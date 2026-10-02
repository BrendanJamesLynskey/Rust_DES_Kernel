//! The event engine for prefill/decode-disaggregated serving: a port of the
//! SimPy model in `disagg_sim/sim.py`.
//!
//! SimPy expresses each instance as a generator that `yield`s timeouts. Rust
//! has no stable generators, so each instance is an explicit state machine:
//! an event resumes it, it does what the Python process would do up to its
//! next `yield`, and it schedules the event that resumes it next. The rules
//! that decide *which* event runs first at equal times are SimPy's (see
//! [`crate::kernel`]), so the two produce the same timestamps to the bit.
//!
//! Ownership pattern: the simulation owns every request and instance in a
//! `Vec`, and events carry indices. Nothing holds a reference across an
//! event, so the borrow checker has nothing to object to, and there is no
//! `Rc<RefCell<…>>` anywhere.
//!
//! Topology (`Mode::Disagg`):
//!
//! ```text
//! arrivals -> router -> prefill[0..P) -> KV link (C channels) -> router -> decode[0..D) -> done
//! ```
//!
//! `Mode::Colocated` replaces the two pools with N instances that do both jobs,
//! a waiting prefill pre-empting the next decode step (vLLM v0 style).

use std::collections::VecDeque;

use serde::Deserialize;

use super::hardware::{
    self, Accelerator, Bound, ConfigError, CostModel, Link, ModelSpec, StepCost,
};
use super::workload::Request;
use crate::kernel::Scheduler;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Disagg,
    Colocated,
}

/// Every knob of `SimConfig` in Python, with the same defaults.
#[derive(Debug, Clone)]
pub struct SimConfig {
    pub model: ModelSpec,
    pub device: Accelerator,
    pub devices_per_instance: i64,
    pub mode: Mode,
    pub n_prefill: usize,
    pub n_decode: usize,
    pub n_colocated: usize,
    pub link: Link,
    pub max_prefill_tokens: i64,
    pub max_decode_batch: usize,
    pub step_overhead: f64,
    pub ttft_slo: f64,
    pub tpot_slo: f64,
    pub warmup_frac: f64,
    pub power_cap_w: Option<f64>,
    pub prefill_power_cap_w: Option<f64>,
    pub decode_power_cap_w: Option<f64>,
    pub dvfs: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        ConfigSpec::default().build().expect("defaults are valid")
    }
}

/// The serialisable form of [`SimConfig`]: names instead of structs. This is
/// what crosses the Python boundary and what the CLI reads.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConfigSpec {
    pub model: String,
    pub device: String,
    pub devices_per_instance: i64,
    pub mode: String,
    pub n_prefill: usize,
    pub n_decode: usize,
    pub n_colocated: usize,
    pub link: String,
    pub link_channels: usize,
    pub max_prefill_tokens: i64,
    pub max_decode_batch: usize,
    pub step_overhead: f64,
    pub ttft_slo: f64,
    pub tpot_slo: f64,
    pub warmup_frac: f64,
    pub power_cap_w: Option<f64>,
    pub prefill_power_cap_w: Option<f64>,
    pub decode_power_cap_w: Option<f64>,
    pub dvfs: bool,
}

impl Default for ConfigSpec {
    fn default() -> Self {
        ConfigSpec {
            model: "llama3-70b".into(),
            device: "h100".into(),
            devices_per_instance: 4,
            mode: "disagg".into(),
            n_prefill: 1,
            n_decode: 1,
            n_colocated: 2,
            link: "ib-ndr".into(),
            link_channels: 1,
            max_prefill_tokens: 8192,
            max_decode_batch: 256,
            step_overhead: 0.5e-3,
            ttft_slo: 1.0,
            tpot_slo: 0.025,
            warmup_frac: 0.1,
            power_cap_w: None,
            prefill_power_cap_w: None,
            decode_power_cap_w: None,
            dvfs: false,
        }
    }
}

impl ConfigSpec {
    pub fn build(&self) -> Result<SimConfig, ConfigError> {
        let unknown = |what: &str, v: &str| ConfigError(format!("unknown {what} {v:?}"));
        let mode = match self.mode.as_str() {
            "disagg" => Mode::Disagg,
            "colocated" => Mode::Colocated,
            m => return Err(unknown("mode", m)),
        };
        let mut link = hardware::link(&self.link).ok_or_else(|| unknown("link", &self.link))?;
        link.channels = self.link_channels.max(1);
        let n = match mode {
            Mode::Disagg => (self.n_prefill, self.n_decode),
            Mode::Colocated => (self.n_colocated, 1),
        };
        if n.0 == 0 || n.1 == 0 {
            return Err(ConfigError("every pool needs at least one instance".into()));
        }
        Ok(SimConfig {
            model: hardware::model(&self.model).ok_or_else(|| unknown("model", &self.model))?,
            device: hardware::device(&self.device)
                .ok_or_else(|| unknown("device", &self.device))?,
            devices_per_instance: self.devices_per_instance,
            mode,
            n_prefill: self.n_prefill,
            n_decode: self.n_decode,
            n_colocated: self.n_colocated,
            link,
            max_prefill_tokens: self.max_prefill_tokens,
            max_decode_batch: self.max_decode_batch,
            step_overhead: self.step_overhead,
            ttft_slo: self.ttft_slo,
            tpot_slo: self.tpot_slo,
            warmup_frac: self.warmup_frac,
            power_cap_w: self.power_cap_w,
            prefill_power_cap_w: self.prefill_power_cap_w,
            decode_power_cap_w: self.decode_power_cap_w,
            dvfs: self.dvfs,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Prefill,
    Decode,
    Colocated,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::Prefill => "prefill",
            Role::Decode => "decode",
            Role::Colocated => "colocated",
        }
    }
}

/// The step an instance is executing: what to do when its timeout fires.
#[derive(Debug, Clone)]
enum InFlight {
    Prefill(Vec<usize>, StepCost),
    Decode(StepCost),
}

/// One serving instance and its accounting (`Instance` in sim.py).
#[derive(Debug, Clone)]
pub struct Instance {
    pub role: Role,
    pub name: String,
    pub cost: CostModel,
    queue: VecDeque<usize>,
    running: Vec<usize>,
    kv_used: i64,
    pub kv_cap: i64,
    pub busy: f64,
    pub steps: u64,
    pub compute_bound_time: f64,
    pub flops: f64,
    pub bytes: f64,
    pub batch_sum: i64,
    pub compute_j: f64,
    pub memory_j: f64,
    pub peak_power: f64,
    pub power_bound_time: f64,
    waiting: bool,        // blocked on its idle event (SimPy: `_wake is not None`)
    wake_triggered: bool, // that event has been triggered but not yet processed
    inflight: Option<InFlight>,
}

impl Instance {
    /// KV-cache tokens reserved right now (zero once every sequence has left).
    pub fn kv_used(&self) -> i64 {
        self.kv_used
    }
}

/// KV-link statistics (`LinkStats` in sim.py).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinkStats {
    pub busy: f64,
    pub bytes: f64,
    pub transfers: u64,
    pub wait: f64,
    pub energy: f64,
}

/// Everything the engine's events can be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ev {
    Arrival(usize),
    Wake(usize),
    StepDone(usize),
    KvDone(usize),
    Stop,
}

/// A finished run (`SimResult` in sim.py).
#[derive(Debug, Clone)]
pub struct SimResult {
    pub cfg: SimConfig,
    pub requests: Vec<Request>,
    pub instances: Vec<Instance>,
    pub link: LinkStats,
    pub horizon: f64,
    pub in_system_area: f64,
    pub rejected: Vec<usize>,
    pub events: u64,
}

pub struct Simulation {
    cfg: SimConfig,
    reqs: Vec<Request>,
    inst: Vec<Instance>,
    front: Vec<usize>, // indices into `inst`: the pool arrivals go to (prefill or colocated)
    decode: Vec<usize>,
    kv_in_use: usize,
    kv_queue: VecDeque<usize>,
    link: LinkStats,
    rejected: Vec<usize>,
    n_done: usize,
    n_in_system: i64,
    area: f64,
    area_t: f64,
    kv_cap: i64,
    stop_scheduled: bool,
}

impl Simulation {
    pub fn new(cfg: SimConfig, workload: Vec<Request>) -> Result<Self, ConfigError> {
        if workload.is_empty() {
            return Err(ConfigError("empty workload".into()));
        }
        let mut inst = Vec::new();
        let mut mk = |role: Role, idx: usize| -> Result<usize, ConfigError> {
            let cap = match role {
                Role::Prefill => cfg.prefill_power_cap_w,
                Role::Decode => cfg.decode_power_cap_w,
                Role::Colocated => None,
            }
            .or(cfg.power_cap_w);
            let cost = CostModel::new(
                cfg.model.clone(),
                cfg.device.clone(),
                cfg.devices_per_instance,
                cfg.step_overhead,
                cap,
                cfg.dvfs,
            )?;
            let kv_cap = cost.kv_capacity_tokens()?;
            inst.push(Instance {
                role,
                name: format!("{}-{idx}", role.as_str()),
                cost,
                queue: VecDeque::new(),
                running: Vec::new(),
                kv_used: 0,
                kv_cap,
                busy: 0.0,
                steps: 0,
                compute_bound_time: 0.0,
                flops: 0.0,
                bytes: 0.0,
                batch_sum: 0,
                compute_j: 0.0,
                memory_j: 0.0,
                peak_power: 0.0,
                power_bound_time: 0.0,
                waiting: true, // every process starts by finding an empty queue
                wake_triggered: false,
                inflight: None,
            });
            Ok(inst.len() - 1)
        };
        let (front, decode) = match cfg.mode {
            Mode::Disagg => (
                (0..cfg.n_prefill)
                    .map(|i| mk(Role::Prefill, i))
                    .collect::<Result<Vec<_>, _>>()?,
                (0..cfg.n_decode)
                    .map(|i| mk(Role::Decode, i))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Mode::Colocated => (
                (0..cfg.n_colocated)
                    .map(|i| mk(Role::Colocated, i))
                    .collect::<Result<Vec<_>, _>>()?,
                Vec::new(),
            ),
        };
        let kv_cap = inst[*decode.first().unwrap_or(&front[0])].kv_cap;
        Ok(Simulation {
            cfg,
            reqs: workload,
            inst,
            front,
            decode,
            kv_in_use: 0,
            kv_queue: VecDeque::new(),
            link: LinkStats::default(),
            rejected: Vec::new(),
            n_done: 0,
            n_in_system: 0,
            area: 0.0,
            area_t: 0.0,
            kv_cap,
            stop_scheduled: false,
        })
    }

    /// Run to completion.
    pub fn run(mut self) -> SimResult {
        let mut s = Scheduler::new();
        // The arrivals process starts at t = 0 and sleeps until the first arrival.
        s.schedule_in(self.reqs[0].arrival.max(0.0), Ev::Arrival(0));
        while let Some((_, ev)) = s.pop() {
            if ev == Ev::Stop {
                break;
            }
            self.handle(&mut s, ev);
        }
        self.population(s.now(), 0);
        SimResult {
            cfg: self.cfg,
            requests: self.reqs,
            instances: self.inst,
            link: self.link,
            horizon: s.now(),
            in_system_area: self.area,
            rejected: self.rejected,
            events: s.processed(),
        }
    }

    fn handle(&mut self, s: &mut Scheduler<Ev>, ev: Ev) {
        match ev {
            Ev::Arrival(k) => self.arrival(s, k),
            Ev::Wake(i) => {
                let inst = &mut self.inst[i];
                inst.waiting = false;
                inst.wake_triggered = false;
                self.resume(s, i);
            }
            Ev::StepDone(i) => self.step_done(s, i),
            Ev::KvDone(rid) => self.kv_done(s, rid),
            Ev::Stop => unreachable!(),
        }
    }

    // ── population accounting (exact time-average for Little's law) ──────
    fn population(&mut self, now: f64, delta: i64) {
        self.area += self.n_in_system as f64 * (now - self.area_t);
        self.area_t = now;
        self.n_in_system += delta;
    }

    fn finish(&mut self, s: &mut Scheduler<Ev>, rid: usize) {
        let now = s.now();
        self.reqs[rid].finish = Some(now);
        self.population(now, -1);
        self.n_done += 1;
        self.maybe_stop(s);
    }

    fn maybe_stop(&mut self, s: &mut Scheduler<Ev>) {
        if self.n_done + self.rejected.len() == self.reqs.len() && !self.stop_scheduled {
            self.stop_scheduled = true;
            s.schedule_now(Ev::Stop); // SimPy: done.succeed(), then run(until=done) returns
        }
    }

    fn arrival(&mut self, s: &mut Scheduler<Ev>, k: usize) {
        let now = s.now();
        self.population(now, 1);
        let r = &self.reqs[k];
        if r.prompt_len + r.output_len > self.kv_cap {
            self.population(now, -1);
            self.rejected.push(k);
            self.maybe_stop(s);
        } else {
            let target = self.least_loaded(&self.front);
            self.submit(s, target, k);
        }
        // The arrivals process loops straight on to its next timeout.
        if let Some(next) = self.reqs.get(k + 1) {
            s.schedule_in((next.arrival - now).max(0.0), Ev::Arrival(k + 1));
        }
    }

    fn load(&self, i: usize) -> i64 {
        let inst = &self.inst[i];
        match inst.role {
            Role::Prefill => inst.queue.iter().map(|&r| self.reqs[r].prompt_len).sum(),
            _ => (inst.queue.len() + inst.running.len()) as i64,
        }
    }

    /// `min(pool, key=load)`: the first instance with the smallest load.
    fn least_loaded(&self, pool: &[usize]) -> usize {
        let mut best = pool[0];
        let mut best_load = self.load(best);
        for &i in &pool[1..] {
            let l = self.load(i);
            if l < best_load {
                best = i;
                best_load = l;
            }
        }
        best
    }

    fn submit(&mut self, s: &mut Scheduler<Ev>, i: usize, rid: usize) {
        let inst = &mut self.inst[i];
        inst.queue.push_back(rid);
        if inst.waiting && !inst.wake_triggered {
            inst.wake_triggered = true;
            s.schedule_now(Ev::Wake(i));
        }
    }

    fn kv_need(&self, rid: usize) -> i64 {
        let r = &self.reqs[rid];
        r.prompt_len + r.output_len // reserve the whole sequence up front
    }

    /// The instance's process runs from its last `yield` to its next one.
    fn resume(&mut self, s: &mut Scheduler<Ev>, i: usize) {
        let now = s.now();
        match self.inst[i].role {
            Role::Prefill => {
                if self.inst[i].queue.is_empty() {
                    self.inst[i].waiting = true;
                    return;
                }
                let budget = self.cfg.max_prefill_tokens;
                let (mut batch, mut tokens) = (Vec::new(), 0i64);
                while let Some(&r) = self.inst[i].queue.front() {
                    if !batch.is_empty() && tokens + self.reqs[r].prompt_len > budget {
                        break;
                    }
                    self.inst[i].queue.pop_front();
                    batch.push(r);
                    tokens += self.reqs[r].prompt_len;
                }
                self.start_prefill(s, i, batch, now);
            }
            Role::Decode => {
                self.admit_decode(i, now);
                if self.inst[i].running.is_empty() {
                    self.inst[i].waiting = true;
                } else {
                    self.start_decode(s, i);
                }
            }
            Role::Colocated => {
                let (max_b, max_t) = (self.cfg.max_decode_batch, self.cfg.max_prefill_tokens);
                let (mut batch, mut tokens) = (Vec::new(), 0i64);
                while let Some(&r) = self.inst[i].queue.front() {
                    let inst = &self.inst[i];
                    if inst.running.len() + batch.len() >= max_b
                        || (!batch.is_empty() && tokens + self.reqs[r].prompt_len > max_t)
                        || inst.kv_used + self.kv_need(r) > inst.kv_cap
                    {
                        break;
                    }
                    let need = self.kv_need(r);
                    let inst = &mut self.inst[i];
                    inst.queue.pop_front();
                    inst.kv_used += need;
                    batch.push(r);
                    tokens += self.reqs[r].prompt_len;
                }
                if !batch.is_empty() {
                    self.start_prefill(s, i, batch, now);
                } else if !self.inst[i].running.is_empty() {
                    self.start_decode(s, i);
                } else {
                    self.inst[i].waiting = true;
                }
            }
        }
    }

    fn start_prefill(&mut self, s: &mut Scheduler<Ev>, i: usize, batch: Vec<usize>, now: f64) {
        for &r in &batch {
            self.reqs[r].prefill_start = Some(now);
        }
        let reqs = &self.reqs;
        let cost = self.inst[i]
            .cost
            .prefill(batch.iter().map(|&r| reqs[r].prompt_len));
        s.schedule_in(cost.time, Ev::StepDone(i));
        self.inst[i].inflight = Some(InFlight::Prefill(batch, cost));
    }

    fn start_decode(&mut self, s: &mut Scheduler<Ev>, i: usize) {
        let inst = &self.inst[i];
        let ctx: i64 = inst
            .running
            .iter()
            .map(|&r| self.reqs[r].prompt_len + self.reqs[r].tokens_out)
            .sum();
        let cost = inst.cost.decode_sum(ctx, inst.running.len() as i64);
        s.schedule_in(cost.time, Ev::StepDone(i));
        self.inst[i].inflight = Some(InFlight::Decode(cost));
    }

    fn admit_decode(&mut self, i: usize, now: f64) {
        let max_b = self.cfg.max_decode_batch;
        while let Some(&r) = self.inst[i].queue.front() {
            let need = self.kv_need(r);
            let inst = &mut self.inst[i];
            if inst.running.len() >= max_b || inst.kv_used + need > inst.kv_cap {
                break;
            }
            inst.queue.pop_front();
            inst.kv_used += need;
            inst.running.push(r);
            self.reqs[r].decode_start = Some(now);
        }
    }

    /// `Instance.step` after its timeout, plus `account_power`.
    fn account(&mut self, i: usize, c: &StepCost, batch: i64) {
        let inst = &mut self.inst[i];
        inst.busy += c.time;
        inst.steps += 1;
        inst.flops += c.flops;
        inst.bytes += c.bytes;
        inst.batch_sum += batch;
        inst.compute_j += c.compute_j;
        inst.memory_j += c.memory_j;
        let p = inst.cost.idle_w + (c.compute_j + c.memory_j) / c.time;
        if p > inst.peak_power {
            inst.peak_power = p;
        }
        match c.bound {
            Bound::Power => inst.power_bound_time += c.time,
            Bound::Compute => inst.compute_bound_time += c.time,
            Bound::Memory => {}
        }
    }

    fn step_done(&mut self, s: &mut Scheduler<Ev>, i: usize) {
        let now = s.now();
        match self.inst[i].inflight.take().expect("a step was in flight") {
            InFlight::Prefill(batch, cost) => {
                self.account(i, &cost, batch.len() as i64);
                for r in batch {
                    let q = &mut self.reqs[r];
                    q.first_token = Some(now);
                    q.last_token = now;
                    q.tokens_out = 1;
                    let short = q.output_len <= 1;
                    match (self.inst[i].role, short) {
                        (Role::Prefill, true) => self.finish(s, r),
                        (Role::Prefill, false) => self.kv_request(s, r),
                        (_, true) => {
                            self.inst[i].kv_used -= self.kv_need(r);
                            self.finish(s, r);
                        }
                        (_, false) => {
                            self.reqs[r].decode_start = Some(now);
                            self.inst[i].running.push(r);
                        }
                    }
                }
            }
            InFlight::Decode(cost) => {
                self.account(i, &cost, self.inst[i].running.len() as i64);
                // Take the batch out so the loop can call `finish(&mut self)`.
                let running = std::mem::take(&mut self.inst[i].running);
                let mut still = Vec::with_capacity(running.len());
                for r in running {
                    let q = &mut self.reqs[r];
                    q.itls.push(now - q.last_token);
                    q.tokens_out += 1;
                    q.last_token = now;
                    if q.tokens_out >= q.output_len {
                        self.inst[i].kv_used -= self.kv_need(r);
                        self.finish(s, r);
                    } else {
                        still.push(r);
                    }
                }
                self.inst[i].running = still;
            }
        }
        self.resume(s, i);
    }

    // ── the KV link: a FIFO resource with `channels` servers ─────────────
    fn kv_request(&mut self, s: &mut Scheduler<Ev>, rid: usize) {
        if self.kv_in_use < self.cfg.link.channels {
            self.kv_start(s, rid);
        } else {
            self.kv_queue.push_back(rid);
        }
    }

    fn kv_bytes(&self, rid: usize) -> f64 {
        self.reqs[rid].prompt_len as f64 * self.cfg.model.kv_bytes_per_token()
    }

    fn kv_start(&mut self, s: &mut Scheduler<Ev>, rid: usize) {
        let now = s.now();
        self.kv_in_use += 1;
        let r = &mut self.reqs[rid];
        r.kv_start = Some(now);
        self.link.wait += now - r.first_token.unwrap();
        s.schedule_in(
            self.cfg.link.transfer_time(self.kv_bytes(rid)),
            Ev::KvDone(rid),
        );
    }

    fn kv_done(&mut self, s: &mut Scheduler<Ev>, rid: usize) {
        let nbytes = self.kv_bytes(rid);
        let link = &self.cfg.link;
        self.kv_in_use -= 1;
        self.link.busy += link.transfer_time(nbytes);
        self.link.bytes += nbytes;
        self.link.transfers += 1;
        self.link.energy += nbytes * 8.0 * link.pj_per_bit * 1e-12;
        self.reqs[rid].kv_ready = Some(s.now());
        if let Some(next) = self.kv_queue.pop_front() {
            self.kv_start(s, next);
        }
        let target = self.least_loaded(&self.decode);
        self.submit(s, target, rid);
    }
}

/// Convenience wrapper: build, run, return.
pub fn simulate(cfg: SimConfig, workload: Vec<Request>) -> Result<SimResult, ConfigError> {
    Ok(Simulation::new(cfg, workload)?.run())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disagg::workload::{LengthDist, poisson_workload};

    #[test]
    fn every_request_finishes_with_ordered_stamps() {
        let wl = poisson_workload(
            4.0,
            200,
            LengthDist::new(2048.0, 0.5),
            LengthDist::new(128.0, 0.5),
            1,
        );
        let res = simulate(SimConfig::default(), wl).unwrap();
        for r in &res.requests {
            let s = [
                r.arrival,
                r.prefill_start.unwrap(),
                r.first_token.unwrap(),
                r.kv_start.unwrap(),
                r.kv_ready.unwrap(),
                r.decode_start.unwrap(),
                r.finish.unwrap(),
            ];
            assert!(s.windows(2).all(|w| w[0] <= w[1]), "{s:?}");
            assert_eq!(r.itls.len() as i64, r.output_len - 1);
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn events_are_small() {
        // A tag and an index; the kernel's heap entry adds time, priority and sequence.
        assert_eq!(std::mem::size_of::<Ev>(), 16);
        let mut s = crate::kernel::Scheduler::new();
        s.schedule_now(Ev::Stop);
        assert_eq!(s.entry_size(), 40);
    }

    #[test]
    fn unknown_names_are_errors() {
        let spec = ConfigSpec {
            link: "carrier-pigeon".into(),
            ..Default::default()
        };
        assert!(spec.build().is_err());
    }
}
