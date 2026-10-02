//! Models, devices, links and the roofline cost model: a line-by-line port of
//! `disagg_sim/hardware.py`.
//!
//! Parity notes. Python keeps FLOP counts as exact integers and converts them
//! to float only when they meet a float; this port does the same with `i128`.
//! Every float expression keeps Python's operation order, because floating-point
//! addition and multiplication are not associative.

use crate::pymath::floordiv;

pub const GB: f64 = 1e9;
pub const TB: f64 = 1e12;

/// A decoder-only transformer, described by its shape alone.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub name: &'static str,
    pub n_layers: i64,
    pub d_model: i64,
    pub n_heads: i64,
    pub n_kv_heads: i64,
    pub d_ff: i64,
    pub vocab: i64,
    pub weight_bytes: f64,
    pub kv_bytes: f64,
}

impl ModelSpec {
    pub fn head_dim(&self) -> i64 {
        self.d_model / self.n_heads
    }

    pub fn params_per_layer(&self) -> i64 {
        let (d, kv) = (self.d_model, self.n_kv_heads * self.head_dim());
        let attn = 2 * d * d + 2 * d * kv; // Wq, Wo + Wk, Wv (GQA-narrow)
        let mlp = 3 * d * self.d_ff; // SwiGLU: gate, up, down
        attn + mlp
    }

    /// Total parameters (untied input embedding and LM head).
    pub fn params(&self) -> i64 {
        self.n_layers * self.params_per_layer() + 2 * self.vocab * self.d_model
    }

    /// Parameters that take part in a matmul per token (the embedding is a lookup).
    pub fn matmul_params(&self) -> i64 {
        self.n_layers * self.params_per_layer() + self.vocab * self.d_model
    }

    pub fn weight_bytes_total(&self) -> f64 {
        self.params() as f64 * self.weight_bytes
    }

    /// K and V, for every layer, for one token.
    pub fn kv_bytes_per_token(&self) -> f64 {
        (2 * self.n_layers * self.n_kv_heads * self.head_dim()) as f64 * self.kv_bytes
    }
}

pub const LLAMA3_8B: ModelSpec = ModelSpec {
    name: "Llama-3-8B",
    n_layers: 32,
    d_model: 4096,
    n_heads: 32,
    n_kv_heads: 8,
    d_ff: 14336,
    vocab: 128256,
    weight_bytes: 2.0,
    kv_bytes: 2.0,
};
pub const LLAMA3_70B: ModelSpec = ModelSpec {
    name: "Llama-3-70B",
    n_layers: 80,
    d_model: 8192,
    n_heads: 64,
    n_kv_heads: 8,
    d_ff: 28672,
    vocab: 128256,
    weight_bytes: 2.0,
    kv_bytes: 2.0,
};

pub fn model(key: &str) -> Option<ModelSpec> {
    match key {
        "llama3-8b" => Some(LLAMA3_8B),
        "llama3-70b" => Some(LLAMA3_70B),
        _ => None,
    }
}

/// One device. Peak numbers are datasheet values; efficiencies derate them.
/// Power coefficients are illustrative, as in the Python model.
#[derive(Debug, Clone, PartialEq)]
pub struct Accelerator {
    pub name: &'static str,
    pub peak_flops: f64,
    pub mem_bw: f64,
    pub mem_capacity: f64,
    pub flops_eff: f64,
    pub bw_eff: f64,
    pub tdp_w: f64,
    pub idle_w: f64,
    pub pj_per_flop: f64,
    pub pj_per_byte: f64,
}

pub fn device(key: &str) -> Option<Accelerator> {
    let base = Accelerator {
        name: "H100-SXM",
        peak_flops: 989.0 * TB,
        mem_bw: 3.35 * TB,
        mem_capacity: 80.0 * GB,
        flops_eff: 0.55,
        bw_eff: 0.80,
        tdp_w: 700.0,
        idle_w: 100.0,
        pj_per_flop: 1.0,
        pj_per_byte: 60.0,
    };
    match key {
        "h100" => Some(base),
        "a100" => Some(Accelerator {
            name: "A100-SXM",
            peak_flops: 312.0 * TB,
            mem_bw: 2.039 * TB,
            tdp_w: 400.0,
            idle_w: 60.0,
            pj_per_flop: 1.6,
            pj_per_byte: 70.0,
            ..base
        }),
        "optical" => Some(Accelerator {
            name: "Hypothetical-optical-MAC",
            peak_flops: 4000.0 * TB,
            flops_eff: 0.4,
            idle_w: 180.0,
            pj_per_flop: 0.1,
            ..base
        }),
        _ => None,
    }
}

/// The fabric that carries KV cache from prefill to decode instances.
#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    pub name: &'static str,
    pub bandwidth: f64,
    pub latency: f64,
    pub channels: usize,
    pub pj_per_bit: f64,
}

impl Link {
    pub fn transfer_time(&self, nbytes: f64) -> f64 {
        self.latency + nbytes / self.bandwidth
    }
}

pub fn link(key: &str) -> Option<Link> {
    let (name, bandwidth, latency, pj_per_bit) = match key {
        "nvlink4" => ("NVLink 4 (one direction)", 450.0 * GB, 5e-6, 5.0),
        "ib-ndr" => ("InfiniBand NDR 400G", 50.0 * GB, 10e-6, 15.0),
        "pcie5" => ("PCIe Gen5 x16", 64.0 * GB, 5e-6, 6.0),
        "eth-100g" => ("100 GbE", 12.5 * GB, 20e-6, 15.0),
        "eth-25g" => ("25 GbE", 3.125 * GB, 20e-6, 15.0),
        _ => return None,
    };
    Some(Link {
        name,
        bandwidth,
        latency,
        channels: 1,
        pj_per_bit,
    })
}

/// Which roof set a step's time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    Compute,
    Memory,
    Power,
}

/// Time and energy of one batch step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepCost {
    pub flops: f64,
    pub bytes: f64,
    pub time: f64,
    pub bound: Bound,
    pub compute_j: f64,
    pub memory_j: f64,
}

/// `math.copysign(abs(x) ** (1 / 3), x)`: `pow`, not `cbrt`, to match Python.
fn py_cbrt(x: f64) -> f64 {
    x.abs().powf(1.0 / 3.0).copysign(x)
}

/// Roofline timing for one forward pass of a batch on one instance
/// (`n_devices` accelerators with tensor parallelism, treated as one device).
#[derive(Debug, Clone)]
pub struct CostModel {
    pub model: ModelSpec,
    pub device: Accelerator,
    pub n_devices: i64,
    pub step_overhead: f64,
    pub mem_util: f64,
    pub power_cap_w: Option<f64>,
    pub enforce_tdp: bool,
    pub dvfs: bool,
    pub s_min: f64,
    // Derived once, as Python's cached_property does.
    flops_rate: f64,
    byte_rate: f64,
    joules_per_flop: f64,
    joules_per_byte: f64,
    pub idle_w: f64,
    budget: Option<f64>,
    weight_bytes: f64,
    kv_per_token: f64,
    matmul2: i128,
    attn: i128,
}

/// The configuration is invalid (for example a power cap below idle power).
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

impl CostModel {
    pub fn new(
        model: ModelSpec,
        device: Accelerator,
        n_devices: i64,
        step_overhead: f64,
        power_cap_w: Option<f64>,
        dvfs: bool,
    ) -> Result<Self, ConfigError> {
        let caps: Vec<f64> = [power_cap_w, Some(device.tdp_w)]
            .into_iter()
            .flatten()
            .collect();
        let budget = caps
            .into_iter()
            .reduce(f64::min)
            .map(|c| (c - device.idle_w) * n_devices as f64);
        if budget.is_some_and(|b| b <= 0.0) {
            return Err(ConfigError("power cap is below idle power".into()));
        }
        Ok(CostModel {
            flops_rate: device.peak_flops * device.flops_eff * n_devices as f64,
            byte_rate: device.mem_bw * device.bw_eff * n_devices as f64,
            joules_per_flop: device.pj_per_flop * 1e-12,
            joules_per_byte: device.pj_per_byte * 1e-12,
            idle_w: device.idle_w * n_devices as f64,
            budget,
            weight_bytes: model.weight_bytes_total(),
            kv_per_token: model.kv_bytes_per_token(),
            matmul2: 2 * model.matmul_params() as i128,
            attn: 2 * (model.n_layers * model.d_model) as i128,
            model,
            device,
            n_devices,
            step_overhead,
            mem_util: 0.9,
            power_cap_w,
            enforce_tdp: true,
            dvfs,
            s_min: 0.4,
        })
    }

    /// Sequences' worth of KV cache that fit beside the weights.
    pub fn kv_capacity_tokens(&self) -> Result<i64, ConfigError> {
        let free =
            self.device.mem_capacity * self.n_devices as f64 * self.mem_util - self.weight_bytes;
        if free <= 0.0 {
            return Err(ConfigError(format!(
                "{} does not fit on {}x {}",
                self.model.name, self.n_devices, self.device.name
            )));
        }
        Ok(floordiv(free, self.kv_per_token) as i64)
    }

    /// The DVFS / power-cap step model of `CostModel.step_time` (see hardware.py
    /// for the derivation). Returns (seconds, compute J, memory J, bound).
    pub fn step_time(&self, flops: f64, nbytes: f64) -> (f64, f64, f64, Bound) {
        let (tc, tm) = (flops / self.flops_rate, nbytes / self.byte_rate);
        let (mut ec, em) = (flops * self.joules_per_flop, nbytes * self.joules_per_byte);
        let mut s = if self.dvfs && tc < tm {
            self.s_min.max(tc / tm)
        } else {
            1.0
        };
        let mut bound = if tc >= tm {
            Bound::Compute
        } else {
            Bound::Memory
        };
        if let Some(budget) = self.budget
            && (ec * s * s + em) / (tc / s).max(tm) > budget
        {
            bound = Bound::Power;
            // Memory still sets the time: (Ec s^2 + Em) / tm <= B
            let x = (budget * tm - em) / ec;
            if x > 0.0 && x.sqrt() * tm >= tc {
                s = s.min(x.sqrt());
            } else {
                // Compute sets the time: Cardano for s^3 + p s + q = 0, p > 0.
                let (p, q) = (em / ec, -budget * tc / ec);
                let r = (q * q / 4.0 + p * p * p / 27.0).sqrt();
                s = py_cbrt(-q / 2.0 + r) + py_cbrt(-q / 2.0 - r);
            }
            s = s.max(self.s_min);
        }
        let mut t = (tc / s).max(tm);
        ec = ec * s * s;
        if let Some(budget) = self.budget
            && (ec + em) / t > budget
        {
            t = (ec + em) / budget; // still too hot at s_min: stretch the step
        }
        (t + self.step_overhead, ec, em, bound)
    }

    fn cost(&self, flops: i128, nbytes: f64) -> StepCost {
        let flops = flops as f64;
        let (time, compute_j, memory_j, bound) = self.step_time(flops, nbytes);
        StepCost {
            flops,
            bytes: nbytes,
            time,
            bound,
            compute_j,
            memory_j,
        }
    }

    /// One prefill step over whole prompts (no chunking).
    pub fn prefill(&self, prompt_lens: impl Iterator<Item = i64> + Clone) -> StepCost {
        let tokens: i64 = prompt_lens.clone().sum();
        let mut flops = self.matmul2 * tokens as i128;
        // causal attention: QK^T and AV, each 2*d*c FLOPs at position c
        flops += prompt_lens
            .map(|s| self.attn * s as i128 * (s as i128 + 1))
            .sum::<i128>();
        let nbytes = self.weight_bytes + tokens as f64 * self.kv_per_token;
        self.cost(flops, nbytes)
    }

    /// One decode step; the cost depends only on the batch size and total context.
    pub fn decode_sum(&self, ctx: i64, batch: i64) -> StepCost {
        let flops = self.matmul2 * batch as i128 + 2 * self.attn * ctx as i128;
        let nbytes = self.weight_bytes + (ctx + batch) as f64 * self.kv_per_token;
        self.cost(flops, nbytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_70b_shape() {
        // Same as test_model_sizes in the Python package.
        let m = LLAMA3_70B;
        assert!((m.params() as f64 / 70.6e9 - 1.0).abs() < 0.01);
        assert_eq!(m.kv_bytes_per_token(), 327_680.0); // 2 * 80 * 8 * 128 * 2
    }

    #[test]
    fn decode_is_memory_bound_and_prefill_compute_bound() {
        let cm =
            CostModel::new(LLAMA3_70B, device("h100").unwrap(), 4, 0.5e-3, None, false).unwrap();
        assert_eq!(cm.decode_sum(4096, 8).bound, Bound::Memory);
        assert_eq!(cm.prefill([4096i64].into_iter()).bound, Bound::Compute);
    }

    #[test]
    fn cap_below_idle_is_an_error() {
        assert!(
            CostModel::new(
                LLAMA3_70B,
                device("h100").unwrap(),
                4,
                0.5e-3,
                Some(50.0),
                false
            )
            .is_err()
        );
    }
}
