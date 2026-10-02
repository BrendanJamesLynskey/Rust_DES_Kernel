//! Turning a finished run into numbers: a port of `disagg_sim/metrics.py`.
//!
//! `summarise` returns the same nested structure as Python's `summarise()`, as
//! JSON, so the differential test can compare the two dictionaries whole.
//! Means use `fsum` (as `statistics.fmean` does); NaN and infinity become
//! `null`, because JSON has no spelling for them.

use serde_json::{Map, Value, json};

use super::engine::{Mode, SimResult};
use super::workload::Request;
use crate::pymath::fmean;

pub const STAGES: [&str; 6] = [
    "prefill_queue",
    "prefill",
    "kv_wait",
    "kv_transfer",
    "decode_queue",
    "decode",
];

/// Which resource "owns" each stage: the one to upgrade if that stage dominates.
fn stage_owner(stage: &str) -> &'static str {
    match stage {
        "prefill_queue" | "prefill" => "prefill",
        "kv_wait" | "kv_transfer" => "kv-link",
        "decode_queue" | "decode" => "decode",
        _ => "",
    }
}

/// Linear-interpolated percentile (numpy's default), on unsorted data.
pub fn percentile(xs: &[f64], p: u32) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    percentile_sorted(&s, p)
}

fn percentile_sorted(s: &[f64], p: u32) -> f64 {
    let k = ((s.len() - 1) as u64 * p as u64) as f64 / 100.0;
    let (lo, hi) = (k.floor(), k.ceil());
    let (a, b) = (s[lo as usize], s[hi as usize]);
    a + (b - a) * (k - lo)
}

/// A JSON number, or `null` for NaN and infinities.
fn num(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

fn dist(xs: &[f64]) -> Value {
    if xs.is_empty() {
        return json!({"mean": null, "p50": null, "p90": null, "p99": null, "max": null});
    }
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    json!({
        "mean": num(fmean(xs)),
        "p50": num(percentile_sorted(&s, 50)),
        "p90": num(percentile_sorted(&s, 90)),
        "p99": num(percentile_sorted(&s, 99)),
        "max": num(*s.last().unwrap()),
    })
}

/// `max(d, key=d.get)`: the first key holding the largest value.
fn argmax<'a>(items: impl Iterator<Item = (&'a str, f64)>) -> Option<(&'a str, f64)> {
    items.fold(None, |best, (k, v)| match best {
        Some((_, bv)) if v.partial_cmp(&bv) != Some(std::cmp::Ordering::Greater) => best,
        _ => Some((k, v)),
    })
}

pub fn summarise(res: &SimResult) -> Value {
    let cfg = &res.cfg;
    let h = res.horizon;
    let mut done: Vec<&Request> = res.requests.iter().filter(|r| r.finish.is_some()).collect();
    done.sort_by(|a, b| a.arrival.total_cmp(&b.arrival)); // stable, like sorted()
    let steady = &done[(done.len() as f64 * cfg.warmup_frac) as usize..];

    let ttft: Vec<f64> = steady.iter().map(|r| r.ttft()).collect();
    let tpot: Vec<f64> = steady.iter().filter_map(|r| r.tpot()).collect();
    let itl: Vec<f64> = steady.iter().flat_map(|r| r.itls.iter().copied()).collect();
    let e2e: Vec<f64> = steady.iter().map(|r| r.e2e()).collect();
    let met = steady
        .iter()
        .filter(|r| r.ttft() <= cfg.ttft_slo && r.tpot().is_none_or(|t| t <= cfg.tpot_slo))
        .count();

    let window = if steady.len() > 1 {
        steady[steady.len() - 1].arrival - steady[0].arrival
    } else {
        f64::NAN
    };
    let out_tokens: i64 = done.iter().map(|r| r.output_len).sum();

    let mut util: Vec<(&str, f64)> = res
        .instances
        .iter()
        .map(|i| (i.name.as_str(), i.busy / h))
        .collect();
    util.push(("kv-link", res.link.busy / (cfg.link.channels as f64 * h)));

    let stages: Vec<[f64; 6]> = steady.iter().map(|r| r.stages()).collect();
    let stage_means: Vec<(&str, f64)> = if steady.is_empty() {
        Vec::new()
    } else {
        STAGES
            .iter()
            .enumerate()
            .map(|(j, &s)| (s, fmean(&stages.iter().map(|x| x[j]).collect::<Vec<_>>())))
            .collect()
    };
    let mean_e2e = fmean(&e2e);

    // Little's law: L = lambda W, using the exact time-average population.
    let lam = done.len() as f64 / h;
    let w = fmean(&done.iter().map(|r| r.e2e()).collect::<Vec<_>>());
    let little_l = res.in_system_area / h;

    // The hot-spot: the stage with the longest mean wait (decode time excluded),
    // mapped to the resource that owns it.
    let hottest =
        argmax(stage_means.iter().filter(|(s, _)| *s != "decode").copied()).map(|(s, _)| s);
    let mut owner = hottest.map_or("", stage_owner);
    if cfg.mode == Mode::Colocated && owner != "kv-link" {
        owner = "colocated";
    }
    let (hot_res, hot_util) = argmax(util.iter().filter(|(k, _)| k.starts_with(owner)).copied())
        .or_else(|| argmax(util.iter().copied()))
        .unwrap();

    let energy = energy_report(res, out_tokens, met);

    let obj = |pairs: &[(&str, f64)]| -> Value {
        Value::Object(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), num(*v)))
                .collect::<Map<_, _>>(),
        )
    };
    let mut efficiency = Map::new();
    let mut cbf = Map::new();
    for i in &res.instances {
        let n = i.cost.n_devices as f64;
        efficiency.insert(
            i.name.clone(),
            json!({
                "mfu": num(i.flops / (h * i.cost.device.peak_flops * n)),
                "mbu": num(i.bytes / (h * i.cost.device.mem_bw * n)),
                "mean_batch": num(if i.steps > 0 { i.batch_sum as f64 / i.steps as f64 } else { 0.0 }),
                "steps": i.steps,
            }),
        );
        cbf.insert(
            i.name.clone(),
            num(if i.busy != 0.0 {
                i.compute_bound_time / i.busy
            } else {
                0.0
            }),
        );
    }
    let share: Vec<(&str, f64)> = stage_means
        .iter()
        .map(|&(s, v)| (s, v / mean_e2e))
        .collect();
    let link = &res.link;
    json!({
        "mode": if cfg.mode == Mode::Disagg { "disagg" } else { "colocated" },
        "requests": {"completed": done.len(), "rejected": res.rejected.len(), "measured": steady.len()},
        "latency_s": {"ttft": dist(&ttft), "tpot": dist(&tpot), "itl": dist(&itl), "e2e": dist(&e2e)},
        "throughput": {
            "output_tok_per_s": num(out_tokens as f64 / h),
            "req_per_s": num(lam),
            "goodput_req_per_s": num(if window > 0.0 { met as f64 / window } else { f64::NAN }),
            "slo_attainment": num(if steady.is_empty() { f64::NAN } else { met as f64 / steady.len() as f64 }),
        },
        "utilisation": obj(&util),
        "kv_link": {
            "bytes_GB": num(link.bytes / 1e9),
            "transfers": link.transfers,
            "mean_wait_ms": num(1e3 * link.wait / link.transfers.max(1) as f64),
            "achieved_GBps": num(link.bytes / h / 1e9),
        },
        "efficiency": efficiency,
        "compute_bound_fraction": cbf,
        "stage_breakdown_s": obj(&stage_means),
        "stage_share": obj(&share),
        "hotspots": {
            "stage": hottest,
            "resource": hot_res,
            "resource_util": num(hot_util),
            "busy_over_90pct": util.iter().filter(|(_, v)| *v > 0.9).map(|(k, _)| *k).collect::<Vec<_>>(),
        },
        "littles_law": {"L_measured": num(little_l), "lambda_W": num(lam * w)},
        "energy": energy,
        "sim_time_s": num(h),
    })
}

/// Static power for the whole run + dynamic energy of the work + link energy.
pub fn energy_report(res: &SimResult, out_tokens: i64, slo_met: usize) -> Value {
    let h = res.horizon;
    let mut per = Map::new();
    let (mut stat, mut compute, mut memory) = (0.0, 0.0, 0.0);
    for i in &res.instances {
        let s_j = i.cost.idle_w * h;
        let (c_j, m_j) = (i.compute_j, i.memory_j);
        (stat, compute, memory) = (stat + s_j, compute + c_j, memory + m_j);
        per.insert(
            i.name.clone(),
            json!({
                "avg_w": num((s_j + c_j + m_j) / h),
                "peak_step_w": num(i.peak_power),
                "power_bound_frac": num(if i.busy != 0.0 { i.power_bound_time / i.busy } else { 0.0 }),
            }),
        );
    }
    let link = res.link.energy;
    let total = stat + compute + memory + link;
    json!({
        "total_J": num(total),
        "avg_power_W": num(total / h),
        "J_per_output_token": num(if out_tokens != 0 { total / out_tokens as f64 } else { f64::NAN }),
        "output_tokens_per_J": num(if total != 0.0 { out_tokens as f64 / total } else { f64::NAN }),
        "J_per_slo_met_request": num(if slo_met != 0 { total / slo_met as f64 } else { f64::INFINITY }),
        "breakdown": {"static": num(stat / total), "compute": num(compute / total),
                      "memory": num(memory / total), "link": num(link / total)},
        "per_instance": per,
    })
}

/// A short text report, in the style of `format_report`.
pub fn format_report(m: &Value) -> String {
    let f = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
    let ms = |v: &Value| format!("{:8.1}", 1e3 * f(v));
    let mut out = vec![format!(
        "── {} ── {} done, {} rejected, sim {:.1}s",
        m["mode"].as_str().unwrap(),
        m["requests"]["completed"],
        m["requests"]["rejected"],
        f(&m["sim_time_s"])
    )];
    out.push("latency (ms)       mean      p50      p90      p99".into());
    for k in ["ttft", "tpot", "itl", "e2e"] {
        let d = &m["latency_s"][k];
        out.push(format!(
            "  {k:<10} {} {} {} {}",
            ms(&d["mean"]),
            ms(&d["p50"]),
            ms(&d["p90"]),
            ms(&d["p99"])
        ));
    }
    let t = &m["throughput"];
    out.push(format!(
        "throughput   {:9.0} tok/s   {:.2} req/s   goodput {:.2} req/s   SLO met {:.1}%",
        f(&t["output_tok_per_s"]),
        f(&t["req_per_s"]),
        f(&t["goodput_req_per_s"]),
        100.0 * f(&t["slo_attainment"])
    ));
    let util: Vec<String> = m["utilisation"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| format!("{k} {:.0}%", 100.0 * f(v)))
        .collect();
    out.push(format!("utilisation  {}", util.join("  ")));
    let e = &m["energy"];
    out.push(format!(
        "power        avg {:.0} W   {:.2} J/token",
        f(&e["avg_power_W"]),
        f(&e["J_per_output_token"])
    ));
    let h = &m["hotspots"];
    out.push(format!(
        "hot-spot     stage={} -> {} (busy {:.0}%)",
        h["stage"].as_str().unwrap_or("None"),
        h["resource"].as_str().unwrap(),
        100.0 * f(&h["resource_util"])
    ));
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy_linear() {
        let xs = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(percentile(&xs, 50), 2.5);
        assert_eq!(percentile(&xs, 90), 3.7);
        assert!(percentile(&[], 50).is_nan());
    }

    #[test]
    fn argmax_keeps_the_first_of_equals() {
        assert_eq!(
            argmax([("a", 1.0), ("b", 2.0), ("c", 2.0)].into_iter()),
            Some(("b", 2.0))
        );
    }
}
