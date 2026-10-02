//! Property-based invariants of the serving engine (proptest).
//!
//! Each property must hold for *every* configuration and workload, so proptest
//! generates them at random and shrinks any failure to a minimal case.

use proptest::prelude::*;
use rust_des_kernel::disagg::{ConfigSpec, LengthDist, poisson_workload, simulate, summarise};

fn spec() -> impl Strategy<Value = ConfigSpec> {
    (
        prop::bool::ANY,
        1usize..4,
        1usize..4,
        prop::sample::select(vec!["ib-ndr", "eth-25g", "nvlink4", "pcie5", "eth-100g"]),
        1usize..4,
        prop::option::of(200.0f64..700.0),
        prop::bool::ANY,
        prop::sample::select(vec![8usize, 64, 256]),
    )
        .prop_map(|(colo, a, b, link, ch, cap, dvfs, maxb)| ConfigSpec {
            mode: if colo { "colocated" } else { "disagg" }.into(),
            n_prefill: a,
            n_decode: b,
            n_colocated: a,
            link: link.into(),
            link_channels: ch,
            power_cap_w: cap,
            dvfs,
            max_decode_batch: maxb,
            ..Default::default()
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn serving_invariants(spec in spec(), rate in 0.5f64..10.0, seed in 0u64..1000,
                          pcv in 0.0f64..1.2, ocv in 0.0f64..1.2) {
        let wl = poisson_workload(rate, 120, LengthDist::new(1500.0, pcv), LengthDist::new(100.0, ocv), seed);
        let cfg = spec.build().unwrap();
        let tdp = cfg.device.tdp_w;
        let res = simulate(cfg, wl).unwrap();

        // Every request either finishes or is rejected, never both.
        let finished = res.requests.iter().filter(|r| r.finish.is_some()).count();
        prop_assert_eq!(finished + res.rejected.len(), res.requests.len());

        for r in res.requests.iter().filter(|r| r.finish.is_some()) {
            // Timestamps never go backwards along a request's path.
            let path: Vec<f64> = [Some(r.arrival), r.prefill_start, r.first_token, r.kv_start,
                                  r.kv_ready, r.decode_start, r.finish].into_iter().flatten().collect();
            prop_assert!(path.windows(2).all(|w| w[0] <= w[1]), "{:?}", path);
            // One inter-token latency per token after the first, all positive.
            prop_assert_eq!(r.itls.len() as i64, (r.output_len - 1).max(0));
            prop_assert!(r.itls.iter().all(|&x| x > 0.0));
            // Stages partition end-to-end latency.
            let sum: f64 = r.stages().iter().sum();
            prop_assert!((sum - r.e2e()).abs() <= 1e-9 * r.e2e().max(1.0));
        }
        for i in &res.instances {
            // All KV reservations are returned; no instance is busy more than 100%.
            prop_assert_eq!(i.kv_used(), 0);
            prop_assert!(i.busy <= res.horizon * (1.0 + 1e-12));
            // The board power limit (and any cap) is respected on every step.
            let limit = spec.power_cap_w.unwrap_or(tdp).min(tdp) * i.cost.n_devices as f64;
            prop_assert!(i.peak_power <= limit * (1.0 + 1e-9), "{} > {}", i.peak_power, limit);
        }

        let m = summarise(&res);
        let parts: f64 = ["static", "compute", "memory", "link"].iter()
            .map(|k| m["energy"]["breakdown"][k].as_f64().unwrap()).sum();
        prop_assert!((parts - 1.0).abs() < 1e-12);
        if res.rejected.is_empty() {
            // Little's law holds exactly over a run that ends empty.
            let l = m["littles_law"]["L_measured"].as_f64().unwrap();
            let lw = m["littles_law"]["lambda_W"].as_f64().unwrap();
            prop_assert!((l - lw).abs() <= 1e-9 * l.max(1e-9), "L {} vs lambda W {}", l, lw);
        }
    }

    #[test]
    fn runs_are_deterministic(seed in 0u64..1000) {
        let run = || {
            let wl = poisson_workload(6.0, 80, LengthDist::new(2048.0, 0.6), LengthDist::new(128.0, 0.6), seed);
            simulate(ConfigSpec::default().build().unwrap(), wl).unwrap().requests
        };
        prop_assert_eq!(run(), run());
    }
}
