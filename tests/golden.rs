//! Bit-exactness without Python: replay the cases in `fixtures/golden.json`
//! (written by `pytests/make_golden.py` from the Python simulator) and require
//! identical timestamps and an identical summary.

use rust_des_kernel::disagg::{
    ConfigSpec, LengthDist, Request, poisson_workload, simulate, summarise,
};
use serde_json::Value;

#[test]
fn matches_the_python_simulator_exactly() {
    let cases: Vec<Value> = serde_json::from_str(include_str!("fixtures/golden.json")).unwrap();
    assert!(cases.len() >= 5);
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let spec: ConfigSpec = serde_json::from_value(case["config"].clone()).unwrap();
        let rows = case["rows"].as_array().unwrap();
        let wl = rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                Request::new(
                    i,
                    r[0].as_f64().unwrap(),
                    r[1].as_i64().unwrap(),
                    r[2].as_i64().unwrap(),
                )
            })
            .collect();
        let res = simulate(spec.build().unwrap(), wl).unwrap();
        for (r, want) in res.requests.iter().zip(case["stamps"].as_array().unwrap()) {
            let got = [
                r.prefill_start,
                r.first_token,
                r.kv_start,
                r.kv_ready,
                r.decode_start,
                r.finish,
            ];
            let want: Vec<Option<f64>> =
                want.as_array().unwrap().iter().map(Value::as_f64).collect();
            assert_eq!(got.to_vec(), want, "{name}: request {}", r.rid);
        }
        assert_eq!(summarise(&res), case["summary"], "{name}: summary differs");
    }
}

#[test]
fn workload_generator_matches_python_random() {
    let cases: Vec<Value> = serde_json::from_str(include_str!("fixtures/workloads.json")).unwrap();
    for c in &cases {
        let f = |v: &Value, i: usize| v[i].as_f64().unwrap();
        let wl = poisson_workload(
            c["rate"].as_f64().unwrap(),
            500,
            LengthDist::new(f(&c["prompt"], 0), f(&c["prompt"], 1)),
            LengthDist::new(f(&c["output"], 0), f(&c["output"], 1)),
            c["seed"].as_u64().unwrap(),
        );
        for (r, want) in wl.iter().zip(c["rows"].as_array().unwrap()) {
            assert_eq!(
                r.arrival,
                want[0].as_f64().unwrap(),
                "seed {}, request {}",
                c["seed"],
                r.rid
            );
            assert_eq!(
                r.prompt_len,
                want[1].as_i64().unwrap(),
                "seed {}, request {}",
                c["seed"],
                r.rid
            );
            assert_eq!(
                r.output_len,
                want[2].as_i64().unwrap(),
                "seed {}, request {}",
                c["seed"],
                r.rid
            );
        }
    }
}
