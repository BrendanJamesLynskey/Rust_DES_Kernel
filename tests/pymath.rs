//! `fsum` and float floor division against CPython's own answers
//! (`fixtures/pymath.json`, written by `pytests/make_pymath_fixture.py`).
//! Added after mutation testing found branches the unit tests never checked.

use rust_des_kernel::pymath::{floordiv, fsum};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/pymath.json")).unwrap()
}

#[test]
fn fsum_matches_cpython_bit_for_bit() {
    for case in fixture()["fsum"].as_array().unwrap() {
        let xs: Vec<f64> = case["xs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let want = case["sum"].as_f64().unwrap();
        assert_eq!(
            fsum(xs.iter().copied()).to_bits(),
            want.to_bits(),
            "fsum({xs:?})"
        );
    }
}

#[test]
fn floordiv_matches_cpython_bit_for_bit() {
    for case in fixture()["floordiv"].as_array().unwrap() {
        let (a, b) = (case["a"].as_f64().unwrap(), case["b"].as_f64().unwrap());
        let want = case["q"].as_f64().unwrap();
        assert_eq!(floordiv(a, b).to_bits(), want.to_bits(), "{a} // {b}");
    }
}
