//! The Python boundary (PyO3). Built by maturin as `rust_des._native`.
//!
//! What crosses the boundary is deliberately small and plain: a JSON config,
//! a list of `(arrival, prompt_len, output_len)` tuples in, and a list of
//! timestamp tuples plus a JSON summary out. The simulation itself runs with
//! the interpreter detached (`Python::detach`, called `allow_threads` before
//! PyO3 0.26), so Python threads can run simulations in parallel.

use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::disagg::{self, ConfigSpec, LengthDist, Request};

type Row = (f64, i64, i64);
type Stamps = (
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
);

fn spec(config_json: &str) -> PyResult<ConfigSpec> {
    serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(format!("bad config: {e}")))
}

fn requests(rows: &[Row]) -> Vec<Request> {
    rows.iter()
        .enumerate()
        .map(|(i, &(a, p, o))| Request::new(i, a, p, o))
        .collect()
}

/// (per-request stamps, summary JSON, events, seconds simulating, seconds summarising)
type Output = (Vec<Stamps>, String, u64, f64, f64);

/// Plain Rust: no Python objects, so it can run on any thread.
fn run_one(spec: &ConfigSpec, rows: &[Row], with_summary: bool) -> Result<Output, String> {
    let cfg = spec.build().map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let res = disagg::simulate(cfg, requests(rows)).map_err(|e| e.to_string())?;
    let t1 = Instant::now();
    let summary = if with_summary {
        disagg::summarise(&res).to_string()
    } else {
        String::new()
    };
    let t2 = Instant::now();
    let stamps = res
        .requests
        .iter()
        .map(|r| {
            (
                r.prefill_start,
                r.first_token,
                r.kv_start,
                r.kv_ready,
                r.decode_start,
                r.finish,
            )
        })
        .collect();
    Ok((
        stamps,
        summary,
        res.events,
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
    ))
}

/// Run one simulation. Returns (per-request stamps, summary JSON, events processed,
/// seconds simulating, seconds summarising).
#[pyfunction]
#[pyo3(signature = (config_json, rows, with_summary=true))]
fn simulate_json(
    py: Python<'_>,
    config_json: &str,
    rows: Vec<Row>,
    with_summary: bool,
) -> PyResult<Output> {
    let spec = spec(config_json)?;
    // The GIL is released for the whole run: nothing in here touches Python.
    py.detach(|| run_one(&spec, &rows, with_summary))
        .map_err(PyValueError::new_err)
}

/// Run many simulations in parallel on Rust threads (rayon), GIL released.
#[pyfunction]
fn simulate_many_json(
    py: Python<'_>,
    configs: Vec<String>,
    rows: Vec<Vec<Row>>,
) -> PyResult<Vec<Output>> {
    if configs.len() != rows.len() {
        return Err(PyValueError::new_err("configs and rows differ in length"));
    }
    let specs = configs
        .iter()
        .map(|c| spec(c))
        .collect::<PyResult<Vec<_>>>()?;
    py.detach(|| {
        specs
            .par_iter()
            .zip(rows.par_iter())
            .map(|(s, r)| run_one(s, r, true))
            .collect::<Result<Vec<_>, _>>()
    })
    .map_err(PyValueError::new_err)
}

/// `poisson_workload` with Python's own random stream: the same rows as
/// `disagg_sim.workload.poisson_workload` for the same arguments.
#[pyfunction]
#[pyo3(signature = (rate, n, prompt_mean, prompt_cv, output_mean, output_cv, seed=0))]
fn poisson_workload(
    rate: f64,
    n: usize,
    prompt_mean: f64,
    prompt_cv: f64,
    output_mean: f64,
    output_cv: f64,
    seed: u64,
) -> Vec<Row> {
    disagg::poisson_workload(
        rate,
        n,
        LengthDist::new(prompt_mean, prompt_cv),
        LengthDist::new(output_mean, output_cv),
        seed,
    )
    .into_iter()
    .map(|r| (r.arrival, r.prompt_len, r.output_len))
    .collect()
}

/// Rust core of rust_des: a bit-exact port of Disaggregated_Inference_Sim's engine.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_json, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_many_json, m)?)?;
    m.add_function(wrap_pyfunction!(poisson_workload, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
