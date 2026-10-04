//! A Rust port of the core of `Disaggregated_Inference_Sim`: the roofline cost
//! model (with DVFS and power caps), the workload generator, the event engine
//! for disaggregated and colocated serving, and the metrics.
//!
//! The port is *bit-exact*: for the same configuration and workload it stamps
//! every request with the same timestamps as the SimPy model, and `summarise`
//! returns the same numbers (tested in `tests/golden.rs` and
//! `pytests/test_differential.py`). Heterogeneous pools (a different device per pool) are
//! ported; the FFT-mixing models, the optical transform engine and KV hand-off compression
//! are Python and JS only, and a configuration that asks for them is rejected.

pub mod engine;
pub mod hardware;
pub mod metrics;
pub mod workload;

pub use engine::{ConfigSpec, Mode, Role, SimConfig, SimResult, Simulation, simulate};
pub use metrics::{format_report, summarise};
pub use workload::{LengthDist, Request, poisson_workload};
