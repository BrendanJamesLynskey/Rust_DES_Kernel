//! # rust_des_kernel
//!
//! A small discrete-event simulation kernel in Rust ([`kernel`]), and on top of
//! it a bit-exact port of the engine and cost model of the Python simulator
//! `Disaggregated_Inference_Sim` ([`disagg`]), exposed to Python with PyO3
//! (module `rust_des`, built with maturin).

pub mod disagg;
pub mod kernel;
pub mod pymath;
pub mod pyrand;
pub mod queueing;

#[cfg(feature = "python")]
mod python;
