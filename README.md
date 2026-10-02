# Rust_DES_Kernel

A small **discrete-event simulation kernel in Rust**, and on top of it a
**bit-exact Rust port** of the engine and cost model of
[Disaggregated_Inference_Sim](https://github.com/BrendanJamesLynskey/Disaggregated_Inference_Sim)
(a SimPy simulator of prefill/decode-disaggregated LLM serving), exposed to
Python with [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs).

It is the companion code for the
[Simulation Engineering Toolkit](https://github.com/BrendanJamesLynskey/SimEng_Hub_Toolkit)
series, decks 01 (Rust for simulation engineers), 02 (porting a simulator core with PyO3)
and 06 (testing frameworks for simulators).

The point is not "Rust is fast". It is to show the whole job of moving a
simulator core across languages *without changing a single answer*:

* **Same answers, to the bit.** For the same configuration and workload, every
  timestamp on every request and every number in the summary are identical to
  the Python simulator's, including the power-capped DVFS model. Tested by
  differential tests (Python reference in the loop) and by golden files (no
  Python needed).
* **Same workloads.** Python's `random.Random` (MT19937, `init_by_array`, the
  53-bit `random()`, `expovariate`, `lognormvariate`) is reproduced exactly, so
  the Rust generator emits the same requests as the Python one for the same seed.
* **Measured, not claimed.** The Rust core runs 53–69× faster than the SimPy
  model and 29–31× faster than its exact fast path; releasing the GIL lets
  Python threads run simulations in parallel (2.8× on 4 cores). All numbers are in
  [`examples/results.md`](examples/results.md).

---

## Quick start

```bash
# Rust only
cargo test --release                       # unit, golden parity, proptest, kernel vs theory
cargo run --release --bin disagg-rs -- --prefill 2 --rate 6 --link eth-25g
cargo run --release --bin disagg-rs -- --sweep 2 3 4 5 6 8      # parallel sweep (rayon), CSV
cargo bench --bench engine                 # criterion

# Python
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release -E dev           # builds rust_des, installs disagg-sim for the tests
pytest pytests                             # differential tests against the Python simulator
python examples/results.py                 # regenerates examples/results.md
```

```python
import rust_des
from disagg_sim.sim import SimConfig

rows = rust_des.poisson_workload(4.0, 1000, 2048, 0.5, 256, 0.5, seed=0)
run = rust_des.simulate(SimConfig(n_prefill=2), rows)      # a SimConfig or a plain dict
print(run.summary["latency_s"]["ttft"]["p99"], run.events, run.core_s)
```

Example report (`disagg-rs --prefill 2 --rate 6 --link eth-25g`):

```
utilisation  prefill-0 47%  prefill-1 13%  decode-0 100%  kv-link 97%
hot-spot     stage=kv_wait -> kv-link (busy 97%)
```

---

## What's in the box

| Path | Role |
|------|------|
| `src/kernel.rs` | The generic kernel: a `BinaryHeap<Reverse<…>>` ordered by (time, priority, sequence), SimPy's tie-breaking rule; a `Model` trait; `run` |
| `src/queueing.rs` | An M/D/1 queue on the kernel, checked against the Pollaczek–Khinchine formula |
| `src/disagg/hardware.rs` | Models, devices, links and the roofline cost model with DVFS and power caps (exact integer FLOP counts, Python's operation order) |
| `src/disagg/workload.rs` | Requests, their timestamps, and the Poisson/lognormal workload generator |
| `src/disagg/engine.rs` | The serving engine: each SimPy process becomes an explicit state machine; events carry indices, not references |
| `src/disagg/metrics.rs` | `summarise()` and `energy_report()`, returning the same nested structure as the Python |
| `src/pyrand.rs`, `src/pymath.rs` | Python's `random.Random`, `math.fsum`, float `//` and `round()`, reproduced exactly |
| `src/python.rs`, `python/rust_des/` | The PyO3 boundary (`rust_des._native`) and a thin Python wrapper |
| `src/bin/disagg-rs.rs` | Command-line runner and parallel sweeps |
| `tests/`, `pytests/` | The verification ladder (below) |
| `benches/engine.rs` | Criterion benchmarks of the kernel and the engine |
| `examples/results.py` | Writes `examples/results.md`, the source of every number quoted anywhere |
| `.github/workflows/ci.yml`, `Jenkinsfile`, `ci/perf_gate.py` | CI: GitHub Actions, and a Jenkins pipeline with JUnit, coverage and a performance-regression gate |

### The verification ladder

| Rung | Where | What it proves |
|------|-------|----------------|
| Unit | `src/**` `#[cfg(test)]` | Kernel ordering, cost-model regimes, CPython's random stream, `fsum`, floor division |
| Analytic | `tests/kernel.rs` | The M/D/1 mean wait matches Pollaczek–Khinchine within 3% |
| Property-based | `tests/kernel.rs`, `tests/props.rs` (proptest) | For random configurations: events pop in order; every request finishes or is rejected; timestamps never go backwards; stages sum to end-to-end latency; KV reservations all return; TDP and caps are never exceeded; Little's law holds exactly; runs are deterministic |
| Golden | `tests/golden.rs`, `tests/pymath.rs` | Fourteen recorded Python runs (every preset, link and power-cap mode, ties, bursts, rejections) replayed with identical timestamps and summaries; four Python-generated workloads reproduced by the Rust generator; `fsum` and float `//` against 1,416 CPython-computed cases |
| Differential | `pytests/test_differential.py` | Ten named configurations, tie-heavy and rejection-heavy workloads, and Hypothesis-generated configurations, all bit-identical to the live Python simulator |
| Performance | `benches/`, `ci/perf_gate.py` | Criterion timings, gated against a stored baseline |
| Mutation | `cargo mutants` | Do the tests notice deliberate bugs? Survivors led to `tests/pymath.rs`, the kernel's equality test and the tight-KV golden cases |

Test quality is measured, and recorded in `examples/results.md` (sections 6–8):

```bash
cargo llvm-cov --release --json --summary-only --output-path target/llvm-cov-summary.json
systemd-run --user --scope -p MemoryMax=5G \
  cargo mutants -j 2 --exclude src/python.rs --timeout 30 -o target/mutants-after-run
python examples/results.py
```

### Recorded results (from `examples/results.md`)

| Workload | Python SimPy | Python fast path | Rust core | Speed-up vs SimPy | Rust events/s |
|----------|--------------|------------------|-----------|-------------------|---------------|
| 1P1D, 1,000 requests | 0.264 s | 0.132 s | 4.3 ms | 62× | 4.73 M |
| Colocated ×2, 1,000 requests | 0.306 s | n/a | 4.5 ms | 69× | 6.05 M |
| 2P2D, 10,000 requests | 2.753 s | 1.494 s | 52.2 ms | 53× | 3.91 M |

Machine: Intel i7-3770 (4 cores, 8 threads), Python 3.12, SimPy 4.1.2, rustc 1.99.
Parity: 6 configurations × 1,000 requests, 0 differing timestamps, identical summaries.
Tests: 46 (29 Rust, 17 Python); line coverage 97.5%; mutation score 95% (786 of 830 viable, non-timeout mutants caught).

---

## Parity notes (why bit-exact is possible, and what it took)

* **Same float operation order.** `a*b*c` is `(a*b)*c` in both languages, and
  is kept that way; FLOP counts stay exact integers (`i128`) until Python would
  convert them.
* **SimPy's event order.** Ties at equal times are broken by priority, then
  by scheduling order. The kernel does the same, and each Python `yield` is
  mirrored by scheduling the same event in the same order.
* **SimPy's "trigger once" rule.** A wake-up already scheduled is not scheduled
  again. Ordinary workloads never test this; a bursty one does (two KV transfers
  finishing together into an idle decoder), so it is a golden case. Without the
  guard the engine double-wakes the decoder and its invariant check panics.
* **`statistics.fmean` uses `math.fsum`**, which is correctly rounded; a plain
  loop differs in the last bit. `pymath::fsum` is a port of CPython's.
* **`serde_json` needs `float_roundtrip`.** Its default float parser can be one
  ulp off; the golden test caught it.
* **Transcendentals.** `ln`, `exp` and `pow(x, 1/3)` come from the platform libm
  in both languages, so they agree on Linux/glibc (tested). On another platform
  the differential test falls back to a 1e-12 relative tolerance for the power
  model.

## Not ported (yet)

The time-series probe and Chrome trace export (passive in Python, so they do
not affect results), the exact fast path (`FastDecodeInstance`; the Rust baseline
is already 29–31× faster than it), and the search/sweep helpers.

---

## Part of

The [Simulation Engineering Toolkit](https://github.com/BrendanJamesLynskey/SimEng_Hub_Toolkit)
series. Sister series: [LLM Inference Simulators](https://github.com/BrendanJamesLynskey/LLM_Hub_Inference_Simulators)
and [FHE Accelerator Simulators](https://github.com/BrendanJamesLynskey/FHE_Hub_Accelerator_Simulators).
Rust interview questions: [Interview_Rust](https://github.com/BrendanJamesLynskey/Interview_Rust).
