"""Regenerate examples/results.md: every number quoted in the README and the decks.

    maturin develop --release
    cargo bench --bench engine          # optional: fills the criterion table
    python examples/results.py
    python examples/results.py --keep-timings   # rerun everything except sections 2-4 (wall clock)

Timings are wall-clock medians on whatever machine runs this; the header of
results.md records which one. ``--keep-timings`` copies sections 2-4 from the current
results.md: use it when nothing on the timed path changed (2026-10-04: heterogeneous pools
only choose each instance's device at construction).
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import simpy

import rust_des
from disagg_sim.hardware import A100_SXM, H100_SXM, LINKS, LLAMA3_8B
from disagg_sim.metrics import summarise
from disagg_sim.sim import SimConfig, simulate
from disagg_sim.workload import LengthDist, poisson_workload

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "results.md"
PROMPT, OUTPUT = LengthDist(2048, 0.5), LengthDist(256, 0.5)


def median_time(fn, repeats):
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def workload(rate, n, seed=0):
    return poisson_workload(rate, n, PROMPT, OUTPUT, seed=seed)


def rows_of(wl):
    return [(r.arrival, r.prompt_len, r.output_len) for r in wl]


def normalise(x):
    """JSON has no NaN or infinity; the Rust summary sends None for both."""
    if isinstance(x, dict):
        return {k: normalise(v) for k, v in x.items()}
    if isinstance(x, list):
        return [normalise(v) for v in x]
    return None if isinstance(x, float) and not math.isfinite(x) else x


def count_simpy_events(cfg, wl):
    """Events SimPy processes for one run (counted in a separate, untimed run)."""
    orig, n = simpy.Environment.step, [0]

    def step(self):
        n[0] += 1
        return orig(self)

    simpy.Environment.step = step
    try:
        simulate(cfg, wl)
    finally:
        simpy.Environment.step = orig
    return n[0]


# ── 1. parity ────────────────────────────────────────────────────────────
PARITY = {
    "1P1D, InfiniBand NDR": SimConfig(),
    "2P2D, 25 GbE, 2 channels": SimConfig(n_prefill=2, n_decode=2, link=replace(LINKS["eth-25g"], channels=2)),
    "Colocated, 2 instances": SimConfig(mode="colocated", n_colocated=2),
    "1P1D, 350 W cap + DVFS": SimConfig(power_cap_w=350.0, dvfs=True),
    "Colocated, 300 W cap": SimConfig(mode="colocated", power_cap_w=300.0),
    "Fixed lengths (simultaneous events)": SimConfig(n_prefill=2, n_decode=2,
                                                    link=replace(LINKS["ib-ndr"], channels=2)),
}
# Heterogeneous pools (2026-10-04): a device and a device count per pool.
PARITY_HETERO = {
    "8B: H100 prefill + A100 decode": SimConfig(model=LLAMA3_8B, devices_per_instance=1, prefill_device=H100_SXM,
                                                decode_device=A100_SXM),
    "8B: 2x A100 prefill + H100 decode, co-packaged optics": SimConfig(
        model=LLAMA3_8B, devices_per_instance=1, n_prefill=2, prefill_device=A100_SXM, decode_device=H100_SXM,
        link=LINKS["cpo-optical"]),
    "70B: 4x H100 prefill + 8x A100 decode, per-pool caps + DVFS": SimConfig(
        decode_device=A100_SXM, decode_devices_per_instance=8, prefill_power_cap_w=450.0, decode_power_cap_w=300.0,
        dvfs=True),
    "70B: explicit identical pools": SimConfig(prefill_device=H100_SXM, decode_device=H100_SXM),
}


def parity(table=None):
    lines = ["| Configuration | Requests | Timestamps compared | Differing | Summary dict identical |",
             "|---|---|---|---|---|"]
    for name, cfg in (table or PARITY).items():
        cv = 0.0 if name.startswith("Fixed") else 0.6
        wl = poisson_workload(5.0, 1000, LengthDist(2048, cv), LengthDist(128, cv), seed=3)
        rows = rows_of(wl)
        res = simulate(cfg, wl)
        rs = rust_des.simulate(cfg, rows)
        py_stamps = [tuple(getattr(r, k) for k in rust_des.STAMPS) for r in res.requests]
        n_cmp = sum(x is not None for s in py_stamps for x in s)
        n_diff = sum(a != b for s, t in zip(py_stamps, rs.stamps) for a, b in zip(s, t))
        same = normalise(summarise(res)) == rs.summary
        lines.append(f"| {name} | {len(rows):,} | {n_cmp:,} | {n_diff} | {'yes' if same else 'NO'} |")
    return "\n".join(lines)


# ── 2. speed ─────────────────────────────────────────────────────────────
SPEED = [
    ("1P1D, 1,000 requests at 4 req/s", SimConfig(sample_dt=1e9), 4.0, 1000),
    ("Colocated x2, 1,000 requests at 4 req/s", SimConfig(mode="colocated", sample_dt=1e9), 4.0, 1000),
    ("2P2D, 10,000 requests at 8 req/s", SimConfig(n_prefill=2, n_decode=2, sample_dt=1e9), 8.0, 10000),
]


def speed():
    rows_md = ["| Workload | Python SimPy | Python fast path | Rust core | Rust via PyO3 (total) | Core speed-up vs SimPy | vs fast path | Rust events/s | SimPy events/s |",
               "|---|---|---|---|---|---|---|---|---|"]
    split = ["| Workload | Rust simulate | Rust summarise | Boundary (conversion) | Python `summarise()` |",
             "|---|---|---|---|---|"]
    for name, cfg, rate, n in SPEED:
        reps = 3 if n > 2000 else 5
        rows = rows_of(workload(rate, n))
        t_py = median_time(lambda: simulate(cfg, workload(rate, n)), reps)
        fast = replace(cfg, fast_forward=True) if cfg.mode == "disagg" else None
        t_fast = median_time(lambda: simulate(fast, workload(rate, n)), reps) if fast else None
        res = simulate(cfg, workload(rate, n))
        t_sum_py = median_time(lambda: summarise(res), reps)
        runs, totals = [], []
        for _ in range(reps):                                # total, core and summary from the same call
            t = time.perf_counter()
            runs.append(rust_des.simulate(cfg, rows))
            totals.append(time.perf_counter() - t)
        t_total = statistics.median(totals)
        core = statistics.median(r.core_s for r in runs)
        summ = statistics.median(r.summary_s for r in runs)
        boundary = statistics.median(t - r.core_s - r.summary_s for t, r in zip(totals, runs))
        ev_rs = runs[0].events
        ev_py = count_simpy_events(cfg, workload(rate, n))
        fast_s = f"{t_fast:.3f} s" if fast else "n/a"
        vs_fast = f"{t_fast / core:.0f}x" if fast else "n/a"
        rows_md.append(f"| {name} | {t_py:.3f} s | {fast_s} | {1e3 * core:.1f} ms | {1e3 * t_total:.1f} ms | "
                       f"{t_py / core:.0f}x | {vs_fast} | {ev_rs / core / 1e6:.2f} M | {ev_py / t_py / 1e3:.0f} k |")
        split.append(f"| {name} | {1e3 * core:.1f} ms | {1e3 * summ:.1f} ms | {1e3 * (t_total - core - summ):.1f} ms | "
                     f"{1e3 * t_sum_py:.0f} ms |")
    return "\n".join(rows_md), "\n".join(split)


# ── 3. the GIL ───────────────────────────────────────────────────────────
def _py_run(seed):
    return simulate(SimConfig(sample_dt=1e9), workload(4.0, 1000, seed)).horizon


def gil():
    k = 8
    rows = [rows_of(workload(4.0, 4000, seed)) for seed in range(k)]
    cfg = SimConfig()
    one = lambda i: rust_des.simulate(cfg, rows[i])            # with the summary, as a user would call it
    t_seq = median_time(lambda: [one(i) for i in range(k)], 3)
    with ThreadPoolExecutor(k) as ex:
        t_thr = median_time(lambda: list(ex.map(one, range(k))), 3)
    t_many = median_time(lambda: rust_des.simulate_many([cfg] * k, rows), 3)
    m = 4
    t_py_seq = median_time(lambda: [_py_run(s) for s in range(m)], 1)
    with ThreadPoolExecutor(m) as ex:
        t_py_thr = median_time(lambda: list(ex.map(_py_run, range(m))), 1)
    with ProcessPoolExecutor(m) as ex:
        list(ex.map(_py_run, range(m)))                     # warm the pool
        t_py_proc = median_time(lambda: list(ex.map(_py_run, range(m))), 1)
    return "\n".join([
        "| What | Runs | Wall time | Speed-up over sequential |", "|---|---|---|---|",
        f"| Rust via PyO3, sequential | {k} x 4,000 requests | {t_seq:.3f} s | 1.0x |",
        f"| Rust via PyO3, {k} Python threads (GIL released) | {k} x 4,000 requests | {t_thr:.3f} s | {t_seq / t_thr:.1f}x |",
        f"| Rust `simulate_many` (rayon, one call) | {k} x 4,000 requests | {t_many:.3f} s | {t_seq / t_many:.1f}x |",
        f"| Python SimPy, sequential | {m} x 1,000 requests | {t_py_seq:.2f} s | 1.0x |",
        f"| Python SimPy, {m} threads (GIL held) | {m} x 1,000 requests | {t_py_thr:.2f} s | {t_py_seq / t_py_thr:.1f}x |",
        f"| Python SimPy, {m} processes | {m} x 1,000 requests | {t_py_proc:.2f} s | {t_py_seq / t_py_proc:.1f}x |",
    ])


# ── 4. criterion ─────────────────────────────────────────────────────────
def criterion():
    base = ROOT / "target" / "criterion"
    out = ["| Benchmark | Mean time | Throughput |", "|---|---|---|"]
    found = False
    for group, name in [("kernel", "md1_100k_customers"), ("disagg", "1P1D_1000req"), ("disagg", "colocated_1000req")]:
        est = base / group / name / "new" / "estimates.json"
        bench = base / group / name / "new" / "benchmark.json"
        if not est.exists():
            continue
        found = True
        mean_ns = json.loads(est.read_text())["mean"]["point_estimate"]
        elems = json.loads(bench.read_text()).get("throughput", {}).get("Elements")
        thr = f"{elems / (mean_ns * 1e-9) / 1e6:.1f} M events/s" if elems else ""
        out.append(f"| `{group}/{name}` | {mean_ns / 1e6:.2f} ms | {thr} |")
    return "\n".join(out) if found else "_Run `cargo bench --bench engine` first._"


def cli_example():
    """The report quoted in the README (deterministic: no timings)."""
    args = ["--prefill", "2", "--rate", "6", "--link", "eth-25g"]
    out = subprocess.run(["cargo", "run", "--release", "-q", "--bin", "disagg-rs", "--", *args],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return f"`disagg-rs {' '.join(args)}`\n\n```\n{out.rstrip()}\n```"


# ── 6-8. test inventory, coverage, mutation testing ──────────────────────
def test_inventory():
    out = subprocess.run(["cargo", "test", "--release"], cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True).stdout
    rows, target = [], None
    for line in out.splitlines():
        m = re.search(r"Running (unittests src/lib\.rs|tests/\S+|unittests src/bin/\S+)", line)
        if m:
            target = m[1]
        elif "Doc-tests" in line:
            target = "doc tests"
        m = re.search(r"test result: ok\. (\d+) passed", line)
        if m and target:
            rows.append((target, int(m[1])))
            target = None
    py = subprocess.run([sys.executable, "-m", "pytest", "pytests", "-p", "no:cacheprovider"],
                        cwd=ROOT, capture_output=True, text=True).stdout
    n_py = int(re.findall(r"(\d+) passed", py)[-1])
    what = {"unittests src/lib.rs": "Unit tests inside the modules", "tests/cli.rs": "The disagg-rs binary end to end",
            "tests/golden.rs": "Recorded Python runs replayed bit for bit", "tests/kernel.rs": "M/D/1 against theory; ordering property",
            "tests/props.rs": "proptest invariants over random configurations", "tests/pymath.rs": "fsum and floor division against CPython",
            "doc tests": "Examples in documentation comments"}
    lines = ["| Test target | Tests | What it checks |", "|---|---|---|"]
    lines += [f"| `{t}` | {n} | {what.get(t, '')} |" for t, n in rows if n]
    lines.append(f"| `pytests/` (pytest + Hypothesis) | {n_py} | Differential tests against the live Python simulator |")
    lines.append(f"| **Total** | **{sum(n for _, n in rows) + n_py}** | |")
    return "\n".join(lines)


def coverage():
    f = ROOT / "target" / "llvm-cov-summary.json"
    if not f.exists():
        return "_Run `cargo llvm-cov --release --json --summary-only --output-path target/llvm-cov-summary.json` first._"
    d = json.loads(f.read_text())["data"][0]
    lines = ["| File | Lines | Regions | Functions |", "|---|---|---|---|"]
    for fe in d["files"]:
        name = fe["filename"].split("/src/", 1)[-1]
        sm = fe["summary"]
        lines.append(f"| `src/{name}` | {sm['lines']['percent']:.1f}% | {sm['regions']['percent']:.1f}% | {sm['functions']['percent']:.1f}% |")
    t = d["totals"]
    lines.append(f"| **Total** | **{t['lines']['percent']:.1f}%** | **{t['regions']['percent']:.1f}%** | **{t['functions']['percent']:.1f}%** |")
    return "\n".join(lines)


def mutation_table(path: Path, files=None):
    d = json.loads(path.read_text())
    per = {}
    for o in d["outcomes"]:
        sc = o["scenario"]
        if not isinstance(sc, dict):
            continue
        f = sc["Mutant"]["file"]
        if files and f not in files:
            continue
        per.setdefault(f, {"CaughtMutant": 0, "MissedMutant": 0, "Timeout": 0, "Unviable": 0})
        per[f][o["summary"]] = per[f].get(o["summary"], 0) + 1
    lines = ["| File | Mutants | Caught | Missed | Timeout | Unviable | Score (caught / (caught + missed)) |",
             "|---|---|---|---|---|---|---|"]
    tot = {"CaughtMutant": 0, "MissedMutant": 0, "Timeout": 0, "Unviable": 0}
    for f in sorted(per):
        c = per[f]
        for k in tot:
            tot[k] += c[k]
        n = sum(c.values())
        score = 100 * c["CaughtMutant"] / max(1, c["CaughtMutant"] + c["MissedMutant"])
        lines.append(f"| `{f}` | {n} | {c['CaughtMutant']} | {c['MissedMutant']} | {c['Timeout']} | {c['Unviable']} | {score:.0f}% |")
    n = sum(tot.values())
    score = 100 * tot["CaughtMutant"] / max(1, tot["CaughtMutant"] + tot["MissedMutant"])
    lines.append(f"| **Total** | **{n}** | **{tot['CaughtMutant']}** | **{tot['MissedMutant']}** | **{tot['Timeout']}** | "
                 f"**{tot['Unviable']}** | **{score:.0f}%** |")
    return "\n".join(lines), d.get("cargo_mutants_version", "?")


def mutation():
    """Three rounds: each set of survivors led to new tests before the next run."""
    first = ROOT / "target" / "mutants-before" / "outcomes.json"
    second = ROOT / "target" / "mutants-round2" / "outcomes.json"
    final = ROOT / "target" / "mutants-final-run" / "mutants.out" / "outcomes.json"
    if not final.exists():
        return "_Run cargo-mutants first (see the README)._"
    t_final, ver = mutation_table(final)
    text = (f"cargo-mutants {ver}, whole crate except the PyO3 bindings (`-j 2 --timeout 30`), with the final tests,"
            " rerun on 2026-10-03 after the cost-model correction:"
            f"\n\n{t_final}")
    pre = ROOT / "target" / "mutants-final-run-pre-correction" / "mutants.out" / "outcomes.json"
    if pre.exists():
        tp, _ = mutation_table(pre)
        text += ("\n\nThe same tests before the correction (the survivors are the same 44 mutants; the new"
                 " weight-traffic code added mutants and every one is caught). Totals row:\n\n"
                 + "\n".join(tp.splitlines()[:2]) + "\n" + tp.splitlines()[-1])
    third = ROOT / "target" / "mutants-round3" / "mutants.out" / "outcomes.json"
    if third.exists():
        t3, _ = mutation_table(third)
        total = t3.splitlines()[-1]
        text += ("\n\nThe run before that differed by one fix: the M/D/1 test divided by the theoretical value, so a "
                 "mutant making the theory negative passed. Its totals row:\n\n" + "\n".join(t3.splitlines()[:2]) + "\n" + total)
    if second.exists():
        t2, _ = mutation_table(second)
        text += ("\n\nThe previous full run, before the last round of tests (golden cases for every hardware preset, "
                 "link and power-cap mode; a fixture of Python-generated workloads; the M/D/1 check at a service time "
                 f"other than 1 s):\n\n{t2}")
    if first.exists():
        t1, _ = mutation_table(first, {"src/kernel.rs", "src/pymath.rs", "src/pyrand.rs", "src/queueing.rs"})
        text += ("\n\nThe first run, before `tests/pymath.rs`, the kernel's equality test and the tight-KV golden "
                 "cases (it was interrupted; these four files had finished):\n\n" + t1)
    return text


def rejected():
    """Python-and-JS-only configurations, and the error each gets from the Rust side."""
    from disagg_sim.hardware import KV_PRESETS, MODELS, OPTICAL_FFT, KVTransit
    lines = ["| Configuration | Rust's answer |", "|---|---|"]
    for name, cfg in [("Hyena-2 model", SimConfig(model=MODELS["llama3-8b-hyena"], devices_per_instance=1)),
                      ("block-circulant model", SimConfig(model=MODELS["llama3-8b-hyena-circ"], devices_per_instance=1)),
                      ("optical transform device in the prefill pool", SimConfig(prefill_device=OPTICAL_FFT)),
                      ("fp8 KV compression in transit", SimConfig(kv_transit=KVTransit(KV_PRESETS["fp8"])))]:
        try:
            rust_des.simulate(cfg, [(0.1, 100, 10)])
            lines.append(f"| {name} | ran (unexpected) |")
        except ValueError as e:
            lines.append(f"| {name} | `ValueError: {e}` |")
    return "\n".join(lines)


def hetero_mutation():
    first = ROOT / "target" / "mutants-hetero-first" / "mutants.out" / "outcomes.json"
    again = ROOT / "target" / "mutants-hetero" / "mutants.out" / "outcomes.json"
    if not first.exists():
        return "_Run cargo-mutants on the changed files first (see section 9)._"
    t, ver = mutation_table(first)
    text = (f"cargo-mutants {ver}, the changed modules only (`-j 2 --timeout 30 --file src/disagg/engine.rs --file "
            f"src/disagg/hardware.rs --file src/disagg/metrics.rs`), run alone:\n\n{t}")
    if again.exists():
        d = json.loads(again.read_text())
        n = {}
        for o in d["outcomes"]:
            if isinstance(o["scenario"], dict):
                n[o["summary"]] = n.get(o["summary"], 0) + 1
        old = ROOT / "target" / "mutants-final-run" / "mutants.out" / "missed.txt"
        strip = lambda line: re.sub(r":\d+:\d+:", ":", line.strip())
        known = {strip(x) for x in old.read_text().splitlines()} if old.exists() else set()
        missed = [o for o in d["outcomes"] if o["summary"] == "MissedMutant"]
        new_missed = [o for o in missed if strip(o["scenario"]["Mutant"]["name"]) not in known] if known else missed
        text += (f"\n\nFour survivors were in the new code (three `||` in `SimConfig::heterogeneous`, and the position of"
                 f" `pools` in the summary). Two tests were added for them, and `--iterate` re-tested the"
                 f" {sum(n.values())} survivors and timeouts: {n.get('CaughtMutant', 0)} caught, {n.get('MissedMutant', 0)}"
                 f" missed, {n.get('Timeout', 0)} timeouts. Of the missed, {len(missed) - len(new_missed)} also survived the"
                 f" 2026-10-03 whole-crate run (section 8); {len(new_missed)} are new.")
    return text


def machine():
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")),
               platform.processor())
    try:
        rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        rustc = "rustc not on PATH"
    return (f"{cpu}, {os.cpu_count()} logical CPUs; {platform.system()} {platform.release()}; "
            f"Python {platform.python_version()}; SimPy {simpy.__version__}; {rustc}; rust_des {rust_des.__version__}")


def kept(section: str) -> str:
    """A section of the current results.md, verbatim (``--keep-timings``)."""
    m = re.search(rf"(## {section}\..*?)(?=\n## \d+\.|\Z)", OUT.read_text(), re.S)
    return m[1].rstrip()


def main():
    keep = "--keep-timings" in sys.argv
    if keep:
        timed = "\n\n".join(kept(n) for n in ("2", "3", "4"))
    else:
        speed_md, split_md = speed()
        timed = f"""## 2. Speed

Python runs have the time-series probe off (`sample_dt=1e9`), because the Rust
engine has no probe. "Rust core" is the engine alone, timed inside Rust;
"via PyO3" is the whole call from Python, including converting the workload,
the summary and the result. Event counts differ: SimPy also processes process
start-ups, resource grants and releases.

{speed_md}

### Where the time goes in one PyO3 call

{split_md}

## 3. Releasing the GIL

`rust_des.simulate` runs with the interpreter detached (`Python::detach`), so
Python threads run simulations in parallel. SimPy holds the GIL, so threads
do not help it; processes do.

{gil()}

## 4. Criterion benchmarks (Rust only)

{criterion()}"""
    text = f"""# Recorded results

Generated by `examples/results.py`. Every number in the README and in the
Simulation Engineering Toolkit decks comes from this file.

Machine: {machine()}.

**2026-10-03: cost model corrected.** Disaggregated_Inference_Sim's closed form charged every step
the whole input-embedding table and left out each new decode token's attention to itself; an
operator trace of the real model (Torch_Sim_Frontend) found both. This port was corrected in the
same way (`hardware.rs`), the golden fixture regenerated from the corrected Python, and every
number below re-measured. The summary's single sort is now `sort_unstable_by` (identical output:
values equal under `total_cmp` are bit-identical); unlike the Python original, this port always
sorted once per distribution.

**2026-10-04: heterogeneous pools.** Each pool can have its own device and device count (section 9).
{"Sections 2-4 (wall clock) are kept from the 2026-10-03 run: pools only choose each instance's device when it is built, so nothing on the timed path changed. " if keep else ""}Everything else was rerun.

## 1. Bit-exact parity with the Python simulator

Every timestamp of every request, compared with `==`, and the whole summary
dictionary (`disagg_sim.metrics.summarise`) compared for equality.

{parity()}

{timed}

## 5. Command-line example

{cli_example()}

## 6. Test inventory

{test_inventory()}

## 7. Coverage

`cargo llvm-cov` (LLVM source-based coverage) over every Rust test target, including the CLI tests.

{coverage()}

## 8. Mutation testing

{mutation()}

## 9. Heterogeneous pools (2026-10-04)

A different device, or device count, per pool (the Splitwise idea): H100 prefill with A100 decode, and so on.
Ported bit-exactly; the same comparison as section 1, 1,000 requests each:

{parity(PARITY_HETERO)}

The Python package also has FFT-mixing model variants, an optical transform device and KV hand-off compression.
They are **Python and JS only, not in the Rust port** (owner decision, 2026-10-04), and the Rust side rejects
them by name rather than simulate something else:

{rejected()}

### Mutation testing of the changed modules

{hetero_mutation()}
"""
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
