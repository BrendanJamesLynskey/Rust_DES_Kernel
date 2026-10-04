"""Regenerate tests/fixtures/golden.json from the Python simulator.

The fixture lets `cargo test` check bit-exactness without Python installed:
each case is a config, a workload, and the timestamps and summary that
Disaggregated_Inference_Sim produced for it.

    python pytests/make_golden.py
"""

import json
import math
from dataclasses import replace
from pathlib import Path

import rust_des
from disagg_sim.hardware import A100_SXM, H100_SXM, HYPOTHETICAL_OPTICAL, LINKS, LLAMA3_8B
from disagg_sim.metrics import summarise
from disagg_sim.sim import SimConfig, simulate
from disagg_sim.workload import LengthDist, poisson_workload

# (name, config, rate, length cv, output mean, burst grid in seconds or None)
CASES = [
    ("1P1D", SimConfig(), 5.0, 0.6, 128, None),
    ("2P2D eth-25g x2", SimConfig(n_prefill=2, n_decode=2, link=replace(LINKS["eth-25g"], channels=2)), 5.0, 0.6, 128, None),
    ("colocated x2", SimConfig(mode="colocated", n_colocated=2), 5.0, 0.6, 128, None),
    ("cap 350 W + DVFS", SimConfig(power_cap_w=350.0, dvfs=True), 5.0, 0.6, 128, None),
    ("fixed lengths (ties)", SimConfig(n_prefill=2, n_decode=2, link=replace(LINKS["ib-ndr"], channels=2)), 8.0, 0.0, 128, None),
    ("simultaneous arrivals", SimConfig(n_prefill=2, n_decode=2), 6.0, 0.0, 128, 0.5),
    # Bursts queue behind a busy prefill, leave in one batch, and their KV transfers finish
    # together into an idle decoder: two submissions before its wake-up runs.
    ("simultaneous KV completions into an idle decoder", SimConfig(link=replace(LINKS["ib-ndr"], channels=2)),
     1.0, 0.0, 16, 3.0),
    # Llama-3-70B on two H100s leaves little room for KV cache: admission control binds,
    # long prompts are rejected, and some requests want a single token. (Added after
    # mutation testing showed no Rust test exercised these paths.)
    ("tight KV capacity, rejections", SimConfig(devices_per_instance=2), 3.0, 1.2, 4, None),
    ("tight KV capacity, colocated", SimConfig(mode="colocated", devices_per_instance=2), 3.0, 1.2, 4, None),
    # Every hardware preset and link, and the per-pool power caps. (Added after a second
    # mutation run: the presets were only exercised by the Python differential suite.)
    ("8B on A100 over PCIe", SimConfig(model=LLAMA3_8B, device=A100_SXM, devices_per_instance=1,
                                       link=LINKS["pcie5"], n_prefill=2), 5.0, 0.6, 128, None),
    ("optical part over 100 GbE", SimConfig(device=HYPOTHETICAL_OPTICAL, link=LINKS["eth-100g"]), 5.0, 0.6, 128, None),
    ("per-pool caps + DVFS over NVLink", SimConfig(prefill_power_cap_w=450.0, decode_power_cap_w=250.0, dvfs=True,
                                                   link=LINKS["nvlink4"]), 5.0, 0.6, 128, None),
    ("colocated, 300 W cap", SimConfig(mode="colocated", power_cap_w=300.0), 5.0, 0.6, 128, None),
    ("1P1D, 200 W cap (compute-bound throttling)", SimConfig(power_cap_w=200.0), 3.0, 0.6, 128, None),
    # Heterogeneous pools (2026-10-04): a device and a device count per pool, and the
    # co-packaged-optics link preset.
    ("8B: H100 prefill + A100 decode", SimConfig(model=LLAMA3_8B, devices_per_instance=1, prefill_device=H100_SXM,
                                                 decode_device=A100_SXM), 6.0, 0.6, 128, None),
    ("70B: 2x A100 prefill (4 each) + H100 decode (8), co-packaged optics",
     SimConfig(n_prefill=2, prefill_device=A100_SXM, decode_devices_per_instance=8, link=LINKS["cpo-optical"]),
     5.0, 0.6, 128, None),
]


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return None if isinstance(x, float) and not math.isfinite(x) else x


def main():
    out = []
    for name, cfg, rate, cv, out_len, grid in CASES:
        prompt = LengthDist(20000, cv) if name.startswith("tight") else LengthDist(2048, cv)
        wl = poisson_workload(rate, 300, prompt, LengthDist(out_len, 1.5 if name.startswith("tight") else cv), seed=5)
        if grid:                                    # requests arrive in bursts on a grid
            for r in wl:
                r.arrival = round(r.arrival / grid) * grid
        rows = [[r.arrival, r.prompt_len, r.output_len] for r in wl]
        res = simulate(cfg, wl)
        if name.startswith("tight"):
            assert res.rejected and any(r.output_len == 1 for r in res.requests)
        out.append({"name": name, "config": rust_des.spec_from_simconfig(cfg), "rows": rows,
                    "stamps": [[getattr(r, k) for k in rust_des.STAMPS] for r in res.requests],
                    "summary": clean(summarise(res))})
    # The Rust workload generator must emit Python's rows exactly (tests/golden.rs).
    workloads = []
    for seed, rate, (pm, pcv), (om, ocv) in [(0, 4.0, (1500, 0.8), (200, 1.0)), (7, 2.5, (2048, 0.5), (128, 0.5)),
                                             (2**33 + 5, 9.0, (512, 0.0), (64, 1.4)), (3, 0.7, (20000, 1.2), (4, 1.5))]:
        wl = poisson_workload(rate, 500, LengthDist(pm, pcv), LengthDist(om, ocv), seed=seed)
        workloads.append({"seed": seed, "rate": rate, "prompt": [pm, pcv], "output": [om, ocv],
                          "rows": [[r.arrival, r.prompt_len, r.output_len] for r in wl]})
    wdest = Path(__file__).parent.parent / "tests" / "fixtures" / "workloads.json"
    wdest.write_text(json.dumps(workloads))
    dest = Path(__file__).parent.parent / "tests" / "fixtures" / "golden.json"
    dest.write_text(json.dumps(out))
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB, {len(out)} cases)")


if __name__ == "__main__":
    main()
