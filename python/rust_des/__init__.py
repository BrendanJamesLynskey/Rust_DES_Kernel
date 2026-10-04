"""rust_des: the Rust core of Disaggregated_Inference_Sim, callable from Python.

The heavy lifting is in the compiled extension ``rust_des._native``. This file
is the thin, Pythonic layer on top: it accepts either a plain dict of config
fields or a ``disagg_sim.SimConfig``, and returns named results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ._native import __version__, poisson_workload, simulate_json, simulate_many_json

__all__ = ["STAMPS", "RunResult", "poisson_workload", "simulate", "simulate_many", "spec_from_simconfig", "__version__"]

STAMPS = ("prefill_start", "first_token", "kv_start", "kv_ready", "decode_start", "finish")


@dataclass
class RunResult:
    stamps: list[tuple]          # one tuple per request, in STAMPS order (None = never reached)
    summary: dict                # same structure as disagg_sim.metrics.summarise() (NaN/inf -> None)
    events: int                  # events the kernel processed
    core_s: float = 0.0          # seconds inside the Rust engine
    summary_s: float = 0.0       # seconds inside the Rust summariser

    def column(self, name: str) -> list:
        j = STAMPS.index(name)
        return [s[j] for s in self.stamps]


def spec_from_simconfig(cfg) -> dict:
    """Translate a ``disagg_sim.SimConfig`` into the config dict the Rust side reads.

    Models, devices and links are passed by name; a hardware object that is not
    one of the named presets is an error rather than a silent approximation.
    Heterogeneous pools cross as device names. The FFT-mixing models, the optical
    transform devices and KV hand-off compression cross by name too, and the Rust
    side rejects them (they are Python and JS only).
    """
    from disagg_sim.hardware import ACCELERATORS, LINKS, MODELS

    def key(table, obj, what, same=lambda a, b: a == b):
        for k, v in table.items():
            if same(v, obj):
                return k
        raise ValueError(f"{what} {obj!r} is not a named preset, so it cannot cross to Rust")

    pd, dd = getattr(cfg, "prefill_device", None), getattr(cfg, "decode_device", None)
    tr = getattr(cfg, "kv_transit", None)
    link = key(LINKS, cfg.link, "link",
               lambda a, b: (a.bandwidth, a.latency, a.pj_per_bit) == (b.bandwidth, b.latency, b.pj_per_bit))
    return {
        "model": key(MODELS, cfg.model, "model"),
        "device": key(ACCELERATORS, cfg.device, "device"),
        "devices_per_instance": cfg.devices_per_instance,
        "mode": cfg.mode,
        "n_prefill": cfg.n_prefill,
        "n_decode": cfg.n_decode,
        "n_colocated": cfg.n_colocated,
        "link": link,
        "link_channels": cfg.link.channels,
        "max_prefill_tokens": cfg.max_prefill_tokens,
        "max_decode_batch": cfg.max_decode_batch,
        "step_overhead": cfg.step_overhead,
        "ttft_slo": cfg.ttft_slo,
        "tpot_slo": cfg.tpot_slo,
        "warmup_frac": cfg.warmup_frac,
        "power_cap_w": cfg.power_cap_w,
        "prefill_power_cap_w": cfg.prefill_power_cap_w,
        "decode_power_cap_w": cfg.decode_power_cap_w,
        "dvfs": cfg.dvfs,
        "prefill_device": None if pd is None else key(ACCELERATORS, pd, "device"),
        "decode_device": None if dd is None else key(ACCELERATORS, dd, "device"),
        "prefill_devices_per_instance": getattr(cfg, "prefill_devices_per_instance", None),
        "decode_devices_per_instance": getattr(cfg, "decode_devices_per_instance", None),
        "kv_transit": None if tr is None else f"{tr.compression.name} at {tr.where}",
    }


def _spec(cfg) -> str:
    return json.dumps(cfg if isinstance(cfg, dict) else spec_from_simconfig(cfg))


def _rows(rows) -> list[tuple]:
    return [(r.arrival, r.prompt_len, r.output_len) if hasattr(r, "arrival") else tuple(r) for r in rows]


def simulate(cfg, rows, summary: bool = True) -> RunResult:
    """Run one simulation. ``rows`` are ``(arrival, prompt_len, output_len)`` tuples
    (or ``disagg_sim`` Requests). The GIL is released while it runs."""
    stamps, s, events, core_s, summary_s = simulate_json(_spec(cfg), _rows(rows), summary)
    return RunResult(stamps, json.loads(s) if summary else {}, events, core_s, summary_s)


def simulate_many(cfgs, rows_list) -> list[RunResult]:
    """Run several simulations in parallel on Rust threads (rayon)."""
    out = simulate_many_json([_spec(c) for c in cfgs], [_rows(rows) for rows in rows_list])
    return [RunResult(s, json.loads(m), e, c, t) for s, m, e, c, t in out]
