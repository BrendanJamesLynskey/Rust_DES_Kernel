"""Differential tests: the Rust port against the Python simulator it ports.

The requirement is *bit-exact*: every timestamp on every request, and every
number in the summary, must be identical, not merely close. The one allowed
exception is the closed-form power-cap solution, which calls ``pow(x, 1/3)``;
that comes from the platform libm in both languages, so it is identical here
(Linux, glibc) but is only promised to within an ulp elsewhere.
"""

from __future__ import annotations

import math
import platform
from dataclasses import replace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import rust_des
from disagg_sim.hardware import A100_SXM, H100_SXM, HYPOTHETICAL_OPTICAL, LINKS, LLAMA3_8B
from disagg_sim.metrics import summarise
from disagg_sim.sim import SimConfig, simulate
from disagg_sim.workload import LengthDist, poisson_workload

STAMPS = rust_des.STAMPS
GLIBC = platform.system() == "Linux" and platform.libc_ver()[0] == "glibc"


def normalise(x):
    """JSON has no NaN or infinity; the Rust side sends None for both."""
    if isinstance(x, dict):
        return {k: normalise(v) for k, v in x.items()}
    if isinstance(x, list):
        return [normalise(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def run_both(cfg: SimConfig, rate=5.0, n=300, prompt=(2048, 0.6), output=(128, 0.6), seed=5):
    wl = poisson_workload(rate, n, LengthDist(*prompt), LengthDist(*output), seed=seed)
    rows = [(r.arrival, r.prompt_len, r.output_len) for r in wl]
    py = simulate(cfg, wl)
    rs = rust_des.simulate(cfg, rows)
    return py, rs


def assert_identical(py, rs):
    for j, name in enumerate(STAMPS):
        a = [getattr(r, name) for r in py.requests]
        b = [s[j] for s in rs.stamps]
        if a != b:
            k = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            pytest.fail(f"{name} differs first at request {k}: python {a[k]!r} rust {b[k]!r}")
    assert normalise(summarise(py)) == rs.summary


CASES = {
    "1P1D, InfiniBand": dict(),
    "2P2D, 25 GbE with 2 channels": dict(n_prefill=2, n_decode=2, link=replace(LINKS["eth-25g"], channels=2)),
    "colocated x2": dict(mode="colocated", n_colocated=2),
    "colocated x3, small batch": dict(mode="colocated", n_colocated=3, max_decode_batch=16),
    "power cap 350 W + DVFS": dict(power_cap_w=350.0, dvfs=True),
    "colocated, power cap 300 W": dict(mode="colocated", power_cap_w=300.0),
    "per-pool caps": dict(prefill_power_cap_w=450.0, decode_power_cap_w=250.0, dvfs=True),
    "8B on A100 over PCIe": dict(model=LLAMA3_8B, device=A100_SXM, devices_per_instance=1,
                                 link=LINKS["pcie5"], n_prefill=2),
    "hypothetical optical part": dict(device=HYPOTHETICAL_OPTICAL),
    "3P1D, NVLink, tight prefill budget": dict(n_prefill=3, link=LINKS["nvlink4"], max_prefill_tokens=4096),
    # Heterogeneous pools (2026-10-04): a different device, or device count, per pool.
    "8B: H100 prefill + A100 decode": dict(model=LLAMA3_8B, devices_per_instance=1, prefill_device=H100_SXM,
                                           decode_device=A100_SXM),
    "8B: 2x A100 prefill + H100 decode, co-packaged optics": dict(
        model=LLAMA3_8B, devices_per_instance=1, n_prefill=2, prefill_device=A100_SXM, decode_device=H100_SXM,
        link=LINKS["cpo-optical"]),
    "70B: 4x H100 prefill + 8x A100 decode, per-pool caps": dict(
        decode_device=A100_SXM, decode_devices_per_instance=8, prefill_power_cap_w=450.0, decode_power_cap_w=300.0,
        dvfs=True),
    "70B: explicit identical pools": dict(prefill_device=H100_SXM, decode_device=H100_SXM,
                                          prefill_devices_per_instance=4, decode_devices_per_instance=4),
    "8B: optical-MAC prefill + A100 decode over 25 GbE": dict(
        model=LLAMA3_8B, devices_per_instance=1, prefill_device=HYPOTHETICAL_OPTICAL, decode_device=A100_SXM,
        link=LINKS["eth-25g"]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_bit_exact_against_python(name):
    py, rs = run_both(SimConfig(**CASES[name]))
    assert_identical(py, rs)


def test_fixed_lengths_create_ties_and_still_match():
    """cv = 0 gives identical prompts, so steps and transfers finish at the same
    instants. Only SimPy's tie-breaking rule gets these in the same order."""
    cfg = SimConfig(n_prefill=2, n_decode=2, link=replace(LINKS["ib-ndr"], channels=2))
    py, rs = run_both(cfg, rate=8.0, prompt=(1024, 0.0), output=(64, 0.0))
    assert_identical(py, rs)


def test_simultaneous_arrivals():
    """Bursts of requests at the same instant: two submissions reach a waiting
    instance before its wake-up is processed, which must not wake it twice."""
    def bursty():
        wl = poisson_workload(6.0, 300, LengthDist(2048, 0.0), LengthDist(128, 0.0), seed=5)
        for r in wl:
            r.arrival = round(r.arrival * 2) / 2
        return wl

    assert len({r.arrival for r in bursty()}) < 300 / 2
    for cfg in (SimConfig(n_prefill=2, n_decode=2), SimConfig(mode="colocated", n_colocated=2)):
        # A fresh workload per run: simulate() mutates requests, and a shallow copy
        # (dataclasses.replace) would still share each request's itls list.
        wl = bursty()
        rows = [(r.arrival, r.prompt_len, r.output_len) for r in wl]
        assert_identical(simulate(cfg, wl), rust_des.simulate(cfg, rows))


def test_simultaneous_kv_completions_into_an_idle_decoder():
    """The case that needs SimPy's 'trigger once' rule: two transfers land on an
    idle decoder at the same instant. Without the wake_triggered guard the Rust
    engine double-wakes the decoder and panics ('a step was in flight')."""
    wl = poisson_workload(1.0, 300, LengthDist(2048, 0.0), LengthDist(16, 0.0), seed=5)
    for r in wl:
        r.arrival = round(r.arrival / 3.0) * 3.0
    cfg = SimConfig(link=replace(LINKS["ib-ndr"], channels=2))
    rows = [(r.arrival, r.prompt_len, r.output_len) for r in wl]
    assert_identical(simulate(cfg, wl), rust_des.simulate(cfg, rows))


def test_rejections_and_single_token_outputs():
    cfg = SimConfig(devices_per_instance=2)
    py, rs = run_both(cfg, rate=3.0, n=150, prompt=(20000, 1.2), output=(4, 1.5))
    assert py.rejected, "the case should exercise rejection"
    assert any(r.output_len == 1 for r in py.requests)
    assert_identical(py, rs)


def test_workload_generator_reproduces_python_random():
    for seed in (0, 1, 7, 2**33 + 5):
        wl = poisson_workload(4.0, 500, LengthDist(1500, 0.8), LengthDist(200, 1.0), seed=seed)
        rows = rust_des.poisson_workload(4.0, 500, 1500, 0.8, 200, 1.0, seed)
        assert rows == [(r.arrival, r.prompt_len, r.output_len) for r in wl]


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    mode=st.sampled_from(["disagg", "colocated"]),
    n_a=st.integers(1, 3), n_b=st.integers(1, 3),
    link=st.sampled_from(sorted(LINKS)), channels=st.integers(1, 3),
    rate=st.floats(0.5, 12.0), seed=st.integers(0, 10_000),
    cap=st.one_of(st.none(), st.floats(200.0, 700.0)), dvfs=st.booleans(),
    prompt_cv=st.sampled_from([0.0, 0.3, 0.8]),
    pools=st.sampled_from([(None, None, None), (A100_SXM, None, None), (None, A100_SXM, 8), (A100_SXM, H100_SXM, None)]),
)
def test_random_configurations_match(mode, n_a, n_b, link, channels, rate, seed, cap, dvfs, prompt_cv, pools):
    pre, dec, n_dec = pools
    cfg = SimConfig(mode=mode, n_prefill=n_a, n_decode=n_b, n_colocated=n_a,
                    link=replace(LINKS[link], channels=channels), power_cap_w=cap, dvfs=dvfs,
                    prefill_device=pre, decode_device=dec, decode_devices_per_instance=n_dec)
    py, rs = run_both(cfg, rate=rate, n=120, prompt=(1500, prompt_cv), seed=seed)
    if cap is None or GLIBC:
        assert_identical(py, rs)
    else:  # pow() from a different libm may differ by an ulp
        for j, name in enumerate(STAMPS):
            for r, s in zip(py.requests, rs.stamps):
                a = getattr(r, name)
                assert (a is None and s[j] is None) or a == pytest.approx(s[j], rel=1e-12)


def test_heterogeneous_pools_with_identical_devices_equal_the_homogeneous_run():
    """In both languages: naming the same device per pool changes nothing but the 'pools' key."""
    base = SimConfig()
    same = replace(base, prefill_device=H100_SXM, decode_device=H100_SXM)
    _, a = run_both(base)
    py, b = run_both(same)
    assert a.stamps == b.stamps
    assert b.summary.pop("pools") == {"prefill": "4x H100-SXM", "decode": "4x H100-SXM"}
    assert a.summary == b.summary
    assert "pools" in summarise(py)


@pytest.mark.parametrize("cfg, needle", [
    ({"model": "llama3-8b-hyena"}, "FFT-mixing"),
    ({"model": "llama3-8b-hyena-circ", "devices_per_instance": 1}, "FFT-mixing"),
    ({"prefill_device": "optical-fft"}, "transform engine"),
    ({"kv_transit": "fp8 at transit"}, "compression"),
])
def test_python_and_js_only_features_are_rejected(cfg, needle):
    with pytest.raises(ValueError, match=f"{needle}.*not in the Rust port"):
        rust_des.simulate(cfg, [(0.1, 100, 10)])


def test_python_and_js_only_features_are_rejected_from_a_simconfig():
    from disagg_sim.hardware import KV_PRESETS, MODELS, OPTICAL_FFT, KVTransit
    for cfg in (SimConfig(model=MODELS["llama3-8b-hybrid"], devices_per_instance=1),
                SimConfig(prefill_device=OPTICAL_FFT),
                SimConfig(kv_transit=KVTransit(KV_PRESETS["fp8"]))):
        with pytest.raises(ValueError, match="not in the Rust port"):
            rust_des.simulate(cfg, [(0.1, 100, 10)])


def test_errors_cross_the_boundary_as_value_errors():
    with pytest.raises(ValueError, match="below idle"):
        rust_des.simulate({"power_cap_w": 50.0}, [(0.1, 100, 10)])
    with pytest.raises(ValueError, match="unknown field"):
        rust_des.simulate({"nonsense": 1}, [(0.1, 100, 10)])
