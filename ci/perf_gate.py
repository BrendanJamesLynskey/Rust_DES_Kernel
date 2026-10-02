"""Fail the build if a criterion benchmark got slower than its stored baseline.

    cargo bench --bench engine -- --noplot
    python ci/perf_gate.py              # compare with ci/perf_baseline.json
    python ci/perf_gate.py --update     # accept the current numbers as the baseline

Timing noise on a shared machine is real, so the gate allows a margin
(default 25%) and reports every benchmark either way.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "ci" / "perf_baseline.json"
BENCHES = ["kernel/md1_100k_customers", "disagg/1P1D_1000req", "disagg/colocated_1000req"]


def current() -> dict:
    out = {}
    for b in BENCHES:
        est = ROOT / "target" / "criterion" / b / "new" / "estimates.json"
        out[b] = json.loads(est.read_text())["mean"]["point_estimate"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--report", default="perf_report.md")
    a = ap.parse_args()
    now = current()
    if a.update:
        BASELINE.write_text(json.dumps(now, indent=2) + "\n")
        print(f"baseline updated: {BASELINE}")
        return 0
    base = json.loads(BASELINE.read_text())
    lines = ["| Benchmark | Baseline | Now | Ratio | Verdict |", "|---|---|---|---|---|"]
    failed = []
    for b in BENCHES:
        r = now[b] / base[b]
        ok = r <= 1 + a.margin
        if not ok:
            failed.append(b)
        lines.append(f"| {b} | {base[b] / 1e6:.2f} ms | {now[b] / 1e6:.2f} ms | {r:.2f} | {'ok' if ok else 'REGRESSION'} |")
    report = "\n".join(lines)
    Path(a.report).write_text(report + "\n")
    print(report)
    if failed:
        print(f"performance regression (> {a.margin:.0%} slower): {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
