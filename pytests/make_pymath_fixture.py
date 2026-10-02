"""Regenerate tests/fixtures/pymath.json: reference answers from CPython itself.

Mutation testing showed that the unit tests for fsum and floor division left
whole branches unchecked (fsum's final half-even correction; floordiv's sign
fix-ups). This fixture pins both against CPython on cases built to reach
those branches.

    python pytests/make_pymath_fixture.py
"""

import json
import math
import random
from pathlib import Path


def fsum_cases(rng):
    cases = [[1.0, 1e100, 1.0, -1e100], [0.1] * 10, [], [5e-324, 5e-324]]
    for _ in range(400):
        # Half-way cases: a value, half an ulp of it, and tiny same-sign or opposite-sign tails
        # exercise the correction step that rounds the exact sum half to even.
        x = rng.uniform(-1, 1) * 2.0 ** rng.randint(-20, 20)
        half = math.ulp(x) / 2
        tail = rng.choice([1, -1]) * half * 2.0 ** -rng.randint(1, 60)
        cases.append(rng.sample([x, half, tail, -x / 3, x / 3], 5))
    for _ in range(400):
        n = rng.randint(1, 40)
        cases.append([rng.choice([-1, 1]) * rng.uniform(0, 1) * 10.0 ** rng.randint(-30, 30) for _ in range(n)])
    return [{"xs": xs, "sum": math.fsum(xs)} for xs in cases]


def floordiv_cases(rng):
    pairs = [(7.0, 2.0), (-7.0, 2.0), (7.0, -2.0), (-7.0, -2.0), (0.3, 0.1), (-0.0, 3.0), (0.0, -3.0),
             (1e300, 1e-3), (5.0, 5.0), (-5.0, 5.0), (1.0, 3.0), (-1.0, 3.0)]
    for _ in range(600):
        a = rng.choice([-1, 1]) * rng.uniform(0, 1) * 10.0 ** rng.randint(-5, 15)
        b = rng.choice([-1, 1]) * rng.uniform(0.001, 1) * 10.0 ** rng.randint(-3, 8)
        pairs.append((a, b))
    return [{"a": a, "b": b, "q": a // b} for a, b in pairs]


def main():
    rng = random.Random(42)
    data = {"fsum": fsum_cases(rng), "floordiv": floordiv_cases(rng)}
    dest = Path(__file__).parent.parent / "tests" / "fixtures" / "pymath.json"
    dest.write_text(json.dumps(data))
    print(f"wrote {dest}: {len(data['fsum'])} fsum cases, {len(data['floordiv'])} floordiv cases")


if __name__ == "__main__":
    main()
