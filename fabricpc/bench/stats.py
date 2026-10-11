"""Two small corrections for reporting a family's contrasts honestly.

Holm's correction. A family tests three contrasts (sPC - backprop,
ePC - backprop, ePC - sPC). Each at 5% gives about a 14% chance that at
least one comes out "significant" by luck alone. Holm's step-down method
(Holm 1979) keeps that chance at 5%: sort the p-values, multiply the
smallest by 3, the next by 2, the last by 1, and never let an adjusted
value drop below the one before it.

Hedges' correction. The paired effect size the framework reports is
d_z = mean difference / SD of the differences. With a handful of seeds it
runs large; multiplying by J(df) = 1 - 3 / (4 df - 1), df = n - 1, removes
most of that bias (Hedges 1981; Lakens 2013 for the paired case).
"""

import math
from typing import List, Sequence


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm-adjusted p-values, in the order given. NaN stays NaN but still
    counts as a test (one that found nothing, p = 1): it was planned, and
    leaving it out would correct the others too little."""
    real = sorted((p, i) for i, p in enumerate(p_values) if not math.isnan(float(p)))
    adjusted = [float("nan")] * len(p_values)
    m, running = len(p_values), 0.0
    for rank, (p, i) in enumerate(real):
        running = max(running, min(1.0, (m - rank) * float(p)))
        adjusted[i] = running
    return adjusted


def hedges_gz(d_z: float, *, n: int) -> float:
    """d_z times Hedges' small-sample factor J(n - 1); NaN below 3 pairs.

    With two pairs J(1) = 0, so every effect would print as 0.
    """
    df = n - 1
    if df < 2:
        return float("nan")
    return float(d_z) * (1.0 - 3.0 / (4.0 * df - 1.0))
