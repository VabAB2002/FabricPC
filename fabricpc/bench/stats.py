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

Equivalence (TOST). "Not significantly different" is no evidence that two
methods match: a small or noisy experiment finds no difference either. Two
one-sided t-tests on the per-seed differences (Schuirmann 1987; Lakens
2017) ask whether the mean difference is above -margin and below +margin;
the larger of the two p-values is the test's. It is below 0.05 exactly
when the 90% t interval of the difference lies inside the margin.
"""

import math
from typing import List, Optional, Sequence

import numpy as np


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


# The equivalence margin used when none is given: half a percentage point
# of accuracy, the same as the pass/fail band's floor. Other metrics have
# no natural unit, so they get none unless one is given (--margin).
ACCURACY_MARGIN = 0.005


def default_margin(metric: str) -> Optional[float]:
    """The margin a metric is tested for equivalence on, or None."""
    return ACCURACY_MARGIN if metric.endswith("accuracy") else None


def tost_paired(diffs: Sequence[float], *, margin: float) -> float:
    """p-value of the paired equivalence test: is the mean of ``diffs``
    inside (-margin, +margin)? Below 0.05 means equivalent at 5%."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2 or not margin > 0 or not np.isfinite(d).all():
        return float("nan")
    mean = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(n)
    if se == 0.0:
        return 0.0 if abs(mean) < margin else 1.0
    from scipy import stats  # optional; only when a test is computed

    p_above = float(stats.t.sf((mean + margin) / se, n - 1))  # mean > -margin
    p_below = float(stats.t.cdf((mean - margin) / se, n - 1))  # mean < +margin
    return max(p_above, p_below)
