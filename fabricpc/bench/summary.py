"""Turn trial files into numbers people can quote.

One trial is one draw from a distribution over random seeds. A row's
result is the mean over its trials, with a standard error, and this module
refuses to produce one from fewer than two trials.

Comparing two rows is a paired test: trial i of row A and trial i of row B
used the same seed and the same data order, so the per-trial differences
are what gets tested. The math comes from ``fabricpc.experiments``, through
``fabricpc.bench.compare``.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from fabricpc.bench.ci import mean_ci95, paired_ci95
from fabricpc.bench.manifest import SCHEMA_VERSION

MIN_TRIALS = 2


def load_trials(results_dir, row_id: str) -> List[dict]:
    """All trial files for a row, sorted by trial number."""
    row_dir = Path(results_dir) / row_id
    files = sorted(row_dir.glob("trial*.json"), key=lambda p: int(p.stem[5:]))
    if not files:
        raise FileNotFoundError(f"no trial files under {row_dir}")
    return [json.loads(p.read_text()) for p in files]


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    low, high = mean_ci95(arr) if n > 1 else (None, None)
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "se": std / math.sqrt(n) if n > 1 else 0.0,
        "ci95_low": low,  # 95% t interval for the mean, see fabricpc.bench.ci
        "ci95_high": high,
        "n": n,
    }


@dataclass(frozen=True)
class RowSummary:
    row_id: str
    algorithm: str
    n_ok: int
    n_failed: int
    seeds: List[int]
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timing: Dict[str, Dict[str, float]] = field(default_factory=dict)
    compute: Dict[str, float] = field(default_factory=dict)
    n_params: int = 0
    num_epochs: float = 0.0
    step_memory: Optional[Dict[str, int]] = None  # same for every trial of a row
    # Peak device memory across trials; None when no trial recorded it.
    peak_memory_bytes: Optional[Dict[str, float]] = None
    # The process-wide peak is mostly GPU compile scratch space and comes out
    # about the same for every algorithm, so it is never a fair way to compare
    # methods (step_memory is). None when there is no peak to compare.
    peak_memory_comparable: Optional[bool] = None
    # Where each successful trial's model was saved, for the model zoo.
    checkpoints: List[Dict[str, object]] = field(default_factory=list)
    # How PC-like the runs were (see regime_summary); None for backprop rows.
    epc_regime: Optional[Dict[str, object]] = None
    backprop_alignment: Optional[Dict[str, object]] = None
    # ePC rows: how far the settle went at the end of training, 0 (backprop)
    # to 1 (the PC equilibrium); the regime's f_weighted, over seeds.
    settled_fraction: Optional[Dict[str, float]] = None
    solver: Optional[Dict[str, object]] = None  # PC rows: as the trials built it
    # Set when trials scored only part of the test set (a rehearsal).
    eval_batches_capped: Optional[int] = None
    band: Optional[Dict[str, object]] = None  # filled in by fabricpc.bench.band
    schema_version: int = SCHEMA_VERSION


def _regime_at(trials: List[dict], when: str) -> Optional[Dict[str, object]]:
    """ePC regime bands, relaxed fraction and unstable count at one point."""
    regimes = [
        (t.get("epc_regime") or {}).get(when)
        for t in trials
        if (t.get("epc_regime") or {}).get(when)
    ]
    if not regimes:
        return None
    bands: Dict[str, int] = {}
    for r in regimes:
        bands[r["band"]] = bands.get(r["band"], 0) + 1
    fractions = [r["f_weighted"] for r in regimes if r.get("f_weighted") is not None]
    return {
        "bands": bands,
        "f_weighted": _stats(fractions) if fractions else None,
        "unstable": sum(bool(r.get("unstable")) for r in regimes),
        "n": len(regimes),
    }


def _alignment_at(trials: List[dict], when: str) -> Optional[Dict[str, object]]:
    """How closely the PC weight update followed backprop's, over seeds."""
    found = [
        ((t.get("diagnostics") or {}).get(when) or {}).get("backprop_alignment")
        for t in trials
    ]
    found = [a for a in found if a]
    if not found:
        return None
    return {
        "mean_cos": _stats([a["mean_cos"] for a in found]),
        "min_cos": _stats([a["min_cos"] for a in found]),
    }


def trial_settled_fraction(trial: dict) -> Optional[float]:
    """An ePC trial's settled fraction at the end of training, else None.

    It is the regime's gradient-weighted relaxed fraction f_weighted: below
    about 0.1 the run trained like backprop, near 1 it reached the PC
    equilibrium (see ``EPCInference.regime``).
    """
    final = (trial.get("epc_regime") or {}).get("final") or {}
    return final.get("f_weighted")


def settled_fraction(trials: List[dict]) -> Optional[Dict[str, float]]:
    """Mean, SE and spread of the good trials' settled fractions; None when
    no trial has one (sPC and backprop rows)."""
    ok = [t for t in trials if t.get("status", "ok") == "ok"]
    values = [f for f in map(trial_settled_fraction, ok) if f is not None]
    return _stats(values) if values else None


def regime_summary(trials: List[dict]) -> Dict[str, object]:
    """How PC-like a row's good trials were, at the start and end of training.

    ``epc_regime`` (ePC rows) counts each band, backprop-like, partially
    relaxed or near PC equilibrium, and averages the gradient-weighted
    relaxed fraction. ``backprop_alignment`` (PC rows with diagnostics)
    averages the cosine between PC's weight update and backprop's. Either
    is None when no trial recorded it: a backprop row, or a run with
    diagnostics off.
    """
    ok = [t for t in trials if t.get("status", "ok") == "ok"]
    out: Dict[str, object] = {}
    for key, at in (("epc_regime", _regime_at), ("backprop_alignment", _alignment_at)):
        parts = {when: at(ok, when) for when in ("init", "final")}
        out[key] = parts if any(parts.values()) else None
    return out


def summarize_row(results_dir, row_id: str) -> RowSummary:
    """Average a row's trials and write ``summary.json`` next to them."""
    trials = load_trials(results_dir, row_id)
    ok = [t for t in trials if t.get("status") == "ok"]
    failed = len(trials) - len(ok)
    if len(ok) < MIN_TRIALS:
        raise ValueError(
            f"{row_id}: need at least {MIN_TRIALS} successful trials to report a "
            f"mean, found {len(ok)}. A single seed is not a result."
        )

    lengths = sorted({float(t.get("num_epochs", 0.0)) for t in ok})
    if len(lengths) > 1:
        raise ValueError(
            f"{row_id}: its trials ran different numbers of epochs {lengths}, so "
            "one mean would describe no run; rerun the short ones (--resume "
            "with the same --epochs) or use another --out"
        )
    # Only metrics every good trial has: trials written before a metric
    # existed (say forward_accuracy) lack it.
    metric_names = sorted(set.intersection(*(set(t["metrics"]) for t in ok)))
    metrics = {name: _stats([t["metrics"][name] for t in ok]) for name in metric_names}
    timing = {
        key: _stats([t[key] for t in ok])
        for key in ("step_time_ms", "compile_time_s", "train_time_s", "achieved_tflops")
    }
    peaks = [
        t["peak_memory_bytes"] for t in ok if t.get("peak_memory_bytes") is not None
    ]
    checkpoints = [
        {"trial": t["trial"], "seed": t["seed"], "checkpoint": t["checkpoint"]}
        for t in ok
        if t.get("checkpoint")
    ]

    summary = RowSummary(
        row_id=row_id,
        algorithm=ok[0]["algorithm"],
        n_ok=len(ok),
        n_failed=failed,
        seeds=[t["seed"] for t in ok],
        metrics=metrics,
        timing=timing,
        compute=ok[0].get("compute", {}),  # same for every trial of a row
        n_params=ok[0]["n_params"],
        num_epochs=float(ok[0].get("num_epochs", 0.0)),
        step_memory=ok[0].get("step_memory"),
        peak_memory_bytes=_stats(peaks) if len(peaks) >= MIN_TRIALS else None,
        peak_memory_comparable=False if len(peaks) >= MIN_TRIALS else None,
        checkpoints=checkpoints,
        **regime_summary(ok),
        settled_fraction=settled_fraction(ok),
        solver=ok[0].get("solver"),
        eval_batches_capped=max(
            (t["eval_batches_capped"] for t in ok if t.get("eval_batches_capped")),
            default=None,
        ),
    )
    write_summary(results_dir, summary)
    return summary


def write_summary(results_dir, summary: RowSummary) -> None:
    path = Path(results_dir) / summary.row_id / "summary.json"
    path.write_text(json.dumps(asdict(summary), indent=2))


@dataclass(frozen=True)
class Contrast:
    row_a: str
    row_b: str
    metric: str
    n: int
    mean_diff: float  # a - b
    se_diff: float
    t_statistic: float
    p_value: float
    significant_at_05: bool
    cohens_d: float
    ci95_low: float = math.nan  # 95% t interval for mean_diff
    ci95_high: float = math.nan
    # Holm over the family's contrasts when both rows are in one family whose
    # rows all have results (``holm_family`` names it); NaN otherwise, since
    # a single pair on its own is not corrected for anything.
    p_holm: float = math.nan
    significant_holm_05: bool = False
    effect_size: str = "d_z (paired)"  # what cohens_d is
    hedges_gz: float = math.nan  # d_z with Hedges' small-sample correction
    holm_family: Optional[str] = None
    schema_version: int = SCHEMA_VERSION


def _family_holm(results_dir, row_a, row_b, metric, p_value):
    """``(p_holm, family id)`` for a pair inside a family, else ``(nan, None)``.

    The pair's p-value is corrected together with the family's other planned
    contrasts, each worked out the same way (two rows, paired on the seeds
    both have). If any of them cannot be worked out, the p stays uncorrected.
    """
    from fabricpc.bench.compare import planned_results
    from fabricpc.bench.registry import COMPARISONS
    from fabricpc.bench.stats import holm

    pair = {row_a, row_b}
    for family in COMPARISONS.values():
        if not any(set(c) == pair for c in family.contrasts):
            continue
        ps = [p_value]
        try:
            for a, b in family.contrasts:
                if {a, b} == pair:
                    continue
                res = planned_results(
                    results_dir, rows=(a, b), contrasts=((a, b),), metric=metric
                )
                ps.append(float(res.contrast_results()[0].p_value))
        except (FileNotFoundError, ValueError):
            return math.nan, None
        return holm(ps)[0], family.id
    return math.nan, None


def compare_rows(results_dir, row_a: str, row_b: str, *, metric: str) -> Contrast:
    """Paired comparison of one metric between two rows, trial by trial.

    The statistics come from fabricpc.experiments' planned-contrast results
    (see ``fabricpc.bench.compare``), the same code every comparison uses.
    """
    from fabricpc.bench.compare import corrected, planned_results

    res = planned_results(
        results_dir, rows=(row_a, row_b), contrasts=((row_a, row_b),), metric=metric
    )
    (c,) = res.contrast_results()
    low, high = paired_ci95(res.per_arm_metrics(row_a), res.per_arm_metrics(row_b))
    contrast = Contrast(
        row_a=row_a,
        row_b=row_b,
        metric=metric,
        n=int(c.n),
        mean_diff=float(c.mean_diff),
        se_diff=float(c.se_diff),
        t_statistic=float(c.t_statistic),
        p_value=float(c.p_value),
        significant_at_05=bool(c.significant_at_05),
        cohens_d=float(c.cohens_d),
        ci95_low=low,
        ci95_high=high,
    )
    (extras,) = corrected([asdict(contrast)])
    p_holm, family = _family_holm(
        results_dir, row_a, row_b, metric, float(extras["p_value"])
    )
    extras.update(
        p_holm=p_holm, significant_holm_05=bool(p_holm < 0.05), holm_family=family
    )
    contrast = Contrast(**{k: extras[k] for k in asdict(contrast)})
    path = Path(results_dir) / f"compare-{row_a}-vs-{row_b}.json"
    path.write_text(json.dumps(asdict(contrast), indent=2))
    return contrast
