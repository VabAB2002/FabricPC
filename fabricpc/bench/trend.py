"""Trend tracking: is tonight's run worse than the nights before? (issue #54)

    python -m fabricpc.bench trend add <results> --history DIR
    python -m fabricpc.bench trend check --history DIR

Every run is kept, one line per row in ``DIR/<row>.jsonl`` (the per-seed step
times and scores, failed seeds, the git sha, the date, the device, the XLA
flags, the epochs, the caps and the machine's speed test). A check judges
the newest night of each row:

- One number per night: the median over its seeds. The seeds of one night
  share that night's machine, so pooling them as separate samples (what this
  module first did) makes ordinary runner noise look significant: in a
  simulation of nine rows at 10% runner noise, every week raised an alarm.
- Step times are divided by the night's speed test (a fixed matmul timed when
  the run started, ``manifest.speed_test_ms``), so a night when the whole
  runner was slow is not a slower FabricPC. Nights without one are left out
  of a speed-adjusted comparison.
- Nights are compared only within one device, XLA flag set, metric, epoch
  count and eval/train cap. A change of any of them starts a new baseline.
- The baseline is the last ``WINDOW`` nights that raised no alarm (at least
  ``MIN_BASELINE``). Nights in an alarm never join it, so a real slowdown
  does not become the new normal.
- An alarm needs ``IN_A_ROW`` nights in a row that are each worse than the
  baseline's median by a practical amount (15% for step time, half a point
  of accuracy, 5% for a lower-is-better score) and by more than
  ``NOISE_MULTIPLE`` times the baseline's own wobble (a robust spread, from
  the median absolute deviation).
- A failed seed or a NaN score alarms at once.

#54 suggests a Welch test. In the same simulation (nine rows, 10% runner
noise, no speed test), a Welch test of two night medians against ten raised
a false alarm in 53% of weeks, the rule above in 32%, so the rule is used
and ``welch_ttest`` stays in ``fabricpc.experiments.statistics``.
How often noise still alarms depends on how much of it the speed test sees;
guide 18 gives the simulated numbers.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

WINDOW = 10
MIN_BASELINE = 5
IN_A_ROW = 2
NOISE_MULTIPLE = 2.0
MIN_SLOWDOWN = 0.15  # relative
MIN_SCORE_DROP = 0.005  # absolute, for higher-is-better scores
MIN_SCORE_WORSE = 0.05  # relative, for lower-is-better scores
LOWER_IS_BETTER = {"reconstruction_mse", "perplexity", "cross_entropy", "forgetting"}
_MAD_TO_SD = 1.4826  # median absolute deviation -> standard deviation (normal)
_GROUP_KEYS = (
    "device",
    "xla_flags",
    "metric",
    "num_epochs",
    "eval_batches_capped",
    "train_steps_capped",
)


@dataclass(frozen=True)
class Alert:
    row_id: str
    what: str  # "step_time_ms", the row's metric, "failed_seeds" or "non_finite"
    baseline: float  # the baseline nights' median (nan for an at-once alarm)
    recent: Tuple[float, ...]  # the nights that tripped it
    change: float  # relative for step time and lower-is-better, else absolute
    threshold: float  # the change that was needed, same units as ``change``
    git_sha: Optional[str]
    speed_adjusted: bool = False
    detail: str = ""

    def line(self) -> str:
        sha = (self.git_sha or "unknown")[:10]
        if self.what in ("failed_seeds", "non_finite"):
            return f"{self.row_id}: {self.what}: {self.detail} at {sha}"
        relative = self.what == "step_time_ms" or self.what in LOWER_IS_BETTER
        size = (
            f"{self.change:+.0%}, alarm at {self.threshold:.0%}"
            if relative
            else f"{self.change:+.4f}, alarm at {self.threshold:.4f}"
        )
        nights = ", ".join(f"{v:.4g}" for v in self.recent)
        adjusted = ", per speed test" if self.speed_adjusted else ""
        return (
            f"{self.row_id}: {self.what} {self.baseline:.4g} -> {nights} "
            f"({size}{adjusted}) over {len(self.recent)} nights at {sha}"
        )


@dataclass(frozen=True)
class _Verdict:
    baseline: float
    recent: Tuple[float, ...]
    change: float
    threshold: float


def _judge(
    values: Sequence[float],
    *,
    higher_is_worse: bool,
    relative: bool,
    min_change: float,
    window: int,
    min_baseline: int,
    in_a_row: int,
    noise_multiple: float,
) -> List[Optional[_Verdict]]:
    """A verdict for every night, oldest first: None, or why it alarms."""
    raw = np.asarray(values, dtype=float)
    ok = np.isfinite(raw) & ((raw > 0) if relative else True)
    x = np.full(raw.shape, np.nan)
    x[ok] = np.log(raw[ok]) if relative else raw[ok]
    if not higher_is_worse:
        x = -x
    min_excess = math.log1p(min_change) if relative else min_change

    verdicts: List[Optional[_Verdict]] = []
    in_alarm = np.zeros(len(x), dtype=bool)  # never joins a baseline
    for i in range(len(x)):
        first = i - in_a_row + 1
        verdict = None
        if first >= 0 and ok[first : i + 1].all():
            base_idx = [j for j in range(first) if ok[j] and not in_alarm[j]]
            base_idx = base_idx[-window:]
            if len(base_idx) >= min_baseline:
                base = x[base_idx]
                center = float(np.median(base))
                spread = _MAD_TO_SD * float(np.median(np.abs(base - center)))
                needed = max(min_excess, noise_multiple * spread)
                excess = x[first : i + 1] - center
                if bool(np.all(excess >= needed)):
                    base_raw = float(np.median(raw[base_idx]))
                    recent = tuple(float(v) for v in raw[first : i + 1])
                    mid = float(np.median(recent))
                    if relative:
                        change, shown = mid / base_raw - 1, math.expm1(needed)
                    else:
                        change, shown = mid - base_raw, needed
                    verdict = _Verdict(base_raw, recent, change, shown)
                    in_alarm[first : i + 1] = True
        verdicts.append(verdict)
    return verdicts


def flag_nights(
    values: Sequence[float],
    *,
    higher_is_worse: bool,
    relative: bool,
    min_change: Optional[float] = None,
    window: int = WINDOW,
    min_baseline: int = MIN_BASELINE,
    in_a_row: int = IN_A_ROW,
    noise_multiple: float = NOISE_MULTIPLE,
) -> List[bool]:
    """For each night (one number each, oldest first): does it raise an alarm?

    ``relative`` judges ratios (step time, MSE, perplexity) instead of
    differences (accuracy). A NaN night never alarms here and never joins a
    baseline. ``min_change`` defaults to the step-time slowdown for relative
    values and the accuracy drop for absolute ones.
    """
    if min_change is None:
        min_change = MIN_SLOWDOWN if relative else MIN_SCORE_DROP
    verdicts = _judge(
        values,
        higher_is_worse=higher_is_worse,
        relative=relative,
        min_change=min_change,
        window=window,
        min_baseline=min_baseline,
        in_a_row=in_a_row,
        noise_multiple=noise_multiple,
    )
    return [v is not None for v in verdicts]


def append_record(history, record: dict) -> bool:
    """Add one run of one row; False if that run is already there."""
    path = Path(history) / f"{record['row_id']}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    key = (record.get("git_sha"), record.get("created_utc"))
    if path.exists():
        for line in path.read_text().splitlines():
            old = json.loads(line)
            if (old.get("git_sha"), old.get("created_utc")) == key:
                return False
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return True


def read_history(history) -> Dict[str, List[dict]]:
    """Every row's runs, oldest first."""
    out = {}
    for path in sorted(Path(history).glob("*.jsonl")):
        records = [json.loads(x) for x in path.read_text().splitlines() if x]
        out[path.stem] = sorted(records, key=lambda r: r.get("created_utc") or "")
    return out


def add_results(results_dir, history) -> List[str]:
    """Record every row of a results folder; returns the rows added."""
    from fabricpc.bench.registry import ROWS

    added = []
    for row_dir in sorted(Path(results_dir).iterdir()):
        manifest_path = row_dir / "manifest.json"
        if not row_dir.is_dir() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        trials = [
            json.loads(p.read_text())
            for p in sorted(row_dir.glob("trial*.json"))
            if p.stem[5:].isdigit()
        ]
        if not trials:
            continue
        ok = [t for t in trials if t.get("status") == "ok"]
        row = ROWS.get(row_dir.name)
        metric = row.metric if row is not None else "accuracy"
        if any(metric not in t.get("metrics", {}) for t in ok):
            continue
        first = trials[0]
        record = {
            "row_id": row_dir.name,
            "metric": metric,
            "created_utc": manifest.get("created_utc"),
            "git_sha": manifest.get("git_sha"),
            "device": ", ".join(manifest.get("devices") or []),
            "xla_flags": manifest.get("xla_flags", ""),
            "speed_test_ms": manifest.get("speed_test_ms"),
            "num_epochs": first.get("num_epochs"),
            "eval_batches_capped": first.get("eval_batches_capped"),
            "train_steps_capped": first.get("train_steps_capped"),
            "n_failed": len(trials) - len(ok),
            "step_time_ms": [float(t["step_time_ms"]) for t in ok],
            "values": [float(t["metrics"][metric]) for t in ok],
        }
        if append_record(history, record):
            added.append(row_dir.name)
    return added


def _median(xs) -> float:
    xs = np.asarray(xs, dtype=float)
    return float(np.median(xs)) if xs.size else float("nan")


def _step_series(group: List[dict]) -> Tuple[List[float], bool]:
    """One step time per night; per speed test when tonight has one."""

    def speed(r):
        s = r.get("speed_test_ms")
        return float(s) if isinstance(s, (int, float)) and s > 0 else None

    if speed(group[-1]) is None:
        return [_median(r["step_time_ms"]) for r in group], False
    return [
        _median(r["step_time_ms"]) / speed(r) if speed(r) else float("nan")
        for r in group
    ], True


def _at_once_alerts(row_id: str, newest: dict) -> List[Alert]:
    sha, alerts = newest.get("git_sha"), []
    n_ok = len(newest.get("values") or [])
    failed = int(newest.get("n_failed") or 0)
    if failed:
        alerts.append(
            Alert(
                row_id,
                "failed_seeds",
                float("nan"),
                (),
                float(failed),
                0.0,
                sha,
                detail=f"{failed} of {failed + n_ok} seeds failed",
            )
        )
    numbers = list(newest.get("values") or []) + list(newest.get("step_time_ms") or [])
    bad = sum(not math.isfinite(float(v)) for v in numbers)
    if bad:
        alerts.append(
            Alert(
                row_id,
                "non_finite",
                float("nan"),
                (),
                float(bad),
                0.0,
                sha,
                detail=f"{bad} NaN or infinite numbers in tonight's scores or step times",
            )
        )
    return alerts


def check(
    history,
    *,
    window: int = WINDOW,
    min_baseline: int = MIN_BASELINE,
    in_a_row: int = IN_A_ROW,
    noise_multiple: float = NOISE_MULTIPLE,
    min_slowdown: float = MIN_SLOWDOWN,
    min_score_drop: float = MIN_SCORE_DROP,
    min_score_worse: float = MIN_SCORE_WORSE,
) -> List[Alert]:
    """Alerts for every row whose newest night is worse than its baseline."""
    common = dict(
        window=window,
        min_baseline=min_baseline,
        in_a_row=in_a_row,
        noise_multiple=noise_multiple,
    )
    alerts = []
    for row_id, records in read_history(history).items():
        newest = records[-1]
        key = tuple(newest.get(k) for k in _GROUP_KEYS)
        group = [r for r in records if tuple(r.get(k) for k in _GROUP_KEYS) == key]
        sha = newest.get("git_sha")
        alerts += _at_once_alerts(row_id, newest)

        steps, adjusted = _step_series(group)
        verdict = _judge(
            steps,
            higher_is_worse=True,
            relative=True,
            min_change=min_slowdown,
            **common,
        )[-1]
        if verdict is not None:
            alerts.append(_alert(row_id, "step_time_ms", verdict, sha, adjusted))

        metric = newest["metric"]
        lower = metric in LOWER_IS_BETTER
        scores = [_median(r["values"]) for r in group]
        verdict = _judge(
            scores,
            higher_is_worse=lower,
            relative=lower,
            min_change=min_score_worse if lower else min_score_drop,
            **common,
        )[-1]
        if verdict is not None:
            alerts.append(_alert(row_id, metric, verdict, sha, False))
    return alerts


def _alert(row_id, what, verdict: _Verdict, sha, adjusted) -> Alert:
    return Alert(
        row_id,
        what,
        verdict.baseline,
        verdict.recent,
        verdict.change,
        verdict.threshold,
        sha,
        speed_adjusted=adjusted,
    )
