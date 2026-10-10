"""Trend tracking: is tonight's run worse than the last few? (issue #54)

    python -m fabricpc.bench trend add <results> --history DIR
    python -m fabricpc.bench trend check --history DIR

Timing on a shared runner wobbles by 5 to 15% from run to run, so "slower
than last night" alerts all the time. Instead every run is kept, one line
per row in ``DIR/<row>.jsonl`` (the per-seed step times and scores, the git
sha, the date, the device and the XLA flags), and a check compares the
newest run of each row with the runs before it:

- Runs are grouped by device and XLA flags. A new device or flag set starts
  a new baseline; it is not a regression.
- The baseline is the previous ``window`` runs of the group, their seeds
  pooled. Tonight's seeds and the baseline's are unpaired and of different
  sizes, so the test is Welch's (``fabricpc.experiments.welch_ttest``).
- An alert needs both: the test says the difference is real (p < alpha),
  and it is big enough to matter: the step time slower by at least
  ``min_slowdown`` (15%), or the score worse by at least ``min_score_drop``
  (half a point of accuracy, or that much MSE or perplexity the wrong way).

A row's solver settings are fixed by its row id; change them under a new id
(or start a new history) so old and new runs are never mixed.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from fabricpc.experiments.statistics import welch_ttest

WINDOW = 5
ALPHA = 0.05
MIN_SLOWDOWN = 0.15
MIN_SCORE_DROP = 0.005
LOWER_IS_BETTER = {"reconstruction_mse", "perplexity", "cross_entropy", "forgetting"}


@dataclass(frozen=True)
class Alert:
    row_id: str
    what: str  # "step_time_ms" or the row's metric
    baseline_mean: float
    new_mean: float
    change: float  # relative for step time, absolute for a score
    p_value: float
    git_sha: Optional[str]

    def line(self) -> str:
        if self.what == "step_time_ms":
            size = f"{self.change:+.0%}"
        else:
            size = f"{self.change:+.4f}"
        return (
            f"{self.row_id}: {self.what} {self.baseline_mean:.4g} -> "
            f"{self.new_mean:.4g} ({size}, Welch p={self.p_value:.3g}) "
            f"at {(self.git_sha or 'unknown')[:10]}"
        )


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
        ok = [t for t in trials if t.get("status") == "ok"]
        row = ROWS.get(row_dir.name)
        metric = row.metric if row is not None else "accuracy"
        if not ok or any(metric not in t.get("metrics", {}) for t in ok):
            continue
        record = {
            "row_id": row_dir.name,
            "metric": metric,
            "created_utc": manifest.get("created_utc"),
            "git_sha": manifest.get("git_sha"),
            "device": ", ".join(manifest.get("devices") or []),
            "xla_flags": manifest.get("xla_flags", ""),
            "step_time_ms": [float(t["step_time_ms"]) for t in ok],
            "values": [float(t["metrics"][metric]) for t in ok],
        }
        if append_record(history, record):
            added.append(row_dir.name)
    return added


def _compare(row_id, what, new, base, sha, *, worse_sign, threshold, relative):
    if len(new) < 2 or len(base) < 2:
        return None
    test = welch_ttest(np.asarray(new), np.asarray(base))
    new_mean, base_mean = float(np.mean(new)), float(np.mean(base))
    change = (new_mean - base_mean) / base_mean if relative else new_mean - base_mean
    if test.p_value < ALPHA and worse_sign * change >= threshold:
        return Alert(row_id, what, base_mean, new_mean, change, test.p_value, sha)
    return None


def check(
    history,
    *,
    window: int = WINDOW,
    min_slowdown: float = MIN_SLOWDOWN,
    min_score_drop: float = MIN_SCORE_DROP,
) -> List[Alert]:
    """Alerts for every row whose newest run is worse than its baseline."""
    alerts = []
    for row_id, records in read_history(history).items():
        groups: Dict[tuple, List[dict]] = {}
        for r in records:
            groups.setdefault((r.get("device"), r.get("xla_flags")), []).append(r)
        newest = records[-1]
        group = groups[(newest.get("device"), newest.get("xla_flags"))]
        baseline = group[:-1][-window:]
        if not baseline:
            continue  # the first run on this device: it is the baseline
        pooled = lambda key: [v for r in baseline for v in r[key]]  # noqa: E731
        sha = newest.get("git_sha")
        found = _compare(
            row_id,
            "step_time_ms",
            newest["step_time_ms"],
            pooled("step_time_ms"),
            sha,
            worse_sign=1,
            threshold=min_slowdown,
            relative=True,
        )
        metric = newest["metric"]
        score = _compare(
            row_id,
            metric,
            newest["values"],
            pooled("values"),
            sha,
            worse_sign=1 if metric in LOWER_IS_BETTER else -1,
            threshold=min_score_drop,
            relative=False,
        )
        alerts += [a for a in (found, score) if a is not None]
    return alerts
