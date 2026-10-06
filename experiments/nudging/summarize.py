"""Summarize nudging runs the way pcx reports them, plus our stricter number.

pcx runs 7 seeds, keeps each seed's best test accuracy over the epochs, and
(most likely) reports the mean of the top 5 of those 7. Our benchmark
reports the last epoch over all seeds. Both are printed so nobody compares
one with the other by mistake.

    python -m experiments.nudging.summarize DIR [config ...]
"""

import json
import sys
from pathlib import Path

import numpy as np


def summarize(out_dir, name):
    runs = [
        json.loads(p.read_text())
        for p in sorted((Path(out_dir) / name).glob("seed*.json"))
    ]
    best = np.array([r["best_acc"] for r in runs])
    last = np.array([r["last_acc"] for r in runs])
    target = runs[0]["config"]["target"] if runs else None
    top5 = np.sort(best)[-5:] if len(best) >= 7 else None
    # std uses numpy's default (ddof=0), as pcx's seed5.py does.
    return {
        "n": len(runs),
        "best_mean": float(best.mean()) if len(best) else None,
        "best_std": float(best.std()) if len(best) else None,
        "top5_mean": float(top5.mean()) if top5 is not None else None,
        "top5_std": float(top5.std()) if top5 is not None else None,
        "last_mean": float(last.mean()) if len(last) else None,
        "last_std": float(last.std()) if len(last) else None,
        "target": target,
        "gap_top5": float(top5.mean() * 100 - target) if top5 is not None else None,
    }


def _pct(x):
    return "   -  " if x is None else f"{100 * x:6.2f}"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out_dir = Path(argv[0])
    names = argv[1:] or sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    print("config    n  best-epoch mean  top5-of-7 (pcx)  last-epoch mean  pcx target")
    for name in names:
        s = summarize(out_dir, name)
        print(
            f"{name:8s} {s['n']:2d}  {_pct(s['best_mean'])}          {_pct(s['top5_mean'])}"
            f"           {_pct(s['last_mean'])}           {s['target']}"
        )


if __name__ == "__main__":
    main()
