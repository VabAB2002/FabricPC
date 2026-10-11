"""Issue #73: does accuracy follow f_max or f-bar as ePC's rate grows?

    python scripts/regime_band_study.py --out regime-band-study
    python scripts/regime_band_study.py --row mnist-mlp-epc --etas 0.001 0.1 1 \
        --seeds 3 --epochs 2 --steps 5 --out regime-band-study

``EPCInference.regime`` labels a run on f-bar, the gradient-weighted relaxed
fraction; ``Regime`` also carries f_max, the relaxed fraction of the top
mode. On the ResNet-18 demo (one seed, one probe batch) the accuracy sweep
followed f_max. The issue asks for more seeds, a second probe batch, and a
second graph family. This script sweeps the ePC rate on a benchmark row with
the safety net off (so the rate stays what was asked), and for every
(rate, seed) records:

- f-bar and f_max at init on two different probe batches (the first and
  the second training batch), and their spread,
- f-bar and f_max after training (on the first batch),
- the test accuracy after training.

It writes ``study.csv`` and a summary table (means over seeds per rate) to
``--out``, and prints the table. Each trial goes through the benchmark's own
``run_trial``, with diagnostics and the learning curve off.
"""

import argparse
import csv
import dataclasses
import statistics
from pathlib import Path

import jax

from fabricpc.bench.measure import epc_regime
from fabricpc.bench.registry import ROWS
from fabricpc.bench.runner import _TIMING_STREAM, run_trial, seed_for_trial
from fabricpc.core.inference_epc import EPCInference


def row_at(row, eta: float, steps: int):
    """The row with an ePC solver at (eta, steps) and no safety net."""

    def build(key):
        params, structure = row.model_factory(key)
        inference = EPCInference(eta_infer=eta, infer_steps=steps)
        return params, structure._replace(
            config={**structure.config, "inference": inference}
        )

    return dataclasses.replace(
        row, id=f"{row.id}-eta{eta:g}", model_factory=build, rate_control=None
    )


def second_batch_regime(row, trial: int):
    """f-bar and f_max at init on the second training batch of the trial,
    with the same key the trial's own regime reading uses, so only the batch
    differs."""
    seed = seed_for_trial(trial)
    master = jax.random.PRNGKey(seed)
    graph_key = jax.random.split(master, 3)[0]
    params, structure = row.model_factory(graph_key)
    train, _ = row.loader_factory(seed)
    batches = iter(train)
    next(batches)
    regime = epc_regime(
        params, structure, next(batches), jax.random.fold_in(master, _TIMING_STREAM)
    )
    return regime["f_weighted"], regime["f_max"]


def study(row_id, etas, seeds, epochs, steps, out):
    base = ROWS[row_id]
    out = Path(out)
    rows = []
    for eta in etas:
        row = row_at(base, eta, steps)
        for trial in range(seeds):
            r = run_trial(
                row,
                trial,
                out / "trials",
                num_epochs=epochs,
                warmup_steps=1,
                timed_steps=2,
                curve_batches=0,
                diagnostics=False,
            )
            if r.status != "ok":
                print(f"eta {eta:g} trial {trial}: failed")
                continue
            f2, fmax2 = second_batch_regime(row, trial)
            init, final = r.epc_regime["init"], r.epc_regime["final"]
            rows.append(
                {
                    "eta": eta,
                    "steps": steps,
                    "seed": r.seed,
                    "accuracy": r.metrics["accuracy"],
                    "f_bar_init": init["f_weighted"],
                    "f_max_init": init["f_max"],
                    "f_bar_init_batch2": f2,
                    "f_max_init_batch2": fmax2,
                    "f_bar_final": final["f_weighted"],
                    "f_max_final": final["f_max"],
                    "eta_lambda_max_init": init["eta_lambda_max"],
                    "eta_lambda_max_final": final["eta_lambda_max"],
                    "band_init": init["band"],
                    "unstable_final": final["unstable"],
                }
            )
            print(
                f"eta {eta:g} seed {r.seed}: acc {r.metrics['accuracy']:.4f} "
                f"f_bar {init['f_weighted']:.3f}/{f2:.3f} "
                f"f_max {init['f_max']:.3f}/{fmax2:.3f}"
            )
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("every trial failed; nothing to summarize")
    with (out / "study.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = summary(rows)
    (out / "summary.md").write_text(table)
    print(table)
    return rows


def summary(rows) -> str:
    """Per rate, over seeds: medians (f above 1 is overshoot and grows without
    bound once eta lambda_max passes 2, so a mean of it says nothing), each
    seed's final eta lambda_max, and how many seeds ended unstable."""
    lines = [
        "| eta | n | accuracy (median) | eta lambda_max init | eta lambda_max end "
        "(per seed) | unstable at end | f_bar init (batch 1 / 2) | f_max init "
        "(batch 1 / 2) | f_bar end | f_max end |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for eta in sorted({r["eta"] for r in rows}):
        rs = sorted((r for r in rows if r["eta"] == eta), key=lambda r: r["seed"])

        def med(key):
            return statistics.median(r[key] for r in rs)

        ends = ", ".join(f"{r['eta_lambda_max_final']:.3g}" for r in rs)
        lines.append(
            f"| {eta:g} | {len(rs)} | {med('accuracy'):.4f} | "
            f"{med('eta_lambda_max_init'):.3g} | {ends} | "
            f"{sum(bool(r['unstable_final']) for r in rs)} of {len(rs)} | "
            f"{med('f_bar_init'):.3f} / {med('f_bar_init_batch2'):.3f} | "
            f"{med('f_max_init'):.3f} / {med('f_max_init_batch2'):.3f} | "
            f"{med('f_bar_final'):.3g} | {med('f_max_final'):.3g} |"
        )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--row", default="mnist-mlp-epc")
    p.add_argument(
        "--etas", type=float, nargs="+", default=[0.001, 0.01, 0.03, 0.1, 0.3, 1.0]
    )
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    study(args.row, args.etas, args.seeds, args.epochs, args.steps, args.out)


if __name__ == "__main__":
    main()
