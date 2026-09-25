"""Command line for the benchmark suite.

    python -m fabricpc.bench list
    python -m fabricpc.bench <row-id> [--trials N] [--epochs E] [--out DIR]
                                      [--resume] [--zoo DIR]
    python -m fabricpc.bench <row-id> --dry-run
    python -m fabricpc.bench compare <row-a> <row-b> [--metric accuracy]
    python -m fabricpc.bench smoke [--floor 0.85]
    python -m fabricpc.bench validate [DIR]

Running a row writes one JSON file per trial, a manifest, ``trials.csv``
(every trial on one line), and (with two or more good trials) a summary
with the mean and standard error. ``validate`` checks that a results
folder is complete and consistent. ``smoke`` is
the short run CI uses: a small row, two seeds, half an epoch, and a check
that accuracy is still above a floor.

A full run (the row's own epochs and seed count) is also checked against
the row's expected score, when it has one, and prints PASS or FAIL. A FAIL
makes the command exit non-zero.

``--resume`` skips trials that already finished with the same epoch count,
so a run cut off by a cloud session limit can pick up where it stopped.
``--zoo DIR`` saves each trial's trained weights there.

Each trial runs in its own Python process (see ``fabricpc.bench.isolate``).
``--in-process`` runs them all in this one instead, which is handy for
debugging; ``--trial i`` runs a single trial and is what each child does.
"""

import argparse
import json
import sys

from fabricpc.bench.band import attach_band
from fabricpc.bench.isolate import run_trial_in_child
from fabricpc.bench.manifest import describe_row, write_manifest
from fabricpc.bench.registry import ROWS
from fabricpc.bench.runner import finished_trial, run_trial, seed_for_trial
from fabricpc.bench.summary import MIN_TRIALS, compare_rows, summarize_row
from fabricpc.bench.writer import validate, write_trials_csv

SMOKE_ROW = "mnist-mlp-spc"
SMOKE_FLOOR = 0.85  # half an epoch of MNIST lands well above this


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fabricpc.bench")
    p.add_argument(
        "target",
        help="'list', 'compare', 'smoke', 'validate', or a row id like mnist-mlp-spc",
    )
    p.add_argument(
        "rows",
        nargs="*",
        help="for 'compare': the two row ids; for 'validate': the results folder",
    )
    p.add_argument("--dry-run", action="store_true", help="show the row, do not train")
    p.add_argument("--out", default="results", help="where result files go")
    p.add_argument("--trials", type=int, help="how many seeds to run")
    p.add_argument("--epochs", type=float, help="override the row's epoch count")
    p.add_argument("--warmup", type=int, default=5, help="untimed steps")
    p.add_argument("--timed", type=int, default=30, help="timed steps")
    p.add_argument("--metric", default="accuracy", help="metric for 'compare'")
    p.add_argument("--row", default=SMOKE_ROW, help="row for 'smoke'")
    p.add_argument("--floor", type=float, default=SMOKE_FLOOR, help="min accuracy")
    p.add_argument(
        "--resume", action="store_true", help="skip trials that already finished"
    )
    p.add_argument("--zoo", help="save each trial's trained weights in this folder")
    p.add_argument("--trial", type=int, help="run only this one trial")
    p.add_argument(
        "--in-process",
        action="store_true",
        help="run trials in this process instead of one process each",
    )
    return p


def _unknown_row(row_id: str) -> int:
    print(
        f"Unknown row '{row_id}'. Run 'python -m fabricpc.bench list' "
        "to see the available rows.",
        file=sys.stderr,
    )
    return 2


def _run_row(
    row,
    out,
    *,
    n_trials,
    epochs,
    warmup,
    timed,
    command,
    resume=False,
    zoo=None,
    in_process=False,
):
    """Run every trial of one row, then summarize. Returns (failed, summary)."""
    run_one = run_trial if in_process else run_trial_in_child
    write_manifest(
        out,
        row,
        seeds=[seed_for_trial(t) for t in range(n_trials)],
        command=command,
    )
    wanted_epochs = epochs if epochs is not None else row.train_config["num_epochs"]
    failed = 0
    for trial in range(n_trials):
        if resume and finished_trial(out, row.id, trial, wanted_epochs):
            print(f"{row.id} trial {trial}: already done, skipping")
            continue
        result = run_one(
            row,
            trial,
            out,
            num_epochs=epochs,
            warmup_steps=warmup,
            timed_steps=timed,
            zoo_dir=zoo,
        )
        if result.status == "ok":
            acc = result.metrics.get("accuracy")
            shown = f"accuracy={acc:.4f}" if acc is not None else ""
            print(
                f"{row.id} trial {trial}: {shown} step={result.step_time_ms:.2f}ms "
                f"train={result.train_time_s:.1f}s"
            )
        else:
            failed += 1
            print(f"{row.id} trial {trial}: FAILED (see {out})", file=sys.stderr)

    write_trials_csv(out, row.id)
    summary = None
    if n_trials - failed >= MIN_TRIALS:
        summary = attach_band(out, row, summarize_row(out, row.id))
        acc = summary.metrics.get("accuracy")
        if acc:
            print(
                f"{row.id}: accuracy {acc['mean']:.4f} ± {acc['se']:.4f} "
                f"(n={acc['n']}), step {summary.timing['step_time_ms']['mean']:.2f}ms"
            )
        band = summary.band
        label = {"pass": "PASS", "fail": "FAIL"}.get(band["status"], "no verdict")
        pending = (
            " (band rule pending sponsor sign-off)" if band["pending_signoff"] else ""
        )
        print(f"{row.id}: band {label}: {band['reason']}{pending}")
    return failed, summary


def cmd_list() -> int:
    for row_id in ROWS:
        print(row_id)
    return 0


def cmd_compare(args) -> int:
    if len(args.rows) != 2:
        print("usage: compare <row-a> <row-b> [--metric NAME]", file=sys.stderr)
        return 2
    row_a, row_b = args.rows
    for row_id in (row_a, row_b):
        if row_id not in ROWS:
            return _unknown_row(row_id)
    c = compare_rows(args.out, row_a, row_b, metric=args.metric)
    sig = "yes" if c.significant_at_05 else "no"
    print(
        f"{row_a} - {row_b} on {args.metric}: {c.mean_diff:+.4f} "
        f"(se {c.se_diff:.4f}, p={c.p_value:.3g}, d={c.cohens_d:.2f}, "
        f"n={c.n}, significant at 5%: {sig})"
    )
    return 0


def cmd_validate(args) -> int:
    folder = args.rows[0] if args.rows else args.out
    problems = validate(folder)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"validate: {len(problems)} problem(s) in {folder}", file=sys.stderr)
        return 1
    print(f"validate: {folder} is complete and consistent")
    return 0


def cmd_smoke(args, command) -> int:
    """A short real run that proves the whole pipeline still works."""
    row = ROWS.get(args.row)
    if row is None:
        return _unknown_row(args.row)
    failed, summary = _run_row(
        row,
        args.out,
        n_trials=2,
        epochs=args.epochs if args.epochs is not None else 0.5,
        warmup=2,
        timed=5,
        command=command,
        in_process=args.in_process,
    )
    if failed or summary is None:
        print("smoke: a trial failed", file=sys.stderr)
        return 1
    mean = summary.metrics["accuracy"]["mean"]
    if mean < args.floor:
        print(
            f"smoke: accuracy {mean:.4f} is below the floor {args.floor}",
            file=sys.stderr,
        )
        return 1
    print(f"smoke: ok (accuracy {mean:.4f} >= {args.floor})")
    return 0


def cmd_one_trial(row, args) -> int:
    """Run a single trial here and write its file. This is what a child runs."""
    result = run_trial(
        row,
        args.trial,
        args.out,
        num_epochs=args.epochs,
        warmup_steps=args.warmup,
        timed_steps=args.timed,
        zoo_dir=args.zoo,
    )
    return 0 if result.status == "ok" else 1


def cmd_run(args, command) -> int:
    row = ROWS.get(args.target)
    if row is None:
        return _unknown_row(args.target)
    if args.dry_run:
        print(json.dumps(describe_row(row), indent=2))
        return 0
    if args.trial is not None:
        return cmd_one_trial(row, args)
    n_trials = args.trials if args.trials is not None else row.n_trials
    failed, summary = _run_row(
        row,
        args.out,
        n_trials=n_trials,
        epochs=args.epochs,
        warmup=args.warmup,
        timed=args.timed,
        command=command,
        resume=args.resume,
        zoo=args.zoo,
        in_process=args.in_process,
    )
    band_failed = summary is not None and summary.band["status"] == "fail"
    return 1 if failed or band_failed else 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    command = [sys.executable, "-m", "fabricpc.bench", *(argv or sys.argv[1:])]
    if args.target == "list":
        return cmd_list()
    if args.target == "compare":
        return cmd_compare(args)
    if args.target == "smoke":
        return cmd_smoke(args, command)
    if args.target == "validate":
        return cmd_validate(args)
    return cmd_run(args, command)


if __name__ == "__main__":
    sys.exit(main())
