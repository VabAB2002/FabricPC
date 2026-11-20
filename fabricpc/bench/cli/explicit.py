"""``run`` and ``list``: pick a benchmark with separate flags.

    python -m fabricpc.bench run --dataset mnist --architecture mlp --method spc
    python -m fabricpc.bench run --dataset mnist --architecture mlp   (all methods)
    python -m fabricpc.bench list [--dataset mnist]

Each has its own ``--help``. ``run`` only works out the row id (or the
family id, when no method is given) and hands over to the normal command
line, so ``run --dataset mnist --architecture mlp --method spc`` does exactly
what ``python -m fabricpc.bench mnist-mlp-spc`` does. The short id form
still works; CI, plan files and the per-trial child processes use it.
"""

import argparse
import sys
from typing import Callable, Dict, List, Sequence

from fabricpc.bench.definitions.registry import COMPARISONS, ROWS

METHOD_ORDER = ("spc", "epc", "backprop")
RunMain = Callable[[Sequence[str]], int]


def _datasets() -> List[str]:
    return sorted({r.dataset for r in ROWS.values()})


def _architectures() -> List[str]:
    return sorted({r.model for r in ROWS.values()})


def _methods() -> List[str]:
    found = {r.algorithm for r in ROWS.values()}
    return [m for m in METHOD_ORDER if m in found] + sorted(found - set(METHOD_ORDER))


# (flag, how it is passed on). Value flags are passed as "--flag value",
# switches as "--flag" when set. Help text matches the main command line.
_VALUE_FLAGS = (
    ("--out", str, "where result files go (default: results)"),
    ("--trials", int, "how many seeds to run (default: the row's own)"),
    ("--epochs", float, "override the row's epoch count"),
    ("--zoo", str, "save each trial's trained weights in this folder"),
    ("--warmup", int, "untimed steps before timing (default 5)"),
    ("--timed", int, "timed steps (default 30)"),
    (
        "--curve-batches",
        int,
        "test batches scored after every epoch for the learning curve "
        "(default 10; 0 turns the curve off)",
    ),
)
_SWITCHES = (
    ("--dry-run", "show the settings, do not train"),
    ("--resume", "skip trials that already finished"),
    ("--fast", "quicker runs: no learning curve and no diagnostics"),
    ("--no-diagnostics", "skip the backprop-likeness and stiffness checks"),
    ("--in-process", "run trials in this process instead of one process each"),
)


def run_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fabricpc.bench run",
        description="Run one benchmark: a dataset, a model architecture and a "
        "learning method. Leave out --method to run all methods on the same "
        "model and compare them. 'python -m fabricpc.bench list' shows what "
        "exists.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=_datasets(),
        metavar="NAME",
        help="dataset: " + ", ".join(_datasets()),
    )
    p.add_argument(
        "--architecture",
        required=True,
        choices=_architectures(),
        metavar="NAME",
        help="model architecture: " + ", ".join(_architectures()),
    )
    p.add_argument(
        "--method",
        choices=_methods(),
        metavar="NAME",
        help="learning method: " + ", ".join(_methods()) + " "
        "(default: all of them, then a comparison)",
    )
    for flag, kind, text in _VALUE_FLAGS:
        p.add_argument(flag, type=kind, help=text)
    for flag, text in _SWITCHES:
        p.add_argument(flag, action="store_true", help=text)
    return p


def list_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fabricpc.bench list",
        description="List every benchmark: dataset, architecture, the methods "
        "it runs with, the metric it is judged on, and its default epochs and "
        "seeds.",
    )
    p.add_argument(
        "--dataset", choices=_datasets(), metavar="NAME", help="only this dataset"
    )
    p.add_argument(
        "--ids",
        action="store_true",
        help="print the short ids instead (one row per line, then the families)",
    )
    return p


def target_for(dataset: str, architecture: str, method=None) -> str:
    """The row id (or family id without a method) for these choices.

    Looked up from the rows' own fields, not by gluing strings, and raises
    ValueError naming what does exist when the combination is not there.
    """
    rows = [r for r in ROWS.values() if (r.dataset, r.model) == (dataset, architecture)]
    if not rows:
        have = sorted({r.model for r in ROWS.values() if r.dataset == dataset})
        raise ValueError(
            f"no {architecture} benchmark on {dataset}; architectures for "
            f"{dataset}: {', '.join(have)}"
        )
    if method is None:
        family = [
            f.id for f in COMPARISONS.values() if {r.id for r in rows} == set(f.rows)
        ]
        if not family:
            raise ValueError(
                f"{dataset} {architecture} has no all-methods comparison; "
                "pick one with --method"
            )
        return family[0]
    match = [r.id for r in rows if r.algorithm == method]
    if not match:
        have = [m for m in _methods() if m in {r.algorithm for r in rows}]
        raise ValueError(
            f"{dataset} {architecture} does not run with {method}; "
            f"methods: {', '.join(have)}"
        )
    return match[0]


def run_cli(argv: Sequence[str], run_main: RunMain) -> int:
    """``python -m fabricpc.bench run ...``. Returns the exit code."""
    parser = run_parser()
    args = parser.parse_args(list(argv))
    try:
        target = target_for(args.dataset, args.architecture, args.method)
    except ValueError as e:
        parser.error(str(e))
    forward = [target]
    for flag, _, _ in _VALUE_FLAGS:
        value = getattr(args, flag[2:].replace("-", "_"))
        if value is not None:
            forward += [flag, str(value)]
    for flag, _ in _SWITCHES:
        if getattr(args, flag[2:].replace("-", "_")):
            forward.append(flag)
    # stderr, so --dry-run's JSON on stdout stays clean.
    print(
        f"run: {target} (same as: python -m fabricpc.bench {' '.join(forward)})",
        file=sys.stderr,
    )
    return run_main(forward)


def benchmark_table(dataset=None) -> List[Dict[str, str]]:
    """One line per dataset + architecture, with its methods."""
    groups: Dict[tuple, list] = {}
    for row in ROWS.values():
        if dataset is None or row.dataset == dataset:
            groups.setdefault((row.dataset, row.model), []).append(row)
    table = []
    for (ds, arch), rows in sorted(groups.items()):
        methods = [m for m in _methods() if m in {r.algorithm for r in rows}]
        first = rows[0]
        table.append(
            {
                "dataset": ds,
                "architecture": arch,
                "methods": ", ".join(methods),
                "metric": first.metric,
                "epochs": f"{first.train_config['num_epochs']:g}",
                "seeds": str(first.n_trials),
            }
        )
    return table


def list_cli(argv: Sequence[str], print_ids: Callable[[], int]) -> int:
    """``python -m fabricpc.bench list ...``. Returns the exit code."""
    args = list_parser().parse_args(list(argv))
    if args.ids:
        return print_ids()
    table = benchmark_table(args.dataset)
    cols = ["dataset", "architecture", "methods", "metric", "epochs", "seeds"]
    width = {c: max(len(c), *(len(r[c]) for r in table)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols).rstrip())
    for r in table:
        print("  ".join(r[c].ljust(width[c]) for c in cols).rstrip())
    print(
        "\nRun one:   python -m fabricpc.bench run --dataset D --architecture A "
        "--method M\nRun all methods and compare: leave out --method. "
        "Details: python -m fabricpc.bench run --help\n"
        "Short ids (like mnist-mlp-spc): python -m fabricpc.bench list --ids"
    )
    return 0
