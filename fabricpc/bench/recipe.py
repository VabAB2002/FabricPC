"""Build a family's rows, one per method, from one graph builder.

A new model joins the suite with one call: say how to build its graph for a
given settling solver, which solver each PC method uses, its data and its
optimizer, and ``pc_family`` makes the rows. Every row gets the same graph,
the same starting weights for a seed, the same data order and the same
optimizer; only the method differs, so the comparison is fair by
construction. ``register_family`` then makes them runnable by name.

See "Adding your own model" in docs/user_guides/18_benchmark_suite.md.
"""

from typing import Callable, Dict, Mapping, Optional, Union

import jax
import optax

from fabricpc.bench.methods import ALGORITHMS, METHODS
from fabricpc.bench.registry import (
    DEFAULT_TRIALS,
    BenchmarkRow,
    LoaderFactory,
    _cifar10_loaders,
    _flat_image_loaders,
    _safety_net,
)
from fabricpc.core.inference import InferenceBase
from fabricpc.graph_initialization import initialize_params

GraphBuilder = Callable[[InferenceBase], object]  # solver -> GraphStructure
Optimizer = Union[
    optax.GradientTransformation, Callable[[int], optax.GradientTransformation]
]


def _model_factory(build_graph: GraphBuilder, solver: InferenceBase):
    def build(rng_key: jax.Array):
        structure = build_graph(solver)
        return initialize_params(structure, rng_key), structure

    return build


def _optimizer_factory(optimizer: Optimizer):
    if isinstance(optimizer, optax.GradientTransformation):
        return lambda total_steps: optimizer
    return optimizer  # already (total_steps) -> transform, for schedules


def pc_family(
    *,
    dataset: str,
    model: str,
    build_graph: GraphBuilder,
    solvers: Mapping[str, InferenceBase],
    loader_factory: LoaderFactory,
    optimizer: Optimizer,
    num_epochs: float,
    batch_size: int,
    tune_loader_factory: Optional[LoaderFactory] = None,
    metric: str = "accuracy",
    methods=ALGORITHMS,
    safety_net: bool = True,
    n_trials: int = DEFAULT_TRIALS,
    **row_options,
) -> Dict[str, BenchmarkRow]:
    """One row per method, ids ``<dataset>-<model>-<method>``.

    ``build_graph(solver)`` returns the graph (a ``GraphStructure``) built
    with that solver; it is called once per trial. ``solvers`` gives each
    settling method's solver, e.g. ``{"spc": InferenceSGD(...), "epc":
    EPCInference(...)}``. Backprop never settles, but its graph still needs
    a solver to be built, so it gets the first one given. ``optimizer`` is
    an optax transform, or ``(total_steps) -> transform`` for a learning
    rate schedule. ``tune_loader_factory`` (train on most of the training
    set, score on the rest) lets ``tune`` tune the rows. ``safety_net``
    gives PC rows the suite's safety net. Other keywords (``tier``,
    ``eval_metrics``, ``reference``, ...) go to every row.
    """
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods {unknown}; known: {list(METHODS)}")
    needs_solver = [m for m in methods if METHODS[m].settles and m not in solvers]
    if needs_solver:
        raise ValueError(f"no solver given for {needs_solver}")
    if not solvers:
        raise ValueError("give at least one solver (backprop's graph needs one)")
    fallback = next(iter(solvers.values()))
    rows = {}
    for method in methods:
        row_id = f"{dataset}-{model}-{method}"
        solver = solvers[method] if METHODS[method].settles else fallback
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset=dataset,
            model=model,
            algorithm=method,
            model_factory=_model_factory(build_graph, solver),
            loader_factory=loader_factory,
            tune_loader_factory=tune_loader_factory,
            optimizer_factory=_optimizer_factory(optimizer),
            train_config={"num_epochs": num_epochs},
            batch_size=batch_size,
            n_trials=n_trials,
            metric=metric,
            rate_control=_safety_net(method) if safety_net else None,
            **row_options,
        )
    return rows


def flat_image_loaders(dataset: str, batch_size: int, *, tuning: bool = False):
    """The suite's MNIST / Fashion-MNIST loaders (flattened 784 pixels):
    ``dataset`` is "mnist" or "fashionmnist"; ``tuning=True`` gives the
    held-out split ``tune`` scores on instead of the test set."""
    return _flat_image_loaders(dataset, batch_size, tuning=tuning)


def cifar10_loaders(batch_size: int, *, tuning: bool = False):
    """The suite's CIFAR-10 loaders (32x32x3, flips and crops for training)."""
    return _cifar10_loaders(batch_size, tuning=tuning)
