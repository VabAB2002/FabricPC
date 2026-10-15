"""Add your own model to the benchmark suite: a worked example.

This file adds a family, ``mnist-wide-mlp`` (784 -> 512 -> 10), trained with
sPC, ePC and backprop. Point the suite at it and use it like any other row:

    FABRICPC_BENCH_ROWS=examples/bench_add_your_model.py python -m fabricpc.bench list
    FABRICPC_BENCH_ROWS=examples/bench_add_your_model.py python -m fabricpc.bench mnist-wide-mlp --preset fast
    FABRICPC_BENCH_ROWS=examples/bench_add_your_model.py python -m fabricpc.bench tune mnist-wide-mlp --budget 10

(``--rows-module examples/bench_add_your_model.py`` does the same as the
variable.) Copy it, change ``build_graph`` and the settings, and your model
gets the suite's seeds, timing, statistics, tuning and report for free.
"""

import optax

from fabricpc.bench.recipe import flat_image_loaders, pc_family
from fabricpc.bench.registry import register_family
from fabricpc.core.activations import ReLUActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.initializers import XavierInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.nodes import IdentityNode, Linear


def build_graph(solver):
    """The model, built with whichever settling solver the method uses."""
    pixels = IdentityNode(shape=(784,), name="pixels")
    hidden = Linear(
        shape=(512,),
        activation=ReLUActivation(),
        name="hidden",
        weight_init=XavierInitializer(),
    )
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        weight_init=XavierInitializer(),
    )
    return graph(
        nodes=[pixels, hidden, output],
        edges=[
            Edge(source=pixels, target=hidden.slot("in")),
            Edge(source=hidden, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=solver,
    )


BATCH_SIZE = 200

FAMILY = register_family(
    pc_family(
        dataset="mnist",
        model="wide-mlp",
        build_graph=build_graph,
        solvers={
            "spc": InferenceSGD(eta_infer=0.05, infer_steps=20),
            "epc": EPCInference(eta_infer=1e-3, infer_steps=5),
        },
        loader_factory=flat_image_loaders("mnist", BATCH_SIZE),
        tune_loader_factory=flat_image_loaders("mnist", BATCH_SIZE, tuning=True),
        optimizer=optax.adamw(1e-3, weight_decay=0.1),
        num_epochs=10,
        batch_size=BATCH_SIZE,
    )
)
