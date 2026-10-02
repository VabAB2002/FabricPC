"""The deep MLP used in the hint experiment.

784 pixels -> `depth` tanh layers of `width` units -> 10-way softmax. Deep
enough that sPC's error signal has to pass through many layers to reach the
first one, which is where the fading shows up.
"""

import jax

from fabricpc.core.activations import SoftmaxActivation, TanhActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import XavierInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import IdentityNode, Linear


def deep_mlp(depth: int, width: int, solver, key: jax.Array):
    """Build the graph and its starting weights. Hidden nodes are h0..h{depth-1}."""
    pixels = IdentityNode(shape=(784,), name="pixels")
    hidden = [
        Linear(
            shape=(width,),
            activation=TanhActivation(),
            name=f"h{i}",
            weight_init=XavierInitializer(),
        )
        for i in range(depth)
    ]
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        weight_init=XavierInitializer(),
    )
    chain = [pixels, *hidden, output]
    edges = [Edge(source=a, target=b.slot("in")) for a, b in zip(chain, chain[1:])]
    structure = graph(
        nodes=chain,
        edges=edges,
        task_map=TaskMap(x=pixels, y=output),
        inference=solver,
    )
    return initialize_params(structure, key), structure
