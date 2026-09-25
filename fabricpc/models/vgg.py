"""VGG-style image classifiers as predictive coding graphs.

A VGG is a chain of 3x3 convolutions with 2x2 max pools between blocks and
one linear classifier at the end. In FabricPC every conv and every pool is
its own node with its own latent state, so the graph is a plain chain:

    input -> conv1 -> pool1 -> conv2 -> pool2 -> ... -> class

Depths supported:

* VGG-5: conv channels 128, 256, 512, 512 with a pool after each conv. This
  is the layout the pcx benchmark paper uses (arXiv:2407.01163), so its
  numbers can be compared with ours.
* VGG-7: two convs then a pool, three times: 128,128 | 256,256 | 512,512.
* VGG-9: two convs then a pool, four times: 128,128 | 256,256 | 512,512 |
  512,512.

VGG-7 and VGG-9 follow the usual VGG pattern; they are our layouts, not
pcx's, and are noted as such in the benchmark suite.
"""

from typing import Optional, Sequence, Tuple

from fabricpc.core.activations import ActivationBase, GeluActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceBase
from fabricpc.core.initializers import InitializerBase, KaimingInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.nodes import ConvNode, ConvPoolNode, IdentityNode, Linear, MaxPool

# Each inner list is one block; a 2x2 max pool follows every block.
_BLOCKS = {
    5: [[128], [256], [512], [512]],
    7: [[128, 128], [256, 256], [512, 512]],
    9: [[128, 128], [256, 256], [512, 512], [512, 512]],
}


def create_vgg(
    depth: int,
    *,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    inference: InferenceBase,
    activation: Optional[ActivationBase] = None,
    weight_init: Optional[InitializerBase] = None,
    scaling=None,
    fuse_pool: bool = False,
):
    """Build a VGG graph of the given depth.

    Args:
        depth: 5, 7, or 9.
        input_shape: (height, width, channels) of one image, e.g. (32, 32, 3).
        num_classes: size of the softmax output.
        inference: the PC solver for the graph (InferenceSGD, EPCInference, ...).
        activation: hidden activation; GELU by default, as in pcx's CIFAR runs.
        weight_init: conv/linear initializer; Kaiming by default.
        scaling: optional MuPCConfig, passed straight to ``graph``.
        fuse_pool: put each block's max pool inside its last conv node
            (``ConvPoolNode``), as pcx does, instead of a separate ``MaxPool``
            node. Backprop computes the same function either way; for
            predictive coding it halves the number of latent layers, which
            state-based PC needs to train VGG-5 well.

    Returns:
        A GraphStructure ready for ``initialize_params``.
    """
    if depth not in _BLOCKS:
        raise ValueError(f"depth must be one of {sorted(_BLOCKS)}, got {depth}")
    activation = activation or GeluActivation()
    weight_init = weight_init or KaimingInitializer()

    height, width, _ = input_shape
    nodes = [IdentityNode(shape=tuple(input_shape), name="input")]
    edges = []
    prev = nodes[0]
    conv_i = 0

    for block_i, channels in enumerate(_BLOCKS[depth], start=1):
        for j, c in enumerate(channels):
            conv_i += 1
            common = dict(
                name=f"conv{conv_i}",
                kernel_size=(3, 3),
                padding="SAME",
                activation=activation,
                weight_init=weight_init,
            )
            if fuse_pool and j == len(channels) - 1:
                conv = ConvPoolNode(shape=(height // 2, width // 2, c), **common)
            else:
                conv = ConvNode(shape=(height, width, c), **common)
            nodes.append(conv)
            edges.append(Edge(source=prev, target=conv.slot("in")))
            prev = conv
        height, width = height // 2, width // 2
        if fuse_pool:
            continue
        pool = MaxPool(
            shape=(height, width, channels[-1]),
            name=f"pool{block_i}",
            window_shape=(2, 2),
        )
        nodes.append(pool)
        edges.append(Edge(source=prev, target=pool.slot("in")))
        prev = pool

    output = Linear(
        shape=(num_classes,),
        name="class",
        flatten_input=True,
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=weight_init,
    )
    nodes.append(output)
    edges.append(Edge(source=prev, target=output.slot("in")))

    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=nodes[0], y=output),
        inference=inference,
        scaling=scaling,
    )


def vgg_channels(depth: int) -> Sequence[int]:
    """The conv channel list for a depth, flattened. Handy for docs and tests."""
    return [c for block in _BLOCKS[depth] for c in block]
