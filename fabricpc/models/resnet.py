"""ResNet-18 for 32x32 images as a predictive coding graph.

Moved here from ``examples/resnet18_cifar10_demo.py`` so the benchmark
suite and the demo build the exact same graph. The layout is the CIFAR
variant (3x3 stem, no max pool):

    input(32,32,3) -> stem(32,32,32)
    -> stage 1: 2 residual blocks (32,32,32)
    -> stage 2: 2 residual blocks (16,16,64)
    -> stage 3: 2 residual blocks (8,8,128)
    -> stage 4: 2 residual blocks (4,4,256)
    -> global average pool -> Linear(10, softmax + cross-entropy)

Each residual block is three nodes (four when the skip path needs a 1x1
projection), so the graph has 31 nodes and 38 edges.

``lean=True`` builds the same network with fewer PC latents. The plain
layout gives nine nodes a latent even though they have no weights: the
eight SkipConnection adds and the global AvgPool. The lean layout folds
each add into the block's second conv (``ConvResidualNode``) and the
average pool into the last block, so every latent has weights: 21 latents
instead of 30, the same 2,795,210 parameters, and, given the same weights,
the same forward pass (muPC scales included). Only how PC settles changes.
"""

import math

from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import XavierInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core.mupc import MuPCConfig
from fabricpc.nodes import (
    AvgPool,
    ConvNode,
    ConvResidualNode,
    IdentityNode,
    Linear,
    SkipConnection,
)


def make_residual_block(
    prev_node,
    channels,
    stride,
    block_name,
    weight_init,
    activation=ReLUActivation(),
    lean_scales=None,
    global_pool=False,
):
    """
    Create one residual block: conv_a -> conv_b(act) -> skip(sum).

    With ``lean_scales`` (a dict from ``_lean_scales``) conv_b and the sum
    are one ``ConvResidualNode`` named ``<block>_res`` instead, and with
    ``global_pool`` too that node also averages over space.

    Activation is applied on the main path before summation. The skip path
    passes through without activation, preserving gradient flow.

    Returns:
        (nodes_list, edges_list, skip_node) where skip_node is the block output.
    """
    in_h, in_w, in_channels = prev_node._shape

    if stride == 1:
        out_h, out_w = in_h, in_w
    else:
        out_h, out_w = in_h // stride, in_w // stride

    nodes = []
    edges = []

    conv_a = ConvNode(
        shape=(out_h, out_w, channels),
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
        name=f"{block_name}_conv_a",
    )

    edges.append(Edge(source=prev_node, target=conv_a.slot("in")))

    if lean_scales is None:
        conv_b = ConvNode(
            shape=(out_h, out_w, channels),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="SAME",
            activation=activation,
            weight_init=weight_init,
            name=f"{block_name}_conv_b",
        )
        skip_node = SkipConnection(
            shape=(out_h, out_w, channels),
            name=f"{block_name}_skip_sum",
        )
        nodes.extend([conv_a, conv_b, skip_node])
        edges.append(Edge(source=conv_a, target=conv_b.slot("in")))
        edges.append(Edge(source=conv_b, target=skip_node.slot("in")))
    else:
        # conv_b and the add in one node; its latent is the block output.
        map_shape = (out_h, out_w, channels)
        skip_node = ConvResidualNode(
            shape=(channels,) if global_pool else map_shape,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="SAME",
            activation=activation,
            weight_init=weight_init,
            name=f"{block_name}_res",
            in_scale=lean_scales["in_scale"](channels),
            branch_scale=lean_scales["branch_scale"],
            global_pool=global_pool,
            map_shape=map_shape if global_pool else None,
            pool_scale=lean_scales["pool_scale"](out_h * out_w),
        )
        nodes.extend([conv_a, skip_node])
        edges.append(Edge(source=conv_a, target=skip_node.slot("in")))

    # Skip connection: the stream enters the merge's unscaled "skip" slot
    # (via a 1x1 projection when the block downsamples).
    needs_downsample = (stride != 1) or (in_channels != channels)
    if needs_downsample:
        conv_skip = ConvNode(
            shape=(out_h, out_w, channels),
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding="SAME",
            activation=IdentityActivation(),
            weight_init=weight_init,
            name=f"{block_name}_skip",
        )
        nodes.append(conv_skip)
        edges.append(Edge(source=prev_node, target=conv_skip.slot("in")))
        edges.append(Edge(source=conv_skip, target=skip_node.slot("skip")))
    else:
        edges.append(Edge(source=prev_node, target=skip_node.slot("skip")))

    return nodes, edges, skip_node


_STAGES = [
    (32, 1, 2),  # (channels, first_stride, num_blocks)
    (64, 2, 2),
    (128, 2, 2),
    (256, 2, 2),
]


def _lean_scales(scaling, activation, n_blocks):
    """The muPC scales the plain layout puts on the edges the lean layout
    folds away, so the lean graph computes the same function.

    Plain layout, from ``compute_mupc_scalings``: conv_b's input edge gets
    gain / sqrt(9 * C) (a 3x3 conv on C channels, not a merge); the
    SkipConnection's branch edge gets 1 / sqrt(L) (weightless, identity
    gain, a merge, with L = the number of blocks on the longest path); the
    global AvgPool's edge gets 1 / sqrt(1/n) = sqrt(n) for n pooled cells.
    Without muPC all three are 1.0.
    """
    if scaling is None:
        return {
            "in_scale": lambda channels: 1.0,
            "branch_scale": 1.0,
            "pool_scale": lambda cells: 1.0,
        }
    if not isinstance(scaling, MuPCConfig):
        raise NotImplementedError(
            f"lean=True knows muPC's scales only, got {type(scaling).__name__}"
        )
    gain = type(activation).variance_gain(activation.config)
    return {
        "in_scale": lambda channels: gain / math.sqrt(9 * channels),
        "branch_scale": 1.0 / math.sqrt(n_blocks),
        "pool_scale": lambda cells: math.sqrt(cells),
    }


def build_resnet18(
    weight_init,
    inference,
    scaling=None,
    output_weight_init=XavierInitializer(),
    activation=ReLUActivation(),
    lean=False,
):
    """
    Build ResNet-18 for CIFAR-10 as a predictive coding graph.

    Args:
        weight_init: InitializerBase for conv/linear weights.
        inference: InferenceBase instance (a plain solver or an
            InferenceSchedule) driving PC inference.
        scaling: Optional MuPCConfig for muPC parameterization.
        output_weight_init: InitializerBase for the output layer
            (default: XavierInitializer).
        activation: Activation for hidden conv layers (default: ReLU).
        lean: fold each block's add and the final average pool into
            weighted nodes, so no PC latent is parameter-free (see the
            module docstring). Off by default; the benchmark's
            ``cifar10-resnet18`` rows use the plain layout.

    Returns:
        GraphStructure ready for initialize_params().
    """
    # Input
    input_node = IdentityNode(shape=(32, 32, 3), name="input")

    # Stem convolution: 3x3, 32 channels, no maxpool (CIFAR is 32x32)
    stem = ConvNode(
        shape=(32, 32, 32),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
        name="stem",
    )

    all_nodes = [input_node, stem]
    all_edges = [Edge(source=input_node, target=stem.slot("in"))]

    # Build 4 stages with [2, 2, 2, 2] blocks
    n_blocks = sum(num_blocks for _, _, num_blocks in _STAGES)
    lean_scales = _lean_scales(scaling, activation, n_blocks) if lean else None

    prev = stem
    for stage_idx, (channels, first_stride, num_blocks) in enumerate(_STAGES, 1):
        for block_idx in range(num_blocks):
            stride = first_stride if block_idx == 0 else 1
            block_name = f"s{stage_idx}b{block_idx + 1}"
            is_last = stage_idx == len(_STAGES) and block_idx == num_blocks - 1

            nodes, edges, add_node = make_residual_block(
                prev_node=prev,
                channels=channels,
                stride=stride,
                block_name=block_name,
                weight_init=weight_init,
                activation=activation,
                lean_scales=lean_scales,
                global_pool=lean and is_last,
            )
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            prev = add_node

    # Global average pooling: (B, 4, 4, 256) -> (B, 256). The lean layout
    # already pooled inside the last block.
    if not lean:
        avg_pool = AvgPool(shape=(256,), name="avgpool", global_pool=True)
        all_nodes.append(avg_pool)
        all_edges.append(Edge(source=prev, target=avg_pool.slot("in")))
        prev = avg_pool

    # Output: Linear(10) with softmax + cross-entropy
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        weight_init=output_weight_init,
        name="output",
    )
    all_nodes.append(output)
    all_edges.append(Edge(source=prev, target=output.slot("in")))

    # Build graph
    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=input_node, y=output),
        inference=inference,
        scaling=scaling,
    )

    return structure
