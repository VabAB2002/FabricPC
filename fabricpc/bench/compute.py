"""How much arithmetic one weight update costs, worked out from the graph.

We count matrix multiplies per edge, because every learning algorithm
here is built from them:

* Predictive coding: each settling step does a forward projection and a
  gradient on every weighted edge (2 matmuls per edge per step), then the
  weight update does one more per edge. That is ``2*E*T + E``.
* Backprop: forward, gradient to the input, gradient to the weight.
  That is ``3*E``.

For a plain chain this is the sponsor's closed form with E = depth. For
any other graph it is the same sum taken edge by edge, which is why we
count edges instead of layers.

Some nodes keep their own named weights instead of one per incoming edge
(a transformer's W_q, W_k, W_v, W_o, W_ff1, W_ff2, W_out). Each such matrix
is counted as one more weighted edge, applied at every position of the
node. Attention also multiplies Q by K^T and the attention weights by V,
two matmuls with no weight at all: their FLOPs are added to each pass, but
they are not counted as edges. An embedding table is a lookup, not a
matmul, and 1-D weights (a layer norm's gain) are not matmuls either.

FLOPs use the usual 2 * (multiply-adds) rule for one pass over an edge.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Compute:
    weighted_edges: int
    infer_steps: int
    matmuls_per_update: int
    matmuls_per_update_backprop: int
    pc_to_backprop_ratio: float
    flops_per_pass: int  # one pass over every weighted edge
    flops_per_update: int  # flops_per_pass times the matmul factor


def _flops_one_pass_for_edge(weight, target_shape, batch_size: int) -> int:
    """2 * batch * multiply-adds for one pass over one weighted edge.

    Linear weights are (fan_in, fan_out). Conv kernels are
    (*kernel, C_in, C_out). Either way the multiply-adds repeat at every
    output position, so we multiply by everything in the output's shape
    but its last dim: the spatial size of a conv map, the sequence length
    of a transformer node, and 1 for a plain vector.
    """
    weight_size = math.prod(weight.shape)
    positions = math.prod(target_shape[:-1])
    return 2 * batch_size * positions * weight_size


# Weights a node keeps under its own name that are not matmuls.
_NOT_MATMULS = ("embeddings",)


def _own_matmuls(node_params, edge_keys):
    """A node's named weight matrices that are not tied to an edge."""
    return [
        w
        for name, w in node_params.weights.items()
        if name not in edge_keys and name not in _NOT_MATMULS and w.ndim >= 2
    ]


def _attention_flops(node, node_params, batch_size: int) -> int:
    """Q K^T and A V of a self-attention node: 2 * (2 * B * L * L * D).

    Splitting D over heads does not change the total.
    """
    weights = node_params.weights
    if "W_q" not in weights or "W_k" not in weights:
        return 0
    seq_len = node.node_info.shape[0]
    width = weights["W_q"].shape[-1]
    return 2 * (2 * batch_size * seq_len * seq_len * width)


def _matmul_output_shape(node, input_shapes=None):
    """Where a node's matmul actually runs.

    Usually the node's own shape. A ConvPoolNode pools after its conv, so its
    shape is the pooled one and its conv runs on the larger, unpooled grid.
    The same goes for a ConvResidualNode with global_pool, whose latent is
    (C,) but whose conv runs on the whole map. The node's input shapes are
    passed along because a pool rounds an odd size down, and only the input
    tells us the real conv size then (7x7 pooled to 3x3, not 6x6).
    """
    shape = node.node_info.shape
    unpooled = getattr(type(node), "_unpooled_shape", None)
    if unpooled is None:
        return shape
    return unpooled(shape, node.node_info.node_config, input_shapes)


def count_edge_weights(params, structure) -> int:
    """Weighted edges by the first rule only: weights keyed by an edge.

    Results written before a node's own weights were counted saved this
    number, so ``refresh`` uses it to tell "the graph changed" apart from
    "the counting changed".
    """
    return sum(
        edge_key in params.nodes[edge.target].weights
        for edge_key, edge in structure.edges.items()
    )


def count_compute(
    params,
    structure,
    *,
    batch_size: int,
    algorithm: str,
    infer_steps: Optional[int] = None,
) -> Compute:
    """Count matmuls and flops per weight update for ``algorithm``."""
    if infer_steps is None:
        infer_steps = int(structure.config["inference"].config["infer_steps"])

    weighted_edges = 0
    flops_per_pass = 0
    for edge_key, edge in structure.edges.items():
        target_params = params.nodes[edge.target]
        weight = target_params.weights.get(edge_key)
        if weight is None:
            continue  # e.g. a skip edge or an identity node: no matmul
        weighted_edges += 1
        source_shape = structure.nodes[edge.source].node_info.shape
        target_shape = _matmul_output_shape(
            structure.nodes[edge.target], {edge_key: source_shape}
        )
        flops_per_pass += _flops_one_pass_for_edge(weight, target_shape, batch_size)

    edge_keys = set(structure.edges)
    for name, node in structure.nodes.items():
        node_params = params.nodes[name]
        for weight in _own_matmuls(node_params, edge_keys):
            weighted_edges += 1
            flops_per_pass += _flops_one_pass_for_edge(
                weight, node.node_info.shape, batch_size
            )
        flops_per_pass += _attention_flops(node, node_params, batch_size)

    backprop_factor = 3
    if algorithm == "backprop":
        factor = backprop_factor
    else:
        factor = 2 * infer_steps + 1

    matmuls = factor * weighted_edges
    matmuls_bp = backprop_factor * weighted_edges
    return Compute(
        weighted_edges=weighted_edges,
        infer_steps=infer_steps,
        matmuls_per_update=matmuls,
        matmuls_per_update_backprop=matmuls_bp,
        pc_to_backprop_ratio=(matmuls / matmuls_bp) if matmuls_bp else float("nan"),
        flops_per_pass=flops_per_pass,
        flops_per_update=factor * flops_per_pass,
    )
