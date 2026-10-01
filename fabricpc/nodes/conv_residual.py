"""
A conv layer and its residual add in one PC node.

In the plain ResNet layout a residual block ends with a ``ConvNode`` and
then a ``SkipConnection`` node that adds the stream back in. The add has no
weights, but it still gets a PC latent of its own, and so does the global
average pool before the classifier. Each such latent is one more hop a
prediction error has to cross while settling, which is what hurt sPC on
the old VGG layout with separate MaxPool nodes (see ``ConvPoolNode``).

``ConvResidualNode`` is the block's second conv with the add inside it:

    z_mu = branch_scale * act(conv(in_scale * x_in) + b) + x_skip

and, with ``global_pool=True``, the result is also averaged over space:

    z_mu = pool_scale * mean_over_space(...)

so the last block can hold the pooled features and the classifier reads
them directly. It has the same weights as the ConvNode it replaces.

Why the scales are arguments. muPC scales each edge before a node's
forward, and it can not see inside this node: the plain layout scales the
conv input by ``gain / sqrt(fan_in)``, damps the branch by ``1 / sqrt(L)``
at the SkipConnection, and scales the pooled map by ``sqrt(n)`` at the
AvgPool. Those three numbers now live inside one node, so the builder
passes them in (``build_resnet18(lean=True)`` works them out with muPC's
own formulas) and both slots are left out of muPC's edge scaling. With
the scales at their default of 1.0 the node is just conv + add. The one
thing that does not carry over is muPC's top-down Jacobian compensation
on the folded edges, which only changes how PC settles, not the forward
pass.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import jax
import jax.lax as lax
import jax.numpy as jnp

from fabricpc.core.activations import ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import (
    KaimingInitializer,
    NormalInitializer,
    ZerosInitializer,
)
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.convolutional import ConvNode

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


def _slot_input(inputs: Dict[str, jnp.ndarray], slot: str) -> Tuple[str, jnp.ndarray]:
    key = next(k for k in inputs if k.endswith(f":{slot}"))
    return key, inputs[key]


class ConvResidualNode(ConvNode):
    """
    conv -> bias -> activation -> scaled -> plus the skip stream, in one node.

    Wire the block's first conv to "in" and the residual stream (the
    previous block, or its 1x1 projection) to "skip". The skip edge is
    required: it is what makes this node a residual merge for muPC's depth
    count, like SkipConnection and MlpResidualNode.

    Args (on top of ConvNode's):
        in_scale: multiplies the "in" input before the conv (muPC's edge
            scale for the conv). Default 1.0.
        branch_scale: multiplies the activated conv before the add (muPC's
            1/sqrt(L) damping at the merge). Default 1.0.
        global_pool: average the block's output over all spatial cells, so
            the latent is (C,). Needs ``map_shape``.
        map_shape: the (H, W, C) block output before pooling.
        pool_scale: multiplies the pooled output (muPC's sqrt(n) for an
            average over n cells). Default 1.0.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        kernel_size: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]] = None,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        activation: "ActivationBase" = ReLUActivation(),
        energy: "EnergyFunctional" = GaussianEnergy(),
        use_bias: bool = True,
        weight_init: "InitializerBase" = KaimingInitializer(),
        bias_init: "InitializerBase" = ZerosInitializer(),
        latent_init: "InitializerBase" = NormalInitializer(),
        in_scale: float = 1.0,
        branch_scale: float = 1.0,
        global_pool: bool = False,
        map_shape: Optional[Tuple[int, ...]] = None,
        pool_scale: float = 1.0,
    ):
        if global_pool:
            if map_shape is None:
                raise ValueError(
                    f"ConvResidualNode '{name}': global_pool=True needs "
                    "map_shape, the (H, W, C) output before pooling."
                )
            if tuple(shape) != (map_shape[-1],):
                raise ValueError(
                    f"ConvResidualNode '{name}': with global_pool the shape "
                    f"must be (C,) = ({map_shape[-1]},), got {tuple(shape)}."
                )
        elif map_shape is not None and tuple(map_shape) != tuple(shape):
            raise ValueError(
                f"ConvResidualNode '{name}': map_shape is only for global_pool."
            )
        map_shape = tuple(map_shape) if global_pool else tuple(shape)
        if stride is None:
            stride = (1,) * (len(map_shape) - 1)
        NodeBase.__init__(
            self,
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            bias_init=bias_init,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            in_scale=float(in_scale),
            branch_scale=float(branch_scale),
            global_pool=bool(global_pool),
            map_shape=map_shape,
            pool_scale=float(pool_scale),
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            # Not muPC-scaled: the node applies in_scale itself (see the
            # module docstring).
            "in": SlotSpec(name="in", is_multi_input=False, is_variance_scalable=False),
            "skip": SlotSpec(
                name="skip",
                is_multi_input=False,
                is_variance_scalable=False,
                is_skip_connection=True,
                require_connected=True,
            ),
        }

    @staticmethod
    def _unpooled_shape(node_shape, config, input_shapes=None) -> Tuple[int, ...]:
        """Where the conv runs: the (H, W, C) map, also when global_pool
        leaves the latent at (C,). Used by the bench's flop count."""
        return tuple(config.get("map_shape") or node_shape)

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: "InitializerBase",
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """One kernel, for the "in" edge; the skip edge has no weights."""
        config = config or {}
        map_shape = tuple(config.get("map_shape") or node_shape)
        conv_inputs = {k: s for k, s in input_shapes.items() if k.endswith(":in")}
        for k, s in input_shapes.items():
            if k.endswith(":skip") and tuple(s) != map_shape:
                raise ValueError(
                    f"ConvResidualNode: skip input {k} has shape {tuple(s)}, "
                    f"but the block output is {map_shape}."
                )
        return ConvNode.initialize_params(
            key, map_shape, conv_inputs, weight_init, config
        )

    @staticmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, None]:
        config = node_info.node_config
        map_shape = config["map_shape"]
        dim_numbers = ConvNode._DIM_NUMBERS[len(map_shape) - 1]

        in_key, x = _slot_input(inputs, "in")
        _, skip = _slot_input(inputs, "skip")

        pre = lax.conv_general_dilated(
            lhs=config["in_scale"] * x,
            rhs=params.weights[in_key],
            window_strides=config.get("stride"),
            padding=config.get("padding"),
            dimension_numbers=dim_numbers,
        )
        if "b" in params.biases and params.biases["b"].size > 0:
            pre = pre + params.biases["b"]

        activation = node_info.activation
        branch = type(activation).forward(pre, activation.config)
        out = config["branch_scale"] * branch + skip

        if config["global_pool"]:
            spatial = tuple(range(1, out.ndim - 1))
            out = config["pool_scale"] * jnp.mean(out, axis=spatial)
        return out, None
