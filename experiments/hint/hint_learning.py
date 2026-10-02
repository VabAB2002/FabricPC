"""Put the hint into what each layer learns from, not into its latent.

First try (hint_inference.py) nudged the latents. That leaks: layer l's error is
measured against a prediction made from layer l-1's latent, which got its own
hint, so the two hints mix. Here sPC settles normally and then each hidden
layer's weight gradient gets a direct-feedback-alignment term added on top:

    mode="none"  plain sPC gradients
    mode="pure"  hidden layers learn from the hint only (textbook DFA)
    mode="all"   every hidden layer: whisper + hint
    mode="far"   only layers whose whisper has faded get the hint

The hint is scaled so it is about as loud as the whisper at the last hidden
layer, where the whisper is still clear.
"""

import jax
import jax.numpy as jnp

from experiments.hint.hint_inference import hint_matrix
from fabricpc.core.inference import run_inference
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.trainer import build_clamps, pc_weight_gradients


def hidden_names(structure):
    return sorted(
        (n for n in structure.nodes if n.startswith("h")), key=lambda n: int(n[1:])
    )


def _forward(params, structure, x):
    """Plain feedforward pass. Returns {node: (input, activation)} for hidden nodes and the output probabilities."""
    acts, prev, h = {}, "pixels", x
    for name in hidden_names(structure) + ["class"]:
        w = params.nodes[name].weights[f"{prev}->{name}:in"]
        b = list(params.nodes[name].biases.values())[0]
        a = h @ w + b
        out = jax.nn.softmax(a) if name == "class" else jnp.tanh(a)
        acts[name] = (h, out)
        prev, h = name, out
    return acts, h


def _replace(grads, name, w_grad, b_grad):
    node = grads.nodes[name]
    (wk,), (bk,) = node.weights.keys(), node.biases.keys()
    new = node._replace(weights={wk: w_grad}, biases={bk: b_grad})
    return grads._replace(nodes={**grads.nodes, name: new})


def dfa_grads(params, structure, batch):
    """Textbook DFA gradients (loss = mean cross-entropy) for the hidden layers."""
    acts, p = _forward(params, structure, batch["x"])
    e = (p - batch["y"]) / batch["x"].shape[0]
    out = params
    for name in hidden_names(structure):
        h_in, h = acts[name]
        b = jnp.asarray(hint_matrix(name, h.shape[-1], 10, 0))
        d = (e @ b) * (1 - h**2)
        out = _replace(out, name, h_in.T @ d, d.sum(0))
    return out


def _norm(tree_node):
    return jnp.sqrt(sum(jnp.sum(w**2) for w in tree_node.weights.values()))


def batch_grads(
    params, structure, batch, key, mode="none", strength=1.0, threshold=0.1
):
    clamps = build_clamps(batch, structure, clamp_target=True)
    state = initialize_graph_state(
        structure, batch["x"].shape[0], key, clamps=clamps, params=params
    )
    state = run_inference(params, state, clamps, structure)
    grads = pc_weight_gradients(params, state, structure, clamps)
    if mode == "none":
        return grads

    hint = dfa_grads(params, structure, batch)
    names = hidden_names(structure)
    last = names[-1]
    # Make the hint as loud as the whisper where the whisper is clear.
    scale = _norm(grads.nodes[last]) / (_norm(hint.nodes[last]) + 1e-30)

    for name in names:
        (wk,), (bk,) = grads.nodes[name].weights.keys(), grads.nodes[name].biases.keys()
        gw, gb = grads.nodes[name].weights[wk], grads.nodes[name].biases[bk]
        hw, hb = hint.nodes[name].weights[wk], hint.nodes[name].biases[bk]
        if mode == "pure":
            new_w, new_b = hw, hb
        else:
            mix = strength * scale
            if mode == "far":
                # Faded = whisper much quieter than at the last hidden layer.
                faded = _norm(grads.nodes[name]) < threshold * _norm(grads.nodes[last])
                mix = jnp.where(faded, mix, 0.0)
            new_w, new_b = gw + mix * hw, gb + mix * hb
        grads = _replace(grads, name, new_w, new_b)
    return grads
