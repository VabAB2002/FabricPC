"""FLOPs per update as XLA compiles them.

``fabricpc.bench.compute`` counts matmuls by hand: ``2*E*T + E`` for
predictive coding, ``3*E`` for backprop. That is the closed form the
sponsor's issue uses, but it is not the arithmetic that runs. Checked
against XLA's own count, it was 8 times too high on the MNIST MLP's sPC
row (XLA computes the first layer's prediction of a clamped input once,
not at every settle step), 13% high on VGG-5 sPC and 26 to 37% low on the
transformer rows (attention and layer norm work outside the counted
matmuls). Backprop never needs the input's gradient, so it does fewer
than 3 passes.

So every trial also asks XLA. One catch: XLA counts a loop's body once,
whatever its trip count, so the compiled count of a PC step is the same at
T = 4 and T = 16. The settle loop (``jax.lax.fori_loop`` in
``fabricpc.core.inference``) is therefore unrolled while two small steps,
T = 1 and T = 2, are compiled for counting only, and the count is
extended in a straight line to the row's T. A fully unrolled compile at
T = 5 gives the same number.

Any count that cannot be made is left out (an empty dict), not an error:
the trial's scores do not depend on it.
"""

from contextlib import contextmanager
from typing import Dict, Optional

import jax

from fabricpc.bench.measure import training_step


def _unrolled_fori_loop(lower, upper, body_fun, init_val, **_):
    value = init_val
    for i in range(int(lower), int(upper)):
        value = body_fun(i, value)
    return value


@contextmanager
def unrolled_settle_loop():
    """Trace ``jax.lax.fori_loop`` as a plain Python loop, then put it back."""
    original = jax.lax.fori_loop
    jax.lax.fori_loop = _unrolled_fori_loop
    try:
        yield
    finally:
        jax.lax.fori_loop = original


def infer_steps(structure) -> Optional[int]:
    inference = structure.config.get("inference")
    steps = (getattr(inference, "config", None) or {}).get("infer_steps")
    return int(steps) if steps is not None else None


def with_infer_steps(structure, steps: int):
    """The same graph with its settle run for ``steps`` steps."""
    inference = structure.config["inference"]
    rebuilt = type(inference)(**{**inference.config, "infer_steps": steps})
    return structure._replace(config={**structure.config, "inference": rebuilt})


def _compiled(params, structure, optimizer, batch, key, algorithm) -> float:
    step = training_step(structure, optimizer, algorithm=algorithm)
    lowered = step.lower(params, optimizer.init(params), batch, key)
    cost = lowered.compile().cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    return float(cost["flops"])


def flops_per_update(params, structure, optimizer, batch, key, algorithm) -> float:
    """XLA's FLOPs for one weight update on ``batch``.

    ``algorithm`` is the trainer's: ``"pc"`` or ``"backprop"``.
    """
    if algorithm == "backprop":
        return _compiled(params, structure, optimizer, batch, key, algorithm)
    steps = infer_steps(structure)
    if steps is None:
        raise ValueError("the graph's inference has no infer_steps to count")
    with unrolled_settle_loop():
        one = _compiled(
            params, with_infer_steps(structure, 1), optimizer, batch, key, algorithm
        )
        two = _compiled(
            params, with_infer_steps(structure, 2), optimizer, batch, key, algorithm
        )
    return one + (steps - 1) * (two - one)


def xla_compute(params, structure, optimizer, batch, key, algorithm) -> Dict:
    """``xla_flops_per_update``, the same graph's backprop count, and their
    ratio; empty if XLA cannot say."""
    try:
        flops = flops_per_update(params, structure, optimizer, batch, key, algorithm)
        backprop = (
            flops
            if algorithm == "backprop"
            else flops_per_update(params, structure, optimizer, batch, key, "backprop")
        )
    except Exception:  # noqa: BLE001 - a missing number, not a failed trial
        return {}
    return {
        "xla_flops_per_update": flops,
        "xla_flops_per_update_backprop": backprop,
        "xla_pc_to_backprop_ratio": flops / backprop,
    }
