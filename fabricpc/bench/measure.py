"""Timing and memory measurement for one benchmark trial.

Two rules keep the numbers honest:

* The first step includes JIT compilation, so it is timed on its own and
  never mixed into the per-step time.
* JAX hands work to the device and returns right away, so every timed step
  ends with ``block_until_ready``. Without that the stopwatch measures
  nothing.
"""

import statistics
import time
from dataclasses import dataclass
from typing import Optional

import jax

from fabricpc.training import make_train_step
from fabricpc.training.trainer import convert_batch


@dataclass(frozen=True)
class Timing:
    compile_time_s: float
    step_time_ms: float  # median over the timed steps
    timed_steps: int


def time_steps(
    params,
    structure,
    optimizer,
    loader,
    rng_key,
    *,
    algorithm: str,
    warmup_steps: int,
    timed_steps: int,
) -> Timing:
    """Time one training step, the way a benchmark should.

    Runs one compile step, then ``warmup_steps`` untimed steps, then
    ``timed_steps`` timed steps. Batches are taken from ``loader`` and
    reused in a cycle if the loader is shorter than the step count.
    """
    step = make_train_step(structure, optimizer, algorithm=algorithm)
    opt_state = optimizer.init(params)
    batches = [convert_batch(b) for b in loader]
    if not batches:
        raise ValueError("loader yielded no batches")

    def batch_at(i):
        return batches[i % len(batches)]

    keys = jax.random.split(rng_key, 1 + warmup_steps + timed_steps)
    k = 0

    # Compile step, timed separately.
    t0 = time.perf_counter()
    params, opt_state, _, _ = step(params, opt_state, batch_at(0), keys[k])
    jax.block_until_ready(params)
    compile_time_s = time.perf_counter() - t0
    k += 1

    # Warmup: run, do not time.
    for i in range(warmup_steps):
        params, opt_state, _, _ = step(params, opt_state, batch_at(i + 1), keys[k])
        k += 1
    jax.block_until_ready(params)

    # Timed steps, each one synced before the clock stops.
    samples = []
    for i in range(timed_steps):
        t0 = time.perf_counter()
        params, opt_state, _, _ = step(
            params, opt_state, batch_at(i + 1 + warmup_steps), keys[k]
        )
        jax.block_until_ready(params)
        samples.append(time.perf_counter() - t0)
        k += 1

    return Timing(
        compile_time_s=compile_time_s,
        step_time_ms=statistics.median(samples) * 1000.0,
        timed_steps=timed_steps,
    )


def memory_bytes_in_use() -> Optional[int]:
    """Bytes the default device has in use right now, or None if it cannot say.

    GPU devices report this. The CPU device does not, so on a laptop this is
    None and the result file says so instead of guessing.
    """
    jax.block_until_ready(jax.numpy.zeros(1))
    device = jax.local_devices()[0]
    stats = device.memory_stats() if hasattr(device, "memory_stats") else None
    if not stats:
        return None
    return int(stats.get("bytes_in_use", 0))
