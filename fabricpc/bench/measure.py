"""Timing and memory measurement for one benchmark trial.

Two rules keep the numbers honest:

* The first step includes JIT compilation, so it is timed on its own and
  never mixed into the per-step time.
* JAX hands work to the device and returns right away, so every timed step
  ends with ``block_until_ready``. Without that the stopwatch measures
  nothing.
"""

import itertools
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Optional

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
    ``timed_steps`` timed steps. Only that many batches are read from
    ``loader``, and they are reused in a cycle if the loader is shorter.
    """
    step = make_train_step(structure, optimizer, algorithm=algorithm)
    opt_state = optimizer.init(params)
    needed = 1 + warmup_steps + timed_steps
    batches = [convert_batch(b) for b in itertools.islice(loader, needed)]
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


@dataclass(frozen=True)
class MemorySnapshot:
    bytes_in_use: Optional[int]  # what is held right now
    peak_bytes: Optional[int]  # the most ever held so far in this process


def memory_snapshot() -> MemorySnapshot:
    """Read the default device's memory numbers, or None where it cannot say.

    GPU devices report both. The CPU device reports nothing, so on a laptop
    both are None and the result file says so instead of guessing.

    The peak counts everything since the process started, compile buffers
    included. Each trial normally gets its own process, so the peak is that
    trial's alone; with ``--in-process`` a later trial's peak can carry over
    from an earlier, bigger one.
    """
    jax.block_until_ready(jax.numpy.zeros(1))
    device = jax.local_devices()[0]
    stats = device.memory_stats() if hasattr(device, "memory_stats") else None
    if not stats:
        return MemorySnapshot(bytes_in_use=None, peak_bytes=None)

    def read(key):
        return int(stats[key]) if key in stats else None

    return MemorySnapshot(
        bytes_in_use=read("bytes_in_use"), peak_bytes=read("peak_bytes_in_use")
    )


def step_memory(
    params, structure, optimizer, loader, rng_key, *, algorithm: str
) -> Optional[Dict[str, int]]:
    """Memory one compiled training step needs, as reported by XLA.

    The process-wide peak from ``memory_snapshot`` also counts scratch space
    used while compiling (on GPU, cuDNN tries several convolution algorithms
    in temporary buffers), and that depends only on the layer shapes, so it
    comes out the same for backprop, sPC and ePC. The compiled step's own
    sizes are what actually differ: its inputs (params, optimizer state,
    batch), its outputs, and the temporary buffers it needs while running.

    Compiles the step once more, so call it after timing. Returns None when
    the step cannot be analysed (for example a step that is not jitted).
    """
    step = make_train_step(structure, optimizer, algorithm=algorithm)
    batch = convert_batch(next(iter(loader)))
    try:
        compiled = step.lower(params, optimizer.init(params), batch, rng_key).compile()
        analysis = compiled.memory_analysis()
    except (AttributeError, NotImplementedError, TypeError):
        return None
    if analysis is None:
        return None
    sizes = {
        "argument_bytes": int(analysis.argument_size_in_bytes),
        "output_bytes": int(analysis.output_size_in_bytes),
        "temp_bytes": int(analysis.temp_size_in_bytes),
        "alias_bytes": int(analysis.alias_size_in_bytes),
    }
    sizes["total_bytes"] = (
        sizes["argument_bytes"]
        + sizes["output_bytes"]
        + sizes["temp_bytes"]
        - sizes["alias_bytes"]
    )
    return sizes
