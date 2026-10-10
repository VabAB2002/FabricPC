"""Tests for how time_steps reads the training data."""

import pytest

from fabricpc.bench import measure
from fabricpc.bench.registry import ROWS
from tests.test_bench_runner import fake_mnist_loaders


class CountingLoader:
    """Wraps a loader and counts how many batches were taken from it."""

    def __init__(self, loader):
        self._loader = loader
        self.taken = 0

    def __iter__(self):
        for batch in self._loader:
            self.taken += 1
            yield batch


def test_timing_reads_only_the_batches_it_needs(rng_key):
    row = ROWS["mnist-mlp-backprop"]
    train, _ = fake_mnist_loaders(rng_key, n_batches=20)
    counting = CountingLoader(train)
    params, structure = row.model_factory(rng_key)

    timing = measure.time_steps(
        params,
        structure,
        row.optimizer_factory(4),
        counting,
        rng_key,
        algorithm="backprop",
        warmup_steps=1,
        timed_steps=2,
    )

    assert counting.taken == 4  # 1 compile + 1 warmup + 2 timed, not all 20
    assert timing.timed_steps == 2
    assert timing.step_time_ms > 0


def _memory_of_training_step(row, params, structure, optimizer, batch, key):
    """What XLA says the step train() really runs needs, for comparison."""
    from fabricpc.training.trainer import _make_step

    step = _make_step(
        structure, optimizer, algorithm="pc", with_state=False, donate=True
    )
    opt_state = optimizer.init(params)
    return measure._memory_of(step.lower(params, opt_state, batch, key).compile())


def test_timing_measures_the_step_training_really_runs(rng_key):
    # train() drops the settled GraphState inside its step. A step that
    # returns it keeps work alive that training never does, so timing and
    # memory have to come from the same program training runs.
    from fabricpc.training.trainer import convert_batch

    row = ROWS["mnist-mlp-spc"]
    train, _ = fake_mnist_loaders(rng_key, n_batches=4)
    params, structure = row.model_factory(rng_key)
    optimizer = row.optimizer_factory(4)
    batch = convert_batch(next(iter(train)))

    timing = measure.time_steps(
        params,
        structure,
        optimizer,
        train,
        rng_key,
        algorithm="pc",
        warmup_steps=1,
        timed_steps=1,
    )

    expected = _memory_of_training_step(
        row, params, structure, optimizer, batch, rng_key
    )
    assert timing.step_memory == expected


def test_timing_leaves_the_callers_params_usable(rng_key):
    # The training step donates its inputs, so timing must work on a copy.
    import jax

    row = ROWS["mnist-mlp-spc"]
    train, _ = fake_mnist_loaders(rng_key, n_batches=4)
    params, structure = row.model_factory(rng_key)
    before = jax.tree_util.tree_map(lambda x: x.copy(), params)

    measure.time_steps(
        params,
        structure,
        row.optimizer_factory(4),
        train,
        rng_key,
        algorithm="pc",
        warmup_steps=1,
        timed_steps=1,
    )

    same = jax.tree_util.tree_map(lambda a, b: bool((a == b).all()), params, before)
    assert all(jax.tree_util.tree_leaves(same))


def test_warmup_and_compile_stay_out_of_the_step_time(monkeypatch, rng_key):
    # A fake step that is slow for its first few calls, the way compiling
    # and first-run costs are, then quick. A fake clock moves by exactly each
    # step's length, so the answer does not depend on how busy the machine is.
    import types

    now = [0.0]
    calls = []

    def fake_training_step(structure, optimizer, *, algorithm):
        def step(params, opt_state, batch, key):
            calls.append(1)
            now[0] += 0.2 if len(calls) <= 3 else 0.01
            return params, opt_state, {}

        return step  # not jitted, so it is used as it is

    monkeypatch.setattr(measure, "training_step", fake_training_step)
    monkeypatch.setattr(
        measure, "time", types.SimpleNamespace(perf_counter=lambda: now[0])
    )
    row = ROWS["mnist-mlp-backprop"]
    train, _ = fake_mnist_loaders(rng_key, n_batches=4)
    params, structure = row.model_factory(rng_key)

    timing = measure.time_steps(
        params,
        structure,
        row.optimizer_factory(4),
        train,
        rng_key,
        algorithm="backprop",
        warmup_steps=2,
        timed_steps=5,
    )

    assert len(calls) == 1 + 2 + 5  # first call, warmup, timed
    assert timing.step_time_ms == pytest.approx(10.0)  # not the 200 ms ones
