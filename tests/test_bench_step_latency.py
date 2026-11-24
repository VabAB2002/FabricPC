import inspect
import statistics
from unittest.mock import patch

import jax
import optax
import pytest

from fabricpc.bench.run.measure import time_steps
from fabricpc.bench.run.runner import run_trial


def test_default_timed_steps_is_30():
    assert inspect.signature(run_trial).parameters["timed_steps"].default == 30
    assert inspect.signature(time_steps).parameters["timed_steps"].default == 30


def test_time_steps_reports_median_wall_clock_latency():
    # Simulate 30 timed steps with a single large outlier (e.g. GC/OS jitter)
    step_durations_s = [0.010] * 29 + [0.500]

    ticks = []
    now = 0.0
    for d in step_durations_s:
        ticks.append(now)        # t0 at start of timed step
        now += d
        ticks.append(now)        # t1 at end of timed step

    tick_iter = iter(ticks)

    def fake_step(params, opt_state, batch, key):
        return params, opt_state, 0.0, {}

    loader = [({"x": jax.numpy.zeros((2, 4)), "y": jax.numpy.zeros((2, 2))})]
    optimizer = optax.sgd(0.01)
    params = {"w": jax.numpy.zeros((4, 2))}

    with (
        patch("fabricpc.bench.run.measure.make_train_step", return_value=fake_step),
        patch("fabricpc.bench.run.measure.convert_batch", side_effect=lambda b: b),
        patch("fabricpc.bench.run.measure.time.perf_counter", side_effect=lambda: next(tick_iter)),
    ):
        result = time_steps(
            params,
            structure=None,
            optimizer=optimizer,
            loader=loader,
            rng_key=jax.random.PRNGKey(0),
            algorithm="backprop",
            warmup_steps=5,
            timed_steps=30,
        )

    assert result.timed_steps == 30
    expected_median_ms = statistics.median(step_durations_s) * 1000.0
    assert result.step_time_ms == pytest.approx(expected_median_ms)