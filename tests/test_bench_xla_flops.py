"""Tests for FLOPs per update as XLA compiles them (fabricpc.bench.xla_flops)."""

import jax
import pytest

from fabricpc.bench import xla_flops
from fabricpc.bench.measure import training_step
from fabricpc.bench.registry import ROWS
from fabricpc.training.trainer import convert_batch


def _setup(row_id, batch_size=8):
    row = ROWS[row_id]
    key = jax.random.PRNGKey(0)
    params, structure = row.model_factory(key)
    x = jax.random.normal(key, (batch_size, 784))
    y = jax.nn.one_hot(jax.numpy.zeros(batch_size, dtype=int), 10)
    return params, structure, row.optimizer_factory(10), convert_batch((x, y)), key


def _compiled_flops(params, structure, optimizer, batch, key, algorithm):
    step = training_step(structure, optimizer, algorithm=algorithm)
    lowered = step.lower(params, optimizer.init(params), batch, key)
    cost = lowered.compile().cost_analysis()
    return (cost[0] if isinstance(cost, list) else cost)["flops"]


def test_backprop_is_the_compiled_steps_own_count():
    params, structure, opt, batch, key = _setup("mnist-mlp-backprop")
    got = xla_flops.flops_per_update(params, structure, opt, batch, key, "backprop")
    assert got == pytest.approx(
        _compiled_flops(params, structure, opt, batch, key, "backprop")
    )


def test_a_pc_count_is_every_settle_step_not_one():
    # XLA counts a while loop's body once, whatever the trip count, so the
    # plain compiled count is the same at T = 4 and T = 16.
    params, structure, opt, batch, key = _setup("mnist-mlp-spc")
    at = {
        t: xla_flops.flops_per_update(
            params, xla_flops.with_infer_steps(structure, t), opt, batch, key, "pc"
        )
        for t in (4, 8, 16)
    }
    assert at[4] < at[8] < at[16]
    assert at[16] - at[8] == pytest.approx(2 * (at[8] - at[4]), rel=1e-6)


def test_the_extrapolation_matches_a_fully_unrolled_compile():
    params, structure, opt, batch, key = _setup("mnist-mlp-epc")
    s5 = xla_flops.with_infer_steps(structure, 5)
    with xla_flops.unrolled_settle_loop():
        direct = _compiled_flops(params, s5, opt, batch, key, "pc")
    got = xla_flops.flops_per_update(params, s5, opt, batch, key, "pc")
    assert got == pytest.approx(direct, rel=1e-6)


def test_the_settle_loop_is_put_back_afterwards():
    original = jax.lax.fori_loop
    params, structure, opt, batch, key = _setup("mnist-mlp-spc")
    xla_flops.flops_per_update(params, structure, opt, batch, key, "pc")
    assert jax.lax.fori_loop is original


def test_xla_compute_gives_the_count_and_the_ratio_to_backprop():
    params, structure, opt, batch, key = _setup("mnist-mlp-spc")
    pc = xla_flops.xla_compute(params, structure, opt, batch, key, "pc")
    bp = xla_flops.xla_compute(params, structure, opt, batch, key, "backprop")

    assert pc["xla_flops_per_update_backprop"] == pytest.approx(
        bp["xla_flops_per_update"]
    )
    assert pc["xla_pc_to_backprop_ratio"] == pytest.approx(
        pc["xla_flops_per_update"] / bp["xla_flops_per_update"]
    )
    assert bp["xla_pc_to_backprop_ratio"] == 1.0


def test_a_count_that_cannot_be_made_is_left_out_not_a_crash(monkeypatch):
    params, structure, opt, batch, key = _setup("mnist-mlp-spc")

    def broken(*a, **k):
        raise RuntimeError("no cost analysis on this backend")

    monkeypatch.setattr(xla_flops, "_compiled", broken)
    assert xla_flops.xla_compute(params, structure, opt, batch, key, "pc") == {}
