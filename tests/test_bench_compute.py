"""Tests for the analytic compute count: how many matrix multiplies one
weight update costs under each algorithm."""

import pytest

from fabricpc.bench.compute import count_compute
from fabricpc.bench.registry import ROWS

# The MNIST MLP: three weight matrices, 784x256, 256x64, 64x10.
MLP_WEIGHTS = 784 * 256 + 256 * 64 + 64 * 10


def build(row_id, key):
    return ROWS[row_id].model_factory(key)


def test_chain_mlp_matches_the_closed_form_for_pc(rng_key):
    # Sponsor's formula for a chain of D layers with T settling steps:
    # PC costs 2*D*T + D matmuls per update, backprop costs 3*D.
    params, structure = build("mnist-mlp-spc", rng_key)
    c = count_compute(params, structure, batch_size=200, algorithm="spc")
    assert c.weighted_edges == 3
    assert c.infer_steps == 20  # read from the graph's solver
    assert c.matmuls_per_update == 2 * 3 * 20 + 3
    assert c.matmuls_per_update_backprop == 3 * 3
    assert c.pc_to_backprop_ratio == pytest.approx(123 / 9)


def test_backprop_row_has_ratio_one(rng_key):
    params, structure = build("mnist-mlp-backprop", rng_key)
    c = count_compute(params, structure, batch_size=200, algorithm="backprop")
    assert c.matmuls_per_update == 9
    assert c.pc_to_backprop_ratio == 1.0


def test_epc_row_uses_its_own_step_count(rng_key):
    params, structure = build("mnist-mlp-epc", rng_key)
    c = count_compute(params, structure, batch_size=200, algorithm="epc")
    assert c.infer_steps == 5
    assert c.matmuls_per_update == 2 * 3 * 5 + 3


def test_flops_count_every_weight_matrix_times_the_batch(rng_key):
    # One pass over a Linear edge is 2 * batch * fan_in * fan_out flops.
    params, structure = build("mnist-mlp-backprop", rng_key)
    c = count_compute(params, structure, batch_size=200, algorithm="backprop")
    one_pass = 2 * 200 * MLP_WEIGHTS
    assert c.flops_per_pass == one_pass
    assert c.flops_per_update == 3 * one_pass


def test_infer_steps_can_be_overridden(rng_key):
    params, structure = build("mnist-mlp-spc", rng_key)
    c = count_compute(params, structure, batch_size=8, algorithm="spc", infer_steps=1)
    assert c.matmuls_per_update == c.matmuls_per_update_backprop
