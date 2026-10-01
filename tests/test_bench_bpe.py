"""Tests for the BPE Tiny Shakespeare transformer rows (fabricpc.bench.bpe)."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import pytest

from fabricpc.bench.registry import ALGORITHMS, COMPARISONS, ROWS

FAMILY = "tinyshakespeare-bpe-transformer"
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bpe_rows_are_registered_as_a_tier_2_perplexity_family():
    for algo in ALGORITHMS:
        row = ROWS[f"{FAMILY}-{algo}"]
        assert row.id == f"{FAMILY}-{algo}"
        assert row.dataset == "tinyshakespeare-bpe"
        assert row.model == "transformer"
        assert row.algorithm == algo
        assert row.metric == "perplexity"
        assert row.tier == 2
        # examples/transformer_v2_demo.py BPE_DEFAULTS
        assert row.batch_size == 32
        assert row.train_config["num_epochs"] == 5


def test_bpe_family_is_compared_apart_from_the_char_family():
    comparison = COMPARISONS[FAMILY]
    assert comparison.rows == tuple(f"{FAMILY}-{algo}" for algo in ALGORITHMS)
    assert comparison.metric == "perplexity"
    assert "tinyshakespeare-transformer" in COMPARISONS
    assert COMPARISONS["tinyshakespeare-transformer"].rows != comparison.rows


def test_bpe_rows_have_no_expected_score_until_our_first_full_run():
    for algo in ALGORITHMS:
        assert ROWS[f"{FAMILY}-{algo}"].reference is None


def test_bpe_factory_builds_the_demos_four_block_model(rng_key):
    for algo in ALGORITHMS:
        _, structure = ROWS[f"{FAMILY}-{algo}"].model_factory(rng_key)
        # 64 tokens per sequence, 11711 BPE ids per position.
        assert tuple(structure.nodes["logits"].node_info.shape) == (64, 11711)
        assert "L3_mha" in structure.nodes  # depth 4
        assert "L4_mha" not in structure.nodes
        init = structure.nodes["L0_mha"].node_info.weight_init
        assert type(init).__name__ == "NormalInitializer"
        assert init.config["std"] == pytest.approx(0.039890499730518045)


def test_bpe_rows_use_the_demos_bpe_solver():
    from fabricpc.bench import bpe

    for algo in ("spc", "backprop"):
        solver = bpe._solver_for(algo)
        assert type(solver).__name__ == "InferenceSGDNormClip"
        assert solver.config["infer_steps"] == 23
        assert solver.config["eta_infer"] == pytest.approx(0.06558512264378524)
        assert solver.config["max_norm"] == 5.0
    assert type(bpe._solver_for("epc")).__name__ == "EPCInference"


def test_bpe_learning_rate_is_a_cosine_decay_to_a_tenth():
    from fabricpc.bench import bpe

    lr = bpe._lr_schedule(1000)
    peak = 1.676456563307537e-05
    assert float(lr(0)) == pytest.approx(peak)
    assert float(lr(500)) < peak
    assert float(lr(1000)) == pytest.approx(0.1 * peak)


def test_bpe_rows_let_adam_scale_up_tiny_pc_gradients():
    # Same reason as the char transformer rows: Adam epsilon 1e-12.
    def last_step(opt, grad):
        params = {"w": jnp.zeros(4)}
        state = opt.init(params)
        for _ in range(200):
            updates, state = opt.update({"w": jnp.full(4, grad)}, state, params)
        return float(jnp.abs(updates["w"]).max())

    for algo in ALGORITHMS:
        make = ROWS[f"{FAMILY}-{algo}"].optimizer_factory
        assert last_step(make(1000), 1e-10) > 0.5 * last_step(make(1000), 1e-2)


def test_bpe_pc_rows_have_the_safety_net():
    assert ROWS[f"{FAMILY}-spc"].rate_control.target == 1.8
    assert ROWS[f"{FAMILY}-epc"].rate_control.target == 1.0
    assert ROWS[f"{FAMILY}-backprop"].rate_control is None


class _FakeBpe:
    made = []
    vocab = 11711

    def __init__(self, split, seq_len, batch_size, shuffle=True, seed=None, **kw):
        _FakeBpe.made.append((split, seq_len, batch_size, shuffle, seed, kw))
        self.vocab_size = _FakeBpe.vocab


def test_bpe_loaders_read_the_shared_cache_outside_the_repo(monkeypatch, tmp_path):
    from fabricpc.bench import bpe
    from fabricpc.utils.data import dataloader

    monkeypatch.setattr(dataloader, "BpeDataLoader", _FakeBpe)
    monkeypatch.setenv("FABRICPC_BPE_DIR", str(tmp_path / "bpe"))
    _FakeBpe.made = []
    _FakeBpe.vocab = 11711
    train, test = ROWS[f"{FAMILY}-spc"].loader_factory(7)
    tr, te = _FakeBpe.made
    assert tr[:5] == ("train", 64, 32, True, 7)
    assert te[:5] == ("test", 64, 32, False, None)
    for *_, kw in (tr, te):
        assert kw["bpe_data_dir"] == str(tmp_path / "bpe")
        assert kw["vocab_size"] == 11711
        assert kw["verbose"] is False
    assert bpe.bpe_cache_dir() == tmp_path / "bpe"


def test_bpe_cache_defaults_to_the_home_cache_not_the_repo(monkeypatch):
    from fabricpc.bench import bpe

    monkeypatch.delenv("FABRICPC_BPE_DIR", raising=False)
    path = bpe.bpe_cache_dir()
    assert path.is_absolute()
    assert REPO_ROOT not in path.parents
    assert path.parts[-2:] == ("fabricpc", "bpe_tokenized")


def test_bpe_loaders_refuse_a_cache_with_a_different_vocabulary(monkeypatch):
    # The model is built for 11711 ids; an old or odd cache would give the
    # loader another size and every batch would index past the logits.
    from fabricpc.utils.data import dataloader

    monkeypatch.setattr(dataloader, "BpeDataLoader", _FakeBpe)
    _FakeBpe.made = []
    _FakeBpe.vocab = 5000
    with pytest.raises(ValueError, match="11711"):
        ROWS[f"{FAMILY}-epc"].loader_factory(0)


def _tiny_shakespeare_on_disk():
    have = all(
        importlib.util.find_spec(m) for m in ("tokenizers", "tensorflow_datasets")
    )
    folder = Path.home() / "tensorflow_datasets" / "tiny_shakespeare"
    return have and folder.is_dir()


@pytest.mark.skipif(
    not _tiny_shakespeare_on_disk(),
    reason="needs tokenizers, tfds and Tiny Shakespeare already downloaded",
)
def test_real_bpe_loader_gives_the_vocabulary_the_model_expects(monkeypatch, tmp_path):
    # Trains the tokenizer into a scratch folder (a few seconds) and checks
    # the batches the rows will see.
    monkeypatch.setenv("FABRICPC_BPE_DIR", str(tmp_path / "bpe"))
    train, test = ROWS[f"{FAMILY}-backprop"].loader_factory(0)
    assert train.vocab_size == test.vocab_size == 11711
    x, y = next(iter(train))
    assert x.shape == y.shape == (32, 64)
    assert int(x.max()) < 11711
    # About 241k training tokens at about 4 characters per token.
    assert 6_000 < len(train) < 9_000
    assert sorted(p.name for p in (tmp_path / "bpe").iterdir()) == [
        "test.npy",
        "tokenizer.json",
        "train.npy",
        "validation.npy",
    ]
