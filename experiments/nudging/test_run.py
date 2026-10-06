"""Tests for the nudging training loop, on tiny fake data."""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np

from experiments.nudging.configs import CONFIGS
from experiments.nudging.data import DropLast
from experiments.nudging.run import run_seed
from experiments.nudging.schedule import beta_for_epoch, cn_factors


class _List:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _fake_loaders(n_train=3, batch=8):
    key = jax.random.PRNGKey(0)
    batches = []
    for i in range(n_train):
        k1, k2, key = jax.random.split(key, 3)
        x = np.asarray(jax.random.normal(k1, (batch, 16, 16, 3)))
        y = np.asarray(jax.nn.one_hot(jax.random.randint(k2, (batch,), 0, 4), 4))
        batches.append((x, y))
    return _List(batches), _List(batches[:1])


def _tiny(name):
    return dataclasses.replace(
        CONFIGS[name], input_shape=(16, 16, 3), num_classes=4, T=2
    )


def test_a_seed_writes_its_json_with_best_and_last(tmp_path):
    cfg = _tiny("c10-nn")
    out = run_seed(cfg, 0, tmp_path, epochs=2, loaders=_fake_loaders())
    saved = json.loads((tmp_path / cfg.name / "seed0.json").read_text())
    assert saved == json.loads(json.dumps(out))
    for key in ("best_acc", "best_epoch", "last_acc", "curve", "config", "device"):
        assert key in saved
    assert len(saved["curve"]) == 2
    assert saved["best_acc"] >= saved["last_acc"]
    assert saved["best_epoch"] in (1, 2)


def test_each_epoch_records_the_pcx_beta(tmp_path):
    cfg = _tiny("c10-nn")
    out = run_seed(cfg, 0, tmp_path, epochs=2, loaders=_fake_loaders())
    want = [beta_for_epoch(cfg.beta0, cfg.beta_ir, cfg.factor, e) for e in range(2)]
    assert [p["beta"] for p in out["curve"]] == want


def test_centered_nudging_uses_the_seeded_signs(tmp_path):
    cfg = _tiny("c10-cn")
    out = run_seed(cfg, 3, tmp_path, epochs=2, loaders=_fake_loaders())
    signs = cn_factors(3, 2)
    want = [beta_for_epoch(cfg.beta0, cfg.beta_ir, signs[e], e) for e in range(2)]
    assert [p["beta"] for p in out["curve"]] == want


def test_a_finished_seed_is_skipped(tmp_path):
    cfg = _tiny("c10-se")
    run_seed(cfg, 0, tmp_path, epochs=1, loaders=_fake_loaders())
    path = tmp_path / cfg.name / "seed0.json"
    before = path.stat().st_mtime_ns
    run_seed(cfg, 0, tmp_path, epochs=1, loaders=_fake_loaders())
    assert path.stat().st_mtime_ns == before


def test_fractional_epoch_runs_part_of_the_batches(tmp_path):
    cfg = _tiny("c10-se")
    out = run_seed(cfg, 0, tmp_path, epochs=0.4, loaders=_fake_loaders(n_train=5))
    assert out["curve"][0]["batches"] == 2


def test_drop_last_skips_the_short_batch():
    full = (np.zeros((8, 2)), np.zeros((8, 3)))
    short = (np.zeros((3, 2)), np.zeros((3, 3)))
    loader = DropLast(_List([full, full, short]), batch_size=8)
    assert len(loader) == 2
    assert [b[0].shape[0] for b in loader] == [8, 8]


def test_configs_hold_the_pcx_yaml_values():
    c = CONFIGS["c10-nn"]
    assert (c.T, c.beta0, c.factor, c.beta_ir) == (7, 0.17877062530418192, -1.0, 0.02)
    assert (c.lr_x, c.momentum) == (0.028016192698375384, 0.5)
    assert (c.lr_w, c.wd) == (0.000261833244376427, 0.0008992487435200682)
    assert CONFIGS["c10-cn"].centered and not c.centered
    assert CONFIGS["tiny-nn"].input_shape == (56, 56, 3)
    assert CONFIGS["tiny-nn"].target == 46.40
    assert jnp.isclose(CONFIGS["c100-cn"].target, 67.19)
