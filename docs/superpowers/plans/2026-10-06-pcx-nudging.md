# pcx Nudging Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce pcx's nudged VGG-5 results (issue #59 targets) inside FabricPC: CIFAR-10 CN 89.47, CIFAR-100 CN 67.19, Tiny-ImageNet NN 46.40, plus the fully-specified checks PC-SE 87.98 / PN 88.42 / NN 88.83 on CIFAR-10.

**Architecture:** An experiment package `experiments/nudging/` on branch `vishal/nudging-experiment` with its own training loop (like `experiments/hint/`), so FabricPC's trainer and the benchmark stay untouched. It reuses FabricPC's graph, state init, energies and `pc_weight_gradients`, and adds (a) a squared-error VGG head option, (b) a momentum state optimizer matching pcx's `optax.sgd(lr_x, momentum)` reset per batch, (c) nudged batch gradients `h = u − β(u − y)` with weight grads × 1/β, (d) pcx's per-epoch β schedule including its quirks, (e) pcx's exact configs, data pipeline, schedule and reporting (best epoch per seed; mean of all seeds and of the top 5 of 7).

**Tech Stack:** JAX 0.10.2, optax 0.2.8, FabricPC (this repo), pytest. Mac CPU for tests/smoke; Colab T4 for full runs.

**Spec:** Research report (pcx `build-docs` branch: `examples/s4_1_discriminative_mode/VGG5/{cifar10,cifar100,tinyimagenet}/{PC_SE.py,*.yaml,run_exps.sh}`, `seed.py`; arXiv 2407.01163 v2 Table 1 and App. B). Key facts copied into Global Constraints below.

## Global Constraints

- Work only on branch `vishal/nudging-experiment`; never push to `upstream`; ask the user before every commit/push and before any paid GPU run.
- Code and comments in plain student-style English; no AI mentions; black + ruff clean; TDD.
- pcx nudged runs use the **SE head**: Linear output with **no activation** and `GaussianEnergy` (0.5·(h−u)²); `u` = logits.
- Output target set once after the feedforward init: `h = u − β(u − y)`, y one-hot; output stays clamped for all T steps.
- Weight gradients multiplied by `1/β` (β < 0 flips sign and scales by 1/|β|).
- β per epoch: `β = factor·(β0 + β_ir·e)`; if `|β| ≥ 1` then `β = +1.0` (pcx's quirk, even for factor −1).
- CN (no pcx code exists): `factor` drawn ±1 with probability ½ each epoch from a seeded RNG; hyperparameters = the dataset's NN yaml.
- State optimizer: SGD with momentum (`v = m·v + g; z ← z − lr_x·v`), velocity reset to 0 every batch; per-sample energy gradients (FabricPC's `latent_grad` already is per-sample).
- Weight optimizer: `optax.adamw(schedule, weight_decay=wd)` (default eps 1e-8), schedule `warmup_cosine_decay_schedule(init=lr, peak=1.1·lr, warmup=0.1·total, decay=total, end=0.1·lr)`.
- No safety net (rate control) in these runs — pcx has none.
- Data: batch 128, `drop_last=True`, 50 epochs. CIFAR flip p=.5 + crop pad 4; normalize CIFAR-10 mean (0.4914,0.4822,0.4465) std (0.2023,0.1994,0.2010); CIFAR-100 mean (0.5071,0.4867,0.4408) std (0.2675,0.2565,0.2761); Tiny: flip + crop 56 no pad, test centre crop 56, ImageNet mean/std.
- Score = forward-pass test accuracy (argmax of logits) after every epoch; per seed keep best and last epoch. Report: mean±std over all seeds of best, mean±std of top-5-of-7 best (pcx `avg5`), mean±std of last.
- pcx configs (verbatim from yamls):

| name | data | T | act | β0 / factor / β_ir | lr_x / mom | lr_w / wd | pcx target |
|---|---|---|---|---|---|---|---|
| c10-se | CIFAR-10 | 7 | gelu | 1.0 / +1 / 0 | 0.02348681827141059 / 0.3 | 0.0002657691155688603 / 0.0066559627972820395 | 87.98 |
| c10-pn | CIFAR-10 | 7 | gelu | 0.32297201469734277 / +1 / 0.02 | 0.03623356587254243 / 0.1 | 0.00029968925891952494 / 3.728852110891977e-05 | 88.42 |
| c10-nn | CIFAR-10 | 7 | gelu | 0.17877062530418192 / −1 / 0.02 | 0.028016192698375384 / 0.5 | 0.000261833244376427 / 0.0008992487435200682 | 88.83 |
| c10-cn | CIFAR-10 | as c10-nn, factor random ±1 per epoch | | | | | 89.47 |
| c100-nn | CIFAR-100 | 12 | gelu | 0.15 / −1 / 0.02 | 0.03325378866925405 / 0.25 | 7.67519702690274e-05 / 0.004202804226490381 | 65.46 |
| c100-cn | CIFAR-100 | as c100-nn, factor random ±1 per epoch | | | | | 67.19 |
| tiny-nn | Tiny-ImageNet | 8 | gelu | 0.5 / −1 / 0.0 | 0.0573599016772709 / 0.1 | 0.0001376789472968569 / 1.6046037514265988e-05 | 46.40 |

## Review Focus

1. β = 1 exactly must reproduce plain clamped PC (h = y) — otherwise every nudged result is suspect.
2. Negative β must flip the weight-gradient sign (a missed `1/β` silently trains the wrong direction and still "learns" a bit).
3. The `|β| ≥ 1 → +1.0` clamp must apply after the sign (pcx's quirk) — reproduce it, don't "fix" it.
4. Momentum velocity must reset every batch and be zero-initialised; momentum 0 must equal FabricPC's `InferenceSGD` exactly.
5. Best-epoch reporting must use the test set exactly as pcx does (forward argmax), and the summary must label best vs last clearly so nobody compares best-epoch numbers with the benchmark's last-epoch numbers.

---

### Task 1: Squared-error head for `create_vgg`

**Files:**
- Modify: `fabricpc/models/vgg.py` (`create_vgg` signature and output node)
- Test: `tests/test_vgg.py`

**Interfaces:**
- Produces: `create_vgg(..., head: str = "ce")`; `head="se"` → output `Linear(..., activation=IdentityActivation(), energy=GaussianEnergy())`; `head="ce"` unchanged; any other value raises `ValueError`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_vgg.py`)

```python
def test_vgg_se_head_has_linear_output_and_gaussian_energy():
    from fabricpc.core.activations import IdentityActivation
    from fabricpc.core.energy import GaussianEnergy
    from fabricpc.core.inference import InferenceSGD
    from fabricpc.models.vgg import create_vgg

    s = create_vgg(5, input_shape=(32, 32, 3), num_classes=10,
                   inference=InferenceSGD(0.01, 2), fuse_pool=True, head="se")
    out = s.nodes["class"].node_info
    assert isinstance(out.activation, IdentityActivation)
    assert isinstance(out.energy, GaussianEnergy)


def test_vgg_ce_head_is_still_the_default():
    from fabricpc.core.activations import SoftmaxActivation
    from fabricpc.core.inference import InferenceSGD
    from fabricpc.models.vgg import create_vgg

    s = create_vgg(5, input_shape=(32, 32, 3), num_classes=10,
                   inference=InferenceSGD(0.01, 2), fuse_pool=True)
    assert isinstance(s.nodes["class"].node_info.activation, SoftmaxActivation)


def test_vgg_rejects_an_unknown_head():
    import pytest
    from fabricpc.core.inference import InferenceSGD
    from fabricpc.models.vgg import create_vgg

    with pytest.raises(ValueError):
        create_vgg(5, input_shape=(32, 32, 3), num_classes=10,
                   inference=InferenceSGD(0.01, 2), head="mse")
```

Before writing, check how node_info exposes activation/energy (`grep -n "activation\|energy" fabricpc/nodes/base.py`) and adjust the attribute path in the tests to the real one.

- [ ] **Step 2: Run, expect FAIL** — `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_vgg.py -k head` → unexpected keyword `head`.
- [ ] **Step 3: Implement** — add `head: str = "ce"` to `create_vgg`; validate `head in ("ce", "se")`; build the output with `activation=SoftmaxActivation(), energy=CrossEntropyEnergy()` for "ce" and `activation=IdentityActivation(), energy=GaussianEnergy()` for "se"; document it in the docstring ("se: linear logits with squared error, as pcx's nudged runs use").
- [ ] **Step 4: Run** the three tests plus all of `tests/test_vgg.py` → PASS.
- [ ] **Step 5: black/ruff; stage** (commit only after the user approves).

---

### Task 2: Momentum state optimizer (pcx's settle)

**Files:**
- Create: `experiments/nudging/__init__.py`, `experiments/nudging/settle.py`
- Test: `experiments/nudging/test_settle.py`

**Interfaces:**
- Produces: `settle(params, state, clamps, structure, *, lr_x: float, momentum: float, steps: int) -> GraphState`. Each step: `zero_grads` → `forward_value_and_grad` (classmethods on `InferenceBase`, use `InferenceSGD`) → for each unclamped node `v = momentum·v + latent_grad; z_latent ← z_latent − lr_x·v`. Velocities start at zero (fresh per call = per batch). Implemented with `jax.lax.fori_loop` carrying `(state, velocities)`. After the loop, run one more `forward_value_and_grad` so `z_mu`/`error` match the final latents (same as FabricPC's solvers leave them; check `InferenceSGD.run_inference`/`finalize_state` and mirror it).

- [ ] **Step 1: Failing tests**

```python
import jax, jax.numpy as jnp, numpy as np
from experiments.nudging.settle import settle
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.models.vgg import create_vgg


def _tiny(steps):
    s = create_vgg(5, input_shape=(8, 8, 3), num_classes=4,
                   inference=InferenceSGD(eta_infer=0.05, infer_steps=steps),
                   fuse_pool=True, head="se")
    p = initialize_params(s, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 3))
    y = jax.nn.one_hot(jnp.array([1, 3]), 4)
    clamps = {"input": x, "class": y}
    st = initialize_graph_state(s, 2, jax.random.PRNGKey(2), clamps=clamps, params=p)
    return s, p, st, clamps


def test_momentum_zero_equals_fabricpc_sgd():
    s, p, st, clamps = _tiny(5)
    ref = run_inference(p, st, clamps, s)
    out = settle(p, st, clamps, s, lr_x=0.05, momentum=0.0, steps=5)
    for n in s.nodes:
        np.testing.assert_allclose(out.nodes[n].z_latent, ref.nodes[n].z_latent, atol=1e-6)


def test_momentum_changes_the_path_but_not_the_clamps():
    s, p, st, clamps = _tiny(5)
    a = settle(p, st, clamps, s, lr_x=0.05, momentum=0.0, steps=5)
    b = settle(p, st, clamps, s, lr_x=0.05, momentum=0.5, steps=5)
    assert float(jnp.abs(a.nodes["conv1"].z_latent - b.nodes["conv1"].z_latent).max()) > 1e-7
    np.testing.assert_array_equal(b.nodes["class"].z_latent, clamps["class"])


def _grad_at(p, st, clamps, s, node):
    st = InferenceSGD.zero_grads(p, st, clamps, s)
    st = InferenceSGD.forward_value_and_grad(p, st, clamps, s)
    return st.nodes[node].latent_grad


def test_two_momentum_steps_match_hand_computation_on_one_node():
    # v1 = g1, z1 = z0 - lr*g1 ; v2 = m*v1 + g2, z2 = z1 - lr*v2
    from fabricpc.core.state_ops import update_node_in_state

    s, p, st, clamps = _tiny(2)
    lr, m = 0.05, 0.5
    one = settle(p, st, clamps, s, lr_x=lr, momentum=m, steps=1)
    g1 = _grad_at(p, st, clamps, s, "conv1")
    np.testing.assert_allclose(one.nodes["conv1"].z_latent,
                               st.nodes["conv1"].z_latent - lr * g1, atol=1e-6)
    # g2 is the gradient at the state after one step (all nodes moved once).
    g2 = _grad_at(p, one, clamps, s, "conv1")
    two = settle(p, st, clamps, s, lr_x=lr, momentum=m, steps=2)
    expected = one.nodes["conv1"].z_latent - lr * (m * g1 + g2)
    np.testing.assert_allclose(two.nodes["conv1"].z_latent, expected, atol=1e-5)
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError`).
- [ ] **Step 3: Implement `settle`.**
- [ ] **Step 4: Run** → PASS (`PYTHONPATH=. .venv/bin/python -m pytest -q experiments/nudging`).
- [ ] **Step 5: black/ruff; stage.**

---

### Task 3: Nudged batch gradients

**Files:**
- Create: `experiments/nudging/nudge.py`
- Test: `experiments/nudging/test_nudge.py`

**Interfaces:**
- Consumes: `settle` (Task 2), `create_vgg(head="se")` (Task 1).
- Produces: `nudged_grads(params, batch, structure, key, *, beta, lr_x, momentum, steps) -> (grads, metrics)`. Steps: `clamps = build_clamps(batch, structure, clamp_target=True)`; `state = initialize_graph_state(..., clamps=clamps, params=params)`; `u = state.nodes[out].z_mu` (out = `structure.task_map["y"]`); `h = jnp.where(beta == 1.0, y, u - beta * (u - y))`; `clamps_n = {**clamps, out: h}`; `state = set_latents_to_clamps(state, clamps_n)`; `state = settle(...)`; `grads = pc_weight_gradients(params, state, structure, clamps_n)`; `grads = tree_map(lambda g: g / beta, grads)`. `beta` is a traced scalar (jnp array) so the jitted function is not recompiled per epoch. Metrics: `{"energy": graph_energy(state, structure)/denom}`.

- [ ] **Step 1: Failing tests**

```python
import jax, jax.numpy as jnp, numpy as np
from experiments.nudging.nudge import nudged_grads
from experiments.nudging.settle import settle  # noqa: F401
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph_initialization import initialize_params
from fabricpc.models.vgg import create_vgg
from fabricpc.training.trainer import _batch_grads

def _setup(steps=3):
    s = create_vgg(5, input_shape=(8, 8, 3), num_classes=4,
                   inference=InferenceSGD(eta_infer=0.05, infer_steps=steps),
                   fuse_pool=True, head="se")
    p = initialize_params(s, jax.random.PRNGKey(0))
    batch = {"x": jax.random.normal(jax.random.PRNGKey(1), (4, 8, 8, 3)),
             "y": jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 4)}
    return s, p, batch

def _flat(g):
    return jnp.concatenate([w.ravel() for n in g.nodes.values() for w in n.weights.values()])

def test_beta_one_equals_plain_clamped_pc():
    s, p, b = _setup()
    ours, _ = nudged_grads(p, b, s, jax.random.PRNGKey(5), beta=jnp.float32(1.0),
                           lr_x=0.05, momentum=0.0, steps=3)
    ref = _batch_grads(p, b, s, jax.random.PRNGKey(5), algorithm="pc")[0]
    np.testing.assert_allclose(_flat(ours), _flat(ref), rtol=1e-5, atol=1e-7)

def test_output_stays_at_the_nudged_target():
    from fabricpc.graph_initialization.state_initializer import initialize_graph_state
    from fabricpc.training.trainer import build_clamps

    s, p, b = _setup()
    beta = jnp.float32(0.3)
    _, _, final = nudged_grads(p, b, s, jax.random.PRNGKey(5), beta=beta, lr_x=0.05,
                               momentum=0.0, steps=3, return_state=True)
    clamps = build_clamps(b, s, clamp_target=True)
    init = initialize_graph_state(s, 4, jax.random.PRNGKey(5), clamps=clamps, params=p)
    u = init.nodes["class"].z_mu
    h = u - beta * (u - b["y"])
    np.testing.assert_allclose(final.nodes["class"].z_latent, h, atol=1e-6)

def test_negative_beta_flips_and_scales_at_zero_steps():
    s, p, b = _setup(steps=0)
    gp, _ = nudged_grads(p, b, s, jax.random.PRNGKey(5), beta=jnp.float32(0.2), lr_x=0.05, momentum=0.0, steps=0)
    gn, _ = nudged_grads(p, b, s, jax.random.PRNGKey(5), beta=jnp.float32(-0.2), lr_x=0.05, momentum=0.0, steps=0)
    np.testing.assert_allclose(_flat(gp), _flat(gn), rtol=1e-5, atol=1e-8)  # 1/beta scaling makes them equal

def test_small_beta_limit_is_finite():
    s, p, b = _setup()
    g1, _ = nudged_grads(p, b, s, jax.random.PRNGKey(5), beta=jnp.float32(1e-3), lr_x=0.05, momentum=0.0, steps=3)
    g2, _ = nudged_grads(p, b, s, jax.random.PRNGKey(5), beta=jnp.float32(2e-3), lr_x=0.05, momentum=0.0, steps=3)
    assert jnp.isfinite(_flat(g1)).all()
    np.testing.assert_allclose(_flat(g1), _flat(g2), rtol=5e-2, atol=1e-6)
```

`nudged_grads(..., return_state=True)` returns `(grads, metrics, final_state)`; default `False` returns `(grads, metrics)`.

- [ ] **Step 2: Run, expect FAIL.**  **Step 3: Implement.**  **Step 4: Run → PASS.**  **Step 5: black/ruff; stage.**

---

### Task 4: β schedule (pcx quirks included)

**Files:**
- Create: `experiments/nudging/schedule.py`
- Test: `experiments/nudging/test_schedule.py`

**Interfaces:**
- Produces: `beta_for_epoch(beta0: float, beta_ir: float, factor: float, epoch: int) -> float` (pcx formula incl. `|β|≥1 → +1.0`); `cn_factors(seed: int, n_epochs: int) -> list[float]` (±1 per epoch, seeded `numpy.random.default_rng(seed)`, p=½).

- [ ] **Step 1: Failing tests**

```python
from experiments.nudging.schedule import beta_for_epoch, cn_factors

def test_c10_nn_schedule_matches_pcx_including_the_plus_one_quirk():
    b0, ir = 0.17877062530418192, 0.02
    assert abs(beta_for_epoch(b0, ir, -1.0, 0) - (-0.17877062530418192)) < 1e-12
    assert abs(beta_for_epoch(b0, ir, -1.0, 41) - (-(b0 + 0.82))) < 1e-12
    assert beta_for_epoch(b0, ir, -1.0, 42) == 1.0   # |beta| >= 1 -> +1.0 even when negative
    assert beta_for_epoch(b0, ir, -1.0, 49) == 1.0

def test_plain_se_is_always_one():
    assert all(beta_for_epoch(1.0, 0.0, 1.0, e) == 1.0 for e in range(50))

def test_cn_factors_are_seeded_and_both_signs():
    a, b = cn_factors(0, 50), cn_factors(0, 50)
    assert a == b and set(a) == {1.0, -1.0}
    assert cn_factors(1, 50) != a
```

- [ ] **Steps 2–5:** fail → implement → pass → black/ruff; stage.

---

### Task 5: Configs, data, training loop, per-epoch scoring

**Files:**
- Create: `experiments/nudging/configs.py`, `experiments/nudging/data.py`, `experiments/nudging/run.py`
- Test: `experiments/nudging/test_run.py`

**Interfaces:**
- `configs.CONFIGS: dict[str, NudgeConfig]` with frozen dataclass `NudgeConfig(name, dataset, num_classes, input_shape, T, activation, beta0, factor, beta_ir, centered, lr_x, momentum, lr_w, wd, target, epochs=50, batch_size=128)` holding the Global Constraints table verbatim (`centered=True` for `*-cn`; activation `"gelu"` → `GeluActivation()`).
- `data.loaders(cfg, seed) -> (train, test)`: CIFAR via `dataloader.Cifar10Loader/Cifar100Loader(..., normalize_mean=pcx, normalize_std=pcx)` wrapped in `AugmentedImageLoader(flip_prob=.5, crop_pad=4, seed=seed)`; Tiny via the existing Tiny-ImageNet loader with `crop_size=56`/centre crop as `fabricpc/bench/tinyimagenet.py` does, ImageNet mean/std. Training loader must drop the last short batch (wrap with a filter on batch size; check `len()` reflects it).
- `run.run_seed(cfg, seed, out_dir, *, epochs=None, loaders=None) -> dict`: build `create_vgg(5, ..., fuse_pool=True, head="se", activation=act)`, weights with `initialize_params(s, PRNGKey(seed))`, `optax.adamw(warmup_cosine(...), weight_decay=wd)`, jitted step = `nudged_grads` + optax update with `beta` passed as a jnp scalar; per epoch: β from `beta_for_epoch` (CN: factor from `cn_factors(seed, epochs)[e]`), train all batches, then forward-pass test accuracy (`evaluate(..., algorithm="backprop")`) → record `{"epoch", "beta", "test_acc", "seconds"}`. Write `out_dir/<cfg.name>/seed<seed>.json` with `best_acc`, `best_epoch`, `last_acc`, `curve`, `config`, `git_sha`, `device`. Skip a seed whose JSON already exists (resume).
- CLI: `python -m experiments.nudging.run <config> --seeds 0,1,2 --out DIR [--epochs E]`.

- [ ] **Step 1: Failing tests** — with tiny fake CIFAR-shaped loaders (`(8, 32, 32, 3)` batches, one-hot y, 3 batches), `epochs=2`: (a) JSON written with keys above and 2 curve points; (b) `best_acc >= last_acc`, `best_epoch in {1,2}`; (c) a second call with the same seed skips (file mtime unchanged); (d) β recorded per epoch equals `beta_for_epoch` for c10-nn; (e) CN config records both signs over 50 fake epochs via `cn_factors` (test the schedule wiring with `epochs=50` but zero-batch loader if supported, else test the factor list directly).
- [ ] **Steps 2–5:** fail → implement → pass → black/ruff; stage.

---

### Task 6: Summary in pcx's reporting style

**Files:**
- Create: `experiments/nudging/summarize.py`
- Test: `experiments/nudging/test_summarize.py`

**Interfaces:**
- `summarize(out_dir, name) -> dict` = `{"n", "best_mean", "best_std", "top5_mean", "top5_std" (only when n ≥ 7: mean/std of the 5 highest best_acc, ddof=0 like numpy default), "last_mean", "last_std", "target", "gap_top5"}`; CLI prints one line per config.

- [ ] **Step 1: Failing tests** with hand-made seed JSONs: 7 seeds best = [.80,.81,.82,.83,.84,.85,.86] → top5_mean = .84; last values given → last_mean exact; n<7 → top5 fields None.
- [ ] **Steps 2–5:** fail → implement → pass → black/ruff; stage.

---

### Task 7: Mac smoke run and GPU run sheet

- [ ] **Step 1:** Mac CPU, low priority: `PYTHONPATH=. taskpolicy -c utility .venv/bin/python -m experiments.nudging.run c10-nn --seeds 0 --epochs 0.02 --out <scratch>` (support fractional epochs = that fraction of batches) → completes, accuracy > chance trend not required; record s/step.
- [ ] **Step 2:** Same for `c10-se` and `c10-cn`.
- [ ] **Step 3:** Write `experiments/nudging/README.md`: what each config is, the CN reconstruction caveat (no pcx code; NN hyperparameters + random ±1 per epoch), best-vs-last reporting, Colab commands, expected T4 time (sPC VGG-5 T=7 ≈ 150 ms/step × 390 steps ≈ 1 h/seed incl. eval; 3 seeds ≈ 3 h per config).
- [ ] **Step 4:** Full experiment tests + full repo suite (`--ignore=tests/test_doc_epc_tutorial.py`) green; ask the user to commit/push the branch.
- [ ] **Step 5 (GPU, needs user go-ahead):** Colab order: c10-nn (3 seeds) → c10-se (3) → c10-cn (3) → c10-pn (3); then 7 seeds for the best; c100/tiny only on bigger GPUs.
