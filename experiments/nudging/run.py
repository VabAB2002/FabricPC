"""Train FabricPC's VGG-5 the way pcx trains its nudged runs, one seed at a time.

    python -m experiments.nudging.run c10-nn --seeds 0,1,2 --out DIR [--epochs E]

For each seed this writes DIR/<config>/seed<seed>.json with the test accuracy
after every epoch, the best epoch (what pcx reports) and the last epoch
(what our benchmark reports). A seed whose file already exists is skipped,
so a stopped run can just be started again.
"""

import argparse
import dataclasses
import json
import math
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from experiments.nudging.configs import CONFIGS
from experiments.nudging.nudge import nudged_grads
from experiments.nudging.schedule import beta_for_epoch, cn_factors
from fabricpc.core.activations import GeluActivation
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph_initialization import initialize_params
from fabricpc.models.vgg import create_vgg
from fabricpc.training.trainer import convert_batch, evaluate


def _schedule(lr, total_steps):
    """pcx's rate: starts at lr, warms up to 1.1*lr over the first tenth,
    then a cosine down to 0.1*lr."""
    return optax.warmup_cosine_decay_schedule(
        init_value=lr,
        peak_value=1.1 * lr,
        warmup_steps=max(1, int(0.1 * total_steps)),
        decay_steps=max(2, total_steps),
        end_value=0.1 * lr,
    )


def _git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_seed(cfg, seed, out_dir, *, epochs=None, loaders=None):
    """Train one seed and write its result file. Returns the result dict."""
    path = Path(out_dir) / cfg.name / f"seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())

    if loaders is None:
        from experiments.nudging.data import loaders as make_loaders

        loaders = make_loaders(cfg, seed)
    train_loader, test_loader = loaders
    epochs = cfg.epochs if epochs is None else epochs
    whole_epochs = max(1, math.ceil(epochs))
    per_epoch = len(train_loader)
    # A fractional run (a quick check) trains that fraction of one epoch.
    batches_per_epoch = per_epoch if epochs >= 1 else max(1, int(per_epoch * epochs))

    structure = create_vgg(
        5,
        input_shape=cfg.input_shape,
        num_classes=cfg.num_classes,
        inference=InferenceSGD(eta_infer=cfg.lr_x, infer_steps=cfg.T),
        activation=GeluActivation(),
        fuse_pool=True,
        head="se",
    )
    master = jax.random.PRNGKey(seed)
    graph_key, train_key, eval_key = jax.random.split(master, 3)
    params = initialize_params(structure, graph_key)
    opt = optax.adamw(
        _schedule(cfg.lr_w, batches_per_epoch * whole_epochs), weight_decay=cfg.wd
    )
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch, key, beta):
        grads, metrics = nudged_grads(
            params,
            batch,
            structure,
            key,
            beta=beta,
            lr_x=cfg.lr_x,
            momentum=cfg.momentum,
            steps=cfg.T,
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics["energy"]

    signs = cn_factors(seed, whole_epochs) if cfg.centered else None
    curve, i = [], 0
    for e in range(whole_epochs):
        factor = signs[e] if cfg.centered else cfg.factor
        beta = beta_for_epoch(cfg.beta0, cfg.beta_ir, factor, e)
        t0 = time.perf_counter()
        n = 0
        for raw in train_loader:
            if n == batches_per_epoch:
                break
            params, opt_state, _ = step(
                params,
                opt_state,
                convert_batch(raw),
                jax.random.fold_in(train_key, i),
                jnp.float32(beta),
            )
            n += 1
            i += 1
        jax.block_until_ready(params)
        seconds = time.perf_counter() - t0
        # pcx scores with a plain forward pass (argmax of the logits).
        acc = float(
            evaluate(
                params, structure, test_loader, {}, eval_key, algorithm="backprop"
            )["accuracy"]
        )
        curve.append(
            {
                "epoch": e + 1,
                "beta": beta,
                "test_acc": acc,
                "seconds": seconds,
                "batches": n,
            }
        )
        print(
            f"{cfg.name} seed {seed} epoch {e + 1}: beta={beta:+.3f} acc={acc:.4f}",
            flush=True,
        )

    best = max(curve, key=lambda p: p["test_acc"])
    result = {
        "config": dataclasses.asdict(cfg),
        "seed": seed,
        "epochs": epochs,
        "best_acc": best["test_acc"],
        "best_epoch": best["epoch"],
        "last_acc": curve[-1]["test_acc"],
        "curve": curve,
        "git_sha": _git_sha(),
        "device": str(jax.devices()[0].device_kind),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", choices=sorted(CONFIGS))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float)
    a = p.parse_args(argv)
    cfg = CONFIGS[a.config]
    for seed in [int(s) for s in a.seeds.split(",")]:
        r = run_seed(cfg, seed, a.out, epochs=a.epochs)
        print(
            f"{cfg.name} seed {seed}: best {r['best_acc']:.4f} (epoch {r['best_epoch']}), "
            f"last {r['last_acc']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
