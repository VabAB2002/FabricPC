"""Hint experiment, round 2: the hint goes into the learning signal.

python -m experiments.hint.run2 --out <dir> [--seeds 1] [--epochs 3] [--depths 4,8,16,32] [--dataset mnist|fashionmnist] [--only a,b]
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import optax

from experiments.hint.hint_learning import batch_grads
from experiments.hint.models import deep_mlp
from fabricpc.bench.diagnostics import _flat_weights
from fabricpc.core.inference import InferenceSGD
from fabricpc.training.trainer import _batch_grads, convert_batch, evaluate
from fabricpc.utils.data.dataloader import FashionMnistLoader, MnistLoader

LOADERS = {"mnist": MnistLoader, "fashionmnist": FashionMnistLoader}

WIDTH, BATCH, LR, ETA = 128, 128, 1e-3, 0.1

# name -> (settle steps, mode, extra settings); "backprop" is special
METHODS = {
    "backprop": (8, "backprop", {}),
    "spc_T8": (8, "none", {}),
    "replay_T20": (20, "none", {}),
    "pure_dfa": (8, "pure", {}),
    "all_1": (8, "all", {"strength": 1.0}),
    "all_0.3": (8, "all", {"strength": 0.3}),
    "far_0.1": (8, "far", {"strength": 1.0, "threshold": 0.1}),
    "far_0.01": (8, "far", {"strength": 1.0, "threshold": 0.01}),
}


def _grad_fn(structure, mode, extra):
    if mode == "backprop":
        return jax.jit(
            lambda p, b, k: _batch_grads(p, b, structure, k, algorithm="backprop")[0]
        )
    return jax.jit(
        partial(_wrap, structure=structure, mode=mode, extra=tuple(extra.items()))
    )


def _wrap(p, b, k, structure, mode, extra):
    return batch_grads(p, structure, b, k, mode=mode, **dict(extra))


def _alignment(params, structure, batch, key, grad_fn):
    import jax.numpy as jnp

    ours = _flat_weights(grad_fn(params, batch, key))
    bp = _flat_weights(
        _batch_grads(params, batch, structure, key, algorithm="backprop")[0]
    )
    return {
        n: float(
            ours[n]
            @ bp[n]
            / (jnp.linalg.norm(ours[n]) * jnp.linalg.norm(bp[n]) + 1e-30)
        )
        for n in ours
    }


def run_one(name, seed, epochs, depth, dataset="mnist"):
    loader_cls = LOADERS[dataset]
    steps, mode, extra = METHODS[name]
    gkey, tkey, ekey, dkey = jax.random.split(jax.random.PRNGKey(seed), 4)
    params, structure = deep_mlp(
        depth, WIDTH, InferenceSGD(eta_infer=ETA, infer_steps=steps), gkey
    )
    grad_fn = _grad_fn(structure, mode, extra)
    opt = optax.adam(LR)
    opt_state = opt.init(params)

    @jax.jit
    def apply(p, s, g):
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s

    start, i = time.time(), 0
    for ep in range(int(epochs)):
        loader = loader_cls(
            "train",
            batch_size=BATCH,
            tensor_format="flat",
            shuffle=True,
            seed=seed * 100 + ep,
        )
        for raw in loader:
            batch = convert_batch(raw)
            g = grad_fn(params, batch, jax.random.fold_in(tkey, i))
            params, opt_state = apply(params, opt_state, g)
            i += 1
    seconds = time.time() - start

    test = loader_cls("test", batch_size=BATCH, tensor_format="flat", shuffle=False)
    acc = float(
        evaluate(params, structure, test, {}, ekey, algorithm="backprop")["accuracy"]
    )
    probe = convert_batch(next(iter(test)))
    align = _alignment(params, structure, probe, dkey, grad_fn)
    return {
        "method": name,
        "seed": seed,
        "depth": depth,
        "dataset": dataset,
        "epochs": epochs,
        "accuracy": acc,
        "train_s": round(seconds, 1),
        "cos": align,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--only", default="")
    p.add_argument("--dataset", default="mnist", choices=sorted(LOADERS))
    p.add_argument("--depths", default="", help="comma list; overrides --depth")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.only.split(",") if a.only else list(METHODS)
    depths = [int(d) for d in a.depths.split(",")] if a.depths else [a.depth]
    for depth in depths:
        for seed in range(a.seeds):
            for name in names:
                path = out / f"{a.dataset}-d{depth}-{name}-seed{seed}.json"
                if path.exists():
                    continue
                row = run_one(name, seed, a.epochs, depth, a.dataset)
                path.write_text(json.dumps(row, indent=1))
                print(
                    f"{a.dataset} d{depth} {name:11s} seed {seed}: "
                    f"acc {row['accuracy']:.4f}  "
                    f"cos h0 {row['cos'].get('h0', 1):.2f}  ({row['train_s']}s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
