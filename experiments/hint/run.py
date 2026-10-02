"""Run the hint experiment: replay vs direct hint vs hybrid, on a deep MNIST MLP.

    python -m experiments.hint.run --out <dir> [--seeds 3] [--epochs 3] [--only name,...]

Each method trains the same deep MLP. We score it with a plain forward pass
(the fair score for every method) and measure, layer by layer, how closely
its weight update points the same way as backprop's (cos 1 = same as backprop).
"""

import argparse
import json
import time
from pathlib import Path

import jax
import optax

from experiments.hint.hint_inference import InferenceSGDHint
from experiments.hint.models import deep_mlp
from fabricpc.bench.diagnostics import backprop_alignment
from fabricpc.core.inference import InferenceSGD
from fabricpc.training.trainer import evaluate, train
from fabricpc.utils.data.dataloader import MnistLoader

DEPTH, WIDTH, BATCH, LR = 8, 128, 128, 1e-3
ETA, T = 0.1, 8

# name -> (algorithm, solver)
METHODS = {
    "backprop": ("backprop", InferenceSGD(eta_infer=ETA, infer_steps=T)),
    "spc_T8": ("pc", InferenceSGD(eta_infer=ETA, infer_steps=T)),
    # A. replay: let the whisper chain run longer
    "replay_T20": ("pc", InferenceSGD(eta_infer=ETA, infer_steps=20)),
    "replay_T50": ("pc", InferenceSGD(eta_infer=ETA, infer_steps=50)),
    # B. direct recording only (direct feedback alignment)
    "direct_only": (
        "pc",
        InferenceSGDHint(eta_infer=ETA, infer_steps=1, hint_strength=1.0, chain=False),
    ),
    # C. hybrid: whisper chain plus a direct hint of different strengths
    "hybrid_0.01": (
        "pc",
        InferenceSGDHint(eta_infer=ETA, infer_steps=T, hint_strength=0.01),
    ),
    "hybrid_0.1": (
        "pc",
        InferenceSGDHint(eta_infer=ETA, infer_steps=T, hint_strength=0.1),
    ),
    "hybrid_1": (
        "pc",
        InferenceSGDHint(eta_infer=ETA, infer_steps=T, hint_strength=1.0),
    ),
}


def run_one(name: str, seed: int, epochs: float) -> dict:
    algorithm, solver = METHODS[name]
    graph_key, train_key, eval_key, diag_key = jax.random.split(
        jax.random.PRNGKey(seed), 4
    )
    params, structure = deep_mlp(DEPTH, WIDTH, solver, graph_key)
    train_loader = MnistLoader(
        "train", batch_size=BATCH, tensor_format="flat", shuffle=True, seed=seed
    )
    test_loader = MnistLoader(
        "test", batch_size=BATCH, tensor_format="flat", shuffle=False
    )
    probe = next(iter(test_loader))

    start = time.time()
    result = train(
        params,
        structure,
        train_loader,
        optax.adam(LR),
        {"num_epochs": epochs},
        train_key,
        algorithm=algorithm,
        verbose=False,
    )
    seconds = time.time() - start
    trained = result.params if hasattr(result, "params") else result[0]

    scores = evaluate(
        trained, structure, test_loader, {}, eval_key, algorithm="backprop"
    )
    align = (
        None
        if algorithm == "backprop"
        else backprop_alignment(trained, structure, probe, diag_key, "spc")
    )
    return {
        "method": name,
        "seed": seed,
        "epochs": epochs,
        "depth": DEPTH,
        "accuracy": float(scores["accuracy"]),
        "train_s": round(seconds, 1),
        "alignment": align,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--only", default="")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.only.split(",") if a.only else list(METHODS)
    for seed in range(a.seeds):
        for name in names:
            path = out / f"{name}-seed{seed}.json"
            if path.exists():
                continue
            row = run_one(name, seed, a.epochs)
            path.write_text(json.dumps(row, indent=1))
            al = row["alignment"]
            print(
                f"{name:12s} seed {seed}: acc {row['accuracy']:.4f}  "
                f"first-layer cos {al['layers']['h0']['cos'] if al else 1.0:.2f}  "
                f"({row['train_s']}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
