"""Step 1: build the bench models, try pickle on them, save each model to a folder,
and keep some reference numbers so step 2 can compare.

Run from the repo root:  python scripts/serialization_test/step1_save.py
"""

import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import jax

import fpcmodel
from fabricpc.bench.definitions import registry as R
from fabricpc.bench.definitions import deep as D
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import run_inference
from fabricpc.nodes import Linear, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import UniformInitializer, XavierInitializer
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params

OUT = Path(__file__).parent / "out"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir()

key = jax.random.PRNGKey(0)


def reference(structure, params, seed=7, batch=4):
    """Run inference on one random batch and return the energy and outputs of every node."""
    rng = np.random.default_rng(seed)
    xn, yn = structure.task_map["x"], structure.task_map["y"]
    x = rng.standard_normal((batch, *structure.nodes[xn].node_info.shape)).astype(
        "float32"
    )
    ydim = structure.nodes[yn].node_info.shape
    y = np.zeros((batch, *ydim), "float32")
    y.reshape(batch, -1)[
        np.arange(batch), rng.integers(0, y.reshape(batch, -1).shape[1], batch)
    ] = 1.0
    clamps = {xn: jax.numpy.asarray(x), yn: jax.numpy.asarray(y)}
    state = initialize_graph_state(
        structure, batch, jax.random.PRNGKey(seed), clamps=clamps, params=params
    )
    state = run_inference(params, state, clamps, structure)
    out = {f"energy|{n}": np.asarray(s.energy) for n, s in state.nodes.items()}
    out.update({f"z_mu|{n}": np.asarray(s.z_mu) for n, s in state.nodes.items()})
    out.update(
        {f"z_latent|{n}": np.asarray(s.z_latent) for n, s in state.nodes.items()}
    )
    return out


print("== 1. pickle probe ==")
params, structure = R._mnist_mlp_factory("spc")(key)
for label, obj in [
    ("structure", structure),
    ("params", params),
    ("activation SigmoidActivation()", structure.nodes["hidden1"].node_info.activation),
    (
        "node_config (MappingProxyType)",
        structure.nodes["hidden1"].node_info.node_config,
    ),
    ("inference InferenceSGD", structure.config["inference"]),
]:
    try:
        b = pickle.dumps(obj)
        print(f"pickle.dumps({label}): OK, {len(b)} bytes")
    except Exception as e:
        print(f"pickle.dumps({label}): {type(e).__name__}: {e}")
for mod in ("cloudpickle", "dill"):
    try:
        __import__(mod)
        print(mod, "installed")
    except ImportError:
        print(mod, "NOT installed -> not tested")

print("== 2. build + save ==")
builders = {
    "mnist-mlp-spc": lambda: R._mnist_mlp_factory("spc")(key),
    "mnist-mlp-epc": lambda: R._mnist_mlp_factory("epc")(key),
    "fcresnet8-mupc-spc": lambda: D._model_factory("spc", 8)(key),
    "cifar10-vgg5-spc": lambda: R._cifar10_vgg5_factory("spc")(key),
    "cifar10-resnet18-spc": lambda: R._cifar10_resnet18_factory("spc")(key),
    "char-transformer-spc": lambda: R._char_transformer_factory("spc")(key),
    "mnist-autoencoder-spc": lambda: R._mnist_autoencoder_factory("spc")(key),
}
try:
    from fabricpc.bench.definitions import rows_hopfield as H

    builders["hopfield-spc"] = lambda: H._recall_graph_factory("spc")(key)
except Exception as e:
    print("hopfield import failed:", e)

FORWARD = {"mnist-mlp-spc", "mnist-mlp-epc", "fcresnet8-mupc-spc", "cifar10-vgg5-spc"}
summary = {}
for name, build in builders.items():
    t = time.time()
    try:
        params, structure = build()
    except Exception as e:
        print(f"{name}: BUILD FAILED {type(e).__name__}: {e}")
        continue
    classes = sorted({type(n).__name__ for n in structure.nodes.values()})
    n_mupc = sum(
        n.node_info.scaling_config is not None for n in structure.nodes.values()
    )
    try:
        fpcmodel.save_model(
            structure, params, OUT / f"{name}.fpc", metadata={"source": name}
        )
    except Exception as e:
        print(f"{name}: SAVE FAILED {type(e).__name__}: {e}")
        continue
    files = sorted(
        p.name + ("/" if p.is_dir() else "") for p in (OUT / f"{name}.fpc").iterdir()
    )
    s2, p2, _ = fpcmodel.load_model(OUT / f"{name}.fpc")
    FIELDS = (
        "name",
        "shape",
        "node_type",
        "node_class",
        "slots",
        "in_degree",
        "out_degree",
        "in_edges",
        "out_edges",
        "scaling_config",
    )
    bad = [
        (n, f)
        for n in structure.nodes
        for f in FIELDS
        if getattr(structure.nodes[n].node_info, f) != getattr(s2.nodes[n].node_info, f)
    ]
    bad += [
        f
        for f in ("edges", "task_map", "node_order", "schedule")
        if getattr(structure, f) != getattr(s2, f)
    ]
    bad += [
        n
        for n in structure.nodes
        if dict(structure.nodes[n].node_info.node_config).keys()
        != dict(s2.nodes[n].node_info.node_config).keys()
    ]
    print(
        f"   python == on NodeInfo fields/edges/task_map/node_order/schedule (no encoder): mismatches={bad}"
    )
    n_arr = sum(1 for _ in jax.tree_util.tree_leaves(params))
    print(
        f"{name}: saved nodes={len(structure.nodes)} edges={len(structure.edges)} "
        f"arrays={n_arr} mupc_nodes={n_mupc} classes={classes} files={files} "
        f"({time.time()-t:.1f}s)"
    )
    # also save the original weights, so step 2 can check they come back exactly the same
    np.savez(OUT / f"{name}.orig_params.npz", **fpcmodel.params_to_flat(params))
    (OUT / f"{name}.orig_structure.json").write_text(
        json.dumps(fpcmodel.structure_to_dict(structure), sort_keys=True)
    )
    if name in FORWARD:
        t = time.time()
        np.savez(OUT / f"{name}.ref.npz", **reference(structure, params))
        print(f"   reference inference recorded ({time.time()-t:.1f}s)")
print(
    "bench.definitions imported in this process:",
    "fabricpc.bench.definitions.registry" in sys.modules,
)

print("== 2b. cyclic graph, unroll=2 (structure only) ==")

a = IdentityNode(shape=(8,), name="a")
b = Linear(shape=(8,), name="b", weight_init=UniformInitializer(-0.2, 0.2))
c = Linear(shape=(8,), name="c", weight_init=XavierInitializer())
d = Linear(shape=(4,), name="d", weight_init=XavierInitializer())
cyc = graph(
    nodes=[a, b, c, d],
    edges=[
        Edge(source=a, target=b.slot("in")),
        Edge(source=b, target=c.slot("in")),
        Edge(source=c, target=b.slot("in")),
        Edge(source=c, target=d.slot("in")),
    ],
    task_map=TaskMap(x=a, y=d),
    inference=InferenceSGD(),
    unroll=2,
)
pc = initialize_params(cyc, key)
fpcmodel.save_model(cyc, pc, OUT / "cyclic-unroll2.fpc")
s2, p2, _ = fpcmodel.load_model(OUT / "cyclic-unroll2.fpc")
print(
    "saved schedule:",
    cyc.schedule,
    "| loaded schedule equal:",
    s2.schedule == cyc.schedule,
    "| unroll:",
    s2.config["unroll"],
    "| UniformInitializer config:",
    dict(s2.nodes["b"].node_info.weight_init.config),
)
np.savez(OUT / "cyclic-unroll2.orig_params.npz", **fpcmodel.params_to_flat(pc))
(OUT / "cyclic-unroll2.orig_structure.json").write_text(
    json.dumps(fpcmodel.structure_to_dict(cyc), sort_keys=True)
)
