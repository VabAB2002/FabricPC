"""Step 2: run this as a separate process after step 1.

It loads every saved folder without any model-building code and compares
the structure, weights and inference results with step 1.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import jax

import fpcmodel
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import run_inference
from fabricpc.core.inference_epc import EPCInference
from fabricpc.training.trainer import _check_swappable

OUT = Path(__file__).parent / "out"


def reference(structure, params, seed=7, batch=4):
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


def node_info_diff(a, b):
    """Compare NodeInfo field by field, without using our own encoder."""
    bad = []
    for f in (
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
    ):
        if getattr(a, f) != getattr(b, f):
            bad.append(f)
    return bad


print("== 3. load in a fresh process ==")
for folder in sorted(OUT.glob("*.fpc")):
    name = folder.name[:-4]
    report = {}
    try:
        structure, params, meta = fpcmodel.load_model(folder, report=report)
    except Exception as e:
        print(f"{name}: LOAD FAILED {type(e).__name__}: {e}")
        continue
    # (a) structure: encode again and compare with what step 1 saved
    orig = json.loads((OUT / f"{name}.orig_structure.json").read_text())
    again = json.loads(
        json.dumps(fpcmodel.structure_to_dict(structure), sort_keys=True)
    )
    struct_equal = orig == again
    # (b) weights: must be exactly the same, bit by bit
    with np.load(OUT / f"{name}.orig_params.npz") as z:
        o = {k: z[k] for k in z.files}
    f = fpcmodel.params_to_flat(params)
    bit = set(o) == set(f) and all(
        o[k].dtype == f[k].dtype
        and o[k].shape == f[k].shape
        and o[k].tobytes() == f[k].tobytes()
        for k in o
    )
    empty_nodes = [n for n, p in params.nodes.items() if not p.weights and not p.biases]
    line = (
        f"{name}: structure_equal={struct_equal} params_bit_identical={bit} "
        f"arrays={len(f)} dtypes={sorted({str(v.dtype) for v in f.values()})} "
        f"weightless_nodes_restored={len(empty_nodes)}"
    )
    if report:
        line += f" NOTE={report}"
    print(line)
    if not struct_equal:
        for i, (x, y) in enumerate(zip(orig["nodes"], again["nodes"])):
            if x != y:
                print(
                    "   first differing node:",
                    x["name"],
                    {k: (x[k], y[k]) for k in x if x[k] != y[k]},
                )
                break
    ref = OUT / f"{name}.ref.npz"
    if ref.exists():
        got = reference(structure, params)
        with np.load(ref) as z:
            want = {k: z[k] for k in z.files}
        same = all(want[k].tobytes() == got[k].tobytes() for k in want)
        maxdiff = max(
            float(np.max(np.abs(want[k] - got[k]))) if want[k].size else 0.0
            for k in want
        )
        tot = float(sum(got[k].sum() for k in got if k.startswith("energy|")))
        print(
            f"   settled state (energy, z_mu, z_latent of {len(structure.nodes)} nodes): bitwise_equal={same} max_abs_diff={maxdiff:.3g} total_energy={tot:.6f}"
        )

print(
    "bench.definitions imported in this process:",
    any(m.startswith("fabricpc.bench.definitions") for m in sys.modules),
)

print("== 4. negative tests ==")
src = json.loads((OUT / "mnist-mlp-spc.fpc" / "structure.json").read_text())
bad = copy.deepcopy(src)
bad["nodes"][1]["activation"]["__class__"] = "os.system"
try:
    fpcmodel.structure_from_dict(bad)
    print("evil class path: LOADED (BAD)")
except Exception as e:
    print(f"evil class path: rejected -> {type(e).__name__}: {e}")
bad = copy.deepcopy(src)
bad["nodes"][1]["class"] = "fabricpc.core.activations.SigmoidActivation"
try:
    fpcmodel.structure_from_dict(bad)
    print("wrong base class: LOADED (BAD)")
except Exception as e:
    print(f"wrong base class: rejected -> {type(e).__name__}: {e}")
bad = copy.deepcopy(src)
bad["nodes"][1]["activation"] = {
    "__class__": "fabricpc.bench.run.zoo.Path",
    "config": {},
}
try:
    fpcmodel.structure_from_dict(bad)
    print("re-exported non-fabricpc class (fabricpc.bench.run.zoo.Path): LOADED (BAD)")
except Exception as e:
    print(f"re-exported non-fabricpc class: rejected -> {type(e).__name__}: {e}")
bad = copy.deepcopy(src)
bad["format_version"] = 2
try:
    fpcmodel.structure_from_dict(bad)
    print("newer format_version: LOADED (BAD)")
except Exception as e:
    print(f"newer format_version: rejected -> {type(e).__name__}: {e}")

print(
    "== 5. swap the solver on a loaded structure (what the bench would do per algorithm) =="
)

s, p, _ = fpcmodel.load_model(OUT / "mnist-mlp-spc.fpc")
s2 = s._replace(config={**s.config, "inference": EPCInference()})
_check_swappable(s, s2)
print(
    "spc->epc swap:",
    type(s.config["inference"]).__name__,
    "->",
    type(s2.config["inference"]).__name__,
    "ok",
)
