"""Step 3: check which classes can't be rebuilt with cls(**obj.config),
and save a test Orbax checkpoint that step 4 uses.
"""

import inspect
import json
import shutil
from pathlib import Path
import numpy as np
import orbax.checkpoint as ocp
import fpcmodel
from fabricpc.core import activations, energy, initializers, inference, inference_epc
from fabricpc.graph_initialization import state_initializer
from fabricpc.core._frozen import FrozenConfig
from fabricpc.core.inference import InferenceBase, InferenceSGD
from fabricpc.graph_initialization.state_initializer import StateInitBase

print("== 6. constructor contract: cls(**obj.config) reproduces obj.config ==")
bases = (FrozenConfig, InferenceBase, StateInitBase)
specials = {
    "InferenceSchedule": lambda c: c(InferenceSGD(), inference_epc.EPCInference()),
    "GlobalStateInit": lambda c: c(initializer=initializers.NormalInitializer(std=0.1)),
}
for mod in (
    activations,
    energy,
    initializers,
    inference,
    inference_epc,
    state_initializer,
):
    for name, cls in vars(mod).items():
        if (
            not (
                inspect.isclass(cls)
                and issubclass(cls, bases)
                and not inspect.isabstract(cls)
            )
            or cls.__module__ != mod.__name__
        ):
            continue
        try:
            obj = specials[name](cls) if name in specials else cls()
        except Exception as e:
            print(f"  {name}: cannot build default ({e})")
            continue
        try:
            again = cls(**dict(obj.config))
            ok = fpcmodel.encode(again) == fpcmodel.encode(obj)
            raw = (
                "OK"
                if ok
                else f"CONFIG CHANGED {dict(obj.config)} -> {dict(again.config)}"
            )
        except Exception as e:
            raw = f"FAILS {type(e).__name__}: {e}"
        try:
            via = fpcmodel.encode(
                fpcmodel.decode(json.loads(json.dumps(fpcmodel.encode(obj))))
            ) == fpcmodel.encode(obj)
        except Exception as e:
            via = f"{type(e).__name__}: {e}"
        if raw != "OK" or via is not True:
            print(
                f"  {name}: raw cls(**config) -> {raw} | prototype JSON round-trip -> {via}"
            )
        else:
            print(f"  {name}: OK")
z = initializers.ZerosInitializer(gain=3.0)
print(
    "  ZerosInitializer(gain=3.0).config =",
    dict(z.config),
    "(gain silently dropped; harmless for zeros)",
)
print(
    "  FrozenConfig defines __eq__:",
    "__eq__" in vars(FrozenConfig),
    "| SigmoidActivation()==SigmoidActivation():",
    activations.SigmoidActivation() == activations.SigmoidActivation(),
)

print("== 7. Orbax restore with NO template ==")

OUT = Path(__file__).parent / "out"
s, p, _ = fpcmodel.load_model(OUT / "fcresnet8-mupc-spc.fpc")
d = OUT / "orbax_probe"
shutil.rmtree(d, ignore_errors=True)
ck = ocp.StandardCheckpointer()
ck.save(d.resolve(), p)
ck.wait_until_finished()
try:
    r = ocp.StandardCheckpointer().restore(d.resolve())
    print(
        "  restore() type:",
        type(r).__name__,
        "| top keys:",
        (
            list(r)[:3]
            if hasattr(r, "keys")
            else r._fields if hasattr(r, "_fields") else "?"
        ),
    )
    nodes = r["nodes"] if isinstance(r, dict) else r.nodes
    print(
        "  node keys saved:",
        len(nodes),
        "of",
        len(p.nodes),
        "| missing (weightless):",
        sorted(set(p.nodes) - set(nodes)),
    )
    k = [n for n in nodes if nodes[n]][0]
    print(
        "  node entry type:",
        type(nodes[k]).__name__,
        "| sub keys:",
        list(nodes[k]) if isinstance(nodes[k], dict) else nodes[k]._fields,
    )
    w = nodes[k]["weights"] if isinstance(nodes[k], dict) else nodes[k].weights
    ek = list(w)[0]
    print(
        "  edge key survived:",
        repr(ek),
        "| leaf type:",
        type(w[ek]).__name__,
        w[ek].dtype,
        "| bit-identical:",
        np.asarray(w[ek]).tobytes() == np.asarray(p.nodes[k].weights[ek]).tobytes(),
    )
except Exception as e:
    print("  template-free restore FAILED:", type(e).__name__, str(e)[:300])
print(
    "  orbax folder contents:", sorted(x.name for x in d.rglob("*") if x.is_file())[:12]
)
