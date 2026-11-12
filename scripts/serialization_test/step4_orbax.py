"""Step 4: check if Orbax can restore the weights without a template."""

import shutil
from pathlib import Path
import numpy as np
import orbax.checkpoint as ocp
import fpcmodel

OUT = Path(__file__).parent / "out"
s, p, _ = fpcmodel.load_model(OUT / "fcresnet8-mupc-spc.fpc")
r = ocp.StandardCheckpointer().restore((OUT / "orbax_probe").resolve())
print(
    "A) GraphParams saved directly, restore() no template ->",
    type(r).__name__,
    "len",
    len(r),
    "| r[0]:",
    type(r[0]).__name__,
    "keys",
    len(r[0]),
    "of",
    len(p.nodes),
)
k = "res" if "res" in r[0] else [n for n in r[0]][1]
print(
    "   node",
    repr(k),
    "->",
    type(r[0][k]).__name__,
    [type(x).__name__ for x in r[0][k]] if isinstance(r[0][k], list) else list(r[0][k]),
)
print("   missing nodes:", sorted(set(p.nodes) - set(r[0])))
# B) same weights saved as a normal nested dict
plain = {
    n: {"weights": dict(v.weights), "biases": dict(v.biases)}
    for n, v in p.nodes.items()
}
d = OUT / "orbax_probe_dict"
shutil.rmtree(d, ignore_errors=True)
ck = ocp.StandardCheckpointer()
ck.save(d.resolve(), plain)
ck.wait_until_finished()
r2 = ocp.StandardCheckpointer().restore(d.resolve())
print(
    "B) plain dict saved, restore() no template ->",
    type(r2).__name__,
    "nodes",
    len(r2),
    "of",
    len(plain),
    "| missing:",
    sorted(set(plain) - set(r2)),
)
bad = 0
n = 0
for node, v in plain.items():
    for kind in ("weights", "biases"):
        for name, arr in v[kind].items():
            n += 1
            got = r2.get(node, {}).get(kind, {}).get(name)
            if (
                got is None
                or np.asarray(got).tobytes() != np.asarray(arr).tobytes()
                or got.dtype != arr.dtype
            ):
                bad += 1
print(
    f"   arrays compared={n} mismatched_or_missing={bad} | sample key:",
    repr(next(iter(r2[[x for x in r2 if r2[x].get("weights")][0]]["weights"]))),
)
print(
    "   files:", len([x for x in d.rglob("*") if x.is_file()]), "| npz format: 1 file"
)
