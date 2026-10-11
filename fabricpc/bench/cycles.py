"""Graphs with loops: refine with sPC, or unroll? (issue #64)

    python -m fabricpc.bench cycles --out DIR [--epochs 1] [--max-eval-batches 5]

A graph with a loop can be settled two ways. Unrolling
(``graph(..., unroll=U)``) visits each loop U times inside one
differentiable program, which ePC relaxes. The composed schedule relaxes
with ePC on a single visit and then refines with T2 steps of sPC on the true
loop energy. This measures both, and a third arm (unrolled ePC then sPC),
on the same trained weights and the same images. Unrolling is swept over U
and its own ePC step count (it has no refinement to finish the settle); the
composed arms over T2:

- energy: the true full-graph energy of the settled state (every node's
  energy at its latents, back edges included, the way sPC sees it), per
  sample, with input and label clamped as in a training step. The anchor is
  sPC settled to convergence; an arm "lands" when it is within ``TOL``
  (1%) of the anchor's energy above it.
- accuracy: the same solver with the label free, scored on the output.
- matmuls per sample: one pass over a weighted edge is one matmul. ePC
  runs a forward pass over the visit schedule to start, two per step
  (forward and backward) and one to finish; sPC two per step over the
  graph's edges.
- wall-clock: the median time of one settle (jitted, compile excluded).

Two loops: ``loop2`` is the demo's (hidden2 -> back -> hidden1, two hidden
nodes in the loop), ``loop4`` puts four hidden nodes in it. The weights are
trained first with the demo's sPC (eta 0.15, 50 steps, unroll 2). For each
arm the cheapest setting that lands is reported.
"""

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import optax

from fabricpc.core import EPCInference, InferenceSchedule, InferenceSGD
from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy, graph_energy
from fabricpc.core.inference import run_inference
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear
from fabricpc.training import train
from fabricpc.training.trainer import build_clamps, convert_batch

TOL = 0.01  # an arm lands within 1% of the converged energy
ANCHOR_STEPS = 2000
EPC_ETA, EPC_STEPS = 0.05, 5  # the demo's ePC settings
SPC_ETA = 0.15  # the demo's sPC rate
LOOPS = {"loop2": 2, "loop4": 4}
UNROLLS = (1, 2, 3, 4, 6, 8)
EPC_SWEEP = (5, 20, 50, 200)  # ePC steps for the unroll arm
REFINES = (5, 10, 20, 50, 100, 200)
SPC_STEPS = (20, 50, 100, 200, 500)


def cyclic_factory(loop: int, *, unroll: int, inference):
    """pixels -> h1 -> ... -> h<loop> -> class, and h<loop> -> back -> h1."""

    def build(key):
        sig = SigmoidActivation()
        pixels = Linear(shape=(784,), activation=sig, name="pixels")
        widths = [256] + [64] * (loop - 1)
        hidden = [
            Linear(shape=(w,), activation=sig, name=f"hidden{i + 1}")
            for i, w in enumerate(widths)
        ]
        back = Linear(shape=(64,), activation=sig, name="back")
        output = Linear(
            shape=(10,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="class",
        )
        chain = [pixels, *hidden, output]
        edges = [Edge(source=a, target=b.slot("in")) for a, b in zip(chain, chain[1:])]
        edges += [
            Edge(source=hidden[-1], target=back.slot("in")),
            Edge(source=back, target=hidden[0].slot("in")),
        ]
        structure = graph(
            nodes=[pixels, *hidden, back, output],
            edges=edges,
            task_map=TaskMap(x=pixels, y=output),
            inference=inference,
            unroll=unroll,
        )
        return initialize_params(structure, key), structure

    return build


@dataclass(frozen=True)
class Arm:
    name: str  # "unroll", "refine", "unroll2+refine" or "spc"
    unroll: int
    epc_steps: int  # 0 when the arm has no ePC segment
    spc_steps: int  # 0 when the arm has no sPC segment

    def inference(self):
        epc = EPCInference(eta_infer=EPC_ETA, infer_steps=self.epc_steps)
        spc = InferenceSGD(eta_infer=SPC_ETA, infer_steps=self.spc_steps)
        if not self.spc_steps:
            return epc
        if not self.epc_steps:
            return spc
        return InferenceSchedule(epc, spc)


def arms() -> List[Arm]:
    # Unrolling has no refinement, so its own step count is swept too:
    # with the demo's 5 steps it cannot settle, whatever U is.
    out = [Arm("unroll", u, t, 0) for t in EPC_SWEEP for u in UNROLLS]
    out += [Arm("refine", 1, EPC_STEPS, t) for t in REFINES]
    out += [Arm("unroll2+refine", 2, EPC_STEPS, t) for t in REFINES]
    out += [Arm("spc", 1, 0, t) for t in SPC_STEPS]
    return out


def _weighted_in_edges(structure, params, node: str) -> int:
    weights = params.nodes[node].weights
    return sum(
        1 for key, e in structure.edges.items() if e.target == node and key in weights
    )


def matmuls_per_sample(arm: Arm, structure, params) -> int:
    """Matmuls one settle of one sample costs (see the module docstring)."""
    visits = sum(_weighted_in_edges(structure, params, n) for n in structure.schedule)
    edges = sum(_weighted_in_edges(structure, params, n) for n in structure.nodes)
    count = visits  # the feedforward start along the visit schedule
    if arm.epc_steps:
        count += (2 * arm.epc_steps + 2) * visits
    count += 2 * arm.spc_steps * edges
    return count


def true_energy(params, state, clamps, structure) -> jnp.ndarray:
    """Every node's energy at its latents, back edges included, per sample."""
    fresh = InferenceSGD.forward_value_and_grad(params, state, clamps, structure)
    return graph_energy(fresh, structure) / state.batch_size


def _settler(structure):
    def settle(params, clamps, key):
        batch = next(iter(clamps.values())).shape[0]
        state = initialize_graph_state(
            structure, batch, key, clamps=clamps, params=params
        )
        return run_inference(params, state, clamps, structure)

    return jax.jit(settle)


def measure(arm: Arm, params, batches, loop: int, key) -> Dict[str, float]:
    """Energy (label clamped), accuracy (label free), matmuls, time."""
    _, structure = cyclic_factory(loop, unroll=arm.unroll, inference=arm.inference())(
        key
    )
    settle = _settler(structure)
    energies, correct, count, times = [], 0, 0, []
    for i, raw in enumerate(batches):
        batch = convert_batch(raw)
        clamped = build_clamps(batch, structure, clamp_target=True)
        free = build_clamps(batch, structure, clamp_target=False)
        k = jax.random.fold_in(key, i)
        state = jax.block_until_ready(settle(params, clamped, k))
        energies.append(float(true_energy(params, state, clamped, structure)))
        t0 = time.perf_counter()
        out = jax.block_until_ready(settle(params, free, k))
        times.append(time.perf_counter() - t0)
        pred = jnp.argmax(out.nodes["class"].z_mu, -1)
        correct += int(jnp.sum(pred == jnp.argmax(batch["y"], -1)))
        count += int(pred.shape[0])
    return {
        "energy": statistics.fmean(energies),
        "accuracy": correct / count,
        "matmuls_per_sample": matmuls_per_sample(arm, structure, params),
        # The first free settle compiles; the rest are timed.
        "settle_ms": statistics.median(times[1:] or times) * 1000.0,
    }


def train_weights(loop, train_loader, epochs, key):
    """The demo's sPC training, to get weights worth settling on."""
    params, structure = cyclic_factory(
        loop, unroll=2, inference=InferenceSGD(eta_infer=SPC_ETA, infer_steps=50)
    )(key)
    trained = train(
        params,
        structure,
        train_loader,
        optax.adamw(0.001, weight_decay=0.001),
        {"num_epochs": epochs},
        key,
        algorithm="pc",
        verbose=False,
    )
    return trained.params


def cheapest_landing(rows: List[dict], anchor: float) -> Dict[str, Optional[dict]]:
    """Per arm, the lowest-matmul setting within TOL of the anchor energy."""
    best: Dict[str, Optional[dict]] = {}
    for r in rows:
        best.setdefault(r["arm"], None)
        if r["arm"] == "anchor":
            continue
        lands = r["energy"] <= anchor + TOL * abs(anchor)
        if lands and (
            best[r["arm"]] is None
            or r["matmuls_per_sample"] < best[r["arm"]]["matmuls_per_sample"]
        ):
            best[r["arm"]] = r
    best.pop("anchor", None)
    return best


def run(out, *, loader_factory, epochs=1.0, n_batches=5, seed=0, loops=None):
    """Train each loop's weights, measure every arm, write the results."""
    key = jax.random.PRNGKey(seed)
    report = {"tol": TOL, "anchor_steps": ANCHOR_STEPS, "loops": {}}
    for loop_name in loops or LOOPS:
        loop = LOOPS[loop_name]
        train_loader, test_loader = loader_factory(seed)
        params = train_weights(loop, train_loader, epochs, key)
        batches = [b for _, b in zip(range(n_batches), test_loader)]
        anchor_arm = Arm("anchor", 1, 0, ANCHOR_STEPS)
        rows = [
            {
                "arm": "anchor",
                **asdict(anchor_arm),
                **measure(anchor_arm, params, batches, loop, key),
            }
        ]
        for arm in arms():
            rows.append(
                {
                    "arm": arm.name,
                    **asdict(arm),
                    **measure(arm, params, batches, loop, key),
                }
            )
        anchor = rows[0]["energy"]
        report["loops"][loop_name] = {
            "anchor_energy": anchor,
            "rows": rows,
            "cheapest_landing": cheapest_landing(rows, anchor),
        }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cycles.json").write_text(json.dumps(report, indent=2))
    (out / "cycles.md").write_text(render(report))
    return report


def render(report) -> str:
    lines = []
    for loop_name, data in report["loops"].items():
        anchor = data["anchor_energy"]
        lines += [
            f"## {loop_name}: anchor energy {anchor:.4f} "
            f"(sPC, {report['anchor_steps']} steps); lands within {report['tol']:.0%}",
            "",
            "| arm | unroll | ePC steps | sPC steps | energy | lands | accuracy "
            "| matmuls/sample | settle ms |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in data["rows"]:
            lands = r["energy"] <= anchor + report["tol"] * abs(anchor)
            lines.append(
                f"| {r['arm']} | {r['unroll']} | {r['epc_steps']} | {r['spc_steps']} "
                f"| {r['energy']:.4f} | {'yes' if lands else 'no'} "
                f"| {r['accuracy']:.4f} | {r['matmuls_per_sample']} "
                f"| {r['settle_ms']:.2f} |"
            )
        lines += ["", "Cheapest setting that lands, per arm:", ""]
        for arm, r in data["cheapest_landing"].items():
            if r is None:
                lines.append(f"- {arm}: none of its settings lands")
            else:
                lines.append(
                    f"- {arm}: unroll {r['unroll']}, ePC {r['epc_steps']}, "
                    f"sPC {r['spc_steps']}: {r['matmuls_per_sample']} matmuls/sample, "
                    f"{r['settle_ms']:.2f} ms, accuracy {r['accuracy']:.4f}"
                )
        lines.append("")
    return "\n".join(lines)
