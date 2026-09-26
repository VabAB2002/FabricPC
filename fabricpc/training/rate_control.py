"""An inference rate that follows the settle's stiffness during training.

Why a fixed rate fails. Both PC solvers settle by gradient descent, which is
stable only while eta_infer * lambda_max < 2, lambda_max the stiffest
direction of the energy the solver descends: in error coordinates for ePC
(``EPCInference``, measured by ``epsilon_spectrum``), in latent coordinates
for the state-based solvers (``InferenceSGD`` and variants, measured by
``latent_curvature``). Training makes networks stiffer, and relaxed PC gives
the weights a reason to: at the equilibrium of a linear chain the energy is
1/2 r S^-1 r with S = I + J J^T, J the map from the hidden errors to the
output prediction, so making J larger lowers the energy without reducing
the output residual r. The more a run relaxes, the more its weight updates
push the stiffness up. With a fixed eta the settle eventually diverges.

Measured cases. ResNet-18 under ePC (report Sections 5.8 and 5.9); VGG-5
under ePC at eta 0.01, T 8 (dead by epoch 5); a 6-layer GELU MLP on MNIST
under ePC at eta 0.041, T 8 (dead in the first epoch, lambda_max 1.4 to
37,000). The character transformer under sPC at eta 0.0175: the settle's
stiffness went from 3.7 at init to about 35,000 after one epoch (about 960
for the same graph trained by backprop), so eta * lambda_max reached 650;
the solver's norm clip kept the latents finite but the weight updates
became noise and training stalled.

What this does. Every ``every`` updates it measures the stiffness on a
fixed probe batch and sets

    eta_infer = eta_max / 2**k,  the largest such value with
    eta_infer * stiffness <= target

using the largest stiffness of the last ``window`` probes; eta_max is the
rate the graph was built with and is never exceeded. For ePC, holding
eta * lambda_max fixed holds the top mode's relaxed fraction
1 - (1 - target)^T fixed, so the run stays in one regime; the ``Regime`` label
is recorded at every probe. It slows the growth of the stiffness; it does
not stop it. On the 6-layer MLP under ePC (target 0.2) the run never
crossed the bound in 15 epochs and matched backprop's accuracy while
lambda_max rose to about 1,800 and eta fell from 0.041 to 1.6e-4. Rates are
powers of two below eta_max so each distinct rate compiles its step once.

A state-based solver's signal to a node d hops from the output shrinks
roughly like (eta * coupling)^d per settle, so a much smaller rate also
starves the early layers; pair it with more steps or an optimizer whose
epsilon is below those gradients.

Use it as ``train(..., structure_callback=controller.on_iter)`` after
``structure = controller.start(params)``; evaluate with
``controller.structure``.
"""

import math
from collections import deque
from typing import Any, Dict, List, Mapping, Optional

import jax

from fabricpc.core.epsilon_spectrum import make_epsilon_spectrum
from fabricpc.core.inference import InferenceBase, InferenceSchedule
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.latent_curvature import make_latent_curvature
from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.graph_initialization.state_initializer import initialize_graph_state


class InferenceRateController:
    """Keeps ``eta_infer * stiffness`` at ``target`` as the stiffness changes.

    Args:
        structure: the graph, built with the solver whose ``eta_infer`` is
            the largest rate allowed; every other setting of the solver is
            kept. ``EPCInference`` or a single state-based solver (not an
            ``InferenceSchedule``).
        probe_clamps: fixed clamps (node name -> array), inputs and targets,
            measured at every probe. Keep the same batch for the whole run.
        target: eta_infer * stiffness to hold, in (0, 2). Stability needs it
            below 2; for ePC at odd infer_steps the output gradient reverses
            above about 1. The stiffness can jump several times over between
            probes, so leave room: 0.2 allows a tenfold jump.
        every: probe after every ``every``-th weight update.
        key: PRNG key for the probe's state initialization and start vector.
        window: the rate uses the largest stiffness of the last ``window``
            probes, so one high reading keeps it down for a while instead of
            the rate flipping with every probe.
        iters: Lanczos (ePC) or power-iteration (state-based) steps per probe.
    """

    def __init__(
        self,
        structure: GraphStructure,
        probe_clamps: Mapping[str, Any],
        *,
        target: float,
        every: int,
        key: jax.Array,
        window: int = 3,
        iters: int = 20,
    ):
        solver = structure.config.get("inference")
        if not isinstance(solver, InferenceBase) or isinstance(
            solver, InferenceSchedule
        ):
            raise ValueError(
                "InferenceRateController needs a graph built with one solver "
                f"(EPCInference or a state-based one), got {type(solver).__name__}."
            )
        if not 0.0 < float(target) < 2.0:
            raise ValueError(
                f"target is eta_infer * stiffness and must be in (0, 2) for a "
                f"stable settle, got {target}."
            )
        if int(every) < 1 or int(window) < 1:
            raise ValueError("every and window must be positive counts.")
        self.target = float(target)
        self.every = int(every)
        self.window = int(window)
        self.key = key
        self.probe_clamps = dict(probe_clamps)
        self.eta_max = float(solver.config["eta_infer"])
        self.is_epc = isinstance(solver, EPCInference)
        self._solver_cls = type(solver)
        self._solver_config = dict(solver.config)
        # The ePC spectrum leaves latent decay out, so it is added to the
        # measured value; the state-based measure already includes it.
        self._decay = (
            float(solver.config.get("latent_decay", 0.0)) if self.is_epc else 0.0
        )
        self._base = structure
        self._by_eta: Dict[float, GraphStructure] = {}
        self.structure = self._with_eta(self.eta_max)
        self.history: List[Dict[str, Any]] = []
        self.crossings = 0
        self._recent: deque = deque(maxlen=self.window)

        batch_size = next(iter(self.probe_clamps.values())).shape[0]
        if self.is_epc:
            measure = make_epsilon_spectrum(structure, int(iters))
        else:
            measure = make_latent_curvature(structure, int(iters))

        def probe(params, clamps, k):
            state = initialize_graph_state(
                structure, batch_size, k, clamps=clamps, params=params
            )
            return measure(params, state, clamps, k)

        self._probe = jax.jit(probe)

    # ------------------------------------------------------------ internals

    def _with_eta(self, eta: float) -> GraphStructure:
        """The graph with only the rate changed; one object per rate."""
        if eta not in self._by_eta:
            solver = self._solver_cls(**{**self._solver_config, "eta_infer": eta})
            self._by_eta[eta] = self._base._replace(
                config={**self._base.config, "inference": solver}
            )
        return self._by_eta[eta]

    def _eta_for(self, stiffness: float) -> Optional[float]:
        """Largest eta_max / 2**k with eta * stiffness <= target."""
        if not math.isfinite(stiffness) or stiffness <= 0.0:
            return None
        ratio = self.eta_max * stiffness / self.target
        k = max(0, math.ceil(math.log2(ratio) - 1e-12)) if ratio > 1.0 else 0
        return self.eta_max / 2**k

    def _measure(self, params: GraphParams, update: int, epoch: int) -> None:
        result = self._probe(params, self.probe_clamps, self.key)
        if self.is_epc:
            spectrum = result.host()
            lam = float(spectrum.lambda_max)
            lam_min = float(spectrum.lambda_min)
        else:
            spectrum = None
            lam = float(result)
            lam_min = float("nan")
        stiffness = lam + self._decay
        eta_before = float(self.structure.config["inference"].config["eta_infer"])
        eta_lambda = eta_before * stiffness
        crossed = bool(eta_lambda > 2.0) or not math.isfinite(lam)
        if crossed and self.history:
            self.crossings += 1

        eta_after = None
        lam_used = float("nan")
        if math.isfinite(stiffness):
            self._recent.append(stiffness)
            lam_used = max(self._recent)
            eta_after = self._eta_for(lam_used)
        if eta_after is None:
            # No usable reading (a diverged or dead network): halve the rate.
            eta_after = eta_before / 2.0
        self.structure = self._with_eta(eta_after)

        row = {
            "update": int(update),
            "epoch": int(epoch),
            "lambda_max": lam,
            "lambda_min": lam_min,
            "lambda_used": lam_used,
            "eta_before": eta_before,
            "eta_after": eta_after,
            "eta_lambda_max": eta_lambda,
            "crossed": crossed,
            "band": None,
            "f_weighted": None,
            "f_max": None,
        }
        if spectrum is not None:
            regime = self.structure.config["inference"].regime(spectrum)
            row.update(
                band=regime.band,
                f_weighted=float(regime.f_weighted),
                f_max=float(regime.f_max),
            )
        self.history.append(row)

    # ------------------------------------------------------------ public API

    def start(self, params: GraphParams) -> GraphStructure:
        """Measure at the initial params and return the structure to train."""
        self._measure(params, update=0, epoch=0)
        return self.structure

    def on_iter(self, ctx) -> Optional[GraphStructure]:
        """``structure_callback``: probe every ``every`` updates.

        Returns the new structure when the rate changes, else None.
        """
        if ctx.step % self.every:
            return None
        before = self.structure
        self._measure(ctx.params, update=ctx.step, epoch=ctx.epoch_idx)
        return None if self.structure is before else self.structure

    def summary(self) -> Dict[str, Any]:
        """Numbers to report with the run."""
        if not self.history:
            return {"probes": 0}
        first, last = self.history[0], self.history[-1]
        etas = [r["eta_after"] for r in self.history]
        return {
            "solver": self._solver_cls.__name__,
            "probes": len(self.history),
            "target": self.target,
            "infer_steps": int(self._solver_config["infer_steps"]),
            "eta_max": self.eta_max,
            "eta_start": first["eta_after"],
            "eta_final": last["eta_after"],
            "eta_min": min(etas),
            "rate_changes": sum(
                1 for r in self.history[1:] if r["eta_after"] != r["eta_before"]
            ),
            "lambda_max_start": first["lambda_max"],
            "lambda_max_final": last["lambda_max"],
            "lambda_max_peak": max(r["lambda_max"] for r in self.history),
            "crossings": self.crossings,
            "band_start": first["band"],
            "band_final": last["band"],
            "f_weighted_final": last["f_weighted"],
            "f_max_final": last["f_max"],
        }
