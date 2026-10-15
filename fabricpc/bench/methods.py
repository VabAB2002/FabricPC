"""The training methods the suite compares, in one place.

A row id ends in its method's name (``mnist-mlp-spc``). Everything that
needs the list of methods, or what a method is, reads it from here.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Method:
    """One way to train a graph.

    ``trainer`` is the algorithm ``fabricpc.training.train`` runs ("pc" or
    "backprop"); a PC method's solver lives in the graph. ``settles`` says
    whether training relaxes latents first, so solver settings and the
    settled fraction mean something. ``safety_target`` is where the safety
    net starts shrinking the settle's rate (eta * stiffness), None for no
    safety net; see ``fabricpc.bench.registry.RateControl``.
    """

    name: str
    trainer: str
    settles: bool
    safety_target: Optional[float]
    about: str


# Safety-net targets come from the end-of-training values of our own runs:
# ePC rows were healthy at 0.07 or below (MNIST, Fashion-MNIST, VGG-5) and
# broke at 3 to 15 (the autoencoder, 3 of 5 seeds); sPC rows were healthy up
# to 1.7 (Fashion-MNIST 1.6, VGG-5 1.0 to 1.7) and degraded above 2
# (autoencoder 1.9 to 2.4, the transformer about 650). So ePC acts at 1.0
# and sPC at 1.8, and runs that were healthy train as before.
METHODS: Dict[str, Method] = {
    m.name: m
    for m in (
        Method("spc", "pc", True, 1.8, "state-based PC: settles the latents"),
        Method("epc", "pc", True, 1.0, "error-based PC: settles the errors"),
        Method("backprop", "backprop", False, None, "backpropagation"),
    )
}

# The order rows of a family are built and listed in.
ALGORITHMS: Tuple[str, ...] = tuple(METHODS)

# The order results are shown in: the baseline first.
DISPLAY_ORDER: Tuple[str, ...] = ("backprop",) + tuple(
    m for m in ALGORITHMS if m != "backprop"
)


def trainer_algorithm(method: str) -> str:
    """What ``train`` / ``evaluate`` call the method: "pc" or "backprop"."""
    return METHODS[method].trainer


def split_row_id(row_id: str) -> Tuple[str, Optional[str]]:
    """``(family, method)`` from a row id; method None if it names none."""
    family, _, last = row_id.rpartition("-")
    if family and last in METHODS:
        return family, last
    return row_id, None
