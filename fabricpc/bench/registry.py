"""The list of benchmark rows.

A row is one dataset, one model, one training algorithm. Rows in the same
family (same dataset and model) share a graph so the algorithms can be
compared on equal footing. Row ids look like ``mnist-mlp-spc``.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple

import jax
import optax

from fabricpc.core.activations import (
    GeluActivation,
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD, InferenceSGDNormClip
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import build_resnet18, create_deep_transformer, create_vgg
from fabricpc.nodes import IdentityNode, Linear

# The three ways we can train the same graph.
ALGORITHMS: Tuple[str, ...] = ("spc", "epc", "backprop")

# What each row's model factory returns: (params, structure).
ModelFactory = Callable[[jax.Array], Tuple[object, object]]
LoaderFactory = Callable[[int], Tuple[object, object]]


@dataclass(frozen=True)
class Reference:
    """The score a full run of a row is expected to land near.

    ``floor`` is the smallest allowed miss, in the metric's own units (0.005
    is half a percentage point of accuracy). ``fabricpc.bench.band`` widens
    it to two standard errors when the run is noisier than that.
    """

    metric: str
    value: float
    source: str  # where the number came from, so anyone can check it
    floor: float = 0.005


@dataclass(frozen=True)
class BenchmarkRow:
    """Everything needed to run one benchmark row."""

    id: str
    dataset: str
    model: str
    algorithm: str
    model_factory: ModelFactory
    loader_factory: LoaderFactory
    optimizer_factory: Callable[[], optax.GradientTransformation]
    train_config: Mapping[str, object]
    batch_size: int
    n_trials: int = 5
    tier: int = 1
    reference: Optional[Reference] = None  # None until a full run sets one


def _solver_for(algorithm: str):
    """Pick the inference solver for an algorithm.

    Backprop does not settle latents, but the graph still needs an inference
    object to build, so it gets the same one as spc.
    """
    if algorithm == "epc":
        return EPCInference()
    return InferenceSGD(eta_infer=0.05, infer_steps=20)


def _mnist_mlp_factory(algorithm: str) -> ModelFactory:
    """Build the MNIST MLP from the mnist demo: 784 -> 256 -> 64 -> 10."""

    def build(rng_key: jax.Array):
        pixels = IdentityNode(shape=(784,), name="pixels")
        hidden1 = Linear(
            shape=(256,),
            activation=SigmoidActivation(),
            name="hidden1",
            weight_init=XavierInitializer(),
        )
        hidden2 = Linear(
            shape=(64,),
            activation=SigmoidActivation(),
            name="hidden2",
            weight_init=XavierInitializer(),
        )
        output = Linear(
            shape=(10,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="class",
            weight_init=XavierInitializer(),
        )
        structure = graph(
            nodes=[pixels, hidden1, hidden2, output],
            edges=[
                Edge(source=pixels, target=hidden1.slot("in")),
                Edge(source=hidden1, target=hidden2.slot("in")),
                Edge(source=hidden2, target=output.slot("in")),
            ],
            task_map=TaskMap(x=pixels, y=output),
            inference=_solver_for(algorithm),
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _flat_image_loaders(dataset: str, batch_size: int) -> LoaderFactory:
    """Loaders for one trial of a flattened-image dataset (MNIST-like).

    ``dataset`` is "mnist" or "fashionmnist". The seed fixes the shuffle order.
    """

    def build(seed: int):
        # Imported here so the registry can be listed without tensorflow.
        from fabricpc.utils.data import dataloader

        loader_cls = {
            "mnist": dataloader.MnistLoader,
            "fashionmnist": dataloader.FashionMnistLoader,
        }[dataset]
        train = loader_cls(
            "train",
            batch_size=batch_size,
            tensor_format="flat",
            shuffle=True,
            seed=seed,
        )
        test = loader_cls(
            "test", batch_size=batch_size, tensor_format="flat", shuffle=False
        )
        return train, test

    return build


# Mean test accuracy from our first full runs: 5 seeds x 20 epochs on a
# Colab T4, 2026-09-23. These are our own numbers, not pcx's.
_MLP_STAGE1_SOURCE = "Team 16 first full run, Colab T4, 5 seeds x 20 epochs, 2026-09-23"
_MLP_STAGE1_ACCURACY = {
    "mnist": {"spc": 0.9818, "epc": 0.9815, "backprop": 0.9816},
    "fashionmnist": {"spc": 0.8840, "epc": 0.8881, "backprop": 0.8888},
}


def _mlp_family(dataset: str) -> Dict[str, BenchmarkRow]:
    """The 784-256-64-10 MLP on a flattened-image dataset, three algorithms."""
    rows = {}
    batch_size = 200
    for algo in ALGORITHMS:
        row_id = f"{dataset}-mlp-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset=dataset,
            model="mlp",
            algorithm=algo,
            model_factory=_mnist_mlp_factory(algo),
            loader_factory=_flat_image_loaders(dataset, batch_size),
            optimizer_factory=lambda: optax.adamw(0.001, weight_decay=0.1),
            train_config={"num_epochs": 20},
            batch_size=batch_size,
            reference=Reference(
                metric="accuracy",
                value=_MLP_STAGE1_ACCURACY[dataset][algo],
                source=_MLP_STAGE1_SOURCE,
            ),
        )
    return rows


def _vgg_solver_for(algorithm: str):
    """Solver settings for the VGG rows.

    pcx trained VGG-5 with 8 settling steps on CIFAR-10. The state learning
    rate here is a starting point for the GPU runs, not a tuned value.
    """
    if algorithm == "epc":
        return EPCInference()
    return InferenceSGD(eta_infer=0.015, infer_steps=8)


def _cifar10_vgg5_factory(algorithm: str) -> ModelFactory:
    """VGG-5 on 32x32x3 images with 10 classes, pcx's channel layout."""

    def build(rng_key: jax.Array):
        structure = create_vgg(
            5,
            input_shape=(32, 32, 3),
            num_classes=10,
            inference=_vgg_solver_for(algorithm),
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _cifar10_loaders(batch_size: int) -> LoaderFactory:
    """CIFAR-10 loaders for one trial. No augmentation yet; see the design doc."""

    def build(seed: int):
        from fabricpc.utils.data.dataloader import Cifar10Loader

        train = Cifar10Loader("train", batch_size=batch_size, shuffle=True, seed=seed)
        test = Cifar10Loader("test", batch_size=batch_size, shuffle=False)
        return train, test

    return build


def _cifar10_vgg5_family() -> Dict[str, BenchmarkRow]:
    # Batch size and epoch count follow pcx's VGG-5 CIFAR-10 runs. The weight
    # learning rate and weight decay are pcx's centered-nudging values, rounded.
    rows = {}
    batch_size = 128
    for algo in ALGORITHMS:
        row_id = f"cifar10-vgg5-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="cifar10",
            model="vgg5",
            algorithm=algo,
            model_factory=_cifar10_vgg5_factory(algo),
            loader_factory=_cifar10_loaders(batch_size),
            optimizer_factory=lambda: optax.adamw(2.6e-4, weight_decay=1.2e-5),
            train_config={"num_epochs": 50},
            batch_size=batch_size,
        )
    return rows


def _resnet_solver_for(algorithm: str):
    """The ResNet-18 demo's solver settings, kept identical here.

    ePC: eta 3e-4, 5 steps (the demo's stable point over 100 epochs).
    sPC: norm-clipped SGD, eta 0.2, 120 steps. Backprop reuses the sPC
    object only because the graph needs one to build.
    """
    if algorithm == "epc":
        return EPCInference(eta_infer=3e-4, infer_steps=5)
    return InferenceSGDNormClip(eta_infer=0.2, infer_steps=120, max_norm=1.0)


def _cifar10_resnet18_factory(algorithm: str) -> ModelFactory:
    """ResNet-18 with muPC parameterization and GELU, as in the demo."""

    def build(rng_key: jax.Array):
        structure = build_resnet18(
            weight_init=MuPCInitializer(),
            inference=_resnet_solver_for(algorithm),
            scaling=MuPCConfig(include_output=False),
            output_weight_init=XavierInitializer(),
            activation=GeluActivation(),
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _cifar10_resnet18_family() -> Dict[str, BenchmarkRow]:
    # Batch size, learning rate, weight decay, and epoch count follow the
    # demo's 100-epoch runs. No augmentation yet; see the design doc.
    rows = {}
    batch_size = 256
    for algo in ALGORITHMS:
        row_id = f"cifar10-resnet18-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="cifar10",
            model="resnet18",
            algorithm=algo,
            model_factory=_cifar10_resnet18_factory(algo),
            loader_factory=_cifar10_loaders(batch_size),
            optimizer_factory=lambda: optax.adamw(1e-3, weight_decay=0.01),
            train_config={"num_epochs": 100},
            batch_size=batch_size,
        )
    return rows


# The transformer demo's tuned character-level settings (val perplexity 12.2).
_CHAR_TRANSFORMER = {
    "embed_dim": 64,
    "num_heads": 8,
    "mlp_dim": 256,
    "depth": 2,
    "seq_len": 128,
    "vocab_size": 65,  # Tiny Shakespeare's character set
    "batch_size": 16,
    "num_epochs": 5,
    "infer_steps": 12,
    "lr": 1.21e-4,
    "eta_infer": 0.0175,
}


def _transformer_solver_for(algorithm: str):
    if algorithm == "epc":
        return EPCInference()
    return InferenceSGDNormClip(
        eta_infer=_CHAR_TRANSFORMER["eta_infer"],
        infer_steps=_CHAR_TRANSFORMER["infer_steps"],
        max_norm=1.0,
    )


def _char_transformer_factory(algorithm: str) -> ModelFactory:
    def build(rng_key: jax.Array):
        c = _CHAR_TRANSFORMER
        structure = create_deep_transformer(
            depth=c["depth"],
            embed_dim=c["embed_dim"],
            num_heads=c["num_heads"],
            mlp_dim=c["mlp_dim"],
            seq_len=c["seq_len"],
            vocab_size=c["vocab_size"],
            inference=_transformer_solver_for(algorithm),
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _tinyshakespeare_loaders(batch_size: int, seq_len: int) -> LoaderFactory:
    def build(seed: int):
        from fabricpc.utils.data.dataloader import CharDataLoader

        train = CharDataLoader(
            "train", seq_len=seq_len, batch_size=batch_size, shuffle=True, seed=seed
        )
        test = CharDataLoader(
            "test", seq_len=seq_len, batch_size=batch_size, shuffle=False
        )
        return train, test

    return build


def _tinyshakespeare_transformer_family() -> Dict[str, BenchmarkRow]:
    # Constant learning rate for now; the demo uses a cosine schedule that
    # needs the loader's length, which the row does not know up front.
    rows = {}
    c = _CHAR_TRANSFORMER
    for algo in ALGORITHMS:
        row_id = f"tinyshakespeare-transformer-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="tinyshakespeare",
            model="transformer",
            algorithm=algo,
            model_factory=_char_transformer_factory(algo),
            loader_factory=_tinyshakespeare_loaders(c["batch_size"], c["seq_len"]),
            optimizer_factory=lambda: optax.adam(_CHAR_TRANSFORMER["lr"]),
            train_config={"num_epochs": c["num_epochs"]},
            batch_size=c["batch_size"],
        )
    return rows


ROWS: Dict[str, BenchmarkRow] = {}
ROWS.update(_mlp_family("mnist"))
ROWS.update(_mlp_family("fashionmnist"))
ROWS.update(_cifar10_vgg5_family())
ROWS.update(_cifar10_resnet18_family())
ROWS.update(_tinyshakespeare_transformer_family())
