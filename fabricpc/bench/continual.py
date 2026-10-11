"""Continual learning: does learning task B make the network forget task A?

    fashionmnist-continual-{spc,epc,backprop}

Inspired by Song et al. (2024, Nature Neuroscience, "Inferring neural
activity before plasticity as a foundation for learning beyond
backpropagation"), where predictive coding forgot less than backprop. The
network shape and the shared 5-unit output follow them; much else does not
(our settings below vs theirs: AdamW 1e-3 with weight decay 0.1 against SGD
with a learning rate tuned per rule; softmax with cross-entropy against a
squared-error output; sPC settling 20 steps at 0.05 against about 64 to 128
steps at 0.1, near equilibrium; a fixed even/odd split against random
classes; 300 images per class and one pass per block against more data and
4 passes per block; scoring after each block, with the PC rows' settled
prediction, against a forward pass after every pass). Results here are not
a replication of theirs:

- Fashion-MNIST split in two tasks: A is the even classes (0, 2, 4, 6, 8),
  B the odd ones (1, 3, 5, 7, 9). Each task is a 5-way problem on one
  shared output layer (label // 2), so learning B writes over A's outputs.
- 300 training images per class, the same ones for all three rows of a
  seed; the whole 10,000-image test set, each image marked with its task.
- 10 blocks, A B A B ..., each one pass over its task's 1,500 images,
  batch 32. The optimizer keeps its state across blocks, as it would in a
  real sequence of tasks.
- 784 -> 32 -> 32 -> 32 -> 5, sigmoid hidden layers. Our output is softmax
  with cross-entropy, like every other classifier here.

How it runs inside the normal trial: ``BlockLoader`` hands out the next
block each time training starts a pass over it, so ``train()`` with
``num_epochs`` = 10 *is* the 10 blocks. The learning curve scores the model
after every block; the test batches are 1,000 images, so its 10 batches
are the whole test set. ``accuracy_task_a`` and ``accuracy_task_b`` read the
``task`` mark on each test image. ``curve_summary`` turns the curve into:

- ``mean_accuracy``: both tasks' accuracy averaged over every block, Song
  et al.'s headline measure;
- ``forgetting``: after each block from the second on, how much accuracy
  the task that was *not* trained lost during that block, averaged;
- ``gain`` and ``forgetting_per_gain``: what the trained task gained per
  block, and forgetting relative to it;
- ``final_task_a`` / ``final_task_b``: each task's accuracy at the end.

The row's own metric is ``accuracy`` on the whole test set at the end,
which is the mean of the two tasks (equal test sizes).
"""

from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.training import metrics as training_metrics

TASKS = ((0, 2, 4, 6, 8), (1, 3, 5, 7, 9))  # A, B
PER_CLASS = 300
BLOCKS = 10
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1000
_MEAN, _STD = 0.2860, 0.3530  # Fashion-MNIST's, as FashionMnistLoader uses
_PROBE_STREAM = 1_000_003  # random stream of the probe batch, apart from blocks


def _task_of(labels: np.ndarray) -> np.ndarray:
    """0 for task A (even classes), 1 for task B (odd)."""
    return (labels % 2).astype(np.int32)


def _prepare(images: np.ndarray, labels: np.ndarray):
    """Flat, normalized pixels; 5-way one-hot targets; the task of each."""
    x = (images.reshape(len(images), -1).astype(np.float32) / 255.0 - _MEAN) / _STD
    y = np.eye(5, dtype=np.float32)[labels // 2]
    return x, y, _task_of(labels)


def pick_per_class(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """``per_class`` indices of every class, the same for the same seed."""
    rng = np.random.default_rng(seed)
    picked = [
        rng.choice(np.flatnonzero(labels == c), size=per_class, replace=False)
        for c in range(10)
    ]
    return np.sort(np.concatenate(picked))


class BlockLoader:
    """The training stream: each pass is the next block, A, B, A, B, ...

    ``len()`` is the batches in one block (both tasks are the same size), so
    ``train()`` sees one block per epoch. Every block is shuffled with its
    own seed, so the stream is the same for every row of a seed.
    """

    def __init__(self, x, y, task, *, batch_size: int, seed: int):
        self._tasks = [(x[task == t], y[task == t]) for t in range(len(TASKS))]
        self._batch_size = batch_size
        self._seed = seed
        self.block = 0  # how many blocks have been handed out
        sizes = {len(tx) for tx, _ in self._tasks}
        if len(sizes) != 1:
            raise ValueError(f"tasks must be the same size, got {sorted(sizes)}")
        self._n = sizes.pop()

    def task_of_block(self, block: int) -> int:
        return block % len(self._tasks)

    def __len__(self):
        return -(-self._n // self._batch_size)

    def probe_batch(self):
        """One fixed batch, half task A and half task B, for the diagnostics
        at the start and end of training and the safety net's probes. It
        does not move the stream on; the first batch of the next pass would
        be whichever task comes next."""
        rng = np.random.default_rng((self._seed, _PROBE_STREAM))
        half = self._batch_size // 2
        xs, ys = [], []
        for (tx, ty), size in zip(self._tasks, (half, self._batch_size - half)):
            pick = rng.permutation(self._n)[:size]
            xs.append(tx[pick])
            ys.append(ty[pick])
        return np.concatenate(xs), np.concatenate(ys)

    def __iter__(self):
        tx, ty = self._tasks[self.task_of_block(self.block)]
        order = np.random.default_rng((self._seed, self.block)).permutation(self._n)
        self.block += 1
        for start in range(0, self._n, self._batch_size):
            idx = order[start : start + self._batch_size]
            yield tx[idx], ty[idx]


class TestLoader:
    """The whole test set in fixed batches, each image marked with its task."""

    def __init__(self, x, y, task, *, batch_size: int):
        self._x, self._y, self._task = x, y, task
        self._batch_size = batch_size

    def __len__(self):
        return -(-len(self._x) // self._batch_size)

    def __iter__(self):
        for start in range(0, len(self._x), self._batch_size):
            end = start + self._batch_size
            yield {
                "x": self._x[start:end],
                "y": self._y[start:end],
                "task": self._task[start:end],
            }


def loaders_from_arrays(train, test, *, seed: int, per_class: int = PER_CLASS):
    """(BlockLoader, TestLoader) from raw (images, labels) pairs."""
    images, labels = train
    keep = pick_per_class(labels, per_class, seed)
    x, y, task = _prepare(images[keep], labels[keep])
    block = BlockLoader(x, y, task, batch_size=BATCH_SIZE, seed=seed)
    tx, ty, ttask = _prepare(*test)
    return block, TestLoader(tx, ty, ttask, batch_size=TEST_BATCH_SIZE)


def loaders(seed: int):
    """The row's loader factory: Fashion-MNIST through tfds, cached."""
    from fabricpc.utils.data.dataloader import FewShotLoader

    train = FewShotLoader._load_raw_split("fashion_mnist", "train")
    test = FewShotLoader._load_raw_split("fashion_mnist", "test")
    return loaders_from_arrays(train, test, seed=seed)


def _task_accuracy(task: int) -> training_metrics.EvalMetric:
    """Accuracy on the test images of one task (weight 0 on the others)."""

    def fn(state, batch, structure):
        correct, _ = training_metrics.accuracy.fn(state, batch, structure)
        mine = (jnp.asarray(batch["task"]) == task).astype(jnp.float32)
        return correct * mine, mine

    return training_metrics.EvalMetric(fn=fn)


EVAL_METRICS = {
    "accuracy": training_metrics.accuracy,
    "accuracy_task_a": _task_accuracy(0),
    "accuracy_task_b": _task_accuracy(1),
}


def curve_summary(points: List[Dict[str, float]]) -> Dict[str, float]:
    """What the per-block curve says about learning one task after another.

    Block k (from 1) trains task A when k is odd and B when it is even.

    - ``mean_accuracy``: the mean over every block of (A + B) / 2, Song et
      al.'s headline measure (they report the mean test error over the run).
    - ``forgetting``: from block 2 on, the accuracy the idle task lost during
      the block, averaged.
    - ``gain``: from block 2 on, the accuracy the trained task gained,
      averaged. A method that learns more per block also has more to lose
      on a shared output layer, so ``forgetting`` alone favours methods that
      learn less; ``forgetting_per_gain`` divides one by the other.
    - ``final_task_a`` / ``final_task_b``: each task at the end (the last
      block trains B, so the final accuracy mostly measures B).
    """
    if not points:
        return {}
    acc = [(p["accuracy_task_a"], p["accuracy_task_b"]) for p in points]
    drops, gains = [], []
    for k in range(1, len(acc)):  # block k + 1, 0-based index k
        idle = 1 if k % 2 == 0 else 0  # A trained in even indices
        trained = 1 - idle
        drops.append(acc[k - 1][idle] - acc[k][idle])
        gains.append(acc[k][trained] - acc[k - 1][trained])
    out = {
        "mean_accuracy": float(np.mean([(a + b) / 2 for a, b in acc])),
        "final_task_a": float(acc[-1][0]),
        "final_task_b": float(acc[-1][1]),
    }
    if drops:
        out["forgetting"] = float(np.mean(drops))
        out["gain"] = float(np.mean(gains))
        if out["gain"] > 0:
            out["forgetting_per_gain"] = out["forgetting"] / out["gain"]
    return out


def model_factory(algorithm: str):
    """784 -> 32 -> 32 -> 32 -> 5, sigmoid hidden layers, softmax output."""
    from fabricpc.bench.registry import _solver_for
    from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
    from fabricpc.core.energy import CrossEntropyEnergy
    from fabricpc.core.initializers import XavierInitializer
    from fabricpc.core.topology import Edge
    from fabricpc.graph_assembly import TaskMap, graph
    from fabricpc.graph_initialization import initialize_params
    from fabricpc.nodes import IdentityNode, Linear

    def build(rng_key: jax.Array):
        pixels = IdentityNode(shape=(784,), name="pixels")
        hidden = [
            Linear(
                shape=(32,),
                activation=SigmoidActivation(),
                name=f"hidden{i}",
                weight_init=XavierInitializer(),
            )
            for i in (1, 2, 3)
        ]
        output = Linear(
            shape=(5,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="class",
            weight_init=XavierInitializer(),
        )
        chain = [pixels, *hidden, output]
        structure = graph(
            nodes=chain,
            edges=[
                Edge(source=a, target=b.slot("in")) for a, b in zip(chain, chain[1:])
            ],
            task_map=TaskMap(x=pixels, y=output),
            inference=_solver_for(algorithm),
        )
        return initialize_params(structure, rng_key), structure

    return build
