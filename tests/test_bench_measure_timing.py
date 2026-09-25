"""Tests for how time_steps reads the training data."""

from fabricpc.bench import measure
from fabricpc.bench.registry import ROWS
from tests.test_bench_runner import fake_mnist_loaders


class CountingLoader:
    """Wraps a loader and counts how many batches were taken from it."""

    def __init__(self, loader):
        self._loader = loader
        self.taken = 0

    def __iter__(self):
        for batch in self._loader:
            self.taken += 1
            yield batch


def test_timing_reads_only_the_batches_it_needs(rng_key):
    row = ROWS["mnist-mlp-backprop"]
    train, _ = fake_mnist_loaders(rng_key, n_batches=20)
    counting = CountingLoader(train)
    params, structure = row.model_factory(rng_key)

    timing = measure.time_steps(
        params,
        structure,
        row.optimizer_factory(),
        counting,
        rng_key,
        algorithm="backprop",
        warmup_steps=1,
        timed_steps=2,
    )

    assert counting.taken == 4  # 1 compile + 1 warmup + 2 timed, not all 20
    assert timing.timed_steps == 2
    assert timing.step_time_ms > 0
