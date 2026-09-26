"""Tests for the memory readings in fabricpc.bench.measure."""

import jax

from fabricpc.bench import measure


class _FakeDevice:
    def __init__(self, stats):
        self._stats = stats

    def memory_stats(self):
        return self._stats


def _use_device(monkeypatch, stats):
    monkeypatch.setattr(jax, "local_devices", lambda: [_FakeDevice(stats)])


def test_memory_reports_the_peak_and_the_current_bytes(monkeypatch):
    _use_device(monkeypatch, {"bytes_in_use": 100, "peak_bytes_in_use": 900})

    mem = measure.memory_snapshot()

    assert mem.bytes_in_use == 100
    assert mem.peak_bytes == 900


def test_memory_is_none_when_the_device_cannot_say(monkeypatch):
    _use_device(monkeypatch, None)

    mem = measure.memory_snapshot()

    assert mem.bytes_in_use is None
    assert mem.peak_bytes is None


def test_missing_peak_is_none_not_zero(monkeypatch):
    # Some backends only report the current number.
    _use_device(monkeypatch, {"bytes_in_use": 100})

    mem = measure.memory_snapshot()

    assert mem.bytes_in_use == 100
    assert mem.peak_bytes is None


def test_step_memory_comes_from_the_compiled_training_step(rng_key):
    # The process-wide peak includes compile scratch space (cuDNN autotuning on
    # GPU), which is the same for every algorithm. The compiled step's own
    # memory is what actually differs between backprop and PC.
    from fabricpc.bench.registry import ROWS
    from tests.test_bench_runner import fake_mnist_loaders

    sizes = {}
    for algo, trainer in (("backprop", "backprop"), ("spc", "pc")):
        row = ROWS[f"mnist-mlp-{algo}"]
        params, structure = row.model_factory(rng_key)
        loader, _ = fake_mnist_loaders(rng_key)
        mem = measure.step_memory(
            params,
            structure,
            row.optimizer_factory(10),
            loader,
            rng_key,
            algorithm=trainer,
        )
        assert mem["argument_bytes"] > 0
        assert mem["temp_bytes"] >= 0
        assert mem["total_bytes"] == (
            mem["argument_bytes"]
            + mem["output_bytes"]
            + mem["temp_bytes"]
            - mem["alias_bytes"]
        )
        sizes[algo] = mem["temp_bytes"]
    # PC keeps latent states and runs settling steps: it needs more scratch.
    assert sizes["spc"] > sizes["backprop"]


def test_step_memory_is_none_when_the_step_cannot_be_analysed(monkeypatch, rng_key):
    from fabricpc.bench.registry import ROWS
    from tests.test_bench_runner import fake_mnist_loaders

    monkeypatch.setattr(measure, "make_train_step", lambda *a, **k: (lambda *x: x))
    row = ROWS["mnist-mlp-backprop"]
    params, structure = row.model_factory(rng_key)
    loader, _ = fake_mnist_loaders(rng_key)
    assert (
        measure.step_memory(
            params,
            structure,
            row.optimizer_factory(10),
            loader,
            rng_key,
            algorithm="backprop",
        )
        is None
    )
