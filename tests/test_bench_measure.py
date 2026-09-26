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


def test_timing_compiles_the_step_once_and_reports_its_memory(monkeypatch, rng_key):
    # The step is compiled once; timing and the memory report share it.
    from fabricpc.bench.registry import ROWS
    from tests.test_bench_runner import fake_mnist_loaders

    lowered = []
    real = measure.make_train_step

    def counting_make_train_step(*args, **kwargs):
        step = real(*args, **kwargs)

        class Counted:
            def lower(self, *a, **k):
                lowered.append(1)
                return step.lower(*a, **k)

            def __call__(self, *a, **k):
                return step(*a, **k)

        return Counted()

    monkeypatch.setattr(measure, "make_train_step", counting_make_train_step)
    row = ROWS["mnist-mlp-spc"]
    params, structure = row.model_factory(rng_key)
    loader, _ = fake_mnist_loaders(rng_key)

    timing = measure.time_steps(
        params,
        structure,
        row.optimizer_factory(10),
        loader,
        rng_key,
        algorithm="pc",
        warmup_steps=1,
        timed_steps=2,
    )

    assert len(lowered) == 1
    assert timing.step_memory["total_bytes"] > 0
    assert timing.compile_time_s > 0 and timing.step_time_ms > 0


def test_epc_regime_labels_how_pc_like_an_epc_run_is(rng_key):
    from fabricpc.bench.registry import ROWS
    from tests.test_bench_runner import fake_mnist_loaders

    loader, _ = fake_mnist_loaders(rng_key)
    batch = next(iter(loader))

    row = ROWS["mnist-mlp-epc"]
    params, structure = row.model_factory(rng_key)
    regime = measure.epc_regime(params, structure, batch, rng_key)
    assert regime["band"] in (
        "backprop-like",
        "partially relaxed",
        "near PC equilibrium",
        "no positive curvature",
    )
    assert 0.0 <= regime["f_weighted"] <= 1.0
    assert isinstance(regime["unstable"], bool)
    assert isinstance(regime["label"], str)

    # Only ePC has a regime; sPC and backprop get None.
    for algo in ("spc", "backprop"):
        params, structure = ROWS[f"mnist-mlp-{algo}"].model_factory(rng_key)
        assert measure.epc_regime(params, structure, batch, rng_key) is None
