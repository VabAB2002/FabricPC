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
