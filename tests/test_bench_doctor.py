"""Tests for ``python -m fabricpc.bench doctor``: is this machine ready?"""

import collections

from fabricpc.bench import doctor
from fabricpc.bench.__main__ import main

Usage = collections.namedtuple("Usage", "total used free")
GB = 1024**3


def _by_name(checks):
    return {c.name: c for c in checks}


def test_low_disk_fails_and_a_little_disk_warns(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: Usage(0, 0, 1 * GB))
    assert doctor.check_disk(tmp_path).status == "fail"

    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: Usage(0, 0, 10 * GB))
    assert doctor.check_disk(tmp_path).status == "warn"

    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: Usage(0, 0, 90 * GB))
    assert doctor.check_disk(tmp_path).status == "ok"


def test_disk_is_measured_where_out_will_be_even_before_it_exists(tmp_path):
    check = doctor.check_disk(tmp_path / "not" / "made" / "yet")
    assert check.status in ("ok", "warn", "fail")
    assert "GB free" in check.detail


def test_both_cuda_plugins_installed_is_flagged(monkeypatch):
    # The clash we hit on Colab: the cuda13 plugin shadows the cuda12 one.
    monkeypatch.setattr(
        doctor,
        "_installed",
        lambda: {"jax-cuda12-plugin", "jax-cuda13-plugin", "jax-cuda13-pjrt"},
    )
    check = doctor.check_cuda_plugins()
    assert check.status == "warn"
    assert "pip uninstall" in check.detail

    monkeypatch.setattr(doctor, "_installed", lambda: {"jax-cuda12-plugin"})
    assert doctor.check_cuda_plugins().status == "ok"


def test_a_missing_dataset_warns_that_it_will_download(monkeypatch, tmp_path):
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("FABRICPC_TINYIMAGENET_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("FABRICPC_BPE_DIR", str(tmp_path / "nobpe"))

    checks = doctor.check_datasets()

    assert {c.status for c in checks} == {"warn"}
    assert all("download" in c.detail or "built" in c.detail for c in checks)
    assert "cifar10" in {c.name.split()[-1] for c in checks}


def test_a_dataset_on_disk_is_ok(monkeypatch, tmp_path):
    (tmp_path / "mnist").mkdir()
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path))

    checks = _by_name(doctor.check_datasets())

    assert checks["dataset mnist"].status == "ok"


def test_an_unwritable_out_folder_fails(tmp_path):
    out = tmp_path / "locked"
    out.mkdir()
    out.chmod(0o500)
    try:
        assert doctor.check_writable(out).status == "fail"
    finally:
        out.chmod(0o700)
    assert doctor.check_writable(tmp_path / "new").status == "ok"


def test_one_training_step_compiles_and_runs_here():
    check = doctor.check_one_step()
    assert check.status == "ok", check.detail


def test_the_command_exits_nonzero_only_on_a_fail(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        doctor, "check_one_step", lambda: doctor.Check("step", "ok", "")
    )
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: Usage(0, 0, 90 * GB))
    assert main(["doctor", "--out", str(tmp_path)]) == 0
    assert "Python" in capsys.readouterr().out

    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: Usage(0, 0, 1 * GB))
    assert main(["doctor", "--out", str(tmp_path)]) == 1


def test_expect_gpu_turns_a_cpu_only_machine_into_a_failure(monkeypatch):
    # Before a GPU campaign, a CUDA install that fell back to the CPU must
    # stop the run, not pass with a warning.
    assert doctor.check_devices(expect_gpu=False).status in ("ok", "warn")
    assert doctor.check_devices(expect_gpu=True).status == "fail"  # this is a CPU


def test_a_jax_that_cannot_start_is_a_failed_check_not_a_crash(monkeypatch):
    import jax

    def broken():
        raise RuntimeError("Unable to initialize backend 'cuda'")

    monkeypatch.setattr(jax, "devices", broken)
    check = doctor.check_devices()
    assert check.status == "fail"
    assert "cuda" in check.detail


def test_the_command_takes_expect_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(
        doctor, "check_one_step", lambda: doctor.Check("step", "ok", "")
    )
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda p: Usage(0, 0, 90 * GB))
    assert main(["doctor", "--out", str(tmp_path), "--expect-gpu"]) == 1
