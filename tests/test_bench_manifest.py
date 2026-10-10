"""Tests for manifest.json: the record of what produced a result."""

import json

import pytest

import jax

import fabricpc
from fabricpc.bench.manifest import SCHEMA_VERSION, write_manifest
from fabricpc.bench.registry import ROWS


def test_manifest_records_versions_hardware_and_the_command(tmp_path):
    row = ROWS["mnist-mlp-spc"]
    path = write_manifest(
        tmp_path,
        row,
        seeds=[0, 1000],
        command=["python", "-m", "fabricpc.bench", "mnist-mlp-spc", "--trials", "2"],
    )

    assert path == tmp_path / row.id / "manifest.json"
    m = json.loads(path.read_text())

    assert m["schema_version"] == SCHEMA_VERSION
    assert m["row"]["id"] == "mnist-mlp-spc"
    assert m["seeds"] == [0, 1000]
    assert m["command"] == "python -m fabricpc.bench mnist-mlp-spc --trials 2"

    v = m["versions"]
    assert v["fabricpc"] == fabricpc.__version__
    assert v["jax"] == jax.__version__
    for name in ("jaxlib", "optax", "python"):
        assert v[name]

    assert m["platform"]["system"]
    assert m["devices"] == [str(d) for d in jax.devices()]
    assert "xla_flags" in m  # may be empty, but must be recorded
    assert m["created_utc"].endswith("Z")


def test_manifest_records_the_git_commit_when_inside_a_repo(tmp_path):
    row = ROWS["mnist-mlp-spc"]
    m = json.loads(write_manifest(tmp_path, row, seeds=[0], command=[]).read_text())
    # This test file lives inside the FabricPC git repo, so a sha is expected.
    assert m["git_sha"] is None or len(m["git_sha"]) == 40


def test_row_description_includes_the_expected_score():
    from fabricpc.bench.manifest import describe_row
    from fabricpc.bench.registry import ROWS

    judged = describe_row(ROWS["mnist-mlp-spc"])["reference"]
    assert judged["metric"] == "accuracy"
    assert "Colab T4" in judged["source"]
    assert describe_row(ROWS["cifar10-vgg5-spc"])["reference"] is None


def test_git_sha_falls_back_to_the_environment_when_git_cannot_say(monkeypatch):
    # Cloud jobs can run a copy of the code without its .git folder.
    from fabricpc.bench import manifest

    def no_git(*args, **kwargs):
        raise OSError("git is not available")

    monkeypatch.setattr(manifest.subprocess, "run", no_git)
    monkeypatch.setenv("FABRICPC_GIT_SHA", "a" * 40)
    assert manifest._git_sha() == "a" * 40

    monkeypatch.setenv("FABRICPC_GIT_SHA", "not-a-sha")
    assert manifest._git_sha() is None

    monkeypatch.delenv("FABRICPC_GIT_SHA")
    assert manifest._git_sha() is None


@pytest.mark.parametrize(
    "porcelain, expected", [(" M fabricpc/bench/band.py\n", True), ("", False)]
)
def test_the_manifest_says_whether_the_code_had_uncommitted_changes(
    monkeypatch, porcelain, expected
):
    # The git sha alone names a commit that may not hold the code that ran.
    import subprocess

    from fabricpc.bench import manifest

    def fake_run(cmd, **kwargs):
        out = porcelain if "status" in cmd else "a" * 40 + "\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(manifest.subprocess, "run", fake_run)
    assert manifest._git_dirty() is expected


def test_git_dirty_is_none_when_git_cannot_say(monkeypatch):
    from fabricpc.bench import manifest

    def no_git(cmd, **kwargs):
        raise OSError("no git here")

    monkeypatch.setattr(manifest.subprocess, "run", no_git)
    assert manifest._git_dirty() is None


def test_the_manifest_records_the_dirty_flag(tmp_path):
    path = write_manifest(tmp_path, ROWS["mnist-mlp-spc"], seeds=[0], command=["x"])
    assert json.loads(path.read_text())["git_dirty"] in (True, False, None)
