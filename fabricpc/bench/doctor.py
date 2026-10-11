"""Is this machine ready to run the suite? One command, before the hours.

    python -m fabricpc.bench doctor [--out DIR] [--expect-gpu]

Checks, each ok, warn or fail:

- Python is 3.11 or newer.
- What JAX sees: a CPU-only machine is fine for smoke, regress and
  rehearsals, not for the GPU campaign. With --expect-gpu it is a fail,
  because a broken CUDA install quietly falls back to the CPU. A JAX that
  cannot start at all is a fail too.
- Both the CUDA 12 and the CUDA 13 JAX plugins installed: the newer one
  shadows the older and JAX falls back or breaks (this happened on Colab).
- XLA_FLAGS, printed so a run's settings are known.
- Every dataset the registry uses is on disk; a missing one downloads on
  first use, which needs the internet and time.
- Free disk where --out will be, and that it can be written.
- Uncommitted changes, which the results' git sha would not show.
- One small training step compiles and runs.

Only a fail makes the command exit non-zero.
"""

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import List, Set

FAIL_BELOW_GB = 2
WARN_BELOW_GB = 20
TFDS_DATASETS = ("mnist", "fashion_mnist", "cifar10", "cifar100", "tiny_shakespeare")


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # "ok", "warn" or "fail"
    detail: str


def check_python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    return Check("Python", "ok" if ok else "fail", f"{v.major}.{v.minor}.{v.micro}")


def check_devices(expect_gpu: bool = False) -> Check:
    """What JAX sees. With ``expect_gpu``, a CPU-only machine fails."""
    import jax

    try:
        devices = jax.devices()
    except Exception as e:  # noqa: BLE001 - a broken install is a fail, not a crash
        return Check("JAX devices", "fail", f"JAX could not start: {e}")
    kinds = sorted({f"{d.platform}:{d.device_kind}" for d in devices})
    text = f"{len(devices)} device(s): {', '.join(kinds)}"
    if all(d.platform == "cpu" for d in devices):
        if expect_gpu:
            return Check(
                "JAX devices",
                "fail",
                text + "; --expect-gpu was given but JAX sees no GPU (a broken "
                "CUDA install falls back to the CPU)",
            )
        return Check(
            "JAX devices",
            "warn",
            text + "; no GPU, fine for smoke, regress and --rehearse only",
        )
    return Check("JAX devices", "ok", text)


def _installed() -> Set[str]:
    return {(d.metadata["Name"] or "").lower() for d in metadata.distributions()}


def check_cuda_plugins() -> Check:
    names = _installed()
    has12 = any(n.startswith("jax-cuda12") for n in names)
    has13 = any(n.startswith("jax-cuda13") for n in names)
    if has12 and has13:
        return Check(
            "CUDA plugins",
            "warn",
            "both the CUDA 12 and CUDA 13 JAX plugins are installed; keep one, "
            "e.g. pip uninstall -y jax-cuda13-plugin jax-cuda13-pjrt",
        )
    found = "cuda12" if has12 else "cuda13" if has13 else "none (CPU only)"
    return Check("CUDA plugins", "ok", found)


def check_xla_flags() -> Check:
    return Check("XLA_FLAGS", "ok", os.environ.get("XLA_FLAGS") or "(not set)")


def check_datasets() -> List[Check]:
    from fabricpc.bench.bpe import bpe_cache_dir

    tfds_dir = Path(
        os.environ.get("TFDS_DATA_DIR") or Path.home() / "tensorflow_datasets"
    ).expanduser()
    checks = []
    for name in TFDS_DATASETS:
        there = (tfds_dir / name).is_dir()
        checks.append(
            Check(
                f"dataset {name}",
                "ok" if there else "warn",
                (
                    str(tfds_dir / name)
                    if there
                    else f"not in {tfds_dir}; tfds will download it on first use"
                ),
            )
        )
    tiny_dir = Path(
        os.environ.get("FABRICPC_TINYIMAGENET_DIR")
        or Path.home() / "tensorflow_datasets" / "tiny_imagenet_200"
    ).expanduser()
    tiny_zip = tiny_dir / "tiny-imagenet-200.zip"
    checks.append(
        Check(
            "dataset tiny_imagenet_200",
            "ok" if tiny_zip.is_file() else "warn",
            (
                str(tiny_zip)
                if tiny_zip.is_file()
                else f"no {tiny_zip}; the first run will download it (about 250 MB)"
            ),
        )
    )
    bpe = bpe_cache_dir()
    checks.append(
        Check(
            "dataset tinyshakespeare-bpe",
            "ok" if bpe.is_dir() else "warn",
            (
                str(bpe)
                if bpe.is_dir()
                else f"no {bpe}; the tokenizer is built on first use (needs tiny_shakespeare)"
            ),
        )
    )
    return checks


def _existing(path: Path) -> Path:
    """``path`` or its nearest parent that exists, where the space is."""
    path = path.expanduser().resolve()
    while not path.exists():
        path = path.parent
    return path


def check_disk(out) -> Check:
    free_gb = shutil.disk_usage(_existing(Path(out))).free / 1024**3
    status = (
        "fail"
        if free_gb < FAIL_BELOW_GB
        else "warn" if free_gb < WARN_BELOW_GB else "ok"
    )
    return Check("disk", status, f"{free_gb:.1f} GB free where --out {out} goes")


def check_writable(out) -> Check:
    target = _existing(Path(out))
    try:
        with tempfile.NamedTemporaryFile(dir=target):
            pass
    except OSError as e:
        return Check("--out writable", "fail", f"cannot write in {target}: {e}")
    return Check("--out writable", "ok", str(target))


def check_git() -> Check:
    from fabricpc.bench.manifest import _git_dirty, _git_sha

    sha, dirty = _git_sha(), _git_dirty()
    if sha is None:
        return Check("git", "warn", "no commit known; results will have no git sha")
    if dirty:
        return Check(
            "git", "warn", f"{sha[:10]} with uncommitted changes (git_dirty: true)"
        )
    return Check("git", "ok", f"{sha[:10]}, clean")


def check_one_step() -> Check:
    """Compile and run one training step of a small model on random data."""
    import time

    import jax

    from fabricpc.bench.measure import training_step
    from fabricpc.bench.registry import ROWS
    from fabricpc.training.trainer import convert_batch

    try:
        row = ROWS["mnist-mlp-spc"]
        key = jax.random.PRNGKey(0)
        params, structure = row.model_factory(key)
        optimizer = row.optimizer_factory(10)
        x = jax.random.normal(key, (8, 784))
        y = jax.nn.one_hot(jax.random.randint(key, (8,), 0, 10), 10)
        step = training_step(structure, optimizer, algorithm="pc")
        t0 = time.perf_counter()
        out = step(
            jax.tree_util.tree_map(jax.numpy.copy, params),
            optimizer.init(params),
            convert_batch((x, y)),
            key,
        )
        jax.block_until_ready(out[0])
    except Exception as e:  # noqa: BLE001 - say what broke, do not crash
        return Check("one training step", "fail", f"{type(e).__name__}: {e}")
    return Check(
        "one training step",
        "ok",
        f"mnist-mlp-spc compiled and ran in {time.perf_counter() - t0:.1f}s",
    )


def run_checks(out, expect_gpu: bool = False) -> List[Check]:
    return [
        check_python(),
        check_devices(expect_gpu),
        check_cuda_plugins(),
        check_xla_flags(),
        *check_datasets(),
        check_disk(out),
        check_writable(out),
        check_git(),
        check_one_step(),
    ]


def format_checks(checks: List[Check]) -> str:
    mark = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}
    lines = [f"[{mark[c.status]}] {c.name}: {c.detail}" for c in checks]
    counts = {s: sum(c.status == s for c in checks) for s in ("ok", "warn", "fail")}
    lines.append(
        f"{counts['ok']} ok, {counts['warn']} warnings, {counts['fail']} failed"
    )
    return "\n".join(lines)
