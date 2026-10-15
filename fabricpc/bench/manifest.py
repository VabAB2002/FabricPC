"""manifest.json: everything someone needs to reproduce a run.

Versions, hardware, the flags JAX was given, the row's settings, the seed
list, and the exact command. A result without this file is not citable.
"""

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import List, Optional

import jax

import fabricpc
from fabricpc.bench.measure import speed_test_ms
from fabricpc.bench.registry import BenchmarkRow

# Bump this when a field is added, renamed, or changes meaning.
# 2: expected score, band verdict, peak memory
# 3: summary lists checkpoints and aggregates peak memory; compare files carry
#    a version too
# 4: summary says how PC-like the runs were (epc_regime, backprop_alignment);
#    trials and summary carry eval_batches_capped; manifest carries git_dirty;
#    contrasts carry p_holm, significant_holm_05, effect_size, hedges_gz
# 5: manifest carries speed_test_ms, the machine's speed when the run started
# 6: a two-row compare file carries holm_family, and its p_holm is NaN
#    (uncorrected) unless both rows are in one family; trials carry
#    timing_version (measure.TIMING_VERSION) and safety_net_s
# 7: trial compute carries xla_flops_per_update, xla_flops_per_update_backprop
#    and xla_pc_to_backprop_ratio; achieved_tflops uses XLA's count
# 8: trials carry requested_epochs, the epochs asked for before
#    max_train_steps cut them (num_epochs is what was trained)
SCHEMA_VERSION = 8


def _speed_test() -> Optional[float]:
    try:
        return speed_test_ms()
    except Exception:  # noqa: BLE001 - a missing number, not a failed run
        return None


def _git_dirty() -> Optional[bool]:
    """True when the code had changes git has not committed, so the git sha
    alone does not say what ran; None when git cannot say."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path(fabricpc.__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_sha() -> Optional[str]:
    """The commit this code came from, or None if nothing can say.

    Asks git first. A cloud job may run a copy of the code without its .git
    folder, so the job can pass the commit in ``FABRICPC_GIT_SHA`` instead.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(fabricpc.__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = out.stdout.strip()
        if out.returncode == 0 and _is_full_sha(sha):
            return sha
    except (OSError, subprocess.TimeoutExpired):
        pass
    sha = os.environ.get("FABRICPC_GIT_SHA", "").strip()
    return sha if _is_full_sha(sha) else None


def _is_full_sha(text: str) -> bool:
    return len(text) == 40 and all(c in "0123456789abcdef" for c in text)


def describe_row(row: BenchmarkRow) -> dict:
    """The plain-data view of a row."""
    return {
        "id": row.id,
        "dataset": row.dataset,
        "model": row.model,
        "algorithm": row.algorithm,
        "n_trials": row.n_trials,
        "batch_size": row.batch_size,
        "train_config": dict(row.train_config),
        "tier": row.tier,
        "reference": asdict(row.reference) if row.reference else None,
    }


def build_manifest(row: BenchmarkRow, seeds: List[int], command: List[str]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "row": describe_row(row),
        "seeds": list(seeds),
        "command": " ".join(command),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "versions": {
            "fabricpc": fabricpc.__version__,
            "jax": jax.__version__,
            "jaxlib": _package_version("jaxlib"),
            "optax": _package_version("optax"),
            "tensorflow_datasets": _package_version("tensorflow-datasets"),
            "python": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "devices": [str(d) for d in jax.devices()],
        "default_backend": jax.default_backend(),
        # Until the library ships named flag profiles, record the raw flags.
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        # A fixed matmul timed now; trend divides step times by it.
        "speed_test_ms": _speed_test(),
        "argv0": sys.executable,
    }


def write_manifest(out_dir, row: BenchmarkRow, *, seeds, command) -> Path:
    """Write ``<out_dir>/<row id>/manifest.json`` and return its path."""
    path = Path(out_dir) / row.id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest(row, seeds, command), indent=2))
    return path
