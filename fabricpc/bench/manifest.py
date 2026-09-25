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
from fabricpc.bench.registry import BenchmarkRow

# Bump this when a field is added, renamed, or changes meaning.
SCHEMA_VERSION = 2  # 2: expected score, band verdict, peak memory


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_sha() -> Optional[str]:
    """The commit this code came from, or None if we are not in a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(fabricpc.__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and len(sha) == 40 else None


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
        "argv0": sys.executable,
    }


def write_manifest(out_dir, row: BenchmarkRow, *, seeds, command) -> Path:
    """Write ``<out_dir>/<row id>/manifest.json`` and return its path."""
    path = Path(out_dir) / row.id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_manifest(row, seeds, command), indent=2))
    return path
