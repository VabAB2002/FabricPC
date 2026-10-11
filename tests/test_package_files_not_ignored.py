"""Every source file in the fabricpc package has to be one git will commit.

A broad .gitignore rule (like ``data/`` for downloaded datasets) can quietly
catch a package folder such as ``fabricpc/utils/data``. A new module there
then works on the machine that wrote it and is missing on every fresh clone,
so any import of it fails in CI.
"""

import fnmatch
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _git_works() -> bool:
    if shutil.which("git") is None:
        return False
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _package_files(pattern):
    return sorted(
        p.relative_to(REPO)
        for p in (REPO / "fabricpc").rglob(pattern)
        if "__pycache__" not in p.parts
    )


@pytest.mark.skipif(not _git_works(), reason="needs a git checkout")
def test_no_package_source_file_is_gitignored():
    sources = [str(p) for p in _package_files("*.py") + _package_files("*.json")]
    assert any(s.endswith(".json") for s in sources)
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=REPO,
        input="\n".join(sources) + "\n",
        capture_output=True,
        text=True,
    )
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert ignored == [], f"git would leave these out of a commit: {ignored}"


def test_every_data_file_in_the_package_gets_installed():
    # setuptools installs only .py files unless package-data names the rest;
    # the GPU plan files were missing from a built wheel.
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    package_data = config["tool"]["setuptools"].get("package-data", {})

    def installed(path):
        for package, patterns in package_data.items():
            package_dir = Path(*package.split("."))
            if package_dir in path.parents:
                inner = path.relative_to(package_dir).as_posix()
                if any(fnmatch.fnmatch(inner, pat) for pat in patterns):
                    return True
        return False

    data = _package_files("*.json")
    assert data
    missing = [str(p) for p in data if not installed(p)]
    assert missing == [], f"pyproject.toml package-data leaves out: {missing}"
