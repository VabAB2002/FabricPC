"""Every source file in the fabricpc package has to be one git will commit.

A broad .gitignore rule (like ``data/`` for downloaded datasets) can quietly
catch a package folder such as ``fabricpc/utils/data``. A new module there
then works on the machine that wrote it and is missing on every fresh clone,
so any import of it fails in CI.
"""

import shutil
import subprocess
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


@pytest.mark.skipif(not _git_works(), reason="needs a git checkout")
def test_no_package_source_file_is_gitignored():
    sources = sorted(
        str(p.relative_to(REPO))
        for p in (REPO / "fabricpc").rglob("*.py")
        if "__pycache__" not in p.parts
    )
    assert sources
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=REPO,
        input="\n".join(sources) + "\n",
        capture_output=True,
        text=True,
    )
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert ignored == [], f"git would leave these out of a commit: {ignored}"
