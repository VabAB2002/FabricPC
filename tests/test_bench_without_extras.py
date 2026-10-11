"""Commands that only print must work without the optional extras.

scipy comes with the ``experiments`` extra, the datasets and the tokenizer
with ``tfds``. ``list`` and ``--dry-run`` never compute a confidence
interval or load data, so a plain install should be able to run them.
"""

import subprocess
import sys

import pytest

OPTIONAL = ("scipy", "optuna", "tensorflow", "tensorflow_datasets", "tokenizers")


def _run_without_extras(argv):
    # sys.modules[name] = None makes any "import name" raise ImportError.
    code = (
        f"import sys; sys.modules.update(dict.fromkeys({OPTIONAL!r})); "
        "from fabricpc.bench.__main__ import main; "
        f"sys.exit(main({argv!r}))"
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )


@pytest.mark.parametrize(
    "argv",
    [["list"], ["mnist-mlp-spc", "--dry-run"], ["cifar10-vgg5", "--dry-run"]],
)
def test_printing_commands_need_no_optional_extras(argv):
    proc = _run_without_extras(argv)
    assert proc.returncode == 0, proc.stderr[-2000:]
