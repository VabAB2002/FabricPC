
import os
from pathlib import Path

import tensorflow_datasets as tfds

from fabricpc.utils.data import tinyimagenet as ti
from fabricpc.bench.definitions.bpe import bpe_cache_dir

class MissingDatasetError(Exception):
    """Raised when a required dataset is unavailable."""
    pass

TFDS_DATASETS = {
    "mnist": "mnist:3.0.1",
    "fashionmnist": "fashion_mnist",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "tinyshakespeare": "tiny_shakespeare",
    "tinyshakespeare-bpe": "tiny_shakespeare",
}

def check_dataset(dataset_name):
    # Synthetic dataset: no download required.
    if dataset_name == "patterns64":
        return

    # Tiny-ImageNet uses a ZIP file, not TFDS.
    if dataset_name == "tinyimagenet":

        folder = os.environ.get(ti._ENV_DIR) or ti._DEFAULT_DIR
        path = Path(folder).expanduser() / ti.ZIP_NAME

        if not path.is_file():
            command = (
                'python -c "import os; from pathlib import Path; '
                'from fabricpc.utils.data import tinyimagenet as ti; '
                'ti._download(Path(os.getenv(ti._ENV_DIR) '
                'or ti._DEFAULT_DIR).expanduser() / ti.ZIP_NAME)"'
            )
            raise MissingDatasetError(
                f"Dataset '{dataset_name}' is unavailable.\n"
                f"Download it using:\n{command}"
            )
        return

    # BPE benchmarks can use an already prepared token cache.
    if dataset_name == "tinyshakespeare-bpe":

        cache = bpe_cache_dir()
        required = ("tokenizer.json", "train.npy", "test.npy")

        if all((cache / name).is_file() for name in required):
            return

    # Check TensorFlow Datasets.
    if dataset_name in TFDS_DATASETS:

        name = TFDS_DATASETS[dataset_name]

        data_dir = os.environ.get("TFDS_DATA_DIR")
        builder = tfds.builder(name, data_dir=data_dir)

        if not builder.is_prepared():
            command = (
                'python -c "import tensorflow_datasets as tfds; '
                f"tfds.builder('{name}').download_and_prepare()\""
            )
            raise MissingDatasetError(
                f"Dataset '{dataset_name}' is unavailable.\n"
                f"Download it using:\n{command}"
            )

    # Other names may be synthetic test fixtures.
    # Possible dataset rejections here.
