
import pytest

from fabricpc.bench.definitions.dataset_check import (
    TFDS_DATASETS,
    MissingDatasetError,
    check_dataset,
)
from fabricpc.bench.definitions.registry import ROWS

def test_all_registered_datasets_are_covered():
    known = set(TFDS_DATASETS) | {"tinyimagenet", "patterns64"}

    registered = {row.dataset for row in ROWS.values()}

    assert registered <= known

@pytest.mark.parametrize("dataset,tfds_name", list(TFDS_DATASETS.items()),)
def test_missing_tfds_dataset(
        tmp_path, monkeypatch, dataset, tfds_name
):
    monkeypatch.setenv(
        "TFDS_DATA_DIR", str(tmp_path / "empty_tfds")
    )
    monkeypatch.setenv(
        "FABRICPC_BPE_DIR", str(tmp_path / "empty_bpe")
    )

    with pytest.raises(MissingDatasetError) as error:
        check_dataset(dataset)

    message = str(error.value)

    assert dataset in message
    assert tfds_name in message
    assert "Download it using:" in message

def test_missing_tinyimagenet(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "FABRICPC_TINYIMAGENET_DIR",
        str(tmp_path / "empty_tinyimagenet"),
    )

    with pytest.raises(MissingDatasetError) as error:
        check_dataset("tinyimagenet")

    assert "tinyimagenet" in str(error.value)
    assert "Download it using:" in str(error.value)

def test_patterns64_does_not_require_download():
    assert check_dataset("patterns64") is None
