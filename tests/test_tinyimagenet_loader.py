"""Tests for TinyImageNetLoader, on a tiny fake copy of the dataset's zip.

The real zip is 248 MB, so these tests build a small zip with the same
folder layout. Most tests store raw pixel bytes instead of JPEGs and swap
in a decoder that reads them back, so they need neither Pillow nor
TensorFlow; one test uses real JPEGs when Pillow is installed.
"""

import hashlib
import io
import zipfile

import numpy as np
import pytest

from fabricpc.utils.data import tinyimagenet
from fabricpc.utils.data.tinyimagenet import TinyImageNetLoader

WNIDS = ["n02", "n01", "n03"]  # not sorted on purpose
TRAIN_PER_CLASS = 4
VAL_LABELS = ["n03", "n01", "n02", "n01", "n03"]  # val_0 .. val_4


def _pixels(value):
    """A 64x64 RGB image where every pixel is ``value``."""
    return np.full((64, 64, 3), value, dtype=np.uint8)


def _train_value(wnid, i):
    # Every training image gets its own grey level, so we can tell them apart.
    return 10 * sorted(WNIDS).index(wnid) + i


def _make_zip(path, encode=lambda img: img.tobytes()):
    """Write a small zip laid out like tiny-imagenet-200.zip."""
    root = "tiny-imagenet-200/"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(root + "wnids.txt", "\n".join(WNIDS) + "\n")
        for wnid in WNIDS:
            z.writestr(root + f"train/{wnid}/{wnid}_boxes.txt", "")
            for i in range(TRAIN_PER_CLASS):
                name = root + f"train/{wnid}/images/{wnid}_{i}.JPEG"
                z.writestr(name, encode(_pixels(_train_value(wnid, i))))
        rows = []
        for i, wnid in enumerate(VAL_LABELS):
            z.writestr(root + f"val/images/val_{i}.JPEG", encode(_pixels(200 + i)))
            rows.append(f"val_{i}.JPEG\t{wnid}\t0\t0\t63\t63")
        z.writestr(root + "val/val_annotations.txt", "\n".join(rows) + "\n")
        # The official test split has no labels; the loader must ignore it.
        z.writestr(root + "test/images/test_0.JPEG", encode(_pixels(1)))
    return path


def _raw_decode(data):
    return np.frombuffer(data, dtype=np.uint8).reshape(64, 64, 3)


@pytest.fixture
def fake_zip(tmp_path, monkeypatch):
    """A fake dataset in tmp_path, read back with the raw-bytes decoder."""
    monkeypatch.setattr(tinyimagenet, "_decode_jpeg", _raw_decode)
    TinyImageNetLoader._split_cache.clear()
    _make_zip(tmp_path / tinyimagenet.ZIP_NAME)
    yield tmp_path
    TinyImageNetLoader._split_cache.clear()


def _plain(split, data_dir, **kw):
    """A loader that hands back raw 0-255 pixels, to check what it read."""
    kw.setdefault("batch_size", 4)
    return TinyImageNetLoader(
        split,
        normalize_mean=0.0,
        normalize_std=1.0 / 255.0,
        data_dir=str(data_dir),
        download=False,
        **kw,
    )


def _grey_levels(images):
    return [int(round(float(v))) for v in np.asarray(images)[:, 0, 0, 0]]


def test_train_batches_have_the_right_shapes_and_one_hot_labels(fake_zip):
    loader = _plain("train", fake_zip, shuffle=False)
    assert loader.num_examples == 3 * TRAIN_PER_CLASS
    assert len(loader) == 3  # 12 images, batches of 4
    batches = list(loader)
    assert len(batches) == 3
    x, y = batches[0]
    assert x.shape == (4, 64, 64, 3)
    assert x.dtype == np.float32
    assert np.asarray(y).shape == (4, 200)
    np.testing.assert_array_equal(np.asarray(y).sum(axis=1), np.ones(4))


def test_labels_follow_the_sorted_class_ids(fake_zip):
    # Class k is the k-th wnid in sorted order, like torchvision's ImageFolder.
    loader = _plain("train", fake_zip, shuffle=False)
    for x, y in loader:
        for level, label in zip(_grey_levels(x), np.argmax(np.asarray(y), axis=1)):
            assert level // 10 == label


def test_val_uses_the_annotation_file_and_test_means_val(fake_zip):
    for split in ("val", "test"):
        loader = _plain(split, fake_zip, shuffle=False)
        assert loader.num_examples == len(VAL_LABELS)
        levels, labels = [], []
        for x, y in loader:
            levels += _grey_levels(x)
            labels += list(np.argmax(np.asarray(y), axis=1))
        assert levels == [200 + i for i in range(len(VAL_LABELS))]
        assert labels == [sorted(WNIDS).index(w) for w in VAL_LABELS]


def test_last_batch_is_kept_even_when_it_is_short(fake_zip):
    loader = _plain("val", fake_zip, shuffle=False)
    assert len(loader) == 2
    assert [x.shape[0] for x, _ in loader] == [4, 1]


def test_images_are_normalized_per_channel(fake_zip):
    loader = TinyImageNetLoader(
        "val",
        batch_size=5,
        shuffle=False,
        normalize_mean=(0.1, 0.2, 0.3),
        normalize_std=(0.5, 0.5, 0.5),
        data_dir=str(fake_zip),
        download=False,
    )
    x, _ = next(iter(loader))
    expected = (200 / 255.0 - np.array([0.1, 0.2, 0.3])) / 0.5
    np.testing.assert_allclose(x[0, 0, 0], expected, rtol=1e-6)


def test_default_normalization_is_the_imagenet_one(fake_zip):
    loader = TinyImageNetLoader("val", 5, data_dir=str(fake_zip), download=False)
    np.testing.assert_allclose(loader.normalize_mean, [0.485, 0.456, 0.406])
    np.testing.assert_allclose(loader.normalize_std, [0.229, 0.224, 0.225])


def test_center_crop_takes_the_middle_of_the_image(fake_zip, monkeypatch):
    def decode_with_marker(data):
        img = _raw_decode(data).copy()
        img[4, 4] = 7  # top-left pixel of a centred 56x56 window
        return img

    monkeypatch.setattr(tinyimagenet, "_decode_jpeg", decode_with_marker)
    loader = _plain("val", fake_zip, shuffle=False, center_crop=56)
    x, _ = next(iter(loader))
    assert x.shape == (4, 56, 56, 3)
    assert int(round(float(x[0, 0, 0, 0]))) == 7


def test_flat_format_flattens_each_image(fake_zip):
    loader = _plain("val", fake_zip, shuffle=False, tensor_format="flat")
    x, _ = next(iter(loader))
    assert x.shape == (4, 64 * 64 * 3)


def test_shuffle_is_seeded_and_changes_every_epoch(fake_zip):
    def order(loader):
        return [lvl for x, _ in loader for lvl in _grey_levels(x)]

    a = _plain("train", fake_zip, shuffle=True, seed=3)
    b = _plain("train", fake_zip, shuffle=True, seed=3)
    first = order(a)
    assert first == order(b)
    assert sorted(first) == sorted(order(_plain("train", fake_zip, shuffle=False)))
    assert order(a) != first  # the second epoch is in a new order
    assert first != order(_plain("train", fake_zip, shuffle=True, seed=4))


def test_max_samples_keeps_only_that_many_images(fake_zip):
    loader = _plain("train", fake_zip, shuffle=False, max_samples=6)
    assert loader.num_examples == 6
    assert len(loader) == 2
    assert sum(x.shape[0] for x, _ in loader) == 6


def test_unknown_split_is_an_error(fake_zip):
    with pytest.raises(ValueError, match="split"):
        _plain("holdout", fake_zip)


def test_missing_data_without_download_is_a_clear_error(tmp_path):
    TinyImageNetLoader._split_cache.clear()
    with pytest.raises(FileNotFoundError, match="tiny-imagenet-200.zip"):
        TinyImageNetLoader("val", 4, data_dir=str(tmp_path), download=False)


def test_download_checks_the_md5_and_keeps_only_the_zip(tmp_path, monkeypatch):
    src = _make_zip(tmp_path / "source.zip")
    md5 = hashlib.md5(src.read_bytes()).hexdigest()
    monkeypatch.setattr(tinyimagenet, "URL", src.as_uri())
    monkeypatch.setattr(tinyimagenet, "MD5", md5)
    monkeypatch.setattr(tinyimagenet, "_decode_jpeg", _raw_decode)
    TinyImageNetLoader._split_cache.clear()

    data_dir = tmp_path / "cache"
    loader = TinyImageNetLoader("val", 4, data_dir=str(data_dir), download=True)
    assert loader.num_examples == len(VAL_LABELS)
    # The zip is kept as is (compressed); nothing is unpacked next to it.
    assert sorted(p.name for p in data_dir.iterdir()) == [tinyimagenet.ZIP_NAME]
    TinyImageNetLoader._split_cache.clear()


def test_download_with_a_wrong_md5_leaves_nothing_behind(tmp_path, monkeypatch):
    src = _make_zip(tmp_path / "source.zip")
    monkeypatch.setattr(tinyimagenet, "URL", src.as_uri())
    monkeypatch.setattr(tinyimagenet, "MD5", "0" * 32)
    TinyImageNetLoader._split_cache.clear()

    data_dir = tmp_path / "cache"
    with pytest.raises(IOError, match="md5"):
        TinyImageNetLoader("val", 4, data_dir=str(data_dir), download=True)
    assert list(data_dir.iterdir()) == []


def test_data_dir_can_come_from_the_environment(fake_zip, monkeypatch):
    monkeypatch.setenv("FABRICPC_TINYIMAGENET_DIR", str(fake_zip))
    loader = TinyImageNetLoader("val", 4, download=False)
    assert loader.num_examples == len(VAL_LABELS)


def test_real_jpegs_decode_to_rgb_including_grey_ones(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")

    def to_jpeg(img):
        buf = io.BytesIO()
        # Tiny-ImageNet has some greyscale JPEGs; store every other one as grey.
        mode = "L" if img[0, 0, 0] % 2 else "RGB"
        pic = Image.fromarray(img[:, :, 0] if mode == "L" else img, mode=mode)
        pic.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    TinyImageNetLoader._split_cache.clear()
    _make_zip(tmp_path / tinyimagenet.ZIP_NAME, encode=to_jpeg)
    loader = _plain("val", tmp_path, shuffle=False, batch_size=5)
    x, _ = next(iter(loader))
    assert x.shape == (5, 64, 64, 3)
    np.testing.assert_allclose(_grey_levels(x), [200, 201, 202, 203, 204], atol=2)
    TinyImageNetLoader._split_cache.clear()


def test_loader_is_exported_from_the_data_package():
    from fabricpc.utils import data
    from fabricpc.utils.data import dataloader

    assert data.TinyImageNetLoader is TinyImageNetLoader
    assert dataloader.TinyImageNetLoader is TinyImageNetLoader
