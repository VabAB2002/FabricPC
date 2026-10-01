"""Tiny-ImageNet loader (64x64 RGB, 200 classes).

Tiny-ImageNet is not in tfds, so this loader fetches the official zip from
Stanford's CS231n site (248 MB, md5 checked) the first time it is used and
then reads the JPEGs straight out of that zip. Nothing is unpacked: the zip
is the only file on disk. A split's JPEG bytes are read into memory once
per process (about 230 MB for train, 23 MB for val) and each batch is
decoded when it is needed, so we never hold the whole split as a big uint8
array (100k x 64 x 64 x 3 would be 1.2 GB).

Splits: ``train`` has 100,000 labelled images, ``val`` 10,000. The official
``test`` split has no labels, so like pcx (and most papers) we score on
``val``; asking for ``test`` gives ``val``.

Class k is the k-th WordNet id in sorted order, which is also the order
torchvision's ImageFolder (and so pcx) uses.
"""

import hashlib
import io
import os
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from fabricpc.utils.data.data_utils import one_hot

URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
MD5 = "90528d7ca1a48142e341f4ef8d21d0de"
ZIP_NAME = "tiny-imagenet-200.zip"
_ROOT = "tiny-imagenet-200/"
_DEFAULT_DIR = "~/tensorflow_datasets/tiny_imagenet_200"
_ENV_DIR = "FABRICPC_TINYIMAGENET_DIR"
_IMAGE_SIZE = 64


def _decode_jpeg(data: bytes) -> np.ndarray:
    """Decode one JPEG to a (64, 64, 3) uint8 array.

    Some Tiny-ImageNet images are greyscale, so everything is converted to
    RGB. Uses Pillow when it is installed and TensorFlow (which the other
    image loaders need anyway) when it is not.
    """
    try:
        from PIL import Image
    except ImportError:
        import tensorflow as tf

        return tf.io.decode_jpeg(data, channels=3).numpy()
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))


_PREFETCHER = []


def _prefetcher() -> ThreadPoolExecutor:
    """One helper thread per process for decoding the next batch."""
    if not _PREFETCHER:
        _PREFETCHER.append(ThreadPoolExecutor(max_workers=1))
    return _PREFETCHER[0]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(dest: Path) -> None:
    """Fetch the zip to ``dest``, keeping it only if its md5 is right."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"TinyImageNetLoader: downloading {URL} to {dest} (248 MB, once)")
    try:
        with urllib.request.urlopen(URL) as response, open(part, "wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        got = _md5(part)
        if got != MD5:
            raise IOError(f"Tiny-ImageNet download has md5 {got}, expected {MD5}")
        part.rename(dest)
    finally:
        if part.exists():
            part.unlink()


def _read_split(zip_path: Path, split: str):
    """All JPEG bytes and integer labels of one split, in a fixed order."""
    with zipfile.ZipFile(zip_path) as z:
        wnids = sorted(z.read(_ROOT + "wnids.txt").decode().split())
        label_of = {wnid: k for k, wnid in enumerate(wnids)}
        if split == "train":
            names = sorted(
                n
                for n in z.namelist()
                if n.startswith(_ROOT + "train/") and n.endswith(".JPEG")
            )
            labels = [label_of[n.split("/")[2]] for n in names]
        else:
            rows = z.read(_ROOT + "val/val_annotations.txt").decode().splitlines()
            pairs = sorted(
                (int(r.split("\t")[0][4:-5]), r.split("\t")[1]) for r in rows if r
            )
            names = [_ROOT + f"val/images/val_{i}.JPEG" for i, _ in pairs]
            labels = [label_of[wnid] for _, wnid in pairs]
        jpegs = [z.read(n) for n in names]
    return jpegs, np.asarray(labels, dtype=np.int32)


class TinyImageNetLoader:
    """Tiny-ImageNet loader (64x64 RGB, 200 classes).

    Same interface and batch format as the CIFAR loaders: yields
    (normalized float32 images, one-hot labels) numpy batches, keeps the
    last short batch, and reshuffles on every pass when ``shuffle=True``.
    Construct a fresh instance per training run.

    Args:
        split: 'train', or 'val' (also 'test', which means 'val').
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        seed: Random seed for the shuffle order; None for a random one.
        tensor_format: 'NHWC' for image tensors or 'flat' for flattened rows.
        normalize_mean: Per-channel mean; ImageNet's by default, as in pcx.
        normalize_std: Per-channel std; ImageNet's by default.
        center_crop: If set, cut the middle ``center_crop`` x ``center_crop``
            window out of every image (pcx scores on the centre 56x56).
        max_samples: If set, use only this many images of the split (a
            fixed random subset), for quick checks.
        data_dir: Where the zip lives. Defaults to $FABRICPC_TINYIMAGENET_DIR,
            else ~/tensorflow_datasets/tiny_imagenet_200.
        download: Fetch the zip if it is not there yet.
    """

    _NUM_CLASSES = 200
    # (zip path, split) -> (JPEG bytes, labels), shared by every instance in
    # this process, so building a loader per trial reads the zip only once.
    _split_cache = {}

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",
        normalize_mean: tuple = (0.485, 0.456, 0.406),
        normalize_std: tuple = (0.229, 0.224, 0.225),
        center_crop: int = None,
        max_samples: int = None,
        data_dir: str = None,
        download: bool = True,
    ):
        if split == "test":
            split = "val"
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = np.asarray(normalize_mean, dtype=np.float32)
        self.normalize_std = np.asarray(normalize_std, dtype=np.float32)
        self.center_crop = center_crop
        self._epoch = 0

        data_dir = data_dir or os.environ.get(_ENV_DIR) or _DEFAULT_DIR
        zip_path = Path(data_dir).expanduser() / ZIP_NAME
        if not zip_path.exists():
            if not download:
                raise FileNotFoundError(f"{zip_path} not found (download=False)")
            _download(zip_path)

        key = (str(zip_path.resolve()), split)
        if key not in self._split_cache:
            self._split_cache[key] = _read_split(zip_path, split)
        self._jpegs, labels = self._split_cache[key]

        self._indices = np.arange(len(labels))
        if max_samples is not None and max_samples < len(labels):
            picked = np.random.default_rng(0).permutation(len(labels))
            self._indices = np.sort(picked[:max_samples])
        self._labels = labels
        self.num_examples = len(self._indices)
        self._num_batches = (self.num_examples + batch_size - 1) // batch_size

    def _batch(self, idx):
        """Decode, crop and normalize the images at ``idx``."""
        images = np.stack([_decode_jpeg(self._jpegs[i]) for i in idx])
        if self.center_crop is not None:
            top = (_IMAGE_SIZE - self.center_crop) // 2
            images = images[
                :, top : top + self.center_crop, top : top + self.center_crop
            ]
        images = images.astype(np.float32) / 255.0
        images = (images - self.normalize_mean) / self.normalize_std
        if self.tensor_format == "flat":
            images = images.reshape(images.shape[0], -1)
        labels = one_hot(self._labels[idx], num_classes=self._NUM_CLASSES)
        return images, labels

    def __iter__(self):
        order = self._indices.copy()
        if self.shuffle:
            epoch_seed = self.seed + self._epoch if self.seed is not None else None
            np.random.default_rng(epoch_seed).shuffle(order)
        self._epoch += 1
        starts = range(0, self.num_examples, self.batch_size)
        chunks = [order[s : s + self.batch_size] for s in starts]
        # Decode the next batch in a helper thread while the caller trains on
        # this one (JPEG decoding releases the GIL).
        pending = _prefetcher().submit(self._batch, chunks[0]) if chunks else None
        for k in range(len(chunks)):
            batch = pending.result()
            if k + 1 < len(chunks):
                pending = _prefetcher().submit(self._batch, chunks[k + 1])
            yield batch

    def __len__(self):
        return self._num_batches
