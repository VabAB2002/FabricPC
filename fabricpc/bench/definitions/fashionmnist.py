"""FashionMNIST dataset loader using TensorFlow Datasets."""

import numpy as np
import tensorflow_datasets as tfds


class TFDSFashionMnistIterator:
    """Wraps a TFDS FashionMNIST dataset into a flat-image batch iterator."""

    def __init__(self, dataset, batch_size: int, shuffle: bool = True, seed: int | None = None):
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=1024, seed=seed, reshuffle_each_iteration=True
            )
        self._dataset = dataset.batch(batch_size, drop_remainder=False)
        self._iter = None

    def __iter__(self):
        self._iter = iter(tfds.as_numpy(self._dataset))
        return self

    def __next__(self):
        images, labels = next(self._iter)
        images = images.astype(np.float32) / 255.0
        # Standard FashionMNIST channel statistics
        images = (images - 0.2860) / 0.3530
        images = images.reshape(images.shape[0], -1)
        return images, labels

    def __len__(self):
        return len(self._dataset)


def load_fashionmnist(
    batch_size: int = 64, train: bool = True, seed: int | None = None
) -> TFDSFashionMnistIterator:
    """Load FashionMNIST dataset via TensorFlow Datasets with standard splits."""
    split = "train" if train else "test"
    ds = tfds.load("fashion_mnist:3.0.1", split=split, as_supervised=True)
    return TFDSFashionMnistIterator(ds, batch_size=batch_size, shuffle=train, seed=seed)