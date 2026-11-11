import tensorflow_datasets as tfds
import numpy as np

class TFDSMnistIterator:
    """Wraps a TFDS dataset into a batch iterator compatible with the benchmark runner."""
    def __init__(self, dataset, batch_size, shuffle=True):
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        self._iter = iter(tfds.as_numpy(dataset))
        self._dataset = dataset

    def __iter__(self):
        # Re-initialize the underlying iterator for new epochs
        self._iter = iter(tfds.as_numpy(self._dataset))
        return self

    def __next__(self):
        images, labels = next(self._iter)
        # Flatten images to match (batch_size, 784) and scale to [0, 1] if needed
        images = images.astype(np.float32) / 255.0
        images = images.reshape(images.shape[0], -1)
        return images, labels

    def __len__(self):
        # Approximate or calculated length if needed by the trainer
        return len(self._dataset)


def load_mnist(batch_size=64, train=True):
    """Load MNIST dataset using TensorFlow Datasets and standard splits."""
    split = "train" if train else "test"
    ds = tfds.load("mnist", split=split, as_supervised=True)
    return TFDSMnistIterator(ds, batch_size=batch_size, shuffle=train)