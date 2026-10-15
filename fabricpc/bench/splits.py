"""Which part of a dataset a run trains on and which part it is scored on.

Tuning trains on most of the training set and scores on the rest, so the
test set is never used to pick settings. The cut is fixed (not per seed),
so every try and every method is scored on the same examples.
"""

from typing import Tuple

TUNE_TRAIN_SPLIT = "train[:90%]"
TUNE_HOLDOUT_SPLIT = "train[90%:]"


def splits(tuning: bool) -> Tuple[str, str]:
    """(split to train on, split to score on) for a run or for tuning."""
    return (TUNE_TRAIN_SPLIT, TUNE_HOLDOUT_SPLIT) if tuning else ("train", "test")
