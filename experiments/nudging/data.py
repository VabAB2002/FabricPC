"""Data loaders matching pcx's VGG-5 pipeline.

pcx normalizes CIFAR-10 with std (0.2023, 0.1994, 0.2010), not the usual
(0.247, 0.243, 0.261), flips and crops training images, and drops the last
short training batch (drop_last=True).
"""

PCX_NORMALIZE = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tinyimagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


class DropLast:
    """Wrap a loader and skip any batch smaller than ``batch_size``."""

    def __init__(self, loader, batch_size):
        self.loader = loader
        self.batch_size = batch_size

    def __iter__(self):
        for batch in self.loader:
            if batch[0].shape[0] == self.batch_size:
                yield batch

    def __len__(self):
        n = getattr(self.loader, "num_examples", None)
        if n is None:
            return sum(1 for b in self.loader if b[0].shape[0] == self.batch_size)
        return n // self.batch_size


def loaders(cfg, seed):
    """(train, test) loaders for one seed, with pcx's normalization."""
    from fabricpc.utils.data import dataloader

    mean, std = PCX_NORMALIZE[cfg.dataset]
    bs = cfg.batch_size
    if cfg.dataset in ("cifar10", "cifar100"):
        base = {
            "cifar10": dataloader.Cifar10Loader,
            "cifar100": dataloader.Cifar100Loader,
        }[cfg.dataset]
        raw = base(
            "train",
            batch_size=bs,
            shuffle=True,
            seed=seed,
            normalize_mean=mean,
            normalize_std=std,
        )
        train = dataloader.AugmentedImageLoader(
            raw, flip_prob=0.5, crop_pad=4, seed=seed
        )
        test = base(
            "test", batch_size=bs, shuffle=False, normalize_mean=mean, normalize_std=std
        )
    else:
        raw = dataloader.TinyImageNetLoader(
            "train",
            batch_size=bs,
            shuffle=True,
            seed=seed,
            normalize_mean=mean,
            normalize_std=std,
        )
        train = dataloader.AugmentedImageLoader(
            raw, flip_prob=0.5, crop_pad=0, crop_size=56, seed=seed
        )
        test = dataloader.TinyImageNetLoader(
            "val",
            batch_size=bs,
            shuffle=False,
            center_crop=56,
            normalize_mean=mean,
            normalize_std=std,
        )
    train.num_examples = raw.num_examples
    return DropLast(train, bs), test
