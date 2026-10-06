"""pcx's per-epoch nudging strength, copied with its quirks.

pcx computes beta once per epoch:

    beta = factor * (beta0 + beta_ir * epoch)
    if abs(beta) >= 1: beta = 1.0

The reset goes to +1.0 even when factor is -1, so a negative-nudging run
with a growing beta turns into plain clamped training near the end. We keep
that, because the published numbers came from that code.

Centered nudging (CN) has no code in pcx. The paper says it alternates
epochs of positive and negative nudging at random, so cn_factors draws the
sign of each epoch with a fair coin from a seeded generator.
"""

import numpy as np


def beta_for_epoch(beta0, beta_ir, factor, epoch):
    """The nudging strength pcx uses for one epoch (epochs count from 0)."""
    beta = factor * (beta0 + beta_ir * epoch)
    if abs(beta) >= 1.0:
        beta = 1.0
    return float(beta)


def cn_factors(seed, n_epochs):
    """Random +1/-1 sign per epoch for centered nudging, the same for a seed."""
    rng = np.random.default_rng(seed)
    return [1.0 if rng.uniform() < 0.5 else -1.0 for _ in range(n_epochs)]
