# Nudging experiment: can FabricPC reach pcx's published VGG-5 numbers?

Issue #59 lists three reference numbers: CIFAR-10 VGG-5 89.47%, CIFAR-100
67.19% and Tiny-ImageNet 46.40%. They come from the pcx benchmark paper
(Pinchetti et al., arXiv 2407.01163, Table 1) and use **nudging**, which
FabricPC did not have. This folder adds nudging and trains FabricPC's VGG-5
the way pcx trained theirs, to see how close FabricPC gets.

It lives on the branch `vishal/nudging-experiment` and does not change the
benchmark (`fabricpc/bench`) or FabricPC's trainer. The only library change
is a `head="se"` option on `create_vgg`.

## What nudging is

Normal predictive coding clamps the output to the label `y`. Nudging clamps
it to a point between the network's own guess `u` and the label:

    h = u - beta * (u - y)

- `beta = 1`: the normal clamp.
- `0 < beta < 1`: positive nudging (PN), a gentler pull towards the label.
- `beta < 0`: negative nudging (NN), a push away from the label. The weight
  gradients are multiplied by `1/beta`, which turns the update back the
  right way.
- Centered nudging (CN): epochs of positive and negative nudging at random.

## Configs

| config | data | pcx target | settings from |
|---|---|---|---|
| `c10-se` | CIFAR-10 | 87.98 (plain PC, squared error) | pcx yaml |
| `c10-pn` | CIFAR-10 | 88.42 | pcx yaml |
| `c10-nn` | CIFAR-10 | 88.83 | pcx yaml |
| `c10-cn` | CIFAR-10 | **89.47** | NN yaml + random sign per epoch |
| `c100-nn` | CIFAR-100 | 65.46 | pcx yaml |
| `c100-cn` | CIFAR-100 | **67.19** | NN yaml + random sign per epoch |
| `tiny-nn` | Tiny-ImageNet | **46.40** | pcx yaml |

pcx has no code or settings for CN, only a sentence in the paper, so the CN
rows are our best reconstruction, not a faithful copy.

## How it matches pcx (and where it does not)

Same as pcx: VGG-5 layout (conv and pool in one PC layer), squared-error
output on raw logits, nudged target set once after the feedforward pass,
weights times `1/beta`, the beta schedule including pcx's "+1.0 once
|beta| >= 1" quirk, latent SGD with momentum reset every batch, AdamW with
pcx's warm-up/cosine rate, batch 128, 50 epochs, flips and 4-pixel crops,
pcx's normalization, dropping the last short batch, scoring with a forward
pass.

Different: weights start from FabricPC's Kaiming init (pcx uses PyTorch's
default), crop padding is added after normalizing, and the CN sign rule is
our guess.

## Reporting

pcx runs 7 seeds, keeps each seed's **best** epoch, and reports the mean of
the top 5. Our benchmark reports the **last** epoch. `summarize` prints
both. Only compare like with like.

## Running

Mac check (minutes):

    PYTHONPATH=. python -m experiments.nudging.run c10-nn --seeds 0 --epochs 0.02 --out /tmp/nudge

GPU (Colab T4, about 1 hour per seed for CIFAR-10):

    PYTHONPATH=. python -m experiments.nudging.run c10-nn --seeds 0,1,2 --out RESULTS
    PYTHONPATH=. python -m experiments.nudging.summarize RESULTS

A stopped run can be started again with the same command; finished seeds
are skipped.

Tests: `PYTHONPATH=. python -m pytest experiments/nudging -q`
