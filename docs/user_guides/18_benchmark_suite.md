# Benchmark Suite

`fabricpc.bench` trains a model several times with different seeds, measures what each run cost, and compares backprop, state-based PC (sPC) and ePC on the same graph. Every number it reports comes from one command, over several seeds, with a mean and a standard error; it never reports a single seed.

A **row** is one dataset, one model and one training algorithm, named `dataset-model-algorithm` (`cifar10-vgg5-spc`). A **family** is the three rows of one dataset and model (`cifar10-vgg5`), and it is compared as a unit.

## Quick start

```bash
python -m fabricpc.bench list                          # every row and family
python -m fabricpc.bench mnist-mlp-spc --dry-run       # a row's settings, no training
python -m fabricpc.bench mnist-mlp-spc --trials 5      # one row, 5 seeds
python -m fabricpc.bench mnist-mlp --trials 5          # a family: all three rows, then compare
python -m fabricpc.bench compare mnist-mlp             # compare results already on disk
python -m fabricpc.bench validate results              # check a results folder is whole
python -m fabricpc.bench smoke                         # the short run CI uses
```

Results go under `--out` (default `results/`). Useful flags:

| Flag | Effect |
|---|---|
| `--trials N` | seeds to run (default: the row's own, 5) |
| `--epochs E` | shorten a run; a shortened run gets no pass/fail verdict |
| `--resume` | skip trials that already finished with the same epoch count |
| `--zoo DIR` | save each trial's trained weights (the model zoo) |
| `--metric NAME` | metric for `compare` (default: the family's own) |
| `--in-process` | run all trials in this process instead of one process each |

## What a run writes

```
results/
  cifar10-vgg5-spc/
    manifest.json     environment: versions, git commit, GPU, XLA flags, command, row settings
    trial0.json ...   one file per seed
    trials.csv        every trial on one line
    summary.json      mean, std, standard error, pass/fail band
  compare-cifar10-vgg5.json   paired contrasts for the family
```

Every file carries `schema_version`. Each trial records:

| Field | Meaning |
|---|---|
| `metrics` | the task metrics from `evaluate` (accuracy, perplexity, cross-entropy, energy), plus the same metrics from one plain forward pass as `forward_<name>` (see below) |
| `step_time_ms` | median of the timed steps, after compile and warmup, each ending in `block_until_ready` |
| `compile_time_s` | compiling the training step, timed on its own |
| `train_time_s` | wall-clock for the full training run |
| `step_memory` | the compiled training step's argument, output and temporary bytes, from XLA's memory analysis |
| `peak_memory_bytes` | the process-wide peak; on GPU this includes compile scratch space and can be the same for every algorithm, so prefer `step_memory` |
| `compute` | weighted edges, matmuls and FLOPs per weight update (PC `2·E·T + E`, backprop `3·E`, summed edge by edge) |
| `achieved_tflops` | FLOPs per update divided by the measured step time |
| `epc_regime` | ePC rows only: the regime label at the start and end of training (see below) |
| `checkpoint` | where the trained weights were saved, with `--zoo` |

## Seeds, pairing and comparisons

Trial `i` uses seed `i·1000`, the same rule as `PlannedMultiContrastExperiment`, so trial `i` of every row in a family sees the same data order and the same initial key. A trial here trains exactly the model the experiment framework would train for that seed; a test checks the accuracies match to the bit.

`compare` loads the finished trials into the framework's own `PlannedMultiContrastResults` and uses its `contrast_results()`: a paired t-test and Cohen's d on the per-seed differences. Each family tests three contrasts: sPC vs backprop, ePC vs backprop, and ePC vs sPC. Rows are paired on the seeds that all of them ran successfully; seeds only some rows have are listed as `unpaired_seeds`, and fewer than two shared seeds is an error.

## The pass/fail band

A full run (the row's own epoch count and seed count) is checked against the row's expected score when it has one:

```
|mean - expected| <= max(floor, 2 * SE)
```

The floor defaults to half a percentage point of accuracy. A FAIL makes the command exit non-zero. Rows without an expected score, and shortened runs, get no verdict. The rule and the default of 5 seeds are a proposal awaiting the maintainers' sign-off, and every verdict says so.

## Settled scores and forward-pass scores

`evaluate` on a PC graph clamps the input, leaves the output free, and runs full inference. With a cross-entropy output the free output is not at rest where it starts, so inference keeps pushing it and the outputs become over-confident. Accuracy barely moves, but loss and perplexity can look much worse than the weights really are. Every trial therefore also scores the trained weights with one plain forward pass (`evaluate(..., algorithm="backprop")`) and records those numbers as `forward_accuracy`, `forward_perplexity` and so on. On a backprop row they are a copy of the normal ones. To compare the three methods on the same footing, use `compare <family> --metric forward_accuracy` (or `forward_perplexity`).

## Reading ePC results

With `EPCInference`'s default rate and step count a run can be *backprop-like*: the errors barely relax, and one ePC step from zero error is backprop's activation gradient (see [Training with ePC](17_training_with_epc.md)). ePC rows therefore record the regime label from `EPCInference.regime` on one training batch, at initialization and after training. Read `epc_regime.final.band` before reporting an ePC result as predictive coding.

## Running on a GPU in the cloud

`scripts/lightning/make_job.sh` builds the command for a Lightning AI job that runs one benchmark at a pinned commit with `--zoo`, and `scripts/lightning/fetch_results.sh` brings the results back and runs `validate`. Long runs can be split across jobs with `--resume`.

## Adding a row

Rows live in `fabricpc/bench/registry.py` as frozen `BenchmarkRow`s: a model factory `(rng_key) -> (params, structure)`, a loader factory `(seed) -> (train, test)`, an optimizer factory `(total_steps) -> optax transform` (so learning-rate schedules can see the run length), the training config, the batch size, and optionally an expected score and the row's main metric. All three rows of a family must build the same graph and differ only in the solver.

## A note on graph layout

How a network is split into PC nodes matters for sPC. On VGG-5, a separate `MaxPool` node after every conv doubles the number of latent layers; sPC then reached 37% on CIFAR-10 after 50 epochs, against 84% with the pool fused into the conv (`ConvPoolNode`, `create_vgg(fuse_pool=True)`). Backprop and ePC were unaffected. The VGG rows use the fused layout. The v2 transformer has the same kind of extra layer: its MLP is two PC nodes, with the wide hidden layer as a latent of its own. `create_deep_transformer(fuse_mlp=True)` builds each MLP as one `MlpResidualNode` (same weights, two latents per block instead of three); the transformer rows do not use it yet.
