# Benchmark Suite

`fabricpc.bench` trains a model several times with different seeds, measures what each run cost, and compares backprop, state-based PC (sPC) and ePC on the same graph. Every number it reports comes from one command, over several seeds, with a mean and a standard error; it never reports a single seed.

A **row** is one dataset, one model and one training algorithm, named `dataset-model-algorithm` (`cifar10-vgg5-spc`). A **family** is the three rows of one dataset and model (`cifar10-vgg5`), and it is compared as a unit.

## Install

The suite needs two of FabricPC's extras: `tfds` (the datasets and the BPE tokenizer) and `experiments` (scipy, for the confidence intervals and p-values). `list` and `--dry-run` work without them.

```bash
pip install -e ".[experiments,tfds]"            # CPU
pip install -e ".[experiments,tfds,cuda12]"     # NVIDIA GPU
```

`.[all]` includes both. For a GPU server, see [Running on a GPU Server](19_running_on_a_gpu_server.md) for the exact pinned install.

## Quick start

```bash
python -m fabricpc.bench doctor --out results          # is this machine ready? (devices, datasets, disk, one step)
python -m fabricpc.bench list                          # every row and family
python -m fabricpc.bench mnist-mlp-spc --dry-run       # a row's settings, no training
python -m fabricpc.bench mnist-mlp-spc                 # one row, 10 seeds (the default)
python -m fabricpc.bench mnist-mlp-spc --preset fast   # one row, 3 seeds
python -m fabricpc.bench mnist-mlp --trials 5          # a family: all three rows, 5 seeds each, then compare
python -m fabricpc.bench compare mnist-mlp             # compare results already on disk
python -m fabricpc.bench tune mnist-mlp --budget 20    # tune each row's settings on held-out training data
python -m fabricpc.bench mnist-mlp --settings tuned    # run the family with the tuned settings
python -m fabricpc.bench validate results              # check a results folder is whole
python -m fabricpc.bench report results --out report.md  # one readable page for a results folder
python -m fabricpc.bench smoke                         # the short run CI uses (into smoke-results/)
python -m fabricpc.bench regress                       # CI's regression checks (into regress-results/)
```

Results go under `--out` (default `results/`). `smoke` and `regress` write to `smoke-results/` and `regress-results/` unless `--out` is typed, so a check never replaces a real run. `--dry-run` on a family prints its three rows and trains nothing. Useful flags:

| Flag | Effect |
|---|---|
| `--trials N` | seeds to run (default: the row's own, 10); overrides `--preset` |
| `--preset fast\|standard` | 3 or 10 seeds |
| `--epochs E` | shorten a run; a shortened run gets no pass/fail verdict |
| `--resume` | skip trials that already finished with the same epoch count asked for and the same `--max-train-steps` / `--max-eval-batches` caps (a shortened trial is never reused for a full run, and the reverse); refused when the folder already holds more trials than asked for, so finished seeds are never deleted |
| `--zoo DIR` | save each trial's trained weights (the model zoo) |
| `--metric NAME` | metric for `compare` (default: the family's own) |
| `--in-process` | run all trials in this process instead of one process each |
| `--settings PATH` | run with tuned settings: a file `tune` wrote, or its output folder (each row reads `<row id>.json`; a row with no file runs as written) |
| `--set NAME=VALUE` | change one setting for this run (`lr_scale`, `eta_infer`, `infer_steps`); repeatable, applied after `--settings`. On a family, `eta_infer` and `infer_steps` change the PC rows only |

## What a run writes

```
results/
  cifar10-vgg5-spc/
    manifest.json     environment: versions, git commit (and git_dirty: uncommitted changes?), GPU, XLA flags, command, row settings
    trial0.json ...   one file per seed
    trials.csv        every trial on one line
    summary.json      mean, std, standard error, 95% CI, pass/fail band
  compare-cifar10-vgg5.json   paired contrasts for the family
```

Every file, compare files included, carries `schema_version` (the history is in `fabricpc/bench/manifest.py`), and `validate` flags any file that does not match the code reading it. `refresh` (below) brings an older folder up to date. Besides the means, `summary.json` lists each successful trial's `checkpoint` with its seed (for the model zoo) and the average `peak_memory_bytes`, which is marked `peak_memory_comparable: false` for the reason in the table below. It also says how PC-like the runs were: `epc_regime` (ePC rows: how many seeds ended in each band, backprop-like, partially relaxed or near PC equilibrium, the mean relaxed fraction `f_weighted`, and how many were unstable, at the start and at the end) and `backprop_alignment` (PC rows: the mean and weakest-layer cosine between PC's weight update and backprop's). ePC rows also get `settled_fraction`, that relaxed fraction at the end of training over seeds: how far the settle went, from 0 (the run trained like backprop) to 1 (it reached the PC equilibrium). Below about 0.1 an ePC row is backprop in all but name. The run prints it, `trials.csv` has it per seed, and a family's compare file lists it per row. Every PC trial also records `solver`, the settling solver it was built with (class, `eta_infer`, `infer_steps` and its other settings); the safety net's last rate is in `rate_control`. Default ePC is backprop-like, so an ePC score should be read with its band; the report shows both in its "PC-likeness at end" column, read from the trial files so older folders show it too. Each trial records:

| Field | Meaning |
|---|---|
| `metrics` | the task metrics from `evaluate` (accuracy, perplexity, cross-entropy, energy), plus the same metrics from one plain forward pass as `forward_<name>` (see below) |
| `step_time_ms` | median of the timed steps, after compile and warmup, each ending in `block_until_ready`. The step timed is the one `train()` runs (inputs donated, settled state dropped inside the step). Runs made before 2026-10-10 timed a step that also returned the settled state: their sPC step times are about twice too slow (MNIST MLP on CPU: 11.8 ms vs 5.9 ms), ePC and backprop about right. `refresh` cannot fix a measured time; rerun those rows |
| `compile_time_s` | compiling the training step, timed on its own |
| `train_time_s` | wall-clock for the full training run. On PC rows it includes the safety net (its share is `safety_net_s`) and the wait for every batch its callback causes, which backprop rows do not have, so compare speed across methods by `step_time_ms` |
| `step_memory` | the compiled training step's argument, output and temporary bytes, from XLA's memory analysis (the same step as `step_time_ms`; runs before 2026-10-10 report about twice the real total) |
| `peak_memory_bytes` | the process-wide peak; on GPU this includes compile scratch space and can be the same for every algorithm, so prefer `step_memory` |
| `compute` | weighted edges, matmuls and FLOPs per weight update (PC `2·E·T + E`, backprop `3·E`, summed edge by edge). A transformer node's own weight matrices (`W_q` … `W_out`) count as edges applied at every position, and attention's `QKᵀ` and `AV` add their FLOPs. Before 2026-10-10 the transformer rows counted 0; `refresh` corrects old files. This hand count is the *nominal* convention of issue #59. `xla_flops_per_update` is what XLA compiled for one update (the settle loop unrolled for counting at T = 1 and 2 and extended to the row's T, since XLA counts a loop's body once), with `xla_flops_per_update_backprop` for the same graph and `xla_pc_to_backprop_ratio`. They differ from the hand count in both directions: on the MNIST MLP, sPC at T = 20 is 5.3 passes over the weights against the nominal 41 (XLA computes the clamped input's prediction once, not every step); VGG-5 sPC 15.0 against 17; the char transformer's sPC 34 against 25 and ePC 17 against 11 (attention and layer norm); backprop 2.1 to 3.2 against 3 |
| `achieved_tflops` | FLOPs per update divided by the measured step time, using XLA's count when the trial has one, else the nominal count |
| `epc_regime` | ePC rows only: the regime label at the start and end of training (see below) |
| `checkpoint` | where the trained weights were saved, with `--zoo` |
| `diagnostics` | at init and after training: `backprop_alignment` (PC rows: per layer, the cosine between PC's weight update and backprop's on one batch, the size ratio, and the weakest layer) and `stiffness` (every row: λ_max of the error-coordinate Hessian; sPC rows also η times the stiffness their settle sees, stable below 2). See below |

## Seeds, pairing and comparisons

Trial `i` uses seed `i·1000`, the same rule as `PlannedMultiContrastExperiment`, so trial `i` of every row in a family sees the same data order and the same initial key. A trial here trains exactly the model the experiment framework would train for that seed; a test checks the accuracies match to the bit.

`compare` loads the finished trials into the framework's own `PlannedMultiContrastResults` and uses its `contrast_results()`: a paired t-test and Cohen's d on the per-seed differences. Each family tests three contrasts: sPC vs backprop, ePC vs backprop, and ePC vs sPC. Their p-values are Holm-adjusted together (`p_holm`), so the chance of a false "significant" in the family stays at 5%; a contrast with no p-value still counts as one of the three. `compare <row-a> <row-b>` on two rows of one family adjusts its p over that family's three contrasts (`holm_family`); for two rows that are not one family's, it reports the p as uncorrected (`p_holm` is NaN). Rows are paired on the seeds that all of them ran successfully, and must have been trained for the same number of epochs (a 1-epoch row is never tested against a 20-epoch one; a row whose own trials differ in length gets no summary); seeds only some rows have are listed as `unpaired_seeds`, and fewer than two shared seeds is an error.

Every mean in `summary.json` (metrics and timing) has `ci95_low` and `ci95_high` beside `mean`, `std`, `se` and `n`: a 95% confidence interval using the Student t value for n-1 degrees of freedom (`fabricpc.bench.ci`). With few seeds t is much bigger than 1.96: 2.78 for five seeds, 4.30 for three, 12.7 for two, so a two-seed interval is very wide on purpose. Each contrast in a compare file also gets `ci95_low` and `ci95_high` for its mean difference, worked out from the same per-seed differences as the paired t-test; the framework's contrast has the SE but no interval. Zero lies outside that interval exactly when p < 0.05. The intervals are not corrected for testing three contrasts at once, but the verdict is: each contrast also gets `p_holm`, Holm's step-down adjustment over the family's contrasts (sort the p-values, multiply by 3, 2, 1, keep them non-decreasing), and `significant_holm_05` is what the CLI and the report call significant. `cohens_d` is the paired effect size d_z (mean difference over the SD of the differences, `effect_size: "d_z (paired)"`), which runs large with few seeds; `hedges_gz` multiplies it by Hedges' factor 1 - 3/(4(n-1) - 1). Rows that score exactly the same on every seed are written as a difference of 0 with p = 1 instead of NaN. The CLI prints the interval as `95% CI [low, high]`.

**"No difference" is not "the same".** A small or noisy experiment fails to find a difference too, so a non-significant contrast is no evidence that PC matches backprop. Each contrast therefore also gets an equivalence test (two one-sided t-tests, TOST, Schuirmann 1987) on the same per-seed differences: is the mean difference above -margin and below +margin? It passes at 5% exactly when the 90% interval of the difference lies inside the margin. `p_equivalence` is its p-value, `p_equivalence_holm` the Holm adjustment over the family's contrasts, and `equivalent_holm_05` the verdict the CLI prints ("equivalent within ±0.005: yes" or "not shown"). The margin is in the metric's units and is written to the compare file as `equivalence_margin`: 0.005 (half a percentage point, the band's floor) on accuracy metrics by default, none on other metrics, where there is no natural unit; set it with `compare <family> --margin M` (or on a family run), and `--margin 0` turns the test off. Only say two methods match when `equivalent_holm_05` is true; the margin itself is a choice to state next to the result.

## The pass/fail band

A full run (the row's own epoch count, and at least 3 good seeds) is checked against the row's expected score when it has one:

```
|mean - expected| <= max(floor, 2 * SE)
```

The floor defaults to half a percentage point of accuracy. A FAIL makes the command exit non-zero. Rows without an expected score, shortened runs, runs with fewer than 3 good seeds and runs where any seed failed get no verdict. Three is the mentor's minimum, so a `--preset fast` run is judged too; with few seeds the SE is bigger, so the band widens on its own. Rows run 10 seeds by default, the mentor's guidance. The band rule is a proposal awaiting the maintainers' sign-off, and every verdict says so.

## The regression checks in CI

`smoke` shows one MNIST model still learns. `regress` is a stronger check that runs on every push: tiny fixed-seed runs of three kinds of model, each with sPC, ePC and backprop, where the mean of two seeds has to stay on the right side of a fixed line.

| Rows | Run | Score | Line |
|---|---|---|---|
| `mnist-mlp-*` | 0.5 epoch | test accuracy | at least 0.85 |
| `mnist-autoencoder-*` | 0.5 epoch | reconstruction MSE | at most 0.055 |
| `patterns64-hopfield-*` | 30 epochs (the full row) | `bit_accuracy_lift_p20` | at least 0.10 |

The lines were set from real runs with a wide margin, so a different CPU or jax version does not make CI flaky; `MEASURED` in `fabricpc/bench/regress.py` has the numbers. For scale: handing back the average image scores an MSE of 0.0675, and a Hopfield row that learned nothing has a lift of 0. After the runs the folder is validated, and `report.md` and `regress.json` (every verdict) are written into it. Any FAIL, failed trial or validate problem makes the command exit non-zero. `regress mnist-autoencoder-epc` runs just that one check. If a change is meant to move a score across its line, change the line in `regress.py` in the same pull request and say why.

## Settled scores and forward-pass scores

`evaluate` on a PC graph clamps the input, leaves the output free, and runs full inference. With a cross-entropy output the free output is not at rest where it starts, so inference keeps pushing it and the outputs become over-confident. Accuracy barely moves, but loss and perplexity can look much worse than the weights really are. Every trial therefore also scores the trained weights with one plain forward pass (`evaluate(..., algorithm="backprop")`) and records those numbers as `forward_accuracy`, `forward_perplexity` and so on. On a backprop row they are a copy of the normal ones. The transformer rows (`tinyshakespeare-transformer-*`, `tinyshakespeare-bpe-transformer-*`) are judged on `forward_perplexity`: it is their summary headline, their family's comparison metric and what `trend` watches, and the settled number stays in each trial as `perplexity`. (In a check on Fashion-MNIST outside the suite, a trained sPC model's settled cross-entropy was 1.88 against 0.42 from a forward pass.) For the other rows, compare on the same footing with `compare <family> --metric forward_accuracy`. Transformer results from before 2026-09-26 have no forward scores; rerun them to compare.

## Reading ePC results

With `EPCInference`'s default rate and step count a run can be *backprop-like*: the errors barely relax, and one ePC step from zero error is backprop's activation gradient (see [Training with ePC](17_training_with_epc.md)). ePC rows therefore record the regime label from `EPCInference.regime` on one training batch, at initialization and after training. Read `epc_regime.final.band` before reporting an ePC result as predictive coding.

Every sPC and ePC row trains with a safety net: `rate_control=RateControl(...)` shrinks the inference rate only when a run nears the settle's stability limit (η times the stiffness reaching 2). The trigger comes from our own runs at the end of training: healthy ePC rows sat at 0.07 or below and the autoencoder's broke at 3 to 15 (3 of 5 seeds, rebuild error up to 10 times its best, half the hidden units dead), so ePC acts at 1.0; healthy sPC rows reached 1.7 and degraded above 2, so sPC acts at 1.8. Rows that were healthy train exactly as before (the test that matches our trials to the experiment framework bit for bit still passes). On the two autoencoder seeds that broke, the safety net kept both on track: rebuild error 0.0097 and 0.0112 at the end (backprop 0.0084 to 0.0098), against 0.15 and 0.14 without it. It probes every 50 updates, about a tenth more training time.

A PC row (sPC or ePC) can also be given its own setting: a rate that follows the settle's stiffness: set `rate_control=RateControl(target=..., every=...)` on the row (`fabricpc.bench.registry`). The trial then records `rate_control` (the `InferenceRateController` summary and every probe), trials.csv adds `eta_final` and `rate_crossings`, and the final score and regime are read at the rate the run ended on.

## Reading the diagnostics

Two numbers say what a PC score means. `backprop_alignment.mean_cos` near 1 says the run learned in backprop's direction: default ePC on VGG-5 reads 1.00, so its accuracy is backprop's reached another way. Lower values say the run learned something else, and `weakest` names the layer furthest off; on VGG-5, sPC reads 0.90 with the output layer at 0.85. Settled PC can lower its energy by making the network stiffer rather than more accurate, and `stiffness.lambda_max` is where that shows, comparable across the three methods because it depends on the weights alone: on VGG-5 about 6.5 at init, 36 after backprop, 53 after default ePC, 143 after sPC. For sPC rows, `eta_times_settle_stiffness` above 2 means the settle diverges; on the character transformer it reached about 650 after one epoch. trials.csv shows `stiffness_init`, `stiffness_final`, `bp_cos_min`, `bp_cos_mean`, and `bp_cos_weakest`. `run_trial(..., diagnostics=False)` skips both.

## Quicker runs with --fast

`--fast` skips the per-epoch learning curve and the start-and-end diagnostics (backprop-likeness and stiffness). Training, seeds and the final scores are exactly the same; only those extra measurements are left out. On a Colab T4 they made the small rows about three times slower (about 3 minutes per MNIST trial instead of about 1). `--no-diagnostics` skips only the diagnostics, and `--curve-batches 0` only the curve. The command line saved in each manifest shows which run used them.

## The autoencoder rows

`mnist-autoencoder` (784-128-32-128-784, ReLU inside, sigmoid out, pixels in [0, 1]) is not a classifier, so it declares its own scores through the row's `eval_metrics` instead of `evaluate`'s defaults. Its main metric is `reconstruction_mse`, the mean squared pixel error (lower is better). `code_sparsity` is the fraction of the 32 bottleneck units that are silent, and `hidden_sparsity` the same over all three hidden layers. A unit counts as silent when its activation is below 1e-6. Sparsity is read from each unit's activation (z_mu), not its latent: in a settled PC state the latent carries a small leftover error, which would make every PC unit look slightly active. `compare mnist-autoencoder` pairs the three methods on `reconstruction_mse`; pass `--metric code_sparsity` to compare sparsity instead.

## The Hopfield rows (trained denoising, not yet attractor recall)

`patterns64-hopfield` is associative memory: store a few patterns, then get a whole one back from a noisy copy. It follows the binary experiment in `examples/storkey_hopfield_recall.py`: seven random ±1 patterns of 64 bits (new ones for every seed, the same for all three rows), the graph `probe -> StorkeyHopfield -> output`, trained for 30 epochs on 100 noisy copies of each pattern per epoch with 15% of the bits flipped. It runs on a laptop CPU: the whole family at 3 seeds took under 3 minutes on a MacBook (about 20 seconds per trial, most of it compiling the per-epoch learning curve; `--curve-batches 0` skips the curve).

The test set flips 0%, 10%, 20%, 30% and 50% of the bits, 20 probes per pattern at each rate. The recalled pattern is the sign of the Hopfield node's latent (0 counts as +1), and each rate gets its own scores, named with the rate (`_p20` is 20%):

| Score | Meaning |
|---|---|
| `bit_accuracy_pNN` | fraction of bits that match the stored pattern (the main metric is `bit_accuracy_p20`) |
| `exact_recall_pNN` | fraction of probes brought back with every bit right |
| `nearest_correct_pNN` | fraction closer to the right pattern than to any other stored one (the example's "exact"; a tie does not count) |
| `storkey_rule_bit_accuracy_pNN`, `storkey_rule_exact_recall_pNN` | the same probes recalled by a classic Hopfield network written with the Storkey (1997) rule |
| `probe_bit_accuracy_pNN`, `probe_exact_recall_pNN` | the do-nothing baseline: the noisy probe handed back unchanged |
| `bit_accuracy_lift_pNN` | `bit_accuracy_pNN` minus the do-nothing baseline; 0 means the row learned nothing |

At 0% any network that keeps the probe's signs is perfect, so p00 only checks that nothing is destroyed, not that the patterns are stored. 50% is chance (bit accuracy near 0.5). Always read a score next to its `probe_*` baseline: a row that copies its input scores about 1 minus the flip rate in bit accuracy and 0 in exact recall. Things to keep in mind when reading the numbers:

- The `StorkeyHopfield` node does not use the Storkey rule. Its weights are learned by gradient descent from noisy copies, so it is a trained denoiser. The `storkey_rule_*` scores are the real rule on the same patterns and probes, written once from the clean patterns. They do not depend on training, so all three rows show the same value: read them as a fixed ruler, not a fourth method.
- The backprop row reads the pattern from one pass through the node, `tanh` of a blend of the probe and `probe @ W`, with no settling. sPC and ePC read it after inference, and every PC row also records `forward_*` scores from one pass. With these settings (20 steps, output free, Hopfield strength 1.0) settling adds essentially nothing: on our Mac check the settled and `forward_*` scores matched to within 0.002 in bit accuracy, and settled was sometimes a little worse. So all three rows are one-layer denoisers that differ only in how W was trained, and all sit well below the Storkey-rule ruler. They do not yet test attractor recall; do not present them as that.
- The rows recall with the training graph's 20 inference steps. The example uses 100; on our check that moved bit accuracy by 0.002 at most.
- The ePC row does not use ePC's defaults. With η = 0.001 and 5 steps W never learned to denoise: every score equalled the do-nothing baseline (bit accuracy 0.796 at 20%, exact recall 0). It now uses η = 0.1 and 20 steps, where it learns like sPC (0.950 vs 0.949 on seed 0). ePC Hopfield results from before this change, such as `hopfield-mac-check`, are the do-nothing score and need a rerun.

`compare patterns64-hopfield` pairs the rows on `bit_accuracy_p20`; pass `--metric exact_recall_p20` (or any score above) to compare on another. Code: `fabricpc/bench/rows_hopfield.py`.

## The report page

`python -m fabricpc.bench report results --out report.md` (or `--format html`) writes one page for a results folder, however nested. Per family it shows:

- a table of rows with the metric (mean ± SE and 95% CI), step time, slowdown against backprop, the band verdict, and how PC-like the runs ended up (ePC bands over the seeds, cosine to backprop's update);
- a "Cost and learning" table: PC's FLOPs per update over backprop's, from XLA's count (older runs show the nominal count, marked), achieved TFLOP/s, step memory, compile time, and the learning curve averaged over seeds as a one-line text chart (`▁▃▅▇█ 0.49 → 0.87`, at most 20 characters);
- the paired comparisons with the p-value, the Holm-adjusted p, Hedges' g_z and a plain yes/no on the adjusted p;
- where the numbers came from, any `NOTES.md`, and what was skipped. sPC rows whose step time was measured before the 2026-10-10 timing fix are listed there, since their step time is about twice too slow.

The report never stops on a messy folder; it reads what it can and says what it skipped.

## Estimating time and cost before a run

`probe` builds a row's model and loaders, times a few training steps and evaluation batches (compile timed on its own, median of the steps), and multiplies out the full run instead of doing it. It counts two compiles per seed, because a real trial compiles the step once for its timing pass and again inside `train()`:

```bash
python -m fabricpc.bench probe mnist-mlp                        # every row of a family
python -m fabricpc.bench probe cifar10-vgg5-spc --trials 5 --epochs 50 --price-per-hour 1.06
```

It prints, per row, the step time, steps per epoch, time per seed and for all seeds, and the family total with its cost (hours times `--price-per-hour`, default 1.06 USD, what a Lightning T4 cost us). `--trials` and `--epochs` default to the row's own; `--warmup` and `--timed` set how many steps are timed (default 2 and 10). With `--out DIR` it also writes `probe-<row-or-family>.json` there. The numbers only hold for the device they were measured on, which the report names, so probe on the machine you will pay for. It also times what a real trial adds on top of training: the diagnostics at the start and end (and the ePC regime), left out with `--fast` or `--no-diagnostics` as in a real run, and the safety net's stiffness probes, one every `every` updates. It leaves out process start-up and the recompile when the safety net changes the rate, so treat it as a floor. Estimates made before 2026-10-10 timed a heavier step than training runs (sPC about twice too slow) and left the extras out: probe again on the GPU before trusting them.

## Tuning a row's settings

The rows' settings come from demos and from pcx and were not tuned for this suite, and a method trained with another method's settings is at a disadvantage. `tune` searches them with Optuna:

```bash
python -m fabricpc.bench tune mnist-mlp --budget 20 --out tuned        # each row of the family, 20 tries each
python -m fabricpc.bench tune cifar10-vgg5-spc --budget 30 --epochs 5  # one row, shorter tries
python -m fabricpc.bench mnist-mlp --settings tuned                    # a normal 10-seed run with what was found
```

- **What is searched.** `lr_scale` on every row: it multiplies the optimizer's update, which for the optax optimizers the rows use is the same as multiplying the learning rate, schedule and all (weight decay included, as in AdamW). On PC rows also `eta_infer`, the settle's step size, and `infer_steps`, the settle steps per batch. Default ranges sit around the row's own values: `lr_scale` 0.1 to 10, `eta_infer` a tenth to ten times the row's, `infer_steps` half to twice the row's. `--space eta_infer=1e-4:1e-1` replaces a range (log scale; add `:lin` for a linear one). The row's own settings are always the first try, so tuning can keep them but never do worse on the tuning data.
- **What it is scored on.** Each try is a normal training run of the row (the same code a benchmark run uses, safety net included) on 90% of the training set, scored with the row's own metric on the other 10% (`train[90%:]`; the text rows use Tiny Shakespeare's own validation split). The test set is never read, so the benchmark score of a tuned row is not inflated by the tuning. Every try trains from the same seed (`--seed`, default 0), so tries differ only in their settings. A try that fails, or scores NaN, is dropped and recorded.
- **Fair between methods.** A family is tuned row by row with the same budget, seed and held-out data. Tuned rows can end up with different learning rates; each method then runs with its own best settings for the same search effort, where the untuned rows share one optimizer.
- **Finding an ePC that really settles.** Default ePC trains like backprop (settled fraction near 0). `tune mnist-mlp-epc --min-settled 0.5` keeps only settings whose settled fraction ends at 0.5 or above; on a family it tunes the ePC row only. Run the result into its own `--out`, since it has the same row id as the default ePC row.
- **Cost.** A try costs one training run of the row, without timing, FLOP counting or diagnostics. `--epochs` and `--max-train-steps` shorten the tries; short tries favour settings that learn fast, so check the winner with a full run.
- **Output.** `tuned/<row id>.json` holds the best settings, the score on the held-out data, the row's own score there, the ranges, budget, seed and commit. `tuned/<row id>/study.json` lists every try. A run with `--settings` records the settings in the manifest and in every trial (`settings`), and `--resume` never reuses a trial that ran with other settings.

Rows that cannot be tuned yet, because their loaders have no held-out split: `tinyimagenet-vgg5`, `fashionmnist-continual`, `patterns64-hopfield`. A row becomes tunable by giving it a `tune_loader_factory` (see Adding a row).

## Running on a GPU in the cloud

`scripts/lightning/make_job.sh` builds the command for a Lightning AI job that runs one benchmark at a pinned commit with `--zoo`, and `scripts/lightning/fetch_results.sh` brings the results back and runs `validate`. Long runs can be split across jobs with `--resume`.

On a shared GPU machine (a JupyterHub), `python -m fabricpc.bench queue <plan.json> --out DIR` works through a list of rows and families in order, with `--resume`, a `queue-status.json` after every job, and `validate` plus `report` at the end; `fabricpc/bench/plans/gpu_campaign.json` is the waiting GPU work, and `gpu_deep_vgg.json` holds VGG-7/9 until their layouts are signed off. Before paying for the hours:

```bash
python -m fabricpc.bench doctor --out runs --expect-gpu                       # GPU visible (fails if not)? both CUDA plugins? datasets, disk, one step
python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json --validate-plan       # unknown ids, and jobs that get no pass/fail verdict
python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json --probe --out campaign  # time every row on this machine, train nothing
python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json --rehearse --out rehearsal  # every job for 3 training steps
```

`--expect-gpu` turns "JAX sees no GPU" from a warning into a failure: a broken CUDA install makes JAX fall back to the CPU with only a warning, and the campaign would then run for days on the CPU. Each trial's own error output, where that warning appears, is kept in `<out>/<row>/trial<i>.stderr.log`.

`--validate-plan` warns about jobs that run fewer than 3 seeds or a shortened epoch count, since those get no verdict. `--probe` on its own times every row and prints the estimate; it never starts the campaign. `--rehearse` runs every job with 2 seeds, 3 real training steps whatever the dataset's size (`--max-train-steps 3`), and 2 test batches (`--max-eval-batches 2`), with the learning curve and the diagnostics on, so a missing dataset, a shape error, a diagnostics or curve bug or a zoo problem shows up before the paid time. Its trials record `train_steps_capped` and `eval_batches_capped` and never get a verdict. On a laptop CPU, rehearse only the small rows: with the diagnostics on, the VGG-size rows are slow there (a CPU rehearsal of just `mnist-mlp` and `cifar10-vgg5` ran for more than 10 minutes before it was stopped). On the GPU the whole plan is the point. See [Running on a GPU Server](19_running_on_a_gpu_server.md).

## Trend tracking

Shared runners wobble by 5 to 15% in timing, so "slower than last night" alerts all the time. `trend` keeps every run and compares the newest with the ones before it (issue #54):

```bash
python -m fabricpc.bench trend add regress-results --history bench-history   # one line per row per run
python -m fabricpc.bench trend check --history bench-history                 # exit 1 on a regression
```

`add` appends each row's per-seed step times and scores, its failed seeds, the git sha, date, device, XLA flags, epochs, eval and train caps, and the machine's speed test from the manifest, to `<history>/<row>.jsonl` (the same run is never added twice). `check` judges each row's newest night:

- **One number per night**, the median over its seeds. The seeds of one night share that night's machine, so they are not independent samples.
- **Divided by the speed test.** Every manifest records `speed_test_ms`, a fixed 512 x 512 matrix product timed when the run started. A night when the whole runner was slow then does not look like a slower FabricPC. Nights without a speed test are left out of a speed-adjusted comparison.
- **Like with like.** Nights are compared only within one device, XLA flag set, metric, epoch count and eval/train cap; a change of any of them starts a new baseline instead of alerting. A row's solver settings are fixed by its id; change them under a new id or start a new history.
- **A frozen baseline.** The last 10 nights that raised no alarm (at least 5). Nights in an alarm never join it, so a real slowdown keeps alarming instead of becoming the new normal.
- **Two nights in a row**, each worse than the baseline's median by a practical amount (step time 15%, accuracy-like scores half a point, MSE, perplexity, cross-entropy and forgetting 5%) and by more than twice the baseline's own wobble (1.4826 x its median absolute deviation).
- **At once** on a failed seed or a NaN or infinite number.

How well this works depends on how much of a runner's noise the speed test sees. Simulated with nine rows, two seeds a night and 1,000 weeks (`tests/test_bench_trend.py` checks the first line):

| Speed test sees | Runner noise | Weeks with a false alarm | 20% / 25% / 30% slowdown caught within 3 nights |
|---|---|---|---|
| all of it | 5 to 15% | 0% | 85% / 99% / 100% |
| half of it | 10% | 3% | 63% / 82% / 95% |
| half of it | 15% | 19% | 50% / 65% / 80% |
| none (no speed test) | 10% | 33% | 38% / 53% / 64% |

Real nightly history will show which line applies. Issue #54 suggests a Welch test. In the same simulation without a speed test, a Welch test of two night medians against ten raised a false alarm in 53% of weeks at 10% noise and 90% at 15%, against 32% and 53% for the rule above, so the rule is used (`fabricpc.experiments.welch_ttest` is still there). The first version of `trend` pooled every seed as a separate sample and raised a false alarm in every simulated week at 10% noise.

## Fixing old results without retraining

When a bug fix changes a number the suite works out rather than measures, `refresh` writes a corrected copy of an old results folder:

```bash
python -m fabricpc.bench refresh results/old-run --out results/old-run-refreshed
```

It recounts `compute` and `achieved_tflops` with today's FLOP counter (keeping the run's own settling steps and its measured step time), makes cloud checkpoint paths relative to the new folder when the zoo is on disk (`--zoo DIR` if it is not next to the old results), and marks `peak_memory_bytes` with `peak_memory_comparable: false` when the run has no `step_memory`. Metrics and times are copied unchanged. Each trial and manifest gets a `refresh` list saying what changed and with which commit; the manifest's own `git_sha` stays the commit that trained. The old folder is never edited, and the new one is checked with `validate`. `--out` is required and must not already hold results for those rows. Every row is checked before anything is written: a row not in the registry, or whose parameter count or weighted-edge count no longer matches the run, stops the refresh. Every compare file in the old folder is rebuilt (family, pairwise, or on another metric); one that cannot be rebuilt is copied unchanged and listed in the new `NOTES.md`, which also carries the old notes under a line saying what was corrected. A band verdict that changes against today's reference is recorded in the manifest's refresh record. Trials and manifests written by an older version of the suite are stamped with today's `schema_version`, and that change is recorded too; a compare file that is copied unchanged gets the stamp plus `copied_unchanged: true`, so `validate` passes and the file still says its numbers are old.

## Continual learning

`fashionmnist-continual-{spc,epc,backprop}` asks whether learning a second task makes a network forget the first, the setting where Song et al. (2024, Nature Neuroscience) found predictive coding forgets less than backprop. The rows are inspired by that paper, not a replication: the network (784 -> 32 -> 32 -> 32 -> 5, sigmoid) and the shared 5-unit output follow it, but the optimizer (AdamW with weight decay, not tuned SGD), the output loss (cross-entropy, not squared error), the settle (sPC 20 steps at 0.05, not about 64 to 128 at 0.1), the class split (fixed even/odd, not random), the amount of data and the scoring schedule differ; `fabricpc/bench/continual.py` lists them.

Fashion-MNIST is split in two 5-way tasks on one shared output layer: A is the even classes, B the odd ones (label // 2). Each seed picks 300 training images per class, and training runs 10 blocks, A B A B ..., each one pass over its task with batch 32; the optimizer keeps its state across blocks. `BlockLoader` hands out the next block on every pass, so `train()` over 10 epochs is the 10 blocks, and the learning curve scores the whole test set after each block (test batches of 1,000). Each trial reports `accuracy` (both tasks at the end, the row's metric; the last block trains B, so it mostly measures B), `mean_accuracy` (both tasks averaged over every block: Song et al.'s headline measure), `forgetting` (from the second block on, what the idle task lost per block), `gain` (what the trained task gained per block), `forgetting_per_gain`, and `final_task_a`/`_b`. These come from the learning curve, so `--fast` leaves them out. PC rows are scored on the curve with their settled prediction, backprop with a forward pass. Compare on any of them with `compare fashionmnist-continual --metric mean_accuracy` (or `forgetting`, ...). Code: `fabricpc/bench/continual.py`.

First Mac CPU run, 10 seeds (`python -m fabricpc.bench fashionmnist-continual`, about 7 minutes; the measures other than `accuracy` and `forgetting` were recomputed from the saved curves with `curve_summary`). Mean ± SE over seeds; paired differences Holm-corrected over the family's three contrasts:

| measure | sPC | ePC | backprop | sPC - backprop |
|---|---|---|---|---|
| mean_accuracy | 0.590 ± 0.005 | 0.575 ± 0.008 | 0.574 ± 0.008 | +0.015, Holm p = 0.013, 9 of 10 seeds |
| gain | 0.245 | 0.224 | 0.223 | +0.022, Holm p = 0.013 |
| forgetting | 0.189 | 0.174 | 0.174 | +0.015, Holm p = 0.026 |
| forgetting_per_gain | 0.767 | 0.775 | 0.775 | -0.009, Holm p = 0.51 |

So sPC learns each new block more and also loses more of the other task; relative to what it learns it forgets about the same as backprop, and averaged over the run it is ahead, in the direction Song et al. report. ePC was backprop-like on all 10 seeds and matched backprop to within 0.1 point on every measure. With one optimizer, one loss and a short settle, these are preliminary; a version closer to Song et al.'s settings would test how far they hold.

## Graphs with loops

`python -m fabricpc.bench cycles --out DIR` measures issue #64: on two looped MNIST graphs, how much a settle costs (matmuls per sample, time) to land within 1% of the true energy (sPC settled for 2000 steps), when unrolling the loop and relaxing with ePC, when relaxing one visit with ePC and refining with sPC, and with plain sPC. It writes `cycles.json` and `cycles.md`. `--epochs` sets the training of the weights first (default 1) and `--max-eval-batches` the test batches settled (default 5). First runs are in `docs/reports/cycles_refine_vs_unroll.md`. They do not pick a default yet: the cheapest arm changed with the trained weights (unrolling alone with 1-epoch weights, unroll 2 plus a short sPC refinement with 0.25-epoch weights), and one visit never lands.

## Adding your own model

A new model joins the suite through `fabricpc.bench.recipe.pc_family`: give it a function that builds the graph for a given settling solver, the solver each PC method uses, the data, the optimizer, and the epochs. It returns one row per method (`<dataset>-<model>-spc`, `-epc`, `-backprop`) that share the graph, the starting weights for a seed, the data order and the optimizer, so the comparison is fair by construction. `fabricpc.bench.registry.register_family(rows)` adds them to the suite. `examples/bench_add_your_model.py` is a complete example, a 784-512-10 MLP on MNIST:

```bash
FABRICPC_BENCH_ROWS=examples/bench_add_your_model.py python -m fabricpc.bench list
FABRICPC_BENCH_ROWS=examples/bench_add_your_model.py python -m fabricpc.bench mnist-wide-mlp --preset fast
FABRICPC_BENCH_ROWS=examples/bench_add_your_model.py python -m fabricpc.bench tune mnist-wide-mlp --budget 10
```

`FABRICPC_BENCH_ROWS` names Python files or importable modules (comma-separated) that are imported before the command runs; `--rows-module FILE` does the same for one command. Child trial processes inherit the variable, so they find the same rows. Your rows then get everything the built-in ones get: seeds and pairing, timing and FLOPs, the settled fraction, tuning (when you give `tune_loader_factory`; `recipe.flat_image_loaders` and `recipe.cifar10_loaders` take `tuning=True`), compare with Holm and equivalence, the report and the model zoo. The methods themselves are listed once, in `fabricpc/bench/methods.py`; a family may use any two or more of them (`methods=("spc", "backprop")`), and its contrasts follow: each settling method against backprop, then the settling methods against each other.

## Adding a row

Rows live in `fabricpc/bench/registry.py` as frozen `BenchmarkRow`s: a model factory `(rng_key) -> (params, structure)`, a loader factory `(seed) -> (train, test)`, an optimizer factory `(total_steps) -> optax transform` (so learning-rate schedules can see the run length), the training config, the batch size, and optionally an expected score and the row's main metric. Give it a `tune_loader_factory` too, the same as its loader factory but training on `fabricpc.bench.splits.TUNE_TRAIN_SPLIT` and scoring on `TUNE_HOLDOUT_SPLIT`, so `tune` can tune it without touching the test set. All three rows of a family must build the same graph and differ only in the solver.

A family can also live in its own module and be added to `ROWS` with one line; `fabricpc/bench/cifar100.py` does this for `cifar100-vgg5`, the CIFAR-10 VGG-5 recipe with 100 classes and pcx's CIFAR-100 settings (12 settling steps, hard tanh; pcx's PC-CE weight settings, rate 2.2e-4 and decay 7.6e-3, for all three rows). Every family trains its three rows with one optimizer, as `examples/PC_backprop_compare.py` does, so a difference between them comes from the learning rule alone; a test checks this for every family. pcx tuned its backprop separately, so our backprop rows on CIFAR-100 and Tiny-ImageNet may score a little below pcx's. Results made before 2026-10-11 for `cifar100-vgg5-backprop` used pcx's separate backprop settings (rate 2.1e-4, decay 2.2e-5). pcx's best VGG-5 CIFAR-100 score is 67.19% (centered nudging, which we do not run); its PC-CE and backprop cross-entropy runs score 60.00% and 60.82%. The rows have no expected score until their first full run.

`fabricpc/bench/tinyimagenet.py` adds `tinyimagenet-vgg5` the same way: VGG-5 with 200 classes on 56x56 crops of Tiny-ImageNet's 64x64 images (random flip and random 56x56 window for training, the centre 56x56 of the 10,000 `val` images for scoring, since the official test split has no labels), with pcx's PC-CE settings (7 settling steps, hard tanh; weights 4.5e-5 / decay 1.4e-3 for the PC rows, 8.7e-5 / 2.1e-5 for backprop). `TinyImageNetLoader` downloads the official 248 MB zip once (md5 checked) to `~/tensorflow_datasets/tiny_imagenet_200` (or `$FABRICPC_TINYIMAGENET_DIR`) and decodes JPEGs from it per batch, so nothing else is written to disk. pcx's best VGG-5 Tiny-ImageNet score, 46.40% top-1, is negative nudging, which we do not run; its PC-CE and backprop cross-entropy runs score 41.29% and 43.72%. These rows also have no expected score yet.

`fabricpc/bench/bpe.py` adds `tinyshakespeare-bpe-transformer` the same way: the char transformer recipe (same builder, cosine decay to a tenth, Adam epsilon 1e-12, safety net) with the BPE settings of `examples/transformer_v2_demo.py` (`BPE_DEFAULTS`: 4 blocks of width 128, 64 tokens, batch 32, 23 settling steps, 11711 BPE ids). It is Tier 2 and heavy: 7,527 steps per epoch, and on a MacBook CPU a step took about 0.7 s for backprop, 3.3 s for ePC and 15 s for sPC. The tokenizer is trained once (a few seconds) and cached with the encoded splits, about 2 MB, in `~/.cache/fabricpc/bpe_tokenized` (set `FABRICPC_BPE_DIR` to move it). The demo's comment gives val perplexity about 1133 and test about 721, but from tuning on a 50k-sequence subset, about a fifth of the training text these rows use, so those are not this row's setting. Guessing each token by its frequency in the training text (unigram, add-one smoothing) scores about 790 on test and about 674 on val. The rows are scored on test, so judge them against the test numbers only: unigram 790, demo 721. BPE perplexities cannot be compared with the char rows' (a token is about 4 characters).

## The deep FC-ResNet rows (muPC past 100 layers)

`mnist-fcresnet{8,16,32,64,128}` answer sponsor issue #59: does muPC keep a predictive coding net trainable at 100+ layers? Each is the network from `examples/mupc_demo.py`, input(784) -> stem(64) -> *depth* `LinearResidual` blocks (one PC node each, `tanh(W x + b) + x`) -> softmax readout, with muPC on (weight path scaled by gain / sqrt(64 · depth), the identity skip unscaled, the readout left out). The settings are the demo's: batch 256, 3 epochs, AdamW 0.002 with weight decay 0.01, sPC settling at rate 0.1 for max(20, 3 · (depth + 2)) steps; Adam's epsilon is 1e-12 like the other deep rows. Each row carries its `depth`, and `report` adds an "Accuracy against depth" table with one line per depth and one column per method. Depths 8 and 16 are tier 1; 32 and up are tier 2, because sPC's settle grows with depth twice over (more nodes, more steps): on a MacBook CPU an sPC step takes about 0.07 s at depth 8 and 8 s at depth 128. This is the demo's recipe, not the muPC paper's (pre-activation, no biases, muPC on a squared-error readout, Adam 0.1, batch 64, one epoch). First Mac CPU checks: at depth 8 (2 seeds, full run) backprop 92.9%, ePC 92.9%, sPC 92.1%, matching the demo's 92.0%. At depth 128 (1 seed) backprop reached 89.6% and ePC 87.8% after 3 epochs, and sPC went from 9% to 76% in its first 200 updates (a full sPC run there is about 2 hours on CPU). Backprop also loses about 3 points from depth 8 to 128, so part of the demo's drop with depth comes from the recipe, not from PC. Code: `fabricpc/bench/deep.py`.

## A note on graph layout

How a network is split into PC nodes matters for sPC. On VGG-5, a separate `MaxPool` node after every conv doubles the number of latent layers; sPC then reached 37% on CIFAR-10 after 50 epochs, against 84% with the pool fused into the conv (`ConvPoolNode`, `create_vgg(fuse_pool=True)`). Backprop and ePC were unaffected. The VGG rows use the fused layout. The v2 transformer has the same kind of extra layer: its MLP is two PC nodes, with the wide hidden layer as a latent of its own. `create_deep_transformer(fuse_mlp=True)` builds each MLP as one `MlpResidualNode` (same weights, two latents per block instead of three); the transformer rows do not use it yet. The VGG-5 and transformer rows give Adam an epsilon of 1e-12 instead of the default 1e-8, for all three methods: sPC's error reaches the first layers of these graphs scaled down by about the inference rate per hop, so their weight gradients are around 1e-8 to 1e-10, and at the default epsilon Adam barely moves them (under sPC the transformer's embedding stayed frozen). Backprop's gradients are far above both values. VGG-5 and transformer results from before this change used the default.

## The deeper conv rows

`cifar10-vgg7` and `cifar10-vgg9` are the `cifar10-vgg5` recipe with a deeper `create_vgg(depth, fuse_pool=True)`: 3x3 convs with a 2x2 max pool fused into the last conv of each block, 128,128 | 256,256 | 512,512 for VGG-7 and one more 512,512 block for VGG-9. Everything else (loaders, optimizer, schedule, Adam epsilon, settling settings, safety net) is VGG-5's. These layouts are ours and need the sponsor's confirmation: pcx's VGG-7 (Table 5 of arXiv 2407.01163) has the same channels but leaves some convs unpadded, pcx only ships VGG-7 configs for CIFAR-100 and Tiny ImageNet, and it has no VGG-9 code. The sPC rows keep VGG-5's 8 settling steps at rate 0.015. A check on one batch at initialization says this is probably not enough: sPC's weight update matched backprop's (cosine near 1) in only the top three or four conv layers, and the layers below got updates about 1e-5 to 1e-8 the size of backprop's with a cosine near 0, which is float noise. 12 steps bought one more layer. Read an sPC VGG-7/9 result with that in mind, and probe the settling rate or step count on a GPU before paying for a full run.

`cifar10-resnet18lean` is `cifar10-resnet18` built with `build_resnet18(lean=True)`. The plain ResNet-18 gives nine nodes a PC latent although they have no weights (eight `SkipConnection` adds and the global `AvgPool`), the same pattern that broke sPC on VGG-5 with separate pool nodes. The lean layout folds each add into the block's second conv (`ConvResidualNode`) and the average pool into the last block: 21 latents instead of 30, the same 2,795,210 parameters, and with the same weights the same forward pass, muPC scales included (the node applies the scales muPC would have put on the folded edges). So its backprop row trains the same network as the plain one, and the PC rows differ only in the latents. The plain rows are unchanged. On one CIFAR-10 batch at initialization, sPC's update agreed with backprop's (cosine above 0.5) in more weight layers with the lean layout than the plain one at 8, 30 and 120 settling steps (10 vs 7, 18 vs 12, 13 vs 12 of 21); whether that turns into better accuracy needs a GPU run. Code: `fabricpc/bench/rows_deep_convnets.py`. None of these rows has an expected score yet.

## Reading the files with other tools

Trial files are strict JSON except for a run that diverged: a metric that became NaN or infinite is written as `NaN`/`Infinity`, which Python's `json` reads but `jq` and JavaScript's `JSON.parse` do not. That only happens when a run broke, which is worth seeing; the safety net's routine NaN (sPC has no `lambda_min`) is written as `null`.
