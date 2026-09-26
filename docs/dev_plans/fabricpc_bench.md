# fabricpc.bench — reproducible benchmark suite and model zoo (v0.2)

Status: v0.2, 2026-09-26. Most of the design below is built and tested on branch
`vishal/bench-skeleton`; see "Status" at the end for what is built, what was measured, and what
changed from v0.1. Written against FabricPC 0.6.0 (commit 8406e6a). Resolves issue #59. Authors: Team 16, Penn State Behrend SWENG 480/481
(Tiffany Hart, Harry Maldonado, Victoria Worthington, Vishal Bidari). Mentor: Matthew Behrend.

## Context

FabricPC has three of the four pieces a definitive predictive-coding library needs (muPC on DAGs, a
graph-general node/edge API, paired significance testing). It has no reproducible benchmark suite,
so none of its numbers are citable and no regression check exists for "the library still trains
models to the same accuracy after this PR". Issue #59 specifies the suite; the 2026-09-22 kickoff
narrowed and clarified it:

- The deliverable is a *pattern* for writing benchmarks that are self-consistent across each
  other, plus a **model zoo** (trained checkpoints) and a results table with accuracy, compute time,
  and a statistical spread over several trials. The specific numbers matter less than the
  infrastructure that produces them (M. Behrend, kickoff).
- The central comparison is **sPC vs ePC vs backprop on one graph** (`InferenceSGD` /
  `EPCInference` / `algorithm="backprop"`), which the unified trainer already makes one flag apart.
- pcx (arXiv:2407.01163) is a reference for *which* tasks to include, not a target to be pinned to.
  The team defines and defends its own model list.
- Output format is the team's design. A numerical regression check in CI does not exist and is a
  known gap; a lightweight one is in scope.
- A user story from the faculty advisor: train an autoencoder under backprop and under PC, run both
  through the suite, and compare on metrics other than accuracy (e.g. hidden-layer sparsity).
- The team will have a shared dual-GPU machine from the SingularityNET pool.

Depends on: XLA flag profiles (#50) for recording the active profile; model checkpointing
(`docs/dev_plans_archive/model_checkpointing.md`, PR #38) for the zoo. Neither has shipped on
`main` at 0.6.0; both are worked around below.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Location | New subpackage `fabricpc/bench/` with a `__main__.py`, so `python -m fabricpc.bench <row>` works from any install. | Issue #59's stated command form; ships with the library; importable by the future nightly job. |
| Row definition | Frozen Python dataclasses in `fabricpc/bench/registry.py`, one `BenchmarkRow` per row, keyed by id `{dataset}-{model}-{algorithm}` (e.g. `cifar10-vgg5-spc`). Optional YAML overrides for hyperparameters. | Type-checked, greppable, testable; no schema language to invent. YAML only for the numbers that tune. |
| Arms | Every row has exactly one algorithm. A *comparison* is a named set of rows sharing dataset and model (e.g. `cifar10-vgg5` = {spc, epc, backprop}) run through `PlannedMultiContrastExperiment` with declared contrasts. | Keeps "one graph in all arms" by construction; reuses the existing paired framework unchanged. |
| Seeds / trials | Default `n_trials=5`, per-row override. Trial `i` uses the framework's `seed_offset + i*1000`. Never report a single-seed accuracy: the writer refuses to emit an aggregate row when `n_trials < 2`. | pcx used 5; 5 is enough for a mean±SE; contrasts get NaN statistics below 2, which the writer rejects. |
| Reproducibility definition | A row reproduces its reference when `abs(mean − ref) <= band`, band per row, default `max(0.5 pp, 2·SE)`. Bitwise agreement is never claimed. | Issue #59 ("within a stated band over a stated seed count, never bitwise"). Default band is a proposal for sponsor sign-off. |
| Isolation | One subprocess per (row, trial). Parent orchestrates, child runs and prints one JSON line. | Fresh JAX memory pool per trial makes peak memory honest; a crashed trial cannot poison the others. Pattern from `examples/scaling/mlp_scaling.py`. |
| Timing protocol | `warmup_steps=5` untimed; then `>=30` timed steps, each ending in `jax.block_until_ready`; report the median. Compile time measured separately as wall-clock of the first step. | Excludes JIT, defeats async dispatch, resists scheduler noise. |
| Memory | `device.memory_stats()["bytes_in_use"]` after warmup, plus `peak_bytes_in_use` reported separately with a caveat that it includes compile buffers. | Matches the scaling script's reasoning; both numbers are useful, labelled honestly. |
| Compute accounting | Per-node analytic matmul-FLOP counter summed over `structure.edges`; per row: `flops_per_update_pc`, `flops_per_update_bp`, `pc_to_bp_ratio`, `achieved_tflops = flops / median_step_time`. Chain-MLP closed form (`2·D·T + D` vs `3·D` matmuls) is a unit test, not the implementation. | Issue #59 requires it; conv, transformer, merge and Hopfield nodes break the closed form. |
| Output | `results/<row_id>/<run_id>/` containing `trials.csv` (one line per trial), `summary.json` (mean, SE, contrasts, band verdict), `manifest.json` (environment). `schema_version: 1` in every file. `run_id = <UTC timestamp>-<git sha7>`. | Human-readable + machine-readable; the nightly trend job (#54) consumes `summary.json`. |
| Manifest | FabricPC version + git SHA, JAX/jaxlib/optax/tfds versions, Python, platform, GPU name and driver, XLA flags actually set (raw `XLA_FLAGS` until #50 lands, then the profile name), full row config, seed list, command line. | What a stranger needs to reproduce. |
| Model zoo | After each trial the final `params` are saved under `zoo/<row_id>/trial<i>/`. Until PR #38 lands: `orbax.checkpoint.StandardCheckpointer` on the params pytree plus a JSON sidecar with the row id and manifest. Switch to `fabricpc.serialization.save_checkpoint` when it merges. | Sponsor wants checkpoints; do not block on an unmerged PR; do not write a second checkpoint format that outlives its purpose. |
| Metrics beyond accuracy | Every row records `accuracy` or `perplexity` (from `evaluate`'s defaults) and `energy`/`target_energy`. Rows may declare extra `EvalMetric`s; the autoencoder rows add `reconstruction_mse` and `hidden_sparsity` (fraction of `z_latent` entries below a threshold at the bottleneck node). | Faculty-advisor user story; the metric system already accepts caller-supplied metrics. |
| CI | `python -m fabricpc.bench smoke` runs `mnist-mlp` for 2 epochs, 1 trial, CPU, in under 3 minutes, and asserts (a) the result files validate against the schema and (b) accuracy exceeds a floor. Wired into `test.yml` as a job. Full rows are never run in per-PR CI. | Closes the "code runs but the math changed" gap on a lightweight model without slowing CI. |
| Library contributions | `fabricpc/models/vgg.py` (`create_vgg(depth in {5,7,9}, ...)` following `create_deep_transformer`'s style) and `TinyImageNetLoader` in `fabricpc/utils/data/dataloader.py` on the `_TfdsImageLoader` base with a custom download of the Stanford zip. `FashionMnistLoader` already exists. | Only true gaps. |
| Version pin | Develop against 0.6.0; rebase on `main` at agreed points with the mentor. | 0.6.0 broke the node contract; a moving base would eat the semester. |

## Scope: the model zoo

Tiers set the order of work, not a deadline: Tier 1 first, then Tier 2, with Tier 3 as stretch.

| Tier | Row family | Dataset | Model | Notes |
|---|---|---|---|---|
| 1 | `mnist-mlp` | MNIST | 784-256-64-10 MLP (`examples/mnist_demo.py`) | Vertical slice; smoke test; CPU-runnable. |
| 1 | `fashionmnist-mlp` | FashionMNIST | same MLP | Loader exists. |
| 1 | `cifar10-vgg5` | CIFAR-10 | VGG-5 (new builder) | pcx's headline task; reference 89.47 (CN) noted, not targeted. |
| 1 | `cifar10-resnet18` | CIFAR-10 | ResNet-18 (`examples/resnet18_cifar10_demo.py`) | Already built; wraps the demo's builder. |
| 1 | `tinyshakespeare-transformer` | Tiny Shakespeare | transformer v2, char-level | Perplexity row; builder exists. |
| 2 | `cifar100-vgg5`, `tinyimagenet-vgg5` | CIFAR-100, Tiny-ImageNet | VGG-5 | Needs `TinyImageNetLoader`. |
| 2 | `cifar10-vgg7`, `cifar10-vgg9` | CIFAR-10 | VGG-7/9 | Depth series for the depth-ceiling story. |
| 2 | `mnist-autoencoder` | MNIST | 784-128-32-128-784 | Faculty-advisor story; `reconstruction_mse`, `hidden_sparsity`. |
| 2 | `mnist-hopfield-retrieval` | MNIST | Storkey-Hopfield node | Associative-memory row; builder in `storkey_hopfield_demo.py`. |
| 2 | `tinyshakespeare-transformer-bpe` | Tiny Shakespeare | transformer v2, BPE | Second perplexity row. |
| 3 | pcx-match rows | as pcx | VGG-5 with nudging | Requires nudging in the trainer (not present at 0.6.0); separate design doc. |
| 3 | `deep-*` | MNIST / CIFAR-10 | 100+ layer muPC chains | Extension arm from #59. |

Every Tier 1 and 2 family has three rows: `-spc`, `-epc`, `-backprop`. That is 30 rows at Tier 1+2.

## Design

### Symbol table

- `BenchmarkRow`: frozen dataclass — `id`, `dataset`, `model_factory` (callable `(rng_key, algorithm) -> (params, structure)`), `loader_factory` (`(seed) -> (train, test)`), `algorithm` in {`spc`, `epc`, `backprop`}, `train_config`, `optimizer_factory`, `n_trials`, `reference` (optional `(value, source, band)`), `extra_metrics`, `tier`.
- `Comparison`: frozen dataclass — `id`, `rows` (tuple of row ids), `contrasts` (tuple of `(a, b)`), `metric`.
- `TrialResult`: one trial's numbers — task metrics, timing block, memory block, compute block, checkpoint path.
- `RunSummary`: aggregate — per-metric mean/std/SE/n, contrast results, band verdict, manifest ref.

### One row, one trial (child process)

1. Parse `--row <id> --trial <i> --out <dir>`; look up the row; derive `trial_seed`.
2. `setup_jax()`; record raw `XLA_FLAGS`, devices, versions into the manifest.
3. `params, structure = row.model_factory(key, row.algorithm)`; count params; run the FLOP counter over `structure`.
4. Build loaders from `row.loader_factory(trial_seed)`.
5. `step = make_train_step(structure, optimizer, algorithm=algo)`; time the first call as `compile_time_s`; run `warmup_steps`; time `timed_steps` with sync; median → `step_time_ms`. Read memory.
6. Full training via `train(...)` with the row's `num_epochs` (the timing loop above is separate from the real training so its steps do not count against the schedule).
7. `evaluate(...)` with default metrics plus `row.extra_metrics`.
8. Save checkpoint to `zoo/`; print one JSON line with everything; exit 0. Any exception → JSON line with `status: "failed"` and the traceback; exit 1.

### One comparison (parent process)

1. For each row and trial, spawn the child; collect JSON lines; write `trials.csv`.
2. Assemble per-arm metric arrays; run `paired_ttest` / `cohens_d` from `fabricpc/experiments/statistics.py` for each declared contrast (same trial seeds across arms guarantee pairing).
3. Compute band verdicts for rows with a `reference`.
4. Write `summary.json` and `manifest.json`; print a summary table.

Note: the parent does not reuse `PlannedMultiContrastExperiment.run()` directly, because that runs all arms in one process. It reuses its seed scheme and its statistics so results are identical to what the in-process runner would produce. (Alternative: subclass it with a subprocess hook. Deferred; see Open.)

### Realized settling steps

`InferenceSGD` / `EPCInference` run a fixed `infer_steps`, so realized `T` equals the config today. The field is still recorded so that adaptive termination (roadmap, Jan 2027) lands without a schema change.

## Public API

```python
# fabricpc/bench/__main__.py
#   python -m fabricpc.bench <row_id|comparison_id> [--trials N] [--out DIR] [--epochs E] [--dry-run]
#   python -m fabricpc.bench list
#   python -m fabricpc.bench smoke
#   python -m fabricpc.bench validate <results_dir>

# fabricpc/bench/registry.py
ROWS: Mapping[str, BenchmarkRow]
COMPARISONS: Mapping[str, Comparison]
def register_row(row: BenchmarkRow) -> None      # for contributors adding rows

# fabricpc/bench/runner.py
def run_trial(row: BenchmarkRow, trial: int, out: Path) -> TrialResult
def run_comparison(cmp: Comparison, out: Path, n_trials: int | None = None) -> RunSummary

# fabricpc/bench/measure.py
def time_steps(step, params, opt_state, batches, key, *, warmup: int, timed: int) -> TimingBlock
def memory_snapshot() -> MemoryBlock
def count_matmul_flops(structure, batch_size, algorithm, infer_steps) -> ComputeBlock

# fabricpc/bench/writer.py
def write_trial(...); def write_summary(...); def write_manifest(...); def validate(results_dir) -> None

# fabricpc/models/vgg.py
def create_vgg(depth: int, num_classes: int, input_shape: tuple, inference, activation=..., scaling=None) -> GraphStructure

# fabricpc/utils/data/dataloader.py
class TinyImageNetLoader(_TfdsImageLoader): ...   # 200 classes, 64x64, val labels from val_annotations.txt
```

## Alternatives considered

| Alternative | Rejected because |
|---|---|
| Rows as YAML files only | Untyped; every row would re-encode model construction as strings; harder to test. Python registry + YAML overrides keeps both. |
| One process for all trials (reuse `PlannedMultiContrastExperiment.run()` as-is) | Peak-memory readings are meaningless after the first trial; one crash loses the run. |
| Hydra / MLflow / W&B for config and tracking | New dependencies with their own state; the sponsor's reproducibility story is pinned versions + one command + recorded flags, not a tracking server. Aim already exists for dashboards. |
| Containers as the reproducibility unit | Useful later; not how the maintainers run things today; adds a build system. Deferred. |
| Pin to pcx's exact recipes and targets | Sponsor explicitly unpinned it at kickoff; nudging is not in the trainer. Kept as Tier 3. |
| Bitwise reproducibility | Impossible under the production XLA profile; issue #59 forbids claiming it. |

## Open / deferred

- Band default (`max(0.5 pp, 2·SE)`) and `n_trials=5` need sponsor sign-off.
- Whether to subclass `PlannedMultiContrastExperiment` with a subprocess hook instead of the parallel parent above.
- Exact `hidden_sparsity` threshold for the autoencoder rows.
- Whether the zoo stores optimizer state (resume) or params only (inference). Proposal: params only.
- Nudging design doc (Tier 3) — separate document.
- Where the dual-GPU machine lives and how results get copied off it.

## File changes

New: `fabricpc/bench/{__init__,__main__,registry,runner,measure,writer,schema}.py`, `fabricpc/bench/rows/*.py` (one file per family), `fabricpc/models/vgg.py`, `tests/test_bench_*.py`, `docs/user_guides/18_benchmark_suite.md`.
Modified: `fabricpc/utils/data/dataloader.py` (+`TinyImageNetLoader`), `fabricpc/models/__init__.py`, `.github/workflows/test.yml` (+smoke job), `CHANGELOG.md`, `pyproject.toml` (optional `[bench]` extra if any new dependency appears; none expected).

## Implementation steps

1. Skeleton: registry, `BenchmarkRow`, `mnist-mlp-{spc,epc,backprop}`, CLI `list` and `--dry-run`. Tests for registry.
2. Runner + writer for one trial on CPU; `trials.csv`, `manifest.json`, schema validation. **Milestone: vertical slice** (target: Week 7, Report 1.0).
3. `measure.py`: timing protocol, memory, FLOP counter with the chain-MLP unit test.
4. Comparison runner: subprocess orchestration, paired statistics, `summary.json`, band verdict.
5. `smoke` command + CI job.
6. `create_vgg`, `cifar10-vgg5-*` rows; first GPU runs on the sponsor machine.
7. Wrap ResNet-18 and transformer v2 builders as rows. **Milestone: Tier 1 complete.**
8. `TinyImageNetLoader`, CIFAR-100, VGG-7/9, autoencoder, Hopfield rows.
9. Zoo publication + `18_benchmark_suite.md` + upstream PRs.

## Test plan

- Registry: every row id parses as `{dataset}-{model}-{algorithm}`; every comparison's rows share dataset and model; no duplicate ids.
- FLOP counter: chain MLP matches `2·D·T + D` (PC) and `3·D` (BP) exactly; conv node count matches a hand computation for one layer.
- Timing: `time_steps` with a mocked step returns the median, excludes warmup, and calls `block_until_ready`.
- Writer: schema round-trip; refuses aggregate output for `n_trials < 2`; `validate` rejects a file missing `schema_version`.
- Runner: `mnist-mlp-spc` 1 epoch on CPU produces all fields; a child that raises produces `status: failed` and a non-zero parent exit.
- Statistics: paired contrast over identical arms yields `mean_diff == 0`.

## Verification

Demo results block, per CONTRIBUTING: the command that produced every number, the machine, JAX version, and the result files committed under `docs/bench_results/` for Tier 1 rows.

## Risks

| Risk | Mitigation |
|---|---|
| GPU machine arrives late | Everything through step 5 runs on CPU; Tier 1 rows have CPU-sized `--epochs` overrides for development. |
| 0.7.0 breaks the node contract again | Pinned base; rebase only at agreed points; the suite touches trainer/evaluate/make_train_step and nothing deeper. |
| Timing accidentally includes compile or async work | Protocol enforced in `measure.py`, not per row; unit-tested with a mock. |
| Single-seed numbers leak into a README or slide | Writer refuses to emit aggregates below 2 trials; docs state the rule. |
| Team over-reaches on the matrix | Tiers; Tier 1 is five families, all with existing builders except VGG-5. |
| Checkpoint PR #38 changes shape | Zoo writer isolated behind one function; swap when it lands. |

## Status (v0.2, 2026-09-26)

### Built

| Piece | Where | Notes |
|---|---|---|
| Rows and families | `registry.py` | 15 Tier 1 rows; each family compares spc vs backprop, epc vs backprop, epc vs spc; each row has a main metric (accuracy, or perplexity for text) |
| One process per trial | `isolate.py` | parent runs `python -m fabricpc.bench <row> --trial i --in-process`; a child that dies still leaves a failed trial with its exit code and stderr |
| Timing | `measure.time_steps` | the step is lowered and compiled once (timed on its own), then warmup, then the median of the timed steps with `block_until_ready` |
| Memory | `measure` | `step_memory` from XLA's memory analysis of the compiled step; `peak_memory_bytes` kept, but on GPU it is set by compile scratch space and is the same for every algorithm |
| Compute | `compute.py` | per-edge matmul and FLOP count; `ConvPoolNode` convs counted at their unpooled size |
| Pass/fail band | `band.py` | `|mean − expected| ≤ max(floor, 2·SE)`; only full runs get a verdict; marked pending sponsor sign-off |
| Comparisons | `compare.py` | finished trials loaded into `PlannedMultiContrastResults`; paired on the seeds all rows share; a test shows a trial here trains exactly what the framework would |
| Output | `writer.py` | `trial<i>.json`, `trials.csv` (rewritten after every trial), `summary.json`, `manifest.json`, `compare-<family>.json`, all with `schema_version: 2`; `validate` checks a folder |
| Learning curve | `runner.py` | test metrics on a fixed slice of the test set after every epoch (`--curve-batches`) |
| ePC regime | `measure.epc_regime` | `EPCInference.regime` at init and after training on ePC rows |
| Resume, zoo | CLI | `--resume` skips finished trials; `--zoo` saves weights (Orbax), path recorded relative to the results folder |
| Library additions | `models/vgg.py`, `models/transformer.py`, `nodes/convolutional.py`, `nodes/transformer_v2.py`, `utils/data` | `create_vgg` (5/7/9, `fuse_pool`), `ConvPoolNode`, `AugmentedImageLoader`, `create_deep_transformer(fuse_mlp=True)` with `MlpResidualNode` |
| Cloud runs | `scripts/lightning/` | job builder and fetcher for Lightning AI |

### Changed from v0.1

- **VGG layout.** v0.1 built each block as a `ConvNode` plus a `MaxPool` node. `MaxPool` carries its own latent state, so VGG-5 had eight latent layers against pcx's four, and sPC reached only 37% on CIFAR-10 after 50 epochs with pcx's own hyperparameters. With the pool fused into the conv node (`ConvPoolNode`) the same settings reach 84.4%. The VGG rows use the fused layout.
- **Seeds.** The runner now splits the seed exactly as `PlannedMultiContrastExperiment` does, and trains on fresh loaders after timing.
- **Timing and memory.** Compile is timed as lower-and-compile; memory is the compiled step's own needs (see above).
- **Recipes.** VGG-5 trains with flips and pad-4 crops and a warmup-cosine rate; the transformer rows use the tuned settings of `examples/transformer_v2_demo.py`.

### Measured so far

| Family | Backprop | ePC | sPC | Notes |
|---|---|---|---|---|
| `mnist-mlp` (Colab T4, 5 seeds, 20 epochs) | 98.16 | 98.15 | 98.18 | before the seed-split change |
| `fashionmnist-mlp` (same) | 88.88 | 88.81 | 88.40 | |
| `cifar10-vgg5` fused (Lightning T4, 50 epochs) | 86.76 ± 0.04 (5 seeds) | 86.80 ± 0.08 (3) | 84.44 ± 0.16 (3) | sPC − BP −2.29 pt, p = 0.002; ePC − BP +0.06 pt, p = 0.20 |

ePC used `EPCInference`'s defaults (η 1e-3, T 5). On the VGG-5 checkpoints its regime is backprop-like at init (f̄ 0.009) and 0.08–0.25 after training, so the VGG-5 result describes default ePC, which is close to backprop, not relaxed PC.

### Open

- Band rule and seed count: sponsor sign-off.
- ePC at a rate chosen from the spectrum, so that runs are near PC equilibrium, as a separate row or arm.
- Transformer cost: the char loader makes one sequence per character (about 62,700 steps per epoch at batch 16); decide epochs or a sequence cap before full runs.
- ResNet-18 sPC (T = 120) needs days of GPU time.
- Upstream pull requests.
