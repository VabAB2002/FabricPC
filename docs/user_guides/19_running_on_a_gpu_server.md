# Running on a GPU Server

This guide is for a shared GPU machine you reach through JupyterHub (or any Linux box with an NVIDIA GPU and a terminal). The goal is one command that works through the whole benchmark campaign, keeps going when you close the browser, and can be restarted without losing finished work.

The tool is the benchmark suite's run queue:

```bash
python -m fabricpc.bench queue <plan.json> --out <dir> [--dry-run] [--only IDS] [--continue-on-failure]
                                          [--validate-plan] [--probe] [--rehearse]
```

A **plan** is a JSON file listing jobs in priority order. Each job is a row (`mnist-mlp-spc`) or a family (`cifar10-vgg5`, all three rows plus the comparison), with optional `trials`, `epochs`, `zoo` (save trained weights, `true`/`false`) and a `note`. `fabricpc/bench/plans/gpu_campaign.json` is the plan for the waiting GPU work. The deeper VGGs (VGG-7 and VGG-9) have their own plan, `fabricpc/bench/plans/gpu_deep_vgg.json`, which waits for the sponsor's sign-off on their layouts and starts with short sPC probes; read its description before running it. See [Benchmark Suite](18_benchmark_suite.md) for what a row and a family are.

## 1. Open a terminal

In JupyterHub: **File → New → Terminal**. Everything below is typed there, not in a notebook cell.

## 2. Get the code

The benchmark suite (`fabricpc/bench`) is not on `main` yet. It lives on the `vishal/bench-next` branch, so clone that branch, not the default one, and check out the exact commit the person who wrote the plan gives you, so every result comes from the same code:

```bash
cd ~
git clone -b vishal/bench-next https://github.com/VabAB2002/FabricPC.git
cd FabricPC
git checkout <commit>         # the commit you were given
ls fabricpc/bench/plans/      # should list gpu_campaign.json; if not, you are on the wrong branch
git log -1 --oneline          # note the commit; every result records it too
```

If you already cloned `main`, run `git fetch origin && git checkout vishal/bench-next && git checkout <commit>` inside `FabricPC`. Do not `git pull` while a campaign is running: a restart would then train its remaining trials with different code than the finished ones. Once the suite is merged into `main`, a plain clone works and this step goes back to normal.

## 3. Make the environment

Use the same install as the Lightning jobs (`scripts/lightning/make_job.sh`): the CUDA 12 extras with pinned JAX and Optax versions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -U pip
pip install --no-cache-dir -e ".[all,cuda12]" "jax[cuda12]==0.10.2" "optax==0.2.8"
```

`--no-cache-dir` keeps pip from filling a small home quota with a copy of the CUDA wheels.

Then put the settings every campaign shell needs in one file, `~/bench_env.sh`, and source it in every terminal and tmux window:

```bash
source ~/FabricPC/.venv/bin/activate
export JAX_PLATFORMS=cuda                    # no GPU is an error, not a silent CPU run
export PYTHONUNBUFFERED=1                    # progress shows up in the log as it happens
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # do not grab 75% of a shared card per trial
export CUDA_VISIBLE_DEVICES=${GPU:-0}        # one queue per GPU: GPU=1 source ~/bench_env.sh
```

On a card other people use, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45` instead of `PREALLOCATE=false` caps our share. Without `JAX_PLATFORMS=cuda`, a broken CUDA install makes JAX fall back to the CPU with only a warning, and the whole campaign would run there.

Use `cuda13` instead of `cuda12` only if `nvidia-smi` shows a driver of 580 or newer (see [Installation](01_installation.md)). If the hub already gives you a Python with JAX installed, still make your own `.venv`: mixing JAX versions is the most common way to end up silently on the CPU.

## 4. Check that JAX sees the GPU

```bash
nvidia-smi -L
python -c "import jax; print(jax.__version__, jax.devices())"
```

You want to see something like `[CudaDevice(id=0)]`. If it says `CpuDevice`, stop and fix the install first; a queue run on the CPU would take weeks. The queue also writes the device into its status file, so a CPU run can never pass for a GPU one.

## 5. Look before you start

Ask the hub's admin first how long an idle or long-running session may live (the "cull" timeout and maximum age). tmux survives closing the browser but not a culled server, and a culled server loses the trial that was running, which for the biggest rows is hours.

```bash
python -m fabricpc.bench doctor --out ~/runs/gpu-campaign --expect-gpu
python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json --validate-plan
python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json --rehearse --out ~/runs/rehearsal --continue-on-failure
python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json --out ~/runs/gpu-campaign --dry-run --probe
```

`doctor --expect-gpu` fails unless JAX sees a GPU, and checks the datasets, the disk and one training step. `--validate-plan` lists job ids that are not in the registry and jobs that cannot get a pass/fail verdict (fewer than 3 seeds, or a shortened epoch count). `--rehearse` runs every job for 3 training steps with 2 seeds and 2 test batches, into its own folder; it downloads every dataset and compiles every model on this GPU, so problems show up in the first hour. Run it inside tmux (step 6), and check that every job in `~/runs/rehearsal/queue-status.json` is `done`. `--dry-run --probe` times a few training steps of every row and prints an estimate in hours; with about 40 rows to compile this can take 30 to 60 minutes. The probe files land in `~/runs/gpu-campaign/probes/`, so later dry runs reuse them. Treat the estimate as a floor and add 30 to 50%.

## 6. Start the queue so it survives closing the browser

A program started in a JupyterHub terminal dies when the terminal session goes away. Start it inside `tmux` (best) or with `nohup`.

With **tmux**:

```bash
tmux new -s bench
source ~/bench_env.sh
mkdir -p ~/runs                   # tee can not create the log's folder itself
pgrep -af fabricpc.bench          # must print nothing: never two queues on one folder
python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json --out ~/runs/gpu-campaign --continue-on-failure 2>&1 | tee -a ~/runs/gpu-campaign.log
```

Detach with `Ctrl-b` then `d`. Close the browser. Later, open a terminal and `tmux attach -t bench` to watch it again.

With **nohup** (if `tmux` is not installed):

```bash
mkdir -p ~/runs
source ~/bench_env.sh
nohup python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json \
    --out ~/runs/gpu-campaign --continue-on-failure >> ~/runs/gpu-campaign.log 2>&1 &
tail -f ~/runs/gpu-campaign.log      # Ctrl-C stops watching, not the queue
```

Other useful flags:

| Flag | Effect |
|---|---|
| `--only cifar10-vgg5,mnist-mlp` | run just these jobs from the plan |
| `--continue-on-failure` | go on to the next job when one fails (default: stop). Recommended: one failure (say a moment out of memory on a shared card) then does not leave the GPU idle overnight; rerunning the same command retries failed trials |
| `--dry-run` | print the plan, run nothing |
| `--validate-plan` | check the plan: unknown ids, and jobs that cannot get a verdict |
| `--probe` | with `--dry-run` or on its own: time every row and estimate the hours; trains nothing |
| `--rehearse` | every job for 3 training steps, 2 seeds, 2 test batches; give it its own `--out` |

**Two GPUs.** The queue uses one GPU. To use two, run two queues in two tmux windows, each with its own GPU and its own `--out`, and split the plan with `--only` so each family stays whole on one card (its timing comparisons stay fair): `GPU=0 source ~/bench_env.sh` with `--out ~/runs/gpu0 --only ...`, and `GPU=1 source ~/bench_env.sh` with `--out ~/runs/gpu1 --only ...`. `report ~/runs` then covers both.

## 7. Check on it

```bash
cat ~/runs/gpu-campaign/queue-status.json
```

The status file is rewritten when each job starts and ends. For every job it shows `pending`, `running`, `done`, `failed`, `skipped` (id not in the registry), `interrupted` or `not selected`, with start and end times, and at the top the git commit, the device and how many times the queue was restarted. A job that trained all its seeds but missed its expected score still counts as `done`; its message says so, and the row's `summary.json` has the verdict.

## 8. Where results land

```
~/runs/gpu-campaign/
  queue-status.json          the queue's own record
  report.md                  one readable page for everything, written at the end
  cifar10-vgg5/              one folder per job, the normal results layout
    cifar10-vgg5-spc/        manifest.json, trial0.json ..., trials.csv, summary.json
    compare-cifar10-vgg5.json
    zoo/                     this job's trained weights, if the job has "zoo": true
  mnist-mlp/ ...
  probes/                    probe files, if you ran --probe
```

When the queue finishes it runs `validate` on every job folder and writes `report.md`. You can also do both yourself at any time:

```bash
python -m fabricpc.bench validate ~/runs/gpu-campaign/cifar10-vgg5
python -m fabricpc.bench report ~/runs/gpu-campaign --out ~/runs/gpu-campaign/report.md
```

## 9. Resume after a stop

If the queue was stopped (you pressed Ctrl-C, the server restarted, the hub killed your session), run **the exact same command again**. Every job runs with `--resume`, so each trial that already finished is skipped (`trial 0: already done, skipping`) and only the trial that was cut off starts over. You lose at most one trial per stop. Finished jobs are checked again in seconds and move on.

Keep the same `--out` folder and the same plan. Changing a job's `epochs` means its old trials no longer count as finished (they trained for a different length), so they are run again.

## 10. Bring the results home

The small files (everything but the weights) are a few megabytes. Pack them without the zoo:

```bash
cd ~/runs
tar -czf gpu-campaign-results.tgz --exclude='*/zoo' gpu-campaign
```

Then either download `gpu-campaign-results.tgz` from the JupyterHub file browser (right click → Download), or copy it from your own computer if you have SSH access:

```bash
scp <user>@<server>:~/runs/gpu-campaign-results.tgz .
mkdir -p gpu-campaign && tar -xzf gpu-campaign-results.tgz
python -m fabricpc.bench validate gpu-campaign/cifar10-vgg5
```

The weights in each job's `zoo/` add up to about 1 to 5 GB for a whole campaign; only fetch them if you need them, and check the hub's disk quota with `du -sh ~/runs/gpu-campaign/*/zoo` while the queue runs.

## Troubleshooting

- **`No module named fabricpc.bench`.** You are on `main`, which does not have the benchmark suite yet. Check out `vishal/bench-next` and the commit you were given (step 2), then reinstall with `pip install -e .`.
- **`jax.devices()` shows only a CPU, or JAX fails to start with `JAX_PLATFORMS=cuda`.** The CUDA wheels did not install or do not match the driver. Reinstall in a fresh `.venv` with the command in step 3.
- **A trial seems stuck.** Nothing prints while a trial trains. Check `nvidia-smi` (is the GPU busy?) and `ls -lt ~/runs/gpu-campaign/*/*/trial*.json` (when did the last trial finish?).
- **Out of GPU memory on a shared GPU.** Someone else may be using the card (`nvidia-smi`). Each trial runs in its own process, so one failed trial does not take the others down; restart the queue when the card is free and it resumes.
- **A job says `skipped`.** Its id is not in the registry of the code you cloned. Pull newer code, then restart; the other jobs keep their finished trials.
- **The queue stopped at a `failed` job.** Read the `error` in that job's `trial*.json`, fix it, and restart, or restart with `--continue-on-failure` to go on and come back to it.
