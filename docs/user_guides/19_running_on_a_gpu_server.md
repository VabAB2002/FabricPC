# Running on a GPU Server

This guide is for a shared GPU machine you reach through JupyterHub (or any Linux box with an NVIDIA GPU and a terminal). The goal is one command that works through the whole benchmark campaign, keeps going when you close the browser, and can be restarted without losing finished work.

The tool is the benchmark suite's run queue:

```bash
python -m fabricpc.bench queue <plan.json> --out <dir> [--dry-run] [--only IDS] [--continue-on-failure]
```

A **plan** is a JSON file listing jobs in priority order. Each job is a row (`mnist-mlp-spc`) or a family (`cifar10-vgg5`, all three rows plus the comparison), with optional `trials`, `epochs`, `zoo` (save trained weights, `true`/`false`) and a `note`. `fabricpc/bench/plans/gpu_campaign.json` is the plan for the waiting GPU work. The deeper VGGs (VGG-7 and VGG-9) have their own plan, `fabricpc/bench/plans/gpu_deep_vgg.json`, which waits for the sponsor's sign-off on their layouts and starts with short sPC probes; read its description before running it. See [Benchmark Suite](18_benchmark_suite.md) for what a row and a family are.

## 1. Open a terminal

In JupyterHub: **File → New → Terminal**. Everything below is typed there, not in a notebook cell.

## 2. Get the code

The benchmark suite (`fabricpc/bench`) is not on `main` yet. It lives on the `vishal/bench-skeleton` branch, so clone that branch, not the default one:

```bash
cd ~
git clone -b vishal/bench-skeleton https://github.com/VabAB2002/FabricPC.git
cd FabricPC
ls fabricpc/bench/plans/      # should list gpu_campaign.json; if not, you are on the wrong branch
git log -1 --oneline          # note the commit; every result records it too
```

If you already cloned `main`, run `git fetch origin && git checkout vishal/bench-skeleton` inside `FabricPC`. To run an exact commit (so every result comes from the same code), `git checkout <commit>` after that; use the commit the person who wrote the plan gives you. Once the suite is merged into `main`, a plain clone works and this step goes back to normal.

## 3. Make the environment

Use the same install as the Lightning jobs (`scripts/lightning/make_job.sh`): the CUDA 12 extras with pinned JAX and Optax versions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[all,cuda12]" "jax[cuda12]==0.10.2" "optax==0.2.8"
```

Use `cuda13` instead of `cuda12` only if `nvidia-smi` shows a driver of 580 or newer (see [Installation](01_installation.md)). If the hub already gives you a Python with JAX installed, still make your own `.venv`: mixing JAX versions is the most common way to end up silently on the CPU.

## 4. Check that JAX sees the GPU

```bash
nvidia-smi -L
python -c "import jax; print(jax.__version__, jax.devices())"
```

You want to see something like `[CudaDevice(id=0)]`. If it says `CpuDevice`, stop and fix the install first; a queue run on the CPU would take weeks. The queue also writes the device into its status file, so a CPU run can never pass for a GPU one.

## 5. Look before you start

```bash
python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json --validate-plan
python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json --out ~/runs/gpu-campaign --dry-run --probe
```

`--validate-plan` lists job ids that are not in the registry yet (families still being added). They are fine to leave in: the queue skips them with a message and runs everything else. `--dry-run` prints the jobs in order. With a bare `--probe` it also times a few training steps of every row on this GPU (a few minutes, and it downloads datasets it needs) and prints an estimate in hours; the probe files land in `~/runs/gpu-campaign/probes/`, so later dry runs reuse them. Treat the estimate as a floor.

## 6. Start the queue so it survives closing the browser

A program started in a JupyterHub terminal dies when the terminal session goes away. Start it inside `tmux` (best) or with `nohup`.

With **tmux**:

```bash
tmux new -s bench
source .venv/bin/activate
mkdir -p ~/runs                   # tee can not create the log's folder itself
python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json --out ~/runs/gpu-campaign 2>&1 | tee -a ~/runs/gpu-campaign.log
```

Detach with `Ctrl-b` then `d`. Close the browser. Later, open a terminal and `tmux attach -t bench` to watch it again.

With **nohup** (if `tmux` is not installed):

```bash
mkdir -p ~/runs
nohup .venv/bin/python -m fabricpc.bench queue fabricpc/bench/plans/gpu_campaign.json \
    --out ~/runs/gpu-campaign >> ~/runs/gpu-campaign.log 2>&1 &
tail -f ~/runs/gpu-campaign.log      # Ctrl-C stops watching, not the queue
```

Other useful flags:

| Flag | Effect |
|---|---|
| `--only cifar10-vgg5,mnist-mlp` | run just these jobs from the plan |
| `--continue-on-failure` | go on to the next job when one fails (default: stop) |
| `--dry-run` | print the plan, run nothing |

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
  mnist-mlp/ ...
  zoo/                       trained weights for jobs with "zoo": true (large)
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
tar -czf gpu-campaign-results.tgz --exclude=gpu-campaign/zoo gpu-campaign
```

Then either download `gpu-campaign-results.tgz` from the JupyterHub file browser (right click → Download), or copy it from your own computer if you have SSH access:

```bash
scp <user>@<server>:~/runs/gpu-campaign-results.tgz .
mkdir -p gpu-campaign && tar -xzf gpu-campaign-results.tgz
python -m fabricpc.bench validate gpu-campaign/cifar10-vgg5
```

The weights in `zoo/` can be large (ResNet and VGG checkpoints for 5 seeds of every row); only fetch them if you need them, and check the hub's disk quota with `du -sh ~/runs/gpu-campaign/zoo` while the queue runs.

## Troubleshooting

- **`No module named fabricpc.bench`.** You are on `main`, which does not have the benchmark suite yet. Check out `vishal/bench-skeleton` (step 2), then reinstall with `pip install -e .`.
- **`jax.devices()` shows only a CPU.** The CUDA wheels did not install or do not match the driver. Reinstall in a fresh `.venv` with the command in step 3.
- **Out of GPU memory on a shared GPU.** Someone else may be using the card (`nvidia-smi`). Each trial runs in its own process, so one failed trial does not take the others down; restart the queue when the card is free and it resumes.
- **A job says `skipped`.** Its id is not in the registry of the code you cloned. Pull newer code, then restart; the other jobs keep their finished trials.
- **The queue stopped at a `failed` job.** Read the `error` in that job's `trial*.json`, fix it, and restart, or restart with `--continue-on-failure` to go on and come back to it.
