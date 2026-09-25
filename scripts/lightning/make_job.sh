#!/bin/bash
# Build the command for a Lightning AI job that runs one benchmark and keeps
# its results. Prints the command; pass it to `lightning job run --command`.
#
#   scripts/lightning/make_job.sh "<bench args>" <training time limit, s> \
#       [commit] [results folder on the Lightning Drive to resume from]
#
# Lightning starts the command with plain `sh`, so the command only writes
# the real script to a file and hands it to bash. The script:
#   * checks out an exact commit, so every job runs the same code;
#   * runs the benchmark with --zoo, so trained weights are saved;
#   * limits only the training with `timeout`, so the steps after it
#     always run, even when training fails or runs out of time;
#   * copies everything (results and weights) to the job's artifacts folder;
#   * prints the small result files (not the weights) into the log as a
#     base64 tarball between markers, as a backup. fetch_results.sh reads it.
#
# With STUDIO_MODE=1 the job is meant for `lightning job run --studio
# fabricpc`: it uses the Studio's own checkout and .venv instead of fetching
# and installing, and writes everything under the Studio folder, in
# runs/<job name>/. Lightning keeps the files a Studio job changes there as
# the job's artifacts, so the weights survive; nothing else does.
set -euo pipefail
args="$1"
limit="$2"
# GitHub only serves a commit by its full id, so expand short ids here.
commit="$(git -C "$(dirname "$0")" rev-parse --verify "${3:-HEAD}^{commit}")"
restore="${4:-}"

if [ -n "${STUDIO_MODE:-}" ]; then
  home=/teamspace/studios/this_studio
  echo "export FABRICPC_REPO=$home/FabricPC PATH=$home/FabricPC/.venv/bin:\$PATH"
  echo "export TFDS_SRC=none ART=none OUT=$home/runs/\$LIGHTNING_JOB_NAME"
fi
cat <<OUTER
cat > /tmp/fabricpc_job.sh <<'FABRICPC_JOB'
#!/bin/bash
set -uo pipefail
export GIT_PAGER=cat PAGER=cat PIP_PROGRESS_BAR=off PYTHONUNBUFFERED=1
exec < /dev/null
BENCH_ARGS="$args"
LIMIT=$limit
COMMIT=$commit
RESTORE="$restore"
export FABRICPC_GIT_SHA=$commit
OUTER
cat <<'SCRIPT'
OUT=${OUT:-/tmp/out}
ART=${ART:-/teamspace/jobs/${LIGHTNING_JOB_NAME:-local}/artifacts}
TFDS_SRC=${TFDS_SRC:-/teamspace/uploads/tfds}
set -x
nvidia-smi -L || true

# Code: an existing checkout (FABRICPC_REPO), or a fresh one at COMMIT.
if [ -n "${FABRICPC_REPO:-}" ]; then
  cd "$FABRICPC_REPO"
else
  git init -q /work && cd /work
  git fetch -q --depth 1 https://github.com/VabAB2002/FabricPC.git "$COMMIT" \
    && git checkout -q FETCH_HEAD || { echo "SETUP FAILED: cannot get $COMMIT"; exit 1; }
  pip install -q -U pip
  pip install -q -e ".[all,cuda12]" "jax[cuda12]==0.10.2" "optax==0.2.8"
fi
# A Studio job's copy of the code has no .git; the manifest then reads
# FABRICPC_GIT_SHA.
git --no-pager log -1 --oneline 2>/dev/null || echo "code at $COMMIT"
python -c "import jax; print('JAX', jax.__version__, jax.devices())"

# Datasets already on the Drive save a slow download.
if [ -d "$TFDS_SRC" ]; then
  mkdir -p ~/tensorflow_datasets && cp -r "$TFDS_SRC"/. ~/tensorflow_datasets/
fi

mkdir -p "$OUT"
if [ -n "$RESTORE" ]; then
  # A Drive folder name, or an absolute path (used when testing locally).
  case "$RESTORE" in /*) src="$RESTORE" ;; *) src="/teamspace/uploads/$RESTORE" ;; esac
  cp -r "$src"/. "$OUT"/
fi

timeout "$LIMIT" python -m fabricpc.bench $BENCH_ARGS --out "$OUT" --zoo "$OUT/zoo"
rc=$?
echo "BENCH EXIT $rc"
python -m fabricpc.bench validate "$OUT" || true

if [ "$ART" != none ]; then
  mkdir -p "$ART" && cp -r "$OUT"/. "$ART"/ && echo "ARTIFACTS COPIED TO $ART" \
    || echo "ARTIFACTS COPY FAILED"
fi
set +x
echo "=== RESULTS TARBALL"
tar -czf - --exclude=./zoo -C "$OUT" . | base64 -w 76
echo "=== END TARBALL"
exit $rc
FABRICPC_JOB
exec bash /tmp/fabricpc_job.sh
SCRIPT
