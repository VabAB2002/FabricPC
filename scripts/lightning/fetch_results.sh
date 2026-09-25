#!/bin/bash
# Copy a finished Lightning job's result files to this computer.
#
#   scripts/lightning/fetch_results.sh <job name> <destination folder> [log file]
#
# With a log file (e.g. from a local test run) it reads that instead of
# asking Lightning for the job's log.
# Reads the base64 tarball that make_job.sh prints between markers in the
# job log, checks it unpacks, and runs `validate` on the result.
set -euo pipefail
job="$1"
dest="$2"
mkdir -p "$dest"
log="$dest/$job.log"
if [ -n "${3:-}" ]; then
  cp "$3" "$log"
else
  lightning job logs "$job" --teamspace "${TEAMSPACE:-vabab2002/general}" > "$log" 2>&1
fi
awk '/=== RESULTS TARBALL$/{f=1; next} /=== END TARBALL$/{f=0} f' "$log" \
  | sed -E 's/^[^ ]+ //; s/^.*[[:space:]]//' \
  | grep -E '^[A-Za-z0-9+/=]+$' \
  | base64 -d | tar -xzf - -C "$dest"
echo "unpacked $job into $dest"
python -m fabricpc.bench validate "$dest"
