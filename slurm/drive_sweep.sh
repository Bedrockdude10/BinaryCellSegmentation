#!/bin/bash
# slurm/drive_sweep.sh — resubmit the sweep array until every run is banked.
#
#   ./slurm/drive_sweep.sh [max_rounds]
#
# The `gpu` partition caps wall clock at 8h and the submit cap allows only 8
# array tasks, so the sweep needs more than one submission. Each round is
# idempotent: completed runs are skipped outright and a part-trained run resumes
# from last.pt with its RNG state, so resubmitting the identical script is always
# safe.
#
# Stops when all runs are complete, when max_rounds is reached, or when two
# consecutive rounds bank nothing new (which means something is failing rather
# than just running out of time — check slurm_logs/).
#
# Progress is appended to slurm/drive_sweep.log on this machine.
set -uo pipefail

HOST="${EXPLORER_HOST:?set EXPLORER_HOST, e.g. user@cluster.example.edu}"
REPO="~/BinaryCellSegmentation"
MAX_ROUNDS="${1:-6}"
EXPECTED=35
LOG="$(cd "$(dirname "$0")" && pwd)/drive_sweep.log"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_done() {
  ssh -o BatchMode=yes "$HOST" \
    "ls -d $REPO/runs/*/seed*/results.json 2>/dev/null | wc -l" 2>/dev/null || echo 0
}

queued() {
  ssh -o BatchMode=yes "$HOST" 'squeue -u $USER -h -o "%T" | wc -l' 2>/dev/null || echo 0
}

prev=-1
stalled=0

for round in $(seq 1 "$MAX_ROUNDS"); do
  done_now=$(count_done)
  say "round $round: $done_now/$EXPECTED complete"
  if [[ "$done_now" -ge "$EXPECTED" ]]; then
    say "all runs banked"
    break
  fi
  if [[ "$done_now" -eq "$prev" ]]; then
    stalled=$((stalled + 1))
    say "no progress since last round (${stalled}x)"
    if [[ "$stalled" -ge 2 ]]; then
      say "STOPPING: two rounds with no progress — inspect slurm_logs/ and runs/*/*/FAILED.txt"
      exit 1
    fi
  else
    stalled=0
  fi
  prev=$done_now

  # Wait for any straggler jobs before submitting, so we stay under the cap.
  while [[ "$(queued)" -gt 0 ]]; do sleep 120; done

  jid=$(ssh -o BatchMode=yes "$HOST" "cd $REPO && sbatch --parsable slurm/sweep.sbatch" 2>&1)
  if [[ ! "$jid" =~ ^[0-9]+$ ]]; then
    say "SUBMIT FAILED: $jid"
    exit 1
  fi
  say "submitted array $jid; waiting for it to drain"
  sleep 60
  while [[ "$(queued)" -gt 0 ]]; do sleep 180; done
  say "array $jid finished"
done

final=$(count_done)
say "driver exiting with $final/$EXPECTED complete"
[[ "$final" -ge "$EXPECTED" ]]
