#!/bin/bash
# slurm/sync_to_explorer.sh — push code (and optionally data) to Explorer.
#
#   ./slurm/sync_to_explorer.sh              # code only
#   ./slurm/sync_to_explorer.sh --data       # code + the 2.3 GB flat caches
#   ./slurm/sync_to_explorer.sh --weights    # code + timm pretrained weights
#   ./slurm/sync_to_explorer.sh --pull       # bring run artifacts back (no push)
#
# WARNING: do not push code while a sweep is in flight. Every results.json
# records a code_hash over src/, scripts/ and configs/; changing those mid-sweep
# splits the runs across two hashes and scripts/aggregate.py will (rightly) warn
# that they may not belong in one table. Push, then launch.
#
# Note the LEADING SLASHES on the directory excludes. An unanchored `data/`
# matches any directory named data at any depth and silently drops `src/data/`,
# which fails later as a confusing `ModuleNotFoundError: No module named
# 'src.data'` on the compute node.
set -euo pipefail

HOST="${EXPLORER_HOST:-rollo.l@explorer.northeastern.edu}"
# Large transfers go through the transfer node; login-node rsync of multi-GB
# files gets reset mid-stream. Same shared home, so only the hostname differs.
XFER="${EXPLORER_XFER:-rollo.l@xfer.discovery.neu.edu}"
REPO_DEST="~/BinaryCellSegmentation"
DATA_DEST="~/nucseg_data"
HF_DEST="~/hf_cache/hub"

cd "$(dirname "$0")/.."

# Pull-only mode: fetch the artifacts the aggregator reads, not the checkpoints.
for arg in "$@"; do
  if [[ "$arg" == "--pull" ]]; then
    echo "=== run artifacts <- $HOST:$REPO_DEST/runs"
    mkdir -p runs
    rsync -az --stats \
      --include='*/' --include='results.json' --include='progress.json' \
      --include='config.resolved.yaml' --include='train.log' --include='table3*.json' \
      --exclude='*' \
      "$HOST:$REPO_DEST/runs/" runs/ | tail -4
    echo "=== done"
    exit 0
  fi
done

echo "=== code -> $HOST:$REPO_DEST"
rsync -az --stats \
  --exclude='/data/' --exclude='/figures/' --exclude='/demo_output/' \
  --exclude='/Checkpoint Export/' --exclude='/outputs/*/tensorboard/' \
  --exclude='/runs/' --exclude='/runs_smoke/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
  --exclude='*.pdf' --exclude='*.aux' --exclude='.DS_Store' \
  ./ "$HOST:$REPO_DEST/" | tail -4

for arg in "$@"; do
  case "$arg" in
    --data)
      echo "=== flat caches -> $XFER:$DATA_DEST (2.3 GB, resumable)"
      rsync -a --partial --stats \
        data/pannuke_flat_fold1 data/pannuke_flat_fold2 data/pannuke_flat_fold3 \
        data/monuseg_flat_train_256 data/monuseg_flat_test_256 \
        "$XFER:$DATA_DEST/" | tail -4
      ;;
    --weights)
      echo "=== timm Swin-T weights -> $HOST:$HF_DEST"
      ssh "$HOST" "mkdir -p $HF_DEST"
      rsync -a --partial \
        "$HOME/.cache/huggingface/hub/models--timm--swin_tiny_patch4_window7_224.ms_in1k" \
        "$HOST:$HF_DEST/"
      ;;
  esac
done

echo "=== done"
