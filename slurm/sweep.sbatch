#!/bin/bash
# slurm/sweep.sbatch — multi-seed sweep on Explorer.
#
#   sbatch slurm/sweep.sbatch
#
# Sizing notes (Explorer specifics):
#   * The `gpu` partition is HETEROGENEOUS (v100-sxm2, v100-pcie, t4, a100,
#     h200). `--gres=gpu:v100-sxm2:1` pins every run in Tables 1, 2 and 3 to one
#     GPU model; without it, hardware variance would be reported as seed
#     variance. v100-sxm2 is also the most abundant (44 GPUs over 12 nodes).
#   * The submit cap is 8 array TASKS per user, measured with
#     `sbatch --test-only` (9+ is rejected outright with
#     QOSMaxSubmitJobPerUserLimit, not queued) — hence `--array=0-7%4` rather
#     than one task per run. Note this account's association grants QOS `normal`,
#     not `gpu`; do not pass `--qos=gpu`, it is an invalid specification here.
#   * Partition max wall clock is 8h, less than the whole sweep, so each task
#     works through runs until its time budget is nearly gone, checkpoints, and
#     exits 0. Resubmitting this identical script continues: completed runs are
#     skipped, a part-trained run resumes from last.pt with its RNG state.
#   * Work is claimed dynamically, so 8 tasks share 35 runs without one task
#     inheriting all the slow configurations.
#
# Resources are measured, not guessed (slurm/preflight.sbatch job 8916611, on a
# Tesla V100-SXM2-32GB, in the fp32 regime the sweep actually uses): peak GPU
# memory 5.82 GiB for the heaviest config (pretrained Swin on PanNuke), host RSS
# ~8 GB, so --mem=24G leaves ample headroom for the memmapped caches and 8
# dataloader workers. Per-epoch cost: PanNuke 76 s (VGG) / 101 s (ResNet) /
# 128 s (Swin); MoNuSeg 12 s / 20 s. The longest single run (pretrained Swin on
# PanNuke) is roughly 3 h, so it fits inside one 8 h allocation without resuming.
# Whole sweep is about 45-50 GPU-hours.
#
# Resubmit until `python scripts/run_sweep.py --list` reports 35/35.

#SBATCH --job-name=nucseg-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%4
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm_logs/sweep_%A_%a.out
#SBATCH --error=slurm_logs/sweep_%A_%a.out

set -euo pipefail
T0=$(date +%s)
WALL_SECONDS=$((8 * 3600))

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p slurm_logs runs

echo "=== task ${SLURM_ARRAY_TASK_ID:-0} of job ${SLURM_ARRAY_JOB_ID:-local} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module load miniconda3/25.9.1 cuda/12.8.0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Pretrained Swin weights are pre-fetched into home; compute nodes stay offline
# so a missing file fails loudly instead of hanging on a download.
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 8 CPUs are for dataloader workers, not BLAS threads fighting each other.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── Stage the flat caches to node-local NVMe ─────────────────────────────────
# ~2.3 GB total. $TMPDIR is per-node NVMe (~298 GB, wiped at job end) and much
# better than hitting shared home for random reads every epoch of every run.
SRC="$HOME/nucseg_data"
DEST="${TMPDIR:-/tmp}/nucseg_cache"

# Guard every input BEFORE copying gigabytes — a 2-second failure beats
# discovering a missing file after a multi-GB pull.
for d in pannuke_flat_fold1 pannuke_flat_fold2 pannuke_flat_fold3 \
         monuseg_flat_train_256 monuseg_flat_test_256; do
  for f in images.npy masks.npy meta.json; do
    [[ -f "$SRC/$d/$f" ]] || { echo "MISSING: $SRC/$d/$f" >&2; exit 1; }
  done
done
[[ -d "$HF_HOME" ]] || { echo "MISSING: $HF_HOME (timm Swin weights)" >&2; exit 1; }

# Two array tasks can land on the same node and share $TMPDIR (observed: tasks 0
# and 1 both on d1022 with /tmp/nucseg_cache). Serialise staging behind a lock
# and gate on a completion marker written only after rsync finishes, so the
# second task waits and then reuses the copy instead of rewriting files the
# first task is already memory-mapping.
mkdir -p "$DEST"
exec 9>"$DEST/.stage.lock"
flock 9
if [[ -f "$DEST/.stage.complete" ]]; then
  echo "=== cache already staged at $DEST by another task on this node"
else
  echo "=== staging $SRC -> $DEST"
  time rsync -a --stats "$SRC/" "$DEST/"
  touch "$DEST/.stage.complete"
fi
flock -u 9
exec 9>&-
df -h "$DEST" | tail -1

# ── Work ─────────────────────────────────────────────────────────────────────
BUDGET=$((WALL_SECONDS - ($(date +%s) - T0)))
echo "=== time budget for runs: ${BUDGET}s"

# Extra arguments passed to sbatch are forwarded to the runner, so a subset can
# be released first:
#     sbatch slurm/sweep.sbatch --seeds 42          # reproduction gate
#     sbatch slurm/sweep.sbatch                     # everything still missing
python scripts/run_sweep.py \
  --worker \
  --run-root runs \
  --cache-root "$DEST" \
  --time-budget "$BUDGET" \
  --reserve 1200 \
  "$@" &
PY=$!
# Slurm sends USR1 300s before the wall clock; forward it so the runner
# checkpoints and exits cleanly rather than being killed mid-epoch.
trap 'echo "forwarding USR1 to $PY"; kill -USR1 "$PY" 2>/dev/null || true' USR1
wait "$PY" || RC=$?
RC=${RC:-0}

echo "=== task done rc=$RC after $(( ($(date +%s) - T0) / 60 )) min"
python scripts/run_sweep.py --list || true
exit "$RC"
