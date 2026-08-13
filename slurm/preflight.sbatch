#!/bin/bash
# slurm/preflight.sbatch — single-task pilot before releasing the full sweep.
#
#   sbatch slurm/preflight.sbatch
#
# Validates, for the cost of one task, everything the 35-run sweep depends on:
# the exact allocation path (same partition/gres/qos as slurm/sweep.sbatch),
# offline pretrained-weight loading, staging, the split assertions, the results
# artifact, and the peak-memory footprint of every configuration — including the
# heaviest, pretrained Swin on PanNuke — so the sweep's --mem request is measured
# rather than guessed.
#
# Writes to runs_smoke/, NOT runs/, and caps epochs at 3. These numbers are
# plumbing checks and are not reportable.

#SBATCH --job-name=nucseg-preflight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/preflight_%j.out
#SBATCH --error=slurm_logs/preflight_%j.out

set -euo pipefail
T0=$(date +%s)
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p slurm_logs runs_smoke

echo "=== preflight on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

module load miniconda3/25.9.1 cuda/12.8.0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SRC="$HOME/nucseg_data"
DEST="${TMPDIR:-/tmp}/nucseg_cache"

echo "=== 1/5 environment"
python - <<'PY'
import torch, numpy, sys
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version(),
      "| numpy", numpy.__version__)
assert torch.cuda.is_available(), "no CUDA on a --gres=gpu node"
print("gpu:", torch.cuda.get_device_name(0),
      "cc", ".".join(map(str, torch.cuda.get_device_capability(0))),
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
import timm
print("timm", timm.__version__)
PY

echo "=== 2/5 pretrained Swin loads OFFLINE (HF_HUB_OFFLINE=1)"
python - <<'PY'
from src.models.encoders import SwinEncoder
enc = SwinEncoder(pretrained=True)
print("swin stage channels:", enc.stage_channels)
PY

echo "=== 3/5 stage caches to node-local NVMe"
for d in pannuke_flat_fold1 pannuke_flat_fold2 pannuke_flat_fold3 \
         monuseg_flat_train_256 monuseg_flat_test_256; do
  for f in images.npy masks.npy meta.json; do
    [[ -f "$SRC/$d/$f" ]] || { echo "MISSING: $SRC/$d/$f" >&2; exit 1; }
  done
done
mkdir -p "$DEST"
time rsync -a "$SRC/" "$DEST/"
du -sh "$DEST"; df -h "$DEST" | tail -1

echo "=== 4/5 splits match the committed manifest"
python scripts/make_split_manifest.py --check --cache-root "$DEST"

echo "=== 5/5 smoke: all 7 configs, seed 42, 3 epochs"
python scripts/run_sweep.py \
  --local \
  --run-root runs_smoke \
  --cache-root "$DEST" \
  --seeds 42 \
  --max-epochs 3

echo "=== peak GPU memory per config (drives the sweep's --mem request)"
python - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("runs_smoke").glob("*/seed42/results.json")):
    r = json.loads(p.read_text())
    print(f"{r['run_id']:34s} {r['peak_gpu_memory_bytes'] / 2**30:5.2f} GiB  "
          f"{r['wall_clock_seconds'] / r['epochs_run']:6.1f} s/epoch  "
          f"{r['params_total'] / 1e6:5.1f}M params  gpu={r['gpu_model']}")
PY

echo "=== preflight done in $(( ($(date +%s) - T0) / 60 )) min"
