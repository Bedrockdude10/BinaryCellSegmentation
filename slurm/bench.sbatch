#!/bin/bash
# slurm/bench.sbatch — Table 3 latency/memory, on an exclusively held GPU node.
#
#   sbatch slurm/bench.sbatch
#
# Run this when no sweep tasks of yours are running (`squeue -u $USER`). Latency
# measured next to a training job is not a number worth printing, and the
# benchmark script itself aborts if another compute process is on the device.
#
# NOT --exclusive: that requests every GPU on the node and this account's QOS
# rejects it (QOSMaxGRESPerJob). `--gres=gpu:v100-sxm2:1` still gives EXCLUSIVE
# USE OF THAT GPU — Slurm does not share an allocated GPU between jobs — which is
# what matters for a GPU latency measurement. Other jobs may share the node's
# CPUs, so the script synchronises the device around every single iteration and
# reports median with p05/p95 rather than a mean, making any host-side jitter
# visible instead of silently averaged in. It also aborts outright if another
# compute process appears on the GPU.
#
# Same GPU model as the sweep, otherwise Table 3 cannot be read alongside
# Tables 1 and 2.

#SBATCH --job-name=nucseg-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/bench_%j.out
#SBATCH --error=slurm_logs/bench_%j.out

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
mkdir -p slurm_logs runs/_bench

echo "=== bench on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

module load miniconda3/25.9.1 cuda/12.8.0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/hf_cache"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1

python scripts/benchmark_table3.py \
  --device cuda \
  --batch-size 1 \
  --warmup 50 \
  --trials 300 \
  --out "runs/_bench/table3_${SLURM_JOB_ID:-local}.json"
