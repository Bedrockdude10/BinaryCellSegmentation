# Multi-seed re-run of Tables 1 and 2

> **Status: complete.** 35/35 runs banked, 50 h 23 min total GPU time, zero
> failures, one GPU model (`Tesla V100-SXM2-32GB`), one split hash per dataset.
> Full output in `runs/summary.txt`; the seed-42 reproduction check is in
> `runs/repro_seed42.txt`; Table 3 is in `runs/_bench/`.
>
> **Four published claims do not survive.** See [Findings](#findings-what-the-5-seed-numbers-change).

Reviewer-requested variance estimates for the MICCAI 2026 MSB EMERGE
camera-ready. Every configuration in Tables 1 and 2 is re-run across **5
training seeds** and reported as mean ± standard deviation.

**The experiment itself is unchanged.** Model architectures, optimizer, LR
schedules, augmentation policy, AMP setting and early-stopping criteria are
identical to the reviewed submission. Each `(base, overrides)` chain in
`configs/sweep.yaml` was verified to rebuild the `config.yaml` that was saved
next to the corresponding published run, field for field. Only `train_seed`
varies, plus data *plumbing* (caching, worker counts) that does not touch
arithmetic.

## The matrix — 7 configurations x 5 seeds = 35 runs

| id | table | dataset | encoder | loss | published Dice |
|---|---|---|---|---|---|
| `pannuke_swin_scratch` | 1 | PanNuke | Swin-T (scratch) | BCE | 0.590 |
| `pannuke_swin_pretrained` | 1 | PanNuke | Swin-T (pretrained) | BCE | 0.802 |
| `pannuke_resnet_bce` | 1 | PanNuke | ResNet | BCE | 0.849 |
| `pannuke_vgg_dice` | 1 | PanNuke | VGG | Dice | 0.851 |
| `pannuke_vgg_bce` | 1 | PanNuke | VGG | BCE | 0.851 |
| `monuseg_vgg_bce` | 2 | MoNuSeg | VGG | BCE | 0.796 |
| `monuseg_swin_pretrained` | 2 | MoNuSeg | Swin-T (pretrained) | BCE | 0.750 |

Not runs: Table 1's **Otsu** row has no parameters and no RNG (one
deterministic number, carried over); Table 2's **MedT** row is a published
figure from another paper.

## Seeds

```yaml
train_seeds: [42, 43, 44, 45, 46]   # identical for every config -> paired comparisons
split_seed: 42                      # FIXED for the whole study, never varied
```

`split_seed` and `train_seed` are separate on purpose. `split_seed` selects which
MoNuSeg images are held out for validation; MoNuSeg has only 37 training images,
so resampling the split per seed would fold split variance into what the paper
reports as training-seed variance. PanNuke uses the dataset's predefined folds,
so no split RNG applies there.

Seed 42 is first because it is the seed the published single-run numbers used, so
it doubles as the reproduction check.

Every run recomputes a hash over the split's file lists and asserts it against
`configs/split_manifest.json` before training starts. A drifted split fails on
line one instead of quietly contributing a non-comparable number to a table.

## What randomness is seeded, and where

`src/seeding.py` is the single source of truth. `set_seed(train_seed)` covers
python `random`, `numpy`, `torch` and all CUDA devices. On top of that:

- the train DataLoader gets an explicit `generator=`, so shuffle order does not
  depend on how many draws model construction happened to consume;
- every worker process runs `worker_init_fn=seed_worker`, which reseeds `numpy`
  and `random` inside the worker.

### What was actually broken (measured, not assumed)

`PreprocessedDataset` does its flips and rotations with `np.random`, and the
reviewed code only ever called `torch.manual_seed(seed)`. The commonly-cited
consequence — "numpy in DataLoader workers is unseeded" — **did not apply**:
PyTorch reseeds `numpy` inside each worker from a `base_seed` it draws when the
iterator is created, so with `num_workers: 4` the published runs' augmentation
was already deterministic. `scripts/check_seeding.py` check 6 confirms this.

Two real defects remained, both now fixed and both covered by that script:

- **The stream was coupled to unrelated global-RNG consumption** (check 5).
  `base_seed` came from the *global* torch RNG because no `generator=` was
  passed, so the augmentation and shuffle streams depended on how many draws
  earlier code had taken. Measured: constructing the model before rather than
  after the loaders changes every batch (`6a7a6d25…` vs `b8224539…`). An
  explicit `generator=` makes the stream a function of `train_seed` alone.
- **`num_workers = 0` was genuinely unseeded** (check 4). Nothing seeds numpy in
  the main process, so that path was not reproducible at all. Measured: two
  identically-configured passes gave `cdf13079…` and `cd53f855…`. The sweep
  configs all use workers > 0, so the published runs avoided this, but it made
  `seed` a partial guarantee rather than a real one.

`random` was never seeded either, and neither was `numpy` in the main process.
`set_seed` now covers both.

## Hardware pinning

The Explorer `gpu` partition is heterogeneous: `v100-sxm2`, `v100-pcie`, `t4`,
`a100`, `h200`. Every job requests `--gres=gpu:v100-sxm2:1`, and every
`results.json` records the GPU model. `scripts/aggregate.py` hard-errors if the
runs it is asked to average did not all land on the same GPU model — otherwise
hardware variance would be reported as seed variance, and Table 3's latency
numbers would not be readable alongside Tables 1 and 2.

## Running it

```bash
# once: build the flat patch caches and verify they match the original pipeline
python scripts/prepare_cache.py

# once: record the splits (commit the result)
python scripts/make_split_manifest.py

# sanity: seeding guarantees hold
python scripts/check_seeding.py

# smoke test, all configs, 3 epochs, into a throwaway run root
python scripts/run_sweep.py --run-root runs_smoke --seeds 42 --max-epochs 3 --local
```

On the cluster (Explorer). One-time setup, then launch:

```bash
./slurm/sync_to_explorer.sh --data --weights   # code + 2.3 GB caches + Swin weights
```

```bash
ssh $EXPLORER_HOST 'cd ~/BinaryCellSegmentation && sbatch slurm/preflight.sbatch'
```

The preflight is a single-task pilot: it validates the exact allocation path,
offline weight loading, staging, the split assertions and the results artifact,
and measures the peak memory of every configuration — for the cost of one task.
Then release seed 42 first (the reproduction gate), and only then the rest:

```bash
ssh $EXPLORER_HOST 'cd ~/BinaryCellSegmentation && sbatch slurm/sweep.sbatch --seeds 42'
```

```bash
ssh $EXPLORER_HOST 'cd ~/BinaryCellSegmentation && sbatch slurm/sweep.sbatch'
```

Resubmit that last one until `--list` reports 35/35, or let the driver do it:

```bash
./slurm/drive_sweep.sh
```

It submits, waits for the array to drain, and resubmits while runs remain —
stopping if two consecutive rounds bank nothing new, which means something is
failing rather than just running out of wall clock. Progress goes to
`slurm/drive_sweep.log`.

**Do not push code between launches** — see the warning in
`slurm/sync_to_explorer.sh`.

Local sequential fallback, no Slurm:

```bash
python scripts/run_sweep.py --local
```

Aggregation runs locally off the pulled artifacts (`src.sweep` is deliberately
torch-free so this works anywhere, including a login node):

```bash
./slurm/sync_to_explorer.sh --pull && python scripts/aggregate.py --json runs/summary.json
```

Table 3 is benchmarked separately, on an idle exclusive node, never while sweep
tasks of yours are running:

```bash
ssh $EXPLORER_HOST 'cd ~/BinaryCellSegmentation && sbatch slurm/bench.sbatch'
```

### Slurm sizing

`slurm/sweep.sbatch` is `--array=0-7%4`, not one task per run, because the `gpu`
QOS allows 8 submitted and 4 running array *tasks* per user and rejects larger
arrays at submit time. Tasks claim runs dynamically (atomic `mkdir` under
`runs/.claims/`, with a heartbeat so a dead task's claim is reclaimed), so 8
tasks share 35 runs without one task inheriting all the slow configurations.

The partition's 8-hour cap is shorter than the whole sweep, so a task stops
starting new epochs when its remaining budget looks too small, checkpoints, and
exits 0. **Resubmitting the identical script continues the sweep**: a run with a
complete `results.json` is skipped outright, and a part-trained run resumes from
`last.pt`.

## Resumability

Two levels:

1. **Completed run** — `results.json` with `status: complete` is skipped without
   even building a model.
2. **Interrupted run** — `last.pt` holds model, optimizer, LR scheduler, epoch
   counter, early-stopping staleness counter, metric history **and RNG state**
   (python, numpy, torch, CUDA, and the DataLoader generator). Resuming without
   RNG state would silently break the seeding guarantee, so it is checkpointed
   with everything else. `last.pt` is written every 5 epochs, on early stop, and
   on the `SIGUSR1` Slurm sends before the wall clock; it is deleted once the run
   is banked.

`scripts/check_resume.py` proves this rather than asserting it: it trains one
config straight through, trains it again with an interruption after epoch 2, and
compares. Verified result (MoNuSeg VGG, 4 epochs):

```
epoch    A train loss   B train loss     A val dice   B val dice  match
    1    0.3816447745   0.3816447745    0.543804228  0.543804228  =
    2    0.3423354104   0.3423354104    0.661755443  0.661755443  =
    3    0.3092822894   0.3092822894    0.666415453  0.666415453  =   <- resumed here
    4    0.2924479236   0.2924479236    0.639347672  0.639347672  =
test metrics  A: 0.7180309083  B: 0.7180309083
```

Running the same test with `--persistent-workers` diverges at exactly epoch 3
(val Dice 0.666 vs 0.649, final test 0.662 vs 0.675), which is why
`persistent_workers` is off — see the note in `configs/sweep.yaml`. A resumed run
is therefore the same experiment as an uninterrupted one, so whether a task
happened to get preempted is not a hidden variable in the reported variance.

## Per-run artifact

One `runs/<config>/seed<N>/results.json` per run, containing: git SHA and dirty
flag, the fully resolved config, split hash and summary, `train_seed`,
`split_seed`, per-epoch validation Dice/IoU and train loss, epochs run,
early-stop flag, best epoch, final test metrics with the protocol used, total
and trainable parameter counts, wall-clock duration, peak GPU memory, GPU model
and compute capability, node ID, Slurm job/array IDs, and python/torch/CUDA/
cuDNN/numpy/timm versions.

## Performance changes, and whether they touch numerics

**Numerics-neutral** (kept):

| change | why it cannot change results |
|---|---|
| flat memmap patch caches (`scripts/prepare_cache.py`) | Storage layout only. PanNuke's cached mask is `binary_mask()` applied to exactly the 5-channel mask the on-the-fly loader built; MoNuSeg's is a byte copy of the per-patch `.npy` files. `--verify` asserts elementwise equality against the original pipeline. |
| staging caches to node-local NVMe (`$TMPDIR`) | Same bytes, different filesystem. |
| `num_workers`, `prefetch_factor`, `persistent_workers` | Change when work happens, not what is computed. |
| `pin_memory` | Affects host-to-device transfer mechanics only. |
| dropping periodic `epoch_N.pt` snapshots | Artifacts, not computation — and 35 runs of them would not fit in the 75 GB home quota. |

**Changed on purpose — the one numerics-altering setting: `performance.amp: false`.**

The reviewed code ran `torch.autocast(dtype=float16)` on any non-CPU device with
no `GradScaler`. On the Apple M3 / MPS backend the published numbers came from,
that autocast was inert. On CUDA it is fully active, and without a scaler it
zeroes **92–98% of the ResNet encoder's gradients**: the loss never leaves ~0.25
and the run lands at Dice **0.667 against 0.849 published**. VGG survives it
(0.2% zero gradients), which is why only one row of Table 1 looked wrong.

Measured by `scripts/diagnose_amp.py` on a V100, 4 epochs of ResNet on PanNuke:

| regime | loss, epoch 1 → 4 | zero-grad | s/epoch |
|---|---|---|---|
| `amp_fp16` (reviewed code on CUDA) | 0.2695 → 0.2541 (stalled) | 92–98% | 60 |
| `amp_fp16` + `GradScaler` | 0.2565 → 0.1680 | 0.0% | 62 |
| **`fp32`** (what the sweep uses) | 0.2565 → 0.1682 | 0.0% | 96 |
| VGG `amp_fp16` (control) | 0.2603 → 0.1711 | 0.2% | 45 |

The published MPS ResNet trajectory was 0.2556 / 0.2372 / 0.2007 / 0.1682 —
matching fp32 to four decimals. fp32 is therefore the regime the reviewed results
were actually produced in, which is why the sweep uses it rather than adding a
`GradScaler` (free, same trajectories, but a component the reviewed code lacked).
Cost: ~1.6x wall clock, ~46 GPU-hours for the sweep.

All 35 runs sit in this one regime. The seven seed-42 runs completed under fp16
before this was found are in `discarded_amp_fp16_regime/` as evidence, not in
`runs/`.

**Not changed, deliberately:**

- `cudnn.benchmark` stays `False` (PyTorch's default). Turning it on would let
  cuDNN autotune per-shape algorithm choice, changing floating-point reduction
  order. Input shapes here are fixed, so it would likely be a modest win — but
  it is a numerics change, so it is not made unilaterally.
- `channels_last`, augmentation order, optimizer, LR schedule, early stopping.

## Reproduction gate

Before the remaining 28 runs are released, the 7 seed-42 runs are compared
against the published single-seed numbers:

```bash
python scripts/check_repro.py --json runs/repro_seed42.json
```

Exact reproduction is not expected, for two specific reasons:

1. **Backend and AMP regime.** The reviewed code enables
   `torch.autocast(dtype=float16)` for any non-CPU device. The published runs
   were on Apple M3 / MPS, where that was at best partially effective; on the
   V100 fp16 autocast is fully active. Different kernels, different reduction
   orders, genuinely different arithmetic.
2. **The augmentation stream necessarily changed.** Giving the DataLoader an
   explicit `generator=` was a required correctness fix, and it changes which
   augmentations each sample receives. Seed 42 is therefore a *fresh draw* from
   the seed distribution, not a replay of the published trajectory.

Tolerance: **±0.015 Dice** per configuration. The paper itself states that
differences below ~0.005 Dice are not meaningful run-to-run noise here; 0.015 is
three times that — loose enough to absorb a backend change and a different seed
draw, still 3x smaller than the 0.049 Dice VGG-over-Swin gap the central claim
rests on.

Because a tolerance can pass while the argument breaks, the script also asserts
the structural claims directly: VGG and ResNet exceed pretrained Swin by ≥0.030
and ≥0.020 Dice on PanNuke, VGG-vs-ResNet and BCE-vs-Dice agree within 0.010,
pretraining lifts Swin by ≥0.150, VGG exceeds Swin by ≥0.030 on MoNuSeg, and
from-scratch Swin still clears the Otsu floor.

### Gate result (seed 42, fp32, Tesla V100-SXM2-32GB)

| config | published | re-run | Δ Dice | within ±0.015 |
|---|---|---|---|---|
| VGG / BCE | 0.8513 | 0.8544 | +0.0030 | yes |
| VGG / Dice | 0.8506 | 0.8568 | +0.0062 | yes |
| ResNet | 0.8486 | 0.8524 | +0.0038 | yes |
| MoNuSeg VGG | 0.7959 | 0.8045 | +0.0086 | yes |
| MoNuSeg Swin-pretrained | 0.7499 | 0.7486 | −0.0013 | yes |
| Swin-pretrained | 0.8017 | 0.8241 | +0.0224 | no |
| Swin from-scratch | 0.5903 | 0.8254 | **+0.2351** | no |

**The from-scratch Swin gap is a property of the experiment, not of this
harness.** `configs/ablations/swin.yaml` overrides only the architecture, so
from-scratch Swin inherits **lr = 0.01 SGD** from `default.yaml` — very high for
a from-scratch transformer, and right on a stability boundary. The published run
fell off the stable side: its per-epoch validation Dice was
`0.278, 0.531, 0.003, 0.023, 0.585, 0.208, 0.523, 0.336, …` with non-monotonic
training loss, and the reported 0.5903 is the peak of that oscillation at epoch 5
before early stopping at 20. The re-run, with a byte-identical config and only a
different data order, climbs monotonically to 0.827 at epoch 61.

Pretrained Swin (lr 1e-4) is stable in both; its +0.022 is ordinary seed and
hardware variance with the same trajectory shape.

This is the single-seed weakness the reviewers flagged, appearing in the paper's
own results. The 5-seed numbers determine whether from-scratch Swin is genuinely
~0.82 or simply high-variance; either way the multi-seed table is the honest
answer, and `pannuke_swin_pretrained - pannuke_swin_scratch >= 0.150` stays in
the structural checks so the change cannot pass unnoticed.

## Aggregation

```bash
python scripts/aggregate.py --json runs/summary.json
```

Emits mean and sample standard deviation (ddof=1) with n for every metric in
Tables 1 and 2, mean epochs-to-early-stop and mean best epoch per config, per-run
and total wall clock, the published-vs-mean delta in units of SD, and
paste-ready LaTeX bodies matching the existing `@{}llcccr@{}` and `@{}lccc@{}`
column structures.

No p-values and no confidence intervals: n=5 does not support them.

It flags any pairwise comparison whose difference in means is smaller than the
pooled standard deviation, and hard-errors on: a config with fewer than 5
completed runs, runs spanning more than one GPU model, or runs on one dataset
with more than one split hash. `--allow-partial` downgrades the completeness
error to a warning for mid-sweep progress checks; its output is not reportable.

## Table 3

`scripts/benchmark_table3.py`, run through `slurm/bench.sbatch` on an
exclusively allocated node, never alongside training. Batch size 1, 3x256x256,
`eval()`/`no_grad`, fp32, 50 discarded warmup iterations then 300 timed ones
with device synchronisation around the clock and per-iteration times retained.
The script aborts if another compute process is resident on the GPU.

Note that the published Table 3 numbers were measured on an Apple M3 via MPS.
Re-measuring on the V100 is required for Table 3 to be comparable with the
re-run Tables 1 and 2, and changes the paper's stated testbed.

## Findings: what the 5-seed numbers change

All numbers mean ± sample SD over `train_seeds: [42, 43, 44, 45, 46]`, n=5.

| config | published | 5-seed mean ± SD |
|---|---|---|
| Swin-T (scratch), PanNuke | 0.5903 | **0.8255 ± 0.0011** |
| Swin-T (pretrained), PanNuke | 0.8017 | 0.8244 ± 0.0009 |
| ResNet, PanNuke | 0.8486 | 0.8526 ± 0.0003 |
| VGG / Dice, PanNuke | 0.8506 | **0.8561 ± 0.0007** |
| VGG / BCE, PanNuke | 0.8513 | 0.8544 ± 0.0003 |
| VGG, MoNuSeg | 0.7959 | 0.7939 ± 0.0113 |
| Swin-T (pretrained), MoNuSeg | 0.7499 | 0.7578 ± 0.0067 |

**1. ImageNet pretraining does nothing for Swin on PanNuke.** From-scratch
0.8255 ± 0.0011 vs pretrained 0.8244 ± 0.0009 — a 0.0011 difference against a
pooled SD of 0.0010 (ratio 1.1), i.e. indistinguishable. The published "+0.212
Dice lift" was an artifact of a single from-scratch run that diverged; all five
re-runs converge to ~0.825 with a very small SD. §4.1's "From-scratch Swin
barely exceeds Otsu … confirming transformers lack sufficient inductive bias at
this scale", the +0.212 figure, and the Discussion's "role of pretraining"
paragraph all rest on that one run.

**2. The VGG-over-Swin claim holds, at a smaller margin.** VGG/BCE 0.8544 vs
pretrained Swin 0.8244 = **+0.030 Dice** (ratio 47.7), not the published 0.049.
The abstract's "4.9 Dice points" becomes ~3.0. On MoNuSeg, VGG 0.7939 vs Swin
0.7578 = +0.036 (ratio 3.9), against the published 0.046.

**3. "VGG ≈ ResNet within noise" and "BCE ≈ Dice" are no longer defensible as
stated — in the opposite direction.** PanNuke seed variance is far tighter than
the paper assumes: SD ≈ 0.0003. So VGG/BCE vs ResNet (Δ0.0018, ratio 6.0) and
VGG/Dice vs VGG/BCE (Δ0.0017, ratio 3.1) are both reliably non-zero, while being
far too small to matter practically. §4.1's "differences below ~0.005 Dice are
not meaningful" is now an overclaim in the other direction. Note also that
**VGG/Dice (0.8561) edges out VGG/BCE (0.8544)**, so the bold cells in Table 1
move.

**4. Pretrained Swin's 100-epoch cap was costing it ~0.005 Dice — measured, not
guessed.** In the main sweep it ran `[88, 100, 95, 100, 98]` epochs against the
100 cap, with seed 45's best epoch AT the cap. A supplementary 5-seed sweep with
the cosine schedule extended to 150 epochs (`configs/sweep_supp.yaml`,
`runs_supp/`, results in `runs_supp/summary.txt`) settles it:

| seed | e100 Dice | ep | stopped by | e150 Dice | ep | stopped by | Δ |
|---|---|---|---|---|---|---|---|
| 42 | 0.8241 | 88 | patience | 0.8300 | 119 | patience | +0.0059 |
| 43 | 0.8243 | 100 | cap | 0.8300 | 135 | patience | +0.0057 |
| 44 | 0.8246 | 95 | patience | 0.8296 | 113 | patience | +0.0049 |
| 45 | 0.8256 | 100 | cap | 0.8302 | 115 | patience | +0.0046 |
| 46 | 0.8233 | 98 | patience | 0.8289 | 118 | patience | +0.0056 |

e100 0.8244 ± 0.0009 → e150 **0.8297 ± 0.0005**; paired Δ **+0.0053 ± 0.0006**,
same sign in 5/5 seeds. All five e150 runs stopped on **patience** at 113-135
epochs, well inside the new cap, so pretrained Swin is now genuinely converged.

Against the e150 number, VGG/BCE (0.8544 ± 0.0003) still leads by **+0.0247**
(pooled SD 0.0004, ratio 59.7). The conclusion is unchanged; the margin moves
from ~3.0 to ~2.5 Dice points.

**Two caveats on how to phrase this.** (a) `build_scheduler` sets
`CosineAnnealingLR(T_max=cfg["training"]["epochs"])`, so raising the cap also
stretches the LR curve. The e150 runs were already ~0.003 ahead of their e100
counterparts *at epoch 100*, before any extra budget existed — so the gain is
mostly the gentler LR decay, not the extra epochs, and this experiment cannot
separate the two. Say "with a longer cosine schedule", not "trained longer".
(b) Swin gains from a gentler decay while VGG (StepLR, converging at ~52 epochs,
cap never in play) has no equivalent headroom. Both schedules are the reviewed
ones and neither was altered for Table 1, but "VGG beats Swin under each
architecture's published recipe" is more defensible than "VGG beats Swin, period."

These supplementary numbers are NOT in Table 1 and cannot be: the LR schedule
differs from the reviewed one. `pannuke_vgg_dice` (1/5 seeds) and
`monuseg_swin_pretrained` (1/5) also touched their caps and were not re-run.

**Mean epochs-to-early-stop** (for the rebuttal's epoch-count argument, which
needs means rather than one run): Swin-scratch 69.0 ± 9.4, Swin-pretrained
96.2 ± 5.0, ResNet 49.8 ± 3.8, VGG/Dice 84.8 ± 17.5, VGG/BCE 51.8 ± 2.6,
MoNuSeg VGG 55.4 ± 10.5, MoNuSeg Swin 121.8 ± 17.6.

**MedT anchor survives.** VGG on MoNuSeg is 0.7939 ± 0.0113 against MedT's
published 0.796 — within one SD, so "VGG reaches MedT-level Dice" still reads
correctly.

## Table 3 on the V100

Measured on the same GPU model as Tables 1 and 2 (`runs/_bench/`), batch 1,
3×256×256, fp32, 50 warmup + 300 timed iterations, device-synchronised per
iteration:

| model | params | latency (ms) | peak memory (MB) | published (M3/MPS) |
|---|---|---|---|---|
| VGG U-Net | 34.0M | 7.1 (p05 7.07 / p95 7.12) | 264.9 | 25.2 ms / 130.5 MB |
| ResNet U-Net | 37.5M | 9.6 (p05 9.60 / p95 9.66) | 277.2 | 34.2 ms / 144.0 MB |
| Swin-T U-Net (pretrained) | 86.8M | 18.4 (p05 18.34 / p95 18.50) | 618.8 | 46.6 ms / 333.6 MB |

The Pareto conclusion is unchanged — VGG is fastest and smallest — but two
caveats: the *ratios* differ (Swin is 2.6× VGG latency on the V100 vs 1.85× on
the M3), and the memory column is **not** directly comparable to the published
one, because CUDA reports `max_memory_allocated` over a forward pass (weights +
activations) while the original MPS numbers used `current_allocated_memory`.
Reporting these numbers means the paper's stated testbed changes from "Apple M3
with the PyTorch MPS backend" to the V100.
