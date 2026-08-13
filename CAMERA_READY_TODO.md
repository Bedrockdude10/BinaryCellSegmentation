# Authorial revisions still open

Seven passages in `WhenSimpleWins.tex` need rewriting rather than number
substitution. Each is marked in place with a `TODO(camera-ready)` or
`TODO(authorial …)` comment carrying the measured facts, so you can work through
them in the file. Find them all with:

```bash
grep -n "TODO(camera-ready)\|TODO(authorial" WhenSimpleWins.tex
```

Everything mechanical is already merged to `main`: Table 1 and 2 bodies, the
abstract and results numbers, the Testbed sentence, and Figure 1's path. All
figures are regenerated from seed-42 checkpoints.

Supporting numbers: `runs/summary.txt` (per-config mean ± SD, 95% CIs, paired
comparisons), `runs_supp/summary.txt` (the 150-epoch Swin check),
`runs/repro_seed42.txt`, and `SWEEP.md` for method and findings.

---

## A1 — §4.1, from-scratch Swin and pretraining *(line ~146)*

**Blocks:** the paragraph currently asserts pretraining lifts Swin by 0.212 and
that from-scratch Swin "barely exceeds Otsu".

Measured at n=5: from-scratch **0.8255 ± 0.0011** vs pretrained
**0.8244 ± 0.0009**. Paired difference **+0.0011, 95% CI [+0.0006, +0.0016]**,
in favour of *from-scratch*. ImageNet pretraining yields no benefit.

The published 0.5903 came from one from-scratch run that **diverged** — its
per-epoch validation Dice went 0.278 / 0.531 / 0.003 / 0.023 / 0.585 / 0.208 …
and 0.5903 is the peak of that oscillation at epoch 5. All five re-runs converge
smoothly to ~0.825. From-scratch Swin inherits lr = 0.01 SGD from
`configs/default.yaml`, which is on a stability boundary for a from-scratch
transformer.

This **strengthens** your "not a matter of weights" reading: if ImageNet weights
change nothing measurable, the residual ~0.03 gap to VGG cannot be about those
weights. Do **not** write "pretraining significantly hurts" — 0.001 Dice is
immaterial regardless of the p-value.

**Reviewer link:** rebuttal point 3 said "ImageNet initialization recovers 0.212
Dice". That premise is gone.

## A2 — Discussion, "The role of pretraining" *(line ~257)*

Same finding, second location. Currently "$+$0.212 Dice" and "the final 0.049 to
VGG". The gap is **0.030** (paired, 95% CI [0.029, 0.031]) and the lift is ~0.
The skip-connection mechanism is unaffected and now better supported.

## A3 — Limitations *(line ~266)*

"All results are single runs without variance estimates" is now false, and it is
the reviewers' central objection (all three raised it).

Replacement facts: every configuration in Tables 1 and 2 re-run over **5 training
seeds** on one GPU model with one fixed split; mean ± SD reported; comparisons
paired per seed with 95% CIs; **all 11 paired CIs on test Dice exclude zero**.

Honest limitations that remain, and are worth stating:

* this measures **training-seed variance only**, on a single fixed split, so it
  says nothing about split or dataset variance;
* n=5 does not support Wilcoxon signed-rank — its two-sided floor is
  **p = 0.0625** at n=5, and it sits at that floor for all 11 comparisons;
* four differences are reproducible but below the 0.005 Dice materiality
  threshold §4.1 itself sets, so they are reported as measurable-but-immaterial
  rather than as rankings.

**Reviewer link:** rebuttal point 4 promised "VGG and ResNet will be described as
indistinguishable". At n=5 they are *distinguishable* — paired difference 0.0018,
95% CI [0.0012, 0.0023], p_Holm = 2.9e-3. Writing "indistinguishable" would
misreport your own data. "Differ by 0.002 Dice, reliably measurable but far too
small to matter; we do not rank them" keeps the promise and stays accurate. The
other half of that promise — withdrawing "widening as data shrinks" — **is**
supported: PanNuke gap 0.030, MoNuSeg 0.036, and MoNuSeg's pooled SD is 0.0093,
so the gaps are not resolvably different.

## A4 — Table 3 and the Testbed sentence *(line ~130)*

Table 3 still holds the **Apple M3** numbers (25.2 / 34.2 / 46.6 ms;
130.5 / 144.0 / 333.6 MB). The V100 measurement is in `runs/_bench/` at
7.1 / 9.6 / 18.4 ms and 264.9 / 277.2 / 618.8 MB.

Arguments for keeping M3: it is the device your deployment framing invokes, it is
what rebuttal point 5 quoted to reviewers, and Table 3's internal comparability
only needs *one* device, which the M3 already was. Arguments for V100:
consistency with the training hardware.

Note the memory columns are **not** comparable between the two: CUDA reports
`max_memory_allocated` over a forward pass (weights + activations) while the MPS
numbers used `current_allocated_memory`.

If you switch to V100, update the Testbed sentence too, and consider adding
"measured on …" to Table 3's caption either way.

## A5 — Abstract, "the strongest option in that space" *(line ~33)*

The abstract twin of the Conclusion phrase you withdrew in rebuttal point 5.
R-Ttru and R-1jr8 both objected that three encoders is too narrow a basis, with
MobileNet / EfficientNet / ConvNeXt-Tiny untested. Scope to what was measured.

## A6 — Abstract, "at a fraction of the parameter count" *(line ~33)*

Holds against Swin-T (34.0M vs 86.8M) but not against MedT, which is **1.4M** —
smaller than VGG. Rebuttal point 5 committed to correcting this.

## A7 — Figure 1 caption *(line ~205)*

The caption claims VGG "delineates nuclear boundaries more tightly". On the
regenerated seed-42 figure the annotated values are VGG **0.793** vs Swin
**0.783** — a 0.010 gap against Table 2's 0.036 mean, so this region
under-represents the effect.

Three renders are available in `figures/`:

| file | what it is |
|---|---|
| `..._seed42_solid.png` | **currently used.** Solid masks on white, same column semantics, no caption change needed |
| `..._seed42_errormap.png` | TP/FP/FN per pixel. Supports the boundary claim better; needs the column description rewritten |
| `..._seed42_rerun.png` | faithful reproduction of the published style, including the low-contrast overlay K9Em objected to |

If you pick a different slide because it shows the effect more clearly, say so in
the caption — otherwise it is cherry-picking.

---

## Not covered by any of the above

* **R-Ttru Q2 / R-1jr8** asked for a resolution-matched Swin or a CNN with
  reduced skip resolution. Not run — it is a new experiment, and rebuttal point 3
  named it as future work.
* **R-Ttru Q4 / R-1jr8** asked for compact CNN baselines. Not run; rebuttal
  point 5 scopes the claim instead.
* **K9Em** noted the writing has "many insertions and long sentences". Not
  addressed in the rebuttal and not a numbers question.
* The abstract says PanNuke has "~5K training patches"; fold1 is 2656 (5179
  counting fold2 as validation). Pre-existing, unrelated to the re-run.
