# Changes after review

Paper: *When Simple Wins: Lightweight CNN Encoders for Resource-Constrained
Nucleus Segmentation* — MICCAI 2026 Workshop MSB EMERGE

**No new experiments or results appear in the camera-ready.** Every number comes
from re-running the configurations that were reviewed, unchanged. Each
`(base, overrides)` configuration chain was verified to reproduce the config file
saved alongside the corresponding original run, field for field.

---

## 1. Multi-seed results and variance estimates
*(all three reviewers; Reviewer Ttru Q1; author response point 4)*

- Every configuration in Tables 1 and 2 re-run across **5 training seeds**
  (42–46, identical for every configuration); tables now report mean and standard
  deviation, n=5.
- Differences are judged against **paired 95% confidence intervals**. Because all
  configurations share the same five seeds, comparisons use per-seed differences.
  All 11 pairwise comparisons on test Dice have intervals excluding zero.
- Limitations rewritten: the single-run disclosure is replaced, and the scope of
  the new estimates is stated — they bound training-seed variance on one fixed
  split, not split variance.
- Limitations notes that at n=5 a two-sided signed-rank *p* cannot fall below
  0.0625, which is why intervals are reported rather than non-parametric tests.
- The re-runs were performed on a single NVIDIA Tesla V100-SXM2-32GB in fp32; the
  original results were produced on an Apple M3 via the PyTorch MPS backend.

## 2. Correction to the from-scratch Swin result
*(follows from point 1)*

- Table 1's from-scratch Swin-T row changes from **0.590 to 0.825 ± 0.001** Dice.
- Cause: the original single run diverged rather than converged. Its per-epoch
  validation Dice oscillated (0.278, 0.531, 0.003, 0.023, 0.585, 0.208, …) and the
  reported 0.590 was the peak of that oscillation at epoch 5. All five re-runs
  converge smoothly to ≈0.825.
- Consequence: **ImageNet pretraining shows no measurable benefit** for the Swin
  encoder (from-scratch 0.825 ± 0.001 vs pretrained 0.824 ± 0.001). The previously
  reported +0.212 Dice "lift" from pretraining is withdrawn.
- The VGG-over-Swin result is unchanged in direction and slightly smaller in
  magnitude: 0.030 Dice on PanNuke (previously 0.049) and 0.036 on MoNuSeg
  (previously 0.046).
- Section 4.1 and the Discussion revised accordingly.  **[PENDING — items A1/A2]**

## 3. Scope of the architectural attribution
*(all three reviewers; author response point 3)*

- The pretraining finding is scoped to **ImageNet** pretraining; the
  skip-connection resolution bottleneck is presented as the mechanism we believe
  responsible rather than one we have isolated.  **[PENDING — items A1/A2]**
- Future work now names **a resolution-matched Swin variant** (or a CNN with
  equally reduced skip resolution) and **pretraining matched to H&E rather than
  ImageNet** as the experiments that would settle the attribution.

## 4. Scope of claims
*(Reviewers Ttru and 1jr8; Area Chair; author response point 5)*

- The Conclusion's "strongest lightweight option" is withdrawn, replaced with
  "the strongest of the three encoders we compare".
- The same superlative in the abstract is scoped identically.
- The abstract's "at a fraction of the parameter count" now reads "at a fraction
  of **Swin-T's** parameter count" — the claim holds against Swin-T (34.0M vs
  86.8M) but not against MedT, which at 1.4M is smaller than our VGG.
- Section 4.2's "widening as data shrinks" is withdrawn, replaced with "the gap is
  comparable at both data scales".
- VGG and ResNet are no longer ranked. They differ by 0.002 Dice, reproducible
  across seeds but below the 0.005 Dice we treat as material.

## 5. MedT comparison
*(Reviewers K9Em and 1jr8; author response point 1)*

- The comparison's lack of control in training-set size and input protocol is
  stated **at the point of comparison** in Section 4.2, not only in Limitations,
  and no parity with MedT is claimed.

## 6. Learning rates, schedules and epoch limits
*(Reviewers Ttru Q3 and 1jr8; author response point 2)*

- Limitations now states that we did not sweep learning rates and therefore cannot
  quantify how sensitive the encoder comparison is to that choice.
- It notes the instability observed in from-scratch Swin at a learning rate of
  10⁻², which indicates the sensitivity is not negligible.
- It notes that pretrained Swin reached its 100-epoch cap in two of five seeds,
  so its schedule may bound it — although those two seeds scored no lower than
  the three that stopped on patience.

## 7. Figure 1 legibility
*(Reviewer K9Em)*

- Figure 1 re-rendered with the H&E image removed from behind the masks, which are
  now solid fills in a high-contrast colour that survives greyscale printing.
- Regenerated from the multi-seed checkpoints, using the same test slide
  (TCGA-2Z-A9J9-01A-01-TS1) as the original figure.

## 8. Claims about foundation models
*(Area Chair: "overbroad or unfair comparisons")*

- Removed from Section 4.5 the quantitative claim that pathology foundation models
  in the 300M–1B range "need about an order of magnitude more inference memory and
  are typically deployed via linear probing on cached embeddings". No foundation
  model is evaluated in this work.
- Removed from the Discussion the assertion that foundation models "are
  memory-hungry and, depending on hardware, may be unusable", which also
  duplicated a sentence in the Introduction. Replaced with our own measurement:
  peak training memory was 3.6 GiB for VGG and 5.4 GiB for pretrained Swin-T, the
  heaviest configuration we ran, so every configuration fits a single 8 GB
  consumer GPU.
- Foundation models are now referred to only as motivation, as cited background,
  and as future work.

## 9. Single measurement platform
*(consistency with point 1)*

- Table 3 (inference latency and peak memory) is re-measured on the same
  NVIDIA V100 used for the accuracy results, replacing the earlier Apple M3 / MPS
  measurement, so the whole paper reports one platform.
- Table 3's caption documents the protocol (median of 300 synchronised forward
  passes after 50 warmup iterations, batch size 1, fp32) and states that peak
  memory covers weights and activations.
- Section 4.5's ratios are corrected for the new platform: Swin uses 2.3× the
  memory and 2.6× the latency of VGG (previously 2.6× and 1.8×).

## 10. Presentation

- Author names, affiliation and contact details de-anonymized; the code link now
  points to the public repository rather than the anonymized mirror.
- Sentence-level tightening in response to Reviewer K9Em's note on insertions and
  long sentences: several em-dash asides were promoted into main clauses or split
  into separate sentences, and one non-sequitur parenthetical was removed from
  Section 3.1.

---

*Items marked **[PENDING]** are not yet complete in this draft and must be
finished before submission.*
