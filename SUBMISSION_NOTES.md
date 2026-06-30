# MICCAI 2026 Submission — Progress & Decisions

Tracking file for converting `final_report.tex` (NeurIPS 2024 style) into a
MICCAI 2026 / EMERGE–compliant submission.

**New source file:** `WhenSimpleWins.tex` → compiles to `WhenSimpleWins.pdf`.
**Original is untouched:** `final_report.tex` / `final_report.pdf` were not modified.

## Change log — four content additions (EMERGE deadline pass)
Added (per request): (1) controlled-design framing sentence before Contributions
(line ~48); (2) noise-magnitude sentence in §4.1 PanNuke (line ~136); (3) new
`\subsection{Limitations}` as last subsection of Discussion (line ~222); (4)
StarDist/Cellpose exclusion sentence in Instance-aware methods (line ~68) with two
new bibitems `schmidt2018stardist` (line ~296) and `stringer2021cellpose` (line ~302),
alphabetical. Refs now 27; compiles clean, no undefined cites, no margin overflow.

These four added ~17 lines to an already-full 8-page body (→ 9 pages). To restore
**8 body + 2 ref pages**, the four additions were tightened and redundant prose was
trimmed: §4.2 MedT-confound enumeration (now consolidated in Limitations), §4.5
Qualitative (overlapped the Fig.1 caption), §3 Methods intro (overlapped the new
Contributions sentence), §3.1 Intuition, Discussion Implications + role-of-pretraining,
Conclusion, the Pathology-Foundation-Models entry, and Future Work (dropped its
non-H&E item, now a stated limitation). No results, numbers, claims, citations, or
section structure were removed — wording only. Final: 10 pages (8 + 2), verified.

## How to build
```
pdflatex WhenSimpleWins.tex
pdflatex WhenSimpleWins.tex   # second pass resolves refs/citations
```
Uses the MICCAI-provided `llncs.cls` (copied into this dir from
`~/Downloads/MICCAI2026-Latex-Template`). `splncs04.bst` is also copied in but
unused — the bibliography is a manual `thebibliography` (no bibtex pass needed).

## Compliance status (verified against the pasted MICCAI 2026 guidelines)
- **Template:** Springer LNCS `\documentclass[runningheads]{llncs}`, MICCAI 2026 version. No margin/spacing changes; no `\vspace`/`\hspace` shrinking.
- **Anonymized author block:** kept verbatim from the template (`Anonymized Authors` / `Anonymized Affiliations` / `email@anonymized.com`). Not modified or removed.
- **Abstract + keywords:** present. Abstract = 159 words (LNCS target 150–250). 5 keywords.
- **Page limit:** main content = pages 1–8; references = pages 9–10. Exactly within "8 + up to 2". Verified via pdftotext page-fill analysis.
- **Citations/refs:** numeric LNCS style; 27 refs sorted alphabetically by first-author surname. No undefined citations; every `\bibitem` is cited.
- **Data use declaration:** added (Methods §Datasets). Both PanNuke and MoNuSeg are CC BY-NC-SA 4.0 (verified via web); de-identified public data, no extra ethics approval. Datasets cited.
- **Margins:** no visible margin overflow (only two cosmetic sub-mm overfull-hbox warnings remain; checked page 1 + page 7 renders).

## ⚠️ ACTION REQUIRED BY AUTHOR before submitting
1. **Code URL — DONE.** Now prints the anonymized
   `https://anonymous.4open.science/r/BinaryCellSegmentation-A99C/`. (Verify that
   anonymous.4open.science snapshot is current before the deadline.)
2. **Paper ID:** insert the CMT paper ID into the author block once assigned
   (the template permits only this edit to that block).
3. **Filename — DONE.** Source renamed to `WhenSimpleWins.tex` (surname removed),
   so the source no longer de-anonymizes if chairs request it.
4. **CMT form** — see the dedicated section below.

## VENUE: EMERGE workshop ≠ main conference (IMPORTANT)
Target venue is the **MICCAI Student Board EMERGE workshop** (student-led, "for
students"), not the main conference. Per the official page
(https://miccaimsb.github.io/emerge/), confirmed 2026-06-21:
- **Submit via OpenReview** (MSB EMERGE 2026 portal) — **NOT CMT**. No CMT
  intent-to-submit registration step.
- **No co-author-reviewer signup mandate** is stated (that is a main-conference
  rule). So sole authorship is **not** the structural blocker it would be for the
  main track. (Confirm on the EMERGE CFP/OpenReview portal to be 100% sure.)
- **Deadline: June 30, 2026, 23:59 PT** (archival track). ~9 days out as of 06-21.
- Format = "same guidelines/process as the MICCAI 2026 main conference":
  **LNCS, 8 pages + 2 refs, double-blind.** => the paper as built is already
  correct for EMERGE. Archival track (original full paper) is the right fit.
- The anonymized author block stays "Anonymized Authors"; OpenReview holds the
  real author identity in metadata.

## CMT submission form (MAIN CONFERENCE ONLY — not needed for EMERGE)
The fields below apply only if submitting to the main MICCAI conference via CMT.
For EMERGE, OpenReview will ask for a subset (title, abstract, authors, possibly
keywords/subject areas); the co-author-reviewer nomination and intent-to-submit
do not apply. The novelty statement and area picks below are reusable if asked.

**Drafted from the paper (ready to paste):**
- *Title:* When Simple Wins: Lightweight CNN Encoders for Resource-Constrained Nucleus Segmentation.
- *Abstract:* paste verbatim from the PDF.
- *Primary area:* **MIC** (segmentation/analysis, not intervention).
- *Secondary areas* (pick closest CMT labels; ≥1 each):
  - Modalities: Microscopy / Histopathology (H&E WSI).
  - Application: Image Segmentation (core); optionally Computer-Aided Diagnosis / Digital Pathology.
  - Body: multi-organ/pan-cancer — use a "multiple/whole-body" option if present, else tag the organs covered (breast is dominant in both datasets).
- *Novelty/Impact statement (477/500 chars, impact-type):*
  > Through a controlled U-Net encoder ablation, we show that a simple from-scratch VGG encoder matches a purpose-built small-data transformer (MedT) and outperforms a pretrained Swin Transformer on nucleus segmentation, at lower compute and memory. The significance is scientific/practical impact: reproducible evidence that lightweight from-scratch CNNs—not transformers or pathology foundation models—are the strongest choice for resource-constrained, low-data histopathology.

**Must come from the author (not in the anonymized PDF, cannot be inferred):**
- Full author list + affiliations (LOCKED after submission — must be complete).
- Domain conflicts (current/recent affiliations, advisors, co-authors, collaborators).
- Nominated co-author reviewer (name + email) — must sign up before the intent-to-submit deadline.
- ⚠️ If sole-authored, the author IS the reviewer and (if a student) must have a
  first-authored paper at MICCAI/MedIA/TMI/MIDL/ISBI/IPMI/CVPR etc. to be eligible,
  or the paper is desk-rejected. Adding an eligible co-author resolves this.

## What was changed / removed (and why) vs. final_report.tex
- **Template swap:** NeurIPS `neurips_2024.sty` → LNCS. Repackaged preamble
  (graphicx, amsmath, amsfonts, booktabs, nicefrac, hyperref). Dropped unused
  packages (subcaption, multirow, microtype, url).
- **De-anonymized text removed:** real author block (name/affiliation/email);
  "formatted with the official NeurIPS 2024 style file"; the closing line
  "made available to the instructional staff for reproducibility verification"
  (revealed a course context). Replaced with a proper code-availability line.
- **Appendix removed:** MICCAI 2026 forbids PDF/supplementary appendices
  (supplementary = multimedia only). The two appendix figures
  (`training_curves.png`, PanNuke `summary_grid.png`) were dropped; their content
  is described in the body text and the figures remain in the repo.
- **Figure 1 swapped:** the tall 3-row `monuseg_comparison.png` → single-row
  `monuseg_comparison_row1.png` to fit the 8-page limit; caption rewritten to
  honestly describe the one region shown. Multi-region behavior stays in prose.
- **Prose tightened for length (no claims/numbers changed):** merged redundant
  subsections (dropped "Encoder Comparison" and "Training Dynamics" — folded key
  points into PanNuke results; merged VGG-vs-ResNet + Loss into one Q4
  subsection); removed dataset paragraphs duplicated between Related Work and
  Methods; compressed Q1–Q5 list, Discussion, Conclusion, Future Work, and
  several intros. Two loss equations moved to a display equation (fixed a 32pt
  margin overflow).

## MedT comparison confound (verified, §4.2 expanded)
- **Our training set = 37 images** (verified by counting `data/MoNuSeg 2018 Training
  Data/`: 37 TIFs + 37 XMLs; train cache = 37×16 patches). Test = 14.
- **MedT = 30 images, 512×512** (verified from the MedT paper, arXiv 2102.10662:
  "The training data contains 30 images… The test data contains 14 images"; F1
  79.55 / IoU 66.17). So the existing 37/30 and 0.7955/0.6617 claims are correct.
- The ${+}7$ images (${+}23\%$) bias the MedT comparison in our favor, so §4.2 now
  states we only claim VGG *reaches* MedT-level Dice, and stresses the confound is
  confined to the cross-paper MedT anchor — VGG/ResNet/Swin all share the identical
  37-image set, so the controlled VGG\,>\,Swin result is unaffected.
- **30-vs-37 provenance:** the official MoNuSeg page / 2019 challenge paper quote 30
  (the 2017 Kumar release); the *distributed* 2018 training ZIP — and our data —
  contains 37 (30 + 7 appended; the 6 `-01A-…-TS1/TSB` barcodes in our set are
  almost certainly among the additions). The literature is genuinely split 30/37.
  A footnote in §Datasets now states this to pre-empt reviewer confusion. MedT's
  paper reports 30 with no stated reason for not using the full 37.

## Facts verified (anti-hallucination)
- All numeric results (Dice/IoU/params/latency/memory) copied verbatim from
  `final_report.tex` tables — nothing recomputed or invented.
- PanNuke + MoNuSeg licenses (CC BY-NC-SA 4.0) confirmed via web search.
- No fabricated software version numbers (kept "exact versions in the repo").
- Reference bibliographic data preserved from the original; long author lists
  with "et al." left as-is rather than inventing missing names.
