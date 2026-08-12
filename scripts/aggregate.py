#!/usr/bin/env python3
# scripts/aggregate.py
"""Aggregate the multi-seed sweep into mean/std and paste-ready LaTeX bodies.

    python scripts/aggregate.py                       # full report, fails if incomplete
    python scripts/aggregate.py --allow-partial       # mid-sweep progress view
    python scripts/aggregate.py --json summary.json

Reports mean and sample standard deviation (ddof=1) over the 5 train seeds, plus
mean epochs-to-early-stop and wall clock. Deliberately no p-values and no
confidence intervals: n=5 does not support them.

Three hard errors, because each one means the numbers are not comparable:
  * a configuration with fewer than the expected number of completed runs
  * runs in the same table on different GPU models
  * runs on the same dataset with different split hashes
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sweep import build_specs, load_sweep, sweep_entries  # noqa: E402

# Published figures from another paper, reproduced unchanged (MedT, arXiv 2102.10662).
MEDT_ROW = {"label": r"MedT~\cite{valanarasu2021medt} (published, 30 imgs)",
            "dice": 0.796, "iou": 0.662, "params": "1.4M"}

# Table 1 and 2 row order, matching WhenSimpleWins.tex.
TABLE1_ORDER = ["pannuke_swin_scratch", "pannuke_swin_pretrained", "pannuke_resnet_bce",
                "pannuke_vgg_dice", "pannuke_vgg_bce"]
TABLE2_ORDER = ["monuseg_swin_pretrained", "monuseg_vgg_bce"]
# Supplementary configs: shown in the text report so their numbers are visible,
# deliberately absent from TABLE1_ORDER/TABLE2_ORDER so they can never be emitted
# into a paper table body.
SUPP_ORDER = ["pannuke_swin_pretrained_e150"]
LOSS_LABEL = {"bce_logits": "BCE", "dice": "Dice"}


class AggregationError(RuntimeError):
    pass


# ── Load ─────────────────────────────────────────────────────────────────────

def collect(run_root: Path, sweep_path: str) -> tuple[dict, list[str]]:
    """Group completed results by sweep id; return (groups, problems)."""
    groups: dict[str, list[dict]] = {}
    problems: list[str] = []
    expected_seeds = set(load_sweep(sweep_path)["train_seeds"])

    by_id: dict[str, list[tuple[int, Path]]] = {}
    for spec, run_dir in build_specs(sweep_path, run_root=run_root):
        by_id.setdefault(spec.cfg["experiment"]["sweep_id"], []).append((spec.train_seed, run_dir))

    for sweep_id, entries in by_id.items():
        found = []
        for seed, run_dir in entries:
            path = run_dir / "results.json"
            if not path.exists():
                problems.append(f"{sweep_id} seed {seed}: no results.json ({run_dir})")
                continue
            res = json.loads(path.read_text())
            if res.get("status") != "complete":
                problems.append(f"{sweep_id} seed {seed}: status={res.get('status')}")
                continue
            if res["train_seed"] != seed:
                problems.append(f"{sweep_id} seed {seed}: results.json says {res['train_seed']}")
                continue
            found.append(res)
        seeds_found = {r["train_seed"] for r in found}
        missing = expected_seeds - seeds_found
        if missing:
            problems.append(f"{sweep_id}: missing seeds {sorted(missing)} "
                            f"({len(seeds_found)}/{len(expected_seeds)} complete)")
        if found:
            groups[sweep_id] = sorted(found, key=lambda r: r["train_seed"])
    return groups, problems


def check_regime(groups: dict[str, list[dict]]) -> list[str]:
    """One GPU model everywhere, one split hash per dataset. Non-negotiable.

    Returns non-fatal warnings. A differing ``code_hash`` warns rather than
    errors: the code can change in ways that provably cannot touch numerics
    (moving a dataclass, renaming a script), and only the operator can judge
    that. GPU model and split hash admit no such judgement, so they raise.
    """
    gpus: dict[str, list[str]] = {}
    splits: dict[str, dict[str, list[str]]] = {}
    codes: dict[str, list[str]] = {}
    for sweep_id, runs in groups.items():
        for r in runs:
            gpus.setdefault(r.get("gpu_model", "<unknown>"), []).append(r["run_id"])
            splits.setdefault(r["dataset"], {}).setdefault(r["split_hash"], []).append(r["run_id"])
            codes.setdefault(r.get("code_hash", "<unknown>"), []).append(r["run_id"])

    if len(gpus) > 1:
        detail = "\n".join(f"  {g}: {len(ids)} runs (e.g. {ids[0]})" for g, ids in gpus.items())
        raise AggregationError(
            "Runs span multiple GPU models — hardware variance would be reported as seed "
            f"variance, and Table 3 would be meaningless:\n{detail}"
        )
    for dataset, by_hash in splits.items():
        if len(by_hash) > 1:
            detail = "\n".join(f"  {h[:12]}: {len(ids)} runs (e.g. {ids[0]})"
                               for h, ids in by_hash.items())
            raise AggregationError(
                f"{dataset}: runs used different splits, so their metrics are not "
                f"comparable:\n{detail}"
            )

    warnings = []
    if len(codes) > 1:
        detail = "; ".join(f"{h[:12]} x{len(ids)}" for h, ids in codes.items())
        warnings.append(
            f"runs span {len(codes)} code_hash values ({detail}). Confirm the "
            f"differences cannot affect numerics before reporting these together."
        )
    versions = {json.dumps(r.get("versions", {}), sort_keys=True)
                for runs in groups.values() for r in runs}
    if len(versions) > 1:
        warnings.append(f"runs span {len(versions)} distinct library-version sets")
    amp = {bool(r.get("amp_enabled")) for runs in groups.values() for r in runs}
    if len(amp) > 1:
        warnings.append("runs mix AMP-enabled and AMP-disabled — different numerics regimes")
    return warnings


# ── Statistics ───────────────────────────────────────────────────────────────

def _mstd(values: list[float]) -> dict:
    vals = [float(v) for v in values]
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "values": vals,
    }


def summarise(sweep_id: str, runs: list[dict], entry: dict) -> dict:
    params = {r["params_total"] for r in runs}
    if len(params) != 1:
        raise AggregationError(f"{sweep_id}: differing parameter counts {params}")
    peak = [r.get("peak_gpu_memory_bytes") or 0 for r in runs]
    return {
        "sweep_id": sweep_id,
        "table": runs[0]["table"],
        "dataset": runs[0]["dataset"],
        "encoder": runs[0]["encoder"],
        "loss": runs[0]["loss"],
        "row_label": entry.get("row_label", sweep_id),
        "n": len(runs),
        "seeds": [r["train_seed"] for r in runs],
        "params_total": params.pop(),
        "gpu_model": runs[0].get("gpu_model"),
        "split_hash": runs[0]["split_hash"],
        "test_dice": _mstd([r["test"]["dice"] for r in runs]),
        "test_iou": _mstd([r["test"]["iou"] for r in runs]),
        "best_val_dice": _mstd([r.get("best_val_dice", float("nan")) for r in runs]),
        "epochs_run": _mstd([r["epochs_run"] for r in runs]),
        "best_epoch": _mstd([r["best_epoch"] for r in runs]),
        "early_stopped_count": sum(bool(r.get("early_stopped")) for r in runs),
        "wall_clock_seconds": _mstd([r["wall_clock_seconds"] for r in runs]),
        "wall_clock_total_seconds": sum(r["wall_clock_seconds"] for r in runs),
        "peak_gpu_memory_bytes": _mstd(peak),
        "published": entry.get("published"),
    }


def pooled_std(a: dict, b: dict) -> float:
    """Equal-n pooled SD — the yardstick for 'is this difference meaningful'."""
    return ((a["std"] ** 2 + b["std"] ** 2) / 2) ** 0.5


def comparisons(summaries: dict[str, dict], order: list[str], metric: str) -> list[dict]:
    out = []
    present = [s for s in order if s in summaries]
    for i, a_id in enumerate(present):
        for b_id in present[i + 1:]:
            a, b = summaries[a_id][metric], summaries[b_id][metric]
            delta = abs(a["mean"] - b["mean"])
            ps = pooled_std(a, b)
            out.append({
                "a": a_id, "b": b_id, "metric": metric,
                "delta_mean": a["mean"] - b["mean"],
                "abs_delta": delta, "pooled_std": ps,
                "warning": delta < ps,
            })
    return out


# ── LaTeX ────────────────────────────────────────────────────────────────────

# Table 1 already has 6 columns in a page-limited LNCS layout, so mean+-std may
# not fit. "compact" writes the SD in parentheses, which is ~4 characters
# narrower per cell and a conventional alternative.
_STYLES = ("pm", "compact")


def _pm(stat: dict, places: int = 3, style: str = "pm") -> str:
    if style == "compact":
        return f"{stat['mean']:.{places}f}\\,({stat['std']:.{places}f})"
    return f"{stat['mean']:.{places}f} $\\pm$ {stat['std']:.{places}f}"


def _bold_pm(stat: dict, places: int = 3, style: str = "pm") -> str:
    if style == "compact":
        return f"\\textbf{{{stat['mean']:.{places}f}}}\\,({stat['std']:.{places}f})"
    return f"\\textbf{{{stat['mean']:.{places}f}}} $\\pm$ \\textbf{{{stat['std']:.{places}f}}}"


def _params(n: int) -> str:
    return f"{n / 1e6:.1f}M"


def latex_table1(summaries: dict[str, dict], otsu: dict | None, style: str = "pm") -> str:
    """Body for tab:pannuke_results — columns @{}llcccr@{}:
    Encoder & Loss & Dice & IoU & Epochs & Params."""
    present = [s for s in TABLE1_ORDER if s in summaries]
    if not present:
        return "% no PanNuke runs aggregated\n"
    best_dice = max(present, key=lambda s: summaries[s]["test_dice"]["mean"])
    best_iou = max(present, key=lambda s: summaries[s]["test_iou"]["mean"])
    sep = "\\," if style == "compact" else " $\\pm$ "

    lines = []
    if otsu:
        lines.append(f"Otsu (classical) & --- & {otsu['dice']:.3f} & {otsu['iou']:.3f} "
                     f"& --- & 0 \\\\")
    for sid in present:
        s = summaries[sid]
        dice = (_bold_pm(s["test_dice"], style=style) if sid == best_dice
                else _pm(s["test_dice"], style=style))
        iou = (_bold_pm(s["test_iou"], style=style) if sid == best_iou
               else _pm(s["test_iou"], style=style))
        ep = (f"{s['epochs_run']['mean']:.1f}\\,({s['epochs_run']['std']:.1f})"
              if style == "compact"
              else f"{s['epochs_run']['mean']:.1f}{sep}{s['epochs_run']['std']:.1f}")
        lines.append(
            f"{s['row_label']} & {LOSS_LABEL.get(s['loss'], s['loss'])} & {dice} & {iou} "
            f"& {ep} & {_params(s['params_total'])} \\\\"
        )
    return "\n".join(lines) + "\n"


def latex_table2(summaries: dict[str, dict], style: str = "pm") -> str:
    """Body for tab:monuseg_results — columns @{}lccc@{}:
    Model & Dice & IoU & Params."""
    present = [s for s in TABLE2_ORDER if s in summaries]
    if not present:
        return "% no MoNuSeg runs aggregated\n"
    best_dice = max(present, key=lambda s: summaries[s]["test_dice"]["mean"])
    best_iou = max(present, key=lambda s: summaries[s]["test_iou"]["mean"])

    lines = [f"{MEDT_ROW['label']} & {MEDT_ROW['dice']:.3f} & {MEDT_ROW['iou']:.3f} "
             f"& {MEDT_ROW['params']} \\\\"]
    for sid in present:
        s = summaries[sid]
        dice = (_bold_pm(s["test_dice"], style=style) if sid == best_dice
                else _pm(s["test_dice"], style=style))
        iou = (_bold_pm(s["test_iou"], style=style) if sid == best_iou
               else _pm(s["test_iou"], style=style))
        lines.append(f"{s['row_label']} & {dice} & {iou} & {_params(s['params_total'])} \\\\")
    return "\n".join(lines) + "\n"


# ── Report ───────────────────────────────────────────────────────────────────

def _fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m"


def report(summaries: dict[str, dict], comps: list[dict], otsu: dict | None) -> str:
    out: list[str] = []
    gpu = next(iter(summaries.values()))["gpu_model"] if summaries else "?"
    out.append(f"GPU model (all runs): {gpu}")
    for dataset in sorted({s["dataset"] for s in summaries.values()}):
        h = next(s["split_hash"] for s in summaries.values() if s["dataset"] == dataset)
        out.append(f"split hash [{dataset}]: {h}")
    out.append("")

    out.append(f"{'config':26s} {'n':>2s}  {'test Dice':>18s} {'test IoU':>18s} "
               f"{'epochs':>13s} {'best ep':>8s} {'wall/run':>9s}")
    for sid in TABLE1_ORDER + TABLE2_ORDER + SUPP_ORDER:
        if sid not in summaries:
            continue
        s = summaries[sid]
        out.append(
            f"{sid:26s} {s['n']:2d}  "
            f"{s['test_dice']['mean']:.4f}+-{s['test_dice']['std']:.4f}  "
            f"{s['test_iou']['mean']:.4f}+-{s['test_iou']['std']:.4f}  "
            f"{s['epochs_run']['mean']:6.1f}+-{s['epochs_run']['std']:4.1f} "
            f"{s['best_epoch']['mean']:8.1f} "
            f"{_fmt_hms(s['wall_clock_seconds']['mean']):>9s}"
        )
    total = sum(s["wall_clock_total_seconds"] for s in summaries.values())
    n_runs = sum(s["n"] for s in summaries.values())
    out.append("")
    out.append(f"total wall clock: {_fmt_hms(total)} over {n_runs} runs "
               f"({_fmt_hms(total / max(n_runs, 1))} mean)")

    out.append("")
    out.append("Published single-seed value vs multi-seed mean (delta in units of SD):")
    for sid in TABLE1_ORDER + TABLE2_ORDER + SUPP_ORDER:
        s = summaries.get(sid)
        if not s or not s.get("published"):
            continue
        pub, got = s["published"], s["test_dice"]
        d = got["mean"] - pub["dice"]
        sd = f"{d / got['std']:+.1f} SD" if got["std"] > 0 else "n/a"
        out.append(f"  {sid:26s} published {pub['dice']:.4f}  mean {got['mean']:.4f}  "
                   f"delta {d:+.4f} ({sd})  seed-42 {_seed42(s):.4f}")

    # A hard flag at exactly 1x pooled SD is a brittle line to stand on, so every
    # comparison's ratio is printed too: a delta of 1.1x the SD is no more
    # claimable than one of 0.9x, and only showing the flagged ones would hide it.
    out.append("")
    out.append("Pairwise differences, |delta| / pooled SD (smallest first):")
    for c in sorted(comps, key=lambda x: abs(x["delta_mean"]) / max(x["pooled_std"], 1e-12)):
        ratio = abs(c["delta_mean"]) / max(c["pooled_std"], 1e-12)
        note = ("  <-- BELOW pooled SD, do not claim" if c["warning"]
                else "  <-- within 2x pooled SD, treat as indistinguishable" if ratio < 2
                else "")
        out.append(f"   [{c['metric'][5:]:4s}] {c['a']:24s} vs {c['b']:24s} "
                   f"delta {c['delta_mean']:+.4f}  pooledSD {c['pooled_std']:.4f}  "
                   f"ratio {ratio:6.1f}{note}")

    for style in _STYLES:
        label = ("mean $\\pm$ SD" if style == "pm" else "mean (SD) — narrower, if the "
                 "6-column Table 1 overflows the LNCS text width")
        out.append("")
        out.append("=" * 78)
        out.append(f"% Table 1 body (tab:pannuke_results) — columns: @{{}}llcccr@{{}} — {label}")
        out.append("=" * 78)
        out.append(latex_table1(summaries, otsu, style))
        out.append("=" * 78)
        out.append(f"% Table 2 body (tab:monuseg_results) — columns: @{{}}lccc@{{}} — {label}")
        out.append("=" * 78)
        out.append(latex_table2(summaries, style))
    return "\n".join(out)


def _seed42(s: dict) -> float:
    try:
        return s["test_dice"]["values"][s["seeds"].index(42)]
    except (ValueError, IndexError):
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--otsu", default="outputs/classical_otsu/eval_fold3.json")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Report what exists instead of failing on missing runs")
    ap.add_argument("--json", default=None, help="Also write the summary as JSON")
    args = ap.parse_args()

    groups, problems = collect(Path(args.run_root), args.sweep)
    if problems and not args.allow_partial:
        print("Incomplete sweep — refusing to average fewer seeds than expected:",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("\nRe-run the missing seeds, or pass --allow-partial for a progress view.",
              file=sys.stderr)
        return 1
    if not groups:
        print("No completed runs found.", file=sys.stderr)
        return 1
    if problems:
        print("WARNING: partial aggregation, these are NOT reportable numbers:",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("", file=sys.stderr)

    try:
        regime_warnings = check_regime(groups)
    except AggregationError as e:
        # Expected failure mode, not a crash — no traceback.
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    for w in regime_warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if regime_warnings:
        print("", file=sys.stderr)
    entries = sweep_entries(args.sweep)
    try:
        summaries = {sid: summarise(sid, runs, entries[sid]) for sid, runs in groups.items()}
    except AggregationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    comps: list[dict] = []
    for order in (TABLE1_ORDER, TABLE2_ORDER):
        for metric in ("test_dice", "test_iou"):
            comps += comparisons(summaries, order, metric)

    otsu = None
    otsu_path = Path(args.otsu)
    if otsu_path.exists():
        otsu = json.loads(otsu_path.read_text())

    text = report(summaries, comps, otsu)
    print(text)

    if args.json:
        Path(args.json).write_text(json.dumps({
            "summaries": summaries, "comparisons": comps, "otsu": otsu,
            "problems": problems, "partial": bool(problems),
            "regime_warnings": regime_warnings,
        }, indent=2, sort_keys=True, default=str))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
