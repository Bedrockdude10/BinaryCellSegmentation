#!/usr/bin/env python3
# scripts/make_split_manifest.py
"""Record the train/val/test splits so every run can assert against them.

    python scripts/make_split_manifest.py            # write configs/split_manifest.json
    python scripts/make_split_manifest.py --check     # verify, change nothing

Run this ONCE and commit the result. Every training run recomputes the
fingerprint and dies immediately if it does not match, which is what stops a
drifted split (rebuilt cache, extra image, changed val_fraction) from silently
contaminating a table halfway through the sweep.

Regenerating the manifest invalidates every already-completed run: if the split
changed, all 5 seeds of every configuration must be re-run, because a table may
not mix splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.builders import split_fingerprint  # noqa: E402
from src.splits import MANIFEST_PATH, fingerprint_hash  # noqa: E402
from src.sweep import build_specs  # noqa: E402


def build_manifest(sweep_path: str, cache_root: str | None) -> dict:
    manifest: dict[str, dict] = {}
    for spec, _ in build_specs(sweep_path, cache_root=cache_root):
        if spec.dataset in manifest:
            continue
        detail = split_fingerprint(spec.cfg, spec.split_seed)
        manifest[spec.dataset] = {"hash": fingerprint_hash(detail), "detail": detail}
        print(f"{spec.dataset:9s} {manifest[spec.dataset]['hash'][:16]}  "
              f"{_describe(detail)}")
    return manifest


def _describe(detail: dict) -> str:
    if detail["dataset"] == "monuseg":
        return (f"split_seed={detail['split_seed']} val_fraction={detail['val_fraction']} "
                f"{detail['n_source_images']} imgs -> "
                f"{detail['n_train_patches']}/{detail['n_val_patches']} patches, "
                f"val={','.join(s[:12] for s in detail['val_images'])}")
    return f"folds={detail['folds']} n={detail['num_examples']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--out", default=str(MANIFEST_PATH))
    ap.add_argument("--check", action="store_true",
                    help="Compare against the committed manifest instead of writing")
    args = ap.parse_args()

    manifest = build_manifest(args.sweep, args.cache_root)
    out = Path(args.out)

    if args.check:
        if not out.exists():
            print(f"{out} does not exist", file=sys.stderr)
            return 1
        existing = json.loads(out.read_text())
        bad = [d for d in manifest
               if existing.get(d, {}).get("hash") != manifest[d]["hash"]]
        if bad:
            for d in bad:
                print(f"MISMATCH {d}: manifest={existing.get(d, {}).get('hash', '<absent>')[:16]} "
                      f"computed={manifest[d]['hash'][:16]}", file=sys.stderr)
            return 1
        print(f"\n{out}: all {len(manifest)} splits match")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nWrote {out} — commit this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
