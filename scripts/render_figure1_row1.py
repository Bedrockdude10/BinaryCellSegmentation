"""Re-render Figure 1 (row 1) from sweep checkpoints.

    python scripts/render_figure1_row1.py <stage_dir> <out_dir>

<stage_dir> holds one subdirectory per model, each with `config.yaml` and
`best.pt`, named `unet_vgg_monuseg` and `unet_swin_pretrained_monuseg` so
visualize.py picks the published column labels. The published checkpoints no
longer exist anywhere in the repo, so the figure cannot be regenerated from the
original weights — these come from the multi-seed sweep instead.

Reuses scripts/visualize.py unchanged. Image selection replicates the published
figure exactly: RandomState(42).choice(14, size=3) picks the same three test
slides, and row 1 is the first of them (TCGA-2Z-A9J9-01A-01-TS1). Rendered as a
native 1-row figure rather than cropped out of the 3-row one.
"""
import importlib.util, sys, pathlib
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("viz", REPO / "scripts/visualize.py")
viz = importlib.util.module_from_spec(spec); spec.loader.exec_module(viz)

stage = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
device = viz.resolve_device("auto")
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

models = []
for d in ["unet_vgg_monuseg", "unet_swin_pretrained_monuseg"]:
    m, disp, raw = viz.load_model(stage / d, device)
    assert m is not None, f"failed to load {d}"
    models.append((viz.MONUSEG_DISPLAY.get(raw, disp), m))
print("columns:", [n for n, _ in models])

tifs = sorted((REPO / "data/MoNuSegTestData").glob("*.tif"))
chosen = sorted(np.random.RandomState(42).choice(len(tifs), size=3, replace=False))
row1 = [tifs[chosen[0]]]
print("row 1 image:", row1[0].stem)
viz.plot_monuseg_comparison(row1, models, device, mean, std, out)
