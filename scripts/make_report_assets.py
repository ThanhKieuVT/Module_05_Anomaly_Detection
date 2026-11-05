"""Illustrations for the report, made from real VisA images and the project's own code.

    python scripts/make_report_assets.py --out ../report/figures

Writes PNGs plus ``assets.json`` with the exact image files used, so every caption can name them:

* dataset_samples.png  normal and defective test images of four classes, ground truth outlined
* fewshot_pcb1.png     the 4 shots of the fixed split visa/pcb1 k4_s0 and two of its validation images
* vit_patches.png      one shot cut into the 32x32 grid of 14-pixel patches DINOv2 sees at 448 px
* synthetic.png        Perlin and CutPaste anomalies made by fscad.data.synthetic on that shot
* masking.png          background masks from fscad.scoring.masking on frozen DINOv2 tokens

These are illustrations of inputs and preprocessing, not results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from fscad.data.datasets import list_samples, load_image, load_mask, normalize  # noqa: E402
from fscad.data.splits import load_split  # noqa: E402
from fscad.data.synthetic import synthesize  # noqa: E402
from fscad.pipeline import setup  # noqa: E402
from fscad.scoring.masking import foreground_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "visa"
SIZE = 448
CLASSES = ("pcb1", "macaroni2", "capsules", "candle")


def rel(p: str | Path) -> str:
    return str(Path(p).relative_to(DATA))


def img(p) -> np.ndarray:
    return load_image(p, SIZE).permute(1, 2, 0).numpy()


def off(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def pick_defect(samples):
    """Defective test image whose defect is clearly visible: largest mask among the first 30."""
    bad = [s for s in samples if s.label == 1][:30]
    return max(bad, key=lambda s: float(load_mask(s.mask, 112).sum()))


def dataset_samples(out: Path) -> dict:
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(2.4 * len(CLASSES), 5.0))
    used = {}
    for j, cls in enumerate(CLASSES):
        test = list_samples("visa", DATA, cls, "test")
        good = next(s for s in test if s.label == 0)
        bad = pick_defect(test)
        axes[0, j].imshow(img(good.path))
        off(axes[0, j], f"{cls}: bình thường\n{Path(good.path).name}")
        axes[1, j].imshow(img(bad.path))
        axes[1, j].contour(load_mask(bad.mask, SIZE).numpy(), levels=[0.5], colors="red", linewidths=1.2)
        off(axes[1, j], f"{cls}: có lỗi\n{Path(bad.path).name}")
        used[cls] = {"normal": rel(good.path), "anomaly": rel(bad.path), "mask": rel(bad.mask)}
    fig.tight_layout()
    fig.savefig(out / "dataset_samples.png", dpi=150)
    plt.close(fig)
    return used


def fewshot(out: Path, splits_dir: Path) -> dict:
    split = load_split(splits_dir, "visa", "pcb1", 4, 0, root=DATA)
    shots, val = split["train_paths"], split["val_paths"][:2]
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
    for ax, p, tag in zip(axes, shots + val, ["shot"] * 4 + ["validation"] * 2, strict=True):
        ax.imshow(img(p))
        off(ax, f"{tag}\n{Path(p).name}")
        for s in ax.spines.values():
            s.set_edgecolor("tab:green" if tag == "shot" else "tab:gray")
            s.set_linewidth(2.5)
    fig.tight_layout()
    fig.savefig(out / "fewshot_pcb1.png", dpi=150)
    plt.close(fig)
    return {
        "split": "splits/visa/pcb1/k4_s0.json",
        "shots": [rel(p) for p in shots],
        "val": [rel(p) for p in val],
    }


def vit_patches(out: Path, path: str) -> None:
    x = img(path)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(x)
    for k in range(0, SIZE + 1, 14):
        ax.axhline(k - 0.5, color="white", lw=0.35, alpha=0.8)
        ax.axvline(k - 0.5, color="white", lw=0.35, alpha=0.8)
    r, c = 14, 9  # one patch highlighted
    ax.add_patch(plt.Rectangle((c * 14 - 0.5, r * 14 - 0.5), 14, 14, fill=False, ec="red", lw=1.5))
    ax.set_xlim(-0.5, SIZE - 0.5)
    ax.set_ylim(SIZE - 0.5, -0.5)
    ax.axis("off")
    fig.savefig(out / "vit_patches.png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def synthetic(out: Path, shots: list[str]) -> dict:
    x = load_image(shots[0], SIZE)
    src = load_image(shots[1], SIZE)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    axes[0, 0].imshow(x.permute(1, 2, 0))
    off(axes[0, 0], f"shot gốc\n{Path(shots[0]).name}")
    axes[1, 0].axis("off")
    for j, kind in enumerate(("perlin", "cutpaste"), start=1):
        g = torch.Generator().manual_seed(3 + j)
        a, m = synthesize(x.clone(), src, kind, g)
        axes[0, j].imshow(a.clamp(0, 1).permute(1, 2, 0))
        off(axes[0, j], f"lỗi tổng hợp: {kind}")
        axes[1, j].imshow(m, cmap="gray")
        off(axes[1, j], "mask vùng lỗi")
    fig.tight_layout()
    fig.savefig(out / "synthetic.png", dpi=150)
    plt.close(fig)
    return {"shot": rel(shots[0]), "texture_source": rel(shots[1])}


@torch.no_grad()
def masking(out: Path, used: dict) -> dict:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["method=knn", "tracking.enabled=false"])
    backbone, device = setup(cfg)
    paths = [DATA / used[c]["anomaly"] for c in ("pcb1", "macaroni2", "capsules")]
    x = torch.stack([load_image(p, SIZE) for p in paths])
    f = backbone(normalize(x.to(device)), adapted=False)
    m = foreground_mask(f.patches, f.grid).reshape(-1, *f.grid).float()
    fig, axes = plt.subplots(2, len(paths), figsize=(2.6 * len(paths), 5.0))
    for j, p in enumerate(paths):
        axes[0, j].imshow(x[j].permute(1, 2, 0))
        off(axes[0, j], f"{p.parts[-5]} / {p.name}")
        axes[1, j].imshow(x[j].permute(1, 2, 0))
        axes[1, j].imshow(m[j], cmap="Greens", alpha=0.45, extent=(0, SIZE, SIZE, 0))
        off(axes[1, j], f"giữ {m[j].mean():.0%} patch (xanh)")
    fig.tight_layout()
    fig.savefig(out / "masking.png", dpi=150)
    plt.close(fig)
    return {"images": [rel(p) for p in paths], "kept": [round(float(v), 3) for v in m.mean((1, 2))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/report_assets")
    ap.add_argument("--splits-dir", default=str(ROOT / "splits"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assets = {"dataset_samples": dataset_samples(out)}
    assets["fewshot"] = fewshot(out, Path(a.splits_dir))
    shots = [str(DATA / p) for p in assets["fewshot"]["shots"]]
    vit_patches(out, shots[0])
    assets["vit_patches"] = assets["fewshot"]["shots"][0]
    assets["synthetic"] = synthetic(out, shots)
    assets["masking"] = masking(out, assets["dataset_samples"])
    (out / "assets.json").write_text(json.dumps(assets, indent=2, ensure_ascii=False))
    print(json.dumps(assets, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
