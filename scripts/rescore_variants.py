"""Re-score saved products with variants of the image score and of the background mask.

    python scripts/rescore_variants.py --products checkpoints --dataset visa --shots 4 --out outputs/rescore

No product is refitted: every variant reuses the patch distances of the exported checkpoints, so
differences come only from how patch distances become an image score. Variants:

* ``default``    dilated + closed foreground mask, mean of the top 1% patches (as evaluated)
* ``nodilate``   foreground mask without dilation / closing (drops the dark band around objects)
* ``max``        dilated mask, highest patch only
* ``nodilate_max``

For each variant, validation images (normal, from the split) are scored too, so thresholds can be
recomputed: ``q99`` (the evaluated rule, 99% quantile of the 30 validation scores) and
``mean3sd`` (mean + 3 standard deviations of the validation scores). Output: one CSV row per
(method, seed, class, variant) with I-AUROC and, per threshold rule, FPR and recall on the test set.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from scipy import ndimage

from fscad.data.datasets import list_samples, normalize
from fscad.data.splits import load_split
from fscad.metrics import image_auroc
from fscad.pipeline import dataset_root, load_images, setup
from fscad.product import ProductModel, activate
from fscad.scoring.masking import foreground_mask
from fscad.scoring.scoring import image_score

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("default", "nodilate", "max", "nodilate_max")


def undilate(mask: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Erode the dilated + closed mask back by one 3x3 step (removes the added one-patch rim)."""
    h, w = grid
    s = np.ones((3, 3), dtype=bool)
    out = [ndimage.binary_erosion(m.reshape(h, w).numpy(), structure=s, border_value=1) for m in mask]
    return torch.from_numpy(np.stack(out).reshape(len(out), -1))


@torch.no_grad()
def variant_scores(backbone, product, images, device, bs, top_q) -> dict[str, np.ndarray]:
    activate(backbone, product)
    product.memory.to(device)
    out = {v: [] for v in VARIANTS}
    for i in range(0, len(images), bs):
        frozen, adapted = backbone.forward_both(normalize(images[i : i + bs].to(device)))
        d = product.memory.query(adapted.patches)
        if product.meta.get("masking"):
            fg = foreground_mask(frozen.patches, frozen.grid)
            tight = undilate(fg, frozen.grid)
        else:
            fg = tight = torch.ones(d.shape, dtype=torch.bool)
        dd = d * fg.to(d.device, d.dtype)
        dt = d * tight.to(d.device, d.dtype)
        out["default"].append(image_score(dd, "top_mean", top_q).cpu())
        out["nodilate"].append(image_score(dt, "top_mean", top_q).cpu())
        out["max"].append(image_score(dd, "max").cpu())
        out["nodilate_max"].append(image_score(dt, "max").cpu())
    product.memory.to("cpu")
    return {v: torch.cat(s).numpy() for v, s in out.items()}


def rates(test: np.ndarray, labels: np.ndarray, thr: float) -> tuple[float, float]:
    flag = test > thr
    return float(flag[labels == 0].mean()), float(flag[labels == 1].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--dataset", default="visa")
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--methods", nargs="+", default=["knn", "lora"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "rescore")
    a = ap.parse_args()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"dataset={a.dataset}", "tracking.enabled=false"])
    backbone, device = setup(cfg)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.dataset}_k{a.shots}.csv"
    done = set()
    if path.exists():
        with open(path) as f:
            done = {(r["method"], int(r["seed"]), r["cls"]) for r in csv.DictReader(f)}
    root = dataset_root(cfg, a.dataset)
    for method in a.methods:
        for seed in a.seeds:
            bundle = a.products / f"{a.dataset}-{method}-k{a.shots}-s{seed}"
            for m in sorted(bundle.glob("*/model.pt")):
                cls = m.parent.name.split("__", 1)[1]
                if (a.classes and cls not in a.classes) or (method, seed, cls) in done:
                    continue
                product = ProductModel.load(m)
                test = list_samples(a.dataset, root, cls, "test")
                labels = np.array([s.label for s in test])
                split = load_split(cfg.paths.splits_dir, a.dataset, cls, a.shots, seed, root=root)
                size, bs, q = cfg.model.img_size, cfg.eval.batch_size, cfg.eval.top_q
                st = variant_scores(
                    backbone, product, load_images([s.path for s in test], size), device, bs, q
                )
                sv = variant_scores(backbone, product, load_images(split["val_paths"], size), device, bs, q)
                rows = []
                for v in VARIANTS:
                    row = {"method": method, "seed": seed, "cls": cls, "variant": v}
                    row["i_auroc"] = image_auroc(labels, st[v])
                    for rule, thr in (
                        ("q99", float(np.quantile(sv[v], 0.99))),
                        ("mean3sd", float(sv[v].mean() + 3 * sv[v].std(ddof=1))),
                    ):
                        row[f"fpr_{rule}"], row[f"recall_{rule}"] = rates(st[v], labels, thr)
                    rows.append(row)
                new = not path.exists()
                with open(path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    if new:
                        w.writeheader()
                    w.writerows(rows)
                d = rows[0]
                print(
                    f"{method} s{seed} {cls:<12} default {d['i_auroc']:.4f}  nodilate {rows[1]['i_auroc']:.4f}  "
                    f"max {rows[2]['i_auroc']:.4f}  nodilate_max {rows[3]['i_auroc']:.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
