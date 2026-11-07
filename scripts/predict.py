"""Run inference with saved checkpoints on image files, without starting the API.

    # route every image among the products of a bundle, write predictions.csv and heatmaps
    python scripts/predict.py --products checkpoints/visa-lora-k4-s0 --out outputs/predict \
        data/visa/pcb1/Data/Images/Anomaly/000.JPG data/visa/candle/Data/Images/Normal/

    # skip the router: score every image as one known product
    python scripts/predict.py --products checkpoints/visa-lora-k4-s0 --product visa/pcb1 img.png

``--products`` is a directory of ``<product id>/model.pt`` (a bundle written by
``scripts/export_checkpoints.py``, or the API's products directory). The backbone, LoRA shape and
scoring settings are read from the checkpoints, so they always match how the products were fitted.
Scoring is the one of the evaluation (``fscad.product.score_product``), background masking included.

Per image the CSV has: routed product (empty if rejected), nearest product and its CLS similarity,
whether the router rejected it, the anomaly score, the product's threshold delta and the decision:
``ok``, ``anomaly`` or ``rejected`` (unknown product: an alert for a person, never passed silently).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from fscad.data.datasets import load_image  # noqa: E402
from fscad.product import ProductModel  # noqa: E402
from fscad.service import AnomalyService  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def images(args: list[str]) -> list[Path]:
    out = []
    for a in map(Path, args):
        out += sorted(p for p in a.rglob("*") if p.suffix.lower() in EXTS) if a.is_dir() else [a]
    return out


def service_cfg(products: Path, device: str):
    """serve.yaml + the model / LoRA / scoring settings stored in the checkpoints."""
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="serve")
    OmegaConf.set_struct(cfg, False)
    metas = [ProductModel.load(p).meta for p in sorted(products.glob("*/model.pt"))]
    if not metas:
        raise SystemExit(f"no <product>/model.pt under {products}")
    stored = [m.get("config") for m in metas if m.get("config")]
    if stored:
        keys = ("method", "model", "lora", "eval")
        if any({k: s.get(k) for k in keys} != {k: stored[0].get(k) for k in keys} for s in stored):
            raise SystemExit("products in this directory were fitted with different configurations")
        c = stored[0]
        cfg = OmegaConf.merge(cfg, {k: c[k] for k in ("model", "memory") if c.get(k)})
        cfg.method = c.get("method", cfg.method)
        if c.get("lora"):
            cfg.lora = OmegaConf.merge(cfg.lora, c["lora"])
        cfg.eval = OmegaConf.merge(cfg.eval, {k: v for k, v in c["eval"].items() if k != "masking"})
    cfg.products_dir, cfg.device, cfg.cache_size = str(products), device, len(metas)
    return cfg


def heatmap(img, amap, title: str, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.4))
    ax[0].imshow(img.permute(1, 2, 0))
    ax[1].imshow(img.permute(1, 2, 0))
    ax[1].imshow(amap, cmap="jet", alpha=0.45)
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files or directories")
    ap.add_argument("--products", required=True, type=Path, help="directory of <product>/model.pt")
    ap.add_argument("--product", help="score every image as this product (no routing)")
    ap.add_argument("--out", type=Path, default=Path("outputs/predict"))
    ap.add_argument("--no-maps", action="store_true", help="do not write heatmap PNGs")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    a = ap.parse_args()

    cfg = service_cfg(a.products, a.device)
    svc = AnomalyService(cfg)
    print(
        f"{len(svc.router)} products on {svc.device}: {', '.join(svc.router.classes)}; tau {svc.router.tau:.4f}"
    )
    if a.product and a.product not in svc.router.classes:
        raise SystemExit(f"unknown product {a.product}")
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(images(a.inputs)):
        x = load_image(path, cfg.model.img_size)
        p = svc.predict(x, return_map=not a.no_maps, product=a.product)
        decision = "rejected" if p.rejected else ("anomaly" if p.is_anomaly else "ok")
        row = {
            "image": str(path),
            "product": p.product or "",
            "nearest": p.best_match,
            "similarity": round(p.similarity, 4),
            "rejected": p.rejected,
            "score": None if p.score is None else round(p.score, 4),
            "threshold": None if p.threshold is None else round(p.threshold, 4),
            "decision": decision,
        }
        rows.append(row)
        if p.anomaly_map is not None:
            title = f"{path.name}: {row['product']} S={row['score']} delta={row['threshold']} -> {decision}"
            heatmap(x, p.anomaly_map, title, a.out / f"{i:04d}_{path.stem}.png")
        print(
            f"{path.name:<28} {decision:<9} {row['product'] or '(nearest ' + p.best_match + ')':<22} S={row['score']}"
        )
    with open(a.out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {d: sum(r["decision"] == d for r in rows) for d in ("ok", "anomaly", "rejected")}
    (a.out / "summary.json").write_text(json.dumps({"images": len(rows), **summary}, indent=2))
    print(f"{len(rows)} images: {summary}; results in {a.out}")


if __name__ == "__main__":
    main()
