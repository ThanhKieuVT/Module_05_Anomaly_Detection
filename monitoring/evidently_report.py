"""Offline drift report with Evidently on CLS embeddings (reference = onboarding, current = stream).

    python monitoring/evidently_report.py --product artifacts/serving/mvtec_ad__bottle/model.pt \
        --current-dir data/mvtec_ad2/vial/test_public/good --out results/evidently_bottle.html

Embeddings are reduced with PCA (fit on the reference) so the per-column drift tests stay readable.
Supports both the legacy (<0.7) and the new (>=0.7) Evidently APIs.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from sklearn.decomposition import PCA

from fscad.data.datasets import IMG_EXTS, normalize
from fscad.models.backbone import build_backbone
from fscad.pipeline import load_images
from fscad.product import ProductModel
from fscad.utils import get_device

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def run_report(ref: pd.DataFrame, cur: pd.DataFrame, out: Path) -> None:
    try:  # Evidently >= 0.7
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(reference_data=ref, current_data=cur)
        snapshot.save_html(str(out))
    except ImportError:  # legacy API
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        r = Report(metrics=[DataDriftPreset()])
        r.run(reference_data=ref, current_data=cur)
        r.save_html(str(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True)
    ap.add_argument("--current-dir", required=True)
    ap.add_argument("--components", type=int, default=8)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--out", default="results/evidently_report.html")
    a = ap.parse_args()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="serve", overrides=shlex.split(a.overrides))
    p = ProductModel.load(a.product)
    ref_cls = torch.cat([p.cls_feats] + ([p.val_cls] if p.val_cls is not None else []))
    device = get_device(cfg.device)
    backbone = build_backbone(cfg).to(device)
    paths = sorted(str(x) for x in Path(a.current_dir).rglob("*") if x.suffix.lower() in IMG_EXTS)
    cur = []
    with torch.no_grad():
        for i in range(0, len(paths), 8):
            x = normalize(load_images(paths[i : i + 8], cfg.model.img_size).to(device))
            cur.append(backbone(x, adapted=False).cls.cpu())
    cur_cls = torch.cat(cur)
    k = min(a.components, len(ref_cls) - 1, ref_cls.shape[1])
    pca = PCA(n_components=k).fit(ref_cls.numpy())
    cols = [f"pc{i}" for i in range(k)]
    ref_df = pd.DataFrame(pca.transform(ref_cls.numpy()), columns=cols)
    cur_df = pd.DataFrame(pca.transform(cur_cls.numpy()), columns=cols)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_report(ref_df, cur_df, out)
    print(
        f"reference n={len(ref_df)}  current n={len(cur_df)}  mean |shift| per PC: "
        f"{np.abs(cur_df.mean() - ref_df.mean()).round(3).tolist()}"
    )
    print(f"report: {out}")
