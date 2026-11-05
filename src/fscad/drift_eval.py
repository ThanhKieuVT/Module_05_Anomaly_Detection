"""H3 offline: does the label-free drift monitor alarm before detection quality drops?

For each product, lighting is changed synthetically on its labelled test set with increasing
severity (``drift.levels``). At every level we measure

* I-AUROC on the whole relit test set (needs ground truth, hence ``test_public`` on MVTec AD 2),
* the drift monitor after it has seen a production-like stream of that level (normal images plus
  anomalies at ``drift.anomaly_rate``), continuing the sliding window from the previous level.

The summary gives the first severity that raises an alarm, the first severity where I-AUROC is
more than ``drift.delta`` below its clean value, and whether the monitor already fires on the
clean stream (false alarm).

    python -m fscad.drift_eval dataset=mvtec_ad2 'classes=[vial]' shots=4 drift.kind=mixed
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from fscad.data.datasets import list_samples, normalize
from fscad.data.photometric import relight
from fscad.metrics import image_auroc
from fscad.monitoring import DriftMonitor
from fscad.pipeline import artifact_path, dataset_root, fit_from_split, load_images, resolve_classes, setup
from fscad.product import ProductModel, activate
from fscad.scoring.masking import foreground_mask
from fscad.scoring.scoring import image_score
from fscad.tracking import Tracker
from fscad.utils import seed_everything

log = logging.getLogger("fscad.drift_eval")


@torch.no_grad()
def cls_and_scores(backbone, product: ProductModel, paths, cfg, device, kind: str, level: float):
    """Frozen CLS and anomaly scores of the images at one drift level."""
    activate(backbone, product)
    product.memory.to(device)
    cls, scores = [], []
    bs = cfg.eval.batch_size
    for i in range(0, len(paths), bs):
        x = relight(load_images(paths[i : i + bs], cfg.model.img_size), kind, level)
        frozen, adapted = backbone.forward_both(normalize(x.to(device)))
        cls.append(frozen.cls.cpu())
        d = product.memory.query(adapted.patches)
        if product.meta.get("masking"):  # same scoring as score_product
            d = d * foreground_mask(frozen.patches, frozen.grid).to(d.device, d.dtype)
        scores.append(image_score(d, cfg.eval.image_score, cfg.eval.top_q).cpu())
    product.memory.to("cpu")
    return torch.cat(cls), torch.cat(scores).numpy()


def stream_order(labels: np.ndarray, anomaly_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Indices of a production-like stream: all normal images plus anomalies at the given rate."""
    normal = np.flatnonzero(labels == 0)
    anom = np.flatnonzero(labels == 1)
    n_anom = min(len(anom), int(round(anomaly_rate / max(1e-9, 1 - anomaly_rate) * len(normal))))
    idx = np.r_[normal, rng.choice(anom, n_anom, replace=False)] if n_anom else normal
    return rng.permutation(idx)


def first_level(levels, flags) -> float | None:
    return next((lv for lv, f in zip(levels, flags, strict=True) if f), None)


def evaluate_class(cfg, backbone, device, dataset: str, cls: str, seed: int, tracker) -> tuple[list, dict]:
    dc = cfg.drift
    apath = artifact_path(cfg, dataset, cls, cfg.shots, seed)
    if cfg.reuse_artifacts and apath.exists():
        product = ProductModel.load(apath, "cpu")
    else:
        product = fit_from_split(cfg, backbone, device, dataset, cls, cfg.shots, seed)
    samples = list_samples(dataset, dataset_root(cfg, dataset), cls, "test")
    paths = [s.path for s in samples]
    labels = np.array([s.label for s in samples])
    prototype = F.normalize(product.cls_feats.float(), dim=-1).mean(0)
    monitor = DriftMonitor(
        prototype,
        product.val_cls,
        np.asarray(product.val_scores),
        window=dc.window,
        min_samples=dc.min_samples,
        alpha=dc.alpha,
    )
    rng = np.random.default_rng(seed)
    rows = []
    for level in dc.levels:
        cls_feats, scores = cls_and_scores(backbone, product, paths, cfg, device, dc.kind, level)
        for i in stream_order(labels, dc.anomaly_rate, rng):
            monitor.update(cls_feats[i : i + 1], float(scores[i]))
        st = monitor.status()
        rows.append(
            {
                "dataset": dataset,
                "class": cls,
                "kind": dc.kind,
                "level": float(level),
                "i_auroc": image_auroc(labels, scores),
                "drift": st.drift,
                "cls_p": st.cls_p,
                "score_p": st.score_p,
                "mean_similarity": st.mean_similarity,
            }
        )
    levels = [r["level"] for r in rows]
    clean = rows[0]["i_auroc"]
    alarm = first_level(levels[1:], [r["drift"] for r in rows[1:]])
    drop = first_level(levels, [r["i_auroc"] < clean - dc.delta for r in rows])
    summary = {
        "dataset": dataset,
        "class": cls,
        "kind": dc.kind,
        "clean_i_auroc": clean,
        "false_alarm_clean": bool(rows[0]["drift"]),
        "alarm_level": alarm,
        "drop_level": drop,
        "alarm_before_drop": alarm is not None and (drop is None or alarm < drop),
    }
    tracker.log_metrics(
        {k: float(v) for k, v in summary.items() if isinstance(v, (bool, int, float)) and v is not None}
    )
    return rows, summary


def run(cfg: DictConfig) -> tuple[list[dict], list[dict]]:
    tracker = Tracker.from_cfg(cfg)
    backbone, device = setup(cfg)
    dataset, seed = cfg.dataset.name, cfg.seeds[0]
    seed_everything(seed)
    all_rows, summaries = [], []
    for cls in resolve_classes(cfg):
        tags = {
            "phase": cfg.phase,
            "dataset": dataset,
            "class": cls,
            "shots": cfg.shots,
            "seed": seed,
            "method": f"{cfg.method}-drift-{cfg.drift.kind}",
        }
        with tracker.run(
            f"drift-{dataset}-{cls}-{cfg.drift.kind}", tags, OmegaConf.to_container(cfg, resolve=True)
        ):
            rows, summary = evaluate_class(cfg, backbone, device, dataset, cls, seed, tracker)
            tracker.log_dict({"levels": rows, "summary": summary}, "drift_eval.json")
        log.info("%s/%s %s", dataset, cls, json.dumps(summary))
        all_rows += rows
        summaries.append(summary)
    return all_rows, summaries


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rows, summaries = run(cfg)
    out = Path(HydraConfig.get().runtime.output_dir)
    for name, data in (("drift_levels.csv", rows), ("drift_summary.csv", summaries)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)


if __name__ == "__main__":
    main()
