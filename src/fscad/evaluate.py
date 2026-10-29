"""Single-task few-shot evaluation (one adapter per class, oracle class identity).

    python -m fscad.evaluate dataset=mvtec_ad method=lora shots=4 seeds=[0,1,2]
    python -m fscad.evaluate -m dataset=mvtec_ad,visa method=knn,lora shots=1,2,4,8,16 seeds=[0,1,2]

``method=knn`` is the frozen DINOv2 + kNN baseline (in-house AnomalyDINO); ``method=lora`` trains
an adapter per class. Results go to MLflow and to ``results.csv`` in the Hydra output directory.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from fscad.pipeline import artifact_path, evaluate_product, fit_from_split, resolve_classes, setup
from fscad.product import ProductModel
from fscad.tracking import Tracker
from fscad.utils import seed_everything

log = logging.getLogger("fscad.evaluate")


def run(cfg: DictConfig) -> list[dict]:
    tracker = Tracker.from_cfg(cfg)
    backbone, device = setup(cfg)
    dataset = cfg.dataset.name
    rows = []
    for cls in resolve_classes(cfg):
        for seed in cfg.seeds:
            seed_everything(seed)
            tags = {
                "phase": cfg.phase,
                "dataset": dataset,
                "class": cls,
                "shots": cfg.shots,
                "seed": seed,
                "method": cfg.method,
            }
            with tracker.run(
                f"{cfg.method}-{dataset}-{cls}-k{cfg.shots}-s{seed}",
                tags,
                params=OmegaConf.to_container(cfg, resolve=True),
            ):
                apath = artifact_path(cfg, dataset, cls, cfg.shots, seed)
                if cfg.reuse_artifacts and apath.exists():
                    product = ProductModel.load(apath, device)
                else:
                    product = fit_from_split(cfg, backbone, device, dataset, cls, cfg.shots, seed, tracker)
                    if cfg.save_artifacts:
                        product.save(apath)
                metrics = evaluate_product(cfg, backbone, device, product, dataset, cls)
                metrics.update(
                    {
                        k: product.meta[k]
                        for k in ("fit_seconds", "train_seconds", "memory_size")
                        if k in product.meta
                    }
                )
                metrics["mb_per_class"] = product.nbytes() / 2**20
                tracker.log_metrics(metrics)
            row = {**tags, **metrics}
            rows.append(row)
            log.info(
                "%s/%s k=%d s=%d  I-AUROC=%.4f  AU-PRO=%s",
                dataset,
                cls,
                cfg.shots,
                seed,
                metrics["i_auroc"],
                f"{metrics.get('au_pro', float('nan')):.4f}",
            )
    return rows


def write_results(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for m in ("i_auroc", "p_auroc", "au_pro", "f1_max", "latency_ms_per_image"):
        vals = [r[m] for r in rows if m in r and r[m] == r[m]]
        if vals:
            summary[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps({k: round(v["mean"], 4) for k, v in summary.items()}))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rows = run(cfg)
    write_results(rows, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
