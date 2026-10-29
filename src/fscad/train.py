"""Train (onboard) one product: adapter + memory bank + prototype, saved as a single artifact.

    python -m fscad.train dataset=mvtec_ad classes=[bottle] shots=4 seeds=[0]

The artifact is written to ``artifacts/<run_name>/<dataset>/<class>/k<shots>_s<seed>.pt``. Use
``scripts/register_model.py`` to push it to the MLflow Model Registry.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from fscad.pipeline import artifact_path, fit_from_split, resolve_classes, setup
from fscad.tracking import Tracker
from fscad.utils import seed_everything

log = logging.getLogger("fscad.train")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    tracker = Tracker.from_cfg(cfg)
    backbone, device = setup(cfg)
    dataset = cfg.dataset.name
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
                "stage": "train",
            }
            with tracker.run(
                f"train-{dataset}-{cls}-k{cfg.shots}-s{seed}",
                tags,
                params=OmegaConf.to_container(cfg, resolve=True),
            ):
                product = fit_from_split(cfg, backbone, device, dataset, cls, cfg.shots, seed, tracker)
                path = product.save(artifact_path(cfg, dataset, cls, cfg.shots, seed))
                tracker.log_metrics({k: v for k, v in product.meta.items() if isinstance(v, int | float)})
                tracker.log_artifact(path)
            log.info(
                "saved %s (%.2f MB, fit %.1fs)", path, product.nbytes() / 2**20, product.meta["fit_seconds"]
            )


if __name__ == "__main__":
    main()
