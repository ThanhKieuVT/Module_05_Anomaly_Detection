"""Shared helpers for the train / evaluate / continual entry points."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fscad.data.datasets import ADDataset, dataset_classes, list_samples, load_image
from fscad.data.splits import load_split
from fscad.metrics import all_metrics
from fscad.models.backbone import DinoBackbone, build_backbone
from fscad.product import ProductModel, fit_product, score_product
from fscad.utils import get_device, timer


def dataset_root(cfg, dataset: str | None = None) -> Path:
    return Path(cfg.paths.data_root) / (dataset or cfg.dataset.name)


def resolve_classes(cfg, dataset: str | None = None) -> list[str]:
    name = dataset or cfg.dataset.name
    if dataset is None and cfg.classes:
        return list(cfg.classes) if not isinstance(cfg.classes, str) else cfg.classes.split(",")
    return dataset_classes(name, dataset_root(cfg, name))


def load_images(paths: list[str], size: int) -> torch.Tensor:
    return torch.stack([load_image(p, size) for p in paths]) if paths else torch.empty(0, 3, size, size)


def artifact_path(cfg, dataset: str, cls: str, shots: int, seed: int) -> Path:
    return Path(cfg.paths.artifacts_dir) / cfg.run_name / dataset / cls / f"k{shots}_s{seed}.pt"


def setup(cfg) -> tuple[DinoBackbone, torch.device]:
    device = get_device(cfg.device)
    backbone = build_backbone(cfg).to(device)
    return backbone, device


def fit_from_split(
    cfg, backbone, device, dataset: str, cls: str, shots: int, seed: int, tracker=None
) -> ProductModel:
    root = dataset_root(cfg, dataset)
    split = load_split(cfg.paths.splits_dir, dataset, cls, shots, seed, root=root)
    img_size = cfg.model.img_size
    product = fit_product(
        f"{dataset}/{cls}",
        backbone,
        load_images(split["train_paths"], img_size),
        cfg,
        device,
        val_images=load_images(split["val_paths"], img_size),
        seed=seed,
        tracker=tracker,
    )
    product.meta.update({"dataset": dataset, "class": cls, "split": split["train"]})
    return product


def test_loader(cfg, dataset: str, cls: str) -> DataLoader:
    samples = list_samples(dataset, dataset_root(cfg, dataset), cls, "test")
    if cfg.eval.max_test_images:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(samples))[: cfg.eval.max_test_images]
        samples = [samples[i] for i in sorted(idx)]
    ds = ADDataset(samples, cfg.model.img_size, cfg.eval.mask_size)
    return DataLoader(ds, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=cfg.eval.num_workers)


def evaluate_product(cfg, backbone, device, product: ProductModel, dataset: str, cls: str) -> dict:
    labels, scores, masks, maps = [], [], [], []
    t: dict = {}
    n = 0
    for batch in test_loader(cfg, dataset, cls):
        with timer(t, "inference", device):
            s, m = score_product(backbone, product, batch["image"], cfg, device, with_maps=cfg.eval.pixel)
        labels.append(batch["label"].numpy())
        scores.append(s)
        n += len(s)
        if cfg.eval.pixel:
            masks.append(batch["mask"].numpy())
            maps.append(m)
    labels = np.concatenate(labels)
    scores = np.concatenate(scores)
    out = all_metrics(
        labels,
        scores,
        np.concatenate(masks) if masks else None,
        np.concatenate(maps) if maps else None,
        pixel=cfg.eval.pixel,
    )
    out["latency_ms_per_image"] = 1000 * t["inference"] / max(1, n)
    out["n_test"] = n
    return out


__all__ = [
    "artifact_path",
    "dataset_root",
    "evaluate_product",
    "fit_from_split",
    "load_images",
    "resolve_classes",
    "setup",
    "test_loader",
    "build_backbone",
]
