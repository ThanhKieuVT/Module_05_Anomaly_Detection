"""One registered product = LoRA adapter + patch memory bank + CLS features + score threshold.

``fit_product`` is the whole onboarding step for a new product from K normal images:

1. (optional) train a LoRA adapter on the shots + synthetic anomalies,
2. build the patch memory bank from adapted features of the shots and their augmentations,
3. keep the frozen CLS features (router prototype) and a score threshold from validation normals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fscad.data import augment
from fscad.data.datasets import normalize
from fscad.data.synthetic import patch_labels, synthesize
from fscad.models.backbone import DinoBackbone
from fscad.models.lora import (
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    reset_lora,
)
from fscad.models.memory_bank import MemoryBank
from fscad.scoring.scoring import anomaly_map, image_score
from fscad.training.losses import compactness_margin_loss
from fscad.utils import log


@dataclass
class ProductModel:
    name: str
    lora: dict[str, torch.Tensor] | None
    memory: MemoryBank
    cls_feats: torch.Tensor  # (K', D) frozen CLS of shots (+ augmentations)
    score_threshold: float | None = None
    val_scores: list[float] = field(default_factory=list)
    val_cls: torch.Tensor | None = None  # frozen CLS of validation normals (router tau, drift reference)
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "lora": self.lora,
                "memory": self.memory.state_dict(),
                "cls_feats": self.cls_feats.cpu(),
                "score_threshold": self.score_threshold,
                "val_scores": self.val_scores,
                "val_cls": None if self.val_cls is None else self.val_cls.cpu(),
                "meta": self.meta,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> ProductModel:
        s = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            s["name"],
            s["lora"],
            MemoryBank.from_state_dict(s["memory"], device),
            s["cls_feats"],
            s["score_threshold"],
            s.get("val_scores", []),
            s.get("val_cls"),
            s.get("meta", {}),
        )

    def nbytes(self) -> int:
        lora = sum(v.numel() * v.element_size() for v in (self.lora or {}).values())
        return lora + self.memory.nbytes() + self.cls_feats.numel() * self.cls_feats.element_size()


def activate(backbone: DinoBackbone, product: ProductModel) -> None:
    """Load a product's adapter into the shared backbone."""
    if backbone.use_lora:
        if product.lora is None:
            reset_lora(backbone.vit)  # B = 0 -> identical to the frozen backbone
        else:
            load_lora_state_dict(backbone.vit, product.lora)


# --------------------------------------------------------------------------- training
def train_adapter(
    backbone: DinoBackbone, shots: torch.Tensor, cfg, device: torch.device, seed: int = 0, tracker=None
) -> dict:
    """shots: (K, 3, H, W) in [0, 1]. Returns training statistics."""
    tc = cfg.train
    g = torch.Generator().manual_seed(seed)
    reset_lora(backbone.vit)
    params = lora_parameters(backbone.vit)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, tc.steps))
    k = shots.shape[0]
    stats = {"loss": [], "compact": [], "margin": [], "spread": []}
    t0 = time.perf_counter()
    for step in range(tc.steps):
        # half normal views, half synthetic anomalies; at least two distinct normal views
        n_norm = max(2, tc.batch_size // 2)
        n_anom = max(1, tc.batch_size - n_norm)
        idx = torch.randint(k, (n_norm + n_anom,), generator=g)
        imgs, masks, ids = [], [], []
        for j, i in enumerate(idx.tolist()):
            view = augment.random_view(shots[i], list(cfg.memory.augment), g)
            if j < n_norm:
                imgs.append(view)
                masks.append(torch.zeros(view.shape[1:]))
                ids.append(j)  # each normal view counts as its own "image" for the kNN reference
            else:
                src = shots[int(torch.randint(k, (1,), generator=g))]
                a, m = synthesize(view, src, tc.synthetic, g)
                imgs.append(a)
                masks.append(m)
                ids.append(-1 - j)
        x = normalize(torch.stack(imgs).to(device))
        feats = backbone(x, adapted=True)
        labels = patch_labels(torch.stack(masks).to(device), feats.grid)
        if not tc.use_synthetic:
            labels[n_norm:] = -1
        out = compactness_margin_loss(
            feats.patches,
            labels,
            torch.tensor(ids, device=device),
            margin=tc.margin,
            lam=tc.lam,
            center=tc.center,
        )
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        sched.step()
        for key in stats:
            stats[key].append(float(getattr(out, key)))
        if tracker is not None and (step % tc.log_every == 0 or step == tc.steps - 1):
            tracker.log_metrics({f"train/{key}": v[-1] for key, v in stats.items()}, step=step)
    for p in params:
        p.requires_grad_(False)
    stats["train_seconds"] = time.perf_counter() - t0
    if stats["spread"] and stats["spread"][-1] < tc.collapse_spread:
        log.warning("normal patch features look collapsed (spread %.4f)", stats["spread"][-1])
    return stats


# --------------------------------------------------------------------------- fit / score
@torch.no_grad()
def extract(backbone: DinoBackbone, images: torch.Tensor, device: torch.device, batch_size: int = 8):
    """Frozen CLS and adapted patch features for (B, 3, H, W) images in [0, 1]."""
    cls, patches, grid = [], [], None
    for i in range(0, len(images), batch_size):
        x = normalize(images[i : i + batch_size].to(device))
        frozen, adapted = backbone.forward_both(x)
        cls.append(frozen.cls.cpu())
        patches.append(adapted.patches)
        grid = adapted.grid
    return torch.cat(cls), torch.cat(patches), grid


def fit_product(
    name: str,
    backbone: DinoBackbone,
    shots: torch.Tensor,
    cfg,
    device: torch.device,
    val_images: torch.Tensor | None = None,
    seed: int = 0,
    tracker=None,
) -> ProductModel:
    meta: dict = {"shots": int(shots.shape[0]), "seed": seed, "model": cfg.model.name}
    t0 = time.perf_counter()
    lora = None
    if backbone.use_lora and cfg.train.steps > 0:
        stats = train_adapter(backbone, shots, cfg, device, seed=seed, tracker=tracker)
        meta.update(
            {
                "train_seconds": stats["train_seconds"],
                "final_loss": stats["loss"][-1],
                "final_spread": stats["spread"][-1],
            }
        )
        lora = lora_state_dict(backbone.vit)
    elif backbone.use_lora:
        reset_lora(backbone.vit)

    views = augment.expand(shots, list(cfg.memory.augment))
    cls, patches, _ = extract(backbone, views, device, cfg.eval.batch_size)
    memory = MemoryBank(k=cfg.memory.k, device=device)
    memory.add(patches)
    if cfg.memory.coreset_ratio or cfg.memory.max_size:
        memory.subsample(cfg.memory.coreset_ratio, cfg.memory.max_size, seed=seed)
    product = ProductModel(name, lora, memory, cls[: shots.shape[0]], meta=meta)

    if val_images is not None and len(val_images):
        scores, _ = score_product(backbone, product, val_images, cfg, device, with_maps=False)
        product.val_cls = extract(backbone, val_images, device, cfg.eval.batch_size)[0]
        product.val_scores = scores.tolist()
        product.score_threshold = float(np.quantile(scores, cfg.eval.threshold_quantile))
    meta["fit_seconds"] = time.perf_counter() - t0
    meta["memory_size"] = len(memory)
    meta["bytes"] = product.nbytes()
    return product


@torch.no_grad()
def score_product(
    backbone: DinoBackbone,
    product: ProductModel,
    images: torch.Tensor,
    cfg,
    device: torch.device,
    with_maps: bool = True,
    map_size: int | None = None,
):
    """Image scores (B,) and optional pixel maps (B, S, S) for images known to be this product."""
    activate(backbone, product)
    ec = cfg.eval
    size = map_size or ec.mask_size
    scores, maps = [], []
    for i in range(0, len(images), ec.batch_size):
        x = normalize(images[i : i + ec.batch_size].to(device))
        feats = backbone(x, adapted=True)
        d = product.memory.query(feats.patches)
        scores.append(image_score(d, ec.image_score, ec.top_q).cpu())
        if with_maps:
            maps.append(anomaly_map(d, feats.grid, (size, size), ec.sigma).cpu())
    s = torch.cat(scores).numpy()
    return s, (torch.cat(maps).numpy() if with_maps else None)
