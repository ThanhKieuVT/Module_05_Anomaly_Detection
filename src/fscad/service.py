"""Multi-adapter inference service core (framework-free; FastAPI wraps it in serving/app.py).

One frozen backbone is loaded once. Products live on disk as ``<products_dir>/<id>/model.pt``;
at most ``cache_size`` of them (adapter + memory bank) are kept in memory, least recently used
first out. The router only needs the small CLS prototypes, so it always knows every product.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from fscad.data.datasets import normalize
from fscad.models.backbone import build_backbone
from fscad.models.router import PrototypeRouter
from fscad.monitoring import DriftMonitor
from fscad.product import ProductModel, activate, fit_product
from fscad.scoring.scoring import anomaly_map, image_score
from fscad.utils import get_device, sync


def product_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name)


@dataclass
class Prediction:
    product: str | None
    best_match: str
    similarity: float
    rejected: bool
    score: float | None = None
    threshold: float | None = None
    is_anomaly: bool | None = None
    anomaly_map: np.ndarray | None = None
    timings_ms: dict = field(default_factory=dict)
    cache_hit: bool | None = None

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "anomaly_map"}
        return d


class AnomalyService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.backbone = build_backbone(cfg).to(self.device)
        self.dir = Path(cfg.products_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.router = PrototypeRouter(n_prototypes=cfg.router.n_prototypes)
        self.cache: OrderedDict[str, ProductModel] = OrderedDict()
        self.thresholds: dict[str, float | None] = {}
        self.monitors: dict[str, DriftMonitor] = {}
        self.val_cls: dict[str, torch.Tensor] = {}
        self.lock = threading.Lock()  # one backbone, adapters are swapped in place
        self.stats = {"cache_hit": 0, "cache_miss": 0}
        self.reload()

    # ------------------------------------------------------------------ registry
    def reload(self) -> list[str]:
        """(Re)build the router from every product on disk."""
        self.router = PrototypeRouter(n_prototypes=self.cfg.router.n_prototypes)
        self.cache.clear()
        for path in sorted(self.dir.glob("*/model.pt")):
            self._register(ProductModel.load(path, "cpu"))
        self._calibrate()
        return self.router.classes

    def _register(self, p: ProductModel) -> None:
        self.router.register(p.name, p.cls_feats)
        self.thresholds[p.name] = p.score_threshold
        if p.val_cls is not None and len(p.val_cls):
            self.val_cls[p.name] = p.val_cls
        ref_cls = p.val_cls if p.val_cls is not None else p.cls_feats
        self.monitors[p.name] = DriftMonitor(
            self.router.prototypes[p.name][0],
            ref_cls,
            np.asarray(p.val_scores),
            self.cfg.drift.window,
            self.cfg.drift.min_samples,
        )

    def _calibrate(self) -> None:
        if self.val_cls:
            cls = torch.cat(list(self.val_cls.values()))
            labels = [n for n, v in self.val_cls.items() for _ in range(len(v))]
            self.router.calibrate(cls, labels, self.cfg.router.tau_quantile)

    def products(self) -> list[dict]:
        return [
            {"name": n, "threshold": self.thresholds.get(n), "cached": n in self.cache}
            for n in self.router.classes
        ]

    def _get(self, name: str) -> tuple[ProductModel, bool]:
        if name in self.cache:
            self.cache.move_to_end(name)
            self.stats["cache_hit"] += 1
            return self.cache[name], True
        self.stats["cache_miss"] += 1
        p = ProductModel.load(self.dir / product_id(name) / "model.pt", self.device)
        self.cache[name] = p
        while len(self.cache) > self.cfg.cache_size:
            self.cache.popitem(last=False)
        return p, False

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(
        self, image: torch.Tensor, return_map: bool = False, product: str | None = None
    ) -> Prediction:
        """image: (3, H, W) in [0, 1] at the model resolution. ``product`` skips routing (oracle)."""
        t: dict[str, float] = {}

        def tick(key, t0):
            sync(self.device)
            t[key] = 1000 * (time.perf_counter() - t0)

        with self.lock:
            t0 = time.perf_counter()
            x = normalize(image[None].to(self.device))
            h, grid = self.backbone.forward_prefix(x)
            cls = self.backbone.forward_suffix(h, adapted=False)[:, 0]
            rr = self.router.route(cls.cpu())
            tick("route", t0)
            sim, best, rejected = float(rr.similarity[0]), rr.best[0], bool(rr.rejected[0])
            target = product or (None if rejected else best)
            if target is None:
                return Prediction(None, best, sim, True, timings_ms=t)

            t0 = time.perf_counter()
            p, hit = self._get(target)
            activate(self.backbone, p)
            tick("adapter", t0)

            t0 = time.perf_counter()
            tokens = self.backbone.forward_suffix(h, adapted=True)
            patches = tokens[:, self.backbone.n_prefix :]
            d = p.memory.query(patches)
            score = float(image_score(d, self.cfg.eval.image_score, self.cfg.eval.top_q)[0])
            amap = None
            if return_map:
                amap = anomaly_map(d, grid, tuple(image.shape[1:]), self.cfg.eval.sigma)[0].cpu().numpy()
            tick("score", t0)

        t["total"] = sum(t.values())
        thr = self.thresholds.get(target)
        self.monitors[target].update(cls.cpu(), score)
        return Prediction(
            target,
            best,
            sim,
            rejected and product is None,
            score,
            thr,
            None if thr is None else score > thr,
            amap,
            t,
            hit,
        )

    # ------------------------------------------------------------------ onboarding
    def onboard(self, name: str, shots: torch.Tensor, val: torch.Tensor | None = None) -> dict:
        """Register a new product from a few normal images (B, 3, H, W) in [0, 1]."""
        cfg = OmegaConf.merge(self.cfg, {"train": {"steps": self.cfg.onboard_steps}})
        t0 = time.perf_counter()
        with self.lock:
            if val is None or len(val) == 0:  # no extra images: reuse the shots for the threshold
                val = shots
            p = fit_product(name, self.backbone, shots, cfg, self.device, val_images=val)
            p.memory.to("cpu")
            path = p.save(self.dir / product_id(name) / "model.pt")
            (path.parent / "meta.json").write_text(
                json.dumps({"name": name, **p.meta}, default=str, indent=2)
            )
            self.cache.pop(name, None)
            self._register(p)
            self._calibrate()
        return {
            "name": name,
            "seconds": time.perf_counter() - t0,
            "bytes": p.nbytes(),
            "threshold": p.score_threshold,
            "tau": self.router.tau,
            **{k: v for k, v in p.meta.items() if isinstance(v, int | float)},
        }

    def drift(self) -> dict[str, dict]:
        return {n: m.status().as_dict() for n, m in self.monitors.items()}
