"""Prototype router with rejection.

Each registered product keeps one (or a few, via k-means) L2-normalised prototypes of the frozen
CLS token. A test image goes to the product with the highest cosine similarity, or is rejected as
an unregistered product when that similarity is below the threshold tau.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RouteResult:
    names: list[str | None]  # None when rejected
    best: list[str]  # argmax class even when rejected (for oracle comparison)
    similarity: torch.Tensor  # (B,) best cosine similarity
    rejected: torch.Tensor  # (B,) bool


class PrototypeRouter:
    def __init__(self, n_prototypes: int = 1, tau: float | None = None, seed: int = 0):
        self.n_prototypes = n_prototypes
        self.tau = tau
        self.seed = seed
        self.prototypes: dict[str, torch.Tensor] = {}  # name -> (P, D)

    def __len__(self) -> int:
        return len(self.prototypes)

    @property
    def classes(self) -> list[str]:
        return list(self.prototypes)

    def register(self, name: str, cls_feats: torch.Tensor) -> None:
        z = F.normalize(cls_feats.float().cpu(), dim=-1)
        k = min(self.n_prototypes, len(z))
        protos = z.mean(0, keepdim=True) if k == 1 else _kmeans(z, k, self.seed)
        self.prototypes[name] = F.normalize(protos, dim=-1)

    def unregister(self, name: str) -> None:
        self.prototypes.pop(name, None)

    def similarities(self, cls_feats: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        """(B, C) max-over-prototypes cosine similarity to every registered class."""
        if not self.prototypes:
            raise RuntimeError("no class registered in the router")
        z = F.normalize(cls_feats.float().cpu(), dim=-1)
        names = self.classes
        sims = torch.stack([(z @ self.prototypes[n].T).max(dim=1).values for n in names], dim=1)
        return sims, names

    def route(self, cls_feats: torch.Tensor) -> RouteResult:
        sims, names = self.similarities(cls_feats)
        best_sim, best_idx = sims.max(dim=1)
        best = [names[i] for i in best_idx.tolist()]
        rejected = (
            (best_sim < self.tau) if self.tau is not None else torch.zeros_like(best_sim, dtype=torch.bool)
        )
        routed = [None if r else n for n, r in zip(best, rejected.tolist(), strict=True)]
        return RouteResult(routed, best, best_sim, rejected)

    def calibrate(self, cls_feats: torch.Tensor, labels: list[str], quantile: float = 0.01) -> float:
        """Set tau to the given quantile of correct-class similarity on normal validation images."""
        sims, names = self.similarities(cls_feats)
        idx = torch.tensor([names.index(lbl) for lbl in labels])
        own = sims[torch.arange(len(labels)), idx].numpy()
        self.tau = float(np.quantile(own, quantile))
        return self.tau

    def state_dict(self) -> dict:
        return {"n_prototypes": self.n_prototypes, "tau": self.tau, "prototypes": dict(self.prototypes)}

    @classmethod
    def from_state_dict(cls, state: dict) -> PrototypeRouter:
        r = cls(n_prototypes=state["n_prototypes"], tau=state["tau"])
        r.prototypes = dict(state["prototypes"])
        return r


def _kmeans(z: torch.Tensor, k: int, seed: int, iters: int = 50) -> torch.Tensor:
    """Spherical k-means on unit vectors."""
    g = torch.Generator().manual_seed(seed)
    c = z[torch.randperm(len(z), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = (z @ c.T).argmax(dim=1)
        new = torch.stack([z[assign == j].mean(0) if (assign == j).any() else c[j] for j in range(k)])
        new = F.normalize(new, dim=-1)
        if torch.allclose(new, c, atol=1e-6):
            break
        c = new
    return c
