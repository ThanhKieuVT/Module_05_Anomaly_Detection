"""Patch memory bank with cosine kNN scoring (PatchCore / AnomalyDINO style)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MemoryBank:
    def __init__(self, k: int = 1, device: str | torch.device = "cpu"):
        self.k = k
        self.device = torch.device(device)
        self.bank: torch.Tensor | None = None  # (M, D), L2-normalised

    def __len__(self) -> int:
        return 0 if self.bank is None else self.bank.shape[0]

    def add(self, feats: torch.Tensor) -> None:
        feats = F.normalize(feats.reshape(-1, feats.shape[-1]).float(), dim=-1).to(self.device)
        self.bank = feats if self.bank is None else torch.cat([self.bank, feats])

    def subsample(self, ratio: float | None = None, max_size: int | None = None, seed: int = 0) -> None:
        """Greedy coreset (k-center) subsampling, as in PatchCore."""
        if self.bank is None:
            return
        n = self.bank.shape[0]
        target = n
        if ratio is not None:
            target = min(target, max(1, int(n * ratio)))
        if max_size is not None:
            target = min(target, max_size)
        if target >= n:
            return
        self.bank = self.bank[greedy_coreset(self.bank, target, seed=seed)]

    @torch.no_grad()
    def query(self, feats: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Cosine distance to the k-th nearest neighbours (mean over k). feats: (..., D)."""
        if self.bank is None:
            raise RuntimeError("memory bank is empty")
        shape = feats.shape[:-1]
        q = F.normalize(feats.reshape(-1, feats.shape[-1]).float(), dim=-1).to(self.device)
        out = []
        for i in range(0, q.shape[0], chunk):
            sim = q[i : i + chunk] @ self.bank.T
            top = sim.topk(min(self.k, sim.shape[1]), dim=1).values
            out.append(1.0 - top.mean(dim=1))
        return torch.cat(out).reshape(shape)

    def to(self, device: str | torch.device) -> MemoryBank:
        self.device = torch.device(device)
        if self.bank is not None:
            self.bank = self.bank.to(self.device)
        return self

    def state_dict(self) -> dict:
        return {"k": self.k, "bank": None if self.bank is None else self.bank.cpu()}

    @classmethod
    def from_state_dict(cls, state: dict, device: str | torch.device = "cpu") -> MemoryBank:
        mb = cls(k=state["k"], device=device)
        mb.bank = None if state["bank"] is None else state["bank"].to(mb.device)
        return mb

    def nbytes(self) -> int:
        return 0 if self.bank is None else self.bank.numel() * self.bank.element_size()


@torch.no_grad()
def greedy_coreset(x: torch.Tensor, n: int, seed: int = 0, proj_dim: int = 128) -> torch.Tensor:
    """Return indices of an approximate k-center coreset of size n (random projection for speed)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if x.shape[1] > proj_dim:
        proj = torch.randn(x.shape[1], proj_dim, generator=g).to(x.device) / proj_dim**0.5
        z = x @ proj
    else:
        z = x
    idx = [int(torch.randint(len(z), (1,), generator=g))]
    dist = torch.cdist(z, z[idx[-1]][None]).squeeze(1)
    for _ in range(n - 1):
        nxt = int(dist.argmax())
        idx.append(nxt)
        dist = torch.minimum(dist, torch.cdist(z, z[nxt][None]).squeeze(1))
    return torch.tensor(idx, device=x.device)
