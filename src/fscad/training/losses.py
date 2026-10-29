"""Compactness + margin loss for training a LoRA adapter on normal + synthetic patches.

    L = mean_{p in N} d(h_p, c_p) + lambda * mean_{p in A} max(0, gamma - d(h_p, c_p))

``d`` is cosine distance. Two choices for the reference c_p:

* ``center="knn"`` (default): c_p is the nearest normal patch from *another* image in the batch.
  This matches the kNN scoring used at test time and keeps different object parts apart.
* ``center="mean"``: c is the mean normal patch feature (the single-centre form in the report).
  Simpler, but on object classes it pulls all parts towards one point.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    loss: torch.Tensor
    compact: torch.Tensor
    margin: torch.Tensor
    spread: torch.Tensor  # mean pairwise cosine distance of normal patches (collapse monitor)


def _nn_distance(
    q: torch.Tensor, q_img: torch.Tensor, ref: torch.Tensor, ref_img: torch.Tensor
) -> torch.Tensor:
    """Cosine distance from each query patch to its nearest reference patch of a different image."""
    sim = q @ ref.T
    same = q_img[:, None] == ref_img[None, :]
    sim = sim.masked_fill(same, -2.0)
    return 1.0 - sim.max(dim=1).values


def compactness_margin_loss(
    feats: torch.Tensor,  # (B, N, D) adapted patch features
    labels: torch.Tensor,  # (B, N) 0 normal, 1 anomalous, -1 ignore
    image_ids: torch.Tensor,  # (B,) id of the source normal image (views of one shot share an id)
    margin: float = 0.5,
    lam: float = 1.0,
    center: str = "knn",
    max_ref: int = 8192,
) -> LossOutput:
    b, n, d = feats.shape
    z = F.normalize(feats, dim=-1).reshape(b * n, d)
    lab = labels.reshape(-1)
    pid = image_ids[:, None].expand(b, n).reshape(-1)

    normal = lab == 0
    anom = lab == 1
    zn, idn = z[normal], pid[normal]
    if zn.shape[0] > max_ref:  # bound memory of the similarity matrix
        sel = torch.randperm(zn.shape[0], device=z.device)[:max_ref]
        ref, ref_id = zn[sel], idn[sel]
    else:
        ref, ref_id = zn, idn

    if center == "mean":
        c = F.normalize(zn.mean(0, keepdim=True), dim=-1)
        d_norm = 1.0 - (zn @ c.T).squeeze(1)
        d_anom = 1.0 - (z[anom] @ c.T).squeeze(1)
    elif center == "knn":
        if torch.unique(ref_id).numel() < 2:
            raise ValueError("knn compactness needs normal patches from at least two images per batch")
        d_norm = _nn_distance(zn, idn, ref, ref_id)
        # anomalous patches are compared with all normal patches (their own image included)
        d_anom = 1.0 - (z[anom] @ ref.T).max(dim=1).values if anom.any() else z.new_zeros(0)
    else:
        raise ValueError(f"unknown center: {center}")

    compact = d_norm.mean()
    marg = F.relu(margin - d_anom).mean() if d_anom.numel() else z.new_zeros(())
    with torch.no_grad():
        k = min(1024, zn.shape[0])
        s = zn[torch.randperm(zn.shape[0], device=z.device)[:k]]
        spread = (1.0 - s @ s.T).mean()
    return LossOutput(compact + lam * marg, compact.detach(), marg.detach(), spread)
