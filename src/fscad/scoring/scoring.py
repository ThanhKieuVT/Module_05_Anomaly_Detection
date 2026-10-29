"""Turn patch-level kNN distances into image scores and pixel anomaly maps."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def image_score(patch_scores: torch.Tensor, mode: str = "top_mean", top_q: float = 0.01) -> torch.Tensor:
    """patch_scores: (B, N). 'max' or mean of the top-q fraction (AnomalyDINO uses top 1%)."""
    if mode == "max":
        return patch_scores.max(dim=1).values
    if mode == "top_mean":
        k = max(1, int(math.ceil(patch_scores.shape[1] * top_q)))
        return patch_scores.topk(k, dim=1).values.mean(dim=1)
    raise ValueError(f"unknown image score mode: {mode}")


def gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = max(1, int(4 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def gaussian_blur(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B, H, W)."""
    if sigma <= 0:
        return maps
    k = gaussian_kernel(sigma).to(maps.device)
    pad = len(k) // 2
    x = maps[:, None]
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), k.view(1, 1, -1, 1))
    return x[:, 0]


def anomaly_map(
    patch_scores: torch.Tensor, grid: tuple[int, int], out_size: tuple[int, int], sigma: float = 4.0
) -> torch.Tensor:
    """(B, N) patch scores -> (B, H, W) pixel map: bilinear upsampling + Gaussian smoothing."""
    b = patch_scores.shape[0]
    m = patch_scores.reshape(b, 1, grid[0], grid[1]).float()
    m = F.interpolate(m, size=out_size, mode="bilinear", align_corners=False)[:, 0]
    return gaussian_blur(m, sigma)
