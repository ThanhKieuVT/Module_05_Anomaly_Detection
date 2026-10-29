"""Synthetic anomalies generated from normal images only.

* Perlin: a thresholded Perlin-noise mask (as in DRAEM) blended with a texture. DRAEM takes the
  texture from DTD; here it is a colour-jittered, shuffled copy of another normal image so the
  method needs no external data.
* CutPaste: a rectangle cut from the image, jittered, and pasted at another position.

All functions take and return images in [0, 1] with shape (3, H, W) and a mask (H, W) in {0, 1}.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _fade(t: torch.Tensor) -> torch.Tensor:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def rand_perlin_2d(shape: tuple[int, int], res: tuple[int, int], g: torch.Generator) -> torch.Tensor:
    """Classic gradient Perlin noise; shape must be divisible by res."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1]), indexing="ij"
            ),
            dim=-1,
        )
        % 1
    )
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, generator=g)
    grads = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile(s0, s1):
        return grads[s0[0] : s0[1], s1[0] : s1[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)

    def dot(g_, sx, sy):
        return (torch.stack((grid[..., 0] + sx, grid[..., 1] + sy), dim=-1) * g_[: shape[0], : shape[1]]).sum(
            -1
        )

    n00 = dot(tile([0, -1], [0, -1]), 0, 0)
    n10 = dot(tile([1, None], [0, -1]), -1, 0)
    n01 = dot(tile([0, -1], [1, None]), 0, -1)
    n11 = dot(tile([1, None], [1, None]), -1, -1)
    t = _fade(grid[: shape[0], : shape[1]])
    lerp = torch.lerp
    return math.sqrt(2) * lerp(lerp(n00, n10, t[..., 0]), lerp(n01, n11, t[..., 0]), t[..., 1])


def perlin_mask(
    size: int, g: torch.Generator, threshold: float = 0.5, min_frac: float = 0.005, max_frac: float = 0.3
) -> torch.Tensor:
    """Binary blob mask covering a reasonable fraction of the image."""
    work = 256  # generate at a power of two, then resize
    for _ in range(20):
        sx = 2 ** int(torch.randint(0, 6, (1,), generator=g))
        sy = 2 ** int(torch.randint(0, 6, (1,), generator=g))
        noise = rand_perlin_2d((work, work), (sx, sy), g)
        mask = (noise > threshold).float()
        frac = mask.mean().item()
        if min_frac <= frac <= max_frac:
            break
    mask = F.interpolate(mask[None, None], size=(size, size), mode="nearest")[0, 0]
    return mask


def color_jitter(x: torch.Tensor, g: torch.Generator, strength: float = 0.4) -> torch.Tensor:
    b = 1 + (torch.rand(1, generator=g).item() * 2 - 1) * strength
    c = 1 + (torch.rand(1, generator=g).item() * 2 - 1) * strength
    shift = (torch.rand(3, 1, 1, generator=g) * 2 - 1) * strength * 0.5
    mean = x.mean(dim=(1, 2), keepdim=True)
    return ((x - mean) * c + mean * b + shift).clamp(0, 1)


def perlin_anomaly(
    img: torch.Tensor, source: torch.Tensor, g: torch.Generator, beta_range: tuple[float, float] = (0.2, 0.8)
) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = img.shape
    mask = perlin_mask(h, g)
    tex = color_jitter(source, g, 0.6)
    # scramble the texture spatially so it is not a copy of normal structure
    k = int(torch.randint(1, 4, (1,), generator=g))
    tex = torch.rot90(tex, k, dims=(1, 2))
    if tex.shape[1:] != (h, w):
        tex = F.interpolate(tex[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    beta = beta_range[0] + (beta_range[1] - beta_range[0]) * torch.rand(1, generator=g).item()
    out = img * (1 - mask) + ((1 - beta) * tex + beta * img) * mask
    return out.clamp(0, 1), mask


def cutpaste_anomaly(
    img: torch.Tensor,
    g: torch.Generator,
    area: tuple[float, float] = (0.02, 0.15),
    aspect: tuple[float, float] = (0.3, 3.3),
) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = img.shape
    a = (area[0] + (area[1] - area[0]) * torch.rand(1, generator=g).item()) * h * w
    log_r = (
        math.log(aspect[0]) + (math.log(aspect[1]) - math.log(aspect[0])) * torch.rand(1, generator=g).item()
    )
    r = math.exp(log_r)
    ph = max(2, min(h - 1, int(round(math.sqrt(a * r)))))
    pw = max(2, min(w - 1, int(round(math.sqrt(a / r)))))
    y0 = int(torch.randint(0, h - ph + 1, (1,), generator=g))
    x0 = int(torch.randint(0, w - pw + 1, (1,), generator=g))
    patch = color_jitter(img[:, y0 : y0 + ph, x0 : x0 + pw], g, 0.2)
    y1 = int(torch.randint(0, h - ph + 1, (1,), generator=g))
    x1 = int(torch.randint(0, w - pw + 1, (1,), generator=g))
    out = img.clone()
    out[:, y1 : y1 + ph, x1 : x1 + pw] = patch
    mask = torch.zeros(h, w)
    mask[y1 : y1 + ph, x1 : x1 + pw] = 1
    return out, mask


def synthesize(img: torch.Tensor, source: torch.Tensor, kind: str, g: torch.Generator):
    if kind == "perlin":
        return perlin_anomaly(img, source, g)
    if kind == "cutpaste":
        return cutpaste_anomaly(img, g)
    if kind == "mixed":
        return synthesize(img, source, "perlin" if torch.rand(1, generator=g).item() < 0.5 else "cutpaste", g)
    raise ValueError(f"unknown synthetic anomaly kind: {kind}")


def patch_labels(mask: torch.Tensor, grid: tuple[int, int], pos: float = 0.3) -> torch.Tensor:
    """(B, H, W) pixel masks -> (B, N) patch labels: 1 anomalous, 0 normal, -1 ignored (boundary)."""
    frac = F.adaptive_avg_pool2d(mask[:, None].float(), grid)[:, 0].flatten(1)
    lab = torch.full_like(frac, -1.0)
    lab[frac == 0] = 0
    lab[frac >= pos] = 1
    return lab
