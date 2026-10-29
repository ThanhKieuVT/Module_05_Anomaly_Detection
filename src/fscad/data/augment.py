"""Geometric augmentations applied to the few normal shots (training and memory bank)."""

from __future__ import annotations

import torch

AUGMENTATIONS = ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")


def apply(x: torch.Tensor, name: str) -> torch.Tensor:
    """x: (..., H, W)."""
    if name == "identity":
        return x
    if name == "hflip":
        return x.flip(-1)
    if name == "vflip":
        return x.flip(-2)
    if name.startswith("rot"):
        return torch.rot90(x, int(name[3:]) // 90, dims=(-2, -1))
    raise ValueError(f"unknown augmentation {name}")


def expand(images: torch.Tensor, names: list[str]) -> torch.Tensor:
    """(B, C, H, W) -> (B * len(names), C, H, W)."""
    names = ["identity"] + [n for n in names if n != "identity"]
    return torch.cat([apply(images, n) for n in names])


def random_view(img: torch.Tensor, names: list[str], g: torch.Generator) -> torch.Tensor:
    pool = ["identity"] + [n for n in names if n != "identity"]
    return apply(img, pool[int(torch.randint(len(pool), (1,), generator=g))])
