"""Controlled lighting changes for drift experiments.

MVTec AD 2 ``test_private_mixed`` has real lighting variation but no ground truth, and
``test_public`` (with ground truth) does not say which lighting each image was taken under. To
measure both when a drift alarm fires and when detection quality drops, lighting is changed
synthetically on ``test_public`` with a severity in [0, 1] (0 = unchanged).
"""

from __future__ import annotations

import torch

KINDS = ("brightness", "gamma", "color_temp", "mixed")


def relight(x: torch.Tensor, kind: str, severity: float) -> torch.Tensor:
    """x: (..., 3, H, W) in [0, 1]. At severity 1: brightness x0.5, gamma 2.0, warm shift of 20%."""
    s = float(severity)
    if kind not in KINDS:
        raise ValueError(f"unknown drift kind {kind}; expected one of {KINDS}")
    if kind in ("gamma", "mixed"):
        x = x.clamp(0, 1) ** (1.0 + s)
    if kind in ("brightness", "mixed"):
        x = x * (1.0 - 0.5 * s)
    if kind in ("color_temp", "mixed"):
        gain = torch.tensor([1.0 + 0.2 * s, 1.0, 1.0 - 0.2 * s], dtype=x.dtype, device=x.device)
        x = x * gain.view(3, 1, 1)
    return x.clamp(0, 1)
