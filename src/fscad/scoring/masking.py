"""Zero-shot foreground masking of patch tokens, as in AnomalyDINO (Damm et al., 2025).

The first principal component of an image's patch tokens separates object from background on
centred objects. Patches with first PC > ``threshold`` are kept; if that keeps at most 35% of the
central crop, the sign is flipped. The mask is dilated and closed with a 3x3 kernel. Background
patches get anomaly score 0, so dust or stains on the conveyor or mat cannot raise an alarm.

Per-class defaults follow AnomalyDINO's "agnostic" preprocessing: on for every VisA class, and on
MVTec AD only for capsule, hazelnut, pill, screw and toothbrush (``masking: auto``).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

MVTEC_AD_MASKED = {"capsule", "hazelnut", "pill", "screw", "toothbrush"}


def masking_default(dataset: str, cls: str, setting) -> bool:
    """``setting``: true | false | 'auto' (AnomalyDINO's per-class choice)."""
    if isinstance(setting, bool):
        return setting
    if str(setting).lower() in ("true", "false"):
        return str(setting).lower() == "true"
    if setting != "auto":
        raise ValueError(f"eval.masking must be true, false or auto, got {setting!r}")
    if dataset == "visa":
        return True
    if dataset == "mvtec_ad":
        return cls in MVTEC_AD_MASKED
    return False


@torch.no_grad()
def foreground_mask(
    patches: torch.Tensor, grid: tuple[int, int], threshold: float = 10.0, border: float = 0.2
) -> torch.Tensor:
    """(B, N, D) patch tokens -> (B, N) bool, True = foreground (scored)."""
    out = []
    h, w = grid
    struct = np.ones((3, 3), dtype=bool)
    for x in patches.float().cpu():
        xc = x - x.mean(0, keepdim=True)
        _, _, v = torch.linalg.svd(xc, full_matrices=False)
        pc = (xc @ v[0]).numpy()
        # SVD fixes the sign arbitrarily; orient it like sklearn's PCA (largest |loading| positive)
        if v[0][v[0].abs().argmax()] < 0:
            pc = -pc
        m = (pc > threshold).reshape(h, w)
        c = m[int(h * border) : int(h * (1 - border)), int(w * border) : int(w * (1 - border))]
        if c.sum() <= c.size * 0.35:
            m = (-pc > threshold).reshape(h, w)
        m = ndimage.binary_dilation(m, structure=struct)
        m = ndimage.binary_closing(m, structure=struct)
        out.append(torch.from_numpy(m.reshape(-1)))
    return torch.stack(out)
