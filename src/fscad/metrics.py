"""Evaluation metrics: I-AUROC, P-AUROC, AU-PRO, F1-max, and continual-learning measures."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_curve, roc_auc_score

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    p, r, _ = precision_recall_curve(np.asarray(labels), np.asarray(scores))
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(f1.max())


def pixel_auroc(masks: np.ndarray, maps: np.ndarray, max_pixels: int = 5_000_000, seed: int = 0) -> float:
    y = np.asarray(masks).reshape(-1).astype(np.uint8)
    s = np.asarray(maps).reshape(-1)
    if y.min() == y.max():
        return float("nan")
    if y.size > max_pixels:  # uniform subsample keeps memory bounded; AUROC is unbiased under it
        idx = np.random.default_rng(seed).choice(y.size, max_pixels, replace=False)
        y, s = y[idx], s[idx]
    return float(roc_auc_score(y, s))


def au_pro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3, n_thresholds: int = 1000) -> float:
    """Area under the per-region-overlap curve up to ``fpr_limit``, normalised to [0, 1].

    PRO(t) is the mean over all connected ground-truth regions of the fraction of the region whose
    score is >= t; FPR(t) is computed on all normal pixels (Bergmann et al., MVTec AD).
    """
    masks = np.asarray(masks).astype(bool)
    maps = np.asarray(maps, dtype=np.float64)
    if not masks.any():
        return float("nan")
    normal_scores = np.sort(maps[~masks])
    region_scores = []
    for m, a in zip(masks, maps, strict=True):
        lab, n = ndimage.label(m)
        for r in range(1, n + 1):
            region_scores.append(np.sort(a[lab == r]))

    # sweep thresholds from high to low; ties are handled because every threshold is a real score
    all_scores = np.concatenate([normal_scores] + region_scores)
    thr = np.unique(np.quantile(all_scores, np.linspace(0.0, 1.0, n_thresholds)))[::-1]
    thr = np.concatenate([[np.inf], thr])
    x = 1.0 - np.searchsorted(normal_scores, thr, side="left") / normal_scores.size  # FPR(t)
    y = np.zeros_like(thr)
    for rs in region_scores:
        y += 1.0 - np.searchsorted(rs, thr, side="left") / rs.size  # overlap(t)
    y /= len(region_scores)
    # cut the curve at fpr_limit with linear interpolation
    cut = int(np.searchsorted(x, fpr_limit, side="right"))
    if cut < len(x):
        x0, x1, y0, y1 = x[cut - 1], x[cut], y[cut - 1], y[cut]
        y_lim = y0 + (y1 - y0) * (fpr_limit - x0) / (x1 - x0) if x1 > x0 else y1
        x, y = np.append(x[:cut], fpr_limit), np.append(y[:cut], y_lim)
    return float(_trapezoid(y, x) / fpr_limit)


def forgetting_measure(acc: np.ndarray) -> float:
    """acc[l, j]: performance on task j after learning task l (NaN where j > l).

    FM = 1/(T-1) * sum_{j<T} max_{l in [j, T-1)} (acc[l, j] - acc[T-1, j])   (Chaudhry et al., 2018)
    """
    acc = np.asarray(acc, dtype=float)
    t = acc.shape[0]
    if t < 2:
        return 0.0
    drops = []
    for j in range(t - 1):
        best = np.nanmax(acc[j : t - 1, j])
        drops.append(max(0.0, best - acc[t - 1, j]))
    return float(np.mean(drops))


def all_metrics(labels, scores, masks=None, maps=None, pixel: bool = True) -> dict[str, float]:
    out = {"i_auroc": image_auroc(labels, scores), "f1_max": f1_max(labels, scores)}
    if pixel and masks is not None and maps is not None:
        out["p_auroc"] = pixel_auroc(masks, maps)
        out["au_pro"] = au_pro(masks, maps)
    return out
