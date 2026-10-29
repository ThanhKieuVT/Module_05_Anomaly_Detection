"""Label-free drift detection for one product.

The reference is what the product looked like at onboarding: frozen CLS embeddings and anomaly
scores of its normal validation images. Over a sliding window of production predictions we track

* ``cls_similarity``: cosine similarity of each CLS embedding to the product prototype; a drop in
  this distribution (KS test) means the images no longer look like the training images
  (lighting, camera, new variant).
* ``score``: anomaly scores; a shift up without a real defect wave is the usual early symptom
  before detection quality falls.

Both tests use the two-sample Kolmogorov-Smirnov statistic. ``drift`` is raised when either
p-value falls below ``alpha`` once the window holds ``min_samples`` predictions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp


@dataclass
class DriftStatus:
    n: int
    cls_ks: float
    cls_p: float
    score_ks: float
    score_p: float
    mean_similarity: float
    drift: bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class DriftMonitor:
    def __init__(
        self,
        prototype: torch.Tensor,
        ref_cls: torch.Tensor,
        ref_scores: np.ndarray,
        window: int = 200,
        min_samples: int = 50,
        alpha: float = 0.01,
    ):
        self.prototype = F.normalize(prototype.float().reshape(-1), dim=0)
        self.ref_sim = self._sim(ref_cls) if ref_cls is not None and len(ref_cls) else np.array([])
        self.ref_scores = np.asarray(ref_scores, dtype=float)
        self.sims: deque[float] = deque(maxlen=window)
        self.scores: deque[float] = deque(maxlen=window)
        self.min_samples = min_samples
        self.alpha = alpha

    def _sim(self, cls: torch.Tensor) -> np.ndarray:
        z = F.normalize(cls.float().reshape(-1, self.prototype.numel()), dim=-1)
        return (z @ self.prototype).numpy()

    def update(self, cls: torch.Tensor, score: float) -> None:
        self.sims.extend(self._sim(cls).tolist())
        self.scores.append(float(score))

    def status(self) -> DriftStatus:
        n = len(self.scores)
        nan = float("nan")
        cls_ks = cls_p = score_ks = score_p = nan
        if n >= self.min_samples and len(self.ref_sim) >= 2:
            r = ks_2samp(np.asarray(self.sims), self.ref_sim)
            cls_ks, cls_p = float(r.statistic), float(r.pvalue)
        if n >= self.min_samples and len(self.ref_scores) >= 2:
            r = ks_2samp(np.asarray(self.scores), self.ref_scores)
            score_ks, score_p = float(r.statistic), float(r.pvalue)
        drift = (cls_p == cls_p and cls_p < self.alpha) or (score_p == score_p and score_p < self.alpha)
        mean_sim = float(np.mean(self.sims)) if self.sims else nan
        return DriftStatus(n, cls_ks, cls_p, score_ks, score_p, mean_sim, bool(drift))
