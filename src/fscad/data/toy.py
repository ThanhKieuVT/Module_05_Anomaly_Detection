"""A tiny synthetic dataset in MVTec AD layout, for tests, CI and smoke runs (no download needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _texture(rng: np.random.Generator, size: int, hue: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = 0.5 + 0.25 * np.sin(2 * np.pi * (6 * xx + rng.uniform(-0.05, 0.05))) * np.cos(2 * np.pi * 6 * yy)
    img = base[..., None] * hue[None, None] + rng.normal(0, 0.03, (size, size, 3))
    return np.clip(img, 0, 1)


def make_toy_dataset(
    root: str | Path,
    classes: tuple[str, ...] = ("alpha", "beta"),
    size: int = 112,
    n_train: int = 12,
    n_test_good: int = 4,
    n_test_bad: int = 4,
    seed: int = 0,
) -> Path:
    root = Path(root)
    rng = np.random.default_rng(seed)
    hues = {c: rng.uniform(0.2, 1.0, 3) for c in classes}
    for c in classes:
        for sub in ("train/good", "test/good", "test/defect", "ground_truth/defect"):
            (root / c / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n_train):
            _save(_texture(rng, size, hues[c]), root / c / "train/good" / f"{i:03d}.png")
        for i in range(n_test_good):
            _save(_texture(rng, size, hues[c]), root / c / "test/good" / f"{i:03d}.png")
        for i in range(n_test_bad):
            img = _texture(rng, size, hues[c])
            mask = np.zeros((size, size), dtype=np.uint8)
            s = size // 5
            y, x = rng.integers(0, size - s, 2)
            img[y : y + s, x : x + s] = rng.uniform(0, 1, 3)
            mask[y : y + s, x : x + s] = 255
            _save(img, root / c / "test/defect" / f"{i:03d}.png")
            Image.fromarray(mask).save(root / c / "ground_truth/defect" / f"{i:03d}_mask.png")
    return root


def _save(img: np.ndarray, path: Path) -> None:
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
