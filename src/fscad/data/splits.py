"""Fixed few-shot splits: the same K normal images for every method, per (class, shots, seed).

Files are written to ``splits/<dataset>/<class>/k<shots>_s<seed>.json`` and tracked with DVC.
Each file also lists a disjoint set of normal images used only to calibrate thresholds
(router tau and the per-product score threshold); they are never used for training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from fscad.data.datasets import list_samples


def split_path(splits_dir: str | Path, dataset: str, cls: str, shots: int, seed: int) -> Path:
    return Path(splits_dir) / dataset / cls / f"k{shots}_s{seed}.json"


def make_split(dataset: str, root: str | Path, cls: str, shots: int, seed: int, n_val: int = 10) -> dict:
    train = sorted(s.path for s in list_samples(dataset, root, cls, "train"))
    if len(train) < shots:
        raise ValueError(f"{dataset}/{cls}: only {len(train)} train images, need {shots}")
    rng = random.Random(f"{dataset}/{cls}/{seed}")  # same order for every K -> nested splits
    order = train[:]
    rng.shuffle(order)
    val_pool = sorted(s.path for s in list_samples(dataset, root, cls, "validation"))
    if not val_pool:  # no official validation split: hold out from the tail of the train order
        val_pool = order[shots:]
    val = [p for p in val_pool if p not in set(order[:shots])][:n_val]
    root = Path(root)
    rel = lambda p: str(Path(p).relative_to(root))  # noqa: E731
    return {
        "dataset": dataset,
        "class": cls,
        "shots": shots,
        "seed": seed,
        "train": [rel(p) for p in order[:shots]],
        "val": [rel(p) for p in val],
    }


def write_split(split: dict, splits_dir: str | Path) -> Path:
    path = split_path(splits_dir, split["dataset"], split["class"], split["shots"], split["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))
    return path


def load_split(
    splits_dir: str | Path,
    dataset: str,
    cls: str,
    shots: int,
    seed: int,
    root: str | Path | None = None,
    create: bool = True,
) -> dict:
    path = split_path(splits_dir, dataset, cls, shots, seed)
    if not path.exists():
        if not create or root is None:
            raise FileNotFoundError(f"{path} (run scripts/make_splits.py)")
        write_split(make_split(dataset, root, cls, shots, seed), splits_dir)
    split = json.loads(path.read_text())
    if root is not None:
        split["train_paths"] = [str(Path(root) / p) for p in split["train"]]
        split["val_paths"] = [str(Path(root) / p) for p in split["val"]]
    return split
