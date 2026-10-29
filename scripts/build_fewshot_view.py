"""Build an MVTec-layout folder of symlinks that holds exactly the K shots of one split.

The official AnomalyDINO and PatchCore code read ``<root>/<class>/train/good``. Pointing them at
this view makes every baseline use the same K images as our methods (and converts VisA / MVTec AD 2
to the MVTec layout on the way).

    python scripts/build_fewshot_view.py --dataset mvtec_ad --shots 4 --seed 0 --out data_views
    -> data_views/mvtec_ad/k4_s0/<class>/{train/good, test/<defect>, ground_truth/<defect>}
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fscad.data.datasets import dataset_classes, list_samples
from fscad.data.splits import load_split


def link(src: str | Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(Path(src).resolve(), dst)


def build(dataset: str, data_root: Path, splits_dir: Path, shots: int, seed: int, out: Path) -> Path:
    root = data_root / dataset
    view = out / dataset / f"k{shots}_s{seed}"
    for cls in dataset_classes(dataset, root):
        split = load_split(splits_dir, dataset, cls, shots, seed, root=root)
        for i, p in enumerate(split["train_paths"]):
            link(p, view / cls / "train" / "good" / f"{i:03d}{Path(p).suffix.lower()}")
        for j, s in enumerate(list_samples(dataset, root, cls, "test")):
            name = f"{j:04d}{Path(s.path).suffix.lower()}"
            link(s.path, view / cls / "test" / s.defect / name)
            if s.mask:
                link(s.mask, view / cls / "ground_truth" / s.defect / f"{Path(name).stem}_mask.png")
    return view


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="data_views")
    a = ap.parse_args()
    print(build(a.dataset, Path(a.data_root), Path(a.splits_dir), a.shots, a.seed, Path(a.out)))
