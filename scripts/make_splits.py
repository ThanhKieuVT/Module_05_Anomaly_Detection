"""Write the fixed few-shot splits (tracked by DVC).

    python scripts/make_splits.py --dataset mvtec_ad --data-root data --shots 1 2 4 8 16 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fscad.data.datasets import dataset_classes
from fscad.data.splits import make_split, write_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-val", type=int, default=10)
    args = ap.parse_args()
    root = Path(args.data_root) / args.dataset
    n = 0
    for cls in dataset_classes(args.dataset, root):
        for k in args.shots:
            for s in args.seeds:
                write_split(make_split(args.dataset, root, cls, k, s, n_val=args.n_val), args.splits_dir)
                n += 1
    print(f"wrote {n} splits to {args.splits_dir}/{args.dataset}")


if __name__ == "__main__":
    main()
