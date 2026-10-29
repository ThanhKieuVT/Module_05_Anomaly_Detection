"""Generate the synthetic toy dataset used by tests, CI and smoke runs."""

import argparse

from fscad.data.toy import make_toy_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/toy")
    ap.add_argument("--classes", nargs="+", default=["alpha", "beta", "gamma"])
    ap.add_argument("--size", type=int, default=112)
    a = ap.parse_args()
    print(make_toy_dataset(a.root, tuple(a.classes), size=a.size))
