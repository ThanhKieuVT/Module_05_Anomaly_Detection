"""Latency / memory benchmark of the serving path (route + adapter swap + score).

    python scripts/benchmark_latency.py --products-dir artifacts/serving --n 200 --overrides "model=dinov2_vits14"
Reports p50/p95 per stage for cache hits and misses, and peak GPU memory.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from fscad.service import AnomalyService

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--products-dir", default="artifacts/serving")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cache-size", type=int, default=4)
    ap.add_argument("--overrides", default="")
    a = ap.parse_args()
    ov = shlex.split(a.overrides) + [f"products_dir={a.products_dir}", f"cache_size={a.cache_size}"]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="serve", overrides=ov)
    svc = AnomalyService(cfg)
    names = svc.router.classes
    if not names:
        raise SystemExit("no products in " + a.products_dir)
    size = cfg.model.img_size
    img = torch.rand(3, size, size)
    for _ in range(5):  # warm-up
        svc.predict(img, product=names[0])
    rows = {"hit": [], "miss": []}
    rng = random.Random(0)
    for _ in range(a.n):
        p = svc.predict(img, product=rng.choice(names))
        rows["hit" if p.cache_hit else "miss"].append(p.timings_ms)
    out = {}
    for kind, ts in rows.items():
        if not ts:
            continue
        out[kind] = {
            k: {
                "p50": float(np.percentile([t[k] for t in ts], 50)),
                "p95": float(np.percentile([t[k] for t in ts], 95)),
            }
            for k in ts[0]
        }
        out[kind]["n"] = len(ts)
    if torch.cuda.is_available():
        out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    print(json.dumps(out, indent=2))
