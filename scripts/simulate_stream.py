"""Simulate a production line against the running API and record when drift alarms fire.

Phases (each ``--per-phase`` images):
  1. normal   : test images of the registered product under the original lighting
  2. lighting : the same stream with a lighting change (gamma + brightness), or real MVTec AD 2
                lighting variants via --drift-dir
  3. unknown  : lighting change, mixed with images of an unregistered product (--unknown-dir)

    python scripts/simulate_stream.py --api http://localhost:8000 \
        --images data/mvtec_ad2/vial/test_public/good --unknown-dir data/mvtec_ad2/can/test_public/good
Writes a CSV log and prints the time from drift onset to the first alarm.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageEnhance

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def images(folder: str) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)


def relight(img: Image.Image, gamma: float, brightness: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    img = Image.fromarray((np.clip(arr**gamma, 0, 1) * 255).astype(np.uint8))
    return ImageEnhance.Brightness(img).enhance(brightness)


def post(api: str, img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    r = requests.post(f"{api}/predict", files={"file": ("x.png", buf.getvalue(), "image/png")}, timeout=60)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--images", required=True)
    ap.add_argument(
        "--drift-dir", default=None, help="real drifted images (e.g. MVTec AD 2 lighting variants)"
    )
    ap.add_argument("--unknown-dir", default=None)
    ap.add_argument("--per-phase", type=int, default=150)
    ap.add_argument("--gamma", type=float, default=1.8)
    ap.add_argument("--brightness", type=float, default=0.6)
    ap.add_argument("--unknown-rate", type=float, default=0.2)
    ap.add_argument("--out", default="results/drift_stream.csv")
    a = ap.parse_args()
    rng = random.Random(0)
    base = images(a.images)
    drifted = images(a.drift_dir) if a.drift_dir else None
    unknown = images(a.unknown_dir) if a.unknown_dir else []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows, t_start = [], time.time()
    drift_onset = alarm_at = None
    for phase in ("normal", "lighting", "unknown"):
        if phase == "unknown" and not unknown:
            continue
        for _ in range(a.per_phase):
            src, kind = rng.choice(base), "known"
            if phase == "unknown" and rng.random() < a.unknown_rate:
                src, kind = rng.choice(unknown), "unknown"
            if phase != "normal" and drifted and kind == "known":
                img = Image.open(rng.choice(drifted))
            else:
                img = Image.open(src)
                if phase != "normal":
                    img = relight(img, a.gamma, a.brightness)
            res = post(a.api, img)
            now = time.time() - t_start
            if phase == "lighting" and drift_onset is None:
                drift_onset = now
            d = requests.get(f"{a.api}/drift", timeout=30).json()
            flagged = any(v["drift"] for v in d.values())
            if drift_onset is not None and flagged and alarm_at is None:
                alarm_at = now
            rows.append(
                {
                    "t": round(now, 3),
                    "phase": phase,
                    "kind": kind,
                    "product": res.get("product"),
                    "rejected": res.get("rejected"),
                    "score": res.get("score"),
                    "similarity": res.get("similarity"),
                    "drift_flag": flagged,
                }
            )
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_lighting = sum(r["phase"] == "lighting" for r in rows)
    if alarm_at is not None:
        idx = next(i for i, r in enumerate(rows) if r["drift_flag"] and r["phase"] != "normal")
        first_drift = next(i for i, r in enumerate(rows) if r["phase"] == "lighting")
        print(
            f"drift alarm after {alarm_at - drift_onset:.1f}s / {idx - first_drift} images "
            f"(of {n_lighting} drifted)"
        )
    else:
        print("no drift alarm raised")
    print(f"log: {a.out}")
