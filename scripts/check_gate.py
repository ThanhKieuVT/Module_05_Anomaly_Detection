"""Fail (exit 1) when an evaluation summary misses the promotion thresholds. Used by CI."""

import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--min-i-auroc", type=float, default=None)
    ap.add_argument("--min-au-pro", type=float, default=None)
    ap.add_argument("--max-latency-ms", type=float, default=None)
    a = ap.parse_args()
    s = json.loads(Path(a.summary).read_text())
    checks = [
        ("i_auroc", a.min_i_auroc, ">="),
        ("au_pro", a.min_au_pro, ">="),
        ("latency_ms_per_image", a.max_latency_ms, "<="),
    ]
    ok = True
    for key, thr, op in checks:
        if thr is None:
            continue
        v = s[key]["mean"]
        passed = v >= thr if op == ">=" else v <= thr
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} {key} = {v:.4f} ({op} {thr})")
    sys.exit(0 if ok else 1)
