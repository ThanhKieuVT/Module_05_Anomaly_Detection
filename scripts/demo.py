"""End-to-end demo of the inference service on generated toy data (no dataset download).

Starts the FastAPI app, onboards two products from 4 normal images each, then sends good and
defective images of both products and images of a third, unregistered product:

    python scripts/demo.py                       # real DINOv2 ViT-S/14 (weights from the HF hub)
    python scripts/demo.py --overrides "model=tiny onboard_steps=5"   # seconds, random weights

Prints one line per prediction (routed product, nearest product and CLS similarity, score,
decision), the routing and detection summary, and writes the anomaly map of one defective image
to <out>/anomaly_map.png. A rejected image is an ALERT (the ``alarm`` policy of the report): it is
not scored, a person has to look at it. Toy defects cover 20% of the image and shift the CLS token a
lot, so the router tends to reject them; this is the failure mode the split rejection rates of
fscad.continual measure. Toy numbers only show that the system works end to end.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from fscad.data.toy import make_toy_dataset

ROOT = Path(__file__).resolve().parents[1]


def wait_ready(api: str, proc: subprocess.Popen, timeout: float = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            sys.exit("server exited during startup")
        try:
            return requests.get(f"{api}/health", timeout=2).json()
        except requests.ConnectionError:
            time.sleep(1)
    sys.exit("server did not start in time")


def files(field: str, paths: list[Path]) -> list[tuple]:
    return [(field, (p.name, p.read_bytes(), "image/png")) for p in paths]


def save_overlay(img_path: Path, map_b64: str, out: Path) -> None:
    img = Image.open(img_path).convert("RGB")
    m = Image.open(io.BytesIO(base64.b64decode(map_b64))).convert("L").resize(img.size)
    heat = np.asarray(m, dtype=np.float32) / 255.0
    red = np.zeros((*heat.shape, 3), dtype=np.float32)
    red[..., 0] = heat
    blend = 0.6 * np.asarray(img, dtype=np.float32) / 255.0 + 0.4 * red
    side = np.concatenate([np.asarray(img) / 255.0, np.clip(blend, 0, 1)], axis=1)
    Image.fromarray((side * 255).astype(np.uint8)).resize((side.shape[1] * 2, side.shape[0] * 2)).save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--overrides", default="onboard_steps=50", help="Hydra overrides for configs/serve.yaml")
    ap.add_argument("--out", default="outputs/demo")
    a = ap.parse_args()
    api = f"http://127.0.0.1:{a.port}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="fscad_demo_"))
    data = make_toy_dataset(tmp / "toy", classes=("alpha", "beta", "gamma"), size=224, n_train=8)
    env = {
        **os.environ,
        "FSCAD_PRODUCTS_DIR": str(tmp / "products"),
        "FSCAD_OVERRIDES": a.overrides,
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "serving.app:app",
        "--port",
        str(a.port),
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    try:
        health = wait_ready(api, proc)
        print(f"service up on {health['device']}")
        tau = None

        for name in ("alpha", "beta"):
            train = sorted((data / name / "train/good").glob("*.png"))
            r = requests.post(
                f"{api}/onboard",
                data={"name": f"toy/{name}"},
                files=files("files", train[:4]) + files("val_files", train[4:]),
                timeout=600,
            ).json()
            print(f"onboarded toy/{name} from 4 shots in {r['seconds']:.1f}s")
        tau = requests.get(f"{api}/health", timeout=10).json()["tau"]
        print(f"router tau = {tau:.4f}")

        print(f"\n{'image':<20}{'routed to':<11}{'nearest':<11}{'cos':>7}{'score':>8}{'oracle':>8}  decision")
        rows = []
        for name in ("alpha", "beta", "gamma"):
            for kind in ("good", "defect"):
                if name == "gamma" and kind == "defect":
                    continue
                for p in sorted((data / name / "test" / kind).glob("*.png")):
                    r = requests.post(f"{api}/predict", files=files("file", [p]), timeout=120).json()
                    # oracle: force the true product, to separate routing errors from detection errors
                    o = None
                    if name != "gamma":
                        o = requests.post(
                            f"{api}/predict",
                            files=files("file", [p]),
                            params={"product": f"toy/{name}", "return_map": kind == "defect"},
                            timeout=120,
                        ).json()
                        r["oracle_anomaly"] = o["is_anomaly"]
                        if (
                            kind == "defect"
                            and o.get("anomaly_map_png")
                            and not (out / "anomaly_map.png").exists()
                        ):
                            save_overlay(p, o["anomaly_map_png"], out / "anomaly_map.png")
                    if r["rejected"]:
                        decision = "ALERT: rejected, unknown product"
                    else:
                        decision = "ALERT: anomaly" if r["is_anomaly"] else "ok"
                    score = f"{r['score']:.4f}" if r.get("score") is not None else "-"
                    oracle = f"{o['score']:.4f}" if o else "-"
                    print(
                        f"{name + '/' + kind + '/' + p.stem:<20}{str(r['product']):<11}"
                        f"{r['best_match']:<11}{r['similarity']:>7.4f}{score:>8}{oracle:>8}  {decision}"
                    )
                    rows.append((name, kind, r))

        known = [(n, k, r) for n, k, r in rows if n != "gamma"]
        route_ok = np.mean([r["product"] == f"toy/{n}" for n, _, r in known])
        unk_rej = np.mean([r["rejected"] for n, _, r in rows if n == "gamma"])
        nearest_ok = np.mean([r["best_match"] == f"toy/{n}" for n, _, r in known])
        rej_good = np.mean([r["rejected"] for _, k, r in known if k == "good"])
        rej_bad = np.mean([r["rejected"] for _, k, r in known if k == "defect"])
        alert = [(k == "defect", r["rejected"] or bool(r["is_anomaly"])) for _, k, r in known]
        print(
            f"\nknown products: routed correctly {route_ok:.2f}, nearest product correct {nearest_ok:.2f}, "
            f"rejected good {rej_good:.2f} / defective {rej_bad:.2f}"
        )
        print(f"unregistered product rejected: {unk_rej:.2f}")
        oracle = [(k == "defect", r["oracle_anomaly"]) for _, k, r in known]
        print(
            "oracle routing (true product forced): "
            f"defects flagged {np.mean([p for y, p in oracle if y]):.2f}, "
            f"good images flagged {np.mean([p for y, p in oracle if not y]):.2f}"
        )
        print(
            "alerts (anomaly or rejection) on known products: "
            f"defects caught {np.mean([p for y, p in alert if y]):.2f}, "
            f"false alerts on good images {np.mean([p for y, p in alert if not y]):.2f}"
        )
        drift = requests.get(f"{api}/drift", timeout=30).json()
        for prod, st in drift.items():
            print(f"drift {prod}: n={st['n']} drift={st['drift']} (monitor needs min_samples before testing)")
        if (out / "anomaly_map.png").exists():
            print(f"anomaly map of a defective image: {out / 'anomaly_map.png'}")
        print(f"API docs while running: {api}/docs")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()
