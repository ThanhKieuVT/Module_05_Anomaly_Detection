"""FastAPI front-end for the multi-adapter anomaly detection service.

    uvicorn serving.app:app --host 0.0.0.0 --port 8000

Environment:
    FSCAD_PRODUCTS_DIR   where products live (default artifacts/serving)
    FSCAD_OVERRIDES      extra Hydra overrides for configs/serve.yaml, e.g. "model=dinov2_vitb14 device=cuda"

Endpoints:
    POST /predict   image -> product, rejection, anomaly score, decision (+ optional map as PNG/base64)
    POST /onboard   name + a few normal images -> new adapter registered in place
    GET  /products  registered products
    GET  /drift     drift statistics per product
    POST /reload    rescan the products directory (after pulling new versions from the registry)
    GET  /metrics   Prometheus metrics
"""

from __future__ import annotations

import base64
import io
import os
import shlex
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from hydra import compose, initialize_config_dir
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from fscad.service import AnomalyService

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

PRED_LATENCY = Histogram(
    "fscad_latency_seconds",
    "Latency per stage",
    ["stage"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)
PREDICTIONS = Counter("fscad_predictions_total", "Predictions", ["product", "decision"])
REJECTIONS = Counter("fscad_rejections_total", "Images rejected as unknown product")
CACHE = Counter("fscad_adapter_cache_total", "Adapter cache lookups", ["result"])
SCORES = Histogram(
    "fscad_anomaly_score", "Anomaly score", ["product"], buckets=tuple(np.round(np.linspace(0.0, 1.0, 21), 2))
)
SIMILARITY = Histogram(
    "fscad_router_similarity",
    "Best router cosine similarity",
    buckets=tuple(np.round(np.linspace(0.0, 1.0, 21), 2)),
)
DRIFT_CLS = Gauge("fscad_drift_cls_ks", "KS statistic of CLS similarity vs reference", ["product"])
DRIFT_SCORE = Gauge("fscad_drift_score_ks", "KS statistic of anomaly scores vs reference", ["product"])
DRIFT_FLAG = Gauge("fscad_drift_flag", "1 if drift detected", ["product"])
ONBOARD = Histogram(
    "fscad_onboard_seconds", "Time to onboard a product", buckets=(5, 10, 30, 60, 120, 300, 600, 1200)
)
N_PRODUCTS = Gauge("fscad_products", "Registered products")


def load_cfg():
    overrides = shlex.split(os.environ.get("FSCAD_OVERRIDES", ""))
    if "FSCAD_PRODUCTS_DIR" in os.environ:
        overrides.append(f"products_dir={os.environ['FSCAD_PRODUCTS_DIR']}")
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="serve", overrides=overrides)


_service: AnomalyService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service()  # load backbone and products before accepting traffic
    yield


app = FastAPI(title="fscad", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


def service() -> AnomalyService:
    global _service
    if _service is None:
        _service = AnomalyService(load_cfg())
        N_PRODUCTS.set(len(_service.router))
    return _service


def read_image(data: bytes, size: int) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB").resize((size, size), Image.BICUBIC)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"cannot decode image: {e}") from e
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def map_png(m: np.ndarray) -> str:
    m = (m - m.min()) / max(1e-8, m.max() - m.min())
    buf = io.BytesIO()
    Image.fromarray((m * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health() -> dict:
    s = service()
    return {"status": "ok", "device": str(s.device), "products": len(s.router), "tau": s.router.tau}


@app.get("/products")
def products() -> list[dict]:
    return service().products()


@app.post("/predict")
async def predict(file: UploadFile = File(...), return_map: bool = False, product: str | None = None) -> dict:
    s = service()
    if len(s.router) == 0:
        raise HTTPException(409, "no product registered yet, call /onboard first")
    img = read_image(await file.read(), s.cfg.model.img_size)
    pred = s.predict(img, return_map=return_map, product=product)
    for stage, ms in pred.timings_ms.items():
        PRED_LATENCY.labels(stage).observe(ms / 1000)
    SIMILARITY.observe(pred.similarity)
    if pred.rejected:
        REJECTIONS.inc()
        PREDICTIONS.labels("unknown", "rejected").inc()
    else:
        CACHE.labels("hit" if pred.cache_hit else "miss").inc()
        SCORES.labels(pred.product).observe(pred.score)
        PREDICTIONS.labels(pred.product, "anomaly" if pred.is_anomaly else "normal").inc()
        st = s.monitors[pred.product].status()
        if st.cls_ks == st.cls_ks:
            DRIFT_CLS.labels(pred.product).set(st.cls_ks)
        if st.score_ks == st.score_ks:
            DRIFT_SCORE.labels(pred.product).set(st.score_ks)
        DRIFT_FLAG.labels(pred.product).set(int(st.drift))
    out = pred.as_dict()
    if return_map and pred.anomaly_map is not None:
        out["anomaly_map_png"] = map_png(pred.anomaly_map)
    return out


@app.post("/onboard")
async def onboard(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    val_files: list[UploadFile] | None = File(None),
) -> dict:
    s = service()
    size = s.cfg.model.img_size
    shots = torch.stack([read_image(await f.read(), size) for f in files])
    val = torch.stack([read_image(await f.read(), size) for f in val_files]) if val_files else None
    res = s.onboard(name, shots, val)
    ONBOARD.observe(res["seconds"])
    N_PRODUCTS.set(len(s.router))
    return res


@app.get("/drift")
def drift() -> dict:
    return service().drift()


@app.post("/reload")
def reload() -> dict:
    names = service().reload()
    N_PRODUCTS.set(len(names))
    return {"products": names}
