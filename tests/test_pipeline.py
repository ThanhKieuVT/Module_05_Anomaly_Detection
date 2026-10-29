"""End-to-end smoke tests on the toy dataset with a tiny random ViT (no download)."""

import io

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from fscad import continual, evaluate
from fscad.pipeline import load_images
from fscad.product import ProductModel
from tests.conftest import make_cfg


def test_evaluate_lora_and_knn(toy_root, tmp_path):
    for method in ("lora", "knn"):
        cfg = make_cfg(toy_root, tmp_path, f"method={method}", "classes=[alpha]")
        rows = evaluate.run(cfg)
        assert len(rows) == 1
        r = rows[0]
        for k in ("i_auroc", "p_auroc", "au_pro", "f1_max", "latency_ms_per_image"):
            assert np.isfinite(r[k]), k
    art = list((tmp_path / "artifacts").rglob("*.pt"))
    assert len(art) == 2
    p = ProductModel.load(art[0])
    assert len(p.memory) > 0 and p.score_threshold is not None and p.val_cls is not None


def test_continual_with_holdout(toy_root, tmp_path):
    cfg = make_cfg(
        toy_root, tmp_path, "continual.datasets=[toy]", "continual.holdout=[toy/gamma]", "eval.pixel=false"
    )
    res = continual.run(cfg)
    f = res["final"]
    assert res["order"] == ["toy/alpha", "toy/beta"]
    assert f["fm_oracle"] == 0.0
    assert 0.0 <= f["routing_acc"] <= 1.0 and "reject_auroc" in f
    assert len(res["steps"]) == 2


def test_service_onboard_and_predict(toy_root, tmp_path):
    from fscad.service import AnomalyService

    cfg = make_cfg(toy_root, tmp_path)
    OmegaConf.set_struct(cfg, False)
    serve = OmegaConf.merge(
        cfg,
        {
            "products_dir": str(tmp_path / "serving"),
            "cache_size": 1,
            "onboard_steps": 2,
            "drift": {"window": 20, "min_samples": 3},
        },
    )
    svc = AnomalyService(serve)
    root = toy_root / "toy"
    for name in ("alpha", "beta"):
        shots = load_images([str(p) for p in sorted((root / name / "train/good").glob("*.png"))[:2]], 112)
        res = svc.onboard(name, shots)
        assert res["name"] == name and res["bytes"] > 0
    img = load_images([str(sorted((root / "alpha/test/defect").glob("*.png"))[0])], 112)[0]
    pred = svc.predict(img, return_map=True, product="alpha")
    assert pred.score is not None and pred.anomaly_map.shape == (112, 112)
    svc.predict(img, product="beta")  # evicts alpha (cache_size=1)
    assert list(svc.cache) == ["beta"]
    for _ in range(3):
        svc.predict(img, product="alpha")
    assert svc.drift()["alpha"]["n"] == 4
    # a fresh service sees the products saved on disk
    assert sorted(AnomalyService(serve).router.classes) == ["alpha", "beta"]


def test_api(toy_root, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("FSCAD_PRODUCTS_DIR", str(tmp_path / "api"))
    monkeypatch.setenv(
        "FSCAD_OVERRIDES",
        "model=tiny device=cpu onboard_steps=2 lora.blocks=2 " "train.batch_size=4 eval.batch_size=4",
    )
    import serving.app as appmod

    appmod._service = None
    client = TestClient(appmod.app)
    root = toy_root / "toy" / "alpha"

    def png(path):
        buf = io.BytesIO()
        Image.open(path).save(buf, format="PNG")
        return buf.getvalue()

    files = [
        ("files", (p.name, png(p), "image/png")) for p in sorted((root / "train/good").glob("*.png"))[:2]
    ]
    r = client.post("/onboard", data={"name": "alpha"}, files=files)
    assert r.status_code == 200, r.text
    test_img = sorted((root / "test/defect").glob("*.png"))[0]
    r = client.post("/predict?return_map=true", files={"file": ("x.png", png(test_img), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "score" in body and "anomaly_map_png" in body or body["rejected"]
    assert client.get("/products").json()[0]["name"] == "alpha"
    m = client.get("/metrics/").text
    assert "fscad_predictions_total" in m
