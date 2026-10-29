"""Download every ``fscad-*`` model at a given alias into the serving products directory.

    python scripts/pull_registry.py --alias production --out artifacts/serving
    curl -X POST localhost:8000/reload
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlflow

from fscad.service import product_id

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="production")
    ap.add_argument("--out", default="artifacts/serving")
    a = ap.parse_args()
    client = mlflow.MlflowClient()
    for rm in client.search_registered_models(filter_string="name LIKE 'fscad-%'"):
        try:
            mv = client.get_model_version_by_alias(rm.name, a.alias)
        except mlflow.exceptions.MlflowException:
            continue
        local = Path(mlflow.artifacts.download_artifacts(f"models:/{rm.name}@{a.alias}"))
        src = next(local.rglob("*.pt"))
        from fscad.product import ProductModel

        name = ProductModel.load(src).name
        dst = Path(a.out) / product_id(name) / "model.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"{rm.name} v{mv.version} -> {dst}")
