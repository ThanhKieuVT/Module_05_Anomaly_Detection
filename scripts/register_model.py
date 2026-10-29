"""Register a product artifact in the MLflow Model Registry and (optionally) promote it.

    python scripts/register_model.py artifacts/<run>/mvtec_ad/bottle/k4_s0.pt --alias staging
    python scripts/register_model.py ... --alias production --gate results/summary.json --min-au-pro 0.85

A product = adapter + memory bank + prototype in one file, logged as an MLflow pyfunc model named
``fscad-<dataset>-<class>``. Aliases (``staging`` / ``production``) replace the deprecated stages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
from mlflow.pyfunc import PythonModel


class ProductPyfunc(PythonModel):
    """Lets MLflow load the artifact; serving loads the .pt file directly."""

    def load_context(self, context):
        from fscad.product import ProductModel

        self.product = ProductModel.load(context.artifacts["product"])

    def predict(self, context, model_input, params=None):  # pragma: no cover - not used for serving
        raise NotImplementedError("use the fscad service (serving/app.py) for inference")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--name", default=None, help="registry name, default fscad-<dataset>-<class>")
    ap.add_argument("--alias", default="staging")
    ap.add_argument("--gate", default=None, help="summary.json from fscad.evaluate to check before promoting")
    ap.add_argument("--min-au-pro", type=float, default=None)
    ap.add_argument("--max-latency-ms", type=float, default=None)
    a = ap.parse_args()

    if a.gate:
        s = json.loads(Path(a.gate).read_text())
        if a.min_au_pro is not None and s["au_pro"]["mean"] < a.min_au_pro:
            sys.exit(f"gate failed: AU-PRO {s['au_pro']['mean']:.4f} < {a.min_au_pro}")
        if a.max_latency_ms is not None and s["latency_ms_per_image"]["mean"] > a.max_latency_ms:
            sys.exit(f"gate failed: latency {s['latency_ms_per_image']['mean']:.1f} ms > {a.max_latency_ms}")

    from fscad.product import ProductModel

    p = ProductModel.load(a.artifact)
    name = a.name or "fscad-" + p.name.replace("/", "-")
    with mlflow.start_run(run_name=f"register-{name}", tags={"phase": "system", "stage": "register"}):
        mlflow.log_params({k: v for k, v in p.meta.items() if isinstance(v, int | float | str)})
        info = mlflow.pyfunc.log_model(
            name="product",
            python_model=ProductPyfunc(),
            artifacts={"product": str(a.artifact)},
            registered_model_name=name,
        )
    client = mlflow.MlflowClient()
    version = info.registered_model_version
    client.set_registered_model_alias(name, a.alias, version)
    print(f"registered {name} v{version} as @{a.alias}")


if __name__ == "__main__":
    main()
