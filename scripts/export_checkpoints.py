"""Collect the checkpoints of official runs into ready-to-serve bundles with a manifest.

    python scripts/export_checkpoints.py dataset=visa method=lora shots=4 'seeds=[0,1,2]' \
        +export.out=checkpoints

Only MLflow runs tagged final=true whose logged parameters match the product configuration of the
given overrides (method, model, LoRA, training, memory, scoring) are exported, so a checkpoint of an
older configuration is never shipped by mistake. Each run is linked to its file by the
``checkpoint`` / ``checkpoint_sha256`` tags (runs from fscad >= the provenance change) or, for older
runs, by the file whose modification time falls inside the run. Each exported file gets the
provenance of its run in ``meta`` (config, git commit, MLflow run id, metrics).

Output, one serving bundle per (dataset, method, shots, seed):

    <out>/<dataset>-<method>-k<K>-s<seed>/<product id>/model.pt   (+ meta.json)
    <out>/manifest.json                                            every file, run, sha256, metrics

A bundle is a ``products_dir`` for the API or for ``scripts/predict.py``.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from hydra import compose, initialize_config_dir

from fscad.pipeline import file_sha256, product_config
from fscad.product import ProductModel
from fscad.service import product_id
from fscad.tracking import _flatten

ROOT = Path(__file__).resolve().parents[1]
SLACK_S = 5.0  # file system vs MLflow clock


def expected_params(cfg) -> dict[str, str]:
    """Flattened product config as MLflow logged it (str(value))."""
    return {k: str(v)[:500] for k, v in _flatten(product_config(cfg)).items()}


def find_runs(cfg, seeds: list[int]):
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg.tracking.uri))
    filt = (
        f"tags.final='true' and tags.dataset='{cfg.dataset.name}' and tags.method='{cfg.method}' "
        f"and tags.shots='{cfg.shots}'"
    )
    df = mlflow.search_runs(experiment_names=[cfg.tracking.experiment], filter_string=filt)
    df = df[(df.status == "FINISHED") & df["tags.seed"].astype(int).isin(seeds)]
    want = expected_params(cfg)
    ok = df.apply(lambda r: all(str(r.get(f"params.{k}")) == v for k, v in want.items()), axis=1)
    skipped = df[~ok]
    df = df[ok].sort_values("start_time")
    # one run per (class, seed): the latest
    df = df.groupby(["tags.class", "tags.seed"], as_index=False).last()
    return df, skipped


def locate(run, artifacts: Path, dataset: str) -> Path:
    tag = run.get("tags.checkpoint")
    if isinstance(tag, str) and tag:
        p = Path(tag) if Path(tag).is_absolute() else ROOT / tag
        if file_sha256(p) != run.get("tags.checkpoint_sha256"):
            raise RuntimeError(f"{p} changed since run {run.run_id} (sha256 mismatch)")
        return p
    # legacy runs: the file written while the run was open
    name = f"k{run['tags.shots']}_s{run['tags.seed']}.pt"
    t0, t1 = run.start_time.timestamp() - SLACK_S, run.end_time.timestamp() + SLACK_S
    cands = [
        p for p in artifacts.glob(f"**/{dataset}/{run['tags.class']}/{name}") if t0 <= p.stat().st_mtime <= t1
    ]
    if len(cands) != 1:
        raise RuntimeError(f"run {run.run_id}: {len(cands)} checkpoint files written during the run")
    return cands[0]


def main(overrides: list[str]) -> None:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    out = Path(cfg.get("export", {}).get("out", "checkpoints"))
    seeds = list(cfg.seeds)
    runs, skipped = find_runs(cfg, seeds)
    if runs.empty:
        sys.exit("no final run matches this configuration")
    artifacts = Path(cfg.paths.artifacts_dir)
    if not artifacts.is_absolute():
        artifacts = ROOT / artifacts
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    done = {f["path"] for f in manifest["files"]}
    for _, run in runs.iterrows():
        cls, seed = run["tags.class"], int(run["tags.seed"])
        src = locate(run, artifacts, cfg.dataset.name)
        product = ProductModel.load(src)
        name = f"{cfg.dataset.name}/{cls}"
        if product.name != name or product.meta.get("shots") != cfg.shots or product.meta.get("seed") != seed:
            raise RuntimeError(f"{src} does not hold {name} k{cfg.shots} s{seed}")
        metrics = {k[8:]: float(v) for k, v in run.items() if k.startswith("metrics.") and v == v}
        product.meta.setdefault("config", product_config(cfg))
        product.meta.update(
            {"git_commit": run["tags.git_commit"], "mlflow_run_id": run.run_id, "metrics": metrics}
        )
        bundle = f"{cfg.dataset.name}-{cfg.method}-k{cfg.shots}-s{seed}"
        dst = out / bundle / product_id(name) / "model.pt"
        product.save(dst)
        (dst.parent / "meta.json").write_text(
            json.dumps({"name": name, **product.meta}, indent=2, default=str)
        )
        rel = str(dst.relative_to(out))
        entry = {
            "path": rel,
            "product": name,
            "bundle": bundle,
            "method": cfg.method,
            "shots": cfg.shots,
            "seed": seed,
            "sha256": file_sha256(dst),
            "source": str(src),
            "mlflow_run_id": run.run_id,
            "git_commit": run["tags.git_commit"],
            "i_auroc": metrics.get("i_auroc"),
            "au_pro": metrics.get("au_pro"),
            "threshold": product.score_threshold,
        }
        manifest["files"] = [f for f in manifest["files"] if f["path"] != rel] + [entry]
        print(f"{'updated' if rel in done else 'exported'} {rel}  I-AUROC {entry['i_auroc']:.4f}  <- {src}")
    manifest["files"].sort(key=lambda f: f["path"])
    manifest["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"{len(runs)} checkpoints -> {out}; {len(skipped)} final runs of other configurations skipped")


if __name__ == "__main__":
    main(sys.argv[1:])
