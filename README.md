# fscad: Few-Shot Continual Anomaly Detection with Routed Low-Rank Adapters

A frozen DINOv2 ViT is shared by every product. Each new product is onboarded from a few normal
images (1-16) as a small LoRA adapter + a patch memory bank + a CLS prototype. At inference a
prototype router picks the product's adapter, or rejects the image as an unregistered product.
Nothing learned earlier is modified, so oracle-routed performance never forgets; the system error
splits cleanly into routing error and detection error. The repo also contains the MLOps side:
experiment tracking, data versioning, a multi-adapter inference service, and drift monitoring.

```
image ─► ViT blocks 1..L-a (frozen, shared) ─┬─► blocks L-a+1..L, no LoRA ─► CLS ─► router ─► product p / reject
                                             └─► blocks L-a+1..L + LoRA_p ─► patch tokens ─► kNN(memory_p) ─► score, map
```

## Layout

```
configs/              Hydra: dataset, model, lora, train, method (lora | knn), serve
src/fscad/
  data/               MVTec AD, VisA, MVTec AD 2 readers; fixed few-shot splits; synthetic anomalies; toy data
  models/             backbone (timm DINOv2, shared prefix), lora (Q/V-only LoRA on fused qkv), memory_bank, router
  scoring/            image score (top-1% mean), anomaly maps
  training/           compactness + margin loss
  product.py          onboarding a product (train adapter, build memory, threshold) and scoring
  evaluate.py         single-task few-shot evaluation          -> MLflow + results.csv
  train.py            train and save product artifacts
  continual.py        task-agnostic continual run, real vs oracle routing, rejection of held-out products
  service.py          multi-adapter inference core (LRU adapter cache, router, drift monitors)
  monitoring.py       label-free drift detection (KS tests on CLS similarity and scores)
  metrics.py          I-AUROC, P-AUROC, AU-PRO, F1-max, Forgetting Measure
serving/app.py        FastAPI: /predict /onboard /products /drift /reload /metrics
monitoring/           Prometheus config + alerts, Grafana dashboard, Evidently report
scripts/              splits, experiment matrix, baselines, LaTeX tables, registry, ONNX, latency, drift stream
third_party/          official AnomalyDINO and PatchCore code (git submodules) for the baselines
tests/                unit + end-to-end tests on a generated toy dataset (no download)
dvc.yaml              splits -> train -> evaluate
```

## Setup

```bash
git clone --recurse-submodules <repo-url> && cd fscad   # or: git submodule update --init
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision                             # pick the CUDA build for your machine
pip install -e ".[mlops,dev]"
pre-commit install
pytest                                                    # ~10 s, CPU, no download
```

DINOv2 weights are downloaded by timm from the Hugging Face hub on first use
(`vit_small_patch14_dinov2.lvd142m`, `vit_base_patch14_dinov2.lvd142m`).

### Data

Download MVTec AD, VisA and MVTec AD 2 (registration needed on the MVTec site) into:

```
data/mvtec_ad/<class>/{train,test,ground_truth}
data/visa/<class>/Data/...  + data/visa/split_csv/1cls.csv       (official VisA release)
data/mvtec_ad2/<class>/{train,validation,test_public}
```

Track them with DVC (never with git) and configure a remote:

```bash
dvc init
dvc add data/mvtec_ad data/visa data/mvtec_ad2
dvc remote add -d storage gdrive://<folder-id>      # or s3://bucket/path
dvc push
```

Few-shot splits are fixed per (class, shots, seed) so every method sees the same images:

```bash
dvc repro splits          # = python scripts/make_splits.py --dataset <d> --shots 1 2 4 8 16 --seeds 0 1 2
```

## Running experiments

Tracking goes to `sqlite:///mlflow.db` (view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`) unless `MLFLOW_TRACKING_URI` is set (e.g. the compose MLflow server
`http://localhost:5000`). In zsh, quote overrides that contain brackets.

```bash
# frozen DINOv2 + kNN (in-house AnomalyDINO) and the LoRA method
python -m fscad.evaluate dataset=mvtec_ad method=knn shots=4 'seeds=[0,1,2]'
python -m fscad.evaluate dataset=mvtec_ad method=lora shots=4 'seeds=[0,1,2]'

# sweeps (Hydra multirun)
python -m fscad.evaluate -m dataset=mvtec_ad,visa method=lora shots=1,4,16 'seeds=[0,1,2]'

# whole experiment matrix of the plan
python scripts/run_matrix.py               # prints the commands
python scripts/run_matrix.py --run main    # runs one group: baseline | main | ablation | continual

# continual: reuse the single-task adapters, then route
python -m fscad.continual reuse_artifacts=true shots=4 'continual.datasets=[mvtec_ad,visa]' \
    'continual.holdout=[visa/pcb4,visa/macaroni2,visa/capsules]' continual.order=random
```

Held-out products should resemble registered ones (pcb4 vs pcb1-3, macaroni2 vs macaroni1),
otherwise the rejection AUROC is high without saying much. An image rejected by the router gets no
anomaly score; `continual.reject_policy=alarm|miss` decides how it counts in the real-routing
AUROC (both are logged), and rejection rates are logged separately for normal and anomalous images.
`continual_result.json` also lists the most frequent routing confusions.

```bash
# H3: synthetic lighting drift on MVTec AD 2 test_public (the only split with ground truth):
# at which severity does the KS monitor alarm, and at which does I-AUROC drop?
python -m fscad.drift_eval dataset=mvtec_ad2 'classes=[vial]' shots=4 drift.kind=mixed
```

Useful knobs: `lora.r`, `lora.blocks`, `train.steps`, `train.synthetic=perlin|cutpaste|mixed`,
`train.use_synthetic=false`, `train.center=knn|mean`, `model=dinov2_vitb14`, `memory.augment=[...]`,
`router.n_prototypes=3` (k-means prototypes, the gate-3 fallback), `eval.max_test_images=20` for quick runs.

### Run conventions

* Every MLflow run is tagged `phase, dataset, class, shots, seed, method, git_commit, final`.
* Do not run experiments from uncommitted code: `tracking.require_clean=true` refuses to start.
* Paper numbers come only from runs with `tracking.final=true` (which also requires a clean tree);
  `python scripts/make_tables.py --phase single_task --metric au_pro` builds the LaTeX table from them.

### Official baselines (CUDA)

```bash
pip install -r third_party/AnomalyDINO/requirements.txt
pip install -e third_party/patchcore-inspection
bash scripts/run_baselines.sh mvtec_ad     # AnomalyDINO + PatchCore on our splits
```

`scripts/build_fewshot_view.py` builds a symlinked MVTec-layout folder containing exactly the K
shots of a split, so the official code uses the same images as our runs.

## Checkpoints and inference

Vietnamese handover guide (setup, training, reproduction, checkpoints, inference): [docs/HUONG_DAN.md](docs/HUONG_DAN.md).

Every evaluated product is saved to
`artifacts/<run_name>/<config fingerprint>/<dataset>/<class>/k<K>_s<seed>.pt`; the fingerprint is a
hash of the product configuration, so different configurations never overwrite each other. The
checkpoint's `meta` holds the config, git commit, MLflow run id and metrics, and the MLflow run is
tagged with the checkpoint path and sha256.

```bash
# collect the checkpoints of the official runs (final=true, current config) into serving bundles
python scripts/export_checkpoints.py dataset=visa method=lora shots=4 'seeds=[0,1,2]' +export.out=checkpoints
#   -> checkpoints/visa-lora-k4-s<seed>/<product>/model.pt + checkpoints/manifest.json (sha256, run, metrics)

# inference from the command line: routing, score, threshold, decision, heatmaps -> predictions.csv
python scripts/predict.py --products checkpoints/visa-lora-k4-s0 --out outputs/predict path/to/images/

# or serve a bundle
FSCAD_PRODUCTS_DIR=checkpoints/visa-lora-k4-s0 uvicorn serving.app:app --port 8000
```

The service and `predict.py` score exactly like the evaluation (background masking included), so
the stored threshold applies; `tests/test_pipeline.py::test_service_scores_like_evaluation` checks it.

## Demo (no download)

```bash
python scripts/demo.py                                    # real DINOv2 ViT-S/14, ~3 min on a laptop GPU/MPS
python scripts/demo.py --overrides "model=tiny onboard_steps=5"   # seconds, random weights
```

Starts the API, onboards two toy products from 4 normal images each, sends good and defective
images of both and images of an unregistered third product, and prints per image: routed product,
nearest product, CLS similarity, score, oracle score (true product forced) and the decision. It
writes the anomaly map of one defective image to `outputs/demo/anomaly_map.png`.

On toy data the large defects (20% of the image) shift the CLS token far enough that the router
rejects them as unknown products, while the oracle scores flag all of them. This is why a rejection
counts as an alert (`reject_policy=alarm`) and why rejection rates are reported separately for
normal and defective images.

## Serving and monitoring

```bash
docker compose up --build
# API http://localhost:8000/docs | MLflow :5000 | Prometheus :9090 | Grafana :3000 (dashboard "fscad")

# onboard a product from a few normal images
curl -F name=mvtec_ad/bottle -F files=@a.png -F files=@b.png -F files=@c.png -F files=@d.png localhost:8000/onboard
# predict (router picks the product; add ?return_map=true for the anomaly map as base64 PNG)
curl -F file=@test.png "localhost:8000/predict"
```

Without Docker: `uvicorn serving.app:app --port 8000` (configure with `FSCAD_PRODUCTS_DIR` and
`FSCAD_OVERRIDES="model=dinov2_vitb14 device=cuda"`).

Registry flow: `python scripts/register_model.py artifacts/<run>/<dataset>/<class>/k4_s0.pt --alias staging`,
promote with `--alias production --gate summary.json --min-au-pro 0.85`, then
`python scripts/pull_registry.py --alias production && curl -X POST localhost:8000/reload`.

System measurements:

```bash
python scripts/benchmark_latency.py --products-dir artifacts/serving          # p50/p95, cache hit vs miss
python scripts/export_onnx.py +out=artifacts/onnx/prefix.onnx +int8=true      # shared prefix in ONNX / INT8
python scripts/simulate_stream.py --images data/mvtec_ad2/vial/test_public/good \
    --unknown-dir data/mvtec_ad2/can/test_public/good                         # time from drift onset to alarm
python monitoring/evidently_report.py --product artifacts/serving/<id>/model.pt --current-dir <images>
```

## CI

`.github/workflows/ci.yml`: ruff + black, pytest, and a smoke run (toy data -> train one class ->
evaluate -> gate). A manual `promote` job evaluates on real data on a self-hosted GPU runner and
registers the product only if AU-PRO and latency pass the gate.

## Implementation notes

* **LoRA on Q and V only.** timm and the official DINOv2 fuse Q, K, V into one `Linear(D, 3D)`;
  `peft` can only adapt that layer as a whole. `models/lora.py` wraps it and updates the Q and V
  slices, so the trainable parameter count is exactly `L * 2 * r * 2D` (ViT-S, r=8, L=4: 49,152).
* **Shared prefix.** The first `12 - L` blocks run once per image; only the last `L` blocks run twice
  (frozen for the router's CLS, adapted for the patch tokens).
* **Compactness reference.** `train.center=knn` (default) pulls each normal patch towards its nearest
  normal patch of another view, which matches kNN scoring and keeps object parts apart.
  `train.center=mean` is the single-centre form.
* **Background masking.** `eval.masking=auto` scores only foreground patches (first PCA component
  of the patch tokens, as in AnomalyDINO): on for every VisA class and for MVTec AD capsule,
  hazelnut, pill, screw, toothbrush. Stains on the mat or conveyor then cannot raise an alarm.
* **Thresholds.** Router `tau` is the 1% quantile of correct-class CLS similarity on held-out normal
  images (30 per class, disjoint from the shots); the per-product decision threshold is the 99%
  quantile of their anomaly scores. `evaluate` logs recall and FPR at that threshold.
