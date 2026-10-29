#!/usr/bin/env bash
# Official baselines on our fixed few-shot splits (needs a CUDA GPU).
#   bash scripts/run_baselines.sh mvtec_ad   |   bash scripts/run_baselines.sh visa
# Install their requirements first:
#   pip install -r third_party/AnomalyDINO/requirements.txt
#   pip install -e third_party/patchcore-inspection
set -euo pipefail
DATASET=${1:-mvtec_ad}
SHOTS=${SHOTS:-"1 2 4 8 16"}
SEEDS=${SEEDS:-"0 1 2"}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
case "$DATASET" in
  mvtec_ad) ADINO_NAME=MVTec ;;
  visa) ADINO_NAME=VisA ;;
  *) echo "unsupported dataset $DATASET" >&2; exit 1 ;;
esac

for K in $SHOTS; do
  for S in $SEEDS; do
    VIEW=$(python "$ROOT/scripts/build_fewshot_view.py" --dataset "$DATASET" --shots "$K" --seed "$S" \
             --data-root "$ROOT/data" --splits-dir "$ROOT/splits" --out "$ROOT/data_views")
    OUT="$ROOT/results/baselines/$DATASET/k${K}_s${S}"
    mkdir -p "$OUT"

    # AnomalyDINO: train/good holds exactly K images, so --shots K uses all of them.
    (cd "$ROOT/third_party/AnomalyDINO" && \
      python run_anomalydino.py --dataset "$ADINO_NAME" --data_root "$VIEW" --shots "$K" \
        --num_seeds 1 --just_seed "$S" --preprocess agnostic --eval_segm --no-save_examples \
        --tag "fscad_k${K}_s${S}" 2>&1 | tee "$OUT/anomalydino.log")

    # PatchCore (WRN50, layer2+3, 10% coreset, 224 px) on the same K images.
    CLASSES=$(ls "$VIEW")
    FLAGS=$(for c in $CLASSES; do printf -- "-d %s " "$c"; done)
    (cd "$ROOT/third_party/patchcore-inspection" && \
      python bin/run_patchcore.py --gpu 0 --seed "$S" --log_group "fscad_${DATASET}_k${K}_s${S}" \
        --log_project fscad_baselines "$OUT/patchcore" \
        patch_core -b wideresnet50 -le layer2 -le layer3 --pretrain_embed_dimension 1024 \
          --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
        sampler -p 0.1 approx_greedy_coreset \
        dataset --resize 256 --imagesize 224 $FLAGS mvtec "$VIEW" 2>&1 | tee "$OUT/patchcore.log")
  done
done
echo "logs and metrics in $ROOT/results/baselines/$DATASET (AnomalyDINO also writes results_* inside its folder)"
