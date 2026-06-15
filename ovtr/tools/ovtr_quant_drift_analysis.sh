#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_VARIANT="${MODEL_VARIANT:-5_frame}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
QUANT_MODE="${QUANT_MODE:-ptq}"
QUANT_PARTITION="${QUANT_PARTITION:-exp_a3}"
QUANT_WEIGHT_BITS="${QUANT_WEIGHT_BITS:-4}"
QUANT_ACTIVATION_BITS="${QUANT_ACTIVATION_BITS:-4}"
QUANT_ATTENTION_BITS="${QUANT_ATTENTION_BITS:-8}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
OUTPUT="${OUTPUT:-./results}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CALIB_SAMPLES="${CALIB_SAMPLES:-512}"
ANALYSIS_MAX_FRAMES="${ANALYSIS_MAX_FRAMES:-0}"
ANALYSIS_PLOT_MAX_AGE="${ANALYSIS_PLOT_MAX_AGE:-0}"
ANALYSIS_IOU_DIVERGENCE_THRESH="${ANALYSIS_IOU_DIVERGENCE_THRESH:-0.5}"
ANALYSIS_SAMPLE_SEQS_PER_DATASET="${ANALYSIS_SAMPLE_SEQS_PER_DATASET:-10}"
ANALYSIS_SAMPLE_DATASETS="${ANALYSIS_SAMPLE_DATASETS:-YFCC100M HACS BDD ArgoVerse AVA LaSOT Charades}"
ANALYSIS_EMBEDDING_VIZ="${ANALYSIS_EMBEDDING_VIZ:-0}"
ANALYSIS_EMBEDDING_ONLY="${ANALYSIS_EMBEDDING_ONLY:-0}"
ANALYSIS_COMPARE_QUANT="${ANALYSIS_COMPARE_QUANT:-}"
ANALYSIS_EMBEDDING_NUM_CLASSES="${ANALYSIS_EMBEDDING_NUM_CLASSES:-100}"
ANALYSIS_EMBEDDING_SCORE_TOPK="${ANALYSIS_EMBEDDING_SCORE_TOPK:-10}"
ANALYSIS_EMBEDDING_SCORE_MAX_BARS="${ANALYSIS_EMBEDDING_SCORE_MAX_BARS:-12}"
ANALYSIS_EMBEDDING_MAX_REGIONS_PER_GROUP="${ANALYSIS_EMBEDDING_MAX_REGIONS_PER_GROUP:-32}"
ANALYSIS_EMBEDDING_MAX_GROUPS_PER_CLASS="${ANALYSIS_EMBEDDING_MAX_GROUPS_PER_CLASS:-20}"
ANALYSIS_EMBEDDING_MAX_EXAMPLES_PER_CLASS="${ANALYSIS_EMBEDDING_MAX_EXAMPLES_PER_CLASS:-2}"
ANALYSIS_EMBEDDING_MIN_REGIONS="${ANALYSIS_EMBEDDING_MIN_REGIONS:-2}"
ANALYSIS_EMBEDDING_SCORE_TEMPERATURE="${ANALYSIS_EMBEDDING_SCORE_TEMPERATURE:-0.007}"

case "${MODEL_VARIANT}:${EVAL_SPLIT}" in
    lite:val)
        CONFIG_FILE="./config/ovtr_lite_train_val.py"
        FP32_PRETRAIN_DEFAULT="../model_zoo/ovtr_lite.pth"
        SCORE_THRESH="0.19 0.19 0.19 0.19 0.19 0.19 0.19"
        FILTER_SCORE_THRESH="0.19 0.19 0.19 0.19 0.19 0.19 0.19"
        IOUS_THRESH="0.45 0.45 0.45 0.45 0.45 0.45 0.45"
        ;;
    lite:test)
        CONFIG_FILE="./config/ovtr_lite_test.py"
        FP32_PRETRAIN_DEFAULT="../model_zoo/ovtr_lite.pth"
        SCORE_THRESH="0.19 0.19 0.19 0.19 0.19 0.19 0.19"
        FILTER_SCORE_THRESH="0.19 0.19 0.19 0.19 0.19 0.19 0.19"
        IOUS_THRESH="0.45 0.45 0.45 0.45 0.45 0.45 0.45"
        ;;
    5_frame:val)
        CONFIG_FILE="./config/ovtr_5_frame_train_val.py"
        FP32_PRETRAIN_DEFAULT="../model_zoo/ovtr_5_frame.pth"
        SCORE_THRESH="0.20 0.17 0.17 0.20 0.17 0.20 0.17"
        FILTER_SCORE_THRESH="0.20 0.17 0.17 0.20 0.17 0.20 0.17"
        IOUS_THRESH="0.5 0.45 0.5 0.4 0.45 0.45 0.45"
        ;;
    5_frame:test)
        CONFIG_FILE="./config/ovtr_5_frame_test.py"
        FP32_PRETRAIN_DEFAULT="../model_zoo/ovtr_5_frame.pth"
        SCORE_THRESH="0.20 0.17 0.17 0.20 0.17 0.20 0.17"
        FILTER_SCORE_THRESH="0.20 0.17 0.17 0.20 0.17 0.20 0.17"
        IOUS_THRESH="0.5 0.45 0.5 0.4 0.45 0.45 0.45"
        ;;
    *)
        echo "Unsupported MODEL_VARIANT/EVAL_SPLIT combination: ${MODEL_VARIANT}/${EVAL_SPLIT}" >&2
        exit 1
        ;;
esac

FP32_PRETRAIN_MODEL="${FP32_PRETRAIN_MODEL:-${FP32_PRETRAIN_DEFAULT}}"
if [ "${QUANT_MODE}" = "qat" ]; then
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-./weights_qat_${MODEL_VARIANT}_${QUANT_PARTITION}}"
    PRETRAIN_MODEL="${PRETRAIN_MODEL:-${CHECKPOINT_DIR}/checkpoint.pth}"
else
    PRETRAIN_MODEL="${PRETRAIN_MODEL:-${FP32_PRETRAIN_DEFAULT}}"
fi
if [ -z "${ANALYSIS_OUTPUT_DIR:-}" ]; then
    if [ "${ANALYSIS_EMBEDDING_ONLY}" = "1" ]; then
        ANALYSIS_OUTPUT_DIR="./results/embedding_viz_${MODEL_VARIANT}_${QUANT_MODE}_${QUANT_PARTITION}_${EVAL_SPLIT}"
    else
        ANALYSIS_OUTPUT_DIR="./results/quant_drift_${MODEL_VARIANT}_${QUANT_MODE}_${QUANT_PARTITION}_${EVAL_SPLIT}"
    fi
fi

EXTRA_ANALYSIS_ARGS=()
if [ "${ANALYSIS_EMBEDDING_VIZ}" = "1" ] || [ "${ANALYSIS_EMBEDDING_ONLY}" = "1" ]; then
    EXTRA_ANALYSIS_ARGS+=(--analysis_embedding_viz)
fi
if [ "${ANALYSIS_EMBEDDING_ONLY}" = "1" ]; then
    EXTRA_ANALYSIS_ARGS+=(--analysis_embedding_only)
fi
if [ -n "${ANALYSIS_COMPARE_QUANT}" ]; then
    # shellcheck disable=SC2206
    COMPARE_QUANT_ARGS=(${ANALYSIS_COMPARE_QUANT})
    EXTRA_ANALYSIS_ARGS+=(--analysis_compare_quant "${COMPARE_QUANT_ARGS[@]}")
fi
EXTRA_ANALYSIS_ARGS+=(--analysis_embedding_num_classes "${ANALYSIS_EMBEDDING_NUM_CLASSES}")
EXTRA_ANALYSIS_ARGS+=(--analysis_embedding_score_topk "${ANALYSIS_EMBEDDING_SCORE_TOPK}")
EXTRA_ANALYSIS_ARGS+=(--analysis_embedding_score_max_bars "${ANALYSIS_EMBEDDING_SCORE_MAX_BARS}")
EXTRA_ANALYSIS_ARGS+=(--analysis_embedding_max_regions_per_group "${ANALYSIS_EMBEDDING_MAX_REGIONS_PER_GROUP}")
EXTRA_ANALYSIS_ARGS+=(--analysis_embedding_max_groups_per_class "${ANALYSIS_EMBEDDING_MAX_GROUPS_PER_CLASS}")
EXTRA_ANALYSIS_ARGS+=(--analysis_embedding_max_examples_per_class "${ANALYSIS_EMBEDDING_MAX_EXAMPLES_PER_CLASS}")
EXTRA_ANALYSIS_ARGS+=(--analysis_embedding_min_regions "${ANALYSIS_EMBEDDING_MIN_REGIONS}")
EXTRA_ANALYSIS_ARGS+=(--analysis_embedding_score_temperature "${ANALYSIS_EMBEDDING_SCORE_TEMPERATURE}")

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python ./analyze_quant_drift.py \
    --config_file "${CONFIG_FILE}" \
    --dataset_file lvis_generated_img_seqs \
    --epochs 16 \
    --with_box_refine \
    --two_stage \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --lr_drop 13 \
    --pretrain "${PRETRAIN_MODEL}" \
    --analysis_fp32_pretrain "${FP32_PRETRAIN_MODEL}" \
    --analysis_output_dir "${ANALYSIS_OUTPUT_DIR}" \
    --analysis_max_frames "${ANALYSIS_MAX_FRAMES}" \
    --analysis_plot_max_age "${ANALYSIS_PLOT_MAX_AGE}" \
    --analysis_iou_divergence_thresh "${ANALYSIS_IOU_DIVERGENCE_THRESH}" \
    --analysis_sample_sequences_per_dataset "${ANALYSIS_SAMPLE_SEQS_PER_DATASET}" \
    --analysis_sample_datasets ${ANALYSIS_SAMPLE_DATASETS} \
    --output_dir "${OUTPUT}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --sample_mode random_interval \
    --sample_interval 1 \
    --sampler_steps 4 7 14 \
    --sampler_lengths 2 3 4 5 \
    --merger_dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --track_query_iteration CIP \
    --calculate_negative_samples \
    --quant_mode "${QUANT_MODE}" \
    --quant_partition "${QUANT_PARTITION}" \
    --quant_weight_bits "${QUANT_WEIGHT_BITS}" \
    --quant_activation_bits "${QUANT_ACTIVATION_BITS}" \
    --quant_attention_bits "${QUANT_ATTENTION_BITS}" \
    --quant_calib_samples "${CALIB_SAMPLES}" \
    --score_thresh ${SCORE_THRESH} \
    --filter_score_thresh ${FILTER_SCORE_THRESH} \
    --ious_thresh ${IOUS_THRESH} \
    --miss_tolerance 5 5 5 5 5 5 5 \
    --maximum_quantity 160 \
    "${EXTRA_ANALYSIS_ARGS[@]}" \
    "$@"
