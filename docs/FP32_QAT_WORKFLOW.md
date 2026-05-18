# FP32 모델 기반 QAT Train / Eval / Analysis 실행 문서

이 문서는 현재 OVTR 레포에서 FP32 pretrained checkpoint를 시작점으로 QAT를 학습하고, 학습된 QAT checkpoint를 평가하며, FP32 baseline 대비 quantization drift를 분석하는 shell 실행 방법을 정리합니다.

기준 entry point는 `ovtr` tree입니다.

- QAT train: `ovtr/tools/ovtr_quant_full_model.sh`
- QAT eval: `ovtr/tools/ovtr_qat_eval.sh`
- Quant drift analysis: `ovtr/tools/ovtr_quant_drift_analysis.sh`

아래 명령은 모두 레포 루트에서 `cd ovtr` 한 뒤 실행하는 것을 기준으로 합니다.

```bash
cd ovtr
```

## 사전 준비

실행 전 다음 파일과 환경이 준비되어 있어야 합니다.

- Python / PyTorch / CUDA extension 환경: `docs/BLACKWELL_SETUP.md` 참고
- FP32 checkpoint:
  - lite 기본값: `../model_zoo/ovtr_lite.pth`
  - 5-frame 기본값: `../model_zoo/ovtr_5_frame.pth`
- Quant calibration annotation:
  - config 기본값: `../data/lvis_clear_75_60_calib_512.json`
  - 없으면 `ovtr` 디렉터리에서 아래 명령으로 생성합니다.

```bash
python ../process/create_lvis_calibration_split.py
```

QAT 평가와 analysis는 이미 quant state가 포함된 QAT checkpoint를 요구합니다. 학습 산출물을 평가 기본 경로와 맞추려면 train 시 `OUTPUT=./weights_qat_${MODEL_VARIANT}_${QUANT_PARTITION}` 형태를 권장합니다.

## 모델 Variant와 Partition

`MODEL_VARIANT`는 다음 두 값을 지원합니다.

| 값 | train/val config | test config | 기본 FP32 checkpoint |
| --- | --- | --- | --- |
| `lite` | `./config/ovtr_lite_train_val.py` | `./config/ovtr_lite_test.py` | `../model_zoo/ovtr_lite.pth` |
| `5_frame` | `./config/ovtr_5_frame_train_val.py` | `./config/ovtr_5_frame_test.py` | `../model_zoo/ovtr_5_frame.pth` |

`QUANT_PARTITION`은 다음 값을 지원합니다.

| 값 | 의미 |
| --- | --- |
| `exp_a` | A-side 전체. `exp_a1 + exp_a2 + exp_a3_head`에 가까운 범위입니다. |
| `exp_a1` | `backbone`, `input_proj`, `patch2query` |
| `exp_a2` | `transformer.encoder` aggregation, `level_embed`, encoder output 계열. `fusion_layers`는 제외합니다. |
| `exp_a3` | decoder와 `tgt_embed`. `transformer.decoder.bbox_embed`, `feature_align` head는 제외합니다. |
| `exp_a3_head` | decoder, `tgt_embed`, decoder output head, `feature_align` |
| `exp_b` | `track_embed` |
| `exp_a1_to_b` | `exp_a1 + exp_a2 + exp_a3 + exp_b`. A3 output head는 제외합니다. |
| `exp_a3_b` | `exp_a3 + exp_b`. A3 output head는 제외합니다. |

현재 quant backend가 patch하는 module type은 `Conv2d`, `Linear`, `Embedding`, `MultiheadAttention`, `MultiScaleDeformableAttention`입니다.

## 1. FP32 Checkpoint에서 QAT Train

QAT 학습은 `QUANT_MODE=qat`로 `ovtr/tools/ovtr_quant_full_model.sh`를 실행합니다. 이 스크립트는 내부에서 `torchrun ./main.py`를 호출하고, `--pretrain`으로 FP32 checkpoint를 로드한 뒤 calibration initialization을 수행한 다음 QAT fine-tuning을 시작합니다.

권장 기본 예시는 다음과 같습니다.

```bash
MODEL_VARIANT=lite \
QUANT_MODE=qat \
QUANT_PARTITION=exp_a1_to_b \
PRETRAIN_MODEL=../model_zoo/ovtr_lite.pth \
OUTPUT=./weights_qat_lite_exp_a1_to_b \
CUDA_DEVICES=0,1,2,3 \
NPROC_GPU=4 \
MASTER_PORT=9987 \
BATCH_SIZE=1 \
NUM_WORKERS=8 \
CALIB_SAMPLES=512 \
QAT_EPOCHS=1 \
QAT_LR=4e-5 \
QAT_LR_BACKBONE=4e-6 \
./tools/ovtr_quant_full_model.sh
```

5-frame 모델은 `MODEL_VARIANT=5_frame`과 FP32 checkpoint만 바꾸면 됩니다.

```bash
MODEL_VARIANT=5_frame \
QUANT_MODE=qat \
QUANT_PARTITION=exp_a3_b \
PRETRAIN_MODEL=../model_zoo/ovtr_5_frame.pth \
OUTPUT=./weights_qat_5_frame_exp_a3_b \
CUDA_DEVICES=0,1,2,3 \
NPROC_GPU=4 \
MASTER_PORT=9987 \
BATCH_SIZE=1 \
NUM_WORKERS=8 \
CALIB_SAMPLES=512 \
QAT_EPOCHS=1 \
QAT_LR=4e-5 \
QAT_LR_BACKBONE=4e-6 \
./tools/ovtr_quant_full_model.sh
```

메모리가 부족하면 다음처럼 calibration과 auxiliary loss를 줄여서 먼저 동작 확인을 하실 수 있습니다.

```bash
MODEL_VARIANT=lite \
QUANT_MODE=qat \
QUANT_PARTITION=exp_a1_to_b \
PRETRAIN_MODEL=../model_zoo/ovtr_lite.pth \
OUTPUT=./weights_qat_lite_exp_a1_to_b \
CUDA_DEVICES=0 \
NPROC_GPU=1 \
BATCH_SIZE=1 \
NUM_WORKERS=4 \
CALIB_SAMPLES=32 \
QUANT_PIPELINE=legacy \
QUANT_DISABLE_PSEUDO_SEQUENCE_CALIB=1 \
QAT_EPOCHS=1 \
./tools/ovtr_quant_full_model.sh \
  --no_aux_loss \
  --max_len 100 \
  --quant_mse_bins 0 \
  --quant_mse_candidates 1
```

QAT train에서 자주 바꾸는 환경변수는 다음과 같습니다.

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `MODEL_VARIANT` | `lite` | `lite` 또는 `5_frame` |
| `QUANT_MODE` | `ptq` | QAT train은 반드시 `qat`로 설정합니다. |
| `QUANT_PARTITION` | `exp_a` | quantize하고 QAT에서 trainable로 둘 partition입니다. |
| `PRETRAIN_MODEL` | variant별 FP32 checkpoint | `--pretrain`으로 전달되는 시작 checkpoint입니다. |
| `OUTPUT` | `./results_quant` | `main.py --output_dir`. QAT eval 기본값과 맞추려면 `./weights_qat_${MODEL_VARIANT}_${QUANT_PARTITION}`를 권장합니다. |
| `CUDA_DEVICES` | QAT일 때 `0,1,2,3` | `CUDA_VISIBLE_DEVICES`로 전달됩니다. |
| `NPROC_GPU` | `CUDA_DEVICES` 개수 | `torchrun --nproc_per_node` 값입니다. |
| `MASTER_PORT` | `9987` | `torchrun --master_port` 값입니다. |
| `NUM_WORKERS` | `8` | dataloader worker 수입니다. |
| `BATCH_SIZE` | `1` | QAT는 기본적으로 batch size 1을 권장합니다. |
| `CALIB_SAMPLES` | `512` | `--quant_calib_samples`입니다. |
| `QAT_EPOCHS` | `1` | `--epochs`입니다. |
| `QAT_LR` | `4e-5` | 일반 QAT trainable parameter learning rate입니다. |
| `QAT_LR_BACKBONE` | `4e-6` | backbone parameter learning rate입니다. |
| `QUANT_PIPELINE` | `standard` | `standard` 또는 `legacy`입니다. |
| `QUANT_RANGE_METHOD` | empty | 지정 시 `--quant_range_method {mse,minmax}`로 전달됩니다. |
| `QUANT_WEIGHT_BITS` | `4` | `--quant_weight_bits`입니다. |
| `QUANT_ACTIVATION_BITS` | `4` | `--quant_activation_bits`입니다. |
| `QUANT_ATTENTION_BITS` | `8` | `--quant_attention_bits`입니다. |
| `QUANT_CALIB_SEQUENCE_LENGTH` | `3` | calibration 이미지당 pseudo-video frame 수입니다. |
| `QUANT_CALIB_MAX_TRANSLATE` | `0.08` | pseudo calibration translation 최대 비율입니다. |
| `QUANT_CALIB_MAX_ROTATE` | `6.0` | pseudo calibration rotation 최대 각도입니다. |
| `QUANT_CALIB_SCALE_JITTER` | `0.08` | pseudo calibration scale jitter입니다. |
| `QUANT_CALIB_MOTION_BLUR` | `3` | pseudo calibration motion blur kernel입니다. `<=1`이면 blur를 끕니다. |
| `QUANT_DISABLE_PSEUDO_SEQUENCE_CALIB` | `0` | `1`이면 legacy static-frame calibration을 사용합니다. |
| `QUANT_USE_SCHEDULER` | `0` | `1`이면 QAT 중 LR scheduler를 step합니다. |
| `QUANT_CALIBRATION_ONLY` | `0` | `1`이면 QAT initialization checkpoint만 저장하고 종료합니다. |
| `QAT_ALLOW_BATCH` | `0` | `1`이고 `BATCH_SIZE > 1`이면 experimental batched QAT path를 사용합니다. |

QAT train에서 스크립트가 `main.py`에 넘기는 핵심 인자는 아래와 같습니다.

```bash
torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_GPU}" \
  ./main.py \
  --epochs "${QAT_EPOCHS}" \
  --lr "${QAT_LR}" \
  --lr_backbone "${QAT_LR_BACKBONE}" \
  --lr_drop 13 \
  --max_len 250 \
  --config_file "${CONFIG_FILE}" \
  --dataset_file lvis_generated_img_seqs \
  --with_box_refine \
  --two_stage \
  --pretrain "${PRETRAIN_MODEL}" \
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
  --quant_mode qat \
  --quant_pipeline "${QUANT_PIPELINE}" \
  --quant_partition "${QUANT_PARTITION}" \
  --quant_weight_bits "${QUANT_WEIGHT_BITS}" \
  --quant_activation_bits "${QUANT_ACTIVATION_BITS}" \
  --quant_attention_bits "${QUANT_ATTENTION_BITS}" \
  --quant_calib_samples "${CALIB_SAMPLES}" \
  --quant_calib_sequence_length "${QUANT_CALIB_SEQUENCE_LENGTH}" \
  --quant_calib_max_translate "${QUANT_CALIB_MAX_TRANSLATE}" \
  --quant_calib_max_rotate "${QUANT_CALIB_MAX_ROTATE}" \
  --quant_calib_scale_jitter "${QUANT_CALIB_SCALE_JITTER}" \
  --quant_calib_motion_blur "${QUANT_CALIB_MOTION_BLUR}"
```

조건부로 추가되는 인자는 다음과 같습니다.

- `QUANT_RANGE_METHOD`가 비어 있지 않으면 `--quant_range_method "${QUANT_RANGE_METHOD}"`
- `QUANT_DISABLE_PSEUDO_SEQUENCE_CALIB=1`이면 `--quant_disable_pseudo_sequence_calib`
- `QUANT_USE_SCHEDULER=1`이면 `--quant_use_scheduler`
- `QUANT_CALIBRATION_ONLY=1`이면 `--quant_calibration_only`
- `QAT_ALLOW_BATCH=1`이면 `--quant_qat_allow_batch`
- 스크립트 뒤에 붙인 추가 CLI 인자는 그대로 `main.py`에 전달됩니다.

스크립트 뒤에 추가로 넘길 수 있는 quant CLI 인자는 다음과 같습니다.

| CLI 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--quant_range_method {mse,minmax}` | pipeline별 자동값 | range estimator입니다. `standard`는 기본 `mse`, `legacy`는 기본 `minmax`입니다. |
| `--quant_bn_folding` / `--quant_no_bn_folding` | `standard`에서 on | FrozenBatchNorm2d를 앞 Conv2d에 folding할지 제어합니다. |
| `--quant_cle` / `--quant_no_cle` | `standard`에서 on | safe cross-layer equalization 사용 여부입니다. |
| `--quant_adaround` / `--quant_no_adaround` | `standard`에서 on | AdaRound-style weight rounding 사용 여부입니다. |
| `--quant_adaround_samples N` | `128` | AdaRound-style rounding에 사용할 calibration sample 최대 개수입니다. |
| `--quant_adaround_iters N` | `1000` | AdaRound-style rounding optimization iteration budget입니다. |
| `--quant_mse_bins N` | `2048` | MSE range estimation histogram bin 수입니다. |
| `--quant_mse_candidates N` | `80` | MSE clipping candidate 수입니다. |
| `--quant_bias_correction {auto,on,off}` | `auto` | bias correction policy입니다. |

QAT를 이어서 학습하려면 기존 QAT checkpoint를 `--resume`으로 넘기면 됩니다. `main.py`는 `--pretrain`을 먼저 로드한 뒤 `--resume`을 로드하므로, 최종 weight와 optimizer state는 resume checkpoint가 덮어씁니다.

```bash
MODEL_VARIANT=lite \
QUANT_MODE=qat \
QUANT_PARTITION=exp_a1_to_b \
PRETRAIN_MODEL=../model_zoo/ovtr_lite.pth \
OUTPUT=./weights_qat_lite_exp_a1_to_b \
CUDA_DEVICES=0,1,2,3 \
NPROC_GPU=4 \
QAT_EPOCHS=2 \
./tools/ovtr_quant_full_model.sh \
  --resume ./weights_qat_lite_exp_a1_to_b/checkpoint.pth
```

QAT train 산출물은 `OUTPUT` 아래에 저장됩니다.

- `checkpoint.pth`: 최신 QAT checkpoint
- `checkpoint0000.pth` 등: `--save_period`에 따른 epoch checkpoint
- `quant_manifest.json`: quant mode, partition, bit width, quantized module, quant state 정보를 담은 manifest
- `log.txt`: train log

## 2. QAT Checkpoint Eval

QAT eval은 `ovtr/tools/ovtr_qat_eval.sh` 또는 variant별 wrapper를 사용합니다.

- lite val: `./tools/ovtr_qat_eval_lite_val.sh`
- lite test: `./tools/ovtr_qat_eval_lite_test.sh`
- 5-frame val: `./tools/ovtr_qat_eval_5_frame_val.sh`
- 5-frame test: `./tools/ovtr_qat_eval_5_frame_test.sh`

train에서 권장 `OUTPUT`을 사용했다면 eval은 기본 `CHECKPOINT_DIR`만으로 checkpoint를 찾습니다.

```bash
MODEL_VARIANT=lite \
EVAL_SPLIT=val \
QUANT_PARTITION=exp_a1_to_b \
CHECKPOINT_DIR=./weights_qat_lite_exp_a1_to_b \
CUDA_DEVICES=0 \
NPROC_GPU=1 \
MASTER_PORT=9988 \
OUTPUT=./results \
NUM_WORKERS=48 \
BATCH_SIZE=1 \
./tools/ovtr_qat_eval.sh
```

다른 checkpoint 파일을 직접 지정하려면 `PRETRAIN_MODEL`을 사용합니다.

```bash
MODEL_VARIANT=5_frame \
EVAL_SPLIT=test \
QUANT_PARTITION=exp_a3_b \
PRETRAIN_MODEL=./weights_qat_5_frame_exp_a3_b/checkpoint.pth \
RESULT_PATH=./results/teta_results_5_frame_qat_exp_a3_b_test \
VIS_OUTPUT=./results/vis_output_track_5_frame_qat_exp_a3_b_test \
CUDA_DEVICES=0 \
NPROC_GPU=1 \
./tools/ovtr_qat_eval.sh
```

QAT eval에서 자주 바꾸는 환경변수는 다음과 같습니다.

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `MODEL_VARIANT` | `lite` | `lite` 또는 `5_frame` |
| `EVAL_SPLIT` | `val` | `val` 또는 `test` |
| `QUANT_PARTITION` | `exp_a` | train 때 사용한 partition과 맞춰야 합니다. |
| `QUANT_WEIGHT_BITS` | `4` | train 때 사용한 weight bit와 맞춥니다. |
| `QUANT_ACTIVATION_BITS` | `4` | train 때 사용한 activation bit와 맞춥니다. |
| `QUANT_ATTENTION_BITS` | `8` | train 때 사용한 attention bit와 맞춥니다. |
| `CHECKPOINT_DIR` | `./weights_qat_${MODEL_VARIANT}_${QUANT_PARTITION}` | 기본 QAT checkpoint directory입니다. |
| `PRETRAIN_MODEL` | `${CHECKPOINT_DIR}/checkpoint.pth` | `eval.py --pretrain`으로 전달되는 QAT checkpoint입니다. |
| `CUDA_DEVICES` | `0` | `CUDA_VISIBLE_DEVICES`입니다. |
| `NPROC_GPU` | `1` | `torchrun --nproc_per_node` 값입니다. |
| `MASTER_PORT` | `9988` | `torchrun --master_port` 값입니다. |
| `OUTPUT` | `./results` | `eval.py --output_dir`입니다. |
| `NUM_WORKERS` | `48` | dataloader worker 수입니다. |
| `BATCH_SIZE` | `1` | eval batch size입니다. tracking state는 내부에서 frame 단위로 처리됩니다. |
| `RESULT_PATH` | variant/split별 자동 경로 | TETA tracking result directory입니다. |
| `VIS_OUTPUT` | variant/split별 자동 경로 | `--vis` output directory입니다. |

QAT eval에서 스크립트가 `eval.py`에 넘기는 핵심 인자는 아래와 같습니다.

```bash
torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_GPU}" \
  ./eval.py \
  --config_file "${CONFIG_FILE}" \
  --dataset_file lvis_generated_img_seqs \
  --epochs 16 \
  --with_box_refine \
  --two_stage \
  --lr 4e-5 \
  --lr_backbone 4e-6 \
  --lr_drop 13 \
  --pretrain "${PRETRAIN_MODEL}" \
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
  --quant_mode qat \
  --quant_partition "${QUANT_PARTITION}" \
  --quant_weight_bits "${QUANT_WEIGHT_BITS}" \
  --quant_activation_bits "${QUANT_ACTIVATION_BITS}" \
  --quant_attention_bits "${QUANT_ATTENTION_BITS}" \
  --score_thresh ${SCORE_THRESH} \
  --filter_score_thresh ${FILTER_SCORE_THRESH} \
  --ious_thresh ${IOUS_THRESH} \
  --miss_tolerance 5 5 5 5 5 5 5 \
  --maximum_quantity 160 \
  --vis \
  --result_path_track "${RESULT_PATH}" \
  --vis_output "${VIS_OUTPUT}"
```

`SCORE_THRESH`, `FILTER_SCORE_THRESH`, `IOUS_THRESH`는 variant별로 스크립트 안에서 자동 설정됩니다.

| variant | score/filter threshold | IoU threshold |
| --- | --- | --- |
| `lite` | `0.19 0.19 0.19 0.19 0.19 0.19 0.19` | `0.45 0.45 0.45 0.45 0.45 0.45 0.45` |
| `5_frame` | `0.20 0.17 0.17 0.20 0.17 0.20 0.17` | `0.5 0.45 0.5 0.4 0.45 0.45 0.45` |

QAT eval 산출물은 다음 위치에 생성됩니다.

- `RESULT_PATH`: tracking result 및 TETA 평가 입력/출력
- `VIS_OUTPUT`: visualization 결과
- `OUTPUT/quant_manifest.json`: eval 시점의 quant manifest

## 3. FP32 Baseline 대비 Quant Drift Analysis

Quant drift analysis는 FP32 baseline과 quantized target을 같은 sequence에서 비교합니다. 실행 mode는 세 가지입니다.

- `fp32_free`: FP32 model normal tracking run
- `quant_free`: quantized model normal tracking run
- `quant_teacher_forced`: quantized model에 FP32 recurrent track query를 주입한 run

QAT checkpoint를 분석하려면 `QUANT_MODE=qat`로 실행합니다.

```bash
MODEL_VARIANT=lite \
EVAL_SPLIT=val \
QUANT_MODE=qat \
QUANT_PARTITION=exp_a1_to_b \
FP32_PRETRAIN_MODEL=../model_zoo/ovtr_lite.pth \
CHECKPOINT_DIR=./weights_qat_lite_exp_a1_to_b \
PRETRAIN_MODEL=./weights_qat_lite_exp_a1_to_b/checkpoint.pth \
ANALYSIS_OUTPUT_DIR=./results/quant_drift_lite_qat_exp_a1_to_b_val \
ANALYSIS_MAX_FRAMES=300 \
ANALYSIS_SAMPLE_SEQS_PER_DATASET=3 \
CUDA_DEVICES=0 \
NUM_WORKERS=8 \
BATCH_SIZE=1 \
./tools/ovtr_quant_drift_analysis.sh
```

5-frame test split 예시는 다음과 같습니다.

```bash
MODEL_VARIANT=5_frame \
EVAL_SPLIT=test \
QUANT_MODE=qat \
QUANT_PARTITION=exp_a3_b \
FP32_PRETRAIN_MODEL=../model_zoo/ovtr_5_frame.pth \
PRETRAIN_MODEL=./weights_qat_5_frame_exp_a3_b/checkpoint.pth \
ANALYSIS_OUTPUT_DIR=./results/quant_drift_5_frame_qat_exp_a3_b_test \
ANALYSIS_MAX_FRAMES=0 \
ANALYSIS_SAMPLE_SEQS_PER_DATASET=10 \
CUDA_DEVICES=0 \
NUM_WORKERS=8 \
BATCH_SIZE=1 \
./tools/ovtr_quant_drift_analysis.sh
```

Analysis에서 quant boundary error까지 기록하려면 스크립트 뒤에 analysis CLI 인자를 추가합니다.

```bash
MODEL_VARIANT=lite \
EVAL_SPLIT=val \
QUANT_MODE=qat \
QUANT_PARTITION=exp_a1_to_b \
FP32_PRETRAIN_MODEL=../model_zoo/ovtr_lite.pth \
PRETRAIN_MODEL=./weights_qat_lite_exp_a1_to_b/checkpoint.pth \
ANALYSIS_OUTPUT_DIR=./results/quant_drift_lite_qat_exp_a1_to_b_boundaries \
ANALYSIS_MAX_FRAMES=100 \
CUDA_DEVICES=0 \
./tools/ovtr_quant_drift_analysis.sh \
  --analysis_record_quant_boundaries \
  --analysis_quant_boundary_module_regex '^(backbone|transformer\\.decoder|track_embed)' \
  --analysis_quant_boundary_max_rows_per_frame 200
```

Analysis에서 자주 바꾸는 환경변수는 다음과 같습니다.

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `MODEL_VARIANT` | `5_frame` | `lite` 또는 `5_frame` |
| `EVAL_SPLIT` | `val` | `val` 또는 `test` |
| `QUANT_MODE` | `ptq` | QAT checkpoint 분석은 `qat`로 설정합니다. |
| `QUANT_PARTITION` | `exp_a3` | 분석 대상 quant partition입니다. |
| `FP32_PRETRAIN_MODEL` | variant별 FP32 checkpoint | `--analysis_fp32_pretrain`입니다. |
| `CHECKPOINT_DIR` | `./weights_qat_${MODEL_VARIANT}_${QUANT_PARTITION}` | `QUANT_MODE=qat`일 때 기본 QAT checkpoint directory입니다. |
| `PRETRAIN_MODEL` | QAT이면 `${CHECKPOINT_DIR}/checkpoint.pth` | 분석 대상 quantized checkpoint입니다. |
| `ANALYSIS_OUTPUT_DIR` | `./results/quant_drift_${MODEL_VARIANT}_${QUANT_MODE}_${QUANT_PARTITION}_${EVAL_SPLIT}` | CSV/JSON/PNG 분석 산출물 directory입니다. |
| `ANALYSIS_MAX_FRAMES` | `0` | 처리할 최대 frame 수입니다. `0`은 split 전체입니다. |
| `ANALYSIS_PLOT_MAX_AGE` | `0` | track age plot의 최대 age입니다. `0`은 제한 없음입니다. |
| `ANALYSIS_IOU_DIVERGENCE_THRESH` | `0.5` | matched track을 low-IoU divergence로 표시하는 threshold입니다. |
| `ANALYSIS_SAMPLE_SEQS_PER_DATASET` | `10` | dataset prefix별 sampling할 sequence 수입니다. `0`이면 sampling을 끕니다. |
| `ANALYSIS_SAMPLE_DATASETS` | `YFCC100M HACS BDD ArgoVerse AVA LaSOT Charades` | sampling 대상 dataset prefix 목록입니다. |
| `CUDA_DEVICES` | `0` | analysis는 single-process `python` 실행입니다. |
| `OUTPUT` | `./results` | 공통 output root입니다. |
| `NUM_WORKERS` | `8` | dataloader worker 수입니다. |
| `BATCH_SIZE` | `1` | analysis는 코드에서 `--batch_size 1`을 요구합니다. |
| `CALIB_SAMPLES` | `512` | PTQ 분석에서 target checkpoint에 quant state가 없을 때 calibration에 사용할 sample 수입니다. QAT 분석은 learned quant state가 없는 checkpoint를 허용하지 않습니다. |
| `QUANT_WEIGHT_BITS` | `4` | quantized target의 weight bit입니다. |
| `QUANT_ACTIVATION_BITS` | `4` | quantized target의 activation bit입니다. |
| `QUANT_ATTENTION_BITS` | `8` | quantized target의 attention bit입니다. |

Analysis에서 스크립트가 `analyze_quant_drift.py`에 넘기는 핵심 인자는 아래와 같습니다.

```bash
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
  --maximum_quantity 160
```

Analysis 전용 추가 CLI 인자는 다음과 같습니다.

| CLI 인자 | 설명 |
| --- | --- |
| `--analysis_record_quant_boundaries` | quantized module boundary의 fake-quant 전후 tensor error를 `quant_boundary_errors.csv`에 기록합니다. |
| `--analysis_quant_boundary_module_regex REGEX` | boundary 기록 대상 module name regex입니다. 생략하면 partition에 맞는 기본 regex를 사용합니다. |
| `--analysis_quant_boundary_max_rows_per_frame N` | frame당 boundary row 최대 개수입니다. `0`은 제한 없음입니다. |

Analysis 산출물은 `ANALYSIS_OUTPUT_DIR` 아래에 생성됩니다.

- `metrics_summary.json`: `fp32_free`, `quant_free`, `quant_teacher_forced`별 TETA/IDF1/MOTA 요약
- `track_metrics.csv`: track-level drift metric
- `frame_metrics.csv`: frame-level drift metric
- `divergences.csv`: first divergence row
- `one_step_errors.csv`: teacher-forced one-step error
- `accumulation_gaps.csv`: quant free-run과 teacher-forced run 사이의 accumulated drift gap
- `state_io_errors.csv`: recurrent state input/output 비교
- `recurrent_query_errors.csv`: recurrent query drift
- `quant_boundary_errors.csv`: `--analysis_record_quant_boundaries` 사용 시 module boundary error
- plot PNG 파일들

## 실행 순서 예시

lite 모델의 `exp_a1_to_b` QAT를 학습, val 평가, drift 분석까지 한 번에 이어서 실행하는 전체 예시는 다음과 같습니다.

```bash
cd ovtr

MODEL_VARIANT=lite \
QUANT_MODE=qat \
QUANT_PARTITION=exp_a1_to_b \
PRETRAIN_MODEL=../model_zoo/ovtr_lite.pth \
OUTPUT=./weights_qat_lite_exp_a1_to_b \
CUDA_DEVICES=0,1,2,3 \
NPROC_GPU=4 \
BATCH_SIZE=1 \
CALIB_SAMPLES=512 \
QAT_EPOCHS=1 \
./tools/ovtr_quant_full_model.sh

MODEL_VARIANT=lite \
EVAL_SPLIT=val \
QUANT_PARTITION=exp_a1_to_b \
CHECKPOINT_DIR=./weights_qat_lite_exp_a1_to_b \
CUDA_DEVICES=0 \
NPROC_GPU=1 \
./tools/ovtr_qat_eval.sh

MODEL_VARIANT=lite \
EVAL_SPLIT=val \
QUANT_MODE=qat \
QUANT_PARTITION=exp_a1_to_b \
FP32_PRETRAIN_MODEL=../model_zoo/ovtr_lite.pth \
PRETRAIN_MODEL=./weights_qat_lite_exp_a1_to_b/checkpoint.pth \
ANALYSIS_OUTPUT_DIR=./results/quant_drift_lite_qat_exp_a1_to_b_val \
ANALYSIS_MAX_FRAMES=300 \
CUDA_DEVICES=0 \
./tools/ovtr_quant_drift_analysis.sh
```

## 주의 사항

- QAT eval은 checkpoint 안에 `_ovtr_quant_*` quant state가 있어야 합니다. FP32 checkpoint를 `ovtr_qat_eval.sh`에 넣으면 `QAT evaluation expects a checkpoint that already contains learned quant state.` 오류가 납니다.
- QAT analysis도 learned quant state가 포함된 QAT checkpoint가 필요합니다. FP32 checkpoint만 넣으면 `QAT drift analysis expects a checkpoint that already contains learned quant state.` 오류가 납니다.
- QAT train은 선택한 partition의 FP32 weight와 quant parameter만 trainable로 풀고 나머지는 freeze합니다.
- QAT mode에서 `main.py`는 fixed 5-frame sampling을 강제하고, `QAT_ALLOW_BATCH=1`이 아닌 경우 transformer/frame-wise checkpointing을 자동으로 켭니다.
- Analysis는 single-process 실행만 지원합니다. `torchrun`이 아니라 스크립트처럼 `python ./analyze_quant_drift.py`로 실행해야 합니다.
- Analysis는 `BATCH_SIZE=1`만 허용합니다.
- `QUANT_PARTITION`, bit width, FP32 baseline checkpoint는 train/eval/analysis 사이에서 일관되게 맞추는 것이 좋습니다.
