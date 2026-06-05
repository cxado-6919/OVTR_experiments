# QAT-Implement 브랜치 변경 사항

이 문서는 `main`과 비교했을 때 `QAT-Implement` 브랜치에 누적된 변경 사항을 요약합니다.

비교 기준:

- 확인한 Git 범위: `main...HEAD`
- 현재 브랜치: `QAT-Implement`
- 작업트리 기준: 아직 커밋되지 않은 decoder checkpointing 변경과 QAT 메모리 문서화 변경도 현재 브랜치 상태를 설명하는 범위에 포함했습니다.
- `repomix-output.xml` 같은 untracked 로컬 산출물은 브랜치 변경 사항으로 보지 않았습니다.

## 전체 요약

이 브랜치는 OVTR과 detection-pretraining 트리를 Blackwell GPU에서 사용할 수 있는 최신 환경으로 포팅하고, full-model PTQ/QAT 지원, quantization partition 확장, quantization drift 분석 도구, CUDA extension build/deploy 경로 개선, quantized training/evaluation workflow용 스크립트를 추가합니다.

주요 변경 축:

- Blackwell GPU를 위한 최신 Python/PyTorch/CUDA 환경 지원.
- `ovtr`와 `ovtr_det_bs2_pretrain` 양쪽의 package-local CUDA extension build 지원.
- PTQ, QAT, calibration, manifest 생성, checkpoint load, 선택적 int-MSDA deploy를 포함한 full-model OVTR quantization 인프라.
- `exp_a1_to_b`, `exp_a3_b`를 포함한 새 quantization partition.
- forced checkpointing과 decoder activation checkpointing을 통한 QAT 메모리 대응.
- FP32 free-run, quantized free-run, quantized teacher-forced tracking을 비교하는 quantization drift 분석.
- 최신 dependency 호환성을 위한 evaluation/tracking/data pipeline 호환 shim 업데이트.

## 실행 환경 및 Blackwell 포팅

이 브랜치는 기존 legacy 환경에서 Blackwell GPU에 적합한 CUDA 12.8 / PyTorch 2.7 기반 환경으로 실행 전제를 업데이트합니다.

중요 변경 사항:

- setup, CUDA op build, CLIP dependency, smoke test, troubleshooting 내용을 담은 `docs/BLACKWELL_SETUP.md`를 추가했습니다.
- environment 생성과 local CUDA op build를 위한 `scripts/setup_blackwell.sh`를 추가했습니다.
- `TORCH_CUDA_ARCH_LIST=12.0+PTX`를 포함해 CUDA compiler/toolchain 환경 변수를 설정하는 `tools/ovtr_blackwell_env.sh`를 추가했습니다.
- 최신 Python/PyTorch dependency set에 맞게 `requirements.txt`를 업데이트했습니다.
- precomputed embedding 파일이 있을 때 normal training/eval에서 CLIP package import가 필수가 아니도록 했습니다.

CUDA extension 변경:

- `ovtr/models/ops`와 `ovtr_det_bs2_pretrain/models/ops`가 각각 package-local `_C` extension을 build합니다.
- setup script가 기존 CUDA 11.x stack을 가정하지 않고 CUDA architecture 설정을 직접 존중하거나 추론합니다.
- 최신 PyTorch/CUDA toolchain에 맞게 MSDeformAttn extension 경로를 업데이트했습니다.

참고: 현재 브랜치 diff에는 ops tree 아래의 compiled `.so`, `.o`, `build/` artifact가 포함되어 있습니다. 프로젝트에서 build binary를 의도적으로 추적하는 경우에만 유지하고, 그렇지 않다면 merge 범위에서 제거해야 합니다.

## OVTR 양자화

이 브랜치는 `ovtr/models/quant_utils.py`와 `ovtr/util/quantization.py`에 주요 quantization runtime을 추가합니다.

핵심 기능:

- `--quant_mode {none,ptq,qat}`로 normal FP32, PTQ, QAT 실행을 제어합니다.
- `--quant_partition`으로 quantize할 model partition과 QAT에서 trainable로 둘 partition을 선택합니다.
- `--quant_pipeline {legacy,standard}`로 preparation pipeline을 선택합니다.
- weight, activation, attention bit width를 설정할 수 있습니다.
- PTQ는 calibration, quantized checkpoint save, quant manifest 생성을 지원합니다.
- QAT는 calibration initialization, learned LSQ-style quantization parameter, quantized checkpoint continuation을 지원합니다.
- Quant state는 `_ovtr_quant_*` key로 직렬화되며 normal model weight와 분리해서 reload할 수 있습니다.
- Quant manifest에는 mode, partition, bit width, 선택된 module, trainable quant param, int-MSDA export metadata가 포함됩니다.

Quantized module 적용 범위:

- `nn.Conv2d`
- `nn.Linear`
- `nn.Embedding`
- `nn.MultiheadAttention`
- `MultiScaleDeformableAttention`

Standard quantization preparation이 추가하는 기능:

- FrozenBatchNorm을 앞선 convolution에 folding.
- Safe cross-layer equalization.
- AdaRound-style weight rounding.
- MSE histogram range search.
- Bias materialization / correction 지원.

Calibration 변경:

- `cfg.data.calib` 기반 전용 quant calibration loader를 추가했습니다.
- Pseudo-sequence calibration 제어 옵션을 추가했습니다. 현재 기본 calibration path는 static-frame이며 pseudo-sequence calibration은 비활성화되어 있습니다:
  - `--quant_calib_sequence_length`
  - `--quant_calib_max_translate`
  - `--quant_calib_max_rotate`
  - `--quant_calib_scale_jitter`
  - `--quant_calib_motion_blur`
  - `--quant_disable_pseudo_sequence_calib`
- Calibration split 생성을 위한 `process/create_lvis_calibration_split.py`를 추가했습니다.

Deploy 변경:

- eval/inference-only int-MSDA deployment를 위한 `--quant_deploy int_msda`를 추가했습니다.
- int-MSDA deploy가 `weight_bits=4`, `activation_bits=4`, `attention_bits=8`을 사용하도록 검사합니다.
- int4 weight와 uint4/uint8 activation/attention quantization metadata를 위한 low-bit packing/export buffer를 추가했습니다.

## 양자화 Partition

이 브랜치는 quantized OVTR fine-tuning을 위한 experiment partition을 추가하고 일부 의미를 정리합니다.

지원 partition:

- `exp_a`
- `exp_a1`
- `exp_a1_backbone`
- `exp_a1_backbone_input_proj`
- `exp_a2`
- `exp_a3`
- `exp_a3_head`
- `exp_b`
- `exp_a1_to_b`
- `exp_a3_b`

Partition 의미:

- `exp_a1`: backbone, `input_proj`, `patch2query`.
- `exp_a1_backbone`: backbone만 포함합니다.
- `exp_a1_backbone_input_proj`: backbone과 `input_proj`를 포함하고 `patch2query`는 제외합니다.
- `exp_a2`: transformer encoder aggregation과 encoder output head. 단, fusion layer는 제외합니다.
- `exp_a3`: decoder와 `tgt_embed`. 단, decoder output head는 제외합니다.
- `exp_a3_head`: decoder, `tgt_embed`, output-head component.
- `exp_b`: `track_embed`.
- `exp_a`: 기존 full A 정의를 사용하는 combined A-side quantization.
- `exp_a1_to_b`: `a1 + a2 + exp_a3 + b`. A3 output head는 의도적으로 제외합니다.
- `exp_a3_b`: `exp_a3 + b`. 이 partition도 A3 output head를 제외합니다.

Head-exclusion 동작은 `exp_a3`, `exp_a1_to_b`, `exp_a3_b`에서 중요합니다. Head-excluded variant에는 `transformer.decoder.bbox_embed`와 `feature_align`이 포함되지 않습니다.

## QAT 학습 동작 및 메모리

QAT는 선택된 floating-point weight와 learned quantization parameter를 함께 trainable로 유지합니다. 이 방식은 QAT semantics를 보존하지만, 원본 OVTR fine-tuning 경로보다 메모리 사용량을 증가시킵니다.

QAT 관련 동작:

- `main.py`는 QAT를 fixed 5-frame sampling으로 강제합니다.
- `--quant_qat_allow_batch`가 `batch_size > 1`과 함께 사용되지 않는 한 QAT에서 transformer checkpointing과 frame-wise checkpointing을 자동으로 활성화합니다.
- Batched QAT는 experimental로 표시되어 있으며 QAT-only checkpoint forcing path를 비활성화합니다.
- `use_transformer_ckpt`를 `TransformerDecoder`까지 확장하여 decoder activation checkpointing을 추가했습니다.

Decoder checkpointing 세부 사항:

- training mode에서만 적용됩니다.
- `torch.utils.checkpoint(..., use_reentrant=False)`를 사용합니다.
- decoder layer body만 checkpoint합니다.
- reference point update, bbox head, class logit, aux-output collection은 checkpoint boundary 밖에 둡니다.
- Calibration/eval path는 direct forward path를 유지하므로 observer/calibration side effect가 checkpoint recomputation으로 반복되지 않습니다.

권장 low-memory QAT baseline은 `docs/BLACKWELL_SETUP.md`에 문서화되어 있습니다.

## 양자화 Drift 분석

이 브랜치는 quantization 상태에서 tracking quality와 state drift를 분석하기 위해 `ovtr/analyze_quant_drift.py`와 `ovtr/util/quant_drift_analysis.py`를 추가합니다.

분석 mode:

- `fp32_free`: FP32 model이 normal run을 수행하고 reference trace를 기록합니다.
- `quant_free`: quantized model이 normal run을 수행합니다.
- `quant_teacher_forced`: quantized model이 FP32 trace에서 active recurrent track query를 받습니다. Detection query는 quantized model에서 그대로 나옵니다.

출력 artifact:

- TETA, IDF1, MOTA 관련 run-level metric summary
- frame-level metric row
- track-level metric row
- divergence row
- state I/O row
- recurrent query drift row
- 선택적 quant boundary error row
- 선택한 analysis output directory 아래의 CSV artifact와 plot

주요 옵션:

- `--analysis_fp32_pretrain`
- `--analysis_output_dir`
- `--analysis_max_frames`
- `--analysis_iou_divergence_thresh`
- `--analysis_sample_sequences_per_dataset`
- `--analysis_sample_datasets`
- `--analysis_record_quant_boundaries`
- `--analysis_quant_boundary_module_regex`
- `--analysis_quant_boundary_max_rows_per_frame`

Wrapper script:

- `ovtr/tools/ovtr_quant_drift_analysis.sh`

## Script 및 실행 진입점

새로 추가되었거나 업데이트된 OVTR script:

- `ovtr/tools/ovtr_quant_full_model.sh`: 주요 PTQ/QAT 실행 진입점.
- `ovtr/tools/ovtr_quant_lite_example.sh`: lite-model quantization wrapper.
- `ovtr/tools/ovtr_quant_5_frame_example.sh`: 5-frame quantization wrapper.
- `ovtr/tools/ovtr_qat_eval.sh`: QAT checkpoint evaluation 실행 진입점.
- `ovtr/tools/ovtr_qat_eval_lite_{val,test}.sh`: lite QAT eval wrapper.
- `ovtr/tools/ovtr_qat_eval_5_frame_{val,test}.sh`: 5-frame QAT eval wrapper.
- `ovtr/tools/ovtr_quant_drift_analysis.sh`: quant drift analysis wrapper.
- `ovtr/tools/quant_smoke_test.py`: quantization smoke test.
- `ovtr/tools/smoke_test.py`: repo-local runtime smoke test.

Standard training/eval/demo script도 이 브랜치에서 사용하는 최신 config path, model variant, result path, random-interval sequence sampling에 맞게 업데이트되었습니다.

## Tracking, Evaluation, Dataset 호환성

이 브랜치는 원래 코드가 의존하던 legacy external dependency 전제를 줄이기 위해 tracking/evaluation/data stack의 많은 부분을 업데이트합니다.

중요 영역:

- 현재 code path에 필요한 local `mmcv`, `mmdet`, Detectron2 compatibility shim을 추가했습니다.
- COCO/LVIS/TAO dataset loader, parser, sampler, pipeline transform을 업데이트했습니다.
- TETA evaluation module과 tracking metric utility를 업데이트했습니다.
- Visualization utility와 demo/eval path를 업데이트했습니다.
- Checked-in binary artifact였던 `ovtr/results/track_demo.gif`를 branch에서 제거했습니다.

이 변경들은 단일 quantization feature라기보다 넓은 범위의 호환성 개선 및 현대화 작업입니다.

## Detection Pretraining 트리

`ovtr_det_bs2_pretrain` tree에도 `ovtr`와 같은 modern runtime/CUDA compatibility 작업이 적용되었습니다.

주요 변경:

- Dataset, transform, utility module, backbone, transformer, MSDeformAttn code path를 최신 환경에 맞게 정리했습니다.
- Package-local CUDA op build를 지원합니다.
- `ovtr_det_bs2_pretrain/tools/smoke_test.py`를 추가했습니다.
- `ovtr_det_bs2_pretrain/models/qat_utils.py`가 추가되었지만, detection-pretraining Group-A QAT entry point는 현재 deprecated 상태입니다.
- `ovtr_det_bs2_pretrain/tools/ovtr_detection_pretrain_group_a_qat.sh`는 detection-pretraining Group-A QAT가 제거되었음을 알리고 `ovtr/tools/ovtr_quant_full_model.sh` 사용을 안내합니다.

## 추가 또는 업데이트된 검증

Smoke test:

- `ovtr/tools/smoke_test.py`
- `ovtr_det_bs2_pretrain/tools/smoke_test.py`
- `ovtr/tools/quant_smoke_test.py`

Quant smoke coverage:

- MSE range observer behavior.
- AdaRound commit path.
- PTQ legacy min-max path.
- bias correction path.
- QAT quant parameter creation.
- low-bit packing/unpacking reconstruction.
- uint4 zero-point correction formula.
- `exp_a1_to_b`, `exp_a3_b` combined partition coverage.

## Merge 및 정리 참고 사항

이 브랜치를 merge하기 전에 다음 항목을 검토해야 합니다.

- 두 CUDA op tree 아래에 build artifact가 diff에 포함되어 있습니다. Compiled `.so`, `.o`, `.ninja_*`, `build/` file을 계속 추적할지 확인해야 합니다.
- `ovtr/main.py.before_cleanup`가 tracked added file로 존재합니다. 의도적인 historical context인지 제거 대상인지 확인해야 합니다.
- `repomix-output.xml`는 현재 local untracked 상태이며 branch comparison에는 포함되지 않았습니다.
- 이 브랜치는 vendored/shim file과 dataset/evaluation module을 많이 변경합니다. Review는 file-by-file diff보다 behavior-level validation 중심으로 진행하는 것이 좋습니다.
- 큰 partition에서 QAT는 여전히 OOM이 날 수 있습니다. Decoder checkpointing은 activation memory를 줄이지만 trainable floating-point weight에 대한 optimizer state는 줄이지 않습니다.
