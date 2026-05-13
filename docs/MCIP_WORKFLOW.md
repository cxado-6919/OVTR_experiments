# M-CIP 변경 사항 및 실행 방법

이 문서는 OVTR에 추가된 M-CIP(Memory-calibrated Category Information Propagation) 연구 모듈의 변경점, 주요 인자, 학습/평가 실행 방법을 정리합니다.

M-CIP는 양자화 실험이 아니라 tracking 성능 개선을 위한 opt-in 모듈입니다. 기본값은 모두 비활성화되어 있으며, 기존 baseline config를 그대로 쓰면 원래 OVTR/CIP 경로가 유지됩니다.

## 변경 요약

- `track_query_iteration=CIP`일 때만 M-CIP가 개입합니다.
- `mcip_enable=False`이면 기존 `Category_Information_Propagator`가 그대로 사용됩니다.
- `mcip_enable=True`이면 `MemoryCalibratedCategoryInformationPropagator`가 사용됩니다.
- 기존 baseline config 파일은 수정하지 않았습니다.
- 새 config는 두 개입니다.
  - `config/ovtr_5_frame_mcip_train_val.py`: M-CIP 활성화, attention protection은 기존 KL 유지
  - `config/ovtr_5_frame_mcip_topk_train_val.py`: M-CIP 활성화, top-k sparse attention protection 사용
- M-CIP와 top-k attention protection 효과를 따로 ablation할 수 있도록 main M-CIP config는 top-k를 켜지 않습니다.

## 구현된 기능

M-CIP는 기존 CIP updater를 확장하여 track별 compact memory를 유지합니다.

- image memory: `img_memory`
- semantic memory: `semantic_memory`
- class confidence memory: `cls_conf_memory`
- class entropy memory: `cls_entropy_memory`
- previous box: `prev_boxes`
- box velocity: `box_velocity`
- memory age: `memory_age`

semantic memory는 full class distribution을 저장하지 않고 hidden-dim feature만 저장합니다. 현재 frame의 class logits와 text feature 길이가 다를 수 있으므로 다음 방식으로 안전하게 맞춥니다.

```python
cls_len = min(pred_logits.shape[-1], text_feat.shape[0])
logits_for_memory = pred_logits[..., :cls_len]
text_feat_for_memory = text_feat[:cls_len]
prob = softmax(logits_for_memory.float(), dim=-1)
semantic_obs = prob @ text_feat_for_memory.float()
```

`cls_len == 0`이면 오류를 냅니다. 길이 mismatch는 `debug_mcip=True`일 때 debug stat으로 기록하며, 실행을 중단하지 않습니다.

motion ref propagation은 예측 box와 velocity를 이용하지만, 초기 learned scale은 `0.0`입니다. 따라서 학습 초반에는 baseline ref point update와 가깝게 시작합니다. `inverse_sigmoid` 전 box는 `[1e-4, 1 - 1e-4]`로 clamp됩니다.

## 주요 인자

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--mcip_enable` / `--no_mcip_enable` | config/default 기준 `False` | M-CIP updater 활성화 |
| `--mcip_detach_memory` / `--no_mcip_detach_memory` | `True` | persistent memory를 detach하여 긴 temporal graph/OOM 방지 |
| `--mcip_memory_momentum` | `0.8` | learned gate update를 감쇠하는 memory momentum |
| `--mcip_use_semantic_memory` / `--no_mcip_use_semantic_memory` | `True` | semantic memory 사용 여부 |
| `--mcip_use_motion_ref` / `--no_mcip_use_motion_ref` | `True` | motion-conditioned ref point propagation 사용 여부 |
| `--mcip_motion_momentum` | `0.7` | box velocity EMA momentum |
| `--mcip_motion_scale_init` | `0.0` | motion ref learned scale 초기값 |
| `--mcip_gate_use_txt` | `False` | 예약 인자. v1 gate에는 `output_embedding_txt`를 넣지 않음 |
| `--debug_mcip` | `False` | M-CIP debug stat 및 trainable M-CIP parameter 이름 출력 |
| `--attention_protection_mode` | `kl` | `kl`, `topk`, `none` |
| `--attention_protection_topk` | `3` | top-k attention protection에서 query별 class top-k |
| `--attention_protection_conf_thresh` | `0.25` | top-k mask 적용 confidence threshold |

## Config 선택

Baseline:

```bash
--config_file ./config/ovtr_5_frame_train_val.py
```

M-CIP only:

```bash
--config_file ./config/ovtr_5_frame_mcip_train_val.py
```

M-CIP + top-k attention protection:

```bash
--config_file ./config/ovtr_5_frame_mcip_topk_train_val.py
```

두 M-CIP config는 tracking-stage freeze filter에서 새 M-CIP parameter가 얼지 않도록 `train_tracking_only`에 `track_embed.gate_mlp`, `track_embed.memory_img_proj`, `track_embed.memory_sem_proj`, `track_embed.motion_scale`를 포함합니다.

## 학습 실행

학습 스크립트는 `CONFIG_FILE`, `OUTPUT`, `BATCH_SIZE`, `EXTRA_ARGS` 환경변수를 받을 수 있습니다.

```bash
cd ovtr
bash tools/ovtr_multi_frame_train.sh
```

예를 들어 세 ablation 조합은 다음처럼 실행합니다.

```bash
cd ovtr

# 1. M-CIP + KL protection
CONFIG_FILE=./config/ovtr_5_frame_mcip_train_val.py \
OUTPUT=./weights_mcip_kl \
bash tools/ovtr_multi_frame_train.sh

# 2. Baseline + top-k protection
CONFIG_FILE=./config/ovtr_5_frame_train_val.py \
EXTRA_ARGS="--attention_protection_mode topk" \
OUTPUT=./weights_baseline_topk \
bash tools/ovtr_multi_frame_train.sh

# 3. M-CIP + top-k protection
CONFIG_FILE=./config/ovtr_5_frame_mcip_topk_train_val.py \
OUTPUT=./weights_mcip_topk \
bash tools/ovtr_multi_frame_train.sh
```

`EXTRA_ARGS`는 단순한 추가 CLI 인자를 뒤에 붙이는 용도입니다. 공백이 들어가는 path/value에는 쓰지 않는 것을 권장합니다.

`debug_mcip`로 새 M-CIP parameter가 실제 trainable인지 확인할 수 있습니다.

```bash
EXTRA_ARGS="--debug_mcip" bash tools/ovtr_multi_frame_train.sh
```

이 옵션을 켜면 freeze filter 적용 후 trainable M-CIP parameter 이름이 출력됩니다.

## 평가 실행

평가 스크립트도 `CONFIG_FILE`, `PRETRAIN_MODEL`, `RESULT_PATH`, `VIS_OUTPUT`, `EXTRA_ARGS` 환경변수를 받을 수 있습니다.

```bash
cd ovtr
bash tools/ovtr_ovmot_eval_e15_val.sh
```

학습한 세 조합을 같은 순서로 평가하는 예시는 다음과 같습니다.

```bash
cd ovtr

# 1. M-CIP + KL protection
CONFIG_FILE=./config/ovtr_5_frame_mcip_train_val.py \
PRETRAIN_MODEL=./weights_mcip_kl/checkpoint0015.pth \
RESULT_PATH=./results/teta_results_mcip_kl_val \
VIS_OUTPUT=./results/vis_output_track_mcip_kl_val \
bash tools/ovtr_ovmot_eval_e15_val.sh

# 2. Baseline + top-k protection
CONFIG_FILE=./config/ovtr_5_frame_train_val.py \
EXTRA_ARGS="--attention_protection_mode topk" \
PRETRAIN_MODEL=./weights_baseline_topk/checkpoint0015.pth \
RESULT_PATH=./results/teta_results_baseline_topk_val \
VIS_OUTPUT=./results/vis_output_track_baseline_topk_val \
bash tools/ovtr_ovmot_eval_e15_val.sh

# 3. M-CIP + top-k protection
CONFIG_FILE=./config/ovtr_5_frame_mcip_topk_train_val.py \
PRETRAIN_MODEL=./weights_mcip_topk/checkpoint0015.pth \
RESULT_PATH=./results/teta_results_mcip_topk_val \
VIS_OUTPUT=./results/vis_output_track_mcip_topk_val \
bash tools/ovtr_ovmot_eval_e15_val.sh
```

## Checkpoint 호환성

`mcip_enable=True` 상태에서 기존 OVTR checkpoint를 `--pretrain` 또는 `--resume`으로 로드하면 새 M-CIP parameter는 checkpoint에 없을 수 있습니다. 이 경우 새 M-CIP key만 허용 missing key로 분류하여 명확히 출력합니다.

허용되는 missing key 범위:

- `track_embed.gate_mlp.*`
- `track_embed.memory_img_proj.*`
- `track_embed.memory_sem_proj.*`
- `track_embed.motion_scale`

이 외의 missing/unexpected key는 숨기지 않고 기존처럼 출력합니다.

## 검증

baseline 경로 검증:

```bash
cd /home/pjh/clone_repo/OVTR
conda run -n OVTR python -m compileall ovtr/models ovtr/main.py ovtr/eval.py

cd ovtr
conda run -n OVTR python tools/smoke_test.py
```

M-CIP 최소 검증:

```bash
cd /home/pjh/clone_repo/OVTR/ovtr
conda run -n OVTR python tools/mcip_smoke_test.py
```

`tools/mcip_smoke_test.py`는 다음을 확인합니다.

- dummy `MemoryCalibratedCategoryInformationPropagator` forward
- `ref_pts`, `query_tgt`, `img_memory`, `semantic_memory`, `box_velocity` shape
- NaN/Inf 없음
- inference-style `RuntimeTrackerBase.update` 후 M-CIP field 보존
- field가 누락된 경우 lazy restore 동작

## Ablation 권장 순서

1. Baseline OVTR/CIP: `ovtr_5_frame_train_val.py`
2. M-CIP only: `ovtr_5_frame_mcip_train_val.py`
3. M-CIP + top-k attention protection: `ovtr_5_frame_mcip_topk_train_val.py`

이 순서로 비교하면 M-CIP memory 효과와 top-k attention protection 효과를 분리해서 볼 수 있습니다.
