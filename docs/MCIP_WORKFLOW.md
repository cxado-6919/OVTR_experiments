# M-CIP v2 변경 사항 및 실행 방법

이 문서는 OVTR에 추가된 M-CIP(Memory-calibrated Category Information Propagation) v2 연구 모듈의 변경점, 주요 인자, 학습/평가 실행 방법을 정리합니다.

M-CIP는 양자화 실험이 아니라 tracking 성능 개선을 위한 opt-in 모듈입니다. 기본값은 모두 비활성화되어 있으며, 기존 baseline config를 그대로 쓰면 원래 OVTR/CIP 경로가 유지됩니다.

## 변경 요약

- `track_query_iteration=CIP`일 때만 M-CIP가 개입합니다.
- `mcip_enable=False`이면 기존 `Category_Information_Propagator`가 그대로 사용됩니다.
- `mcip_enable=True`이면 `MemoryCalibratedCategoryInformationPropagator`가 사용됩니다.
- M-CIP v2는 `O_img` replacement가 아닙니다.
- M-CIP v2는 baseline CIP output인 `base_query_tgt`에 bounded residual bias만 더합니다.
- final post-residual LayerNorm은 추가하지 않습니다. baseline preservation은 M-CIP v2의 핵심 설계 조건입니다.
- motion ref propagation은 default off입니다.
- 기존 baseline config 파일 `ovtr/config/ovtr_5_frame_train_val.py`는 수정하지 않습니다.

## 구현된 기능

M-CIP v2는 기존 CIP updater의 self-attention/FFN 경로를 먼저 그대로 실행합니다.

```python
base_query_tgt = self._cip_core(track_instances, out_embed_img)
query_tgt = base_query_tgt + inject_gate[:, None] * delta
```

여기서 `delta`는 old memory만 보고 만든 residual이며, 같은 forward에서 새로 갱신된 memory는 injection에 사용하지 않습니다. residual adapter의 마지막 `Linear` weight/bias는 zero init입니다. 따라서 초기 상태에서는 `inject_gate`가 nonzero여도 `delta == 0`이어서 M-CIP v2 output이 baseline CIP output과 수치적으로 거의 동일해야 합니다.

Residual은 다음 조건으로 clamp됩니다.

```python
||delta|| <= mcip_max_residual_ratio * ||base_query_tgt||
```

기본값은 `mcip_max_residual_ratio=0.05`입니다.

Memory update는 learned overwrite gate 대신 analytic reliability update를 사용합니다.

```python
entropy_norm = entropy / math.log(num_classes)  # raw entropy일 때
entropy_norm = entropy_norm.clamp(0.0, 1.0)
current_reliability = scores * (1.0 - entropy_norm)
current_reliability = current_reliability.clamp(0.0, 1.0)
update_gate = valid_mem * mcip_max_memory_update * current_reliability * img_consistency
```

`memory_age == 0`인 new track은 현재 `out_embed_img`/`semantic_obs`로 memory를 초기화하지만, 같은 forward의 residual injection은 0입니다.

Semantic memory는 full expectation 또는 top-k prototype expectation을 지원합니다. 기본은 `mcip_semantic_topk=5`입니다. `mcip_semantic_topk <= 0`이면 기존 full expectation 방식을 사용합니다. 생성된 `semantic_obs`는 `F.normalize(..., dim=-1)`로 정규화됩니다.

Motion ref propagation은 기본 비활성화입니다. `mcip_use_motion_ref=False`일 때 ref point는 current `pred_boxes` 기반입니다. optional motion은 inference에서만 적용되며 xy center velocity만 쓰고, offset은 `mcip_motion_offset_cap=0.02`로 clamp됩니다. learned scale은 `torch.tanh(self.motion_scale)`로 제한됩니다.

## 주요 인자

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--mcip_enable` / `--no_mcip_enable` | config/default 기준 `False` | M-CIP updater 활성화 |
| `--mcip_detach_memory` / `--no_mcip_detach_memory` | `True` | persistent memory를 detach하여 긴 temporal graph/OOM 방지 |
| `--mcip_max_memory_update` | `0.05` | analytic memory update gate 상한 |
| `--mcip_max_residual_ratio` | `0.05` | memory residual delta norm 상한 |
| `--mcip_use_semantic_memory` / `--no_mcip_use_semantic_memory` | `True` | semantic memory를 residual adapter 입력에 사용할지 여부 |
| `--mcip_semantic_topk` | `5` | semantic observation에 사용할 class prototype top-k. `<=0`이면 full expectation |
| `--mcip_use_motion_ref` / `--no_mcip_use_motion_ref` | `False` | inference-only motion-conditioned ref point propagation 사용 여부 |
| `--mcip_motion_momentum` | `0.7` | box velocity EMA momentum |
| `--mcip_motion_scale_init` | `0.0` | motion ref learned scale 초기값 |
| `--mcip_motion_offset_cap` | `0.02` | optional motion xy offset clamp |
| `--debug_mcip` | `False` | M-CIP debug stat 및 trainable M-CIP parameter 이름 출력 |
| `--attention_protection_mode` | `kl` | `kl`, `topk`, `none` |
| `--attention_protection_topk` | `3` | top-k attention protection에서 query별 class top-k |
| `--attention_protection_conf_thresh` | `0.25` | top-k mask 적용 confidence threshold |

## Config 선택

Baseline:

```bash
--config_file ./config/ovtr_5_frame_train_val.py
```

M-CIP v2 memory-only, no motion:

```bash
--config_file ./config/ovtr_5_frame_mcip_train_val.py
```

M-CIP v2 + top-k attention protection:

```bash
--config_file ./config/ovtr_5_frame_mcip_topk_train_val.py
```

## 학습 실행

학습 스크립트는 `CONFIG_FILE`, `OUTPUT`, `BATCH_SIZE`, `EXTRA_ARGS` 환경변수를 받을 수 있습니다.

```bash
cd ovtr
bash tools/ovtr_multi_frame_train.sh
```

예시:

```bash
cd ovtr

CONFIG_FILE=./config/ovtr_5_frame_mcip_train_val.py OUTPUT=./weights_mcip_v2 bash tools/ovtr_multi_frame_train.sh

CONFIG_FILE=./config/ovtr_5_frame_mcip_topk_train_val.py OUTPUT=./weights_mcip_v2_topk bash tools/ovtr_multi_frame_train.sh
```

`debug_mcip`로 새 M-CIP parameter가 실제 trainable인지 확인할 수 있습니다.

```bash
EXTRA_ARGS="--debug_mcip" bash tools/ovtr_multi_frame_train.sh
```

## 평가 실행

평가 스크립트도 `CONFIG_FILE`, `PRETRAIN_MODEL`, `RESULT_PATH`, `VIS_OUTPUT`, `EXTRA_ARGS` 환경변수를 받을 수 있습니다.

```bash
cd ovtr
bash tools/ovtr_ovmot_eval_e15_val.sh
```

Optional inference-only motion ref를 평가할 때만 다음처럼 켭니다.

```bash
EXTRA_ARGS="--mcip_use_motion_ref" bash tools/ovtr_ovmot_eval_e15_val.sh
```

## Checkpoint 호환성

`mcip_enable=True` 상태에서 기존 OVTR checkpoint 또는 M-CIP v1 checkpoint를 `--pretrain` 또는 `--resume`으로 로드할 수 있습니다. 새 v2 parameter가 checkpoint에 없으면 allowed missing M-CIP key로 출력합니다. v1의 `gate_mlp`, `memory_img_proj`, `memory_sem_proj` key가 새 모델에 없으면 allowed unexpected M-CIP key로 출력하며 실행을 막지 않습니다.

허용되는 M-CIP key 범위는 다음 계열입니다.

- v1 compatibility: `track_embed.gate_mlp.*`, `track_embed.memory_img_proj.*`, `track_embed.memory_sem_proj.*`, `track_embed.motion_scale`
- v2: `track_embed.memory_residual_adapter.*`, `track_embed.mcip_img_obs_norm.*`, `track_embed.mcip_img_memory_norm.*`, `track_embed.mcip_sem_memory_norm.*`, `track_embed.memory_inject_logit`, `track_embed.motion_scale`

이 외의 missing/unexpected key는 숨기지 않고 기존처럼 출력합니다.

## Debug acceptance

초기 zero-init 상태에서 다음 기준을 만족해야 합니다.

- `query_tgt_cosine_to_baseline >= 0.999`
- `relative_query_tgt_l2_error <= 1e-4`

항상 다음 기준을 확인합니다.

- `residual_norm_ratio_mean <= mcip_max_residual_ratio`
- `residual_norm_ratio_max <= 0.10`
- `motion_offset_l1_mean == 0` when `mcip_use_motion_ref=False`

## 검증

M-CIP 최소 검증:

```bash
cd /home/pjh/clone_repo/OVTR/ovtr
conda run -n OVTR python tools/mcip_smoke_test.py
```

`tools/mcip_smoke_test.py`는 다음을 확인합니다.

- zero-init M-CIP v2 output이 baseline CIP와 수치적으로 거의 동일한지
- residual adapter final `Linear` weight/bias가 zero init인지
- `inject_gate`가 nonzero여도 `delta=0`이면 `query_tgt`가 baseline과 거의 동일한지
- `memory_age == 0` new track의 same-forward injection이 0인지
- normalized/raw entropy reliability가 `[0, 1]`로 clamp되는지
- `mcip_use_motion_ref=False`일 때 motion offset이 0이고 ref point가 current boxes 기반인지
- semantic top-k/full expectation path의 shape와 finite 여부
- runtime tracker update 이후 M-CIP field 보존

## Ablation 권장 순서

1. Baseline OVTR/CIP: `ovtr_5_frame_train_val.py`
2. M-CIP v2 memory-only, no motion: `ovtr_5_frame_mcip_train_val.py`
3. M-CIP v2 semantic top-k ablation: `--mcip_semantic_topk 0`, `1`, `5`, `10`
4. Optional inference-only motion ref: evaluation에서만 `--mcip_use_motion_ref`
