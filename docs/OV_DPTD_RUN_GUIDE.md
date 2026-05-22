# OV-DPTD 실행 가이드

이 문서는 OV-DPTD v1-v7를 학습/평가할 때 사용하는 config 조합, shell script, 주요 인자를 정리합니다.

OV-DPTD 인자는 현재 config field로 관리됩니다. `tools/*.sh`의 `EXTRA_ARGS`는 기존 CLI 인자를 뒤에 붙이는 용도이며, DPTD field는 config 파일에 명시하는 방식을 권장합니다.

## 기본 원칙

- baseline 확인은 `use_ov_dptd=False` 그대로 실행합니다.
- DPTD training은 `BATCH_SIZE=1`로 실행합니다. v4 구현은 DPTD training batch size > 1을 RuntimeError로 막습니다.
- DPTD는 `use_checkpoint_track=True`, `use_transformer_ckpt=True`, `quant_deploy="int_msda"`와 같이 쓰지 않습니다.
- semantic update suppression은 inference-only입니다. training config에서는 `use_dptd_update_suppression=False`, `use_dptd_semantic_update_suppression=False`를 유지합니다.
- CTI fusion은 v4에서도 지원하지 않습니다. `ov_dptd_fuse_cti=True`는 NotImplementedError입니다.

## Config 준비

기존 baseline config를 복사해서 실험별 config를 만드는 방식을 권장합니다.

```bash
cd ovtr
cp config/ovtr_5_frame_train_val.py config/ovtr_5_frame_dptd_v4_train_val.py
cp config/ovtr_5_frame_test.py config/ovtr_5_frame_dptd_v4_test.py
```

### v1 linear_sum training config

```python
use_ov_dptd = True
ov_dptd_fusion = "linear_sum"
ov_dptd_fuse_cti = False
ov_dptd_id_path_text = "none"
ov_dptd_store_debug = True

use_dptd_update_suppression = False
use_dptd_semantic_memory = False
use_dptd_semantic_gate = False
use_dptd_semantic_update_suppression = False

use_checkpoint_track = False
use_transformer_ckpt = False
```

### v3 memory training config

```python
use_ov_dptd = True
ov_dptd_fusion = "linear_sum"
ov_dptd_fuse_cti = False
ov_dptd_store_debug = True

use_dptd_semantic_memory = True
dptd_memory_debug = True
dptd_memory_use_alignment_feature = True
dptd_memory_allow_untrained_visual_projection = False

use_dptd_update_suppression = False
use_dptd_semantic_gate = False
use_dptd_semantic_update_suppression = False

use_checkpoint_track = False
use_transformer_ckpt = False
```

### v4 semantic_gate training config

```python
use_ov_dptd = True
ov_dptd_fusion = "semantic_gate"
ov_dptd_fuse_cti = False
ov_dptd_id_path_text = "none"
ov_dptd_store_debug = True

use_dptd_semantic_memory = True
dptd_memory_debug = True
dptd_memory_use_alignment_feature = True
dptd_memory_allow_untrained_visual_projection = False

use_dptd_semantic_gate = True
dptd_gate_mode = "heuristic"
dptd_gate_debug = True
dptd_gate_min_appearance = 0.1

ov_dptd_semantic_gate_id_proj_init = "small_random"
ov_dptd_semantic_gate_id_proj_init_std = 1e-3
ov_dptd_reinit_dead_semantic_gate_id_proj = False

# inference-only 기능이므로 training에서는 끕니다.
use_dptd_update_suppression = False
use_dptd_semantic_update_suppression = False

use_checkpoint_track = False
use_transformer_ckpt = False
```

`semantic_gate`에서는 `ov_dptd_gate_alpha`가 0으로 시작하고 ID projection은 small-random으로 시작합니다. 초기 forward는 baseline과 같지만 첫 backward에서 alpha gradient가 흐르도록 하기 위한 설정입니다.

### v5 topk_memory training config

v5 top-k text memory interaction은 semantic gate 전용입니다. training config 예시는 다음과 같습니다.

```python
use_ov_dptd = True
ov_dptd_fusion = "semantic_gate"
ov_dptd_id_path_text = "topk_memory"
ov_dptd_fuse_cti = False
ov_dptd_store_debug = True

use_dptd_semantic_memory = True
dptd_store_topk_text_embeddings = True
dptd_id_text_topk = 5
dptd_id_text_num_heads = 4
dptd_id_text_dropout = 0.0
dptd_id_text_out_zero_init = True
dptd_id_text_score_eps = 1e-8
dptd_id_text_debug = True

use_dptd_semantic_gate = True
dptd_gate_mode = "heuristic"
dptd_gate_debug = True

# inference-only 기능이므로 training에서는 끕니다.
use_dptd_update_suppression = False
use_dptd_semantic_update_suppression = False

use_checkpoint_track = False
use_transformer_ckpt = False
```

`ov_dptd_fusion="linear_sum"`과 `ov_dptd_id_path_text="topk_memory"` 조합은 v5에서 RuntimeError입니다. semantic_gate의 small-random ID projection과 nonzero alpha gradient 흐름을 전제로 adapter gradient를 확인하기 때문입니다.

### v6 auxiliary-loss fine-tuning config

v6 auxiliary losses는 DPTD 구조를 바꾸지 않고 training loss만 추가합니다. 기본값은 모두 off이며, `use_dptd_losses=True`일 때도 weight가 0이면 engine에서 0이 곱해집니다.

```python
use_ov_dptd = True
ov_dptd_fusion = "semantic_gate"
ov_dptd_fuse_cti = False

use_dptd_semantic_memory = True
use_dptd_semantic_gate = True

# v5 top-k text adapter는 선택 사항입니다.
ov_dptd_id_path_text = "topk_memory"
dptd_store_topk_text_embeddings = True

use_dptd_losses = True
dptd_loss_ofa_consistency_weight = 0.02
dptd_loss_semantic_memory_weight = 0.01
dptd_loss_visual_memory_weight = 0.01
dptd_loss_offset_consistency_weight = 1e-4
dptd_loss_same_category_contrast_weight = 0.0

dptd_loss_min_reliability = 0.5
dptd_loss_max_entropy = 0.75
dptd_loss_min_score = 0.3
dptd_loss_query_scope = "track_only"
dptd_loss_apply_aux = False
dptd_loss_ofa_target = "ada_stopgrad"
dptd_loss_memory_target = "previous_stopgrad"
dptd_loss_offset_target = "historical_stopgrad"
dptd_contrast_temperature = 0.07
dptd_contrast_min_negatives = 1
dptd_loss_store_debug = True

# inference-only 기능이므로 training에서는 끕니다.
use_dptd_update_suppression = False
use_dptd_semantic_update_suppression = False

use_checkpoint_track = False
use_transformer_ckpt = False
```

권장 시작점은 OFA consistency `0.02-0.05`, semantic/visual memory `0.01-0.02`, offset consistency `1e-4` 이하, same-category contrast는 `0.0`부터입니다. Offset loss는 LocA에 영향을 줄 수 있으므로 특히 작게 시작하세요.

`use_dptd_losses=True`이면 frame별 다음 loss key가 weight_dict에 항상 들어갑니다. weight가 0이어도 criterion이 같은 key를 반환하므로 logging/engine contract가 안정적입니다.

- `frame_{i}_loss_dptd_ofa_consistency`
- `frame_{i}_loss_dptd_semantic_memory`
- `frame_{i}_loss_dptd_visual_memory`
- `frame_{i}_loss_dptd_offset_consistency`
- `frame_{i}_loss_dptd_same_category_contrast`

주의할 guard:

- `use_dptd_losses=True`는 `use_ov_dptd=True`가 필요합니다.
- semantic/visual/contrast memory loss weight를 0보다 크게 쓰려면 `use_dptd_semantic_memory=True`가 필요합니다.
- `dptd_loss_apply_aux=True`는 v6에서 지원하지 않습니다. final decoder layer loss만 계산합니다.
- `dptd_loss_query_scope="track_only"`, target mode는 `ada_stopgrad`/`previous_stopgrad`/`historical_stopgrad`만 지원합니다.

### v7 semantic offset residual config

v7 residual은 semantic gate 기반 ID path에서만 사용합니다. LocA를 흔들 수 있으므로 scale/clamp를 작게 시작하고, zero-init을 유지합니다.

```python
use_ov_dptd = True
ov_dptd_fusion = "semantic_gate"
ov_dptd_fuse_cti = False
use_dptd_semantic_memory = True
use_dptd_semantic_gate = True
ov_dptd_use_historical_offsets = True

use_dptd_semantic_offset_residual = True
dptd_offset_residual_scale = 0.05
dptd_offset_residual_clamp = 0.1
dptd_offset_residual_hidden_dim = 256
dptd_offset_residual_memory_dim = 512  # build에서 loaded embedding dim으로 auto 정규화
dptd_offset_residual_use_semantic_proto = True
dptd_offset_residual_use_visual_memory = True
dptd_offset_residual_use_box_delta = True
dptd_offset_residual_use_memory_age = True
dptd_offset_residual_detach_memory = True
dptd_offset_residual_zero_init = True
dptd_offset_residual_debug = True

use_checkpoint_track = False
use_transformer_ckpt = False
```

`use_dptd_semantic_offset_residual=False`이면 `ov_dptd_offset_*` parameter가 생성되지 않습니다. 기존 v6 checkpoint를 residual-off config로 로드할 때 missing key나 trainable parameter count가 늘지 않는 것이 의도입니다.

주의할 점:

- v7은 `query_dim=4`만 지원합니다.
- `dptd_info["sampling_offsets"]`는 AD path final offsets이고, residual-applied ID offsets가 아닙니다.
- v6 offset consistency target은 residual 적용 전 original historical offsets입니다.
- raw embedding dim은 `text_embeddings.shape[-1]` 기준입니다. class count인 `shape[0]`를 memory dim으로 쓰면 안 됩니다.

### v4 eval config with suppression

평가에서 semantic-aware update suppression까지 보려면 eval용 config에서만 다음을 켭니다.

```python
use_ov_dptd = True
ov_dptd_fusion = "semantic_gate"
use_dptd_semantic_memory = True
use_dptd_semantic_gate = True

use_dptd_update_suppression = True
dptd_update_suppression_thresh = 0.4

use_dptd_semantic_update_suppression = True
dptd_semantic_update_suppression_thresh = 0.3
```

`use_dptd_semantic_update_suppression=True`는 `use_dptd_update_suppression=True`가 필요합니다. base suppression이 꺼져 있으면 RuntimeError가 납니다.

## Training Script

5-frame 학습 스크립트:

```bash
cd ovtr
CONFIG_FILE=./config/ovtr_5_frame_dptd_v4_train_val.py \
OUTPUT=./weights_ov_dptd_v4 \
BATCH_SIZE=1 \
CUDA_DEVICES=0,1,2,3 \
NPROC_GPU=4 \
bash tools/ovtr_multi_frame_train.sh
```

`tools/ovtr_multi_frame_train.sh`는 두 stage를 연속 실행합니다.

1. Stage 1: `--pretrain "${PRETRAIN_MODEL}"`, 기본 1 epoch
2. Stage 2: `--resume "${OUTPUT}/checkpoint0000.pth"`, 기본 16 epoch

주요 환경변수:

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `CONFIG_FILE` | `./config/ovtr_5_frame_train_val.py` | 학습 config |
| `PRETRAIN_MODEL` | `../model_zoo/ovtr_det_pretrain.pth` | stage 1 pretrain checkpoint |
| `OUTPUT` | `./weights` | checkpoint/log 출력 경로 |
| `BATCH_SIZE` | `4` | DPTD training에서는 `1`로 설정 |
| `CUDA_DEVICES` | `0,1,2,3` | 사용할 GPU |
| `NPROC_GPU` | `4` | torchrun process 수 |
| `MASTER_PORT` | `9982` | distributed port |
| `EXTRA_ARGS` | empty | 추가 CLI 인자 |

DPTD training에서 `BATCH_SIZE=1`을 명시하지 않으면 config가 켜진 상태에서 RuntimeError가 날 수 있습니다.

### Lite training

Lite 학습 스크립트는 stage별 epoch/lr도 환경변수로 조정할 수 있습니다.

```bash
cd ovtr
CONFIG_FILE=./config/ovtr_lite_dptd_v4_train_val.py \
OUTPUT=./weights_ov_dptd_v4_lite \
BATCH_SIZE=1 \
STAGE1_EPOCHS=1 \
STAGE2_EPOCHS=16 \
CUDA_DEVICES=0 \
NPROC_GPU=1 \
bash tools/ovtr_multi_frame_lite_train.sh
```

주요 추가 환경변수:

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `STAGE1_EPOCHS` | `1` | pretrain stage epoch |
| `STAGE2_EPOCHS` | `16` | resume stage epoch. `0`이면 stage 2 skip |
| `STAGE1_LR` | `2e-4` | stage 1 lr |
| `STAGE2_LR` | `4e-5` | stage 2 lr |
| `NUM_WORKERS` | `4` | dataloader worker 수 |

## Fine-Tuning Trainability 확인

v4 semantic gate가 실제로 학습되는지 보려면 log에서 다음 값을 확인합니다.

- `ov_dptd_gate_alpha`
- `ov_dptd_gate_alpha_grad_norm`
- `ov_dptd_id_proj_weight_norm`
- `ov_dptd_id_proj_grad_norm`
- `dptd_gate_raw_valid_mean`
- `dptd_gate_valid_mean`

예시:

```bash
tail -f ./weights_ov_dptd_v4/log.txt | rg "ov_dptd_gate_alpha|ov_dptd_id_proj|dptd_gate"
```

기대 흐름:

- reset 직후 `ov_dptd_gate_alpha`는 0입니다.
- `semantic_gate` + small-random ID projection이면 first backward 후 `ov_dptd_gate_alpha_grad_norm`이 0보다 커질 수 있습니다.
- optimizer step 이후 `ov_dptd_gate_alpha`가 0에서 벗어나면 다음 step부터 ID projection에도 gradient가 흐를 수 있습니다.

`tools/dptd_smoke_test.py`에도 optimizer membership과 alpha update smoke가 포함되어 있습니다.

```bash
python ovtr/tools/dptd_smoke_test.py
```

## Evaluation Script

5-frame val 평가 스크립트는 `CONFIG_FILE`, `PRETRAIN_MODEL`, `RESULT_PATH`, `VIS_OUTPUT`, `EXTRA_ARGS`를 받을 수 있습니다.

```bash
cd ovtr
CONFIG_FILE=./config/ovtr_5_frame_dptd_v4_test.py \
PRETRAIN_MODEL=./weights_ov_dptd_v4/checkpoint0015.pth \
RESULT_PATH=./results/teta_results_ov_dptd_v4_val \
VIS_OUTPUT=./results/vis_output_track_ov_dptd_v4_val \
CUDA_DEVICES=0 \
NPROC_GPU=1 \
bash tools/ovtr_ovmot_eval_e15_val.sh
```

주요 환경변수:

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `CONFIG_FILE` | `./config/ovtr_5_frame_train_val.py` | val 평가 config |
| `PRETRAIN_MODEL` | `../model_zoo/ovtr_5_frame.pth` | 평가 checkpoint |
| `OUTPUT` | `./results` | eval output dir |
| `RESULT_PATH` | `./results/teta_results_5_frame_val` | tracking result path |
| `VIS_OUTPUT` | `./results/vis_output_track_5_frame_val` | visualization output |
| `BATCH_SIZE` | `1` | eval batch size |
| `CUDA_DEVICES` | `0` | 사용할 GPU |
| `NPROC_GPU` | `1` | torchrun process 수 |
| `EXTRA_ARGS` | empty | 추가 CLI 인자 |

5-frame test 스크립트 `tools/ovtr_ovmot_eval_e15_test.sh`는 현재 `--config_file ./config/ovtr_5_frame_test.py`가 하드코딩되어 있습니다. custom DPTD test config를 쓰려면 스크립트를 복사해 config 경로를 바꾸거나, 아래처럼 `eval.py`를 직접 실행합니다.

```bash
cd ovtr
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=9983 --nproc_per_node=1 \
    ./eval.py \
    --config_file ./config/ovtr_5_frame_dptd_v4_test.py \
    --dataset_file lvis_generated_img_seqs \
    --epochs 16 \
    --with_box_refine \
    --two_stage \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --lr_drop 13 \
    --pretrain ./weights_ov_dptd_v4/checkpoint0015.pth \
    --output_dir ./results \
    --num_workers 48 \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 1 \
    --sampler_steps 4 7 14 \
    --sampler_lengths 2 3 4 5 \
    --merger_dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --track_query_iteration CIP \
    --calculate_negative_samples \
    --score_thresh 0.20 0.17 0.17 0.20 0.17 0.20 0.17 \
    --filter_score_thresh 0.20 0.17 0.17 0.20 0.17 0.20 0.17 \
    --ious_thresh 0.5 0.45 0.5 0.4 0.45 0.45 0.45 \
    --miss_tolerance 5 5 5 5 5 5 5 \
    --maximum_quantity 160 \
    --result_path_track ./results/teta_results_ov_dptd_v4_test \
    --vis_output ./results/vis_output_track_ov_dptd_v4_test
```

Lite val/test 스크립트도 config path가 각각 `ovtr_lite_train_val.py`, `ovtr_lite_test.py`로 하드코딩되어 있습니다. Lite DPTD custom config를 쓰려면 직접 `eval.py`를 실행하거나 script copy를 만들어 `--config_file`만 교체합니다.

## Checkpoint Load

기존 OVTR checkpoint를 DPTD config로 로드하면 새 DPTD parameter는 checkpoint에 없을 수 있습니다. 다음 key들은 expected missing으로 분류되어 일반 missing key와 분리 출력됩니다.

- `ov_dptd_*`
- `dptd_visual_memory_proj.*`

기존 dead v4 checkpoint를 semantic_gate로 로드했을 때 `ov_dptd_gate_alpha == 0`이고 ID projection norm이 0이면 warning이 출력됩니다.

기본은 checkpoint 값을 보존합니다.

```python
ov_dptd_reinit_dead_semantic_gate_id_proj = False
```

dead checkpoint의 ID projection만 재초기화하고 싶을 때 eval/training config에 명시합니다.

```python
ov_dptd_reinit_dead_semantic_gate_id_proj = True
ov_dptd_semantic_gate_id_proj_init_std = 1e-3
```

## 빠른 검증

문법/기본 smoke:

```bash
python -m py_compile ovtr/models/ovtr.py ovtr/models/transformer.py ovtr/tools/dptd_smoke_test.py
python ovtr/tools/dptd_smoke_test.py
cd ovtr
python tools/smoke_test.py
```

변경 전 baseline contract를 확인하려면 config에서 `use_ov_dptd=False`를 유지한 채 `tools/smoke_test.py`를 실행합니다.

## Troubleshooting

### `batch_size > 1` RuntimeError

DPTD training은 현재 single-image path만 지원합니다.

```bash
BATCH_SIZE=1 bash tools/ovtr_multi_frame_train.sh
```

### `semantic_gate requires use_dptd_semantic_gate=True`

`ov_dptd_fusion="semantic_gate"`를 쓸 때는 다음 두 option이 같이 필요합니다.

```python
use_dptd_semantic_memory = True
use_dptd_semantic_gate = True
```

### `pred_embed` missing RuntimeError

기본 visual memory source는 alignment feature `pred_embed`입니다. fallback projection을 실험 목적으로 허용하려면 명시적으로 켭니다.

```python
dptd_memory_allow_untrained_visual_projection = True
```

기본 실험에서는 fallback을 켜지 않는 것을 권장합니다.

### `topk_memory`가 RuntimeError를 내는 경우

`topk_memory`는 semantic gate 전용입니다. 다음 네 가지가 모두 필요합니다.

```python
use_ov_dptd = True
ov_dptd_fusion = "semantic_gate"
ov_dptd_id_path_text = "topk_memory"
use_dptd_semantic_memory = True
use_dptd_semantic_gate = True
```

`dptd_topk_text_embeddings.shape[-1]`은 decoder CTI가 쓰는 text feature dim과 같아야 합니다. 이 dim이 다르면 adapter k/v projection input dim과 맞지 않아 RuntimeError가 납니다.

모든 top-k text score가 `dptd_id_text_score_eps` 이하인 query는 attention을 실행하지 않고 residual 0으로 skip됩니다.

### v7 residual parameter가 보이지 않는 경우

정상입니다. `use_dptd_semantic_offset_residual=False`이면 `ov_dptd_offset_*` module은 빈 `ModuleList`라 parameter가 없습니다. residual을 학습하려면 다음 조합이 모두 필요합니다.

```python
use_ov_dptd = True
use_dptd_semantic_memory = True
use_dptd_semantic_gate = True
ov_dptd_fusion = "semantic_gate"
ov_dptd_use_historical_offsets = True
use_dptd_semantic_offset_residual = True
```

gradient smoke에서는 `ov_dptd_gate_alpha`가 0이면 fused output loss에서 residual branch gradient가 0일 수 있습니다. alpha를 nonzero로 두거나 ID-path loss를 사용해 확인하세요.

### DPTD loss가 RuntimeError를 내는 경우

v6 auxiliary loss는 DPTD opt-in path 전용입니다. 다음 조합을 먼저 확인합니다.

```python
use_ov_dptd = True
use_dptd_losses = True
dptd_loss_apply_aux = False
dptd_loss_query_scope = "track_only"
dptd_loss_ofa_target = "ada_stopgrad"
dptd_loss_memory_target = "previous_stopgrad"
dptd_loss_offset_target = "historical_stopgrad"
```

semantic/visual/contrast loss weight를 0보다 크게 켤 때는 memory state가 필요합니다.

```python
use_dptd_semantic_memory = True
```

DPTD loss tensor는 training 중에만 생성됩니다. eval/inference에서 `dptd_loss_tensors`가 없는 것은 정상입니다.

### Alpha가 계속 0인 경우

다음을 확인합니다.

- `ov_dptd_fusion="semantic_gate"`
- `ov_dptd_semantic_gate_id_proj_init="small_random"`
- `ov_dptd_semantic_gate_id_proj_init_std > 0`
- freeze filter에 `ov_dptd_`가 포함되어 있는지
- log에 `ov_dptd_gate_alpha_grad_norm > 0`이 찍히는지

tracking-only freeze list에는 `ov_dptd_`, `dptd_visual_memory_proj` substring이 포함되어 있어 DPTD parameter가 optimizer param group에 들어가도록 처리되어 있습니다.
