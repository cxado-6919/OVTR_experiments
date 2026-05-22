# OV-DPTD v1-v4 변경 사항

이 문서는 OVTR에 추가된 Open-Vocabulary Dual-Path Temporal Decoder(OV-DPTD) v1부터 v4까지의 변경점을 정리합니다.

OV-DPTD는 기본 OVTR/M-CIP/QAT/int_msda/TensorRT 경로를 바꾸지 않는 opt-in 기능입니다. `use_ov_dptd=False`이면 기존 OVTR decoder, tracking update, config 기본 동작이 유지됩니다.

## 전체 구조

OV-DPTD는 기존 OVTR decoder를 appearance-adaptive path로 유지하고, track identity를 보존하기 위한 identity-preserving path를 같은 decoder layer parameter로 병렬 실행합니다.

- Appearance-adaptive path: 기존 DeformableTransformerDecoderLayer의 CTI/OFA 흐름을 유지합니다.
- Identity-preserving path: text cross-attention 없이 image cross-attention과 OFA-style image FFN만 수행합니다.
- Query split: detect query가 앞쪽 `[0:num]`, track query가 뒤쪽 `[num:]`에 있다고 가정합니다.
- Track query identity: ID path의 track query는 decoder layer를 거치며 갱신된 query가 아니라 frame decode 진입 시점의 initial track query를 고정해서 사용합니다.
- Classification/category isolation: 항상 AD CTI output 기준입니다.
- Box/feature alignment/CIP 입력: DPTD가 켜진 경우 fused OFA output 기준입니다.
- CTI fusion: v4까지 기본 off이며 `ov_dptd_fuse_cti=True`는 지원하지 않습니다.

## v1: Dual-Path Temporal Decoder

v1은 decoder 안에 AD path와 ID path를 병렬로 넣고, OFA feature만 fusion하는 구조를 추가했습니다.

### MSDeformAttn offset override

`MultiScaleDeformableAttention.forward`에 optional 인자가 추가되었습니다.

- `override_sampling_offsets=None`
- `return_sampling_offsets=False`

기본값에서는 기존 반환값과 동일합니다. DPTD가 켜진 경우 AD path에서 sampling offset을 반환받아 다음 frame track query의 historical offset으로 재사용할 수 있습니다. `quant_deploy="int_msda"`와 DPTD 조합은 지원하지 않으며 RuntimeError로 막습니다.

### Decoder 동작

- 기존 decoder forward는 AD path로 유지합니다.
- `forward_identity_path(...)`는 self-attention, image cross-attention, image FFN만 수행합니다.
- ID path는 current-frame text cross-attention과 CTI FFN을 수행하지 않습니다.
- historical offset은 track query `[num:]`에만 적용합니다.
- historical offset이 없거나 shape이 맞지 않으면 current predicted offset으로 fallback합니다.

### OFA fusion

기본 fusion은 `ov_dptd_fusion="linear_sum"`입니다.

```python
fused_ofa = out_proj(ada_proj(ada_ofa) + id_proj(id_ofa))
fused_cti = ada_cti
```

CTI는 기본적으로 fusion하지 않습니다. 따라서 classification, `pre_class_embed`, category isolation은 기존 AD CTI 기준이고, bbox/feature alignment/CIP 입력은 fused OFA 기준입니다.

## v2: Confidence-Guided Update Suppression

v2는 inference-only update suppression을 추가했습니다.

- `use_dptd_update_suppression=True`일 때만 활성화됩니다.
- `use_ov_dptd=False`와 함께 쓰면 RuntimeError입니다.
- training forward에서 켜면 RuntimeError입니다.
- 새 track 생성에는 suppression을 적용하지 않습니다.

복원은 tensor index가 아니라 `obj_idxes` 기반 track id 매칭으로 수행합니다. `track_base.update(...)` 이후 row가 바뀔 수 있기 때문입니다.

기본 복원 대상 field:

- `query_tgt`
- `query_pos`
- `ref_pts`
- `dptd_sampling_offsets`
- `output_embedding_img`
- `output_embedding_txt`

Debug stat:

- `dptd_update_suppressed_count`
- `dptd_update_suppressed_ids`
- `dptd_update_suppression_restore_success_count`
- `dptd_update_suppression_restore_skip_count`

중복된 valid `obj_idxes`는 silent first-match로 처리하지 않고 RuntimeError로 막습니다.

## v3: Track-Wise Semantic/Visual Memory

v3는 track별 semantic/visual memory state를 추가했습니다. full class probability vector는 저장하지 않습니다. OVTR training에서는 frame마다 selected category set이 바뀔 수 있으므로, 장기 memory는 compact feature만 저장합니다.

### Long-term Instances field

`use_dptd_semantic_memory=True`일 때 다음 field가 `Instances`에 붙습니다.

| Field | Shape | 설명 |
| --- | --- | --- |
| `dptd_semantic_proto` | `[num_tracks, clip_dim]` | selected class score와 raw CLIP text embedding의 weighted sum |
| `dptd_visual_memory` | `[num_tracks, clip_dim]` | alignment feature 또는 허용된 fallback projection feature |
| `dptd_semantic_conf` | `[num_tracks]` | current classification confidence |
| `dptd_semantic_entropy` | `[num_tracks]` | normalized entropy |
| `dptd_memory_age` | `[num_tracks]` | 마지막 reliable memory update 이후 지난 frame 수 |
| `dptd_topk_class_indices` | `[num_tracks, topk]` | global category id |
| `dptd_topk_class_scores` | `[num_tracks, topk]` | normalized probability `p` 기준 top-k score |

모든 memory state는 `torch.no_grad()` 안에서 계산하고 detach된 tensor로 저장합니다. 다음 frame으로 autograd graph가 이어지지 않게 하기 위한 제한입니다.

### Semantic proto 계산

`num_cls = len(select_id)` 기준으로 background/no-object를 제외한 selected class만 사용합니다.

```python
logits = pred_logits[..., :num_cls]
score = sigmoid(logits)
p = score / score.sum(-1, keepdim=True).clamp_min(1e-6)
semantic_proto = normalize(p @ raw_clip_text_embeddings, eps=1e-6)
```

Entropy는 `num_cls == 1`일 때 denominator를 `1.0`으로 두고 NaN을 방지합니다.

```python
entropy_den = math.log(num_cls) if num_cls > 1 else 1.0
entropy = (-(p * p.clamp_min(1e-6).log()).sum(-1) / entropy_den).clamp(0.0, 1.0)
```

`dptd_text_memory_embeddings`는 `_post_process_single_image`에서 semantic candidate 계산에만 쓰는 ephemeral key입니다. 계산 직후 `frame_res`에서 제거되며, eval json이나 long-term `Instances` field에 저장하지 않습니다.

### Visual memory

기본 visual source는 `frame_res["pred_embed"]`입니다. `dptd_memory_use_alignment_feature=True`인데 `pred_embed`가 없으면 RuntimeError를 냅니다. fallback projection은 `dptd_memory_allow_untrained_visual_projection=True`일 때만 허용됩니다.

### Memory update

New track 판단에는 `dptd_memory_age == 0`을 쓰지 않습니다.

- new track: old snapshot에 해당 `obj_idxes`가 없거나 old semantic/visual memory norm이 0인 경우
- existing reliable track: EMA update 후 age를 0으로 리셋
- unreliable track: content/top-k/conf/entropy를 유지하고 age만 증가
- v2에서 suppressed된 track: current candidate가 있어도 memory content를 갱신하지 않고 age만 증가

Row alignment를 위해 `_dptd_current_*` temp field를 잠깐 붙인 뒤, memory update 직후 제거합니다. `track_embed`에는 temp field가 넘어가지 않습니다.

## v4: Semantic-Reliability Guided Fusion

v4는 parameter-free heuristic semantic gate를 추가했습니다. Trainable gate MLP는 구현하지 않았습니다.

### Gate 조건

`use_dptd_semantic_gate=True`는 다음 조건을 요구합니다.

- `use_ov_dptd=True`
- `use_dptd_semantic_memory=True`
- `dptd_gate_mode="heuristic"`

### Raw gate와 fusion gate

v4 보완에서 update-control/debug용 raw gate와 fusion용 clamped gate를 분리했습니다.

```python
raw_gate = score_conf * entropy_conf * semantic_conf * offset_conf
fusion_gate = raw_gate.clamp(min=dptd_gate_min_appearance, max=1.0)
```

- `raw_gate`: semantic-aware update suppression에 사용합니다.
- `fusion_gate`: OFA semantic gate fusion에 사용합니다.
- detect query, no-memory track, new track은 gate를 `1.0`으로 둡니다.
- `semantic_gate_memory_valid` mask는 valid existing track 기준 debug 통계와 low-count 계산에 사용합니다.

Decoder 내부 visual consistency는 v3 visual memory가 CLIP/alignment space이고 OFA는 hidden space이므로 v4 fusion gate에는 포함하지 않습니다. Box consistency도 fused box에 의존하면 circular dependency가 생기므로 v4에서는 deferred로 두었습니다.

Debug에는 다음이 기록됩니다.

- `dptd_gate_box_conf_deferred=True`
- `dptd_gate_box_conf_included=False`

### Semantic-gate OFA fusion

`ov_dptd_fusion="semantic_gate"`일 때만 다음 fusion을 사용합니다.

```python
id_delta = ov_dptd_ofa_id_proj[layer_id](id_ofa)
fused_ofa = ada_ofa + ov_dptd_gate_alpha * (1 - fusion_gate) * id_delta
```

Detect query는 gate가 1이므로 ID correction이 들어가지 않습니다. CTI에는 ID feature를 직접 섞지 않습니다.

### Dead-branch fix

초기 구현에서는 `ov_dptd_gate_alpha=0`이고 `ov_dptd_ofa_id_proj=0`이라 first backward에서 alpha/id projection gradient가 모두 0이 되는 dead branch 문제가 있었습니다.

현재는 `semantic_gate`에서만 ID projection을 small-random init합니다.

- `ov_dptd_gate_alpha`: 계속 0.0
- `ov_dptd_ofa_id_proj.weight`: `Normal(0, ov_dptd_semantic_gate_id_proj_init_std)`
- `ov_dptd_ofa_id_proj.bias`: 0
- `linear_sum`: 기존 zero-init 유지

alpha가 0이므로 initial forward는 baseline과 동일하게 시작하지만, first backward에서 alpha로 gradient가 흐를 수 있습니다.

기존 dead v4 checkpoint는 load 직후 감지합니다. 기본은 warning만 출력하고 checkpoint 값을 보존합니다. `ov_dptd_reinit_dead_semantic_gate_id_proj=True`일 때만 dead ID projection을 small-random으로 재초기화합니다.

## 주요 Config 기본값

| Config | 기본값 | 설명 |
| --- | --- | --- |
| `use_ov_dptd` | `False` | OV-DPTD 전체 활성화 |
| `ov_dptd_use_historical_offsets` | `True` | track query에 historical sampling offset 사용 |
| `ov_dptd_fusion` | `"linear_sum"` | `"linear_sum"` 또는 `"semantic_gate"` |
| `ov_dptd_id_path_text` | `"none"` | v1-v4에서는 ID path text attention 미지원 |
| `ov_dptd_fuse_cti` | `False` | v4까지 CTI fusion 금지 |
| `ov_dptd_store_debug` | `False` | DPTD debug stat 저장 |
| `use_dptd_update_suppression` | `False` | v2 confidence suppression |
| `dptd_update_suppression_thresh` | `0.4` | score 기반 suppression threshold |
| `use_dptd_semantic_memory` | `False` | v3 semantic/visual memory |
| `dptd_memory_ema` | `0.8` | memory EMA 계수 |
| `dptd_memory_min_score` | `0.4` | reliable memory update 최소 score |
| `dptd_memory_max_entropy` | `0.75` | reliable memory update 최대 entropy |
| `dptd_memory_use_alignment_feature` | `True` | visual memory에 `pred_embed` 사용 |
| `dptd_memory_allow_untrained_visual_projection` | `False` | fallback projection 허용 |
| `dptd_memory_store_topk` | `5` | 저장할 top-k class 수 |
| `use_dptd_semantic_gate` | `False` | v4 heuristic semantic gate |
| `dptd_gate_mode` | `"heuristic"` | v4는 heuristic만 지원 |
| `dptd_gate_min_score` | `0.3` | score confidence lower bound |
| `dptd_gate_max_entropy` | `0.8` | entropy confidence upper bound |
| `dptd_gate_semantic_cos_tau` | `0.25` | semantic cosine threshold |
| `dptd_gate_offset_tau` | `0.2` | offset consistency temperature |
| `dptd_gate_temperature` | `10.0` | sigmoid consistency temperature |
| `dptd_gate_min_appearance` | `0.1` | fusion gate clamp lower bound |
| `use_dptd_semantic_update_suppression` | `False` | raw gate 기반 update suppression 확장 |
| `dptd_semantic_update_suppression_thresh` | `0.3` | semantic update suppression threshold |
| `ov_dptd_semantic_gate_id_proj_init` | `"small_random"` | semantic_gate ID projection init |
| `ov_dptd_semantic_gate_id_proj_init_std` | `1e-3` | small-random init std |
| `ov_dptd_reinit_dead_semantic_gate_id_proj` | `False` | dead checkpoint ID projection 재초기화 |

## Guard 및 호환성

다음 조합은 v4 범위에서 명확히 막습니다.

- `use_ov_dptd=True` and `use_checkpoint_track=True`
- `use_ov_dptd=True` and `use_transformer_ckpt=True`
- `use_ov_dptd=True` and training `batch_size > 1`
- `use_ov_dptd=True` and `quant_deploy="int_msda"`
- `ov_dptd_fuse_cti=True`
- `ov_dptd_id_path_text!="none"`
- `use_dptd_update_suppression=True` and `use_ov_dptd=False`
- training mode and `use_dptd_update_suppression=True`
- `use_dptd_semantic_memory=True` and `use_ov_dptd=False`
- `use_dptd_semantic_gate=True` and `use_dptd_semantic_memory=False`
- `ov_dptd_fusion="semantic_gate"` and `use_dptd_semantic_gate=False`
- `use_dptd_semantic_update_suppression=True` and `use_dptd_update_suppression=False`

## Debug/Training Log

학습 loop는 다음 DPTD stat을 수집합니다.

- `ov_dptd_gate_alpha`
- `ov_dptd_gate_alpha_grad_norm`
- `ov_dptd_id_proj_weight_norm`
- `ov_dptd_id_proj_grad_norm`
- `dptd_gate_raw_valid_mean`
- `dptd_gate_valid_mean`

Gate debug stat은 valid existing track이 0개여도 NaN을 만들지 않고 0.0 또는 0 count로 기록합니다.

## Smoke Test

DPTD 관련 smoke/unit check는 다음 파일에 모여 있습니다.

```bash
python ovtr/tools/dptd_smoke_test.py
```

기본 OVTR smoke는 별도로 유지됩니다.

```bash
cd ovtr
python tools/smoke_test.py
```
