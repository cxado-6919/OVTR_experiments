# OV-DPTD v1-v7 변경 사항

이 문서는 OVTR에 추가된 Open-Vocabulary Dual-Path Temporal Decoder(OV-DPTD) v1부터 v7까지의 변경점을 정리합니다.

OV-DPTD는 기본 OVTR/QAT/int_msda/TensorRT 경로를 바꾸지 않는 opt-in 기능입니다. `use_ov_dptd=False`이면 기존 OVTR decoder, tracking update, config 기본 동작이 유지됩니다.

## 전체 구조

OV-DPTD는 기존 OVTR decoder를 appearance-adaptive path로 유지하고, track identity를 보존하기 위한 identity-preserving path를 같은 decoder layer parameter로 병렬 실행합니다.

- Appearance-adaptive path: 기존 DeformableTransformerDecoderLayer의 CTI/OFA 흐름을 유지합니다.
- Identity-preserving path: text cross-attention 없이 image cross-attention과 OFA-style image FFN만 수행합니다.
- Query split: detect query가 앞쪽 `[0:num]`, track query가 뒤쪽 `[num:]`에 있다고 가정합니다.
- Track query identity: ID path의 track query는 decoder layer를 거치며 갱신된 query가 아니라 frame decode 진입 시점의 initial track query를 고정해서 사용합니다.
- Classification/category isolation: 항상 AD CTI output 기준입니다.
- Box/feature alignment/CIP 입력: DPTD가 켜진 경우 fused OFA output 기준입니다.
- CTI fusion: v7까지 기본 off이며 `ov_dptd_fuse_cti=True`는 지원하지 않습니다.

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

## v5: Top-k Text Memory Interaction

v5는 identity-preserving path에 track별 top-k text memory interaction을 추가했습니다. 이 기능은 ID path 보조용이며 AD CTI/classification path를 직접 바꾸지 않습니다.

### 사용 조건

`ov_dptd_id_path_text="topk_memory"`는 v5에서 다음 조합에서만 허용됩니다.

- `use_ov_dptd=True`
- `use_dptd_semantic_memory=True`
- `use_dptd_semantic_gate=True`
- `ov_dptd_fusion="semantic_gate"`

`linear_sum + topk_memory`는 RuntimeError로 막습니다. `linear_sum`에서는 ID projection이 zero-init이므로 text adapter gradient가 dead branch가 될 수 있기 때문입니다.

### 새 long-term field

기존 class top-k field와 별도로 adapter용 text top-k field를 저장합니다.

| Field | Shape | 설명 |
| --- | --- | --- |
| `dptd_topk_text_embeddings` | `[num_tracks, dptd_id_text_topk, text_feature_dim]` | decoder CTI가 사용하는 selected text feature 기준 top-k embedding |
| `dptd_topk_text_scores` | `[num_tracks, dptd_id_text_topk]` | normalized probability `p` 기준 top-k score |

`dptd_memory_store_topk`는 기존 `dptd_topk_class_indices/scores` 전용입니다. v5 text adapter field shape는 `dptd_id_text_topk`만 따르므로 두 값이 달라도 shape mismatch가 나지 않아야 합니다.

Text feature dim은 TransformerDecoder 생성 시점에 `dptd_id_text_feature_dim`으로 고정됩니다. runtime `dptd_topk_text_embeddings.shape[-1]`이 adapter k/v projection input dim과 다르면 RuntimeError를 냅니다. LazyLinear는 사용하지 않습니다.

### Adapter attention

ID text adapter는 각 query가 자기 top-k text token만 attend합니다. selected class 전체 attention이나 query 간 attention은 없습니다.

```python
q:   [bs * nq, 1, hidden]
k/v: [bs * nq, topk, hidden]
key_padding_mask = dptd_topk_text_scores <= dptd_id_text_score_eps
```

모든 top-k token이 invalid인 query는 MultiheadAttention 계산에 넣지 않고 residual 0으로 skip합니다. detect query, no-memory/new track, `memory_valid=False` track에도 residual을 적용하지 않습니다.

Residual은 ID OFA에만 더합니다.

```python
id_ofa_enhanced = id_ofa + id_text_residual
```

이후 기존 semantic gate OFA fusion이 `id_ofa_enhanced`를 사용합니다. CTI output, classification logits, category isolation에는 text residual을 직접 섞지 않습니다.

### 초기화와 compatibility

q/k/v projection은 일반 Linear 초기화이고, `dptd_id_text_out_zero_init=True`에서는 out projection을 zero-init합니다. 따라서 `topk_memory`를 켜도 initial forward는 v4 semantic_gate와 allclose여야 합니다.

Debug stat:

- `dptd_id_text_applied_count`
- `dptd_id_text_skipped_no_memory_count`
- `dptd_id_text_skipped_no_topk_count`
- `dptd_id_text_residual_norm_mean`
- `dptd_id_text_residual_norm_max`

## v6: Auxiliary Losses

v6는 v1-v5 decoder 구조를 새로 바꾸는 단계가 아니라, 이미 구현된 DPTD branch를 학습시키기 위한 optional auxiliary loss를 추가한 단계입니다. 기본값은 `use_dptd_losses=False`이고 모든 loss weight는 `0.0`입니다. 이 상태에서는 v5와 output key, loss key, tensor shape, training/eval behavior가 같아야 합니다.

Loss tensor는 training 중 `use_dptd_losses=True`일 때만 보존합니다. eval/inference에서는 loss용 tensor를 만들지 않으며, long-term `Instances` state에도 저장하지 않습니다.

### Loss tensor flow

DPTD decoder는 state 저장용 sampling offset과 loss용 sampling offset을 분리합니다.

- `dptd_info["sampling_offsets"]`: 다음 frame state 저장용 detached final AD sampling offsets
- `dptd_info["loss_tensors"]["ad_sampling_offsets_final"]`: training loss용 non-detached AD sampling offsets

`loss_tensors`에는 final decoder layer 기준으로 다음 tensor가 들어갑니다.

| Key | Shape | Gradient | 설명 |
| --- | --- | --- | --- |
| `ada_ofa_final` | `[bs, nq, hidden_dim]` | target에서 detach 사용 | AD OFA stop-gradient target |
| `id_ofa_final` | `[bs, nq, hidden_dim]` | yes | top-k text residual 적용 후 ID OFA |
| `fused_ofa_final` | `[bs, nq, hidden_dim]` | yes | bbox/align/CIP 흐름의 fused OFA |
| `ad_sampling_offsets_final` | `[bs, nq, nheads, nlevels, npoints, 2]` | yes | offset consistency loss 입력 |
| `historical_sampling_offsets` | shape-compatible | detached target | 이전 frame offset target |
| `semantic_gate` | `[bs, nq]` | detached weight | reliability weight |
| `semantic_gate_memory_valid` | `[bs, nq]` | detached mask | valid memory mask |
| `num_det` | scalar/int | n/a | detect/track query split |

`_post_process_single_image`는 memory update 전에 previous semantic/visual memory snapshot을 detach target으로 만들고, criterion 계산 뒤 다음 frame으로 넘어가기 전에 `_dptd_prev_*_loss`, `_dptd_current_*_loss`, `_dptd_original_query_idx_loss` 같은 loss temp field를 제거합니다.

### Loss keys

`use_dptd_losses=True`이면 weight가 0이어도 frame별 weight_dict와 criterion output에 다음 key가 항상 포함됩니다. 실제 scaling은 engine의 기존 `loss_dict[key] * weight_dict[key]` 경로를 따르며 criterion 내부에서는 weight를 곱하지 않습니다.

- `frame_{i}_loss_dptd_ofa_consistency`
- `frame_{i}_loss_dptd_semantic_memory`
- `frame_{i}_loss_dptd_visual_memory`
- `frame_{i}_loss_dptd_offset_consistency`
- `frame_{i}_loss_dptd_same_category_contrast`

### Loss별 valid/skip 조건

- OFA consistency: track query, existing/memory-valid track만 사용합니다. `normalize(id_ofa_final)`을 `normalize(ada_ofa_final.detach())`에 맞춥니다. v1-v5는 AD/ID decoder layer parameter를 공유하므로 ID path loss gradient가 shared decoder parameter에도 흐를 수 있으며, v6 기본 동작으로 허용합니다.
- Semantic memory consistency: previous semantic memory valid, score/entropy/reliability 조건을 통과한 existing track만 사용합니다. full class probability vector가 아니라 semantic prototype vector끼리 비교합니다.
- Visual memory consistency: current visual memory candidate와 detached previous visual memory를 비교합니다. 기본 visual source는 v3와 동일하게 `pred_embed`입니다.
- Offset consistency: current final AD sampling offsets와 historical offsets shape가 정확히 맞을 때만 L1 loss를 계산합니다. mismatch는 crash 없이 zero/skip하고 debug count를 올립니다.
- Same-category contrast: category 비교는 local selected-class index가 아니라 global category id 기준입니다. positive는 같은 obj의 previous visual memory, negative는 같은 global category이면서 obj가 다른 visual feature/memory입니다. negative 수가 부족하면 zero/skip합니다.

Reliability는 detached semantic gate가 있으면 이를 우선 사용하고, 없으면 score/entropy/previous-memory-valid mask로 계산합니다. detect query는 항상 invalid입니다.

### 권장 초기 weight

모든 weight 기본값은 0입니다. 실험을 시작할 때는 작은 값부터 켜는 것을 권장합니다.

| Loss | 권장 시작값 | 메모 |
| --- | --- | --- |
| `dptd_loss_ofa_consistency_weight` | `0.02` 또는 `0.05` | ID path 안정화용 |
| `dptd_loss_semantic_memory_weight` | `0.01` 또는 `0.02` | semantic proto drift 완화 |
| `dptd_loss_visual_memory_weight` | `0.01` 또는 `0.02` | visual memory 안정화 |
| `dptd_loss_offset_consistency_weight` | `1e-4` 이하 | LocA 영향이 커질 수 있어 작게 시작 |
| `dptd_loss_same_category_contrast_weight` | `0.0`부터 | conservative ablation 권장 |

### v6 Debug stat

`use_dptd_losses=True` and `dptd_loss_store_debug=True`일 때 다음 stat을 criterion/model debug dict에 남깁니다.

- `dptd_loss_valid_track_count`
- `dptd_loss_ofa_valid_count`
- `dptd_loss_semantic_valid_count`
- `dptd_loss_visual_valid_count`
- `dptd_loss_offset_valid_count`
- `dptd_loss_contrast_valid_count`
- `dptd_loss_reliability_mean`
- `dptd_loss_offset_shape_skip_count`
- `dptd_loss_no_positive_skip_count`
- `dptd_loss_no_negative_skip_count`

## v7: Semantic Historical Offset Residual

v7은 ID path가 재사용하는 historical sampling offsets를 대체하지 않고, valid track query에만 작은 residual correction을 더하는 optional module입니다. 기본값은 `use_dptd_semantic_offset_residual=False`이며, off 상태에서는 v6 checkpoint compatibility와 trainable parameter count를 유지하기 위해 `ov_dptd_offset_*` parameter를 생성하지 않습니다.

### 적용 위치

- AD path는 기존처럼 current sampling offsets를 예측합니다.
- ID path 호출 직전에만 `adjusted_historical_offsets = historical_offsets + residual`을 만듭니다.
- detect query `[0:num_det]`, memory 없는 new track, memory invalid track은 residual이 0입니다.
- `dptd_info["sampling_offsets"]`는 계속 final AD path sampling offsets입니다.
- v6 `loss_tensors["historical_sampling_offsets"]`는 residual 적용 전 원본 historical offsets를 target으로 유지합니다.

### Residual module

`use_dptd_semantic_offset_residual=True`일 때만 per-layer module을 생성합니다. 모든 이름은 checkpoint expected-missing 분류를 위해 `ov_dptd_` prefix를 사용합니다.

- `ov_dptd_offset_semantic_proj`: semantic proto projection
- `ov_dptd_offset_visual_proj`: visual memory projection
- `ov_dptd_offset_box_proj`: current reference와 previous box delta projection
- `ov_dptd_offset_age_proj`: normalized memory age projection
- `ov_dptd_offset_residual_out`: offset residual output

semantic/visual/box/age projection은 Xavier init이고, residual output은 weight/bias zero-init입니다. 따라서 기능을 켠 직후 initial forward는 v6와 allclose여야 합니다.

### Memory dim

raw `load_embeddings(...)`의 text embedding은 보통 `[num_classes, clip_dim]`이므로 v7 memory dim은 `text_embeddings.shape[-1]`입니다. build 단계에서 기본값 `512`는 auto value로 보고 loaded embedding dim으로 정규화합니다. explicit non-default value가 loaded text/image embedding dim과 다르면 RuntimeError입니다.

### Debug stat

Residual debug key는 residual이 enabled이거나 `dptd_offset_residual_debug=True`일 때만 merge됩니다.

- `dptd_offset_residual_applied_count`
- `dptd_offset_residual_skipped_count`
- `dptd_offset_residual_norm_mean`
- `dptd_offset_residual_norm_max`
- `dptd_offset_residual_scale`
- `dptd_offset_residual_clamp`
- `dptd_offset_residual_zero_init`

## 주요 Config 기본값

| Config | 기본값 | 설명 |
| --- | --- | --- |
| `use_ov_dptd` | `False` | OV-DPTD 전체 활성화 |
| `ov_dptd_use_historical_offsets` | `True` | track query에 historical sampling offset 사용 |
| `ov_dptd_fusion` | `"linear_sum"` | `"linear_sum"` 또는 `"semantic_gate"` |
| `ov_dptd_id_path_text` | `"none"` | `"none"` 또는 v5 `"topk_memory"` |
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
| `dptd_id_text_topk` | `5` | v5 text adapter용 top-k text token 수 |
| `dptd_id_text_num_heads` | `4` | v5 text adapter head 수 |
| `dptd_id_text_dropout` | `0.0` | v5 text adapter attention dropout |
| `dptd_id_text_out_zero_init` | `True` | initial forward v4 호환을 위한 out projection zero-init |
| `dptd_id_text_score_eps` | `1e-8` | top-k text token validity mask threshold |
| `dptd_id_text_debug` | `False` | v5 text adapter debug stat 저장 |
| `dptd_store_topk_text_embeddings` | `True` | v5 long-term top-k text field 저장 |
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
| `use_dptd_losses` | `False` | v6 auxiliary loss 활성화 |
| `dptd_loss_ofa_consistency_weight` | `0.0` | OFA cross-path consistency weight |
| `dptd_loss_semantic_memory_weight` | `0.0` | semantic memory consistency weight |
| `dptd_loss_visual_memory_weight` | `0.0` | visual memory consistency weight |
| `dptd_loss_offset_consistency_weight` | `0.0` | sampling offset consistency weight |
| `dptd_loss_same_category_contrast_weight` | `0.0` | same-category instance contrast weight |
| `dptd_loss_min_reliability` | `0.5` | v6 loss valid track 최소 reliability |
| `dptd_loss_max_entropy` | `0.75` | v6 semantic/visual loss 최대 entropy |
| `dptd_loss_min_score` | `0.3` | v6 loss 최소 score |
| `dptd_loss_query_scope` | `"track_only"` | v6은 track query loss만 지원 |
| `dptd_loss_apply_aux` | `False` | v6은 final decoder layer loss만 지원 |
| `dptd_loss_ofa_target` | `"ada_stopgrad"` | OFA target mode |
| `dptd_loss_memory_target` | `"previous_stopgrad"` | memory target mode |
| `dptd_loss_offset_target` | `"historical_stopgrad"` | offset target mode |
| `dptd_contrast_temperature` | `0.07` | contrast CE temperature |
| `dptd_contrast_min_negatives` | `1` | contrast 최소 negative 수 |
| `dptd_loss_store_debug` | `False` | v6 loss debug stat 저장 |
| `use_dptd_semantic_offset_residual` | `False` | v7 historical offset residual 활성화 |
| `dptd_offset_residual_scale` | `0.05` | residual tanh output scale |
| `dptd_offset_residual_clamp` | `0.1` | residual absolute clamp |
| `dptd_offset_residual_hidden_dim` | `256` | residual MLP hidden dim |
| `dptd_offset_residual_memory_dim` | `512` | default auto memory dim; build에서 embedding dim으로 정규화 |
| `dptd_offset_residual_use_semantic_proto` | `True` | semantic proto feature 사용 |
| `dptd_offset_residual_use_visual_memory` | `True` | visual memory feature 사용 |
| `dptd_offset_residual_use_box_delta` | `True` | reference/previous box delta feature 사용 |
| `dptd_offset_residual_use_memory_age` | `True` | memory age feature 사용 |
| `dptd_offset_residual_detach_memory` | `True` | memory input detach |
| `dptd_offset_residual_zero_init` | `True` | residual output zero-init |
| `dptd_offset_residual_debug` | `False` | residual debug stat 저장 |

## Guard 및 호환성

다음 조합은 v6 범위에서 명확히 막습니다.

- `use_ov_dptd=True` and `use_checkpoint_track=True`
- `use_ov_dptd=True` and `use_transformer_ckpt=True`
- `use_ov_dptd=True` and training `batch_size > 1`
- `use_ov_dptd=True` and `quant_deploy="int_msda"`
- `ov_dptd_fuse_cti=True`
- `ov_dptd_id_path_text`가 `"none"` 또는 `"topk_memory"`가 아닌 경우
- `ov_dptd_id_path_text="topk_memory"` and `ov_dptd_fusion!="semantic_gate"`
- `ov_dptd_id_path_text="topk_memory"` and `use_dptd_semantic_memory=False`
- `use_dptd_update_suppression=True` and `use_ov_dptd=False`
- training mode and `use_dptd_update_suppression=True`
- `use_dptd_semantic_memory=True` and `use_ov_dptd=False`
- `use_dptd_semantic_gate=True` and `use_dptd_semantic_memory=False`
- `ov_dptd_fusion="semantic_gate"` and `use_dptd_semantic_gate=False`
- `use_dptd_semantic_update_suppression=True` and `use_dptd_update_suppression=False`
- `use_dptd_losses=True` and `use_ov_dptd=False`
- semantic/visual/contrast memory loss weight > 0 and `use_dptd_semantic_memory=False`
- `dptd_loss_apply_aux=True`
- `dptd_loss_query_scope`가 `"track_only"`가 아닌 경우
- `dptd_loss_ofa_target`, `dptd_loss_memory_target`, `dptd_loss_offset_target`가 v6 지원값이 아닌 경우
- `use_dptd_semantic_offset_residual=True` without `use_ov_dptd`, semantic memory, semantic gate, `semantic_gate` fusion, or historical offsets
- `use_dptd_semantic_offset_residual=True` and `query_dim != 4`
- invalid v7 residual scale/clamp/hidden dim/memory dim, or `dptd_offset_residual_zero_init=False`

## Debug/Training Log

학습 loop는 다음 DPTD stat을 수집합니다. v6 loss debug는 위 v6 섹션의 `dptd_loss_*` stat도 함께 사용합니다.

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
