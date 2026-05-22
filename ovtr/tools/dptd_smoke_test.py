import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectron2.structures import Instances  # noqa: E402
from models.ovtr import DPTD_CURRENT_MEMORY_FIELDS, OVTR, RuntimeTrackerBase, resolve_ov_dptd_options  # noqa: E402
from models.transformer import DeformableTransformerDecoderLayer, TransformerDecoder  # noqa: E402


def _make_decoder(use_ov_dptd=False, num_layers=2, d_model=256, nheads=4, nlevels=2, npoints=2, num_det=3, **decoder_kwargs):
    layer = DeformableTransformerDecoderLayer(
        d_model=d_model,
        d_ffn=512,
        dropout=0.0,
        activation="relu",
        n_levels=nlevels,
        n_heads=nheads,
        n_points=npoints,
        use_text_cross_attention=True,
        extra_track_attn=True,
    )
    decoder = TransformerDecoder(
        layer,
        num_layers,
        nn.LayerNorm(d_model),
        d_model=d_model,
        query_dim=4,
        num_feature_levels=nlevels,
        text_dim=d_model,
        num_queries=num_det,
        attention_protection=False,
        computed_aux=list(range(num_layers)),
        use_ov_dptd=use_ov_dptd,
        ov_dptd_store_debug=True,
        **decoder_kwargs,
    )
    decoder.bbox_embed = nn.ModuleList([nn.Linear(d_model, 4) for _ in range(num_layers)])
    if use_ov_dptd:
        decoder.reset_ov_dptd_fusion_parameters()
    decoder.eval()
    return decoder


def _inputs(num_queries, batch_size=1, d_model=256, nlevels=2):
    spatial_shapes = torch.tensor([[2, 2], [1, 1]], dtype=torch.long)
    level_start_index = torch.tensor([0, 4], dtype=torch.long)
    src = torch.randn(5, batch_size, d_model)
    src_padding_mask = torch.zeros(batch_size, 5, dtype=torch.bool)
    src_valid_ratios = torch.ones(batch_size, nlevels, 2)
    tgt = torch.randn(num_queries, batch_size, d_model)
    reference_points = torch.rand(num_queries, batch_size, 4).clamp(0.05, 0.95)
    pos = torch.randn(5, batch_size, d_model)
    text_dict = {
        "encoded_text": torch.randn(batch_size, 7, d_model),
        "text_token_mask": torch.ones(batch_size, 7, dtype=torch.bool),
        "select_text_num": 7,
    }
    return {
        "tgt": tgt,
        "reference_points": reference_points,
        "src": src,
        "src_spatial_shapes": spatial_shapes,
        "src_level_start_index": level_start_index,
        "src_valid_ratios": src_valid_ratios,
        "src_padding_mask": src_padding_mask,
        "pos": pos,
        "text_dict": text_dict,
    }


def _run_decoder(decoder, num_queries, num_det, historical_offsets=None, return_dptd_info=False, dptd_gate_state=None):
    kwargs = _inputs(num_queries)
    return decoder(
        kwargs["tgt"],
        kwargs["reference_points"],
        kwargs["src"],
        kwargs["src_spatial_shapes"],
        kwargs["src_level_start_index"],
        kwargs["src_valid_ratios"],
        src_padding_mask=kwargs["src_padding_mask"],
        pos=kwargs["pos"],
        text_dict=kwargs["text_dict"],
        num=num_det,
        dptd_sampling_offsets=historical_offsets,
        dptd_gate_state=dptd_gate_state,
        return_dptd_info=return_dptd_info,
    )


def test_baseline_decoder_contract():
    decoder = _make_decoder(use_ov_dptd=False)
    out = _run_decoder(decoder, num_queries=3, num_det=3)
    assert len(out) == 5
    hs_cti, hs_ofa, inter_refs, pre_classes, query_pos = out
    assert hs_cti.shape == (2, 1, 3, 256)
    assert hs_ofa.shape == (2, 1, 3, 256)
    assert inter_refs.shape == (2, 1, 3, 4)
    assert pre_classes.shape == (2, 1, 3, 7)
    assert query_pos.shape == (3, 1, 256)


def test_dptd_first_and_second_frame_offsets():
    decoder = _make_decoder(use_ov_dptd=True)

    first = _run_decoder(decoder, num_queries=3, num_det=3, return_dptd_info=True)
    assert len(first) == 6
    first_info = first[-1]
    assert first_info["sampling_offsets"].shape == (1, 3, 4, 2, 2, 2)
    assert first_info["sampling_offsets"][0].shape == (3, 4, 2, 2, 2)

    num_det = 3
    num_tracks = 2
    num_queries = num_det + num_tracks
    historical = torch.zeros(num_queries, 4, 2, 2, 2)
    second = _run_decoder(
        decoder,
        num_queries=num_queries,
        num_det=num_det,
        historical_offsets=historical,
        return_dptd_info=True,
    )
    second_info = second[-1]
    assert second_info["sampling_offsets"].shape == (1, num_queries, 4, 2, 2, 2)
    assert second_info["sampling_offsets"][0].shape == (num_queries, 4, 2, 2, 2)
    assert second_info["debug"]["ov_dptd_num_track_queries"] == num_tracks
    assert second_info["debug"]["ov_dptd_historical_offset_used_count"] > 0

    fake_model = OVTR.__new__(OVTR)
    fake_model.use_ov_dptd = True
    fake_model.transformer = SimpleNamespace(decoder=SimpleNamespace(layers=decoder.layers))
    track_instances = Instances((1, 1))
    track_instances.query_tgt = torch.zeros(num_queries, 256)
    OVTR._attach_dptd_sampling_offsets(
        fake_model,
        {"dptd_sampling_offsets": second_info["sampling_offsets"]},
        track_instances,
    )
    assert track_instances.has("dptd_sampling_offsets")
    assert track_instances.dptd_sampling_offsets.shape == (num_queries, 4, 2, 2, 2)


def _args_cfg(**overrides):
    args = SimpleNamespace(quant_deploy="none")
    cfg = SimpleNamespace(
        use_ov_dptd=True,
        ov_dptd_use_historical_offsets=True,
        ov_dptd_fusion="linear_sum",
        ov_dptd_id_path_text="none",
        ov_dptd_fuse_cti=False,
        ov_dptd_store_debug=False,
        use_dptd_update_suppression=False,
        dptd_update_suppression_thresh=0.4,
        dptd_update_suppression_restore_fields=[
            "query_tgt",
            "query_pos",
            "ref_pts",
            "dptd_sampling_offsets",
            "output_embedding_img",
            "output_embedding_txt",
        ],
        dptd_update_suppression_track_id_based=True,
        use_dptd_semantic_memory=False,
        dptd_memory_ema=0.8,
        dptd_memory_min_score=0.4,
        dptd_memory_max_entropy=0.75,
        dptd_memory_use_alignment_feature=True,
        dptd_memory_allow_untrained_visual_projection=False,
        dptd_memory_store_topk=5,
        dptd_memory_debug=False,
        use_dptd_semantic_gate=False,
        dptd_gate_mode="heuristic",
        dptd_gate_min_score=0.3,
        dptd_gate_max_entropy=0.8,
        dptd_gate_semantic_cos_tau=0.25,
        dptd_gate_visual_cos_tau=0.25,
        dptd_gate_offset_tau=0.2,
        dptd_gate_box_iou_tau=0.3,
        dptd_gate_temperature=10.0,
        dptd_gate_min_appearance=0.1,
        dptd_gate_debug=False,
        use_dptd_semantic_update_suppression=False,
        dptd_semantic_update_suppression_thresh=0.3,
        use_checkpoint_track=False,
        use_transformer_ckpt=False,
    )
    for name, value in overrides.items():
        target = args if name.startswith("args__") else cfg
        key = name.split("args__", 1)[1] if name.startswith("args__") else name
        setattr(target, key, value)
    return args, cfg


def _assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__}")


def test_dptd_guards():
    _assert_raises(RuntimeError, lambda: resolve_ov_dptd_options(*_args_cfg(use_checkpoint_track=True)))
    _assert_raises(RuntimeError, lambda: resolve_ov_dptd_options(*_args_cfg(use_transformer_ckpt=True)))
    _assert_raises(RuntimeError, lambda: resolve_ov_dptd_options(*_args_cfg(args__quant_deploy="int_msda")))
    _assert_raises(RuntimeError, lambda: resolve_ov_dptd_options(*_args_cfg(use_ov_dptd=False, use_dptd_update_suppression=True)))
    _assert_raises(RuntimeError, lambda: resolve_ov_dptd_options(*_args_cfg(use_dptd_update_suppression=True, dptd_update_suppression_track_id_based=False)))
    _assert_raises(RuntimeError, lambda: resolve_ov_dptd_options(*_args_cfg(use_ov_dptd=False, use_dptd_semantic_memory=True)))
    _assert_raises(RuntimeError, lambda: resolve_ov_dptd_options(*_args_cfg(use_dptd_semantic_memory=True, dptd_memory_use_alignment_feature=False)))
    _assert_raises(RuntimeError, lambda: resolve_ov_dptd_options(*_args_cfg(use_dptd_semantic_gate=True, use_dptd_semantic_memory=False)))
    _assert_raises(RuntimeError, lambda: resolve_ov_dptd_options(*_args_cfg(use_dptd_semantic_gate=True, use_dptd_semantic_memory=True, use_dptd_semantic_update_suppression=True)))
    _assert_raises(RuntimeError, lambda: resolve_ov_dptd_options(*_args_cfg(ov_dptd_fusion="semantic_gate")))
    _assert_raises(NotImplementedError, lambda: resolve_ov_dptd_options(*_args_cfg(use_dptd_semantic_gate=True, use_dptd_semantic_memory=True, dptd_gate_mode="mlp")))
    _assert_raises(NotImplementedError, lambda: resolve_ov_dptd_options(*_args_cfg(ov_dptd_fuse_cti=True)))

    fake_model = OVTR.__new__(OVTR)
    fake_model.training = True
    fake_model.use_ov_dptd = True
    fake_model.use_checkpoint = False
    fake_model.use_dptd_update_suppression = False
    _assert_raises(RuntimeError, lambda: OVTR._check_ov_dptd_forward_supported(fake_model, 2))
    _assert_raises(RuntimeError, lambda: OVTR._forward_hybrid_batched(fake_model, [], []))

    fake_model.use_dptd_update_suppression = True
    _assert_raises(RuntimeError, lambda: OVTR._check_ov_dptd_forward_supported(fake_model, 1))


def _make_suppression_model():
    fake_model = OVTR.__new__(OVTR)
    fake_model.training = False
    fake_model.use_ov_dptd = True
    fake_model.use_dptd_update_suppression = True
    fake_model.dptd_update_suppression_thresh = 0.4
    fake_model.dptd_update_suppression_restore_fields = [
        "query_tgt",
        "query_pos",
        "ref_pts",
        "dptd_sampling_offsets",
        "output_embedding_img",
        "output_embedding_txt",
    ]
    fake_model.dptd_update_suppression_track_id_based = True
    fake_model.ov_dptd_debug_stats = {
        "dptd_update_suppressed_count": 0,
        "dptd_update_suppressed_ids": [],
        "dptd_update_suppression_restore_success_count": 0,
        "dptd_update_suppression_restore_skip_count": 0,
    }
    return fake_model


def _make_suppression_tracks():
    num_tracks = 3
    hidden_dim = 4
    inst = Instances((1, 1))
    inst.obj_idxes = torch.tensor([10, 11, -1], dtype=torch.long)
    inst.scores = torch.zeros(num_tracks)
    inst.query_tgt = torch.arange(num_tracks * hidden_dim, dtype=torch.float32).view(num_tracks, hidden_dim)
    inst.query_pos = inst.query_tgt + 10.0
    inst.ref_pts = torch.arange(num_tracks * 4, dtype=torch.float32).view(num_tracks, 4) + 20.0
    inst.dptd_sampling_offsets = torch.arange(num_tracks * 2 * 1 * 2 * 2, dtype=torch.float32).view(num_tracks, 2, 1, 2, 2)
    inst.output_embedding_img = inst.query_tgt + 30.0
    inst.output_embedding_txt = inst.query_tgt + 40.0
    inst.pred_boxes = torch.zeros(num_tracks, 4)
    inst.pred_logits = torch.zeros(num_tracks, 5)
    inst.disappear_time = torch.zeros(num_tracks, dtype=torch.long)
    inst.cls_idxes = torch.zeros(num_tracks, dtype=torch.long)
    return inst


def test_dptd_update_suppression_restore_by_track_id():
    model = _make_suppression_model()
    tracks = _make_suppression_tracks()
    old_values = {
        name: tracks.get(name).detach().clone()
        for name in model.dptd_update_suppression_restore_fields
    }

    snapshot = OVTR._snapshot_dptd_update_suppression_state(model, tracks)
    for name in model.dptd_update_suppression_restore_fields:
        tracks.set(name, tracks.get(name) + 1000.0)
    tracks.scores = torch.tensor([0.2, 0.9, 0.95])

    suppressed_ids = OVTR._select_dptd_update_suppressed_ids(model, snapshot, tracks)
    assert suppressed_ids.detach().cpu().tolist() == [10]

    restored = OVTR._restore_dptd_update_suppressed_state(model, snapshot, tracks)
    for name in model.dptd_update_suppression_restore_fields:
        value = restored.get(name)
        assert torch.equal(value[0], old_values[name][0]), name
        assert torch.equal(value[1], old_values[name][1] + 1000.0), name
        assert torch.equal(value[2], old_values[name][2] + 1000.0), name

    assert model.ov_dptd_debug_stats["dptd_update_suppressed_count"] == 1
    assert model.ov_dptd_debug_stats["dptd_update_suppressed_ids"] == [10]
    assert model.ov_dptd_debug_stats["dptd_update_suppression_restore_success_count"] == len(model.dptd_update_suppression_restore_fields)


def test_dptd_update_suppression_after_track_base_update():
    model = _make_suppression_model()
    tracks = _make_suppression_tracks()
    old_values = {
        name: tracks.get(name).detach().clone()
        for name in model.dptd_update_suppression_restore_fields
    }
    snapshot = OVTR._snapshot_dptd_update_suppression_state(model, tracks)

    for name in model.dptd_update_suppression_restore_fields:
        tracks.set(name, tracks.get(name) + 1000.0)
    tracks.scores = torch.tensor([0.2, 0.9, 0.95])
    OVTR._select_dptd_update_suppressed_ids(model, snapshot, tracks)

    tracker = RuntimeTrackerBase(score_thresh=0.6, filter_score_thresh=0.4, miss_tolerance=5)
    tracker.max_obj_id = 100
    updated = tracker.update(tracks, torch.zeros(len(tracks), dtype=torch.bool))
    restored = OVTR._restore_dptd_update_suppressed_state(model, snapshot, updated)

    ids = restored.obj_idxes.detach().cpu().tolist()
    low_idx = ids.index(10)
    high_idx = ids.index(11)
    new_idx = ids.index(100)
    for name in model.dptd_update_suppression_restore_fields:
        value = restored.get(name)
        assert torch.equal(value[low_idx], old_values[name][0]), name
        assert torch.equal(value[high_idx], old_values[name][1] + 1000.0), name
        assert torch.equal(value[new_idx], old_values[name][2] + 1000.0), name



def _make_memory_model(topk=2, allow_projection=False):
    fake_model = OVTR.__new__(OVTR)
    fake_model.training = False
    fake_model.use_ov_dptd = True
    fake_model.use_dptd_semantic_memory = True
    fake_model.dptd_memory_ema = 0.8
    fake_model.dptd_memory_min_score = 0.4
    fake_model.dptd_memory_max_entropy = 0.75
    fake_model.dptd_memory_use_alignment_feature = True
    fake_model.dptd_memory_allow_untrained_visual_projection = allow_projection
    fake_model.dptd_memory_store_topk = topk
    fake_model.dptd_memory_debug = True
    fake_model.use_dptd_semantic_gate = True
    fake_model.dptd_gate_visual_cos_tau = 0.25
    fake_model.dptd_gate_temperature = 10.0
    fake_model.use_dptd_semantic_update_suppression = False
    fake_model.dptd_semantic_update_suppression_thresh = 0.3
    fake_model.dptd_memory_dim = 4
    fake_model.dptd_visual_memory_proj = nn.Linear(4, 4) if allow_projection else None
    fake_model.ov_dptd_debug_stats = {
        "dptd_memory_update_count": 0,
        "dptd_memory_keep_count": 0,
        "dptd_memory_init_count": 0,
        "dptd_memory_entropy_mean": 0.0,
        "dptd_memory_semantic_proto_cosine_delta_mean": 0.0,
        "dptd_memory_topk_change_rate": 0.0,
        "dptd_memory_visual_source": "none",
    }
    return fake_model


def _make_memory_tracks(num_tracks=1, obj_idxes=None):
    if obj_idxes is None:
        obj_idxes = list(range(10, 10 + num_tracks))
    inst = Instances((1, 1))
    inst.obj_idxes = torch.tensor(obj_idxes, dtype=torch.long)
    inst.scores = torch.ones(num_tracks)
    inst.query_tgt = torch.zeros(num_tracks, 4)
    inst.query_pos = torch.zeros(num_tracks, 4)
    inst.ref_pts = torch.zeros(num_tracks, 4)
    inst.pred_boxes = torch.zeros(num_tracks, 4)
    inst.pred_logits = torch.zeros(num_tracks, 3)
    inst.output_embedding_img = torch.randn(num_tracks, 4)
    inst.output_embedding_txt = torch.randn(num_tracks, 4)
    inst.disappear_time = torch.zeros(num_tracks, dtype=torch.long)
    inst.cls_idxes = torch.zeros(num_tracks, dtype=torch.long)
    return inst


def _text_memory(num_cls):
    base = torch.eye(4)
    if num_cls <= 4:
        return base[:num_cls]
    extra = torch.randn(num_cls - 4, 4)
    return torch.cat([base, extra], dim=0)


def _attach_memory_candidates(model, tracks, logits, select_id, pred_embed=None):
    tracks.pred_logits = logits.clone()
    frame_res = {
        "select_id": torch.tensor(select_id, dtype=torch.long),
        "dptd_text_memory_embeddings": _text_memory(len(select_id)),
    }
    if pred_embed is not None:
        frame_res["pred_embed"] = pred_embed.unsqueeze(0)
    OVTR._attach_dptd_memory_candidates(model, frame_res, tracks)
    assert "dptd_text_memory_embeddings" not in frame_res
    return frame_res


def test_dptd_memory_init_topk_detach_and_entropy_single_class():
    model = _make_memory_model(topk=2)
    tracks = _make_memory_tracks(num_tracks=1, obj_idxes=[10])
    OVTR._ensure_dptd_memory_fields(model, tracks)
    snapshot = OVTR._snapshot_dptd_memory_state(model, tracks)

    logits = torch.tensor([[2.0]], requires_grad=True)
    pred_embed = torch.tensor([[0.0, 1.0, 0.0, 0.0]], requires_grad=True)
    _attach_memory_candidates(model, tracks, logits, [42], pred_embed=pred_embed)
    assert not torch.isnan(tracks._dptd_current_semantic_entropy).any()
    for field_name in DPTD_CURRENT_MEMORY_FIELDS:
        value = tracks.get(field_name)
        assert value.grad_fn is None, field_name

    updated = OVTR._update_dptd_memory_state(model, snapshot, None, tracks)
    assert updated.dptd_semantic_proto.shape == (1, 4)
    assert updated.dptd_visual_memory.shape == (1, 4)
    assert updated.dptd_memory_age.tolist() == [0]
    assert updated.dptd_topk_class_indices.tolist() == [[42, -1]]
    assert torch.allclose(updated.dptd_topk_class_scores[0, 0], torch.tensor(1.0))
    for field_name in DPTD_CURRENT_MEMORY_FIELDS:
        assert not updated.has(field_name), field_name
    for field_name in ["dptd_semantic_proto", "dptd_visual_memory", "dptd_semantic_conf", "dptd_semantic_entropy", "dptd_topk_class_scores"]:
        assert updated.get(field_name).grad_fn is None, field_name


def test_dptd_memory_existing_ema_not_confused_by_zero_age():
    model = _make_memory_model(topk=2)
    tracks = _make_memory_tracks(num_tracks=1, obj_idxes=[10])
    OVTR._ensure_dptd_memory_fields(model, tracks)
    tracks.dptd_semantic_proto[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    tracks.dptd_visual_memory[0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    tracks.dptd_memory_age[0] = 5

    snapshot = OVTR._snapshot_dptd_memory_state(model, tracks)
    _attach_memory_candidates(model, tracks, torch.tensor([[3.0]]), [5], pred_embed=torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
    tracks = OVTR._update_dptd_memory_state(model, snapshot, None, tracks)
    assert tracks.dptd_memory_age.tolist() == [0]

    prev_semantic = tracks.dptd_semantic_proto.detach().clone()
    prev_visual = tracks.dptd_visual_memory.detach().clone()
    snapshot = OVTR._snapshot_dptd_memory_state(model, tracks)
    _attach_memory_candidates(model, tracks, torch.tensor([[-5.0, 5.0]]), [8, 9], pred_embed=torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    current_semantic = tracks._dptd_current_semantic_proto.detach().clone()
    current_visual = tracks._dptd_current_visual_memory.detach().clone()
    tracks = OVTR._update_dptd_memory_state(model, snapshot, None, tracks)

    expected_semantic = torch.nn.functional.normalize(0.8 * prev_semantic + 0.2 * current_semantic, dim=-1, eps=1e-6)
    expected_visual = torch.nn.functional.normalize(0.8 * prev_visual + 0.2 * current_visual, dim=-1, eps=1e-6)
    assert torch.allclose(tracks.dptd_semantic_proto, expected_semantic, atol=1e-6)
    assert torch.allclose(tracks.dptd_visual_memory, expected_visual, atol=1e-6)
    assert not torch.allclose(tracks.dptd_semantic_proto, current_semantic)
    assert tracks.dptd_memory_age.tolist() == [0]


def test_dptd_memory_keep_low_conf_high_entropy_and_suppressed():
    model = _make_memory_model(topk=2)
    tracks = _make_memory_tracks(num_tracks=2, obj_idxes=[10, 11])
    OVTR._ensure_dptd_memory_fields(model, tracks)
    tracks.dptd_semantic_proto[:] = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    tracks.dptd_visual_memory[:] = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    tracks.dptd_memory_age[:] = torch.tensor([2, 3])
    old_semantic = tracks.dptd_semantic_proto.clone()
    old_visual = tracks.dptd_visual_memory.clone()
    snapshot = OVTR._snapshot_dptd_memory_state(model, tracks)

    logits = torch.tensor([[-4.0, -5.0], [2.0, 2.0]])
    pred_embed = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    _attach_memory_candidates(model, tracks, logits, [3, 9], pred_embed=pred_embed)
    tracks.scores = torch.tensor([0.1, 0.9])
    tracks = OVTR._update_dptd_memory_state(model, snapshot, None, tracks)
    assert torch.equal(tracks.dptd_semantic_proto, old_semantic)
    assert torch.equal(tracks.dptd_visual_memory, old_visual)
    assert tracks.dptd_memory_age.tolist() == [3, 4]

    snapshot = OVTR._snapshot_dptd_memory_state(model, tracks)
    _attach_memory_candidates(model, tracks, torch.tensor([[4.0], [4.0]]), [7], pred_embed=pred_embed)
    tracks.scores = torch.tensor([0.9, 0.9])
    suppressed = {"suppressed_ids": torch.tensor([10], dtype=torch.long)}
    tracks = OVTR._update_dptd_memory_state(model, snapshot, suppressed, tracks)
    assert torch.equal(tracks.dptd_semantic_proto[0], old_semantic[0])
    assert tracks.dptd_memory_age[0].item() == 4
    assert tracks.dptd_memory_age[1].item() == 0


def test_dptd_memory_missing_pred_embed_and_selected_class_changes():
    model = _make_memory_model(topk=2)
    tracks = _make_memory_tracks(num_tracks=1, obj_idxes=[10])
    OVTR._ensure_dptd_memory_fields(model, tracks)
    _assert_raises(RuntimeError, lambda: _attach_memory_candidates(model, tracks, torch.tensor([[1.0, 2.0]]), [1, 2], pred_embed=None))

    snapshot = OVTR._snapshot_dptd_memory_state(model, tracks)
    frame_res = _attach_memory_candidates(model, tracks, torch.tensor([[3.0, 1.0, -2.0]]), [30, 10, 20], pred_embed=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert "dptd_text_memory_embeddings" not in frame_res
    score = torch.tensor([[3.0, 1.0, -2.0]]).sigmoid()
    prob = score / score.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    assert tracks._dptd_current_topk_class_indices.tolist() == [[30, 10]]
    assert torch.allclose(tracks._dptd_current_topk_class_scores, prob[:, [0, 1]])
    tracks = OVTR._update_dptd_memory_state(model, snapshot, None, tracks)
    assert tracks.dptd_semantic_proto.shape == (1, 4)

    snapshot = OVTR._snapshot_dptd_memory_state(model, tracks)
    _attach_memory_candidates(model, tracks, torch.tensor([[2.0]]), [99], pred_embed=torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    tracks = OVTR._update_dptd_memory_state(model, snapshot, None, tracks)
    assert tracks.dptd_semantic_proto.shape == (1, 4)
    assert tracks.dptd_topk_class_indices.tolist() == [[99, -1]]


def test_dptd_memory_track_base_new_row_and_instances_cat():
    model = _make_memory_model(topk=2)
    tracks = _make_memory_tracks(num_tracks=2, obj_idxes=[10, -1])
    OVTR._ensure_dptd_memory_fields(model, tracks)
    tracks.dptd_semantic_proto[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    tracks.dptd_visual_memory[0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    snapshot = OVTR._snapshot_dptd_memory_state(model, tracks)
    _attach_memory_candidates(
        model,
        tracks,
        torch.tensor([[3.0], [4.0]]),
        [5],
        pred_embed=torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    new_candidate = tracks._dptd_current_semantic_proto[1].detach().clone()
    tracks.scores = torch.tensor([0.9, 0.95])
    tracker = RuntimeTrackerBase(score_thresh=0.6, filter_score_thresh=0.4, miss_tolerance=5)
    tracker.max_obj_id = 100
    updated = tracker.update(tracks, torch.zeros(len(tracks), dtype=torch.bool))
    updated = OVTR._update_dptd_memory_state(model, snapshot, None, updated)
    ids = updated.obj_idxes.detach().cpu().tolist()
    new_idx = ids.index(100)
    assert torch.allclose(updated.dptd_semantic_proto[new_idx], new_candidate)
    for field_name in DPTD_CURRENT_MEMORY_FIELDS:
        assert not updated.has(field_name), field_name

    init_tracks = _make_memory_tracks(num_tracks=2, obj_idxes=[-1, -1])
    init_tracks.pred_logits = torch.zeros(2, updated.pred_logits.shape[1])
    OVTR._ensure_dptd_memory_fields(model, init_tracks)
    merged = Instances.cat([init_tracks, updated[updated.obj_idxes >= 0]])
    assert merged.dptd_semantic_proto.shape == (4, 4)
    assert merged.dptd_visual_memory.shape == (4, 4)
    assert merged.dptd_topk_class_indices.shape == (4, 2)


def _gate_state(num_queries=5, num_det=3, memory_valid=True, semantic_sign=1.0):
    semantic = torch.zeros(num_queries, 4)
    semantic[num_det:] = semantic_sign * torch.tensor([1.0, 0.0, 0.0, 0.0])
    visual = torch.ones(num_queries, 4)
    valid = torch.zeros(num_queries, dtype=torch.bool)
    if memory_valid:
        valid[num_det:] = True
    return {
        "semantic_proto": semantic,
        "visual_memory": visual,
        "memory_age": torch.zeros(num_queries, dtype=torch.long),
        "pred_boxes": torch.zeros(num_queries, 4),
        "historical_offsets": torch.zeros(num_queries, 4, 2, 2, 2),
        "obj_idxes": torch.tensor([-1, -1, -1, 10, 11], dtype=torch.long)[:num_queries],
        "memory_valid": valid,
        "text_memory_embeddings": torch.eye(4)[:1],
    }


def test_dptd_semantic_gate_helper_and_deferred_box():
    decoder = _make_decoder(
        use_ov_dptd=True,
        use_dptd_semantic_gate=True,
        dptd_semantic_update_suppression_thresh=0.3,
    )
    logits = torch.full((1, 5, 1), 5.0)
    ad_offsets = torch.zeros(1, 5, 4, 2, 2, 2)
    historical = torch.zeros_like(ad_offsets)
    tgt = torch.zeros(5, 1, 256)
    debug = {}
    gate = decoder._compute_dptd_semantic_gate(logits, ad_offsets, historical, _gate_state(semantic_sign=1.0), 3, tgt, debug)
    assert gate.shape == (1, 5)
    assert torch.allclose(gate[:, :3], torch.ones(1, 3))
    assert not torch.isnan(gate).any()
    assert debug["dptd_gate_box_conf_deferred"] is True
    assert debug["box_consistency_mean"] == 1.0

    low_gate = decoder._compute_dptd_semantic_gate(logits, ad_offsets, historical, _gate_state(semantic_sign=-1.0), 3, tgt, {})
    assert float(low_gate[0, 3]) < float(gate[0, 3])

    no_memory_gate = decoder._compute_dptd_semantic_gate(logits, ad_offsets, historical, _gate_state(memory_valid=False), 3, tgt, {})
    assert torch.allclose(no_memory_gate, torch.ones_like(no_memory_gate))

    one_cls_debug = {}
    one_cls_gate = decoder._compute_dptd_semantic_gate(torch.zeros(1, 5, 1), ad_offsets, historical, _gate_state(), 3, tgt, one_cls_debug)
    assert not torch.isnan(one_cls_gate).any()


def test_dptd_semantic_gate_fusion_track_only():
    decoder = _make_decoder(use_ov_dptd=True, use_dptd_semantic_gate=True, ov_dptd_fusion="semantic_gate")
    decoder._reset_linear_identity(decoder.ov_dptd_ofa_id_proj[0])
    decoder.ov_dptd_gate_alpha.data.fill_(1.0)
    ada = torch.zeros(5, 1, 256)
    ident = torch.ones(5, 1, 256)
    gate = torch.ones(1, 5)
    gate[:, 3:] = 0.0
    fused = decoder._fuse_ov_dptd_ofa_semantic_gate(0, ada, ident, gate)
    assert torch.equal(fused[:3], ada[:3])
    assert torch.allclose(fused[3:], torch.ones_like(fused[3:]))


def test_dptd_semantic_gate_linear_sum_debug_contract():
    decoder = _make_decoder(use_ov_dptd=True, use_dptd_semantic_gate=True, ov_dptd_fusion="linear_sum")
    gate_state = _gate_state(num_queries=5, num_det=3)
    out = _run_decoder(decoder, num_queries=5, num_det=3, historical_offsets=torch.zeros(5, 4, 2, 2, 2), return_dptd_info=True, dptd_gate_state=gate_state)
    info = out[-1]
    assert info["semantic_gate"].shape == (1, 5)
    assert "dptd_gate_mean" in info["debug"]
    assert info["debug"]["dptd_gate_box_conf_deferred"] is True
    assert out[1].shape == (2, 1, 5, 256)


def test_dptd_semantic_update_suppression_and_visual_consistency_order():
    model = _make_memory_model(topk=2)
    model.use_dptd_update_suppression = True
    model.use_dptd_semantic_update_suppression = True
    model.dptd_semantic_update_suppression_thresh = 0.5
    model.dptd_update_suppression_thresh = 0.0
    model.dptd_update_suppression_track_id_based = True
    model.dptd_update_suppression_restore_fields = ["query_tgt"]
    model.ov_dptd_debug_stats.update({"dptd_update_suppressed_count": 0, "dptd_update_suppressed_ids": []})
    tracks = _make_memory_tracks(num_tracks=1, obj_idxes=[10])
    OVTR._ensure_dptd_memory_fields(model, tracks)
    tracks.dptd_visual_memory[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    snapshot = OVTR._snapshot_dptd_memory_state(model, tracks)
    _attach_memory_candidates(model, tracks, torch.tensor([[4.0]]), [5], pred_embed=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    tracks.set("_dptd_current_gate", torch.tensor([0.2]))
    tracks = OVTR._compute_dptd_post_visual_consistency(model, snapshot, tracks)
    assert tracks.has("_dptd_current_visual_consistency")
    assert model.ov_dptd_debug_stats["visual_consistency_mean"] > 0.5
    suppression_snapshot = {"ids": torch.tensor([10], dtype=torch.long), "fields": {}, "suppressed_ids": torch.empty(0, dtype=torch.long)}
    suppressed = OVTR._extend_dptd_semantic_update_suppression(model, suppression_snapshot, tracks)
    assert suppressed.detach().cpu().tolist() == [10]
    tracks = OVTR._update_dptd_memory_state(model, snapshot, suppression_snapshot, tracks)
    for field_name in DPTD_CURRENT_MEMORY_FIELDS:
        assert not tracks.has(field_name), field_name
    assert not tracks.has("_dptd_current_gate")
    assert not tracks.has("_dptd_current_visual_consistency")

def main():
    torch.manual_seed(0)
    test_baseline_decoder_contract()
    test_dptd_first_and_second_frame_offsets()
    test_dptd_guards()
    test_dptd_update_suppression_restore_by_track_id()
    test_dptd_update_suppression_after_track_base_update()
    test_dptd_memory_init_topk_detach_and_entropy_single_class()
    test_dptd_memory_existing_ema_not_confused_by_zero_age()
    test_dptd_memory_keep_low_conf_high_entropy_and_suppressed()
    test_dptd_memory_missing_pred_embed_and_selected_class_changes()
    test_dptd_memory_track_base_new_row_and_instances_cat()
    test_dptd_semantic_gate_helper_and_deferred_box()
    test_dptd_semantic_gate_fusion_track_only()
    test_dptd_semantic_gate_linear_sum_debug_contract()
    test_dptd_semantic_update_suppression_and_visual_consistency_order()
    print("DPTD smoke passed")


if __name__ == "__main__":
    main()
