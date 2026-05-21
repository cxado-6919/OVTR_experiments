import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectron2.structures import Instances  # noqa: E402
from models.ovtr import OVTR, RuntimeTrackerBase, resolve_ov_dptd_options  # noqa: E402
from models.transformer import DeformableTransformerDecoderLayer, TransformerDecoder  # noqa: E402


def _make_decoder(use_ov_dptd=False, num_layers=2, d_model=256, nheads=4, nlevels=2, npoints=2, num_det=3):
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


def _run_decoder(decoder, num_queries, num_det, historical_offsets=None, return_dptd_info=False):
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


def main():
    torch.manual_seed(0)
    test_baseline_decoder_contract()
    test_dptd_first_and_second_frame_offsets()
    test_dptd_guards()
    test_dptd_update_suppression_restore_by_track_id()
    test_dptd_update_suppression_after_track_base_update()
    print("DPTD smoke passed")


if __name__ == "__main__":
    main()
