import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectron2.structures import Instances  # noqa: E402
from models.ovtr import OVTR, RuntimeTrackerBase  # noqa: E402
from models.updater import (  # noqa: E402
    Category_Information_Propagator,
    MemoryCalibratedCategoryInformationPropagator,
    build as build_updater,
    ensure_mcip_track_fields,
)
from util.misc import inverse_sigmoid  # noqa: E402


def _make_args(**overrides):
    args = SimpleNamespace(
        random_drop=0.0,
        fp_ratio=0.0,
        update_query_pos=False,
        merger_dropout=0.0,
        mcip_enable=True,
        mcip_detach_memory=True,
        mcip_memory_momentum=0.8,
        mcip_use_semantic_memory=True,
        mcip_use_motion_ref=False,
        mcip_motion_momentum=0.7,
        mcip_motion_scale_init=0.0,
        mcip_max_memory_update=0.05,
        mcip_max_residual_ratio=0.05,
        mcip_motion_offset_cap=0.02,
        mcip_semantic_topk=5,
        mcip_gate_use_txt=False,
        debug_mcip=True,
        mcip_debug_log_interval=1,
        mcip_debug_stats_file=None,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _empty_init(num_classes, hidden_dim, device):
    inst = Instances((1, 1))
    inst.ref_pts = torch.zeros((0, 4), device=device)
    inst.query_tgt = torch.zeros((0, hidden_dim), device=device)
    inst.query_pos = torch.zeros((0, hidden_dim), device=device)
    inst.obj_idxes = torch.zeros((0,), dtype=torch.long, device=device)
    inst.matched_gt_idxes = torch.zeros((0,), dtype=torch.long, device=device)
    inst.iou = torch.zeros((0,), device=device)
    inst.scores = torch.zeros((0,), device=device)
    inst.pred_boxes = torch.zeros((0, 4), device=device)
    inst.pred_logits = torch.zeros((0, num_classes), device=device)
    inst.img_memory = torch.zeros((0, hidden_dim), device=device)
    inst.semantic_memory = torch.zeros((0, hidden_dim), device=device)
    inst.cls_conf_memory = torch.zeros((0,), device=device)
    inst.cls_entropy_memory = torch.zeros((0,), device=device)
    inst.prev_boxes = torch.zeros((0, 4), device=device)
    inst.box_velocity = torch.zeros((0, 4), device=device)
    inst.memory_age = torch.zeros((0,), device=device)
    return inst


def _tracks(num_tracks=5, hidden_dim=256, num_classes=7, device=torch.device("cpu")):
    inst = Instances((1, 1))
    inst.ref_pts = torch.zeros((num_tracks, 4), device=device)
    inst.query_tgt = torch.randn((num_tracks, hidden_dim), device=device)
    inst.query_pos = torch.randn((num_tracks, hidden_dim), device=device)
    inst.obj_idxes = torch.arange(num_tracks, dtype=torch.long, device=device)
    inst.matched_gt_idxes = torch.full((num_tracks,), -1, dtype=torch.long, device=device)
    inst.iou = torch.ones((num_tracks,), device=device)
    inst.scores = torch.linspace(0.9, 0.5, num_tracks, device=device)
    inst.pred_boxes = torch.rand((num_tracks, 4), device=device).clamp(1e-3, 1 - 1e-3)
    inst.pred_logits = torch.randn((num_tracks, num_classes), device=device)
    inst.output_embedding_img = torch.randn((num_tracks, hidden_dim), device=device)
    inst.output_embedding_txt = torch.randn((num_tracks, hidden_dim), device=device)
    inst.semantic_obs = torch.nn.functional.normalize(torch.randn((num_tracks, hidden_dim), device=device), dim=-1)
    inst.cls_conf_obs = torch.rand((num_tracks,), device=device)
    inst.cls_entropy_obs = torch.rand((num_tracks,), device=device)
    ensure_mcip_track_fields(inst, hidden_dim, device=device, dtype=torch.float32)
    return inst


def _clone_instances(inst):
    clone = Instances(inst.image_size)
    for name, value in inst.get_fields().items():
        if isinstance(value, torch.Tensor):
            value = value.clone()
        clone.set(name, value)
    return clone


def _forward(updater, track_instances, num_classes, hidden_dim, device):
    return updater({
        "track_instances": track_instances,
        "init_track_instances": _empty_init(num_classes, hidden_dim, device),
    })


def _matched_updaters(hidden_dim, args=None):
    args = _make_args() if args is None else args
    torch.manual_seed(123)
    baseline = Category_Information_Propagator(
        args,
        dim_in=hidden_dim,
        hidden_dim=hidden_dim,
        dim_out=hidden_dim * 2,
    )
    mcip = MemoryCalibratedCategoryInformationPropagator(
        args,
        dim_in=hidden_dim,
        hidden_dim=hidden_dim,
        dim_out=hidden_dim * 2,
    )
    mcip.load_state_dict(baseline.state_dict(), strict=False)
    baseline.eval()
    mcip.eval()
    return baseline, mcip


def _relative_l2(a, b):
    return ((a - b).float().norm() / b.float().norm().clamp_min(1e-6)).item()


def test_zero_init_mcip_matches_baseline_cip():
    device = torch.device("cpu")
    hidden_dim = 64
    num_classes = 9
    baseline, mcip = _matched_updaters(hidden_dim)
    tracks = _tracks(6, hidden_dim, num_classes, device)
    tracks.memory_age = torch.ones((6,), device=device)
    tracks.img_memory = tracks.output_embedding_img.clone()
    tracks.semantic_memory = tracks.semantic_obs.clone()
    tracks.cls_entropy_obs = torch.zeros((6,), device=device)
    tracks.scores = torch.ones((6,), device=device)

    with torch.no_grad():
        base_out = _forward(baseline, _clone_instances(tracks), num_classes, hidden_dim, device)
        mcip_out = _forward(mcip, _clone_instances(tracks), num_classes, hidden_dim, device)

    cosine = torch.nn.functional.cosine_similarity(
        mcip_out.query_tgt.float(),
        base_out.query_tgt.float(),
        dim=-1,
    ).mean().item()
    rel_l2 = _relative_l2(mcip_out.query_tgt, base_out.query_tgt)
    assert cosine >= 0.999, cosine
    assert rel_l2 <= 1e-4, rel_l2
    assert torch.allclose(mcip_out.query_tgt, base_out.query_tgt, rtol=1e-4, atol=1e-5)
    assert mcip.mcip_debug_stats["inject_gate_max"] > 0.0
    assert mcip.mcip_debug_stats["query_tgt_cosine_to_baseline"] >= 0.999
    assert mcip.mcip_debug_stats["relative_query_tgt_l2_error"] <= 1e-4


def test_residual_adapter_final_linear_zero_init():
    updater = MemoryCalibratedCategoryInformationPropagator(
        _make_args(),
        dim_in=32,
        hidden_dim=32,
        dim_out=64,
    )
    final = updater.memory_residual_adapter[-1]
    assert torch.allclose(final.weight, torch.zeros_like(final.weight))
    assert torch.allclose(final.bias, torch.zeros_like(final.bias))


def test_residual_norm_clamp():
    device = torch.device("cpu")
    hidden_dim = 32
    args = _make_args(mcip_max_residual_ratio=0.05)
    _, updater = _matched_updaters(hidden_dim, args)
    with torch.no_grad():
        updater.memory_residual_adapter[-1].bias.fill_(100.0)
    tracks = _tracks(4, hidden_dim, 11, device)
    tracks.memory_age = torch.ones((4,), device=device)
    tracks.img_memory = torch.randn_like(tracks.img_memory)
    tracks.cls_entropy_obs = torch.zeros((4,), device=device)
    tracks.scores = torch.ones((4,), device=device)
    with torch.no_grad():
        out = _forward(updater, tracks, 11, hidden_dim, device)
    assert torch.isfinite(out.query_tgt).all()
    assert updater.mcip_debug_stats["residual_norm_ratio_max"] <= args.mcip_max_residual_ratio + 1e-5


def test_new_track_initializes_memory_without_same_forward_injection():
    device = torch.device("cpu")
    hidden_dim = 32
    num_classes = 6
    baseline, mcip = _matched_updaters(hidden_dim)
    tracks = _tracks(3, hidden_dim, num_classes, device)
    tracks.memory_age = torch.zeros((3,), device=device)
    tracks.img_memory.zero_()
    tracks.semantic_memory.zero_()

    with torch.no_grad():
        base_out = _forward(baseline, _clone_instances(tracks), num_classes, hidden_dim, device)
        mcip_out = _forward(mcip, _clone_instances(tracks), num_classes, hidden_dim, device)

    assert mcip.mcip_debug_stats["inject_gate_max"] == 0.0
    assert _relative_l2(mcip_out.query_tgt, base_out.query_tgt) <= 1e-4
    assert torch.allclose(mcip_out.img_memory, tracks.output_embedding_img, atol=1e-6)
    assert torch.allclose(mcip_out.semantic_memory, tracks.semantic_obs, atol=1e-6)
    assert torch.allclose(mcip_out.memory_age, torch.ones_like(mcip_out.memory_age))


def test_entropy_normalization_and_reliability_clamp():
    device = torch.device("cpu")
    hidden_dim = 16
    num_classes = 100
    updater = MemoryCalibratedCategoryInformationPropagator(
        _make_args(),
        dim_in=hidden_dim,
        hidden_dim=hidden_dim,
        dim_out=hidden_dim * 2,
    )
    tracks = _tracks(3, hidden_dim, num_classes, device)
    normalized = torch.tensor([0.0, 0.5, 1.0], device=device)
    raw = torch.tensor([math.log(num_classes) * 0.25, math.log(num_classes), math.log(num_classes) * 2.0], device=device)
    norm_out = updater._normalize_entropy(normalized, tracks)
    raw_out = updater._normalize_entropy(raw, tracks)
    assert torch.allclose(norm_out, normalized, atol=1e-6)
    assert torch.allclose(raw_out, torch.tensor([0.25, 1.0, 1.0], device=device), atol=1e-6)
    scores = torch.tensor([1.2, 0.5, -1.0], device=device)
    reliability = (scores.clamp(0.0, 1.0) * (1.0 - raw_out)).clamp(0.0, 1.0)
    assert torch.isfinite(reliability).all()
    assert reliability.min().item() >= 0.0
    assert reliability.max().item() <= 1.0


def test_motion_off_ref_pts_uses_current_boxes():
    device = torch.device("cpu")
    hidden_dim = 32
    updater = MemoryCalibratedCategoryInformationPropagator(
        _make_args(mcip_use_motion_ref=False),
        dim_in=hidden_dim,
        hidden_dim=hidden_dim,
        dim_out=hidden_dim * 2,
    ).to(device)
    updater.eval()
    tracks = _tracks(4, hidden_dim, 7, device)
    tracks.memory_age = torch.ones((4,), device=device)
    tracks.prev_boxes = (tracks.pred_boxes - 0.05).clamp(1e-3, 1 - 1e-3)
    tracks.box_velocity = torch.full((4, 4), 0.08, device=device)
    with torch.no_grad():
        out = _forward(updater, tracks, 7, hidden_dim, device)
    expected_ref = inverse_sigmoid(tracks.pred_boxes[:, :4].detach().clamp(1e-4, 1.0 - 1e-4))
    assert torch.allclose(out.ref_pts, expected_ref, atol=1e-6)
    assert updater.mcip_debug_stats["motion_offset_l1_mean"] == 0.0
    assert updater.mcip_debug_stats["motion_offset_l1_max"] == 0.0


def test_mcip_updater_dummy_forward_and_debug_file():
    device = torch.device("cpu")
    hidden_dim = 64
    num_tracks = 5
    num_classes = 7
    required_debug_keys = [
        "residual_norm_ratio_mean",
        "residual_norm_ratio_max",
        "residual_cosine_to_base",
        "query_tgt_cosine_to_baseline",
        "relative_query_tgt_l2_error",
        "gate_saturation_frac",
        "memory_update_gate_mean",
        "memory_update_gate_max",
        "memory_update_gate_min",
        "inject_gate_mean",
        "inject_gate_max",
        "inject_gate_min",
        "motion_offset_l1_mean",
        "motion_offset_l1_max",
        "motion_scale",
        "cls_conf_obs_mean",
        "cls_entropy_norm_mean",
        "current_reliability_mean",
        "memory_reliability_mean",
        "motion_reliability_mean",
        "motion_reliability_max",
        "active_track_count",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        debug_stats_file = Path(tmpdir) / "mcip_debug_stats.jsonl"
        updater = MemoryCalibratedCategoryInformationPropagator(
            _make_args(mcip_debug_stats_file=str(debug_stats_file)),
            dim_in=hidden_dim,
            hidden_dim=hidden_dim,
            dim_out=hidden_dim * 2,
        ).to(device)
        updater.eval()
        tracks = _tracks(num_tracks, hidden_dim, num_classes, device)
        with torch.no_grad():
            out = _forward(updater, tracks, num_classes, hidden_dim, device)

        assert len(out) == num_tracks
        for name, shape in {
            "ref_pts": (num_tracks, 4),
            "query_tgt": (num_tracks, hidden_dim),
            "img_memory": (num_tracks, hidden_dim),
            "semantic_memory": (num_tracks, hidden_dim),
            "box_velocity": (num_tracks, 4),
        }.items():
            assert out.has(name), name
            assert tuple(out.get(name).shape) == shape
            assert torch.isfinite(out.get(name)).all(), name

        for key in required_debug_keys:
            assert key in updater.mcip_debug_stats, key
            assert math.isfinite(float(updater.mcip_debug_stats[key])), key
        assert updater.mcip_debug_stats["active_track_count"] > 0
        assert updater.mcip_debug_stats["residual_norm_ratio_max"] <= updater.mcip_max_residual_ratio + 1e-6
        assert debug_stats_file.exists()
        debug_record = json.loads(debug_stats_file.read_text().strip().splitlines()[0])
        for key in required_debug_keys:
            assert key in debug_record, key


def test_first_velocity_update():
    device = torch.device("cpu")
    hidden_dim = 32
    args = _make_args(mcip_motion_momentum=0.7)
    updater = MemoryCalibratedCategoryInformationPropagator(
        args,
        dim_in=hidden_dim,
        hidden_dim=hidden_dim,
        dim_out=hidden_dim * 2,
    ).to(device)
    updater.eval()

    first_track = _tracks(1, hidden_dim, 5, device)
    first_track.memory_age = torch.zeros((1,), device=device)
    first_track.prev_boxes = torch.zeros((1, 4), device=device)
    first_track.box_velocity = torch.ones((1, 4), device=device)
    with torch.no_grad():
        first_out = _forward(updater, first_track, 5, hidden_dim, device)
    assert torch.allclose(first_out.box_velocity, torch.zeros_like(first_out.box_velocity), atol=1e-6)
    assert torch.allclose(first_out.prev_boxes, first_track.pred_boxes[:, :4].detach(), atol=1e-6)

    next_track = _tracks(1, hidden_dim, 5, device)
    delta = torch.tensor([[0.04, -0.02, 0.03, -0.01]], device=device)
    next_track.memory_age = torch.ones((1,), device=device)
    next_track.prev_boxes = (next_track.pred_boxes[:, :4] - delta).clamp(1e-3, 1 - 1e-3)
    next_track.box_velocity = torch.zeros((1, 4), device=device)
    expected_delta = next_track.pred_boxes[:, :4].detach() - next_track.prev_boxes
    expected_velocity = (1.0 - args.mcip_motion_momentum) * expected_delta
    with torch.no_grad():
        next_out = _forward(updater, next_track, 5, hidden_dim, device)
    assert torch.allclose(next_out.box_velocity, expected_velocity, atol=1e-6)


def _call_semantic_helper(track_instances, text_feat, debug_mcip=True, semantic_topk=5):
    holder = SimpleNamespace(
        mcip_enable=True,
        mcip_detach_memory=True,
        debug_mcip=debug_mcip,
        mcip_semantic_topk=semantic_topk,
        mcip_debug_stats={},
    )
    OVTR._mcip_attach_semantic_observations(holder, {"text_feat": text_feat}, track_instances)
    return holder


def test_semantic_observation_topk_and_full_expectation():
    device = torch.device("cpu")
    hidden_dim = 32
    tracks_topk = _tracks(3, hidden_dim, 7, device)
    text_feat = torch.randn((7, hidden_dim), device=device)
    _call_semantic_helper(tracks_topk, text_feat, debug_mcip=True, semantic_topk=3)
    assert tracks_topk.semantic_obs.shape == (3, hidden_dim)
    assert torch.isfinite(tracks_topk.semantic_obs).all()
    assert torch.allclose(tracks_topk.semantic_obs.norm(dim=-1), torch.ones(3, device=device), atol=1e-5)
    assert tracks_topk.cls_entropy_obs.min().item() >= 0.0
    assert tracks_topk.cls_entropy_obs.max().item() <= 1.0

    tracks_full = _tracks(3, hidden_dim, 7, device)
    _call_semantic_helper(tracks_full, text_feat, debug_mcip=True, semantic_topk=0)
    assert tracks_full.semantic_obs.shape == (3, hidden_dim)
    assert torch.isfinite(tracks_full.semantic_obs).all()
    assert torch.allclose(tracks_full.semantic_obs.norm(dim=-1), torch.ones(3, device=device), atol=1e-5)


def test_semantic_observation_length_mismatch():
    device = torch.device("cpu")
    hidden_dim = 32

    more_logits = _tracks(3, hidden_dim, 7, device)
    holder = _call_semantic_helper(more_logits, torch.randn((5, hidden_dim), device=device), debug_mcip=True)
    assert more_logits.semantic_obs.shape == (3, hidden_dim)
    assert torch.isfinite(more_logits.semantic_obs).all()
    assert holder.mcip_debug_stats["semantic_cls_len_mismatch"] == {
        "pred_logits": 7,
        "text_feat": 5,
        "used": 5,
    }

    fewer_logits = _tracks(3, hidden_dim, 4, device)
    holder = _call_semantic_helper(fewer_logits, torch.randn((6, hidden_dim), device=device), debug_mcip=True)
    assert fewer_logits.semantic_obs.shape == (3, hidden_dim)
    assert torch.isfinite(fewer_logits.cls_conf_obs).all()
    assert holder.mcip_debug_stats["semantic_cls_len_mismatch"] == {
        "pred_logits": 4,
        "text_feat": 6,
        "used": 4,
    }

    no_debug = _tracks(3, hidden_dim, 7, device)
    holder = _call_semantic_helper(no_debug, torch.randn((5, hidden_dim), device=device), debug_mcip=False)
    assert "semantic_cls_len_mismatch" not in holder.mcip_debug_stats

    zero_classes = _tracks(3, hidden_dim, 0, device)
    try:
        _call_semantic_helper(zero_classes, torch.randn((0, hidden_dim), device=device), debug_mcip=True)
    except RuntimeError as exc:
        assert "zero classes" in str(exc)
    else:
        raise AssertionError("cls_len == 0 should raise RuntimeError")


def test_runtime_tracker_preserves_mcip_fields():
    device = torch.device("cpu")
    hidden_dim = 256
    num_tracks = 5
    tracker = RuntimeTrackerBase(score_thresh=0.4, filter_score_thresh=0.2, miss_tolerance=3, maximum_quantity=10)
    track_instances = _tracks(num_tracks, hidden_dim, 7, device)
    track_instances.disappear_time = torch.zeros((num_tracks,), dtype=torch.long, device=device)
    updated = tracker.update(track_instances, torch.zeros((0,), dtype=torch.long, device=device))

    for name in [
        "img_memory",
        "semantic_memory",
        "cls_conf_memory",
        "cls_entropy_memory",
        "prev_boxes",
        "box_velocity",
        "memory_age",
    ]:
        assert updated.has(name), name
        assert len(updated.get(name)) == len(updated)
        assert updated.get(name).device == updated.scores.device
        assert torch.isfinite(updated.get(name)).all(), name

    updated.remove("semantic_memory")
    ensure_mcip_track_fields(updated, hidden_dim, device=updated.scores.device, dtype=updated.query_tgt.dtype)
    assert updated.has("semantic_memory")
    assert tuple(updated.semantic_memory.shape) == (len(updated), hidden_dim)


def test_factory_preserves_baseline_cip():
    hidden_dim = 32
    updater = build_updater(
        _make_args(mcip_enable=False),
        "CIP",
        dim_in=hidden_dim,
        hidden_dim=hidden_dim,
        dim_out=hidden_dim * 2,
    )
    assert type(updater) is Category_Information_Propagator


def main():
    test_zero_init_mcip_matches_baseline_cip()
    test_residual_adapter_final_linear_zero_init()
    test_residual_norm_clamp()
    test_new_track_initializes_memory_without_same_forward_injection()
    test_entropy_normalization_and_reliability_clamp()
    test_motion_off_ref_pts_uses_current_boxes()
    test_mcip_updater_dummy_forward_and_debug_file()
    test_first_velocity_update()
    test_semantic_observation_topk_and_full_expectation()
    test_semantic_observation_length_mismatch()
    test_runtime_tracker_preserves_mcip_fields()
    test_factory_preserves_baseline_cip()
    print("M-CIP smoke passed")


if __name__ == "__main__":
    main()
