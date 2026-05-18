import math
import sys
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
        mcip_use_motion_ref=True,
        mcip_motion_momentum=0.7,
        mcip_motion_scale_init=0.0,
        mcip_gate_use_txt=False,
        debug_mcip=True,
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
    inst.semantic_obs = torch.randn((num_tracks, hidden_dim), device=device)
    inst.cls_conf_obs = torch.rand((num_tracks,), device=device)
    inst.cls_entropy_obs = torch.rand((num_tracks,), device=device)
    ensure_mcip_track_fields(inst, hidden_dim, device=device, dtype=torch.float32)
    return inst


def test_mcip_updater_dummy_forward():
    device = torch.device("cpu")
    hidden_dim = 256
    num_tracks = 5
    num_classes = 7
    updater = MemoryCalibratedCategoryInformationPropagator(
        _make_args(),
        dim_in=hidden_dim,
        hidden_dim=hidden_dim,
        dim_out=hidden_dim * 2,
    ).to(device)
    updater.eval()

    track_instances = _tracks(num_tracks, hidden_dim, num_classes, device)
    init_track_instances = _empty_init(num_classes, hidden_dim, device)
    with torch.no_grad():
        out = updater({
            "track_instances": track_instances,
            "init_track_instances": init_track_instances,
        })

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

    required_debug_keys = [
        "img_proj_norm_ratio",
        "sem_proj_norm_ratio",
        "total_delta_norm_ratio",
        "mcip_img_cosine",
        "motion_offset_l1_mean",
        "motion_offset_l1_max",
        "motion_offset_norm_mean",
        "motion_offset_norm_max",
        "motion_scale",
        "effective_gate_mean",
        "effective_gate_max",
        "effective_gate_min",
        "cls_conf_obs_mean",
        "cls_conf_obs_min",
        "cls_conf_obs_max",
        "cls_entropy_obs_mean",
        "cls_entropy_obs_min",
        "cls_entropy_obs_max",
        "current_reliability_mean",
        "memory_reliability_mean",
        "motion_reliability_mean",
        "motion_reliability_max",
        "active_track_count",
    ]
    assert isinstance(updater.mcip_debug_stats, dict)
    for key in required_debug_keys:
        assert key in updater.mcip_debug_stats, key
        value = updater.mcip_debug_stats[key]
        assert isinstance(value, (float, int)), (key, type(value))
        assert math.isfinite(float(value)), (key, value)

    assert updater.mcip_debug_stats["active_track_count"] > 0
    assert -1.0 <= updater.mcip_debug_stats["mcip_img_cosine"] <= 1.0
    for key in [
        "effective_gate_mean",
        "cls_conf_obs_mean",
        "cls_entropy_obs_mean",
        "motion_reliability_mean",
    ]:
        assert 0.0 <= updater.mcip_debug_stats[key] <= 1.0, (key, updater.mcip_debug_stats[key])


def test_motion_scale_gradient():
    device = torch.device("cpu")
    hidden_dim = 32
    updater = MemoryCalibratedCategoryInformationPropagator(
        _make_args(),
        dim_in=hidden_dim,
        hidden_dim=hidden_dim,
        dim_out=hidden_dim * 2,
    ).to(device)
    updater.train()

    track_instances = _tracks(2, hidden_dim, 5, device)
    track_instances.memory_age = torch.ones((2,), device=device)
    track_instances.prev_boxes = (track_instances.pred_boxes - 0.05).clamp(1e-3, 1 - 1e-3)
    track_instances.box_velocity = torch.full((2, 4), 0.03, device=device)
    track_instances.cls_conf_obs = torch.ones((2,), device=device)
    track_instances.cls_entropy_obs = torch.zeros((2,), device=device)
    track_instances.cls_conf_memory = torch.ones((2,), device=device)
    track_instances.cls_entropy_memory = torch.zeros((2,), device=device)

    out = updater({
        "track_instances": track_instances,
        "init_track_instances": _empty_init(5, hidden_dim, device),
    })
    loss = out.ref_pts.sum()
    loss.backward()

    assert updater.motion_scale.grad is not None
    assert torch.isfinite(updater.motion_scale.grad).all()
    grad_abs_sum = updater.motion_scale.grad.abs().sum().item()
    assert grad_abs_sum > 0.0, f"motion_scale gradient is zero: {grad_abs_sum}"
    print(f"motion_scale_grad_abs_sum={grad_abs_sum:.12g}")


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
        first_out = updater({
            "track_instances": first_track,
            "init_track_instances": _empty_init(5, hidden_dim, device),
        })
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
        next_out = updater({
            "track_instances": next_track,
            "init_track_instances": _empty_init(5, hidden_dim, device),
        })
    assert torch.allclose(next_out.box_velocity, expected_velocity, atol=1e-6)


def _call_semantic_helper(track_instances, text_feat, debug_mcip=True):
    holder = SimpleNamespace(
        mcip_enable=True,
        mcip_detach_memory=True,
        debug_mcip=debug_mcip,
        mcip_debug_stats={},
    )
    OVTR._mcip_attach_semantic_observations(holder, {"text_feat": text_feat}, track_instances)
    return holder


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
    test_mcip_updater_dummy_forward()
    test_motion_scale_gradient()
    test_first_velocity_update()
    test_semantic_observation_length_mismatch()
    test_runtime_tracker_preserves_mcip_fields()
    test_factory_preserves_baseline_cip()
    print("M-CIP smoke passed")


if __name__ == "__main__":
    main()
