import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectron2.structures import Instances  # noqa: E402
from models.ovtr import RuntimeTrackerBase  # noqa: E402
from models.updater import (  # noqa: E402
    MemoryCalibratedCategoryInformationPropagator,
    ensure_mcip_track_fields,
)


def _make_args():
    return SimpleNamespace(
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

    updated.remove("semantic_memory")
    ensure_mcip_track_fields(updated, hidden_dim, device=updated.scores.device, dtype=updated.query_tgt.dtype)
    assert updated.has("semantic_memory")
    assert tuple(updated.semantic_memory.shape) == (len(updated), hidden_dim)


def main():
    test_mcip_updater_dummy_forward()
    test_runtime_tracker_preserves_mcip_fields()
    print("M-CIP smoke passed")


if __name__ == "__main__":
    main()
