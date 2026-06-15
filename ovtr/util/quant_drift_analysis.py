import copy
import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from detectron2.structures import Instances

from core import eval_mot


TETA_FIELDS = [
    "TETA",
    "LocA",
    "AssocA",
    "ClsA",
    "LocRe",
    "LocPr",
    "AssocRe",
    "AssocPr",
    "ClsRe",
    "ClsPr",
]

FEEDBACK_INPUT_FIELDS = (
    "ref_pts",
    "query_tgt",
    "query_pos",
    "obj_idxes",
    "matched_gt_idxes",
    "iou",
    "scores",
    "cls_idxes",
    "disappear_time",
)

POSTPROCESS_ONLY_FIELDS = ("boxes", "labels", "pred_logits", "pred_boxes")

TRACK_METRIC_COLUMNS = [
    "run_mode",
    "file_path",
    "frame_id",
    "sequence_key",
    "obj_id",
    "track_age",
    "query_tgt_cosine_distance",
    "query_pos_cosine_distance",
    "ref_pts_l1_distance",
    "pred_box_iou",
    "score_fp32",
    "score_quant",
    "class_fp32",
    "class_quant",
    "disappear_time_fp32",
    "disappear_time_quant",
    "alternate_obj_id",
    "alternate_iou",
    "divergence_type",
]

FRAME_METRIC_COLUMNS = [
    "run_mode",
    "file_path",
    "frame_id",
    "active_fp32",
    "active_quant",
    "matched_tracks",
    "missing_tracks",
    "mean_query_tgt_cosine_distance",
    "mean_pred_box_iou",
]

QUERY_STATE_IO_COLUMNS = [
    "run_mode",
    "file_path",
    "frame_id",
    "sequence_key",
    "obj_id",
    "track_age",
    "output_present",
    "input_output_query_tgt_cosine_distance",
    "input_output_query_pos_cosine_distance",
    "input_output_ref_pts_l1_distance",
]

RECURRENT_QUERY_DRIFT_COLUMNS = [
    "run_mode",
    "file_path",
    "frame_id",
    "sequence_key",
    "obj_id",
    "track_age",
    "input_present",
    "query_tgt_cosine_distance",
    "query_pos_cosine_distance",
    "ref_pts_l1_distance",
    "query_tgt_distance_delta_from_prev",
    "missing_recurrent_input",
]

ACCUMULATION_GAP_COLUMNS = [
    "file_path",
    "frame_id",
    "sequence_key",
    "obj_id",
    "track_age",
    "free_query_tgt_cosine_distance",
    "one_step_query_tgt_cosine_distance",
    "query_tgt_distance_gap",
    "free_query_pos_cosine_distance",
    "one_step_query_pos_cosine_distance",
    "query_pos_distance_gap",
    "free_ref_pts_l1_distance",
    "one_step_ref_pts_l1_distance",
    "ref_pts_l1_gap",
    "free_pred_box_iou",
    "one_step_pred_box_iou",
    "pred_box_iou_gap",
    "free_divergence_type",
    "one_step_divergence_type",
]

QUANT_BOUNDARY_ERROR_COLUMNS = [
    "run_mode",
    "file_path",
    "frame_id",
    "module_name",
    "tensor_role",
    "shape",
    "numel",
    "mean_abs_error",
    "max_abs_error",
    "mse",
    "relative_l2_error",
    "cosine_distance",
]

ALIGNMENT_GROUP_COLUMNS = [
    "sample_index",
    "image_id",
    "filename",
    "class_id",
    "class_name",
    "model",
    "num_regions",
    "region_text_mae",
    "region_region_mae",
    "region_region_pearson",
    "mean_confidence",
    "mean_region_text",
]

ALIGNMENT_SUMMARY_COLUMNS = [
    "model",
    "aggregation",
    "num_groups",
    "total_regions",
    "region_text_mae",
    "region_region_mae",
    "region_region_pearson",
    "mean_confidence",
    "mean_region_text",
]

ALIGNMENT_CLASS_SUMMARY_COLUMNS = [
    "rank",
    "class_id",
    "class_name",
    "instance_count",
    "image_count",
    "first_index",
    "processed_groups",
    "positive_groups",
    "skipped_no_match",
    "model",
    "aggregation",
    "num_groups",
    "total_regions",
    "region_text_mae",
    "region_region_mae",
    "region_region_pearson",
    "mean_confidence",
    "mean_region_text",
]

ALIGNMENT_MODEL_MACRO_COLUMNS = [
    "model",
    "num_classes",
    "total_groups",
    "total_regions",
    "region_text_mae",
    "region_region_mae",
    "region_region_pearson",
    "mean_confidence",
    "mean_region_text",
]

ALIGNMENT_SELECTED_CLASS_COLUMNS = [
    "rank",
    "class_id",
    "class_name",
    "instance_count",
    "image_count",
    "first_index",
]

CLASS_SCORE_COLUMNS = [
    "model",
    "class_id",
    "class_name",
    "mean_cosine_score",
    "is_target",
    "is_background",
    "mean_observed_confidence",
    "mean_target_prob_fg_softmax",
    "mean_target_prob_all_softmax",
    "mean_target_rank",
    "mean_target_rank_normalized",
]

EMPTY_TRACK_RESULT = np.zeros((0, 5), dtype=np.float32)
EMPTY_SCORED_TRACK_RESULT = np.zeros((0, 6), dtype=np.float32)


@dataclass
class TrackState:
    obj_id: int
    track_age: int
    query_tgt: Optional[torch.Tensor]
    query_pos: Optional[torch.Tensor]
    ref_pts: Optional[torch.Tensor]
    box: Optional[torch.Tensor]
    score: float
    cls: int
    disappear_time: int


@dataclass
class FrameTrace:
    file_path: str
    frame_id: int
    sequence_key: str
    score_threshold: float
    max_obj_id: int
    active: Dict[int, TrackState]
    feedback_active: Optional[Instances]


@dataclass
class ReferenceTrace:
    frames: Dict[Tuple[str, int], FrameTrace] = field(default_factory=dict)

    def get(self, file_path: str, frame_id: int) -> Optional[FrameTrace]:
        return self.frames.get((file_path, int(frame_id)))


@dataclass
class EmbeddingFrameTrace:
    run_mode: str
    file_path: str
    frame_id: int
    sequence_key: str
    score_threshold: float
    query_indices: torch.Tensor
    obj_ids: torch.Tensor
    cls_idxes: torch.Tensor
    scores: torch.Tensor
    disappear_time: torch.Tensor
    boxes: torch.Tensor
    pred_embed: torch.Tensor


@dataclass
class RunResult:
    run_mode: str
    track_results: List[List[np.ndarray]]
    processed_frames: int
    total_detect_time: float
    trace: Optional[ReferenceTrace] = None
    track_metric_rows: List[dict] = field(default_factory=list)
    frame_metric_rows: List[dict] = field(default_factory=list)
    divergence_rows: List[dict] = field(default_factory=list)
    state_io_rows: List[dict] = field(default_factory=list)
    recurrent_query_rows: List[dict] = field(default_factory=list)
    quant_boundary_rows: List[dict] = field(default_factory=list)
    embedding_frames: Dict[Tuple[str, int], EmbeddingFrameTrace] = field(default_factory=dict)
    embedding_class_anchors: Optional[torch.Tensor] = None

    @property
    def avg_latency_ms(self) -> float:
        if self.processed_frames <= 0:
            return float("nan")
        return (self.total_detect_time / self.processed_frames) * 1000.0

    @property
    def fps(self) -> float:
        if self.total_detect_time <= 0:
            return float("nan")
        return self.processed_frames / self.total_detect_time


def sequence_key_from_file_path(file_path: str) -> str:
    return str(Path(file_path).parent)


def clone_tensor(value: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    out = value.detach().clone()
    if device is not None:
        out = out.to(device)
    return out


def clone_field(value, device: Optional[torch.device] = None):
    if isinstance(value, torch.Tensor):
        return clone_tensor(value, device=device)
    if isinstance(value, list):
        return copy.deepcopy(value)
    if hasattr(value, "to"):
        out = value.to(device) if device is not None else value
        return copy.deepcopy(out)
    return copy.deepcopy(value)


def clone_instances(
    instances: Optional[Instances],
    *,
    fields: Optional[Iterable[str]] = None,
    device: Optional[torch.device] = None,
) -> Optional[Instances]:
    if instances is None:
        return None
    wanted = None if fields is None else set(fields)
    out = Instances(instances.image_size)
    for name, value in instances.get_fields().items():
        if wanted is not None and name not in wanted:
            continue
        out.set(name, clone_field(value, device=device))
    return out


def strip_feedback_only_fields(instances: Optional[Instances]) -> Optional[Instances]:
    if instances is None:
        return None
    for name in POSTPROCESS_ONLY_FIELDS:
        if instances.has(name):
            instances.remove(name)
    return instances


def active_track_instances(instances: Optional[Instances]) -> Optional[Instances]:
    if instances is None or not instances.has("obj_idxes"):
        return None
    if len(instances) == 0:
        return instances
    active_mask = instances.obj_idxes >= 0
    if active_mask.sum().item() == 0:
        return None
    return instances[active_mask]


def snapshot_feedback_active(instances: Optional[Instances]) -> Optional[Instances]:
    active = active_track_instances(instances)
    if active is None:
        return None
    fields = [name for name in FEEDBACK_INPUT_FIELDS if active.has(name)]
    return clone_instances(active, fields=fields, device=torch.device("cpu"))


def build_teacher_forced_input(model, fp32_frame: Optional[FrameTrace]) -> Optional[Instances]:
    if fp32_frame is None or fp32_frame.feedback_active is None:
        return None

    device = model.transformer.level_embed.device
    init_tracks = model._generate_empty_tracks()
    active = clone_instances(fp32_frame.feedback_active, fields=FEEDBACK_INPUT_FIELDS, device=device)

    out = Instances(init_tracks.image_size)
    for name in FEEDBACK_INPUT_FIELDS:
        if not init_tracks.has(name):
            continue
        init_value = init_tracks.get(name)
        if active is not None and active.has(name) and len(active) > 0:
            out.set(name, torch.cat([init_value, active.get(name)], dim=0))
        else:
            out.set(name, init_value)
    return out


def _get_tensor_field(instances: Instances, name: str, index: int) -> Optional[torch.Tensor]:
    if not instances.has(name):
        return None
    value = instances.get(name)[index]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _get_int_field(instances: Instances, name: str, index: int, default: int = -1) -> int:
    if not instances.has(name):
        return default
    value = instances.get(name)[index]
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def _get_float_field(instances: Instances, name: str, index: int, default: float = float("nan")) -> float:
    if not instances.has(name):
        return default
    value = instances.get(name)[index]
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def snapshot_frame_state(
    instances: Optional[Instances],
    *,
    file_path: str,
    frame_id: int,
    score_threshold: float,
    track_ages: Dict[Tuple[str, int], int],
    update_ages: bool,
    max_obj_id: int,
) -> FrameTrace:
    sequence_key = sequence_key_from_file_path(file_path)
    active: Dict[int, TrackState] = {}
    feedback_active = snapshot_feedback_active(instances)

    if instances is not None and instances.has("obj_idxes"):
        cpu_instances = instances.to(torch.device("cpu"))
        for index, obj_idx in enumerate(cpu_instances.obj_idxes):
            obj_id = int(obj_idx.item())
            if obj_id < 0:
                continue

            age_key = (sequence_key, obj_id)
            if update_ages:
                track_ages[age_key] = track_ages.get(age_key, 0) + 1
            track_age = track_ages.get(age_key, 0)

            active[obj_id] = TrackState(
                obj_id=obj_id,
                track_age=track_age,
                query_tgt=_get_tensor_field(cpu_instances, "query_tgt", index),
                query_pos=_get_tensor_field(cpu_instances, "query_pos", index),
                ref_pts=_get_tensor_field(cpu_instances, "ref_pts", index),
                box=_get_tensor_field(cpu_instances, "boxes", index),
                score=_get_float_field(cpu_instances, "scores", index),
                cls=_get_int_field(cpu_instances, "cls_idxes", index),
                disappear_time=_get_int_field(cpu_instances, "disappear_time", index, default=0),
            )

    return FrameTrace(
        file_path=file_path,
        frame_id=int(frame_id),
        sequence_key=sequence_key,
        score_threshold=float(score_threshold),
        max_obj_id=int(max_obj_id),
        active=active,
        feedback_active=feedback_active,
    )


def filter_dt_by_score(dt_instances: Instances, score_threshold: float) -> Instances:
    keep = (dt_instances.scores > score_threshold) & (dt_instances.disappear_time == 0)
    return dt_instances[keep]


def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
    wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
    areas = wh[:, 0] * wh[:, 1]
    keep = areas > area_threshold
    return dt_instances[keep]


def track_results_from_instances(dt_instances: Instances, num_classes: int) -> List[np.ndarray]:
    empty = [EMPTY_TRACK_RESULT for _ in range(num_classes)]
    if len(dt_instances) == 0 or not dt_instances.has("boxes"):
        return empty

    boxes = dt_instances.boxes.detach().cpu().numpy().astype(np.float32)
    labels = dt_instances.cls_idxes.detach().cpu().numpy().astype(np.int64)
    obj_ids = dt_instances.obj_idxes.detach().cpu().numpy().astype(np.float32)

    out = []
    for cls_idx in range(num_classes):
        keep = labels == cls_idx
        if not keep.any():
            out.append(EMPTY_TRACK_RESULT)
            continue
        out.append(np.concatenate([obj_ids[keep, None], boxes[keep]], axis=1).astype(np.float32))
    return out


def make_embedding_frame_trace(
    *,
    run_mode: str,
    file_path: str,
    frame_id: int,
    score_threshold: float,
    raw_trace: Optional[dict],
) -> Optional[EmbeddingFrameTrace]:
    if raw_trace is None:
        return None
    required = ("query_indices", "obj_ids", "cls_idxes", "scores", "disappear_time", "boxes", "pred_embed")
    if any(name not in raw_trace for name in required):
        return None
    return EmbeddingFrameTrace(
        run_mode=run_mode,
        file_path=file_path,
        frame_id=int(frame_id),
        sequence_key=sequence_key_from_file_path(file_path),
        score_threshold=float(score_threshold),
        query_indices=raw_trace["query_indices"].detach().cpu().long(),
        obj_ids=raw_trace["obj_ids"].detach().cpu().long(),
        cls_idxes=raw_trace["cls_idxes"].detach().cpu().long(),
        scores=raw_trace["scores"].detach().cpu().float(),
        disappear_time=raw_trace["disappear_time"].detach().cpu().long(),
        boxes=raw_trace["boxes"].detach().cpu().float(),
        pred_embed=raw_trace["pred_embed"].detach().cpu().float(),
    )


def track_results_with_dummy_scores(track_results: Sequence[List[np.ndarray]]) -> List[List[np.ndarray]]:
    scored = []
    for frame_result in track_results:
        scored_frame = []
        for cls_result in frame_result:
            if cls_result.shape[0] == 0:
                scored_frame.append(EMPTY_SCORED_TRACK_RESULT)
            elif cls_result.shape[1] == 5:
                score = np.ones((cls_result.shape[0], 1), dtype=cls_result.dtype)
                scored_frame.append(np.concatenate([cls_result, score], axis=1).astype(np.float32))
            else:
                scored_frame.append(cls_result.astype(np.float32))
        scored.append(scored_frame)
    return scored


def set_tracking_thresholds(model, file_path: str, args) -> Tuple[float, float]:
    dataset_list = ["YFCC100M", "HACS", "BDD", "ArgoVerse", "AVA", "LaSOT", "Charades"]
    parts = file_path.split("/")
    dataset_name = parts[1] if len(parts) > 1 else dataset_list[0]
    if dataset_name in dataset_list:
        index = dataset_list.index(dataset_name)
    else:
        index = 0

    score_thresh = args.score_thresh[index]
    filter_score_thresh = args.filter_score_thresh[index]
    miss_tolerance = args.miss_tolerance[index]
    ious_thresh = args.ious_thresh[index]

    model.track_base.score_thresh = score_thresh
    model.track_base.filter_score_thresh = filter_score_thresh
    model.track_base.miss_tolerance = miss_tolerance
    model.track_base.maximum_quantity = args.maximum_quantity
    model.transformer.decoder.isol_ratio = 5
    model.ious_thresh = ious_thresh
    return float(score_thresh), float(filter_score_thresh)


def box_iou_xyxy(box_a: Optional[torch.Tensor], box_b: Optional[torch.Tensor]) -> float:
    if box_a is None or box_b is None:
        return float("nan")
    a = box_a.float()
    b = box_b.float()
    x1 = torch.maximum(a[0], b[0])
    y1 = torch.maximum(a[1], b[1])
    x2 = torch.minimum(a[2], b[2])
    y2 = torch.minimum(a[3], b[3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_a = torch.clamp(a[2] - a[0], min=0) * torch.clamp(a[3] - a[1], min=0)
    area_b = torch.clamp(b[2] - b[0], min=0) * torch.clamp(b[3] - b[1], min=0)
    denom = area_a + area_b - inter
    if denom.item() <= 0:
        return float("nan")
    return float((inter / denom).item())


def cosine_distance(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    if a is None or b is None or a.numel() == 0 or b.numel() == 0:
        return float("nan")
    return float((1.0 - F.cosine_similarity(a.flatten(), b.flatten(), dim=0)).item())


def l1_distance(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> float:
    if a is None or b is None:
        return float("nan")
    return float(torch.mean(torch.abs(a.float() - b.float())).item())


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _numeric_gap(left, right) -> float:
    left_value = _safe_float(left)
    right_value = _safe_float(right)
    if math.isnan(left_value) or math.isnan(right_value):
        return float("nan")
    return left_value - right_value


def compare_query_state_io(
    *,
    run_mode: str,
    file_path: str,
    frame_id: int,
    input_instances: Optional[Instances],
    output_frame: FrameTrace,
    reference_frame: Optional[FrameTrace] = None,
) -> List[dict]:
    if input_instances is None or not input_instances.has("obj_idxes"):
        return []

    sequence_key = sequence_key_from_file_path(file_path)
    cpu_instances = input_instances.to(torch.device("cpu"))
    rows = []
    for index, obj_idx in enumerate(cpu_instances.obj_idxes):
        obj_id = int(obj_idx.item())
        if obj_id < 0:
            continue

        output_state = output_frame.active.get(obj_id)
        reference_state = reference_frame.active.get(obj_id) if reference_frame is not None else None
        track_age = float("nan")
        if reference_state is not None:
            track_age = reference_state.track_age
        elif output_state is not None:
            track_age = output_state.track_age
        rows.append(
            {
                "run_mode": run_mode,
                "file_path": file_path,
                "frame_id": int(frame_id),
                "sequence_key": sequence_key,
                "obj_id": obj_id,
                "track_age": track_age,
                "output_present": output_state is not None,
                "input_output_query_tgt_cosine_distance": cosine_distance(
                    _get_tensor_field(cpu_instances, "query_tgt", index),
                    output_state.query_tgt if output_state is not None else None,
                ),
                "input_output_query_pos_cosine_distance": cosine_distance(
                    _get_tensor_field(cpu_instances, "query_pos", index),
                    output_state.query_pos if output_state is not None else None,
                ),
                "input_output_ref_pts_l1_distance": l1_distance(
                    _get_tensor_field(cpu_instances, "ref_pts", index),
                    output_state.ref_pts if output_state is not None else None,
                ),
            }
        )
    return rows


def _index_instances_by_obj_id(instances: Optional[Instances]) -> Tuple[Optional[Instances], Dict[int, int]]:
    if instances is None or not instances.has("obj_idxes"):
        return None, {}
    cpu_instances = instances.to(torch.device("cpu"))
    by_obj_id = {}
    for index, obj_idx in enumerate(cpu_instances.obj_idxes):
        obj_id = int(obj_idx.item())
        if obj_id >= 0:
            by_obj_id[obj_id] = index
    return cpu_instances, by_obj_id


def compare_recurrent_query_to_reference(
    *,
    run_mode: str,
    file_path: str,
    frame_id: int,
    input_instances: Optional[Instances],
    fp32_frame: Optional[FrameTrace],
    previous_query_tgt_distances: Dict[Tuple[str, int], float],
) -> List[dict]:
    if fp32_frame is None or fp32_frame.feedback_active is None:
        return []

    sequence_key = sequence_key_from_file_path(file_path)
    input_cpu, input_by_obj_id = _index_instances_by_obj_id(input_instances)
    reference_cpu, reference_by_obj_id = _index_instances_by_obj_id(fp32_frame.feedback_active)
    if reference_cpu is None:
        return []

    rows = []
    for obj_id, reference_index in reference_by_obj_id.items():
        input_index = input_by_obj_id.get(obj_id)
        reference_state = fp32_frame.active.get(obj_id)
        track_age = reference_state.track_age if reference_state is not None else float("nan")
        input_present = input_cpu is not None and input_index is not None

        if input_present:
            query_tgt_distance = cosine_distance(
                _get_tensor_field(input_cpu, "query_tgt", input_index),
                _get_tensor_field(reference_cpu, "query_tgt", reference_index),
            )
            query_pos_distance = cosine_distance(
                _get_tensor_field(input_cpu, "query_pos", input_index),
                _get_tensor_field(reference_cpu, "query_pos", reference_index),
            )
            ref_pts_distance = l1_distance(
                _get_tensor_field(input_cpu, "ref_pts", input_index),
                _get_tensor_field(reference_cpu, "ref_pts", reference_index),
            )
        else:
            query_tgt_distance = float("nan")
            query_pos_distance = float("nan")
            ref_pts_distance = float("nan")

        prev_key = (sequence_key, obj_id)
        previous_distance = previous_query_tgt_distances.get(prev_key, float("nan"))
        query_tgt_delta = _numeric_gap(query_tgt_distance, previous_distance)
        if not math.isnan(_safe_float(query_tgt_distance)):
            previous_query_tgt_distances[prev_key] = float(query_tgt_distance)

        rows.append(
            {
                "run_mode": run_mode,
                "file_path": file_path,
                "frame_id": int(frame_id),
                "sequence_key": sequence_key,
                "obj_id": obj_id,
                "track_age": track_age,
                "input_present": bool(input_present),
                "query_tgt_cosine_distance": query_tgt_distance,
                "query_pos_cosine_distance": query_pos_distance,
                "ref_pts_l1_distance": ref_pts_distance,
                "query_tgt_distance_delta_from_prev": query_tgt_delta,
                "missing_recurrent_input": not input_present,
            }
        )
    return rows


def build_recurrent_query_age_summary(rows: Sequence[dict]) -> List[dict]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    if df.empty or "track_age" not in df or "query_tgt_cosine_distance" not in df:
        return []
    grouped = (
        df.dropna(subset=["track_age", "query_tgt_cosine_distance"])
        .groupby(["run_mode", "track_age"], dropna=True)["query_tgt_cosine_distance"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    return grouped.to_dict("records")


def build_accumulation_gap_rows(free_rows: Sequence[dict], one_step_rows: Sequence[dict]) -> List[dict]:
    one_step_by_key = {
        (row.get("sequence_key"), int(row.get("frame_id")), int(row.get("obj_id"))): row
        for row in one_step_rows
        if row.get("sequence_key") is not None and row.get("frame_id") is not None and row.get("obj_id") is not None
    }
    gap_rows = []
    for free_row in free_rows:
        if free_row.get("sequence_key") is None or free_row.get("frame_id") is None or free_row.get("obj_id") is None:
            continue
        key = (free_row.get("sequence_key"), int(free_row.get("frame_id")), int(free_row.get("obj_id")))
        one_step_row = one_step_by_key.get(key)
        if one_step_row is None:
            continue

        gap_rows.append(
            {
                "file_path": free_row.get("file_path", one_step_row.get("file_path")),
                "frame_id": int(free_row.get("frame_id")),
                "sequence_key": free_row.get("sequence_key"),
                "obj_id": int(free_row.get("obj_id")),
                "track_age": free_row.get("track_age", one_step_row.get("track_age")),
                "free_query_tgt_cosine_distance": free_row.get("query_tgt_cosine_distance"),
                "one_step_query_tgt_cosine_distance": one_step_row.get("query_tgt_cosine_distance"),
                "query_tgt_distance_gap": _numeric_gap(
                    free_row.get("query_tgt_cosine_distance"),
                    one_step_row.get("query_tgt_cosine_distance"),
                ),
                "free_query_pos_cosine_distance": free_row.get("query_pos_cosine_distance"),
                "one_step_query_pos_cosine_distance": one_step_row.get("query_pos_cosine_distance"),
                "query_pos_distance_gap": _numeric_gap(
                    free_row.get("query_pos_cosine_distance"),
                    one_step_row.get("query_pos_cosine_distance"),
                ),
                "free_ref_pts_l1_distance": free_row.get("ref_pts_l1_distance"),
                "one_step_ref_pts_l1_distance": one_step_row.get("ref_pts_l1_distance"),
                "ref_pts_l1_gap": _numeric_gap(
                    free_row.get("ref_pts_l1_distance"),
                    one_step_row.get("ref_pts_l1_distance"),
                ),
                "free_pred_box_iou": free_row.get("pred_box_iou"),
                "one_step_pred_box_iou": one_step_row.get("pred_box_iou"),
                "pred_box_iou_gap": _numeric_gap(
                    free_row.get("pred_box_iou"),
                    one_step_row.get("pred_box_iou"),
                ),
                "free_divergence_type": free_row.get("divergence_type", ""),
                "one_step_divergence_type": one_step_row.get("divergence_type", ""),
            }
        )
    return gap_rows


def _best_alternate_iou(fp_state: TrackState, quant_frame: FrameTrace) -> Tuple[float, Optional[int]]:
    best_iou = float("nan")
    best_obj_id = None
    for obj_id, quant_state in quant_frame.active.items():
        if fp_state.cls != -1 and quant_state.cls != fp_state.cls:
            continue
        cur_iou = box_iou_xyxy(fp_state.box, quant_state.box)
        if math.isnan(cur_iou):
            continue
        if best_obj_id is None or cur_iou > best_iou:
            best_iou = cur_iou
            best_obj_id = obj_id
    return best_iou, best_obj_id


def compare_frame_to_reference(
    *,
    run_mode: str,
    fp32_frame: Optional[FrameTrace],
    quant_frame: FrameTrace,
    iou_divergence_thresh: float,
    first_divergences: Dict[Tuple[str, str, int], dict],
) -> Tuple[List[dict], dict, List[dict]]:
    if fp32_frame is None:
        return [], {
            "run_mode": run_mode,
            "file_path": quant_frame.file_path,
            "frame_id": quant_frame.frame_id,
            "active_fp32": 0,
            "active_quant": len(quant_frame.active),
            "matched_tracks": 0,
            "missing_tracks": 0,
            "mean_query_tgt_cosine_distance": float("nan"),
            "mean_pred_box_iou": float("nan"),
        }, []

    rows = []
    new_divergences = []
    query_distances = []
    box_ious = []
    matched = 0
    missing = 0

    for obj_id, fp_state in fp32_frame.active.items():
        quant_state = quant_frame.active.get(obj_id)
        divergence_type = ""
        alternate_obj_id = None
        alternate_iou = float("nan")

        if quant_state is None:
            missing += 1
            alternate_iou, alternate_obj_id = _best_alternate_iou(fp_state, quant_frame)
            divergence_type = "id_switch" if not math.isnan(alternate_iou) and alternate_iou >= iou_divergence_thresh else "missing"
            query_distance = float("nan")
            query_pos_distance = float("nan")
            ref_pts_l1 = float("nan")
            box_iou = alternate_iou
            score_quant = float("nan")
            class_quant = -1
            disappear_time_quant = -1
        else:
            matched += 1
            query_distance = cosine_distance(fp_state.query_tgt, quant_state.query_tgt)
            query_pos_distance = cosine_distance(fp_state.query_pos, quant_state.query_pos)
            ref_pts_l1 = l1_distance(fp_state.ref_pts, quant_state.ref_pts)
            box_iou = box_iou_xyxy(fp_state.box, quant_state.box)
            score_quant = quant_state.score
            class_quant = quant_state.cls
            disappear_time_quant = quant_state.disappear_time

            if score_quant < fp32_frame.score_threshold:
                divergence_type = "threshold_drop"
            elif fp_state.cls != -1 and class_quant != -1 and fp_state.cls != class_quant:
                divergence_type = "class_mismatch"
            elif not math.isnan(box_iou) and box_iou < iou_divergence_thresh:
                divergence_type = "low_iou"

            if not math.isnan(query_distance):
                query_distances.append(query_distance)
            if not math.isnan(box_iou):
                box_ious.append(box_iou)

        row = {
            "run_mode": run_mode,
            "file_path": quant_frame.file_path,
            "frame_id": quant_frame.frame_id,
            "sequence_key": quant_frame.sequence_key,
            "obj_id": obj_id,
            "track_age": fp_state.track_age,
            "query_tgt_cosine_distance": query_distance,
            "query_pos_cosine_distance": query_pos_distance,
            "ref_pts_l1_distance": ref_pts_l1,
            "pred_box_iou": box_iou,
            "score_fp32": fp_state.score,
            "score_quant": score_quant,
            "class_fp32": fp_state.cls,
            "class_quant": class_quant,
            "disappear_time_fp32": fp_state.disappear_time,
            "disappear_time_quant": disappear_time_quant,
            "alternate_obj_id": alternate_obj_id,
            "alternate_iou": alternate_iou,
            "divergence_type": divergence_type,
        }
        rows.append(row)

        if divergence_type:
            div_key = (run_mode, quant_frame.sequence_key, obj_id)
            if div_key not in first_divergences:
                first_divergences[div_key] = row
                new_divergences.append(row)

    frame_row = {
        "run_mode": run_mode,
        "file_path": quant_frame.file_path,
        "frame_id": quant_frame.frame_id,
        "active_fp32": len(fp32_frame.active),
        "active_quant": len(quant_frame.active),
        "matched_tracks": matched,
        "missing_tracks": missing,
        "mean_query_tgt_cosine_distance": float(np.mean(query_distances)) if query_distances else float("nan"),
        "mean_pred_box_iou": float(np.mean(box_ious)) if box_ious else float("nan"),
    }
    return rows, frame_row, new_divergences


def evaluate_teta(dataset, track_results: Sequence[List[np.ndarray]], output_dir: Path, metric: Sequence[str]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"track_results": list(track_results), "bbox_results": None}
    dataset.evaluate(outputs, metric=metric, resfile_path=str(output_dir))
    summary_path = output_dir / "OVTR" / "teta_summary_results.pth"
    if not summary_path.exists():
        return {}

    with summary_path.open("rb") as handle:
        raw = pickle.load(handle)

    combined = raw.get("COMBINED_SEQ", raw)
    if "average" in combined and "TETA" in combined["average"]:
        values = combined["average"]["TETA"].get(50) or combined["average"]["TETA"].get("50")
    else:
        values_by_class = []
        for class_result in combined.values():
            if isinstance(class_result, dict) and "TETA" in class_result:
                class_values = class_result["TETA"].get(50) or class_result["TETA"].get("50")
                if class_values is not None:
                    values_by_class.append(np.asarray(class_values, dtype=float))
        values = np.mean(np.stack(values_by_class), axis=0) if values_by_class else None

    if values is None:
        return {}

    values = np.asarray(values, dtype=float)
    metrics = {f"TETA50_{name}": float(values[idx]) for idx, name in enumerate(TETA_FIELDS[: len(values)])}
    metrics["TETA50_all"] = metrics.get("TETA50_TETA", float("nan"))
    metrics["LocA"] = metrics.get("TETA50_LocA", float("nan"))
    metrics["AssocA"] = metrics.get("TETA50_AssocA", float("nan"))
    metrics["ClsA"] = metrics.get("TETA50_ClsA", float("nan"))
    return metrics


def _split_by_processed_videos(items: Sequence, data_infos: Sequence[dict]) -> List[list]:
    if not data_infos:
        return []
    starts = [idx for idx, info in enumerate(data_infos) if info.get("frame_id", -1) == 0]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    starts.append(len(data_infos))
    return [list(items[starts[idx] : starts[idx + 1]]) for idx in range(len(starts) - 1)]


def evaluate_id_metrics(dataset, track_results: Sequence[List[np.ndarray]], processed_frames: int, nproc: int = 1) -> dict:
    if processed_frames <= 0:
        return {}

    data_infos = dataset.data_infos[:processed_frames]
    scored_results = track_results_with_dummy_scores(track_results[:processed_frames])
    split_results = _split_by_processed_videos(scored_results, data_infos)
    ann_infos = [dataset.get_ann_info(info) for info in data_infos]
    split_ann_infos = _split_by_processed_videos(ann_infos, data_infos)
    if not split_results or not split_ann_infos:
        return {}

    return eval_mot(
        results=split_results,
        annotations=split_ann_infos,
        logger=None,
        classes=dataset.CLASSES,
        iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_by_classes=False,
        nproc=max(1, int(nproc)),
    )


def summarize_run_metrics(
    *,
    run_result: RunResult,
    dataset,
    output_dir: Path,
    metric: Sequence[str],
    id_nproc: int,
) -> dict:
    run_dir = output_dir / "eval" / run_result.run_mode
    summary = {
        "processed_frames": run_result.processed_frames,
        "avg_latency_ms": run_result.avg_latency_ms,
        "fps": run_result.fps,
    }
    try:
        summary.update(evaluate_teta(dataset, run_result.track_results, run_dir, metric))
    except Exception as exc:  # keep drift artifacts even when metric tooling fails
        summary["TETA_error"] = str(exc)
    try:
        summary.update(evaluate_id_metrics(dataset, run_result.track_results, run_result.processed_frames, nproc=id_nproc))
    except Exception as exc:
        summary["IDF1_error"] = str(exc)
    return summary


def _metric_delta(left: dict, right: dict, keys: Sequence[str]) -> dict:
    out = {}
    for key in keys:
        if key not in left or key not in right:
            continue
        try:
            out[key] = float(left[key]) - float(right[key])
        except (TypeError, ValueError):
            continue
    return out


def write_metrics_summary(summaries: Dict[str, dict], output_dir: Path) -> dict:
    metric_keys = ["TETA50_all", "LocA", "AssocA", "ClsA", "IDF1", "MOTA", "avg_latency_ms", "fps"]
    payload = dict(summaries)
    payload["delta"] = {}
    if "fp32_free" in summaries and "quant_free" in summaries:
        payload["delta"]["quant_free_minus_fp32_free"] = _metric_delta(
            summaries["quant_free"], summaries["fp32_free"], metric_keys
        )
    if "quant_free" in summaries and "quant_teacher_forced" in summaries:
        payload["delta"]["quant_teacher_forced_minus_quant_free"] = _metric_delta(
            summaries["quant_teacher_forced"], summaries["quant_free"], metric_keys
        )
    payload["notes"] = {
        "HOTA": "not implemented in this repo",
        "teacher_forcing": "only active recurrent track queries are forced from fp32; detection queries remain from the quant model",
    }
    path = output_dir / "metrics_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
    return payload


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _save_csv(
    rows: List[dict],
    path: Path,
    columns: Optional[List[str]] = None,
    *,
    append: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if columns is not None:
        for column in columns:
            if column not in df:
                df[column] = np.nan
        df = df[columns]
    df.to_csv(path, index=False, mode="a" if append else "w", header=not append)


def write_metric_csvs(
    *,
    track_rows: List[dict],
    frame_rows: List[dict],
    divergence_rows: List[dict],
    output_dir: Path,
    append: bool = False,
) -> None:
    _save_csv(track_rows, output_dir / "track_metrics.csv", columns=TRACK_METRIC_COLUMNS, append=append)
    _save_csv(frame_rows, output_dir / "frame_metrics.csv", columns=FRAME_METRIC_COLUMNS, append=append)
    _save_csv(divergence_rows, output_dir / "divergences.csv", columns=TRACK_METRIC_COLUMNS, append=append)


def write_research_csvs(
    *,
    one_step_rows: List[dict],
    accumulation_gap_rows: List[dict],
    state_io_rows: List[dict],
    recurrent_query_rows: List[dict],
    quant_boundary_rows: List[dict],
    output_dir: Path,
    append: bool = False,
) -> None:
    _save_csv(one_step_rows, output_dir / "one_step_metrics.csv", columns=TRACK_METRIC_COLUMNS, append=append)
    _save_csv(
        accumulation_gap_rows,
        output_dir / "accumulation_gap.csv",
        columns=ACCUMULATION_GAP_COLUMNS,
        append=append,
    )
    _save_csv(
        state_io_rows,
        output_dir / "query_state_io_metrics.csv",
        columns=QUERY_STATE_IO_COLUMNS,
        append=append,
    )
    _save_csv(
        recurrent_query_rows,
        output_dir / "recurrent_query_drift.csv",
        columns=RECURRENT_QUERY_DRIFT_COLUMNS,
        append=append,
    )
    _save_csv(
        quant_boundary_rows,
        output_dir / "quant_boundary_errors.csv",
        columns=QUANT_BOUNDARY_ERROR_COLUMNS,
        append=append,
    )


def _plot_age_curve(df: pd.DataFrame, value: str, ylabel: str, title: str, output_path: Path, max_age: int) -> None:
    plt.figure(figsize=(9, 5))
    plot_df = df.dropna(subset=[value, "track_age"])
    if max_age > 0:
        plot_df = plot_df[plot_df["track_age"] <= max_age]
    if plot_df.empty:
        plt.text(0.5, 0.5, "No matched tracks", ha="center", va="center")
    else:
        for run_mode, group in plot_df.groupby("run_mode"):
            grouped = group.groupby("track_age")[value].mean().sort_index()
            plt.plot(grouped.index, grouped.values, marker="o", linewidth=1.5, label=run_mode)
        plt.legend()
    plt.xlabel("Track age")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_metric_bars(summary: dict, output_path: Path) -> None:
    run_modes = [mode for mode in ["fp32_free", "quant_free", "quant_teacher_forced"] if mode in summary]
    metric_keys = ["TETA50_all", "IDF1", "MOTA"]
    values = []
    labels = []
    for run_mode in run_modes:
        for metric in metric_keys:
            if metric in summary[run_mode]:
                labels.append(f"{run_mode}\n{metric}")
                values.append(float(summary[run_mode][metric]))

    plt.figure(figsize=(10, 5))
    if not values:
        plt.text(0.5, 0.5, "No metric summary available", ha="center", va="center")
        plt.xticks([])
    else:
        plt.bar(np.arange(len(values)), values)
        plt.xticks(np.arange(len(values)), labels, rotation=35, ha="right")
    plt.ylabel("Metric value")
    plt.title("Teacher-forced vs free-run metrics")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_quant_boundary_error(df: pd.DataFrame, output_path: Path, max_modules: int = 20) -> None:
    plt.figure(figsize=(11, 5))
    plot_df = df.dropna(subset=["module_name", "mean_abs_error"]) if not df.empty else df
    if plot_df.empty:
        plt.text(0.5, 0.5, "No quant boundary errors", ha="center", va="center")
        plt.xticks([])
    else:
        grouped = plot_df.groupby("module_name")["mean_abs_error"].mean().sort_values(ascending=False)
        grouped = grouped.head(max_modules)
        plt.bar(np.arange(len(grouped)), grouped.values)
        plt.xticks(np.arange(len(grouped)), grouped.index, rotation=45, ha="right")
    plt.ylabel("Mean absolute error")
    plt.title("Quant boundary error by module")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_plots(output_dir: Path, metrics_summary: dict, plot_max_age: int) -> None:
    track_path = output_dir / "track_metrics.csv"
    if track_path.exists():
        track_df = pd.read_csv(track_path)
    else:
        track_df = pd.DataFrame()

    if track_df.empty:
        track_df = pd.DataFrame(
            columns=["run_mode", "track_age", "query_tgt_cosine_distance", "pred_box_iou"]
        )

    _plot_age_curve(
        track_df,
        "query_tgt_cosine_distance",
        "Cosine distance",
        "query_tgt drift by track age",
        output_dir / "query_tgt_cosine_by_age.png",
        plot_max_age,
    )
    _plot_age_curve(
        track_df,
        "pred_box_iou",
        "IoU",
        "pred_box agreement by track age",
        output_dir / "pred_box_iou_by_age.png",
        plot_max_age,
    )

    recurrent_path = output_dir / "recurrent_query_drift.csv"
    if recurrent_path.exists():
        recurrent_df = pd.read_csv(recurrent_path)
    else:
        recurrent_df = pd.DataFrame()
    if recurrent_df.empty:
        recurrent_df = pd.DataFrame(columns=["run_mode", "track_age", "query_tgt_cosine_distance"])
    _plot_age_curve(
        recurrent_df,
        "query_tgt_cosine_distance",
        "Cosine distance",
        "recurrent query_tgt drift by track age",
        output_dir / "recurrent_query_tgt_by_age.png",
        plot_max_age,
    )

    boundary_path = output_dir / "quant_boundary_errors.csv"
    if boundary_path.exists():
        boundary_df = pd.read_csv(boundary_path)
    else:
        boundary_df = pd.DataFrame()
    _plot_quant_boundary_error(boundary_df, output_dir / "quant_boundary_error_by_module.png")
    _plot_metric_bars(metrics_summary, output_dir / "teacher_forced_vs_free_metrics.png")




@dataclass
class AlignmentOutput:
    embeddings: torch.Tensor
    region_text: torch.Tensor
    confidence: torch.Tensor
    relation: torch.Tensor

    @property
    def mean_confidence(self) -> float:
        if self.confidence.numel() == 0:
            return float("nan")
        return float(self.confidence.mean().item())

    @property
    def mean_region_text(self) -> float:
        if self.region_text.numel() == 0:
            return float("nan")
        return float(self.region_text.mean().item())


@dataclass
class EmbeddingGroupExample:
    sample_index: int
    image_id: object
    filename: str
    class_id: int
    class_name: str
    image: np.ndarray
    rois: torch.Tensor
    outputs: Dict[str, AlignmentOutput]

    @property
    def num_regions(self) -> int:
        return int(self.rois.shape[0])


def _class_name(class_id: int, class_names: Optional[Sequence[str]]) -> str:
    if class_names is not None and 0 <= int(class_id) < len(class_names):
        return str(class_names[int(class_id)])
    return f"class_{int(class_id)}"


def _sanitize_name(value: str) -> str:
    out = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_")[:80] or "class"


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = [
        (float(v), float(w))
        for v, w in zip(values, weights)
        if math.isfinite(float(v)) and float(w) > 0
    ]
    if not pairs:
        return float("nan")
    total_weight = sum(weight for _, weight in pairs)
    return float(sum(value * weight for value, weight in pairs) / total_weight)


def _off_diagonal_values(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square relation matrix, got {tuple(matrix.shape)}")
    if matrix.shape[0] < 2:
        return matrix.new_empty((0,))
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def pearson_correlation(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape:
        raise ValueError(
            "Pearson inputs must have the same shape: "
            f"{tuple(reference.shape)} vs {tuple(candidate.shape)}"
        )
    if reference.numel() == 0:
        return float("nan")

    reference = reference.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    if torch.allclose(reference, candidate, atol=1e-7, rtol=1e-6):
        return 1.0

    ref_centered = reference - reference.mean()
    cand_centered = candidate - candidate.mean()
    denom = torch.linalg.vector_norm(ref_centered) * torch.linalg.vector_norm(cand_centered)
    if denom.item() == 0:
        return float("nan")
    return float(torch.dot(ref_centered, cand_centered).div(denom).item())


def compute_alignment_output(
    region_embeddings: torch.Tensor,
    text_features: torch.Tensor,
    class_id: int,
    *,
    temperature: float = 0.007,
    bg_embedding: Optional[torch.Tensor] = None,
) -> AlignmentOutput:
    if region_embeddings.ndim != 2:
        raise ValueError(f"region_embeddings must be 2D, got {tuple(region_embeddings.shape)}")
    if text_features.ndim != 2:
        raise ValueError(f"text_features must be 2D, got {tuple(text_features.shape)}")
    if region_embeddings.shape[1] != text_features.shape[1]:
        raise ValueError(
            "Region embedding dimension must match text feature dimension: "
            f"{region_embeddings.shape[1]} vs {text_features.shape[1]}"
        )
    if class_id < 0 or class_id >= text_features.shape[0]:
        raise ValueError(
            f"class_id {class_id} is outside text feature range [0, {text_features.shape[0]})"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    embeddings = F.normalize(region_embeddings.float(), p=2, dim=1)
    base_text = F.normalize(
        text_features.to(device=embeddings.device, dtype=embeddings.dtype),
        p=2,
        dim=1,
    )
    class_text = base_text[class_id]
    region_text = embeddings @ class_text
    relation = embeddings @ embeddings.t()

    all_text = base_text
    if bg_embedding is not None:
        bg_embedding = bg_embedding.to(device=embeddings.device, dtype=embeddings.dtype)
        if bg_embedding.ndim == 1:
            bg_embedding = bg_embedding.unsqueeze(0)
        if bg_embedding.ndim != 2 or bg_embedding.shape[1] != embeddings.shape[1]:
            raise ValueError(
                "bg_embedding must be compatible with region embeddings: "
                f"{tuple(bg_embedding.shape)} vs {tuple(embeddings.shape)}"
            )
        all_text = torch.cat([all_text, F.normalize(bg_embedding, p=2, dim=1)], dim=0)

    logits = (embeddings @ all_text.t()) / float(temperature)
    confidence = logits.softmax(dim=1)[:, class_id]
    return AlignmentOutput(
        embeddings=embeddings.detach().cpu(),
        region_text=region_text.detach().cpu(),
        confidence=confidence.detach().cpu(),
        relation=relation.detach().cpu(),
    )


def compute_distortion_metrics(reference: AlignmentOutput, candidate: AlignmentOutput) -> Dict[str, float]:
    if reference.region_text.shape != candidate.region_text.shape:
        raise ValueError(
            "region_text shapes must match: "
            f"{tuple(reference.region_text.shape)} vs {tuple(candidate.region_text.shape)}"
        )
    if reference.relation.shape != candidate.relation.shape:
        raise ValueError(
            "relation matrix shapes must match: "
            f"{tuple(reference.relation.shape)} vs {tuple(candidate.relation.shape)}"
        )

    if reference.region_text.numel() == 0:
        region_text_mae = float("nan")
    else:
        region_text_mae = float(
            torch.mean(torch.abs(candidate.region_text - reference.region_text)).item()
        )

    ref_relation = _off_diagonal_values(reference.relation)
    cand_relation = _off_diagonal_values(candidate.relation)
    if ref_relation.numel() == 0:
        region_region_mae = float("nan")
        region_region_pearson = float("nan")
    else:
        region_region_mae = float(torch.mean(torch.abs(cand_relation - ref_relation)).item())
        region_region_pearson = pearson_correlation(ref_relation, cand_relation)

    return {
        "region_text_mae": region_text_mae,
        "region_region_mae": region_region_mae,
        "region_region_pearson": region_region_pearson,
    }


def _target_alignment(frame: EmbeddingFrameTrace, anchors: torch.Tensor, class_id: int) -> torch.Tensor:
    if frame.pred_embed.numel() == 0 or class_id < 0 or class_id >= anchors.shape[0]:
        return torch.empty((0,), dtype=torch.float32)
    emb = F.normalize(frame.pred_embed.float(), p=2, dim=1)
    anchor = F.normalize(anchors[class_id].float().view(1, -1), p=2, dim=1)
    return (emb @ anchor.t()).squeeze(1).cpu()


def _positive_group_candidates(fp32_result: RunResult) -> Tuple[Dict[int, dict], Dict[int, List[dict]]]:
    anchors = fp32_result.embedding_class_anchors
    class_stats: Dict[int, dict] = {}
    groups_by_class: Dict[int, List[dict]] = {}
    if anchors is None:
        return class_stats, groups_by_class

    for sample_index, (frame_key, frame) in enumerate(fp32_result.embedding_frames.items()):
        if frame.pred_embed.numel() == 0:
            continue
        valid = (frame.scores >= frame.score_threshold) & (frame.disappear_time == 0) & (frame.cls_idxes >= 0)
        if valid.numel() == 0 or not bool(valid.any()):
            continue
        for class_id_tensor in torch.unique(frame.cls_idxes[valid]):
            class_id = int(class_id_tensor.item())
            if class_id < 0 or class_id >= anchors.shape[0]:
                continue
            mask = valid & (frame.cls_idxes == class_id)
            indices = torch.nonzero(mask, as_tuple=False).flatten().cpu()
            if indices.numel() == 0:
                continue
            alignments = _target_alignment(frame, anchors, class_id).index_select(0, indices)
            mean_alignment = float(alignments.mean().item()) if alignments.numel() else float("nan")
            stat = class_stats.setdefault(
                class_id,
                {
                    "instance_count": 0,
                    "image_count": 0,
                    "first_index": sample_index,
                    "alignment_sum": 0.0,
                },
            )
            stat["instance_count"] += int(indices.numel())
            stat["image_count"] += 1
            stat["first_index"] = min(int(stat["first_index"]), sample_index)
            stat["alignment_sum"] += float(alignments.sum().item()) if alignments.numel() else 0.0
            groups_by_class.setdefault(class_id, []).append(
                {
                    "sample_index": sample_index,
                    "frame_key": frame_key,
                    "indices": indices,
                    "count": int(indices.numel()),
                    "mean_alignment": mean_alignment,
                }
            )
    return class_stats, groups_by_class


def _select_embedding_targets(
    class_stats: Dict[int, dict],
    class_names: Optional[Sequence[str]],
    num_classes: int,
) -> List[dict]:
    targets = []
    for class_id, stat in class_stats.items():
        instance_count = int(stat.get("instance_count", 0))
        image_count = int(stat.get("image_count", 0))
        first_index = int(stat.get("first_index", 0))
        targets.append(
            {
                "class_id": int(class_id),
                "class_name": _class_name(class_id, class_names),
                "instance_count": instance_count,
                "image_count": image_count,
                "first_index": first_index,
            }
        )
    targets.sort(
        key=lambda item: (
            -item["instance_count"],
            -item["image_count"],
            item["first_index"],
            item["class_name"],
        )
    )
    selected = targets[: max(0, int(num_classes))]
    for rank, target in enumerate(selected, start=1):
        target["rank"] = rank
    return selected


def _frame_index_by_keys(frame: Optional[EmbeddingFrameTrace]) -> Dict[Tuple[str, int], int]:
    if frame is None:
        return {}
    out: Dict[Tuple[str, int], int] = {}
    for idx in range(int(frame.query_indices.numel())):
        obj_id = int(frame.obj_ids[idx].item())
        query_idx = int(frame.query_indices[idx].item())
        if obj_id >= 0:
            out.setdefault(("obj", obj_id), idx)
        out.setdefault(("query", query_idx), idx)
    return out


def _positive_match_keys(frame: EmbeddingFrameTrace, indices: torch.Tensor) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
    keys = []
    for idx_tensor in indices:
        idx = int(idx_tensor.item())
        obj_id = int(frame.obj_ids[idx].item())
        query_idx = int(frame.query_indices[idx].item())
        query_key = ("query", query_idx)
        primary = ("obj", obj_id) if obj_id >= 0 else query_key
        keys.append((primary, query_key))
    return keys


def _lookup_match_index(by_key: Dict[Tuple[str, int], int], keys: Tuple[Tuple[str, int], Tuple[str, int]]) -> Optional[int]:
    primary, query_key = keys
    found = by_key.get(primary)
    if found is not None:
        return found
    if primary != query_key:
        return by_key.get(query_key)
    return None


def _indices_for_match_keys(
    frame: Optional[EmbeddingFrameTrace],
    keys: Sequence[Tuple[Tuple[str, int], Tuple[str, int]]],
) -> List[Optional[int]]:
    by_key = _frame_index_by_keys(frame)
    return [_lookup_match_index(by_key, key_pair) for key_pair in keys]


def _gather_rows(tensor: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if not indices:
        shape = (0,) + tuple(tensor.shape[1:])
        return torch.empty(shape, dtype=tensor.dtype)
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    return tensor.index_select(0, index_tensor).detach().cpu()


def _embedding_rois_from_boxes(boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.detach().cpu().float()
    batch = torch.zeros((boxes.shape[0], 1), dtype=boxes.dtype)
    return torch.cat([batch, boxes], dim=1)


def _load_embedding_image(image_root: Optional[str], file_path: str, boxes: torch.Tensor) -> Tuple[np.ndarray, Optional[str]]:
    candidates = []
    path = Path(file_path)
    if path.is_absolute():
        candidates.append(path)
    if image_root:
        candidates.append(Path(image_root) / file_path)
    candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            try:
                image = plt.imread(candidate)
                if image.ndim == 2:
                    image = np.repeat(image[..., None], 3, axis=2)
                if image.shape[-1] > 3:
                    image = image[..., :3]
                if image.dtype != np.uint8:
                    image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
                return image, str(candidate)
            except Exception:
                continue
    if boxes.numel() > 0:
        max_xy = boxes[:, 2:4].max(dim=0).values
        height = max(256, int(math.ceil(float(max_xy[1].item()))) + 16)
        width = max(256, int(math.ceil(float(max_xy[0].item()))) + 16)
    else:
        height, width = 800, 1333
    return np.full((height, width, 3), 245, dtype=np.uint8), None


def _heat_overlay(image: np.ndarray, rois: np.ndarray, values: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    heat = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for roi, value in zip(rois, values):
        x1, y1, x2, y2 = roi[1:5]
        x1 = int(np.clip(np.floor(x1), 0, max(width - 1, 0)))
        x2 = int(np.clip(np.ceil(x2), 0, width))
        y1 = int(np.clip(np.floor(y1), 0, max(height - 1, 0)))
        y2 = int(np.clip(np.ceil(y2), 0, height))
        if x2 <= x1 or y2 <= y1:
            continue
        heat[y1:y2, x1:x2] += float(value)
        count[y1:y2, x1:x2] += 1.0
    valid = count > 0
    heat[valid] /= count[valid]
    heat[~valid] = np.nan
    return heat


def _plot_alignment_scatter(path: Path, summary_rows: Sequence[Dict[str, object]], *, title: str) -> None:
    rows = [row for row in summary_rows if row.get("aggregation") == "macro" and str(row.get("model")) != "fp32"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=170)
    ax.axhline(0, color="0.85", linewidth=1)
    ax.axvline(0, color="0.85", linewidth=1)
    colors = {"qat": "#ff7f0e", "cr_qat": "#2ca02c", "CR-QAT": "#2ca02c", "QAT": "#ff7f0e"}
    for model_name in sorted({str(row.get("model")) for row in rows}):
        model_rows = [row for row in rows if str(row.get("model")) == model_name]
        points = []
        for row in model_rows:
            x = float(row.get("region_region_mae", float("nan")))
            y = float(row.get("region_text_mae", float("nan")))
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
        if not points:
            continue
        ax.scatter(
            [x for x, _ in points],
            [y for _, y in points],
            s=26 if len(points) > 1 else 62,
            alpha=0.6,
            label=model_name,
            color=colors.get(model_name),
        )
    ax.set_xlabel("MAE of inter-region relation vs FP32")
    ax.set_ylabel("MAE of region-text alignment vs FP32")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_example(path: Path, example: EmbeddingGroupExample, model_names: Sequence[str]) -> None:
    rois = example.rois.detach().cpu().numpy()
    outputs = [example.outputs[name] for name in model_names]
    confidences = [out.confidence.numpy() for out in outputs]
    relations = [out.relation.numpy() for out in outputs]
    heat_values = np.concatenate([vals for vals in confidences if vals.size > 0]) if confidences else np.array([])
    heat_vmax = float(np.nanmax(heat_values)) if heat_values.size and np.isfinite(heat_values).any() else 1.0
    if heat_vmax <= 0:
        heat_vmax = 1.0

    finite_rel = np.concatenate([rel.reshape(-1) for rel in relations if rel.size > 0]) if relations else np.array([])
    finite_rel = finite_rel[np.isfinite(finite_rel)]
    rel_vmin = float(np.nanmin(finite_rel)) if finite_rel.size else -1.0
    rel_vmax = float(np.nanmax(finite_rel)) if finite_rel.size else 1.0
    if math.isclose(rel_vmin, rel_vmax):
        rel_vmin = rel_vmax - 1e-3

    fig, axes = plt.subplots(2, len(model_names), figsize=(3.2 * len(model_names), 5.9), dpi=160)
    if len(model_names) == 1:
        axes = np.array(axes).reshape(2, 1)

    reference = example.outputs[model_names[0]]
    im = None
    for col, model_name in enumerate(model_names):
        output = example.outputs[model_name]
        heat = _heat_overlay(example.image, rois, output.confidence.numpy())
        ax = axes[0, col]
        ax.imshow(example.image)
        ax.imshow(heat, cmap="turbo", alpha=0.55, vmin=0.0, vmax=heat_vmax)
        for roi in rois:
            x1, y1, x2, y2 = roi[1:5]
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    max(0.0, x2 - x1),
                    max(0.0, y2 - y1),
                    fill=False,
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.85,
                )
            )
        ax.set_title(f"{model_name}\nPbar={output.mean_confidence:.3f}", fontsize=10)
        ax.axis("off")

        ax = axes[1, col]
        im = ax.imshow(output.relation.numpy(), cmap="RdBu_r", vmin=rel_vmin, vmax=rel_vmax)
        if model_name == model_names[0]:
            subtitle = "reference"
        else:
            r = compute_distortion_metrics(reference, output)["region_region_pearson"]
            subtitle = f"r={r:.3f}" if math.isfinite(float(r)) else "r=nan"
        ax.set_title(subtitle, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{example.class_name} | image_id={example.image_id} | N={example.num_regions}",
        fontsize=12,
    )
    if im is not None:
        fig.colorbar(im, ax=axes[1, :].ravel().tolist(), fraction=0.025, pad=0.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_example_npz(path: Path, example: EmbeddingGroupExample, model_names: Sequence[str]) -> None:
    payload = {
        "sample_index": np.array(example.sample_index),
        "image_id": np.array(str(example.image_id)),
        "filename": np.array(example.filename),
        "class_id": np.array(example.class_id),
        "class_name": np.array(example.class_name),
        "rois": example.rois.detach().cpu().numpy(),
        "model_names": np.array(model_names),
    }
    for model_name in model_names:
        key = _sanitize_name(model_name)
        output = example.outputs[model_name]
        payload[f"{key}__embeddings"] = output.embeddings.numpy()
        payload[f"{key}__region_text"] = output.region_text.numpy()
        payload[f"{key}__confidence"] = output.confidence.numpy()
        payload[f"{key}__relation"] = output.relation.numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)



def _normalize_anchor_matrix(anchors: torch.Tensor) -> torch.Tensor:
    if not isinstance(anchors, torch.Tensor) or anchors.ndim != 2:
        raise ValueError(f"class anchors must be a 2D tensor, got {type(anchors)}")
    anchors = anchors.detach().cpu().float()
    if anchors.shape[0] < anchors.shape[1]:
        anchors = anchors.t().contiguous()
    return F.normalize(anchors, p=2, dim=1)


def _foreground_class_count(anchors: torch.Tensor, class_names: Optional[Sequence[str]]) -> int:
    if class_names is None:
        return int(anchors.shape[0])
    return min(int(anchors.shape[0]), len(class_names))


def _score_class_name(class_id: int, class_names: Optional[Sequence[str]]) -> str:
    if class_id < 0:
        return "background"
    if class_names is not None and class_id < len(class_names):
        return str(class_names[class_id])
    return f"class_{class_id}"


def _model_color(model_name: str) -> Optional[str]:
    colors = {
        "fp32": "#1f77b4",
        "FP32": "#1f77b4",
        "qat": "#ff7f0e",
        "QAT": "#ff7f0e",
        "cr_qat": "#2ca02c",
        "CR-QAT": "#2ca02c",
        "cr-qat": "#2ca02c",
    }
    return colors.get(str(model_name))


def _compute_class_score_payload(
    *,
    class_id: int,
    class_name: str,
    image_id: object,
    num_regions: int,
    outputs: Dict[str, AlignmentOutput],
    model_names: Sequence[str],
    class_names: Optional[Sequence[str]],
    class_anchors_by_model: Dict[str, torch.Tensor],
    score_topk: int,
    score_max_bars: int,
    score_temperature: float,
) -> Dict[str, object]:
    if score_temperature <= 0:
        raise ValueError(f"score_temperature must be positive, got {score_temperature}")

    fallback_anchors = class_anchors_by_model.get("fp32")
    selected_ids = [int(class_id)]
    per_model: Dict[str, Dict[str, object]] = {}
    background_available = False
    background_index_by_model: Dict[str, Optional[int]] = {}

    for model_name in model_names:
        if model_name not in outputs:
            continue
        anchors = class_anchors_by_model.get(model_name, fallback_anchors)
        if anchors is None:
            continue
        anchors = _normalize_anchor_matrix(anchors)
        foreground_count = _foreground_class_count(anchors, class_names)
        if class_id < 0 or class_id >= foreground_count:
            continue

        embeddings = F.normalize(outputs[model_name].embeddings.float(), p=2, dim=1)
        scores = embeddings @ anchors.t()
        foreground_scores = scores[:, :foreground_count]
        mean_foreground_scores = foreground_scores.mean(dim=0).detach().cpu().numpy()
        target_scores = foreground_scores[:, class_id]
        target_rank = (foreground_scores > target_scores[:, None]).sum(dim=1).float() + 1.0
        target_prob_fg = torch.softmax(foreground_scores / float(score_temperature), dim=1)[:, class_id]
        target_prob_all = torch.softmax(scores / float(score_temperature), dim=1)[:, class_id]
        background_index = foreground_count if anchors.shape[0] > foreground_count else None
        background_score = None
        if background_index is not None:
            background_available = True
            background_score = float(scores[:, background_index].mean().item())
        background_index_by_model[model_name] = background_index

        k = min(max(int(score_topk), 0), foreground_count)
        if k > 0:
            for idx in torch.topk(foreground_scores.mean(dim=0), k=k).indices.tolist():
                idx = int(idx)
                if idx not in selected_ids:
                    selected_ids.append(idx)

        per_model[model_name] = {
            "mean_foreground_scores": mean_foreground_scores,
            "background_score": background_score,
            "mean_observed_confidence": outputs[model_name].mean_confidence,
            "mean_target_prob_fg_softmax": float(target_prob_fg.mean().item()),
            "mean_target_prob_all_softmax": float(target_prob_all.mean().item()),
            "mean_target_rank": float(target_rank.mean().item()),
            "mean_target_rank_normalized": float(target_rank.mean().item() / max(float(foreground_count), 1.0)),
            "foreground_count": foreground_count,
        }

    if not per_model:
        raise ValueError("No class score payload could be computed; missing outputs or class anchors.")

    max_bars = max(int(score_max_bars), 1)
    if len(selected_ids) > max_bars:
        rest = selected_ids[1:]
        rest.sort(
            key=lambda idx: max(
                float(payload["mean_foreground_scores"][idx])
                for payload in per_model.values()
                if idx < len(payload["mean_foreground_scores"])
            ),
            reverse=True,
        )
        selected_ids = [int(class_id)] + rest[: max_bars - 1]

    labels = []
    for idx in selected_ids:
        label = _score_class_name(idx, class_names)
        if idx == class_id:
            label = f"{label} (target)"
        labels.append(label)
    if background_available:
        labels.append("background")

    return {
        "class_id": int(class_id),
        "class_name": class_name,
        "image_id": image_id,
        "num_regions": int(num_regions),
        "model_names": [name for name in model_names if name in per_model],
        "selected_ids": selected_ids,
        "labels": labels,
        "per_model": per_model,
        "background_available": background_available,
        "background_index_by_model": background_index_by_model,
    }


def _write_class_score_csv(path: Path, payload: Dict[str, object], class_names: Optional[Sequence[str]]) -> None:
    rows = []
    selected_ids = payload["selected_ids"]
    for model_name in payload["model_names"]:
        model_payload = payload["per_model"][model_name]
        for idx in selected_ids:
            rows.append(
                {
                    "model": model_name,
                    "class_id": int(idx),
                    "class_name": _score_class_name(int(idx), class_names),
                    "mean_cosine_score": float(model_payload["mean_foreground_scores"][idx]),
                    "is_target": int(int(idx) == int(payload["class_id"])),
                    "is_background": 0,
                    "mean_observed_confidence": float(model_payload["mean_observed_confidence"]),
                    "mean_target_prob_fg_softmax": float(model_payload["mean_target_prob_fg_softmax"]),
                    "mean_target_prob_all_softmax": float(model_payload["mean_target_prob_all_softmax"]),
                    "mean_target_rank": float(model_payload["mean_target_rank"]),
                    "mean_target_rank_normalized": float(model_payload["mean_target_rank_normalized"]),
                }
            )
        if payload["background_available"] and model_payload.get("background_score") is not None:
            rows.append(
                {
                    "model": model_name,
                    "class_id": -1,
                    "class_name": "background",
                    "mean_cosine_score": float(model_payload["background_score"]),
                    "is_target": 0,
                    "is_background": 1,
                    "mean_observed_confidence": float(model_payload["mean_observed_confidence"]),
                    "mean_target_prob_fg_softmax": float(model_payload["mean_target_prob_fg_softmax"]),
                    "mean_target_prob_all_softmax": float(model_payload["mean_target_prob_all_softmax"]),
                    "mean_target_rank": float(model_payload["mean_target_rank"]),
                    "mean_target_rank_normalized": float(model_payload["mean_target_rank_normalized"]),
                }
            )
    _save_csv(rows, path, columns=CLASS_SCORE_COLUMNS)


def _plot_class_score_png(path: Path, payload: Dict[str, object]) -> None:
    labels = payload["labels"]
    model_names = payload["model_names"]
    selected_ids = payload["selected_ids"]
    x = np.arange(len(labels))
    width = min(0.8 / max(len(model_names), 1), 0.25)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(11, 0.92 * len(labels)), 8),
        dpi=170,
        gridspec_kw={"height_ratios": [3.2, 1.2]},
    )
    ax = axes[0]
    for j, model_name in enumerate(model_names):
        model_payload = payload["per_model"][model_name]
        values = [float(model_payload["mean_foreground_scores"][idx]) for idx in selected_ids]
        if payload["background_available"]:
            background_score = model_payload.get("background_score")
            values.append(float(background_score) if background_score is not None else float("nan"))
        offset = (j - (len(model_names) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=model_name,
            color=_model_color(model_name),
            alpha=0.9,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean cosine score over selected positive queries")
    ax.set_title(
        "Class score comparison | "
        f"{payload['class_name']} | image_id={payload['image_id']} | N={payload['num_regions']}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(loc="upper right")
    note = "Background bar is omitted because OVTR has no learned background class anchor."
    if payload["background_available"]:
        note = "Background bar is shown because class anchors include an extra background row."
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=8.5, color="dimgray")

    ax2 = axes[1]
    metric_labels = ["observed P\n(target)", "target P\n(fg softmax)", "target rank\n(norm.)"]
    mx = np.arange(len(metric_labels))
    for j, model_name in enumerate(model_names):
        model_payload = payload["per_model"][model_name]
        values = [
            float(model_payload["mean_observed_confidence"]),
            float(model_payload["mean_target_prob_fg_softmax"]),
            float(model_payload["mean_target_rank_normalized"]),
        ]
        offset = (j - (len(model_names) - 1) / 2.0) * width
        ax2.bar(
            mx + offset,
            values,
            width=width,
            label=model_name,
            color=_model_color(model_name),
            alpha=0.9,
        )
    ax2.set_xticks(mx)
    ax2.set_xticklabels(metric_labels)
    ax2.set_ylabel("Value")
    ax2.set_ylim(0, 1.02)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)
    ax2.text(
        2,
        0.95,
        "rank normalized by #foreground classes\n(lower is better)",
        ha="center",
        va="top",
        fontsize=8.5,
        color="dimgray",
    )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_embedding_class_score_artifacts(
    png_path: Path,
    csv_path: Path,
    *,
    class_id: int,
    class_name: str,
    image_id: object,
    num_regions: int,
    outputs: Dict[str, AlignmentOutput],
    model_names: Sequence[str],
    class_names: Optional[Sequence[str]],
    class_anchors_by_model: Dict[str, torch.Tensor],
    score_topk: int,
    score_max_bars: int,
    score_temperature: float,
) -> None:
    payload = _compute_class_score_payload(
        class_id=class_id,
        class_name=class_name,
        image_id=image_id,
        num_regions=num_regions,
        outputs=outputs,
        model_names=model_names,
        class_names=class_names,
        class_anchors_by_model=class_anchors_by_model,
        score_topk=score_topk,
        score_max_bars=score_max_bars,
        score_temperature=score_temperature,
    )
    _plot_class_score_png(png_path, payload)
    _write_class_score_csv(csv_path, payload, class_names)


def _npz_scalar_text(value: np.ndarray) -> str:
    item = value.item() if getattr(value, "shape", ()) == () else value
    return str(item)


def write_embedding_class_score_artifacts_from_npz(
    npz_path: Path,
    class_anchors: torch.Tensor,
    *,
    class_names: Optional[Sequence[str]],
    score_topk: int = 10,
    score_max_bars: int = 12,
    score_temperature: float = 0.007,
    overwrite: bool = False,
) -> bool:
    npz_path = Path(npz_path)
    png_path = npz_path.with_name(f"{npz_path.stem}_class_scores.png")
    csv_path = npz_path.with_name(f"{npz_path.stem}_class_scores.csv")
    if not overwrite and png_path.exists() and csv_path.exists():
        return False

    data = np.load(npz_path)
    model_names = [str(name) for name in data["model_names"].tolist()]
    outputs: Dict[str, AlignmentOutput] = {}
    for model_name in model_names:
        key = _sanitize_name(model_name)
        outputs[model_name] = AlignmentOutput(
            embeddings=torch.from_numpy(data[f"{key}__embeddings"]).float(),
            region_text=torch.from_numpy(data[f"{key}__region_text"]).float(),
            confidence=torch.from_numpy(data[f"{key}__confidence"]).float(),
            relation=torch.from_numpy(data[f"{key}__relation"]).float(),
        )
    class_anchors_by_model = {model_name: class_anchors for model_name in model_names}
    write_embedding_class_score_artifacts(
        png_path,
        csv_path,
        class_id=int(data["class_id"]),
        class_name=_npz_scalar_text(data["class_name"]),
        image_id=_npz_scalar_text(data["image_id"]),
        num_regions=int(data["rois"].shape[0]),
        outputs=outputs,
        model_names=model_names,
        class_names=class_names,
        class_anchors_by_model=class_anchors_by_model,
        score_topk=score_topk,
        score_max_bars=score_max_bars,
        score_temperature=score_temperature,
    )
    return True


def _maybe_add_example(
    examples: List[EmbeddingGroupExample],
    candidate: EmbeddingGroupExample,
    *,
    max_examples: int,
    min_regions: int,
) -> None:
    if candidate.num_regions < min_regions:
        return
    examples.append(candidate)
    examples.sort(key=lambda item: item.num_regions, reverse=True)
    del examples[max(0, int(max_examples)):]


def _format_metric_row(
    sample_index: int,
    filename: str,
    class_id: int,
    class_name: str,
    model_name: str,
    output: AlignmentOutput,
    metrics: Dict[str, float],
) -> Dict[str, object]:
    return {
        "sample_index": sample_index,
        "image_id": filename,
        "filename": filename,
        "class_id": class_id,
        "class_name": class_name,
        "model": model_name,
        "num_regions": int(output.region_text.numel()),
        "region_text_mae": metrics["region_text_mae"],
        "region_region_mae": metrics["region_region_mae"],
        "region_region_pearson": metrics["region_region_pearson"],
        "mean_confidence": output.mean_confidence,
        "mean_region_text": output.mean_region_text,
    }


def _summary_rows(metric_rows: List[Dict[str, object]], model_names: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for model_name in model_names:
        model_rows = [row for row in metric_rows if row["model"] == model_name]
        if not model_rows:
            continue
        weights = [float(row["num_regions"]) for row in model_rows]
        total_regions = int(sum(weights))
        for aggregation in ("macro", "region_weighted"):
            if aggregation == "region_weighted":
                reducer = _weighted_mean
            else:
                reducer = lambda values, _: _finite_mean(values)
            rows.append(
                {
                    "model": model_name,
                    "aggregation": aggregation,
                    "num_groups": len(model_rows),
                    "total_regions": total_regions,
                    "region_text_mae": reducer([row["region_text_mae"] for row in model_rows], weights),
                    "region_region_mae": reducer([row["region_region_mae"] for row in model_rows], weights),
                    "region_region_pearson": reducer([row["region_region_pearson"] for row in model_rows], weights),
                    "mean_confidence": reducer([row["mean_confidence"] for row in model_rows], weights),
                    "mean_region_text": reducer([row["mean_region_text"] for row in model_rows], weights),
                }
            )
    return rows


def _aggregate_model_rows(class_summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = [row for row in class_summary_rows if row.get("aggregation") == "macro"]
    final_rows = []
    for model_name in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if str(row["model"]) == model_name]
        final_rows.append(
            {
                "model": model_name,
                "num_classes": len(model_rows),
                "total_groups": int(sum(int(row["num_groups"]) for row in model_rows)),
                "total_regions": int(sum(int(row["total_regions"]) for row in model_rows)),
                "region_text_mae": _finite_mean([row["region_text_mae"] for row in model_rows]),
                "region_region_mae": _finite_mean([row["region_region_mae"] for row in model_rows]),
                "region_region_pearson": _finite_mean([row["region_region_pearson"] for row in model_rows]),
                "mean_confidence": _finite_mean([row["mean_confidence"] for row in model_rows]),
                "mean_region_text": _finite_mean([row["mean_region_text"] for row in model_rows]),
            }
        )
    return final_rows


def _analyze_embedding_target(
    target: dict,
    *,
    viz_dir: Path,
    fp32_result: RunResult,
    compare_results: Dict[str, RunResult],
    class_names: Optional[Sequence[str]],
    image_root: Optional[str],
    groups: Sequence[dict],
    model_names: Sequence[str],
    max_groups_per_class: int,
    max_examples_per_class: int,
    min_regions: int,
    max_regions_per_group: int,
    score_topk: int,
    score_max_bars: int,
    score_temperature: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    class_id = int(target["class_id"])
    class_name = str(target["class_name"])
    class_out_dir = viz_dir / _sanitize_name(class_name)
    examples_dir = class_out_dir / "examples"
    class_out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    sorted_groups = sorted(
        groups,
        key=lambda item: (-int(item["count"]), -float(item["mean_alignment"]), int(item["sample_index"])),
    )[: max(0, int(max_groups_per_class))]
    class_anchors_by_model: Dict[str, torch.Tensor] = {"fp32": fp32_result.embedding_class_anchors}
    for label, result in compare_results.items():
        class_anchors_by_model[label] = (
            result.embedding_class_anchors
            if result.embedding_class_anchors is not None
            else fp32_result.embedding_class_anchors
        )
    metric_rows: List[Dict[str, object]] = []
    examples: List[EmbeddingGroupExample] = []
    skipped_no_match = 0
    positive_groups = 0

    for group in sorted_groups:
        frame_key = group["frame_key"]
        fp_frame = fp32_result.embedding_frames.get(frame_key)
        if fp_frame is None:
            skipped_no_match += 1
            continue
        candidate_indices = group["indices"]
        if candidate_indices.numel() > max_regions_per_group:
            scores = fp_frame.scores.index_select(0, candidate_indices)
            order = torch.argsort(scores, descending=True)[:max_regions_per_group]
            candidate_indices = candidate_indices.index_select(0, order)
        match_keys = _positive_match_keys(fp_frame, candidate_indices)
        compare_index_lists = {
            label: _indices_for_match_keys(result.embedding_frames.get(frame_key), match_keys)
            for label, result in compare_results.items()
        }
        keep_positions = []
        for pos in range(len(match_keys)):
            if all(index_list[pos] is not None for index_list in compare_index_lists.values()):
                keep_positions.append(pos)
        if not keep_positions:
            skipped_no_match += 1
            continue

        fp_keep_indices = [int(candidate_indices[pos].item()) for pos in keep_positions]
        fp_embeddings = _gather_rows(fp_frame.pred_embed, fp_keep_indices)
        fp_boxes = _gather_rows(fp_frame.boxes, fp_keep_indices)
        fp_anchors = fp32_result.embedding_class_anchors
        if fp_anchors is None or class_id >= fp_anchors.shape[0]:
            skipped_no_match += 1
            continue

        outputs: Dict[str, AlignmentOutput] = {
            "fp32": compute_alignment_output(
                fp_embeddings,
                fp_anchors,
                class_id,
                temperature=score_temperature,
            )
        }
        valid_group = True
        for label, result in compare_results.items():
            frame = result.embedding_frames.get(frame_key)
            index_list = compare_index_lists[label]
            matched_indices = [int(index_list[pos]) for pos in keep_positions if index_list[pos] is not None]
            if frame is None or len(matched_indices) != len(keep_positions):
                valid_group = False
                break
            anchors = result.embedding_class_anchors if result.embedding_class_anchors is not None else fp_anchors
            if anchors is None or class_id >= anchors.shape[0]:
                valid_group = False
                break
            embeddings = _gather_rows(frame.pred_embed, matched_indices)
            outputs[label] = compute_alignment_output(
                embeddings,
                anchors,
                class_id,
                temperature=score_temperature,
            )
        if not valid_group:
            skipped_no_match += 1
            continue

        reference_output = outputs["fp32"]
        for model_name in model_names:
            output = outputs[model_name]
            metrics = compute_distortion_metrics(reference_output, output)
            metric_rows.append(
                _format_metric_row(
                    int(group["sample_index"]),
                    fp_frame.file_path,
                    class_id,
                    class_name,
                    model_name,
                    output,
                    metrics,
                )
            )

        image, _ = _load_embedding_image(image_root, fp_frame.file_path, fp_boxes)
        _maybe_add_example(
            examples,
            EmbeddingGroupExample(
                sample_index=int(group["sample_index"]),
                image_id=fp_frame.file_path,
                filename=fp_frame.file_path,
                class_id=class_id,
                class_name=class_name,
                image=image,
                rois=_embedding_rois_from_boxes(fp_boxes),
                outputs=outputs,
            ),
            max_examples=max_examples_per_class,
            min_regions=min_regions,
        )
        positive_groups += 1

    summary_rows = _summary_rows(metric_rows, model_names)
    _save_csv(metric_rows, class_out_dir / "metrics_per_group.csv", columns=ALIGNMENT_GROUP_COLUMNS)
    _save_csv(summary_rows, class_out_dir / "metrics_summary.csv", columns=ALIGNMENT_SUMMARY_COLUMNS)
    _plot_alignment_scatter(class_out_dir / "distortion_scatter.png", summary_rows, title=f"Detection alignment distortion: {class_name}")
    for idx, example in enumerate(examples):
        stem = f"{idx:03d}_{_sanitize_name(example.class_name)}_{example.sample_index}"
        _plot_example(examples_dir / f"{stem}.png", example, model_names)
        npz_path = examples_dir / f"{stem}.npz"
        _write_example_npz(npz_path, example, model_names)
        write_embedding_class_score_artifacts(
            examples_dir / f"{stem}_class_scores.png",
            examples_dir / f"{stem}_class_scores.csv",
            class_id=example.class_id,
            class_name=example.class_name,
            image_id=example.image_id,
            num_regions=example.num_regions,
            outputs=example.outputs,
            model_names=model_names,
            class_names=class_names,
            class_anchors_by_model=class_anchors_by_model,
            score_topk=score_topk,
            score_max_bars=score_max_bars,
            score_temperature=score_temperature,
        )

    enriched_rows = []
    for row in summary_rows:
        enriched = {
            "rank": target["rank"],
            "class_id": class_id,
            "class_name": class_name,
            "instance_count": target["instance_count"],
            "image_count": target["image_count"],
            "first_index": target["first_index"],
            "processed_groups": len(sorted_groups),
            "positive_groups": positive_groups,
            "skipped_no_match": skipped_no_match,
        }
        enriched.update(row)
        enriched_rows.append(enriched)

    class_summary = {
        "class_id": class_id,
        "class_name": class_name,
        "processed_groups": len(sorted_groups),
        "positive_groups": positive_groups,
        "skipped_no_match": skipped_no_match,
        "metric_rows": len(metric_rows),
        "examples": len(examples),
    }
    return enriched_rows, class_summary


def write_embedding_visualizations(
    *,
    output_dir: Path,
    fp32_result: RunResult,
    compare_results: Dict[str, RunResult],
    class_names: Optional[Sequence[str]],
    image_root: Optional[str],
    num_classes: int,
    score_topk: int,
    score_max_bars: int,
    max_regions_per_group: int,
    max_groups_per_class: int = 20,
    max_examples_per_class: int = 2,
    min_regions: int = 2,
    score_temperature: float = 0.007,
) -> dict:
    viz_dir = output_dir / "embedding_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    summary_path = viz_dir / "alignment_summary.json"

    if fp32_result.embedding_class_anchors is None or not fp32_result.embedding_frames:
        for csv_name, columns in (
            ("selected_classes.csv", ALIGNMENT_SELECTED_CLASS_COLUMNS),
            ("metrics_summary_by_class.csv", ALIGNMENT_CLASS_SUMMARY_COLUMNS),
            ("metrics_model_macro_mean.csv", ALIGNMENT_MODEL_MACRO_COLUMNS),
        ):
            _save_csv([], viz_dir / csv_name, columns=columns)
        summary = {"status": "empty", "reason": "missing fp32 embedding traces"}
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary

    compare_results = dict(compare_results)
    class_stats, groups_by_class = _positive_group_candidates(fp32_result)
    targets = _select_embedding_targets(class_stats, class_names, num_classes)
    _save_csv(targets, viz_dir / "selected_classes.csv", columns=ALIGNMENT_SELECTED_CLASS_COLUMNS)

    model_names = ["fp32"] + list(compare_results.keys())
    all_class_summary_rows: List[Dict[str, object]] = []
    class_summaries = []
    for target in targets:
        class_rows, class_summary = _analyze_embedding_target(
            target,
            viz_dir=viz_dir,
            fp32_result=fp32_result,
            compare_results=compare_results,
            class_names=class_names,
            image_root=image_root,
            groups=groups_by_class.get(int(target["class_id"]), []),
            model_names=model_names,
            max_groups_per_class=max_groups_per_class,
            max_examples_per_class=max_examples_per_class,
            min_regions=min_regions,
            max_regions_per_group=max_regions_per_group,
            score_topk=score_topk,
            score_max_bars=score_max_bars,
            score_temperature=score_temperature,
        )
        all_class_summary_rows.extend(class_rows)
        class_summaries.append(class_summary)

    _save_csv(
        all_class_summary_rows,
        viz_dir / "metrics_summary_by_class.csv",
        columns=ALIGNMENT_CLASS_SUMMARY_COLUMNS,
    )
    model_macro_rows = _aggregate_model_rows(all_class_summary_rows)
    _save_csv(
        model_macro_rows,
        viz_dir / "metrics_model_macro_mean.csv",
        columns=ALIGNMENT_MODEL_MACRO_COLUMNS,
    )
    _plot_alignment_scatter(
        viz_dir / "distortion_scatter_100classes.png",
        all_class_summary_rows,
        title="Detection alignment distortion by class",
    )

    generated_classes = sum(1 for item in class_summaries if int(item.get("positive_groups", 0)) > 0)
    summary = {
        "status": "ok",
        "style": "ovtrack_alignment_distortion",
        "requested_classes": int(num_classes),
        "generated_classes": generated_classes,
        "available_positive_classes": len(class_stats),
        "compare_labels": list(compare_results.keys()),
        "score_temperature": float(score_temperature),
        "score_topk": int(score_topk),
        "score_max_bars": int(score_max_bars),
        "class_score_background": "omitted_unless_anchor_count_exceeds_class_count",
        "max_groups_per_class": int(max_groups_per_class),
        "max_examples_per_class": int(max_examples_per_class),
        "min_regions": int(min_regions),
        "max_regions_per_group": int(max_regions_per_group),
        "classes": class_summaries,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, sort_keys=True)
    return summary


def sync_timing_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def run_inference_frame(
    *,
    model,
    data,
    track_instances: Optional[Instances],
    info,
    file_path: str,
    num_classes: int,
    args,
    run_mode: str = "",
    collect_embedding_trace: bool = False,
) -> Tuple[Instances, List[np.ndarray], float, float, Optional[EmbeddingFrameTrace]]:
    frame_id = int(info[0])
    score_threshold, _ = set_tracking_thresholds(model, file_path, args)
    res = model.inference_single_image(
        data,
        track_instances,
        frame_id=frame_id,
        ori_img_size=info[1],
        return_embedding_trace=collect_embedding_trace,
    )
    next_track_instances = res["track_instances"]
    dt_instances = next_track_instances.to(torch.device("cpu"))
    dt_instances = filter_dt_by_score(dt_instances, score_threshold)
    dt_instances = filter_dt_by_area(dt_instances, args.area_threshold)
    track_results = track_results_from_instances(dt_instances, num_classes)
    embedding_trace = None
    if collect_embedding_trace:
        embedding_trace = make_embedding_frame_trace(
            run_mode=run_mode,
            file_path=file_path,
            frame_id=frame_id,
            score_threshold=score_threshold,
            raw_trace=res.get("embedding_trace"),
        )
    return next_track_instances, track_results, score_threshold, float(model.track_base.max_obj_id), embedding_trace
