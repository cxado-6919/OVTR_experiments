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
) -> Tuple[Instances, List[np.ndarray], float, float]:
    frame_id = int(info[0])
    score_threshold, _ = set_tracking_thresholds(model, file_path, args)
    res = model.inference_single_image(data, track_instances, frame_id=frame_id, ori_img_size=info[1])
    next_track_instances = res["track_instances"]
    dt_instances = next_track_instances.to(torch.device("cpu"))
    dt_instances = filter_dt_by_score(dt_instances, score_threshold)
    dt_instances = filter_dt_by_area(dt_instances, args.area_threshold)
    track_results = track_results_from_instances(dt_instances, num_classes)
    return next_track_instances, track_results, score_threshold, float(model.track_base.max_obj_id)
