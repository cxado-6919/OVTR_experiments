from collections import defaultdict
from typing import Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


BACKBONE_DISTILL_FEATURE_KEY = "_cr_qat_backbone_features"
CR_QAT_BACKBONE_FD_LOSS_KEY = "loss_cr_qat_backbone_fd"
CR_QAT_BACKBONE_FD_TARGET = "projected_srcs"
CR_QAT_BACKBONE_FD_LOSS_TYPE = "pkd_normalized_mse"
TRKD_TRACE_KEY = "_cr_qat_trkd_trace"
CR_QAT_TRKD_LOSS_KEY = "loss_cr_qat_trkd"
CR_QAT_TRKD_ANCHOR = "image_feat"
CR_QAT_TRKD_ASSIGNMENT = "teacher_positive_obj_id_intersection"


def make_trkd_trace(frame_index, sample_index, labels, obj_ids, embeddings, anchors) -> dict:
    return {
        "frame_index": int(frame_index),
        "sample_index": -1 if sample_index is None else int(sample_index),
        "labels": labels.detach().long(),
        "obj_ids": obj_ids.detach().long(),
        "embeddings": embeddings,
        "anchors": anchors.detach(),
    }


def make_projected_src_feature_frame(srcs, masks=None) -> dict:
    frame = {"srcs": tuple(srcs)}
    if masks is not None:
        frame["masks"] = tuple(masks)
    return frame


def _resize_mask(mask: torch.Tensor, size) -> torch.Tensor:
    if mask.shape[-2:] == size:
        return mask
    return F.interpolate(mask[:, None].float(), size=size, mode="nearest")[:, 0].to(torch.bool)


def _combined_valid_mask(student_mask, teacher_mask, size, device) -> torch.Tensor:
    combined = None
    for mask in (student_mask, teacher_mask):
        if mask is None:
            continue
        mask = _resize_mask(mask.to(device=device), size)
        combined = mask if combined is None else (combined | mask)
    if combined is None:
        return None
    return (~combined).unsqueeze(1).float()


def _normalize_projected_feature(feature: torch.Tensor, valid_mask: torch.Tensor, eps: float) -> torch.Tensor:
    feature = feature.float()
    if valid_mask is None:
        mean = feature.mean(dim=(2, 3), keepdim=True)
        var = (feature - mean).pow(2).mean(dim=(2, 3), keepdim=True)
    else:
        denom = valid_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = (feature * valid_mask).sum(dim=(2, 3), keepdim=True) / denom
        var = ((feature - mean).pow(2) * valid_mask).sum(dim=(2, 3), keepdim=True) / denom
    return (feature - mean) / (var + eps).sqrt()


def _frame_srcs(frame: Mapping[str, Sequence[torch.Tensor]]) -> Sequence[torch.Tensor]:
    if "srcs" not in frame:
        raise KeyError("Backbone distillation feature frame is missing 'srcs'.")
    return frame["srcs"]


def _frame_masks(frame: Mapping[str, Sequence[torch.Tensor]]) -> Sequence[torch.Tensor]:
    return frame.get("masks", ())


def projected_backbone_feature_distillation_loss(
    student_frames: Sequence[Mapping[str, Sequence[torch.Tensor]]],
    teacher_frames: Sequence[Mapping[str, Sequence[torch.Tensor]]],
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if len(student_frames) != len(teacher_frames):
        raise ValueError(
            f"Student/teacher feature frame count mismatch: {len(student_frames)} vs {len(teacher_frames)}"
        )
    losses = []
    for student_frame, teacher_frame in zip(student_frames, teacher_frames):
        student_srcs = _frame_srcs(student_frame)
        teacher_srcs = _frame_srcs(teacher_frame)
        student_masks = _frame_masks(student_frame)
        teacher_masks = _frame_masks(teacher_frame)
        if len(student_srcs) != len(teacher_srcs):
            raise ValueError(
                f"Student/teacher feature level count mismatch: {len(student_srcs)} vs {len(teacher_srcs)}"
            )
        for level_idx, (student_src, teacher_src) in enumerate(zip(student_srcs, teacher_srcs)):
            if student_src.shape[1] != teacher_src.shape[1]:
                raise ValueError(
                    "Student/teacher projected feature channel mismatch at "
                    f"level {level_idx}: {student_src.shape[1]} vs {teacher_src.shape[1]}"
                )
            teacher_src = teacher_src.detach().to(device=student_src.device)
            if teacher_src.shape[-2:] != student_src.shape[-2:]:
                teacher_src = F.interpolate(
                    teacher_src.float(),
                    size=student_src.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            student_mask = student_masks[level_idx] if level_idx < len(student_masks) else None
            teacher_mask = teacher_masks[level_idx] if level_idx < len(teacher_masks) else None
            valid_mask = _combined_valid_mask(
                student_mask,
                teacher_mask,
                student_src.shape[-2:],
                student_src.device,
            )
            student_norm = _normalize_projected_feature(student_src, valid_mask, eps)
            teacher_norm = _normalize_projected_feature(teacher_src, valid_mask, eps)
            level_loss = (student_norm - teacher_norm).pow(2)
            if valid_mask is not None:
                denom = (valid_mask.sum() * student_src.shape[1]).clamp_min(1.0)
                level_loss = (level_loss * valid_mask).sum() / denom
            else:
                level_loss = level_loss.mean()
            losses.append(level_loss)
    if not losses:
        raise ValueError("No projected backbone features were provided for distillation.")
    return torch.stack(losses).mean()



def _iter_trkd_records(traces):
    for trace in traces or ():
        labels = trace.get("labels")
        obj_ids = trace.get("obj_ids")
        embeddings = trace.get("embeddings")
        anchors = trace.get("anchors")
        if labels is None or obj_ids is None or embeddings is None or anchors is None:
            continue
        for idx in range(int(labels.numel())):
            obj_id = int(obj_ids[idx].detach().cpu().item())
            if obj_id < 0:
                continue
            yield {
                "key": (int(trace.get("sample_index", -1)), int(trace.get("frame_index", -1)), obj_id),
                "label": int(labels[idx].detach().cpu().item()),
                "embedding": embeddings[idx],
                "anchor": anchors[idx],
            }


def trkd_similarity_matrix_loss(student_anchor, student_regions, teacher_anchor, teacher_regions) -> torch.Tensor:
    student_x = torch.cat([student_anchor[None], student_regions], dim=0).float()
    teacher_x = torch.cat([teacher_anchor[None], teacher_regions], dim=0).float()
    student_x = F.normalize(student_x, dim=1)
    teacher_x = F.normalize(teacher_x, dim=1)
    student_matrix = student_x @ student_x.t()
    teacher_matrix = teacher_x @ teacher_x.t()
    return F.smooth_l1_loss(student_matrix, teacher_matrix, reduction="mean")


def _zero_trkd_loss(student_traces, teacher_traces) -> torch.Tensor:
    for traces in (student_traces, teacher_traces):
        for trace in traces or ():
            embeddings = trace.get("embeddings")
            if isinstance(embeddings, torch.Tensor):
                return embeddings.sum() * 0.0
    return torch.tensor(0.0)


def trkd_relational_distillation_loss(student_traces, teacher_traces) -> torch.Tensor:
    student_by_key = {record["key"]: record for record in _iter_trkd_records(student_traces)}
    grouped = defaultdict(lambda: {"student": [], "teacher": [], "student_anchor": None, "teacher_anchor": None})
    for teacher_record in _iter_trkd_records(teacher_traces):
        student_record = student_by_key.get(teacher_record["key"])
        if student_record is None:
            continue
        label = teacher_record["label"]
        group = grouped[label]
        group["student"].append(student_record["embedding"])
        group["teacher"].append(teacher_record["embedding"].detach().to(device=student_record["embedding"].device))
        if group["student_anchor"] is None:
            group["student_anchor"] = student_record["anchor"].to(device=student_record["embedding"].device)
            group["teacher_anchor"] = teacher_record["anchor"].detach().to(device=student_record["embedding"].device)

    losses = []
    for group in grouped.values():
        if not group["student"]:
            continue
        student_regions = torch.stack(group["student"], dim=0)
        teacher_regions = torch.stack(group["teacher"], dim=0)
        losses.append(
            trkd_similarity_matrix_loss(
                group["student_anchor"],
                student_regions,
                group["teacher_anchor"],
                teacher_regions,
            )
        )
    if not losses:
        return _zero_trkd_loss(student_traces, teacher_traces)
    return torch.stack(losses).mean()


class BackboneFeatureDistiller:
    def __init__(
        self,
        teacher_model: torch.nn.Module,
        *,
        eps: float = 1e-6,
        enable_trkd: bool = False,
        stage_getter: Optional[Callable[[], int]] = None,
    ):
        self.teacher_model = teacher_model
        self.eps = eps
        self.enable_trkd = enable_trkd
        self.stage_getter = stage_getter

    def _current_stage(self) -> int:
        if self.stage_getter is None:
            return 1
        return int(self.stage_getter())

    def _teacher_forward_with_trace(self, data: Mapping[str, object]):
        previous_backbone_flag = bool(getattr(self.teacher_model, "return_backbone_distill_features", False))
        previous_trkd_flag = bool(getattr(self.teacher_model, "return_trkd_trace", False))
        self.teacher_model.return_backbone_distill_features = True
        self.teacher_model.return_trkd_trace = True
        self.teacher_model.eval()
        try:
            with torch.no_grad():
                return self.teacher_model(data)
        finally:
            self.teacher_model.return_backbone_distill_features = previous_backbone_flag
            self.teacher_model.return_trkd_trace = previous_trkd_flag
            self.teacher_model.eval()

    def __call__(self, student_outputs: Mapping[str, object], data: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
        if BACKBONE_DISTILL_FEATURE_KEY not in student_outputs:
            raise KeyError(
                f"Student outputs are missing {BACKBONE_DISTILL_FEATURE_KEY}; "
                "enable backbone feature capture on the student model."
            )
        losses = {}
        use_trkd = self.enable_trkd and self._current_stage() >= 2
        if use_trkd:
            teacher_outputs = self._teacher_forward_with_trace(data)
            teacher_frames = teacher_outputs[BACKBONE_DISTILL_FEATURE_KEY]
            losses[CR_QAT_TRKD_LOSS_KEY] = trkd_relational_distillation_loss(
                student_outputs.get(TRKD_TRACE_KEY, ()),
                teacher_outputs.get(TRKD_TRACE_KEY, ()),
            )
        else:
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_frames = self.teacher_model.extract_projected_backbone_features(data)
        losses[CR_QAT_BACKBONE_FD_LOSS_KEY] = projected_backbone_feature_distillation_loss(
            student_outputs[BACKBONE_DISTILL_FEATURE_KEY],
            teacher_frames,
            eps=self.eps,
        )
        return losses
