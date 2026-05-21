# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import json
import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from util import box_ops
from util.misc import get_rank, inverse_sigmoid
from detectron2.structures import Boxes, Instances, pairwise_iou


MCIP_MEMORY_FIELDS = {
    'img_memory': ('hidden',),
    'semantic_memory': ('hidden',),
    'cls_conf_memory': (),
    'cls_entropy_memory': (),
    'prev_boxes': (4,),
    'box_velocity': (4,),
    'memory_age': (),
}


def ensure_mcip_track_fields(track_instances: Instances, hidden_dim: int, device=None, dtype=None) -> Instances:
    """Ensure M-CIP persistent memory fields exist and match the current track set."""
    if len(track_instances) == 0:
        return track_instances

    if device is None:
        if track_instances.has('query_tgt'):
            device = track_instances.query_tgt.device
        elif track_instances.has('scores'):
            device = track_instances.scores.device
        else:
            device = torch.device('cpu')
    if dtype is None:
        if track_instances.has('query_tgt'):
            dtype = track_instances.query_tgt.dtype
        else:
            dtype = torch.float32

    num_tracks = len(track_instances)
    field_shapes = {
        'img_memory': (num_tracks, hidden_dim),
        'semantic_memory': (num_tracks, hidden_dim),
        'cls_conf_memory': (num_tracks,),
        'cls_entropy_memory': (num_tracks,),
        'prev_boxes': (num_tracks, 4),
        'box_velocity': (num_tracks, 4),
        'memory_age': (num_tracks,),
    }

    for name, shape in field_shapes.items():
        needs_init = not track_instances.has(name)
        if not needs_init:
            value = track_instances.get(name)
            needs_init = (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(shape)
                or value.device != device
            )
        if needs_init:
            track_instances.set(name, torch.zeros(shape, device=device, dtype=dtype))
        else:
            value = track_instances.get(name)
            if value.dtype != dtype:
                track_instances.set(name, value.to(dtype=dtype))
    return track_instances


def random_drop_tracks(track_instances: Instances, drop_probability: float) -> Instances:
    if drop_probability > 0 and len(track_instances) > 0:
        keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = track_instances[keep_idxes]
    return track_instances


class TrackEmbeddingBase(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)
        self._reset_parameters()

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt


class QueryInteractionModule(TrackEmbeddingBase):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if args.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if args.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if args.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
            inactive_instances = track_instances[track_instances.obj_idxes < 0]

            # add fp for each active track in a specific probability.
            fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
            selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

            if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
                num_fp = len(selected_active_track_instances)
                if num_fp >= len(inactive_instances):
                    fp_track_instances = inactive_instances
                else:
                    inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                    selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                    ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                    # select the fp with the largest IoU for each active track.
                    fp_indexes = ious.max(dim=0).indices

                    # remove duplicate fp.
                    fp_indexes = torch.unique(fp_indexes)
                    fp_track_instances = inactive_instances[fp_indexes]

                merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
                return merged_track_instances

            return active_track_instances

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            # set -2 instead of -1 to ensure that these tracks will not be selected in matching.
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        dim = track_instances.query_pos.shape[1]
        out_embed = track_instances.output_embedding
        query_pos = track_instances.query_pos[:, :dim // 2]
        query_feat = track_instances.query_pos[:, dim//2:]
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos[:, :dim // 2] = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[:, dim//2:] = query_feat

        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
        return track_instances

    def forward(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data['init_track_instances']
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances
    

class Category_Information_Propagator(QueryInteractionModule):
    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__(args, dim_in, hidden_dim, dim_out)
        self.score_thresh = 0.8
        self.random_drop = args.random_drop
        self.fp_ratio = args.fp_ratio
        self.update_query_pos = args.update_query_pos

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        dropout = args.merger_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def _cip_core(self, track_instances: Instances, cip_img: torch.Tensor) -> torch.Tensor:
        query_pos = track_instances.query_pos
        query_feat = track_instances.query_tgt
        q = k = query_pos + cip_img
        tgt = cip_img

        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        return query_feat

    def _aggregate_category_info(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances

        track_instances.query_tgt = self._cip_core(
            track_instances,
            track_instances.output_embedding_img,
        )
        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :4].detach().clone())
        return track_instances

    def _select_active_tracks(self, data: dict, id_gt=None) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.iou > 0.5)
            active_track_instances = track_instances[active_idxes]
            active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(track_instances, active_track_instances)
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]
        return active_track_instances

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances: Instances, active_track_instances: Instances) -> Instances:
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        fp_prob = torch.ones_like(active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[torch.bernoulli(fp_prob).bool()]

        if len(inactive_instances) > 0 and len(selected_active_track_instances) > 0:
            num_fp = len(selected_active_track_instances)
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                inactive_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(inactive_instances.pred_boxes))
                selected_active_boxes = Boxes(box_ops.box_cxcywh_to_xyxy(selected_active_track_instances.pred_boxes))
                ious = pairwise_iou(inactive_boxes, selected_active_boxes)
                fp_indexes = ious.max(dim=0).indices
                fp_indexes = torch.unique(fp_indexes)
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat([active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances

    def forward(self, data, id_gt=None) -> Instances:
        active_track_instances = self._select_active_tracks(data, id_gt=id_gt)
        active_track_instances = self._aggregate_category_info(active_track_instances)
        init_track_instances: Instances = data['init_track_instances']
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances


class MemoryCalibratedCategoryInformationPropagator(Category_Information_Propagator):
    """Safe Residual M-CIP v2: opt-in bounded memory bias after baseline CIP."""

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        super()._build_layers(args, dim_in, hidden_dim, dim_out)

        def opt(name, default):
            value = getattr(args, name, default)
            return default if value is None else value

        self.mcip_detach_memory = bool(opt('mcip_detach_memory', True))
        self.mcip_memory_momentum = float(opt('mcip_memory_momentum', 0.8))
        self.mcip_use_semantic_memory = bool(opt('mcip_use_semantic_memory', True))
        self.mcip_use_motion_ref = bool(opt('mcip_use_motion_ref', False))
        self.mcip_motion_momentum = float(opt('mcip_motion_momentum', 0.7))
        self.mcip_max_memory_update = float(opt('mcip_max_memory_update', 0.05))
        self.mcip_max_residual_ratio = float(opt('mcip_max_residual_ratio', 0.05))
        self.mcip_motion_offset_cap = float(opt('mcip_motion_offset_cap', 0.02))
        self.debug_mcip = bool(opt('debug_mcip', False))
        self.mcip_debug_log_interval = max(1, int(opt('mcip_debug_log_interval', 1) or 1))
        self.mcip_debug_log_calls = 0
        self.mcip_debug_stats_file = None
        if self.debug_mcip:
            self.mcip_debug_stats_file = self._resolve_debug_stats_file(args)

        self.mcip_img_obs_norm = nn.LayerNorm(dim_in)
        self.mcip_img_memory_norm = nn.LayerNorm(dim_in)
        self.mcip_sem_memory_norm = nn.LayerNorm(dim_in)
        adapter_input_dim = dim_in * 3 + 3
        self.memory_residual_adapter = nn.Sequential(
            nn.Linear(adapter_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim_in),
        )
        self.memory_inject_logit = nn.Parameter(torch.tensor(-4.0))
        self.motion_scale = nn.Parameter(torch.tensor(float(opt('mcip_motion_scale_init', 0.0))))
        self.mcip_debug_stats = {}
        self.last_mcip_debug_stats = self.mcip_debug_stats
        self.last_debug_stats = self.mcip_debug_stats

    def _resolve_debug_stats_file(self, args):
        debug_stats_file = getattr(args, 'mcip_debug_stats_file', None)
        output_dir = getattr(args, 'output_dir', None)
        if debug_stats_file is None and output_dir:
            debug_stats_file = os.path.join(output_dir, 'mcip_debug_stats.jsonl')
        if not debug_stats_file:
            return None

        rank = get_rank()
        if rank == 0:
            return debug_stats_file

        base, ext = os.path.splitext(debug_stats_file)
        return '{}_rank{}{}'.format(base, rank, ext or '.jsonl')

    def _reset_parameters(self):
        super()._reset_parameters()
        final = self.memory_residual_adapter[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _detach_if_needed(self, tensor):
        return tensor.detach() if self.mcip_detach_memory else tensor

    def _normalize_entropy(self, entropy: torch.Tensor, track_instances: Instances) -> torch.Tensor:
        entropy_norm = entropy.to(device=track_instances.scores.device, dtype=track_instances.scores.dtype)
        num_classes = 0
        if track_instances.has('pred_logits'):
            num_classes = int(track_instances.pred_logits.shape[-1])
        if num_classes > 1:
            log_classes = math.log(num_classes)
            entropy_norm = torch.where(entropy_norm > 1.0, entropy_norm / log_classes, entropy_norm)
        return entropy_norm.clamp(0.0, 1.0)

    def _clamp_residual_delta(self, delta: torch.Tensor, base_query_tgt: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        delta_norm = delta.float().norm(dim=-1, keepdim=True)
        base_norm = base_query_tgt.detach().float().norm(dim=-1, keepdim=True)
        max_norm = float(self.mcip_max_residual_ratio) * base_norm
        scale = torch.minimum(
            torch.ones_like(delta_norm),
            max_norm / delta_norm.clamp_min(eps),
        ).to(dtype=delta.dtype)
        return delta * scale

    def _update_debug_stats(self, **stats):
        clean_stats = {}
        for name, value in stats.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
                if value.numel() != 1:
                    value = value.float().mean()
                value = value.item()
            if isinstance(value, bool):
                clean_stats[name] = value
            elif isinstance(value, int):
                clean_stats[name] = int(value)
            else:
                clean_stats[name] = float(value)
        self.mcip_debug_stats = clean_stats
        self.last_mcip_debug_stats = clean_stats
        self.last_debug_stats = clean_stats
        self._write_debug_stats(clean_stats)

    def _write_debug_stats(self, stats):
        if not self.mcip_debug_stats_file:
            return
        self.mcip_debug_log_calls += 1
        if self.mcip_debug_log_calls % self.mcip_debug_log_interval != 0:
            return

        stats_dir = os.path.dirname(self.mcip_debug_stats_file)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        with open(self.mcip_debug_stats_file, 'a') as handle:
            handle.write(json.dumps(stats, sort_keys=True) + '\n')

    def _get_observation(self, track_instances: Instances, name: str, like: torch.Tensor) -> torch.Tensor:
        if track_instances.has(name):
            value = track_instances.get(name)
            if value.shape[0] == len(track_instances):
                return value.to(device=like.device, dtype=like.dtype)
        return torch.zeros((len(track_instances),) + tuple(like.shape[1:]), device=like.device, dtype=like.dtype)

    def _aggregate_category_info(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            if getattr(self, 'debug_mcip', False):
                self._update_debug_stats(
                    residual_norm_ratio_mean=0.0,
                    residual_norm_ratio_max=0.0,
                    residual_cosine_to_base=0.0,
                    query_tgt_cosine_to_baseline=1.0,
                    relative_query_tgt_l2_error=0.0,
                    gate_saturation_frac=0.0,
                    memory_update_gate_mean=0.0,
                    memory_update_gate_max=0.0,
                    memory_update_gate_min=0.0,
                    inject_gate_mean=0.0,
                    inject_gate_max=0.0,
                    inject_gate_min=0.0,
                    motion_offset_l1_mean=0.0,
                    motion_offset_l1_max=0.0,
                    motion_scale=0.0,
                    cls_conf_obs_mean=0.0,
                    cls_conf_obs_min=0.0,
                    cls_conf_obs_max=0.0,
                    cls_entropy_norm_mean=0.0,
                    cls_entropy_norm_min=0.0,
                    cls_entropy_norm_max=0.0,
                    current_reliability_mean=0.0,
                    memory_reliability_mean=0.0,
                    motion_reliability_mean=0.0,
                    motion_reliability_max=0.0,
                    active_track_count=0,
                )
            return track_instances

        dim = track_instances.output_embedding_img.shape[1]
        ensure_mcip_track_fields(
            track_instances,
            dim,
            device=track_instances.output_embedding_img.device,
            dtype=track_instances.output_embedding_img.dtype,
        )

        out_embed_img = track_instances.output_embedding_img
        semantic_obs = self._get_observation(track_instances, 'semantic_obs', out_embed_img)
        cls_conf_obs = self._get_observation(track_instances, 'cls_conf_obs', track_instances.scores).reshape(-1)
        cls_entropy_obs = self._get_observation(track_instances, 'cls_entropy_obs', track_instances.scores).reshape(-1)

        img_memory_old = track_instances.img_memory
        semantic_memory_old = track_instances.semantic_memory
        box_velocity_old = track_instances.box_velocity
        memory_age = track_instances.memory_age
        cls_conf_memory_old = track_instances.cls_conf_memory.to(dtype=out_embed_img.dtype)
        cls_entropy_memory_old = track_instances.cls_entropy_memory.to(dtype=out_embed_img.dtype)
        scores = track_instances.scores.to(dtype=out_embed_img.dtype).clamp(0.0, 1.0)
        memory_age_normalized = memory_age.float().clamp(max=100.0).to(dtype=out_embed_img.dtype) / 100.0

        base_query_tgt = self._cip_core(track_instances, out_embed_img)
        cls_entropy_norm = self._normalize_entropy(cls_entropy_obs, track_instances).to(dtype=out_embed_img.dtype)
        cls_entropy_memory_norm = self._normalize_entropy(cls_entropy_memory_old, track_instances).to(dtype=out_embed_img.dtype)
        current_reliability = (scores * (1.0 - cls_entropy_norm)).clamp(0.0, 1.0)
        memory_reliability = (
            cls_conf_memory_old.clamp(0.0, 1.0) * (1.0 - cls_entropy_memory_norm)
        ).clamp(0.0, 1.0)
        img_consistency = (
            F.cosine_similarity(out_embed_img.float(), img_memory_old.float(), dim=-1)
            .to(dtype=out_embed_img.dtype)
            .add(1.0)
            .mul(0.5)
            .clamp(0.0, 1.0)
        )
        valid_mem = (memory_age > 0).to(dtype=out_embed_img.dtype)

        semantic_memory_for_adapter = semantic_memory_old if self.mcip_use_semantic_memory else torch.zeros_like(semantic_memory_old)
        adapter_input = torch.cat([
            self.mcip_img_obs_norm(out_embed_img),
            self.mcip_img_memory_norm(img_memory_old),
            self.mcip_sem_memory_norm(semantic_memory_for_adapter),
            current_reliability[:, None],
            img_consistency[:, None],
            memory_age_normalized[:, None],
        ], dim=-1)
        delta = self.memory_residual_adapter(adapter_input)
        delta = self._clamp_residual_delta(delta, base_query_tgt)
        inject_gate = (
            valid_mem
            * current_reliability
            * img_consistency
            * torch.sigmoid(self.memory_inject_logit).to(dtype=out_embed_img.dtype)
        )
        residual = inject_gate[:, None] * delta
        track_instances.query_tgt = base_query_tgt + residual

        update_gate = (
            valid_mem
            * float(self.mcip_max_memory_update)
            * current_reliability
            * img_consistency
        ).clamp(0.0, float(self.mcip_max_memory_update))
        update_gate_col = update_gate[:, None]
        is_new = memory_age <= 0
        img_memory_updated = (1.0 - update_gate_col) * img_memory_old + update_gate_col * out_embed_img
        img_memory_new = torch.where(is_new[:, None], out_embed_img, img_memory_updated)
        if self.mcip_use_semantic_memory:
            semantic_memory_updated = (
                (1.0 - update_gate_col) * semantic_memory_old
                + update_gate_col * semantic_obs
            )
            semantic_memory_new = torch.where(is_new[:, None], semantic_obs, semantic_memory_updated)
        else:
            semantic_memory_new = torch.where(is_new[:, None], semantic_obs, semantic_memory_old)

        cls_conf_obs = cls_conf_obs.to(dtype=out_embed_img.dtype).clamp(0.0, 1.0)
        cls_conf_memory_new = torch.where(
            is_new,
            cls_conf_obs,
            (1.0 - update_gate) * cls_conf_memory_old + update_gate * cls_conf_obs,
        )
        cls_entropy_memory_new = torch.where(
            is_new,
            cls_entropy_norm,
            (1.0 - update_gate) * cls_entropy_memory_norm + update_gate * cls_entropy_norm,
        )

        pred_boxes_det = track_instances.pred_boxes[:, :4].detach()
        prev_boxes_old = track_instances.prev_boxes
        box_delta = pred_boxes_det - prev_boxes_old
        box_delta = torch.where(is_new[:, None], torch.zeros_like(box_delta), box_delta)
        box_velocity_new = (
            float(self.mcip_motion_momentum) * box_velocity_old
            + (1.0 - float(self.mcip_motion_momentum)) * box_delta
        )
        box_velocity_new = torch.where(is_new[:, None], torch.zeros_like(box_velocity_new), box_velocity_new)
        memory_age_new = memory_age + 1

        next_boxes = pred_boxes_det.clamp(1e-4, 1.0 - 1e-4)
        motion_offset = torch.zeros_like(pred_boxes_det)
        motion_reliability = current_reliability
        if self.mcip_use_motion_ref and not self.training:
            velocity_det = box_velocity_new.detach()
            valid_motion = (memory_age > 0).to(pred_boxes_det.dtype)[:, None]
            motion_reliability = torch.where(
                memory_age > 0,
                0.5 * current_reliability + 0.5 * memory_reliability,
                current_reliability,
            ).clamp(0.0, 1.0)
            xy_offset = (
                valid_motion
                * motion_reliability[:, None].to(dtype=pred_boxes_det.dtype)
                * torch.tanh(self.motion_scale).to(dtype=pred_boxes_det.dtype)
                * velocity_det[:, :2]
            )
            xy_offset = xy_offset.clamp(-float(self.mcip_motion_offset_cap), float(self.mcip_motion_offset_cap))
            motion_offset[:, :2] = xy_offset
            next_boxes = (pred_boxes_det + motion_offset).clamp(1e-4, 1.0 - 1e-4)
        track_instances.ref_pts = inverse_sigmoid(next_boxes)

        track_instances.img_memory = self._detach_if_needed(img_memory_new)
        track_instances.semantic_memory = self._detach_if_needed(semantic_memory_new)
        track_instances.cls_conf_memory = self._detach_if_needed(cls_conf_memory_new)
        track_instances.cls_entropy_memory = self._detach_if_needed(cls_entropy_memory_new)
        track_instances.prev_boxes = pred_boxes_det.clone()
        track_instances.box_velocity = self._detach_if_needed(box_velocity_new)
        track_instances.memory_age = self._detach_if_needed(memory_age_new)

        if getattr(self, 'debug_mcip', False):
            with torch.no_grad():
                eps = 1e-6
                base_norm = base_query_tgt.detach().float().norm(dim=-1).clamp_min(eps)
                delta_norm_ratio = delta.detach().float().norm(dim=-1) / base_norm
                residual_cosine_to_base = F.cosine_similarity(
                    delta.detach().float(),
                    base_query_tgt.detach().float(),
                    dim=-1,
                ).mean()
                query_tgt_cosine = F.cosine_similarity(
                    track_instances.query_tgt.detach().float(),
                    base_query_tgt.detach().float(),
                    dim=-1,
                ).mean()
                relative_query_tgt_l2_error = (
                    (track_instances.query_tgt.detach().float() - base_query_tgt.detach().float()).norm()
                    / base_query_tgt.detach().float().norm().clamp_min(eps)
                )
                motion_offset_abs = motion_offset.detach().float().abs()

                self._update_debug_stats(
                    residual_norm_ratio_mean=delta_norm_ratio.mean(),
                    residual_norm_ratio_max=delta_norm_ratio.max(),
                    residual_cosine_to_base=residual_cosine_to_base,
                    query_tgt_cosine_to_baseline=query_tgt_cosine,
                    relative_query_tgt_l2_error=relative_query_tgt_l2_error,
                    gate_saturation_frac=0.0,
                    memory_update_gate_mean=update_gate.detach().float().mean(),
                    memory_update_gate_max=update_gate.detach().float().max(),
                    memory_update_gate_min=update_gate.detach().float().min(),
                    inject_gate_mean=inject_gate.detach().float().mean(),
                    inject_gate_max=inject_gate.detach().float().max(),
                    inject_gate_min=inject_gate.detach().float().min(),
                    motion_offset_l1_mean=motion_offset_abs.mean(),
                    motion_offset_l1_max=motion_offset_abs.max(),
                    motion_scale=torch.tanh(self.motion_scale.detach()),
                    cls_conf_obs_mean=cls_conf_obs.detach().float().mean(),
                    cls_conf_obs_min=cls_conf_obs.detach().float().min(),
                    cls_conf_obs_max=cls_conf_obs.detach().float().max(),
                    cls_entropy_norm_mean=cls_entropy_norm.detach().float().mean(),
                    cls_entropy_norm_min=cls_entropy_norm.detach().float().min(),
                    cls_entropy_norm_max=cls_entropy_norm.detach().float().max(),
                    current_reliability_mean=current_reliability.detach().float().mean(),
                    memory_reliability_mean=memory_reliability.detach().float().mean(),
                    motion_reliability_mean=motion_reliability.detach().float().mean(),
                    motion_reliability_max=motion_reliability.detach().float().max(),
                    active_track_count=len(track_instances),
                )
        return track_instances


def build(args, layer_name, dim_in, hidden_dim, dim_out):
    embedding_layers = {
        'CIP': Category_Information_Propagator,
        'QIM': QueryInteractionModule,
    }
    assert layer_name in embedding_layers, 'invalid updater: {}'.format(layer_name)
    if layer_name == 'CIP' and getattr(args, 'mcip_enable', False):
        return MemoryCalibratedCategoryInformationPropagator(args, dim_in, hidden_dim, dim_out)
    return embedding_layers[layer_name](args, dim_in, hidden_dim, dim_out)
