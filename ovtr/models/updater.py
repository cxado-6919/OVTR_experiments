# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from torch import nn
from torch.nn import functional as F

from util import box_ops
from util.misc import inverse_sigmoid
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
        
        # if args.update_query_pos:
        # self.dropout3 = nn.Dropout(dropout)
        # self.dropout4 = nn.Dropout(dropout)

        self.activation = F.relu

    def _aggregate_category_info(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
            return track_instances
        
        out_embed_img = track_instances.output_embedding_img
        query_pos = track_instances.query_pos
        query_feat = track_instances.query_tgt
        q = k = query_pos + out_embed_img
        tgt = out_embed_img
        
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_tgt = query_feat

        track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :4].detach().clone())
        return track_instances
    
    def _select_active_tracks(self, data: dict, id_gt=None) -> Instances:
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
    
    def forward(self, data, id_gt=None) -> Instances:
        active_track_instances = self._select_active_tracks(data, id_gt=id_gt)
        active_track_instances = self._aggregate_category_info(active_track_instances)
        init_track_instances: Instances = data['init_track_instances']
        merged_track_instances = Instances.cat([init_track_instances, active_track_instances])
        return merged_track_instances


class MemoryCalibratedCategoryInformationPropagator(Category_Information_Propagator):
    """M-CIP: opt-in category propagation with detached compact track memory."""

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        super()._build_layers(args, dim_in, hidden_dim, dim_out)
        self.mcip_detach_memory = getattr(args, 'mcip_detach_memory', True)
        self.mcip_memory_momentum = getattr(args, 'mcip_memory_momentum', 0.8)
        self.mcip_use_semantic_memory = getattr(args, 'mcip_use_semantic_memory', True)
        self.mcip_use_motion_ref = getattr(args, 'mcip_use_motion_ref', True)
        self.mcip_motion_momentum = getattr(args, 'mcip_motion_momentum', 0.7)
        self.debug_mcip = getattr(args, 'debug_mcip', False)

        gate_input_dim = dim_in * 4 + 3 + 4
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.memory_img_proj = nn.Linear(dim_in, dim_in)
        self.memory_sem_proj = nn.Linear(dim_in, dim_in)
        self.motion_scale = nn.Parameter(
            torch.tensor(float(getattr(args, 'mcip_motion_scale_init', 0.0)))
        )
        self.last_debug_stats = {}

    def _reset_parameters(self):
        super()._reset_parameters()
        nn.init.zeros_(self.memory_img_proj.weight)
        nn.init.zeros_(self.memory_img_proj.bias)
        nn.init.zeros_(self.memory_sem_proj.weight)
        nn.init.zeros_(self.memory_sem_proj.bias)

    def _detach_if_needed(self, tensor):
        return tensor.detach() if self.mcip_detach_memory else tensor

    def _get_observation(self, track_instances: Instances, name: str, like: torch.Tensor) -> torch.Tensor:
        if track_instances.has(name):
            value = track_instances.get(name)
            if value.shape[0] == len(track_instances):
                return value.to(device=like.device, dtype=like.dtype)
        return torch.zeros((len(track_instances),) + tuple(like.shape[1:]), device=like.device, dtype=like.dtype)

    def _aggregate_category_info(self, track_instances: Instances) -> Instances:
        if len(track_instances) == 0:
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
        scores = track_instances.scores.to(dtype=out_embed_img.dtype)

        gate_input = torch.cat([
            out_embed_img,
            img_memory_old,
            semantic_obs,
            semantic_memory_old,
            cls_conf_obs[:, None],
            cls_entropy_obs[:, None],
            scores[:, None],
            box_velocity_old,
        ], dim=-1)
        gate = self.gate_mlp(gate_input)
        # Momentum damps the learned overwrite gate so memory changes slowly by default.
        effective_gate = gate * (1.0 - float(self.mcip_memory_momentum))

        img_memory_new = (1.0 - effective_gate) * img_memory_old + effective_gate * out_embed_img
        semantic_memory_new = (
            (1.0 - effective_gate) * semantic_memory_old + effective_gate * semantic_obs
            if self.mcip_use_semantic_memory
            else semantic_memory_old
        )
        gate_scalar = effective_gate.squeeze(-1)
        cls_conf_memory_new = (
            (1.0 - gate_scalar) * track_instances.cls_conf_memory + gate_scalar * cls_conf_obs
        )
        cls_entropy_memory_new = (
            (1.0 - gate_scalar) * track_instances.cls_entropy_memory + gate_scalar * cls_entropy_obs
        )

        pred_boxes = self._detach_if_needed(track_instances.pred_boxes[:, :4])
        prev_boxes_old = track_instances.prev_boxes
        box_delta = pred_boxes - prev_boxes_old
        box_velocity_new = (
            float(self.mcip_motion_momentum) * box_velocity_old
            + (1.0 - float(self.mcip_motion_momentum)) * box_delta
        )
        memory_age_new = track_instances.memory_age + 1

        query_pos = track_instances.query_pos
        query_feat = track_instances.query_tgt
        mcip_img = out_embed_img + self.memory_img_proj(img_memory_new) + self.memory_sem_proj(semantic_memory_new)
        q = k = query_pos + mcip_img
        tgt = mcip_img

        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_tgt = query_feat

        if self.mcip_use_motion_ref:
            next_boxes = pred_boxes + self.motion_scale * box_velocity_new
            next_boxes = next_boxes.clamp(1e-4, 1.0 - 1e-4)
            track_instances.ref_pts = inverse_sigmoid(next_boxes.detach().clone())
        else:
            track_instances.ref_pts = inverse_sigmoid(
                track_instances.pred_boxes[:, :4].detach().clone().clamp(1e-4, 1.0 - 1e-4)
            )

        track_instances.img_memory = self._detach_if_needed(img_memory_new)
        track_instances.semantic_memory = self._detach_if_needed(semantic_memory_new)
        track_instances.cls_conf_memory = self._detach_if_needed(cls_conf_memory_new)
        track_instances.cls_entropy_memory = self._detach_if_needed(cls_entropy_memory_new)
        track_instances.prev_boxes = self._detach_if_needed(pred_boxes).clone()
        track_instances.box_velocity = self._detach_if_needed(box_velocity_new)
        track_instances.memory_age = self._detach_if_needed(memory_age_new)

        if self.debug_mcip:
            self.last_debug_stats = {
                'avg_gate': gate.detach().mean().item(),
                'avg_cls_entropy_memory': track_instances.cls_entropy_memory.detach().mean().item(),
                'avg_box_velocity_norm': track_instances.box_velocity.detach().norm(dim=-1).mean().item(),
                'num_active_tracks': len(track_instances),
            }
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
