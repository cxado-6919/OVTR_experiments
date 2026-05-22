# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
"""
DETR model and criterion classes.
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import List
import copy
import math
from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid, all_reduce_tensor,)

from detectron2.structures import Instances, Boxes, matched_boxlist_iou
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .updater import build as build_updater
from .deformable_detr import SetCriterion
from .segmentation import sigmoid_focal_loss
from .quant_utils import maybe_get_quantized_embedding_weight

from util.clip_utils import load_embeddings
from .utils import MLP, protect_det_preds, protect_track_preds, preprocess_for_masks
from util.list_LVIS import Frequency_list_total_1, Frequency_list_70, novel_class


ATTENTION_PROTECTION_OPTION_DEFAULTS = {
    'attention_protection_mode': 'kl',
    'attention_protection_topk': 3,
    'attention_protection_conf_thresh': 0.25,
}

OV_DPTD_OPTION_DEFAULTS = {
    'use_ov_dptd': False,
    'ov_dptd_use_historical_offsets': True,
    'ov_dptd_fusion': 'linear_sum',
    'ov_dptd_id_path_text': 'none',
    'ov_dptd_fuse_cti': False,
    'ov_dptd_store_debug': False,
    'use_dptd_update_suppression': False,
    'dptd_update_suppression_thresh': 0.4,
    'dptd_update_suppression_restore_fields': [
        'query_tgt',
        'query_pos',
        'ref_pts',
        'dptd_sampling_offsets',
        'output_embedding_img',
        'output_embedding_txt',
    ],
    'dptd_update_suppression_track_id_based': True,
    'use_dptd_semantic_memory': False,
    'dptd_memory_ema': 0.8,
    'dptd_memory_min_score': 0.4,
    'dptd_memory_max_entropy': 0.75,
    'dptd_memory_use_alignment_feature': True,
    'dptd_memory_allow_untrained_visual_projection': False,
    'dptd_memory_store_topk': 5,
    'dptd_memory_debug': False,
    'use_dptd_semantic_gate': False,
    'dptd_gate_mode': 'heuristic',
    'dptd_gate_min_score': 0.3,
    'dptd_gate_max_entropy': 0.8,
    'dptd_gate_semantic_cos_tau': 0.25,
    'dptd_gate_visual_cos_tau': 0.25,
    'dptd_gate_offset_tau': 0.2,
    'dptd_gate_box_iou_tau': 0.3,
    'dptd_gate_temperature': 10.0,
    'dptd_gate_min_appearance': 0.1,
    'dptd_gate_debug': False,
    'use_dptd_semantic_update_suppression': False,
    'dptd_semantic_update_suppression_thresh': 0.3,
    'ov_dptd_semantic_gate_id_proj_init': 'small_random',
    'ov_dptd_semantic_gate_id_proj_init_std': 1e-3,
    'ov_dptd_reinit_dead_semantic_gate_id_proj': False,
}

DPTD_MEMORY_FIELDS = (
    'dptd_semantic_proto',
    'dptd_visual_memory',
    'dptd_semantic_conf',
    'dptd_semantic_entropy',
    'dptd_memory_age',
    'dptd_topk_class_indices',
    'dptd_topk_class_scores',
)

DPTD_CURRENT_MEMORY_FIELDS = (
    '_dptd_current_semantic_proto',
    '_dptd_current_visual_memory',
    '_dptd_current_semantic_conf',
    '_dptd_current_semantic_entropy',
    '_dptd_current_topk_class_indices',
    '_dptd_current_topk_class_scores',
)

DPTD_TEMP_FIELDS = DPTD_CURRENT_MEMORY_FIELDS + (
    '_dptd_current_gate',
    '_dptd_current_gate_raw',
    '_dptd_current_gate_memory_valid',
    '_dptd_current_visual_consistency',
)


def _validate_dptd_memory_options(container):
    if not getattr(container, 'use_dptd_semantic_memory', False):
        return
    if not getattr(container, 'use_ov_dptd', False):
        raise RuntimeError('DPTD semantic memory requires use_ov_dptd=True.')
    if (
        not getattr(container, 'dptd_memory_use_alignment_feature', True)
        and not getattr(container, 'dptd_memory_allow_untrained_visual_projection', False)
    ):
        raise RuntimeError(
            'DPTD semantic memory needs alignment features or '
            'dptd_memory_allow_untrained_visual_projection=True.'
        )
    ema = float(getattr(container, 'dptd_memory_ema', 0.8))
    if ema < 0.0 or ema >= 1.0:
        raise RuntimeError('dptd_memory_ema must satisfy 0 <= ema < 1.')
    min_score = float(getattr(container, 'dptd_memory_min_score', 0.4))
    max_entropy = float(getattr(container, 'dptd_memory_max_entropy', 0.75))
    if min_score < 0.0 or min_score > 1.0:
        raise RuntimeError('dptd_memory_min_score must be in [0, 1].')
    if max_entropy < 0.0 or max_entropy > 1.0:
        raise RuntimeError('dptd_memory_max_entropy must be in [0, 1].')
    if int(getattr(container, 'dptd_memory_store_topk', 5)) < 1:
        raise RuntimeError('dptd_memory_store_topk must be >= 1.')


def _validate_dptd_gate_options(container):
    if getattr(container, 'ov_dptd_fuse_cti', False):
        raise NotImplementedError('OV-DPTD v4 keeps CTI fusion disabled; ov_dptd_fuse_cti=True is unsupported.')
    if getattr(container, 'dptd_gate_mode', 'heuristic') != 'heuristic':
        raise NotImplementedError("OV-DPTD v4 only supports dptd_gate_mode='heuristic'.")
    if getattr(container, 'use_dptd_semantic_update_suppression', False) and not getattr(container, 'use_dptd_update_suppression', False):
        raise RuntimeError('DPTD semantic update suppression requires use_dptd_update_suppression=True.')
    if getattr(container, 'ov_dptd_fusion', 'linear_sum') == 'semantic_gate' and not getattr(container, 'use_dptd_semantic_gate', False):
        raise RuntimeError("ov_dptd_fusion='semantic_gate' requires use_dptd_semantic_gate=True.")
    if not getattr(container, 'use_dptd_semantic_gate', False):
        return
    if not getattr(container, 'use_ov_dptd', False):
        raise RuntimeError('DPTD semantic gate requires use_ov_dptd=True.')
    if not getattr(container, 'use_dptd_semantic_memory', False):
        raise RuntimeError('DPTD semantic gate requires use_dptd_semantic_memory=True.')
    if getattr(container, 'ov_dptd_fusion', 'linear_sum') not in ('linear_sum', 'semantic_gate'):
        raise NotImplementedError("OV-DPTD semantic gate supports ov_dptd_fusion in {'linear_sum', 'semantic_gate'}.")
    init_mode = getattr(container, 'ov_dptd_semantic_gate_id_proj_init', 'small_random')
    if init_mode not in ('small_random', 'zero'):
        raise RuntimeError("ov_dptd_semantic_gate_id_proj_init must be one of {'small_random', 'zero'}.")
    init_std = float(getattr(container, 'ov_dptd_semantic_gate_id_proj_init_std', 1e-3))
    if init_mode == 'small_random' and init_std <= 0.0:
        raise RuntimeError('ov_dptd_semantic_gate_id_proj_init_std must be > 0 for small_random init.')
    if init_std < 0.0:
        raise RuntimeError('ov_dptd_semantic_gate_id_proj_init_std must be >= 0.')
    for name in [
        'dptd_gate_min_score',
        'dptd_gate_max_entropy',
        'dptd_gate_semantic_cos_tau',
        'dptd_gate_visual_cos_tau',
        'dptd_gate_offset_tau',
        'dptd_gate_box_iou_tau',
        'dptd_gate_temperature',
        'dptd_gate_min_appearance',
        'dptd_semantic_update_suppression_thresh',
    ]:
        value = float(getattr(container, name))
        if name.endswith('_tau') or name == 'dptd_gate_temperature':
            if value <= 0.0:
                raise RuntimeError(f'{name} must be > 0.')
        elif value < 0.0 or value > 1.0:
            raise RuntimeError(f'{name} must be in [0, 1].')


def resolve_attention_protection_options(args, cfg):
    """Apply config/CLI/default attention protection options to both args and cfg."""
    for name, default in ATTENTION_PROTECTION_OPTION_DEFAULTS.items():
        value = getattr(args, name, None)
        if value is None:
            value = getattr(cfg, name, default)
        setattr(args, name, value)
        setattr(cfg, name, value)


def resolve_ov_dptd_options(args, cfg):
    """Apply config/CLI/default OV-DPTD options and reject unsupported v1 combinations."""
    for name, default in OV_DPTD_OPTION_DEFAULTS.items():
        value = getattr(args, name, None)
        if value is None:
            value = getattr(cfg, name, default)
        setattr(args, name, value)
        setattr(cfg, name, value)

    if getattr(args, 'use_dptd_update_suppression', False) and not getattr(args, 'use_ov_dptd', False):
        raise RuntimeError('DPTD update suppression requires use_ov_dptd=True.')
    if (
        getattr(args, 'use_dptd_update_suppression', False)
        and not getattr(args, 'dptd_update_suppression_track_id_based', True)
    ):
        raise RuntimeError('DPTD update suppression v2 requires track-id based restoration.')
    _validate_dptd_memory_options(args)
    _validate_dptd_gate_options(args)

    if not getattr(args, 'use_ov_dptd', False):
        return

    if getattr(cfg, 'use_checkpoint_track', False):
        raise RuntimeError('OV-DPTD v1 does not support use_checkpoint_track=True.')
    if getattr(cfg, 'use_transformer_ckpt', False):
        raise RuntimeError('OV-DPTD v1 does not support use_transformer_ckpt=True.')
    if getattr(args, 'quant_deploy', 'none') == 'int_msda':
        raise RuntimeError('OV-DPTD v1 does not support --quant_deploy int_msda.')
    if getattr(args, 'ov_dptd_fusion', 'linear_sum') not in ('linear_sum', 'semantic_gate'):
        raise NotImplementedError("OV-DPTD supports ov_dptd_fusion in {'linear_sum', 'semantic_gate'}.")
    if getattr(args, 'ov_dptd_id_path_text', 'none') != 'none':
        raise NotImplementedError("OV-DPTD v1 only supports ov_dptd_id_path_text='none'.")
    init_mode = getattr(args, 'ov_dptd_semantic_gate_id_proj_init', 'small_random')
    if init_mode not in ('small_random', 'zero'):
        raise RuntimeError("ov_dptd_semantic_gate_id_proj_init must be one of {'small_random', 'zero'}.")
    init_std = float(getattr(args, 'ov_dptd_semantic_gate_id_proj_init_std', 1e-3))
    if init_mode == 'small_random' and init_std <= 0.0:
        raise RuntimeError('ov_dptd_semantic_gate_id_proj_init_std must be > 0 for small_random init.')
    if init_std < 0.0:
        raise RuntimeError('ov_dptd_semantic_gate_id_proj_init_std must be >= 0.')

class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, processor_dct=None):
        super().__init__()
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        prob = out_logits.sigmoid()
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes.clamp(0, 1)  
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = labels

        track_instances.remove('pred_logits')
        track_instances.remove('pred_boxes')
        return track_instances

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.6, miss_tolerance=5, maximum_quantity=50):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0
        self.maximum_quantity = maximum_quantity

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances, _track_discard, is_repeat=False):
        cancel_disappear = track_instances.scores >= self.score_thresh
        cancel_disappear[_track_discard] = False
        track_instances.disappear_time[cancel_disappear] = 0
        # Found valid index
        score_indx = track_instances.scores >= self.score_thresh
        obj_indx = track_instances.obj_idxes != -1 
        valid_indx = score_indx | obj_indx
        track_instances = track_instances[valid_indx]

        if len(track_instances) > self.maximum_quantity:
            top_indices = self.quantity_filter(track_instances, self.maximum_quantity) 
            track_instances = track_instances[top_indices]

        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -2:
                continue
            elif track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # print("track {} has score {:.2f}, assign obj_id {}, cls is {}".format(i, track_instances.scores[i], self.max_obj_id, track_instances.cls_idxes[i]))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh and is_repeat is False:
                track_instances.disappear_time[i] += 1
                # print(track_instances.obj_idxes[i])
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1
                    # print("track {} has score {:.2f}, disappear".format(i, track_instances.scores[i], self.max_obj_id))
            # elif (track_instances.obj_idxes[i] >= 0) and (track_instances.scores[i] >= self.filter_score_thresh) and (track_instances.keep_cls[i] == False):
            #     print("track {} keeps origin obj_id {}, cls changes to {}".format(i, track_instances.obj_idxes[i], track_instances.cls_idxes[i]))
                # track_instances.obj_idxes[i] = self.max_obj_id
                # self.max_obj_id += 1
        return track_instances
    
    @staticmethod
    def quantity_filter(track_instances, maximum_quantity):
        scores = track_instances.scores
        _, top_indices = torch.topk(scores, k=maximum_quantity, sorted=False)
        top_indices = torch.sort(top_indices).values
        return top_indices


class OVFrameMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses,
                        random_drop=0,
                        calculate_negative_samples=True,
                        num_queries=900,
                        train_with_artificial_img_seqs=False,
                        ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses,)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.random_drop = random_drop

        self.num_queries = num_queries
        self.calculate_negative_samples = calculate_negative_samples
        self.train_with_artificial_img_seqs = train_with_artificial_img_seqs

    def initialize(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.gt_instances_batch = None
        self._current_frame_idx_batch = None
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def initialize_batch(self, gt_instances_batch: List[List[Instances]]):
        self.gt_instances = None
        self.gt_instances_batch = gt_instances_batch
        self._current_frame_idx_batch = [0 for _ in gt_instances_batch]
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self, sample_idx=None):
        if sample_idx is None:
            self._current_frame_idx += 1
        else:
            self._current_frame_idx_batch[sample_idx] += 1

    def _get_current_gt_instances(self, sample_idx=None):
        if sample_idx is None:
            return self.gt_instances[self._current_frame_idx]
        return self.gt_instances_batch[sample_idx][self._current_frame_idx_batch[sample_idx]]

    def _accumulate_losses(self, prefix, loss_dict):
        for key, value in loss_dict.items():
            loss_key = f"{prefix}{key}"
            if loss_key in self.losses_dict:
                self.losses_dict[loss_key] = self.losses_dict[loss_key] + value
            else:
                self.losses_dict[loss_key] = value

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            num_boxes = all_reduce_tensor(num_boxes, average=True)
        else:
            num_boxes = num_boxes / get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            "align": self.loss_align,
            "align_pre": self.loss_align_pre,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']

        def get_src_permutation_idx(indices):
            batch_idx = torch.cat([torch.full_like(src[filt != -1], i) for i, (src, filt) in enumerate(indices)])
            src_idx = torch.cat([src[filt != -1] for (src, filt) in indices])
            return batch_idx, src_idx
        idx = get_src_permutation_idx(indices)
        
        if self.calculate_negative_samples:
            num_class = len(outputs["select_id"])
        else:
            num_class = len(torch.unique(gt_instances[0].labels))
        
        target_classes = torch.full(src_logits.shape[:2], num_class,
                                    dtype=torch.int64, device=src_logits.device)

        select_id = outputs["select_id"]
        labels_ori = torch.cat([t.labels[J[J != -1]] for t, (_, J) in zip(gt_instances, indices)])
        tgt_ids_all = torch.cat([(select_id == lid).nonzero(as_tuple=False)[0] for lid in labels_ori])
        target_classes[idx] = tgt_ids_all
      
        if self.focal_loss:                                                                    
            gt_labels_target = F.one_hot(target_classes, num_classes=num_class + 1)[:, :,
                               :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits[:, :, :num_class],
                                             gt_labels_target,
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=True)* src_logits.shape[1]
        else:
            loss_ce = F.cross_entropy(src_logits[:, :, num_class].transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses
    
    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
    
    def loss_align(self, outputs, targets, indices, num_boxes, l1_distillation=False):
        """Alignment mechanism guides generalization capabilities and aligned queries.
        """
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx

        idx = self._get_src_permutation_idx(indices)
        src_feature = outputs["pred_embed"][idx]

        select_id = outputs["select_id"]
        image_feat = outputs["image_feat"]
        target_feature = []
        for t, (_, i) in zip(targets, indices):
            for c in t.labels[i]:
                index = (select_id == c).nonzero(as_tuple=False)[0]
                target_feature.append(image_feat[index])
        target_feature = torch.cat(target_feature, dim=0)
        # l1 normalize the feature
        src_feature = nn.functional.normalize(src_feature, dim=1)
        if l1_distillation:
            loss_feature = F.l1_loss(src_feature, target_feature, reduction="none")
        else:
            loss_feature = F.mse_loss(src_feature, target_feature, reduction="none")
        losses = {"loss_align": loss_feature.sum() / num_boxes}
        return losses
    
    def loss_align_pre(self, outputs, targets, indices, num_boxes):
        """Preserve text features without sudden variations.
        """
        input_feat = outputs["input_feat"]
        loss_feature_all = []

        select_id = outputs["select_id"]
        uniq_labels = [torch.unique(t.labels) for t in targets]
        tgt_ids_all = []
        embed_bs_index = []
        for i, uniq_label in enumerate(uniq_labels):
            tgt_ids = []
            if len(uniq_label)==0:
                continue
            else:
                embed_bs_index.append(i)
                for lid in uniq_label:
                    index = (select_id == lid).nonzero(as_tuple=False)[0]
                    tgt_ids.append(index)
                tgt_ids = torch.cat(tgt_ids)
            tgt_ids_all.append(tgt_ids)
        
        input_feats = torch.cat([input_feat[i] for i in tgt_ids_all])
        encoder_embeds = outputs["text_embed"][:, embed_bs_index]

        for encoder_embed in encoder_embeds:
            src_feature = torch.cat([enc_embed[tgt_id] for enc_embed, tgt_id in zip(encoder_embed,tgt_ids_all)])
            # l2 normalize the feature
            src_feature = nn.functional.normalize(src_feature, dim=1)
            loss_feature = F.mse_loss(src_feature, input_feats, reduction="none")
            loss_feature_all.append(loss_feature.sum() / num_boxes)
        loss_feature_all = torch.stack(loss_feature_all)
        loss_encoder_align = loss_feature_all.sum()
        losses = {"loss_align_pre": loss_encoder_align}
        return losses
    
    def match_for_single_frame(self, outputs: dict, is_first=None, sample_idx=None):
        outputs_without_aux = {k: v for k, v in outputs.items() if 
                               k != 'aux_outputs' and k != 'enc_outputs'}

        def select_unmatched_indexes(matched_indexes: torch.Tensor, num_total_indexes: int) -> torch.Tensor:
            matched_indexes_set = set(matched_indexes.detach().cpu().numpy().tolist())
            all_indexes_set = set(list(range(num_total_indexes)))
            unmatched_indexes_set = all_indexes_set - matched_indexes_set
            unmatched_indexes = torch.as_tensor(list(unmatched_indexes_set), dtype=torch.long).to(matched_indexes)
            return unmatched_indexes

        gt_instances_i = self._get_current_gt_instances(sample_idx)  # gt instances of i-th image.
        track_instances_last: Instances = outputs_without_aux['track_instances']

        if self.train_with_artificial_img_seqs:
            shielded_ids = protect_det_preds(outputs_without_aux, num_queries=self.num_queries)
            keep_indices = torch.ones(len(track_instances_last), dtype=torch.bool, device=shielded_ids.device)
            keep_indices[shielded_ids] = False
            track_instances = track_instances_last[keep_indices]
        else:
            keep_indices = torch.ones(len(track_instances_last), dtype=torch.bool, device=track_instances_last.obj_idxes.device)
            track_instances = track_instances_last

        outputs_i = {
            'pred_logits': track_instances.pred_logits.unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes.unsqueeze(0),
            'pred_embed': outputs_without_aux['pred_embed'][0, keep_indices].unsqueeze(0),
            'select_id':outputs_without_aux['select_id'],
            'image_feat':outputs_without_aux['image_feat'],
        }

        obj_idxes = gt_instances_i.obj_ids
        device = obj_idxes.device
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        track_instances.matched_gt_idxes[:] = -1
        valid_track_mask = track_instances.obj_idxes >= 0
        valid_track_idxes = torch.arange(len(track_instances), device=device)[valid_track_mask]
        valid_obj_idxes = track_instances.obj_idxes[valid_track_idxes]
        for j in range(len(valid_obj_idxes)):
            obj_id = valid_obj_idxes[j].item()
            if obj_id in obj_idx_to_gt_idx:
                track_instances.matched_gt_idxes[valid_track_idxes[j]] = obj_idx_to_gt_idx[obj_id]
            else:
                num_disappear_track += 1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long, device=device)
        matched_track_idxes = (track_instances.obj_idxes >= 0) # occu 
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(device)

        # step2. select the unmatched slots.
        # note that the fp tracks (obj_idxes == -2) will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the unmatched gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        unmatched_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        unmatched_gt_instances = gt_instances_i[unmatched_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher, unmatched_track_idxes):
            new_track_indices = matcher(unmatched_outputs,
                                             [unmatched_gt_instances])

            # map the matched pair indexes to original index-space.
            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt for loss calculation.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], unmatched_tgt_indexes[tgt_idx]],
                                              dim=1).to(device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
            'select_id':outputs_without_aux['select_id'],
        }

        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher, unmatched_track_idxes)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0) 

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = device

        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self._accumulate_losses(f'frame_{self._current_frame_idx if sample_idx is None else self._current_frame_idx_batch[sample_idx]}_', new_track_loss)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):

                # Shield and match individually for each layer.
                if self.train_with_artificial_img_seqs:
                    _shielded_ids_layer = protect_det_preds(aux_outputs, num_queries=self.num_queries)
                    _keep_indices_layer = torch.ones(len(track_instances_last), dtype=torch.bool, device=shielded_ids.device)
                    _keep_indices_layer[_shielded_ids_layer] = False
                    track_instances_layer = track_instances_last[_keep_indices_layer]
                else:
                    _keep_indices_layer = torch.ones(len(track_instances_last), dtype=torch.bool, device=track_instances_last.obj_idxes.device)
                    track_instances_layer = track_instances_last

                # step1*. inherit and update the previous tracks.
                track_instances_layer.matched_gt_idxes[:] = -1
                valid_track_mask = track_instances_layer.obj_idxes >= 0
                valid_track_idxes = torch.arange(len(track_instances_layer), device=device)[valid_track_mask]
                valid_obj_idxes = track_instances_layer.obj_idxes[valid_track_idxes]
                for j in range(len(valid_obj_idxes)):
                    obj_id = valid_obj_idxes[j].item()
                    if obj_id in obj_idx_to_gt_idx:
                        track_instances_layer.matched_gt_idxes[valid_track_idxes[j]] = obj_idx_to_gt_idx[obj_id]

                full_track_idxes = torch.arange(len(track_instances_layer), dtype=torch.long, device=device)
                matched_track_idxes_layer = (track_instances_layer.obj_idxes >= 0)
                prev_matched_indices_layer = torch.stack(
                    [full_track_idxes[matched_track_idxes_layer], track_instances_layer.matched_gt_idxes[matched_track_idxes_layer]], dim=1).to(device)

                # step2*. select the unmatched slots.
                unmatched_track_idxes_layer = full_track_idxes[track_instances_layer.obj_idxes == -1]

                # step3*. do matching between the unmatched slots and GTs.
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, _keep_indices_layer][unmatched_track_idxes_layer].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, _keep_indices_layer][unmatched_track_idxes_layer].unsqueeze(0),
                    'select_id': aux_outputs['select_id'],
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher, unmatched_track_idxes_layer)

                # step4*. merge the unmatched pairs and the matched pairs.
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices_layer], dim=0)

                # step5*. calculate losses.
                _keep_aux_outputs = {
                    'pred_logits': aux_outputs['pred_logits'][0, _keep_indices_layer].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, _keep_indices_layer].unsqueeze(0),
                    'pred_embed': aux_outputs['pred_embed'][0, _keep_indices_layer].unsqueeze(0),
                    'select_id': aux_outputs['select_id'],
                    'image_feat': aux_outputs['image_feat'],
                }
                for loss in self.losses:
                    l_dict = self.get_loss(loss,
                                           _keep_aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    frame_idx = self._current_frame_idx if sample_idx is None else self._current_frame_idx_batch[sample_idx]
                    self._accumulate_losses(f'frame_{frame_idx}_aux{i}_', l_dict)
            
        self._step(sample_idx)
        return track_instances

    def forward(self, outputs):
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        loss_avg = {}
        for loss_name, _ in losses.items():
            loss_avg[loss_name] = losses[loss_name] / num_samples
        return loss_avg
    

class OVTR(nn.Module):
    def __init__(self, backbone, transformer, num_feature_levels, criterion, track_embed,
                    aux_loss=True, with_box_refine=False, two_stage=False,
                    two_stage_bbox_embed_share=False,
                    dec_pred_bbox_embed_share=True,
                    use_checkpoint=None,
                    distribution_based_sampling=None,
                    text_embeddings=None,
                    image_embeddings=None,
                    max_len=None,
                    novel_cls_cpu=None,
                    computed_aux=None,
                    score_thresh=None,
                    filter_score_thresh=None,
                    miss_tolerance=None,
                    train_with_artificial_img_seqs=False,
                    use_ov_dptd=False,
                    ov_dptd_store_debug=False,
                    use_dptd_update_suppression=False,
                    dptd_update_suppression_thresh=0.4,
                    dptd_update_suppression_restore_fields=None,
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
                    dptd_gate_mode='heuristic',
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
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        self.num_queries = transformer.num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
     
        self.max_pad_len = max_len
        self.text_embeddings=text_embeddings.t()
        self.image_embeddings=image_embeddings.t()
        self.patch2query = nn.Linear(512, 256)
        self.all_ids = torch.tensor(range(self.text_embeddings.shape[-1]))
        self.all_ids = [i + 1 for i in self.all_ids]
        self.select_id = list(range(0, len(Frequency_list_total_1)))

        self.frequency = torch.tensor(Frequency_list_70, dtype=torch.float32, device='cpu')
        print("0.7 power sampling | Training excludes rare categories.")
        self.frequency_eval = torch.tensor(Frequency_list_total_1, dtype=torch.float32, device='cpu')
        self.novel_cls_cpu = novel_cls_cpu
        self.computed_aux = computed_aux
   
        for layer in [self.patch2query]:
            nn.init.xavier_uniform_(self.patch2query.weight)
            nn.init.constant_(self.patch2query.bias, 0)
        
        # feature alignment
        self.feature_align = nn.Linear(256, 512) # alignment head
        nn.init.xavier_uniform_(self.feature_align.weight)
        nn.init.constant_(self.feature_align.bias, 0)
        num_pred = len(self.computed_aux)
        if with_box_refine:
            self.feature_align = _get_clones(self.feature_align, num_pred)
        else:
            self.feature_align = nn.ModuleList([self.feature_align for _ in range(num_pred)])

        # bbox
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        
        if two_stage:
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.refpoint_embed = None

        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase(score_thresh=score_thresh, 
                                             filter_score_thresh=filter_score_thresh, 
                                             miss_tolerance=miss_tolerance)
        
        self.use_checkpoint = use_checkpoint
        self.distribution_based_sampling = distribution_based_sampling
        self.criterion = criterion
        self.train_with_artificial_img_seqs = train_with_artificial_img_seqs
        self.use_ov_dptd = use_ov_dptd
        self.ov_dptd_store_debug = ov_dptd_store_debug
        self.use_dptd_update_suppression = use_dptd_update_suppression
        self.dptd_update_suppression_thresh = dptd_update_suppression_thresh
        self.dptd_update_suppression_restore_fields = (
            list(dptd_update_suppression_restore_fields)
            if dptd_update_suppression_restore_fields is not None
            else list(OV_DPTD_OPTION_DEFAULTS['dptd_update_suppression_restore_fields'])
        )
        self.dptd_update_suppression_track_id_based = dptd_update_suppression_track_id_based
        self.use_dptd_semantic_memory = use_dptd_semantic_memory
        self.dptd_memory_ema = dptd_memory_ema
        self.dptd_memory_min_score = dptd_memory_min_score
        self.dptd_memory_max_entropy = dptd_memory_max_entropy
        self.dptd_memory_use_alignment_feature = dptd_memory_use_alignment_feature
        self.dptd_memory_allow_untrained_visual_projection = dptd_memory_allow_untrained_visual_projection
        self.dptd_memory_store_topk = int(dptd_memory_store_topk)
        self.dptd_memory_debug = dptd_memory_debug
        self.use_dptd_semantic_gate = use_dptd_semantic_gate
        self.dptd_gate_mode = dptd_gate_mode
        self.dptd_gate_min_score = dptd_gate_min_score
        self.dptd_gate_max_entropy = dptd_gate_max_entropy
        self.dptd_gate_semantic_cos_tau = dptd_gate_semantic_cos_tau
        self.dptd_gate_visual_cos_tau = dptd_gate_visual_cos_tau
        self.dptd_gate_offset_tau = dptd_gate_offset_tau
        self.dptd_gate_box_iou_tau = dptd_gate_box_iou_tau
        self.dptd_gate_temperature = dptd_gate_temperature
        self.dptd_gate_min_appearance = dptd_gate_min_appearance
        self.dptd_gate_debug = dptd_gate_debug
        self.use_dptd_semantic_update_suppression = use_dptd_semantic_update_suppression
        self.dptd_semantic_update_suppression_thresh = dptd_semantic_update_suppression_thresh
        self.dptd_memory_dim = int(self.text_embeddings.shape[0])
        self.dptd_visual_memory_proj = None
        if self.use_dptd_semantic_memory and self.dptd_memory_allow_untrained_visual_projection:
            self.dptd_visual_memory_proj = nn.Linear(hidden_dim, self.dptd_memory_dim)
            nn.init.xavier_uniform_(self.dptd_visual_memory_proj.weight)
            nn.init.constant_(self.dptd_visual_memory_proj.bias, 0)
        if self.use_ov_dptd and self.use_checkpoint:
            raise RuntimeError('OV-DPTD v1 does not support use_checkpoint_track=True.')
        if self.use_dptd_update_suppression and not self.use_ov_dptd:
            raise RuntimeError('DPTD update suppression requires use_ov_dptd=True.')
        if self.use_dptd_update_suppression and not self.dptd_update_suppression_track_id_based:
            raise RuntimeError('DPTD update suppression v2 requires track-id based restoration.')
        _validate_dptd_memory_options(self)
        _validate_dptd_gate_options(self)
        self.supports_mot_batch = (
            (not use_checkpoint)
            and (len(self.transformer.encoder.fusion_layers) == 0)
            and (not self.use_ov_dptd)
        )
        self.ov_dptd_debug_stats = {
            'ov_dptd_enabled': bool(self.use_ov_dptd),
            'ov_dptd_num_track_queries': 0,
            'ov_dptd_historical_offset_used_count': 0,
            'ov_dptd_historical_offset_fallback_count': 0,
            'dptd_update_suppressed_count': 0,
            'dptd_update_suppressed_ids': [],
            'dptd_update_suppression_restore_success_count': 0,
            'dptd_update_suppression_restore_skip_count': 0,
            'dptd_memory_update_count': 0,
            'dptd_memory_keep_count': 0,
            'dptd_memory_init_count': 0,
            'dptd_memory_entropy_mean': 0.0,
            'dptd_memory_semantic_proto_cosine_delta_mean': 0.0,
            'dptd_memory_topk_change_rate': 0.0,
            'dptd_memory_visual_source': 'none',
            'dptd_gate_mean': 1.0,
            'dptd_gate_min': 1.0,
            'dptd_gate_max': 1.0,
            'dptd_gate_track_mean': 1.0,
            'dptd_gate_track_min': 1.0,
            'dptd_gate_track_max': 1.0,
            'dptd_gate_valid_mean': 0.0,
            'dptd_gate_valid_min': 0.0,
            'dptd_gate_valid_max': 0.0,
            'dptd_gate_raw_valid_mean': 0.0,
            'dptd_gate_raw_valid_min': 0.0,
            'dptd_gate_raw_valid_max': 0.0,
            'dptd_gate_low_count': 0,
            'semantic_consistency_mean': 1.0,
            'visual_consistency_mean': 1.0,
            'offset_consistency_mean': 1.0,
            'box_consistency_mean': 1.0,
            'dptd_gate_box_conf_deferred': True,
            'dptd_gate_box_conf_included': False,
            'ov_dptd_gate_alpha': 0.0,
            'ov_dptd_gate_alpha_grad_norm': 0.0,
            'ov_dptd_id_proj_weight_norm': 0.0,
            'ov_dptd_id_proj_grad_norm': 0.0,
            'ov_dptd_id_proj_init_mode': 'none',
        } if self.use_ov_dptd and (
            self.ov_dptd_store_debug
            or self.use_dptd_update_suppression
            or self.dptd_memory_debug
            or self.dptd_gate_debug
        ) else {}

    def _dptd_offset_shape(self, num_queries):
        decoder_layers = getattr(self.transformer.decoder, 'layers', [])
        if len(decoder_layers) == 0:
            raise RuntimeError('OV-DPTD requires at least one decoder layer.')
        cross_attn = decoder_layers[0].cross_attn
        return (
            num_queries,
            cross_attn.num_heads,
            cross_attn.num_levels,
            cross_attn.num_points,
            2,
        )

    def _make_empty_dptd_sampling_offsets(self, num_queries, device, dtype):
        return torch.zeros(self._dptd_offset_shape(num_queries), device=device, dtype=dtype)

    def _attach_dptd_sampling_offsets(self, frame_res, track_instances):
        if not self.use_ov_dptd:
            return
        if 'dptd_sampling_offsets' not in frame_res:
            raise RuntimeError('OV-DPTD decoder did not return dptd_sampling_offsets.')
        offsets = frame_res['dptd_sampling_offsets']
        if offsets.dim() == 6:
            offsets = offsets[0]
        expected_shape = self._dptd_offset_shape(len(track_instances))
        if tuple(offsets.shape) != expected_shape:
            raise RuntimeError(
                'OV-DPTD sampling offset shape mismatch: '
                f'expected {expected_shape}, got {tuple(offsets.shape)}'
            )
        track_instances.dptd_sampling_offsets = offsets.to(
            device=track_instances.query_tgt.device,
            dtype=track_instances.query_tgt.dtype,
        )

    def _update_ov_dptd_debug_stats(self, dptd_info):
        if not (self.use_ov_dptd and self.ov_dptd_store_debug and dptd_info is not None):
            return
        debug = dptd_info.get('debug') if isinstance(dptd_info, dict) else None
        if not debug:
            return
        for key in [
            'ov_dptd_enabled',
            'ov_dptd_num_track_queries',
            'ov_dptd_historical_offset_used_count',
            'ov_dptd_historical_offset_fallback_count',
            'dptd_gate_mean',
            'dptd_gate_min',
            'dptd_gate_max',
            'dptd_gate_track_mean',
            'dptd_gate_track_min',
            'dptd_gate_track_max',
            'dptd_gate_valid_mean',
            'dptd_gate_valid_min',
            'dptd_gate_valid_max',
            'dptd_gate_raw_valid_mean',
            'dptd_gate_raw_valid_min',
            'dptd_gate_raw_valid_max',
            'dptd_gate_low_count',
            'semantic_consistency_mean',
            'visual_consistency_mean',
            'offset_consistency_mean',
            'box_consistency_mean',
            'dptd_gate_box_conf_deferred',
            'dptd_gate_box_conf_included',
            'ov_dptd_gate_alpha',
            'ov_dptd_id_proj_weight_norm',
            'ov_dptd_id_proj_init_mode',
        ]:
            if key in debug:
                self.ov_dptd_debug_stats[key] = debug[key]

    def _reset_dptd_update_suppression_debug(self):
        if not self.ov_dptd_debug_stats:
            return
        self.ov_dptd_debug_stats['dptd_update_suppressed_count'] = 0
        self.ov_dptd_debug_stats['dptd_update_suppressed_ids'] = []
        self.ov_dptd_debug_stats['dptd_update_suppression_restore_success_count'] = 0
        self.ov_dptd_debug_stats['dptd_update_suppression_restore_skip_count'] = 0

    def _add_dptd_update_suppression_debug(self, success_count=0, skip_count=0):
        if not self.ov_dptd_debug_stats:
            return
        self.ov_dptd_debug_stats['dptd_update_suppression_restore_success_count'] += int(success_count)
        self.ov_dptd_debug_stats['dptd_update_suppression_restore_skip_count'] += int(skip_count)

    @staticmethod
    def _validate_dptd_unique_obj_ids(ids, context):
        if not isinstance(ids, torch.Tensor) or ids.numel() == 0:
            return
        valid_ids = ids[ids >= 0].detach().cpu().tolist()
        seen = set()
        duplicates = []
        for track_id in valid_ids:
            track_id = int(track_id)
            if track_id in seen and track_id not in duplicates:
                duplicates.append(track_id)
            seen.add(track_id)
        if duplicates:
            raise RuntimeError(f'DPTD duplicate valid obj_idxes in {context}: {duplicates}')

    def _validate_dptd_track_instance_unique_ids(self, track_instances, context):
        if track_instances is not None and track_instances.has('obj_idxes'):
            self._validate_dptd_unique_obj_ids(track_instances.obj_idxes, context)

    def _snapshot_dptd_update_suppression_state(self, track_instances):
        if not self.use_dptd_update_suppression:
            return None
        if self.training:
            raise RuntimeError('DPTD update suppression is inference-only.')
        if not self.use_ov_dptd:
            raise RuntimeError('DPTD update suppression requires use_ov_dptd=True.')
        if not self.dptd_update_suppression_track_id_based:
            raise RuntimeError('DPTD update suppression v2 requires track-id based restoration.')

        self._reset_dptd_update_suppression_debug()
        if not track_instances.has('obj_idxes') or len(track_instances) == 0:
            return {'ids': torch.empty(0, dtype=torch.long), 'fields': {}, 'suppressed_ids': torch.empty(0, dtype=torch.long)}

        self._validate_dptd_track_instance_unique_ids(track_instances, 'suppression snapshot')
        valid_mask = track_instances.obj_idxes >= 0
        old_ids = track_instances.obj_idxes[valid_mask].detach().clone()
        snapshot = {
            'ids': old_ids,
            'fields': {},
            'suppressed_ids': old_ids.new_empty((0,)),
        }
        if old_ids.numel() == 0:
            return snapshot

        for field_name in self.dptd_update_suppression_restore_fields:
            if not track_instances.has(field_name):
                continue
            value = track_instances.get(field_name)
            if not isinstance(value, torch.Tensor) or value.shape[0] != len(track_instances):
                continue
            snapshot['fields'][field_name] = value[valid_mask].detach().clone()
        return snapshot

    def _select_dptd_update_suppressed_ids(self, snapshot, track_instances):
        if snapshot is None:
            return None
        if snapshot['ids'].numel() == 0 or not track_instances.has('obj_idxes') or not track_instances.has('scores'):
            snapshot['suppressed_ids'] = snapshot['ids'].new_empty((0,))
            return snapshot['suppressed_ids']

        suppressed_ids = []
        self._validate_dptd_unique_obj_ids(snapshot['ids'], 'suppression snapshot ids')
        self._validate_dptd_track_instance_unique_ids(track_instances, 'suppression selection')
        current_ids = track_instances.obj_idxes
        current_scores = track_instances.scores
        for track_id in snapshot['ids'].detach().cpu().tolist():
            matches = torch.nonzero(current_ids == int(track_id), as_tuple=False).flatten()
            if matches.numel() == 0:
                continue
            current_idx = int(matches[0].item())
            if float(current_scores[current_idx].detach().item()) < float(self.dptd_update_suppression_thresh):
                suppressed_ids.append(int(track_id))

        device = current_ids.device
        dtype = current_ids.dtype
        snapshot['suppressed_ids'] = torch.tensor(suppressed_ids, device=device, dtype=dtype)
        if self.ov_dptd_debug_stats:
            self.ov_dptd_debug_stats['dptd_update_suppressed_count'] = len(suppressed_ids)
            self.ov_dptd_debug_stats['dptd_update_suppressed_ids'] = suppressed_ids
        return snapshot['suppressed_ids']

    def _restore_dptd_update_suppressed_state(self, snapshot, track_instances):
        if snapshot is None:
            return track_instances
        suppressed_ids = snapshot.get('suppressed_ids')
        if suppressed_ids is None or suppressed_ids.numel() == 0:
            return track_instances
        if not track_instances.has('obj_idxes'):
            self._add_dptd_update_suppression_debug(skip_count=len(snapshot.get('fields', {})) * int(suppressed_ids.numel()))
            return track_instances

        success_count = 0
        skip_count = 0
        self._validate_dptd_unique_obj_ids(snapshot['ids'], 'suppression restore snapshot ids')
        self._validate_dptd_track_instance_unique_ids(track_instances, 'suppression restore')
        current_ids = track_instances.obj_idxes
        snapshot_ids = snapshot['ids'].to(device=current_ids.device, dtype=current_ids.dtype)
        for track_id in suppressed_ids.to(device=current_ids.device, dtype=current_ids.dtype).detach().cpu().tolist():
            current_matches = torch.nonzero(current_ids == int(track_id), as_tuple=False).flatten()
            snapshot_matches = torch.nonzero(snapshot_ids == int(track_id), as_tuple=False).flatten()
            if current_matches.numel() == 0 or snapshot_matches.numel() == 0:
                skip_count += max(1, len(snapshot.get('fields', {})))
                continue
            current_idx = int(current_matches[0].item())
            snapshot_idx = int(snapshot_matches[0].item())

            for field_name, old_values in snapshot.get('fields', {}).items():
                if not track_instances.has(field_name):
                    skip_count += 1
                    continue
                current_value = track_instances.get(field_name)
                if not isinstance(current_value, torch.Tensor) or current_value.shape[0] != len(track_instances):
                    skip_count += 1
                    continue
                old_value = old_values[snapshot_idx].to(device=current_value.device, dtype=current_value.dtype)
                if tuple(current_value[current_idx].shape) != tuple(old_value.shape):
                    skip_count += 1
                    continue
                current_value[current_idx] = old_value
                success_count += 1

        self._add_dptd_update_suppression_debug(success_count=success_count, skip_count=skip_count)
        return track_instances

    def _ensure_dptd_memory_fields(self, track_instances):
        if not self.use_dptd_semantic_memory:
            return track_instances
        num_tracks = len(track_instances)
        if track_instances.has('query_tgt'):
            device = track_instances.query_tgt.device
            dtype = track_instances.query_tgt.dtype
        elif track_instances.has('scores'):
            device = track_instances.scores.device
            dtype = track_instances.scores.dtype
        else:
            device = self.text_embeddings.device
            dtype = self.text_embeddings.dtype
        memory_dim = getattr(self, 'dptd_memory_dim', None)
        if memory_dim is None:
            memory_dim = self.text_embeddings.shape[0]
        memory_dim = int(memory_dim)
        topk = int(self.dptd_memory_store_topk)

        def _needs_tensor(name, shape, expected_dtype):
            needs_init = not track_instances.has(name)
            if not needs_init:
                value = track_instances.get(name)
                needs_init = not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            if needs_init:
                fill = -1 if name == 'dptd_topk_class_indices' else 0
                value = torch.full(shape, fill, device=device, dtype=expected_dtype)
                track_instances.set(name, value)
            else:
                value = track_instances.get(name)
                if value.device != device or value.dtype != expected_dtype:
                    track_instances.set(name, value.to(device=device, dtype=expected_dtype))

        _needs_tensor('dptd_semantic_proto', (num_tracks, memory_dim), dtype)
        _needs_tensor('dptd_visual_memory', (num_tracks, memory_dim), dtype)
        _needs_tensor('dptd_semantic_conf', (num_tracks,), torch.float32)
        _needs_tensor('dptd_semantic_entropy', (num_tracks,), torch.float32)
        _needs_tensor('dptd_memory_age', (num_tracks,), torch.long)
        _needs_tensor('dptd_topk_class_indices', (num_tracks, topk), torch.long)
        _needs_tensor('dptd_topk_class_scores', (num_tracks, topk), torch.float32)
        return track_instances

    def _snapshot_dptd_memory_state(self, track_instances):
        if not self.use_dptd_semantic_memory:
            return None
        track_instances = self._ensure_dptd_memory_fields(track_instances)
        if not track_instances.has('obj_idxes') or len(track_instances) == 0:
            return {'ids': torch.empty(0, dtype=torch.long), 'fields': {}}
        self._validate_dptd_track_instance_unique_ids(track_instances, 'memory snapshot')
        valid_mask = track_instances.obj_idxes >= 0
        ids = track_instances.obj_idxes[valid_mask].detach().clone()
        snapshot = {'ids': ids, 'fields': {}}
        for field_name in DPTD_MEMORY_FIELDS:
            if not track_instances.has(field_name):
                continue
            value = track_instances.get(field_name)
            if isinstance(value, torch.Tensor) and value.shape[0] == len(track_instances):
                snapshot['fields'][field_name] = value[valid_mask].detach().clone()
        return snapshot

    def _make_dptd_gate_state(self, track_instances, text_memory_embeddings):
        if not self.use_dptd_semantic_gate:
            return None
        track_instances = self._ensure_dptd_memory_fields(track_instances)
        semantic = track_instances.dptd_semantic_proto.detach()
        visual = track_instances.dptd_visual_memory.detach()
        memory_valid = (
            (track_instances.obj_idxes >= 0)
            & (semantic.float().norm(dim=-1) > 1e-6)
            & (visual.float().norm(dim=-1) > 1e-6)
        )
        return {
            'semantic_proto': semantic,
            'visual_memory': visual,
            'memory_age': track_instances.dptd_memory_age.detach(),
            'pred_boxes': track_instances.pred_boxes.detach() if track_instances.has('pred_boxes') else None,
            'historical_offsets': track_instances.dptd_sampling_offsets.detach() if track_instances.has('dptd_sampling_offsets') else None,
            'obj_idxes': track_instances.obj_idxes.detach(),
            'memory_valid': memory_valid.detach(),
            'text_memory_embeddings': text_memory_embeddings.detach() if text_memory_embeddings is not None else None,
        }

    def _normalize_dptd_gate_vector(self, value, field_name, track_instances, dtype=None, default=None):
        if value is None:
            value = default
        if value is None:
            raise RuntimeError(f'DPTD semantic gate missing {field_name}.')
        if isinstance(value, torch.Tensor) and value.dim() == 2:
            value = value[0]
        if not isinstance(value, torch.Tensor) or value.shape[0] != len(track_instances):
            raise RuntimeError(f'DPTD semantic gate row mismatch for {field_name}.')
        target_dtype = dtype or track_instances.scores.dtype
        return value.to(device=track_instances.scores.device, dtype=target_dtype).detach()

    def _attach_dptd_gate_values(self, frame_res, track_instances):
        if not self.use_dptd_semantic_gate:
            return track_instances
        default_gate = torch.ones(len(track_instances), device=track_instances.scores.device, dtype=track_instances.scores.dtype)
        gate = self._normalize_dptd_gate_vector(
            frame_res.get('dptd_gate_values'),
            'dptd_gate_values',
            track_instances,
            default=default_gate,
        )
        raw_value = frame_res.get('dptd_gate_raw_values')
        if raw_value is None and self.use_dptd_semantic_update_suppression:
            raise RuntimeError('DPTD semantic update suppression requires raw semantic gate values.')
        raw_gate = self._normalize_dptd_gate_vector(
            raw_value,
            'dptd_gate_raw_values',
            track_instances,
            default=gate,
        )
        default_valid = torch.zeros(len(track_instances), device=track_instances.scores.device, dtype=torch.bool)
        memory_valid = self._normalize_dptd_gate_vector(
            frame_res.get('dptd_gate_memory_valid'),
            'dptd_gate_memory_valid',
            track_instances,
            dtype=torch.bool,
            default=default_valid,
        )
        track_instances.set('_dptd_current_gate', gate)
        track_instances.set('_dptd_current_gate_raw', raw_gate)
        track_instances.set('_dptd_current_gate_memory_valid', memory_valid.bool())
        return track_instances

    def _compute_dptd_post_visual_consistency(self, memory_snapshot, track_instances):
        if not (self.use_dptd_semantic_gate and track_instances.has('_dptd_current_visual_memory')):
            return track_instances
        current_visual = track_instances.get('_dptd_current_visual_memory')
        consistency = torch.ones(len(track_instances), device=current_visual.device, dtype=torch.float32)
        if memory_snapshot is not None and 'dptd_visual_memory' in memory_snapshot.get('fields', {}) and track_instances.has('obj_idxes'):
            snapshot_ids = memory_snapshot['ids'].to(device=track_instances.obj_idxes.device, dtype=track_instances.obj_idxes.dtype)
            old_visual = memory_snapshot['fields']['dptd_visual_memory'].to(device=current_visual.device, dtype=current_visual.dtype)
            measured = []
            for row_idx in range(len(track_instances)):
                track_id = track_instances.obj_idxes[row_idx]
                if int(track_id.detach().item()) < 0:
                    continue
                matches = torch.nonzero(snapshot_ids == track_id, as_tuple=False).flatten()
                if matches.numel() == 0:
                    continue
                old_value = old_visual[int(matches[0].item())]
                if old_value.float().norm().item() <= 1e-6:
                    continue
                cosine = F.cosine_similarity(
                    current_visual[row_idx].float().unsqueeze(0),
                    old_value.float().unsqueeze(0),
                    dim=-1,
                    eps=1e-6,
                )[0].clamp(-1.0, 1.0)
                conf = torch.sigmoid((cosine - float(self.dptd_gate_visual_cos_tau)) * float(self.dptd_gate_temperature)).clamp(0.0, 1.0)
                consistency[row_idx] = conf.to(dtype=consistency.dtype)
                measured.append(conf.detach().float())
            if measured and self.ov_dptd_debug_stats:
                self.ov_dptd_debug_stats['visual_consistency_mean'] = float(torch.stack(measured).mean().item())
        track_instances.set('_dptd_current_visual_consistency', consistency.detach())
        return track_instances

    def _extend_dptd_semantic_update_suppression(self, snapshot, track_instances):
        if not self.use_dptd_semantic_update_suppression:
            return None
        if snapshot is None:
            return None
        if not track_instances.has('_dptd_current_gate_raw'):
            raise RuntimeError('DPTD semantic update suppression requires _dptd_current_gate_raw.')
        if not track_instances.has('obj_idxes'):
            return None
        old_ids = snapshot.get('ids')
        if not isinstance(old_ids, torch.Tensor) or old_ids.numel() == 0:
            return old_ids
        self._validate_dptd_unique_obj_ids(old_ids, 'semantic suppression snapshot ids')
        self._validate_dptd_track_instance_unique_ids(track_instances, 'semantic suppression')
        semantic_ids = []
        current_ids = track_instances.obj_idxes
        raw_gate = track_instances.get('_dptd_current_gate_raw')
        for track_id in old_ids.detach().cpu().tolist():
            matches = torch.nonzero(current_ids == int(track_id), as_tuple=False).flatten()
            if matches.numel() == 0:
                continue
            row_idx = int(matches[0].item())
            if float(raw_gate[row_idx].detach().item()) < float(self.dptd_semantic_update_suppression_thresh):
                semantic_ids.append(int(track_id))
        previous = snapshot.get('suppressed_ids')
        device = current_ids.device
        dtype = current_ids.dtype
        if previous is not None and isinstance(previous, torch.Tensor) and previous.numel() > 0:
            all_ids = set(int(v) for v in previous.detach().cpu().tolist())
        else:
            all_ids = set()
        all_ids.update(semantic_ids)
        snapshot['suppressed_ids'] = torch.tensor(sorted(all_ids), device=device, dtype=dtype)
        if self.ov_dptd_debug_stats:
            ids = snapshot['suppressed_ids'].detach().cpu().tolist()
            self.ov_dptd_debug_stats['dptd_update_suppressed_count'] = len(ids)
            self.ov_dptd_debug_stats['dptd_update_suppressed_ids'] = ids
        return snapshot['suppressed_ids']


    def _dptd_memory_visual_projection(self, image_feature):
        proj = getattr(self, 'dptd_visual_memory_proj', None)
        if proj is None:
            raise RuntimeError('DPTD visual memory projection fallback is not initialized.')
        return proj(image_feature)

    def _compute_dptd_semantic_memory_candidates(self, frame_res, track_instances):
        text_embeddings = frame_res.pop('dptd_text_memory_embeddings', None)
        if text_embeddings is None:
            raise RuntimeError('DPTD semantic memory requires frame_res["dptd_text_memory_embeddings"].')
        select_id = frame_res['select_id']
        if not isinstance(select_id, torch.Tensor):
            select_id = torch.as_tensor(select_id, device=track_instances.pred_logits.device, dtype=torch.long)
        else:
            select_id = select_id.to(device=track_instances.pred_logits.device, dtype=torch.long)
        num_cls = int(select_id.numel())
        if num_cls <= 0:
            raise RuntimeError('DPTD semantic memory received an empty selected class set.')
        if track_instances.pred_logits.shape[-1] < num_cls:
            raise RuntimeError(
                'DPTD semantic memory logits shorter than selected classes: '
                f'{track_instances.pred_logits.shape[-1]} < {num_cls}'
            )

        logits = track_instances.pred_logits[..., :num_cls].float()
        score = logits.sigmoid()
        prob = score / score.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        text_embeddings = text_embeddings.to(device=prob.device, dtype=prob.dtype)
        semantic_proto = F.normalize(prob @ text_embeddings, dim=-1, eps=1e-6)
        semantic_conf = score.max(dim=-1).values
        entropy_den = math.log(num_cls) if num_cls > 1 else 1.0
        entropy = -(prob * prob.clamp_min(1e-6).log()).sum(dim=-1) / entropy_den
        entropy = entropy.clamp(0.0, 1.0)

        store_topk = int(self.dptd_memory_store_topk)
        actual_topk = min(store_topk, num_cls)
        _, topk_local = torch.topk(score, k=actual_topk, dim=-1)
        topk_scores = torch.gather(prob, dim=-1, index=topk_local)
        topk_indices = select_id[topk_local]
        if actual_topk < store_topk:
            pad_shape = (len(track_instances), store_topk - actual_topk)
            topk_indices = torch.cat([
                topk_indices,
                torch.full(pad_shape, -1, device=topk_indices.device, dtype=topk_indices.dtype),
            ], dim=-1)
            topk_scores = torch.cat([
                topk_scores,
                torch.zeros(pad_shape, device=topk_scores.device, dtype=topk_scores.dtype),
            ], dim=-1)

        target_dtype = track_instances.query_tgt.dtype if track_instances.has('query_tgt') else semantic_proto.dtype
        return {
            '_dptd_current_semantic_proto': semantic_proto.to(dtype=target_dtype).detach(),
            '_dptd_current_semantic_conf': semantic_conf.to(dtype=torch.float32).detach(),
            '_dptd_current_semantic_entropy': entropy.to(dtype=torch.float32).detach(),
            '_dptd_current_topk_class_indices': topk_indices.to(dtype=torch.long).detach(),
            '_dptd_current_topk_class_scores': topk_scores.to(dtype=torch.float32).detach(),
        }

    def _compute_dptd_visual_memory_candidates(self, frame_res, track_instances):
        visual_source = 'alignment_feature'
        if self.dptd_memory_use_alignment_feature and 'pred_embed' in frame_res:
            visual_memory = frame_res['pred_embed'][0]
        elif self.dptd_memory_allow_untrained_visual_projection:
            visual_source = 'projection_fallback'
            visual_memory = self._dptd_memory_visual_projection(track_instances.output_embedding_img)
        else:
            raise RuntimeError(
                'DPTD visual memory requires frame_res["pred_embed"] unless '
                'dptd_memory_allow_untrained_visual_projection=True.'
            )
        if visual_memory.shape[0] != len(track_instances):
            raise RuntimeError(
                'DPTD visual memory row mismatch: '
                f'{visual_memory.shape[0]} != {len(track_instances)}'
            )
        memory_dim = int(getattr(self, 'dptd_memory_dim', visual_memory.shape[-1]))
        if visual_memory.shape[-1] != memory_dim:
            raise RuntimeError(
                'DPTD visual memory dim mismatch: '
                f'{visual_memory.shape[-1]} != {memory_dim}'
            )
        visual_memory = F.normalize(visual_memory.float(), dim=-1, eps=1e-6)
        target_dtype = track_instances.query_tgt.dtype if track_instances.has('query_tgt') else visual_memory.dtype
        if getattr(self, 'ov_dptd_debug_stats', {}):
            self.ov_dptd_debug_stats['dptd_memory_visual_source'] = visual_source
        return visual_memory.to(dtype=target_dtype).detach()

    def _attach_dptd_memory_candidates(self, frame_res, track_instances):
        if not self.use_dptd_semantic_memory:
            return track_instances
        with torch.no_grad():
            semantic_candidates = self._compute_dptd_semantic_memory_candidates(frame_res, track_instances)
            visual_candidate = self._compute_dptd_visual_memory_candidates(frame_res, track_instances)
            for name, value in semantic_candidates.items():
                track_instances.set(name, value.detach())
            track_instances.set('_dptd_current_visual_memory', visual_candidate.detach())
        return track_instances

    def _remove_dptd_memory_candidate_fields(self, track_instances):
        if track_instances is None:
            return track_instances
        for field_name in DPTD_TEMP_FIELDS:
            if track_instances.has(field_name):
                track_instances.remove(field_name)
        return track_instances

    def _set_dptd_memory_debug_stats(self, **stats):
        if not (self.dptd_memory_debug and getattr(self, 'ov_dptd_debug_stats', {})):
            return
        for name, value in stats.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    value = 0.0
                elif value.numel() == 1:
                    value = value.detach().item()
                else:
                    value = value.detach().float().mean().item()
            if isinstance(value, bool):
                self.ov_dptd_debug_stats[name] = bool(value)
            elif isinstance(value, int):
                self.ov_dptd_debug_stats[name] = int(value)
            elif isinstance(value, str):
                self.ov_dptd_debug_stats[name] = value
            else:
                self.ov_dptd_debug_stats[name] = float(value)

    def _update_dptd_memory_state(self, memory_snapshot, suppression_snapshot, track_instances):
        if not self.use_dptd_semantic_memory:
            return track_instances
        with torch.no_grad():
            track_instances = self._ensure_dptd_memory_fields(track_instances)
            self._validate_dptd_track_instance_unique_ids(track_instances, 'memory update')
            if memory_snapshot is not None:
                self._validate_dptd_unique_obj_ids(memory_snapshot.get('ids'), 'memory update snapshot ids')
            missing = [name for name in DPTD_CURRENT_MEMORY_FIELDS if not track_instances.has(name)]
            if missing:
                raise RuntimeError(f'DPTD memory candidate fields missing: {missing}')

            suppressed_ids = set()
            if suppression_snapshot is not None:
                suppressed = suppression_snapshot.get('suppressed_ids')
                if isinstance(suppressed, torch.Tensor) and suppressed.numel() > 0:
                    suppressed_ids = set(int(v) for v in suppressed.detach().cpu().tolist())

            snapshot_ids = memory_snapshot['ids'] if memory_snapshot is not None else torch.empty(0, dtype=torch.long)
            snapshot_ids_cpu = snapshot_ids.detach().cpu().tolist() if isinstance(snapshot_ids, torch.Tensor) else []
            snapshot_index = {int(track_id): idx for idx, track_id in enumerate(snapshot_ids_cpu)}
            snapshot_fields = memory_snapshot.get('fields', {}) if memory_snapshot is not None else {}

            update_count = 0
            keep_count = 0
            init_count = 0
            entropy_values = []
            cosine_deltas = []
            topk_changed = 0
            topk_compared = 0
            ema = float(self.dptd_memory_ema)
            eps = 1e-6

            current_semantic = track_instances.get('_dptd_current_semantic_proto')
            current_visual = track_instances.get('_dptd_current_visual_memory')
            current_conf = track_instances.get('_dptd_current_semantic_conf')
            current_entropy = track_instances.get('_dptd_current_semantic_entropy')
            current_topk_indices = track_instances.get('_dptd_current_topk_class_indices')
            current_topk_scores = track_instances.get('_dptd_current_topk_class_scores')

            for row_idx in range(len(track_instances)):
                if not track_instances.has('obj_idxes'):
                    continue
                track_id = int(track_instances.obj_idxes[row_idx].detach().item())
                if track_id < 0:
                    continue

                old_idx = snapshot_index.get(track_id)
                has_old = old_idx is not None
                old_semantic = (
                    snapshot_fields['dptd_semantic_proto'][old_idx].to(
                        device=track_instances.dptd_semantic_proto.device,
                        dtype=track_instances.dptd_semantic_proto.dtype,
                    )
                    if has_old and 'dptd_semantic_proto' in snapshot_fields
                    else track_instances.dptd_semantic_proto[row_idx]
                )
                old_visual = (
                    snapshot_fields['dptd_visual_memory'][old_idx].to(
                        device=track_instances.dptd_visual_memory.device,
                        dtype=track_instances.dptd_visual_memory.dtype,
                    )
                    if has_old and 'dptd_visual_memory' in snapshot_fields
                    else track_instances.dptd_visual_memory[row_idx]
                )
                old_age = (
                    snapshot_fields['dptd_memory_age'][old_idx].to(
                        device=track_instances.dptd_memory_age.device,
                        dtype=track_instances.dptd_memory_age.dtype,
                    )
                    if has_old and 'dptd_memory_age' in snapshot_fields
                    else track_instances.dptd_memory_age[row_idx]
                )
                old_topk = (
                    snapshot_fields['dptd_topk_class_indices'][old_idx].to(
                        device=track_instances.dptd_topk_class_indices.device,
                        dtype=track_instances.dptd_topk_class_indices.dtype,
                    )
                    if has_old and 'dptd_topk_class_indices' in snapshot_fields
                    else track_instances.dptd_topk_class_indices[row_idx]
                )

                is_new = (
                    not has_old
                    or old_semantic.float().norm().item() <= eps
                    or old_visual.float().norm().item() <= eps
                )
                reliable = (
                    float(track_instances.scores[row_idx].detach().item()) >= float(self.dptd_memory_min_score)
                    and float(current_entropy[row_idx].detach().item()) <= float(self.dptd_memory_max_entropy)
                )
                entropy_values.append(float(current_entropy[row_idx].detach().item()))

                if track_id in suppressed_ids:
                    track_instances.dptd_semantic_proto[row_idx] = old_semantic
                    track_instances.dptd_visual_memory[row_idx] = old_visual
                    if has_old and 'dptd_semantic_conf' in snapshot_fields:
                        track_instances.dptd_semantic_conf[row_idx] = snapshot_fields['dptd_semantic_conf'][old_idx].to(
                            device=track_instances.dptd_semantic_conf.device,
                            dtype=track_instances.dptd_semantic_conf.dtype,
                        )
                    if has_old and 'dptd_semantic_entropy' in snapshot_fields:
                        track_instances.dptd_semantic_entropy[row_idx] = snapshot_fields['dptd_semantic_entropy'][old_idx].to(
                            device=track_instances.dptd_semantic_entropy.device,
                            dtype=track_instances.dptd_semantic_entropy.dtype,
                        )
                    if has_old and 'dptd_topk_class_indices' in snapshot_fields:
                        track_instances.dptd_topk_class_indices[row_idx] = old_topk
                    if has_old and 'dptd_topk_class_scores' in snapshot_fields:
                        track_instances.dptd_topk_class_scores[row_idx] = snapshot_fields['dptd_topk_class_scores'][old_idx].to(
                            device=track_instances.dptd_topk_class_scores.device,
                            dtype=track_instances.dptd_topk_class_scores.dtype,
                        )
                    track_instances.dptd_memory_age[row_idx] = old_age + 1
                    keep_count += 1
                    continue

                if is_new:
                    track_instances.dptd_semantic_proto[row_idx] = current_semantic[row_idx]
                    track_instances.dptd_visual_memory[row_idx] = current_visual[row_idx]
                    track_instances.dptd_semantic_conf[row_idx] = current_conf[row_idx]
                    track_instances.dptd_semantic_entropy[row_idx] = current_entropy[row_idx]
                    track_instances.dptd_topk_class_indices[row_idx] = current_topk_indices[row_idx]
                    track_instances.dptd_topk_class_scores[row_idx] = current_topk_scores[row_idx]
                    track_instances.dptd_memory_age[row_idx] = 0
                    init_count += 1
                    continue

                if reliable:
                    new_semantic = F.normalize(
                        ema * old_semantic.float() + (1.0 - ema) * current_semantic[row_idx].float(),
                        dim=-1,
                        eps=1e-6,
                    ).to(dtype=track_instances.dptd_semantic_proto.dtype)
                    new_visual = F.normalize(
                        ema * old_visual.float() + (1.0 - ema) * current_visual[row_idx].float(),
                        dim=-1,
                        eps=1e-6,
                    ).to(dtype=track_instances.dptd_visual_memory.dtype)
                    track_instances.dptd_semantic_proto[row_idx] = new_semantic
                    track_instances.dptd_visual_memory[row_idx] = new_visual
                    track_instances.dptd_semantic_conf[row_idx] = current_conf[row_idx]
                    track_instances.dptd_semantic_entropy[row_idx] = current_entropy[row_idx]
                    track_instances.dptd_topk_class_indices[row_idx] = current_topk_indices[row_idx]
                    track_instances.dptd_topk_class_scores[row_idx] = current_topk_scores[row_idx]
                    track_instances.dptd_memory_age[row_idx] = 0
                    cosine_delta = 1.0 - F.cosine_similarity(
                        old_semantic.float().unsqueeze(0),
                        new_semantic.float().unsqueeze(0),
                        dim=-1,
                        eps=1e-6,
                    )[0]
                    cosine_deltas.append(float(cosine_delta.detach().item()))
                    topk_changed += int(not torch.equal(old_topk.detach().cpu(), current_topk_indices[row_idx].detach().cpu()))
                    topk_compared += 1
                    update_count += 1
                else:
                    track_instances.dptd_semantic_proto[row_idx] = old_semantic
                    track_instances.dptd_visual_memory[row_idx] = old_visual
                    if has_old and 'dptd_semantic_conf' in snapshot_fields:
                        track_instances.dptd_semantic_conf[row_idx] = snapshot_fields['dptd_semantic_conf'][old_idx].to(
                            device=track_instances.dptd_semantic_conf.device,
                            dtype=track_instances.dptd_semantic_conf.dtype,
                        )
                    if has_old and 'dptd_semantic_entropy' in snapshot_fields:
                        track_instances.dptd_semantic_entropy[row_idx] = snapshot_fields['dptd_semantic_entropy'][old_idx].to(
                            device=track_instances.dptd_semantic_entropy.device,
                            dtype=track_instances.dptd_semantic_entropy.dtype,
                        )
                    if has_old and 'dptd_topk_class_indices' in snapshot_fields:
                        track_instances.dptd_topk_class_indices[row_idx] = old_topk
                    if has_old and 'dptd_topk_class_scores' in snapshot_fields:
                        track_instances.dptd_topk_class_scores[row_idx] = snapshot_fields['dptd_topk_class_scores'][old_idx].to(
                            device=track_instances.dptd_topk_class_scores.device,
                            dtype=track_instances.dptd_topk_class_scores.dtype,
                        )
                    track_instances.dptd_memory_age[row_idx] = old_age + 1
                    keep_count += 1

            self._set_dptd_memory_debug_stats(
                dptd_memory_update_count=update_count,
                dptd_memory_keep_count=keep_count,
                dptd_memory_init_count=init_count,
                dptd_memory_entropy_mean=(sum(entropy_values) / len(entropy_values)) if entropy_values else 0.0,
                dptd_memory_semantic_proto_cosine_delta_mean=(sum(cosine_deltas) / len(cosine_deltas)) if cosine_deltas else 0.0,
                dptd_memory_topk_change_rate=(topk_changed / topk_compared) if topk_compared else 0.0,
            )
        return self._remove_dptd_memory_candidate_fields(track_instances)


    def _generate_empty_tracks(self, cls_pad_len=1203):
        track_instances = Instances((1, 1))
        num_queries = self.num_queries
        dim_h = self.transformer.d_model
        device = self.transformer.level_embed.device

        track_instances.ref_pts = torch.zeros((num_queries, 4), device=device)
        track_instances.query_tgt = maybe_get_quantized_embedding_weight(self.transformer.tgt_embed)
        track_instances.query_pos = torch.zeros((num_queries, dim_h), device=device)

        track_instances.obj_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((num_queries,), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((num_queries, 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((num_queries, cls_pad_len), dtype=torch.float, device=device)
        if self.use_ov_dptd:
            track_instances.dptd_sampling_offsets = self._make_empty_dptd_sampling_offsets(
                num_queries,
                device,
                track_instances.query_tgt.dtype,
            )
        if self.use_dptd_semantic_memory:
            self._ensure_dptd_memory_fields(track_instances)

        if not self.training:
            track_instances.cls_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
            track_instances.disappear_time = torch.zeros((num_queries, ), dtype=torch.long, device=device)
        return track_instances.to(device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_embed):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_embed':c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_embed[:-1])]
        
    def _distribution_based_sampling(self, pad_len, uniq_labels=None):
        frequency = self.frequency.clone()
        frequency[uniq_labels] = 0
        extra_labels = torch.multinomial(frequency, pad_len)
        extra_labels = extra_labels[torch.isin(extra_labels,self.novel_cls_cpu, invert=True)]
        extra_labels = extra_labels[torch.randperm(len(extra_labels))]
        return extra_labels
    
    def get_select_id(self, cls_num, labels_list, extra_labels, is_first):
        max_pad_len = max(cls_num, self.max_pad_len)
        # get input categories
        uniq_labels = torch.unique(labels_list).to("cpu")    
        if is_first: # first frame detection
            if len(uniq_labels) < max_pad_len:
                pad_len = max_pad_len - len(uniq_labels)
                if self.distribution_based_sampling: # Sample negative categories based on the distribution.
                    extra_labels = self._distribution_based_sampling(pad_len, uniq_labels=uniq_labels)
                else:
                    extra_list = torch.tensor([i for i in self.all_ids if i not in uniq_labels])
                    extra_labels = extra_list[torch.randperm(len(extra_list))][:pad_len]
                select_id = uniq_labels.tolist() + extra_labels.tolist()
            else:
                select_id = uniq_labels.tolist()
                extra_labels = torch.LongTensor([])
        else: # subsequent frame tracking
            extra_label_notin = torch.isin(extra_labels, uniq_labels, invert=True)
            extra_labels_cur = extra_labels[extra_label_notin]
            select_id = uniq_labels.tolist() + extra_labels_cur.tolist()
            if len(select_id) < max_pad_len:
                sampled_labels = self._distribution_based_sampling(max_pad_len-len(select_id), uniq_labels=torch.tensor(select_id))
                if extra_labels is not None:
                    extra_labels = torch.cat([extra_labels_cur, sampled_labels])
                else:
                    extra_labels = sampled_labels
                select_id = uniq_labels.tolist() + extra_labels.tolist()
            elif len(select_id) > max_pad_len:
                select_id = select_id[:max_pad_len]
        return select_id, extra_labels

    def _extract_backbone_features(self, samples):
        features, pos = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            num_srcs = len(srcs)
            for l in range(num_srcs, self.num_feature_levels):
                if l == num_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                mask = F.interpolate(samples.mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        return srcs, masks, pos

    def _prepare_text_conditioning(self, targets, extra_labels, is_first, cls_num, batch_size):
        if self.training:
            labels_list = torch.cat([targets.labels])
            select_id, extra_labels = self.get_select_id(cls_num, labels_list, extra_labels, is_first)
        else:
            select_id, extra_labels = self.select_id, None

        text_query = self.text_embeddings[:, select_id].to(self.patch2query.weight.device).t()
        image_align = self.image_embeddings[:, select_id].to(text_query.device).t()
        image_feat_ori = image_align.float().detach()

        dtype = self.patch2query.weight.dtype
        text_query = self.patch2query(text_query.type(dtype))
        select_id = torch.tensor(select_id, device=text_query.device)
        text_dict = preprocess_for_masks(batch_size, select_id, text_query)
        return text_dict, image_feat_ori, select_id, extra_labels

    def _slice_encoder_cache(self, encoder_cache, sample_idx):
        return {
            'memory': encoder_cache['memory'][sample_idx:sample_idx + 1],
            'mask_flatten': encoder_cache['mask_flatten'][sample_idx:sample_idx + 1],
            'lvl_pos_embed_flatten': encoder_cache['lvl_pos_embed_flatten'][sample_idx:sample_idx + 1],
            'spatial_shapes': encoder_cache['spatial_shapes'],
            'level_start_index': encoder_cache['level_start_index'],
            'valid_ratios': encoder_cache['valid_ratios'][sample_idx:sample_idx + 1],
        }

    def _transpose_batch_inputs(self, data):
        frames = data['imgs']
        targets = data['gt_instances']
        if len(frames) == 0:
            return [], [], 0
        if isinstance(frames[0], list):
            batch_size = len(frames)
            num_frames = len(frames[0])
            frames_by_time = [[frames[sample_idx][frame_idx] for sample_idx in range(batch_size)] for frame_idx in range(num_frames)]
            targets_by_time = [[targets[sample_idx][frame_idx] for sample_idx in range(batch_size)] for frame_idx in range(num_frames)]
            return frames_by_time, targets_by_time, batch_size
        return [[frame] for frame in frames], [[target] for target in targets], 1

    def _forward_single_image_from_memory(self, encoder_cache, sample_idx, track_instances: Instances, targets=None, extra_labels=None, is_first=True, cls_num=0):
        if self.use_ov_dptd:
            raise RuntimeError('OV-DPTD v1 does not support _forward_single_image_from_memory.')
        text_dict, image_feat_ori, select_id, extra_labels = self._prepare_text_conditioning(
            targets, extra_labels, is_first, cls_num, batch_size=1
        )
        sample_cache = self._slice_encoder_cache(encoder_cache, sample_idx)

        (hs_cti, hs_ofa, init_reference, inter_references, pre_outputs_classes, query_pos_track) = self.transformer.decode_from_memory(
            sample_cache['memory'],
            sample_cache['mask_flatten'],
            sample_cache['lvl_pos_embed_flatten'],
            sample_cache['spatial_shapes'],
            sample_cache['level_start_index'],
            sample_cache['valid_ratios'],
            query_pos=track_instances.query_pos,
            query_tgt=track_instances.query_tgt,
            ref_pts=track_instances.ref_pts,
            text_dict=text_dict,
        )

        outputs_coords = []
        outputs_embeds = []

        for lvl in range(hs_cti.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs_ofa[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_embeds.append(self.feature_align[lvl](hs_ofa[lvl]))
        outputs_class = pre_outputs_classes
        outputs_coord = torch.stack(outputs_coords)
        outputs_embed = torch.stack(outputs_embeds)

        if init_reference.shape[-1] == 4:
            ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :4]], dim=0)
        else:
            ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'ref_pts': ref_pts_all[-2],
            'pred_embed': outputs_embed[-1],
            'select_id': select_id,
            'image_feat': image_feat_ori,
            'extra_labels': extra_labels,
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_embed)
            for temp in out['aux_outputs']:
                temp['select_id'] = select_id
                temp['image_feat'] = image_feat_ori

        out['query_pos_track'] = query_pos_track.transpose(0, 1)
        out['hs_ofa'] = hs_ofa[-1]
        out['hs_cti'] = hs_cti[-1]
        return out

    def _forward_hybrid_batched(self, frames_by_time, targets_by_time):
        if self.use_ov_dptd:
            raise RuntimeError('OV-DPTD v1 does not support hybrid batched training.')
        batch_size = len(targets_by_time[0])
        sample_gt_instances = [
            [targets_by_time[frame_idx][sample_idx] for frame_idx in range(len(targets_by_time))]
            for sample_idx in range(batch_size)
        ]
        cls_nums = [
            max(len(torch.unique(gt_instance.labels)) for gt_instance in gt_instances_per_sample)
            for gt_instances_per_sample in sample_gt_instances
        ]
        self.criterion.initialize_batch(sample_gt_instances)

        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
            'track_instances': []
        }
        track_instances_batch = [self._generate_empty_tracks() for _ in range(batch_size)]
        extra_labels_batch = [None] * batch_size

        for frame_index, (frame_batch, targets_batch) in enumerate(zip(frames_by_time, targets_by_time)):
            is_last = frame_index == len(frames_by_time) - 1
            is_first = frame_index == 0
            for frame in frame_batch:
                frame.requires_grad = False

            samples = nested_tensor_from_tensor_list(frame_batch)
            srcs, masks, pos = self._extract_backbone_features(samples)
            encoder_cache = self.transformer.encode_image(srcs, masks, pos)
            next_extra_labels_batch = []

            for sample_idx, (track_instances, targets) in enumerate(zip(track_instances_batch, targets_batch)):
                frame_res = self._forward_single_image_from_memory(
                    encoder_cache,
                    sample_idx,
                    track_instances,
                    targets,
                    extra_labels_batch[sample_idx],
                    is_first,
                    cls_nums[sample_idx],
                )
                frame_res = self._post_process_single_image(
                    frame_res,
                    track_instances,
                    is_last,
                    is_first=is_first,
                    sample_idx=sample_idx,
                )

                track_instances_batch[sample_idx] = frame_res['track_instances']
                next_extra_labels_batch.append(frame_res['extra_labels'])

                if sample_idx == 0:
                    outputs['pred_logits'].append(frame_res['pred_logits'])
                    outputs['pred_boxes'].append(frame_res['pred_boxes'])
                    outputs['track_instances'].append(frame_res['track_instances_pre'])

            extra_labels_batch = next_extra_labels_batch

        outputs['losses_dict'] = self.criterion.losses_dict
        return outputs
    
    def _forward_single_image(self, samples, track_instances: Instances, targets=None, extra_labels=None ,is_first=True, cls_num=0):
        srcs, masks, pos = self._extract_backbone_features(samples)

        # Get the selected category id
        if self.training:
            labels_list = torch.cat([targets.labels])
            select_id, extra_labels = self.get_select_id(cls_num, labels_list, extra_labels, is_first)
        else:
            select_id, extra_labels = self.select_id, None

        # Prepare queries and embeddings for alignment
        text_query = self.text_embeddings[:, select_id].to(masks[0].device).t()
        dptd_text_memory_embeddings = text_query.detach() if self.use_dptd_semantic_memory else None
        image_align = self.image_embeddings[:, select_id].to(masks[0].device).t()
        
        image_feat_ori = (image_align.float()).detach()

        dtype = self.patch2query.weight.dtype
        text_query = self.patch2query(text_query.type(dtype))
        select_id = torch.tensor(select_id).to(text_query.device)
        text_dict = preprocess_for_masks(srcs[0].shape[0], select_id, text_query)

        dptd_sampling_offsets = (
            track_instances.dptd_sampling_offsets
            if self.use_ov_dptd and track_instances.has('dptd_sampling_offsets')
            else None
        )
        dptd_gate_state = self._make_dptd_gate_state(track_instances, dptd_text_memory_embeddings)
        transformer_outputs = self.transformer(
            srcs,
            masks,
            pos,
            track_instances.query_pos,
            track_instances.query_tgt,
            ref_pts=track_instances.ref_pts,
            text_dict=text_dict,
            dptd_sampling_offsets=dptd_sampling_offsets,
            dptd_gate_state=dptd_gate_state,
            return_dptd_info=self.use_ov_dptd,
        )
        dptd_info = None
        if self.use_ov_dptd:
            (hs_cti, hs_ofa, init_reference, inter_references, pre_outputs_classes, query_pos_track, dptd_info) = transformer_outputs
        else:
            (hs_cti, hs_ofa, init_reference, inter_references, pre_outputs_classes, query_pos_track) = transformer_outputs

        outputs_coords = []
        outputs_embeds = []

        for lvl in range(hs_cti.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs_ofa[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_embeds.append(self.feature_align[lvl](hs_ofa[lvl]))
        outputs_class = pre_outputs_classes
        outputs_coord = torch.stack(outputs_coords)
        outputs_embed = torch.stack(outputs_embeds)

        if init_reference.shape[-1]==4:
            ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :4]], dim=0)
        else:
            ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)
        out = {
            'pred_logits': outputs_class[-1], 
            'pred_boxes': outputs_coord[-1], 
            'ref_pts': ref_pts_all[-2],
            "pred_embed": outputs_embed[-1],
            "select_id": select_id,
            "image_feat": image_feat_ori,
            "extra_labels": extra_labels,
            }
        if self.use_dptd_semantic_memory:
            out['dptd_text_memory_embeddings'] = dptd_text_memory_embeddings
        if self.use_ov_dptd:
            if dptd_info is None or dptd_info.get('sampling_offsets') is None:
                raise RuntimeError('OV-DPTD decoder did not return final sampling offsets.')
            out['dptd_sampling_offsets'] = dptd_info['sampling_offsets']
            if dptd_info.get('semantic_gate') is not None:
                out['dptd_gate_values'] = dptd_info['semantic_gate'].detach()
            if dptd_info.get('semantic_gate_raw') is not None:
                out['dptd_gate_raw_values'] = dptd_info['semantic_gate_raw'].detach()
            if dptd_info.get('semantic_gate_memory_valid') is not None:
                out['dptd_gate_memory_valid'] = dptd_info['semantic_gate_memory_valid'].detach()
            self._update_ov_dptd_debug_stats(dptd_info)
            
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_embed)
            for temp in out["aux_outputs"]:
                temp["select_id"] = select_id
                temp["image_feat"] = image_feat_ori
            
        out['query_pos_track'] = query_pos_track.transpose(0, 1)
        out['hs_ofa'] = hs_ofa[-1]
        out['hs_cti'] = hs_cti[-1]
        return out
     
    def _post_process_single_image(self, frame_res, track_instances, is_last, is_repeat=None, is_first=False, target_size=None, sample_idx=None):
        with torch.no_grad():
            track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values

        dptd_suppression_snapshot = self._snapshot_dptd_update_suppression_state(track_instances)
        dptd_memory_snapshot = self._snapshot_dptd_memory_state(track_instances)

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding_txt = frame_res['hs_cti'][0]
        track_instances.output_embedding_img = frame_res['hs_ofa'][0]
        track_instances.query_pos = frame_res["query_pos_track"][0]
        self._attach_dptd_sampling_offsets(frame_res, track_instances)
        track_instances = self._attach_dptd_memory_candidates(frame_res, track_instances)
        track_instances = self._attach_dptd_gate_values(frame_res, track_instances)
        self._select_dptd_update_suppressed_ids(dptd_suppression_snapshot, track_instances)
        track_instances = self._restore_dptd_update_suppressed_state(dptd_suppression_snapshot, track_instances)

        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res, is_first, sample_idx=sample_idx)
        else:
            if self.train_with_artificial_img_seqs:
                track_instances, _track_discard = protect_track_preds(track_instances, num_queries=self.num_queries, miss_tolerance=self.track_base.miss_tolerance, ious_thresh=self.ious_thresh) 
            track_instances = self.post_process_pre(track_instances, frame_res['select_id'], is_first)
            # each track will be assigned an unique global id by the track base.
            if is_first:
                self.track_base.clear()
            track_instances = self.track_base.update(track_instances, _track_discard, is_repeat=is_repeat)
            track_instances = self._restore_dptd_update_suppressed_state(dptd_suppression_snapshot, track_instances)

        track_instances = self._compute_dptd_post_visual_consistency(dptd_memory_snapshot, track_instances)
        self._extend_dptd_semantic_update_suppression(dptd_suppression_snapshot, track_instances)
        track_instances = self._restore_dptd_update_suppressed_state(dptd_suppression_snapshot, track_instances)
        track_instances = self._compute_dptd_post_visual_consistency(dptd_memory_snapshot, track_instances)
        track_instances = self._update_dptd_memory_state(
            dptd_memory_snapshot,
            dptd_suppression_snapshot,
            track_instances,
        )
        frame_res['track_instances'] = track_instances

        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks(cls_pad_len=track_instances.pred_logits.shape[1])
        tmp['track_instances'] = track_instances

        if not is_last:
            out_track_instances = self.track_embed(tmp)
            out_track_instances = self._restore_dptd_update_suppressed_state(
                dptd_suppression_snapshot,
                out_track_instances,
            )
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        frame_res['track_instances_pre'] = track_instances
        return frame_res
    
    def post_process_pre(self, track_instances, select_id, is_first):
        out_logits = track_instances.pred_logits

        prob = out_logits.sigmoid()
        scores, labels = prob.max(-1)
 
        track_instances.scores = scores
        cur_cls_idxes = select_id[labels]
        # track_instances.keep_cls = torch.eq(cur_cls_idxes, track_instances.cls_idxes)

        if is_first:
            track_instances.cls_idxes = cur_cls_idxes
        else:
            track_instances.cls_idxes[scores >= self.track_base.filter_score_thresh] = cur_cls_idxes[scores >= self.track_base.filter_score_thresh]
        return track_instances

    @torch.no_grad()
    def inference_single_image(self, data, track_instances=None, is_repeat=False, frame_id=None, ori_img_size=None, extra_labels=None):
        img = nested_tensor_from_tensor_list([data['imgs'][0]])
        if (track_instances is None) or (frame_id == 0):
            track_instances = self._generate_empty_tracks()
        if frame_id == 0:
            is_first = True
        else:
            is_first = False

        res = self._forward_single_image(img, track_instances, None, extra_labels, is_first, cls_num=None)
        res = self._post_process_single_image(res, track_instances, False, is_repeat=is_repeat, is_first=is_first, target_size=ori_img_size[:-1])

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size[:-1])
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts'] 
            img_h, img_w = ori_img_size[:-1]
            # scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def _check_ov_dptd_forward_supported(self, batch_size):
        if self.training and self.use_ov_dptd and batch_size > 1:
            raise RuntimeError('OV-DPTD v1 does not support batch_size > 1 training.')
        if self.use_ov_dptd and self.use_checkpoint:
            raise RuntimeError('OV-DPTD v1 does not support use_checkpoint_track=True.')
        if self.training and self.use_dptd_update_suppression:
            raise RuntimeError('DPTD update suppression is inference-only.')
        if self.use_dptd_update_suppression and not self.use_ov_dptd:
            raise RuntimeError('DPTD update suppression requires use_ov_dptd=True.')
        if getattr(self, 'use_dptd_semantic_memory', False) and not self.use_ov_dptd:
            raise RuntimeError('DPTD semantic memory requires use_ov_dptd=True.')
        if getattr(self, 'use_dptd_semantic_gate', False) and not getattr(self, 'use_dptd_semantic_memory', False):
            raise RuntimeError('DPTD semantic gate requires use_dptd_semantic_memory=True.')
        if getattr(self, 'use_dptd_semantic_update_suppression', False) and not getattr(self, 'use_dptd_update_suppression', False):
            raise RuntimeError('DPTD semantic update suppression requires use_dptd_update_suppression=True.')

    def forward(self, data):
        frames_by_time, targets_by_time, batch_size = self._transpose_batch_inputs(data)
        self._check_ov_dptd_forward_supported(batch_size)
        if self.training and batch_size > 1 and self.supports_mot_batch:
            return self._forward_hybrid_batched(frames_by_time, targets_by_time)

        if self.training:
            self.criterion.initialize(data['gt_instances'])
        frames = data['imgs']
        cls_num = max([len(torch.unique(gt_instance.labels)) for gt_instance in data['gt_instances']])
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
            'track_instances': []
        }
        track_instances = self._generate_empty_tracks()

        keys = list(track_instances._fields.keys())
        for frame_index, (frame, targets) in enumerate(zip(frames, data['gt_instances'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            is_first = frame_index == 0
            if is_first:
                extra_labels = None
            else:
                extra_labels = frame_res["extra_labels"]
            if self.use_checkpoint and frame_index < len(frames) - 3:
                def fn(frame, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp, targets, extra_labels, is_first, cls_num)
                    ret = (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['ref_pts'],
                        frame_res['pred_embed'],
                        frame_res['select_id'],
                        frame_res['image_feat'],
                        frame_res['extra_labels'],
                        frame_res['query_pos_track'],
                        frame_res['hs_cti'],
                        frame_res['hs_ofa'],
                    )
                    return ret + (
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_embed'] for aux in frame_res['aux_outputs']],
                        *[aux['select_id'] for aux in frame_res['aux_outputs']],
                        *[aux['image_feat'] for aux in frame_res['aux_outputs']],
                    )
                args = [frame] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'ref_pts': tmp[2],
                    'pred_embed': tmp[3],
                    'select_id': tmp[4],
                    'image_feat': tmp[5],
                    'extra_labels': tmp[6],
                    'query_pos_track': tmp[7],
                    'hs_cti': tmp[8],
                    'hs_ofa': tmp[9],
                }
                aux_offset = 10
                frame_res.update({
                    'aux_outputs': [{
                        'pred_logits': tmp[aux_offset+i],
                        'pred_boxes': tmp[aux_offset+5+i],
                        'pred_embed': tmp[aux_offset+10+i],
                        'select_id': tmp[aux_offset+15+i],
                        'image_feat': tmp[aux_offset+20+i],
                    } for i in range(len(self.computed_aux)-1)],
                })
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances, targets, extra_labels, is_first, cls_num)
            frame_res = self._post_process_single_image(frame_res, track_instances, is_last, is_first=is_first)

            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])
            outputs['track_instances'].append(frame_res['track_instances_pre'])

        outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args, cfg):
    resolve_attention_protection_options(args, cfg)
    resolve_ov_dptd_options(args, cfg)
    
    assert cfg.Clip_text_embeddings and cfg.Clip_image_embeddings, "Clip_text_embeddings or Clip_image_embeddings should not be None"
    text_embeddings, image_embeddings = load_embeddings(cfg.Clip_text_embeddings, cfg.Clip_image_embeddings)  

    device = torch.device(args.device)
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)
    d_model = transformer.d_model
    hidden_dim = cfg.dim_feedforward
    updater = build_updater(args, args.track_query_iteration, d_model, hidden_dim, d_model*2)
    matcher = build_matcher(args)

    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    
    for i in range(0, num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            'frame_{}_loss_align'.format(i): args.align_loss_coef,
                            })

    if args.aux_loss:
        for i in range(0, num_frames_per_batch):
            for j in range(cfg.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    'frame_{}_aux{}_loss_align'.format(i, j): args.align_loss_coef,
                                    })

    losses = ['labels', 'boxes', 'align']

    criterion = OVFrameMatcher(None, matcher=matcher, weight_dict=weight_dict, losses=losses, random_drop=args.random_drop,
                                train_with_artificial_img_seqs=cfg.train_with_artificial_img_seqs,
                                calculate_negative_samples=args.calculate_negative_samples,
                                num_queries=cfg.num_queries,
                                )
    criterion.to(device)

    model = OVTR(
        backbone,
        transformer,
        track_embed=updater,
        num_feature_levels=cfg.num_feature_levels,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        max_len=args.max_len,
        use_checkpoint=cfg.use_checkpoint_track,
        train_with_artificial_img_seqs=cfg.train_with_artificial_img_seqs,
        distribution_based_sampling=cfg.distribution_based_sampling,
        novel_cls_cpu=novel_class,
        computed_aux=cfg.computed_aux,
        score_thresh=args.score_thresh,
        filter_score_thresh=args.filter_score_thresh,
        miss_tolerance=args.miss_tolerance,
        use_ov_dptd=args.use_ov_dptd,
        ov_dptd_store_debug=args.ov_dptd_store_debug,
        use_dptd_update_suppression=args.use_dptd_update_suppression,
        dptd_update_suppression_thresh=args.dptd_update_suppression_thresh,
        dptd_update_suppression_restore_fields=args.dptd_update_suppression_restore_fields,
        dptd_update_suppression_track_id_based=args.dptd_update_suppression_track_id_based,
        use_dptd_semantic_memory=args.use_dptd_semantic_memory,
        dptd_memory_ema=args.dptd_memory_ema,
        dptd_memory_min_score=args.dptd_memory_min_score,
        dptd_memory_max_entropy=args.dptd_memory_max_entropy,
        dptd_memory_use_alignment_feature=args.dptd_memory_use_alignment_feature,
        dptd_memory_allow_untrained_visual_projection=args.dptd_memory_allow_untrained_visual_projection,
        dptd_memory_store_topk=args.dptd_memory_store_topk,
        dptd_memory_debug=args.dptd_memory_debug,
        use_dptd_semantic_gate=args.use_dptd_semantic_gate,
        dptd_gate_mode=args.dptd_gate_mode,
        dptd_gate_min_score=args.dptd_gate_min_score,
        dptd_gate_max_entropy=args.dptd_gate_max_entropy,
        dptd_gate_semantic_cos_tau=args.dptd_gate_semantic_cos_tau,
        dptd_gate_visual_cos_tau=args.dptd_gate_visual_cos_tau,
        dptd_gate_offset_tau=args.dptd_gate_offset_tau,
        dptd_gate_box_iou_tau=args.dptd_gate_box_iou_tau,
        dptd_gate_temperature=args.dptd_gate_temperature,
        dptd_gate_min_appearance=args.dptd_gate_min_appearance,
        dptd_gate_debug=args.dptd_gate_debug,
        use_dptd_semantic_update_suppression=args.use_dptd_semantic_update_suppression,
        dptd_semantic_update_suppression_thresh=args.dptd_semantic_update_suppression_thresh,
    )
    return model, criterion
