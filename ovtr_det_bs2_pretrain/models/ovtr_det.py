# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
"""
DETR model and criterion classes.
"""
import copy
import torch
import torch.nn.functional as F
from torch import nn
from typing import List
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, get_world_size, is_dist_avail_and_initialized, inverse_sigmoid)
from detectron2.structures import Instances
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .deformable_detr import SetCriterion
from .segmentation import sigmoid_focal_loss
from util.clip_utils import load_embeddings

from .utils import MLP, preprocess_for_masks
from util.list_LVIS import Frequency_list_no_rare, novel_class


class OVMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses,
                        losses_enc=None,
                        random_drop=0,
                        calculate_negative_samples=True,
                        num_queries=900,
                        ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.losses_enc = losses_enc
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.random_drop = random_drop

        self.num_queries = num_queries
        self.calculate_negative_samples = calculate_negative_samples

    def initialize(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
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
        idx = self._get_src_permutation_idx(indices)
        
        if self.calculate_negative_samples:
            num_class = len(outputs["select_id"])

        target_classes = torch.full(src_logits.shape[:2], num_class,
                                    dtype=torch.int64, device=src_logits.device)
        
        select_id = outputs["select_id"]
        labels_ori = torch.cat([t["labels"][J] for t, (_, J) in zip(gt_instances, indices)])
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
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(gt_instances, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
    
    def loss_align(self, outputs, targets, indices, num_boxes, l1_distillation=False):
        """Alignment mechanism guides generalization capabilities and aligned queries.
        """
        idx = self._get_src_permutation_idx(indices)
        src_feature = outputs["pred_embed"][idx]

        select_id = outputs["select_id"]
        image_feat = outputs["image_feat"]
        target_feature = []
        for t, (_, i) in zip(targets, indices):
            for c in t["labels"][i]:
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
        uniq_labels = [torch.unique(t['labels']) for t in targets]
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
    
    def forward(self, outputs, targets):

        if not self.training:
            return {
                "loss_ce": outputs["pred_logits"].sum() * 0.0,
                "class_error": outputs["pred_logits"].sum() * 0.0,
            }

        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"
        }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        masks = []
        for t in targets:
            mask = t["labels"] == -2
            for ind, v in enumerate(t["labels"]):
                if v in outputs["select_id"]:
                    mask[ind] = True
            masks.append(mask)
        num_boxes = sum(len(t["labels"][m]) for t, m in zip(targets, masks))
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Retrieve the matching between the outputs of the last layer and the targets
        select_id = outputs["select_id"]
        indices = self.matcher(outputs_without_aux, targets, select_id)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, select_id)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs["log"] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]        
            indices = self.matcher(enc_outputs, targets, enc_outputs['select_id'])
            for loss in self.losses_enc:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        return losses


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class OVTR_det(nn.Module):
    def __init__(self, backbone, transformer, num_feature_levels, criterion, track_embed,
                    aux_loss=True, with_box_refine=False, two_stage=False,
                    two_stage_bbox_embed_share=False,
                    dec_pred_bbox_embed_share=True,
                    distribution_based_sampling=None,
                    text_embeddings=None,
                    image_embeddings=None,
                    max_len=None,
                    novel_cls_cpu=None
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For LVIS, we recommend 900 queries.
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
        self.frequency = torch.tensor(Frequency_list_no_rare, dtype=torch.float32, device='cpu')
        print("Training excludes rare categories.")
        self.novel_cls_cpu = novel_cls_cpu

        for layer in [self.patch2query]:
            nn.init.xavier_uniform_(self.patch2query.weight)
            nn.init.constant_(self.patch2query.bias, 0)

        # feature alignment
        self.feature_align = nn.Linear(256, 512)  # alignment head
        nn.init.xavier_uniform_(self.feature_align.weight)
        nn.init.constant_(self.feature_align.bias, 0)
        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.feature_align = _get_clones(self.feature_align, num_pred)
        else:
            self.feature_align = nn.ModuleList([self.feature_align for _ in range(num_pred)])
        self.input_align = copy.deepcopy(self.feature_align)
    
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

        self.distribution_based_sampling = distribution_based_sampling
        self.criterion = criterion

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
        extra_labels = extra_labels[torch.randperm(pad_len)]
        extra_labels = extra_labels[torch.isin(extra_labels, self.novel_cls_cpu, invert=True)]
        return extra_labels
    
    def forward_train(self, samples, targets=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)      
        src, mask = features[-1].decompose()
        assert mask is not None
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
                          
        # Get input categories
        labels_list = torch.cat([t["labels"] for t in targets])
        uniq_labels = torch.unique(labels_list).to("cpu")
        uniq_labels = uniq_labels[torch.randperm(len(uniq_labels))] # detection:random
        if len(uniq_labels) < self.max_pad_len:
            pad_len = self.max_pad_len - len(uniq_labels)
            if self.distribution_based_sampling: # Sample negative categories based on the distribution.
                extra_labels = self._distribution_based_sampling(pad_len, uniq_labels=uniq_labels)
            else:
                extra_list = torch.tensor([i for i in self.all_ids if i not in uniq_labels])
                extra_labels = extra_list[torch.randperm(len(extra_list))][:pad_len]
            select_id = uniq_labels.tolist() + extra_labels.tolist()
        else:
            select_id = uniq_labels.tolist()

        # Prepare queries and embeddings for alignment
        text_query = self.text_embeddings[:, select_id].to(masks[0].device).t()
        image_align = self.image_embeddings[:, select_id].to(masks[0].device).t()
        
        input_feat_ori = (text_query.float()).detach()
        image_feat_ori = (image_align.float()).detach()

        dtype = self.patch2query.weight.dtype
        text_query = self.patch2query(text_query.type(dtype))
        select_id = torch.tensor(select_id).to(text_query.device)
        text_dict = preprocess_for_masks(srcs[0].shape[0], select_id, text_query)
        query_embed = None

        (hs_cti, hs_ofa, init_reference, inter_references, ref_enc, pre_outputs_classes, encoder_outputs_class) = self.transformer(
            srcs, masks, pos, query_embed, text_dict=text_dict)

        outputs_coords = []
        outputs_embeds = []
        input_embeds = []

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
            input_embeds.append(self.input_align[lvl](text_dict["encoded_text_all"][lvl]))
        outputs_class = pre_outputs_classes
        outputs_coord = torch.stack(outputs_coords)
        outputs_embed = torch.stack(outputs_embeds)
        input_embed = torch.stack(input_embeds)

        if inter_references.shape[3]==4:
            ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :4]], dim=0)
        else:
            ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)
        out = {
            'pred_logits': outputs_class[-1], 
            'pred_boxes': outputs_coord[-1], 
            'ref_pts': ref_pts_all[5],
            "pred_embed": outputs_embed[-1],
            "select_id": select_id,
            "image_feat": image_feat_ori}
            
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_embed)
            for temp in out["aux_outputs"]:
                temp["select_id"] = select_id
                temp["image_feat"] = image_feat_ori

        if self.two_stage:
            out["enc_outputs"] = {
                "pred_logits": encoder_outputs_class[0],
                "pred_boxes": ref_enc[-1],
                "select_id": select_id,
                "text_embed": input_embed,
                "input_feat": input_feat_ori,
            }
        return out
    
    def forward(self, samples: NestedTensor, targets=None):
            return self.forward_train(samples, targets)
    

def build(args, cfg):
    
    assert cfg.Clip_text_embeddings and cfg.Clip_image_embeddings, "Clip_text_embeddings or Clip_image_embeddings should not be None"
    text_embeddings, image_embeddings = load_embeddings(cfg.Clip_text_embeddings, cfg.Clip_image_embeddings)  

    device = torch.device(args.device)
    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)
    track_embedding_layer = None # detection
    matcher = build_matcher(args)

    weight_dict = { 'loss_ce': args.cls_loss_coef,
                    'loss_bbox': args.bbox_loss_coef,
                    'loss_giou': args.giou_loss_coef,
                    'loss_align': args.align_loss_coef,
                    }

    if args.aux_loss and args.two_stage:
        aux_weight_dict = {}
        for i in range(cfg.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        weight_dict.update({'loss_align_pre_enc':args.align_pre_loss_coef})

    losses = ['labels', 'boxes', 'align']
    losses_enc = ['labels', 'boxes', 'align_pre']

    criterion = OVMatcher(None, matcher=matcher, weight_dict=weight_dict, losses=losses, random_drop=args.random_drop,
                            calculate_negative_samples=args.calculate_negative_samples,
                            num_queries=cfg.num_queries,
                            losses_enc=losses_enc
                            )
    criterion.to(device)
    
    model = OVTR_det(
        backbone,
        transformer,
        track_embed=track_embedding_layer,
        num_feature_levels=cfg.num_feature_levels,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        max_len=args.max_len,
        distribution_based_sampling=cfg.distribution_based_sampling,
        novel_cls_cpu=novel_class
    )
    return model, criterion
