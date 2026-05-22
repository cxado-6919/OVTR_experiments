# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Grounding DINO (https://github.com/IDEA-Research/GroundingDINO)
# Copyright (c) 2023 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Optional
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor, nn
from torch.nn.init import xavier_uniform_, constant_, normal_
from util.misc import inverse_sigmoid

from .fuse_modules import BiAttentionBlock
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    ContrastiveEmbed,
    attention_protection,
)
import copy
import math

class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=6,
        num_unicoder_layers=0,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        query_dim=4,
        # for deformable encoder
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        # two stage
        two_stage_type="no",
        embed_init_tgt=False,
        extra_track_attn=True,
        # for text
        use_fusion_layer=False,
        use_checkpoint=False,
        use_transformer_ckpt=False,
        use_text_cross_attention=False,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
        prior_prob=0.01,
        log_scale=0.0,
        text_dim=256,
        attention_protection=False,
        attention_protection_mode="kl",
        attention_protection_topk=3,
        attention_protection_conf_thresh=0.25,
        computed_aux=None,
        use_ov_dptd=False,
        ov_dptd_use_historical_offsets=True,
        ov_dptd_fusion="linear_sum",
        ov_dptd_id_path_text="none",
        ov_dptd_fuse_cti=False,
        ov_dptd_store_debug=False,
        use_dptd_semantic_gate=False,
        dptd_gate_mode="heuristic",
        dptd_gate_min_score=0.3,
        dptd_gate_max_entropy=0.8,
        dptd_gate_semantic_cos_tau=0.25,
        dptd_gate_visual_cos_tau=0.25,
        dptd_gate_offset_tau=0.2,
        dptd_gate_box_iou_tau=0.3,
        dptd_gate_temperature=10.0,
        dptd_gate_min_appearance=0.1,
        dptd_gate_debug=False,
        dptd_semantic_update_suppression_thresh=0.3,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        assert query_dim == 4

        # encoder layer
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )

        if use_fusion_layer:
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,
            use_transformer_ckpt=use_transformer_ckpt,
        )

        # decoder layer
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            use_text_cross_attention=use_text_cross_attention,
            extra_track_attn=extra_track_attn,
        )

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
            prior_prob=prior_prob,
            log_scale=log_scale,
            text_dim=text_dim,
            num_queries=num_queries,
            attention_protection=attention_protection,
            attention_protection_mode=attention_protection_mode,
            attention_protection_topk=attention_protection_topk,
            attention_protection_conf_thresh=attention_protection_conf_thresh,
            computed_aux=computed_aux,
            use_transformer_ckpt=use_transformer_ckpt,
            use_ov_dptd=use_ov_dptd,
            ov_dptd_use_historical_offsets=ov_dptd_use_historical_offsets,
            ov_dptd_fusion=ov_dptd_fusion,
            ov_dptd_id_path_text=ov_dptd_id_path_text,
            ov_dptd_fuse_cti=ov_dptd_fuse_cti,
            ov_dptd_store_debug=ov_dptd_store_debug,
            use_dptd_semantic_gate=use_dptd_semantic_gate,
            dptd_gate_mode=dptd_gate_mode,
            dptd_gate_min_score=dptd_gate_min_score,
            dptd_gate_max_entropy=dptd_gate_max_entropy,
            dptd_gate_semantic_cos_tau=dptd_gate_semantic_cos_tau,
            dptd_gate_visual_cos_tau=dptd_gate_visual_cos_tau,
            dptd_gate_offset_tau=dptd_gate_offset_tau,
            dptd_gate_box_iou_tau=dptd_gate_box_iou_tau,
            dptd_gate_temperature=dptd_gate_temperature,
            dptd_gate_min_appearance=dptd_gate_min_appearance,
            dptd_gate_debug=dptd_gate_debug,
            dptd_semantic_update_suppression_thresh=dptd_semantic_update_suppression_thresh,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type == "standard":
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.two_stage_wh_embedding = None

        if two_stage_type == "no":
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self.attention_protection = attention_protection

        self._reset_parameters()
        if hasattr(self.decoder, "reset_ov_dptd_fusion_parameters"):
            self.decoder.reset_ov_dptd_fusion_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.two_stage_type != "standard":
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def _prepare_encoder_inputs(self, srcs, masks, pos_embeds):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        return {
            "src_flatten": src_flatten,
            "mask_flatten": mask_flatten,
            "lvl_pos_embed_flatten": lvl_pos_embed_flatten,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
        }

    def encode_image(self, srcs, masks, pos_embeds):
        encoder_inputs = self._prepare_encoder_inputs(srcs, masks, pos_embeds)
        memory, _ = self.encoder(
            encoder_inputs["src_flatten"],
            encoder_inputs["spatial_shapes"],
            encoder_inputs["level_start_index"],
            encoder_inputs["valid_ratios"],
            encoder_inputs["lvl_pos_embed_flatten"],
            encoder_inputs["mask_flatten"],
        )
        encoder_inputs["memory"] = memory
        return encoder_inputs

    def _ensure_encoded_text(self, text_dict):
        if "encoded_text" in text_dict:
            return text_dict

        text_dict = dict(text_dict)
        text_dict["encoded_text"] = text_dict["text_features"]
        text_dict["encoded_text_all"] = text_dict["text_features"].unsqueeze(0)
        return text_dict

    def decode_from_memory(
        self,
        memory,
        mask_flatten,
        lvl_pos_embed_flatten,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        query_pos=None,
        query_tgt=None,
        ref_pts=None,
        text_dict=None,
        dptd_sampling_offsets=None,
        dptd_gate_state=None,
        return_dptd_info=False,
    ):
        text_dict = self._ensure_encoded_text(text_dict)
        bs, _, _ = memory.shape

        if self.two_stage_type == "standard":
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            if text_dict is not None:
                enc_outputs_class_unselected = self.decoder.advance_enc_class_embed(output_memory, text_dict)
            else:
                enc_outputs_class_unselected = self.decoder.advance_enc_class_embed(output_memory)

            topk_logits = enc_outputs_class_unselected.max(-1)[0]
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory) + output_proposals
            topk = self.num_queries

            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]
            topk_coords_unact_undetach = torch.gather(
                enc_outputs_coord_unselected,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )
            topk_coords_unact = topk_coords_unact_undetach.detach()
            reference_points = topk_coords_unact.sigmoid()

            if self.embed_init_tgt:
                tgt = query_tgt[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            else:
                tgt_undetach = torch.gather(
                    output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
                )
                tgt = tgt_undetach.detach()

            if len(query_tgt) != self.num_queries:
                reference_points_track = ref_pts[self.num_queries:].unsqueeze(0).repeat(bs, 1, 1).sigmoid()
                reference_points = torch.cat([reference_points, reference_points_track], dim=1)
            init_reference_out = reference_points
        else:
            query_embed = query_pos.unsqueeze(0).expand(bs, -1, -1)
            tgt = query_tgt.unsqueeze(0).expand(bs, -1, -1)

            if ref_pts is None:
                reference_points = self.reference_points(query_embed).sigmoid()
            else:
                reference_points = ref_pts.unsqueeze(0).repeat(bs, 1, 1).sigmoid()
            init_reference_out = reference_points

        isolation_mask = None
        decoder_outputs = self.decoder(
            tgt.transpose(0, 1),
            reference_points.transpose(0, 1),
            memory.transpose(0, 1),
            spatial_shapes,
            level_start_index,
            valid_ratios,
            mask_flatten,
            tgt_mask=isolation_mask,
            text_dict=text_dict,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            num=self.num_queries,
            enc_output_undetach=None,
            dptd_sampling_offsets=dptd_sampling_offsets,
            dptd_gate_state=dptd_gate_state,
            return_dptd_info=return_dptd_info,
        )

        dptd_info = None
        if return_dptd_info:
            hs_cti, hs_ofa, inter_references, pre_outputs_classes, query_pos_track, dptd_info = decoder_outputs
        else:
            hs_cti, hs_ofa, inter_references, pre_outputs_classes, query_pos_track = decoder_outputs

        inter_references_out = inter_references
        if self.two_stage_type == "standard":
            ret = (hs_cti, hs_ofa, init_reference_out, inter_references_out, pre_outputs_classes, query_pos_track)
            if return_dptd_info:
                ret = ret + (dptd_info,)
            return ret

        ret = (hs_cti, hs_ofa, init_reference_out, pre_outputs_classes, query_pos_track)
        if return_dptd_info:
            ret = ret + (dptd_info,)
        return ret

    def forward(
        self,
        srcs,
        masks,
        pos_embeds,
        query_pos=None,
        query_tgt=None,
        ref_pts=None,
        text_dict=None,
        cache=None,
        dptd_sampling_offsets=None,
        dptd_gate_state=None,
        return_dptd_info=False,
    ):

        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        encoder_inputs = self._prepare_encoder_inputs(srcs, masks, pos_embeds)

        text_features = text_dict["text_features"]
        text_attention_mask = text_dict["text_token_mask"]

        ###### Begin Encoder ######
        memory, memory_text_all = self.encoder(
            encoder_inputs["src_flatten"],
            encoder_inputs["spatial_shapes"],
            encoder_inputs["level_start_index"],
            encoder_inputs["valid_ratios"],
            encoder_inputs["lvl_pos_embed_flatten"],
            encoder_inputs["mask_flatten"],
            text_features,
            ~text_attention_mask,
        )
        memory_text = memory_text_all[-1]
        text_dict["encoded_text"] = memory_text
        text_dict["encoded_text_all"] = memory_text_all
        return self.decode_from_memory(
            memory,
            encoder_inputs["mask_flatten"],
            encoder_inputs["lvl_pos_embed_flatten"],
            encoder_inputs["spatial_shapes"],
            encoder_inputs["level_start_index"],
            encoder_inputs["valid_ratios"],
            query_pos=query_pos,
            query_tgt=query_tgt,
            ref_pts=ref_pts,
            text_dict=text_dict,
            dptd_sampling_offsets=dptd_sampling_offsets,
            dptd_gate_state=dptd_gate_state,
            return_dptd_info=return_dptd_info,
        )


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []

        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
        else:
            self.layers = []
            del encoder_layer

            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, text_feature=None, text_attention_mask=None):

        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - padding_mask: [bs, sum(hi*wi)]

            - text_feature: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src
        memory_text = text_feature
        memory_text_all = []

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                if memory_text is None or text_attention_mask is None:
                    raise ValueError("text_feature and text_attention_mask are required when fusion layers are enabled")
                if self.use_checkpoint:
                    output, memory_text = checkpoint.checkpoint(
                        self.fusion_layers[layer_id],
                        output,
                        memory_text,
                        padding_mask,
                        text_attention_mask,
                        use_reentrant=False,
                    )
                else:
                    output, memory_text = self.fusion_layers[layer_id](
                        v=output,
                        l=memory_text,
                        attention_mask_v=padding_mask,
                        attention_mask_l=text_attention_mask,
                    )

            if memory_text is not None:
                memory_text_all.append(memory_text)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    padding_mask,
                    use_reentrant=False,
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=padding_mask,
                )

        if not memory_text_all:
            return output, None
        return output, torch.stack(memory_text_all)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        d_model=256,
        query_dim=4,
        num_feature_levels=1,
        prior_prob=0.01,
        log_scale=0.0,
        text_dim=256,
        num_queries=900,
        attention_protection=False,
        attention_protection_mode="kl",
        attention_protection_topk=3,
        attention_protection_conf_thresh=0.25,
        computed_aux=None,
        use_transformer_ckpt=False,
        use_ov_dptd=False,
        ov_dptd_use_historical_offsets=True,
        ov_dptd_fusion="linear_sum",
        ov_dptd_id_path_text="none",
        ov_dptd_fuse_cti=False,
        ov_dptd_store_debug=False,
        use_dptd_semantic_gate=False,
        dptd_gate_mode="heuristic",
        dptd_gate_min_score=0.3,
        dptd_gate_max_entropy=0.8,
        dptd_gate_semantic_cos_tau=0.25,
        dptd_gate_visual_cos_tau=0.25,
        dptd_gate_offset_tau=0.2,
        dptd_gate_box_iou_tau=0.3,
        dptd_gate_temperature=10.0,
        dptd_gate_min_appearance=0.1,
        dptd_gate_debug=False,
        dptd_semantic_update_suppression_thresh=0.3,
    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None

        self.query_scale = None
        self.bbox_embed = None

        self.d_model = d_model

        self.ref_anchor_head = None

        self.computed_aux = computed_aux
        self.attention_protection = attention_protection
        self.attention_protection_mode = attention_protection_mode
        self.attention_protection_topk = attention_protection_topk
        self.attention_protection_conf_thresh = attention_protection_conf_thresh
        self.use_transformer_ckpt = use_transformer_ckpt
        self.num_queries_det = num_queries
        self.isol_ratio = 10

        _class_embed = ContrastiveEmbed()
        class_embed_layerlist = [_class_embed for i in range(self.num_layers)]
        self.advance_class_embed = nn.ModuleList(class_embed_layerlist)
        self.advance_enc_class_embed = copy.deepcopy(_class_embed)

        # Logits scale transformation
        self.num_logits_layer = num_layers + 1
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.text_dim = text_dim
        self.log_scale = nn.Parameter(torch.full((self.num_logits_layer,), log_scale), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(self.text_dim), requires_grad=True)
        self.bias0 = nn.Parameter(torch.full((self.num_logits_layer,), bias_value), requires_grad=True)
        self.eps: float = 1e-05

        self.use_ov_dptd = use_ov_dptd
        self.ov_dptd_use_historical_offsets = ov_dptd_use_historical_offsets
        self.ov_dptd_fusion = ov_dptd_fusion
        self.ov_dptd_id_path_text = ov_dptd_id_path_text
        self.ov_dptd_fuse_cti = ov_dptd_fuse_cti
        self.ov_dptd_store_debug = ov_dptd_store_debug
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
        self.dptd_semantic_update_suppression_thresh = dptd_semantic_update_suppression_thresh
        self.ov_dptd_gate_alpha = nn.Parameter(torch.tensor(0.0)) if self.use_ov_dptd else None
        if self.use_ov_dptd:
            if self.ov_dptd_fusion not in ("linear_sum", "semantic_gate"):
                raise NotImplementedError("OV-DPTD only supports ov_dptd_fusion in {'linear_sum', 'semantic_gate'}.")
            if self.ov_dptd_fusion == "semantic_gate" and not self.use_dptd_semantic_gate:
                raise RuntimeError("ov_dptd_fusion='semantic_gate' requires use_dptd_semantic_gate=True.")
            if self.ov_dptd_fuse_cti:
                raise NotImplementedError("OV-DPTD v4 keeps CTI fusion disabled; ov_dptd_fuse_cti=True is unsupported.")
            if self.use_dptd_semantic_gate and self.dptd_gate_mode != "heuristic":
                raise NotImplementedError("OV-DPTD v4 only supports dptd_gate_mode='heuristic'.")
            if self.ov_dptd_id_path_text != "none":
                raise NotImplementedError("OV-DPTD v1 only supports ov_dptd_id_path_text='none'.")
            self.ov_dptd_ofa_ada_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
            self.ov_dptd_ofa_id_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
            self.ov_dptd_ofa_out_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
            self.ov_dptd_cti_ada_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
            self.ov_dptd_cti_id_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
            self.ov_dptd_cti_out_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        else:
            self.ov_dptd_ofa_ada_proj = nn.ModuleList()
            self.ov_dptd_ofa_id_proj = nn.ModuleList()
            self.ov_dptd_ofa_out_proj = nn.ModuleList()
            self.ov_dptd_cti_ada_proj = nn.ModuleList()
            self.ov_dptd_cti_id_proj = nn.ModuleList()
            self.ov_dptd_cti_out_proj = nn.ModuleList()

    @staticmethod
    def _reset_linear_identity(linear):
        nn.init.eye_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)

    @staticmethod
    def _reset_linear_zero(linear):
        nn.init.constant_(linear.weight, 0.0)
        nn.init.constant_(linear.bias, 0.0)

    def reset_ov_dptd_fusion_parameters(self):
        if not self.use_ov_dptd:
            return
        for layer_id in range(self.num_layers):
            self._reset_linear_identity(self.ov_dptd_ofa_ada_proj[layer_id])
            self._reset_linear_zero(self.ov_dptd_ofa_id_proj[layer_id])
            self._reset_linear_identity(self.ov_dptd_ofa_out_proj[layer_id])
            self._reset_linear_identity(self.ov_dptd_cti_ada_proj[layer_id])
            self._reset_linear_zero(self.ov_dptd_cti_id_proj[layer_id])
            self._reset_linear_identity(self.ov_dptd_cti_out_proj[layer_id])

    def _fuse_ov_dptd_ofa(self, layer_id, ada_ofa, id_ofa):
        return self.ov_dptd_ofa_out_proj[layer_id](
            self.ov_dptd_ofa_ada_proj[layer_id](ada_ofa)
            + self.ov_dptd_ofa_id_proj[layer_id](id_ofa)
        )

    def _fuse_ov_dptd_cti(self, layer_id, ada_cti, id_ofa):
        return self.ov_dptd_cti_out_proj[layer_id](
            self.ov_dptd_cti_ada_proj[layer_id](ada_cti)
            + self.ov_dptd_cti_id_proj[layer_id](id_ofa)
        )

    @staticmethod
    def _new_ov_dptd_debug(enabled):
        if not enabled:
            return None
        return {
            "ov_dptd_enabled": True,
            "ov_dptd_num_track_queries": 0,
            "ov_dptd_historical_offset_used_count": 0,
            "ov_dptd_historical_offset_fallback_count": 0,
            "dptd_gate_mean": 1.0,
            "dptd_gate_min": 1.0,
            "dptd_gate_max": 1.0,
            "dptd_gate_low_count": 0,
            "semantic_consistency_mean": 1.0,
            "visual_consistency_mean": 1.0,
            "offset_consistency_mean": 1.0,
            "box_consistency_mean": 1.0,
            "dptd_gate_box_conf_deferred": True,
        }

    def _normalize_ov_dptd_offsets(self, offsets, tgt):
        if offsets is None:
            return None
        bs = tgt.shape[1]
        num_queries = tgt.shape[0]
        if offsets.dim() == 5:
            offsets = offsets.unsqueeze(0)
        if offsets.dim() != 6:
            return None
        if offsets.shape[0] == 1 and bs != 1:
            offsets = offsets.expand(bs, -1, -1, -1, -1, -1)
        if offsets.shape[:2] != (bs, num_queries):
            return None
        return offsets.to(device=tgt.device, dtype=tgt.dtype)

    def _gate_state_tensor(self, gate_state, name, like, expected_dim=None, dtype=None):
        if not isinstance(gate_state, dict) or name not in gate_state:
            return None
        value = gate_state[name]
        if value is None or not isinstance(value, torch.Tensor):
            return None
        value = value.to(device=like.device, dtype=dtype or like.dtype)
        if expected_dim is not None and value.dim() == expected_dim - 1:
            value = value.unsqueeze(0)
        return value

    def _semantic_gate_ones(self, tgt):
        return torch.ones(tgt.shape[1], tgt.shape[0], device=tgt.device, dtype=tgt.dtype)

    def _compute_dptd_semantic_gate(
        self,
        logits,
        ad_sampling_offsets,
        historical_offsets,
        gate_state,
        track_start,
        tgt,
        debug=None,
    ):
        gate = self._semantic_gate_ones(tgt)
        if not self.use_dptd_semantic_gate or not isinstance(gate_state, dict):
            return gate

        bs, num_queries = gate.shape
        if track_start >= num_queries:
            return gate

        text_memory = self._gate_state_tensor(gate_state, 'text_memory_embeddings', logits, expected_dim=3, dtype=logits.dtype)
        prev_semantic = self._gate_state_tensor(gate_state, 'semantic_proto', logits, expected_dim=3, dtype=logits.dtype)
        memory_valid = self._gate_state_tensor(gate_state, 'memory_valid', logits, expected_dim=2, dtype=torch.bool)
        if text_memory is None or prev_semantic is None or memory_valid is None:
            return gate
        if text_memory.shape[0] == 1 and bs != 1:
            text_memory = text_memory.expand(bs, -1, -1)
        if prev_semantic.shape[0] == 1 and bs != 1:
            prev_semantic = prev_semantic.expand(bs, -1, -1)
        if memory_valid.shape[0] == 1 and bs != 1:
            memory_valid = memory_valid.expand(bs, -1)
        if prev_semantic.shape[:2] != (bs, num_queries) or memory_valid.shape[:2] != (bs, num_queries):
            return gate

        num_cls = min(int(logits.shape[-1]), int(text_memory.shape[1]))
        if num_cls <= 0:
            return gate
        score = logits[..., :num_cls].float().sigmoid()
        prob = score / score.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        current_semantic = F.normalize(prob @ text_memory[:, :num_cls].float(), dim=-1, eps=1e-6)
        entropy_den = math.log(num_cls) if num_cls > 1 else 1.0
        entropy = -(prob * prob.clamp_min(1e-6).log()).sum(dim=-1) / entropy_den
        entropy = entropy.clamp(0.0, 1.0)

        score_max = score.max(dim=-1).values
        score_conf = ((score_max - float(self.dptd_gate_min_score)) / max(1.0 - float(self.dptd_gate_min_score), 1e-6)).clamp(0.0, 1.0)
        entropy_conf = ((float(self.dptd_gate_max_entropy) - entropy) / max(float(self.dptd_gate_max_entropy), 1e-6)).clamp(0.0, 1.0)
        semantic_cos = F.cosine_similarity(current_semantic, prev_semantic.float(), dim=-1, eps=1e-6).clamp(-1.0, 1.0)
        semantic_conf = torch.sigmoid((semantic_cos - float(self.dptd_gate_semantic_cos_tau)) * float(self.dptd_gate_temperature)).clamp(0.0, 1.0)

        offset_conf = torch.ones_like(score_conf)
        if historical_offsets is not None and tuple(historical_offsets.shape) == tuple(ad_sampling_offsets.shape):
            offset_l1 = (ad_sampling_offsets.float() - historical_offsets.float()).abs().mean(dim=(2, 3, 4, 5))
            offset_conf = torch.exp(-offset_l1 / max(float(self.dptd_gate_offset_tau), 1e-6)).clamp(0.0, 1.0)

        valid = memory_valid.bool()
        track_mask = torch.zeros_like(valid)
        track_mask[:, track_start:] = True
        valid = valid & track_mask
        raw_gate = (score_conf * entropy_conf * semantic_conf * offset_conf).clamp(0.0, 1.0)
        gate = torch.where(valid, raw_gate, torch.ones_like(raw_gate))
        gate[:, :track_start] = 1.0
        gate = gate.clamp(min=float(self.dptd_gate_min_appearance), max=1.0).to(dtype=tgt.dtype)

        if debug is not None:
            track_values = gate[:, track_start:]
            valid_values = gate[valid]
            if track_values.numel() > 0:
                debug['dptd_gate_mean'] = float(track_values.detach().float().mean().item())
                debug['dptd_gate_min'] = float(track_values.detach().float().min().item())
                debug['dptd_gate_max'] = float(track_values.detach().float().max().item())
                debug['dptd_gate_low_count'] = int((track_values.detach().float() < float(self.dptd_semantic_update_suppression_thresh)).sum().item())
            if valid_values.numel() > 0:
                debug['semantic_consistency_mean'] = float(semantic_cos[valid].detach().float().mean().item())
                debug['offset_consistency_mean'] = float(offset_conf[valid].detach().float().mean().item())
            debug['visual_consistency_mean'] = float(debug.get('visual_consistency_mean', 1.0))
            debug['box_consistency_mean'] = 1.0
            debug['dptd_gate_box_conf_deferred'] = True
        return gate

    def _fuse_ov_dptd_ofa_semantic_gate(self, layer_id, ada_ofa, id_ofa, gate):
        id_delta = self.ov_dptd_ofa_id_proj[layer_id](id_ofa)
        correction = (1.0 - gate.transpose(0, 1).unsqueeze(-1).to(dtype=ada_ofa.dtype)) * id_delta
        return ada_ofa + self.ov_dptd_gate_alpha.to(dtype=ada_ofa.dtype) * correction


    def _forward_ov_dptd(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        src_padding_mask=None,
        tgt_mask: Optional[Tensor] = None,
        num=None,
        pos: Optional[Tensor] = None,
        text_dict=None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        enc_output_undetach=None,
        dptd_sampling_offsets=None,
        dptd_gate_state=None,
        return_dptd_info=False,
    ):
        if self.use_transformer_ckpt:
            raise RuntimeError("OV-DPTD v1 does not support use_transformer_ckpt=True.")

        output = tgt
        fixed_track_tgt = tgt.clone()
        intermediate_cti = []
        intermediate_ofa = []
        ref_points = []
        text_attention_mask = ~text_dict["text_token_mask"]
        pre_outputs_classes = []
        historical_offsets = self._normalize_ov_dptd_offsets(dptd_sampling_offsets, tgt)
        dptd_debug = self._new_ov_dptd_debug(self.ov_dptd_store_debug)
        last_ad_sampling_offsets = None
        last_gate_values = None

        self.num_queries_cur = tgt.shape[0]
        self.select_text_num = text_dict["select_text_num"]
        self.text_bias, self.log_scale_cls = self.get_logits_bias(text_dict["encoded_text"], self.num_queries_cur)

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[None, :]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[None, :]

            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            ada_cti, ada_ofa, ad_sampling_offsets = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=text_dict["encoded_text"],
                text_attention_mask=text_attention_mask,
                memory=src,
                memory_key_padding_mask=src_padding_mask,
                memory_level_start_index=src_level_start_index,
                memory_spatial_shapes=src_spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                num=num,
                return_sampling_offsets=True,
            )
            last_ad_sampling_offsets = ad_sampling_offsets

            id_tgt = output
            track_start = self.num_queries_cur if num is None else min(max(int(num), 0), self.num_queries_cur)
            if track_start < self.num_queries_cur:
                id_tgt = output.clone()
                id_tgt[track_start:] = fixed_track_tgt[track_start:]

            id_ofa, _, id_debug = layer.forward_identity_path(
                tgt=id_tgt,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory=src,
                memory_key_padding_mask=src_padding_mask,
                memory_level_start_index=src_level_start_index,
                memory_spatial_shapes=src_spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                num=num,
                historical_sampling_offsets=historical_offsets,
                use_historical_offsets=self.ov_dptd_use_historical_offsets,
                store_debug=self.ov_dptd_store_debug,
            )
            if dptd_debug is not None:
                dptd_debug["ov_dptd_num_track_queries"] = max(
                    dptd_debug["ov_dptd_num_track_queries"],
                    id_debug.get("num_track_queries", 0),
                )
                dptd_debug["ov_dptd_historical_offset_used_count"] += id_debug.get("used_count", 0)
                dptd_debug["ov_dptd_historical_offset_fallback_count"] += id_debug.get("fallback_count", 0)

            output = ada_cti
            output_norm = self.norm(output)
            pre_outputs_class = self.pre_class_embed(output_norm.transpose(0, 1), text_dict, layer_id)
            gate_values = self._compute_dptd_semantic_gate(
                pre_outputs_class,
                ad_sampling_offsets,
                historical_offsets,
                dptd_gate_state,
                track_start,
                tgt,
                debug=dptd_debug,
            )
            last_gate_values = gate_values
            if self.ov_dptd_fusion == "semantic_gate":
                output_ofa = self._fuse_ov_dptd_ofa_semantic_gate(layer_id, ada_ofa, id_ofa, gate_values)
            else:
                output_ofa = self._fuse_ov_dptd_ofa(layer_id, ada_ofa, id_ofa)

            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output_ofa)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                if layer_id in self.computed_aux:
                    reference_points = new_reference_points.detach()
                    ref_points.append(new_reference_points)
                else:
                    reference_points = new_reference_points

            if self.attention_protection:
                tgt_mask = attention_protection(
                    pre_outputs_class,
                    self.num_queries_det,
                    layer_id,
                    isol_ratio=self.isol_ratio,
                    mode=self.attention_protection_mode,
                    topk=self.attention_protection_topk,
                    conf_thresh=self.attention_protection_conf_thresh,
                )
            else:
                tgt_mask = None

            if layer_id in self.computed_aux:
                intermediate_cti.append(output_norm)
                intermediate_ofa.append(output_ofa)
                pre_outputs_classes.append(pre_outputs_class)

        ret = [
            torch.stack([itm_out_cti.transpose(0, 1) for itm_out_cti in intermediate_cti]),
            torch.stack([itm_out_ofa.transpose(0, 1) for itm_out_ofa in intermediate_ofa]),
            torch.stack([itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]),
            torch.stack(pre_outputs_classes),
            query_pos,
        ]
        if return_dptd_info:
            ret.append({
                "sampling_offsets": last_ad_sampling_offsets,
                "semantic_gate": last_gate_values,
                "debug": dptd_debug,
            })
        return ret

    def get_logits_bias(self, embedding, num_real_queries):
        bs,cls_len,_ = embedding.shape
        dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang).repeat(self.num_logits_layer,1,1) + self.bias0.repeat(bs,cls_len,1).permute(2,0,1)
        bias = dot_product_proj_tokens_bias.repeat(num_real_queries, 1, 1, 1).permute(1,2,0,3)
        log_scale = (self.log_scale.exp() + self.eps).repeat(bs, num_real_queries, cls_len, 1).permute(3,0,1,2)
        return bias, log_scale

    def logits_with_bias(self, dot_product_logit, bias, log_scale):
        dot_product_logit = dot_product_logit[..., : self.select_text_num]
        dot_product_logit = (dot_product_logit / log_scale) + bias
        dot_product_logit = torch.clamp(dot_product_logit, max=500)
        dot_product_logit = torch.clamp(dot_product_logit, min=-500)
        return dot_product_logit

    def pre_class_embed(self, output, text_dict, layer_id=-1, encoder=False):
        if encoder:
            outputs_class = self.advance_enc_class_embed(output, text_dict)
            outputs_class = self.logits_with_bias(outputs_class, self.text_bias[layer_id,:,:self.num_queries_det,:], self.log_scale_cls[layer_id,:,:self.num_queries_det,:])
        else:
            outputs_class = self.advance_class_embed[layer_id](output, text_dict)
            outputs_class = self.logits_with_bias(outputs_class, self.text_bias[layer_id,:,:self.num_queries_cur,:], self.log_scale_cls[layer_id,:,:self.num_queries_cur,:])
        return outputs_class

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                src_padding_mask=None, tgt_mask: Optional[Tensor] = None, num=None, pos: Optional[Tensor] = None,
                text_dict=None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                enc_output_undetach=None, dptd_sampling_offsets=None, dptd_gate_state=None, return_dptd_info=False
                ):
        if self.use_ov_dptd:
            return self._forward_ov_dptd(
                tgt,
                reference_points,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_valid_ratios,
                src_padding_mask=src_padding_mask,
                tgt_mask=tgt_mask,
                num=num,
                pos=pos,
                text_dict=text_dict,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                enc_output_undetach=enc_output_undetach,
                dptd_sampling_offsets=dptd_sampling_offsets,
                dptd_gate_state=dptd_gate_state,
                return_dptd_info=return_dptd_info,
            )

        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt
        intermediate_cti = []
        intermediate_ofa = []
        ref_points = []
        text_attention_mask = ~text_dict["text_token_mask"]
        pre_outputs_classes = []

        self.num_queries_cur = tgt.shape[0]
        self.select_text_num = text_dict["select_text_num"]
        self.text_bias, self.log_scale_cls = self.get_logits_bias(text_dict["encoded_text"], self.num_queries_cur)

        for layer_id, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[None, :]

            query_sine_embed = gen_sineembed_for_position(
                    reference_points_input[:, :, 0, :]
            )  # nq, bs, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            def _run_decoder_layer(
                tgt_arg,
                query_pos_arg,
                query_sine_arg,
                reference_points_arg,
                memory_text_arg,
                memory_arg,
                memory_pos_arg,
                *,
                decoder_layer=layer,
                tgt_key_padding_mask_arg=tgt_key_padding_mask,
                text_attention_mask_arg=text_attention_mask,
                memory_key_padding_mask_arg=src_padding_mask,
                memory_level_start_index_arg=src_level_start_index,
                memory_spatial_shapes_arg=src_spatial_shapes,
                self_attn_mask_arg=tgt_mask,
                cross_attn_mask_arg=memory_mask,
                num_arg=num,
            ):
                return decoder_layer(
                    tgt=tgt_arg,
                    tgt_query_pos=query_pos_arg,
                    tgt_query_sine_embed=query_sine_arg,
                    tgt_key_padding_mask=tgt_key_padding_mask_arg,
                    tgt_reference_points=reference_points_arg,
                    memory_text=memory_text_arg,
                    text_attention_mask=text_attention_mask_arg,
                    memory=memory_arg,
                    memory_key_padding_mask=memory_key_padding_mask_arg,
                    memory_level_start_index=memory_level_start_index_arg,
                    memory_spatial_shapes=memory_spatial_shapes_arg,
                    memory_pos=memory_pos_arg,
                    self_attn_mask=self_attn_mask_arg,
                    cross_attn_mask=cross_attn_mask_arg,
                    num=num_arg,
                )

            # main process
            if self.training and self.use_transformer_ckpt:
                output, output_ofa = checkpoint.checkpoint(
                    _run_decoder_layer,
                    output,
                    query_pos,
                    query_sine_embed,
                    reference_points_input,
                    text_dict["encoded_text"],
                    src,
                    pos,
                    use_reentrant=False,
                )
            else:
                output, output_ofa = _run_decoder_layer(
                    output,
                    query_pos,
                    query_sine_embed,
                    reference_points_input,
                    text_dict["encoded_text"],
                    src,
                    pos,
                )

            # iter update
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output_ofa)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                if layer_id in self.computed_aux:
                    reference_points = new_reference_points.detach()
                    ref_points.append(new_reference_points)
                else:
                    reference_points = new_reference_points

            output_norm = self.norm(output)

            pre_outputs_class = self.pre_class_embed(output_norm.transpose(0, 1), text_dict, layer_id)
            if self.attention_protection:
                tgt_mask = attention_protection(
                    pre_outputs_class,
                    self.num_queries_det,
                    layer_id,
                    isol_ratio=self.isol_ratio,
                    mode=self.attention_protection_mode,
                    topk=self.attention_protection_topk,
                    conf_thresh=self.attention_protection_conf_thresh,
                )
            else:
                tgt_mask = None

            if layer_id in self.computed_aux:
                intermediate_cti.append(output_norm)
                intermediate_ofa.append(output_ofa)
                pre_outputs_classes.append(pre_outputs_class)


        return [
            torch.stack([itm_out_cti.transpose(0, 1) for itm_out_cti in intermediate_cti]),
            torch.stack([itm_out_ofa.transpose(0, 1) for itm_out_ofa in intermediate_ofa]),
            torch.stack([itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]),
            torch.stack(pre_outputs_classes),
            query_pos,
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None
    ):
        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_text_feat_guide=False,
        use_text_cross_attention=False,
        extra_track_attn=True
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        # image ffn
        self.linear3 = nn.Linear(d_model, d_ffn)
        self.dropout6 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear4 = nn.Linear(d_ffn, d_model)
        self.dropout7 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm5 = nn.LayerNorm(d_model)

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide
        self.use_text_cross_attention = use_text_cross_attention
        self.num_heads = n_heads

        # update track query_embed
        self.extra_track_attn = extra_track_attn
        if self.extra_track_attn:
            print('Training with Extra Self Attention in Every Decoder.', flush=True)
            self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout5 = nn.Dropout(dropout)
            self.norm4 = nn.LayerNorm(d_model)



    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.amp.autocast("cuda", enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_ffn_align(self, tgt):
        with torch.amp.autocast("cuda", enabled=False):
            tgt2 = self.linear4(self.dropout6(self.activation(self.linear3(tgt))))
        tgt = tgt + self.dropout7(tgt2)
        tgt = self.norm5(tgt)
        return tgt

    def _forward_track_attn(self, tgt, query_pos, attn_mask=None, num=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if q.shape[1] > num:
            tgt2 = self.update_attn(q[:,num:].transpose(0,1),
                                    k[:,num:].transpose(0,1),
                                    tgt[:,num:].transpose(0,1))[0].transpose(0,1)
            tgt = torch.cat([tgt[:,:num],self.norm4(tgt[:,num:]+self.dropout5(tgt2))], dim=1)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
        num=None,
        return_sampling_offsets=False,
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
        """
        assert cross_attn_mask is None

        # extra track attention
        if self.extra_track_attn:
            tgt = self._forward_track_attn(
                tgt.transpose(0, 1),
                tgt_query_pos.transpose(0, 1),
                attn_mask=None,
                num=num).transpose(0,1)

        if self_attn_mask is not None:
            self_attn_mask = self_attn_mask.squeeze(dim=0)

        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # image cross-attention
        cross_attn_out = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
            return_sampling_offsets=return_sampling_offsets,
        )
        sampling_offsets = None
        if return_sampling_offsets:
            tgt2, sampling_offsets = cross_attn_out
        else:
            tgt2 = cross_attn_out
        tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt_aligned = tgt # aligned queries

        # text cross-attention in the CTI branch
        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, None),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        # ffn in the CTI branch
        tgt_cti  = self.forward_ffn(tgt)

        # image ffn in the OFA branch
        tgt_aligned = self.forward_ffn_align(tgt_aligned)

        if return_sampling_offsets:
            return tgt_cti, tgt_aligned, sampling_offsets
        return tgt_cti, tgt_aligned

    def forward_identity_path(
        self,
        # for tgt
        tgt: Optional[Tensor],
        tgt_query_pos: Optional[Tensor] = None,
        tgt_query_sine_embed: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,
        # for memory
        memory: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,
        memory_spatial_shapes: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        # sa
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
        num=None,
        historical_sampling_offsets: Optional[Tensor] = None,
        use_historical_offsets: bool = True,
        store_debug: bool = False,
    ):
        assert cross_attn_mask is None

        track_start = tgt.shape[0] if num is None else min(max(int(num), 0), tgt.shape[0])
        debug = {
            "num_track_queries": int(tgt.shape[0] - track_start),
            "used_count": 0,
            "fallback_count": 0,
        }

        if self.extra_track_attn:
            tgt = self._forward_track_attn(
                tgt.transpose(0, 1),
                tgt_query_pos.transpose(0, 1),
                attn_mask=None,
                num=track_start,
            ).transpose(0, 1)

        if self_attn_mask is not None:
            self_attn_mask = self_attn_mask.squeeze(dim=0)

        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        cross_query = self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1)
        bs, num_query, _ = cross_query.shape
        predicted_offsets = self.cross_attn.sampling_offsets(cross_query).view(
            bs,
            num_query,
            self.cross_attn.num_heads,
            self.cross_attn.num_levels,
            self.cross_attn.num_points,
            2,
        )
        override_offsets = predicted_offsets
        if use_historical_offsets and debug["num_track_queries"] > 0:
            if historical_sampling_offsets is not None and tuple(historical_sampling_offsets.shape) == tuple(predicted_offsets.shape):
                override_offsets = predicted_offsets.clone()
                override_offsets[:, track_start:] = historical_sampling_offsets[:, track_start:].to(
                    device=predicted_offsets.device,
                    dtype=predicted_offsets.dtype,
                )
                debug["used_count"] = int(bs * debug["num_track_queries"])
            else:
                debug["fallback_count"] = int(bs * debug["num_track_queries"])

        tgt2, sampling_offsets = self.cross_attn(
            query=cross_query,
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
            override_sampling_offsets=override_offsets,
            return_sampling_offsets=True,
        )
        tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        id_ofa = self.forward_ffn_align(tgt)
        return id_ofa, sampling_offsets, debug


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        two_stage_type=args.two_stage_type,
        embed_init_tgt=args.embed_init_tgt,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,
        extra_track_attn=args.extra_track_attn,
        prior_prob=args.prior_prob,
        log_scale=args.log_scale,
        text_dim=args.text_dim,
        attention_protection=getattr(args, "attention_protection", False),
        attention_protection_mode=getattr(args, "attention_protection_mode", "kl"),
        attention_protection_topk=getattr(args, "attention_protection_topk", 3),
        attention_protection_conf_thresh=getattr(args, "attention_protection_conf_thresh", 0.25),
        computed_aux=args.computed_aux,
        use_ov_dptd=getattr(args, "use_ov_dptd", False),
        ov_dptd_use_historical_offsets=getattr(args, "ov_dptd_use_historical_offsets", True),
        ov_dptd_fusion=getattr(args, "ov_dptd_fusion", "linear_sum"),
        ov_dptd_id_path_text=getattr(args, "ov_dptd_id_path_text", "none"),
        ov_dptd_fuse_cti=getattr(args, "ov_dptd_fuse_cti", False),
        ov_dptd_store_debug=getattr(args, "ov_dptd_store_debug", False),
        use_dptd_semantic_gate=getattr(args, "use_dptd_semantic_gate", False),
        dptd_gate_mode=getattr(args, "dptd_gate_mode", "heuristic"),
        dptd_gate_min_score=getattr(args, "dptd_gate_min_score", 0.3),
        dptd_gate_max_entropy=getattr(args, "dptd_gate_max_entropy", 0.8),
        dptd_gate_semantic_cos_tau=getattr(args, "dptd_gate_semantic_cos_tau", 0.25),
        dptd_gate_visual_cos_tau=getattr(args, "dptd_gate_visual_cos_tau", 0.25),
        dptd_gate_offset_tau=getattr(args, "dptd_gate_offset_tau", 0.2),
        dptd_gate_box_iou_tau=getattr(args, "dptd_gate_box_iou_tau", 0.3),
        dptd_gate_temperature=getattr(args, "dptd_gate_temperature", 10.0),
        dptd_gate_min_appearance=getattr(args, "dptd_gate_min_appearance", 0.1),
        dptd_gate_debug=getattr(args, "dptd_gate_debug", False),
        dptd_semantic_update_suppression_thresh=getattr(args, "dptd_semantic_update_suppression_thresh", 0.3),
    )
