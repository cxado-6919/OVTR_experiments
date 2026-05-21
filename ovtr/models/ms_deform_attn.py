# ------------------------------------------------------------------------
# Modified from Grounding DINO (https://github.com/IDEA-Research/GroundingDINO)
# Copyright (c) 2023 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_

from .ops import HAS_MSDA_EXT, MSDA
from .quant_utils import (
    maybe_observe_and_quantize_attention,
    maybe_observe_and_quantize_msda_aggregation,
    pack_uint4,
    quantize_affine_uint,
)


# helpers
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention Module used in Deformable-DETR

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dim (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): The number of attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query
            in each head. Default: 4.
        img2col_steps (int): The step used in image_to_column. Defualt: 64.
            dropout (float): Dropout layer used in output. Default: 0.1.
        batch_first (bool): if ``True``, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        batch_first: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads, but got {} and {}".format(
                    embed_dim, num_heads
                )
            )
        head_dim = embed_dim // num_heads

        self.batch_first = batch_first

        if not _is_power_of_2(head_dim):
            warnings.warn(
                """
                You'd better set d_model in MSDeformAttn to make sure that
                each dim of the attention head a power of 2, which is more efficient.
                """
            )

        self.im2col_step = img2col_step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.init_weights()

    def _reset_parameters(self):
        return self.init_weights()

    def init_weights(self):
        """
        Default initialization for Parameters of Module.
        """
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def freeze_sampling_offsets(self):
        print("Freeze sampling offsets")
        self.sampling_offsets.weight.requires_grad = False
        self.sampling_offsets.bias.requires_grad = False

    def freeze_attention_weights(self):
        print("Freeze attention weights")
        self.attention_weights.weight.requires_grad = False
        self.attention_weights.bias.requires_grad = False

    @staticmethod
    def _require_lowbit_symbol(name: str):
        if not HAS_MSDA_EXT or MSDA is None:
            raise RuntimeError(f"--quant_deploy int_msda requires CUDA extension symbol {name}.")
        if not hasattr(MSDA, name) and name == "ms_deform_attn_lowbit_forward" and hasattr(MSDA, "ms_deform_attn_int_forward"):
            return getattr(MSDA, "ms_deform_attn_int_forward")
        if not hasattr(MSDA, name):
            raise RuntimeError(f"--quant_deploy int_msda requires CUDA extension symbol {name}.")
        return getattr(MSDA, name)

    @staticmethod
    def _quantizer_scalar(quantizer: nn.Module, attr: str, device: torch.device) -> torch.Tensor:
        value = getattr(quantizer, attr)
        if value is None or value.numel() != 1:
            raise RuntimeError(f"Expected scalar {attr} for int_msda activation quantizer.")
        return value.detach().reshape(1).to(device=device, dtype=torch.float32)

    def _lowbit_linear(self, linear: nn.Linear, x: torch.Tensor, input_quantizer: nn.Module) -> torch.Tensor:
        lowbit_linear_forward = self._require_lowbit_symbol("lowbit_linear_forward")
        if not x.is_cuda:
            raise RuntimeError("--quant_deploy int_msda requires CUDA tensors.")
        if not hasattr(linear, "_ovtr_quant_int4_weight_packed"):
            raise RuntimeError(f"Missing int_msda export key for {getattr(linear, '_ovtr_quant_name', 'linear')}.")

        rows = x.reshape(-1, linear.in_features).contiguous()
        x_scale = self._quantizer_scalar(input_quantizer, "scale", rows.device)
        x_zero_point = self._quantizer_scalar(input_quantizer, "zero_point", rows.device)
        x_q = quantize_affine_uint(rows.float(), x_scale, x_zero_point, bit_width=4)
        x_packed = pack_uint4(x_q)
        out = lowbit_linear_forward(
            x_packed,
            linear._ovtr_quant_int4_weight_packed,
            linear._ovtr_quant_int4_weight_sum,
            x_scale,
            x_zero_point,
            linear._ovtr_quant_int4_weight_scale,
            linear._ovtr_quant_bias_fp32,
            rows.shape[0],
            linear.in_features,
            linear.out_features,
        )
        return out.reshape(*x.shape[:-1], linear.out_features)

    def _quantize_attention_per_head(self, attention_weights: torch.Tensor) -> torch.Tensor:
        quantizer = getattr(self, "_ovtr_quant_attention_quantizer", None)
        if quantizer is None or quantizer.scale.numel() != self.num_heads:
            raise RuntimeError("Missing per-head UINT8 attention qparams for int_msda.")
        scale = quantizer.scale.detach().to(device=attention_weights.device, dtype=attention_weights.dtype)
        zero_point = torch.round(
            quantizer.zero_point.detach().to(device=attention_weights.device, dtype=attention_weights.dtype)
        ).clamp(0, 255)
        scale = scale.view(1, 1, self.num_heads, 1).clamp(min=1e-8)
        zero_point = zero_point.view(1, 1, self.num_heads, 1)
        q = torch.round(attention_weights / scale + zero_point).clamp(0, 255)
        return q.to(torch.uint8)

    def _forward_int_msda(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            raise RuntimeError("--quant_deploy int_msda is eval/inference only.")
        lowbit_msda_forward = self._require_lowbit_symbol("ms_deform_attn_lowbit_forward")

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        head_dim = self.embed_dim // self.num_heads

        value_proj_input_q = self.value_proj._ovtr_quant_input_quantizer
        value_proj_output_q = self.value_proj._ovtr_quant_output_quantizer
        value_real = self._lowbit_linear(self.value_proj, value, value_proj_input_q)
        value_real = value_real.view(bs, num_value, self.num_heads, head_dim)
        value_q = quantize_affine_uint(
            value_real.float(),
            self._quantizer_scalar(value_proj_output_q, "scale", value_real.device),
            self._quantizer_scalar(value_proj_output_q, "zero_point", value_real.device),
            bit_width=4,
        )
        value_zero_point = self._quantizer_scalar(value_proj_output_q, "zero_point", value_real.device)
        if key_padding_mask is not None:
            z_v = int(round(float(value_zero_point.item())))
            value_q = value_q.masked_fill(key_padding_mask[:, :, None, None].to(device=value_q.device), z_v)
        value_packed = pack_uint4(value_q)

        sampling_offsets = self._lowbit_linear(
            self.sampling_offsets,
            query,
            self.sampling_offsets._ovtr_quant_input_quantizer,
        ).view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_logits = self._lowbit_linear(
            self.attention_weights,
            query,
            self.attention_weights._ovtr_quant_input_quantizer,
        ).view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_logits.softmax(-1)
        attention_q = self._quantize_attention_per_head(attention_weights).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        attention_quantizer = self._ovtr_quant_attention_quantizer
        aggregation = lowbit_msda_forward(
            value_packed,
            spatial_shapes.contiguous(),
            level_start_index.contiguous(),
            sampling_locations.contiguous(),
            attention_q.contiguous(),
            self._quantizer_scalar(value_proj_output_q, "scale", value_real.device),
            value_zero_point,
            attention_quantizer.scale.detach().float().to(value_real.device).contiguous(),
            attention_quantizer.zero_point.detach().float().to(value_real.device).contiguous(),
            head_dim,
            self.im2col_step,
        )

        aggregation_q = self._ovtr_quant_aggregation_output_quantizer
        aggregation_uint4 = quantize_affine_uint(
            aggregation.float(),
            self._quantizer_scalar(aggregation_q, "scale", aggregation.device),
            self._quantizer_scalar(aggregation_q, "zero_point", aggregation.device),
            bit_width=4,
        )
        aggregation_packed = pack_uint4(aggregation_uint4)
        output = self._require_lowbit_symbol("lowbit_linear_forward")(
            aggregation_packed,
            self.output_proj._ovtr_quant_int4_weight_packed,
            self.output_proj._ovtr_quant_int4_weight_sum,
            self._quantizer_scalar(aggregation_q, "scale", aggregation.device),
            self._quantizer_scalar(aggregation_q, "zero_point", aggregation.device),
            self.output_proj._ovtr_quant_int4_weight_scale,
            self.output_proj._ovtr_quant_bias_fp32,
            bs * num_query,
            self.output_proj.in_features,
            self.output_proj.out_features,
        )
        self._ovtr_int_msda_dispatch_count = getattr(self, "_ovtr_int_msda_dispatch_count", 0) + 1
        return output.view(bs, num_query, self.output_proj.out_features)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        override_sampling_offsets: Optional[torch.Tensor] = None,
        return_sampling_offsets: bool = False,
        **kwargs
    ) -> torch.Tensor:

        """Forward Function of MultiScaleDeformableAttention

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)`
            value (torch.Tensor): Value embeddings with shape
                `(num_key, bs, embed_dim)`
            query_pos (torch.Tensor): The position embedding for `query`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with shape `(bs, num_key)`,
                indicating which elements within `key` to be ignored in attention.
            reference_points (torch.Tensor): The normalized reference points
                with shape `(bs, num_query, num_levels, 2)`,
                all elements is range in [0, 1], top-left (0, 0),
                bottom-right (1, 1), including padding are.
                or `(N, Length_{query}, num_levels, 4)`, add additional
                two dimensions `(h, w)` to form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in different levels.
                With shape `(num_levels, 2)`, last dimension represents `(h, w)`.
            level_start_index (torch.Tensor): The start index of each level. A tensor with
                shape `(num_levels, )` which can be represented as
                `[0, h_0 * w_0, h_0 * w_0 + h_1 * w_1, ...]`.

        Returns:
            torch.Tensor: forward results with shape `(num_query, bs, embed_dim)`
        """

        if value is None:
            value = query

        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        if getattr(self, "_ovtr_quant_deploy_mode", "none") == "int_msda":
            if override_sampling_offsets is not None or return_sampling_offsets:
                raise RuntimeError(
                    "OV-DPTD sampling offset override/return is not supported with "
                    "--quant_deploy int_msda."
                )
            output = self._forward_int_msda(
                query,
                value,
                key_padding_mask,
                reference_points,
                spatial_shapes,
                level_start_index,
            )
            if not self.batch_first:
                output = output.permute(1, 0, 2)
            return output

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        expected_offsets_shape = (
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        if override_sampling_offsets is None:
            sampling_offsets = self.sampling_offsets(query).view(*expected_offsets_shape)
        else:
            if tuple(override_sampling_offsets.shape) != expected_offsets_shape:
                raise ValueError(
                    "override_sampling_offsets must have shape "
                    f"{expected_offsets_shape}, got {tuple(override_sampling_offsets.shape)}"
                )
            sampling_offsets = override_sampling_offsets.to(device=query.device, dtype=query.dtype)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = maybe_observe_and_quantize_attention(self, attention_weights)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        # bs, num_query, num_heads, num_levels, num_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
    
        if value.is_cuda and HAS_MSDA_EXT and not getattr(self, "_ovtr_quant_force_pytorch_msda", False):
            halffloat = False
            if value.dtype == torch.float16:
                halffloat = True
                value = value.float()
                sampling_locations = sampling_locations.float()
                attention_weights = attention_weights.float()

            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )

            if halffloat:
                output = output.half()
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        output = maybe_observe_and_quantize_msda_aggregation(self, output)
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        if return_sampling_offsets:
            return output, sampling_offsets.detach()
        return output


def create_dummy_class(klass, dependency, message=""):
    """
    When a dependency of a class is not available, create a dummy class which throws ImportError
    when used.

    Args:
        klass (str): name of the class.
        dependency (str): name of the dependency.
        message: extra message to print
    Returns:
        class: a class object
    """
    err = "Cannot import '{}', therefore '{}' is not available.".format(dependency, klass)
    if message:
        err = err + " " + message

    class _DummyMetaClass(type):
        # throw error on class attribute access
        def __getattr__(_, __):  # noqa: B902
            raise ImportError(err)

    class _Dummy(object, metaclass=_DummyMetaClass):
        # throw error on constructor
        def __init__(self, *args, **kwargs):
            raise ImportError(err)

    return _Dummy


def create_dummy_func(func, dependency, message=""):
    """
    When a dependency of a function is not available, create a dummy function which throws
    ImportError when used.

    Args:
        func (str): name of the function.
        dependency (str or list[str]): name(s) of the dependency.
        message: extra message to print
    Returns:
        function: a function object
    """
    err = "Cannot import '{}', therefore '{}' is not available.".format(dependency, func)
    if message:
        err = err + " " + message

    if isinstance(dependency, (list, tuple)):
        dependency = ",".join(dependency)

    def _dummy(*args, **kwargs):
        raise ImportError(err)

    return _dummy
