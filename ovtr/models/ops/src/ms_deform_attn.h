/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#pragma once

#include "cpu/ms_deform_attn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/ms_deform_attn_cuda.h"
#endif


at::Tensor
ms_deform_attn_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    if (value.is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
#else
        TORCH_CHECK(false, "Not compiled with GPU support");
#endif
    }
    TORCH_CHECK(false, "ms_deform_attn_forward is only implemented for CUDA tensors");
}

std::vector<at::Tensor>
ms_deform_attn_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    if (value.is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_backward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
#else
        TORCH_CHECK(false, "Not compiled with GPU support");
#endif
    }
    TORCH_CHECK(false, "ms_deform_attn_backward is only implemented for CUDA tensors");
}

at::Tensor
lowbit_linear_forward(
    const at::Tensor &x_packed,
    const at::Tensor &weight_packed,
    const at::Tensor &weight_sum,
    const at::Tensor &x_scale,
    const at::Tensor &x_zero_point,
    const at::Tensor &weight_scale,
    const at::Tensor &bias_fp32,
    const int64_t rows,
    const int64_t in_features,
    const int64_t out_features)
{
    if (x_packed.is_cuda())
    {
#ifdef WITH_CUDA
        return lowbit_linear_cuda_forward(
            x_packed, weight_packed, weight_sum, x_scale, x_zero_point,
            weight_scale, bias_fp32, rows, in_features, out_features);
#else
        TORCH_CHECK(false, "Not compiled with GPU support");
#endif
    }
    TORCH_CHECK(false, "lowbit_linear_forward is only implemented for CUDA tensors");
}

at::Tensor
ms_deform_attn_lowbit_forward(
    const at::Tensor &value_packed,
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_uint8,
    const at::Tensor &value_scale,
    const at::Tensor &value_zero_point,
    const at::Tensor &attn_scale,
    const at::Tensor &attn_zero_point,
    const int channels,
    const int im2col_step)
{
    if (value_packed.is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_lowbit_cuda_forward(
            value_packed, spatial_shapes, level_start_index, sampling_loc, attn_uint8,
            value_scale, value_zero_point, attn_scale, attn_zero_point, channels, im2col_step);
#else
        TORCH_CHECK(false, "Not compiled with GPU support");
#endif
    }
    TORCH_CHECK(false, "ms_deform_attn_lowbit_forward is only implemented for CUDA tensors");
}

