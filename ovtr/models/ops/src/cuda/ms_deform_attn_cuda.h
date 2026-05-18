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
#include <torch/extension.h>

at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step);

std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step);

at::Tensor lowbit_linear_cuda_forward(
    const at::Tensor &x_packed,
    const at::Tensor &weight_packed,
    const at::Tensor &weight_sum,
    const at::Tensor &x_scale,
    const at::Tensor &x_zero_point,
    const at::Tensor &weight_scale,
    const at::Tensor &bias_fp32,
    const int64_t rows,
    const int64_t in_features,
    const int64_t out_features);

at::Tensor ms_deform_attn_lowbit_cuda_forward(
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
    const int im2col_step);

