/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <vector>
#include "cuda/ms_deform_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ unsigned char get_uint4_value(const unsigned char* data, const int64_t index)
{
    const unsigned char byte = data[index >> 1];
    return (index & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
}

__device__ __forceinline__ int get_int4_value(const unsigned char* data, const int64_t index)
{
    const int nibble = static_cast<int>(get_uint4_value(data, index));
    return nibble >= 8 ? nibble - 16 : nibble;
}

__global__ void lowbit_linear_kernel(
    const unsigned char* __restrict__ x_packed,
    const unsigned char* __restrict__ weight_packed,
    const int32_t* __restrict__ weight_sum,
    const float* __restrict__ x_scale,
    const float* __restrict__ x_zero_point,
    const float* __restrict__ weight_scale,
    const float* __restrict__ bias,
    const bool has_bias,
    const int64_t rows,
    const int64_t in_features,
    const int64_t out_features,
    float* __restrict__ output)
{
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = rows * out_features;
    if (index >= total) {
        return;
    }
    const int64_t row = index / out_features;
    const int64_t out = index - row * out_features;
    int32_t dot = 0;
    for (int64_t k = 0; k < in_features; ++k) {
        const int x_q = static_cast<int>(get_uint4_value(x_packed, row * in_features + k));
        const int w_q = get_int4_value(weight_packed, out * in_features + k);
        dot += x_q * w_q;
    }
    const int z_x = __float2int_rn(x_zero_point[0]);
    const int32_t acc = dot - z_x * weight_sum[out];
    float value = x_scale[0] * weight_scale[out] * static_cast<float>(acc);
    if (has_bias) {
        value += bias[out];
    }
    output[index] = value;
}

__device__ __forceinline__ float lowbit_value_real(
    const unsigned char* __restrict__ value_packed,
    const int64_t value_index,
    const float value_scale,
    const int value_zero_point)
{
    const int q = static_cast<int>(get_uint4_value(value_packed, value_index));
    return value_scale * static_cast<float>(q - value_zero_point);
}

__global__ void ms_deform_attn_lowbit_kernel(
    const unsigned char* __restrict__ value_packed,
    const int64_t* __restrict__ spatial_shapes,
    const int64_t* __restrict__ level_start_index,
    const float* __restrict__ sampling_loc,
    const unsigned char* __restrict__ attn_uint8,
    const float* __restrict__ value_scale,
    const float* __restrict__ value_zero_point,
    const float* __restrict__ attn_scale,
    const float* __restrict__ attn_zero_point,
    const int batch,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    float* __restrict__ output)
{
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(batch) * num_query * num_heads * channels;
    if (index >= total) {
        return;
    }

    const int c = index % channels;
    const int h = (index / channels) % num_heads;
    const int q = (index / (channels * num_heads)) % num_query;
    const int b = index / (channels * num_heads * num_query);
    const float v_scale = value_scale[0];
    const int v_zp = __float2int_rn(value_zero_point[0]);
    const float a_scale = attn_scale[h];
    const int a_zp = __float2int_rn(attn_zero_point[h]);

    float acc = 0.0f;
    for (int level = 0; level < num_levels; ++level) {
        const int height = static_cast<int>(spatial_shapes[level * 2]);
        const int width = static_cast<int>(spatial_shapes[level * 2 + 1]);
        const int64_t level_start = level_start_index[level];
        for (int point = 0; point < num_point; ++point) {
            const int64_t loc_base =
                (((((static_cast<int64_t>(b) * num_query + q) * num_heads + h) * num_levels + level) * num_point + point) * 2);
            const float loc_w = sampling_loc[loc_base] * width - 0.5f;
            const float loc_h = sampling_loc[loc_base + 1] * height - 0.5f;
            const int w_low = floorf(loc_w);
            const int h_low = floorf(loc_h);
            const int w_high = w_low + 1;
            const int h_high = h_low + 1;
            const float lw = loc_w - w_low;
            const float lh = loc_h - h_low;
            const float hw = 1.0f - lw;
            const float hh = 1.0f - lh;

            float sampled = 0.0f;
            if (h_low >= 0 && w_low >= 0 && h_low < height && w_low < width) {
                const int64_t spatial = level_start + h_low * width + w_low;
                const int64_t value_index = (((static_cast<int64_t>(b) * spatial_size + spatial) * num_heads + h) * channels + c);
                sampled += hh * hw * lowbit_value_real(value_packed, value_index, v_scale, v_zp);
            }
            if (h_low >= 0 && w_high >= 0 && h_low < height && w_high < width) {
                const int64_t spatial = level_start + h_low * width + w_high;
                const int64_t value_index = (((static_cast<int64_t>(b) * spatial_size + spatial) * num_heads + h) * channels + c);
                sampled += hh * lw * lowbit_value_real(value_packed, value_index, v_scale, v_zp);
            }
            if (h_high >= 0 && w_low >= 0 && h_high < height && w_low < width) {
                const int64_t spatial = level_start + h_high * width + w_low;
                const int64_t value_index = (((static_cast<int64_t>(b) * spatial_size + spatial) * num_heads + h) * channels + c);
                sampled += lh * hw * lowbit_value_real(value_packed, value_index, v_scale, v_zp);
            }
            if (h_high >= 0 && w_high >= 0 && h_high < height && w_high < width) {
                const int64_t spatial = level_start + h_high * width + w_high;
                const int64_t value_index = (((static_cast<int64_t>(b) * spatial_size + spatial) * num_heads + h) * channels + c);
                sampled += lh * lw * lowbit_value_real(value_packed, value_index, v_scale, v_zp);
            }

            const int64_t attn_index =
                ((((static_cast<int64_t>(b) * num_query + q) * num_heads + h) * num_levels + level) * num_point + point);
            const float attn_real = a_scale * static_cast<float>(static_cast<int>(attn_uint8[attn_index]) - a_zp);
            acc += sampled * attn_real;
        }
    }

    const int64_t out_index = (static_cast<int64_t>(b) * num_query + q) * (num_heads * channels) + h * channels + c;
    output[out_index] = acc;
}

} // namespace


at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
    TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    TORCH_CHECK(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    TORCH_CHECK(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    TORCH_CHECK(batch % im2col_step_ == 0, "batch(", batch, ") must divide im2col_step(", im2col_step_, ")");
    
    auto output = at::zeros({batch, num_query, num_heads, channels}, value.options());

    const int batch_n = im2col_step_;
    auto output_n = output.view({batch/im2col_step_, batch_n, num_query, num_heads, channels});
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto columns = output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
            ms_deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data_ptr<int64_t>(),
                level_start_index.data_ptr<int64_t>(),
                sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                columns.data_ptr<scalar_t>());

        }));
    }

    output = output.view({batch, num_query, num_heads*channels});

    return output;
}


std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{

    TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
    TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    TORCH_CHECK(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    TORCH_CHECK(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    TORCH_CHECK(batch % im2col_step_ == 0, "batch(", batch, ") must divide im2col_step(", im2col_step_, ")");

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    const int batch_n = im2col_step_;
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    auto grad_output_n = grad_output.view({batch/im2col_step_, batch_n, num_query, num_heads, channels});
    
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto grad_output_g = grad_output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
            ms_deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(),
                                    grad_output_g.data_ptr<scalar_t>(),
                                    value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                                    spatial_shapes.data_ptr<int64_t>(),
                                    level_start_index.data_ptr<int64_t>(),
                                    sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                    attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                                    batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                                    grad_value.data_ptr<scalar_t>() +  n * im2col_step_ * per_value_size,
                                    grad_sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                    grad_attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size);

        }));
    }

    return {
        grad_value, grad_sampling_loc, grad_attn_weight
    };
}

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
    const int64_t out_features)
{
    TORCH_CHECK(x_packed.is_cuda(), "x_packed must be CUDA");
    TORCH_CHECK(weight_packed.is_cuda(), "weight_packed must be CUDA");
    TORCH_CHECK(weight_sum.is_cuda(), "weight_sum must be CUDA");
    TORCH_CHECK(x_scale.is_cuda(), "x_scale must be CUDA");
    TORCH_CHECK(x_zero_point.is_cuda(), "x_zero_point must be CUDA");
    TORCH_CHECK(weight_scale.is_cuda(), "weight_scale must be CUDA");
    TORCH_CHECK(x_packed.scalar_type() == at::kByte, "x_packed must be uint8");
    TORCH_CHECK(weight_packed.scalar_type() == at::kByte, "weight_packed must be uint8");
    TORCH_CHECK(weight_sum.scalar_type() == at::kInt, "weight_sum must be int32");
    TORCH_CHECK(x_scale.scalar_type() == at::kFloat, "x_scale must be float32");
    TORCH_CHECK(x_zero_point.scalar_type() == at::kFloat, "x_zero_point must be float32");
    TORCH_CHECK(weight_scale.scalar_type() == at::kFloat, "weight_scale must be float32");
    TORCH_CHECK(bias_fp32.numel() == 0 || bias_fp32.is_cuda(), "bias_fp32 must be empty or CUDA");
    TORCH_CHECK(bias_fp32.numel() == 0 || bias_fp32.scalar_type() == at::kFloat, "bias_fp32 must be float32");

    auto output = at::empty({rows, out_features}, x_packed.options().dtype(at::kFloat));
    const int threads = 256;
    const int64_t total = rows * out_features;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    lowbit_linear_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_packed.data_ptr<unsigned char>(),
        weight_packed.data_ptr<unsigned char>(),
        weight_sum.data_ptr<int32_t>(),
        x_scale.data_ptr<float>(),
        x_zero_point.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        bias_fp32.numel() == 0 ? nullptr : bias_fp32.data_ptr<float>(),
        bias_fp32.numel() != 0,
        rows,
        in_features,
        out_features,
        output.data_ptr<float>());
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}

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
    const int im2col_step)
{
    TORCH_CHECK(value_packed.is_cuda(), "value_packed must be CUDA");
    TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be CUDA");
    TORCH_CHECK(level_start_index.is_cuda(), "level_start_index must be CUDA");
    TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be CUDA");
    TORCH_CHECK(attn_uint8.is_cuda(), "attn_uint8 must be CUDA");
    TORCH_CHECK(value_scale.is_cuda(), "value_scale must be CUDA");
    TORCH_CHECK(value_zero_point.is_cuda(), "value_zero_point must be CUDA");
    TORCH_CHECK(attn_scale.is_cuda(), "attn_scale must be CUDA");
    TORCH_CHECK(attn_zero_point.is_cuda(), "attn_zero_point must be CUDA");
    TORCH_CHECK(value_packed.scalar_type() == at::kByte, "value_packed must be uint8");
    TORCH_CHECK(attn_uint8.scalar_type() == at::kByte, "attn_uint8 must be uint8");
    TORCH_CHECK(spatial_shapes.scalar_type() == at::kLong, "spatial_shapes must be int64");
    TORCH_CHECK(level_start_index.scalar_type() == at::kLong, "level_start_index must be int64");
    TORCH_CHECK(sampling_loc.scalar_type() == at::kFloat, "sampling_loc must be float32");
    TORCH_CHECK(value_scale.scalar_type() == at::kFloat, "value_scale must be float32");
    TORCH_CHECK(value_zero_point.scalar_type() == at::kFloat, "value_zero_point must be float32");
    TORCH_CHECK(attn_scale.scalar_type() == at::kFloat, "attn_scale must be float32");
    TORCH_CHECK(attn_zero_point.scalar_type() == at::kFloat, "attn_zero_point must be float32");

    const int batch = sampling_loc.size(0);
    const int num_query = sampling_loc.size(1);
    const int num_heads = sampling_loc.size(2);
    const int num_levels = sampling_loc.size(3);
    const int num_point = sampling_loc.size(4);
    const int spatial_size = static_cast<int>(value_packed.numel() * 2 / (batch * num_heads * channels));
    TORCH_CHECK(spatial_size > 0, "Invalid packed value size for ms_deform_attn_lowbit_forward");
    TORCH_CHECK(attn_scale.numel() == num_heads, "attn_scale must have shape [num_heads]");
    TORCH_CHECK(attn_zero_point.numel() == num_heads, "attn_zero_point must have shape [num_heads]");
    (void)im2col_step;

    auto output = at::empty({batch, num_query, num_heads * channels}, value_packed.options().dtype(at::kFloat));
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(batch) * num_query * num_heads * channels;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    ms_deform_attn_lowbit_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        value_packed.data_ptr<unsigned char>(),
        spatial_shapes.data_ptr<int64_t>(),
        level_start_index.data_ptr<int64_t>(),
        sampling_loc.data_ptr<float>(),
        attn_uint8.data_ptr<unsigned char>(),
        value_scale.data_ptr<float>(),
        value_zero_point.data_ptr<float>(),
        attn_scale.data_ptr<float>(),
        attn_zero_point.data_ptr<float>(),
        batch,
        spatial_size,
        num_heads,
        channels,
        num_levels,
        num_query,
        num_point,
        output.data_ptr<float>());
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}
