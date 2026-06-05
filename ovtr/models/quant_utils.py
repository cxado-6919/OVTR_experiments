import math
import re
from types import MethodType
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _reshape_qparam_for_tensor(param: torch.Tensor, x: torch.Tensor, axis: Optional[int]) -> torch.Tensor:
    if axis is None:
        return param
    shape = [1] * x.ndim
    shape[axis] = param.shape[0]
    return param.reshape(shape)


def _grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def _round_pass(x: torch.Tensor) -> torch.Tensor:
    return (torch.round(x) - x).detach() + x


def _reduce_minmax(x: torch.Tensor, axis: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    if axis is None:
        return x.amin(), x.amax()
    reduce_dims = tuple(dim for dim in range(x.ndim) if dim != axis)
    return x.amin(dim=reduce_dims), x.amax(dim=reduce_dims)


def _evenly_spaced_indices(length: int, count: int, device: torch.device) -> torch.Tensor:
    if count >= length:
        return torch.arange(length, device=device, dtype=torch.long)
    if count <= 1:
        return torch.zeros((max(count, 0),), device=device, dtype=torch.long)
    steps = torch.arange(count, device=device, dtype=torch.long)
    return torch.div(steps * (length - 1), count - 1, rounding_mode="floor")


def _quant_bounds(bit_width: int, symmetric: bool) -> Tuple[int, int]:
    if symmetric:
        qmax = (1 << (bit_width - 1)) - 1
        return -qmax, qmax
    return 0, (1 << bit_width) - 1


def _as_scalar_tensor(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return value.detach().reshape(-1)[0].to(device=device, dtype=dtype)


def quantize_affine_uint(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    bit_width: int,
) -> torch.Tensor:
    qmin, qmax = _quant_bounds(bit_width, symmetric=False)
    scale = _as_scalar_tensor(scale, device=x.device, dtype=x.dtype).clamp(min=1e-8)
    zero_point = torch.round(_as_scalar_tensor(zero_point, device=x.device, dtype=x.dtype)).clamp(qmin, qmax)
    q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return q.to(torch.uint8)


def dequantize_affine_uint(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    scale = _as_scalar_tensor(scale, device=q.device, dtype=dtype)
    zero_point = torch.round(_as_scalar_tensor(zero_point, device=q.device, dtype=dtype))
    return (q.to(dtype) - zero_point) * scale


def _pack_4bit_values(q: torch.Tensor) -> torch.Tensor:
    q = q.reshape(-1).to(torch.uint8) & 0x0F
    if q.numel() % 2:
        q = torch.cat([q, q.new_zeros(1)], dim=0)
    low = q[0::2]
    high = q[1::2] << 4
    return (low | high).contiguous()


def _unpack_4bit_values(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    packed = packed.reshape(-1).to(torch.uint8)
    out = torch.empty(packed.numel() * 2, device=packed.device, dtype=torch.uint8)
    out[0::2] = packed & 0x0F
    out[1::2] = (packed >> 4) & 0x0F
    return out[:num_values].contiguous()


def pack_uint4(q: torch.Tensor) -> torch.Tensor:
    """Pack unsigned 4-bit values; element 0 is stored in the low nibble."""
    if q.numel() and bool(((q < 0) | (q > 15)).any().item()):
        raise ValueError("UINT4 pack expects values in [0, 15].")
    return _pack_4bit_values(q)


def unpack_uint4(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    return _unpack_4bit_values(packed, num_values)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit two's-complement values; element 0 is stored in the low nibble."""
    if q.numel() and bool(((q < -8) | (q > 7)).any().item()):
        raise ValueError("INT4 pack expects values in [-8, 7].")
    return _pack_4bit_values(q.to(torch.int16) & 0x0F)


def unpack_int4(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    q = _unpack_4bit_values(packed, num_values).to(torch.int8)
    return torch.where(q >= 8, q - 16, q)


def quantize_weight_int4_per_channel(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    qmin, qmax = _quant_bounds(4, symmetric=True)
    scale = scale.detach().abs().clamp(min=1e-8).to(device=weight.device, dtype=weight.dtype)
    view_shape = [1] * weight.ndim
    view_shape[0] = scale.numel()
    q = torch.round(weight.detach() / scale.reshape(view_shape)).clamp(qmin, qmax)
    return q.to(torch.int8)


def fake_quant_from_packed_uint4(
    packed: torch.Tensor,
    num_values: int,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    q = unpack_uint4(packed, num_values)
    out = dequantize_affine_uint(q, scale, zero_point)
    return out.reshape(shape) if shape is not None else out


def _safe_minmax(min_val: torch.Tensor, max_val: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    min_val = torch.minimum(min_val, max_val - eps)
    max_val = torch.maximum(max_val, min_val + eps)
    return min_val, max_val


def _affine_qparams(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    *,
    bit_width: int,
    symmetric: bool,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    qmin, qmax = _quant_bounds(bit_width, symmetric)
    if symmetric:
        max_abs = torch.maximum(min_val.abs(), max_val.abs())
        scale = torch.clamp(max_abs / float(qmax), min=eps)
        zero_point = torch.zeros_like(scale)
        return scale, zero_point, qmin, qmax

    min_val = torch.minimum(min_val, torch.zeros_like(min_val))
    max_val = torch.maximum(max_val, torch.zeros_like(max_val))
    min_val, max_val = _safe_minmax(min_val, max_val, eps)
    scale = torch.clamp((max_val - min_val) / float(qmax - qmin), min=eps)
    zero_point = torch.round(qmin - (min_val / scale)).clamp(qmin, qmax)
    return scale, zero_point, qmin, qmax


def _fake_quant_with_range(
    x: torch.Tensor,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    *,
    bit_width: int,
    symmetric: bool,
    axis: Optional[int],
    eps: float,
    use_ste: bool = False,
) -> torch.Tensor:
    scale, zero_point, qmin, qmax = _affine_qparams(
        min_val.to(device=x.device, dtype=x.dtype),
        max_val.to(device=x.device, dtype=x.dtype),
        bit_width=bit_width,
        symmetric=symmetric,
        eps=eps,
    )
    scale = _reshape_qparam_for_tensor(scale, x, axis)
    zero_point = _reshape_qparam_for_tensor(zero_point, x, axis)
    round_fn = _round_pass if use_ste else torch.round
    quantized = torch.clamp(round_fn(x / scale + zero_point), qmin, qmax)
    return (quantized - zero_point) * scale


class MinMaxObserver(nn.Module):
    def __init__(
        self,
        bit_width: int,
        symmetric: bool,
        axis: Optional[int] = None,
        eps: float = 1e-8,
        sample_limit: int = 65536,
    ):
        super().__init__()
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.axis = axis
        self.eps = eps
        self.sample_limit = sample_limit
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self._sample_values = None

    def reset(self) -> None:
        self.min_val = self.min_val.new_empty(0)
        self.max_val = self.max_val.new_empty(0)
        self.initialized.fill_(False)
        self._sample_values = None

    def observe(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return

        with torch.no_grad():
            cur_min, cur_max = _reduce_minmax(x.detach(), self.axis)
            if not self.initialized.item():
                self.min_val = cur_min
                self.max_val = cur_max
                self.initialized.fill_(True)
            else:
                self.min_val = torch.minimum(self.min_val, cur_min)
                self.max_val = torch.maximum(self.max_val, cur_max)
            self._append_samples(x.detach())

    def _append_samples(self, x: torch.Tensor) -> None:
        if self.sample_limit <= 0:
            return
        x = x.float()
        if self.axis is None:
            flat = x.reshape(-1)
            if flat.numel() == 0:
                return
            take = min(flat.numel(), min(self.sample_limit, 4096))
            if take < flat.numel():
                index = _evenly_spaced_indices(flat.numel(), take, flat.device)
                flat = flat.index_select(0, index)
            sample = flat.detach().cpu()
            if self._sample_values is None:
                self._sample_values = sample
            else:
                self._sample_values = torch.cat([self._sample_values, sample], dim=0)
                if self._sample_values.numel() > self.sample_limit:
                    index = _evenly_spaced_indices(
                        self._sample_values.numel(),
                        self.sample_limit,
                        self._sample_values.device,
                    )
                    self._sample_values = self._sample_values.index_select(0, index)
            return

        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        if axis < 0 or axis >= x.ndim:
            return
        moved = x.movedim(axis, 0).reshape(x.shape[axis], -1)
        if moved.numel() == 0:
            return
        per_channel_limit = max(1, self.sample_limit // max(moved.shape[0], 1))
        take = min(moved.shape[1], min(per_channel_limit, 2048))
        if take < moved.shape[1]:
            index = _evenly_spaced_indices(moved.shape[1], take, moved.device)
            moved = moved.index_select(1, index)
        sample = moved.detach().cpu()
        if self._sample_values is None:
            self._sample_values = sample
        else:
            self._sample_values = torch.cat([self._sample_values, sample], dim=1)
            if self._sample_values.shape[1] > per_channel_limit:
                index = _evenly_spaced_indices(
                    self._sample_values.shape[1],
                    per_channel_limit,
                    self._sample_values.device,
                )
                self._sample_values = self._sample_values.index_select(1, index)

    def _mse_range_from_samples(self, num_candidates: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._sample_values is None or self._sample_values.numel() == 0 or num_candidates <= 1:
            return self.min_val, self.max_val

        samples = self._sample_values.to(device=self.min_val.device, dtype=self.min_val.dtype)
        if self.axis is None:
            sample_min = samples.amin()
            sample_max = samples.amax()
        else:
            sample_min = samples.amin(dim=1)
            sample_max = samples.amax(dim=1)

        base_min = torch.minimum(self.min_val, sample_min)
        base_max = torch.maximum(self.max_val, sample_max)
        if self.symmetric:
            base_abs = torch.maximum(base_min.abs(), base_max.abs()).clamp(min=self.eps)
            best_abs = base_abs.clone()
            best_error = torch.full_like(base_abs, float("inf"))
            for factor in torch.linspace(1.0, 0.2, num_candidates, device=base_abs.device, dtype=base_abs.dtype):
                cur_abs = base_abs * factor
                cur_min = -cur_abs
                cur_max = cur_abs
                fake = _fake_quant_with_range(
                    samples,
                    cur_min,
                    cur_max,
                    bit_width=self.bit_width,
                    symmetric=True,
                    axis=0 if samples.ndim == 2 else None,
                    eps=self.eps,
                )
                error = (fake - samples).pow(2)
                error = error.mean(dim=1) if samples.ndim == 2 else error.mean()
                best_abs = torch.where(error < best_error, cur_abs, best_abs)
                best_error = torch.minimum(best_error, error)
            return -best_abs, best_abs

        base_min = torch.minimum(base_min, torch.zeros_like(base_min))
        base_max = torch.maximum(base_max, torch.zeros_like(base_max))
        base_min, base_max = _safe_minmax(base_min, base_max, self.eps)
        best_min = base_min.clone()
        best_max = base_max.clone()
        best_error = torch.full_like(base_min, float("inf"))
        for factor in torch.linspace(1.0, 0.2, num_candidates, device=base_min.device, dtype=base_min.dtype):
            cur_min = base_min * factor
            cur_max = base_max * factor
            cur_min, cur_max = _safe_minmax(cur_min, cur_max, self.eps)
            fake = _fake_quant_with_range(
                samples,
                cur_min,
                cur_max,
                bit_width=self.bit_width,
                symmetric=False,
                axis=0 if samples.ndim == 2 else None,
                eps=self.eps,
            )
            error = (fake - samples).pow(2)
            error = error.mean(dim=1) if samples.ndim == 2 else error.mean()
            best_min = torch.where(error < best_error, cur_min, best_min)
            best_max = torch.where(error < best_error, cur_max, best_max)
            best_error = torch.minimum(best_error, error)
        return best_min, best_max

    def finalize_range(self, method: str = "minmax", num_candidates: int = 80) -> None:
        if not self.initialized.item():
            return
        if method == "mse":
            min_val, max_val = self._mse_range_from_samples(num_candidates)
            self.min_val = min_val.detach().clone()
            self.max_val = max_val.detach().clone()

    def _get_qparams(self, x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, int]]:
        if not self.initialized.item():
            return None

        scale, zero_point, qmin, qmax = _affine_qparams(
            self.min_val,
            self.max_val,
            bit_width=self.bit_width,
            symmetric=self.symmetric,
            eps=self.eps,
        )

        scale = _reshape_qparam_for_tensor(scale, x, self.axis)
        zero_point = _reshape_qparam_for_tensor(zero_point, x, self.axis)
        return scale, zero_point, qmin, qmax

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        qparams = self._get_qparams(x)
        if qparams is None:
            return x

        scale, zero_point, qmin, qmax = qparams
        q = torch.round(x / scale + zero_point)
        q = torch.clamp(q, qmin, qmax)
        return (q - zero_point) * scale


class MSEHistogramObserver(MinMaxObserver):
    pass


class StaticAffineQuantizer(nn.Module):
    def __init__(self, bit_width: int, symmetric: bool, axis: Optional[int] = None, eps: float = 1e-8):
        super().__init__()
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.axis = axis
        self.eps = eps
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))
        self.register_buffer("scale", torch.tensor([]))
        self.register_buffer("zero_point", torch.tensor([]))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def reset(self) -> None:
        self.min_val = self.min_val.new_empty(0)
        self.max_val = self.max_val.new_empty(0)
        self.scale = self.scale.new_empty(0)
        self.zero_point = self.zero_point.new_empty(0)
        self.initialized.fill_(False)

    def initialize_from_range(self, min_val: torch.Tensor, max_val: torch.Tensor) -> None:
        with torch.no_grad():
            scale, zero_point, _, _ = _affine_qparams(
                min_val.detach(),
                max_val.detach(),
                bit_width=self.bit_width,
                symmetric=self.symmetric,
                eps=self.eps,
            )
            self.min_val = min_val.detach().clone()
            self.max_val = max_val.detach().clone()
            self.scale = scale.detach().clone()
            self.zero_point = zero_point.detach().clone()
            self.initialized.fill_(True)

    def initialize_from_tensor(
        self,
        x: torch.Tensor,
        *,
        range_method: str = "minmax",
        mse_candidates: int = 80,
        sample_limit: int = 65536,
    ) -> None:
        if self.initialized.item():
            return
        observer = MSEHistogramObserver(
            bit_width=self.bit_width,
            symmetric=self.symmetric,
            axis=self.axis,
            eps=self.eps,
            sample_limit=sample_limit,
        ).to(x.device)
        observer.observe(x.detach())
        observer.finalize_range(method=range_method, num_candidates=mse_candidates)
        self.initialize_from_range(observer.min_val, observer.max_val)

    def quantize(self, x: torch.Tensor, *, use_ste: bool = False) -> torch.Tensor:
        if not self.initialized.item():
            self.initialize_from_tensor(x)
        scale = _reshape_qparam_for_tensor(self.scale.to(device=x.device, dtype=x.dtype), x, self.axis)
        zero_point = _reshape_qparam_for_tensor(self.zero_point.to(device=x.device, dtype=x.dtype), x, self.axis)
        qmin, qmax = _quant_bounds(self.bit_width, self.symmetric)
        round_fn = _round_pass if use_ste else torch.round
        quantized = torch.clamp(round_fn(x / scale + zero_point), qmin, qmax)
        return (quantized - zero_point) * scale


class LSQQuantizer(nn.Module):
    def __init__(
        self,
        bit_width: int,
        symmetric: bool,
        axis: Optional[int],
        param_size: int,
        eps: float = 1e-8,
        sample_limit: int = 65536,
    ):
        super().__init__()
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.axis = axis
        self.param_size = param_size
        self.eps = eps
        self.sample_limit = sample_limit
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.scale = nn.Parameter(torch.ones(param_size))
        if symmetric:
            self.zero_point = None
        else:
            self.zero_point = nn.Parameter(torch.zeros(param_size))
        self._sample_values = None

    @property
    def qmin(self) -> int:
        if self.symmetric:
            return -((1 << (self.bit_width - 1)) - 1)
        return 0

    @property
    def qmax(self) -> int:
        if self.symmetric:
            return (1 << (self.bit_width - 1)) - 1
        return (1 << self.bit_width) - 1

    def reset_observer(self) -> None:
        self.min_val = self.min_val.new_empty(0)
        self.max_val = self.max_val.new_empty(0)
        self._sample_values = None

    def observe(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return

        with torch.no_grad():
            cur_min, cur_max = _reduce_minmax(x.detach(), self.axis)
            if self.min_val.numel() == 0:
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                self.min_val = torch.minimum(self.min_val, cur_min)
                self.max_val = torch.maximum(self.max_val, cur_max)
            self._append_samples(x.detach())

    def _append_samples(self, x: torch.Tensor) -> None:
        if self.sample_limit <= 0:
            return
        x = x.float()
        if self.axis is None:
            flat = x.reshape(-1)
            if flat.numel() == 0:
                return
            take = min(flat.numel(), min(self.sample_limit, 4096))
            if take < flat.numel():
                index = _evenly_spaced_indices(flat.numel(), take, flat.device)
                flat = flat.index_select(0, index)
            sample = flat.detach().cpu()
            if self._sample_values is None:
                self._sample_values = sample
            else:
                self._sample_values = torch.cat([self._sample_values, sample], dim=0)
                if self._sample_values.numel() > self.sample_limit:
                    index = _evenly_spaced_indices(
                        self._sample_values.numel(),
                        self.sample_limit,
                        self._sample_values.device,
                    )
                    self._sample_values = self._sample_values.index_select(0, index)
            return

        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        if axis < 0 or axis >= x.ndim:
            return
        moved = x.movedim(axis, 0).reshape(x.shape[axis], -1)
        if moved.numel() == 0:
            return
        per_channel_limit = max(1, self.sample_limit // max(moved.shape[0], 1))
        take = min(moved.shape[1], min(per_channel_limit, 2048))
        if take < moved.shape[1]:
            index = _evenly_spaced_indices(moved.shape[1], take, moved.device)
            moved = moved.index_select(1, index)
        sample = moved.detach().cpu()
        if self._sample_values is None:
            self._sample_values = sample
        else:
            self._sample_values = torch.cat([self._sample_values, sample], dim=1)
            if self._sample_values.shape[1] > per_channel_limit:
                index = _evenly_spaced_indices(
                    self._sample_values.shape[1],
                    per_channel_limit,
                    self._sample_values.device,
                )
                self._sample_values = self._sample_values.index_select(1, index)

    def _mse_range_from_samples(self, num_candidates: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._sample_values is None or self._sample_values.numel() == 0 or num_candidates <= 1:
            return self.min_val, self.max_val

        samples = self._sample_values.to(device=self.min_val.device, dtype=self.min_val.dtype)
        if self.axis is None:
            sample_min = samples.amin()
            sample_max = samples.amax()
        else:
            sample_min = samples.amin(dim=1)
            sample_max = samples.amax(dim=1)

        base_min = torch.minimum(self.min_val, sample_min)
        base_max = torch.maximum(self.max_val, sample_max)
        if self.symmetric:
            base_abs = torch.maximum(base_min.abs(), base_max.abs()).clamp(min=self.eps)
            best_abs = base_abs.clone()
            best_error = torch.full_like(base_abs, float("inf"))
            for factor in torch.linspace(1.0, 0.2, num_candidates, device=base_abs.device, dtype=base_abs.dtype):
                cur_abs = base_abs * factor
                cur_min = -cur_abs
                cur_max = cur_abs
                fake = _fake_quant_with_range(
                    samples,
                    cur_min,
                    cur_max,
                    bit_width=self.bit_width,
                    symmetric=True,
                    axis=0 if samples.ndim == 2 else None,
                    eps=self.eps,
                )
                error = (fake - samples).pow(2)
                error = error.mean(dim=1) if samples.ndim == 2 else error.mean()
                best_abs = torch.where(error < best_error, cur_abs, best_abs)
                best_error = torch.minimum(best_error, error)
            return -best_abs, best_abs

        base_min = torch.minimum(base_min, torch.zeros_like(base_min))
        base_max = torch.maximum(base_max, torch.zeros_like(base_max))
        base_min, base_max = _safe_minmax(base_min, base_max, self.eps)
        best_min = base_min.clone()
        best_max = base_max.clone()
        best_error = torch.full_like(base_min, float("inf"))
        for factor in torch.linspace(1.0, 0.2, num_candidates, device=base_min.device, dtype=base_min.dtype):
            cur_min = base_min * factor
            cur_max = base_max * factor
            cur_min, cur_max = _safe_minmax(cur_min, cur_max, self.eps)
            fake = _fake_quant_with_range(
                samples,
                cur_min,
                cur_max,
                bit_width=self.bit_width,
                symmetric=False,
                axis=0 if samples.ndim == 2 else None,
                eps=self.eps,
            )
            error = (fake - samples).pow(2)
            error = error.mean(dim=1) if samples.ndim == 2 else error.mean()
            best_min = torch.where(error < best_error, cur_min, best_min)
            best_max = torch.where(error < best_error, cur_max, best_max)
            best_error = torch.minimum(best_error, error)
        return best_min, best_max

    def _initialize(self, min_val: torch.Tensor, max_val: torch.Tensor) -> None:
        with torch.no_grad():
            if self.symmetric:
                max_abs = torch.maximum(min_val.abs(), max_val.abs())
                scale = torch.clamp(max_abs / float(self.qmax), min=self.eps)
                self.scale.copy_(scale.reshape(-1))
            else:
                scale = torch.clamp((max_val - min_val) / float(self.qmax - self.qmin), min=self.eps)
                zero_point = torch.round(self.qmin - (min_val / scale)).clamp(self.qmin, self.qmax)
                self.scale.copy_(scale.reshape(-1))
                self.zero_point.copy_(zero_point.reshape(-1))
            self.initialized.fill_(True)

    def initialize_from_observer(self, range_method: str = "minmax", mse_candidates: int = 80) -> None:
        if self.initialized.item() or self.min_val.numel() == 0:
            return
        min_val, max_val = self.min_val, self.max_val
        if range_method == "mse":
            min_val, max_val = self._mse_range_from_samples(mse_candidates)
        self._initialize(min_val, max_val)

    def maybe_initialize_from_tensor(
        self,
        x: torch.Tensor,
        *,
        range_method: str = "minmax",
        mse_candidates: int = 80,
        sample_limit: int = 65536,
    ) -> None:
        if self.initialized.item():
            return
        if range_method == "mse":
            observer = MSEHistogramObserver(
                bit_width=self.bit_width,
                symmetric=self.symmetric,
                axis=self.axis,
                eps=self.eps,
                sample_limit=sample_limit,
            ).to(x.device)
            observer.observe(x.detach())
            observer.finalize_range(method="mse", num_candidates=mse_candidates)
            min_val, max_val = observer.min_val, observer.max_val
        else:
            min_val, max_val = _reduce_minmax(x.detach(), self.axis)
        self._initialize(min_val, max_val)

    def _gradient_factor(self, x: torch.Tensor) -> float:
        q_range = float(max(abs(self.qmin), abs(self.qmax)))
        if self.axis is None:
            elems_per_param = float(x.numel())
        else:
            elems_per_param = float(x.numel()) / float(x.shape[self.axis])
        return 1.0 / math.sqrt(max(elems_per_param * q_range, 1.0))

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        self.maybe_initialize_from_tensor(x)

        grad_factor = self._gradient_factor(x)
        scale = _grad_scale(self.scale.abs().clamp(min=self.eps), grad_factor)
        scale = _reshape_qparam_for_tensor(scale, x, self.axis)

        if self.symmetric:
            scaled = x / scale
            quantized = torch.clamp(_round_pass(scaled), self.qmin, self.qmax)
            return quantized * scale

        zero_point = _grad_scale(self.zero_point, grad_factor)
        zero_point = torch.clamp(_round_pass(zero_point), self.qmin, self.qmax)
        zero_point = _reshape_qparam_for_tensor(zero_point, x, self.axis)
        quantized = torch.clamp(_round_pass(x / scale + zero_point), self.qmin, self.qmax)
        return (quantized - zero_point) * scale


def fake_quantize_symmetric_weight(
    weight: torch.Tensor,
    bit_width: int,
    axis: Optional[int] = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    qmax = (1 << (bit_width - 1)) - 1
    qmin = -qmax

    if axis is None:
        max_abs = weight.detach().abs().amax()
    else:
        reduce_dims = tuple(dim for dim in range(weight.ndim) if dim != axis)
        max_abs = weight.detach().abs().amax(dim=reduce_dims)

    scale = torch.clamp(max_abs / float(qmax), min=eps)
    scale = _reshape_qparam_for_tensor(scale, weight, axis)
    q = torch.round(weight / scale)
    q = torch.clamp(q, qmin, qmax)
    return q * scale


def _maybe_observe_and_quantize_activation_ptq(
    x: torch.Tensor,
    observer: MinMaxObserver,
    calibrating: bool,
    quant_enabled: bool,
) -> torch.Tensor:
    if calibrating:
        observer.observe(x)
    if quant_enabled:
        return observer.fake_quant(x)
    return x


def _maybe_observe_and_quantize_activation_qat(
    x: torch.Tensor,
    quantizer: LSQQuantizer,
    observer_enabled: bool,
    quant_enabled: bool,
) -> torch.Tensor:
    if observer_enabled:
        quantizer.observe(x)
    if quant_enabled:
        return quantizer.quantize(x)
    return x


def _record_quant_boundary(module: nn.Module, tensor_role: str, before: torch.Tensor, after: torch.Tensor) -> None:
    recorder = getattr(module, "_ovtr_quant_boundary_recorder", None)
    if recorder is None:
        return
    recorder.record(getattr(module, "_ovtr_quant_name", ""), tensor_role, before, after)


def _ovtr_quant_conv2d_forward(self, x: torch.Tensor) -> torch.Tensor:
    if self._ovtr_quant_backend == "ptq":
        input_before = x
        x = _maybe_observe_and_quantize_activation_ptq(
            x,
            self._ovtr_quant_input_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "input", input_before, x)
        weight_before = self.weight
        weight = self._ovtr_quant_weight_quantizer.quantize(weight_before)
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "weight", weight_before, weight)
        out = F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        output_before = out
        out = _maybe_observe_and_quantize_activation_ptq(
            out,
            self._ovtr_quant_output_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "output", output_before, out)
        return out

    input_before = x
    x = _maybe_observe_and_quantize_activation_qat(
        x,
        self._ovtr_quant_input_quantizer,
        self._ovtr_quant_observer_enabled,
        self._ovtr_quant_quant_enabled,
    )
    if self._ovtr_quant_quant_enabled:
        _record_quant_boundary(self, "input", input_before, x)
    weight = self.weight
    if self._ovtr_quant_quant_enabled:
        weight_before = weight
        weight = self._ovtr_quant_weight_quantizer.quantize(weight)
        _record_quant_boundary(self, "weight", weight_before, weight)
    out = F.conv2d(
        x,
        weight,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )
    output_before = out
    out = _maybe_observe_and_quantize_activation_qat(
        out,
        self._ovtr_quant_output_quantizer,
        self._ovtr_quant_observer_enabled,
        self._ovtr_quant_quant_enabled,
    )
    if self._ovtr_quant_quant_enabled:
        _record_quant_boundary(self, "output", output_before, out)
    return out


def _ovtr_quant_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    if self._ovtr_quant_backend == "ptq":
        input_before = x
        if not getattr(self, "_ovtr_quant_skip_input_quantizer", False):
            x = _maybe_observe_and_quantize_activation_ptq(
                x,
                self._ovtr_quant_input_observer,
                self._ovtr_quant_calibration_enabled,
                self._ovtr_quant_quant_enabled,
            )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "input", input_before, x)
        weight_before = self.weight
        weight = self._ovtr_quant_weight_quantizer.quantize(weight_before)
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "weight", weight_before, weight)
        out = F.linear(x, weight, self.bias)
        output_before = out
        out = _maybe_observe_and_quantize_activation_ptq(
            out,
            self._ovtr_quant_output_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "output", output_before, out)
        return out

    input_before = x
    if not getattr(self, "_ovtr_quant_skip_input_quantizer", False):
        x = _maybe_observe_and_quantize_activation_qat(
            x,
            self._ovtr_quant_input_quantizer,
            self._ovtr_quant_observer_enabled,
            self._ovtr_quant_quant_enabled,
        )
    if self._ovtr_quant_quant_enabled:
        _record_quant_boundary(self, "input", input_before, x)
    weight = self.weight
    if self._ovtr_quant_quant_enabled:
        weight_before = weight
        weight = self._ovtr_quant_weight_quantizer.quantize(weight)
        _record_quant_boundary(self, "weight", weight_before, weight)
    out = F.linear(x, weight, self.bias)
    output_before = out
    out = _maybe_observe_and_quantize_activation_qat(
        out,
        self._ovtr_quant_output_quantizer,
        self._ovtr_quant_observer_enabled,
        self._ovtr_quant_quant_enabled,
    )
    if self._ovtr_quant_quant_enabled:
        _record_quant_boundary(self, "output", output_before, out)
    return out


def _ovtr_quant_embedding_weight(self) -> torch.Tensor:
    if self._ovtr_quant_backend == "ptq":
        weight_before = self.weight
        weight = self._ovtr_quant_weight_quantizer.quantize(weight_before)
        if self._ovtr_quant_calibration_enabled:
            self._ovtr_quant_output_observer.observe(weight)
        if self._ovtr_quant_quant_enabled:
            weight = self._ovtr_quant_output_observer.fake_quant(weight)
            _record_quant_boundary(self, "embedding_weight", weight_before, weight)
        return weight

    weight_before = self.weight
    weight = weight_before
    if self._ovtr_quant_quant_enabled:
        weight = self._ovtr_quant_weight_quantizer.quantize(weight)
    if self._ovtr_quant_observer_enabled:
        self._ovtr_quant_output_quantizer.observe(weight)
    if self._ovtr_quant_quant_enabled:
        weight = self._ovtr_quant_output_quantizer.quantize(weight)
        _record_quant_boundary(self, "embedding_weight", weight_before, weight)
    return weight


def _ovtr_quant_embedding_forward(self, x: torch.Tensor) -> torch.Tensor:
    if self._ovtr_quant_backend == "ptq":
        weight_before = self.weight
        weight = self._ovtr_quant_weight_quantizer.quantize(weight_before)
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "embedding_weight", weight_before, weight)
        out = F.embedding(
            x,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        output_before = out
        out = _maybe_observe_and_quantize_activation_ptq(
            out,
            self._ovtr_quant_output_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "output", output_before, out)
        return out

    weight_before = self.weight
    weight = weight_before
    if self._ovtr_quant_quant_enabled:
        weight = self._ovtr_quant_weight_quantizer.quantize(weight)
        _record_quant_boundary(self, "embedding_weight", weight_before, weight)
    out = F.embedding(
        x,
        weight,
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
    )
    output_before = out
    out = _maybe_observe_and_quantize_activation_qat(
        out,
        self._ovtr_quant_output_quantizer,
        self._ovtr_quant_observer_enabled,
        self._ovtr_quant_quant_enabled,
    )
    if self._ovtr_quant_quant_enabled:
        _record_quant_boundary(self, "output", output_before, out)
    return out


def _prepare_mha_additive_mask(
    attn_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    num_heads: int,
    tgt_len: int,
    src_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if attn_mask is None:
        return None

    mask = attn_mask.to(device=device)
    if mask.dtype == torch.bool:
        additive_mask = torch.zeros(mask.shape, device=device, dtype=dtype)
        additive_mask = additive_mask.masked_fill(mask, torch.finfo(dtype).min)
        mask = additive_mask
    else:
        mask = mask.to(dtype=dtype)

    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:
        if mask.shape[0] == batch_size * num_heads:
            return mask.view(batch_size, num_heads, tgt_len, src_len)
        if mask.shape[0] == batch_size:
            return mask.unsqueeze(1)
        raise ValueError(
            f"Unsupported 3D attention mask shape for quantized MultiheadAttention: {tuple(mask.shape)}"
        )
    if mask.dim() == 4:
        return mask

    raise ValueError(f"Unsupported attention mask rank for quantized MultiheadAttention: {mask.dim()}")


def _apply_mha_key_padding_mask(
    attention_scores: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if key_padding_mask is None:
        return attention_scores

    mask = key_padding_mask.to(device=attention_scores.device)
    if mask.dtype == torch.bool:
        return attention_scores.masked_fill(mask[:, None, None, :], torch.finfo(attention_scores.dtype).min)
    return attention_scores + mask.to(dtype=attention_scores.dtype)[:, None, None, :]


def _maybe_quantize_mha_weight_ptq(self, weight: torch.Tensor) -> torch.Tensor:
    return self._ovtr_quant_weight_quantizer.quantize(weight)


def _maybe_quantize_mha_weight_qat(self, weight: torch.Tensor) -> torch.Tensor:
    if self._ovtr_quant_quant_enabled:
        return self._ovtr_quant_weight_quantizer.quantize(weight)
    return weight


def _ovtr_quant_multihead_attention_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
):
    if not self._qkv_same_embed_dim:
        raise NotImplementedError("Quantized MultiheadAttention currently expects qkv_same_embed_dim=True")
    if self.bias_k is not None or self.bias_v is not None or self.add_zero_attn:
        raise NotImplementedError("Quantized MultiheadAttention does not support bias_k/bias_v/add_zero_attn")
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        raise NotImplementedError("Quantized MultiheadAttention currently expects batched 3D inputs")

    batch_first = self.batch_first
    if batch_first:
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

    tgt_len, batch_size, embed_dim = query.shape
    src_len = key.shape[0]
    if embed_dim != self.embed_dim:
        raise ValueError(f"Expected query embedding dim {self.embed_dim}, got {embed_dim}")

    if is_causal and attn_mask is None:
        attn_mask = torch.ones(tgt_len, src_len, device=query.device, dtype=torch.bool).triu(diagonal=1)

    if self._ovtr_quant_backend == "ptq":
        query_before = query
        query = _maybe_observe_and_quantize_activation_ptq(
            query,
            self._ovtr_quant_query_input_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "query_input", query_before, query)
        key_before = key
        key = _maybe_observe_and_quantize_activation_ptq(
            key,
            self._ovtr_quant_key_input_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "key_input", key_before, key)
        value_before = value
        value = _maybe_observe_and_quantize_activation_ptq(
            value,
            self._ovtr_quant_value_input_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "value_input", value_before, value)
        in_proj_weight_before = self.in_proj_weight
        in_proj_weight = _maybe_quantize_mha_weight_ptq(self, in_proj_weight_before)
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "weight", in_proj_weight_before, in_proj_weight)
    else:
        query_before = query
        query = _maybe_observe_and_quantize_activation_qat(
            query,
            self._ovtr_quant_query_input_quantizer,
            self._ovtr_quant_observer_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "query_input", query_before, query)
        key_before = key
        key = _maybe_observe_and_quantize_activation_qat(
            key,
            self._ovtr_quant_key_input_quantizer,
            self._ovtr_quant_observer_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "key_input", key_before, key)
        value_before = value
        value = _maybe_observe_and_quantize_activation_qat(
            value,
            self._ovtr_quant_value_input_quantizer,
            self._ovtr_quant_observer_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "value_input", value_before, value)
        in_proj_weight_before = self.in_proj_weight
        in_proj_weight = _maybe_quantize_mha_weight_qat(self, in_proj_weight_before)
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "weight", in_proj_weight_before, in_proj_weight)

    bias_q, bias_k, bias_v = (None, None, None)
    if self.in_proj_bias is not None:
        bias_q, bias_k, bias_v = self.in_proj_bias.chunk(3)

    w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)
    q = F.linear(query, w_q, bias_q)
    k = F.linear(key, w_k, bias_k)
    v = F.linear(value, w_v, bias_v)

    if self._ovtr_quant_backend == "ptq":
        q_before = q
        q = _maybe_observe_and_quantize_activation_ptq(
            q,
            self._ovtr_quant_query_proj_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "query_proj", q_before, q)
        k_before = k
        k = _maybe_observe_and_quantize_activation_ptq(
            k,
            self._ovtr_quant_key_proj_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "key_proj", k_before, k)
        v_before = v
        v = _maybe_observe_and_quantize_activation_ptq(
            v,
            self._ovtr_quant_value_proj_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "value_proj", v_before, v)
    else:
        q_before = q
        q = _maybe_observe_and_quantize_activation_qat(
            q,
            self._ovtr_quant_query_proj_quantizer,
            self._ovtr_quant_observer_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "query_proj", q_before, q)
        k_before = k
        k = _maybe_observe_and_quantize_activation_qat(
            k,
            self._ovtr_quant_key_proj_quantizer,
            self._ovtr_quant_observer_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "key_proj", k_before, k)
        v_before = v
        v = _maybe_observe_and_quantize_activation_qat(
            v,
            self._ovtr_quant_value_proj_quantizer,
            self._ovtr_quant_observer_enabled,
            self._ovtr_quant_quant_enabled,
        )
        if self._ovtr_quant_quant_enabled:
            _record_quant_boundary(self, "value_proj", v_before, v)

    q = q.contiguous().view(tgt_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
    k = k.contiguous().view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
    v = v.contiguous().view(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

    q = q * (self.head_dim ** -0.5)
    attention_scores = torch.matmul(q, k.transpose(-2, -1))

    additive_attn_mask = _prepare_mha_additive_mask(
        attn_mask,
        batch_size=batch_size,
        num_heads=self.num_heads,
        tgt_len=tgt_len,
        src_len=src_len,
        dtype=attention_scores.dtype,
        device=attention_scores.device,
    )
    if additive_attn_mask is not None:
        attention_scores = attention_scores + additive_attn_mask
    attention_scores = _apply_mha_key_padding_mask(attention_scores, key_padding_mask)

    attention_weights = torch.softmax(attention_scores, dim=-1)
    attention_weights = maybe_observe_and_quantize_attention(self, attention_weights)

    if self.dropout > 0:
        attention_probs = F.dropout(attention_weights, p=self.dropout, training=self.training)
    else:
        attention_probs = attention_weights

    attn_output = torch.matmul(attention_probs, v)
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, batch_size, embed_dim)
    attn_output = self.out_proj(attn_output)

    if batch_first:
        attn_output = attn_output.transpose(0, 1)

    if not need_weights:
        return attn_output, None

    if average_attn_weights:
        attn_weights_out = attention_weights.mean(dim=1)
    else:
        attn_weights_out = attention_weights
    return attn_output, attn_weights_out


def _is_msda(module: nn.Module) -> bool:
    return module.__class__.__name__ == "MultiScaleDeformableAttention"


def _is_quant_module(module: nn.Module) -> bool:
    # Partition coverage is defined over trainable parameters, but runtime patching
    # remains limited to the module types handled by the current quant backend.
    return isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention, nn.Embedding)) or _is_msda(module)


SUPPORTED_QUANT_PARTITIONS = (
    "exp_a",
    "exp_a1",
    "exp_a1_backbone",
    "exp_a1_backbone_input_proj",
    "exp_a2",
    "exp_a3",
    "exp_a3_head",
    "exp_b",
    "exp_a1_to_b",
    "exp_a3_b",
)


def _is_encoder_aggregation_param(name: str) -> bool:
    return name.startswith("transformer.encoder") and "fusion_layers" not in name


def _is_exp_a1_trainable_param(name: str) -> bool:
    return (
        name.startswith("backbone")
        or name.startswith("input_proj")
        or name.startswith("patch2query")
    )


def _is_exp_a1_backbone_trainable_param(name: str) -> bool:
    return name.startswith("backbone")


def _is_exp_a1_backbone_input_proj_trainable_param(name: str) -> bool:
    return name.startswith("backbone") or name.startswith("input_proj")


def _is_exp_a2_trainable_param(name: str) -> bool:
    return (
        _is_encoder_aggregation_param(name)
        or name.startswith("transformer.level_embed")
        or name.startswith("transformer.enc_output")
        or name.startswith("transformer.enc_output_norm")
        or name.startswith("transformer.enc_out_bbox_embed")
    )


def _is_exp_a3_output_head_param(name: str) -> bool:
    return (
        name.startswith("transformer.decoder.bbox_embed")
        or name.startswith("feature_align")
    )


def _is_exp_a3_output_head_module(name: str) -> bool:
    return (
        name.startswith("transformer.decoder.bbox_embed")
        or name.startswith("feature_align")
    )


def _is_exp_a3_full_trainable_param(name: str) -> bool:
    return (
        name.startswith("transformer.decoder")
        or name.startswith("transformer.tgt_embed")
        or name.startswith("feature_align")
    )


def _is_exp_a3_trainable_param(name: str) -> bool:
    return _is_exp_a3_full_trainable_param(name) and not _is_exp_a3_output_head_param(name)


def _is_exp_a3_head_trainable_param(name: str) -> bool:
    return _is_exp_a3_full_trainable_param(name)


def _is_exp_a_trainable_param(name: str) -> bool:
    return (
        _is_exp_a1_trainable_param(name)
        or _is_exp_a2_trainable_param(name)
        or _is_exp_a3_full_trainable_param(name)
    )


def _is_exp_b_trainable_param(name: str) -> bool:
    return name.startswith("track_embed")


def _is_exp_a1_to_b_trainable_param(name: str) -> bool:
    return (
        _is_exp_a1_trainable_param(name)
        or _is_exp_a2_trainable_param(name)
        or _is_exp_a3_trainable_param(name)
        or _is_exp_b_trainable_param(name)
    )


def _is_exp_a3_b_trainable_param(name: str) -> bool:
    return _is_exp_a3_trainable_param(name) or _is_exp_b_trainable_param(name)


def _is_quant_excluded_module(name: str) -> bool:
    return False


def _is_quant_excluded_param(name: str) -> bool:
    return False


def is_ovtr_quant_state_key(name: str) -> bool:
    return "_ovtr_quant_" in name


def _to_existing_tensor_metadata(loaded: torch.Tensor, existing: Optional[torch.Tensor]) -> torch.Tensor:
    if existing is None:
        return loaded.detach().clone()
    return loaded.detach().to(device=existing.device, dtype=existing.dtype).clone()


def _apply_ovtr_quant_tensor(model: nn.Module, key: str, value: torch.Tensor) -> bool:
    if not is_ovtr_quant_state_key(key):
        return False
    if "." not in key:
        return False

    module_name, attr_name = key.rsplit(".", 1)
    try:
        target_module = model.get_submodule(module_name) if module_name else model
    except AttributeError:
        return False

    if attr_name in target_module._buffers:
        existing = target_module._buffers[attr_name]
        target_module._buffers[attr_name] = _to_existing_tensor_metadata(value, existing)
        return True

    if attr_name in target_module._parameters:
        existing = target_module._parameters[attr_name]
        loaded = _to_existing_tensor_metadata(value, existing)
        if existing is not None and existing.shape == loaded.shape:
            with torch.no_grad():
                existing.copy_(loaded)
        else:
            requires_grad = existing.requires_grad if existing is not None else True
            target_module._parameters[attr_name] = nn.Parameter(loaded, requires_grad=requires_grad)
        return True

    return False


def apply_ovtr_quant_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remaining_state = {}
    for key, value in state_dict.items():
        if is_ovtr_quant_state_key(key) and _apply_ovtr_quant_tensor(model, key, value):
            continue
        remaining_state[key] = value
    return remaining_state


def materialize_checkpoint_bias_parameters(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> int:
    materialized = 0
    for key, value in state_dict.items():
        if not key.endswith(".bias") or is_ovtr_quant_state_key(key):
            continue
        if "." not in key:
            continue

        module_name, attr_name = key.rsplit(".", 1)
        if attr_name != "bias":
            continue
        try:
            target_module = model.get_submodule(module_name) if module_name else model
        except AttributeError:
            continue
        if not isinstance(target_module, (nn.Conv2d, nn.Linear)):
            continue
        if target_module.bias is not None:
            continue
        if value.ndim != 1:
            continue
        expected = target_module.out_channels if isinstance(target_module, nn.Conv2d) else target_module.out_features
        if value.numel() != expected:
            continue

        loaded = value.detach().to(device=target_module.weight.device, dtype=target_module.weight.dtype).clone()
        target_module.bias = nn.Parameter(loaded, requires_grad=False)
        materialized += 1
    return materialized


def _set_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    value = value.detach().contiguous()
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value)


def _require_lsq_initialized(quantizer: nn.Module, owner: str) -> None:
    if quantizer is None:
        raise RuntimeError(f"Missing quantizer for {owner}.")
    if not getattr(quantizer, "initialized", torch.tensor(False)).item():
        raise RuntimeError(f"Quantizer for {owner} is not initialized; cannot export int_msda state.")


def _require_asymmetric_per_tensor_uint4(quantizer: nn.Module, owner: str) -> None:
    _require_lsq_initialized(quantizer, owner)
    if getattr(quantizer, "bit_width", None) != 4 or getattr(quantizer, "symmetric", None):
        raise RuntimeError(f"{owner} must be 4-bit asymmetric activation quantizer for int_msda.")
    if getattr(quantizer, "axis", None) is not None:
        raise RuntimeError(f"{owner} must be per-tensor activation quantizer for int_msda.")
    if quantizer.scale.numel() != 1 or quantizer.zero_point is None or quantizer.zero_point.numel() != 1:
        raise RuntimeError(f"{owner} must expose scalar scale and zero-point for int_msda.")


def _require_symmetric_per_channel_int4_weight(quantizer: nn.Module, owner: str) -> None:
    _require_lsq_initialized(quantizer, owner)
    if getattr(quantizer, "bit_width", None) != 4 or not getattr(quantizer, "symmetric", None):
        raise RuntimeError(f"{owner} must be 4-bit symmetric weight quantizer for int_msda.")
    if getattr(quantizer, "axis", None) != 0:
        raise RuntimeError(f"{owner} must use per-output-channel weight scales for int_msda.")


def _require_attention_per_head_uint8(quantizer: nn.Module, owner: str, num_heads: int) -> None:
    _require_lsq_initialized(quantizer, owner)
    if getattr(quantizer, "bit_width", None) != 8 or getattr(quantizer, "symmetric", None):
        raise RuntimeError(f"{owner} must be 8-bit asymmetric attention quantizer for int_msda.")
    if getattr(quantizer, "axis", None) != 2:
        raise RuntimeError(f"{owner} attention quantizer axis must be the num_heads dimension.")
    if quantizer.scale.numel() != num_heads or quantizer.zero_point is None or quantizer.zero_point.numel() != num_heads:
        raise RuntimeError(f"{owner} attention qparams must have shape [num_heads].")


def _export_lowbit_linear(linear: nn.Linear, owner: str) -> None:
    weight_quantizer = getattr(linear, "_ovtr_quant_weight_quantizer", None)
    _require_symmetric_per_channel_int4_weight(weight_quantizer, f"{owner}.weight")
    q_weight = quantize_weight_int4_per_channel(linear.weight, weight_quantizer.scale)
    _set_buffer(linear, "_ovtr_quant_int4_weight_packed", pack_int4(q_weight).to(linear.weight.device))
    _set_buffer(linear, "_ovtr_quant_int4_weight_scale", weight_quantizer.scale.detach().float().to(linear.weight.device))
    _set_buffer(
        linear,
        "_ovtr_quant_int4_weight_sum",
        q_weight.to(torch.int32).sum(dim=1, dtype=torch.int32).to(linear.weight.device),
    )
    if linear.bias is None:
        bias = torch.empty(0, device=linear.weight.device, dtype=torch.float32)
    else:
        bias = linear.bias.detach().float().to(linear.weight.device)
    _set_buffer(linear, "_ovtr_quant_bias_fp32", bias)
    linear._ovtr_quant_int4_in_features = linear.in_features
    linear._ovtr_quant_int4_out_features = linear.out_features


def _count_int_msda_export_keys(model: nn.Module) -> int:
    return sum(1 for key in model.state_dict().keys() if "_ovtr_quant_int4_" in key or key.endswith("_ovtr_quant_bias_fp32"))


class OVTRQuantController:
    def __init__(
        self,
        model: nn.Module,
        mode: str,
        partition: str,
        weight_bits: int = 4,
        activation_bits: int = 4,
        attention_bits: int = 8,
        range_method: str = "minmax",
        mse_candidates: int = 80,
        mse_bins: int = 2048,
    ):
        if mode not in {"ptq", "qat"}:
            raise ValueError(f"Unsupported quant mode: {mode}")
        if partition not in SUPPORTED_QUANT_PARTITIONS:
            raise ValueError(f"Unsupported quant partition: {partition}")

        self.model = model
        self.mode = mode
        self.partition = partition
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.attention_bits = attention_bits
        self.range_method = range_method
        self.mse_candidates = mse_candidates
        self.mse_bins = mse_bins
        self.quant_modules = []
        self.attention_modules = []
        self.quant_module_names = []
        self.excluded_module_names = []
        self.quant_enabled = False
        self.calibration_enabled = False
        self.observer_enabled = False
        self.runtime_state = "disabled"
        self.trainable_param_names = self._collect_trainable_param_names()
        self._attach()

    def _collect_trainable_param_names(self) -> Set[str]:
        selected = set()
        for name, _ in self.model.named_parameters():
            if is_partition_trainable_param(name, self.partition):
                selected.add(name)
        return selected

    def is_trainable_param(self, name: str) -> bool:
        return name in self.trainable_param_names

    def is_quant_param(self, name: str) -> bool:
        return "_ovtr_quant_" in name

    def quant_lr_for_param(self, name: str, base_lr: float) -> Optional[float]:
        if not self.is_quant_param(name):
            return None
        if self.mode != "qat":
            return 0.0
        return base_lr * 0.1

    def _matches_partition_module(self, name: str, module: nn.Module) -> bool:
        if not _is_quant_module(module):
            return False

        if self.partition in {
            "exp_a",
            "exp_a1",
            "exp_a1_backbone",
            "exp_a1_backbone_input_proj",
            "exp_a1_to_b",
        }:
            if name.startswith("backbone"):
                return True
            if self.partition == "exp_a1_backbone":
                return False
            if name.startswith("input_proj"):
                return True
            if self.partition == "exp_a1_backbone_input_proj":
                return False
            if name.startswith("patch2query"):
                return True
            if self.partition == "exp_a1":
                return False

        if self.partition in {"exp_a", "exp_a2", "exp_a1_to_b"}:
            if (
                name.startswith("transformer.encoder")
                and "fusion_layers" not in name
            ) or name.startswith("transformer.enc_output") or name.startswith("transformer.enc_out_bbox_embed"):
                return True
            if self.partition == "exp_a2":
                return False

        if self.partition in {"exp_a", "exp_a3", "exp_a3_head", "exp_a1_to_b", "exp_a3_b"}:
            if self.partition in {"exp_a3", "exp_a1_to_b", "exp_a3_b"} and _is_exp_a3_output_head_module(name):
                return False
            if (
                name.startswith("transformer.decoder")
                or name.startswith("transformer.tgt_embed")
                or name.startswith("feature_align")
            ):
                return True
            if self.partition in {"exp_a3", "exp_a3_head"}:
                return False

        if self.partition == "exp_a":
            return False

        if self.partition in {"exp_b", "exp_a1_to_b", "exp_a3_b"}:
            return name.startswith("track_embed")

        return False

    def _should_patch_module(self, name: str, module: nn.Module) -> bool:
        if not self._matches_partition_module(name, module):
            return False
        if _is_quant_excluded_module(name):
            self.excluded_module_names.append(name)
            return False
        return True

    def _patch_quant_module(self, module: nn.Module, module_name: str) -> None:
        if getattr(module, "_ovtr_quant_patched", False):
            return

        module._ovtr_quant_backend = self.mode
        module._ovtr_quant_name = module_name
        module._ovtr_quant_quant_enabled = False

        if self.mode == "ptq":
            module.add_module(
                "_ovtr_quant_weight_quantizer",
                StaticAffineQuantizer(
                    bit_width=self.weight_bits,
                    symmetric=True,
                    axis=0,
                ),
            )
            module.add_module(
                "_ovtr_quant_input_observer",
                MSEHistogramObserver(bit_width=self.activation_bits, symmetric=False, sample_limit=self.mse_bins),
            )
            module.add_module(
                "_ovtr_quant_output_observer",
                MSEHistogramObserver(bit_width=self.activation_bits, symmetric=False, sample_limit=self.mse_bins),
            )
            module._ovtr_quant_weight_quantizer.to(module.weight.device)
            module._ovtr_quant_input_observer.to(module.weight.device)
            module._ovtr_quant_output_observer.to(module.weight.device)
            module._ovtr_quant_weight_bits = self.weight_bits
            module._ovtr_quant_calibration_enabled = False
        else:
            module.add_module(
                "_ovtr_quant_weight_quantizer",
                LSQQuantizer(
                    bit_width=self.weight_bits,
                    symmetric=True,
                    axis=0,
                    param_size=module.weight.shape[0],
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_input_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_output_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
                    sample_limit=self.mse_bins,
                ),
            )
            module._ovtr_quant_weight_quantizer.to(module.weight.device)
            module._ovtr_quant_input_quantizer.to(module.weight.device)
            module._ovtr_quant_output_quantizer.to(module.weight.device)
            module._ovtr_quant_observer_enabled = False

        module._ovtr_quant_patched = True
        if isinstance(module, nn.Conv2d):
            module.forward = MethodType(_ovtr_quant_conv2d_forward, module)
        elif isinstance(module, nn.Linear):
            module.forward = MethodType(_ovtr_quant_linear_forward, module)
        else:
            raise TypeError(f"Unsupported quant patch target: {type(module)}")

        self.quant_modules.append(module)
        self.quant_module_names.append(module_name)

    def _patch_embedding_module(self, module: nn.Embedding, module_name: str) -> None:
        if getattr(module, "_ovtr_quant_patched", False):
            return

        module._ovtr_quant_backend = self.mode
        module._ovtr_quant_name = module_name
        module._ovtr_quant_quant_enabled = False

        if self.mode == "ptq":
            module.add_module(
                "_ovtr_quant_weight_quantizer",
                StaticAffineQuantizer(
                    bit_width=self.weight_bits,
                    symmetric=True,
                    axis=0,
                ),
            )
            module.add_module(
                "_ovtr_quant_output_observer",
                MSEHistogramObserver(bit_width=self.activation_bits, symmetric=False, sample_limit=self.mse_bins),
            )
            module._ovtr_quant_weight_quantizer.to(module.weight.device)
            module._ovtr_quant_output_observer.to(module.weight.device)
            module._ovtr_quant_weight_bits = self.weight_bits
            module._ovtr_quant_calibration_enabled = False
        else:
            module.add_module(
                "_ovtr_quant_weight_quantizer",
                LSQQuantizer(
                    bit_width=self.weight_bits,
                    symmetric=True,
                    axis=0,
                    param_size=module.weight.shape[0],
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_output_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
                    sample_limit=self.mse_bins,
                ),
            )
            module._ovtr_quant_weight_quantizer.to(module.weight.device)
            module._ovtr_quant_output_quantizer.to(module.weight.device)
            module._ovtr_quant_observer_enabled = False

        module._ovtr_quant_patched = True
        module._ovtr_quant_get_weight = MethodType(_ovtr_quant_embedding_weight, module)
        module.forward = MethodType(_ovtr_quant_embedding_forward, module)
        self.quant_modules.append(module)
        self.quant_module_names.append(module_name)

    def _patch_multihead_attention_module(self, module: nn.MultiheadAttention, module_name: str) -> None:
        if getattr(module, "_ovtr_quant_patched", False):
            return

        module._ovtr_quant_backend = self.mode
        module._ovtr_quant_name = module_name
        module._ovtr_quant_quant_enabled = False

        if self.mode == "ptq":
            module.add_module(
                "_ovtr_quant_weight_quantizer",
                StaticAffineQuantizer(
                    bit_width=self.weight_bits,
                    symmetric=True,
                    axis=0,
                ),
            )
            module.add_module(
                "_ovtr_quant_query_input_observer",
                MSEHistogramObserver(bit_width=self.activation_bits, symmetric=False, sample_limit=self.mse_bins),
            )
            module.add_module(
                "_ovtr_quant_key_input_observer",
                MSEHistogramObserver(bit_width=self.activation_bits, symmetric=False, sample_limit=self.mse_bins),
            )
            module.add_module(
                "_ovtr_quant_value_input_observer",
                MSEHistogramObserver(bit_width=self.activation_bits, symmetric=False, sample_limit=self.mse_bins),
            )
            module.add_module(
                "_ovtr_quant_query_proj_observer",
                MSEHistogramObserver(bit_width=self.activation_bits, symmetric=False, sample_limit=self.mse_bins),
            )
            module.add_module(
                "_ovtr_quant_key_proj_observer",
                MSEHistogramObserver(bit_width=self.activation_bits, symmetric=False, sample_limit=self.mse_bins),
            )
            module.add_module(
                "_ovtr_quant_value_proj_observer",
                MSEHistogramObserver(bit_width=self.activation_bits, symmetric=False, sample_limit=self.mse_bins),
            )
            module.add_module(
                "_ovtr_quant_attention_observer",
                MSEHistogramObserver(
                    bit_width=self.attention_bits,
                    symmetric=False,
                    axis=1,
                    sample_limit=self.mse_bins,
                ),
            )
            module._ovtr_quant_weight_quantizer.to(module.in_proj_weight.device)
            module._ovtr_quant_query_input_observer.to(module.in_proj_weight.device)
            module._ovtr_quant_key_input_observer.to(module.in_proj_weight.device)
            module._ovtr_quant_value_input_observer.to(module.in_proj_weight.device)
            module._ovtr_quant_query_proj_observer.to(module.in_proj_weight.device)
            module._ovtr_quant_key_proj_observer.to(module.in_proj_weight.device)
            module._ovtr_quant_value_proj_observer.to(module.in_proj_weight.device)
            module._ovtr_quant_attention_observer.to(module.in_proj_weight.device)
            module._ovtr_quant_weight_bits = self.weight_bits
            module._ovtr_quant_calibration_enabled = False
            module._ovtr_quant_attention_calibration_enabled = False
            module._ovtr_quant_attention_quant_enabled = False
        else:
            module.add_module(
                "_ovtr_quant_weight_quantizer",
                LSQQuantizer(
                    bit_width=self.weight_bits,
                    symmetric=True,
                    axis=0,
                    param_size=module.in_proj_weight.shape[0],
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_query_input_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_key_input_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_value_input_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_query_proj_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_key_proj_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_value_proj_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_attention_quantizer",
                LSQQuantizer(
                    bit_width=self.attention_bits,
                    symmetric=False,
                    axis=1,
                    param_size=module.num_heads,
                    sample_limit=self.mse_bins,
                ),
            )
            module._ovtr_quant_weight_quantizer.to(module.in_proj_weight.device)
            module._ovtr_quant_query_input_quantizer.to(module.in_proj_weight.device)
            module._ovtr_quant_key_input_quantizer.to(module.in_proj_weight.device)
            module._ovtr_quant_value_input_quantizer.to(module.in_proj_weight.device)
            module._ovtr_quant_query_proj_quantizer.to(module.in_proj_weight.device)
            module._ovtr_quant_key_proj_quantizer.to(module.in_proj_weight.device)
            module._ovtr_quant_value_proj_quantizer.to(module.in_proj_weight.device)
            module._ovtr_quant_attention_quantizer.to(module.in_proj_weight.device)
            module._ovtr_quant_observer_enabled = False
            module._ovtr_quant_attention_observer_enabled = False
            module._ovtr_quant_attention_quant_enabled = False

        module._ovtr_quant_patched = True
        module.forward = MethodType(_ovtr_quant_multihead_attention_forward, module)
        self.quant_modules.append(module)
        self.quant_module_names.append(module_name)

    def _patch_attention_module(self, module: nn.Module, module_name: str) -> None:
        if getattr(module, "_ovtr_quant_attention_patched", False):
            return

        module._ovtr_quant_backend = self.mode
        module._ovtr_quant_name = module_name
        if self.mode == "ptq":
            module.add_module(
                "_ovtr_quant_attention_observer",
                MSEHistogramObserver(
                    bit_width=self.attention_bits,
                    symmetric=False,
                    axis=2,
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_aggregation_output_observer",
                MSEHistogramObserver(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    sample_limit=self.mse_bins,
                ),
            )
            module._ovtr_quant_attention_observer.to(module.value_proj.weight.device)
            module._ovtr_quant_aggregation_output_observer.to(module.value_proj.weight.device)
            module._ovtr_quant_attention_calibration_enabled = False
            module._ovtr_quant_attention_quant_enabled = False
            module._ovtr_quant_aggregation_calibration_enabled = False
            module._ovtr_quant_aggregation_quant_enabled = False
        else:
            module.add_module(
                "_ovtr_quant_attention_quantizer",
                LSQQuantizer(
                    bit_width=self.attention_bits,
                    symmetric=False,
                    axis=2,
                    param_size=module.num_heads,
                    sample_limit=self.mse_bins,
                ),
            )
            module.add_module(
                "_ovtr_quant_aggregation_output_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
                    sample_limit=self.mse_bins,
                ),
            )
            module._ovtr_quant_attention_quantizer.to(module.value_proj.weight.device)
            module._ovtr_quant_aggregation_output_quantizer.to(module.value_proj.weight.device)
            module._ovtr_quant_attention_observer_enabled = False
            module._ovtr_quant_attention_quant_enabled = False
            module._ovtr_quant_aggregation_observer_enabled = False
            module._ovtr_quant_aggregation_quant_enabled = False
            module._ovtr_quant_force_pytorch_msda = True

        module._ovtr_quant_attention_patched = True
        module._ovtr_quant_deploy_mode = "none"
        module._ovtr_int_msda_dispatch_count = 0
        self.attention_modules.append(module)
        self.quant_module_names.append(module_name)

    def _attach(self) -> None:
        for name, module in self.model.named_modules():
            if not self._should_patch_module(name, module):
                continue
            if _is_msda(module):
                self._patch_attention_module(module, name)
                continue
            if isinstance(module, nn.Embedding):
                self._patch_embedding_module(module, name)
                continue
            if isinstance(module, nn.MultiheadAttention):
                self._patch_multihead_attention_module(module, name)
                continue
            self._patch_quant_module(module, name)

        self.model._ovtr_quant_controller = self
        self.model.transformer._ovtr_quant_controller = self
        self._configure_msda_projection_modules()

    def _configure_msda_projection_modules(self) -> None:
        for module in self.attention_modules:
            output_proj = getattr(module, "output_proj", None)
            if output_proj is not None and getattr(output_proj, "_ovtr_quant_patched", False):
                output_proj._ovtr_quant_skip_input_quantizer = True

    def _iter_recordable_quant_modules(self):
        seen = set()
        for module in list(self.quant_modules) + list(self.attention_modules):
            module_id = id(module)
            if module_id in seen:
                continue
            seen.add(module_id)
            yield module

    def set_quant_boundary_recorder(self, recorder, module_regex: Optional[str] = None) -> None:
        pattern = re.compile(module_regex) if module_regex else None
        for module in self._iter_recordable_quant_modules():
            module_name = getattr(module, "_ovtr_quant_name", "")
            if pattern is None or pattern.search(module_name):
                module._ovtr_quant_boundary_recorder = recorder
            elif hasattr(module, "_ovtr_quant_boundary_recorder"):
                module._ovtr_quant_boundary_recorder = None

    def clear_quant_boundary_recorder(self) -> None:
        for module in self._iter_recordable_quant_modules():
            if hasattr(module, "_ovtr_quant_boundary_recorder"):
                module._ovtr_quant_boundary_recorder = None

    def reset_calibration(self) -> None:
        for module in self.quant_modules:
            if isinstance(module, nn.Embedding):
                if self.mode == "ptq":
                    module._ovtr_quant_output_observer.reset()
                else:
                    module._ovtr_quant_output_quantizer.reset_observer()
                continue
            if isinstance(module, nn.MultiheadAttention):
                if self.mode == "ptq":
                    module._ovtr_quant_query_input_observer.reset()
                    module._ovtr_quant_key_input_observer.reset()
                    module._ovtr_quant_value_input_observer.reset()
                    module._ovtr_quant_query_proj_observer.reset()
                    module._ovtr_quant_key_proj_observer.reset()
                    module._ovtr_quant_value_proj_observer.reset()
                    module._ovtr_quant_attention_observer.reset()
                else:
                    module._ovtr_quant_query_input_quantizer.reset_observer()
                    module._ovtr_quant_key_input_quantizer.reset_observer()
                    module._ovtr_quant_value_input_quantizer.reset_observer()
                    module._ovtr_quant_query_proj_quantizer.reset_observer()
                    module._ovtr_quant_key_proj_quantizer.reset_observer()
                    module._ovtr_quant_value_proj_quantizer.reset_observer()
                    module._ovtr_quant_attention_quantizer.reset_observer()
                continue
            if self.mode == "ptq":
                module._ovtr_quant_input_observer.reset()
                module._ovtr_quant_output_observer.reset()
            else:
                module._ovtr_quant_input_quantizer.reset_observer()
                module._ovtr_quant_output_quantizer.reset_observer()
        for module in self.attention_modules:
            if self.mode == "ptq":
                module._ovtr_quant_attention_observer.reset()
                module._ovtr_quant_aggregation_output_observer.reset()
            else:
                module._ovtr_quant_attention_quantizer.reset_observer()
                module._ovtr_quant_aggregation_output_quantizer.reset_observer()

    def initialize_weight_quantizers(self) -> None:
        for module in self.quant_modules:
            if self.mode == "ptq":
                quantizer = getattr(module, "_ovtr_quant_weight_quantizer", None)
                if quantizer is None:
                    continue
                if isinstance(module, nn.MultiheadAttention):
                    quantizer.initialize_from_tensor(
                        module.in_proj_weight,
                        range_method=self.range_method,
                        mse_candidates=self.mse_candidates,
                        sample_limit=self.mse_bins,
                    )
                else:
                    quantizer.initialize_from_tensor(
                        module.weight,
                        range_method=self.range_method,
                        mse_candidates=self.mse_candidates,
                        sample_limit=self.mse_bins,
                    )
                continue

            if isinstance(module, nn.Embedding):
                module._ovtr_quant_weight_quantizer.maybe_initialize_from_tensor(
                    module.weight,
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                    sample_limit=self.mse_bins,
                )
            elif isinstance(module, nn.MultiheadAttention):
                module._ovtr_quant_weight_quantizer.maybe_initialize_from_tensor(
                    module.in_proj_weight,
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                    sample_limit=self.mse_bins,
                )
            else:
                module._ovtr_quant_weight_quantizer.maybe_initialize_from_tensor(
                    module.weight,
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                    sample_limit=self.mse_bins,
                )

    def _adaround_weight(self, weight: torch.Tensor, quantizer: nn.Module, num_iters: int) -> torch.Tensor:
        scale = getattr(quantizer, "scale", None)
        if scale is None or scale.numel() == 0:
            return quantizer.quantize(weight)

        eps = getattr(quantizer, "eps", 1e-8)
        axis = getattr(quantizer, "axis", None)
        qmin = getattr(quantizer, "qmin", None)
        qmax = getattr(quantizer, "qmax", None)
        if qmin is None or qmax is None:
            qmin, qmax = _quant_bounds(quantizer.bit_width, quantizer.symmetric)

        scale = scale.detach().abs().clamp(min=eps).to(device=weight.device, dtype=weight.dtype)
        zero_point = getattr(quantizer, "zero_point", None)
        if zero_point is None or zero_point.numel() == 0:
            zero_point = torch.zeros_like(scale)
        else:
            zero_point = zero_point.detach().to(device=weight.device, dtype=weight.dtype)

        scale = _reshape_qparam_for_tensor(scale, weight, axis)
        zero_point = _reshape_qparam_for_tensor(zero_point, weight, axis)
        target = weight.detach()
        scaled = target / scale + zero_point
        floor = torch.floor(scaled)
        frac = (scaled - floor).clamp(1e-4, 1.0 - 1e-4)
        alpha = nn.Parameter(torch.log(frac / (1.0 - frac)))

        if num_iters > 0:
            optimizer = torch.optim.Adam([alpha], lr=1e-2)
            with torch.enable_grad():
                for step in range(num_iters):
                    optimizer.zero_grad(set_to_none=True)
                    soft_round = torch.sigmoid(alpha)
                    quantized = torch.clamp(floor + soft_round, qmin, qmax)
                    dequantized = (quantized - zero_point) * scale
                    reconstruction = (dequantized - target).pow(2).mean()
                    progress = step / float(max(num_iters - 1, 1))
                    beta = 20.0 - 18.0 * progress
                    rounding_regularizer = (1.0 - (2.0 * soft_round - 1.0).abs().pow(beta)).mean()
                    loss = reconstruction + 0.01 * rounding_regularizer
                    loss.backward()
                    optimizer.step()

        hard_round = (torch.sigmoid(alpha.detach()) >= 0.5).to(dtype=weight.dtype)
        quantized = torch.clamp(floor + hard_round, qmin, qmax)
        return (quantized - zero_point) * scale

    @torch.no_grad()
    def apply_adaround(self, *, num_iters: int = 1000, num_samples: int = 128) -> int:
        del num_samples
        updated = 0
        self.initialize_weight_quantizers()
        for module in self.quant_modules:
            if isinstance(module, nn.MultiheadAttention):
                weight = module.in_proj_weight
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight
            else:
                continue

            quantizer = getattr(module, "_ovtr_quant_weight_quantizer", None)
            if quantizer is None:
                continue
            quantized = self._adaround_weight(weight, quantizer, num_iters)
            weight.copy_(quantized)
            if hasattr(quantizer, "reset"):
                quantizer.reset()
            if hasattr(quantizer, "initialize_from_tensor"):
                quantizer.initialize_from_tensor(
                    weight,
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                    sample_limit=self.mse_bins,
                )
            updated += 1
        return updated

    @torch.no_grad()
    def apply_bias_correction(self) -> int:
        updated = 0
        for module in self.quant_modules:
            if isinstance(module, nn.Conv2d) and module.bias is None:
                module.bias = nn.Parameter(
                    torch.zeros(module.out_channels, device=module.weight.device, dtype=module.weight.dtype),
                    requires_grad=False,
                )
                updated += 1
            elif isinstance(module, nn.Linear) and module.bias is None:
                module.bias = nn.Parameter(
                    torch.zeros(module.out_features, device=module.weight.device, dtype=module.weight.dtype),
                    requires_grad=False,
                )
                updated += 1
        return updated

    @torch.no_grad()
    def apply_bias_corrections(self, corrections: Dict[str, torch.Tensor]) -> int:
        updated = 0
        if not corrections:
            return updated
        for module in self.quant_modules:
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            name = getattr(module, "_ovtr_quant_name", None)
            if name not in corrections:
                continue
            correction = corrections[name].to(device=module.weight.device, dtype=module.weight.dtype)
            if module.bias is None:
                module.bias = nn.Parameter(correction.clone(), requires_grad=False)
            else:
                module.bias.add_(correction)
            updated += 1
        return updated

    def finalize_calibration(self) -> None:
        def _finalize(observer: nn.Module) -> None:
            if hasattr(observer, "finalize_range"):
                observer.finalize_range(method=self.range_method, num_candidates=self.mse_candidates)

        for module in self.quant_modules:
            if isinstance(module, nn.Embedding):
                if self.mode == "ptq":
                    _finalize(module._ovtr_quant_output_observer)
                else:
                    module._ovtr_quant_output_quantizer.initialize_from_observer(
                        range_method=self.range_method,
                        mse_candidates=self.mse_candidates,
                    )
                continue
            if isinstance(module, nn.MultiheadAttention):
                if self.mode == "ptq":
                    _finalize(module._ovtr_quant_query_input_observer)
                    _finalize(module._ovtr_quant_key_input_observer)
                    _finalize(module._ovtr_quant_value_input_observer)
                    _finalize(module._ovtr_quant_query_proj_observer)
                    _finalize(module._ovtr_quant_key_proj_observer)
                    _finalize(module._ovtr_quant_value_proj_observer)
                    _finalize(module._ovtr_quant_attention_observer)
                else:
                    module._ovtr_quant_query_input_quantizer.initialize_from_observer(
                        range_method=self.range_method,
                        mse_candidates=self.mse_candidates,
                    )
                    module._ovtr_quant_key_input_quantizer.initialize_from_observer(
                        range_method=self.range_method,
                        mse_candidates=self.mse_candidates,
                    )
                    module._ovtr_quant_value_input_quantizer.initialize_from_observer(
                        range_method=self.range_method,
                        mse_candidates=self.mse_candidates,
                    )
                    module._ovtr_quant_query_proj_quantizer.initialize_from_observer(
                        range_method=self.range_method,
                        mse_candidates=self.mse_candidates,
                    )
                    module._ovtr_quant_key_proj_quantizer.initialize_from_observer(
                        range_method=self.range_method,
                        mse_candidates=self.mse_candidates,
                    )
                    module._ovtr_quant_value_proj_quantizer.initialize_from_observer(
                        range_method=self.range_method,
                        mse_candidates=self.mse_candidates,
                    )
                    module._ovtr_quant_attention_quantizer.initialize_from_observer(
                        range_method=self.range_method,
                        mse_candidates=self.mse_candidates,
                    )
                continue
            if self.mode == "ptq":
                _finalize(module._ovtr_quant_input_observer)
                _finalize(module._ovtr_quant_output_observer)
            else:
                module._ovtr_quant_input_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )
                module._ovtr_quant_output_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )
        for module in self.attention_modules:
            if self.mode == "ptq":
                _finalize(module._ovtr_quant_attention_observer)
                _finalize(module._ovtr_quant_aggregation_output_observer)
            else:
                module._ovtr_quant_attention_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )
                module._ovtr_quant_aggregation_output_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )

    def has_serialized_quant_state(self) -> bool:
        for module in self.quant_modules:
            if isinstance(module, nn.Embedding):
                if self.mode == "ptq":
                    if (
                        module._ovtr_quant_weight_quantizer.initialized.item()
                        or module._ovtr_quant_output_observer.initialized.item()
                    ):
                        return True
                else:
                    if (
                        module._ovtr_quant_weight_quantizer.initialized.item()
                        or module._ovtr_quant_output_quantizer.initialized.item()
                    ):
                        return True
                continue
            if isinstance(module, nn.MultiheadAttention):
                if self.mode == "ptq":
                    if (
                        module._ovtr_quant_weight_quantizer.initialized.item()
                        or module._ovtr_quant_query_input_observer.initialized.item()
                        or module._ovtr_quant_key_input_observer.initialized.item()
                        or module._ovtr_quant_value_input_observer.initialized.item()
                        or module._ovtr_quant_query_proj_observer.initialized.item()
                        or module._ovtr_quant_key_proj_observer.initialized.item()
                        or module._ovtr_quant_value_proj_observer.initialized.item()
                        or module._ovtr_quant_attention_observer.initialized.item()
                    ):
                        return True
                else:
                    if (
                        module._ovtr_quant_weight_quantizer.initialized.item()
                        or module._ovtr_quant_query_input_quantizer.initialized.item()
                        or module._ovtr_quant_key_input_quantizer.initialized.item()
                        or module._ovtr_quant_value_input_quantizer.initialized.item()
                        or module._ovtr_quant_query_proj_quantizer.initialized.item()
                        or module._ovtr_quant_key_proj_quantizer.initialized.item()
                        or module._ovtr_quant_value_proj_quantizer.initialized.item()
                        or module._ovtr_quant_attention_quantizer.initialized.item()
                    ):
                        return True
                continue
            if self.mode == "ptq":
                if (
                    module._ovtr_quant_weight_quantizer.initialized.item()
                    or module._ovtr_quant_input_observer.initialized.item()
                    or module._ovtr_quant_output_observer.initialized.item()
                ):
                    return True
            else:
                if module._ovtr_quant_weight_quantizer.initialized.item():
                    return True
                if module._ovtr_quant_input_quantizer.initialized.item() or module._ovtr_quant_output_quantizer.initialized.item():
                    return True
        for module in self.attention_modules:
            if self.mode == "ptq":
                if (
                    module._ovtr_quant_attention_observer.initialized.item()
                    or module._ovtr_quant_aggregation_output_observer.initialized.item()
                ):
                    return True
            else:
                if (
                    module._ovtr_quant_attention_quantizer.initialized.item()
                    or module._ovtr_quant_aggregation_output_quantizer.initialized.item()
                ):
                    return True
        return False

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.model.state_dict().items()
            if is_ovtr_quant_state_key(key)
        }

    def load_exported_state_dict(self, state_dict: Optional[Dict[str, torch.Tensor]]) -> None:
        if not state_dict:
            return
        remaining_state = apply_ovtr_quant_state_dict(self.model, state_dict)
        if remaining_state:
            self.model.load_state_dict(remaining_state, strict=False)

    def initialize_qat_from_calibration(self) -> None:
        if self.mode != "qat":
            return
        for module in self.quant_modules:
            if isinstance(module, nn.Embedding):
                module._ovtr_quant_weight_quantizer.maybe_initialize_from_tensor(
                    module.weight,
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                    sample_limit=self.mse_bins,
                )
                module._ovtr_quant_output_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )
                continue
            if isinstance(module, nn.MultiheadAttention):
                module._ovtr_quant_weight_quantizer.maybe_initialize_from_tensor(
                    module.in_proj_weight,
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                    sample_limit=self.mse_bins,
                )
                module._ovtr_quant_query_input_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )
                module._ovtr_quant_key_input_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )
                module._ovtr_quant_value_input_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )
                module._ovtr_quant_query_proj_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )
                module._ovtr_quant_key_proj_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )
                module._ovtr_quant_value_proj_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )
                module._ovtr_quant_attention_quantizer.initialize_from_observer(
                    range_method=self.range_method,
                    mse_candidates=self.mse_candidates,
                )
                continue
            module._ovtr_quant_weight_quantizer.maybe_initialize_from_tensor(
                module.weight,
                range_method=self.range_method,
                mse_candidates=self.mse_candidates,
                sample_limit=self.mse_bins,
            )
            module._ovtr_quant_input_quantizer.initialize_from_observer(
                range_method=self.range_method,
                mse_candidates=self.mse_candidates,
            )
            module._ovtr_quant_output_quantizer.initialize_from_observer(
                range_method=self.range_method,
                mse_candidates=self.mse_candidates,
            )
        for module in self.attention_modules:
            module._ovtr_quant_attention_quantizer.initialize_from_observer(
                range_method=self.range_method,
                mse_candidates=self.mse_candidates,
            )
            module._ovtr_quant_aggregation_output_quantizer.initialize_from_observer(
                range_method=self.range_method,
                mse_candidates=self.mse_candidates,
            )

    def enable_calibration(self, reset: bool = True) -> None:
        if reset:
            self.reset_calibration()
        self.initialize_weight_quantizers()
        self.quant_enabled = False
        self.runtime_state = "calibration"
        if self.mode == "ptq":
            self.calibration_enabled = True
            for module in self.quant_modules:
                module._ovtr_quant_quant_enabled = False
                module._ovtr_quant_calibration_enabled = True
                if isinstance(module, nn.MultiheadAttention):
                    module._ovtr_quant_attention_quant_enabled = False
                    module._ovtr_quant_attention_calibration_enabled = True
            for module in self.attention_modules:
                module._ovtr_quant_attention_quant_enabled = False
                module._ovtr_quant_attention_calibration_enabled = True
                module._ovtr_quant_aggregation_quant_enabled = False
                module._ovtr_quant_aggregation_calibration_enabled = True
            return

        self.observer_enabled = True
        for module in self.quant_modules:
            module._ovtr_quant_quant_enabled = False
            module._ovtr_quant_observer_enabled = True
            if isinstance(module, nn.MultiheadAttention):
                module._ovtr_quant_attention_quant_enabled = False
                module._ovtr_quant_attention_observer_enabled = True
        for module in self.attention_modules:
            module._ovtr_quant_attention_quant_enabled = False
            module._ovtr_quant_attention_observer_enabled = True
            module._ovtr_quant_aggregation_quant_enabled = False
            module._ovtr_quant_aggregation_observer_enabled = True

    def enable_quantization(self) -> None:
        if self.mode != "ptq":
            raise RuntimeError("enable_quantization is only valid for PTQ")
        self.quant_enabled = True
        self.calibration_enabled = False
        self.runtime_state = "quantized"
        for module in self.quant_modules:
            module._ovtr_quant_quant_enabled = True
            module._ovtr_quant_calibration_enabled = False
            if isinstance(module, nn.MultiheadAttention):
                module._ovtr_quant_attention_quant_enabled = True
                module._ovtr_quant_attention_calibration_enabled = False
        for module in self.attention_modules:
            module._ovtr_quant_attention_quant_enabled = True
            module._ovtr_quant_attention_calibration_enabled = False
            module._ovtr_quant_aggregation_quant_enabled = True
            module._ovtr_quant_aggregation_calibration_enabled = False

    def enable_qat(self) -> None:
        if self.mode != "qat":
            raise RuntimeError("enable_qat is only valid for QAT")
        self.initialize_qat_from_calibration()
        self.quant_enabled = True
        self.observer_enabled = False
        self.runtime_state = "qat"
        for module in self.quant_modules:
            module._ovtr_quant_quant_enabled = True
            module._ovtr_quant_observer_enabled = False
            if isinstance(module, nn.MultiheadAttention):
                module._ovtr_quant_attention_quant_enabled = True
                module._ovtr_quant_attention_observer_enabled = False
        for module in self.attention_modules:
            module._ovtr_quant_attention_quant_enabled = True
            module._ovtr_quant_attention_observer_enabled = False
            module._ovtr_quant_aggregation_quant_enabled = True
            module._ovtr_quant_aggregation_observer_enabled = False

    def disable(self) -> None:
        self.quant_enabled = False
        self.calibration_enabled = False
        self.observer_enabled = False
        self.runtime_state = "disabled"
        for module in self.quant_modules:
            module._ovtr_quant_quant_enabled = False
            if self.mode == "ptq":
                module._ovtr_quant_calibration_enabled = False
                if isinstance(module, nn.MultiheadAttention):
                    module._ovtr_quant_attention_calibration_enabled = False
                    module._ovtr_quant_attention_quant_enabled = False
            else:
                module._ovtr_quant_observer_enabled = False
                if isinstance(module, nn.MultiheadAttention):
                    module._ovtr_quant_attention_observer_enabled = False
                    module._ovtr_quant_attention_quant_enabled = False
        for module in self.attention_modules:
            module._ovtr_quant_attention_quant_enabled = False
            if self.mode == "ptq":
                module._ovtr_quant_attention_calibration_enabled = False
                module._ovtr_quant_aggregation_quant_enabled = False
                module._ovtr_quant_aggregation_calibration_enabled = False
            else:
                module._ovtr_quant_attention_observer_enabled = False
                module._ovtr_quant_aggregation_quant_enabled = False
                module._ovtr_quant_aggregation_observer_enabled = False

    def prepare_int_msda_deploy(self) -> int:
        if self.mode != "qat":
            raise RuntimeError("--quant_deploy int_msda requires a QAT checkpoint.")
        if self.weight_bits != 4 or self.activation_bits != 4 or self.attention_bits != 8:
            raise RuntimeError(
                "int_msda requires weight_bits=4, activation_bits=4, attention_bits=8."
            )
        if self.model.training:
            raise RuntimeError("--quant_deploy int_msda is eval/inference only.")

        prepared = 0
        for module in self.attention_modules:
            name = getattr(module, "_ovtr_quant_name", module.__class__.__name__)
            _require_attention_per_head_uint8(
                getattr(module, "_ovtr_quant_attention_quantizer", None),
                f"{name}.attention",
                module.num_heads,
            )
            _require_asymmetric_per_tensor_uint4(
                getattr(module, "_ovtr_quant_aggregation_output_quantizer", None),
                f"{name}.aggregation_output",
            )
            for role in ("value_proj", "sampling_offsets", "attention_weights", "output_proj"):
                linear = getattr(module, role, None)
                if not isinstance(linear, nn.Linear):
                    raise RuntimeError(f"{name}.{role} must be nn.Linear for int_msda.")
                if role != "output_proj":
                    _require_asymmetric_per_tensor_uint4(
                        getattr(linear, "_ovtr_quant_input_quantizer", None),
                        f"{name}.{role}.input",
                    )
                _require_asymmetric_per_tensor_uint4(
                    getattr(linear, "_ovtr_quant_output_quantizer", None),
                    f"{name}.{role}.output",
                )
                _export_lowbit_linear(linear, f"{name}.{role}")
            module._ovtr_quant_deploy_mode = "int_msda"
            module._ovtr_int_msda_dispatch_count = 0
            prepared += 1
        if prepared == 0:
            raise RuntimeError("No MSDeformAttn modules were prepared for int_msda deploy.")
        self.runtime_state = "int_msda"
        return prepared

    def summary(self) -> str:
        return (
            f"OVTRQuantController(mode={self.mode}, partition={self.partition}, "
            f"weight_bits={self.weight_bits}, activation_bits={self.activation_bits}, "
            f"attention_bits={self.attention_bits}, quant_modules={len(self.quant_modules)}, "
            f"attention_modules={len(self.attention_modules)}, state={self.runtime_state})"
        )


def maybe_prepare_ovtr_quant_controller(
    model: nn.Module,
    mode: str,
    partition: str,
    weight_bits: int = 4,
    activation_bits: int = 4,
    attention_bits: int = 8,
    range_method: str = "minmax",
    mse_candidates: int = 80,
    mse_bins: int = 2048,
) -> OVTRQuantController:
    controller = getattr(model, "_ovtr_quant_controller", None)
    if controller is not None:
        if controller.mode != mode or controller.partition != partition:
            raise ValueError(
                f"Existing quant controller mode/partition mismatch: "
                f"{controller.mode}/{controller.partition} vs {mode}/{partition}"
            )
        return controller
    return OVTRQuantController(
        model=model,
        mode=mode,
        partition=partition,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        attention_bits=attention_bits,
        range_method=range_method,
        mse_candidates=mse_candidates,
        mse_bins=mse_bins,
    )


class GroupAQATController(OVTRQuantController):
    """Compatibility shim; new QAT state is owned by OVTRQuantController and _ovtr_quant_* keys."""

    def __init__(
        self,
        model: nn.Module,
        weight_bits: int = 4,
        activation_bits: int = 4,
        attention_bits: int = 8,
    ):
        super().__init__(
            model=model,
            mode="qat",
            partition="exp_a1",
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            attention_bits=attention_bits,
        )


def maybe_prepare_group_a_qat(
    model: nn.Module,
    weight_bits: int = 4,
    activation_bits: int = 4,
    attention_bits: int = 8,
) -> OVTRQuantController:
    return maybe_prepare_ovtr_quant_controller(
        model,
        mode="qat",
        partition="exp_a1",
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        attention_bits=attention_bits,
    )


def maybe_observe_and_quantize_attention(module: nn.Module, attention_weights: torch.Tensor) -> torch.Tensor:
    backend = getattr(module, "_ovtr_quant_backend", None)
    if backend == "ptq":
        observer = getattr(module, "_ovtr_quant_attention_observer", None)
        if observer is None:
            return attention_weights
        if getattr(module, "_ovtr_quant_attention_calibration_enabled", False):
            observer.observe(attention_weights)
        if getattr(module, "_ovtr_quant_attention_quant_enabled", False):
            quantized = observer.fake_quant(attention_weights)
            _record_quant_boundary(module, "attention", attention_weights, quantized)
            return quantized
        return attention_weights

    if backend == "qat":
        quantizer = getattr(module, "_ovtr_quant_attention_quantizer", None)
        if quantizer is None:
            return attention_weights
        if getattr(module, "_ovtr_quant_attention_observer_enabled", False):
            quantizer.observe(attention_weights)
        if getattr(module, "_ovtr_quant_attention_quant_enabled", False):
            quantized = quantizer.quantize(attention_weights)
            _record_quant_boundary(module, "attention", attention_weights, quantized)
            return quantized
        return attention_weights

    return attention_weights


def maybe_observe_and_quantize_msda_aggregation(module: nn.Module, output: torch.Tensor) -> torch.Tensor:
    backend = getattr(module, "_ovtr_quant_backend", None)
    if backend == "ptq":
        observer = getattr(module, "_ovtr_quant_aggregation_output_observer", None)
        if observer is None:
            return output
        if getattr(module, "_ovtr_quant_aggregation_calibration_enabled", False):
            observer.observe(output)
        if getattr(module, "_ovtr_quant_aggregation_quant_enabled", False):
            quantized = observer.fake_quant(output)
            _record_quant_boundary(module, "msda_aggregation_output", output, quantized)
            return quantized
        return output

    if backend == "qat":
        quantizer = getattr(module, "_ovtr_quant_aggregation_output_quantizer", None)
        if quantizer is None:
            return output
        if getattr(module, "_ovtr_quant_aggregation_observer_enabled", False):
            quantizer.observe(output)
        if getattr(module, "_ovtr_quant_aggregation_quant_enabled", False):
            quantized = quantizer.quantize(output)
            _record_quant_boundary(module, "msda_aggregation_output", output, quantized)
            return quantized
        return output

    return output


def maybe_get_quantized_embedding_weight(module: nn.Module) -> torch.Tensor:
    getter = getattr(module, "_ovtr_quant_get_weight", None)
    if getter is not None:
        return getter()
    return module.weight


def quant_param_name_fragment() -> str:
    return "_ovtr_quant_"


def is_quant_trainable_param(name: str) -> bool:
    return quant_param_name_fragment() in name


def is_partition_trainable_param(name: str, partition: str) -> bool:
    if partition == "exp_a":
        return _is_exp_a_trainable_param(name)
    if partition == "exp_a1":
        return _is_exp_a1_trainable_param(name)
    if partition == "exp_a1_backbone":
        return _is_exp_a1_backbone_trainable_param(name)
    if partition == "exp_a1_backbone_input_proj":
        return _is_exp_a1_backbone_input_proj_trainable_param(name)
    if partition == "exp_a2":
        return _is_exp_a2_trainable_param(name)
    if partition == "exp_a3":
        return _is_exp_a3_trainable_param(name)
    if partition == "exp_a3_head":
        return _is_exp_a3_head_trainable_param(name)
    if partition == "exp_b":
        return _is_exp_b_trainable_param(name)
    if partition == "exp_a1_to_b":
        return _is_exp_a1_to_b_trainable_param(name)
    if partition == "exp_a3_b":
        return _is_exp_a3_b_trainable_param(name)
    raise ValueError(f"Unsupported partition: {partition}")


def is_excluded_from_quantization_param(name: str) -> bool:
    return _is_quant_excluded_param(name)
