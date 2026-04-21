import math
from types import MethodType
from typing import Optional, Tuple

import torch
import torch.distributed as dist
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


class LSQQuantizer(nn.Module):
    def __init__(
        self,
        bit_width: int,
        symmetric: bool,
        axis: Optional[int],
        param_size: int,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.axis = axis
        self.param_size = param_size
        self.eps = eps
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.scale = nn.Parameter(torch.ones(param_size))
        if symmetric:
            self.zero_point = None
        else:
            self.zero_point = nn.Parameter(torch.zeros(param_size))

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

    def observe(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return

        with torch.no_grad():
            x_detached = x.detach()
            cur_min, cur_max = _reduce_minmax(x_detached, self.axis)
            if self.min_val.numel() == 0:
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                self.min_val = torch.minimum(self.min_val, cur_min)
                self.max_val = torch.maximum(self.max_val, cur_max)

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

    def initialize_from_observer(self) -> None:
        if self.initialized.item() or self.min_val.numel() == 0:
            return
        self._initialize(self.min_val, self.max_val)

    def maybe_initialize_from_tensor(self, x: torch.Tensor) -> None:
        if self.initialized.item():
            return
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


def _maybe_observe_and_lsq_quantize(
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


def _maybe_lsq_quantize_weight(
    weight: torch.Tensor,
    quantizer: LSQQuantizer,
    quant_enabled: bool,
) -> torch.Tensor:
    if not quant_enabled:
        return weight
    return quantizer.quantize(weight)


def _group_a_qat_conv2d_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = _maybe_observe_and_lsq_quantize(
        x,
        self._group_a_qat_input_quantizer,
        self._group_a_qat_observer_enabled,
        self._group_a_qat_quant_enabled,
    )
    weight = _maybe_lsq_quantize_weight(
        self.weight,
        self._group_a_qat_weight_quantizer,
        self._group_a_qat_quant_enabled,
    )
    out = F.conv2d(
        x,
        weight,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )
    out = _maybe_observe_and_lsq_quantize(
        out,
        self._group_a_qat_output_quantizer,
        self._group_a_qat_observer_enabled,
        self._group_a_qat_quant_enabled,
    )
    return out


def _group_a_qat_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = _maybe_observe_and_lsq_quantize(
        x,
        self._group_a_qat_input_quantizer,
        self._group_a_qat_observer_enabled,
        self._group_a_qat_quant_enabled,
    )
    weight = _maybe_lsq_quantize_weight(
        self.weight,
        self._group_a_qat_weight_quantizer,
        self._group_a_qat_quant_enabled,
    )
    out = F.linear(x, weight, self.bias)
    out = _maybe_observe_and_lsq_quantize(
        out,
        self._group_a_qat_output_quantizer,
        self._group_a_qat_observer_enabled,
        self._group_a_qat_quant_enabled,
    )
    return out


class GroupAQATController:
    def __init__(
        self,
        model: nn.Module,
        weight_bits: int = 4,
        activation_bits: int = 4,
        attention_bits: int = 8,
    ):
        self.model = model
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.attention_bits = attention_bits
        self.quant_modules = []
        self.attention_modules = []
        self.quant_enabled = False
        self.observer_enabled = False
        self.mode = "disabled"
        self._attach()

    def _patch_quant_module(self, module: nn.Module, module_name: str) -> None:
        if getattr(module, "_group_a_qat_patched", False):
            return

        weight_quantizer = LSQQuantizer(
            bit_width=self.weight_bits,
            symmetric=True,
            axis=0,
            param_size=module.weight.shape[0],
        )
        input_quantizer = LSQQuantizer(
            bit_width=self.activation_bits,
            symmetric=False,
            axis=None,
            param_size=1,
        )
        output_quantizer = LSQQuantizer(
            bit_width=self.activation_bits,
            symmetric=False,
            axis=None,
            param_size=1,
        )
        module.add_module("_group_a_qat_weight_quantizer", weight_quantizer)
        module.add_module("_group_a_qat_input_quantizer", input_quantizer)
        module.add_module("_group_a_qat_output_quantizer", output_quantizer)
        module._group_a_qat_weight_quantizer.to(module.weight.device)
        module._group_a_qat_input_quantizer.to(module.weight.device)
        module._group_a_qat_output_quantizer.to(module.weight.device)
        module._group_a_qat_patched = True
        module._group_a_qat_name = module_name
        module._group_a_qat_quant_enabled = False
        module._group_a_qat_observer_enabled = False

        if isinstance(module, nn.Conv2d):
            module.forward = MethodType(_group_a_qat_conv2d_forward, module)
        elif isinstance(module, nn.Linear):
            module.forward = MethodType(_group_a_qat_linear_forward, module)
        else:
            raise TypeError(f"Unsupported QAT patch target: {type(module)}")

        self.quant_modules.append(module)

    def _attach_to_backbone(self) -> None:
        skipped_first_conv = False
        for name, module in self.model.backbone[0].named_modules():
            if isinstance(module, nn.Conv2d):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                qualified = f"backbone.0.{name}" if name else "backbone.0"
                self._patch_quant_module(module, qualified)

    def _attach_to_input_proj(self) -> None:
        for name, module in self.model.input_proj.named_modules():
            if isinstance(module, nn.Conv2d):
                qualified = f"input_proj.{name}" if name else "input_proj"
                self._patch_quant_module(module, qualified)

    def _attach_to_encoder(self) -> None:
        for name, module in self.model.transformer.encoder.named_modules():
            if isinstance(module, nn.Linear):
                qualified = f"transformer.encoder.{name}" if name else "transformer.encoder"
                self._patch_quant_module(module, qualified)
            elif module.__class__.__name__ == "MultiScaleDeformableAttention":
                if not hasattr(module, "_group_a_qat_attention_quantizer"):
                    module.add_module(
                        "_group_a_qat_attention_quantizer",
                        LSQQuantizer(
                            bit_width=self.attention_bits,
                            symmetric=False,
                            axis=2,
                            param_size=module.num_heads,
                        ),
                    )
                    module._group_a_qat_attention_quantizer.to(module.value_proj.weight.device)
                module._group_a_qat_attention_quant_enabled = False
                module._group_a_qat_attention_observer_enabled = False
                module._group_a_qat_force_pytorch_msda = True
                self.attention_modules.append(module)

    def _attach(self) -> None:
        self._attach_to_backbone()
        self._attach_to_input_proj()
        self._attach_to_encoder()
        self.model._group_a_qat_controller = self
        self.model.transformer._group_a_qat_controller = self

    def reset_observers(self) -> None:
        for module in self.quant_modules:
            module._group_a_qat_input_quantizer.reset_observer()
            module._group_a_qat_output_quantizer.reset_observer()
        for module in self.attention_modules:
            module._group_a_qat_attention_quantizer.reset_observer()

    def initialize_weight_quantizers(self) -> None:
        for module in self.quant_modules:
            module._group_a_qat_weight_quantizer.maybe_initialize_from_tensor(module.weight)

    def initialize_observed_qparams(self) -> None:
        for module in self.quant_modules:
            module._group_a_qat_input_quantizer.initialize_from_observer()
            module._group_a_qat_output_quantizer.initialize_from_observer()
        for module in self.attention_modules:
            module._group_a_qat_attention_quantizer.initialize_from_observer()

    def synchronize_calibration_stats(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        def _sync_minmax(quantizer: LSQQuantizer) -> None:
            if quantizer.min_val.numel() == 0 or quantizer.max_val.numel() == 0:
                return
            dist.all_reduce(quantizer.min_val, op=dist.ReduceOp.MIN)
            dist.all_reduce(quantizer.max_val, op=dist.ReduceOp.MAX)

        for module in self.quant_modules:
            _sync_minmax(module._group_a_qat_input_quantizer)
            _sync_minmax(module._group_a_qat_output_quantizer)
        for module in self.attention_modules:
            _sync_minmax(module._group_a_qat_attention_quantizer)

    def _apply_runtime_flags(self, *, quant_enabled: bool, observer_enabled: bool) -> None:
        self.quant_enabled = quant_enabled
        self.observer_enabled = observer_enabled
        for module in self.quant_modules:
            module._group_a_qat_quant_enabled = quant_enabled
            module._group_a_qat_observer_enabled = observer_enabled
        for module in self.attention_modules:
            module._group_a_qat_attention_quant_enabled = quant_enabled
            module._group_a_qat_attention_observer_enabled = observer_enabled

    def enable_calibration(self, reset: bool = True) -> None:
        if reset:
            self.reset_observers()
        self.mode = "calibration"
        self._apply_runtime_flags(quant_enabled=False, observer_enabled=True)

    def enable_qat(self, reset_observers: bool = False) -> None:
        if reset_observers:
            self.reset_observers()
        self.initialize_weight_quantizers()
        self.initialize_observed_qparams()
        self.mode = "qat"
        self._apply_runtime_flags(quant_enabled=True, observer_enabled=False)

    def freeze_observers(self) -> None:
        self.initialize_weight_quantizers()
        self.initialize_observed_qparams()
        self.mode = "frozen"
        self._apply_runtime_flags(quant_enabled=True, observer_enabled=False)

    def disable(self) -> None:
        self.mode = "disabled"
        self._apply_runtime_flags(quant_enabled=False, observer_enabled=False)

    def summary(self) -> str:
        return (
            f"GroupAQATController(method=LSQ, weight_bits={self.weight_bits}, "
            f"activation_bits={self.activation_bits}, attention_bits={self.attention_bits}, "
            f"quant_modules={len(self.quant_modules)}, attention_modules={len(self.attention_modules)}, "
            f"mode={self.mode})"
        )


def maybe_prepare_group_a_qat(
    model: nn.Module,
    weight_bits: int = 4,
    activation_bits: int = 4,
    attention_bits: int = 8,
) -> GroupAQATController:
    controller = getattr(model, "_group_a_qat_controller", None)
    if controller is not None:
        return controller
    return GroupAQATController(
        model=model,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        attention_bits=attention_bits,
    )


def maybe_observe_and_fake_quantize_group_a_attention(
    module: nn.Module,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    quantizer = getattr(module, "_group_a_qat_attention_quantizer", None)
    if quantizer is None:
        return attention_weights

    if getattr(module, "_group_a_qat_attention_observer_enabled", False):
        quantizer.observe(attention_weights)
    if getattr(module, "_group_a_qat_attention_quant_enabled", False):
        return quantizer.quantize(attention_weights)
    return attention_weights
