from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _reshape_qparam_for_tensor(param: torch.Tensor, x: torch.Tensor, axis: Optional[int]) -> torch.Tensor:
    if axis is None:
        return param
    shape = [1] * x.ndim
    shape[axis] = param.shape[0]
    return param.reshape(shape)


class MinMaxObserver(nn.Module):
    def __init__(self, bit_width: int, symmetric: bool, axis: Optional[int] = None, eps: float = 1e-8):
        super().__init__()
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.axis = axis
        self.eps = eps
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def reset(self) -> None:
        self.min_val = self.min_val.new_empty(0)
        self.max_val = self.max_val.new_empty(0)
        self.initialized.fill_(False)

    def observe(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return

        with torch.no_grad():
            x_detached = x.detach()
            if self.axis is None:
                cur_min = x_detached.amin()
                cur_max = x_detached.amax()
            else:
                reduce_dims = tuple(dim for dim in range(x_detached.ndim) if dim != self.axis)
                cur_min = x_detached.amin(dim=reduce_dims)
                cur_max = x_detached.amax(dim=reduce_dims)

            if not self.initialized.item():
                self.min_val = cur_min
                self.max_val = cur_max
                self.initialized.fill_(True)
            else:
                self.min_val = torch.minimum(self.min_val, cur_min)
                self.max_val = torch.maximum(self.max_val, cur_max)

    def _get_qparams(self, x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, int]]:
        if not self.initialized.item():
            return None

        if self.symmetric:
            qmax = (1 << (self.bit_width - 1)) - 1
            qmin = -qmax
            max_abs = torch.maximum(self.min_val.abs(), self.max_val.abs())
            scale = torch.clamp(max_abs / float(qmax), min=self.eps)
            zero_point = torch.zeros_like(scale)
        else:
            qmin = 0
            qmax = (1 << self.bit_width) - 1
            scale = torch.clamp((self.max_val - self.min_val) / float(qmax - qmin), min=self.eps)
            zero_point = torch.round(qmin - (self.min_val / scale)).clamp(qmin, qmax)

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


def _maybe_observe_and_quantize_activation(
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


def _group_a_ptq_conv2d_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = _maybe_observe_and_quantize_activation(
        x,
        self._group_a_ptq_input_observer,
        self._group_a_ptq_calibration_enabled,
        self._group_a_ptq_quant_enabled,
    )
    weight = fake_quantize_symmetric_weight(self.weight, self._group_a_ptq_weight_bits, axis=0)
    out = F.conv2d(
        x,
        weight,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )
    out = _maybe_observe_and_quantize_activation(
        out,
        self._group_a_ptq_output_observer,
        self._group_a_ptq_calibration_enabled,
        self._group_a_ptq_quant_enabled,
    )
    return out


def _group_a_ptq_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = _maybe_observe_and_quantize_activation(
        x,
        self._group_a_ptq_input_observer,
        self._group_a_ptq_calibration_enabled,
        self._group_a_ptq_quant_enabled,
    )
    weight = fake_quantize_symmetric_weight(self.weight, self._group_a_ptq_weight_bits, axis=0)
    out = F.linear(x, weight, self.bias)
    out = _maybe_observe_and_quantize_activation(
        out,
        self._group_a_ptq_output_observer,
        self._group_a_ptq_calibration_enabled,
        self._group_a_ptq_quant_enabled,
    )
    return out


class GroupAPTQController:
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
        self.calibration_enabled = False
        self._attach()

    def _patch_quant_module(self, module: nn.Module, module_name: str) -> None:
        if getattr(module, "_group_a_ptq_patched", False):
            return

        module.add_module(
            "_group_a_ptq_input_observer",
            MinMaxObserver(bit_width=self.activation_bits, symmetric=False),
        )
        module.add_module(
            "_group_a_ptq_output_observer",
            MinMaxObserver(bit_width=self.activation_bits, symmetric=False),
        )
        module._group_a_ptq_input_observer.to(module.weight.device)
        module._group_a_ptq_output_observer.to(module.weight.device)
        module._group_a_ptq_patched = True
        module._group_a_ptq_name = module_name
        module._group_a_ptq_weight_bits = self.weight_bits
        module._group_a_ptq_quant_enabled = False
        module._group_a_ptq_calibration_enabled = False

        if isinstance(module, nn.Conv2d):
            module.forward = MethodType(_group_a_ptq_conv2d_forward, module)
        elif isinstance(module, nn.Linear):
            module.forward = MethodType(_group_a_ptq_linear_forward, module)
        else:
            raise TypeError(f"Unsupported PTQ patch target: {type(module)}")

        self.quant_modules.append(module)

    def _attach_to_backbone(self) -> None:
        for name, module in self.model.backbone[0].named_modules():
            if isinstance(module, nn.Conv2d):
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
                if not hasattr(module, "_group_a_ptq_attention_observer"):
                    module.add_module(
                        "_group_a_ptq_attention_observer",
                        MinMaxObserver(bit_width=self.attention_bits, symmetric=False, axis=2),
                    )
                    module._group_a_ptq_attention_observer.to(module.value_proj.weight.device)
                module._group_a_ptq_attention_quant_enabled = False
                module._group_a_ptq_attention_calibration_enabled = False
                self.attention_modules.append(module)

    def _attach(self) -> None:
        self._attach_to_backbone()
        self._attach_to_input_proj()
        self._attach_to_encoder()
        self.model._group_a_ptq_controller = self
        self.model.transformer._group_a_ptq_controller = self

    def reset_calibration(self) -> None:
        for module in self.quant_modules:
            module._group_a_ptq_input_observer.reset()
            module._group_a_ptq_output_observer.reset()
        for module in self.attention_modules:
            module._group_a_ptq_attention_observer.reset()

    def enable_calibration(self, reset: bool = True) -> None:
        if reset:
            self.reset_calibration()
        self.quant_enabled = False
        self.calibration_enabled = True
        for module in self.quant_modules:
            module._group_a_ptq_quant_enabled = False
            module._group_a_ptq_calibration_enabled = True
        for module in self.attention_modules:
            module._group_a_ptq_attention_quant_enabled = False
            module._group_a_ptq_attention_calibration_enabled = True

    def enable_quantization(self) -> None:
        self.quant_enabled = True
        self.calibration_enabled = False
        for module in self.quant_modules:
            module._group_a_ptq_quant_enabled = True
            module._group_a_ptq_calibration_enabled = False
        for module in self.attention_modules:
            module._group_a_ptq_attention_quant_enabled = True
            module._group_a_ptq_attention_calibration_enabled = False

    def disable(self) -> None:
        self.quant_enabled = False
        self.calibration_enabled = False
        for module in self.quant_modules:
            module._group_a_ptq_quant_enabled = False
            module._group_a_ptq_calibration_enabled = False
        for module in self.attention_modules:
            module._group_a_ptq_attention_quant_enabled = False
            module._group_a_ptq_attention_calibration_enabled = False

    def quantize_level_embed(self, level_embed: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return level_embed
        return fake_quantize_symmetric_weight(level_embed, bit_width=self.weight_bits, axis=None)

    def summary(self) -> str:
        return (
            f"GroupAPTQController(weight_bits={self.weight_bits}, "
            f"activation_bits={self.activation_bits}, attention_bits={self.attention_bits}, "
            f"quant_modules={len(self.quant_modules)}, attention_modules={len(self.attention_modules)})"
        )


def maybe_prepare_group_a_ptq(
    model: nn.Module,
    weight_bits: int = 4,
    activation_bits: int = 4,
    attention_bits: int = 8,
) -> GroupAPTQController:
    controller = getattr(model, "_group_a_ptq_controller", None)
    if controller is not None:
        return controller
    return GroupAPTQController(
        model=model,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        attention_bits=attention_bits,
    )


def maybe_quantize_group_a_level_embed(transformer: nn.Module, level_embed: torch.Tensor) -> torch.Tensor:
    controller = getattr(transformer, "_group_a_ptq_controller", None)
    if controller is None:
        return level_embed
    return controller.quantize_level_embed(level_embed)


def maybe_observe_and_quantize_group_a_attention(
    module: nn.Module,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    observer = getattr(module, "_group_a_ptq_attention_observer", None)
    if observer is None:
        return attention_weights

    if getattr(module, "_group_a_ptq_attention_calibration_enabled", False):
        observer.observe(attention_weights)
    if getattr(module, "_group_a_ptq_attention_quant_enabled", False):
        return observer.fake_quant(attention_weights)
    return attention_weights
