import math
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
            cur_min, cur_max = _reduce_minmax(x.detach(), self.axis)
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
            cur_min, cur_max = _reduce_minmax(x.detach(), self.axis)
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


def _ovtr_quant_conv2d_forward(self, x: torch.Tensor) -> torch.Tensor:
    if self._ovtr_quant_backend == "ptq":
        x = _maybe_observe_and_quantize_activation_ptq(
            x,
            self._ovtr_quant_input_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        weight = fake_quantize_symmetric_weight(self.weight, self._ovtr_quant_weight_bits, axis=0)
        out = F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        out = _maybe_observe_and_quantize_activation_ptq(
            out,
            self._ovtr_quant_output_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        return out

    x = _maybe_observe_and_quantize_activation_qat(
        x,
        self._ovtr_quant_input_quantizer,
        self._ovtr_quant_observer_enabled,
        self._ovtr_quant_quant_enabled,
    )
    weight = self.weight
    if self._ovtr_quant_quant_enabled:
        weight = self._ovtr_quant_weight_quantizer.quantize(weight)
    out = F.conv2d(
        x,
        weight,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )
    out = _maybe_observe_and_quantize_activation_qat(
        out,
        self._ovtr_quant_output_quantizer,
        self._ovtr_quant_observer_enabled,
        self._ovtr_quant_quant_enabled,
    )
    return out


def _ovtr_quant_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    if self._ovtr_quant_backend == "ptq":
        x = _maybe_observe_and_quantize_activation_ptq(
            x,
            self._ovtr_quant_input_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        weight = fake_quantize_symmetric_weight(self.weight, self._ovtr_quant_weight_bits, axis=0)
        out = F.linear(x, weight, self.bias)
        out = _maybe_observe_and_quantize_activation_ptq(
            out,
            self._ovtr_quant_output_observer,
            self._ovtr_quant_calibration_enabled,
            self._ovtr_quant_quant_enabled,
        )
        return out

    x = _maybe_observe_and_quantize_activation_qat(
        x,
        self._ovtr_quant_input_quantizer,
        self._ovtr_quant_observer_enabled,
        self._ovtr_quant_quant_enabled,
    )
    weight = self.weight
    if self._ovtr_quant_quant_enabled:
        weight = self._ovtr_quant_weight_quantizer.quantize(weight)
    out = F.linear(x, weight, self.bias)
    out = _maybe_observe_and_quantize_activation_qat(
        out,
        self._ovtr_quant_output_quantizer,
        self._ovtr_quant_observer_enabled,
        self._ovtr_quant_quant_enabled,
    )
    return out


def _is_msda(module: nn.Module) -> bool:
    return module.__class__.__name__ == "MultiScaleDeformableAttention"


def _is_quant_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv2d, nn.Linear)) or _is_msda(module)


SUPPORTED_QUANT_PARTITIONS = ("exp_a", "exp_a1", "exp_a2", "exp_a3", "exp_a4", "exp_b")


def _is_exp_b_trainable_param(name: str) -> bool:
    return (
        name.startswith("track_embed")
        or "update_attn" in name
        or "norm4" in name
        or name.startswith("transformer.decoder.layers.0")
    )


def _is_detector_output_head_param(name: str) -> bool:
    return (
        "bbox_embed" in name
        or name.startswith("feature_align")
        or name.startswith("transformer.enc_out_bbox_embed")
        or name.startswith("transformer.decoder.log_scale")
        or name.startswith("transformer.decoder.bias_lang")
        or name.startswith("transformer.decoder.bias0")
    )


def _is_encoder_aggregation_param(name: str) -> bool:
    return name.startswith("transformer.encoder") and "fusion_layers" not in name


def _is_shared_decoder_param(name: str) -> bool:
    return (
        name.startswith("transformer.decoder")
        and not name.startswith("transformer.decoder.layers.0")
        and "update_attn" not in name
        and "norm4" not in name
        and not _is_detector_output_head_param(name)
    )


def _is_exp_a1_trainable_param(name: str) -> bool:
    return name.startswith("backbone")


def _is_exp_a2_trainable_param(name: str) -> bool:
    return name.startswith("input_proj") or name.startswith("patch2query")


def _is_exp_a3_trainable_param(name: str) -> bool:
    return _is_encoder_aggregation_param(name)


def _is_exp_a4_trainable_param(name: str) -> bool:
    return _is_shared_decoder_param(name)


def _is_exp_a_trainable_param(name: str) -> bool:
    return (
        _is_exp_a1_trainable_param(name)
        or _is_exp_a2_trainable_param(name)
        or _is_exp_a3_trainable_param(name)
        or _is_exp_a4_trainable_param(name)
    )


def _is_quant_excluded_module(name: str) -> bool:
    return (
        "bbox_embed" in name
        or name.startswith("feature_align")
        or name.startswith("transformer.enc_out_bbox_embed")
        or name.startswith("track_embed.linear2")
        or name.startswith("track_embed.linear_feat2")
        or name.startswith("track_embed.linear_pos2")
    )


def _is_quant_excluded_param(name: str) -> bool:
    return (
        _is_detector_output_head_param(name)
        or name.startswith("track_embed.linear2")
        or name.startswith("track_embed.linear_feat2")
        or name.startswith("track_embed.linear_pos2")
    )


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


class OVTRQuantController:
    def __init__(
        self,
        model: nn.Module,
        mode: str,
        partition: str,
        weight_bits: int = 8,
        activation_bits: int = 6,
        attention_bits: int = 4,
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
        self.quant_modules = []
        self.attention_modules = []
        self.quant_module_names = []
        self.excluded_module_names = []
        self.quant_enabled = False
        self.calibration_enabled = False
        self.observer_enabled = False
        self.runtime_state = "disabled"
        self.first_backbone_conv_name = self._find_first_backbone_conv_name()
        self.trainable_param_names = self._collect_trainable_param_names()
        self._attach()

    def _find_first_backbone_conv_name(self) -> Optional[str]:
        for name, module in self.model.named_modules():
            if name.startswith("backbone.0") and isinstance(module, nn.Conv2d):
                return name
        return None

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

        if self.partition in {"exp_a", "exp_a1"}:
            if name == self.first_backbone_conv_name:
                return False
            if name.startswith("backbone.0"):
                return True
            if self.partition == "exp_a1":
                return False

        if self.partition in {"exp_a", "exp_a2"}:
            if name.startswith("input_proj"):
                return True
            if name.startswith("patch2query"):
                return True
            if self.partition == "exp_a2":
                return False

        if self.partition in {"exp_a", "exp_a3"}:
            if name.startswith("transformer.encoder") and "fusion_layers" not in name:
                return True
            if self.partition == "exp_a3":
                return False

        if self.partition in {"exp_a", "exp_a4"}:
            if _is_shared_decoder_param(name):
                return True
            if self.partition == "exp_a4":
                return False

        if self.partition == "exp_a":
            return False

        return self.partition == "exp_b" and (
            name.startswith("track_embed")
            or "update_attn" in name
            or "norm4" in name
            or name.startswith("transformer.decoder.layers.0")
        )

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
                "_ovtr_quant_input_observer",
                MinMaxObserver(bit_width=self.activation_bits, symmetric=False),
            )
            module.add_module(
                "_ovtr_quant_output_observer",
                MinMaxObserver(bit_width=self.activation_bits, symmetric=False),
            )
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
                ),
            )
            module.add_module(
                "_ovtr_quant_input_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
                ),
            )
            module.add_module(
                "_ovtr_quant_output_quantizer",
                LSQQuantizer(
                    bit_width=self.activation_bits,
                    symmetric=False,
                    axis=None,
                    param_size=1,
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

    def _patch_attention_module(self, module: nn.Module, module_name: str) -> None:
        if getattr(module, "_ovtr_quant_attention_patched", False):
            return

        module._ovtr_quant_backend = self.mode
        module._ovtr_quant_name = module_name
        if self.mode == "ptq":
            module.add_module(
                "_ovtr_quant_attention_observer",
                MinMaxObserver(bit_width=self.attention_bits, symmetric=False, axis=2),
            )
            module._ovtr_quant_attention_observer.to(module.value_proj.weight.device)
            module._ovtr_quant_attention_calibration_enabled = False
            module._ovtr_quant_attention_quant_enabled = False
        else:
            module.add_module(
                "_ovtr_quant_attention_quantizer",
                LSQQuantizer(
                    bit_width=self.attention_bits,
                    symmetric=False,
                    axis=2,
                    param_size=module.num_heads,
                ),
            )
            module._ovtr_quant_attention_quantizer.to(module.value_proj.weight.device)
            module._ovtr_quant_attention_observer_enabled = False
            module._ovtr_quant_attention_quant_enabled = False
            module._ovtr_quant_force_pytorch_msda = True

        module._ovtr_quant_attention_patched = True
        self.attention_modules.append(module)
        self.quant_module_names.append(module_name)

    def _attach(self) -> None:
        for name, module in self.model.named_modules():
            if not self._should_patch_module(name, module):
                continue
            if _is_msda(module):
                self._patch_attention_module(module, name)
                continue
            self._patch_quant_module(module, name)

        self.model._ovtr_quant_controller = self
        self.model.transformer._ovtr_quant_controller = self

    def reset_calibration(self) -> None:
        for module in self.quant_modules:
            if self.mode == "ptq":
                module._ovtr_quant_input_observer.reset()
                module._ovtr_quant_output_observer.reset()
            else:
                module._ovtr_quant_input_quantizer.reset_observer()
                module._ovtr_quant_output_quantizer.reset_observer()
        for module in self.attention_modules:
            if self.mode == "ptq":
                module._ovtr_quant_attention_observer.reset()
            else:
                module._ovtr_quant_attention_quantizer.reset_observer()

    def has_serialized_quant_state(self) -> bool:
        for module in self.quant_modules:
            if self.mode == "ptq":
                if module._ovtr_quant_input_observer.initialized.item() or module._ovtr_quant_output_observer.initialized.item():
                    return True
            else:
                if module._ovtr_quant_weight_quantizer.initialized.item():
                    return True
                if module._ovtr_quant_input_quantizer.initialized.item() or module._ovtr_quant_output_quantizer.initialized.item():
                    return True
        for module in self.attention_modules:
            if self.mode == "ptq":
                if module._ovtr_quant_attention_observer.initialized.item():
                    return True
            else:
                if module._ovtr_quant_attention_quantizer.initialized.item():
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
            module._ovtr_quant_weight_quantizer.maybe_initialize_from_tensor(module.weight)
            module._ovtr_quant_input_quantizer.initialize_from_observer()
            module._ovtr_quant_output_quantizer.initialize_from_observer()
        for module in self.attention_modules:
            module._ovtr_quant_attention_quantizer.initialize_from_observer()

    def enable_calibration(self, reset: bool = True) -> None:
        if reset:
            self.reset_calibration()
        self.quant_enabled = False
        self.runtime_state = "calibration"
        if self.mode == "ptq":
            self.calibration_enabled = True
            for module in self.quant_modules:
                module._ovtr_quant_quant_enabled = False
                module._ovtr_quant_calibration_enabled = True
            for module in self.attention_modules:
                module._ovtr_quant_attention_quant_enabled = False
                module._ovtr_quant_attention_calibration_enabled = True
            return

        self.observer_enabled = True
        for module in self.quant_modules:
            module._ovtr_quant_quant_enabled = False
            module._ovtr_quant_observer_enabled = True
        for module in self.attention_modules:
            module._ovtr_quant_attention_quant_enabled = False
            module._ovtr_quant_attention_observer_enabled = True

    def enable_quantization(self) -> None:
        if self.mode != "ptq":
            raise RuntimeError("enable_quantization is only valid for PTQ")
        self.quant_enabled = True
        self.calibration_enabled = False
        self.runtime_state = "quantized"
        for module in self.quant_modules:
            module._ovtr_quant_quant_enabled = True
            module._ovtr_quant_calibration_enabled = False
        for module in self.attention_modules:
            module._ovtr_quant_attention_quant_enabled = True
            module._ovtr_quant_attention_calibration_enabled = False

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
        for module in self.attention_modules:
            module._ovtr_quant_attention_quant_enabled = True
            module._ovtr_quant_attention_observer_enabled = False

    def disable(self) -> None:
        self.quant_enabled = False
        self.calibration_enabled = False
        self.observer_enabled = False
        self.runtime_state = "disabled"
        for module in self.quant_modules:
            module._ovtr_quant_quant_enabled = False
            if self.mode == "ptq":
                module._ovtr_quant_calibration_enabled = False
            else:
                module._ovtr_quant_observer_enabled = False
        for module in self.attention_modules:
            module._ovtr_quant_attention_quant_enabled = False
            if self.mode == "ptq":
                module._ovtr_quant_attention_calibration_enabled = False
            else:
                module._ovtr_quant_attention_observer_enabled = False

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
    weight_bits: int = 8,
    activation_bits: int = 6,
    attention_bits: int = 4,
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
            return observer.fake_quant(attention_weights)
        return attention_weights

    if backend == "qat":
        quantizer = getattr(module, "_ovtr_quant_attention_quantizer", None)
        if quantizer is None:
            return attention_weights
        if getattr(module, "_ovtr_quant_attention_observer_enabled", False):
            quantizer.observe(attention_weights)
        if getattr(module, "_ovtr_quant_attention_quant_enabled", False):
            return quantizer.quantize(attention_weights)
        return attention_weights

    return attention_weights


def quant_param_name_fragment() -> str:
    return "_ovtr_quant_"


def is_quant_trainable_param(name: str) -> bool:
    return quant_param_name_fragment() in name


def is_partition_trainable_param(name: str, partition: str) -> bool:
    if partition == "exp_a":
        return _is_exp_a_trainable_param(name)
    if partition == "exp_a1":
        return _is_exp_a1_trainable_param(name)
    if partition == "exp_a2":
        return _is_exp_a2_trainable_param(name)
    if partition == "exp_a3":
        return _is_exp_a3_trainable_param(name)
    if partition == "exp_a4":
        return _is_exp_a4_trainable_param(name)
    if partition == "exp_b":
        return _is_exp_b_trainable_param(name)
    raise ValueError(f"Unsupported partition: {partition}")


def is_excluded_from_quantization_param(name: str) -> bool:
    return _is_quant_excluded_param(name)
