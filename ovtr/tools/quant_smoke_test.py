import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.quant_utils import (  # noqa: E402
    MSEHistogramObserver,
    is_quant_trainable_param,
    maybe_prepare_ovtr_quant_controller,
)
from util.quantization import _fold_frozen_batch_norms  # noqa: E402
from util.quantization import _finalize_bias_correction_stats, _register_bias_correction_hooks  # noqa: E402


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x):
        scale = self.weight.reshape(1, -1, 1, 1) * (self.running_var.reshape(1, -1, 1, 1) + 1e-5).rsqrt()
        bias = self.bias.reshape(1, -1, 1, 1) - self.running_mean.reshape(1, -1, 1, 1) * scale
        return x * scale + bias


class ToyQuantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        self.transformer = nn.Module()

    def forward(self, x):
        return self.backbone(x)


def test_bn_folding():
    model = nn.Sequential(nn.Conv2d(3, 3, 1, bias=False), FrozenBatchNorm2d(3), nn.ReLU(), nn.Conv2d(3, 2, 1))
    model[1].weight.copy_(torch.tensor([0.5, 1.5, 2.0]))
    model[1].bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    model[1].running_mean.copy_(torch.tensor([0.2, -0.1, 0.4]))
    model[1].running_var.copy_(torch.tensor([0.5, 1.2, 2.0]))
    x = torch.randn(2, 3, 5, 5)
    reference = model(x)
    folded = _fold_frozen_batch_norms(model)
    assert folded == 1
    assert torch.allclose(model[1].weight, torch.ones_like(model[1].weight))
    assert torch.allclose(model[1].bias, torch.zeros_like(model[1].bias))
    assert torch.allclose(model[1].running_mean, torch.zeros_like(model[1].running_mean))
    assert torch.isfinite(model(x)).all()
    assert torch.allclose(reference, model(x), atol=1e-5, rtol=1e-4)


def test_mse_range_not_worse_than_minmax():
    x = torch.cat([torch.linspace(-0.7, 0.8, 2048), torch.tensor([-8.0, 9.0])])
    minmax_observer = MSEHistogramObserver(bit_width=8, symmetric=False)
    minmax_observer.observe(x)
    minmax_fake = minmax_observer.fake_quant(x)
    minmax_error = (minmax_fake - x).pow(2).mean()

    mse_observer = MSEHistogramObserver(bit_width=8, symmetric=False)
    mse_observer.observe(x)
    mse_observer.finalize_range(method="mse", num_candidates=32)
    mse_fake = mse_observer.fake_quant(x)
    mse_error = (mse_fake - x).pow(2).mean()
    assert mse_error <= minmax_error + 1e-8


def test_ptq_adaround_commit():
    model = ToyQuantModel()
    controller = maybe_prepare_ovtr_quant_controller(model, mode="ptq", partition="exp_a1", range_method="mse", mse_candidates=16)
    updated = controller.apply_adaround(num_iters=2, num_samples=2)
    assert updated == 2
    controller.enable_quantization()
    out = model(torch.randn(3, 4))
    assert torch.isfinite(out).all()
    first_linear = model.backbone[0]
    requantized = first_linear._ovtr_quant_weight_quantizer.quantize(first_linear.weight)
    assert torch.allclose(first_linear.weight, requantized, atol=1e-6, rtol=1e-5)


def test_ptq_legacy_minmax_path():
    model = ToyQuantModel()
    controller = maybe_prepare_ovtr_quant_controller(model, mode="ptq", partition="exp_a1", range_method="minmax")
    controller.enable_calibration(reset=True)
    with torch.no_grad():
        model(torch.randn(4, 4))
    controller.finalize_calibration()
    controller.enable_quantization()
    assert torch.isfinite(model(torch.randn(2, 4))).all()


def test_bias_correction_stats_path():
    model = ToyQuantModel()
    controller = maybe_prepare_ovtr_quant_controller(model, mode="ptq", partition="exp_a1", range_method="minmax")
    controller.apply_bias_correction()
    controller.enable_calibration(reset=True)
    stats, handles = _register_bias_correction_hooks(controller)
    with torch.no_grad():
        model(torch.randn(8, 4))
    for handle in handles:
        handle.remove()
    corrections = _finalize_bias_correction_stats(stats)
    updated = controller.apply_bias_corrections(corrections)
    assert updated == 2
    assert all(torch.isfinite(module.bias).all() for module in model.backbone if isinstance(module, nn.Linear))


def test_qat_quant_params():
    model = ToyQuantModel()
    controller = maybe_prepare_ovtr_quant_controller(model, mode="qat", partition="exp_a1", range_method="mse", mse_candidates=16)
    controller.enable_calibration(reset=True)
    with torch.no_grad():
        model(torch.randn(4, 4))
    controller.finalize_calibration()
    controller.enable_qat()
    quant_param_names = [name for name, param in model.named_parameters() if is_quant_trainable_param(name) and param.requires_grad]
    assert any(name.endswith(".scale") for name in quant_param_names)
    assert all("_ovtr_quant_" in name for name in quant_param_names)
    assert torch.isfinite(model(torch.randn(2, 4))).all()


def main():
    torch.manual_seed(0)
    test_bn_folding()
    test_mse_range_not_worse_than_minmax()
    test_ptq_adaround_commit()
    test_ptq_legacy_minmax_path()
    test_bias_correction_stats_path()
    test_qat_quant_params()
    print("quant smoke passed")


if __name__ == "__main__":
    main()
