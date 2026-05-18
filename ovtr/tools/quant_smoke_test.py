import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.quant_utils import (  # noqa: E402
    MSEHistogramObserver,
    dequantize_affine_uint,
    is_partition_trainable_param,
    is_quant_trainable_param,
    maybe_prepare_ovtr_quant_controller,
    pack_int4,
    pack_uint4,
    quantize_affine_uint,
    unpack_int4,
    unpack_uint4,
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


class ToyPartitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.input_proj = nn.Linear(4, 4)
        self.patch2query = nn.Linear(4, 4)
        self.transformer = nn.Module()
        self.transformer.encoder = nn.Sequential(nn.Linear(4, 4))
        self.transformer.encoder.fusion_layers = nn.Sequential(nn.Linear(4, 4))
        self.transformer.enc_output = nn.Linear(4, 4)
        self.transformer.enc_out_bbox_embed = nn.Linear(4, 4)
        self.transformer.decoder = nn.Module()
        self.transformer.decoder.layers = nn.Sequential(nn.Linear(4, 4))
        self.transformer.decoder.bbox_embed = nn.Linear(4, 4)
        self.transformer.tgt_embed = nn.Linear(4, 4)
        self.feature_align = nn.Linear(4, 4)
        self.track_embed = nn.Linear(4, 4)
        self.unrelated = nn.Linear(4, 4)


def _partition_quant_module_names(partition):
    model = ToyPartitionModel()
    controller = maybe_prepare_ovtr_quant_controller(model, mode="ptq", partition=partition, range_method="minmax")
    return set(controller.quant_module_names)


def _partition_trainable_param_names(partition):
    model = ToyPartitionModel()
    return {name for name, _ in model.named_parameters() if is_partition_trainable_param(name, partition)}


def _param_names_for_modules(module_names):
    return {f"{name}.{suffix}" for name in module_names for suffix in ("weight", "bias")}


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


def test_combined_partition_coverage():
    a1_to_b_modules = {
        "backbone",
        "input_proj",
        "patch2query",
        "transformer.encoder.0",
        "transformer.enc_output",
        "transformer.enc_out_bbox_embed",
        "transformer.decoder.layers.0",
        "transformer.tgt_embed",
        "track_embed",
    }
    a3_b_modules = {
        "transformer.decoder.layers.0",
        "transformer.tgt_embed",
        "track_embed",
    }

    assert _partition_quant_module_names("exp_a1_to_b") == a1_to_b_modules
    assert _partition_trainable_param_names("exp_a1_to_b") == _param_names_for_modules(a1_to_b_modules)
    assert _partition_quant_module_names("exp_a3_b") == a3_b_modules
    assert _partition_trainable_param_names("exp_a3_b") == _param_names_for_modules(a3_b_modules)

    exp_a3_modules = {
        "transformer.decoder.layers.0",
        "transformer.tgt_embed",
    }
    assert _partition_quant_module_names("exp_a3") == exp_a3_modules
    assert _partition_trainable_param_names("exp_a3") == _param_names_for_modules(exp_a3_modules)

    exp_a_modules = _partition_quant_module_names("exp_a")
    assert "transformer.decoder.bbox_embed" in exp_a_modules
    assert "feature_align" in exp_a_modules
    assert "track_embed" not in exp_a_modules


def test_lowbit_pack_layout_and_reconstruction():
    uint4 = torch.tensor([0, 15, 2], dtype=torch.uint8)
    packed_uint4 = pack_uint4(uint4)
    assert packed_uint4.tolist() == [0xF0, 0x02]
    assert unpack_uint4(packed_uint4, uint4.numel()).tolist() == uint4.tolist()

    int4 = torch.tensor([-8, -1, 0, 7], dtype=torch.int8)
    packed_int4 = pack_int4(int4)
    assert packed_int4.tolist() == [0xF8, 0x70]
    assert unpack_int4(packed_int4, int4.numel()).tolist() == int4.tolist()

    x = torch.tensor([-0.31, 0.0, 0.23, 0.91])
    scale = torch.tensor([0.1])
    zero_point = torch.tensor([7.0])
    q = quantize_affine_uint(x, scale, zero_point, bit_width=4)
    reconstructed = dequantize_affine_uint(unpack_uint4(pack_uint4(q), q.numel()), scale, zero_point)
    expected = (torch.round(x / scale + zero_point).clamp(0, 15) - zero_point) * scale
    assert torch.allclose(reconstructed, expected)


def test_uint4_zero_point_correction_formula():
    x_q = torch.tensor([[7, 8, 9, 10], [5, 7, 11, 15]], dtype=torch.int32)
    w_q = torch.tensor([[1, -2, 3, -4], [-3, 2, -1, 4]], dtype=torch.int32)
    z_x = torch.tensor(7, dtype=torch.int32)
    s_x = torch.tensor(0.125)
    s_w = torch.tensor([0.25, 0.5])
    bias = torch.tensor([0.1, -0.2])

    direct = ((x_q - z_x).float()[:, None, :] * w_q.float()[None, :, :]).sum(dim=-1)
    direct = direct * (s_x * s_w)[None, :] + bias[None, :]

    corrected_acc = x_q @ w_q.T - z_x * w_q.sum(dim=1)
    corrected = corrected_acc.float() * (s_x * s_w)[None, :] + bias[None, :]
    assert torch.allclose(corrected, direct)


def main():
    torch.manual_seed(0)
    test_bn_folding()
    test_mse_range_not_worse_than_minmax()
    test_ptq_adaround_commit()
    test_ptq_legacy_minmax_path()
    test_bias_correction_stats_path()
    test_qat_quant_params()
    test_combined_partition_coverage()
    test_lowbit_pack_layout_and_reconstruction()
    test_uint4_zero_point_correction_formula()
    print("quant smoke passed")


if __name__ == "__main__":
    main()
