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
from util.quantization import (  # noqa: E402
    CR_QAT_STAGE1_PARTITION,
    CR_QAT_STAGE2_PARTITION,
    _fold_frozen_batch_norms,
    temporarily_disable_trkd_trace,
)
from util.quantization import _finalize_bias_correction_stats, _register_bias_correction_hooks  # noqa: E402
from util.distillation import (  # noqa: E402
    BACKBONE_DISTILL_FEATURE_KEY,
    CR_QAT_BACKBONE_FD_LOSS_KEY,
    CR_QAT_TRKD_LOSS_KEY,
    TRKD_TRACE_KEY,
    BackboneFeatureDistiller,
    make_projected_src_feature_frame,
    make_trkd_trace,
    projected_backbone_feature_distillation_loss,
    trkd_relational_distillation_loss,
    trkd_similarity_matrix_loss,
)
from util.events import EventStorage  # noqa: E402
from engine import train_one_epoch_mot  # noqa: E402


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


class ToyFirstLastModel(nn.Module):
    def __init__(self):
        super().__init__()
        stem = nn.Module()
        stem.body = nn.Module()
        stem.body.conv1 = nn.Conv2d(3, 4, 1)
        stem.body.layer1 = nn.Conv2d(4, 4, 1)
        stem.patch_embed = nn.Module()
        stem.patch_embed.proj = nn.Conv2d(3, 4, 1)
        self.backbone = nn.Sequential(stem)
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        self.transformer.decoder.layers = nn.Sequential(nn.Linear(4, 4))
        self.transformer.decoder.bbox_embed = nn.Linear(4, 4)
        self.transformer.tgt_embed = nn.Linear(4, 4)
        self.feature_align = nn.Linear(4, 4)


class ToyTraceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.return_trkd_trace = True
        self.sync_states = []
        self.recalibration_states = []

    def _sync_trkd_trace_capture(self):
        self.sync_states.append(bool(self.return_trkd_trace))

    def run_transition_recalibration(self):
        self.recalibration_states.append(bool(self.return_trkd_trace))


class ToyStageTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = nn.Parameter(torch.tensor(1.0))
        self.stage2_extra = nn.Parameter(torch.tensor(2.0))
        self.return_trkd_trace = True
        self.current_stage = 1
        self.forward_trace = []
        self.recalibration_states = []

    def _sync_trkd_trace_capture(self):
        pass

    def run_transition_recalibration(self):
        self.recalibration_states.append(bool(self.return_trkd_trace))

    def forward(self, data):
        self.forward_trace.append(
            (int(self.current_stage), bool(self.return_trkd_trace), bool(self.stage2_extra.requires_grad))
        )
        stage_scale = torch.as_tensor(float(self.current_stage), device=self.probe.device)
        loss_probe = self.probe * data["x"].sum() * stage_scale
        if self.current_stage >= 2:
            loss_probe = loss_probe + self.stage2_extra * data["x"].sum()
        return {
            "loss_probe": loss_probe,
            "pred_boxes": [torch.zeros(1, 4, device=self.probe.device)],
            "pred_logits": [torch.zeros(1, 1, device=self.probe.device)],
            "track_instances": None,
        }


class ToyStageCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_dict = {"loss_probe": 1.0}

    def forward(self, outputs):
        return {"loss_probe": outputs["loss_probe"]}


class ToyEngineStageScheduler:
    def __init__(self, model):
        self.model = model
        self.switch_step = 1
        self.current_stage = 1
        self.last_global_step = 0
        self.recalibrated = False
        self._apply_trainability()

    def _apply_trainability(self):
        self.model.probe.requires_grad_(True)
        self.model.stage2_extra.requires_grad_(self.current_stage >= 2)

    def before_step(self, global_step):
        self.last_global_step = int(global_step)
        if global_step >= self.switch_step and self.current_stage != 2:
            with temporarily_disable_trkd_trace(self.model):
                self.model.run_transition_recalibration()
            self.current_stage = 2
        self.model.current_stage = self.current_stage
        self._apply_trainability()

    def state_dict(self):
        return {
            "cr_qat_stage": self.current_stage,
            "cr_qat_global_step": self.last_global_step,
            "cr_qat_switch_step": self.switch_step,
        }


def _partition_quant_module_names(partition):
    model = ToyPartitionModel()
    controller = maybe_prepare_ovtr_quant_controller(model, mode="ptq", partition=partition, range_method="minmax")
    return set(controller.quant_module_names)


def _partition_trainable_param_names(partition):
    model = ToyPartitionModel()
    return {name for name, _ in model.named_parameters() if is_partition_trainable_param(name, partition)}


def _param_names_for_modules(module_names):
    return {f"{name}.{suffix}" for name in module_names for suffix in ("weight", "bias")}


def _named_quant_modules(controller):
    return {getattr(module, "_ovtr_quant_name", ""): module for module in controller.quant_modules}


def _observer_enabled(module):
    return bool(getattr(module, "_ovtr_quant_observer_enabled", False))


def _quant_enabled(module):
    return bool(getattr(module, "_ovtr_quant_quant_enabled", False))


def _run_linear_partition_modules(model, module_names):
    x = torch.randn(3, 4)
    modules = dict(model.named_modules())
    for name in sorted(module_names):
        module = modules[name]
        if isinstance(module, nn.Linear):
            module(x)


def _apply_stage_trainability(model, controller, partition):
    for _, param in model.named_parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if is_quant_trainable_param(name):
            trainable = controller.is_quant_param_in_partition(name, partition)
        elif partition == CR_QAT_STAGE2_PARTITION:
            trainable = True
        else:
            trainable = is_partition_trainable_param(name, partition)
        if trainable:
            param.requires_grad_(True)


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


def test_projected_backbone_feature_distillation_loss_masks_padding():
    feature = torch.randn(1, 3, 4, 4)
    student_feature = feature.clone().requires_grad_(True)
    teacher_feature = feature.clone()
    teacher_feature[:, :, 0, 0] = teacher_feature[:, :, 0, 0] + 100.0
    padding_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    padding_mask[:, 0, 0] = True

    loss = projected_backbone_feature_distillation_loss(
        [make_projected_src_feature_frame([student_feature])],
        [make_projected_src_feature_frame([teacher_feature], [padding_mask])],
    )
    assert loss.item() < 1e-8
    loss.backward()
    assert student_feature.grad is not None
    assert torch.isfinite(student_feature.grad).all()


def test_projected_backbone_feature_distillation_loss_backward():
    student_feature = torch.randn(2, 4, 3, 3, requires_grad=True)
    teacher_feature = student_feature.detach() + 0.25 * torch.randn_like(student_feature)
    loss = projected_backbone_feature_distillation_loss(
        [make_projected_src_feature_frame([student_feature])],
        [make_projected_src_feature_frame([teacher_feature])],
    )
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(student_feature.grad).all()
    assert student_feature.grad.abs().sum().item() > 0


class ToyFeatureTeacher(nn.Module):
    def __init__(self, frames):
        super().__init__()
        self.frames = frames
        self.probe = nn.Parameter(torch.ones(()))

    def extract_projected_backbone_features(self, data):
        return self.frames


def test_backbone_feature_distiller_callable():
    student_feature = torch.randn(1, 2, 2, 2, requires_grad=True)
    teacher_feature = student_feature.detach().clone()
    teacher = ToyFeatureTeacher([make_projected_src_feature_frame([teacher_feature])])
    teacher.train()
    distiller = BackboneFeatureDistiller(teacher)
    losses = distiller(
        {BACKBONE_DISTILL_FEATURE_KEY: [make_projected_src_feature_frame([student_feature])]},
        {},
    )
    assert set(losses) == {CR_QAT_BACKBONE_FD_LOSS_KEY}
    assert losses[CR_QAT_BACKBONE_FD_LOSS_KEY].item() < 1e-8
    assert not teacher.training


def _trace(frame, labels, obj_ids, embeddings, anchors=None, sample=-1):
    labels = torch.as_tensor(labels, dtype=torch.long)
    obj_ids = torch.as_tensor(obj_ids, dtype=torch.long)
    if anchors is None:
        anchors = torch.stack([torch.eye(4)[int(label) % 4] for label in labels], dim=0)
    return make_trkd_trace(frame, sample, labels, obj_ids, embeddings, anchors)


def test_trkd_loss_identical_trace_zero():
    embeddings = torch.randn(3, 4, requires_grad=True)
    trace = [_trace(0, [1, 1, 2], [10, 11, 12], embeddings)]
    loss = trkd_relational_distillation_loss(trace, trace)
    assert loss.item() < 1e-8


def test_trkd_loss_group_average_matches_manual():
    student_a = torch.tensor([[0.8, 0.1, 0.0, 0.0]], requires_grad=True)
    teacher_a = torch.tensor([[0.7, 0.2, 0.0, 0.0]])
    student_b = torch.tensor([[0.0, 0.8, 0.2, 0.0], [0.0, 0.2, 0.8, 0.0]], requires_grad=True)
    teacher_b = torch.tensor([[0.0, 0.7, 0.3, 0.0], [0.0, 0.3, 0.7, 0.0]])
    anchor_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    anchor_b = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    student_trace = [
        _trace(0, [1], [10], student_a, anchor_a),
        _trace(0, [2, 2], [20, 21], student_b, anchor_b),
    ]
    teacher_trace = [
        _trace(0, [1], [10], teacher_a, anchor_a),
        _trace(0, [2, 2], [20, 21], teacher_b, anchor_b),
    ]
    loss = trkd_relational_distillation_loss(student_trace, teacher_trace)
    expected = (
        trkd_similarity_matrix_loss(anchor_a[0], student_a, anchor_a[0], teacher_a)
        + trkd_similarity_matrix_loss(anchor_b[0], student_b, anchor_b[0], teacher_b)
    ) / 2
    assert torch.allclose(loss, expected)


def test_trkd_loss_uses_teacher_obj_id_intersection():
    student_embeddings = torch.randn(1, 4, requires_grad=True)
    teacher_embeddings = student_embeddings.detach().clone()
    unmatched_teacher = torch.randn(1, 4) * 100.0
    student_trace = [_trace(0, [1], [10], student_embeddings)]
    teacher_trace = [_trace(0, [1, 1], [10, 99], torch.cat([teacher_embeddings, unmatched_teacher], dim=0))]
    loss = trkd_relational_distillation_loss(student_trace, teacher_trace)
    assert loss.item() < 1e-8


class ToyStageTeacher(nn.Module):
    def __init__(self, feature_frame, trkd_trace):
        super().__init__()
        self.feature_frame = feature_frame
        self.trkd_trace = trkd_trace

    def extract_projected_backbone_features(self, data):
        return [self.feature_frame]

    def forward(self, data):
        return {
            BACKBONE_DISTILL_FEATURE_KEY: [self.feature_frame],
            TRKD_TRACE_KEY: self.trkd_trace,
        }


def test_backbone_feature_distiller_stage_gates_trkd():
    stage = {"value": 1}
    student_feature = torch.randn(1, 2, 2, 2, requires_grad=True)
    teacher_feature = student_feature.detach().clone()
    student_embedding = torch.randn(1, 4, requires_grad=True)
    teacher_embedding = student_embedding.detach().clone()
    feature_frame = make_projected_src_feature_frame([teacher_feature])
    teacher_trace = [_trace(0, [1], [10], teacher_embedding)]
    student_outputs = {
        BACKBONE_DISTILL_FEATURE_KEY: [make_projected_src_feature_frame([student_feature])],
        TRKD_TRACE_KEY: [_trace(0, [1], [10], student_embedding)],
    }
    distiller = BackboneFeatureDistiller(
        ToyStageTeacher(feature_frame, teacher_trace),
        enable_trkd=True,
        stage_getter=lambda: stage["value"],
    )
    stage1_losses = distiller(student_outputs, {})
    assert set(stage1_losses) == {CR_QAT_BACKBONE_FD_LOSS_KEY}
    stage["value"] = 2
    stage2_losses = distiller(student_outputs, {})
    assert set(stage2_losses) == {CR_QAT_BACKBONE_FD_LOSS_KEY, CR_QAT_TRKD_LOSS_KEY}


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
    exp_a1_backbone_modules = {"backbone"}
    exp_a1_backbone_input_proj_modules = {"backbone", "input_proj"}
    exp_a1_modules = {"backbone", "input_proj", "patch2query"}

    assert _partition_quant_module_names("exp_a1_backbone") == exp_a1_backbone_modules
    assert _partition_trainable_param_names("exp_a1_backbone") == _param_names_for_modules(exp_a1_backbone_modules)
    assert _partition_quant_module_names("exp_a1_backbone_input_proj") == exp_a1_backbone_input_proj_modules
    assert _partition_trainable_param_names("exp_a1_backbone_input_proj") == _param_names_for_modules(exp_a1_backbone_input_proj_modules)
    assert _partition_quant_module_names("exp_a1") == exp_a1_modules
    assert _partition_trainable_param_names("exp_a1") == _param_names_for_modules(exp_a1_modules)

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
    assert "transformer.decoder.bbox_embed" not in exp_a_modules
    assert "feature_align" not in exp_a_modules
    assert "track_embed" not in exp_a_modules


def test_first_last_layers_excluded_from_quant_patching():
    model = ToyFirstLastModel()
    controller = maybe_prepare_ovtr_quant_controller(model, mode="ptq", partition="exp_a", range_method="minmax")
    module_names = set(controller.quant_module_names)
    excluded_names = set(controller.excluded_module_names)

    assert "backbone.0.body.conv1" not in module_names
    assert "backbone.0.patch_embed.proj" not in module_names
    assert "transformer.decoder.bbox_embed" not in module_names
    assert "feature_align" not in module_names
    assert "backbone.0.body.layer1" in module_names
    assert "transformer.decoder.layers.0" in module_names
    assert {
        "backbone.0.body.conv1",
        "backbone.0.patch_embed.proj",
        "transformer.decoder.bbox_embed",
        "feature_align",
    }.issubset(excluded_names)


def test_cr_qat_partition_subset():
    stage1_modules = _partition_quant_module_names(CR_QAT_STAGE1_PARTITION)
    stage2_modules = _partition_quant_module_names(CR_QAT_STAGE2_PARTITION)
    assert stage1_modules < stage2_modules
    assert _partition_trainable_param_names(CR_QAT_STAGE1_PARTITION) < _partition_trainable_param_names(CR_QAT_STAGE2_PARTITION)


def test_cr_qat_stage1_gates_final_controller():
    model = ToyPartitionModel()
    controller = maybe_prepare_ovtr_quant_controller(
        model,
        mode="qat",
        partition=CR_QAT_STAGE2_PARTITION,
        range_method="minmax",
    )
    controller.set_active_qat_partition(CR_QAT_STAGE1_PARTITION)
    controller.configure_calibration_partitions(observer_partition=CR_QAT_STAGE1_PARTITION)
    controller.enable_calibration(reset=True)

    stage1_modules = controller.module_names_for_partition(CR_QAT_STAGE1_PARTITION)
    stage2_modules = controller.module_names_for_partition(CR_QAT_STAGE2_PARTITION)
    named_modules = _named_quant_modules(controller)
    assert stage1_modules < stage2_modules
    for name, module in named_modules.items():
        assert _quant_enabled(module) is False
        assert _observer_enabled(module) == (name in stage1_modules)

    _run_linear_partition_modules(model, stage1_modules)
    controller.finalize_calibration()
    controller.enable_qat()
    for name, module in named_modules.items():
        assert _quant_enabled(module) == (name in stage1_modules)
        assert _observer_enabled(module) is False


def test_cr_qat_transition_recalibrates_new_modules():
    model = ToyPartitionModel()
    controller = maybe_prepare_ovtr_quant_controller(
        model,
        mode="qat",
        partition=CR_QAT_STAGE2_PARTITION,
        range_method="minmax",
    )
    stage1_modules = controller.module_names_for_partition(CR_QAT_STAGE1_PARTITION)
    stage2_modules = controller.module_names_for_partition(CR_QAT_STAGE2_PARTITION)
    new_stage2_modules = stage2_modules - stage1_modules

    controller.set_active_qat_partition(CR_QAT_STAGE1_PARTITION)
    controller.configure_calibration_partitions(observer_partition=CR_QAT_STAGE1_PARTITION)
    controller.enable_calibration(reset=True)
    _run_linear_partition_modules(model, stage1_modules)
    controller.finalize_calibration()
    controller.enable_qat()

    controller.configure_calibration_partitions(
        observer_partition=CR_QAT_STAGE2_PARTITION,
        quantized_partition=CR_QAT_STAGE1_PARTITION,
        exclude_partition=CR_QAT_STAGE1_PARTITION,
    )
    controller.enable_calibration(reset=True)
    named_modules = _named_quant_modules(controller)
    for name, module in named_modules.items():
        assert _quant_enabled(module) == (name in stage1_modules)
        assert _observer_enabled(module) == (name in new_stage2_modules)

    _run_linear_partition_modules(model, new_stage2_modules)
    controller.finalize_calibration()
    controller.set_active_qat_partition(CR_QAT_STAGE2_PARTITION)
    controller.clear_calibration_partitions()
    controller.enable_qat()

    for name in new_stage2_modules:
        module = named_modules[name]
        if isinstance(module, nn.Linear):
            assert module._ovtr_quant_input_quantizer.initialized.item()
            assert module._ovtr_quant_output_quantizer.initialized.item()
    for name, module in named_modules.items():
        assert _quant_enabled(module) == (name in stage2_modules)


def test_cr_qat_stage_trainable_selection():
    model = ToyPartitionModel()
    controller = maybe_prepare_ovtr_quant_controller(
        model,
        mode="qat",
        partition=CR_QAT_STAGE2_PARTITION,
        range_method="minmax",
    )
    _apply_stage_trainability(model, controller, CR_QAT_STAGE1_PARTITION)
    stage1_trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert _partition_trainable_param_names(CR_QAT_STAGE1_PARTITION).issubset(stage1_trainable)
    assert not any(name.startswith("transformer.encoder") for name in stage1_trainable if "_ovtr_quant_" not in name)
    assert not any(name.startswith("transformer.decoder.bbox_embed") for name in stage1_trainable)
    assert not any(name.startswith("feature_align") for name in stage1_trainable)
    assert not any(name.startswith("unrelated") for name in stage1_trainable)

    _apply_stage_trainability(model, controller, CR_QAT_STAGE2_PARTITION)
    stage2_trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    normal_param_names = {name for name, _ in model.named_parameters() if not is_quant_trainable_param(name)}
    expected_quant_trainable = {
        name
        for name, _ in model.named_parameters()
        if is_quant_trainable_param(name) and controller.is_quant_param_in_partition(name, CR_QAT_STAGE2_PARTITION)
    }
    stage2_quant_trainable = {name for name in stage2_trainable if is_quant_trainable_param(name)}

    assert stage1_trainable < stage2_trainable
    assert normal_param_names.issubset(stage2_trainable)
    assert expected_quant_trainable == stage2_quant_trainable
    assert "transformer.decoder.bbox_embed.weight" in stage2_trainable
    assert "feature_align.weight" in stage2_trainable
    assert "unrelated.weight" in stage2_trainable
    assert any(name.startswith("track_embed") for name in stage2_trainable)


def test_regular_qat_exp_a1_to_b_full_trainable_selection():
    model = ToyPartitionModel()
    controller = maybe_prepare_ovtr_quant_controller(
        model,
        mode="qat",
        partition=CR_QAT_STAGE2_PARTITION,
        range_method="minmax",
    )
    _apply_stage_trainability(model, controller, CR_QAT_STAGE2_PARTITION)
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    normal_param_names = {name for name, _ in model.named_parameters() if not is_quant_trainable_param(name)}
    expected_quant_trainable = {
        name
        for name, _ in model.named_parameters()
        if is_quant_trainable_param(name) and controller.is_quant_param_in_partition(name, CR_QAT_STAGE2_PARTITION)
    }

    assert normal_param_names.issubset(trainable)
    assert expected_quant_trainable == {name for name in trainable if is_quant_trainable_param(name)}
    assert "transformer.decoder.bbox_embed.weight" in trainable
    assert "feature_align.weight" in trainable
    assert "unrelated.weight" in trainable
    assert "transformer.decoder.bbox_embed" not in set(controller.quant_module_names)
    assert "feature_align" not in set(controller.quant_module_names)


def test_cr_qat_stage2_recalibration_temporarily_disables_trkd_trace():
    model = ToyTraceModel()
    current_stage = 1
    switch_step = 3

    def activate_stage2(global_step):
        nonlocal current_stage
        assert global_step == switch_step
        with temporarily_disable_trkd_trace(model):
            model.run_transition_recalibration()
        current_stage = 2

    def before_step(global_step):
        if global_step >= switch_step and current_stage != 2:
            activate_stage2(global_step)

    before_step(0)
    assert model.return_trkd_trace is True
    assert current_stage == 1

    before_step(switch_step)
    assert current_stage == 2
    assert model.recalibration_states == [False]
    assert model.return_trkd_trace is True
    assert model.sync_states == [False, True]

    before_step(switch_step + 1)
    assert model.recalibration_states == [False]


def test_train_loop_runs_stage2_step_after_recalibration():
    model = ToyStageTrainModel()
    criterion = ToyStageCriterion()
    scheduler = ToyEngineStageScheduler(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data_loader = [
        {"filename": ["stage1.jpg"], "x": torch.ones(1)},
        {"filename": ["stage2.jpg"], "x": torch.ones(1)},
    ]
    initial_probe = model.probe.detach().clone()
    initial_extra = model.stage2_extra.detach().clone()

    with EventStorage(0):
        stats = train_one_epoch_mot(
            model,
            criterion,
            data_loader,
            optimizer,
            torch.device("cpu"),
            epoch=0,
            max_norm=0,
            stage_scheduler=scheduler,
        )

    assert model.recalibration_states == [False]
    assert model.forward_trace == [(1, True, False), (2, True, True)]
    assert stats["cr_qat_stage"] == 2
    assert stats["cr_qat_global_step"] == 1
    assert not torch.allclose(model.probe.detach(), initial_probe)
    assert not torch.allclose(model.stage2_extra.detach(), initial_extra)


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
    test_projected_backbone_feature_distillation_loss_masks_padding()
    test_projected_backbone_feature_distillation_loss_backward()
    test_backbone_feature_distiller_callable()
    test_trkd_loss_identical_trace_zero()
    test_trkd_loss_group_average_matches_manual()
    test_trkd_loss_uses_teacher_obj_id_intersection()
    test_backbone_feature_distiller_stage_gates_trkd()
    test_combined_partition_coverage()
    test_first_last_layers_excluded_from_quant_patching()
    test_cr_qat_partition_subset()
    test_cr_qat_stage1_gates_final_controller()
    test_cr_qat_transition_recalibrates_new_modules()
    test_cr_qat_stage_trainable_selection()
    test_regular_qat_exp_a1_to_b_full_trainable_selection()
    test_cr_qat_stage2_recalibration_temporarily_disables_trkd_trace()
    test_train_loop_runs_stage2_step_after_recalibration()
    test_lowbit_pack_layout_and_reconstruction()
    test_uint4_zero_point_correction_formula()
    print("quant smoke passed")


if __name__ == "__main__":
    main()
