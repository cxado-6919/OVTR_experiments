import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import build_dataset
from datasets.data_prefetcher import data_dict_to_cuda
from models.quant_utils import SUPPORTED_QUANT_PARTITIONS, maybe_prepare_ovtr_quant_controller
import util.misc as utils


def add_quant_args(parser) -> None:
    parser.add_argument(
        "--quant_mode",
        default="none",
        choices=["none", "ptq", "qat"],
        help="full-model quantization mode",
    )
    parser.add_argument(
        "--quant_deploy",
        default="none",
        choices=["none", "int_msda"],
        help="deploy-time quantized inference backend; int_msda is eval/inference only",
    )
    parser.add_argument(
        "--quant_partition",
        default="exp_a",
        choices=list(SUPPORTED_QUANT_PARTITIONS),
        help="module partition to quantize (supported backends patch Conv2d/Linear/MSDA only)",
    )
    parser.add_argument(
        "--quant_pipeline",
        default="standard",
        choices=["standard", "legacy"],
        help="quantization preparation pipeline; legacy keeps the previous min-max flow",
    )
    parser.add_argument(
        "--quant_range_method",
        default=None,
        choices=["mse", "minmax"],
        help="range estimator; defaults to mse for standard and minmax for legacy",
    )
    parser.add_argument(
        "--quant_bn_folding",
        dest="quant_bn_folding",
        action="store_true",
        default=None,
        help="fold FrozenBatchNorm2d into preceding Conv2d before quantization",
    )
    parser.add_argument(
        "--quant_no_bn_folding",
        dest="quant_bn_folding",
        action="store_false",
        help="disable quantization BN folding",
    )
    parser.add_argument(
        "--quant_cle",
        dest="quant_cle",
        action="store_true",
        default=None,
        help="apply safe cross-layer equalization before quantization",
    )
    parser.add_argument(
        "--quant_no_cle",
        dest="quant_cle",
        action="store_false",
        help="disable cross-layer equalization",
    )
    parser.add_argument(
        "--quant_adaround",
        dest="quant_adaround",
        action="store_true",
        default=None,
        help="apply AdaRound-style weight rounding before calibration",
    )
    parser.add_argument(
        "--quant_no_adaround",
        dest="quant_adaround",
        action="store_false",
        help="disable AdaRound-style weight rounding",
    )
    parser.add_argument(
        "--quant_adaround_samples",
        default=128,
        type=int,
        help="maximum calibration samples used by AdaRound-style rounding",
    )
    parser.add_argument(
        "--quant_adaround_iters",
        default=1000,
        type=int,
        help="optimization iterations budget for AdaRound-style rounding",
    )
    parser.add_argument(
        "--quant_mse_bins",
        default=2048,
        type=int,
        help="histogram bins budget for MSE range estimation",
    )
    parser.add_argument(
        "--quant_mse_candidates",
        default=80,
        type=int,
        help="number of clipping candidates for MSE range estimation",
    )
    parser.add_argument(
        "--quant_bias_correction",
        default="auto",
        choices=["auto", "on", "off"],
        help="bias correction policy; auto uses it when AdaRound is disabled or unavailable",
    )
    parser.add_argument(
        "--quant_calib_samples",
        default=512,
        type=int,
        help="number of calibration samples to use for min-max calibration",
    )
    parser.add_argument(
        "--quant_calib_sequence_length",
        default=3,
        type=int,
        help="number of deterministic pseudo-video frames generated per LVIS calibration image",
    )
    parser.add_argument(
        "--quant_calib_max_translate",
        default=0.08,
        type=float,
        help="maximum pseudo-sequence translation as an image-size fraction",
    )
    parser.add_argument(
        "--quant_calib_max_rotate",
        default=6.0,
        type=float,
        help="maximum pseudo-sequence rotation in degrees",
    )
    parser.add_argument(
        "--quant_calib_scale_jitter",
        default=0.08,
        type=float,
        help="maximum pseudo-sequence multiplicative scale jitter around 1.0",
    )
    parser.add_argument(
        "--quant_calib_motion_blur",
        default=3,
        type=int,
        help="odd motion-blur kernel size for pseudo-sequence calibration; <=1 disables blur",
    )
    parser.add_argument(
        "--quant_disable_pseudo_sequence_calib",
        action="store_true",
        help="disable pseudo-sequence calibration and use the legacy static-frame calibration path",
    )
    parser.add_argument(
        "--quant_calibration_only",
        action="store_true",
        help="run calibration, save a calibrated checkpoint, and exit",
    )
    parser.add_argument(
        "--quant_weight_bits",
        default=4,
        type=int,
        help="weight bit width for quantization",
    )
    parser.add_argument(
        "--quant_activation_bits",
        default=4,
        type=int,
        help="activation bit width for quantization",
    )
    parser.add_argument(
        "--quant_attention_bits",
        default=8,
        type=int,
        help="attention bit width for quantization",
    )
    parser.add_argument(
        "--quant_use_scheduler",
        action="store_true",
        help="step the learning-rate scheduler during QAT fine-tuning",
    )
    parser.add_argument(
        "--quant_qat_allow_batch",
        action="store_true",
        help=(
            "experimental: for batch_size > 1, disable the QAT-only checkpoint forcing "
            "while keeping fixed 5-frame sampling so the batched MOT path can run when "
            "the config supports it; ignored when batch_size <= 1"
        ),
    )


def _first_or_default(value, default):
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) > 0 else default
    if value is None:
        return default
    return value


def _split_mot_batch(data_dict):
    imgs = data_dict.get("imgs")
    if not isinstance(imgs, list) or len(imgs) == 0 or not isinstance(imgs[0], list):
        return [data_dict]

    batch_size = len(imgs)
    sample_dicts = []
    for sample_idx in range(batch_size):
        sample_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, list):
                if len(value) != batch_size:
                    raise ValueError(
                        f"Expected batched field '{key}' to have length {batch_size}, got {len(value)}"
                    )
                sample_dict[key] = value[sample_idx]
            else:
                sample_dict[key] = value
        sample_dicts.append(sample_dict)
    return sample_dicts


def _unwrap_singleton_list(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _stable_unit_values(seed: int, key: str, count: int):
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    values = []
    offset = 0
    while len(values) < count:
        if offset + 4 > len(digest):
            digest = hashlib.sha256(digest).digest()
            offset = 0
        raw = int.from_bytes(digest[offset:offset + 4], byteorder="little", signed=False)
        values.append((raw / float(2**32 - 1)) * 2.0 - 1.0)
        offset += 4
    return values


def _pseudo_sequence_enabled(args) -> bool:
    return args is not None and not getattr(args, "quant_disable_pseudo_sequence_calib", False)


def _pseudo_sequence_length(args) -> int:
    if not _pseudo_sequence_enabled(args):
        return 1
    return max(1, int(getattr(args, "quant_calib_sequence_length", 3)))


def _clone_sample_for_pseudo_frame(sample_data_dict, image: torch.Tensor):
    pseudo_data = dict(sample_data_dict)
    pseudo_data["imgs"] = [image]
    return pseudo_data


def _apply_motion_blur(image: torch.Tensor, kernel_size: int, horizontal: bool) -> torch.Tensor:
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    channels = image.shape[0]
    kernel = image.new_zeros((channels, 1, kernel_size, kernel_size))
    if horizontal:
        kernel[:, 0, kernel_size // 2, :] = 1.0 / float(kernel_size)
    else:
        kernel[:, 0, :, kernel_size // 2] = 1.0 / float(kernel_size)
    return F.conv2d(
        image.unsqueeze(0),
        kernel,
        padding=kernel_size // 2,
        groups=channels,
    ).squeeze(0)


def _augment_pseudo_calibration_image(
    image: torch.Tensor,
    *,
    frame_index: int,
    sequence_length: int,
    file_path: str,
    args,
) -> torch.Tensor:
    if frame_index == 0 or sequence_length <= 1:
        return image.clone()

    alpha = float(frame_index) / float(max(sequence_length - 1, 1))
    seed = int(getattr(args, "seed", 0))
    tx_sign, ty_sign, rot_sign, scale_sign, blur_sign = _stable_unit_values(seed, file_path, 5)
    translate = float(getattr(args, "quant_calib_max_translate", 0.08))
    max_rotate = float(getattr(args, "quant_calib_max_rotate", 6.0))
    scale_jitter = float(getattr(args, "quant_calib_scale_jitter", 0.08))

    tx = 2.0 * translate * tx_sign * alpha
    ty = 2.0 * translate * ty_sign * alpha
    rotate = math.radians(max_rotate * rot_sign * alpha)
    scale = max(0.1, 1.0 + scale_jitter * scale_sign * alpha)
    cos_r = math.cos(rotate) * scale
    sin_r = math.sin(rotate) * scale
    theta = image.new_tensor([[cos_r, -sin_r, tx], [sin_r, cos_r, ty]]).unsqueeze(0)
    grid = F.affine_grid(
        theta,
        size=(1, image.shape[0], image.shape[1], image.shape[2]),
        align_corners=False,
    )
    augmented = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(0)

    blur_kernel = int(getattr(args, "quant_calib_motion_blur", 3))
    if blur_kernel > 1 and frame_index > 0:
        augmented = _apply_motion_blur(augmented, blur_kernel, horizontal=blur_sign >= 0)
    return augmented


def _iter_pseudo_calibration_frames(sample_data_dict, file_path: str, args):
    sequence_length = _pseudo_sequence_length(args)
    base_image = sample_data_dict["imgs"][0]
    for frame_index in range(sequence_length):
        image = _augment_pseudo_calibration_image(
            base_image,
            frame_index=frame_index,
            sequence_length=sequence_length,
            file_path=file_path,
            args=args,
        )
        yield frame_index, _clone_sample_for_pseudo_frame(sample_data_dict, image)


def _format_cfg_path(path_value) -> str:
    if path_value is None:
        return "N/A"
    if isinstance(path_value, (list, tuple)):
        return ", ".join(str(item) for item in path_value)
    return str(path_value)


def resolve_quant_args(args) -> None:
    if args is None:
        return
    pipeline = getattr(args, "quant_pipeline", "standard")
    if getattr(args, "quant_range_method", None) is None:
        args.quant_range_method = "mse" if pipeline == "standard" else "minmax"
    if getattr(args, "quant_bn_folding", None) is None:
        args.quant_bn_folding = pipeline == "standard"
    if getattr(args, "quant_cle", None) is None:
        args.quant_cle = pipeline == "standard"
    if getattr(args, "quant_adaround", None) is None:
        args.quant_adaround = pipeline == "standard"
    if not hasattr(args, "quant_mse_candidates"):
        args.quant_mse_candidates = 80
    if not hasattr(args, "quant_mse_bins"):
        args.quant_mse_bins = 2048
    if not hasattr(args, "quant_adaround_iters"):
        args.quant_adaround_iters = 1000
    if not hasattr(args, "quant_adaround_samples"):
        args.quant_adaround_samples = 128
    if not hasattr(args, "quant_bias_correction"):
        args.quant_bias_correction = "auto"


@torch.no_grad()
def _fold_frozen_batch_norms(model_ref: torch.nn.Module) -> int:
    folded = 0
    eps = 1e-5
    for parent in model_ref.modules():
        children = list(parent.named_children())
        for (_, conv), (_, bn) in zip(children, children[1:]):
            if not isinstance(conv, torch.nn.Conv2d):
                continue
            if bn.__class__.__name__ != "FrozenBatchNorm2d":
                continue
            weight = bn.weight.to(device=conv.weight.device, dtype=conv.weight.dtype)
            bias = bn.bias.to(device=conv.weight.device, dtype=conv.weight.dtype)
            running_mean = bn.running_mean.to(device=conv.weight.device, dtype=conv.weight.dtype)
            running_var = bn.running_var.to(device=conv.weight.device, dtype=conv.weight.dtype)
            scale = weight * (running_var + eps).rsqrt()
            folded_bias = bias - running_mean * scale
            conv.weight.mul_(scale.reshape(-1, 1, 1, 1))
            if conv.bias is None:
                conv.bias = torch.nn.Parameter(folded_bias.clone(), requires_grad=False)
            else:
                conv.bias.mul_(scale).add_(folded_bias)
            bn.weight.fill_(1.0)
            bn.bias.zero_()
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0 - eps)
            folded += 1
    return folded


@torch.no_grad()
def _equalize_linear_pair(first: torch.nn.Linear, second: torch.nn.Linear) -> bool:
    if first.out_features != second.in_features:
        return False
    first_range = first.weight.detach().abs().amax(dim=1).clamp(min=1e-8)
    second_range = second.weight.detach().abs().amax(dim=0).clamp(min=1e-8)
    scale = torch.sqrt(second_range / first_range).clamp(0.1, 10.0)
    first.weight.mul_(scale.reshape(-1, 1))
    if first.bias is not None:
        first.bias.mul_(scale)
    second.weight.div_(scale.reshape(1, -1))
    return True


@torch.no_grad()
def _equalize_conv_pair(first: torch.nn.Conv2d, second: torch.nn.Conv2d) -> bool:
    if first.out_channels != second.in_channels or first.groups != 1 or second.groups != 1:
        return False
    first_range = first.weight.detach().abs().amax(dim=(1, 2, 3)).clamp(min=1e-8)
    second_range = second.weight.detach().abs().amax(dim=(0, 2, 3)).clamp(min=1e-8)
    scale = torch.sqrt(second_range / first_range).clamp(0.1, 10.0)
    first.weight.mul_(scale.reshape(-1, 1, 1, 1))
    if first.bias is not None:
        first.bias.mul_(scale)
    second.weight.div_(scale.reshape(1, -1, 1, 1))
    return True


def _apply_safe_cross_layer_equalization(model_ref: torch.nn.Module) -> int:
    equalized = 0
    for parent in model_ref.modules():
        if not isinstance(parent, torch.nn.Sequential):
            continue
        children = list(parent.children())
        for first, second in zip(children, children[1:]):
            if isinstance(first, torch.nn.Linear) and isinstance(second, torch.nn.Linear):
                equalized += int(_equalize_linear_pair(first, second))
            elif isinstance(first, torch.nn.Conv2d) and isinstance(second, torch.nn.Conv2d):
                equalized += int(_equalize_conv_pair(first, second))
    return equalized


def _should_run_bias_correction(args) -> bool:
    bias_policy = getattr(args, "quant_bias_correction", "auto")
    return bias_policy == "on" or (bias_policy == "auto" and not getattr(args, "quant_adaround", False))


def prepare_quant_model_for_calibration(model: torch.nn.Module, args, *, quant_state_loaded: bool = False) -> None:
    if getattr(args, "quant_mode", "none") == "none" or quant_state_loaded:
        return
    resolve_quant_args(args)
    if getattr(args, "quant_pipeline", "standard") != "standard":
        return

    model_ref = unwrap_model(model)
    if getattr(args, "quant_bn_folding", False):
        folded = _fold_frozen_batch_norms(model_ref)
        if utils.is_main_process() and folded:
            print(f"[Quant] Folded {folded} FrozenBatchNorm2d modules into Conv2d", flush=True)
    if getattr(args, "quant_cle", False):
        equalized = _apply_safe_cross_layer_equalization(model_ref)
        if utils.is_main_process() and equalized:
            print(f"[Quant] Applied cross-layer equalization to {equalized} adjacent module pairs", flush=True)

    controller = getattr(model_ref, "_ovtr_quant_controller", None)
    if controller is not None and getattr(args, "quant_adaround", False):
        updated = controller.apply_adaround(
            num_iters=getattr(args, "quant_adaround_iters", 1000),
            num_samples=getattr(args, "quant_adaround_samples", 128),
        )
        if utils.is_main_process() and updated:
            print(f"[Quant] Applied AdaRound-style weight rounding to {updated} modules", flush=True)

    if controller is not None and _should_run_bias_correction(args):
        corrected = controller.apply_bias_correction()
        if utils.is_main_process() and corrected:
            print(f"[Quant] Materialized bias terms for {corrected} modules", flush=True)


def _configure_tracking_thresholds_for_calibration(model_ref, args) -> None:
    model_ref.track_base.score_thresh = _first_or_default(getattr(args, "score_thresh", None), 0.5)
    model_ref.track_base.filter_score_thresh = _first_or_default(
        getattr(args, "filter_score_thresh", None), 0.5
    )
    model_ref.track_base.miss_tolerance = _first_or_default(getattr(args, "miss_tolerance", None), 5)
    model_ref.ious_thresh = _first_or_default(getattr(args, "ious_thresh", None), 0.5)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def setup_quant_controller(model: torch.nn.Module, args) -> Optional[object]:
    if getattr(args, "quant_mode", "none") == "none":
        return None

    resolve_quant_args(args)
    model_ref = unwrap_model(model)
    controller = maybe_prepare_ovtr_quant_controller(
        model_ref,
        mode=args.quant_mode,
        partition=args.quant_partition,
        weight_bits=args.quant_weight_bits,
        activation_bits=args.quant_activation_bits,
        attention_bits=args.quant_attention_bits,
        range_method=args.quant_range_method,
        mse_candidates=args.quant_mse_candidates,
        mse_bins=args.quant_mse_bins,
    )
    print(f"[Quant] {controller.summary()}", flush=True)
    return controller


def _quant_module_name_list(modules) -> list:
    return [getattr(module, "_ovtr_quant_name", "") for module in modules]


def _int_msda_dispatch_counts(controller) -> dict:
    counts = {}
    if controller is None:
        return counts
    for module in getattr(controller, "attention_modules", []):
        name = getattr(module, "_ovtr_quant_name", "")
        counts[name] = int(getattr(module, "_ovtr_int_msda_dispatch_count", 0))
    return counts


def build_quant_manifest(model: torch.nn.Module, args) -> dict:
    model_ref = unwrap_model(model)
    controller = getattr(model_ref, "_ovtr_quant_controller", None)
    state_keys = list(model_ref.state_dict().keys())
    quant_state_keys = [key for key in state_keys if "_ovtr_quant_" in key]
    int_export_keys = [
        key
        for key in state_keys
        if "_ovtr_quant_int4_" in key or key.endswith("_ovtr_quant_bias_fp32")
    ]
    original_trainable = []
    quant_trainable = []
    for name, param in model_ref.named_parameters():
        if not param.requires_grad:
            continue
        if "_ovtr_quant_" in name:
            quant_trainable.append(name)
        else:
            original_trainable.append(name)

    manifest = {
        "quant_mode": getattr(args, "quant_mode", "none"),
        "quant_deploy": getattr(args, "quant_deploy", "none"),
        "partition": getattr(args, "quant_partition", None),
        "controller_summary": controller.summary() if controller is not None else None,
        "weight_bits": getattr(args, "quant_weight_bits", None),
        "activation_bits": getattr(args, "quant_activation_bits", None),
        "attention_bits": getattr(args, "quant_attention_bits", None),
        "qscheme": {
            "weight": "4-bit symmetric per-output-channel",
            "activation": "4-bit asymmetric per-tensor",
            "attention": "8-bit asymmetric per-head(axis=num_heads)",
        },
        "quant_module_names": list(getattr(controller, "quant_module_names", [])) if controller is not None else [],
        "attention_modules": _quant_module_name_list(getattr(controller, "attention_modules", [])) if controller is not None else [],
        "original_trainable_params": original_trainable,
        "quant_trainable_params": quant_trainable,
        "quant_enabled": bool(getattr(controller, "quant_enabled", False)) if controller is not None else False,
        "observer_enabled": bool(getattr(controller, "observer_enabled", False)) if controller is not None else False,
        "runtime_state": getattr(controller, "runtime_state", None) if controller is not None else None,
        "quant_state_key_count": len(quant_state_keys),
        "int_msda_export_key_count": len(int_export_keys),
        "int_msda_dispatch_counter": _int_msda_dispatch_counts(controller),
    }
    return manifest


def write_quant_manifest(model: torch.nn.Module, args, *, filename: str = "quant_manifest.json") -> Optional[dict]:
    if not getattr(args, "output_dir", None):
        return None
    manifest = build_quant_manifest(model, args)
    if utils.is_main_process():
        output_path = Path(args.output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def enable_int_msda_deploy(model: torch.nn.Module, args) -> None:
    if getattr(args, "quant_deploy", "none") != "int_msda":
        return
    model_ref = unwrap_model(model)
    if model_ref.training:
        raise RuntimeError("--quant_deploy int_msda is eval/inference only.")
    if not torch.cuda.is_available() or not str(getattr(args, "device", "")).startswith("cuda"):
        raise RuntimeError("--quant_deploy int_msda requires CUDA.")
    try:
        from models.ops import HAS_MSDA_EXT, MSDA
    except ImportError as exc:
        raise RuntimeError("Missing CUDA MSDA extension for --quant_deploy int_msda.") from exc
    if not HAS_MSDA_EXT or MSDA is None:
        raise RuntimeError("Missing CUDA MSDA extension for --quant_deploy int_msda.")
    if not (
        hasattr(MSDA, "ms_deform_attn_lowbit_forward")
        or hasattr(MSDA, "ms_deform_attn_int_forward")
    ):
        raise RuntimeError(
            "CUDA extension does not expose ms_deform_attn_lowbit_forward/ms_deform_attn_int_forward."
        )
    if not hasattr(MSDA, "lowbit_linear_forward"):
        raise RuntimeError("CUDA extension does not expose lowbit_linear_forward.")

    controller = getattr(model_ref, "_ovtr_quant_controller", None)
    if controller is None:
        raise RuntimeError("--quant_deploy int_msda requires OVTRQuantController.")
    prepared = controller.prepare_int_msda_deploy()
    if utils.is_main_process():
        print(f"[Quant] Prepared {prepared} MSDeformAttn modules for int_msda deploy", flush=True)


def _cfg_get(cfg_obj, key, default=None):
    if hasattr(cfg_obj, key):
        return getattr(cfg_obj, key)
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    return default


def _cfg_set(cfg_obj, key, value) -> None:
    if hasattr(cfg_obj, key):
        setattr(cfg_obj, key, value)
    else:
        cfg_obj[key] = value


def _resolve_calibration_config(cfg):
    if not hasattr(cfg.data, "calib"):
        raise AttributeError(
            "Configuration must provide cfg.data.calib for quant calibration."
        )

    calib_cfg = copy.deepcopy(cfg.data.calib)
    calib_source = "held-out LVIS calibration"
    calib_ann_file = _cfg_get(calib_cfg, "ann_file")
    if calib_ann_file is not None:
        ann_files = calib_ann_file if isinstance(calib_ann_file, (list, tuple)) else [calib_ann_file]
        missing_files = [ann_file for ann_file in ann_files if not Path(ann_file).exists()]
        if missing_files:
            missing_list = ", ".join(missing_files)
            raise FileNotFoundError(
                "[Quant] Held-out calibration split not found: "
                f"{missing_list}. Generate it with "
                "'python ../process/create_lvis_calibration_split.py' from the ovtr directory."
            )

    _cfg_set(calib_cfg, "test_mode", True)
    return calib_cfg, calib_source


def build_quant_calibration_loader(args, cfg):
    calib_cfg, calib_source = _resolve_calibration_config(cfg)
    dataset_val = build_dataset(image_set="val", args=args, cfg=calib_cfg)
    dataset_val.quant_calibration_description = calib_source
    dataset_val.quant_calibration_ann_file = _cfg_get(calib_cfg, "ann_file")
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader = DataLoader(
        dataset_val,
        batch_size=1,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.mot_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data_loader.quant_calibration_description = calib_source
    data_loader.quant_calibration_ann_file = getattr(dataset_val, "quant_calibration_ann_file", None)
    return data_loader


def _broadcast_quant_state(controller) -> None:
    if not utils.is_dist_avail_and_initialized():
        return
    state = controller.export_state_dict() if utils.is_main_process() else None
    payload = [state]
    dist.broadcast_object_list(payload, src=0)
    controller.load_exported_state_dict(payload[0])


def _register_bias_correction_hooks(controller):
    stats = {}
    handles = []

    def _hook(module, inputs, output):
        if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
            return
        x = inputs[0].detach()
        quantized_out = output.detach()
        if isinstance(module, torch.nn.Conv2d):
            fp_out = F.conv2d(
                x,
                module.weight,
                module.bias,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            )
            if fp_out.shape != quantized_out.shape or fp_out.ndim < 2:
                return
            reduce_dims = (0,) + tuple(range(2, fp_out.ndim))
            error_sum = (fp_out - quantized_out).sum(dim=reduce_dims)
            count = float((fp_out.numel() // max(fp_out.shape[1], 1)))
        elif isinstance(module, torch.nn.Linear):
            fp_out = F.linear(x, module.weight, module.bias)
            if fp_out.shape != quantized_out.shape or fp_out.ndim == 0:
                return
            reduce_dims = tuple(range(fp_out.ndim - 1))
            error_sum = (fp_out - quantized_out).sum(dim=reduce_dims)
            count = float((fp_out.numel() // max(fp_out.shape[-1], 1)))
        else:
            return

        if not torch.isfinite(error_sum).all() or count <= 0:
            return
        if module not in stats:
            stats[module] = [
                torch.zeros_like(error_sum),
                torch.tensor(0.0, device=error_sum.device, dtype=error_sum.dtype),
            ]
        stats[module][0].add_(error_sum)
        stats[module][1].add_(count)

    for module in controller.quant_modules:
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(_hook))
    return stats, handles


def _finalize_bias_correction_stats(stats) -> dict:
    corrections = {}
    for module, (error_sum, count) in stats.items():
        if count.item() <= 0:
            continue
        name = getattr(module, "_ovtr_quant_name", None)
        if not name:
            continue
        corrections[name] = (error_sum / count).detach().cpu()
    return corrections


def _broadcast_bias_corrections(corrections: dict) -> dict:
    if not utils.is_dist_avail_and_initialized():
        return corrections
    payload = [corrections if utils.is_main_process() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0] or {}


@torch.no_grad()
def calibrate_quant_controller_on_val_loader(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_samples: int,
    args=None,
) -> int:
    model_ref = unwrap_model(model)
    controller = getattr(model_ref, "_ovtr_quant_controller", None)
    if controller is None or num_samples <= 0:
        return 0

    was_training = model.training
    model.eval()
    controller.enable_calibration(reset=True)
    if args is not None:
        _configure_tracking_thresholds_for_calibration(model_ref, args)

    calibrated = 0
    bias_stats = {}
    bias_handles = []
    if args is not None and _should_run_bias_correction(args):
        bias_stats, bias_handles = _register_bias_correction_hooks(controller)

    try:
        if utils.is_main_process():
            calibration_description = getattr(
                data_loader,
                "quant_calibration_description",
                getattr(data_loader.dataset, "quant_calibration_description", "calibration dataset"),
            )
            calibration_ann_file = getattr(data_loader, "quant_calibration_ann_file", None)
            sequence_length = _pseudo_sequence_length(args)
            total_frames = num_samples * sequence_length
            if sequence_length > 1:
                print(
                    f"[Quant] Calibrating {controller.mode.upper()} on {num_samples} "
                    f"{calibration_description} base images x {sequence_length} pseudo frames "
                    f"({total_frames} observer frames)",
                    flush=True,
                )
            else:
                print(
                    f"[Quant] Calibrating {controller.mode.upper()} on {num_samples} "
                    f"{calibration_description} static frames",
                    flush=True,
                )
            print(
                f"[Quant] Calibration annotation source: "
                f"{_format_cfg_path(calibration_ann_file)}",
                flush=True,
            )

            for data_dict in data_loader:
                sample_dicts = _split_mot_batch(dict(data_dict))
                for sample_data_dict in sample_dicts:
                    info = _unwrap_singleton_list(sample_data_dict.pop("info"))
                    file_path = _unwrap_singleton_list(sample_data_dict.pop("file_path"))
                    sample_data_dict = data_dict_to_cuda(sample_data_dict, device)
                    track_instances = None
                    for frame_id, pseudo_data_dict in _iter_pseudo_calibration_frames(
                        sample_data_dict,
                        file_path,
                        args,
                    ):
                        result = model_ref.inference_single_image(
                            pseudo_data_dict,
                            track_instances=track_instances,
                            frame_id=frame_id,
                            ori_img_size=info[1],
                        )
                        track_instances = result["track_instances"]
                        if track_instances is not None:
                            if track_instances.has("boxes"):
                                track_instances.remove("boxes")
                            if track_instances.has("labels"):
                                track_instances.remove("labels")
                    calibrated += 1
                    if calibrated >= num_samples:
                        break
                if calibrated >= num_samples:
                    break
    finally:
        for handle in bias_handles:
            handle.remove()

    if args is not None and _should_run_bias_correction(args):
        corrections = _finalize_bias_correction_stats(bias_stats) if utils.is_main_process() else {}
        corrections = _broadcast_bias_corrections(corrections)
        corrected = controller.apply_bias_corrections(corrections)
        if utils.is_main_process() and corrected:
            print(f"[Quant] Applied bias correction to {corrected} modules", flush=True)

    controller.finalize_calibration()
    _broadcast_quant_state(controller)
    if was_training:
        model.train()
    return calibrated


def enable_loaded_quantization(model: torch.nn.Module, *, require_state: bool = False) -> bool:
    model_ref = unwrap_model(model)
    controller = getattr(model_ref, "_ovtr_quant_controller", None)
    if controller is None:
        return False

    if not controller.has_serialized_quant_state():
        if require_state:
            raise ValueError(
                f"Requested {controller.mode} runtime, but the loaded checkpoint does not contain quant state."
            )
        return False

    if controller.mode == "ptq":
        controller.enable_quantization()
    else:
        controller.enable_qat()
    print(f"[Quant] Activated loaded {controller.mode.upper()} state", flush=True)
    return True
