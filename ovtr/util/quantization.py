import copy
import hashlib
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
        "--quant_partition",
        default="exp_a",
        choices=list(SUPPORTED_QUANT_PARTITIONS),
        help="module partition to quantize (supported backends patch Conv2d/Linear/MSDA only)",
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
        default=8,
        type=int,
        help="weight bit width for quantization",
    )
    parser.add_argument(
        "--quant_activation_bits",
        default=6,
        type=int,
        help="activation bit width for quantization",
    )
    parser.add_argument(
        "--quant_attention_bits",
        default=4,
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

    model_ref = unwrap_model(model)
    controller = maybe_prepare_ovtr_quant_controller(
        model_ref,
        mode=args.quant_mode,
        partition=args.quant_partition,
        weight_bits=args.quant_weight_bits,
        activation_bits=args.quant_activation_bits,
        attention_bits=args.quant_attention_bits,
    )
    print(f"[Quant] {controller.summary()}", flush=True)
    return controller


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
