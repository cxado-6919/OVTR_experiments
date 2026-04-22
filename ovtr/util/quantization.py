import copy
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
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
    calib_cfg = None
    calib_source = None
    if hasattr(cfg.data, "calib"):
        calib_cfg = copy.deepcopy(cfg.data.calib)
        calib_source = "held-out LVIS calibration"
    elif hasattr(cfg.data, "test"):
        calib_cfg = copy.deepcopy(cfg.data.test)
        calib_source = "evaluation dataset"
    elif hasattr(cfg.data, "val"):
        calib_cfg = copy.deepcopy(cfg.data.val)
        calib_source = "validation dataset"
    else:
        raise AttributeError(
            "Configuration must provide cfg.data.calib, cfg.data.test, or cfg.data.val for quant calibration."
        )

    calib_ann_file = _cfg_get(calib_cfg, "ann_file")
    if calib_source == "held-out LVIS calibration" and calib_ann_file is not None:
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
        track_instances = None
        prev_file_path = None
        calibration_description = getattr(
            data_loader,
            "quant_calibration_description",
            getattr(data_loader.dataset, "quant_calibration_description", "calibration dataset"),
        )
        print(
            f"[Quant] Calibrating {controller.mode.upper()} on {num_samples} {calibration_description} samples",
            flush=True,
        )
        for data_dict in data_loader:
            sample_dicts = _split_mot_batch(dict(data_dict))
            for sample_data_dict in sample_dicts:
                info = _unwrap_singleton_list(sample_data_dict.pop("info"))
                file_path = _unwrap_singleton_list(sample_data_dict.pop("file_path"))
                frame_id = info[0]
                if frame_id == 0 or file_path != prev_file_path:
                    track_instances = None
                sample_data_dict = data_dict_to_cuda(sample_data_dict, device)
                result = model_ref.inference_single_image(
                    sample_data_dict,
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
                prev_file_path = file_path
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
