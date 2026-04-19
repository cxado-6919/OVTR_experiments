from typing import Optional

import torch

from datasets.data_prefetcher import data_dict_to_cuda
from models.quant_utils import maybe_prepare_group_a_ptq


def _first_or_default(value, default):
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) > 0 else default
    if value is None:
        return default
    return value


def _configure_tracking_thresholds_for_calibration(model_ref, args) -> None:
    model_ref.track_base.score_thresh = _first_or_default(getattr(args, "score_thresh", None), 0.5)
    model_ref.track_base.filter_score_thresh = _first_or_default(
        getattr(args, "filter_score_thresh", None), 0.5
    )
    model_ref.track_base.miss_tolerance = _first_or_default(getattr(args, "miss_tolerance", None), 5)
    model_ref.ious_thresh = _first_or_default(getattr(args, "ious_thresh", None), 0.5)


def setup_group_a_ptq(model: torch.nn.Module, args) -> Optional[object]:
    if not getattr(args, "group_a_ptq", False):
        return None

    model_ref = model.module if hasattr(model, "module") else model
    controller = maybe_prepare_group_a_ptq(
        model_ref,
        weight_bits=args.group_a_weight_bits,
        activation_bits=args.group_a_activation_bits,
        attention_bits=args.group_a_attention_bits,
    )
    print(f"[Group A PTQ] {controller.summary()}", flush=True)
    return controller


@torch.no_grad()
def calibrate_group_a_on_train_loader(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_batches: int,
    args=None,
) -> None:
    model_ref = model.module if hasattr(model, "module") else model
    controller = getattr(model_ref, "_group_a_ptq_controller", None)
    if controller is None or num_batches <= 0:
        return

    was_training = model.training
    model.eval()
    controller.enable_calibration(reset=True)
    if args is not None:
        _configure_tracking_thresholds_for_calibration(model_ref, args)
    print(f"[Group A PTQ] Calibrating on {num_batches} training batches", flush=True)

    for batch_idx, data_dict in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        data_dict = data_dict_to_cuda(data_dict, device)
        model(data_dict)

    controller.enable_quantization()
    if was_training:
        model.train()
    print("[Group A PTQ] Calibration complete; PTQ-only mode enabled", flush=True)


@torch.no_grad()
def calibrate_group_a_on_eval_loader(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    num_batches: int,
    args=None,
) -> None:
    model_ref = model.module if hasattr(model, "module") else model
    controller = getattr(model_ref, "_group_a_ptq_controller", None)
    if controller is None or num_batches <= 0:
        return

    was_training = model.training
    model.eval()
    controller.enable_calibration(reset=True)
    if args is not None:
        _configure_tracking_thresholds_for_calibration(model_ref, args)
    print(f"[Group A PTQ] Calibrating on {num_batches} eval batches", flush=True)

    for batch_idx, data_dict in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        info = data_dict["info"][0] if isinstance(data_dict["info"], list) else data_dict["info"]
        data_dict = dict(data_dict)
        data_dict.pop("info", None)
        data_dict.pop("file_path", None)
        data_dict = data_dict_to_cuda(data_dict, device)
        model_ref.inference_single_image(data_dict, track_instances=None, frame_id=0, ori_img_size=info[1])

    controller.enable_quantization()
    if was_training:
        model.train()
    print("[Group A PTQ] Calibration complete; PTQ-only mode enabled", flush=True)
