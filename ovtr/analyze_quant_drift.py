import argparse
import copy
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import util.misc as utils
from datasets import build_dataset
from datasets.data_prefetcher import data_dict_to_cuda
from main import get_args_parser
from models import build_model
from util.list_LVIS import CLASSES
from util.quant_drift_analysis import (
    ReferenceTrace,
    RunResult,
    build_teacher_forced_input,
    compare_frame_to_reference,
    run_inference_frame,
    sequence_key_from_file_path,
    snapshot_frame_state,
    strip_feedback_only_fields,
    summarize_run_metrics,
    sync_timing_device,
    write_metric_csvs,
    write_metrics_summary,
    write_plots,
)
from util.quantization import (
    build_quant_calibration_loader,
    calibrate_quant_controller_on_val_loader,
    enable_loaded_quantization,
    prepare_quant_model_for_calibration,
    setup_quant_controller,
)
from util.slconfig import SLConfig
from util.tool import load_model


def add_analysis_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--analysis_fp32_pretrain",
        default=None,
        help="FP32 baseline checkpoint. Defaults to --pretrain when omitted.",
    )
    parser.add_argument(
        "--analysis_output_dir",
        default=None,
        help="Directory for drift CSV/JSON/PNG outputs.",
    )
    parser.add_argument(
        "--analysis_max_frames",
        default=0,
        type=int,
        help="Maximum frames to process. 0 means the full validation split.",
    )
    parser.add_argument(
        "--analysis_iou_divergence_thresh",
        default=0.5,
        type=float,
        help="IoU threshold below which matched tracks are marked as low_iou.",
    )
    parser.add_argument(
        "--analysis_plot_max_age",
        default=0,
        type=int,
        help="Maximum track age shown in age plots. 0 means no limit.",
    )
    return parser


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


def _normalize_threshold_list(value, default, length=7):
    if value is None or len(value) == 0:
        return [default for _ in range(length)]
    value = list(value)
    if len(value) == 1:
        return value * length
    if len(value) < length:
        value.extend([value[-1]] * (length - len(value)))
    return value


def normalize_analysis_args(args) -> None:
    if args.batch_size != 1:
        raise ValueError("Quant drift analysis requires --batch_size 1.")
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        raise ValueError("Quant drift analysis is single-process only; run with python, not torchrun.")
    if args.pretrained is None:
        raise ValueError("--pretrain/--pretrained is required for the quant drift target checkpoint.")

    args.distributed = False
    args.gpu = 0
    args.score_thresh = _normalize_threshold_list(args.score_thresh, 0.5)
    args.filter_score_thresh = _normalize_threshold_list(args.filter_score_thresh, 0.5)
    args.ious_thresh = _normalize_threshold_list(args.ious_thresh, 0.3)
    args.miss_tolerance = _normalize_threshold_list(args.miss_tolerance, 5)
    if args.analysis_output_dir is None:
        args.analysis_output_dir = os.path.join(
            args.output_dir or "./results", f"quant_drift_{args.quant_mode}_{args.quant_partition}"
        )
    if args.analysis_fp32_pretrain is None:
        args.analysis_fp32_pretrain = args.pretrained


def build_validation_loader(args, cfg):
    cfg.data.test.test_mode = True
    dataset_val = build_dataset(image_set="val", args=args, cfg=cfg.data.test)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.mot_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return dataset_val, data_loader_val


def build_loaded_model(args, cfg, device: torch.device, *, checkpoint_path: str, quant_mode: str):
    run_args = copy.deepcopy(args)
    run_args.pretrained = checkpoint_path
    run_args.quant_mode = quant_mode

    model, _ = build_model(run_args, cfg)
    quant_controller = setup_quant_controller(model, run_args)
    model = load_model(model, checkpoint_path)
    model.eval()
    model = model.to(device)

    quant_state_loaded = False
    if quant_controller is not None:
        quant_state_loaded = enable_loaded_quantization(model, require_state=False)
        if not quant_state_loaded:
            prepare_quant_model_for_calibration(
                model,
                run_args,
                quant_state_loaded=quant_state_loaded,
            )

    if quant_mode == "ptq":
        if quant_controller is None:
            raise ValueError("PTQ requested but quant controller was not initialized.")
        if not quant_state_loaded:
            data_loader_calib = build_quant_calibration_loader(run_args, cfg)
            calibrated = calibrate_quant_controller_on_val_loader(
                model,
                data_loader_calib,
                device,
                run_args.quant_calib_samples,
                args=run_args,
            )
            quant_controller.enable_quantization()
            print(f"[QuantDrift] PTQ calibration complete on {calibrated} samples", flush=True)
    elif quant_mode == "qat" and quant_controller is not None and not quant_state_loaded:
        raise ValueError("QAT drift analysis expects a checkpoint that already contains learned quant state.")

    return model


def run_analysis_pass(
    *,
    run_mode: str,
    model,
    data_loader,
    device: torch.device,
    args,
    fp32_trace: Optional[ReferenceTrace] = None,
    teacher_forced: bool = False,
) -> RunResult:
    if hasattr(model, "clear"):
        model.clear()

    trace = ReferenceTrace() if run_mode == "fp32_free" else None
    track_results = []
    track_metric_rows = []
    frame_metric_rows = []
    divergence_rows = []
    first_divergences = {}
    track_ages = {}
    track_instances = None
    previous_sequence = None
    previous_fp32_frame = None
    total_detect_time = 0.0
    processed_frames = 0

    progress = tqdm(data_loader, desc=run_mode)
    with torch.no_grad():
        for data_dict in progress:
            sample_dicts = _split_mot_batch(dict(data_dict))
            for sample_data_dict in sample_dicts:
                info = _unwrap_singleton_list(sample_data_dict.pop("info"))
                file_path = _unwrap_singleton_list(sample_data_dict.pop("file_path"))
                frame_id = int(info[0])
                sequence_key = sequence_key_from_file_path(file_path)
                sequence_reset = frame_id == 0 or sequence_key != previous_sequence

                if sequence_reset:
                    track_instances = None
                    previous_fp32_frame = None
                    if hasattr(model, "clear"):
                        model.clear()

                sample_data_dict = data_dict_to_cuda(sample_data_dict, device=device)

                if teacher_forced:
                    if previous_fp32_frame is None:
                        track_instances = None
                    else:
                        track_instances = build_teacher_forced_input(model, previous_fp32_frame)
                        model.track_base.max_obj_id = previous_fp32_frame.max_obj_id

                sync_timing_device(device)
                start_time = time.perf_counter()
                next_track_instances, frame_track_results, score_threshold, max_obj_id = run_inference_frame(
                    model=model,
                    data=sample_data_dict,
                    track_instances=track_instances,
                    info=info,
                    file_path=file_path,
                    num_classes=len(CLASSES),
                    args=args,
                )
                sync_timing_device(device)
                total_detect_time += time.perf_counter() - start_time
                processed_frames += 1
                track_results.append(frame_track_results)

                frame_trace = snapshot_frame_state(
                    next_track_instances,
                    file_path=file_path,
                    frame_id=frame_id,
                    score_threshold=score_threshold,
                    track_ages=track_ages,
                    update_ages=run_mode == "fp32_free",
                    max_obj_id=max_obj_id,
                )

                if run_mode == "fp32_free":
                    trace.frames[(file_path, frame_id)] = frame_trace
                else:
                    fp32_frame = fp32_trace.get(file_path, frame_id) if fp32_trace is not None else None
                    rows, frame_row, new_divergences = compare_frame_to_reference(
                        run_mode=run_mode,
                        fp32_frame=fp32_frame,
                        quant_frame=frame_trace,
                        iou_divergence_thresh=args.analysis_iou_divergence_thresh,
                        first_divergences=first_divergences,
                    )
                    track_metric_rows.extend(rows)
                    frame_metric_rows.append(frame_row)
                    divergence_rows.extend(new_divergences)

                track_instances = strip_feedback_only_fields(next_track_instances)
                if fp32_trace is not None:
                    previous_fp32_frame = fp32_trace.get(file_path, frame_id)
                previous_sequence = sequence_key

                if args.analysis_max_frames > 0 and processed_frames >= args.analysis_max_frames:
                    break
            if args.analysis_max_frames > 0 and processed_frames >= args.analysis_max_frames:
                break

    return RunResult(
        run_mode=run_mode,
        track_results=track_results,
        processed_frames=processed_frames,
        total_detect_time=total_detect_time,
        trace=trace,
        track_metric_rows=track_metric_rows,
        frame_metric_rows=frame_metric_rows,
        divergence_rows=divergence_rows,
    )


def main(args) -> None:
    normalize_analysis_args(args)
    output_dir = Path(args.analysis_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg = SLConfig.fromfile(args.config_file)
    cfg.device = args.device
    dataset_val, data_loader_val = build_validation_loader(args, cfg)
    device = torch.device(args.device)

    print("[QuantDrift] HOTA is not implemented in this repo; summary will include TETA, IDF1, and MOTA.")
    print("[QuantDrift] Running fp32_free pass", flush=True)
    fp32_model = build_loaded_model(
        args,
        cfg,
        device,
        checkpoint_path=args.analysis_fp32_pretrain,
        quant_mode="none",
    )
    fp32_result = run_analysis_pass(
        run_mode="fp32_free",
        model=fp32_model,
        data_loader=data_loader_val,
        device=device,
        args=args,
    )
    fp32_trace = fp32_result.trace
    del fp32_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[QuantDrift] Running quant_free pass", flush=True)
    quant_model = build_loaded_model(
        args,
        cfg,
        device,
        checkpoint_path=args.pretrained,
        quant_mode=args.quant_mode,
    )
    quant_free_result = run_analysis_pass(
        run_mode="quant_free",
        model=quant_model,
        data_loader=data_loader_val,
        device=device,
        args=args,
        fp32_trace=fp32_trace,
    )

    print("[QuantDrift] Running quant_teacher_forced pass", flush=True)
    quant_teacher_result = run_analysis_pass(
        run_mode="quant_teacher_forced",
        model=quant_model,
        data_loader=data_loader_val,
        device=device,
        args=args,
        fp32_trace=fp32_trace,
        teacher_forced=True,
    )

    track_rows = quant_free_result.track_metric_rows + quant_teacher_result.track_metric_rows
    frame_rows = quant_free_result.frame_metric_rows + quant_teacher_result.frame_metric_rows
    divergence_rows = quant_free_result.divergence_rows + quant_teacher_result.divergence_rows
    write_metric_csvs(
        track_rows=track_rows,
        frame_rows=frame_rows,
        divergence_rows=divergence_rows,
        output_dir=output_dir,
    )

    id_nproc = max(1, min(int(args.num_workers), 4))
    summaries = {
        "fp32_free": summarize_run_metrics(
            run_result=fp32_result,
            dataset=dataset_val,
            output_dir=output_dir,
            metric=args.eval,
            id_nproc=id_nproc,
        ),
        "quant_free": summarize_run_metrics(
            run_result=quant_free_result,
            dataset=dataset_val,
            output_dir=output_dir,
            metric=args.eval,
            id_nproc=id_nproc,
        ),
        "quant_teacher_forced": summarize_run_metrics(
            run_result=quant_teacher_result,
            dataset=dataset_val,
            output_dir=output_dir,
            metric=args.eval,
            id_nproc=id_nproc,
        ),
    }
    metrics_summary = write_metrics_summary(summaries, output_dir)
    write_plots(output_dir, metrics_summary, args.analysis_plot_max_age)
    print(f"[QuantDrift] Wrote drift analysis artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("OVTR quant drift analysis", parents=[add_analysis_args(get_args_parser())])
    parsed_args = parser.parse_args()
    main(parsed_args)
