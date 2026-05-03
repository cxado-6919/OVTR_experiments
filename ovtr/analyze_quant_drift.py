import argparse
import copy
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
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
    build_accumulation_gap_rows,
    build_teacher_forced_input,
    compare_query_state_io,
    compare_frame_to_reference,
    run_inference_frame,
    sequence_key_from_file_path,
    snapshot_frame_state,
    snapshot_feedback_active,
    strip_feedback_only_fields,
    summarize_run_metrics,
    sync_timing_device,
    write_metric_csvs,
    write_metrics_summary,
    write_plots,
    write_research_csvs,
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


DEFAULT_ANALYSIS_SAMPLE_DATASETS = ["YFCC100M", "HACS", "BDD", "ArgoVerse", "AVA", "LaSOT", "Charades"]


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
        help="Maximum frames to process. 0 means the full validation/test split.",
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
    parser.add_argument(
        "--analysis_sample_sequences_per_dataset",
        default=0,
        type=int,
        help="Maximum video sequences sampled per dataset prefix. 0 disables sequence sampling.",
    )
    parser.add_argument(
        "--analysis_sample_datasets",
        nargs="+",
        default=DEFAULT_ANALYSIS_SAMPLE_DATASETS,
        help="Dataset prefixes eligible for sequence sampling.",
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


def build_validation_dataset(args, cfg):
    cfg.data.test.test_mode = True
    dataset_val = build_dataset(image_set="val", args=args, cfg=cfg.data.test)
    return dataset_val


def build_sequence_ranges(dataset, max_frames: int) -> Tuple[List[Tuple[int, int]], int]:
    data_infos = getattr(dataset, "data_infos", None)
    if data_infos is None:
        raise ValueError("Quant drift analysis requires dataset.data_infos to split sequences.")

    total_frames = len(data_infos)
    frame_limit = total_frames if max_frames <= 0 else min(int(max_frames), total_frames)
    if frame_limit <= 0:
        return [], 0

    starts = [
        idx
        for idx, info in enumerate(data_infos[:frame_limit])
        if int(info.get("frame_id", -1)) == 0
    ]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)

    ranges = []
    for pos, start in enumerate(starts):
        if start >= frame_limit:
            continue
        next_start = starts[pos + 1] if pos + 1 < len(starts) else frame_limit
        end = min(next_start, frame_limit)
        if start < end:
            ranges.append((start, end))
    return ranges, frame_limit


def _data_info_file_name(info: dict) -> str:
    return str(info.get("file_name") or info.get("filename") or info.get("file_path") or "")


def _dataset_prefix_from_info(info: dict, dataset_names: Sequence[str]) -> str:
    metadata = info.get("metadata")
    if isinstance(metadata, dict) and metadata.get("dataset"):
        return str(metadata["dataset"])

    path = _data_info_file_name(info).replace("\\", "/")
    parts = [part for part in path.split("/") if part]
    for part in parts:
        if part in dataset_names:
            return part
    if len(parts) >= 2 and parts[0] in {"train", "val", "validation", "test"}:
        return parts[1]
    return parts[0] if parts else "unknown"


def _sequence_ranges_from_infos(data_infos: Sequence[dict]) -> List[Tuple[int, int]]:
    starts = [idx for idx, info in enumerate(data_infos) if int(info.get("frame_id", -1)) == 0]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    starts.append(len(data_infos))
    return [(starts[idx], starts[idx + 1]) for idx in range(len(starts) - 1) if starts[idx] < starts[idx + 1]]


def _ordered_unique(values: Sequence[int]) -> List[int]:
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _write_sampled_annotation(
    *,
    dataset,
    output_dir: Path,
    selected_image_ids: Sequence[int],
    selected_video_ids: Sequence[int],
) -> Optional[Path]:
    ann_file = getattr(dataset, "ann_file", None)
    if not ann_file or not selected_image_ids:
        return None

    ann_path = Path(ann_file)
    if not ann_path.exists():
        print(f"[QuantDrift] Warning: annotation file not found; skipped sampled annotation: {ann_file}", flush=True)
        return None

    with ann_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)

    selected_image_ids = set(selected_image_ids)
    selected_video_ids = set(selected_video_ids)
    selected_track_ids = {
        ann.get("track_id")
        for ann in source.get("annotations", [])
        if ann.get("image_id") in selected_image_ids and ann.get("video_id") in selected_video_ids
    }

    subset = {}
    for key, value in source.items():
        if key == "images" and isinstance(value, list):
            subset[key] = [item for item in value if item.get("id") in selected_image_ids]
        elif key == "videos" and isinstance(value, list):
            subset[key] = [item for item in value if item.get("id") in selected_video_ids]
        elif key == "annotations" and isinstance(value, list):
            subset[key] = [
                item
                for item in value
                if item.get("image_id") in selected_image_ids and item.get("video_id") in selected_video_ids
            ]
        elif key == "tracks" and isinstance(value, list):
            subset[key] = [
                item
                for item in value
                if item.get("video_id") in selected_video_ids
                and (item.get("id") in selected_track_ids or item.get("track_id") in selected_track_ids)
            ]
        else:
            subset[key] = value

    sampled_ann_path = output_dir / "sampled_annotations.json"
    with sampled_ann_path.open("w", encoding="utf-8") as handle:
        json.dump(subset, handle)
    dataset.ann_file = str(sampled_ann_path)
    return sampled_ann_path


def apply_sequence_sampling_to_dataset(dataset, args) -> bool:
    per_dataset = int(getattr(args, "analysis_sample_sequences_per_dataset", 0))
    if per_dataset <= 0:
        return False

    data_infos = getattr(dataset, "data_infos", None)
    if data_infos is None:
        raise ValueError("Quant drift analysis sequence sampling requires dataset.data_infos.")

    dataset_names = list(getattr(args, "analysis_sample_datasets", None) or DEFAULT_ANALYSIS_SAMPLE_DATASETS)
    sequence_ranges = _sequence_ranges_from_infos(data_infos)
    grouped_ranges: Dict[str, List[Tuple[int, int]]] = {name: [] for name in dataset_names}
    skipped_ranges = 0
    for start, end in sequence_ranges:
        dataset_name = _dataset_prefix_from_info(data_infos[start], dataset_names)
        if dataset_name not in grouped_ranges:
            skipped_ranges += 1
            continue
        grouped_ranges[dataset_name].append((start, end))

    rng = random.Random(args.seed)
    selected_ranges = []
    for dataset_name in dataset_names:
        ranges = grouped_ranges[dataset_name]
        if len(ranges) > per_dataset:
            ranges = rng.sample(ranges, per_dataset)
        selected_ranges.extend(ranges)
        print(
            f"[QuantDrift] Sampled {len(ranges)}/{len(grouped_ranges[dataset_name])} "
            f"sequences from {dataset_name}",
            flush=True,
        )

    if not selected_ranges:
        raise ValueError(
            "No sequences matched --analysis_sample_datasets: " + " ".join(dataset_names)
        )

    selected_ranges = sorted(selected_ranges)
    selected_data_infos = [info for start, end in selected_ranges for info in data_infos[start:end]]
    selected_image_ids = [info["id"] for info in selected_data_infos if "id" in info]
    selected_video_ids = _ordered_unique([info["video_id"] for info in selected_data_infos if "video_id" in info])

    dataset.data_infos = selected_data_infos
    if hasattr(dataset, "img_ids"):
        dataset.img_ids = selected_image_ids
    if hasattr(dataset, "vid_ids"):
        dataset.vid_ids = selected_video_ids
    if hasattr(dataset, "ids"):
        dataset.ids = None
    if hasattr(dataset, "num_samples"):
        dataset.num_samples = len(dataset)

    print(
        f"[QuantDrift] Sequence sampling kept {len(selected_ranges)}/{len(sequence_ranges)} sequences "
        f"and {len(selected_data_infos)} frames; skipped {skipped_ranges} unmatched sequences",
        flush=True,
    )
    return True


def build_sequence_loader(dataset, start: int, end: int, args):
    subset = Subset(dataset, range(start, end))
    sampler_val = torch.utils.data.SequentialSampler(subset)
    return DataLoader(
        subset,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.mot_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def empty_run_result(run_mode: str) -> RunResult:
    return RunResult(
        run_mode=run_mode,
        track_results=[],
        processed_frames=0,
        total_detect_time=0.0,
    )


def accumulate_run_result(total: RunResult, partial: RunResult) -> None:
    total.track_results.extend(partial.track_results)
    total.processed_frames += partial.processed_frames
    total.total_detect_time += partial.total_detect_time


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
    progress_desc: Optional[str] = None,
    progress_leave: bool = True,
) -> RunResult:
    if hasattr(model, "clear"):
        model.clear()

    trace = ReferenceTrace() if run_mode == "fp32_free" else None
    track_results = []
    track_metric_rows = []
    frame_metric_rows = []
    divergence_rows = []
    state_io_rows = []
    first_divergences = {}
    track_ages = {}
    track_instances = None
    previous_sequence = None
    previous_fp32_frame = None
    total_detect_time = 0.0
    processed_frames = 0

    progress = tqdm(data_loader, desc=progress_desc or run_mode, leave=progress_leave)
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

                input_feedback_active = snapshot_feedback_active(track_instances)
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
                reference_frame = fp32_trace.get(file_path, frame_id) if fp32_trace is not None else None
                state_io_rows.extend(
                    compare_query_state_io(
                        run_mode=run_mode,
                        file_path=file_path,
                        frame_id=frame_id,
                        input_instances=input_feedback_active,
                        output_frame=frame_trace,
                        reference_frame=reference_frame,
                    )
                )

                if run_mode == "fp32_free":
                    trace.frames[(file_path, frame_id)] = frame_trace
                else:
                    rows, frame_row, new_divergences = compare_frame_to_reference(
                        run_mode=run_mode,
                        fp32_frame=reference_frame,
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
        state_io_rows=state_io_rows,
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
    dataset_val = build_validation_dataset(args, cfg)
    sampled_dataset = apply_sequence_sampling_to_dataset(dataset_val, args)
    sequence_ranges, frame_limit = build_sequence_ranges(dataset_val, args.analysis_max_frames)
    if not sequence_ranges:
        raise ValueError("No frames available for quant drift analysis.")
    if sampled_dataset:
        sampled_infos = dataset_val.data_infos[:frame_limit]
        sampled_ann_path = _write_sampled_annotation(
            dataset=dataset_val,
            output_dir=output_dir,
            selected_image_ids=[info["id"] for info in sampled_infos if "id" in info],
            selected_video_ids=_ordered_unique([info["video_id"] for info in sampled_infos if "video_id" in info]),
        )
        if sampled_ann_path is not None:
            print(f"[QuantDrift] Wrote sampled annotation to {sampled_ann_path}", flush=True)
    device = torch.device(args.device)

    print("[QuantDrift] HOTA is not implemented in this repo; summary will include TETA, IDF1, and MOTA.")
    print(
        f"[QuantDrift] Processing {frame_limit} frames in {len(sequence_ranges)} sequences "
        f"from {args.config_file}",
        flush=True,
    )
    write_metric_csvs(
        track_rows=[],
        frame_rows=[],
        divergence_rows=[],
        output_dir=output_dir,
        append=False,
    )
    write_research_csvs(
        one_step_rows=[],
        accumulation_gap_rows=[],
        state_io_rows=[],
        output_dir=output_dir,
        append=False,
    )

    print("[QuantDrift] Loading fp32_free model", flush=True)
    fp32_model = build_loaded_model(
        args,
        cfg,
        device,
        checkpoint_path=args.analysis_fp32_pretrain,
        quant_mode="none",
    )
    print("[QuantDrift] Loading quantized model", flush=True)
    quant_model = build_loaded_model(
        args,
        cfg,
        device,
        checkpoint_path=args.pretrained,
        quant_mode=args.quant_mode,
    )

    fp32_result = empty_run_result("fp32_free")
    quant_free_result = empty_run_result("quant_free")
    quant_teacher_result = empty_run_result("quant_teacher_forced")

    for sequence_idx, (start, end) in enumerate(tqdm(sequence_ranges, desc="sequences"), start=1):
        sequence_loader = build_sequence_loader(dataset_val, start, end, args)
        sequence_desc = f"seq {sequence_idx}/{len(sequence_ranges)} [{start}:{end}]"

        fp32_sequence_result = run_analysis_pass(
            run_mode="fp32_free",
            model=fp32_model,
            data_loader=sequence_loader,
            device=device,
            args=args,
            progress_desc=f"fp32_free {sequence_desc}",
            progress_leave=False,
        )
        fp32_trace = fp32_sequence_result.trace
        if fp32_trace is None:
            raise RuntimeError("fp32_free pass did not produce a reference trace.")

        quant_free_sequence_result = run_analysis_pass(
            run_mode="quant_free",
            model=quant_model,
            data_loader=sequence_loader,
            device=device,
            args=args,
            fp32_trace=fp32_trace,
            progress_desc=f"quant_free {sequence_desc}",
            progress_leave=False,
        )

        quant_teacher_sequence_result = run_analysis_pass(
            run_mode="quant_teacher_forced",
            model=quant_model,
            data_loader=sequence_loader,
            device=device,
            args=args,
            fp32_trace=fp32_trace,
            teacher_forced=True,
            progress_desc=f"quant_teacher_forced {sequence_desc}",
            progress_leave=False,
        )

        accumulation_gap_rows = build_accumulation_gap_rows(
            quant_free_sequence_result.track_metric_rows,
            quant_teacher_sequence_result.track_metric_rows,
        )
        write_metric_csvs(
            track_rows=(
                quant_free_sequence_result.track_metric_rows
                + quant_teacher_sequence_result.track_metric_rows
            ),
            frame_rows=(
                quant_free_sequence_result.frame_metric_rows
                + quant_teacher_sequence_result.frame_metric_rows
            ),
            divergence_rows=(
                quant_free_sequence_result.divergence_rows
                + quant_teacher_sequence_result.divergence_rows
            ),
            output_dir=output_dir,
            append=True,
        )
        write_research_csvs(
            one_step_rows=quant_teacher_sequence_result.track_metric_rows,
            accumulation_gap_rows=accumulation_gap_rows,
            state_io_rows=(
                fp32_sequence_result.state_io_rows
                + quant_free_sequence_result.state_io_rows
                + quant_teacher_sequence_result.state_io_rows
            ),
            output_dir=output_dir,
            append=True,
        )

        accumulate_run_result(fp32_result, fp32_sequence_result)
        accumulate_run_result(quant_free_result, quant_free_sequence_result)
        accumulate_run_result(quant_teacher_result, quant_teacher_sequence_result)

        del (
            sequence_loader,
            fp32_trace,
            fp32_sequence_result,
            quant_free_sequence_result,
            quant_teacher_sequence_result,
            accumulation_gap_rows,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del fp32_model, quant_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
