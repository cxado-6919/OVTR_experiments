# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from collections import defaultdict
from torch.utils.data import DataLoader
import os
import numpy as np
import random
import argparse
import time
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from models import build_model
from util.slconfig import SLConfig
from util.tool import load_model
from util.quantization import (
    build_quant_calibration_loader,
    calibrate_quant_controller_on_val_loader,
    enable_loaded_quantization,
    setup_quant_controller,
)
from main import get_args_parser
from detectron2.structures import Instances
from datasets import build_dataset
import datasets.samplers as samplers
import util.misc as utils
from datasets.data_prefetcher import data_dict_to_cuda
from util.list_LVIS import CLASSES, COLORS
from mmcv.runner import get_dist_info
np.random.seed(2024)


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


def _sync_timing_device(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None, mask=None):
    # Plots one bounding box on image img
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3.5, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3.5, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return img

def draw_bboxes(ori_img, bbox, identities=None, mask=None, offset=(0, 0), cvt_color=False, img_path=None):
    img = ori_img
    for i, box in enumerate(bbox):
        if mask is not None and mask.shape[0] > 0:
            m = mask[i]
        else:
            m = None
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
            label = int(box[5])
        else:
            score = None
            label = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS[id % len(COLORS)]
        label_str = '{:d} {:s}'.format(id, CLASSES[label])
        img = plot_one_box([x1, y1, x2, y2], img, color, label_str, score=score, mask=m)
    return img

def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class Track(object):
    track_cnt = 0

    def __init__(self, box):
        self.box = box
        self.time_since_update = 0
        self.id = Track.track_cnt
        Track.track_cnt += 1
        self.miss = 0

    def miss_one_frame(self):
        self.miss += 1

    def clear_miss(self):
        self.miss = 0

    def update(self, box):
        self.box = box
        self.clear_miss()


class TRTR(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.active_trackers = {}
        self.inactive_trackers = {}
        self.disappeared_tracks = []

    def _remove_track(self, slot_id):
        self.inactive_trackers.pop(slot_id)
        self.disappeared_tracks.append(slot_id)

    def clear_disappeared_track(self):
        self.disappeared_tracks = []

    def update(self, dt_instances: Instances, target_size=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        dt_idxes = set(dt_instances.obj_idxes.tolist())
        track_idxes = set(self.active_trackers.keys()).union(set(self.inactive_trackers.keys()))
        matched_idxes = dt_idxes.intersection(track_idxes)

        unmatched_tracker = track_idxes - matched_idxes
        for track_id in unmatched_tracker:
            # miss in this frame, move to inactive_trackers.
            if track_id in self.active_trackers:
                self.inactive_trackers[track_id] = self.active_trackers.pop(track_id)
            self.inactive_trackers[track_id].miss_one_frame()
            if self.inactive_trackers[track_id].miss > 10:
                self._remove_track(track_id)

        for i in range(len(dt_instances)):
            idx = dt_instances.obj_idxes[i]
            bbox = np.concatenate([dt_instances.boxes[i], dt_instances.scores[i:i + 1]], axis=-1)
            label = dt_instances.cls_idxes[i]
            if label != -1:

                # get a positive track.
                if idx in self.inactive_trackers:
                    # set state of track active.
                    self.active_trackers[idx] = self.inactive_trackers.pop(idx)
                if idx not in self.active_trackers:
                    # create a new track.
                    self.active_trackers[idx] = Track(idx)
                self.active_trackers[idx].update(bbox)

        ret = []
        if dt_instances.has('masks'):
            mask = []
        for i in range(len(dt_instances)):
            label = dt_instances.cls_idxes[i]
            if label != -1:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate(
                    [dt_instances.boxes[i], dt_instances.scores[i:i + 1], dt_instances.cls_idxes[i:i + 1]], axis=-1)
                ret.append(
                    np.concatenate((box_with_score, [id])).reshape(1, -1)) # TETA does not require +1
                if dt_instances.has('masks'):
                    mask.append(dt_instances.masks[i])

        if len(ret) > 0:
            if dt_instances.has('masks'):
                return np.concatenate(ret), np.concatenate(mask)
            return np.concatenate(ret)
        if dt_instances.has('masks'):      
            img_h, img_w = target_size
            return np.empty((0, 7)), np.empty((0, 1, img_h, img_w))
        return np.empty((0, 7))


class OVTR_inference(object):
    def __init__(self, args, cfg, model=None):
        self.args = args
        self.detr = model

        self.tr_tracker = TRTR()

        self.img_height = 800
        self.img_width = 1333

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.results = defaultdict(list)
        self.result_path_track = args.result_path_track
        self.cur_vis_img_path = args.vis_output
        self.video_output_root = None
        self.video_writers = {}
        self.vis_video_fps = 10.0
        self.root = cfg.data.val.img_prefix
        self.num_classes = len(CLASSES)
        self.vis_points = args.vis_points
        self.dataset_list = ["YFCC100M", "HACS", "BDD", "ArgoVerse", "AVA", "LaSOT", "Charades"]
        if self.cur_vis_img_path is not None:
            self.video_output_root = os.path.join(self.cur_vis_img_path, "videos")

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float, score_threshold: float) -> Instances:
        keep = (dt_instances.scores > score_threshold) & (dt_instances.disappear_time == 0)
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]
    
    def tracking_state_hyperparams(self, file_path, prob_threshold, score_threshold, filter_score_thresh, miss_tolerance, maximum_quantity, ious_thresh):
        indd = self.dataset_list.index(file_path.split("/")[1])
        self.detr.track_base.filter_score_thresh = filter_score_thresh[indd]
        self.detr.track_base.score_thresh = score_threshold[indd]
        self.detr.track_base.miss_tolerance = miss_tolerance[indd]
        self.detr.track_base.maximum_quantity = maximum_quantity
        self.detr.transformer.decoder.isol_ratio = 5
        self.detr.ious_thresh = ious_thresh[indd]
        return prob_threshold[indd], score_threshold[indd]
        
    def update_results_teta(self, bbox_xyxy, identities, labels, scores=None, masks=None, dt_instances=None):     
        if dt_instances.boxes.shape[0] == 0:
            bbox_result = [np.zeros((0, 5), dtype=np.float32) for i in range(self.num_classes)]
        else:
            if isinstance(dt_instances.boxes, torch.Tensor):
                bboxes1 = dt_instances.boxes.detach().cpu().numpy()
                labels1 = dt_instances.cls_idxes.detach().cpu().numpy()
                labels1 = np.array(labels1, dtype=int)
            bbox_result = [bboxes1[labels1 == i, :]
                for i in range(self.num_classes)]    

        if bbox_xyxy.shape[0] == 0:
            track_result = [np.zeros((0, 6), dtype=np.float32) for i in range(self.num_classes)]
        else:
            bboxes2 = np.array(bbox_xyxy, dtype=np.float32)
            labels2 = np.array(labels, dtype=int)
            ids = np.array(identities, dtype=int)
            track_result = [
                np.concatenate((ids[labels2 == i, None], bboxes2[labels2 == i, :]), axis=1)
                for i in range(self.num_classes)
                ]
                
        result = dict(bbox_results=bbox_result, track_results=track_result)
    
        for k, v in result.items():
            self.results[k].append(v)

    def update(self, dt_instances: Instances):
        ret = []
        if dt_instances.has('masks'):
            mask = []
        for i in range(len(dt_instances)):
            label = dt_instances.cls_idxes[i]
            if label != -1:
                id = dt_instances.obj_idxes[i]
                box_with_score = np.concatenate(
                    [dt_instances.boxes[i], dt_instances.scores[i:i + 1], dt_instances.cls_idxes[i:i + 1]], axis=-1)
                ret.append(                     
                    np.concatenate((box_with_score, [id])).reshape(1, -1))
                if dt_instances.has('masks'):
                    mask.append(dt_instances.masks[i])

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    @staticmethod
    def _get_vis_output_parts(img_path):
        img_path_parts = Path(img_path).parts
        frame_name = img_path_parts[-1]
        sequence_name = img_path_parts[-2] if len(img_path_parts) >= 2 else "sequence"
        dataset_name = img_path_parts[-3] if len(img_path_parts) >= 3 else "dataset"
        return dataset_name, sequence_name, frame_name

    def _get_video_writer(self, img_path, frame_width, frame_height):
        if self.video_output_root is None:
            return None

        dataset_name, sequence_name, _ = self._get_vis_output_parts(img_path)
        writer_key = (dataset_name, sequence_name)
        writer = self.video_writers.get(writer_key)
        if writer is not None:
            return writer

        output_dir = os.path.join(self.video_output_root, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{sequence_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.vis_video_fps, (frame_width, frame_height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not initialize VideoWriter for {output_path}")
        self.video_writers[writer_key] = writer
        return writer

    def close_visualization_writers(self):
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()

    def visualize_img_with_bbox(self, save_path, img_path, dt_instances: Instances, ref_pts=None, vis_points=None):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image for visualization: {img_path}")
        img_show = img
        if dt_instances.has('scores'):
            img_show = draw_bboxes(img, np.concatenate(
                [dt_instances.boxes, dt_instances.scores.reshape(-1, 1), dt_instances.cls_idxes.reshape(-1, 1)],
                axis=-1), dt_instances.obj_idxes, img_path = img_path)
        if vis_points:
            img_show = draw_points(img_show, ref_pts)
        writer = self._get_video_writer(img_path, img_show.shape[1], img_show.shape[0])
        if writer is not None:
            writer.write(img_show)

    def detect(self, prob_threshold=0.6, score_threshold=0.5, filter_score_thresh=0.5, miss_tolerance=5, maximum_quantity=60, area_threshold=100, ious_thresh=0.3,
               vis=False, data=None, track_instances=None, info=None, file_path=None):
        frame_id = info[0]
        prob_threshold, score_threshold = self.tracking_state_hyperparams(file_path, prob_threshold, score_threshold, filter_score_thresh, miss_tolerance, maximum_quantity, ious_thresh)

        res = self.detr.inference_single_image(data, track_instances, frame_id=frame_id, ori_img_size=info[1])
        track_instances = res['track_instances']
        dt_instances = track_instances.to(torch.device('cpu'))

        # filter det instances by score.
        dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold, score_threshold)
        dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

        if vis:
            all_ref_pts = tensor_to_numpy(res['ref_pts'][0, :, :2])
            self.visualize_img_with_bbox(self.cur_vis_img_path, os.path.join(self.root, file_path), dt_instances, ref_pts=all_ref_pts, vis_points=self.vis_points)

        tracker_outputs = self.update(dt_instances)

        self.update_results_teta(
                            bbox_xyxy=tracker_outputs[:, :4],
                            identities=tracker_outputs[:, 6],
                            labels=tracker_outputs[:, 5],
                            scores=tracker_outputs[:, 4],
                            masks=None,
                            dt_instances=dt_instances)

        if track_instances is not None:
            track_instances.remove('boxes')
            track_instances.remove('labels')
        return track_instances


def eval(args, cfg):
    utils.init_distributed_mode(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.vis and not args.vis_output:
        raise ValueError("--vis requires --vis_output so eval visualizations have a destination.")

    cfg.data.test.test_mode = True
    cfg.device = args.device
    torch.manual_seed(args.seed)

    # load model and weights
    model, _, = build_model(args, cfg)
    quant_controller = setup_quant_controller(model, args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    model = load_model(model, args.pretrained)
    quant_state_loaded = False
    if quant_controller is not None:
        quant_state_loaded = enable_loaded_quantization(model, require_state=False)
    model.eval()
    model = model.to(torch.device(args.device))

    dataset_val = build_dataset(image_set='val', args=args, cfg=cfg.data.test)
    if args.distributed:
        if args.cache_mode:
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    collate_fn = utils.mot_collate_fn
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    if args.quant_mode == "ptq" and not quant_state_loaded:
        data_loader_calib = build_quant_calibration_loader(args, cfg)
        calibrated = calibrate_quant_controller_on_val_loader(
            model,
            data_loader_calib,
            torch.device(args.device),
            args.quant_calib_samples,
            args=args,
        )
        quant_controller.enable_quantization()
        print(f"[Quant] PTQ calibration complete on {calibrated} samples", flush=True)
        if args.quant_calibration_only:
            if args.output_dir:
                utils.save_on_master(
                    {"model": model.state_dict(), "args": args},
                    Path(args.output_dir) / "checkpoint_quant_calibrated.pth",
                )
            print("[Quant] Exiting after PTQ calibration as requested.", flush=True)
            return
    elif args.quant_mode == "qat" and quant_controller is not None and not quant_state_loaded:
        raise ValueError("QAT evaluation expects a checkpoint that already contains learned quant state.")
    
    tracker = OVTR_inference(args, cfg, model=model)

    torch.manual_seed(args.seed)

    tracker.result_path_track = os.path.abspath(tracker.result_path_track)
    os.makedirs((tracker.result_path_track), exist_ok = True)
    track_instances = None
    timing_device = model.text_embeddings.device
    total_detect_time = 0.0
    processed_frames = 0

    try:
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(data_loader_val)):
                sample_dicts = _split_mot_batch(dict(data_dict))
                for sample_data_dict in sample_dicts:
                    # Tracking state is frame-sequential, so consume loader batches one sample at a time.
                    info = _unwrap_singleton_list(sample_data_dict.pop('info'))
                    file_path = _unwrap_singleton_list(sample_data_dict.pop('file_path'))
                    sample_data_dict = data_dict_to_cuda(sample_data_dict, device=timing_device)
                    _sync_timing_device(timing_device)
                    start_time = time.perf_counter()
                    track_instances = tracker.detect(vis=args.vis, data=sample_data_dict, track_instances=track_instances, info=info,
                                                     prob_threshold=args.score_thresh, score_threshold=args.score_thresh, filter_score_thresh=args.filter_score_thresh,
                                                     miss_tolerance=args.miss_tolerance, maximum_quantity=args.maximum_quantity, area_threshold=1, ious_thresh=args.ious_thresh,
                                                     file_path=file_path)
                    _sync_timing_device(timing_device)
                    total_detect_time += time.perf_counter() - start_time
                    processed_frames += 1
    finally:
        tracker.close_visualization_writers()

    timing_stats_device = timing_device if timing_device.type == "cuda" else torch.device("cpu")
    timing_stats = torch.tensor([total_detect_time, processed_frames], dtype=torch.float64, device=timing_stats_device)
    if utils.is_dist_avail_and_initialized():
        timing_stats = utils.all_reduce_tensor(timing_stats, average=False)
    total_detect_time = timing_stats[0].item()
    processed_frames = int(timing_stats[1].item())
    if processed_frames > 0 and total_detect_time > 0:
        avg_latency_ms = (total_detect_time / processed_frames) * 1000.0
        fps = processed_frames / total_detect_time
    else:
        avg_latency_ms = float("nan")
        fps = float("nan")

    resfile_path = tracker.result_path_track
    print('Inference completed')

    print('Start TETA')
    content_track = tracker.results['track_results']
    outputs={"track_results":content_track, "bbox_results":None}
    
    rank, _ = get_dist_info()
    if rank == 0:
        print(f'Average per-frame latency: {avg_latency_ms:.2f} ms')
        print(f'FPS: {fps:.2f}')
        kwargs = {} if args.eval_options is None else args.eval_options
        
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        eval_kwargs.resfile_path = resfile_path
        eval_results = dataset_val.evaluate(outputs, **eval_kwargs)
        eval_results['avg_latency_ms'] = avg_latency_ms
        eval_results['fps'] = fps
        print(eval_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OVTR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    cfg = SLConfig.fromfile(args.config_file)

    eval(args, cfg)
