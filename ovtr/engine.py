# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import cv2
import os
import sys
import math
from contextlib import nullcontext
from typing import Iterable
from util.list_LVIS import CLASSES
import torch
import util.misc as utils
from util import box_ops
from pathlib import Path
from util.events import get_event_storage, TensorboardXWriter
from util.plot_utils import draw_boxes, draw_ref_pts, image_hwc2chw
from datasets.data_prefetcher import data_prefetcher, data_dict_to_cuda


def _debug_cuda_sync(device, label):
    if os.environ.get("OVTR_DEBUG_CUDA_SYNC") == "1" and torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
        print(f"[CUDA-SYNC] {label}", flush=True)


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


def visualize(track_instances, filename):
    for i, _track_instance in enumerate(track_instances):
        prob = _track_instance.pred_logits.sigmoid()
        scores, labels = prob.max(-1)
        
        boxes = box_ops.box_cxcywh_to_xyxy(_track_instance.pred_boxes)
        boxes = boxes.clamp(0, 1)  
        # and from relative [0, 1] to absolute [0, height] coordinates
        root = str(Path.cwd())
        name = filename.split('/')[-1].split('.')[-2] + f'_v{i}.jpg'
        image_path = os.path.join(root, './data_vis', name)
        img = cv2.imread(image_path)
        img_h, img_w, _ = img.shape
          
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        Path('./data_vis_out').mkdir(parents=True, exist_ok=True)
        for score, label, x, obj_idxes in zip(scores, labels, boxes, _track_instance.obj_idxes):
            if obj_idxes == -1:
                continue
            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            cv2.rectangle(img, c1, c2, (144, 238, 144), thickness=1)
            cv2.putText(img, str(obj_idxes.item()) + CLASSES[label] + "{:.2f}".format(score.item()), (c1[0], c1[1] + 3), 0, 0.5, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(os.path.join(root, './data_vis_out', name), img)

def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    writer=None, amp: bool = False, clip_gradients: bool = False,
                    manual_grad_sync: bool = False, stage_scheduler=None,
                    global_step_start: int = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    storage = get_event_storage()
    step = 0

    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        global_step = global_step_start + step
        if stage_scheduler is not None:
            stage_scheduler.before_step(global_step)
            model.train()
            criterion.train()
        filename = data_dict.pop('filename') # for visualization
        data_dict = data_dict_to_cuda(data_dict, device)
        weight_dict = criterion.weight_dict
        model_ref = model.module if hasattr(model, "module") else model
        supports_mot_batch = getattr(model_ref, "supports_mot_batch", False)
        sample_dicts = [data_dict] if supports_mot_batch else _split_mot_batch(data_dict)
        micro_batch_count = len(sample_dicts)
        filename = filename[0] if isinstance(filename, list) else filename

        optimizer.zero_grad()
        loss_dict = {}
        vis_outputs = None
        vis_data_dict = None

        for micro_batch_idx, sample_data_dict in enumerate(sample_dicts):
            outputs = model(sample_data_dict)
            _debug_cuda_sync(device, f"after model forward ({micro_batch_idx + 1}/{micro_batch_count})")

            track_instances = outputs.pop('track_instances')
            # visualize(track_instances, filename)

            loss_dict_i = criterion(outputs)
            _debug_cuda_sync(device, f"after criterion forward ({micro_batch_idx + 1}/{micro_batch_count})")
            losses = sum(loss_dict_i[k] * weight_dict[k] for k in loss_dict_i.keys() if k in weight_dict)

            loss_value_local = losses.detach().item()
            if not math.isfinite(loss_value_local):
                print("Loss is {}, stopping training".format(loss_value_local))
                print(loss_dict_i)
                sys.exit(1)

            scaled_losses = losses / micro_batch_count

            for key, value in loss_dict_i.items():
                if key in loss_dict:
                    loss_dict[key] = loss_dict[key] + value.detach() / micro_batch_count
                else:
                    loss_dict[key] = value.detach() / micro_batch_count

            if vis_outputs is None:
                vis_outputs = {
                    'pred_boxes': [pred_boxes.detach() for pred_boxes in outputs['pred_boxes']],
                    'pred_logits': [pred_logits.detach() for pred_logits in outputs['pred_logits']],
                }
                vis_data_dict = _split_mot_batch(sample_data_dict)[0] if supports_mot_batch else sample_data_dict

            _debug_cuda_sync(device, f"before backward ({micro_batch_idx + 1}/{micro_batch_count})")
            backward_ctx = nullcontext()
            if hasattr(model, "no_sync") and (
                os.environ.get("OVTR_DDP_NO_SYNC_DEBUG") == "1"
                or micro_batch_idx < micro_batch_count - 1
            ):
                backward_ctx = model.no_sync()
            anomaly_ctx = (
                torch.autograd.detect_anomaly()
                if os.environ.get("OVTR_AUTOGRAD_ANOMALY") == "1"
                else nullcontext()
            )
            with anomaly_ctx:
                with backward_ctx:
                    scaled_losses.backward()
            _debug_cuda_sync(device, f"after backward ({micro_batch_idx + 1}/{micro_batch_count})")

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if manual_grad_sync:
            utils.average_gradients(model.named_parameters())
            _debug_cuda_sync(device, "after manual grad sync")
        if max_norm > 0:
            grad_total_norm = utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # gather the stats from all processes
        if writer is not None:
            if step % 20 == 0:
                img = vis_data_dict['ori_img'][0].permute(1, 2, 0).contiguous()
                h, w = img.shape[:2]
                gt_boxes = box_ops.box_cxcywh_to_xyxy(vis_data_dict['gt_instances'][0].boxes)
                gt_boxes[:, ::2] *= w
                gt_boxes[:, 1::2] *= h
                vis_img = draw_boxes(img, gt_boxes, color=(0, 1, 0))
                dt_boxes = vis_outputs['pred_boxes'][0].clone()
                dt_scores = vis_outputs['pred_logits'][0].clone().sigmoid().max(dim=-1)[0]
                keep = dt_scores > 0.4
                dt_boxes = dt_boxes[keep]
                dt_scores = dt_scores[keep]
                dt_boxes[:, 0::2] *= w
                dt_boxes[:, 1::2] *= h
                # print("gt_boxes={} dt_boxes={}".format(gt_boxes, dt_boxes))
                dt_boxes = box_ops.box_cxcywh_to_xyxy(dt_boxes)
                vis_img = draw_boxes(vis_img, dt_boxes, color=(0, 0, 1), texts=[str(score.item()) for score in dt_scores])
                vis_img = image_hwc2chw(vis_img)
                storage.put_image('image_with_gt', vis_img)
            storage.put_scalar("loss", loss_value)
            for loss_name, loss_value in loss_dict_reduced_scaled.items():
                storage.put_scalar(loss_name, loss_value.item())
            writer.write()
        if storage is not None:
            storage.step()
        step += 1
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if stage_scheduler is not None:
        stats.update(stage_scheduler.state_dict())
    return stats
