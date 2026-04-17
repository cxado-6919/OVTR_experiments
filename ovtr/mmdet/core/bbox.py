import numpy as np
import torch


def _to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data)


def _to_output(data, like):
    if isinstance(like, torch.Tensor):
        return data.to(device=like.device, dtype=data.dtype if data.is_floating_point() else like.dtype)
    return data.cpu().numpy()


def bbox_overlaps(bboxes1, bboxes2, mode="iou", eps=1e-6):
    boxes1 = _to_tensor(bboxes1).float()
    boxes2 = _to_tensor(bboxes2).float()

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        out = boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
        return _to_output(out, bboxes1)

    if boxes1.dim() == 3:
        lt = torch.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
        rb = torch.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = (boxes1[..., :, 2] - boxes1[..., :, 0]).clamp(min=0) * (
            boxes1[..., :, 3] - boxes1[..., :, 1]
        ).clamp(min=0)
        area2 = (boxes2[..., :, 2] - boxes2[..., :, 0]).clamp(min=0) * (
            boxes2[..., :, 3] - boxes2[..., :, 1]
        ).clamp(min=0)
        if mode == "iou":
            union = area1[..., :, None] + area2[..., None, :] - inter
        elif mode == "iof":
            union = area1[..., :, None]
        else:
            raise ValueError(f"Unsupported overlap mode: {mode}")
        overlaps = inter / union.clamp(min=eps)
        return _to_output(overlaps, bboxes1)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    if mode == "iou":
        union = area1[:, None] + area2[None, :] - inter
    elif mode == "iof":
        union = area1[:, None]
    else:
        raise ValueError(f"Unsupported overlap mode: {mode}")

    overlaps = inter / union.clamp(min=eps)
    return _to_output(overlaps, bboxes1)


def bbox2result(bboxes, labels, num_classes):
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    return [bboxes[labels == i, :] for i in range(num_classes)]


def find_inside_bboxes(bboxes, img_h, img_w):
    boxes = _to_tensor(bboxes).float()
    keep = (boxes[:, 0] < img_w) & (boxes[:, 1] < img_h) & (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
    if isinstance(bboxes, torch.Tensor):
        return keep.to(device=bboxes.device)
    return keep.cpu().numpy()
