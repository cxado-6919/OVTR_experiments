import copy
import os
import random

import cv2
import numpy as np
import torch

import mmcv
from mmcv.parallel import DataContainer as DC

from ...core import find_inside_bboxes
from ..builder import PIPELINES


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, (list, tuple)):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.tensor([data], dtype=torch.long)
    if isinstance(data, float):
        return torch.tensor([data], dtype=torch.float32)
    raise TypeError(f"type {type(data)} cannot be converted to tensor")


def _maybe_copy_array(data):
    if isinstance(data, np.ndarray):
        return data.copy()
    return data


def _pick_fill_value(fill_candidates):
    fill_value = random.choice(fill_candidates)
    if isinstance(fill_value, tuple):
        return list(fill_value)
    return fill_value


class Compose:
    def __init__(self, transforms):
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                self.transforms.append(mmcv.build_from_cfg(transform, PIPELINES))
            else:
                self.transforms.append(transform)

    def __call__(self, data):
        for transform in self.transforms:
            if data is None:
                return None
            data = transform(data)
        return data


@PIPELINES.register_module(force=True)
class LoadImageFromFile:
    def __init__(self, to_float32=False, color_type="color", file_client_args=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args or {"backend": "disk"}
        self.file_client = None

    def _get_filename(self, results):
        filename = results["img_info"]["filename"]
        prefix = results.get("img_prefix")
        if prefix:
            filename = os.path.join(prefix, filename)
        return filename

    def _load_image(self, filename):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        data = self.file_client.get(filename)

        # Case 1: already-decoded HWC image array
        if isinstance(data, np.ndarray) and data.ndim == 3:
            img = data.copy()

        # Case 2: encoded bytes stored in a scalar ndarray, e.g. np.array(b'...'), common for HDF5 string datasets
        elif isinstance(data, np.ndarray) and data.ndim == 0:
            scalar = data.item()
            if isinstance(scalar, (bytes, bytearray, memoryview, np.bytes_)):
                buffer = np.frombuffer(bytes(scalar), dtype=np.uint8)
                img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            else:
                raise TypeError(
                    f"Unsupported scalar ndarray payload for image loading: "
                    f"type={type(scalar)}, filename={filename}"
                )

        # Case 3: raw encoded image buffer as uint8 ndarray
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            if data.dtype != np.uint8:
                data = data.astype(np.uint8, copy=False)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Case 4: plain python bytes-like payload
        elif isinstance(data, (bytes, bytearray, memoryview, np.bytes_)):
            buffer = np.frombuffer(bytes(data), dtype=np.uint8)
            img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        else:
            raise TypeError(
                f"Unsupported image payload type={type(data)} "
                f"shape={getattr(data, 'shape', None)} filename={filename}"
            )

        if img is None:
            raise RuntimeError(f"cv2.imdecode failed for filename={filename}")

        return img

    def __call__(self, results):
        filename = self._get_filename(results)
        img = self._load_image(filename)
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        results["ori_filename"] = results["img_info"].get("filename", filename)
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["img_fields"] = ["img"]
        return results


@PIPELINES.register_module(force=True)
class LoadAnnotations:
    def __init__(self, with_bbox=True, with_label=True, with_mask=False, **kwargs):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask

    def __call__(self, results):
        ann_info = results["ann_info"]
        if self.with_bbox:
            results["gt_bboxes"] = ann_info["bboxes"].copy()
            results.setdefault("bbox_fields", []).append("gt_bboxes")
            if "bboxes_ignore" in ann_info:
                results["gt_bboxes_ignore"] = ann_info["bboxes_ignore"].copy()
                results["bbox_fields"].append("gt_bboxes_ignore")
        if self.with_label:
            results["gt_labels"] = ann_info["labels"].copy()
        if self.with_mask and "masks" in ann_info:
            results["gt_masks"] = ann_info["masks"]
            results.setdefault("mask_fields", []).append("gt_masks")
        if "seg_map" in ann_info:
            results["seg_map"] = ann_info["seg_map"]
        return results


@PIPELINES.register_module(force=True)
class FilterAnnotations:
    def __init__(self, min_gt_bbox_wh=(1, 1), keep_empty=False):
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.keep_empty = keep_empty

    def _filter(self, results):
        if "gt_bboxes" not in results:
            return results

        gt_bboxes = results["gt_bboxes"]
        if gt_bboxes.shape[0] == 0:
            return results

        widths = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        heights = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (widths > self.min_gt_bbox_wh[0]) & (heights > self.min_gt_bbox_wh[1])

        if not keep.any():
            return None if self.keep_empty else results

        for key in ("gt_bboxes", "gt_labels", "gt_masks", "gt_semantic_seg", "gt_match_indices"):
            if key in results:
                results[key] = results[key][keep]
        return results

    def __call__(self, results):
        return self._filter(results)


@PIPELINES.register_module(force=True)
class Collect:
    def __init__(self, keys, meta_keys=()):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {key: results[key] for key in self.meta_keys if key in results}
        data["img_metas"] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data


@PIPELINES.register_module(force=True)
class DefaultFormatBundle:
    def __call__(self, results):
        if "img" in results:
            img = results["img"]
            if img.ndim == 3:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results["img"] = DC(to_tensor(img), stack=True)

        for key in ["gt_bboxes", "gt_bboxes_ignore", "gt_labels", "gt_masks"]:
            if key in results:
                results[key] = DC(to_tensor(results[key]))
        return results


@PIPELINES.register_module(force=True)
class ImageToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if img.ndim == 3:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results[key] = to_tensor(img)
        return results


@PIPELINES.register_module(force=True)
class Resize:
    def __init__(self, img_scale=None, multiscale_mode="value", keep_ratio=False, bbox_clip_border=True):
        self.img_scale = img_scale
        self.multiscale_mode = multiscale_mode
        self.keep_ratio = keep_ratio
        self.bbox_clip_border = bbox_clip_border

    def _sample_scale(self):
        if isinstance(self.img_scale, list):
            return random.choice(self.img_scale)
        return self.img_scale

    def __call__(self, results):
        img = results["img"]
        scale = results.get("scale") or self._sample_scale()
        if scale is None:
            return results

        target_w, target_h = scale
        h, w = img.shape[:2]
        if self.keep_ratio:
            ratio = min(target_w / w, target_h / h)
            new_w = max(int(round(w * ratio)), 1)
            new_h = max(int(round(h * ratio)), 1)
            scale_factor = np.array([ratio, ratio, ratio, ratio], dtype=np.float32)
        else:
            new_w = target_w
            new_h = target_h
            scale_factor = np.array(
                [target_w / w, target_h / h, target_w / w, target_h / h], dtype=np.float32
            )

        results["img"] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        results["img_shape"] = results["img"].shape
        results["pad_shape"] = results["img"].shape
        results["scale"] = scale
        results["scale_factor"] = scale_factor
        results["keep_ratio"] = self.keep_ratio

        for key in results.get("bbox_fields", []):
            bboxes = results[key].copy()
            bboxes[:, 0::2] *= scale_factor[0]
            bboxes[:, 1::2] *= scale_factor[1]
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, new_w)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, new_h)
            results[key] = bboxes
        return results


@PIPELINES.register_module(force=True)
class RandomFlip:
    def __init__(self, flip_ratio=0.5, direction="horizontal"):
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        flip = results.get("flip")
        flip_direction = results.get("flip_direction")
        if flip is None:
            flip = random.random() < self.flip_ratio
            flip_direction = self.direction if flip else None
            results["flip"] = flip
            results["flip_direction"] = flip_direction

        if not flip:
            return results

        img = results["img"]
        if flip_direction == "horizontal":
            results["img"] = img[:, ::-1, ...].copy()
        elif flip_direction == "vertical":
            results["img"] = img[::-1, :, ...].copy()
        else:
            raise ValueError(f"Unsupported flip direction: {flip_direction}")

        h, w = results["img"].shape[:2]
        for key in results.get("bbox_fields", []):
            bboxes = results[key].copy()
            if flip_direction == "horizontal":
                x1 = w - bboxes[:, 2]
                x2 = w - bboxes[:, 0]
                bboxes[:, 0] = x1
                bboxes[:, 2] = x2
            else:
                y1 = h - bboxes[:, 3]
                y2 = h - bboxes[:, 1]
                bboxes[:, 1] = y1
                bboxes[:, 3] = y2
            results[key] = bboxes
        return results


@PIPELINES.register_module(force=True)
class Normalize:
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        img = results["img"].astype(np.float32)
        if self.to_rgb:
            img = img[..., ::-1]
        results["img"] = (img - self.mean) / self.std
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module(force=True)
class Pad:
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        img = results["img"]
        h, w = img.shape[:2]
        if self.size is not None:
            target_h, target_w = self.size
        elif self.size_divisor is not None:
            target_h = int(np.ceil(h / self.size_divisor)) * self.size_divisor
            target_w = int(np.ceil(w / self.size_divisor)) * self.size_divisor
        else:
            return results

        pad_h = target_h - h
        pad_w = target_w - w
        if img.ndim == 3:
            padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=self.pad_val)
        else:
            padded = np.pad(img, ((0, pad_h), (0, pad_w)), constant_values=self.pad_val)
        results["img"] = padded
        results["pad_shape"] = padded.shape
        return results


@PIPELINES.register_module(force=True)
class CutOut:
    def __init__(self, n_holes=(1, 1), cutout_ratio=(0.5, 0.5), fill_in=(0, 0, 0), **kwargs):
        if isinstance(n_holes, int):
            self.n_holes = (n_holes, n_holes)
        else:
            self.n_holes = n_holes
        if isinstance(cutout_ratio, (int, float)):
            cutout_ratio = (float(cutout_ratio), float(cutout_ratio))
        self.candidates = [cutout_ratio]
        self.fill_in = fill_in if isinstance(fill_in, list) else [fill_in]

    def __call__(self, results):
        img = results["img"]
        img_h, img_w = img.shape[:2]
        hole_count = random.randint(self.n_holes[0], self.n_holes[1])
        ratio_min, ratio_max = self.candidates[0]
        fill_value = _pick_fill_value(self.fill_in)

        for _ in range(hole_count):
            cut_ratio = random.uniform(ratio_min, ratio_max)
            cutout_w = max(int(img_w * cut_ratio), 1)
            cutout_h = max(int(img_h * cut_ratio), 1)
            x1 = random.randint(0, max(img_w - cutout_w, 0))
            y1 = random.randint(0, max(img_h - cutout_h, 0))
            x2 = min(x1 + cutout_w, img_w)
            y2 = min(y1 + cutout_h, img_h)
            results["img"][y1:y2, x1:x2, ...] = fill_value
        return results


@PIPELINES.register_module(force=True)
class Mosaic:
    def __init__(
        self,
        img_scale,
        center_ratio_range=(0.5, 1.5),
        min_bbox_size=0,
        bbox_clip_border=True,
        pad_val=114.0,
        skip_filter=False,
        **kwargs,
    ):
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.skip_filter = skip_filter

    def get_indexes(self, dataset):
        return [random.randint(0, len(dataset) - 1) for _ in range(3)]

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
        assert loc in ("top_left", "top_right", "bottom_left", "bottom_right")

        if loc == "top_left":
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                max(center_position_xy[1] - img_shape_wh[1], 0),
                center_position_xy[0],
                center_position_xy[1],
            )
            crop_coord = (
                img_shape_wh[0] - (x2 - x1),
                img_shape_wh[1] - (y2 - y1),
                img_shape_wh[0],
                img_shape_wh[1],
            )
        elif loc == "top_right":
            x1, y1, x2, y2 = (
                center_position_xy[0],
                max(center_position_xy[1] - img_shape_wh[1], 0),
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2),
                center_position_xy[1],
            )
            crop_coord = (0, img_shape_wh[1] - (y2 - y1), min(img_shape_wh[0], x2 - x1), img_shape_wh[1])
        elif loc == "bottom_left":
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                center_position_xy[1],
                center_position_xy[0],
                min(center_position_xy[1] + img_shape_wh[1], self.img_scale[0] * 2),
            )
            crop_coord = (img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(y2 - y1, img_shape_wh[1]))
        else:
            x1, y1, x2, y2 = (
                center_position_xy[0],
                center_position_xy[1],
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2),
                min(center_position_xy[1] + img_shape_wh[1], self.img_scale[0] * 2),
            )
            crop_coord = (0, 0, min(img_shape_wh[0], x2 - x1), min(y2 - y1, img_shape_wh[1]))

        return (x1, y1, x2, y2), crop_coord

    def _filter_box_candidates(self, bboxes, labels, match_indices=None):
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid = (bbox_w > self.min_bbox_size) & (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid)[0]
        filtered = [bboxes[valid_inds], labels[valid_inds]]
        if match_indices is not None:
            filtered.append(match_indices[valid_inds])
        return filtered

    def __call__(self, results):
        if "mix_results" not in results:
            raise NotImplementedError(
                "Unsupported base transform 'Mosaic': expected 'mix_results' from a dataset wrapper."
            )

        if len(results["mix_results"]) != 3:
            raise NotImplementedError(
                "Unsupported base transform 'Mosaic': expected exactly 3 auxiliary images."
            )

        locs = ("top_left", "top_right", "bottom_left", "bottom_right")
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])

        if len(results["img"].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results["img"].dtype,
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results["img"].dtype,
            )

        mosaic_bboxes = []
        mosaic_labels = []
        mosaic_match_indices = []
        patches = [results] + list(results["mix_results"])

        for loc, patch in zip(locs, patches):
            patch = copy.deepcopy(patch)
            img_i = patch["img"]
            h_i, w_i = img_i.shape[:2]
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
            img_i = mmcv.imresize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            paste_coord, crop_coord = self._mosaic_combine(loc, (center_x, center_y), img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            gt_bboxes_i = patch.get("gt_bboxes")
            gt_labels_i = patch.get("gt_labels")
            gt_match_indices_i = patch.get("gt_match_indices")
            if gt_bboxes_i is None or gt_labels_i is None:
                continue

            gt_bboxes_i = gt_bboxes_i.copy()
            if gt_bboxes_i.shape[0] > 0:
                pad_w = x1_p - x1_c
                pad_h = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = scale_ratio_i * gt_bboxes_i[:, 0::2] + pad_w
                gt_bboxes_i[:, 1::2] = scale_ratio_i * gt_bboxes_i[:, 1::2] + pad_h

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(_maybe_copy_array(gt_labels_i))
            if gt_match_indices_i is not None:
                mosaic_match_indices.append(_maybe_copy_array(gt_match_indices_i))

        if mosaic_bboxes:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, axis=0)
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
            mosaic_match_indices = (
                np.concatenate(mosaic_match_indices, axis=0) if mosaic_match_indices else None
            )

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0, 2 * self.img_scale[1])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0, 2 * self.img_scale[0])

            if not self.skip_filter:
                filtered = self._filter_box_candidates(mosaic_bboxes, mosaic_labels, mosaic_match_indices)
                mosaic_bboxes = filtered[0]
                mosaic_labels = filtered[1]
                mosaic_match_indices = filtered[2] if len(filtered) > 2 else None

            inside_inds = find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0], 2 * self.img_scale[1])
            mosaic_bboxes = mosaic_bboxes[inside_inds]
            mosaic_labels = mosaic_labels[inside_inds]
            if mosaic_match_indices is not None:
                mosaic_match_indices = mosaic_match_indices[inside_inds]
        else:
            mosaic_bboxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_labels = np.zeros((0,), dtype=np.int64)
            mosaic_match_indices = None

        results["img"] = mosaic_img
        results["img_shape"] = mosaic_img.shape
        results["pad_shape"] = mosaic_img.shape
        results["gt_bboxes"] = mosaic_bboxes
        results["gt_labels"] = mosaic_labels
        if mosaic_match_indices is not None:
            results["gt_match_indices"] = mosaic_match_indices
        return results


@PIPELINES.register_module(force=True)
class RandomAffine:
    def __init__(
        self,
        scaling_ratio_range=(0.5, 1.5),
        max_rotate_degree=10.0,
        max_shear_degree=2.0,
        max_translate_ratio=0.1,
        border=(0, 0),
        border_val=(114, 114, 114),
        bbox_clip_border=True,
        skip_filter=False,
        min_bbox_size=2,
        **kwargs,
    ):
        self.scaling_ratio_range = scaling_ratio_range
        self.max_rotate_degree = max_rotate_degree
        self.max_shear_degree = max_shear_degree
        self.max_translate_ratio = max_translate_ratio
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.min_bbox_size = min_bbox_size

    @staticmethod
    def _get_rotation_matrix(rotate_degrees, width, height):
        radian = np.radians(rotate_degrees)
        center_x, center_y = width / 2, height / 2
        rotation_matrix = np.array(
            [
                [np.cos(radian), -np.sin(radian), 0.0],
                [np.sin(radian), np.cos(radian), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        translate_to_origin = np.array(
            [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        translate_back = np.array(
            [[1.0, 0.0, center_x], [0.0, 1.0, center_y], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return translate_back @ rotation_matrix @ translate_to_origin

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        return np.array([[scale_ratio, 0.0, 0.0], [0.0, scale_ratio, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    @staticmethod
    def _get_shear_matrix(x_degree, y_degree):
        x_radian = np.tan(np.radians(x_degree))
        y_radian = np.tan(np.radians(y_degree))
        return np.array([[1.0, x_radian, 0.0], [y_radian, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    @staticmethod
    def _get_translation_matrix(translate_x, translate_y):
        return np.array([[1.0, 0.0, translate_x], [0.0, 1.0, translate_y], [0.0, 0.0, 1.0]], dtype=np.float32)

    def filter_gt_bboxes(self, orig_bboxes, warp_bboxes):
        if orig_bboxes.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        widths = warp_bboxes[:, 2] - warp_bboxes[:, 0]
        heights = warp_bboxes[:, 3] - warp_bboxes[:, 1]
        return (widths > self.min_bbox_size) & (heights > self.min_bbox_size)

    def __call__(self, results):
        img = results["img"]
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        rotation_degree = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree, width, height)

        scaling_ratio = random.uniform(self.scaling_ratio_range[0], self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        trans_x = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
        img = cv2.warpPerspective(img, warp_matrix, dsize=(width, height), borderValue=self.border_val)
        results["img"] = img
        results["img_shape"] = img.shape
        results["pad_shape"] = img.shape

        for key in results.get("bbox_fields", []):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            if not num_bboxes:
                continue

            xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
            ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
            ones = np.ones_like(xs)
            points = np.vstack([xs, ys, ones])

            warp_points = warp_matrix @ points
            warp_points = warp_points[:2] / warp_points[2]
            xs = warp_points[0].reshape(num_bboxes, 4)
            ys = warp_points[1].reshape(num_bboxes, 4)
            warp_bboxes = np.vstack((xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

            if self.bbox_clip_border:
                warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)

            valid_index = find_inside_bboxes(warp_bboxes, height, width)
            if not self.skip_filter:
                valid_index = valid_index & self.filter_gt_bboxes(bboxes * scaling_ratio, warp_bboxes)

            results[key] = warp_bboxes[valid_index]
            if key == "gt_bboxes":
                if "gt_labels" in results:
                    results["gt_labels"] = results["gt_labels"][valid_index]
                if "gt_match_indices" in results:
                    results["gt_match_indices"] = results["gt_match_indices"][valid_index]

            if "gt_masks" in results:
                raise NotImplementedError("Unsupported base transform 'RandomAffine': gt_masks are not supported.")
        return results


@PIPELINES.register_module(force=True)
class MixUp:
    def __init__(
        self,
        dynamic_scale=(640, 640),
        ratio_range=(0.5, 1.5),
        flip_ratio=0.5,
        pad_val=114.0,
        bbox_clip_border=True,
        skip_filter=False,
        min_bbox_size=2,
        **kwargs,
    ):
        self.dynamic_scale = dynamic_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.min_bbox_size = min_bbox_size

    def get_indexes(self, dataset):
        return [random.randint(0, len(dataset) - 1)]

    def _filter_box_candidates(self, orig_bboxes, new_bboxes):
        if orig_bboxes.shape[0] == 4:
            new_boxes = new_bboxes.T
        else:
            new_boxes = new_bboxes
        widths = new_boxes[:, 2] - new_boxes[:, 0]
        heights = new_boxes[:, 3] - new_boxes[:, 1]
        return (widths > self.min_bbox_size) & (heights > self.min_bbox_size)

    def _mixup_transform(self, results):
        retrieve_results = results["mix_results"][0]
        if retrieve_results["gt_bboxes"].shape[0] == 0:
            return results

        retrieve_img = retrieve_results["img"]
        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones((self.dynamic_scale[0], self.dynamic_scale[1], 3), dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        scale_ratio = min(
            self.dynamic_scale[0] / retrieve_img.shape[0],
            self.dynamic_scale[1] / retrieve_img.shape[1],
        )
        retrieve_img = mmcv.imresize(
            retrieve_img,
            (int(retrieve_img.shape[1] * scale_ratio), int(retrieve_img.shape[0] * scale_ratio)),
        )
        out_img[: retrieve_img.shape[0], : retrieve_img.shape[1]] = retrieve_img

        scale_ratio *= jit_factor
        out_img = mmcv.imresize(
            out_img,
            (int(out_img.shape[1] * jit_factor), int(out_img.shape[0] * jit_factor)),
        )

        if is_flip:
            out_img = out_img[:, ::-1, :]

        ori_img = results["img"]
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset = random.randint(0, padded_img.shape[1] - target_w) if padded_img.shape[1] > target_w else 0
        y_offset = random.randint(0, padded_img.shape[0] - target_h) if padded_img.shape[0] > target_h else 0
        padded_cropped_img = padded_img[y_offset : y_offset + target_h, x_offset : x_offset + target_w]

        retrieve_gt_bboxes = retrieve_results["gt_bboxes"].copy()
        retrieve_gt_bboxes[:, 0::2] *= scale_ratio
        retrieve_gt_bboxes[:, 1::2] *= scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_bboxes[:, 0::2] = np.clip(retrieve_gt_bboxes[:, 0::2], 0, origin_w)
            retrieve_gt_bboxes[:, 1::2] = np.clip(retrieve_gt_bboxes[:, 1::2], 0, origin_h)

        if is_flip:
            retrieve_gt_bboxes[:, 0::2] = origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1]

        clipped_bboxes = retrieve_gt_bboxes.copy()
        clipped_bboxes[:, 0::2] -= x_offset
        clipped_bboxes[:, 1::2] -= y_offset
        if self.bbox_clip_border:
            clipped_bboxes[:, 0::2] = np.clip(clipped_bboxes[:, 0::2], 0, target_w)
            clipped_bboxes[:, 1::2] = np.clip(clipped_bboxes[:, 1::2], 0, target_h)

        mixup_img = 0.5 * ori_img.astype(np.float32) + 0.5 * padded_cropped_img.astype(np.float32)
        retrieve_gt_labels = retrieve_results["gt_labels"]
        retrieve_gt_match_indices = retrieve_results.get("gt_match_indices")

        if not self.skip_filter:
            keep_list = self._filter_box_candidates(retrieve_gt_bboxes.T, clipped_bboxes.T)
            retrieve_gt_labels = retrieve_gt_labels[keep_list]
            clipped_bboxes = clipped_bboxes[keep_list]
            if retrieve_gt_match_indices is not None:
                retrieve_gt_match_indices = retrieve_gt_match_indices[keep_list]

        mixup_gt_bboxes = np.concatenate((results["gt_bboxes"], clipped_bboxes), axis=0)
        mixup_gt_labels = np.concatenate((results["gt_labels"], retrieve_gt_labels), axis=0)
        if retrieve_gt_match_indices is not None and "gt_match_indices" in results:
            mixup_gt_match_indices = np.concatenate((results["gt_match_indices"], retrieve_gt_match_indices), axis=0)
        else:
            mixup_gt_match_indices = None

        inside_inds = find_inside_bboxes(mixup_gt_bboxes, target_h, target_w)
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_labels = mixup_gt_labels[inside_inds]
        if mixup_gt_match_indices is not None:
            mixup_gt_match_indices = mixup_gt_match_indices[inside_inds]

        results["img"] = mixup_img.astype(np.uint8)
        results["img_shape"] = mixup_img.shape
        results["pad_shape"] = mixup_img.shape
        results["gt_bboxes"] = mixup_gt_bboxes
        results["gt_labels"] = mixup_gt_labels
        if mixup_gt_match_indices is not None:
            results["gt_match_indices"] = mixup_gt_match_indices
        return results

    def __call__(self, results):
        if "mix_results" not in results:
            raise NotImplementedError(
                "Unsupported base transform 'MixUp': expected 'mix_results' from a dataset wrapper."
            )
        if len(results["mix_results"]) != 1:
            raise NotImplementedError(
                "Unsupported base transform 'MixUp': expected exactly 1 auxiliary image."
            )
        return self._mixup_transform(results)


@PIPELINES.register_module(force=True)
class YOLOXHSVRandomAug:
    def __init__(self, hgain=5, sgain=30, vgain=30):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, results):
        img = results["img"]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-self.hgain, self.hgain)) % 180
        hsv[..., 1] = np.clip(
            hsv[..., 1] * random.uniform(1 - self.sgain / 255, 1 + self.sgain / 255), 0, 255
        )
        hsv[..., 2] = np.clip(
            hsv[..., 2] * random.uniform(1 - self.vgain / 255, 1 + self.vgain / 255), 0, 255
        )
        results["img"] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return results


@PIPELINES.register_module(force=True)
class MultiScaleFlipAug:
    def __init__(self, img_scale, flip=False, transforms=None):
        self.img_scale = img_scale
        self.flip = flip
        self.transforms = Compose(transforms or [])

    def __call__(self, results):
        results = copy.deepcopy(results)
        results["scale"] = self.img_scale
        results["flip"] = self.flip
        results["flip_direction"] = "horizontal" if self.flip else None
        outputs = self.transforms(results)
        return {key: [value] for key, value in outputs.items()}


__all__ = [
    "Collect",
    "Compose",
    "CutOut",
    "DefaultFormatBundle",
    "FilterAnnotations",
    "ImageToTensor",
    "LoadAnnotations",
    "LoadImageFromFile",
    "MixUp",
    "Mosaic",
    "MultiScaleFlipAug",
    "Normalize",
    "Pad",
    "RandomAffine",
    "RandomFlip",
    "Resize",
    "YOLOXHSVRandomAug",
    "to_tensor",
]
