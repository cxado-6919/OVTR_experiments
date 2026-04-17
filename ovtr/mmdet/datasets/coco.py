import os

import numpy as np

from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module(force=True)
class CocoDataset:
    CLASSES = None

    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        **kwargs,
    ):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.proposals = None

        if classes is not None:
            self.CLASSES = self.get_classes(classes)

        self.data_infos = self.load_annotations(self.ann_file)
        if not self.test_mode and hasattr(self, "_filter_imgs"):
            valid_inds = self._filter_imgs()
            if valid_inds is not None:
                self.data_infos = [self.data_infos[i] for i in valid_inds]
                if hasattr(self, "img_ids"):
                    self.img_ids = [self.img_ids[i] for i in valid_inds]

        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
        self.pipeline = Compose(pipeline)

    def get_classes(self, classes):
        if isinstance(classes, str):
            with open(classes, "r", encoding="utf-8") as handle:
                return tuple(line.strip() for line in handle if line.strip())
        if isinstance(classes, (list, tuple)):
            return tuple(classes)
        raise TypeError(f"Unsupported classes spec: {type(classes)}")

    def load_annotations(self, ann_file):
        raise NotImplementedError

    def pre_pipeline(self, results):
        results["img_prefix"] = self.img_prefix
        results["seg_prefix"] = self.seg_prefix
        results["proposal_file"] = self.proposal_file
        results.setdefault("bbox_fields", [])
        results.setdefault("mask_fields", [])
        results.setdefault("seg_fields", [])
        return results

    @staticmethod
    def xyxy2xywh(bbox):
        x1, y1, x2, y2 = bbox[:4]
        return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

    def __len__(self):
        return len(self.data_infos)
