from mmcv.utils import Registry


DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")

__all__ = ["DATASETS", "PIPELINES"]
