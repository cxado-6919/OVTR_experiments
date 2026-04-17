import math

from torch.utils.data import ConcatDataset as TorchConcatDataset


class ConcatDataset(TorchConcatDataset):
    def __init__(self, datasets, separate_eval=True):
        super().__init__(datasets)
        self.separate_eval = separate_eval


class RepeatDataset:
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = getattr(dataset, "CLASSES", None)
        self.flag = getattr(dataset, "flag", None)

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

    def __len__(self):
        return len(self.dataset) * self.times


class ClassBalancedDataset(RepeatDataset):
    def __init__(self, dataset, oversample_thr):
        super().__init__(dataset, times=max(1, math.ceil(1 / max(oversample_thr, 1e-6))))
        self.oversample_thr = oversample_thr


class MultiImageMixDataset:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.CLASSES = getattr(dataset, "CLASSES", None)
        self.flag = getattr(dataset, "flag", None)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
