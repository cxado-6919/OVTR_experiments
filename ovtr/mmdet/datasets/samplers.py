import itertools
import math
import random

from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler


class DistributedSampler(TorchDistributedSampler):
    pass


class GroupSampler(Sampler):
    def __init__(self, dataset, samples_per_gpu):
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


class DistributedGroupSampler(DistributedSampler):
    def __init__(self, dataset, samples_per_gpu, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.samples_per_gpu = samples_per_gpu


class InfiniteBatchSampler(Sampler):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            iterator = iter(self.sampler)
            while True:
                batch = list(itertools.islice(iterator, self.batch_size))
                if len(batch) < self.batch_size:
                    break
                yield batch

    def __len__(self):
        return math.inf


class InfiniteGroupBatchSampler(InfiniteBatchSampler):
    pass
