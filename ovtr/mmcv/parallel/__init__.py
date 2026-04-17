from torch.utils.data._utils.collate import default_collate

from .data_container import DataContainer


def collate(batch, samples_per_gpu=1):
    if not batch:
        return batch

    first = batch[0]
    if isinstance(first, DataContainer):
        if first.cpu_only:
            return DataContainer([sample.data for sample in batch], cpu_only=True)
        data = [sample.data for sample in batch]
        if first.stack:
            return DataContainer(default_collate(data), stack=True, padding_value=first.padding_value)
        return DataContainer(data, stack=False, padding_value=first.padding_value)

    if isinstance(first, dict):
        return {key: collate([sample[key] for sample in batch], samples_per_gpu) for key in first}

    if isinstance(first, (list, tuple)):
        transposed = zip(*batch)
        return [collate(list(items), samples_per_gpu) for items in transposed]

    return default_collate(batch)


__all__ = ["DataContainer", "collate"]
