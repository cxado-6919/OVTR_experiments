import inspect

import torch
import torch.distributed as dist

from mmcv.parallel import DataContainer

try:
    from detectron2.structures import Instances
except ImportError:  # pragma: no cover
    Instances = tuple()


def _unwrap_batch(data):
    if isinstance(data, DataContainer):
        return _unwrap_batch(data.data)
    if isinstance(data, dict):
        return {key: _unwrap_batch(value) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_unwrap_batch(value) for value in data)
    if isinstance(data, list):
        return [_unwrap_batch(value) for value in data]
    return data


def _model_device(model):
    module = model.module if hasattr(model, "module") else model
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if Instances and isinstance(data, Instances):
        return data.to(device)
    if isinstance(data, dict):
        return {key: _move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_move_to_device(value, device) for value in data)
    if isinstance(data, list):
        return [_move_to_device(value, device) for value in data]
    return data


def _infer_batch_size(data):
    if isinstance(data, dict):
        if "img" in data and isinstance(data["img"], torch.Tensor):
            return int(data["img"].shape[0])
        if "imgs" in data and isinstance(data["imgs"], list):
            return len(data["imgs"])
        for value in data.values():
            size = _infer_batch_size(value)
            if size is not None:
                return size
    elif isinstance(data, (list, tuple)):
        if data and isinstance(data[0], torch.Tensor):
            return int(data[0].shape[0])
        for value in data:
            size = _infer_batch_size(value)
            if size is not None:
                return size
    return None


def _call_model(model, data):
    module = model.module if hasattr(model, "module") else model
    forward_sig = inspect.signature(module.forward)
    param_names = [name for name in forward_sig.parameters.keys() if name != "self"]

    if isinstance(data, dict):
        if "imgs" in data or "gt_instances" in data:
            return model(data)

        if "img" in data or "img_metas" in data:
            for kwargs in (
                {"return_loss": False, "rescale": True, **data},
                data,
            ):
                try:
                    return model(**kwargs)
                except TypeError:
                    continue

        if len(param_names) == 1:
            return model(data)

        try:
            return model(**data)
        except TypeError:
            return model(data)

    if isinstance(data, tuple):
        try:
            return model(*data)
        except TypeError:
            return model(data)

    if isinstance(data, list):
        try:
            return model(*data)
        except TypeError:
            return model(data)

    return model(data)


def _run_local_inference(model, data_loader):
    device = _model_device(model)
    results = []
    for data in data_loader:
        batch = _move_to_device(_unwrap_batch(data), device)
        outputs = _call_model(model, batch)
        batch_size = _infer_batch_size(batch)
        if isinstance(outputs, list) and batch_size is not None and len(outputs) == batch_size:
            results.extend(outputs)
        elif isinstance(outputs, tuple) and batch_size is not None and len(outputs) == batch_size:
            results.extend(list(outputs))
        else:
            results.append(outputs)
    return results


def single_gpu_test(model, data_loader, show=False):
    del show
    model.eval()
    with torch.no_grad():
        return _run_local_inference(model, data_loader)


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    del tmpdir, gpu_collect
    model.eval()
    with torch.no_grad():
        local_results = _run_local_inference(model, data_loader)

    if not (dist.is_available() and dist.is_initialized()):
        return local_results

    world_size = dist.get_world_size()
    if world_size == 1:
        return local_results

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_results)
    if dist.get_rank() == 0:
        merged = []
        for part in gathered:
            merged.extend(part or [])
        dataset_len = len(data_loader.dataset) if hasattr(data_loader, "dataset") else None
        return merged[:dataset_len] if dataset_len is not None else merged
    return None
