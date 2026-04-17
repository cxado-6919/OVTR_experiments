import os
from typing import Any, Dict, Optional

import torch


TORCH_VERSION = torch.__version__


def digit_version(version_str: str):
    core = version_str.split("+")[0]
    digits = []
    for part in core.split("."):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        digits.append(int(num or 0))
    return tuple(digits)


def mkdir_or_exist(dir_name: str):
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def print_log(msg: str, logger=None):
    if logger is None:
        print(msg)
        return
    if hasattr(logger, "info"):
        logger.info(msg)
        return
    print(msg)


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Any] = {}

    def get(self, key: str):
        return self._module_dict.get(key)

    def register_module(self, module=None, force=False, name: Optional[str] = None):
        def _register(obj):
            module_name = name or obj.__name__
            if not force and module_name in self._module_dict:
                raise KeyError(f"{module_name} is already registered in {self._name}")
            self._module_dict[module_name] = obj
            return obj

        if module is not None:
            return _register(module)
        return _register


def build_from_cfg(cfg: Dict[str, Any], registry: Registry, default_args: Optional[Dict[str, Any]] = None):
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "type" not in cfg:
        raise KeyError("`cfg` must contain the key `type`")

    args = cfg.copy()
    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not registered in {registry._name}")
    else:
        obj_cls = obj_type

    if default_args:
        for key, value in default_args.items():
            args.setdefault(key, value)

    return obj_cls(**args)
