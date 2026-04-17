import json
import os
import pickle

import cv2
import numpy as np

from .fileio import BaseStorageBackend, FileClient
from .utils import (
    Registry,
    TORCH_VERSION,
    build_from_cfg,
    digit_version,
    mkdir_or_exist,
    print_log,
)


def dump(obj, file):
    ext = os.path.splitext(str(file))[1].lower()
    mkdir_or_exist(os.path.dirname(os.path.abspath(str(file))))
    if ext == ".json":
        with open(file, "w", encoding="utf-8") as handle:
            json.dump(obj, handle)
        return

    with open(file, "wb") as handle:
        pickle.dump(obj, handle)


def imread(img):
    return cv2.imread(img, cv2.IMREAD_COLOR)


def imwrite(img, file):
    mkdir_or_exist(os.path.dirname(os.path.abspath(file)))
    return cv2.imwrite(file, img)


def imresize(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bgr2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hsv2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def imshow(img, win_name="", wait_time=0):
    cv2.imshow(win_name or "mmcv", img)
    cv2.waitKey(wait_time)


__all__ = [
    "BaseStorageBackend",
    "FileClient",
    "Registry",
    "TORCH_VERSION",
    "bgr2hsv",
    "bgr2rgb",
    "build_from_cfg",
    "digit_version",
    "dump",
    "hsv2bgr",
    "imread",
    "imresize",
    "imshow",
    "imwrite",
    "mkdir_or_exist",
    "print_log",
]
