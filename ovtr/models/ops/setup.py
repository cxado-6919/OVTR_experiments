# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import glob
import os

import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension


DEFAULT_CUDA_ARCH_LIST = "8.0;8.6;8.9;9.0;12.0+PTX"


def _detect_arch_list():
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return os.environ["TORCH_CUDA_ARCH_LIST"]

    if torch.cuda.is_available():
        archs = sorted(
            {f"{major}.{minor}" for major, minor in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))},
            key=lambda item: tuple(int(part) for part in item.split(".")),
        )
        if archs:
            archs[-1] = f"{archs[-1]}+PTX"
            return ";".join(archs)

    return DEFAULT_CUDA_ARCH_LIST

def get_extensions():
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA toolkit was not found. Install a CUDA-enabled PyTorch environment with nvcc "
            "to build the MultiScaleDeformableAttention extension, or rely on the slower "
            "pure-PyTorch fallback by skipping this build step."
        )

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _detect_arch_list())

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu + source_cuda
    include_dirs = [extensions_dir]
    return [
        CUDAExtension(
            name="_C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-lineinfo",
                ],
            },
        )
    ]

setup(
    name="ovtr-msda",
    version="2.0",
    description="OVTR package-local CUDA extension for multi-scale deformable attention",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
