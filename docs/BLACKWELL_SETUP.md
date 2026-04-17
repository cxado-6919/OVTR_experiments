# Modern / Blackwell Setup

This repository has been ported to a modern Linux stack that is suitable for NVIDIA Blackwell GPUs such as RTX 5090 and RTX PRO 6000 Blackwell.

## Target Stack

- Ubuntu 22.04 or 24.04
- Python 3.11
- PyTorch 2.7.1
- torchvision 0.22.1
- torchaudio 2.7.1
- CUDA-enabled wheel channel: `cu128`
- Custom CUDA extensions built locally with a CUDA 12.8-capable toolkit / driver stack

PyTorch 2.7 is the first PyTorch release that explicitly announced Blackwell support and CUDA 12.8 wheels. This port targets `torch==2.7.1` rather than the original `torch 1.10.1 + cu111` stack because the legacy stack is not realistic on Blackwell hardware.

Sources:

- [PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/)
- [PyTorch previous versions install matrix](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-tuning-guide/index.html)

## Create The Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install \
  torch==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt
```

Optional helper:

```bash
bash scripts/setup_blackwell.sh
```

## Build The Custom CUDA Ops

The two duplicated op trees are now package-local in-place extensions. Build both of them:

```bash
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"

cd ovtr/models/ops
python setup.py build_ext --inplace

cd ../../../ovtr_det_bs2_pretrain/models/ops
python setup.py build_ext --inplace
```

Notes:

- `TORCH_CUDA_ARCH_LIST` is respected if you set it yourself.
- If you do not set it, the setup scripts try to derive an architecture list from visible GPUs and otherwise fall back to a modern list that includes Blackwell (`12.0+PTX`).
- The extension is no longer installed as a single global `MultiScaleDeformableAttention` module; each tree now builds its own local `_C` extension in place.

## Optional CLIP Dependency

Core training / eval now only require the precomputed embedding files. Regenerating CLIP embeddings is optional.

If you want to regenerate CLIP text or image embeddings, install either:

```bash
python -m pip install open-clip-torch
```

or:

```bash
python -m pip install git+https://github.com/openai/CLIP.git
```

## Smoke Tests

The smallest repo-local smoke tests do not require the full dataset:

```bash
cd ovtr
python tools/smoke_test.py

cd ../ovtr_det_bs2_pretrain
python tools/smoke_test.py
```

These tests verify:

- Python import path health
- config parsing
- `MultiScaleDeformableAttention` import
- a minimal forward pass through the attention module
- optional CUDA-path attention execution when CUDA is available

## Example Entry Points

Training:

```bash
cd ovtr
bash tools/ovtr_multi_frame_lite_train.sh
```

Evaluation:

```bash
cd ovtr
bash tools/ovtr_ovmot_eval_lite_val.sh
```

Detection pretraining:

```bash
cd ovtr_det_bs2_pretrain
bash tools/ovtr_detection_pretrain.sh
```

## Troubleshooting

- If the CUDA extension build says CUDA is unavailable, confirm that:
  - the PyTorch environment is the CUDA wheel build, not CPU-only
  - `nvcc` from a modern CUDA toolkit is installed and visible
  - the NVIDIA driver is new enough for CUDA 12.8 / Blackwell
- If the extension build fails on architectures:
  - set `TORCH_CUDA_ARCH_LIST=12.0+PTX` explicitly on Blackwell
- If you skip building the extension:
  - the repo can still use the pure-PyTorch deformable attention fallback, but it will be slower
