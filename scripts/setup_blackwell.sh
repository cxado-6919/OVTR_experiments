#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url "${TORCH_CUDA_INDEX_URL}"
python -m pip install -r requirements.txt

export TORCH_CUDA_ARCH_LIST

pushd ovtr/models/ops >/dev/null
python setup.py build_ext --inplace
popd >/dev/null

pushd ovtr_det_bs2_pretrain/models/ops >/dev/null
python setup.py build_ext --inplace
popd >/dev/null

echo "Environment ready in ${VENV_DIR}"
