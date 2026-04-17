#!/usr/bin/env bash
set -euo pipefail

export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
python setup.py build_ext --inplace
