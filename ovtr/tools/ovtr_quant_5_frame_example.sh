#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_VARIANT=5_frame exec "${SCRIPT_DIR}/ovtr_quant_full_model.sh" "$@"
