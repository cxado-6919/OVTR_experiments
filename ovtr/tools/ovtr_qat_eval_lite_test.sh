#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_VARIANT=lite EVAL_SPLIT=test exec "${SCRIPT_DIR}/ovtr_qat_eval.sh" "$@"
