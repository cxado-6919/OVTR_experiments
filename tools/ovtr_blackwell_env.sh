fail() {
  echo "[ovtr_blackwell_env] $1" >&2
  return 1 2>/dev/null || exit 1
}

prepend_path() {
  value="$1"
  current="${2:-}"
  if [ -z "${current}" ]; then
    printf '%s' "${value}"
  else
    printf '%s:%s' "${value}" "${current}"
  fi
}

[ -n "${CONDA_PREFIX:-}" ] || fail "No active conda env."

nvcc_path=""
for candidate in "${CONDA_PREFIX}/bin/nvcc" "${CONDA_PREFIX}/targets/x86_64-linux/bin/nvcc"; do
  if [ -x "${candidate}" ]; then
    nvcc_path="${candidate}"
    break
  fi
done
[ -n "${nvcc_path}" ] || fail "Could not find nvcc under ${CONDA_PREFIX}"

gcc_path=""
gxx_path=""
for candidate in "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" "${CONDA_PREFIX}/bin/gcc"; do
  if [ -x "${candidate}" ]; then
    gcc_path="${candidate}"
    break
  fi
done
for candidate in "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++" "${CONDA_PREFIX}/bin/g++"; do
  if [ -x "${candidate}" ]; then
    gxx_path="${candidate}"
    break
  fi
done

[ -n "${gcc_path}" ] || fail "Missing gcc under ${CONDA_PREFIX}"
[ -n "${gxx_path}" ] || fail "Missing g++ under ${CONDA_PREFIX}"

cuda_target_dir=""
if [ -d "${CONDA_PREFIX}/targets/x86_64-linux" ]; then
  cuda_target_dir="${CONDA_PREFIX}/targets/x86_64-linux"
else
  cuda_target_dir="${CONDA_PREFIX}"
fi

export CUDA_HOME="${CONDA_PREFIX}"
export CUDA_PATH="${CONDA_PREFIX}"
export OVTR_CUDA_TARGET_DIR="${cuda_target_dir}"
export CUDACXX="${nvcc_path}"
export CC="${gcc_path}"
export CXX="${gxx_path}"
export CUDAHOSTCXX="${gxx_path}"
export TORCH_CUDA_ARCH_LIST="12.0+PTX"

if [ -d "${CONDA_PREFIX}/bin" ]; then
  PATH="$(prepend_path "${CONDA_PREFIX}/bin" "${PATH:-}")"
fi
if [ -d "${cuda_target_dir}/bin" ]; then
  PATH="$(prepend_path "${cuda_target_dir}/bin" "${PATH}")"
fi
export PATH

ld_paths=""
for candidate in "${cuda_target_dir}/lib64" "${cuda_target_dir}/lib" "${CONDA_PREFIX}/lib"; do
  if [ -d "${candidate}" ]; then
    ld_paths="$(prepend_path "${candidate}" "${ld_paths}")"
  fi
done
if [ -n "${ld_paths}" ]; then
  export LD_LIBRARY_PATH="$(prepend_path "${ld_paths}" "${LD_LIBRARY_PATH:-}")"
fi

echo "python=$(command -v python)"
echo "pip=$(command -v pip)"
echo "nvcc=${CUDACXX}"
echo "gcc=${CC}"
echo "g++=${CXX}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "OVTR_CUDA_TARGET_DIR=${OVTR_CUDA_TARGET_DIR}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
