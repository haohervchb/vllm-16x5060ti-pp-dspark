#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

command -v uv >/dev/null 2>&1 || {
  echo "error: uv is required; install uv first" >&2
  exit 1
}

# Never inherit a CUDA_HOME that pointed at a deleted checkout/venv.
unset CUDA_HOME CUDACXX CUDA_PATH CUDA_BIN_PATH

if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.12
fi

# This fork builds against CUDA 13.0 / torch 2.13. Install the complete Python
# CUDA toolkit first because vLLM's build metadata probes nvcc before the
# editable install can resolve its own runtime dependencies.
uv pip install --python .venv/bin/python 'cuda-toolkit[all]==13.0.3'

site_packages=$(.venv/bin/python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)

export CUDA_HOME="$site_packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export CUDA_BIN_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "error: CUDA bootstrap did not provide $CUDA_HOME/bin/nvcc" >&2
  exit 1
fi

# NVIDIA's pip CUDA packages use a split layout: nvcc/base headers are under
# nvidia/cu13 while component libraries/headers are spread across sibling
# nvidia/* directories. PyTorch still calls its bundled legacy FindCUDA.cmake,
# which clears user-provided CUDA_CUDART_LIBRARY on the first toolkit-root
# change and then searches for ordinary linker names such as libcudart.so under
# <CUDA_HOME>/lib64. Build a conventional compatibility view so both legacy
# FindCUDA and modern FindCUDAToolkit can discover the installed components.
mkdir -p "$CUDA_HOME/lib" "$CUDA_HOME/lib64" "$CUDA_HOME/include"

while IFS= read -r lib; do
  base=$(basename "$lib")
  ln -sf "$lib" "$CUDA_HOME/lib/$base"
  ln -sf "$lib" "$CUDA_HOME/lib64/$base"

  # Python CUDA wheels may only ship SONAME files (e.g. libcudart.so.13).
  # find_library(NAMES cudart) looks for the unversioned linker name.
  if [[ "$base" =~ ^(lib.+\.so)\.[0-9].*$ ]]; then
    linker_name=${BASH_REMATCH[1]}
    ln -sf "$lib" "$CUDA_HOME/lib/$linker_name"
    ln -sf "$lib" "$CUDA_HOME/lib64/$linker_name"
  fi
done < <(find "$site_packages/nvidia" -type f \
  \( -path '*/lib/lib*.so*' -o -path '*/lib/lib*.a' \) -print)

# Make component headers visible from the conventional toolkit include root.
while IFS= read -r header; do
  case "$header" in
    "$CUDA_HOME/include/"*) continue ;;
  esac
  rel=${header#*/include/}
  mkdir -p "$CUDA_HOME/include/$(dirname "$rel")"
  ln -sf "$header" "$CUDA_HOME/include/$rel"
done < <(find "$site_packages/nvidia" -type f -path '*/include/*' -print)

cudart="$CUDA_HOME/lib64/libcudart.so"
if [[ ! -f "$cudart" ]]; then
  echo "error: failed to create legacy-compatible libcudart linker name" >&2
  find "$site_packages/nvidia" -name 'libcudart.so*' -print >&2 || true
  exit 1
fi

export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_LIBRARY_PATH="$CUDA_HOME/lib64${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
export CMAKE_INCLUDE_PATH="$CUDA_HOME/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
export CMAKE_PREFIX_PATH="$CUDA_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# setup.py appends CMAKE_ARGS to the configure command. The explicit values are
# useful for modern FindCUDAToolkit; the conventional lib/include view above is
# what makes PyTorch's legacy FindCUDA re-discovery succeed after it clears its
# cache variables.
export CMAKE_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCUDAToolkit_ROOT=$CUDA_HOME -DCUDA_CUDART_LIBRARY=$cudart ${CMAKE_ARGS:-}"

echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_CUDART_LIBRARY=$cudart -> $(readlink -f "$cudart")"
"$CUDA_HOME/bin/nvcc" --version

uv pip install --python .venv/bin/python -r requirements/build/cuda.txt
uv pip install --python .venv/bin/python --no-build-isolation -e .

echo
echo "vLLM editable environment created at $repo_root/.venv"
echo "Next: examples/online_serving/deepseek_v4_flash_dspark/setup_flashinfer_sm120.sh"
