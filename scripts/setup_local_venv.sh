#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

command -v uv >/dev/null 2>&1 || {
  echo "error: uv is required; install uv first" >&2
  exit 1
}

unset CUDA_HOME CUDACXX CUDA_PATH

if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.12
fi

uv pip install --python .venv/bin/python 'cuda-toolkit[all]==13.0.3'

site_packages=$(.venv/bin/python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)
export CUDA_HOME="$site_packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

mkdir -p "$CUDA_HOME/lib64"
while IFS= read -r lib; do
  ln -sf "$lib" "$CUDA_HOME/lib64/$(basename "$lib")"
done < <(find "$site_packages/nvidia" -type f -path '*/lib/lib*.so*' -print)

export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$CUDA_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "error: CUDA bootstrap did not provide $CUDA_HOME/bin/nvcc" >&2
  exit 1
fi

cudart=$(find "$site_packages/nvidia" -type f -name 'libcudart.so*' -print | head -n1)
if [[ -z "$cudart" || ! -f "$cudart" ]]; then
  echo "error: CUDA bootstrap did not provide libcudart.so" >&2
  exit 1
fi

# setup.py appends CMAKE_ARGS verbatim to its cmake invocation. Pip-installed
# CUDA has a nonstandard split layout, so tell both legacy FindCUDA and modern
# FindCUDAToolkit exactly where the toolkit and runtime library are.
export CMAKE_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCUDAToolkit_ROOT=$CUDA_HOME -DCUDA_CUDART_LIBRARY=$cudart ${CMAKE_ARGS:-}"

echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_CUDART_LIBRARY=$cudart"
"$CUDA_HOME/bin/nvcc" --version

uv pip install --python .venv/bin/python -r requirements/build/cuda.txt
uv pip install --python .venv/bin/python --no-build-isolation -e .

echo
echo "vLLM editable environment created at $repo_root/.venv"
echo "Next: examples/online_serving/deepseek_v4_flash_dspark/setup_flashinfer_sm120.sh"
