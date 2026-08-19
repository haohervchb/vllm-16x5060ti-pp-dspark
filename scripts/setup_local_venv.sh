#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

command -v uv >/dev/null 2>&1 || {
  echo "error: uv is required; install uv first" >&2
  exit 1
}

# A fresh clone may inherit CUDA_HOME from an older deleted .venv. Never let
# setuptools probe a path that does not exist during bootstrap.
unset CUDA_HOME CUDACXX

if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.12
fi

# vLLM's build metadata invokes nvcc before the editable install can resolve
# runtime dependencies, so install a complete, internally matched CUDA 13.0
# toolkit into the venv first. CUDA 13.0 matches the torch==2.13.0 build used by
# this fork and supports SM120/compute_120f.
uv pip install --python .venv/bin/python 'cuda-toolkit[all]==13.0.3'

site_packages=$(.venv/bin/python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)
export CUDA_HOME="$site_packages/nvidia/cu13"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "error: CUDA bootstrap did not provide $CUDA_HOME/bin/nvcc" >&2
  exit 1
fi

"$CUDA_HOME/bin/nvcc" --version

# Install the build backend requirements into the target venv, then disable
# build isolation so setuptools uses the CUDA toolkit we just bootstrapped.
uv pip install --python .venv/bin/python -r requirements/build/cuda.txt
uv pip install --python .venv/bin/python --no-build-isolation -e .

echo
echo "vLLM editable environment created at $repo_root/.venv"
echo "CUDA_HOME=$CUDA_HOME"
echo "Next: examples/online_serving/deepseek_v4_flash_dspark/setup_flashinfer_sm120.sh"
