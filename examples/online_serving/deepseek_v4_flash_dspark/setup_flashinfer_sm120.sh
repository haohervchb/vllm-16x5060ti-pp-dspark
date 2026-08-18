#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
FLASHINFER_COMMIT="24d7dfb2639083c5a4d418881099421fc800b7bb"
FLASHINFER_SRC="${FLASHINFER_SRC:-${REPO_ROOT}/../flashinfer-dsv4}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing vLLM environment: ${VENV_PYTHON}" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "The uv executable is required to install FlashInfer." >&2
  exit 1
fi

site_packages="$(${VENV_PYTHON} - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
export CUDA_HOME="${CUDA_HOME:-${site_packages}/nvidia/cu13}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0f}"

if [[ ! -d "${FLASHINFER_SRC}/.git" ]]; then
  git clone https://github.com/flashinfer-ai/flashinfer.git "${FLASHINFER_SRC}"
fi

git -C "${FLASHINFER_SRC}" fetch origin "${FLASHINFER_COMMIT}"
git -C "${FLASHINFER_SRC}" checkout --detach "${FLASHINFER_COMMIT}"

BUILD_NVEP=0 uv pip install \
  --python "${VENV_PYTHON}" \
  --no-deps \
  --reinstall \
  "${FLASHINFER_SRC}"

"${VENV_PYTHON}" - <<'PY'
from importlib.metadata import version

from flashinfer.mla._sparse_mla_sm120 import _DECODE_DSV4_DISPATCH

required = {(8, 192), (16, 192)}
missing = required.difference(_DECODE_DSV4_DISPATCH)
if missing:
    raise SystemExit(f"FlashInfer is missing required SM120 dispatch shapes: {missing}")
print(f"FlashInfer {version('flashinfer-python')} supports {sorted(required)}")
PY

installed_source="$(${VENV_PYTHON} - <<'PY'
import flashinfer
print(flashinfer.__file__)
PY
)"
echo "Imported FlashInfer from: ${installed_source}"
echo "Pinned source commit: $(git -C "${FLASHINFER_SRC}" rev-parse HEAD)"
