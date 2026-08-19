#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
VENV_PYTHON="${VENV_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/.venv/bin/vllm}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash-0731}"
LAYOUT="${1:-}"

if [[ ! -x "${VENV_PYTHON}" || ! -x "${VLLM_BIN}" ]]; then
  echo "Expected the vLLM environment at ${REPO_ROOT}/.venv" >&2
  exit 2
fi

case "${LAYOUT}" in
  tp4pp4)
    TP=4
    PP=4
    PP_PARTITION="11,12,12,8"
    DEFAULT_GPU_UTILIZATION="0.90"
    ;;
  tp8pp2)
    TP=8
    PP=2
    PP_PARTITION="23,20"
    DEFAULT_GPU_UTILIZATION="0.92"
    ;;
  *)
    echo "Usage: $0 {tp4pp4|tp8pp2}" >&2
    exit 2
    ;;
esac

gpu_count="$(nvidia-smi -L | wc -l)"
if [[ "${gpu_count}" -ne 16 ]]; then
  echo "Expected 16 visible GPUs before launch; found ${gpu_count}." >&2
  exit 1
fi

bar1_count="$(nvidia-smi -q -d MEMORY | awk '
  /BAR1 Memory Usage/ { in_bar1 = 1; next }
  in_bar1 && /Total/ { if ($(NF - 1) + 0 >= 16000) good++; in_bar1 = 0 }
  END { print good + 0 }
')"
if [[ "${bar1_count}" -ne 16 ]]; then
  echo "Expected framebuffer-sized BAR1 on all 16 GPUs; found ${bar1_count}." >&2
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

"${VENV_PYTHON}" - <<'PY'
from flashinfer.mla._sparse_mla_sm120 import _DECODE_DSV4_DISPATCH

required = {(8, 192), (16, 192)}
missing = required.difference(_DECODE_DSV4_DISPATCH)
if missing:
    raise SystemExit(
        f"FlashInfer lacks required SM120 dispatch shapes {missing}; "
        "run setup_flashinfer_sm120.sh"
    )
PY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-PXB}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-${PP_PARTITION}}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"

ENABLE_PLX_CUSTOM_ALLREDUCE="${ENABLE_PLX_CUSTOM_ALLREDUCE:-1}"
if [[ "${ENABLE_PLX_CUSTOM_ALLREDUCE}" == 1 ]]; then
  export VLLM_ALLOW_PCI_CUSTOM_ALLREDUCE=1
  export VLLM_CUSTOM_ALLREDUCE_ALGO="${VLLM_CUSTOM_ALLREDUCE_ALGO:-2stage}"
  # cudaMalloc IPC registration used by custom all-reduce graph capture is not
  # compatible with expandable segments or cudaMallocAsync.
  export PYTORCH_CUDA_ALLOC_CONF=backend:native
elif [[ "${ENABLE_PLX_CUSTOM_ALLREDUCE}" == 0 ]]; then
  export VLLM_ALLOW_PCI_CUSTOM_ALLREDUCE=0
  unset VLLM_CUSTOM_ALLREDUCE_ALGO
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
else
  echo "ENABLE_PLX_CUSTOM_ALLREDUCE must be 0 or 1." >&2
  exit 2
fi

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${DEFAULT_GPU_UTILIZATION}}"
PROFILE="${PROFILE:-baseline}"
case "${PROFILE}" in
  baseline)
    DEFAULT_MAX_MODEL_LEN=32768
    DEFAULT_MAX_NUM_SEQS=8
    DEFAULT_MAX_NUM_BATCHED_TOKENS=2048
    DEFAULT_PREFIX_CACHING=0
    DEFAULT_KV_CACHE_MEMORY_BYTES=""
    ;;
  context)
    DEFAULT_MAX_MODEL_LEN=auto
    DEFAULT_MAX_NUM_SEQS=4
    DEFAULT_MAX_NUM_BATCHED_TOKENS=512
    DEFAULT_PREFIX_CACHING=1
    DEFAULT_KV_CACHE_MEMORY_BYTES=1500000000
    ;;
  *)
    echo "PROFILE must be 'baseline' or 'context'; got '${PROFILE}'." >&2
    exit 2
    ;;
esac

MAX_MODEL_LEN="${MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${DEFAULT_MAX_NUM_SEQS}}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${DEFAULT_MAX_NUM_BATCHED_TOKENS}}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-${DEFAULT_PREFIX_CACHING}}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-${DEFAULT_KV_CACHE_MEMORY_BYTES}}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
ENABLE_DSPARK="${ENABLE_DSPARK:-1}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8099}"

if [[ "${ENABLE_DSPARK}" == 1 ]]; then
  if ! [[ "${NUM_SPECULATIVE_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SPECULATIVE_TOKENS must be a positive integer." >&2
    exit 2
  fi
  speculative_config="$(printf '{"method":"dspark","num_speculative_tokens":%d}' \
    "${NUM_SPECULATIVE_TOKENS}")"
  speculative_args=(--speculative-config "${speculative_config}")
elif [[ "${ENABLE_DSPARK}" == 0 ]]; then
  speculative_args=()
else
  echo "ENABLE_DSPARK must be 0 or 1." >&2
  exit 2
fi
if [[ "${ENABLE_PREFIX_CACHING}" == 1 ]]; then
  prefix_caching_args=(--enable-prefix-caching)
elif [[ "${ENABLE_PREFIX_CACHING}" == 0 ]]; then
  prefix_caching_args=(--no-enable-prefix-caching)
else
  echo "ENABLE_PREFIX_CACHING must be 0 or 1." >&2
  exit 2
fi
if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then
  if ! [[ "${KV_CACHE_MEMORY_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "KV_CACHE_MEMORY_BYTES must be a positive integer byte count." >&2
    exit 2
  fi
  kv_cache_args=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
else
  kv_cache_args=()
fi
# EXTRA_VLLM_ARGS is intentionally word-split so operators can add ordinary
# vLLM CLI switches without modifying this reproducibility script.
# shellcheck disable=SC2206
extra_args=(${EXTRA_VLLM_ARGS:-})

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --pipeline-parallel-size "${PP}" \
  --tensor-parallel-size "${TP}" \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  "${prefix_caching_args[@]}" \
  --enable-chunked-prefill \
  --compilation-config '{"fast_moe_cold_start":false}' \
  --distributed-executor-backend mp \
  --load-format auto \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  "${speculative_args[@]}" \
  "${kv_cache_args[@]}" \
  "${extra_args[@]}"
