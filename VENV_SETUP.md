# Local vLLM virtual environment setup

This fork is built from source against Python 3.12, PyTorch 2.13, and CUDA 13.0.
Do not create the environment with a bare `uv pip install -e .`; vLLM probes
CUDA during its build metadata/configure phase, before an empty environment has
the CUDA compiler and libraries it needs.

## Fresh clone

From the repository root:

```bash
bash scripts/setup_local_venv.sh
```

The script is idempotent. It:

1. Creates `.venv` with Python 3.12 if it does not exist.
2. Installs the pinned CUDA 13.0 Python toolkit.
3. Constructs a conventional CUDA compatibility view under
   `.venv/lib/python3.12/site-packages/nvidia/cu13` so PyTorch's legacy
   `FindCUDA.cmake` can discover unversioned linker names such as
   `libcudart.so` even though the NVIDIA Python wheels use versioned/split
   component paths.
4. Installs the vLLM CUDA build requirements.
5. Builds this checkout as an editable vLLM installation.

You do not need to activate the environment to use the repo scripts; they call
`.venv/bin/python`, `.venv/bin/vllm`, or `.venv/bin/torchrun` directly.

## Verify

```bash
.venv/bin/python --version
.venv/bin/vllm --version
.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc --version
```

## Install the pinned SM120 FlashInfer build

After the vLLM environment builds successfully:

```bash
examples/online_serving/deepseek_v4_flash_dspark/setup_flashinfer_sm120.sh
```

Then launch TP8/PP2 long-context serving with:

```bash
PROFILE=context \
  examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp8pp2
```

The launcher enables this fork's validated PLX custom-all-reduce path and uses
PyTorch's native allocator. Do not replace it with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when PLX custom all-reduce is
enabled.

## Retrying after a failed build

Do not delete `.venv` just because the editable vLLM compilation failed. The
CUDA toolkit, PyTorch, and other downloaded dependencies can be reused. Pull
the latest `main` and rerun the bootstrap:

```bash
git pull --ff-only origin main
bash scripts/setup_local_venv.sh
```

Delete `.venv` only when you intentionally want a completely clean rebuild.
