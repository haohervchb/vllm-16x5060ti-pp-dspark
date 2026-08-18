# DeepSeek-V4-Flash DSpark on 16 x RTX 5060 Ti

This setup runs `deepseek-ai/DeepSeek-V4-Flash-0731` with DSpark speculative
decoding and pipeline parallelism on one host with sixteen 16 GiB RTX 5060 Ti
GPUs. It has been validated in both of these layouts:

| Layout | Target-layer partition | Last PP stage | GPU-memory utilization |
|---|---|---|---|
| TP4/PP4 | `11,12,12,8` | GPUs 12-15 | `0.90` |
| TP8/PP2 | `23,20` | GPUs 8-15 | `0.92` |

The GPU order is deliberate. GPUs 0-7 are below the first PLX switch and GPUs
8-15 are below the second. Consequently, each TP8 group stays inside one PLX
island. With TP4/PP4, every TP group also stays inside one island.

## Why the drafter belongs on the last stage

The checkpoint declares target taps 41, 42, and 43. Internally these are
zero-based layers 40, 41, and 42. DSpark needs all three hidden-state taps plus
the target `lm_head`. The implementation therefore creates one complete TP-
sharded drafter on the final PP stage; it does not pipeline-shard the drafter.

The last target stage must own all three taps. The startup check rejects an
invalid `VLLM_PP_LAYER_PARTITION` rather than silently reading missing hidden
states. It also prints the local target-layer range and the validated taps.

The last stage proposes draft tokens and broadcasts them back to the earlier PP
stages. Those stages need the exact same proposals on the next target forward.
The relay has fixed-width sampled-token messages so a rejection cannot change
the collective size and desynchronize NCCL. The draft loads its own TP-sharded
embedding because the target embedding exists only on the first PP stage; it
shares the target `lm_head`, which is materialized on the last stage.

This is the reusable part of the CMP 170HX approach. The Ampere-specific model
backend, sparse-index workarounds, and eager-mode workarounds are intentionally
not used. SM120 runs the native DeepSeek-V4 backend and both target and DSpark
CUDA graphs remain enabled.

## Prerequisites

1. The vLLM virtual environment is at `.venv` and this branch is checked out.
2. `nvidia-smi -L` reports exactly 16 GPUs in PLX-island order: first GPUs 0-7,
   then GPUs 8-15.
3. Every GPU has a 16 GiB BAR1 aperture. Check with:

   ```bash
   nvidia-smi -q -d MEMORY | grep -A3 "BAR1 Memory Usage"
   ```

4. ACS redirect is disabled on the PLX bridges after every boot, and the direct
   CUDA peer-copy probe passes inside both islands. `nvidia-smi topo -p2p` alone
   is not a data-integrity test.
5. CUDA peer access is required within each TP group only. There is no TP
   collective between the two PLX islands in the TP8/PP2 layout.

## Install the required FlashInfer revision

The released FlashInfer build originally installed on this host did not expose
DeepSeek-V4 sparse-MLA `(heads=8, topk=192)` or `(heads=16, topk=192)` dispatch.
TP8 needs the first shape and TP4 needs the second when five DSpark tokens are
enabled. Install the pinned revision containing native safe H8 support:

```bash
examples/online_serving/deepseek_v4_flash_dspark/setup_flashinfer_sm120.sh
```

The script installs into this repository's `.venv` without altering the base
Conda environment. It checks the source commit and both required dispatch
shapes after installation.

## Launch

Pass either layout and the local model path:

```bash
MODEL_PATH=/path/to/DeepSeek-V4-Flash-0731 \
  examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp4pp4
```

```bash
MODEL_PATH=/path/to/DeepSeek-V4-Flash-0731 \
  examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp8pp2
```

The defaults reproduce the single-request baseline: 32,768 maximum model
length, five DSpark tokens, FP8 KV cache, block size 256, CUDA graphs enabled,
and prefix caching disabled. Production settings can be overridden, for
example:

```bash
MODEL_PATH=/path/to/model MAX_MODEL_LEN=32768 MAX_NUM_SEQS=8 \
  EXTRA_VLLM_ARGS="--enable-prefix-caching" \
  examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp8pp2
```

Do not reduce TP8/PP2 to `gpu-memory-utilization=0.90` at 32K. The measured
available KV memory was 1.18 GiB while one 32K request required 1.42 GiB. At
`0.92`, initialization produced a 41,448-token cache and completed CUDA-graph
capture.

## Measured single-request baseline

These are end-to-end HTTP measurements at temperature zero. Speculative decode
is content-dependent, so the mixed result is more representative than the
high-acceptance continuation alone.

| Layout/workload | Result |
|---|---:|
| TP4/PP4, high-acceptance continuation, 512 output tokens | 117-129 tok/s |
| TP4/PP4, mixed technical/prose/code, 400 tokens each | 69.0 tok/s aggregate |
| TP4/PP4, 16K repeated technical context | 94.4 decode tok/s, 7,083 prompt tok/s |
| TP8/PP2, high-acceptance continuation, 512 output tokens | 130-146 tok/s |
| TP8/PP2, mixed technical/prose/code, 400 tokens each | 120.6 tok/s aggregate |
| TP8/PP2, 16K repeated technical context | 227.5 decode tok/s, 6,388 prompt tok/s |

The 16K TP8 continuation achieved 97.5% draft acceptance and is a best case.
The TP8 mixed benchmark ranged from 94.9 to 154.3 tok/s as acceptance changed.
On TP4/PP4, the same mixed benchmark ranged from 59.6 to 99.4 tok/s: the extra
pipeline stages and relay latency consume most of the benefit when acceptance is
low. Run `bench_decode.py` against a live server to measure the same three
content classes on a new build.

## Failure signatures

- `missing [40, 41, 42]`: the last PP stage does not contain all DSpark target
  taps; correct `VLLM_PP_LAYER_PARTITION`.
- KV-cache memory error during startup: raise `GPU_MEMORY_UTILIZATION` slightly
  or lower `MAX_MODEL_LEN`. Do not disable CUDA graphs merely to hide an
  incorrect memory plan.
- Unsupported sparse-MLA dispatch at TP8: the pinned FlashInfer revision is not
  the package imported by `.venv/bin/python`; rerun the setup script.
- Very low acceptance or corrupted output only with PP: verify the draft-token
  relay changes are present and all ranks agree on the speculative config.
- NCCL failure across unrelated GPUs: verify `CUDA_VISIBLE_DEVICES` preserves
  the two eight-GPU island blocks and use `NCCL_P2P_LEVEL=PXB`.
