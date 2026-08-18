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

Pass either layout. Hugging Face downloads the checkpoint automatically when
it is not already cached:

```bash
examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp4pp4
```

```bash
examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp8pp2
```

The default `PROFILE=baseline` reproduces the single-request decode baseline:
32,768 maximum model length, five DSpark tokens, FP8 KV cache, block size 256,
CUDA graphs enabled, and prefix caching disabled. Production settings can be
overridden, for example:

```bash
MAX_MODEL_LEN=32768 MAX_NUM_SEQS=8 \
  EXTRA_VLLM_ARGS="--enable-prefix-caching" \
  examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp8pp2
```

Do not reduce TP8/PP2 to `gpu-memory-utilization=0.90` at 32K. The measured
available KV memory was 1.18 GiB while one 32K request required 1.42 GiB. At
`0.92`, initialization produced a 41,448-token cache and completed CUDA-graph
capture.

## PLX custom all-reduce

The launcher enables this fork's opt-in two-stage custom all-reduce for every
TP group. Upstream vLLM normally disables its custom kernel on more than two
PCIe-only GPUs. That default is sensible for ordinary multi-root PCIe systems,
but it leaves performance on the table here: every TP4 or TP8 group is confined
to one PLX island with working all-pairs P2P and framebuffer-sized BAR1.

The exact launch settings are:

```bash
export VLLM_ALLOW_PCI_CUSTOM_ALLREDUCE=1
export VLLM_CUSTOM_ALLREDUCE_ALGO=2stage
export PYTORCH_CUDA_ALLOC_CONF=backend:native
```

The native allocator is mandatory for CUDA-graph IPC registration. Do not use
`expandable_segments:True` or `backend:cudaMallocAsync` with this path. Disable
the optimization for an A/B or an unvalidated topology with:

```bash
ENABLE_PLX_CUSTOM_ALLREDUCE=0 \
  examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp8pp2
```

Validate all groups after changing the GPU order, driver, firmware, or PLX
configuration. This exercises exact eager and CUDA-graph reductions repeatedly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/torchrun --standalone --nproc-per-node=8 \
  scripts/validate_plx_custom_allreduce.py --ops-per-graph 20

CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  .venv/bin/torchrun --standalone --nproc-per-node=8 \
  scripts/validate_plx_custom_allreduce.py --ops-per-graph 20
```

On this host, a 6-by-8192 BF16 reduction took 0.022 ms at TP8 versus 0.041 ms
through PyNCCL (1.89x), and 0.019 ms at TP4 versus 0.029 ms (1.49x).

## Maximum-context profile

The long-context profile is deliberately separate from the decode-throughput
baseline:

```bash
PROFILE=context \
  examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp4pp4
```

```bash
PROFILE=context \
  examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp8pp2
```

It enables prefix caching, sets `max-num-seqs=4`, keeps CUDA graphs enabled,
uses five DSpark tokens, reserves exactly 1,500,000,000 bytes per GPU for KV
cache, and reduces `max-num-batched-tokens` to 512. It keeps PyTorch's native
allocator so custom all-reduce can register CUDA-graph buffers. `max-model-len`
`auto` then fits the model length to the explicit cache instead of relying on
the less reproducible free-memory percentage.

The 512-token prefill chunk is essential. DeepGEMM's sparse indexer creates a
logits workspace proportional to both the prefill chunk and current context.
At a 2,048-token chunk, a 260K TP4 prompt attempted a 470 MiB allocation and
OOMed. At a 1,024-token chunk, a 780K prompt still OOMed. Halving the chunk to
512 made a 1M TP4 prompt and a 480K TP8 prompt usable, while TP4's allocated KV
cache covers the full native model window. A larger chunk can improve cold-
prefill throughput, but its auto-fitted model length is not a usable-context
guarantee.

Measured capacity and real HTTP validation with this profile:

| Layout | vLLM fitted limit | Validated request | Prefix-cache validation |
|---|---:|---:|---:|
| TP4/PP4 | 1,048,576 (native model cap) | 1,000,000 prompt + 64 output | 499,968 cached tokens |
| TP8/PP2 | 502,784 | 480,000 prompt + 64 output | 249,856 cached tokens |

TP4's 500K cold request ran at 4,703 prompt tok/s; extending the same prefix to
1M completed in 237.30 seconds. TP8's 250K cold request ran at 3,700 prompt
tok/s; extending it to 480K completed in 96.83 seconds. These are context
capacity checks, not decode benchmarks. The TP4 target leaves 48,576 tokens
below the architectural limit; TP8 leaves 22,784 tokens below its fitted
limit for output and runtime margin.

Reproduce both the cold and cached-extension checks against a live server:

```bash
TARGET_TOKENS=500000 EXTEND_TOKENS=1000000 \
  examples/online_serving/deepseek_v4_flash_dspark/bench_context.py
```

Use `TARGET_TOKENS=250000 EXTEND_TOKENS=480000` for TP8/PP2. The script sends
exact token-ID prompts and streams 64 generated tokens. Confirm cache reuse
from `/metrics`: `vllm:prefix_cache_hits_total` is rounded down to the last
complete 256-token cache block.

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

With five DSpark tokens, a controlled PLX custom-all-reduce A/B raised the
mixed 800-token-per-workload TP8/PP2 run from 136.5 to 153-159 tok/s. TP4/PP4
rose from about 69 to 86.7 tok/s. Without speculative decoding, TP8/PP2 rose
from 78-79 to 91.6 tok/s. With the final baseline launch defaults, two repeats
measured 161.2 and 163.5 tok/s at TP8/PP2, and 92.7 and 93.0 tok/s at TP4/PP4.
These are end-to-end HTTP results, not isolated collective timings.

Power draw is not a useful saturation target for this single-request workload.
During the final TP8 run, the GPUs reported roughly 80-90% SM activity and
2.75-2.85 GHz core clocks while drawing only about 54-69 W each. TP4 showed the
expected PP imbalance: the first stage was around 93-96% SM utilization, then
roughly 80-90%, 55-75%, and 25-55% on successive stages. Redistributing target
layers did not improve the fixed drafted-token rate, so the memory-safe layer
partitions remain the defaults.

### Target-only comparison at 131K context

For an acceptance-independent comparison with a non-speculative vLLM server,
DSpark was disabled and an exact 131,000-token prompt plus 400 generated tokens
was sent with prefix caching enabled:

| Layout | Cold prefill | Post-TTFT decode |
|---|---:|---:|
| TP8/PP2 | 5,559 tok/s | 91.7-92.2 tok/s |
| TP4/PP4 | 8,944 tok/s | 64.6-64.9 tok/s cached; 71.2 tok/s cold |

The throughput-test launch used these overrides with either layout:

```bash
PROFILE=context ENABLE_DSPARK=0 MAX_MODEL_LEN=131500 \
  MAX_NUM_BATCHED_TOKENS=2048 KV_CACHE_MEMORY_BYTES=2000000000 \
  examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp8pp2
```

The 2,048-token chunk is deliberately aggressive for prefill. At 131K it can
emit recoverable allocator-retry warnings because the 2 GB KV reservation
leaves little transient workspace. Use the context profile's default 512-token
chunk for maximum-context operation; this command is the speed-comparison
profile, not the maximum-capacity profile.

The 16K TP8 continuation achieved 97.5% draft acceptance and is a best case.
The TP8 mixed benchmark ranged from 94.9 to 154.3 tok/s as acceptance changed.
On TP4/PP4, the same mixed benchmark ranged from 59.6 to 99.4 tok/s: the extra
pipeline stages and relay latency consume most of the benefit when acceptance is
low. Run `bench_decode.py` against a live server to measure the same three
content classes on a new build.

### Seven speculative tokens

Seven-token DSpark was validated with CUDA graphs in both PP layouts. It is not
the default because its benefit is acceptance-dependent:

| Layout/workload | Five tokens | Seven tokens |
|---|---:|---:|
| TP8/PP2, mixed technical/prose/code | 120.6 tok/s | 116.0 tok/s mean |
| TP8/PP2, 16K repeated continuation | 227.5 decode tok/s | 274.1 decode tok/s |
| TP4/PP4, 16K repeated continuation | 94.4 decode tok/s | 135.6 decode tok/s |

On TP8, seven tokens improved the high-acceptance 16K continuation by 20.5%
but reduced the mixed aggregate by about 3.8%. Select it explicitly for a
predictable high-acceptance workload:

```bash
NUM_SPECULATIVE_TOKENS=7 \
  examples/online_serving/deepseek_v4_flash_dspark/serve.sh tp4pp4
```

## Failure signatures

- `missing [40, 41, 42]`: the last PP stage does not contain all DSpark target
  taps; correct `VLLM_PP_LAYER_PARTITION`.
- KV-cache memory error during baseline startup: raise
  `GPU_MEMORY_UTILIZATION` slightly or lower `MAX_MODEL_LEN`. For long context,
  use explicit `KV_CACHE_MEMORY_BYTES` and lower `MAX_NUM_BATCHED_TOKENS`; an
  auto-fit success alone does not account for the context-dependent sparse
  indexer workspace. Do not disable CUDA graphs merely to hide an incorrect
  memory plan.
- Unsupported sparse-MLA dispatch at TP8: the pinned FlashInfer revision is not
  the package imported by `.venv/bin/python`; rerun the setup script.
- Very low acceptance or corrupted output only with PP: verify the draft-token
  relay changes are present and all ranks agree on the speculative config.
- NCCL failure across unrelated GPUs: verify `CUDA_VISIBLE_DEVICES` preserves
  the two eight-GPU island blocks and use `NCCL_P2P_LEVEL=PXB`.
