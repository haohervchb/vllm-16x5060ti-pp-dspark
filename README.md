<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Reproducing the 16-GPU ASRock Rack host

This is the complete host configuration used for the DeepSeek-V4 measurements
below. It was validated on 2026-08-18; it is intentionally specific to this
machine rather than a generic multi-GPU recipe.

| Component | Validated configuration |
|---|---|
| Motherboard | ASRock Rack `SPC621D8U-2T/OVH` |
| GPU fabric | Two Broadcom/PLX PEX88096 islands, eight GPUs per island |
| GPUs | 16 x RTX 5060 Ti 16 GB, PCI device `10de:2d04` |
| OS | Ubuntu 22.04.5 LTS, UEFI boot |
| Kernel | `6.8.0-106-generic` |
| NVIDIA driver | Aikitoria patched open driver `610.43.02-p2p` |
| Required BAR1 | 16,384 MiB on every GPU |

The [ASRock Rack product/download page](https://www.asrockrack.com/general/productdetail.asp?Model=SPC621D8U-2T#Download)
is the authoritative source for the generic board manual and firmware. This
host uses the OVH-specific board/firmware variant, so do not assume that a
generic ASRock image can safely replace its BIOS.

### Exact topology

The persistent preboot tool deliberately refuses to modify a different PCI
layout. The verified addresses are:

```text
PLX island 0 / GPU indexes 0-7:
  8f:00.0 92:00.0 93:00.0 94:00.0
  95:00.0 98:00.0 9a:00.0 9b:00.0

PLX island 1 / GPU indexes 8-15:
  c8:00.0 cb:00.0 cc:00.0 cd:00.0
  d1:00.0 d2:00.0 d3:00.0 d4:00.0
```

Each device must expose a **Physical Resizable BAR** capability for BAR1 whose
supported-size list includes 16 GB:

```bash
sudo lspci -s 8f:00.0 -vv | grep -A3 -E 'Physical Resizable BAR'
```

Do not adapt the supplied EFI binary merely by changing the device count. Its
safety checks, BDF list, PCI device ID, BAR number, and size-code validation
are all part of the protection against modifying the wrong PCI function.

### Firmware settings

Boot into Setup and use these settings before installing Ubuntu-side support:

- UEFI boot enabled; CSM disabled.
- Secure Boot disabled. The locally built EFI application and patched NVIDIA
  modules are unsigned.
- Above 4G Decoding enabled.
- MMIO High Granularity set to `1024G`.
- MMIO High Base set around `56T`.
- SR-IOV disabled on this machine.
- Enable **Hot Plug Capable** on every populated CPU PCIe root/stack feeding a
  PLX fabric when the option is exposed. In particular, physical slot/root 2
  feeds Linux root port `89:02.0`; slot/root 10 feeds `c2:02.0`. “Slot 2” does
  not mean GPU index 2.

Hot-plug capability influences the firmware-provided bridge windows, but the
final method also makes Linux reconstruct them. The validated prefetchable
windows are 192 GiB at `89:02.0` and 512 GiB at `c2:02.0`.

### Do not flash ReBarUEFI for this procedure

The earlier bring-up included an OVH-specific BIOS experiment with ReBarDxe.
That experiment is not required by the final reproducible solution. Do not
flash the generic ASRock `P622U2T1.20` image, do not flash a 16 MiB BIOS-region
dump as a complete 32 MiB SPI image, and do not use `flashrom` to write the
running host's SMM-protected BIOS.

The final solution is a standalone EFI application. It performs a read-only
validation of all 16 GPUs, selects physical BAR1 size code 14 (16 GiB), clears
the old BAR1 addresses, disables endpoint memory decoding, and returns to
GRUB. Linux then allocates the BARs and upstream bridge windows. The EFI change
is volatile: it does not rewrite motherboard firmware.

### Install the patched NVIDIA driver

Consumer Blackwell PCIe P2P on this system uses the
[Aikitoria `610.43.02-p2p` branch](https://github.com/aikitoria/open-gpu-kernel-modules/tree/610.43.02-p2p).
Its kernel modules must be paired with the userspace components and GSP
firmware from the exact NVIDIA 610.43.02 release.

```bash
sudo apt-get install -y build-essential linux-headers-"$(uname -r)"

mkdir -p ~/nvidia-610.43.02
cd ~/nvidia-610.43.02
curl -fLO \
  https://download.nvidia.com/XFree86/Linux-x86_64/610.43.02/NVIDIA-Linux-x86_64-610.43.02.run
echo '3034a054bb4cdf7752ff8dc272564cb105513804bff53538945901b16ca77463  NVIDIA-Linux-x86_64-610.43.02.run' \
  | sha256sum -c -
chmod +x NVIDIA-Linux-x86_64-610.43.02.run
sudo ./NVIDIA-Linux-x86_64-610.43.02.run --no-kernel-modules

cd ~
git clone --branch 610.43.02-p2p \
  https://github.com/aikitoria/open-gpu-kernel-modules.git
cd ~/open-gpu-kernel-modules
./install.sh
sudo reboot
```

The patched modules are built for the running kernel. Rebuild/reinstall them
after a kernel ABI change and before booting that new kernel. Verify the exact
module version after reboot:

```bash
nvidia-smi --query-gpu=driver_version,name,pci.bus_id --format=csv,noheader
modinfo nvidia | grep -E '^(filename|version|vermagic):'
```

Do not add `ForceP2P`, `RMPcieP2PType`, `RMForceP2PType`, or
`RMForceStaticBar1` registry profiles. This patched branch already selects its
P2P/BAR1 implementation, and experimental extra keys caused
`NV_ERR_INVALID_REGISTRY_KEY` during GSP initialization. The only persistent
module option used here is:

```text
options nvidia NVreg_EnableResizableBar=1
```

The tested Intel host disables IOMMU translation with `intel_iommu=off`.
Disabling translation and ACS reduces device isolation; do not run untrusted
devices, drivers, containers, or workloads on this configuration.

### One-command persistent BAR1, MMIO, and ACS installation

All setup sources are included in this repository. Starting from an Ubuntu
install booted in UEFI mode, with its EFI System Partition mounted at
`/boot/efi`:

```bash
git clone https://github.com/haohervchb/vllm-16x5060ti-pp-dspark.git \
  ~/vllm-16x5060ti-pp-dspark
cd ~/vllm-16x5060ti-pp-dspark
sudo bash tools/efi-rebar-preboot/install-persistent.sh --install
sudo reboot
```

The installer first checks the exact 16 BDFs, `10de:2d04` device IDs, physical
ReBAR capabilities, and at least 32 PEX bridge functions. It installs missing
Ubuntu build dependencies only when necessary, then configures:

- `/boot/efi/EFI/sglang/ReBarPrebootAuto.efi` for unattended boots;
- `/boot/efi/EFI/sglang/ReBarPreboot.efi` for manual recovery/testing;
- one-shot GRUB entries that run the EFI pass and then reload normal Ubuntu;
- `intel_iommu=off pci=realloc=on,hpmmioprefsize=512G` in GRUB;
- `NVreg_EnableResizableBar=1` for the NVIDIA module;
- `sglang-rebar-rearm.service`, which arms the EFI pass for the next boot;
- `sglang-plx-acs.service`, which clears volatile ACS redirect controls on all
  supported PEX bridges after every boot.

The automatic GRUB entry cannot loop: GRUB consumes its one-shot `next_entry`
before chain-loading the EFI application. A successful Linux boot arms the
next one-shot pass. If Linux never reaches the rearm service, the following
boot falls back to the ordinary default entry.

The 512 GiB hot-plug MMIO reservation is per upstream root and includes bridge
alignment/headroom for eight 16 GiB BARs. It reserves address space, not 512
GiB of physical RAM.

For a manual, interactive preboot pass:

```bash
sudo grub-reboot rebar-preboot
sudo reboot
```

The interactive tool changes nothing until all validation lines pass and the
operator presses uppercase `A`. After it succeeds, return to GRUB and boot
Linux without power-cycling; the PCI change is volatile.

### Validate BAR1 and real P2P data movement

After reboot, validate the persistent state:

```bash
cd ~/vllm-16x5060ti-pp-dspark
sudo bash tools/efi-rebar-preboot/install-persistent.sh --check
```

A successful result has all of the following:

- exactly 16 RTX 5060 Ti devices;
- `BAR1 16384 MiB` on every GPU;
- 192 GiB and 512 GiB prefetchable windows at `89:02.0` and `c2:02.0`;
- zero BAR resize/allocation, invalid registry-key, GSP reset, and P2P mailbox
  errors for the current boot;
- `ACSCtl 0000` on all 32 readable PEX bridge functions;
- both persistence services enabled and active.

`nvidia-smi topo -p2p` reports capability only. Compile and run the direct
payload-integrity probe once inside each PLX island:

```bash
cd ~/vllm-16x5060ti-pp-dspark
nvcc -O2 -arch=sm_120 scripts/cuda_p2p_copy_probe.cu \
  -o /tmp/cuda_p2p_copy_probe
CUDA_VISIBLE_DEVICES=0,1 /tmp/cuda_p2p_copy_probe 0 1
CUDA_VISIBLE_DEVICES=8,9 /tmp/cuda_p2p_copy_probe 0 1
```

Both runs must report bidirectional capability and `PASS` for both the peer
kernel copy and `cudaMemcpyPeer` data validation. Then exercise the actual
TP8/PP2 NCCL grouping:

```bash
cd ~/vllm-16x5060ti-pp-dspark
NCCL_CUMEM_ENABLE=0 NCCL_P2P_LEVEL=PXB \
  conda run --no-capture-output -n sglang-dev \
  torchrun --standalone --nproc-per-node=16 \
  scripts/nccl_tp8_pp2_probe.py
```

The verified host measured approximately 7.29 GB/s TP8 all-reduce algorithm
bandwidth per island and passed all 16 ranks. `NCCL_P2P_LEVEL=PXB` keeps direct
P2P inside GPUs 0-7 and 8-15 while the cross-island PP pairs use shared-memory
host staging. Do not globally set `NCCL_P2P_DISABLE=1` for TP8/PP2.

### Removal and recovery

To remove the automatic EFI files, GRUB entries, services, kernel arguments,
and NVIDIA module fragment installed by the persistent tool:

```bash
cd ~/vllm-16x5060ti-pp-dspark
sudo bash tools/efi-rebar-preboot/install-persistent.sh --uninstall
sudo reboot
```

This does not flash or restore motherboard firmware. The normal Ubuntu GRUB
entry remains available if the EFI pass fails validation. Keep working BMC
console access before changing firmware or PCI resource settings.

## Local DeepSeek-V4-Flash serving on 16 RTX 5060 Ti GPUs

These commands serve the local `DeepSeek-V4-Flash-0731` snapshot with DSpark,
prefix caching, CUDA graphs, and a maximum concurrency of four. They are the
validated maximum-context configurations for this host. Install the required
SM120 FlashInfer revision once before launching:

```bash
cd ~/vllm-16x5060ti-pp-dspark
examples/online_serving/deepseek_v4_flash_dspark/setup_flashinfer_sm120.sh
```

The CUDA 13 toolkit selection below is required. `/usr/local/cuda-12.8/bin/nvcc`
cannot compile FlashInfer's `compute_120f` target.

### TP8/PP2

This layout uses GPUs 0-7 for PP0 and GPUs 8-15 for PP1, keeping each TP group
inside one PLX island. It fitted a 502,784-token context limit and was validated
with a 480,000-token prompt plus 64 output tokens.

```bash
cd ~/vllm-16x5060ti-pp-dspark
export MODEL_PATH=deepseek-ai/DeepSeek-V4-Flash-0731
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NCCL_P2P_LEVEL=PXB
export NCCL_CUMEM_ENABLE=0
export NCCL_SHM_DISABLE=0
export VLLM_PP_LAYER_PARTITION=23,20
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/vllm serve \
  "$MODEL_PATH" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 8 \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --host 0.0.0.0 \
  --port 8099 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --compilation-config '{"fast_moe_cold_start":false}' \
  --distributed-executor-backend mp \
  --load-format auto \
  --gpu-memory-utilization 0.92 \
  --kv-cache-memory-bytes 1500000000 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

### TP4/PP4

This layout fitted the checkpoint's native 1,048,576-token context limit and
was validated with a 1,000,000-token prompt plus 64 output tokens.

```bash
cd ~/vllm-16x5060ti-pp-dspark
export MODEL_PATH=deepseek-ai/DeepSeek-V4-Flash-0731
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST=12.0f
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NCCL_P2P_LEVEL=PXB
export NCCL_CUMEM_ENABLE=0
export NCCL_SHM_DISABLE=0
export VLLM_PP_LAYER_PARTITION=11,12,12,8
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/vllm serve \
  "$MODEL_PATH" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --pipeline-parallel-size 4 \
  --tensor-parallel-size 4 \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --host 0.0.0.0 \
  --port 8099 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --compilation-config '{"fast_moe_cold_start":false}' \
  --distributed-executor-backend mp \
  --load-format auto \
  --gpu-memory-utilization 0.90 \
  --kv-cache-memory-bytes 1500000000 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

Five speculative tokens gave the better mixed-workload result. For a
predictable high-acceptance workload, change `num_speculative_tokens` to `7`.
Implementation details, measured performance, and failure signatures are in
[the local DeepSeek-V4 DSpark guide](examples/online_serving/deepseek_v4_flash_dspark/README.md).

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
