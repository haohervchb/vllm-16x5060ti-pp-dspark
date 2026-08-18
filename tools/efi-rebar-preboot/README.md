# Persistent 16 GiB BAR1 on the 16-GPU SPC621D8U-2T/OVH server

This directory contains the reproducible configuration that changed all 16
RTX 5060 Ti cards from a 256 MiB BAR1 aperture to a 16 GiB BAR1 aperture on the
SPC621D8U-2T/OVH server with two Broadcom/PLX PEX switch islands.

The solution does **not** depend on flashing a modified motherboard BIOS. The
ReBarDxe firmware experiment by itself left every GPU at 256 MiB. The change
that produced the verified 16 GiB result was the standalone EFI application in
this directory, followed by Linux PCI resource reallocation. Avoid another
firmware flash when reproducing this procedure.

## Verified hardware layout

This implementation intentionally refuses to run on a different layout. It
requires 16 NVIDIA `10de:2d04` devices at these stable PCI addresses:

```text
PLX island 0: 8f:00.0 92:00.0 93:00.0 94:00.0
              95:00.0 98:00.0 9a:00.0 9b:00.0
PLX island 1: c8:00.0 cb:00.0 cc:00.0 cd:00.0
              d1:00.0 d2:00.0 d3:00.0 d4:00.0
```

Every device must expose a physical Resizable BAR capability whose BAR 1
supported-size bitmap includes 16 GiB. On the verified GPUs, the capability is
at extended configuration offset `0x134` and its BAR1 control is at `0x13c`.
The initial control had size code 8 (256 MiB); the EFI application selects size
code 14 (16 GiB).

The successfully tested firmware settings were:

- UEFI boot mode and Secure Boot disabled;
- Above 4G Decoding enabled;
- MMIO High Granularity set to 1024G;
- MMIO High Base around 56T;
- SR-IOV disabled.

Per-stack PCIe Hot Plug capability can affect the firmware-provided bridge
windows, but it is not relied upon for the final allocation. Linux receives
`pci=realloc=on,hpmmioprefsize=512G` and reconstructs the upstream windows after
the EFI application leaves BAR1 unassigned. Both PLX host/root ports still need
enough address space above 4G; the validated Linux windows were 192 GiB for
`89:02.0` and 512 GiB for `c2:02.0`.

## Why the normal settings were insufficient

These settings solve different parts of the problem:

1. `NVreg_EnableResizableBar=1` allows the NVIDIA driver to use a large BAR,
   but it does not select a new physical BAR size.
2. `pci=realloc=on,hpmmioprefsize=512G` lets Linux assign large device BARs and
   grow upstream bridge windows, but the firmware originally presented each
   GPU BAR1 as only 256 MiB.
3. Disabling PLX ACS redirect controls allows peer transactions to stay within
   each switch island, but ACS does not resize BARs.
4. ReBarUEFI/ReBarDxe adds firmware ReBAR policy support on some boards, but on
   this server it did not select 16 GiB BAR1 for the GPUs.

The missing operation had to happen before Linux enumerated PCI resources. The
EFI app performs that operation directly with the UEFI PCI Root Bridge I/O
protocol.

## Exact preboot operation

`rebar_preboot.c` first performs a read-only safety pass. It requires all of the
following before changing anything:

- every expected BDF is present in PCI segment 0;
- every function has vendor/device ID `10de:2d04`;
- the physical Resizable BAR extended capability can be traversed safely;
- the selected control describes BAR 1;
- the capability bitmap advertises the 16 GiB size.

It saves each GPU's PCI command word, BAR1 low/high address, and ReBAR control.
Only after all 16 devices validate does it:

1. clear bit 1 (Memory Space Enable) in every GPU's PCI command register;
2. replace bits 12:8 of the physical BAR1 ReBAR control with size code 14;
3. read the control back and verify code 14;
4. write zero to both halves of the 64-bit BAR1 address, making it unassigned;
5. return to GRUB without re-enabling memory decoding.

Linux then boots with PCI reallocation enabled, assigns the sixteen 16 GiB
apertures above 4G, expands the bridge windows, and the NVIDIA driver enables
and uses the resources. A write/readback failure causes a best-effort rollback
of the saved controls, addresses, and command words.

Extended PCI configuration is important here. The ReBAR control is beyond the
legacy 256-byte configuration area. GRUB 2.06 `setpci` uses the legacy CF8/CFC
path on this machine and cannot safely reach offset `0x13c`; that is why the EFI
application uses `EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL` with the extended register
offset encoded in address bits 32 and above.

## One-command persistent installation

Start from an Ubuntu installation booted in UEFI mode with the BIOS settings
above and this repository present. Run:

```bash
cd ~/vllm-16x5060ti-pp-dspark
sudo bash tools/efi-rebar-preboot/install-persistent.sh --install
sudo reboot
```

The installer validates the exact hardware before writing system files. On a
fresh Ubuntu installation it installs the required Ubuntu build packages,
builds manual and automatic EFI binaries, and configures all of the following:

- `/boot/efi/EFI/sglang/ReBarPrebootAuto.efi` for unattended boots;
- `/boot/efi/EFI/sglang/ReBarPreboot.efi` for manual validation/recovery;
- `/etc/grub.d/09_sglang_rebar` with stable GRUB menu-entry IDs;
- `intel_iommu=off pci=realloc=on,hpmmioprefsize=512G` in a GRUB fragment;
- `NVreg_EnableResizableBar=1` in a modprobe fragment;
- `sglang-rebar-rearm.service` to arm the automatic EFI entry for the next
  boot immediately after every successful Linux boot;
- `sglang-plx-acs.service` to clear ACS redirect controls after every boot.

The installer does not reboot automatically. Its first run arms the first
automatic EFI boot with `grub-reboot rebar-preboot-auto`.

### Why the persistent boot does not loop

The automatic entry is always a GRUB **one-shot** entry. GRUB consumes and
clears `next_entry` before chain-loading `ReBarPrebootAuto.efi`. When the EFI
application returns, the same menu entry reloads the normal `grub.cfg`; because
the one-shot value is already gone, GRUB selects the administrator's normal
Ubuntu default. Once Ubuntu is running, `sglang-rebar-rearm.service` arms the
EFI entry for the following boot.

Because the next boot is armed near the beginning of every successful Linux
boot, the mechanism also survives a subsequent hard power loss. If Linux never
reaches the rearm service, GRUB falls back to the ordinary boot entry rather
than looping.

## Manual one-shot mode

The interactive binary makes no changes until all 16 validation lines pass and
the operator presses uppercase `A`. To request it for one boot:

```bash
sudo grub-reboot rebar-preboot
sudo reboot
```

After success, press another key to return to GRUB. Do not power-cycle between
the EFI application and Linux because the PCI configuration change is volatile.

## Validation after reboot

Check the complete persistent installation and live BAR state:

```bash
cd ~/vllm-16x5060ti-pp-dspark
sudo bash tools/efi-rebar-preboot/install-persistent.sh --check
```

A successful boot reports 16,384 MiB BAR1 for all 16 GPUs and no BAR resize or
allocation errors. Then validate actual data movement; `nvidia-smi topo -p2p`
alone is not a data-integrity test:

```bash
nvcc -O2 -arch=sm_120 scripts/cuda_p2p_copy_probe.cu \
  -o /tmp/cuda_p2p_copy_probe
CUDA_VISIBLE_DEVICES=0,1 /tmp/cuda_p2p_copy_probe 0 1
CUDA_VISIBLE_DEVICES=8,9 /tmp/cuda_p2p_copy_probe 0 1

NCCL_CUMEM_ENABLE=0 NCCL_P2P_LEVEL=PXB \
  conda run --no-capture-output -n sglang-dev \
  torchrun --standalone --nproc-per-node=16 \
  scripts/nccl_tp8_pp2_probe.py
```

The validated machine produced successful kernel-copy and `cudaMemcpyPeer`
checks in both islands, approximately 7.29 GB/s TP8 all-reduce algorithm
bandwidth per island, and a passing 16-rank TP8/PP2 topology probe.

`NCCL_P2P_LEVEL=PXB` is deliberate. It retains direct P2P inside GPUs 0-7 and
inside GPUs 8-15, while the NODE-distance PP pairs use shared-memory host
staging. Do not set `NCCL_P2P_DISABLE=1` globally for TP8/PP2.

## Recovery and removal

The EFI operation is volatile and cannot itself brick or rewrite motherboard
firmware. If it fails validation it returns to GRUB without applying a partial
new configuration. The normal Ubuntu entry remains bootable.

To remove the automatic EFI, GRUB, service, kernel, and modprobe configuration:

```bash
cd ~/vllm-16x5060ti-pp-dspark
sudo bash tools/efi-rebar-preboot/install-persistent.sh --uninstall
sudo reboot
```

The uninstall action removes only files installed by this tool. It does not
change, restore, or flash the motherboard firmware and does not remove Ubuntu
packages.
