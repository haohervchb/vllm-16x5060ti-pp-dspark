#!/usr/bin/env bash
# Configure the boot-time MMIO reservation required for real (data-validating)
# CUDA P2P through aikitoria's patched driver behind nested PLX switches.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRUB_SOURCE="${SCRIPT_DIR}/config/99-sglang-rtx5060-p2p.cfg"
MODPROBE_SOURCE="${SCRIPT_DIR}/config/sglang-rtx5060-p2p.conf"
ACS_SOURCE="${SCRIPT_DIR}/configure_plx_acs.sh"
ACS_UNIT_SOURCE="${SCRIPT_DIR}/config/sglang-plx-acs.service"
DOC_SOURCE="${SCRIPT_DIR}/../tools/efi-rebar-preboot/README.md"
GRUB_TARGET="/etc/default/grub.d/99-sglang-rtx5060-p2p.cfg"
MODPROBE_TARGET="/etc/modprobe.d/sglang-rtx5060-p2p.conf"
ACS_TARGET="/usr/local/sbin/sglang-configure-plx-acs"
ACS_UNIT_TARGET="/etc/systemd/system/sglang-plx-acs.service"
DOC_TARGET="/usr/local/share/doc/sglang-rtx5060-rebar/README.md"

usage() {
    cat <<'EOF'
Usage: configure_rtx5060_p2p_host.sh [--check|--apply|--rollback]

  --check     Inspect the active boot arguments, persistent ACS service,
              driver keys, BAR1 sizes, and BAR resize errors. This is the
              default and does not need root (the ACS register check may).
  --apply     Install the GRUB/modprobe fragments and persistent ACS boot
              service, then rebuild GRUB and initramfs. Reboot afterward.
  --rollback  Remove only files installed by this script, disable the ACS
              service, and rebuild GRUB/initramfs. Reboot afterward.
EOF
}

run_nvidia_smi() {
    # A failed GSP initialization can make nvidia-smi block for a long time.
    timeout 8s nvidia-smi "$@"
}

require_root() {
    if (( EUID != 0 )); then
        echo "error: $1 must be run as root (use sudo)" >&2
        exit 1
    fi
}

bar1_totals() {
    run_nvidia_smi -q -d MEMORY 2>/dev/null |
        awk '
            /^GPU [0-9]+:/ { gpu=$2; sub(":", "", gpu) }
            /BAR1 Memory Usage/ { in_bar1=1; next }
            in_bar1 && /^[[:space:]]+Total/ {
                printf "GPU %-2s BAR1 %s %s\n", gpu, $3, $4
                in_bar1=0
            }
        '
}

check_host() {
    echo "== Hardware =="
    local gpu_count gpu_ids
    gpu_ids="$(run_nvidia_smi --query-gpu=pci.device_id --format=csv,noheader 2>/dev/null)" || true
    gpu_count="$(awk 'toupper($0) ~ /2D0410DE/ { count++ } END { print count+0 }' <<<"${gpu_ids}")"
    echo "RTX 5060 Ti / GB206 devices: ${gpu_count}"
    if ! run_nvidia_smi --query-gpu=index,pci.bus_id,name,display_active \
        --format=csv,noheader 2>/dev/null; then
        echo "nvidia-smi cannot initialize the GPUs"
        echo "PCI functions present: $(lspci -Dn 2>/dev/null | awk '$2 ~ /^03/ && toupper($3) == "10DE:2D04" { count++ } END { print count+0 }')"
    fi

    echo
    echo "== Installed fragments =="
    for target in "${GRUB_TARGET}" "${MODPROBE_TARGET}"; do
        if [[ -f "${target}" ]]; then
            echo "${target}: installed"
            sed 's/^/  /' "${target}"
        else
            echo "${target}: not installed"
        fi
    done

    echo
    echo "== Persistent PLX ACS service =="
    for target in "${ACS_TARGET}" "${ACS_UNIT_TARGET}" "${DOC_TARGET}"; do
        if [[ -f "${target}" ]]; then
            echo "${target}: installed"
        else
            echo "${target}: not installed"
        fi
    done
    if command -v systemctl >/dev/null 2>&1; then
        printf 'sglang-plx-acs.service enabled: '
        systemctl is-enabled sglang-plx-acs.service 2>/dev/null || true
        printf 'sglang-plx-acs.service active:  '
        systemctl is-active sglang-plx-acs.service 2>/dev/null || true
    fi
    if [[ -x "${ACS_TARGET}" ]]; then
        "${ACS_TARGET}" --check 2>/dev/null || \
            echo "ACS register check unavailable without root; retry --check with sudo"
    elif [[ -x "${ACS_SOURCE}" ]]; then
        "${ACS_SOURCE}" --check 2>/dev/null || \
            echo "ACS register check unavailable without root; retry --check with sudo"
    fi

    echo
    echo "== Active kernel command line =="
    cat /proc/cmdline
    if grep -qw 'intel_iommu=off' /proc/cmdline &&
       grep -qw 'pci=realloc=on,hpmmioprefsize=512G' /proc/cmdline; then
        echo "boot-time MMIO reservation: ACTIVE"
    else
        echo "boot-time MMIO reservation: NOT ACTIVE"
    fi

    echo
    echo "== Loaded NVIDIA parameters =="
    if [[ -r /proc/driver/nvidia/params ]]; then
        grep -E '^(EnableResizableBar|RegistryDwords):' \
            /proc/driver/nvidia/params || true
    else
        echo "NVIDIA driver is not loaded"
    fi

    echo
    echo "== BAR1 apertures =="
    bar1_totals || true

    echo
    echo "== PLX-island root bridge windows =="
    for root_port in 89:02.0 c2:02.0; do
        printf '%s: ' "${root_port}"
        lspci -s "${root_port}" -vv 2>/dev/null |
            awk -F': ' '/Prefetchable memory behind bridge/ { print $2; found=1 }
                       END { if (!found) print "unavailable" }'
    done

    echo
    echo "== BAR resize messages from this boot =="
    if command -v journalctl >/dev/null 2>&1; then
        journalctl -k -b --no-pager 2>/dev/null |
            awk '
                /BAR resizing failed/ { resize_failed++ }
                /No address space to allocate resized BAR1/ { no_space++ }
                /BAR1 already at requested size/ { already_sized++ }
                /NV_ERR_INVALID_REGISTRY_KEY/ { invalid_key++ }
                /unexpected WPR2 already up/ { wpr2++ }
                /remoteWMBoxLocalAddr != ~0ULL/ { mailbox++ }
                /Static bar1 mapped/ { static_bar1++ }
                END {
                    printf "BAR resizing failed: %d\n", resize_failed+0
                    printf "BAR1 allocation no-space errors: %d\n", no_space+0
                    printf "BAR1 already correctly sized: %d\n", already_sized+0
                    printf "invalid NVIDIA registry-key errors: %d\n", invalid_key+0
                    printf "GSP WPR2/reset-state errors: %d\n", wpr2+0
                    printf "legacy P2P mailbox failures: %d\n", mailbox+0
                    printf "static BAR1 mappings created: %d\n", static_bar1+0
                }
            ' || true
    fi

    echo
    echo "Success requires every GPU to report a framebuffer-sized BAR1 and the"
    echo "direct CUDA probe to validate data in both PLX islands. topo -p2p OK alone"
    echo "is not a data-integrity test."
}

install_fragment() {
    local source="$1"
    local target="$2"

    if [[ -e "${target}" ]] && ! cmp -s "${source}" "${target}"; then
        # Permit replacement of the exact experimental GB206 profile installed
        # by an earlier version of this script, or the earlier safe baseline
        # whose only active directive was identical.  Continue refusing
        # arbitrary pre-existing administrator configuration.
        if [[ "${target}" == "${MODPROBE_TARGET}" ]] &&
           grep -Fqx 'options nvidia NVreg_EnableResizableBar=1' "${target}" &&
           [[ "$(grep -Ec '^[[:space:]]*options[[:space:]]+nvidia([[:space:]]|$)' "${target}")" -eq 1 ]]; then
            echo "Refreshing prior safe ReBAR profile: ${target}"
        elif [[ "${target}" == "${MODPROBE_TARGET}" ]] &&
             grep -Fq 'RTX 5060 Ti / GB206 experimental PCIe P2P path.' "${target}" &&
             grep -Fq 'RMForceStaticBar1=1' "${target}"; then
            echo "Replacing incompatible 610.43.03 experimental profile: ${target}"
        else
            echo "error: refusing to overwrite existing non-matching ${target}" >&2
            exit 1
        fi
    fi
    install -D -m 0644 "${source}" "${target}"
}

apply_host() {
    require_root --apply
    command -v update-grub >/dev/null
    command -v update-initramfs >/dev/null
    command -v systemctl >/dev/null
    command -v lspci >/dev/null
    command -v setpci >/dev/null

    install_fragment "${GRUB_SOURCE}" "${GRUB_TARGET}"
    install_fragment "${MODPROBE_SOURCE}" "${MODPROBE_TARGET}"
    install -D -m 0755 "${ACS_SOURCE}" "${ACS_TARGET}"
    install -D -m 0644 "${ACS_UNIT_SOURCE}" "${ACS_UNIT_TARGET}"
    install -D -m 0644 "${DOC_SOURCE}" "${DOC_TARGET}"
    systemctl daemon-reload
    systemctl enable sglang-plx-acs.service
    # The operation is idempotent and the current host is already using these
    # registers, so applying it immediately is safe as well as boot-persistent.
    systemctl restart sglang-plx-acs.service
    update-grub
    update-initramfs -u -k all

    echo
    echo "Configuration installed. Reboot is required. After reboot:"
    echo "  1. ${SCRIPT_DIR}/configure_rtx5060_p2p_host.sh --check"
    echo "  2. nvcc -O2 -arch=sm_120 ${SCRIPT_DIR}/cuda_p2p_copy_probe.cu -o /tmp/cuda_p2p_copy_probe"
    echo "  3. CUDA_VISIBLE_DEVICES=0,1 /tmp/cuda_p2p_copy_probe 0 1"
    echo "  4. CUDA_VISIBLE_DEVICES=8,9 /tmp/cuda_p2p_copy_probe 0 1"
}

rollback_host() {
    require_root --rollback
    if command -v systemctl >/dev/null 2>&1; then
        systemctl disable --now sglang-plx-acs.service 2>/dev/null || true
    fi
    rm -f -- "${GRUB_TARGET}" "${MODPROBE_TARGET}" "${ACS_TARGET}" \
        "${ACS_UNIT_TARGET}" "${DOC_TARGET}"
    systemctl daemon-reload 2>/dev/null || true
    update-grub
    update-initramfs -u -k all
    echo "Configuration fragments removed. Reboot to complete the rollback."
}

action="${1:---check}"
case "${action}" in
    --check)
        (( $# <= 1 )) || { usage >&2; exit 2; }
        check_host
        ;;
    --apply)
        (( $# == 1 )) || { usage >&2; exit 2; }
        apply_host
        ;;
    --rollback)
        (( $# == 1 )) || { usage >&2; exit 2; }
        rollback_host
        ;;
    --help|-h)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
