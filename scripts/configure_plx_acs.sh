#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: configure_plx_acs.sh [--check | --apply]

Check or disable PCIe ACS redirect controls on Broadcom/LSI PEX88048
(vendor/device 1000:c010) bridges. Disabling ACS is required for direct GPU
P2P on some bare-metal PLX systems. The register change resets at reboot; use
configure_rtx5060_p2p_host.sh --apply to install this helper as a boot service.

  --check   Show ACS control words without changing them (default).
  --apply   Set supported ACS control words to 0000. Must run as root.
EOF
}

mode="${1:---check}"
case "${mode}" in
  --check | --apply) ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if ! command -v lspci >/dev/null || ! command -v setpci >/dev/null; then
  echo "lspci and setpci (the pciutils package) are required" >&2
  exit 1
fi

if [[ "${mode}" == "--apply" && "${EUID}" -ne 0 ]]; then
  echo "--apply must run as root: sudo $0 --apply" >&2
  exit 1
fi

mapfile -t bridges < <(
  lspci -Dn -d 1000:c010 | awk '$2 == "0604:" {print $1}'
)

if [[ "${#bridges[@]}" -eq 0 ]]; then
  echo "No Broadcom/LSI PEX88048 bridges (1000:c010) found" >&2
  exit 1
fi

supported=0
changed=0
for bdf in "${bridges[@]}"; do
  if ! current="$(setpci -s "${bdf}" ECAP_ACS+0x6.w 2>/dev/null)"; then
    continue
  fi
  supported=$((supported + 1))

  if [[ "${mode}" == "--apply" && "${current}" != "0000" ]]; then
    setpci -s "${bdf}" ECAP_ACS+0x6.w=0000
    updated="$(setpci -s "${bdf}" ECAP_ACS+0x6.w)"
    printf '%s ACSCtl %s -> %s\n' "${bdf}" "${current}" "${updated}"
    changed=$((changed + 1))
  else
    printf '%s ACSCtl %s\n' "${bdf}" "${current}"
  fi
done

if [[ "${supported}" -eq 0 ]]; then
  echo "The bridges were found, but ACS capability registers were not readable" >&2
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Retry the check with: sudo $0 --check" >&2
  fi
  exit 1
fi

if [[ "${mode}" == "--apply" ]]; then
  echo "Updated ${changed} of ${supported} ACS-capable PLX bridges."
  echo "This live setting resets at reboot. The sglang-plx-acs systemd service"
  echo "installed by configure_rtx5060_p2p_host.sh reapplies it automatically."
fi
