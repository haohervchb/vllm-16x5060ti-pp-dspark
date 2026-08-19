#!/usr/bin/env bash
# Install the complete pre-OS ReBAR and post-boot ACS configuration for the
# verified SPC621D8U-2T/OVH + 16 x RTX 5060 Ti layout.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
host_config="$repo_root/scripts/configure_rtx5060_p2p_host.sh"
acs_source="$repo_root/scripts/configure_plx_acs.sh"
grub_target=/etc/grub.d/09_sglang_rebar
efi_dir=/boot/efi/EFI/sglang
acs_target=/usr/local/sbin/sglang-configure-plx-acs
doc_dir=/usr/local/share/doc/sglang-rtx5060-rebar
rearm_unit=/etc/systemd/system/sglang-rebar-rearm.service
acs_unit=/etc/systemd/system/sglang-plx-acs.service

expected_bdfs=(
  8f:00.0 92:00.0 93:00.0 94:00.0
  95:00.0 98:00.0 9a:00.0 9b:00.0
  c8:00.0 cb:00.0 cc:00.0 cd:00.0
  d1:00.0 d2:00.0 d3:00.0 d4:00.0
)

usage() {
  cat <<'EOF'
Usage: sudo install-persistent.sh [--install | --check | --uninstall]

  --install    Validate the exact hardware, build and install both EFI apps,
               install the GRUB/kernel/NVIDIA configuration, enable automatic
               ReBAR rearming and PLX ACS services, and arm the first reboot.
               This is the default. It does not reboot the machine.
  --check      Show installed files, service state, next GRUB entry, and the
               active BAR/MMIO host configuration.
  --uninstall  Remove only files and services installed by this tool. It does
               not restore or alter motherboard firmware.
EOF
}

require_root() {
  if (( EUID != 0 )); then
    echo "error: run this command with sudo" >&2
    exit 1
  fi
}

clear_rebar_grub_state() {
  # Never leave either ReBAR entry as a saved/default GRUB choice. The EFI
  # helper returns to GRUB, so a persistent ReBAR saved_entry can recurse back
  # into the helper instead of reaching Ubuntu.
  local saved
  saved=$(grub-editenv list 2>/dev/null | awk -F= '$1 == "saved_entry" {print $2}')
  case "$saved" in
    rebar-preboot|rebar-preboot-auto)
      grub-editenv /boot/grub/grubenv unset saved_entry || true
      ;;
  esac
  grub-editenv /boot/grub/grubenv unset next_entry 2>/dev/null || true
}

install_dependencies() {
  local packages=()

  command -v gcc >/dev/null 2>&1 || packages+=(build-essential)
  command -v objcopy >/dev/null 2>&1 || packages+=(binutils)
  command -v lspci >/dev/null 2>&1 || packages+=(pciutils)
  command -v grub-reboot >/dev/null 2>&1 || packages+=(grub-common)
  command -v mokutil >/dev/null 2>&1 || packages+=(mokutil)
  if [[ ! -e /usr/include/efi/efi.h ||
        ! -e /usr/lib/crt0-efi-x86_64.o ||
        ! -e /usr/lib/libefi.a ]]; then
    packages+=(gnu-efi)
  fi

  if (( ${#packages[@]} > 0 )); then
    echo "Installing required Ubuntu packages: ${packages[*]}"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
  fi
}

verify_platform() {
  local bdf device_id details plx_count

  if [[ ! -d /sys/firmware/efi ]]; then
    echo "error: this installation is not booted in UEFI mode" >&2
    exit 1
  fi
  if ! mountpoint -q /boot/efi; then
    echo "error: the EFI system partition is not mounted at /boot/efi" >&2
    exit 1
  fi
  if mokutil --sb-state 2>/dev/null | grep -qi '^SecureBoot enabled'; then
    echo "error: Secure Boot is enabled; the locally built EFI app is unsigned" >&2
    echo "disable Secure Boot or sign the EFI binary before installing" >&2
    exit 1
  fi

  echo "Validating the exact 16-GPU PCI layout..."
  for bdf in "${expected_bdfs[@]}"; do
    device_id=$(lspci -Dn -s "$bdf" | awk 'NR == 1 { print tolower($3) }')
    if [[ "$device_id" != 10de:2d04 ]]; then
      echo "error: expected RTX 5060 Ti 10de:2d04 at $bdf, found ${device_id:-nothing}" >&2
      exit 1
    fi

    details=$(lspci -s "$bdf" -vv)
    if ! grep -q 'Physical Resizable BAR' <<<"$details" ||
       ! grep -Eq 'BAR 1:.*supported:.*16GB' <<<"$details"; then
      echo "error: $bdf does not expose physical BAR1 with 16GB support" >&2
      exit 1
    fi
    printf '  %s  10de:2d04  BAR1 supports 16GB\n' "$bdf"
  done

  plx_count=$(lspci -Dn -d 1000:c010 |
    awk '$2 == "0604:" { count++ } END { print count+0 }')
  if (( plx_count < 32 )); then
    echo "error: expected at least 32 PEX bridge functions, found $plx_count" >&2
    exit 1
  fi
  echo "Validated $plx_count ACS-capable PEX bridge functions."
}

install_configuration() {
  local esp_uuid rendered_grub

  install_dependencies
  verify_platform
  clear_rebar_grub_state

  esp_uuid=$(findmnt -n -o UUID /boot/efi)
  if [[ -z "$esp_uuid" ]]; then
    echo "error: could not determine the EFI system partition UUID" >&2
    exit 1
  fi

  bash "$script_dir/build.sh"
  install -D -m 0644 "$script_dir/build/ReBarPreboot.efi" \
    "$efi_dir/ReBarPreboot.efi"
  install -D -m 0644 "$script_dir/build/ReBarPrebootAuto.efi" \
    "$efi_dir/ReBarPrebootAuto.efi"

  rendered_grub=$(mktemp)
  trap 'rm -f -- "$rendered_grub"' EXIT
  sed "s/@ESP_UUID@/$esp_uuid/g" \
    "$script_dir/grub-menuentry.template" >"$rendered_grub"
  install -m 0755 "$rendered_grub" "$grub_target"

  install -D -m 0755 "$acs_source" "$acs_target"
  install -D -m 0644 "$script_dir/sglang-rebar-rearm.service" "$rearm_unit"
  install -D -m 0644 "$script_dir/sglang-plx-acs.service" "$acs_unit"
  install -D -m 0644 "$script_dir/README.md" "$doc_dir/README.md"

  "$host_config" --apply

  systemctl daemon-reload
  systemctl enable --now sglang-plx-acs.service
  # Enable persistence for later boots, but arm this boot explicitly only after
  # all files and GRUB state are known-good.
  systemctl enable sglang-rebar-rearm.service
  grub-reboot rebar-preboot-auto

  echo
  echo "Persistent RTX 5060 Ti ReBAR configuration installed."
  echo "GRUB next boot: $(grub-editenv list | awk -F= '$1 == "next_entry" {print $2}')"
  echo "Reboot once. The automatic EFI pass will validate all 16 GPUs and apply"
  echo "16 GiB BAR1. If the EFI helper returns, GRUB will not recursively reload"
  echo "the ReBAR menu entry."
}

check_configuration() {
  local path

  echo "== Persistent installation =="
  for path in \
    "$efi_dir/ReBarPreboot.efi" \
    "$efi_dir/ReBarPrebootAuto.efi" \
    "$grub_target" \
    "$rearm_unit" \
    "$acs_unit" \
    "$acs_target"; do
    if [[ -e "$path" ]]; then
      echo "installed: $path"
    else
      echo "missing:   $path"
    fi
  done

  echo
  echo "== Services =="
  systemctl is-enabled sglang-rebar-rearm.service 2>/dev/null || true
  systemctl is-active sglang-rebar-rearm.service 2>/dev/null || true
  systemctl is-enabled sglang-plx-acs.service 2>/dev/null || true
  systemctl is-active sglang-plx-acs.service 2>/dev/null || true

  echo
  echo "== GRUB environment =="
  grub-editenv list || true

  echo
  "$host_config" --check
}

uninstall_configuration() {
  systemctl disable --now sglang-rebar-rearm.service \
    sglang-plx-acs.service 2>/dev/null || true
  clear_rebar_grub_state

  rm -f -- \
    "$efi_dir/ReBarPreboot.efi" \
    "$efi_dir/ReBarPrebootAuto.efi" \
    "$grub_target" \
    "$rearm_unit" \
    "$acs_unit" \
    "$acs_target" \
    "$doc_dir/README.md"
  rmdir --ignore-fail-on-non-empty -- "$efi_dir" "$doc_dir" 2>/dev/null || true
  systemctl daemon-reload
  "$host_config" --rollback

  echo "Persistent configuration removed. Reboot to return to firmware-sized BARs."
}

require_root
action=${1:---install}
case "$action" in
  --install)
    (( $# == 1 || $# == 0 )) || { usage >&2; exit 2; }
    install_configuration
    ;;
  --check)
    (( $# == 1 )) || { usage >&2; exit 2; }
    check_configuration
    ;;
  --uninstall)
    (( $# == 1 )) || { usage >&2; exit 2; }
    uninstall_configuration
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
