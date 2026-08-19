#!/usr/bin/env bash
# Install persistent, non-interactive pre-OS ReBAR and post-boot ACS setup for
# the verified SPC621D8U-2T/OVH + 16 x RTX 5060 Ti layout.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
host_config="$repo_root/scripts/configure_rtx5060_p2p_host.sh"
acs_source="$repo_root/scripts/configure_plx_acs.sh"
grub_target=/etc/grub.d/41_sglang_rebar_auto
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

  --install    Install one-time persistent configuration. Every later reboot
               automatically applies 16 GiB BAR1 before Ubuntu, with no key
               prompt or interactive ReBAR menu.
  --check      Show installation, service, GRUB and BAR/MMIO state.
  --uninstall  Remove files/services installed by this tool.
EOF
}

require_root() {
  (( EUID == 0 )) || { echo "error: run this command with sudo" >&2; exit 1; }
}

clear_rebar_grub_state() {
  grub-editenv /boot/grub/grubenv unset next_entry 2>/dev/null || true
  local saved
  saved=$(grub-editenv /boot/grub/grubenv list 2>/dev/null |
    awk -F= '$1 == "saved_entry" {print $2}')
  case "$saved" in
    rebar-preboot|rebar-preboot-auto)
      grub-editenv /boot/grub/grubenv unset saved_entry 2>/dev/null || true
      ;;
  esac
}

install_dependencies() {
  local packages=()
  command -v gcc >/dev/null 2>&1 || packages+=(build-essential)
  command -v objcopy >/dev/null 2>&1 || packages+=(binutils)
  command -v lspci >/dev/null 2>&1 || packages+=(pciutils)
  command -v grub-reboot >/dev/null 2>&1 || packages+=(grub-common)
  command -v mokutil >/dev/null 2>&1 || packages+=(mokutil)
  if [[ ! -e /usr/include/efi/efi.h || ! -e /usr/lib/crt0-efi-x86_64.o ||
        ! -e /usr/lib/libefi.a ]]; then
    packages+=(gnu-efi)
  fi
  if (( ${#packages[@]} )); then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
  fi
}

verify_platform() {
  local bdf device_id details plx_count
  [[ -d /sys/firmware/efi ]] || { echo "error: not booted in UEFI mode" >&2; exit 1; }
  mountpoint -q /boot/efi || { echo "error: /boot/efi is not mounted" >&2; exit 1; }
  if mokutil --sb-state 2>/dev/null | grep -qi '^SecureBoot enabled'; then
    echo "error: Secure Boot is enabled; ReBarPrebootAuto.efi is unsigned" >&2
    exit 1
  fi
  for bdf in "${expected_bdfs[@]}"; do
    device_id=$(lspci -Dn -s "$bdf" | awk 'NR==1 {print tolower($3)}')
    [[ "$device_id" == 10de:2d04 ]] || {
      echo "error: expected RTX 5060 Ti at $bdf, found ${device_id:-nothing}" >&2; exit 1; }
    details=$(lspci -s "$bdf" -vv)
    grep -q 'Physical Resizable BAR' <<<"$details" &&
      grep -Eq 'BAR 1:.*supported:.*16GB' <<<"$details" || {
        echo "error: $bdf does not expose 16GB physical BAR1 support" >&2; exit 1; }
  done
  plx_count=$(lspci -Dn -d 1000:c010 | awk '$2=="0604:"{n++}END{print n+0}')
  (( plx_count >= 32 )) || { echo "error: expected >=32 PEX bridges, found $plx_count" >&2; exit 1; }
}

install_configuration() {
  local esp_uuid rendered_grub
  install_dependencies
  verify_platform
  clear_rebar_grub_state

  # Remove every older interactive/early version first. There must never be an
  # uppercase-A entry in a persistent installation.
  rm -f /etc/grub.d/09_sglang_rebar /etc/grub.d/41_sglang_rebar_auto
  rm -f "$efi_dir/ReBarPreboot.efi"

  esp_uuid=$(findmnt -n -o UUID /boot/efi)
  [[ -n "$esp_uuid" ]] || { echo "error: cannot determine ESP UUID" >&2; exit 1; }

  bash "$script_dir/build.sh"
  install -D -m 0644 "$script_dir/build/ReBarPrebootAuto.efi" \
    "$efi_dir/ReBarPrebootAuto.efi"

  rendered_grub=$(mktemp)
  trap 'rm -f -- "$rendered_grub"' EXIT
  sed "s/@ESP_UUID@/$esp_uuid/g" "$script_dir/grub-menuentry.template" >"$rendered_grub"
  install -m 0755 "$rendered_grub" "$grub_target"

  install -D -m 0755 "$acs_source" "$acs_target"
  install -D -m 0644 "$script_dir/sglang-rebar-rearm.service" "$rearm_unit"
  install -D -m 0644 "$script_dir/sglang-plx-acs.service" "$acs_unit"
  install -D -m 0644 "$script_dir/README.md" "$doc_dir/README.md"

  "$host_config" --apply
  update-grub

  systemctl daemon-reload
  systemctl unmask sglang-rebar-rearm.service 2>/dev/null || true
  systemctl enable sglang-rebar-rearm.service
  systemctl enable --now sglang-plx-acs.service

  # Arm exactly the next reboot. After each successful Ubuntu boot the enabled
  # oneshot service arms the following reboot. grub-reboot is one-shot state,
  # while the EFI helper itself is completely non-interactive.
  grub-reboot rebar-preboot-auto

  echo "Persistent automatic 16 GiB ReBAR installed."
  echo "No interactive ReBAR EFI image is installed."
  echo "Every successful Ubuntu boot rearms the automatic preboot for the next reboot."
}

check_configuration() {
  echo "== Files =="
  ls -l "$efi_dir/ReBarPrebootAuto.efi" "$grub_target" "$rearm_unit" "$acs_unit" 2>/dev/null || true
  echo "== Interactive image (must be absent) =="
  test ! -e "$efi_dir/ReBarPreboot.efi" && echo absent || echo ERROR-present
  echo "== Services =="
  systemctl is-enabled sglang-rebar-rearm.service 2>/dev/null || true
  systemctl is-enabled sglang-plx-acs.service 2>/dev/null || true
  echo "== GRUB environment =="
  grub-editenv /boot/grub/grubenv list || true
  echo "== Host =="
  "$host_config" --check
}

uninstall_configuration() {
  systemctl disable --now sglang-rebar-rearm.service sglang-plx-acs.service 2>/dev/null || true
  clear_rebar_grub_state
  rm -f "$efi_dir/ReBarPreboot.efi" "$efi_dir/ReBarPrebootAuto.efi" \
    /etc/grub.d/09_sglang_rebar "$grub_target" "$rearm_unit" "$acs_unit" \
    "$acs_target" "$doc_dir/README.md"
  systemctl daemon-reload
  "$host_config" --rollback
  update-grub
  echo "Persistent ReBAR configuration removed."
}

require_root
action=${1:---install}
case "$action" in
  --install) install_configuration ;;
  --check) check_configuration ;;
  --uninstall) uninstall_configuration ;;
  --help|-h) usage ;;
  *) usage >&2; exit 2 ;;
esac
