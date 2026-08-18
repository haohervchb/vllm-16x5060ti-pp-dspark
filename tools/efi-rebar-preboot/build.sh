#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gnu_efi_root=${GNU_EFI_ROOT:-/}
efi_include="$gnu_efi_root/usr/include/efi"
efi_lib="$gnu_efi_root/usr/lib"
build_dir="$script_dir/build"

for required in \
  "$efi_include/efi.h" \
  "$efi_include/efipciio.h" \
  "$efi_lib/crt0-efi-x86_64.o" \
  "$efi_lib/elf_x86_64_efi.lds" \
  "$efi_lib/libefi.a" \
  "$efi_lib/libgnuefi.a"; do
  if [[ ! -e "$required" ]]; then
    echo "missing GNU-EFI file: $required" >&2
    echo "install gnu-efi, or set GNU_EFI_ROOT to an extracted package root" >&2
    exit 1
  fi
done

mkdir -p "$build_dir"

build_image() {
  local image_name="$1"
  local object_name="$2"
  shift 2

  gcc \
    -I"$efi_include" \
    -I"$efi_include/x86_64" \
    -I"$efi_include/protocol" \
    -DEFI_FUNCTION_WRAPPER \
    -fpic -ffreestanding -fno-stack-protector -fno-stack-check \
    -fshort-wchar -mno-red-zone -maccumulate-outgoing-args \
    -Wall -Wextra -Werror -O2 \
    "$@" \
    -c "$script_dir/rebar_preboot.c" \
    -o "$build_dir/${object_name}.o"

  ld \
    -nostdlib -znocombreloc \
    -T "$efi_lib/elf_x86_64_efi.lds" \
    -shared -Bsymbolic \
    "$efi_lib/crt0-efi-x86_64.o" \
    "$build_dir/${object_name}.o" \
    -L"$efi_lib" -lefi -lgnuefi \
    -o "$build_dir/${object_name}.so"

  objcopy \
    -j .text -j .sdata -j .data -j .dynamic -j .dynsym \
    -j .rel -j .rela -j .reloc \
    --target=efi-app-x86_64 \
    "$build_dir/${object_name}.so" \
    "$build_dir/${image_name}.efi"

  sha256sum "$build_dir/${image_name}.efi"
  file "$build_dir/${image_name}.efi"
}

build_image ReBarPreboot rebar_preboot
build_image ReBarPrebootAuto rebar_preboot_auto -DREBAR_AUTO_APPLY=1
