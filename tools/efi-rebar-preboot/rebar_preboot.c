#include <efi.h>
#include <efilib.h>
#include <efipciio.h>

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

#ifndef REBAR_AUTO_APPLY
#define REBAR_AUTO_APPLY 0
#endif

#define NVIDIA_VENDOR_ID       0x10de
#define RTX_5060_TI_DEVICE_ID  0x2d04
#define PCI_COMMAND_OFFSET     0x04
#define PCI_COMMAND_MEMORY     0x0002
#define PCI_BAR1_LOW_OFFSET    0x14
#define PCI_BAR1_HIGH_OFFSET   0x18
#define PCI_EXT_CAP_START      0x100
#define PCI_EXT_CAP_LIMIT      0xffc
#define PCI_EXT_CAP_ID_REBAR   0x0015
#define REBAR_CAP_OFFSET       0x04
#define REBAR_CTRL_OFFSET      0x08
#define REBAR_CTRL_BAR_MASK    0x00000007
#define REBAR_CTRL_SIZE_MASK   0x00001f00
#define REBAR_CTRL_SIZE_SHIFT  8
#define REBAR_CAP_SIZE_SHIFT   4
#define REBAR_SIZE_16_GIB      14
#define EXPECTED_REBAR_BAR     1

/* Stable BDFs on the SPC621D8U-2T/OVH with the two populated PLX islands. */
static const UINT8 gpu_buses[] = {
    0x8f, 0x92, 0x93, 0x94, 0x95, 0x98, 0x9a, 0x9b,
    0xc8, 0xcb, 0xcc, 0xcd, 0xd1, 0xd2, 0xd3, 0xd4,
};

typedef struct {
    EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *root;
    UINT8 bus;
    UINT16 command;
    UINT32 bar1_low;
    UINT32 bar1_high;
    UINT32 rebar_control;
    UINT16 rebar_offset;
    BOOLEAN control_changed;
    BOOLEAN bars_cleared;
} GPU_STATE;

#if !REBAR_AUTO_APPLY
static EFI_SYSTEM_TABLE *system_table;
#endif

/*
 * EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL encodes extended config offsets in bits
 * 32..63.  The old three-argument GNU-EFI EFI_PCI_ADDRESS macro cannot do
 * this, so do not replace this helper with that macro.
 */
static UINT64
pci_address(UINT8 bus, UINT8 device, UINT8 function, UINT16 reg)
{
    UINT64 address;

    address = ((UINT64)bus << 24) |
              ((UINT64)device << 16) |
              ((UINT64)function << 8);
    if (reg < 0x100)
        return address | reg;
    return address | ((UINT64)reg << 32);
}

static EFI_STATUS
pci_read16(EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *root, UINT8 bus, UINT16 reg,
           UINT16 *value)
{
    return uefi_call_wrapper(root->Pci.Read, 5, root,
                             EfiPciIoWidthUint16,
                             pci_address(bus, 0, 0, reg), 1, value);
}

static EFI_STATUS
pci_read32(EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *root, UINT8 bus, UINT16 reg,
           UINT32 *value)
{
    return uefi_call_wrapper(root->Pci.Read, 5, root,
                             EfiPciIoWidthUint32,
                             pci_address(bus, 0, 0, reg), 1, value);
}

static EFI_STATUS
pci_write16(EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *root, UINT8 bus, UINT16 reg,
            UINT16 *value)
{
    return uefi_call_wrapper(root->Pci.Write, 5, root,
                             EfiPciIoWidthUint16,
                             pci_address(bus, 0, 0, reg), 1, value);
}

static EFI_STATUS
pci_write32(EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *root, UINT8 bus, UINT16 reg,
            UINT32 *value)
{
    return uefi_call_wrapper(root->Pci.Write, 5, root,
                             EfiPciIoWidthUint32,
                             pci_address(bus, 0, 0, reg), 1, value);
}

static EFI_STATUS
find_rebar_capability(EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *root, UINT8 bus,
                      UINT16 *offset)
{
    UINT16 current = PCI_EXT_CAP_START;
    UINT16 next;
    UINT32 header;
    UINTN hops;
    EFI_STATUS status;

    for (hops = 0; hops < 256; ++hops) {
        status = pci_read32(root, bus, current, &header);
        if (EFI_ERROR(status))
            return status;
        if (header == 0 || header == 0xffffffff)
            return EFI_NOT_FOUND;
        if ((header & 0xffff) == PCI_EXT_CAP_ID_REBAR) {
            *offset = current;
            return EFI_SUCCESS;
        }

        next = (UINT16)((header >> 20) & 0x0fff);
        if (next < PCI_EXT_CAP_START || next > PCI_EXT_CAP_LIMIT ||
            (next & 3) != 0 || next == current)
            return EFI_NOT_FOUND;
        current = next;
    }
    return EFI_VOLUME_CORRUPTED;
}

#if !REBAR_AUTO_APPLY
static EFI_STATUS
wait_for_key(EFI_INPUT_KEY *key)
{
    EFI_STATUS status;
    UINTN event_index;

    for (;;) {
        status = uefi_call_wrapper(system_table->ConIn->ReadKeyStroke, 2,
                                   system_table->ConIn, key);
        if (status != EFI_NOT_READY)
            return status;
        status = uefi_call_wrapper(BS->WaitForEvent, 3, 1,
                                   &system_table->ConIn->WaitForKey,
                                   &event_index);
        if (EFI_ERROR(status))
            return status;
    }
}
#endif

static VOID
rollback(GPU_STATE *gpus, UINTN count)
{
    INTN index;

    Print(L"\r\nRolling back changed PCI configuration...\r\n");
    for (index = (INTN)count - 1; index >= 0; --index) {
        GPU_STATE *gpu = &gpus[index];

        if (gpu->control_changed) {
            pci_write32(gpu->root, gpu->bus,
                        (UINT16)(gpu->rebar_offset + REBAR_CTRL_OFFSET),
                        &gpu->rebar_control);
        }
        if (gpu->bars_cleared) {
            pci_write32(gpu->root, gpu->bus, PCI_BAR1_LOW_OFFSET,
                        &gpu->bar1_low);
            pci_write32(gpu->root, gpu->bus, PCI_BAR1_HIGH_OFFSET,
                        &gpu->bar1_high);
        }
        pci_write16(gpu->root, gpu->bus, PCI_COMMAND_OFFSET,
                    &gpu->command);
    }
}

static EFI_STATUS
apply_rebar(GPU_STATE *gpus, UINTN count)
{
    UINT16 command;
    UINT32 control;
    UINT32 zero = 0;
    UINTN index;
    EFI_STATUS status;

    /* First stop every GPU from decoding its old BAR1 address. */
    for (index = 0; index < count; ++index) {
        command = (UINT16)(gpus[index].command & ~PCI_COMMAND_MEMORY);
        status = pci_write16(gpus[index].root, gpus[index].bus,
                             PCI_COMMAND_OFFSET, &command);
        if (EFI_ERROR(status)) {
            Print(L"GPU %u (%02x:00.0): command write failed: %r\r\n",
                  index, gpus[index].bus, status);
            rollback(gpus, index + 1);
            return status;
        }
    }

    for (index = 0; index < count; ++index) {
        control = (gpus[index].rebar_control & ~REBAR_CTRL_SIZE_MASK) |
                  (REBAR_SIZE_16_GIB << REBAR_CTRL_SIZE_SHIFT);
        status = pci_write32(gpus[index].root, gpus[index].bus,
                             (UINT16)(gpus[index].rebar_offset +
                                      REBAR_CTRL_OFFSET),
                             &control);
        if (EFI_ERROR(status)) {
            Print(L"GPU %u (%02x:00.0): ReBAR write failed: %r\r\n",
                  index, gpus[index].bus, status);
            rollback(gpus, count);
            return status;
        }
        gpus[index].control_changed = TRUE;

        status = pci_read32(gpus[index].root, gpus[index].bus,
                            (UINT16)(gpus[index].rebar_offset +
                                     REBAR_CTRL_OFFSET),
                            &control);
        if (EFI_ERROR(status) ||
            ((control & REBAR_CTRL_SIZE_MASK) >> REBAR_CTRL_SIZE_SHIFT) !=
                REBAR_SIZE_16_GIB) {
            Print(L"GPU %u (%02x:00.0): ReBAR readback failed (%08x, %r)\r\n",
                  index, gpus[index].bus, control, status);
            rollback(gpus, count);
            return EFI_DEVICE_ERROR;
        }

        status = pci_write32(gpus[index].root, gpus[index].bus,
                             PCI_BAR1_LOW_OFFSET, &zero);
        if (!EFI_ERROR(status)) {
            /* Mark early so rollback also covers a subsequent high write. */
            gpus[index].bars_cleared = TRUE;
            status = pci_write32(gpus[index].root, gpus[index].bus,
                                 PCI_BAR1_HIGH_OFFSET, &zero);
        }
        if (EFI_ERROR(status)) {
            Print(L"GPU %u (%02x:00.0): BAR1 clear failed: %r\r\n",
                  index, gpus[index].bus, status);
            rollback(gpus, count);
            return status;
        }
        Print(L"GPU %u (%02x:00.0): BAR1 set to 16 GiB and unassigned\r\n",
              index, gpus[index].bus);
    }

    return EFI_SUCCESS;
}

EFI_STATUS
efi_main(EFI_HANDLE image_handle, EFI_SYSTEM_TABLE *st)
{
    EFI_GUID root_bridge_guid = EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL_GUID;
    EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *root;
    EFI_HANDLE *handles = NULL;
    GPU_STATE gpus[ARRAY_SIZE(gpu_buses)];
#if !REBAR_AUTO_APPLY
    EFI_INPUT_KEY key;
#endif
    EFI_STATUS status;
    UINTN handle_count = 0;
    UINTN found = 0;
    UINTN bus_index;
    UINTN handle_index;
    UINT32 id;
    UINT32 capability;
    UINT32 control;
    UINT16 offset;

    InitializeLib(image_handle, st);
#if !REBAR_AUTO_APPLY
    system_table = st;
#endif
    SetMem(gpus, sizeof(gpus), 0);

    Print(L"SPC621D8U-2T/OVH RTX 5060 Ti ReBAR preboot tool\r\n");
    Print(L"Read-only validation first; target is BAR1 = 16 GiB.\r\n\r\n");
#if REBAR_AUTO_APPLY
    Print(L"Automatic mode: changes occur only after all safety checks pass.\r\n\r\n");
#endif

    status = uefi_call_wrapper(BS->LocateHandleBuffer, 5, ByProtocol,
                               &root_bridge_guid, NULL, &handle_count,
                               &handles);
    if (EFI_ERROR(status)) {
        Print(L"Could not locate PCI root bridges: %r\r\n", status);
        goto done;
    }

    for (bus_index = 0; bus_index < ARRAY_SIZE(gpu_buses); ++bus_index) {
        root = NULL;
        for (handle_index = 0; handle_index < handle_count; ++handle_index) {
            EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL *candidate = NULL;

            status = uefi_call_wrapper(BS->HandleProtocol, 3,
                                       handles[handle_index],
                                       &root_bridge_guid,
                                       (VOID **)&candidate);
            if (EFI_ERROR(status) || candidate == NULL ||
                candidate->SegmentNumber != 0)
                continue;
            status = pci_read32(candidate, gpu_buses[bus_index], 0, &id);
            if (!EFI_ERROR(status) && id != 0xffffffff && id != 0) {
                root = candidate;
                break;
            }
        }

        if (root == NULL || (id & 0xffff) != NVIDIA_VENDOR_ID ||
            ((id >> 16) & 0xffff) != RTX_5060_TI_DEVICE_ID) {
            Print(L"MISSING/MISMATCH: 0000:%02x:00.0 (ID %08x)\r\n",
                  gpu_buses[bus_index], root == NULL ? 0xffffffff : id);
            status = EFI_NOT_FOUND;
            goto done;
        }

        status = find_rebar_capability(root, gpu_buses[bus_index], &offset);
        if (EFI_ERROR(status)) {
            Print(L"GPU %u (%02x:00.0): physical ReBAR cap not found: %r\r\n",
                  bus_index, gpu_buses[bus_index], status);
            goto done;
        }
        status = pci_read32(root, gpu_buses[bus_index],
                            (UINT16)(offset + REBAR_CAP_OFFSET), &capability);
        if (!EFI_ERROR(status))
            status = pci_read32(root, gpu_buses[bus_index],
                                (UINT16)(offset + REBAR_CTRL_OFFSET), &control);
        if (EFI_ERROR(status)) {
            Print(L"GPU %u (%02x:00.0): ReBAR read failed: %r\r\n",
                  bus_index, gpu_buses[bus_index], status);
            goto done;
        }
        if ((control & REBAR_CTRL_BAR_MASK) != EXPECTED_REBAR_BAR ||
            (capability & (1U << (REBAR_SIZE_16_GIB +
                                  REBAR_CAP_SIZE_SHIFT))) == 0) {
            Print(L"GPU %u (%02x:00.0): unsafe ReBAR layout cap=%08x ctrl=%08x\r\n",
                  bus_index, gpu_buses[bus_index], capability, control);
            status = EFI_UNSUPPORTED;
            goto done;
        }

        gpus[found].root = root;
        gpus[found].bus = gpu_buses[bus_index];
        gpus[found].rebar_offset = offset;
        gpus[found].rebar_control = control;
        status = pci_read16(root, gpu_buses[bus_index], PCI_COMMAND_OFFSET,
                            &gpus[found].command);
        if (!EFI_ERROR(status))
            status = pci_read32(root, gpu_buses[bus_index],
                                PCI_BAR1_LOW_OFFSET, &gpus[found].bar1_low);
        if (!EFI_ERROR(status))
            status = pci_read32(root, gpu_buses[bus_index],
                                PCI_BAR1_HIGH_OFFSET, &gpus[found].bar1_high);
        if (EFI_ERROR(status)) {
            Print(L"GPU %u (%02x:00.0): PCI state read failed: %r\r\n",
                  bus_index, gpu_buses[bus_index], status);
            goto done;
        }

        Print(L"GPU %u  0000:%02x:00.0  cap=%03x ctrl=%08x size-code=%u\r\n",
              bus_index, gpu_buses[bus_index], offset, control,
              (control & REBAR_CTRL_SIZE_MASK) >> REBAR_CTRL_SIZE_SHIFT);
        ++found;
    }

    if (found != ARRAY_SIZE(gpu_buses)) {
        status = EFI_NOT_FOUND;
        goto done;
    }

    Print(L"\r\nValidated all 16 exact 10de:2d04 GPUs and 16 GiB support.\r\n");
#if REBAR_AUTO_APPLY
    Print(L"Automatic mode: applying the validated BAR1 configuration.\r\n");
#else
    Print(L"Press uppercase A to apply for this boot; any other key exits.\r\n");
    status = wait_for_key(&key);
    if (EFI_ERROR(status) || key.UnicodeChar != L'A') {
        Print(L"No changes made.\r\n");
        if (!EFI_ERROR(status))
            status = EFI_SUCCESS;
        goto done;
    }
#endif

    status = apply_rebar(gpus, found);
    if (!EFI_ERROR(status)) {
        Print(L"\r\nSUCCESS: all GPU BAR1 controls now request 16 GiB.\r\n");
        Print(L"BAR1 addresses are unassigned and memory decode is disabled.\r\n");
        Print(L"Return to GRUB and boot Linux; pci=realloc=on must assign them.\r\n");
    }

done:
    if (handles != NULL)
        uefi_call_wrapper(BS->FreePool, 1, handles);
#if REBAR_AUTO_APPLY
    if (EFI_ERROR(status))
        Print(L"\r\nFAILED: no partial configuration is intentionally retained.\r\n");
    Print(L"Returning to GRUB.\r\n");
#else
    Print(L"\r\nPress any key to return to GRUB.\r\n");
    wait_for_key(&key);
#endif
    return status;
}
