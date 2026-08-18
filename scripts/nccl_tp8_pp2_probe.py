#!/usr/bin/env python3
"""Exercise the NCCL communicator layout used by TP=8, PP=2.

Run with 16 local workers after verifying raw CUDA P2P, for example::

    NCCL_CUMEM_ENABLE=0 NCCL_P2P_LEVEL=PXB \
        torchrun --standalone --nproc-per-node=16 \
        scripts/nccl_tp8_pp2_probe.py

The two tensor-parallel groups are ranks 0..7 and 8..15. Pipeline-parallel
traffic runs between matching ranks in the two groups.
"""

import os
import time

import torch
import torch.distributed as dist


TP_SIZE = 8
PP_SIZE = 2
WORLD_SIZE = TP_SIZE * PP_SIZE


def _new_group_for_rank(group_ranks):
    selected = None
    rank = dist.get_rank()
    for ranks in group_ranks:
        group = dist.new_group(ranks=ranks, backend="nccl")
        if rank in ranks:
            selected = group
    assert selected is not None
    return selected


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != WORLD_SIZE:
        raise RuntimeError(f"This probe requires exactly {WORLD_SIZE} workers")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    tp_ranks = [
        list(range(pp_rank * TP_SIZE, (pp_rank + 1) * TP_SIZE))
        for pp_rank in range(PP_SIZE)
    ]
    pp_ranks = [[tp_rank, TP_SIZE + tp_rank] for tp_rank in range(TP_SIZE)]
    tp_group = _new_group_for_rank(tp_ranks)
    pp_group = _new_group_for_rank(pp_ranks)

    # A reasonably large TP all-reduce verifies and times the intra-island
    # path. Values also prove that every rank participated correctly.
    tensor = torch.full(
        (4 * 1024 * 1024,), rank + 1, dtype=torch.float32, device="cuda"
    )
    expected = sum(r + 1 for r in tp_ranks[rank // TP_SIZE])
    for _ in range(3):
        dist.all_reduce(tensor, group=tp_group)
        tensor.fill_(rank + 1)
    torch.cuda.synchronize()

    tensor.fill_(rank + 1)
    start = time.perf_counter()
    iterations = 10
    for _ in range(iterations):
        dist.all_reduce(tensor, group=tp_group)
        tensor.fill_(rank + 1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    dist.all_reduce(tensor, group=tp_group)
    if tensor[0].item() != expected:
        raise RuntimeError(
            f"rank {rank}: TP all-reduce got {tensor[0].item()}, expected {expected}"
        )

    # Exercise both directions of each cross-island PP pair. NCCL should use
    # SHM here when NCCL_P2P_LEVEL=PXB because these links have NODE distance.
    peer = rank + TP_SIZE if rank < TP_SIZE else rank - TP_SIZE
    pp_send = torch.full((1024 * 1024,), rank, dtype=torch.int32, device="cuda")
    pp_recv = torch.full((1024 * 1024,), -1, dtype=torch.int32, device="cuda")
    if rank < TP_SIZE:
        dist.send(pp_send, dst=peer, group=pp_group)
        dist.recv(pp_recv, src=peer, group=pp_group)
    else:
        dist.recv(pp_recv, src=peer, group=pp_group)
        dist.send(pp_send, dst=peer, group=pp_group)
    torch.cuda.synchronize()

    if pp_recv[0].item() != peer:
        raise RuntimeError(
            f"rank {rank}: PP transfer got {pp_recv[0].item()}, expected {peer}"
        )

    # SGLang creates a world NCCL group before its model-parallel groups. A
    # final world collective checks that this communicator also remains usable.
    world_value = torch.tensor([rank + 1], dtype=torch.int32, device="cuda")
    dist.all_reduce(world_value)
    if world_value.item() != sum(range(1, WORLD_SIZE + 1)):
        raise RuntimeError(f"rank {rank}: world all-reduce validation failed")

    if rank in (0, TP_SIZE):
        size_bytes = tensor.numel() * tensor.element_size()
        algorithm_gbps = size_bytes * iterations / elapsed / 1e9
        print(
            f"rank {rank}: TP8 all-reduce passed, "
            f"{elapsed / iterations * 1e3:.2f} ms/iteration, "
            f"{algorithm_gbps:.2f} GB/s algorithm bandwidth",
            flush=True,
        )

    dist.barrier()
    if rank == 0:
        print("TP=8, PP=2 NCCL topology probe passed on all 16 ranks", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
