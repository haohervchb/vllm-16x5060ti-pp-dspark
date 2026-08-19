#!/usr/bin/env python3
"""Validate vLLM custom all-reduce on a fully-P2P PLX GPU group.

Run with torchrun and one process per GPU. The test covers the eager buffer
path and the registered-buffer CUDA-graph path used during vLLM decoding.
"""

import argparse
import os

os.environ.setdefault("VLLM_ALLOW_PCI_CUSTOM_ALLREDUCE", "1")
os.environ.setdefault("VLLM_CUSTOM_ALLREDUCE_ALGO", "2stage")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def validate_tensor(actual: torch.Tensor, expected: float, label: str) -> None:
    torch.accelerator.synchronize()
    reference = torch.full_like(actual, expected)
    if not torch.equal(actual, reference):
        max_error = (actual.float() - reference.float()).abs().max().item()
        raise RuntimeError(f"{label} failed; maximum absolute error: {max_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--ops-per-graph", type=int, default=1)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[1, 6, 8, 16])
    parser.add_argument("--hidden-size", type=int, default=7168)
    args = parser.parse_args()

    allocator_config = os.environ["PYTORCH_CUDA_ALLOC_CONF"].lower()
    if "expandable_segments:true" in allocator_config or (
        "backend:cudamallocasync" in allocator_config
    ):
        raise SystemExit(
            "PLX custom all-reduce CUDA graphs require the native allocator; "
            "set PYTORCH_CUDA_ALLOC_CONF=backend:native"
        )

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    cpu_group = dist.new_group(backend="gloo")
    communicator = CustomAllreduce(group=cpu_group, device=device)
    if communicator.disabled:
        raise RuntimeError("custom all-reduce did not initialize")

    rank_value = float(rank + 1)
    expected = float(world_size * (world_size + 1) // 2)

    for token_count in args.token_counts:
        shape = (token_count, args.hidden_size)

        eager_input = torch.full(shape, rank_value, dtype=torch.bfloat16, device=device)
        for iteration in range(args.iterations):
            eager_output = communicator.custom_all_reduce(eager_input)
            assert eager_output is not None
            validate_tensor(
                eager_output, expected, f"eager {shape} iteration {iteration}"
            )

        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            graph_input = torch.full(
                shape, rank_value, dtype=torch.bfloat16, device=device
            )
            for _ in range(3):
                communicator.custom_all_reduce(graph_input)
            with communicator.capture():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(args.ops_per_graph):
                        graph_output = communicator.custom_all_reduce(graph_input)
        assert graph_output is not None
        torch.accelerator.synchronize()
        for iteration in range(args.iterations):
            graph.replay()
            validate_tensor(
                graph_output, expected, f"CUDA graph {shape} iteration {iteration}"
            )

        if rank == 0:
            print(
                f"PASS: {shape}, eager and CUDA graph, {args.iterations} "
                f"iterations, {args.ops_per_graph} collectives per graph"
            )

    dist.barrier()
    communicator.close()
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
