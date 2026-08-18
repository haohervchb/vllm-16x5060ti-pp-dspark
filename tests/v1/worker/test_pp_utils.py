# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.pp_utils import PPHandler

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PPHandler uses a CUDA side stream"
)


def _make_handler(
    monkeypatch,
    *,
    is_last_rank: bool,
    num_speculative_steps: int,
    relay_draft_tokens: bool,
    world_size: int = 2,
) -> PPHandler:
    pp_group = SimpleNamespace(
        is_last_rank=is_last_rank,
        last_rank=world_size - 1,
        world_size=world_size,
        make_sibling_device_group=lambda group_desc: object(),
    )
    monkeypatch.setattr(pp_utils, "get_pp_group", lambda: pp_group)
    return PPHandler(
        max_num_reqs=8,
        num_speculative_steps=num_speculative_steps,
        device=torch.device("cuda"),
        relay_draft_tokens=relay_draft_tokens,
    )


def _record_broadcasts(monkeypatch) -> list[torch.Tensor]:
    calls: list[torch.Tensor] = []
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, src, group: calls.append(tensor)
    )
    return calls


def _make_input_batch(num_reqs: int = 3, *, needs_sample: bool = True):
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        prefill_len_np=np.full(num_reqs, 4, dtype=np.int32),
        num_scheduled_tokens=np.full(num_reqs, 4, dtype=np.int32),
        max_seq_len_np=np.full(num_reqs, 100 if needs_sample else 1, dtype=np.int32),
        idx_mapping=torch.arange(num_reqs, device="cuda"),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
    )


def _send_step(
    handler: PPHandler, input_batch, *, width: int, with_draft: bool
) -> None:
    num_reqs = input_batch.num_reqs
    sampled = torch.zeros(num_reqs, width, dtype=torch.int64, device="cuda")
    counts = torch.zeros(num_reqs, dtype=torch.int32, device="cuda")
    handler.broadcast(sampled, counts, counts, input_batch)
    if with_draft:
        draft = torch.zeros(
            num_reqs, handler.max_sample_len - 1, dtype=torch.int64, device="cuda"
        )
        handler.broadcast_draft(draft, input_batch)


@requires_cuda
@pytest.mark.parametrize("width,num_spec", [(1, 1), (1, 3), (2, 3)])
def test_broadcast_pads_sampled_tokens(monkeypatch, width, num_spec):
    handler = _make_handler(
        monkeypatch,
        is_last_rank=True,
        num_speculative_steps=num_spec,
        relay_draft_tokens=True,
    )
    calls = _record_broadcasts(monkeypatch)
    input_batch = _make_input_batch()

    _send_step(handler, input_batch, width=width, with_draft=False)

    sampled = calls[0]
    assert sampled.shape == (input_batch.num_reqs, handler.max_sample_len)
    assert (sampled[:, :width] == 0).all()
    assert (sampled[:, width:] == -1).all()


@requires_cuda
@pytest.mark.parametrize("relay_draft_tokens,expected_calls", [(True, 3), (False, 2)])
def test_send_and_receive_collective_counts_match(
    monkeypatch, relay_draft_tokens, expected_calls
):
    sender = _make_handler(
        monkeypatch,
        is_last_rank=True,
        num_speculative_steps=3,
        relay_draft_tokens=relay_draft_tokens,
    )
    calls = _record_broadcasts(monkeypatch)
    _send_step(
        sender,
        _make_input_batch(),
        width=1,
        with_draft=relay_draft_tokens,
    )
    assert len(calls) == expected_calls

    receiver = _make_handler(
        monkeypatch,
        is_last_rank=False,
        num_speculative_steps=3,
        relay_draft_tokens=relay_draft_tokens,
    )
    calls.clear()
    assert receiver.receive(_make_input_batch())
    assert len(calls) == expected_calls


@requires_cuda
def test_relayed_draft_tokens_survive_deferred_receive(monkeypatch):
    receiver = _make_handler(
        monkeypatch,
        is_last_rank=False,
        num_speculative_steps=3,
        relay_draft_tokens=True,
        world_size=2,
    )
    _record_broadcasts(monkeypatch)
    receiver.receive(_make_input_batch())

    outputs = None
    for _ in range(3):
        outputs = receiver.get_prev_sampled_outputs()
        if outputs is not None:
            break

    assert outputs is not None
    assert outputs["draft_tokens"] is not None
    assert outputs["draft_tokens"].shape == (3, receiver.max_sample_len - 1)
