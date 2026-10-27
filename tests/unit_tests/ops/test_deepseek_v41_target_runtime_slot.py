# SPDX-License-Identifier: Apache-2.0
"""Stable global state slots for compact target execution lanes."""
import pytest
import torch

from vllm_gaudi.ops import deepseek_v41_batch_attention as attention
from vllm_gaudi.ops.deepseek_v41_batch_target import (
    compressor_sequence_runtime_slot_publish_read,
    compressor_sequence_slot_publish_read,
)


@pytest.fixture
def state_ops(monkeypatch):
    def write(bank, values, rows):
        # Functional CPU facade of the native no-write sentinel contract.
        valid = (rows >= 0) & (rows < bank.shape[0])
        indices = rows.clamp(0, bank.shape[0] - 1).long()
        one_hot = torch.nn.functional.one_hot(indices, num_classes=bank.shape[0]).bool()
        mask = one_hot & valid[:, None]
        bank.copy_(torch.where(
            mask.any(0)[:, None], (values.T @ mask.to(values.dtype)).T, bank))
        return torch.where(valid, rows, -1)

    def read(bank, rows, completion):
        valid = (rows >= 0) & (rows < bank.shape[0])
        return torch.where(valid[:, None], bank.index_select(0, rows.clamp_min(0).long()), 0)

    monkeypatch.setattr(attention, "write_state_rows", write)
    monkeypatch.setattr(attention, "read_state_rows", read)


def test_runtime_slot31_matches_static_c6_and_preserves_other_slots(state_ops):
    history = torch.arange(32 * 8 * 512, dtype=torch.float32).reshape(32 * 8, 512)
    scores = history + 9
    reference_history, reference_scores = history.clone(), scores.clone()
    values = torch.arange(6 * 512, dtype=torch.float32).reshape(6, 512) + 100
    positions = torch.tensor([255, 256, 257, -1, -1, -1], dtype=torch.int32)
    expected = compressor_sequence_slot_publish_read(
        reference_history, reference_scores, 31, values, values + 1, positions)
    actual = compressor_sequence_runtime_slot_publish_read(
        history, scores, torch.tensor(31, dtype=torch.int32), values, values + 1, positions)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert torch.equal(history, reference_history)
    assert torch.equal(scores, reference_scores)


@pytest.mark.parametrize("slot", [-1, 32, 900])
def test_invalid_runtime_owner_is_no_write_zero_read(state_ops, slot):
    history = torch.randn(32 * 8, 512)
    scores = torch.randn_like(history)
    before = history.clone(), scores.clone()
    result = compressor_sequence_runtime_slot_publish_read(
        history, scores, torch.tensor(slot, dtype=torch.int32), torch.ones(6, 512),
        torch.ones(6, 512), torch.arange(6, dtype=torch.int32))
    assert all(torch.count_nonzero(value) == 0 for value in result)
    assert torch.equal(history, before[0])
    assert torch.equal(scores, before[1])


def test_runtime_slot_values_reuse_one_dynamo_graph(state_ops, monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    torch._dynamo.reset()
    graphs = []

    def backend(graph, args):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(compressor_sequence_runtime_slot_publish_read,
                             backend=backend, fullgraph=True, dynamic=False)
    history, scores = torch.zeros(32 * 8, 512), torch.zeros(32 * 8, 512)
    values = torch.ones(6, 512)
    positions = torch.arange(6, dtype=torch.int32)
    for slot in (31, 3, 17, -1, 0):
        compiled(history, scores, torch.tensor(slot, dtype=torch.int32), values, values, positions)
    assert len(graphs) == 1
    torch._dynamo.reset()
