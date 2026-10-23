# SPDX-License-Identifier: Apache-2.0
"""Packed draft decode must not require prefill-only FP8 sidecar weights."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops import deepseek_v41_paged_attention as attention_module


def _attention():
    weight = SimpleNamespace(weight=torch.arange(4096 * 4, dtype=torch.float32).reshape(4096, 4) / 32768)
    return SimpleNamespace(
        weights=SimpleNamespace(wq_b=weight),
        heads=8,
        draft_query_fp8=False,
        dspark_q_bf16_rope=False,
        q_scale_rope=False,
        c6_query_fusion=False,
        linear=lambda value, owner: F.linear(value, owner.weight),
        _rope=lambda value, positions, **kwargs: value,
    )


def test_packed_draft_ten_rows_use_checkpoint_projection_with_prefill_optimization_enabled(monkeypatch):
    monkeypatch.setattr(attention_module.gaudi_envs, 'VLLM_HPU_DSV41_PREFILL_Q_PROJECTION', True)
    attention = _attention()
    value = torch.arange(40, dtype=torch.float32).reshape(10, 4) / 40
    positions = torch.arange(10, dtype=torch.int32)
    actual = attention_module.PagedCSA2Attention.project_query(attention, value, positions, request_batch=True)
    expected = F.linear(value, attention.weights.wq_b.weight).reshape(10, 8, 512)
    assert torch.equal(actual, expected)


def test_serial_prefill_still_requires_prepared_fp8_weights(monkeypatch):
    monkeypatch.setattr(attention_module.gaudi_envs, 'VLLM_HPU_DSV41_PREFILL_Q_PROJECTION', True)
    with pytest.raises(ValueError, match='Prefill Q projection requires prepared FP8 weights'):
        attention_module.PagedCSA2Attention.project_query(
            _attention(), torch.zeros(10, 4), torch.arange(10, dtype=torch.int32))


def test_serial_c5_still_uses_checkpoint_projection(monkeypatch):
    monkeypatch.setattr(attention_module.gaudi_envs, 'VLLM_HPU_DSV41_PREFILL_Q_PROJECTION', True)
    attention = _attention()
    value = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 20
    actual = attention_module.PagedCSA2Attention.project_query(
        attention, value, torch.arange(5, dtype=torch.int32))
    expected = F.linear(value, attention.weights.wq_b.weight).reshape(5, 8, 512)
    assert torch.equal(actual, expected)


def test_packed_prepared_fp8_keeps_the_qualified_projection(monkeypatch):
    monkeypatch.setattr(attention_module.gaudi_envs, 'VLLM_HPU_DSV41_PREFILL_Q_PROJECTION', True)
    monkeypatch.setattr(attention_module, 'prefill_event_span', lambda *args: nullcontext())
    attention = _attention()
    attention.layer = 0
    attention._rotary_native_table = lambda: None
    attention.weights.wq_b.dense_fp8 = True
    attention.weights.wq_b.channel_scale = torch.ones(4096)
    marker = torch.ones((10, 8, 512))
    calls = []

    def prepared_projection(value, weight, channels, positions, rotary):
        calls.append(value.shape[0])
        return marker

    monkeypatch.setattr(attention_module, 'prefill_q_projection', prepared_projection)
    actual = attention_module.PagedCSA2Attention.project_query(
        attention, torch.zeros(10, 4), torch.arange(10, dtype=torch.int32), request_batch=True)
    assert actual is marker
    assert calls == [10]
