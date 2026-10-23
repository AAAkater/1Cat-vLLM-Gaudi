# SPDX-License-Identifier: Apache-2.0
"""Packed tentative KV preserves the serial SWA quantization boundary."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops import deepseek_v41_batch_draft as module
from vllm_gaudi.ops.deepseek_v41_math import pack_swa, unpack_swa


@pytest.mark.parametrize('active', [(True, True), (False, True)])
def test_tentative_rows_use_codec_without_mutating_persistent_state(monkeypatch, active):
    generator = torch.Generator().manual_seed(134)
    kv = torch.randn((10, 512), generator=generator, dtype=torch.bfloat16)
    quantized = unpack_swa(pack_swa(kv))
    assert not torch.equal(kv, quantized)
    ring = pack_swa(torch.randn((512, 512), generator=generator, dtype=torch.bfloat16))
    original = ring.clone()
    observed = {}

    def decode(packed, unused, selected, compressed):
        assert packed.shape[0] <= 512
        rows = selected.flatten().long()
        observed['rows'] = rows.clone()
        decoded = unpack_swa(packed).index_select(0, rows.clamp_min(0))
        decoded = torch.where((rows >= 0)[:, None], decoded, 0)
        return decoded, torch.zeros(rows.numel(), dtype=torch.int32)

    def sparse(query, cache, indices, sink, scale):
        observed['cache'] = cache.clone()
        return torch.zeros((10, 1, 512), dtype=torch.bfloat16), None, None

    monkeypatch.setattr(module, 'rms_norm', lambda value, weight, eps: value)
    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v41_selected_kv_bf16_gaudi2', decode, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, 'custom_deepseek_v4_sparse_attn_bf16_gaudi2', sparse, raising=False)
    attention = SimpleNamespace(
        draft_dense_fp8=False,
        _project_qkv_input=lambda value: (value, kv),
        weights=SimpleNamespace(q_norm=SimpleNamespace(weight=None), kv_norm=SimpleNamespace(weight=None),
                                attn_sink=None),
        eps=1e-6,
        project_query=lambda value, positions, **kwargs: value.reshape(10, 1, 512),
        _rope=lambda value, positions, **kwargs: value,
        swa_only_main=torch.zeros(1, 288, dtype=torch.uint8),
        scale=1.0,
        _finish_output=lambda value, positions, **kwargs: value,
    )
    draft = SimpleNamespace(layers=[SimpleNamespace(attention=attention)])
    state = SimpleNamespace(ring=lambda index: ring)
    positions = torch.tensor([[300, 301, 302, 303, 304], [700, 701, 702, 703, 704]], dtype=torch.int32)
    module.draft_attention_batch(
        draft, 0, kv, positions, torch.tensor([1, 0], dtype=torch.int32), torch.tensor(active), state)
    assert torch.equal(ring, original)
    assert observed['cache'].shape == (266, 512)
    for request in range(2):
        span = slice(request * 5, (request + 1) * 5)
        actual = observed['cache'][256 + request * 5:256 + (request + 1) * 5]
        expected = quantized[span] if active[request] else torch.zeros_like(quantized[span])
        assert torch.equal(actual, expected)
        expected_rows = torch.arange(request * 5, (request + 1) * 5)
        if not active[request]:
            expected_rows.fill_(-1)
        assert torch.equal(observed['rows'][request * 5:(request + 1) * 5], expected_rows)
