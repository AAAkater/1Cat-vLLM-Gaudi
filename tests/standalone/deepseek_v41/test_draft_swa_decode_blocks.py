# SPDX-License-Identifier: Apache-2.0
"""SWA cache/selection blocking preserves codecs, order, and invalid rows."""
import math

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_batch_draft import decode_draft_swa_rows
from vllm_gaudi.ops.deepseek_v41_math import pack_swa, unpack_swa


@pytest.mark.parametrize('source_rows', [256, 512, 513, 522, 1024, 2048])
@pytest.mark.parametrize('selected_rows', [17, 1024, 1031])
def test_decoder_blocks_preserve_global_row_order_and_native_limits(monkeypatch, source_rows, selected_rows):
    generator = torch.Generator().manual_seed(source_rows + selected_rows)
    packed = pack_swa(torch.randn((source_rows, 512), generator=generator, dtype=torch.bfloat16))
    before = packed.clone()
    ids = torch.arange(selected_rows, dtype=torch.int32).remainder(source_rows).unsqueeze(0)
    ids[0, :9] = torch.tensor(
        [-1, -(2**31), 2**31 - 1, source_rows, source_rows - 1, 0, 511, 512, source_rows - 1],
        dtype=torch.int32)
    calls = []

    def native_decode(source, main, local, vector):
        assert source.shape[0] <= 512
        assert 0 < local.shape[1] <= 1024
        assert source.is_contiguous() and local.is_contiguous()
        calls.append((source.shape[0], local.shape[1]))
        valid = (local >= 0) & (local < source.shape[0])
        values = unpack_swa(source).index_select(0, local.flatten().clamp(0, source.shape[0] - 1).long())
        values = torch.where(valid.flatten()[:, None], values, 0)
        return values, torch.zeros_like(local)

    monkeypatch.setattr(
        torch.ops.custom_op, 'custom_deepseek_v41_selected_kv_bf16_gaudi2', native_decode, raising=False)
    output = decode_draft_swa_rows(packed, torch.zeros(1, 288, dtype=torch.uint8), ids)
    valid = (ids >= 0) & (ids < source_rows)
    expected = unpack_swa(packed).index_select(0, ids.flatten().clamp(0, source_rows - 1).long())
    expected = torch.where(valid.flatten()[:, None], expected, 0)
    assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(packed, before)
    assert len(calls) == math.ceil(source_rows / 512) * math.ceil(selected_rows / 1024)


def test_decoder_rejects_wrong_index_dtype_before_native_call():
    with pytest.raises(ValueError, match='int32'):
        decode_draft_swa_rows(
            torch.zeros(256, 528, dtype=torch.uint8), torch.zeros(1, 288, dtype=torch.uint8), torch.zeros(1, 1))
