# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch
import pytest

from vllm_gaudi.ops import deepseek_v41_ordered_peer_sum as ordered_peer_sum
from vllm_gaudi.ops.deepseek_v41_ordered_peer_sum import ordered_peer_sum_reference


class _HpuShards:
    """CPU values with the device tag used by the compiled reduction."""

    device = SimpleNamespace(type="hpu")

    def __init__(self, values):
        self._values = values

    @property
    def shape(self):
        return self._values.shape

    def __getitem__(self, index):
        return self._values[index]


def _expected_rank_sum(values):
    total = values[0].float()
    for row in values[1:]:
        total = total + row.float()
    return total.bfloat16()


def _assert_rank_sum(actual, values):
    expected = _expected_rank_sum(values)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_compiled_rank_sum_falls_back_if_compile_construction_fails(monkeypatch):
    state = {"compiled": None, "disabled_reason": None}
    monkeypatch.setattr(ordered_peer_sum, "_rank_sum_graph", state)
    compile_calls = []

    def unavailable(*args, **kwargs):
        compile_calls.append((args, kwargs))
        raise RuntimeError("hpu_backend unavailable")

    monkeypatch.setattr(ordered_peer_sum.torch, "compile", unavailable)
    values = torch.tensor([[256., 1.], [1., -256.], [-256., 0.125], [0.5, 512.]], dtype=torch.bfloat16)
    shards = _HpuShards(values)

    _assert_rank_sum(ordered_peer_sum._compiled_rank_sum(shards), values)
    assert len(compile_calls) == 1
    assert compile_calls[0][0] == (ordered_peer_sum.ordered_rank_sum, )
    assert compile_calls[0][1] == {"backend": "hpu_backend", "fullgraph": True, "dynamic": False}
    assert state["compiled"] is None
    assert state["disabled_reason"] == "RuntimeError: hpu_backend unavailable"

    # The disabled graph is never retried and continues with the eager reduction.
    _assert_rank_sum(ordered_peer_sum._compiled_rank_sum(shards), values)
    assert len(compile_calls) == 1


def test_compiled_rank_sum_falls_back_if_first_compiled_call_fails(monkeypatch):
    state = {"compiled": None, "disabled_reason": None}
    monkeypatch.setattr(ordered_peer_sum, "_rank_sum_graph", state)
    compile_calls = []
    compiled_calls = []

    def failing_compile(function, **kwargs):
        compile_calls.append((function, kwargs))

        def fail(shards):
            compiled_calls.append(shards.shape)
            raise RuntimeError("first graph execution failed")

        return fail

    monkeypatch.setattr(ordered_peer_sum.torch, "compile", failing_compile)
    values = torch.tensor([[256., 1.], [1., -256.], [-256., 0.125], [0.5, 512.]], dtype=torch.bfloat16)
    shards = _HpuShards(values)

    _assert_rank_sum(ordered_peer_sum._compiled_rank_sum(shards), values)
    assert len(compile_calls) == 1
    assert compiled_calls == [values.shape]
    assert callable(state["compiled"])
    assert state["disabled_reason"] == "RuntimeError: first graph execution failed"

    _assert_rank_sum(ordered_peer_sum._compiled_rank_sum(shards), values)
    assert len(compile_calls) == 1
    assert compiled_calls == [values.shape]


def test_compiled_rank_sum_caches_successful_graph(monkeypatch):
    state = {"compiled": None, "disabled_reason": None}
    monkeypatch.setattr(ordered_peer_sum, "_rank_sum_graph", state)
    compile_calls = []
    compiled_calls = []

    def recording_compile(function, **kwargs):
        compile_calls.append((function, kwargs))

        def run(shards):
            compiled_calls.append(shards.shape)
            return function(shards)

        return run

    monkeypatch.setattr(ordered_peer_sum.torch, "compile", recording_compile)
    values = torch.tensor([[256., 1.], [1., -256.], [-256., 0.125], [0.5, 512.]], dtype=torch.bfloat16)
    shards = _HpuShards(values)

    _assert_rank_sum(ordered_peer_sum._compiled_rank_sum(shards), values)
    _assert_rank_sum(ordered_peer_sum._compiled_rank_sum(shards), values)
    assert len(compile_calls) == 1
    assert compile_calls[0][0] is ordered_peer_sum.ordered_rank_sum
    assert compile_calls[0][1] == {"backend": "hpu_backend", "fullgraph": True, "dynamic": False}
    assert compiled_calls == [values.shape, values.shape]
    assert state["disabled_reason"] is None
    assert callable(state["compiled"])


@pytest.mark.parametrize('ranks', [2, 4, 8])
def test_reference_keeps_rank_order_and_final_bf16_boundary(ranks):
    values = torch.randn((ranks, 5120), generator=torch.Generator().manual_seed(42)).bfloat16()
    # Exact cancellations expose reordered addition and intermediate BF16 rounding.
    values[0, :4] = torch.tensor([256., 1., -0., 512.])
    values[1, :4] = torch.tensor([1., -256., 0., -512.])
    if ranks > 2:
        values[2, :4] = torch.tensor([-256., 256., -0., 0.125])
    expected = values[0].float()
    for row in values[1:]:
        expected = expected + row.float()
    actual = ordered_peer_sum_reference(values)
    assert torch.equal(actual.view(torch.int16), expected.bfloat16().reshape(1, -1).view(torch.int16))
