# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the DSpark target-attention qualification tool."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_tool():
    path = Path(__file__).resolve().parents[3] / "tools" / "check_deepseek_v41_dspark_target_attention.py"
    spec = importlib.util.spec_from_file_location("check_deepseek_v41_dspark_target_attention", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_target_attention_tool_metadata_contract_supports_mixed_counts():
    tool = _load_tool()
    assert tool.COMPONENT_SEARCH_LENGTH // 2 >= 512
    for lengths in ((1, 6), (2, 5), (6, 6)):
        metadata, queries, pages = tool.make_metadata(2, lengths, (252, 508))
        assert tuple(metadata.target_query_start_loc.tolist()) == (0, lengths[0], sum(lengths))
        assert queries.shape == (2, 6)
        assert pages.shape == (2, 8)
        assert not metadata.target_padding_mask.any().item()
        assert not queries[0, lengths[0]:].any().item()
        assert not queries[1, lengths[1]:].any().item()


def test_target_attention_tool_comparison_reports_raw_difference():
    tool = _load_tool()
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    actual = torch.tensor([[1.0, 2.5], [3.0, 4.0]])
    report = tool.compare_tensors("output", expected, actual)
    assert report["exact"] is False
    assert report["mismatch"] == 1
    assert report["max_abs"] == 0.5
    exact = tool.compare_tensors("output", expected, expected.clone())
    assert exact["exact"] is True
    assert exact["mismatch"] == 0
    assert exact["max_abs"] == 0.0


@pytest.mark.parametrize(("capacity", "rows"), ((2, 12), (4, 24)))
def test_target_attention_tool_inputs_follow_fixed_capacity(capacity, rows):
    tool = _load_tool()
    value, selected, candidates, done = tool._make_inputs(
        {"hidden_size": 5120}, capacity=capacity, seed=7, device="cpu")
    assert value.shape == (rows, 5120)
    assert value.dtype == torch.bfloat16
    assert selected.shape == (rows, 512)
    assert candidates.shape == (rows, 2048)
    assert done.shape == (rows,)
    assert selected.dtype == candidates.dtype == done.dtype == torch.int32
    assert torch.equal(selected, torch.full_like(selected, -1))
    assert torch.equal(candidates, torch.full_like(candidates, -1))
    assert torch.equal(done, torch.zeros_like(done))


def test_target_attention_tool_mixed_rows_place_padding_as_no_write_sentinels():
    tool = _load_tool()
    batch = tool.make_target_batch(2, (1, 6), (252, 508), device="cpu")
    assert batch.request_lengths == (1, 6)
    assert batch.host_row_slots[:7] == (0, -1, -1, -1, -1, -1, 1)
    assert batch.host_row_valid == (True, False, False, False, False, False, True, True, True, True, True, True)
    assert batch.positions.tolist() == [252, -1, -1, -1, -1, -1, 508, 509, 510, 511, 512, 513]
    assert torch.equal(batch.row_pages[0], batch.pages[0])
    assert torch.equal(batch.row_pages[-1], batch.pages[1])


def test_target_attention_tool_partial_restore_keeps_only_accepted_prefix():
    tool = _load_tool()
    live = torch.tensor([10, 20, 30, 40, 50, 60])
    buffers = [("state", live)]
    baseline = {"state": live.clone()}
    target = {"state": torch.tensor([1, 2, 3, 4, 5, 6])}
    assert baseline["state"].data_ptr() != live.data_ptr()
    rows = (
        SimpleNamespace(destination=live, indices=torch.tensor([0]),
                        owners=(((0, 0), (1, 5)),)),
        SimpleNamespace(destination=live, indices=torch.tensor([1]),
                        owners=(((0, 4), (1, 0)),)),
        SimpleNamespace(destination=live, indices=torch.tensor([2]),
                        owners=(((0, 2), (1, 1)),)),
        SimpleNamespace(destination=live, indices=torch.tensor([3]),
                        owners=(((0, 1),),)),
    )
    snapshot = SimpleNamespace(rows=rows)
    expected = tool._restore_expected(snapshot, buffers, baseline, target, (2, 1))
    assert expected["state"].tolist() == [1, 2, 30, 4, 5, 6]


def test_target_attention_tool_partial_restore_rejects_unknown_destination():
    tool = _load_tool()
    live = torch.tensor([10, 20, 30])
    buffers = [("state", live)]
    baseline = {"state": live.clone()}
    target = {"state": torch.tensor([1, 2, 3])}
    snapshot = SimpleNamespace(rows=(SimpleNamespace(
        destination=live.clone(), indices=torch.tensor([0]), owners=(((0, 0),),)),))
    with pytest.raises(ValueError, match="not one of the tracked buffers"):
        tool._restore_expected(snapshot, buffers, baseline, target, (1, 1))


def test_target_attention_tool_state_compare_excludes_only_unowned_cache_scratch():
    tool = _load_tool()
    saved_main = torch.zeros((128, 288), dtype=torch.uint8)
    saved_index = torch.zeros((128, 68), dtype=torch.uint8)
    target_main, target_index = saved_main.clone(), saved_index.clone()
    target_main[:64, 0] = 19
    target_index[:64, 0] = 23
    buffers = [("main", target_main), ("index", target_index)]
    saved = {"main": saved_main, "index": saved_index}
    checks = tool._compare_state(buffers, saved, 2)
    assert all(check["exact"] for check in checks)
    target_main[64, 0] = 1
    checks = tool._compare_state(buffers, saved, 2)
    assert next(check for check in checks if check["name"] == "main")["exact"] is False


def test_target_attention_tool_cli_and_capacity_scenarios_are_fixed(tmp_path):
    tool = _load_tool()
    default = tool.parse_args([str(tmp_path / "prepared"), "--output", str(tmp_path / "default.json")])
    explicit = tool.parse_args([
        str(tmp_path / "prepared"), "--output", str(tmp_path / "n4.json"), "--batch-capacity", "4",
    ])
    assert default.batch_capacity == 2
    assert explicit.batch_capacity == 4
    with pytest.raises(SystemExit):
        tool.parse_args([
            str(tmp_path / "prepared"), "--output", str(tmp_path / "bad.json"), "--batch-capacity", "1",
        ])

    n2 = tool.scenarios_for_capacity(2)
    assert [item["name"] for item in n2] == ["c6_c6_ringwrap", "mixed_c1_c6", "mixed_c2_c5"]
    n4 = tool.scenarios_for_capacity(4)
    assert [sum(item["active"]) for item in n4] == [4, 2, 1]
    assert [item["lengths"] for item in n4] == [(6, 6, 1, 2), (6, 0, 1, 0), (0, 6, 0, 0)]
    assert [item["slots"] for item in n4] == [
        (0, 1, 2, 3),
        (0, -1, 2, -1),
        (-1, 1, -1, -1),
    ]


def test_target_attention_tool_n4_inactive_lanes_are_fixed_no_write_rows():
    tool = _load_tool()
    batch = tool.make_target_batch(
        4,
        (6, 0, 1, 0),
        (252, 0, 764, 0),
        active=(True, False, True, False),
        slots=(0, -1, 2, -1),
        device="cpu",
    )
    assert batch.batch_capacity == 4
    assert batch.request_lengths == (6, 0, 1, 0)
    assert batch.host_request_slots == (0, 1, 2, 3)
    assert batch.row_valid.reshape(4, 6).sum(1).tolist() == [6, 0, 1, 0]
    assert torch.equal(batch.positions.reshape(4, 6)[1], torch.full((6, ), -1, dtype=torch.int32))
    assert torch.equal(batch.row_slots.reshape(4, 6)[3], torch.full((6, ), -1, dtype=torch.int32))
    assert batch.pages.shape == (4, 8)


def test_target_attention_tool_packed_graph_gate_rejects_recompile_or_failure():
    tool = _load_tool()
    good = {
        "entry_calls": 5,
        "failed_entry_calls": 0,
        "backend_compiles": 1,
        "entry_replays": 4,
    }
    assert tool._packed_graph_gate(good, expected_calls=5, expected_compiles=1)["exact"]
    recompiled = dict(good, backend_compiles=2, entry_replays=3)
    assert not tool._packed_graph_gate(recompiled, expected_calls=5, expected_compiles=1)["exact"]
    failed = dict(good, failed_entry_calls=1, entry_calls=4, entry_replays=3)
    assert not tool._packed_graph_gate(failed, expected_calls=5, expected_compiles=1)["exact"]


class _FakeState(torch.nn.Module):

    def __init__(self, capacity):
        super().__init__()
        self.register_buffer("swa", torch.zeros((capacity * 256, 528), dtype=torch.uint8))
        self.register_buffer("kv_history", torch.zeros((capacity * 8, 512), dtype=torch.float32))
        self.register_buffer("score_history", torch.zeros((capacity * 8, 512), dtype=torch.float32))


class _FakeCache:

    def __init__(self, capacity):
        rows = (capacity * 8 + 1) * 64
        self.main = torch.zeros((rows, 288), dtype=torch.uint8)
        self.index = torch.zeros((rows, 68), dtype=torch.uint8)


class _FakeAttention(torch.nn.Module):
    ratio = 2
    layer = 0

    def __init__(self, capacity, packed_slots):
        super().__init__()
        self.batch_state = _FakeState(capacity)
        self.packed_slots = packed_slots

    def forward_batch_target(self, value, positions, slots, pages, selected, candidates, main_done, index_done,
                             *, row_valid, request_lengths, request_slots):
        del positions, pages, main_done, index_done, row_valid, request_lengths
        assert request_slots == tuple(range(len(request_slots)))
        self.packed_slots.append(slots.reshape(-1, 6)[:, 0].tolist())
        return (torch.zeros_like(value), torch.full_like(selected, -1), candidates,
                torch.zeros(value.shape[0], dtype=torch.int32),
                torch.zeros(value.shape[0], dtype=torch.int32))


@pytest.mark.parametrize(("recompile_every_call", "expected_exact"), ((False, True), (True, False)))
def test_target_attention_tool_n4_graph_gate_controls_qualification(monkeypatch, recompile_every_call, expected_exact):
    tool = _load_tool()
    from vllm_gaudi.ops import deepseek_v41_batch_target as target_module

    capacity = 4
    packed_slots = []
    cache = _FakeCache(capacity)
    shared = SimpleNamespace(sources={"0": cache})

    class FakeTicket:
        snapshot = SimpleNamespace(rows=(), bytes=0)

        def restore(self, committed=None):
            assert len(committed) == capacity

    monkeypatch.setattr(tool, "_build_attention", lambda *args, **kwargs: (
        _FakeAttention(capacity, packed_slots), 2))
    monkeypatch.setattr(tool, "_make_reference", lambda *args, **kwargs: [])
    monkeypatch.setattr(tool, "_run_serial_reference", lambda *args, **kwargs: (
        torch.zeros((24, 5120), dtype=torch.bfloat16),
        torch.full((24, 512), -1, dtype=torch.int32),
    ))
    monkeypatch.setattr(target_module, "make_target_ticket", lambda program, batch: FakeTicket())
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)

    backend_calls = []

    def mock_backend(graph, inputs):
        del inputs
        backend_calls.append(graph)
        return graph.forward

    def mock_compile(module, backend):
        compiled = False

        def call(*args, **kwargs):
            nonlocal compiled
            if recompile_every_call or not compiled:
                backend(module, args)
                compiled = True
            return module(*args, **kwargs)

        return call

    monkeypatch.setattr(tool, "_shared_counting_backend", lambda name: tool._CountingBackend(mock_backend))
    monkeypatch.setattr(tool, "_compile", mock_compile)
    result = tool._qualify_layer(
        SimpleNamespace(tp_size=8),
        layer=0,
        weights=object(),
        config={"hidden_size": 5120},
        shared=shared,
        linear=object(),
        reduce=object(),
        gather=object(),
        capacity=capacity,
        woa_layers=(),
        woa_output_roundtrip=False,
    )

    assert packed_slots == [[0, 1, 2, 3], [0, -1, 2, -1], [0, -1, 2, -1], [-1, 1, -1, -1],
                            [-1, 1, -1, -1]]
    assert result["graph_runtime"]["packed"]["entry_calls"] == 5
    assert result["graph_runtime"]["packed"]["backend_compiles"] == (5 if recompile_every_call else 1)
    assert result["graph_runtime"]["packed"]["entry_replays"] == (0 if recompile_every_call else 4)
    assert result["exact"] is expected_exact
    if recompile_every_call:
        assert result["qualification"]["packed_graph_exact"] is False
