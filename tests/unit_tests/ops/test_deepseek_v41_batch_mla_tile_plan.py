# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for optional packed-MLA tile compaction."""

import pytest
import torch

from vllm_gaudi.ops import deepseek_v41_batch_attention as batch_attention


ROWS_PER_LANE = 6
CAPACITY = 32
TOTAL_ROWS = CAPACITY * ROWS_PER_LANE
HEADS = 2
NATIVE_NAME = "custom_deepseek_v41_batch_paged_mla_mme_gaudi2"
TILE_BUCKETS = (1, 2, 4, 8, 16, 24)


def _active_rows(active_lanes):
    return torch.cat(
        [torch.arange(lane * ROWS_PER_LANE, (lane + 1) * ROWS_PER_LANE, dtype=torch.int64)
         for lane in active_lanes]
    )


def _make_inputs(ratio, active_lanes, *, capacity=CAPACITY):
    rows = capacity * ROWS_PER_LANE
    active_rows = _active_rows(active_lanes)
    active = torch.zeros(rows, dtype=torch.bool)
    active[active_rows] = True
    positions = torch.arange(rows, dtype=torch.int32) + 101
    positions[~active] = -1
    slots = torch.zeros(rows, dtype=torch.int32)
    slots[~active] = -1
    pages = torch.ones((rows, 8), dtype=torch.int32)
    selected = torch.arange(512, dtype=torch.int32).expand(rows, -1).clone()
    selected[:, 7::13] = -1
    query = torch.randn((rows, HEADS, 512), dtype=torch.bfloat16)
    return {
        "query": query,
        "swa": torch.zeros((256, 528), dtype=torch.uint8),
        "main": torch.zeros((1, 288), dtype=torch.uint8),
        "selected": selected,
        "pages": pages,
        "positions": positions,
        "slots": slots,
        "sink": torch.zeros(HEADS),
        "scale": torch.tensor([0.5]),
        "done": torch.zeros(rows, dtype=torch.int32),
        "active": active,
        "active_rows": active_rows,
        "ratio": ratio,
    }


def _recording_native(records):
    def native(query, swa, main, rows, indices, sink, scale, lengths):
        del swa, main, sink, scale
        records.append((query.clone(), rows.clone(), indices.clone(), lengths.clone()))
        return query.clone()

    return native


def _run(data, *, tile_indices=None):
    return batch_attention.batch_packed_mla(
        data["query"],
        data["swa"],
        data["main"],
        data["selected"],
        data["pages"],
        data["positions"],
        data["slots"],
        data["sink"],
        data["scale"],
        data["done"],
        data["done"],
        ratio=data["ratio"],
        tile_indices=tile_indices,
    )


@pytest.mark.parametrize("lanes, expected", [
    ((0, ), [0]),
    ((1, ), [0, 1]),
    ((2, ), [1, 2]),
    ((5, ), [3, 4]),
    ((31, ), [23]),
    ((0, 31), [0, 23]),
    ((0, 1, 31), [0, 1, 2, 23]),
    (tuple(range(9)), list(range(8))),
    (tuple(range(CAPACITY)), list(range(24))),
])
def test_tile_plan_uses_fixed_buckets_and_unique_original_tiles(lanes, expected):
    plan = batch_attention.batch_packed_mla_tile_plan(lanes, total_rows=TOTAL_ROWS)

    assert plan.dtype == torch.int32
    assert plan.shape == (len(expected), )
    assert plan.tolist() == expected
    assert plan.numel() in TILE_BUCKETS
    assert len(set(plan.tolist())) == plan.numel()


@pytest.mark.parametrize("ratio", (0, 1, 2))
def test_compact_tiles_preserve_full_path_active_rows_and_native_arguments(monkeypatch, ratio):
    records = []
    monkeypatch.setattr(torch.ops.custom_op, NATIVE_NAME, _recording_native(records), raising=False)
    data = _make_inputs(ratio, (0, 1, 31))

    full = _run(data)
    full_records = list(records)
    assert len(full_records) == 24

    records.clear()
    plan = batch_attention.batch_packed_mla_tile_plan((0, 1, 31), total_rows=TOTAL_ROWS)
    compact = _run(data, tile_indices=plan)
    compact_records = list(records)

    assert plan.tolist() == [0, 1, 2, 23]
    assert len(compact_records) == len(plan)
    for record, tile_id in zip(compact_records, plan.tolist(), strict=True):
        expected = full_records[tile_id]
        for actual, wanted in zip(record, expected, strict=True):
            assert torch.equal(actual, wanted)
        assert record[0].shape == (8, HEADS, 512)
        assert record[1].shape == (1, 8 * (128 if ratio == 0 else 640))
        assert record[2].shape == (8, 128 if ratio == 0 else 640)
        assert record[3].shape == (8, )

    active_rows = data["active_rows"]
    assert torch.equal(compact[active_rows], full[active_rows])
    assert not compact[~data["active"]].any()


def test_default_path_keeps_the_partial_tail(monkeypatch):
    records = []
    monkeypatch.setattr(torch.ops.custom_op, NATIVE_NAME, _recording_native(records), raising=False)
    data = _make_inputs(0, (0, ), capacity=11)

    output = _run(data)

    assert [record[0].shape[0] for record in records] == [8] * 8 + [2]
    assert torch.equal(output[data["active_rows"]], data["query"][data["active_rows"]])


def test_compact_path_rejects_duplicate_or_out_of_range_tile_ids(monkeypatch):
    monkeypatch.setattr(torch.ops.custom_op, NATIVE_NAME, _recording_native([]), raising=False)
    data = _make_inputs(0, (0, 31))

    for bad in (
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([23, 0], dtype=torch.int32),
        torch.tensor([0, 24], dtype=torch.int32),
    ):
        with pytest.raises(ValueError, match="tile_indices"):
            _run(data, tile_indices=bad)


def test_compact_path_rejects_invalid_tensor_contracts(monkeypatch):
    monkeypatch.setattr(torch.ops.custom_op, NATIVE_NAME, _recording_native([]), raising=False)
    data = _make_inputs(0, (0, ))

    for bad in (
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([0, 1, 2], dtype=torch.int32),
    ):
        with pytest.raises(ValueError, match="tile_indices"):
            _run(data, tile_indices=bad)


@pytest.mark.parametrize("lanes", [(), (0, 0), (1, 0), (-1, ), (CAPACITY, )])
def test_tile_plan_rejects_invalid_active_lanes(lanes):
    with pytest.raises((TypeError, ValueError), match="active lane|active_lanes|at least one"):
        batch_attention.batch_packed_mla_tile_plan(lanes, total_rows=TOTAL_ROWS)


def test_tile_plan_rejects_partial_or_unsupported_total_rows():
    with pytest.raises(ValueError, match="complete eight-row|supported"):
        batch_attention.batch_packed_mla_tile_plan((0, ), total_rows=11 * ROWS_PER_LANE)
    with pytest.raises(ValueError, match="tile size"):
        batch_attention.batch_packed_mla_tile_plan((0, ), total_rows=TOTAL_ROWS, tile=4)


def test_compact_path_does_not_call_collectives(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("tile compaction must not call collectives")

    monkeypatch.setattr(torch.distributed, "all_gather", fail)
    monkeypatch.setattr(torch.distributed, "all_reduce", fail)
    monkeypatch.setattr(torch.ops.custom_op, NATIVE_NAME, _recording_native([]), raising=False)
    data = _make_inputs(0, (0, 31))
    plan = batch_attention.batch_packed_mla_tile_plan((0, 31), total_rows=TOTAL_ROWS)

    _run(data, tile_indices=plan)


def test_tile_plan_compiles_by_bucket_length_not_lane_values(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.ops.custom_op,
        NATIVE_NAME,
        lambda query, swa, main, rows, indices, sink, scale, lengths: query.clone(),
        raising=False,
    )
    data = _make_inputs(0, (0, ))
    graphs = []

    def backend(graph, example_inputs):
        del example_inputs
        graphs.append(graph)
        return graph.forward

    def run(query, tile_indices):
        return batch_attention.batch_packed_mla(
            query,
            data["swa"],
            data["main"],
            data["selected"],
            data["pages"],
            data["positions"],
            data["slots"],
            data["sink"],
            data["scale"],
            data["done"],
            data["done"],
            ratio=0,
            tile_indices=tile_indices,
        )

    torch._dynamo.reset()
    try:
        compiled = torch.compile(run, backend=backend, fullgraph=True, dynamic=False)
        lane0 = batch_attention.batch_packed_mla_tile_plan((0, ), total_rows=TOTAL_ROWS)
        lane31 = batch_attention.batch_packed_mla_tile_plan((31, ), total_rows=TOTAL_ROWS)
        compiled(data["query"], lane0)
        compiled(data["query"], lane31)
        assert len(graphs) == 1

        lane1 = batch_attention.batch_packed_mla_tile_plan((1, ), total_rows=TOTAL_ROWS)
        compiled(data["query"], lane1)
        assert len(graphs) == 2
        graph_text = str(graphs[0].graph)
        assert "tolist" not in graph_text
        assert "active_lanes" not in graph_text
    finally:
        torch._dynamo.reset()
