# SPDX-License-Identifier: Apache-2.0
"""Capacity contracts for packed DSpark target execution."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_gaudi.models.deepseek_v41_batch_target_program import CompiledTargetBatchStage, DSparkTargetStage
from vllm_gaudi.ops import deepseek_v41_batch_attention as batch_attention
from vllm_gaudi.ops import deepseek_v41_batch_target as target_module
from vllm_gaudi.ops import deepseek_v41_paged_attention as paged_module
from vllm_gaudi.ops.deepseek_v41_batch_target import (
    DSparkTargetBatch,
    DSparkTargetSnapshot,
    initialize_target_batch_state,
    target_request_offsets,
    target_rows_for_capacity,
)


def _metadata(capacity, lengths, active, starts, slots, generations=None):
    logical_query_loc = [0]
    for length in lengths:
        logical_query_loc.append(logical_query_loc[-1] + length)
    padding = torch.ones((capacity, 6), dtype=torch.bool)
    for lane, (is_active, length) in enumerate(zip(active, lengths, strict=True)):
        if is_active:
            padding[lane, :length] = False
    if generations is None:
        generations = tuple(lane + 3 if active[lane] else 0 for lane in range(capacity))
    return SimpleNamespace(
        request_ids=tuple(f"request-{lane}" for lane in range(capacity)),
        request_active=torch.tensor(active, dtype=torch.bool),
        target_start_positions=torch.tensor(starts, dtype=torch.int64),
        target_query_start_loc=torch.tensor(logical_query_loc, dtype=torch.int32),
        target_padding_mask=padding,
        request_slots=torch.tensor(slots, dtype=torch.int32),
        slot_generations=torch.tensor(generations, dtype=torch.int64),
        remaining_output_budget=torch.tensor([8 if is_active else 0 for is_active in active], dtype=torch.int32),
        num_draft_tokens=torch.tensor([max(length - 1, 0) for length in lengths], dtype=torch.int32),
        round_generation=torch.tensor(7, dtype=torch.int64),
    )


def _batch(capacity, lengths, active, starts, slots, *, explicit_capacity=True):
    queries = torch.zeros((capacity, 6), dtype=torch.int32)
    for lane, (is_active, length) in enumerate(zip(active, lengths, strict=True)):
        if is_active:
            queries[lane, :length] = torch.arange(10 + lane * 10, 10 + lane * 10 + length)
    pages = torch.arange(1, capacity * 4 + 1, dtype=torch.int32).reshape(capacity, 4)
    metadata = _metadata(capacity, lengths, active, starts, slots)
    return DSparkTargetBatch.from_metadata(
        metadata,
        queries,
        pages,
        batch_capacity=capacity if explicit_capacity else None,
    )


@pytest.mark.parametrize(
    ("capacity", "lengths", "starts"),
    [
        (1, (1, ), (31, )),
        (2, (1, 6), (63, 127)),
        (4, (1, 2, 5, 6), (255, 256, 511, 512)),
    ],
)
def test_all_active_capacity_uses_fixed_six_rows_per_lane(capacity, lengths, starts):
    slots = tuple(range(capacity))
    batch = _batch(capacity, lengths, (True, ) * capacity, starts, slots)
    rows = target_rows_for_capacity(capacity)

    assert batch.batch_capacity == capacity
    assert batch.target_rows == rows
    assert tuple(batch.request_offsets) == target_request_offsets(capacity)
    assert batch.input_ids.shape == (rows, )
    assert batch.positions.shape == (rows, )
    assert batch.row_slots.shape == (rows, )
    assert batch.row_pages.shape == (rows, 4)
    assert batch.row_valid.reshape(capacity, 6).sum(1).tolist() == list(lengths)
    assert batch.row_valid.sum().item() == sum(lengths)
    assert batch.host_request_slots == slots


@pytest.mark.parametrize("capacity", [1, 4])
def test_capacity_can_be_inferred_from_fixed_metadata(capacity):
    lengths = (1, ) * capacity
    batch = _batch(capacity, lengths, (True, ) * capacity, tuple(range(10, 10 + capacity)),
                   tuple(range(capacity)), explicit_capacity=False)
    assert batch.batch_capacity == capacity
    assert batch.host_request_slots == tuple(range(capacity))


def test_from_metadata_integrates_explicit_active_metadata_capacity_bucket():
    from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata

    metadata = DSparkBatchMetadata.build(
        requests=[None, "request-b", None, "request-d"],
        active=[False, True, False, True],
        target_query_lengths=[0, 3, 0, 6],
        draft_token_ids=[[], [11, 12], [], [21, 22, 23, 24, 25]],
        request_slots=[-1, 1, -1, 3],
        slot_generations=[0, 7, 0, 9],
        remaining_output_budget=[0, 4, 0, 6],
        target_padding_mask=[False] * 9,
        target_start_positions=[0, 50, 0, 100],
        round_generation=23,
    )
    source = torch.arange(9, dtype=torch.int32)
    packed = metadata.pack_target_queries(source)
    pages = torch.arange(1, 17, dtype=torch.int32).reshape(4, 4)

    batch = DSparkTargetBatch.from_metadata(metadata, packed, pages)

    assert batch.batch_capacity == 4
    assert batch.request_lengths == (0, 3, 0, 6)
    assert batch.request_ids == (None, "request-b", None, "request-d")
    assert batch.host_request_active == (False, True, False, True)
    assert batch.host_request_slots == (0, 1, 2, 3)
    assert batch.row_valid.reshape(4, 6).sum(1).tolist() == [0, 3, 0, 6]
    assert batch.positions.reshape(4, 6)[1, :3].tolist() == [50, 51, 52]
    assert batch.positions.reshape(4, 6)[3].tolist() == [100, 101, 102, 103, 104, 105]


@pytest.mark.parametrize(
    ("active", "lengths", "starts", "slots"),
    [
        ((True, False, False, False), (1, 0, 0, 0), (301, 0, 0, 0), (0, -1, -1, -1)),
        ((True, False, True, False), (6, 0, 2, 0), (127, 0, 511, 0), (0, -1, 2, -1)),
    ],
)
def test_inactive_lanes_are_six_no_write_rows_with_canonical_host_slots(active, lengths, starts, slots):
    batch = _batch(4, lengths, active, starts, slots)
    row_valid = batch.row_valid.reshape(4, 6)
    positions = batch.positions.reshape(4, 6)
    row_slots = batch.row_slots.reshape(4, 6)

    assert batch.host_request_slots == (0, 1, 2, 3)
    assert batch.request_slots.tolist() == list(slots)
    assert batch.host_request_active == active
    for lane, is_active in enumerate(active):
        if is_active:
            assert row_valid[lane].sum().item() == lengths[lane]
            assert torch.all(positions[lane, :lengths[lane]] >= starts[lane])
        else:
            assert not row_valid[lane].any()
            assert torch.equal(positions[lane], torch.full((6, ), -1, dtype=torch.int32))
            assert torch.equal(row_slots[lane], torch.full((6, ), -1, dtype=torch.int32))
            assert batch.slot_generations[lane].item() == 0


def test_legacy_b2_permutation_is_preserved_but_canonical_slots_are_lane_order():
    legacy = _batch(2, (1, 6), (True, True), (20, 40), (1, 0), explicit_capacity=False)
    assert legacy.host_request_slots == (1, 0)
    assert legacy.row_slots[:6].tolist() == [1, -1, -1, -1, -1, -1]
    assert legacy.row_slots[6:].tolist() == [0] * 6

    canonical = _batch(2, (1, 6), (True, True), (20, 40), (0, 1), explicit_capacity=True)
    assert canonical.host_request_slots == (0, 1)

    with pytest.raises(ValueError, match="canonical target lanes"):
        _batch(2, (1, 6), (True, True), (20, 40), (1, 0), explicit_capacity=True)


class _LaneState(nn.Module):
    def __init__(self, capacity):
        super().__init__()
        self.register_buffer("swa", torch.zeros((capacity * 256, 528), dtype=torch.uint8))
        self.register_buffer("kv_history", torch.zeros((capacity * 8, 512), dtype=torch.float32))
        self.register_buffer("score_history", torch.zeros((capacity * 8, 512), dtype=torch.float32))


class _SnapshotLayer(nn.Module):
    def __init__(self, index, capacity):
        super().__init__()
        self.layer = index
        self.attention = nn.Module()
        self.attention.ratio = 2
        self.attention.batch_state = _LaneState(capacity)


class _Cache(nn.Module):
    def __init__(self):
        super().__init__()
        self.ratio = 2
        self.register_buffer("main", torch.zeros((2048, 288), dtype=torch.uint8))
        self.register_buffer("index", torch.zeros((2048, 68), dtype=torch.uint8))


def _snapshot_program(capacity):
    program = nn.Module()
    program.layers = nn.ModuleList([_SnapshotLayer(1, capacity), _SnapshotLayer(14, capacity)])
    program.shared = nn.Module()
    program.shared.sources = nn.ModuleDict({"2": _Cache()})
    return program


def _fill_snapshot_program(program):
    with torch.no_grad():
        for layer in program.layers:
            state = layer.attention.batch_state
            state.swa.copy_(torch.arange(state.swa.numel()).reshape_as(state.swa).remainder(251).to(torch.uint8))
            state.kv_history.copy_(torch.arange(state.kv_history.numel()).reshape_as(state.kv_history).float())
            state.score_history.copy_(torch.arange(state.score_history.numel()).reshape_as(
                state.score_history).float())
        program.shared.sources["2"].main.fill_(17)
        program.shared.sources["2"].index.fill_(19)


def test_capacity_snapshot_rolls_back_only_active_rejected_suffix():
    capacity = 4
    program = _snapshot_program(capacity)
    _fill_snapshot_program(program)
    batch = _batch(capacity, (6, 0, 2, 0), (True, False, True, False),
                   (255, 0, 511, 0), (0, -1, 2, -1))
    snapshot = DSparkTargetSnapshot.capture(program, batch)
    state = program.layers[0].attention.batch_state
    saved = (state.swa.clone(), state.kv_history.clone(), state.score_history.clone())
    state.swa.fill_(99)
    state.kv_history.fill_(99)
    state.score_history.fill_(99)

    snapshot.restore(committed=(2, 0, 0, 0))

    # The first two rows of lane 0 are accepted; its rejected suffix and both
    # rows of lane 2 roll back. Inactive lanes never owned snapshot entries.
    assert torch.equal(state.swa[255], torch.full((528, ), 99, dtype=torch.uint8))
    assert torch.equal(state.swa[0], torch.full((528, ), 99, dtype=torch.uint8))
    for row in (1, 2, 3, 4):
        assert torch.equal(state.swa[row], saved[0][row])
    for row in (2 * 256 + 255, 2 * 256):
        assert torch.equal(state.swa[row], saved[0][row])
    for slot in (1, 3):
        assert torch.equal(state.swa[slot * 256:(slot + 1) * 256],
                           torch.full((256, 528), 99, dtype=torch.uint8))
    assert all(owner[0] != 1 and owner[0] != 3 for row in snapshot.rows for owners in row.owners for owner in owners)


def test_target_state_capacity_must_match_stage_capacity():
    existing = SimpleNamespace(capacity=2)
    program = SimpleNamespace(batch_state=existing)
    assert initialize_target_batch_state(program, 2) is existing
    with pytest.raises(ValueError, match="capacity differs"):
        initialize_target_batch_state(program, 4)


def test_target_stage_derives_state_capacity_from_fixed_slots_and_rejects_mismatch():
    stage, _calls = _fake_stage(4)
    stage.batch_state = SimpleNamespace(capacity=4)
    batch = _batch(4, (1, 0, 2, 0), (True, False, True, False),
                   (301, 0, 511, 0), (0, -1, 2, -1))

    compiled = DSparkTargetStage.from_batch(stage, batch, backend="eager")
    assert compiled.batch_capacity == 4
    assert compiled.rows == 24
    assert compiled.state.capacity == 4
    assert compiled.stage_interface().request_offsets == (0, 6, 12, 18, 24)

    stage.batch_state = SimpleNamespace(capacity=2)
    with pytest.raises(ValueError, match="capacity differs"):
        DSparkTargetStage.from_batch(stage, batch, backend="eager")


class _FakeTargetAttention(nn.Module):
    def __init__(self, capacity):
        super().__init__()
        self.ratio = 0
        self.batch_state = _LaneState(capacity)


class _FakeTargetLayer(nn.Module):
    def __init__(self, index, capacity, calls):
        super().__init__()
        self.layer = index
        self.attention = _FakeTargetAttention(capacity)
        self.calls = calls

    def forward_batch_target(self, residual, pre_mix, positions, image_mask, engram_rows, slots, pages, selected,
                             candidates, main_ready, index_ready, row_valid, request_lengths, request_slots):
        self.calls.append((self.layer, tuple(residual.shape), tuple(request_lengths), tuple(request_slots)))
        target = residual.mean(1) if self.layer else None
        return residual + 1, pre_mix, selected, candidates, main_ready, index_ready, target


def _fake_stage(capacity):
    stage = nn.Module()
    stage.dspark = True
    stage.pp_rank = 0
    stage.tp_rank = 0
    stage.is_last_stage = True
    stage.reduce = lambda value: value
    stage.config = {"text_config": {"rms_norm_eps": 1e-6}}
    stage.weights = nn.Module()
    stage.weights.embed = nn.Embedding(32, 8)
    stage.weights.norm = nn.Module()
    stage.weights.norm.register_buffer("weight", torch.ones(8))
    stage.shared = nn.Module()
    stage.shared.topk = nn.ModuleDict({"1": nn.Module()})
    stage.shared.sources = nn.ModuleDict({"1": nn.Module()})
    calls = []
    stage.layers = nn.ModuleList([_FakeTargetLayer(index, capacity, calls) for index in range(4)])
    return stage, calls


def _mapping_stage(capacity):
    stage, calls = _fake_stage(capacity)
    sources = tuple(range(0, 40, 4))
    # Keep the fake input projection BF16 so the two Dynamo modes differ by
    # dispatch/autograd state rather than by a test-only dtype promotion.
    stage.weights.embed = nn.Embedding(32, 8).bfloat16()
    stage.shared = nn.Module()
    stage.shared.topk = nn.ModuleDict({str(source): nn.Module() for source in sources})
    stage.shared.sources = nn.ModuleDict({str(source): nn.Module() for source in sources})
    stage.layers = nn.ModuleList([_FakeTargetLayer(index, capacity, calls) for index in range(40)])
    for layer in stage.layers:
        layer.attention.ratio = 1
    return stage, calls


@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_compiled_program_runs_one_six_n_projection_pass(capacity):
    lengths = tuple(1 + (lane % 6) for lane in range(capacity))
    batch = _batch(capacity, lengths, (True, ) * capacity, tuple(range(10, 10 + capacity)),
                   tuple(range(capacity)))
    stage, calls = _fake_stage(capacity)
    compiled = CompiledTargetBatchStage(stage, batch.request_lengths, batch.host_request_slots, backend="eager")
    rows = target_rows_for_capacity(capacity)
    hidden = torch.zeros((rows, 4, 8), dtype=torch.bfloat16)
    pre_mix = torch.zeros((rows, 4), dtype=torch.float32)

    output, aux = compiled(hidden, pre_mix, batch.input_ids, batch.positions, batch.row_slots, batch.row_pages,
                           batch.row_valid)

    assert output.shape == (rows, 8)
    assert aux.shape == (rows, 24)
    assert len(calls) == 4
    assert {call[0] for call in calls} == {0, 1, 2, 3}
    assert all(call[1] == (rows, 4, 8) for call in calls)
    assert all(call[2] == (6, ) * capacity and call[3] == tuple(range(capacity)) for call in calls)
    assert torch.count_nonzero(output[~batch.row_valid]).item() == 0
    assert torch.count_nonzero(aux[~batch.row_valid]).item() == 0


def test_compiled_program_temporarily_raises_recompile_limit_for_unique_group_mappings():
    capacity = 1
    batch = _batch(capacity, (1, ), (True, ), (31, ), (0, ))
    stage, calls = _mapping_stage(capacity)
    torch._dynamo.reset()
    try:
        compiled = CompiledTargetBatchStage(stage, batch.request_lengths, batch.host_request_slots, backend="eager")
        rows = target_rows_for_capacity(capacity)
        hidden = torch.zeros((rows, 4, 8), dtype=torch.bfloat16)
        pre_mix = torch.zeros((rows, 4), dtype=torch.float32)

        assert len(compiled.chunks) == 10
        original_limit = torch._dynamo.config.recompile_limit
        with torch._dynamo.config.patch(recompile_limit=2):
            args = (
                hidden,
                pre_mix,
                batch.input_ids,
                batch.positions,
                batch.row_slots,
                batch.row_pages,
                batch.row_valid,
            )
            with torch.inference_mode():
                inference_output, inference_aux = compiled(*args)
            output, aux = compiled(*args)
            assert torch._dynamo.config.recompile_limit == 2

        assert torch._C._dispatch_keys(inference_output) != torch._C._dispatch_keys(output)
        assert inference_output.shape == output.shape == (rows, 8)
        assert inference_aux.shape == aux.shape == (rows, 32)
        assert len(calls) == 80
        assert torch._dynamo.config.recompile_limit == original_limit
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize(
    ("active", "lengths", "starts", "slots"),
    [
        ((True, False, False, False), (1, 0, 0, 0), (301, 0, 0, 0), (0, -1, -1, -1)),
        ((True, False, True, False), (6, 0, 2, 0), (127, 0, 511, 0), (0, -1, 2, -1)),
    ],
)
def test_compiled_program_keeps_n4_scratch_while_masking_inactive_lanes(active, lengths, starts, slots):
    capacity = 4
    batch = _batch(capacity, lengths, active, starts, slots)
    stage, calls = _fake_stage(capacity)
    compiled = CompiledTargetBatchStage(stage, batch.request_lengths, batch.host_request_slots, backend="eager")
    rows = target_rows_for_capacity(capacity)
    hidden = torch.zeros((rows, 4, 8), dtype=torch.bfloat16)
    pre_mix = torch.zeros((rows, 4), dtype=torch.float32)

    output, aux = compiled(hidden, pre_mix, batch.input_ids, batch.positions, batch.row_slots, batch.row_pages,
                           batch.row_valid)

    assert output.shape == (rows, 8)
    assert aux.shape == (rows, 24)
    assert len(calls) == 4
    assert all(call[1][0] == rows and call[2] == (6, 6, 6, 6) for call in calls)
    assert torch.count_nonzero(output[~batch.row_valid]).item() == 0
    assert torch.count_nonzero(aux[~batch.row_valid]).item() == 0


class _CompressorHarness:
    ratio = 2
    eps = 1e-6

    def __init__(self, capacity):
        self.projection_calls = []
        self.batch_state = SimpleNamespace(
            kv_history=torch.zeros((capacity * 8, 512), dtype=torch.float32),
            score_history=torch.zeros((capacity * 8, 512), dtype=torch.float32),
        )
        self.weights = SimpleNamespace(
            compressor=SimpleNamespace(
                wkv=object(),
                norm=SimpleNamespace(weight=torch.ones(512)),
            ),
            indexer=SimpleNamespace(
                wk=object(),
                k_norm=SimpleNamespace(weight=torch.ones(512)),
            ),
        )
        self.cache = SimpleNamespace(
            main=torch.zeros((2048, 288), dtype=torch.uint8),
            index=torch.zeros((2048, 68), dtype=torch.uint8),
        )

    def _project_compressor_input(self, value):
        self.projection_calls.append(tuple(value.shape))
        return value, value + 1

    def linear(self, value, _weight):
        return value

    def _rope(self, value, _positions, *, request_batch):
        assert request_batch
        return value


def test_compressor_projects_once_and_runs_one_t6_state_step_per_fixed_slot(monkeypatch):
    capacity = 4
    rows = capacity * 6
    harness = _CompressorHarness(capacity)
    publish_calls = []
    mix_calls = []
    writes = []

    def fake_publish(history, score, slot, kv, score_rows, positions):
        assert history.shape == (capacity * 8, 512)
        assert score.shape == history.shape
        assert kv.shape == (6, 512)
        assert score_rows.shape == (6, 512)
        assert positions.shape == (6, )
        publish_calls.append((slot, positions.tolist()))
        return torch.zeros((8, 512)), torch.zeros((8, 512))

    def fake_native(history, score_history, kv, score, positions, **_kwargs):
        mix_calls.append(("native", kv.shape[0]))
        return torch.zeros((6, 512), dtype=torch.bfloat16)

    def fake_compiled(history, score_history, positions):
        mix_calls.append(("compiled", positions.shape[0]))
        return torch.zeros((6, 512), dtype=torch.bfloat16)

    def fake_write(cache, values, row_ids):
        writes.append((tuple(cache.shape), tuple(values.shape), row_ids.clone()))
        return torch.zeros(row_ids.shape, dtype=torch.int32)

    monkeypatch.setattr(target_module, "compressor_sequence_slot_publish_read", fake_publish)
    monkeypatch.setattr(target_module, "compressor_sequence_slot_native_mix", fake_native)
    monkeypatch.setattr(target_module, "compressor_sequence_slot_torch_mix", fake_compiled)
    monkeypatch.setattr(batch_attention, "write_state_rows", fake_write)
    monkeypatch.setattr(paged_module, "rms_norm", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(paged_module, "pack_fp4", lambda value, _group: value)

    positions = torch.full((rows, ), -1, dtype=torch.int32)
    positions[:6] = torch.arange(120, 126)
    positions[12:14] = torch.tensor([254, 255], dtype=torch.int32)
    slots = torch.full((rows, ), -1, dtype=torch.int32)
    slots[:6] = 0
    slots[12:14] = 2
    row_valid = slots >= 0
    pages = torch.arange(1, rows * 4 + 1, dtype=torch.int32).reshape(rows, 4)
    value = torch.randn((rows, 512), dtype=torch.bfloat16)

    paged_module.PagedCSA2Attention._compress_batch_target(
        harness,
        value,
        positions,
        slots,
        pages,
        row_valid,
        (6, ) * capacity,
        tuple(range(capacity)),
    )

    assert harness.projection_calls == [(rows, 512)]
    assert [slot for slot, _positions in publish_calls] == [0, 1, 2, 3]
    assert all(len(row_positions) == 6 for _slot, row_positions in publish_calls)
    assert publish_calls[1][1] == [-1] * 6
    assert publish_calls[3][1] == [-1] * 6
    assert mix_calls.count(("native", 6)) == capacity
    assert mix_calls.count(("compiled", 6)) == capacity
    assert [shape[0] for _cache, shape, _rows in writes] == [rows, rows]
    assert torch.equal(writes[0][2][6:12], torch.full((6, ), -1, dtype=torch.int32))
    assert torch.equal(writes[0][2][18:24], torch.full((6, ), -1, dtype=torch.int32))
    assert torch.equal(writes[1][2][6:12], torch.full((6, ), -1, dtype=torch.int32))
    assert torch.equal(writes[1][2][18:24], torch.full((6, ), -1, dtype=torch.int32))


class _ForwardHarness:
    ratio = 0
    fused_norm = False
    owns_index = False
    owns_kv = False
    batch_c1_numerics = False
    q_scale_rope = False
    native_rope = False
    scale = 0.125
    window = 128
    batch_packed_mla = False
    batch_packed_mla_sram = False
    batch_packed_mla_vector = False
    batch_reindex_group = False
    eps = 1e-6

    def __init__(self, capacity):
        self.batch_state = SimpleNamespace(swa=torch.zeros((capacity * 256, 528), dtype=torch.uint8))
        self.batch_main_unused = torch.zeros((1, 288), dtype=torch.uint8)
        self.weights = SimpleNamespace(
            attn_sink=torch.zeros(512),
            q_norm=SimpleNamespace(weight=torch.ones(512)),
            kv_norm=SimpleNamespace(weight=torch.ones(512)),
        )

    def _project_qkv_input(self, value):
        return value, value

    def project_query(self, value, _positions, *, request_batch):
        assert request_batch
        return value

    def _rope(self, value, _positions, *, request_batch):
        assert request_batch
        return value

    def _finish_output(self, output, _positions, _ready_outputs, *, request_batch):
        assert request_batch
        return output


def test_forward_target_n4_uses_safe_rope_positions_and_no_write_inactive_rows(monkeypatch):
    capacity = 4
    rows = capacity * 6
    harness = _ForwardHarness(capacity)
    writes = []
    calls = []

    def fake_write(cache, values, row_ids):
        writes.append(row_ids.clone())
        return torch.zeros(row_ids.shape, dtype=torch.int32)

    def fake_mla(query, swa, main, selected, pages, positions, slots, sink, scale, swa_done, main_done, **kwargs):
        calls.append({"query_shape": tuple(query.shape), "positions": positions.clone(), "slots": slots.clone()})
        return torch.ones((query.shape[0], 4), dtype=torch.bfloat16)

    monkeypatch.setattr(batch_attention, "write_state_rows", fake_write)
    monkeypatch.setattr(batch_attention, "batch_packed_mla", fake_mla)
    monkeypatch.setattr(paged_module, "rms_norm", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(paged_module, "pack_swa", lambda value: value)

    positions = torch.full((rows, ), -1, dtype=torch.int32)
    positions[:6] = torch.arange(100, 106)
    positions[12:14] = torch.tensor([200, 201], dtype=torch.int32)
    slots = torch.full((rows, ), -1, dtype=torch.int32)
    slots[:6] = 0
    slots[12:14] = 2
    row_valid = slots >= 0
    value = torch.randn((rows, 512), dtype=torch.bfloat16)
    pages = torch.arange(1, rows * 4 + 1, dtype=torch.int32).reshape(rows, 4)
    selected = torch.full((rows, 512), -1, dtype=torch.int32)
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32)
    done = torch.zeros((rows, ), dtype=torch.int32)

    result = paged_module.PagedCSA2Attention.forward_batch_target(
        harness,
        value,
        positions,
        slots,
        pages,
        selected,
        candidates,
        done,
        done,
        row_valid=row_valid,
        request_lengths=(6, ) * capacity,
        request_slots=(0, 1, 2, 3),
    )

    assert calls[0]["query_shape"] == (rows, 512)
    assert torch.equal(calls[0]["positions"], positions.clamp_min(0))
    assert torch.equal(writes[0][6:12], torch.full((6, ), -1, dtype=torch.int32))
    assert torch.equal(writes[0][18:24], torch.full((6, ), -1, dtype=torch.int32))
    assert torch.count_nonzero(result[0][~row_valid]).item() == 0
