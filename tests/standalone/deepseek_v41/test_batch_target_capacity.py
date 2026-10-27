# SPDX-License-Identifier: Apache-2.0
"""Capacity contracts for packed DSpark target execution."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_gaudi.models import deepseek_v41_batch_target_program as target_program
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


@pytest.fixture(autouse=True)
def _cpu_dynamo_only(monkeypatch):
    # Dynamo stream discovery must not acquire an HPU for these CPU contracts.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)


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
        (3, (1, 2, 6), (63, 127, 255)),
        (4, (1, 2, 5, 6), (255, 256, 511, 512)),
        (6, (1, 2, 3, 4, 5, 6), tuple(32 * lane for lane in range(6))),
        (8, (1, 2, 3, 4, 5, 6, 1, 2), tuple(32 * lane for lane in range(8))),
        (11, (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5), tuple(32 * lane for lane in range(11))),
        (16, tuple(1 + lane % 6 for lane in range(16)), tuple(32 * lane for lane in range(16))),
        (64, tuple(1 + lane % 6 for lane in range(64)), tuple(32 * lane for lane in range(64))),
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


@pytest.mark.parametrize("capacity", [1, 4, 11, 16, 64])
def test_capacity_can_be_inferred_from_fixed_metadata(capacity):
    lengths = (1, ) * capacity
    batch = _batch(capacity, lengths, (True, ) * capacity, tuple(range(10, 10 + capacity)),
                   tuple(range(capacity)), explicit_capacity=False)
    assert batch.batch_capacity == capacity
    assert batch.host_request_slots == tuple(range(capacity))


def test_target_row_contract_stops_at_shared_capacity_bound():
    assert target_rows_for_capacity(64) == 6 * 64
    with pytest.raises(ValueError, match="1..64"):
        target_rows_for_capacity(65)


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
    assert batch.host_active_lanes == (1, 3)
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
    assert batch.host_active_lanes == tuple(lane for lane, enabled in enumerate(active) if enabled)
    for lane, is_active in enumerate(active):
        if is_active:
            assert row_valid[lane].sum().item() == lengths[lane]
            assert torch.all(positions[lane, :lengths[lane]] >= starts[lane])
        else:
            assert not row_valid[lane].any()
            assert torch.equal(positions[lane], torch.full((6, ), -1, dtype=torch.int32))
            assert torch.equal(row_slots[lane], torch.full((6, ), -1, dtype=torch.int32))
            assert batch.slot_generations[lane].item() == 0


def test_host_active_lanes_preserves_sparse_lane31():
    capacity = 32
    active = tuple(lane == 31 for lane in range(capacity))
    lengths = tuple(6 if lane == 31 else 0 for lane in range(capacity))
    starts = tuple(127 if lane == 31 else 0 for lane in range(capacity))
    slots = tuple(31 if lane == 31 else -1 for lane in range(capacity))

    batch = _batch(capacity, lengths, active, starts, slots)

    assert batch.host_request_active == active
    assert batch.host_active_lanes == (31, )


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


@pytest.mark.parametrize("capacity", (3, 6, 8, 11, 16, 64))
def test_target_stage_derives_state_capacity_from_fixed_slots_and_rejects_mismatch(capacity):
    stage, _calls = _fake_stage(capacity)
    stage.batch_state = SimpleNamespace(capacity=capacity)
    active = tuple(index % 2 == 0 for index in range(capacity))
    lengths = tuple(1 if enabled else 0 for enabled in active)
    starts = tuple(301 + index if enabled else 0 for index, enabled in enumerate(active))
    slots = tuple(index if enabled else -1 for index, enabled in enumerate(active))
    batch = _batch(capacity, lengths, active, starts, slots)

    compiled = DSparkTargetStage.from_batch(stage, batch, backend="eager")
    assert compiled.batch_capacity == capacity
    assert compiled.rows == 6 * capacity
    assert compiled.state.capacity == capacity
    assert compiled.stage_interface().request_offsets == tuple(range(0, 6 * capacity + 1, 6))

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


def test_compiled_program_keeps_chunk_based_config_for_non_slice(monkeypatch):
    capacity = 1
    batch = _batch(capacity, (1, ), (True, ), (31, ), (0, ))
    stage, _calls = _fake_stage(capacity)
    observed_configs = []

    class _CompiledGroup:
        def __init__(self, group):
            self.group = group

        def __call__(self, *args, **kwargs):
            observed_configs.append((
                torch._dynamo.config.recompile_limit,
                torch._dynamo.config.allow_unspec_int_on_nn_module,
            ))
            return self.group(*args, **kwargs)

    monkeypatch.setattr(target_program.torch, "compile", lambda group, **_kwargs: _CompiledGroup(group))
    compiled = CompiledTargetBatchStage(stage, batch.request_lengths, batch.host_request_slots, backend="eager")
    rows = target_rows_for_capacity(capacity)
    hidden = torch.zeros((rows, 4, 8), dtype=torch.bfloat16)
    pre_mix = torch.zeros((rows, 4), dtype=torch.float32)

    assert not compiled.slice_target_layers
    original_limit = torch._dynamo.config.recompile_limit
    original_allow_unspec_int = torch._dynamo.config.allow_unspec_int_on_nn_module
    with torch._dynamo.config.patch(
            recompile_limit=2,
            allow_unspec_int_on_nn_module=False,
    ):
        output, aux = compiled(hidden, pre_mix, batch.input_ids, batch.positions, batch.row_slots, batch.row_pages,
                               batch.row_valid)
        assert torch._dynamo.config.recompile_limit == 2
        assert torch._dynamo.config.allow_unspec_int_on_nn_module is False

    assert torch._dynamo.config.recompile_limit == original_limit
    assert torch._dynamo.config.allow_unspec_int_on_nn_module is original_allow_unspec_int
    assert observed_configs == [(max(2, 2 * len(compiled.chunks) + 1), False)]
    assert output.shape == (rows, 8)
    assert aux.shape == (rows, 24)


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

    def _rope(self, value, _positions, *, request_batch, inverse=False):
        del inverse
        assert request_batch
        return value

    def _finish_output(self, output, _positions, _ready_outputs, *, request_batch, decode):
        assert request_batch
        assert decode
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


def test_n11_query_norm_projection_tiles_native_rows(monkeypatch):
    calls = []

    def projection(value, norm, weight, scale, positions, table, eps):
        del norm, weight, scale, table, eps
        assert value.shape[0] == positions.numel() <= 64
        calls.append(value.shape[0])
        return value[:, :512]

    monkeypatch.setattr(
        torch.ops.custom_op,
        "custom_deepseek_v41_q_norm_projection_rope_gaudi2",
        projection,
        raising=False,
    )
    harness = SimpleNamespace(
        weights=SimpleNamespace(
            wq_b=SimpleNamespace(weight=torch.empty((8, 1280)), channel_scale=torch.empty((1, 8))),
            q_norm=SimpleNamespace(weight=torch.empty(1280)),
        ),
        heads=1,
        eps=1e-6,
        _rotary_native_table=lambda: torch.empty((1, 64)),
    )
    rows = 11 * 6
    value = torch.zeros((rows, 1280), dtype=torch.bfloat16)
    positions = torch.arange(rows, dtype=torch.int32)

    result = paged_module.PagedCSA2Attention.project_query_input(harness, value, positions)

    assert calls == [64, 2]
    assert result.shape == (rows, 1, 512)


def test_n11_native_rope_tiles_request_rows(monkeypatch):
    calls = []

    def rope(value, positions, table):
        del table
        assert value.shape[0] == positions.numel() <= 64
        calls.append(value.shape[0])
        return value

    monkeypatch.setattr(
        torch.ops.custom_op,
        "custom_deepseek_v41_rope_bf16_gaudi2",
        rope,
        raising=False,
    )
    harness = SimpleNamespace(
        native_rope=True,
        _rotary_native_table=lambda: torch.empty((1, 64)),
    )
    rows = 11 * 6
    value = torch.zeros((rows, 512), dtype=torch.bfloat16)
    positions = torch.arange(rows, dtype=torch.int32)

    result = paged_module.PagedCSA2Attention._rope(
        harness, value, positions, request_batch=True)

    assert calls == [64, 2]
    assert torch.equal(result, value)


def test_n2_target_finishes_each_lane_as_one_small_native_reduce(monkeypatch):
    capacity = 2
    rows = capacity * 6
    harness = _ForwardHarness(capacity)
    harness.groups = 1
    harness.heads = 10
    harness.woa_fp8 = False
    harness.woa_output_roundtrip = False
    monkeypatch.setattr(paged_module.gaudi_envs, "VLLM_HPU_DSV41_DSPARK_WO_HANDOFF", False)

    mla_calls = []
    reductions = []

    def fake_write(cache, values, row_ids):
        del cache, values
        return torch.zeros(row_ids.shape, dtype=torch.int32)

    def fake_mla(query, *args, **kwargs):
        del args, kwargs
        mla_calls.append(tuple(query.shape))
        return torch.ones((query.shape[0], harness.heads, 512), dtype=torch.bfloat16)

    def fake_reduce(partial, ready_outputs=(), **kwargs):
        del kwargs
        reductions.append(
            (
                tuple(partial.shape),
                partial.numel(),
                tuple(int(ready[0, 0]) for ready in ready_outputs),
            )
        )
        return partial.reshape(partial.shape[0], -1)

    monkeypatch.setattr(batch_attention, "write_state_rows", fake_write)
    monkeypatch.setattr(batch_attention, "batch_packed_mla", fake_mla)
    monkeypatch.setattr(paged_module, "rms_norm", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(paged_module, "pack_swa", lambda value: value)

    for method in ("_finish_output", "_finish_projected_output", "_reduce_output"):
        setattr(harness, method, getattr(paged_module.PagedCSA2Attention, method).__get__(harness, type(harness)))
    harness.project_output = lambda value: value
    harness.project_output_consumer = lambda value: value
    harness.reduce = fake_reduce

    positions = torch.arange(100, 100 + rows, dtype=torch.int32)
    slots = torch.repeat_interleave(torch.arange(capacity, dtype=torch.int32), 6)
    row_valid = torch.ones((rows, ), dtype=torch.bool)
    value = torch.randn((rows, 512), dtype=torch.bfloat16)
    pages = torch.arange(1, rows * 4 + 1, dtype=torch.int32).reshape(rows, 4)
    selected = torch.full((rows, 512), -1, dtype=torch.int32)
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32)
    done = torch.zeros((rows, ), dtype=torch.int32)
    ready_outputs = (
        torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2),
        -torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
    )

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
        request_slots=(0, 1),
        ready_outputs=ready_outputs,
    )

    assert mla_calls == [(rows, 512)]
    assert [shape for shape, _numel, _ready in reductions] == [(6, 1, 5120), (6, 1, 5120)]
    assert [numel for _shape, numel, _ready in reductions] == [30720, 30720]
    assert [ready for _shape, _numel, ready in reductions] == [(0, 0), (12, -18)]
    assert result[0].shape == (rows, 5120)


def _run_n11_index_tiling(monkeypatch, branch):
    from vllm_gaudi.ops import deepseek_v41_full_index_mme as full_index_module
    from vllm_gaudi.ops import deepseek_v41_indexer as indexer_module
    from vllm_gaudi.ops import deepseek_v41_reindex_compact as compact_module
    from vllm_gaudi.ops import deepseek_v41_reindex_mme as reindex_module

    capacity = 11
    rows = capacity * 6
    harness = _ForwardHarness(capacity)
    harness.ratio = 1
    harness.owns_index = True
    harness.owns_kv = False
    harness.layer = 1 if branch in ("reindex", "bounded") else 0
    harness.candidate_source = 0
    harness.cache = SimpleNamespace(
        index=torch.zeros((128, 68), dtype=torch.uint8),
        main=torch.zeros((1, 288), dtype=torch.uint8),
    )
    harness.index_heads = 4
    harness.length = 65536
    harness.batch_full_index_mme = branch == "full"
    harness.batch_full_index_mme_prefix_tokens = 4096
    harness.batch_reindex_mme = branch in ("reindex", "bounded")
    harness.bounded_reindex = branch == "bounded"
    harness.batch_index_tiled_keys = False

    index_calls = []
    read_completions = []

    def fake_read(cache, row_ids, completion):
        assert row_ids.numel() == completion.numel() <= 60
        read_completions.append(completion.clone())
        return cache

    def record_call(name, query, weights, pages, positions, candidates, key_ready=None):
        assert query.shape[0] == weights.shape[0] == pages.shape[0] == positions.numel()
        assert query.shape[0] == candidates.shape[0] <= 64
        if key_ready is not None:
            assert key_ready.numel() == query.shape[0]
        index_calls.append(
            {
                "branch": name,
                "query": query[:, 0, 0].to(torch.int64).tolist(),
                "weights": weights[:, 0].to(torch.int64).tolist(),
                "pages": pages[:, 0].to(torch.int64).tolist(),
                "positions": positions.tolist(),
                "candidates": candidates[:, 0].to(torch.int64).tolist(),
                "key_ready": None if key_ready is None else key_ready.tolist(),
            }
        )

    def selected_rows(positions):
        return positions[:, None].expand(-1, 512).clone()

    def selected_blocks(positions):
        return positions[:, None].expand(-1, 2048).clone()

    def fake_full(query, weights, cache, pages, positions, candidates, **kwargs):
        del cache
        record_call("full", query, weights, pages, positions, candidates, kwargs["key_ready"])
        blocks = selected_blocks(positions) if kwargs["publish_candidates"] else None
        return selected_rows(positions), blocks

    def fake_reindex(query, weights, cache, pages, positions, candidates, **kwargs):
        del cache, kwargs
        record_call("reindex", query, weights, pages, positions, candidates)
        return selected_rows(positions)

    def fake_bounded(query, weights, cache, pages, positions, candidates, **kwargs):
        del cache, kwargs
        record_call("bounded", query, weights, pages, positions, candidates)
        return selected_rows(positions)

    def fake_runtime(query, weights, cache, pages, positions, candidates, **kwargs):
        del cache
        record_call("runtime", query, weights, pages, positions, candidates)
        blocks = selected_blocks(positions) if kwargs["publish_candidates"] else None
        return selected_rows(positions), blocks

    monkeypatch.setattr(full_index_module, "full_index_mme_select", fake_full)
    monkeypatch.setattr(reindex_module, "reindex_mme_select", fake_reindex)
    monkeypatch.setattr(reindex_module, "bounded_reindex_mme_select", fake_bounded)
    monkeypatch.setattr(compact_module, "bounded_reindex_bucket", lambda _rows: True)
    monkeypatch.setattr(indexer_module, "runtime_index_select", fake_runtime)
    monkeypatch.setattr(batch_attention, "read_state_rows", fake_read)
    monkeypatch.setattr(
        batch_attention,
        "write_state_rows",
        lambda _cache, _values, row_ids: torch.zeros(row_ids.shape, dtype=torch.int32),
    )
    monkeypatch.setattr(
        batch_attention,
        "batch_packed_mla",
        lambda query, *_args, **_kwargs: torch.ones((query.shape[0], 1), dtype=torch.bfloat16),
    )
    monkeypatch.setattr(paged_module, "rms_norm", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(paged_module, "pack_swa", lambda value: value)

    marker = torch.arange(rows, dtype=torch.float32)
    harness._prepare_index_queries = lambda _value, _qr, _positions, **_kwargs: (
        marker.reshape(rows, 1, 1).expand(rows, 4, 128).clone(),
        marker.reshape(rows, 1).expand(rows, 4).clone(),
    )

    positions = torch.arange(rows, dtype=torch.int32)
    slots = torch.repeat_interleave(torch.arange(capacity, dtype=torch.int32), 6)
    row_valid = torch.ones((rows, ), dtype=torch.bool)
    pages = torch.arange(rows, dtype=torch.int32)[:, None].expand(rows, 4).clone()
    selected = torch.full((rows, 512), -1, dtype=torch.int32)
    candidates = torch.arange(rows, dtype=torch.int32)[:, None].expand(rows, 2048).clone()
    index_done = (torch.arange(rows, dtype=torch.int32) + 100).contiguous()
    value = torch.zeros((rows, 512), dtype=torch.bfloat16)

    result = paged_module.PagedCSA2Attention.forward_batch_target(
        harness,
        value,
        positions,
        slots,
        pages,
        selected,
        candidates,
        index_done,
        index_done,
        row_valid=row_valid,
        request_lengths=(6, ) * capacity,
        request_slots=tuple(range(capacity)),
    )

    expected_rows = [list(range(60)), list(range(60, 66))]
    assert [len(call["positions"]) for call in index_calls] == [60, 6]
    assert [len(completion) for completion in read_completions] == [60, 6]
    assert [completion.tolist() for completion in read_completions] == [
        list(range(100, 160)),
        list(range(160, 166)),
    ]
    assert [call["branch"] for call in index_calls] == [branch, branch]
    for key in ("query", "weights", "pages", "positions", "candidates"):
        assert [call[key] for call in index_calls] == expected_rows
    if branch == "full":
        assert [call["key_ready"] for call in index_calls] == [
            list(range(100, 160)),
            list(range(160, 166)),
        ]
    else:
        assert all(call["key_ready"] is None for call in index_calls)
    assert result[1][:, 0].tolist() == list(range(rows))
    if branch in ("full", "runtime"):
        assert result[2][:, 0].tolist() == list(range(rows))


@pytest.mark.parametrize("branch", ("full", "reindex", "bounded", "runtime"))
def test_n11_index_selection_tiles_native_rows_and_preserves_order(monkeypatch, branch):
    _run_n11_index_tiling(monkeypatch, branch)


def test_compact_target_attention_runs_eight_queries_and_keeps_c6_compressor(monkeypatch):
    from vllm_gaudi.ops.deepseek_v41_target_queries import pack_target_query_rows

    capacity, body_rows = 11, 66
    harness = _ForwardHarness(capacity)
    harness.ratio = 2
    harness.owns_kv = True
    harness.cache = SimpleNamespace(main=torch.zeros((1, 288), dtype=torch.uint8))
    events, writes = [], []

    def compress(value, positions, slots, pages, active, request_lengths, request_slots,
                 runtime_request_slots=False):
        events.append(("compress", value.shape[0]))
        assert request_lengths == (6, ) * capacity
        return torch.ones(body_rows, dtype=torch.int32), torch.ones(body_rows, dtype=torch.int32)

    def qkv(value):
        events.append(("qkv", value.shape[0]))
        return value, value

    def write(cache, values, row_ids):
        writes.append(row_ids.clone())
        return torch.zeros_like(row_ids)

    def mla(query, swa, main, selected, pages, positions, slots, sink, scale, swa_done, main_done, **kwargs):
        events.append(("mla", query.shape[0]))
        assert kwargs["tile_indices"] is None
        assert main_done.shape == swa_done.shape == (query.shape[0], )
        return query[:, :4].clone()

    harness._compress_batch_target = compress
    harness._project_qkv_input = qkv
    monkeypatch.setattr(batch_attention, "write_state_rows", write)
    monkeypatch.setattr(batch_attention, "batch_packed_mla", mla)
    monkeypatch.setattr(paged_module, "rms_norm", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(paged_module, "pack_swa", lambda value: value)
    valid = torch.zeros(body_rows, dtype=torch.bool)
    valid[6:12] = True
    valid[60] = True
    slots = torch.where(valid, torch.arange(body_rows, dtype=torch.int32) // 6, -1)
    positions = torch.where(valid, torch.arange(body_rows, dtype=torch.int32) + 100, -1)
    value = torch.arange(body_rows * 512, dtype=torch.float32).reshape(body_rows, 512)
    plan = pack_target_query_rows(valid.tolist())
    selected = torch.full((body_rows, 512), -1, dtype=torch.int32)
    candidates = torch.full((body_rows, 2048), -1, dtype=torch.int32)
    done = torch.zeros(body_rows, dtype=torch.int32)
    result = paged_module.PagedCSA2Attention.forward_batch_target(
        harness, value, positions, slots, torch.ones((body_rows, 4), dtype=torch.int32),
        selected, candidates, done, done, row_valid=valid, request_lengths=(6, ) * capacity,
        request_slots=tuple(range(capacity)), query_indices=torch.tensor(plan.source_rows),
        query_valid=torch.tensor(plan.valid))
    assert events == [("compress", 66), ("qkv", 8), ("mla", 8)]
    assert len(writes) == 1 and writes[0].shape == (8, )
    assert writes[0][-1] == -1
    assert torch.equal(result[0][valid], value[valid, :4])
    assert torch.count_nonzero(result[0][~valid]) == 0
    assert result[1].shape == (body_rows, 512)
    assert result[2].shape == (body_rows, 2048)
    assert result[3].shape == result[4].shape == (body_rows, )
