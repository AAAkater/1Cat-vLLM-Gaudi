# SPDX-License-Identifier: Apache-2.0
"""Stable (N,B) target snapshot: device layout, pure restore and fullgraph capture."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_gaudi.ops import deepseek_v41_batch_attention as batch_attention
from vllm_gaudi.ops import deepseek_v41_batch_draft as draft_batch
from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata
from vllm_gaudi.ops.deepseek_v41_batch_target import (
    DSparkStableTargetLayout,
    DSparkStableTargetSnapshot,
    DSparkTargetBatch,
    DSparkTargetSnapshot,
    capture_stable_target_rows,
    make_target_ticket,
    restore_stable_target_rows,
)
from vllm_gaudi.ops.deepseek_v41_target_buckets import pack_target_bucket
from vllm_gaudi.v1.worker import deepseek_v41_dspark_batch_runner as runner_module
from vllm_gaudi.v1.worker.deepseek_v41_dspark_batch_runner import DSparkPackedTargetStages


@pytest.fixture(autouse=True)
def _cpu_dynamo_only(monkeypatch):
    # Dynamo stream discovery must not acquire an HPU for these CPU contracts.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)


class _LaneState(nn.Module):

    def __init__(self, capacity, compressor):
        super().__init__()
        self.register_buffer("swa", torch.zeros((capacity * 256, 4), dtype=torch.uint8))
        if compressor:
            self.register_buffer("kv_history", torch.zeros((capacity * 8, 3), dtype=torch.float32))
            self.register_buffer("score_history", torch.zeros((capacity * 8, 3), dtype=torch.float32))


class _Cache(nn.Module):

    def __init__(self, ratio, rows):
        super().__init__()
        self.ratio = ratio
        self.register_buffer("main", torch.zeros((rows, 5), dtype=torch.uint8))
        self.register_buffer("index", torch.zeros((rows, 3), dtype=torch.uint8))


def _program(capacity, seed=0):
    program = nn.Module()
    program.config = {"text_config": {"hidden_size": 2}}
    program.layers = nn.ModuleList()
    for index, compressor in enumerate((True, False, None, True)):
        layer = nn.Module()
        layer.layer = index
        layer.attention = nn.Module()
        if compressor is not None:
            layer.attention.batch_state = _LaneState(capacity, compressor)
        program.layers.append(layer)
    program.shared = nn.Module()
    # The ratio-128 cache is deliberately short so the row-bound filter of
    # the legacy capture is exercised.
    program.shared.sources = nn.ModuleDict({"2": _Cache(2, 4096), "4": _Cache(4, 4096), "128": _Cache(128, 8)})
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for buffer in program.buffers():
            if buffer.is_floating_point():
                buffer.copy_(torch.randn(buffer.shape, generator=generator))
            else:
                buffer.copy_(torch.randint(0, 250, buffer.shape, generator=generator, dtype=torch.int64))
    return program


def _assert_same_state(left, right):
    for first, second in zip(left.buffers(), right.buffers(), strict=True):
        assert torch.equal(first, second)


def _inputs(capacity, lanes, *, shared_pages=()):
    """``lanes`` maps state lane -> (target length, start position)."""
    metadata = DSparkBatchMetadata.build(
        requests=[f"request-{i}" if i in lanes else None for i in range(capacity)],
        target_query_lengths=[lanes[i][0] if i in lanes else 0 for i in range(capacity)],
        draft_token_ids=[list(range(1, lanes[i][0])) if i in lanes else [] for i in range(capacity)],
        request_slots=[i if i in lanes else -1 for i in range(capacity)],
        slot_generations=[3 + i if i in lanes else 0 for i in range(capacity)],
        remaining_output_budget=[64 if i in lanes else 0 for i in range(capacity)],
        target_start_positions=[lanes[i][1] if i in lanes else 0 for i in range(capacity)],
        round_generation=9,
        active=[i in lanes for i in range(capacity)],
    )
    queries = torch.arange(capacity * 6, dtype=torch.int64).reshape(capacity, 6)
    pages = torch.arange(1, capacity * 4 + 1, dtype=torch.int32).reshape(capacity, 4)
    for alias, source in shared_pages:
        pages[alias] = pages[source]
    return metadata, queries, pages


def _batch(metadata, queries, pages, bucket):
    packed = pack_target_bucket(metadata, queries, pages, bucket)
    compact = bucket != metadata.capacity
    batch = DSparkTargetBatch.from_metadata(packed.metadata, packed.queries, packed.pages, batch_capacity=bucket,
                                            state_capacity=metadata.capacity if compact else None)
    return batch, packed.source_lanes


def _write_target_rows(program, batch, value):
    """Simulate the target's tentative writes on its exact write set."""
    for row in DSparkTargetSnapshot.capture(program, batch).rows:
        row.destination.index_copy_(0, row.indices, torch.full_like(row.values, value))


def _touch_outside(program):
    """Mutate row zero after capture; padding entries must not clobber it."""
    with torch.no_grad():
        for buffer in program.buffers():
            buffer[0].add_(3)


def _legacy_device_restore(snapshot, counts, source_lanes):
    # Mirrors the pre-existing DSparkPackedTargetStages.finish_device.
    if len(source_lanes) != counts.shape[0]:
        indices = torch.tensor([max(0, lane) for lane in source_lanes], dtype=torch.int64)
        valid = torch.tensor([lane >= 0 for lane in source_lanes], dtype=torch.bool)
        counts = torch.where(valid, counts.index_select(0, indices), 0)
    snapshot.restore_device(counts)


# (state capacity N, execution bucket B, active lanes, pages aliases)
_CASES = {
    "full_mixed": (8, 8, {0: (6, 100), 2: (3, 250), 5: (1, 511), 7: (6, 600)}, ()),
    # Lanes 1/3 alias pages and positions, so compressed-page rows have two
    # owners; lane 6 crosses the ratio-4 publication boundary.
    "compact_shared_pages": (8, 4, {1: (6, 124), 3: (6, 124), 6: (2, 30)}, ((3, 1), )),
    "single_lane_n11": (11, 1, {10: (6, 255)}, ()),
    # Slot 0 wraps its SWA ring onto row 0, which every padding entry uses.
    "ring_wrap_row_zero": (4, 4, {0: (6, 254), 1: (5, 6), 2: (6, 1), 3: (4, 127)}, ()),
}


def _case(name):
    capacity, bucket, lanes, aliases = _CASES[name]
    metadata, queries, pages = _inputs(capacity, lanes, shared_pages=aliases)
    return capacity, bucket, lanes, metadata, queries, pages


def _count_patterns(capacity, lanes):
    active = sorted(lanes)
    patterns = [[0] * capacity, [lanes[lane][0] if lane in lanes else 0 for lane in range(capacity)]]
    for offset in range(3):
        counts = [0] * capacity
        for position, lane in enumerate(active):
            counts[lane] = (position + offset) % (lanes[lane][0] + 1)
        patterns.append(counts)
    # Alternate full accept / full reject so shared rows see both orders; a
    # duplicate scatter without the shared-row reduction would diverge.
    for parity in range(2):
        patterns.append([lanes[lane][0] if lane in lanes and active.index(lane) % 2 == parity else 0
                         for lane in range(capacity)])
    return patterns


@pytest.mark.parametrize("name", sorted(_CASES))
def test_device_layout_matches_legacy_owner_map(name):
    capacity, bucket, _lanes, metadata, queries, pages = _case(name)
    program = _program(capacity)
    batch, source_lanes = _batch(metadata, queries, pages, bucket)
    legacy = DSparkTargetSnapshot.capture(program, batch)
    stable = DSparkStableTargetLayout(program, capacity).view(bucket)
    stable.capture(batch, source_lanes)
    layout = stable.layout
    by_destination = {id(row.destination): row for row in legacy.rows}
    shared_rows = 0
    pairs = zip(layout.destinations, layout.destination_groups, strict=True)
    for number, (destination, group) in enumerate(pairs):
        indices = stable.indices[group].tolist()
        requests = stable.owners[0, group].tolist()
        queries_ = stable.owners[1, group].tolist()
        observed = {}
        for index, request, query in zip(indices, requests, queries_, strict=True):
            if query < 6:
                observed.setdefault(index, []).append((request, query))
        row = by_destination.get(id(destination))
        expected = {} if row is None else {
            index: sorted(owners) for index, owners in zip(row.indices.tolist(), row.owners, strict=True)}
        assert {index: sorted(owners) for index, owners in observed.items()} == expected
        assert all(query == 6 for query in queries_ if query >= 6)
        shared_rows += sum(len(owners) > 1 for owners in expected.values())
        entries = torch.tensor([entry for entry, query in enumerate(queries_) if query < 6], dtype=torch.int64)
        rows = torch.tensor([index for index, query in zip(indices, queries_, strict=True) if query < 6],
                            dtype=torch.int64)
        assert torch.equal(stable.saved[number].index_select(0, entries), destination.index_select(0, rows))
    if name == "compact_shared_pages":
        assert shared_rows > 0
    # Inactive execution lanes map to -1 (limit 0) even in the identity layout.
    assert stable.lane_map.tolist() == [
        lane if active else -1 for lane, active in zip(source_lanes, batch.host_request_active, strict=True)]


@pytest.mark.parametrize("name", sorted(_CASES))
def test_device_and_host_restore_match_legacy_row_for_row(name):
    capacity, bucket, lanes, metadata, queries, pages = _case(name)
    for counts in _count_patterns(capacity, lanes):
        for path in ("device", "host"):
            legacy_program, stable_program = _program(capacity, 5), _program(capacity, 5)
            batch, source_lanes = _batch(metadata, queries, pages, bucket)
            legacy = DSparkTargetSnapshot.capture(legacy_program, batch)
            legacy.prepare_device_restore()
            stable = DSparkStableTargetLayout(stable_program, capacity).view(bucket)
            stable.capture(batch, source_lanes)
            for program in (legacy_program, stable_program):
                _write_target_rows(program, batch, 201)
                _touch_outside(program)
            device_counts = torch.tensor(counts, dtype=torch.int64)
            if path == "device":
                _legacy_device_restore(legacy, device_counts, source_lanes)
                stable.restore_device(device_counts)
            else:
                lane_counts = tuple(counts[lane] if lane >= 0 else 0 for lane in source_lanes)
                legacy.restore(lane_counts)
                stable.restore(lane_counts)
            _assert_same_state(legacy_program, stable_program)
            # Abort after the device restore: zero counts are idempotent.
            legacy.restore((0, ) * bucket)
            stable.restore((0, ) * bucket)
            _assert_same_state(legacy_program, stable_program)


@pytest.mark.parametrize("bad", [7, -1])
def test_out_of_range_count_rolls_back_every_lane(bad):
    capacity, bucket, lanes, metadata, queries, pages = _case("full_mixed")
    legacy_program, stable_program = _program(capacity, 2), _program(capacity, 2)
    batch, source_lanes = _batch(metadata, queries, pages, bucket)
    legacy = DSparkTargetSnapshot.capture(legacy_program, batch)
    stable = DSparkStableTargetLayout(stable_program, capacity).view(bucket)
    stable.capture(batch, source_lanes)
    for program in (legacy_program, stable_program):
        _write_target_rows(program, batch, 99)
    counts = [lanes[lane][0] if lane in lanes else 0 for lane in range(capacity)]
    counts[1] = bad  # even an inactive lane cannot authorize tentative rows
    stable.restore_device(torch.tensor(counts))
    legacy.restore((0, ) * bucket)
    _assert_same_state(legacy_program, stable_program)


def test_capture_and_restore_are_fullgraph_without_recompiles():
    capacity, bucket = 8, 4
    program, reference = _program(capacity, 1), _program(capacity, 1)
    layout = DSparkStableTargetLayout(program, capacity)
    stable = layout.view(bucket)
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    torch._dynamo.reset()
    capture = torch.compile(capture_stable_target_rows, backend=backend, fullgraph=True, dynamic=False)
    restore = torch.compile(restore_stable_target_rows, backend=backend, fullgraph=True, dynamic=False)
    rounds = (
        ({1: (6, 124), 3: (6, 124), 6: (2, 30)}, ((3, 1), ), [0, 4, 0, 6, 0, 0, 1, 0]),
        ({0: (1, 511)}, (), [1, 0, 0, 0, 0, 0, 0, 0]),
        ({2: (3, 250), 5: (6, 600), 6: (5, 2), 7: (2, 9)}, (), [0, 0, 2, 0, 0, 6, 3, 0]),
        ({4: (6, 255)}, (), [0, 0, 0, 0, 9, 0, 0, 0]),
    )
    with torch._dynamo.config.patch(error_on_recompile=True):
        for lanes, aliases, counts in rounds:
            metadata, queries, pages = _inputs(capacity, lanes, shared_pages=aliases)
            batch, source_lanes = _batch(metadata, queries, pages, bucket)
            legacy = DSparkTargetSnapshot.capture(reference, batch)
            capture(batch.positions, batch.row_slots, batch.row_valid, batch.row_pages, batch.request_slots,
                    layout.groups, stable.indices, stable.owners, stable.lane_map, layout.destinations,
                    stable.saved, layout.destination_groups)
            layout.owner, stable.captured = stable, True
            for target in (program, reference):
                _write_target_rows(target, batch, 17)
            committed = torch.tensor(counts, dtype=torch.int64)
            restore(committed, *stable.restore_arguments())
            legacy.prepare_device_restore()
            _legacy_device_restore(legacy, committed, source_lanes)
            _assert_same_state(program, reference)
            stable.release()
    assert len(graphs) == 2


def test_capture_and_restore_compile_with_aot_functionalization():
    capacity, bucket = 4, 4
    program, reference = _program(capacity, 3), _program(capacity, 3)
    metadata, queries, pages = _inputs(capacity, {0: (6, 254), 1: (5, 6), 3: (4, 127)})
    batch, source_lanes = _batch(metadata, queries, pages, bucket)
    layout = DSparkStableTargetLayout(program, capacity)
    stable = layout.view(bucket)
    torch._dynamo.reset()
    capture = torch.compile(capture_stable_target_rows, backend="aot_eager", fullgraph=True, dynamic=False)
    capture(batch.positions, batch.row_slots, batch.row_valid, batch.row_pages, batch.request_slots, layout.groups,
            stable.indices, stable.owners, stable.lane_map, layout.destinations, stable.saved,
            layout.destination_groups)
    layout.owner, stable.captured = stable, True
    eager = DSparkStableTargetLayout(_program(capacity, 3), capacity).view(bucket)
    eager.capture(batch, source_lanes)
    assert torch.equal(stable.indices, eager.indices) and torch.equal(stable.owners, eager.owners)
    assert all(torch.equal(left, right) for left, right in zip(stable.saved, eager.saved, strict=True))
    legacy = DSparkTargetSnapshot.capture(reference, batch)
    for target in (program, reference):
        _write_target_rows(target, batch, 33)
    restore = torch.compile(restore_stable_target_rows, backend="aot_eager", fullgraph=True, dynamic=False)
    counts = torch.tensor([3, 5, 0, 1])
    restore(counts, *stable.restore_arguments())
    legacy.restore((3, 5, 0, 1))
    _assert_same_state(program, reference)


def test_persistent_buffers_are_reused_and_views_share_storage():
    capacity = 8
    program = _program(capacity)
    layout = DSparkStableTargetLayout(program, capacity)
    small, full = layout.view(2), layout.view(8)
    assert layout.view(2) is small
    pointers = (layout.indices.data_ptr(), layout.owners.data_ptr(), layout.lane_map.data_ptr(),
                tuple(values.data_ptr() for values in layout.saved))
    for lanes in ({1: (6, 10)}, {4: (2, 20), 7: (6, 30)}):
        metadata, queries, pages = _inputs(capacity, lanes)
        batch, source_lanes = _batch(metadata, queries, pages, 2)
        small.capture(batch, source_lanes)
        small.restore((0, 0))
        small.release()
    assert small.indices.data_ptr() == full.indices.data_ptr() == pointers[0]
    assert (layout.indices.data_ptr(), layout.owners.data_ptr(), layout.lane_map.data_ptr(),
            tuple(values.data_ptr() for values in layout.saved)) == pointers
    assert small.indices.shape == (len(layout.groups), 12)
    assert full.saved[0].shape[0] == 48
    assert layout.bytes == full.bytes


def test_ownership_guards_and_lane_slot_invariant():
    capacity = 4
    program = _program(capacity)
    layout = DSparkStableTargetLayout(program, capacity)
    stable = layout.view(4)
    with pytest.raises(RuntimeError, match="no captured round"):
        stable.restore((0, ) * 4)
    metadata, queries, pages = _inputs(capacity, {1: (6, 10), 2: (1, 3)})
    batch, source_lanes = _batch(metadata, queries, pages, 4)
    with pytest.raises(ValueError, match="state lane == logical slot"):
        stable.capture(batch, (0, 2, 1, 3))
    stable.capture(batch, source_lanes)
    with pytest.raises(RuntimeError, match="still owned"):
        layout.view(2).capture(*_batch(*_inputs(capacity, {1: (2, 4)}), 2))
    with pytest.raises(ValueError, match="every state lane"):
        stable.restore_device(torch.zeros(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="0..6"):
        stable.restore((0, 7, 0, 0))
    stable.release()
    with pytest.raises(RuntimeError, match="no captured round"):
        stable.restore_device(torch.zeros(4, dtype=torch.int64))
    assert layout.matches(program)
    program.layers[0].attention.batch_state.register_buffer("swa", torch.zeros((capacity * 256, 4),
                                                                                 dtype=torch.uint8))
    assert not layout.matches(program)


def test_owner_layout_flag_defaults_off_and_requires_device_finish():
    owner = object.__new__(DSparkPackedTargetStages)
    owner.observed_query_keys = set()
    owner.device_finish = False
    with pytest.raises(TypeError, match="boolean"):
        owner.configure_device_owner_layout("true")
    with pytest.raises(ValueError, match="requires dsv41_dspark_device_target_finish"):
        owner.configure_device_owner_layout(True)
    owner.configure_device_owner_layout(False)
    assert owner.device_owner_layout is False
    owner.device_finish = True
    owner.configure_device_owner_layout(True)
    assert owner.device_owner_layout is True
    owner.observed_query_keys = {(4, 24)}
    with pytest.raises(RuntimeError, match="before warmup"):
        owner.configure_device_owner_layout(False)


@pytest.mark.parametrize(("additional", "expected"), [
    ({}, [("finish", False), ("layout", False)]),
    ({"dsv41_dspark_device_target_finish": True, "dsv41_dspark_device_owner_layout": True},
     [("finish", True), ("layout", True)]),
])
def test_model_runner_forwards_owner_layout_flag_after_device_finish(monkeypatch, additional, expected):
    from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
    from vllm_gaudi.v1.worker import deepseek_v41_runner
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner

    calls = []
    target = SimpleNamespace(configure_device_finish=lambda enabled: calls.append(("finish", enabled)),
                             configure_device_owner_layout=lambda enabled: calls.append(("layout", enabled)))
    state = SimpleNamespace(capacity=4, slots=RequestSlots(4), pending=None, single_owner=None, prefill_owner=None)
    runner = object.__new__(V41ModelRunner)
    runner.vllm_config = SimpleNamespace(additional_config=additional)
    runner.model = SimpleNamespace(batch_state=state, tp_rank=0, pp_rank=0,
                                   initialize_dspark_batch=lambda capacity: (target, SimpleNamespace()))
    monkeypatch.setattr(deepseek_v41_runner, "DSparkBatchRunner",
                        lambda target_stage, draft_stage, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(deepseek_v41_runner, "DSparkBatchTrace", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(deepseek_v41_runner.dist, "is_initialized", lambda: False)
    runner._initialize_dspark_batch_runner(4)
    assert calls == expected


def _stages(monkeypatch, program, capacity, *, owner_layout, fused=False):
    import vllm_gaudi.models.deepseek_v41_batch_target_program as target_program

    class Stage:

        def __init__(self, program, lengths, slots, **kwargs):
            self.program = program
            self.compiled = SimpleNamespace()

        def snapshot(self, batch):
            return make_target_ticket(self.program, batch)

        def __call__(self, hidden, pre_mix, batch, engram):
            _write_target_rows(self.program, batch, 123)
            rows = batch.target_rows
            return torch.ones(rows, 2), torch.full((rows, 3), 9.0)

    monkeypatch.setattr(target_program, "DSparkTargetStage", Stage)
    monkeypatch.setattr(runner_module, "_target_greedy_ids",
                        lambda program, hidden, lanes: torch.zeros(hidden.shape[0], dtype=torch.int64))
    owner = DSparkPackedTargetStages(program, SimpleNamespace(capacity=capacity), device="cpu")
    owner.configure_execution_buckets(True, buckets=[1, 2, 4])
    owner.configure_device_finish(True)
    owner.configure_device_owner_layout(owner_layout)
    owner.configure_fused_target_restore(fused)
    return owner


@pytest.mark.parametrize("outcome", ["finish", "abort", "invalid"])
def test_runner_round_matches_legacy_path_and_releases_storage(monkeypatch, outcome):
    capacity = 8
    programs = {flag: _program(capacity, 11) for flag in (False, True)}
    owners = {flag: _stages(monkeypatch, programs[flag], capacity, owner_layout=flag) for flag in programs}
    rounds = ({1: (6, 124), 3: (6, 124), 6: (2, 30)}, {0: (6, 254)}, {2: (3, 250), 5: (1, 511)})
    for index, lanes in enumerate(rounds):
        metadata, queries, pages = _inputs(capacity, lanes, shared_pages=((3, 1), ) if index == 0 else ())
        counts = [0] * capacity
        for lane, (length, _start) in lanes.items():
            counts[lane] = (lane + index) % (length + 1)
        if outcome == "invalid":
            counts[sorted(lanes)[0]] = 7
        for flag, owner in owners.items():
            result = owner.forward_batch(metadata, queries, None, None, pages)
            snapshot = result.payload["ticket"].snapshot
            assert isinstance(snapshot, DSparkStableTargetSnapshot) is flag
            _touch_outside(programs[flag])
            if outcome == "abort":
                owner.abort(result)
                continue
            owner.finish_device(result, torch.tensor(counts))
            if outcome == "invalid":
                with pytest.raises(RuntimeError, match="outside 0..6"):
                    owner.finish(result, counts)
                owner.abort(result)
            else:
                owner.finish(result, counts)
            assert result.payload["finished"]
        _assert_same_state(programs[False], programs[True])
    assert owners[False].stable_target_layout is None
    layout = owners[True].stable_target_layout
    assert layout is not None and layout.owner is None
    assert set(layout.views) == {(4, False), (1, False), (2, False)}


def test_runner_refuses_overlapping_stable_rounds(monkeypatch):
    capacity = 4
    owner = _stages(monkeypatch, _program(capacity), capacity, owner_layout=True)
    metadata, queries, pages = _inputs(capacity, {1: (6, 10)})
    first = owner.forward_batch(metadata, queries, None, None, pages)
    with pytest.raises(RuntimeError, match="still owned"):
        owner.forward_batch(metadata, queries, None, None, pages)
    owner.abort(first)
    second = owner.forward_batch(metadata, queries, None, None, pages)
    owner.abort(second)
    assert owner.stable_target_layout.owner is None


# ---------------------------------------------------------------------------
# Fused restore inside the packed draft transaction.
# ---------------------------------------------------------------------------


class _Round:
    """Pure tensor speculator stub; the transaction graph stays traceable."""

    def __init__(self, *args, **kwargs):
        self.speculator = self

    def propose_after_verify(self, states, positions, slots, committed, output, output_count, draft_enabled,
                             active, status, **kwargs):
        proposals = committed.to(torch.int64).unsqueeze(1) + torch.arange(5, dtype=torch.int64)
        proposals = torch.where(active.unsqueeze(1), proposals, torch.full_like(proposals, -1))
        return proposals, torch.zeros(proposals.shape, dtype=torch.float32)


def _patch_transaction(monkeypatch, counts_of):
    """Verify stub: committed counts come from ``counts_of(control)``."""

    def verify(target_ids, proposals, control, active, binding_sidecar):
        committed = torch.where(active, counts_of(control).to(torch.int64), torch.zeros_like(control[:, 0]))
        rows = torch.arange(6, dtype=torch.int64).unsqueeze(0)
        output = torch.where(rows < committed.unsqueeze(1), target_ids, torch.full_like(target_ids, -1))
        return target_ids, output, committed, committed, target_ids[:, 0], active.clone(), torch.zeros_like(committed)

    def pack(control, committed, output_count, output, proposal_ids, draft_active, status, binding_sidecar):
        rows = committed.shape[0]
        return torch.cat((torch.zeros((rows, 1), dtype=torch.int64), committed.unsqueeze(1).to(torch.int64),
                          output_count.unsqueeze(1).to(torch.int64), torch.zeros((rows, 13), dtype=torch.int64)), 1)

    monkeypatch.setattr(draft_batch, "PackedDraftRound", _Round)
    monkeypatch.setattr(draft_batch, "verify_control_from_target_batch", verify)
    monkeypatch.setattr(draft_batch, "pack_record_batch", pack)


def _transaction_args(capacity, active, counts):
    control = torch.zeros((capacity, 7), dtype=torch.int64)
    control[:, 0] = torch.tensor(counts, dtype=torch.int64)
    return (torch.arange(capacity * 6, dtype=torch.int64).reshape(capacity, 6),
            torch.zeros((capacity, 6, 3), dtype=torch.float32),
            torch.zeros((capacity, 6), dtype=torch.int32),
            torch.arange(capacity, dtype=torch.int32),
            control,
            torch.full((capacity, 5), -1, dtype=torch.int64),
            torch.tensor(active, dtype=torch.bool),
            torch.zeros((capacity, 2), dtype=torch.int32))


_FUSED_ROUNDS = (
    ({1: (6, 124), 3: (6, 124), 6: (2, 30)}, ((3, 1), ), 4, [0, 4, 0, 6, 0, 0, 1, 0]),
    ({0: (1, 511)}, (), 1, [1, 0, 0, 0, 0, 0, 0, 0]),
    ({2: (3, 250), 5: (6, 600), 6: (5, 2), 7: (2, 9)}, (), 4, [0, 0, 2, 0, 0, 6, 3, 0]),
    ({4: (6, 255), 5: (2, 1)}, (), 2, [0, 0, 0, 0, 9, 1, 0, 0]),
    ({lane: (6, 254 if lane == 0 else 40 * lane) for lane in range(8)}, (), 8, [6, 5, 4, 3, 2, 1, 0, 6]),
)


def test_fused_transaction_restore_is_one_graph_across_buckets_and_counts(monkeypatch):
    capacity = 8
    _patch_transaction(monkeypatch, lambda control: control[:, 0])
    transaction = draft_batch.PackedDraftTransaction(
        SimpleNamespace(), SimpleNamespace(capacity=capacity), batch_capacity=capacity, compile_graph=False)
    program, reference = _program(capacity, 21), _program(capacity, 21)
    layout = DSparkStableTargetLayout(program, capacity)
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    torch._dynamo.reset()
    compiled = torch.compile(transaction._forward, backend=backend, fullgraph=True, dynamic=False)
    with torch._dynamo.config.patch(error_on_recompile=True):
        for lanes, aliases, bucket, counts in _FUSED_ROUNDS:
            metadata, queries, pages = _inputs(capacity, lanes, shared_pages=aliases)
            batch, source_lanes = _batch(metadata, queries, pages, bucket)
            stable = layout.view(bucket, padded=True)
            stable.capture(batch, source_lanes)
            legacy = DSparkTargetSnapshot.capture(reference, batch)
            legacy.prepare_device_restore()
            for target in (program, reference):
                _write_target_rows(target, batch, 61)
                _touch_outside(target)
            active = [lane in lanes for lane in range(capacity)]
            args = _transaction_args(capacity, active, counts)
            values = compiled(*args, None, stable.restore_arguments())
            expected = transaction._forward(*args)
            assert len(values) == 7
            for got, want in zip(values[:6], expected, strict=True):
                assert torch.equal(got, want)
            assert torch.equal(values[0][:, 1], values[1])
            _legacy_device_restore(legacy, values[1], source_lanes)
            _assert_same_state(program, reference)
            stable.release()
    assert len(graphs) == 1


def test_fused_transaction_off_path_is_unchanged(monkeypatch):
    capacity = 4
    _patch_transaction(monkeypatch, lambda control: control[:, 0])
    transaction = draft_batch.PackedDraftTransaction(
        SimpleNamespace(), SimpleNamespace(capacity=capacity), batch_capacity=capacity, compile_graph=False)
    args = _transaction_args(capacity, [True, False, True, True], [3, 0, 6, 1])
    eager = transaction(*args)
    torch._dynamo.reset()
    compiled = torch.compile(transaction._forward, backend="aot_eager", fullgraph=True, dynamic=False)
    for values in (eager, compiled(*args)):
        assert len(values) == 6
        for got, want in zip(values, transaction._forward(*args), strict=True):
            assert torch.equal(got, want)


def test_fused_transaction_restore_with_aot_functionalization(monkeypatch):
    capacity = 8
    _patch_transaction(monkeypatch, lambda control: control[:, 0])
    transaction = draft_batch.PackedDraftTransaction(
        SimpleNamespace(), SimpleNamespace(capacity=capacity), batch_capacity=capacity, compile_graph=False)
    program, reference = _program(capacity, 22), _program(capacity, 22)
    lanes, aliases, bucket, counts = _FUSED_ROUNDS[0]
    metadata, queries, pages = _inputs(capacity, lanes, shared_pages=aliases)
    batch, source_lanes = _batch(metadata, queries, pages, bucket)
    stable = DSparkStableTargetLayout(program, capacity).view(bucket, padded=True)
    stable.capture(batch, source_lanes)
    legacy = DSparkTargetSnapshot.capture(reference, batch)
    for target in (program, reference):
        _write_target_rows(target, batch, 5)
    torch._dynamo.reset()
    compiled = torch.compile(transaction._forward, backend="aot_eager", fullgraph=True, dynamic=False)
    active = [lane in lanes for lane in range(capacity)]
    values = compiled(*_transaction_args(capacity, active, counts), None, stable.restore_arguments())
    legacy.restore(tuple(counts[lane] if lane >= 0 else 0 for lane in source_lanes))
    assert values[1].tolist() == [count if live else 0 for count, live in zip(counts, active, strict=True)]
    _assert_same_state(program, reference)


def test_fused_transaction_rejects_bucket_shaped_restore_arguments(monkeypatch):
    capacity = 8
    _patch_transaction(monkeypatch, lambda control: control[:, 0])
    transaction = draft_batch.PackedDraftTransaction(
        SimpleNamespace(), SimpleNamespace(capacity=capacity), batch_capacity=capacity, compile_graph=False)
    program = _program(capacity)
    metadata, queries, pages = _inputs(capacity, {1: (6, 10)})
    batch, source_lanes = _batch(metadata, queries, pages, 2)
    stable = DSparkStableTargetLayout(program, capacity).view(2)
    stable.capture(batch, source_lanes)
    with pytest.raises(ValueError, match="full state-capacity layout"):
        transaction(*_transaction_args(capacity, [False, True] + [False] * 6, [0, 1] + [0] * 6),
                    target_restore=stable.restore_arguments())


@pytest.mark.parametrize("name", ["compact_shared_pages", "ring_wrap_row_zero", "full_mixed"])
def test_native_write_variant_matches_and_never_duplicates_rows(monkeypatch, name):
    writes = []

    def write(cache, values, rows):
        live = rows >= 0
        selected = rows[live].to(torch.int64)
        assert selected.unique().numel() == selected.numel()
        writes.append(int(live.sum()))
        cache.index_copy_(0, selected, values[live])
        return torch.zeros(rows.shape, dtype=torch.int32)

    monkeypatch.setattr(batch_attention, "write_state_rows", write)
    capacity, bucket, lanes, metadata, queries, pages = _case(name)
    for counts in _count_patterns(capacity, lanes):
        legacy_program, stable_program = _program(capacity, 9), _program(capacity, 9)
        batch, source_lanes = _batch(metadata, queries, pages, bucket)
        legacy = DSparkTargetSnapshot.capture(legacy_program, batch)
        legacy.prepare_device_restore()
        stable = DSparkStableTargetLayout(stable_program, capacity).view(bucket, padded=True)
        stable.capture(batch, source_lanes)
        for program in (legacy_program, stable_program):
            _write_target_rows(program, batch, 44)
            _touch_outside(program)
        device_counts = torch.tensor(counts, dtype=torch.int64)
        _legacy_device_restore(legacy, device_counts, source_lanes)
        restore_stable_target_rows(device_counts, *stable.restore_arguments(), native_writes=True)
        _assert_same_state(legacy_program, stable_program)
    assert writes and max(writes) > 0


class _Ring:
    """Minimal verify ring: one in-flight ticket, host readback of the record."""

    batch_capacity = None

    def __init__(self, capacity, *, fail_stage=False):
        self.batch_capacity = capacity
        self.fail_stage = fail_stage
        self.aborted = 0

    def acquire(self, **kwargs):
        return SimpleNamespace(record=torch.zeros((self.batch_capacity, 16), dtype=torch.int64),
                               host=None, completion=SimpleNamespace(query=lambda: True))

    def stage(self, ticket):
        if self.fail_stage:
            raise RuntimeError("record staging failed")
        ticket.host = ticket.record.clone()

    def consume(self, ticket):
        committed = tuple(int(value) for value in ticket.host[:, 1].tolist())
        return SimpleNamespace(committed=committed, output=tuple(() for _ in committed),
                               draft=tuple(() for _ in committed))

    def release(self, ticket):
        pass

    def abort(self, ticket):
        self.aborted += 1


class _AfterRestoreFailure(draft_batch.PackedDraftTransaction):

    def _forward(self, *args, **kwargs):
        values = super()._forward(*args, **kwargs)
        raise RuntimeError("transaction failed after its in-graph restore")


def _draft_stage(capacity, *, transaction_cls=draft_batch.PackedDraftTransaction, fail_stage=False):
    transaction = transaction_cls(
        SimpleNamespace(), SimpleNamespace(capacity=capacity), batch_capacity=capacity, compile_graph=False)
    return draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(), transaction=transaction, verify_ring=_Ring(capacity, fail_stage=fail_stage),
        batch_capacity=capacity, max_model_len=4096)


_STAGE_ROUNDS = (
    ({1: (6, 124), 3: (6, 124), 6: (2, 30)}, ((3, 1), )),
    ({0: (6, 254)}, ()),
    ({2: (3, 250), 5: (1, 511), 7: (6, 9)}, ()),
)


def _round_counts(lanes, index, capacity):
    counts = [0] * capacity
    for lane, (length, _start) in lanes.items():
        counts[lane] = (lane + index) % (length + 1)
    return counts


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("outcome", ["finish", "fail_before", "fail_after", "fail_stage", "invalid", "host_only"])
def test_fused_round_matches_eager_device_finish(monkeypatch, outcome, packed):
    capacity = 8
    holder = {}
    _patch_transaction(monkeypatch, lambda control: holder["counts"])
    restore_calls = []
    real_restore = draft_batch.restore_stable_target_rows

    def counted_restore(*args, **kwargs):
        restore_calls.append(args[0].clone())
        return real_restore(*args, **kwargs)

    monkeypatch.setattr(draft_batch, "restore_stable_target_rows", counted_restore)
    modes = ("eager", "fused")
    programs = {mode: _program(capacity, 31) for mode in modes}
    owners = {mode: _stages(monkeypatch, programs[mode], capacity, owner_layout=True, fused=mode == "fused")
              for mode in modes}
    transaction_cls = _AfterRestoreFailure if outcome == "fail_after" else draft_batch.PackedDraftTransaction
    stages = {mode: _draft_stage(capacity, transaction_cls=transaction_cls, fail_stage=outcome == "fail_stage")
              for mode in modes}
    # The fused mode may also take its round inputs from one staged upload.
    stages["fused"].configure_packed_inputs(packed)
    for index, (lanes, aliases) in enumerate(_STAGE_ROUNDS):
        metadata, queries, pages = _inputs(capacity, lanes, shared_pages=aliases)
        counts = _round_counts(lanes, index, capacity)
        if outcome == "invalid":
            counts[sorted(lanes)[-1]] = 7
        before = {mode: [buffer.clone() for buffer in programs[mode].buffers()] for mode in modes}
        for mode in modes:
            owner, stage = owners[mode], stages[mode]
            target = owner.forward_batch(metadata, queries, None, None, pages)
            payload = target.payload
            assert ("fused_target_restore" in payload) is (mode == "fused")
            assert (payload["device_finish_callback"] is None) is (mode == "fused")
            assert payload["ticket"].snapshot.padded is (mode == "fused")
            holder["counts"] = torch.tensor(counts, dtype=torch.int64)
            if outcome == "fail_before":
                holder["counts"] = None  # the verify stub raises before any restore
            if outcome == "host_only":
                # A draft path that never ran the fused restore: the host
                # finish restores, exactly like the non-device fallback.
                owner.finish(target, counts)
                continue
            calls_before = len(restore_calls)
            try:
                result = stage.verify_and_propose(metadata, target)
            except (RuntimeError, AttributeError, TypeError):
                assert outcome in ("fail_before", "fail_after", "fail_stage")
                assert stage.verify_ring.aborted == index + 1
                ran_in_graph = len(restore_calls) - calls_before
                # Only the fused transaction restores in-graph, and only once
                # verify produced counts.
                assert ran_in_graph == int(mode == "fused" and outcome != "fail_before")
                if outcome == "fail_after":
                    # Restored on device, unknown to the host: abort must fix it.
                    assert not payload["device_finished"]
                if outcome == "fail_stage":
                    # Both modes finished on device before record staging.
                    assert payload["device_finished"]
                owner.abort(target)
                assert payload["finished"]
                continue
            assert outcome in ("finish", "invalid")
            assert payload["device_finished"]
            assert len(restore_calls) - calls_before == int(mode == "fused")
            assert result.accepted.tolist() == [count if lane in lanes else 0 for lane, count in enumerate(counts)]
            if outcome == "invalid":
                with pytest.raises(RuntimeError, match="outside 0..6"):
                    owner.finish(target, result.accepted)
                owner.abort(target)
            else:
                owner.finish(target, result.accepted)
            assert payload["finished"]
        _assert_same_state(programs["eager"], programs["fused"])
        if outcome in ("fail_before", "fail_after", "fail_stage", "invalid"):
            # Zero-count abort returns every target write to its capture value.
            for mode in modes:
                for buffer, saved in zip(programs[mode].buffers(), before[mode], strict=True):
                    assert torch.equal(buffer, saved)
    for mode in modes:
        assert owners[mode].stable_target_layout.owner is None
    assert all(key[1] for key in owners["fused"].stable_target_layout.views)


def test_fused_restore_flag_validation_and_runner_forwarding(monkeypatch):
    owner = object.__new__(DSparkPackedTargetStages)
    owner.observed_query_keys = set()
    owner.device_finish = True
    owner.device_owner_layout = False
    with pytest.raises(TypeError, match="boolean"):
        owner.configure_fused_target_restore(1)
    with pytest.raises(ValueError, match="requires dsv41_dspark_device_owner_layout"):
        owner.configure_fused_target_restore(True)
    owner.configure_device_owner_layout(True)
    owner.configure_fused_target_restore(True)
    assert owner.fused_target_restore is True
    with pytest.raises(ValueError, match="requires dsv41_dspark_device_owner_layout"):
        owner.configure_device_owner_layout(False)
    owner.observed_query_keys = {(8, 48)}
    with pytest.raises(RuntimeError, match="before warmup"):
        owner.configure_fused_target_restore(False)

    from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
    from vllm_gaudi.v1.worker import deepseek_v41_runner
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner

    for additional, expected in (({}, False), ({"dsv41_dspark_fused_target_restore": True}, True)):
        calls = []
        target = SimpleNamespace(configure_device_owner_layout=lambda enabled: calls.append(("layout", enabled)),
                                 configure_fused_target_restore=lambda enabled: calls.append(("fused", enabled)))
        state = SimpleNamespace(capacity=4, slots=RequestSlots(4), pending=None, single_owner=None,
                                prefill_owner=None)
        runner = object.__new__(V41ModelRunner)
        runner.vllm_config = SimpleNamespace(additional_config=additional)
        runner.model = SimpleNamespace(batch_state=state, tp_rank=0, pp_rank=0,
                                       initialize_dspark_batch=lambda capacity: (target, SimpleNamespace()))
        monkeypatch.setattr(deepseek_v41_runner, "DSparkBatchRunner",
                            lambda target_stage, draft_stage, **kwargs: SimpleNamespace(**kwargs))
        monkeypatch.setattr(deepseek_v41_runner, "DSparkBatchTrace", lambda **kwargs: SimpleNamespace(**kwargs))
        monkeypatch.setattr(deepseek_v41_runner.dist, "is_initialized", lambda: False)
        runner._initialize_dspark_batch_runner(4)
        assert calls == [("layout", False), ("fused", expected)]


def test_draft_stage_off_path_keeps_transaction_call_form(monkeypatch):
    capacity = 4
    calls = []

    class Transaction:
        batch_capacity = capacity

        def __call__(self, *args, **kwargs):
            calls.append(sorted(kwargs))
            committed = torch.where(args[6], 2, 0).to(torch.int64)
            record = torch.zeros((capacity, 16), dtype=torch.int64)
            record[:, 1] = committed
            return record, committed

    stage = draft_batch.DSparkBatchDraftStage(SimpleNamespace(), transaction=Transaction(),
                                              verify_ring=_Ring(capacity), batch_capacity=capacity,
                                              max_model_len=4096)
    metadata, queries, pages = _inputs(capacity, {1: (6, 10), 2: (3, 20)})
    device_calls = []
    target = SimpleNamespace(token_ids=torch.zeros(capacity * 6, dtype=torch.int64), payload={
        "aux": torch.zeros((capacity * 6, 3)), "device_finish_callback": lambda t, c: device_calls.append(c)})
    result = stage.verify_and_propose(metadata, target)
    assert calls == [[]]
    assert len(device_calls) == 1 and device_calls[0].tolist() == [0, 2, 2, 0]
    assert result.accepted.tolist() == [0, 2, 2, 0]
    target.payload["fused_target_restore"] = ("not", "a", "graph")
    stage.transaction = None
    stage.draft_round = lambda *args, **kwargs: None  # eager verify path: no in-graph restore exists
    with pytest.raises(RuntimeError, match="requires the packed draft transaction"):
        stage.verify_and_propose(metadata, target)


# ---------------------------------------------------------------------------
# Draft round inputs staged as one fixed vector (dsv41_dspark_draft_packed_inputs).
# ---------------------------------------------------------------------------


def _recording_stage(monkeypatch, capacity, *, packed, buckets=(1, 2, 4)):
    # Transactions are created lazily from the module global, so each stage
    # records into its own batch-state object rather than a closure.
    calls = []

    class Recorder:

        def __init__(self, draft, batch_state, *, batch_capacity, execution_capacity=None, **kwargs):
            self.batch_capacity = batch_capacity
            self.execution_capacity = execution_capacity or batch_capacity
            self.calls = batch_state.calls
            self.compiled = None

        def __call__(self, *args, **kwargs):
            self.calls.append((self.execution_capacity, args, kwargs))
            committed = torch.where(args[6], 1, 0).to(torch.int64)
            record = torch.zeros((capacity, 16), dtype=torch.int64)
            record[:, 1] = committed
            return record, committed

    monkeypatch.setattr(draft_batch, "PackedDraftTransaction", Recorder)
    stage = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(), batch_state=SimpleNamespace(capacity=capacity, calls=calls), verify_ring=_Ring(capacity),
        batch_capacity=capacity, max_model_len=4096)
    stage.configure_execution_buckets(True, buckets=list(buckets))
    stage.configure_packed_inputs(packed)
    return stage, calls


def _draft_target(capacity, seed):
    generator = torch.Generator().manual_seed(seed)
    return SimpleNamespace(
        token_ids=torch.randint(0, 1000, (capacity * 6, ), generator=generator, dtype=torch.int64),
        payload={"aux": torch.randn((capacity * 6, 3), generator=generator)})


@pytest.mark.parametrize(("capacity", "rounds"), [
    (8, ({lane: (6, 10 * lane) for lane in range(8)}, {1: (6, 124), 3: (2, 7), 6: (1, 30)}, {5: (4, 511)},
         {0: (3, 2), 7: (6, 600)})),
    (11, ({10: (6, 255)}, {lane: (1 + lane % 6, 40 + lane) for lane in range(0, 11, 2)}, {2: (5, 3), 9: (1, 1)})),
])
def test_packed_draft_inputs_match_individual_uploads_bit_for_bit(monkeypatch, capacity, rounds):
    reference, reference_calls = _recording_stage(monkeypatch, capacity, packed=False)
    packed, packed_calls = _recording_stage(monkeypatch, capacity, packed=True)
    for index, lanes in enumerate(rounds):
        metadata, _queries, _pages = _inputs(capacity, lanes)
        target = _draft_target(capacity, index)
        expected = reference.verify_and_propose(metadata, target)
        observed = packed.verify_and_propose(metadata, target)
        assert torch.equal(expected.accepted, observed.accepted)
        (bucket, args, kwargs), (packed_bucket, packed_args, packed_kwargs) = reference_calls[-1], packed_calls[-1]
        assert bucket == packed_bucket
        assert sorted(kwargs) == sorted(packed_kwargs)
        for want, got in zip((*args, *kwargs.values()), (*packed_args, *packed_kwargs.values()), strict=True):
            assert got.dtype == want.dtype and got.shape == want.shape and got.device == want.device
            assert torch.equal(got, want)
        staging = packed.input_staging[0 if bucket == capacity else bucket]
        wide, narrow, active = (buffer.untyped_storage().data_ptr() for buffer in staging.device_buffers)
        # control, proposals and lanes view the staged I64 copy, slots and
        # sidecar the I32 copy, and activity is the staged bool vector.
        for value in (packed_args[4], packed_args[5], *packed_kwargs.values()):
            assert value.untyped_storage().data_ptr() == wide
        for value in (packed_args[3], packed_args[7]):
            assert value.untyped_storage().data_ptr() == narrow
        assert packed_args[6].untyped_storage().data_ptr() == active
        assert not any(host.is_pinned() for host in staging.hosts)

    def pointers():
        return {key: tuple(buffer.data_ptr() for buffer in staging.device_buffers)
                for key, staging in packed.input_staging.items()}

    before = pointers()
    metadata, _queries, _pages = _inputs(capacity, rounds[0])
    packed.verify_and_propose(metadata, _draft_target(capacity, 99))
    assert pointers() == before


def test_staging_layout_and_validation():
    staging = draft_batch.DraftRoundInputStaging(4, 2, "cpu")
    assert staging.size == 4 * 17 + 2
    assert [(buffer.dtype, buffer.numel()) for buffer in staging.device_buffers] == [
        (torch.int64, 4 * 13 + 2), (torch.int32, 4 * 3), (torch.bool, 4)]
    control = torch.arange(28, dtype=torch.int64).reshape(4, 7) - 3
    proposals = torch.arange(20, dtype=torch.int64).reshape(4, 5)
    sidecar = torch.tensor([[-2**31, 2**31 - 1], [-1, 0], [5, -6], [0, 0]], dtype=torch.int32)
    values = staging.upload(control, proposals, (1, 2, 3, 4), (0, -1, 2, 3), (True, False, True, True), sidecar,
                            (3, 0))
    expected = (control, proposals, torch.tensor([1, 2, 3, 4]), torch.tensor([0, -1, 2, 3], dtype=torch.int32),
                torch.tensor([True, False, True, True]), sidecar, torch.tensor([3, 0]))
    for got, want in zip(values, expected, strict=True):
        assert got.dtype == want.dtype and torch.equal(got, want)
    with pytest.raises(ValueError, match="lanes"):
        staging.upload(control, proposals, (1, 2, 3, 4), (0, 1, 2, 3), (True, ) * 4, sidecar)
    with pytest.raises(ValueError, match="fixed"):
        staging.upload(control[:3], proposals, (1, 2, 3, 4), (0, 1, 2, 3), (True, ) * 4, sidecar, (0, 1))


def test_packed_inputs_flag_validation_and_runner_forwarding(monkeypatch):
    stage, _calls = _recording_stage(monkeypatch, 4, packed=False)
    with pytest.raises(TypeError, match="boolean"):
        stage.configure_packed_inputs("yes")
    stage.transactions[4] = SimpleNamespace(compiled=object())
    with pytest.raises(RuntimeError, match="before warmup"):
        stage.configure_packed_inputs(True)

    from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
    from vllm_gaudi.v1.worker import deepseek_v41_runner
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner

    monkeypatch.setattr(deepseek_v41_runner, "DSparkBatchRunner",
                        lambda target_stage, draft_stage, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(deepseek_v41_runner, "DSparkBatchTrace", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(deepseek_v41_runner.dist, "is_initialized", lambda: False)

    def initialize(additional, draft):
        state = SimpleNamespace(capacity=4, slots=RequestSlots(4), pending=None, single_owner=None,
                                prefill_owner=None)
        runner = object.__new__(V41ModelRunner)
        runner.vllm_config = SimpleNamespace(additional_config=additional)
        runner.model = SimpleNamespace(batch_state=state, tp_rank=0, pp_rank=0,
                                       initialize_dspark_batch=lambda capacity: (SimpleNamespace(), draft))
        runner._initialize_dspark_batch_runner(4)

    for additional, expected in (({}, [False]), ({"dsv41_dspark_draft_packed_inputs": True}, [True])):
        calls = []
        initialize(additional, SimpleNamespace(configure_packed_inputs=calls.append))
        assert calls == expected
    with pytest.raises(TypeError, match="dsv41_dspark_draft_packed_inputs must be a boolean"):
        initialize({"dsv41_dspark_draft_packed_inputs": 1}, SimpleNamespace(configure_packed_inputs=print))
    with pytest.raises(RuntimeError, match="requires the packed DSpark draft stage"):
        initialize({"dsv41_dspark_draft_packed_inputs": True}, SimpleNamespace())
