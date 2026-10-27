# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for startup warmup of the fixed-capacity DSpark graphs."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_packed_mla_tile_plan
from vllm_gaudi.v1.worker.deepseek_v41_dspark_batch_runner import (
    DSparkBatchRunner,
    DSparkDraftBatch,
    DSparkTargetBatch,
    warmup_scenarios,
)
from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState, V41ModelRunner


class _Layer:
    def __init__(self):
        self.cleared = []

    def clear_slot(self, slot):
        self.cleared.append(slot)


class _TargetState:
    def __init__(self, slots):
        self.slots = slots
        self.layers = {0: _Layer(), 1: _Layer()}
        self.pages = torch.zeros((slots.capacity, 2), dtype=torch.int32)
        self.page_host = torch.zeros_like(self.pages)
        self.page_versions = {}
        self.pending = None
        self.single_owner = None
        self.prefill_owner = None
        self.restore_count = 0

    def acquire(self, request_id):
        return self.slots.acquire(request_id)

    def publish_pages(self, slot, block_ids, pool_blocks):
        assert tuple(block_ids) == (slot.index + 1, )
        assert pool_blocks > block_ids[0]
        self.pages[slot.index, 0] = block_ids[0]
        self.page_host[slot.index, 0] = block_ids[0]
        self.page_versions[slot.index] = (slot.generation, tuple(block_ids))

    def restore_single_bindings(self):
        self.restore_count += 1


class _Engram:
    def __init__(self):
        self.activated, self.released = [], []

    def activate(self, request_id, *, reset=False):
        self.activated.append((request_id, reset))

    def release_request(self, request_id):
        self.released.append(request_id)


class _TargetStage:
    device = torch.device("cpu")

    def __init__(self, state, engram, capacity):
        self.batch_state, self.engram_host = state, engram
        self.batch_capacity = capacity
        self.stage_key = tuple(range(capacity))
        self.stage = SimpleNamespace(compiled=object())
        self.stages = {self.stage_key: self.stage}
        self.orders, self.activity, self.page_rows = [], [], []
        self.finished = self.aborted = 0

    def enter_batched(self):
        pass

    def forward_batch(self, metadata, input_ids, positions, padding_mask, pages):
        del positions, padding_mask
        self.orders.append(tuple(i for i, request_id in enumerate(metadata.request_ids) if request_id is not None))
        self.activity.append(tuple(metadata.request_active.tolist()))
        self.page_rows.append(tuple(tuple(row) for row in pages.tolist()))
        assert tuple(input_ids.shape) == (self.batch_capacity, 6)
        assert tuple(pages.shape) == (self.batch_capacity, 64)
        return DSparkTargetBatch(token_ids=torch.zeros((self.batch_capacity, 6), dtype=torch.int32))

    def finish(self, target, accepted):
        del target
        assert len(tuple(accepted)) == self.batch_capacity
        self.finished += 1

    def abort(self, target):
        del target
        self.aborted += 1


class _DraftState:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cleared = []
        self.rings = [torch.zeros((capacity * 256, 1), dtype=torch.float32) for _ in range(3)]

    def clear_slot(self, slot):
        self.cleared.append(slot)
        for ring in self.rings:
            ring[slot * 256:(slot + 1) * 256].zero_()


class _DraftStage:
    def __init__(self, *, capacity, fail=False):
        self.batch_capacity = capacity
        self.batch_state = _DraftState(capacity)
        self.verify_ring = SimpleNamespace(generation=0)
        self.transaction = SimpleNamespace(compiled=object(), batch_capacity=capacity)
        self.fail, self.calls = fail, 0

    def verify_and_propose(self, metadata, target):
        del target
        active = tuple(metadata.request_active.tolist())
        starts = metadata.target_query_start_loc.tolist()
        counts = tuple(right - left for left, right in zip(starts[:-1], starts[1:], strict=True))
        generation = int(metadata.round_generation.item())
        assert generation > self.verify_ring.generation
        self.verify_ring.generation = generation
        self.calls += 1
        if self.fail:
            raise RuntimeError("draft warmup failure")
        accepted = [1 if enabled else 0 for enabled in active]
        outputs = [[1] if enabled else [] for enabled in active]
        drafts = [[11, 12, 13, 14, 15] if enabled and count == 6 else [] for enabled, count in zip(active, counts)]
        return DSparkDraftBatch(
            accepted=torch.tensor(accepted, dtype=torch.int64),
            output_token_ids=outputs,
            draft_token_ids=drafts,
            completion_event=SimpleNamespace(query=lambda: True),
        )


class _BucketedDraftStage(_DraftStage):
    def __init__(self, *, capacity, buckets, fail=False):
        super().__init__(capacity=capacity, fail=fail)
        self.execution_buckets = tuple(buckets)
        self.transactions = {
            bucket: SimpleNamespace(
                compiled=None,
                batch_capacity=capacity,
                execution_capacity=bucket,
            )
            for bucket in buckets
        }
        self.transaction = self.transactions[capacity]
        self.last_execution_plan = None
        self.graph_key = (capacity, capacity)
        self.executed_buckets = []

    def verify_and_propose(self, metadata, target):
        active_lanes = tuple(lane for lane, active in enumerate(metadata.request_active.tolist()) if active)
        execution_capacity = next(bucket for bucket in self.execution_buckets if bucket >= len(active_lanes))
        result = super().verify_and_propose(metadata, target)
        inactive_lanes = tuple(lane for lane in range(self.batch_capacity) if lane not in active_lanes)
        padding = execution_capacity - len(active_lanes)
        source_lanes = (tuple(range(self.batch_capacity)) if execution_capacity == self.batch_capacity else
                        active_lanes + inactive_lanes[:padding])
        self.last_execution_plan = SimpleNamespace(
            state_capacity=self.batch_capacity,
            execution_capacity=execution_capacity,
            source_lanes=source_lanes,
            logical_slots=tuple(int(metadata.request_slots[lane]) for lane in source_lanes),
        )
        self.graph_key = (self.batch_capacity, execution_capacity)
        transaction = self.transactions[execution_capacity]
        if transaction.compiled is None:
            transaction.compiled = object()
        self.executed_buckets.append(execution_capacity)
        return result


def _request_factory(req_id, page, token):
    params = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    assert params.temperature == 0 and params.max_tokens == 64 and params.ignore_eos
    return RequestState(
        req_id=req_id,
        prompt=[],
        mm_features=[],
        sampling_params=params,
        block_ids=((page, ), ),
        num_computed_tokens=0,
        output=[token],
    )


def _batch_runner(capacity, *, fail=False):
    slots = RequestSlots(capacity)
    state = _TargetState(slots)
    engram = _Engram()
    target = _TargetStage(state, engram, capacity)
    draft = _DraftStage(capacity=capacity, fail=fail)
    runner = DSparkBatchRunner(
        target,
        draft,
        slots=slots,
        batch_capacity=capacity,
        trace=SimpleNamespace(write=lambda payload: None),
    )
    return runner, target, draft, state, engram


def _bucketed_batch_runner(capacity, *, target_buckets, draft_buckets):
    slots = RequestSlots(capacity)
    state = _TargetState(slots)
    engram = _Engram()
    target = _TargetStage(state, engram, capacity)
    target.execution_buckets = tuple(target_buckets)
    original = target.forward_batch

    def forward(metadata, *args):
        active = int(metadata.request_active.sum())
        bucket = next(value for value in target.execution_buckets if value >= active)
        target.stages.setdefault(tuple(range(bucket)), SimpleNamespace(compiled=object()))
        return original(metadata, *args)

    target.forward_batch = forward
    draft = _BucketedDraftStage(capacity=capacity, buckets=draft_buckets)
    runner = DSparkBatchRunner(
        target,
        draft,
        slots=slots,
        batch_capacity=capacity,
        trace=SimpleNamespace(write=lambda payload: None),
    )
    return runner, target, draft, state, engram


def _scenario_tile_bucket(capacity, lengths_by_lane):
    active_lanes = tuple(sorted(lengths_by_lane))
    plan = batch_packed_mla_tile_plan(active_lanes, total_rows=capacity * 6)
    return plan.numel()


@pytest.mark.parametrize("capacity", (1, 2, 4, 8, 16, 24, 31, 64))
def test_warmup_scenarios_keep_legacy_patterns_for_non_n32(capacity):
    scenarios = warmup_scenarios(capacity)
    gap_lane = capacity - 1

    assert tuple(name for name, _lengths in scenarios) == ("full_mixed_c1_c6", "partial_gap", "reuse_c1")
    assert scenarios[0][1] == {lane: (6 if lane % 2 == 0 else 1) for lane in range(capacity)}
    assert scenarios[1][1] == {gap_lane: 6}
    assert scenarios[2][1] == {gap_lane: 1}


def test_n32_warmup_scenarios_cover_each_tile_bucket_with_bounded_rounds():
    scenarios = warmup_scenarios(32)
    by_name = dict(scenarios)

    assert len(scenarios) == 7
    assert len(by_name) == len(scenarios)
    assert tuple(by_name) == (
        "full_mixed_c1_c6",
        "partial_gap",
        "reuse_c1",
        "tile_bucket_2",
        "tile_bucket_4",
        "tile_bucket_8",
        "tile_bucket_16",
    )
    assert by_name["partial_gap"] == {31: 6}
    assert by_name["reuse_c1"] == {31: 1}
    assert by_name["tile_bucket_2"] == {1: 6}
    assert by_name["tile_bucket_4"] == {1: 6, 5: 6}
    assert by_name["tile_bucket_8"] == {1: 6, 2: 6, 5: 6}
    assert by_name["tile_bucket_16"] == {0: 6, 2: 6, 5: 6, 9: 6, 13: 6}

    buckets = {name: _scenario_tile_bucket(32, lengths) for name, lengths in scenarios}
    assert set(buckets.values()) == {1, 2, 4, 8, 16, 24}
    assert [buckets[name] for name in ("tile_bucket_2", "tile_bucket_4", "tile_bucket_8", "tile_bucket_16")] == [
        2,
        4,
        8,
        16,
    ]
    assert [len(by_name[name]) for name in ("tile_bucket_2", "tile_bucket_4", "tile_bucket_8", "tile_bucket_16")] == [
        1,
        2,
        3,
        5,
    ]

    # The scenario matrix is host-only and rank-independent.
    assert all(warmup_scenarios(32) == scenarios for _ in range(8))


def test_n32_warmup_generation_continuity_ignores_unrelated_slot_histories():
    capacity = 32
    scenarios = warmup_scenarios(capacity)
    runner, _target, _draft, state, _engram = _batch_runner(capacity)
    # Make every physical slot carry a different prior generation.  The
    # original check compared the last active lane of every scenario and
    # would fail when the first three (tail lane 31) are followed by lane 1.
    runner.slots.generations = [lane * 7 for lane in range(capacity)]
    validated, validate = [], runner.slots.validate_batch_identities
    runner.slots.validate_batch_identities = lambda slots, generations, active: (
        validated.append((slots, generations, active)) or validate(slots, generations, active))

    runner.warmup(pool_blocks=capacity + 1, max_model_len=8192, request_factory=_request_factory)

    assert [validated[index][1][-1] for index in range(3)] == [218, 219, 220]
    assert validated[3][0] == (1, )
    assert validated[3][1] == (11, )
    assert runner.round_generation == len(scenarios)
    assert runner.slots.owners == {} and state.page_versions == {}


def test_n32_warmup_covers_tile_scenarios_and_cleans_all_state():
    capacity = 32
    scenarios = warmup_scenarios(capacity)
    runner, target, draft, state, engram = _batch_runner(capacity)
    target.execution_buckets = (capacity, )
    draft.execution_buckets = (1, capacity)
    validated, validate = [], runner.slots.validate_batch_identities
    runner.slots.validate_batch_identities = lambda slots, generations, active: (
        validated.append((slots, generations, active)) or validate(slots, generations, active))

    runner.warmup(pool_blocks=capacity + 1, max_model_len=8192, request_factory=_request_factory)

    active_lanes = [tuple(sorted(lengths_by_lane)) for _name, lengths_by_lane in scenarios]
    activity = [
        tuple(lane in lengths_by_lane for lane in range(capacity))
        for _name, lengths_by_lane in scenarios
    ]
    assert target.orders == active_lanes
    assert target.activity == activity
    assert target.finished == draft.calls == len(scenarios)
    assert target.aborted == 0
    assert (runner.round_generation, draft.verify_ring.generation) == (len(scenarios), len(scenarios))

    for round_index, lanes in enumerate(active_lanes):
        observed_slots, observed_generations, observed_active = validated[round_index]
        assert observed_slots == lanes
        assert observed_generations == (round_index + 1, ) * len(lanes)
        assert observed_active == (True, ) * len(lanes)

    page_columns = (8192 + 127) // 128
    expected_page_rows = []
    for _name, lengths_by_lane in scenarios:
        dummy_page = min(lengths_by_lane) + 1
        expected_page_rows.append(tuple(
            (((lane + 1) if lane in lengths_by_lane else dummy_page), ) * page_columns
            for lane in range(capacity)
        ))
    assert target.page_rows == expected_page_rows
    assert state.pages.count_nonzero() == state.page_host.count_nonzero() == 0
    assert state.page_versions == {} and state.restore_count == 1
    assert state.pending is state.single_owner is state.prefill_owner is None
    assert sorted(draft.batch_state.cleared) == sorted(list(range(capacity)) * len(scenarios))
    assert all(sorted(layer.cleared) == sorted(list(range(capacity)) * len(scenarios))
               for layer in state.layers.values())
    assert runner.slots.owners == {} and runner.pending is False
    assert runner.single_draft_owner is None and runner.draft_bank_generations == {}
    assert len(engram.activated) == len(engram.released) == capacity * len(scenarios)
    assert {request_id for request_id, reset in engram.activated if reset} == set(engram.released)


@pytest.mark.parametrize("capacity", (1, 2, 4))
def test_batch_warmup_covers_bounded_lanes_and_preserves_generation_continuity(capacity):
    runner, target, draft, state, engram = _batch_runner(capacity)
    validated, validate = [], runner.slots.validate_batch_identities
    runner.slots.validate_batch_identities = lambda slots, generations, active: (
        validated.append((slots, generations, active)) or validate(slots, generations, active))

    runner.warmup(pool_blocks=capacity + 1, max_model_len=8192, request_factory=_request_factory)

    active_all = tuple(range(capacity))
    gap_lane = capacity - 1
    expected_validations = [
        (active_all, (1, ) * capacity, (True, ) * capacity),
        ((gap_lane, ), (2, ), (True, )),
        ((gap_lane, ), (3, ), (True, )),
    ]
    assert validated == expected_validations
    assert target.orders == [active_all, (gap_lane, ), (gap_lane, )]
    assert target.activity == [
        (True, ) * capacity,
        tuple(lane == gap_lane for lane in range(capacity)),
        tuple(lane == gap_lane for lane in range(capacity)),
    ]
    page_columns = (8192 + 127) // 128
    assert target.page_rows[0] == tuple((lane + 1, ) * page_columns for lane in range(capacity))
    assert target.page_rows[1] == tuple((capacity, ) * page_columns for _ in range(capacity))
    assert target.page_rows[2] == tuple((capacity, ) * page_columns for _ in range(capacity))
    assert target.finished == draft.calls == 3
    assert target.aborted == 0
    assert (runner.round_generation, draft.verify_ring.generation) == (3, 3)
    assert state.pages.count_nonzero() == state.page_host.count_nonzero() == 0
    assert state.page_versions == {} and state.restore_count == 1
    assert sorted(draft.batch_state.cleared) == sorted(list(range(capacity)) * 3)
    assert all(sorted(layer.cleared) == sorted(list(range(capacity)) * 3) for layer in state.layers.values())
    assert runner.slots.owners == {} and runner.pending is False
    assert len(engram.activated) == len(engram.released) == capacity * 3
    assert {request_id for request_id, reset in engram.activated if reset} == set(engram.released)

    owners = {}
    requests = {}
    counts = {}
    proposals = {}
    for lane in range(capacity):
        req_id = f"live-{lane}"
        owner = runner.slots.acquire(req_id)
        assert owner.index == lane
        owners[req_id] = owner
        requests[req_id] = _request_factory(req_id, lane + 1, 10 + lane)
        counts[req_id] = 1
    scheduled = SimpleNamespace(num_scheduled_tokens=counts, scheduled_spec_decode_tokens=proposals)
    live = runner.build_plan(scheduled, requests, owners, max_model_len=8192)
    assert live is not None
    assert live.capacity == capacity and live.active_lanes == active_all
    assert live.round_generation == 4 and live.round_generation > draft.verify_ring.generation


@pytest.mark.parametrize("capacity", (2, 4, 32))
def test_batch_warmup_cleans_every_owner_when_execution_fails(capacity):
    runner, target, draft, state, engram = _batch_runner(capacity, fail=True)

    with pytest.raises(RuntimeError, match="draft warmup failure"):
        runner.warmup(pool_blocks=capacity + 1, max_model_len=8192, request_factory=_request_factory)

    assert target.finished == 0 and target.aborted == 1 and draft.calls == 1
    assert runner.slots.owners == {} and runner.pending is False
    assert state.pages.count_nonzero() == state.page_host.count_nonzero() == 0
    assert state.restore_count == 1 and len(engram.released) == capacity
    assert sorted(draft.batch_state.cleared) == list(range(capacity))
    assert all(sorted(layer.cleared) == list(range(capacity)) for layer in state.layers.values())


@pytest.mark.parametrize("capacity", (1, 2, 4, 32))
def test_batch_warmup_rejects_the_last_scheduler_page_as_unsafe_dummy(capacity):
    runner, target, draft, state, engram = _batch_runner(capacity)

    with pytest.raises(ValueError, match=f"more than {capacity} non-null scheduler pages"):
        runner.warmup(pool_blocks=capacity, max_model_len=8192, request_factory=_request_factory)

    assert runner.slots.owners == {}
    assert state.pages.count_nonzero() == 0
    assert target.finished == draft.calls == 0
    assert engram.activated == engram.released == []


def _startup_runner(owner, events):
    runner = object.__new__(V41ModelRunner)
    runner.request_batches = runner.prefix_checkpoints = None
    runner.use_dspark = True
    runner.model = SimpleNamespace(
        native=False,
        program=SimpleNamespace(runtime_indexer=False, length=8192),
    )
    runner.model_config = SimpleNamespace(max_model_len=8192)
    runner.vllm_config = SimpleNamespace(additional_config={})
    runner.state = SimpleNamespace(blocks=8, clear=lambda: events.append("state_clear"))
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=False, barrier=lambda: events.append("barrier")))
    runner._dummy_run = lambda *args, **kwargs: events.append("dummy")
    runner._warm_device_round_inputs = lambda: events.append("sampling")
    runner.graphed_buckets = set()
    runner.dspark_batch_runner = owner
    return runner, events


def test_b2_startup_warmup_runs_before_barrier_and_state_clear(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    events = []

    class Owner:
        def warmup(self, **kwargs):
            events.append("b2_warmup")
            request = kwargs["request_factory"]("dummy", 1, 4096)
            assert request.prompt == [] and request.output == [4096]
            assert request.num_computed_tokens == 0 and request.block_ids == ((1, ), )

    runner, startup_events = _startup_runner(Owner(), events)
    runner.warmup_model()
    assert startup_events[-3:] == ["b2_warmup", "barrier", "state_clear"]


def test_b1_startup_without_a_b2_owner_skips_packed_warmup(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    events = []
    runner, events = _startup_runner(None, events)

    runner.warmup_model()

    assert "b2_warmup" not in events
    assert events[-2:] == ["barrier", "state_clear"]


def test_b2_startup_failure_stops_before_barrier_and_state_clear(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    events = []

    class Owner:
        def warmup(self, **kwargs):
            del kwargs
            events.append("b2_warmup")
            raise RuntimeError("warmup failed")

    runner, events = _startup_runner(Owner(), events)
    with pytest.raises(RuntimeError, match="warmup failed"):
        runner.warmup_model()
    assert events[-1] == "b2_warmup"
    assert "barrier" not in events and "state_clear" not in events


@pytest.mark.parametrize(
    ("additional", "expected_target", "expected_draft", "error"),
    (
        (
            {},
            (True, (1, 2, 4, 8, 16, 32, 64)),
            (True, (1, 2, 4, 8, 16, 32, 64)),
            None,
        ),
        (
            {
                "dsv41_dspark_target_warmup_buckets": (1, 2, 4),
                "dsv41_dspark_draft_warmup_buckets": (1, 4),
            },
            (True, (1, 2, 4)),
            (True, (1, 4)),
            None,
        ),
        (
            {
                "dsv41_dspark_active_metadata": False,
                "dsv41_dspark_draft_buckets": True,
                "dsv41_dspark_draft_warmup_buckets": (1, 4),
            },
            (True, (1, 2, 4, 8, 16, 32, 64)),
            (False, (1, 4)),
            None,
        ),
        (
            {
                "dsv41_dspark_active_metadata": False,
                "dsv41_dspark_draft_buckets": "false",
            },
            None,
            None,
            "dsv41_dspark_draft_buckets must be a boolean",
        ),
    ),
)
def test_runner_configures_independent_draft_buckets(monkeypatch, additional, expected_target, expected_draft, error):
    from vllm_gaudi.v1.worker import deepseek_v41_runner

    calls = {}

    class Stage:
        def __init__(self, name):
            self.name = name

        def configure_execution_buckets(self, enabled, *, buckets):
            calls[self.name] = (enabled, buckets)

    target, draft = Stage("target"), Stage("draft")

    class Adapter:
        def __init__(self, target_stage, draft_stage, **kwargs):
            self.target_stage = target_stage
            self.draft_stage = draft_stage
            self.batch_capacity = kwargs["batch_capacity"]

    slots = RequestSlots(4)
    state = SimpleNamespace(
        capacity=4,
        slots=slots,
        pending=None,
        single_owner=None,
        prefill_owner=None,
    )
    runner = object.__new__(V41ModelRunner)
    runner.vllm_config = SimpleNamespace(additional_config=additional)
    runner.model = SimpleNamespace(
        batch_state=state,
        tp_rank=0,
        pp_rank=0,
        initialize_dspark_batch=lambda capacity: (target, draft),
    )
    monkeypatch.setattr(deepseek_v41_runner, "DSparkBatchRunner", Adapter)
    monkeypatch.setattr(
        deepseek_v41_runner,
        "DSparkBatchTrace",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(deepseek_v41_runner.dist, "is_initialized", lambda: False)

    if error is not None:
        with pytest.raises(TypeError, match=error):
            runner._initialize_dspark_batch_runner(4)
        assert "draft" not in calls
        return

    runner._initialize_dspark_batch_runner(4)

    assert calls["target"] == expected_target
    assert calls["draft"] == expected_draft
    assert runner.dspark_batch_runner.target_stage is target
    assert runner.dspark_batch_runner.draft_stage is draft


def test_warmup_covers_independent_target_and_draft_bucket_sets():
    capacity = 8
    runner, target, draft, _state, _engram = _bucketed_batch_runner(
        capacity,
        target_buckets=(1, 2, 8),
        draft_buckets=(1, 4, 8),
    )

    runner.warmup(pool_blocks=capacity + 1, max_model_len=8192, request_factory=_request_factory)

    assert target.orders == [
        tuple(range(capacity)),
        (capacity - 1, ),
        (capacity - 1, ),
        (0, 1),
        (0, 1, 2, 3),
    ]
    # Target stage keys follow target buckets only: the four-lane draft-bucket round executes on target bucket 8,
    # while partial_gap/reuse_c1 cover target bucket 1.
    assert set(target.stages) == {tuple(range(bucket)) for bucket in (1, 2, capacity)}
    assert set(draft.transactions) == {1, 4, 8}
    assert all(transaction.compiled is not None for transaction in draft.transactions.values())
    assert draft.transaction is draft.transactions[capacity]
    assert set(draft.executed_buckets) == {1, 4, 8}
    assert runner.round_generation == draft.verify_ring.generation == len(target.orders)


def test_warmup_rejects_a_mismatched_actual_draft_transaction_bucket():
    capacity = 8
    runner, _target, draft, _state, _engram = _bucketed_batch_runner(
        capacity,
        target_buckets=(1, 8),
        draft_buckets=(1, 4, 8),
    )
    draft.transactions[4].execution_capacity = 2

    with pytest.raises(RuntimeError, match="draft transaction execution capacity changed"):
        runner.warmup(pool_blocks=capacity + 1, max_model_len=8192, request_factory=_request_factory)


@pytest.mark.parametrize("capacity,buckets", [(32, (1, 32)), (11, (1, 2, 4, 8, 11))])
def test_execution_bucket_warmup_covers_only_configured_keys(capacity, buckets):
    runner, target, draft, state, _engram = _batch_runner(capacity)
    target.execution_buckets = buckets
    original = target.forward_batch

    def forward(metadata, *args):
        active = int(metadata.request_active.sum())
        bucket = next(value for value in buckets if value >= active)
        target.stages.setdefault(tuple(range(bucket)), SimpleNamespace(compiled=object()))
        return original(metadata, *args)

    target.forward_batch = forward
    runner.warmup(pool_blocks=capacity + 1, max_model_len=8192, request_factory=_request_factory)
    expected_rounds = 3 + sum(1 < bucket < capacity for bucket in buckets)
    assert len(target.orders) == expected_rounds
    assert runner.round_generation == draft.verify_ring.generation == expected_rounds
    assert set(target.stages) == {tuple(range(bucket)) for bucket in buckets}
    assert not runner.slots.owners
    assert not runner.pending
