# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for startup warmup of the fixed-capacity DSpark graphs."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
from vllm_gaudi.v1.worker.deepseek_v41_dspark_batch_runner import (
    DSparkBatchRunner,
    DSparkDraftBatch,
    DSparkTargetBatch,
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


@pytest.mark.parametrize("capacity", (2, 4))
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


@pytest.mark.parametrize("capacity", (1, 2, 4))
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
