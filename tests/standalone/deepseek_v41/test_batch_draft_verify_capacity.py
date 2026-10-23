# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for verify-ring capacities N=1, 2 and 4."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops import deepseek_v41_batch_draft as draft_batch
from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
from vllm_gaudi.ops.deepseek_v41_batch_verify import (
    DEFAULT_BATCH_CAPACITY,
    MAX_BATCH_REQUESTS,
    BatchVerifyRing,
    batch_binding_sidecar,
    batch_control_metadata,
    pack_record_batch,
    validate_batch_records,
    verify_control_from_target_batch,
)


_CAPACITY_CASES = (
    (1, (True, )),
    (2, (True, False)),
    (4, (True, False, True, False)),
)


def _patch_ring_runtime(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace(
        record=lambda stream: None,
        synchronize=lambda: None,
        query=lambda: True,
    ))
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: None)


def _identities(active, *, slots=None, generations=None):
    if slots is None:
        slots = [index if enabled else -1 for index, enabled in enumerate(active)]
    if generations is None:
        generations = [index + 1 if enabled else 0 for index, enabled in enumerate(active)]
    return tuple(slots), tuple(generations)


def _record(capacity, active, *, round_generation=1, slots=None, generations=None):
    slots, generations = _identities(active, slots=slots, generations=generations)
    target = torch.stack([
        torch.arange(10 + 20 * index, 16 + 20 * index) if enabled else torch.zeros(6, dtype=torch.int64)
        for index, enabled in enumerate(active)
    ])
    proposals = torch.stack([
        target[index, :5] if enabled else torch.full((5, ), -1, dtype=torch.int64)
        for index, enabled in enumerate(active)
    ])
    target_counts = [6 if enabled else 1 for enabled in active]
    proposal_counts = [5 if enabled else 0 for enabled in active]
    metadata = batch_control_metadata(
        round_generation=round_generation,
        request_slots=slots,
        slot_generations=generations,
        target_counts=target_counts,
        proposal_counts=proposal_counts,
        remaining_output_budget=[64 if enabled else 0 for enabled in active],
        target_start_positions=[10 + 20 * index if enabled else 0 for index, enabled in enumerate(active)],
        max_model_len=128,
        need_sample=[1 if enabled else 0 for enabled in active],
        active=active,
    )
    binding_sidecar = batch_binding_sidecar(metadata)
    active_tensor = torch.tensor(active)
    _, output, committed, output_count, _, enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active_tensor, binding_sidecar)
    record = pack_record_batch(
        metadata, committed, output_count, output, proposals, enabled, status, binding_sidecar)
    return record, target, proposals, metadata, slots, generations


@pytest.mark.parametrize(("capacity", "active"), _CAPACITY_CASES)
def test_ring_consumes_one_capacity_sized_batch_readback(monkeypatch, capacity, active):
    _patch_ring_runtime(monkeypatch)
    record, _, _, _, slots, generations = _record(capacity, active)
    ring = BatchVerifyRing("cpu", batch_capacity=capacity)
    ticket = ring.acquire(
        round_generation=1,
        request_slots=slots,
        slot_generations=generations,
        max_committed=[6 if enabled else 0 for enabled in active],
        active=active,
    )
    assert ticket.record.shape == (capacity, 16)
    ticket.record.copy_(record)
    ring.stage(ticket)
    commit = ring.consume(ticket)
    ring.release(ticket)

    assert ring.copy_count == ring.event_record_count == ring.completion_sync_count == 1
    assert commit.committed == tuple(6 if enabled else 0 for enabled in active)
    assert all(len(commit.output[index]) == (6 if enabled else 0) for index, enabled in enumerate(active))
    assert all(len(commit.draft[index]) == (5 if enabled else 0) for index, enabled in enumerate(active))


@pytest.mark.parametrize("inactive_limit", [0, 6])
def test_ring_accepts_zero_or_legacy_inactive_commit_limits(monkeypatch, inactive_limit):
    _patch_ring_runtime(monkeypatch)
    active = (True, False)
    record, _, _, _, slots, generations = _record(2, active)
    ring = BatchVerifyRing("cpu", batch_capacity=2)
    ticket = ring.acquire(
        round_generation=1,
        request_slots=slots,
        slot_generations=generations,
        max_committed=[6, inactive_limit],
        active=active,
    )
    ticket.record.copy_(record)
    ring.stage(ticket)
    commit = ring.consume(ticket)
    ring.release(ticket)
    assert commit.committed == (6, 0)


def test_owner_generation_revocation_cancels_n4_consume_and_reuses_slot(monkeypatch):
    _patch_ring_runtime(monkeypatch)
    owners = RequestSlots(4)
    slots = [owners.acquire(f"request-{index}") for index in range(4)]
    active = (True, True, True, True)
    ring = BatchVerifyRing("cpu", batch_capacity=4, owner_validator=owners.validate_batch_identities)
    first_slots = [slot.index for slot in slots]
    first_generations = [slot.generation for slot in slots]
    record, _, _, _, _, _ = _record(
        4,
        active,
        round_generation=1,
        slots=first_slots,
        generations=first_generations,
    )
    ticket = ring.acquire(
        round_generation=1,
        request_slots=first_slots,
        slot_generations=first_generations,
        max_committed=[6, 6, 6, 6],
        active=active,
    )
    ticket.record.copy_(record)
    ring.stage(ticket)

    owners.release(slots[0])
    reused = owners.acquire("replacement")
    assert reused.index == slots[0].index and reused.generation == slots[0].generation + 1
    with pytest.raises(RuntimeError, match="no longer owns"):
        ring.consume(ticket)
    assert ring.completion_sync_count == 0
    ring.abort(ticket)
    assert ring.completion_sync_count == 1

    second_slots = [reused.index, slots[1].index, slots[2].index, slots[3].index]
    second_generations = [reused.generation, slots[1].generation, slots[2].generation, slots[3].generation]
    record, _, _, _, _, _ = _record(
        4,
        active,
        round_generation=2,
        slots=second_slots,
        generations=second_generations,
    )
    ticket = ring.acquire(
        round_generation=2,
        request_slots=second_slots,
        slot_generations=second_generations,
        max_committed=[6, 6, 6, 6],
        active=active,
    )
    ticket.record.copy_(record)
    ring.stage(ticket)
    assert ring.consume(ticket).committed == (6, 6, 6, 6)
    ring.release(ticket)


def test_stage_capacity_mismatch_fails_before_target_or_ticket(monkeypatch):
    class Ring:
        batch_capacity = 4

        def __init__(self):
            self.acquire_calls = 0

        def acquire(self, **kwargs):
            self.acquire_calls += 1
            raise AssertionError("capacity mismatch acquired a ticket")

    ring = Ring()
    target_calls = []
    stage = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(),
        batch_state=SimpleNamespace(capacity=2),
        verify_ring=ring,
        transaction=lambda *args: None,
        batch_capacity=4,
        max_model_len=128,
        target_ids=lambda target, batch_size: target_calls.append(batch_size) or target.token_ids,
    )
    metadata = SimpleNamespace(
        request_ids=("a", "b", "c", "d"),
        num_draft_tokens=torch.zeros(4, dtype=torch.int32),
        draft_token_ids=torch.empty(0, dtype=torch.int32),
        request_slots=torch.tensor([-1, -1, -1, -1], dtype=torch.int32),
        slot_generations=torch.zeros(4, dtype=torch.int64),
        target_start_positions=torch.zeros(4, dtype=torch.int64),
        remaining_output_budget=torch.zeros(4, dtype=torch.int32),
        target_query_start_loc=torch.zeros(5, dtype=torch.int32),
        round_generation=torch.tensor(1, dtype=torch.int64),
        request_active=torch.zeros(4, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="batch state capacity"):
        stage.verify_and_propose(metadata, SimpleNamespace(token_ids=torch.zeros((4, 6), dtype=torch.int64)))
    assert ring.acquire_calls == 0
    assert target_calls == []


def test_verify_constants_and_capacity_bounds(monkeypatch):
    _patch_ring_runtime(monkeypatch)
    assert DEFAULT_BATCH_CAPACITY == 2
    assert MAX_BATCH_REQUESTS == 8
    assert BatchVerifyRing("cpu").batch_capacity == DEFAULT_BATCH_CAPACITY
    with pytest.raises(ValueError, match="batch capacity"):
        BatchVerifyRing("cpu", batch_capacity=0)
    with pytest.raises(ValueError, match="batch capacity"):
        BatchVerifyRing("cpu", batch_capacity=MAX_BATCH_REQUESTS + 1)


def test_n4_verify_mixes_c1_c6_and_inactive_lanes():
    active = torch.tensor([True, True, False, True])
    slots = torch.tensor([0, 1, -1, 3], dtype=torch.int32)
    generations = torch.tensor([4, 5, 0, 7], dtype=torch.int64)
    metadata = batch_control_metadata(
        round_generation=3,
        request_slots=slots,
        slot_generations=generations,
        target_counts=[1, 6, 1, 6],
        proposal_counts=[0, 5, 0, 5],
        remaining_output_budget=[64, 64, 0, 64],
        target_start_positions=[10, 20, 0, 40],
        max_model_len=128,
        need_sample=[1, 1, 0, 1],
        active=active,
    )
    binding_sidecar = batch_binding_sidecar(metadata)
    target = torch.tensor([
        [10, 0, 0, 0, 0, 0],
        [20, 21, 22, 23, 24, 25],
        [30, 0, 0, 0, 0, 0],
        [40, 41, 42, 43, 44, 45],
    ], dtype=torch.int64)
    proposals = torch.tensor([
        [50, 51, 52, 53, 54],
        [20, 21, 22, 23, 24],
        [-1, -1, -1, -1, -1],
        [40, 41, 42, 43, 44],
    ], dtype=torch.int64)

    _, output, committed, output_count, _, enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active, binding_sidecar)
    record = pack_record_batch(
        metadata, committed, output_count, output, proposals, enabled, status, binding_sidecar)
    commit = validate_batch_records(
        record,
        round_generation=3,
        request_slots=slots,
        slot_generations=generations,
        max_committed=[1, 6, 0, 6],
        active=active,
    )

    assert committed.tolist() == output_count.tolist() == [1, 6, 0, 6]
    assert status.tolist() == [0, 0, 3, 0]
    assert output[2].tolist() == [-1] * 6
    assert commit.committed == (1, 6, 0, 6)
    assert commit.output[2] == commit.draft[2] == ()


@pytest.mark.parametrize(("capacity", "active"), _CAPACITY_CASES)
def test_validation_uses_capacity_shaped_records(capacity, active):
    record, _, _, _, slots, generations = _record(capacity, active)
    commit = validate_batch_records(
        record,
        round_generation=1,
        request_slots=slots,
        slot_generations=generations,
        max_committed=[6 if enabled else 0 for enabled in active],
        active=active,
    )
    assert len(commit.committed) == capacity
    assert tuple(bool(row) for row in commit.output) == tuple(enabled for enabled in active)
