# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for packed DSpark draft and per-request verify commits."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops import deepseek_v41_batch_draft as draft_batch
from vllm_gaudi.ops.deepseek_v41_batch_verify import (
    BATCH_INACTIVE,
    BINDING_SIDECAR_HIGH,
    BINDING_SIDECAR_LOW,
    BatchVerifyRing,
    _validate_binding_sidecar,
    batch_binding_sidecar,
    batch_binding_words,
    batch_control_metadata,
    pack_binding_word,
    pack_record_batch,
    unpack_binding_word,
    validate_batch_records,
    verify_control_from_target_batch,
)
from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
from vllm_gaudi.ops.deepseek_v41_verify import verify_control_from_target


def _metadata(target, proposals, active, *, generation=3):
    batch = target.shape[0]
    proposal_count = (proposals >= 0).sum(1).to(torch.int64)
    target_count = torch.where(proposal_count > 0, proposal_count + 1, torch.ones_like(proposal_count))
    slots = torch.tensor([0, -1][:batch], dtype=torch.int64)
    generations = torch.tensor([4, 0][:batch], dtype=torch.int64)
    if batch == 2 and active[1]:
        slots[1] = 1
        generations[1] = 9
    return batch_control_metadata(
        round_generation=generation,
        request_slots=slots,
        slot_generations=generations,
        target_counts=target_count,
        proposal_counts=proposal_count,
        remaining_output_budget=[64] * batch,
        target_start_positions=[16384, 32768][:batch],
        max_model_len=65536,
        need_sample=[1] * batch,
        active=active,
    )


def _patch_cpu_ring_runtime(monkeypatch):
    syncs = []

    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            syncs.append(1)

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", Event)
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: None)
    return syncs


def _record_for_stage(*, round_generation=1, request_slots=(0, 1), slot_generations=(4, 9)):
    target = torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]], dtype=torch.int64)
    proposals = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int64)
    active = torch.tensor([True, True])
    metadata = batch_control_metadata(
        round_generation=round_generation,
        request_slots=request_slots,
        slot_generations=slot_generations,
        target_counts=[6, 6],
        proposal_counts=[5, 5],
        remaining_output_budget=[64, 64],
        target_start_positions=[10, 20],
        max_model_len=100,
        need_sample=[1, 1],
        active=active,
    )
    binding_sidecar = batch_binding_sidecar(metadata)
    _, output, committed, output_count, _, enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active, binding_sidecar)
    record = pack_record_batch(
        metadata,
        committed,
        output_count,
        output,
        torch.full_like(proposals, 77),
        enabled,
        status,
        binding_sidecar,
    )
    return record


def _stage_metadata(*, round_generation=1, request_slots=(0, 1), slot_generations=(4, 9)):
    from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata

    return DSparkBatchMetadata.build(
        requests=("request-a", "request-b"),
        target_query_lengths=[6, 6],
        draft_token_ids=[[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]],
        request_slots=request_slots,
        slot_generations=slot_generations,
        remaining_output_budget=[64, 64],
        target_start_positions=[10, 20],
        round_generation=round_generation,
    )


def _stage_target():
    return SimpleNamespace(
        token_ids=torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]], dtype=torch.int64),
        states=torch.zeros((2, 6, 8), dtype=torch.bfloat16),
    )


def test_batched_verify_matches_each_serial_greedy_request():
    target = torch.tensor([
        [10, 11, 12, 13, 14, 15],
        [20, 21, 22, 23, 24, 25],
    ])
    proposals = torch.tensor([
        [10, 11, 12, 13, 14],
        [20, 99, -1, -1, -1],
    ])
    active = torch.tensor([True, True])
    metadata = _metadata(target, proposals, active)
    result = verify_control_from_target_batch(target, proposals, metadata, active)
    _, output, committed, count, anchor, enabled, status = result

    for index in range(2):
        count_i = int((proposals[index] >= 0).sum())
        serial_meta = torch.cat((metadata[index, :1], torch.tensor([
            count_i + 1, count_i, 64, metadata[index, 4], 65536, 1
        ])))
        serial = verify_control_from_target(target[index], proposals[index], serial_meta)
        assert output[index].tolist() == serial[1].tolist()
        assert committed[index].item() == serial[2].item()
        assert count[index].item() == serial[3].item()
        assert anchor[index].item() == serial[4].item()
        assert enabled[index].item() == serial[5].item()
        assert status[index].item() == serial[6].item()
    assert output.tolist() == [
        [10, 11, 12, 13, 14, 15],
        [20, 21, -1, -1, -1, -1],
    ]
    assert committed.tolist() == [6, 2]


def test_padded_lane_and_batch_record_validation():
    target = torch.tensor([[10, 11, 12, 13, 14, 15], [30, 31, 32, 33, 34, 35]])
    proposals = torch.tensor([[10, 11, 12, 13, 14], [-1, -1, -1, -1, -1]])
    active = torch.tensor([True, False])
    metadata = _metadata(target, proposals, active)
    target, output, committed, count, anchor, enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active)
    record = pack_record_batch(metadata, committed, count, output, torch.full_like(proposals, 77), enabled, status)
    assert status.tolist() == [0, BATCH_INACTIVE]
    assert record.shape == (2, 16)
    commit = validate_batch_records(
        record,
        round_generation=3,
        request_slots=[0, -1],
        slot_generations=[4, 0],
        max_committed=[6, 6],
        active=active,
    )
    assert commit.committed == (6, 0)
    assert commit.output == ((10, 11, 12, 13, 14, 15), ())
    assert commit.draft == ((77, 77, 77, 77, 77), ())


def test_batch_verify_ring_uses_one_pinned_copy_and_stale_binding_fails(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace(
        record=lambda stream: None,
        synchronize=lambda: None,
    ))
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: None)

    ring = BatchVerifyRing("cpu", batch_capacity=2)
    target = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    proposals = torch.tensor([[1, 2, 3, 4, 5], [7, 99, -1, -1, -1]])
    active = torch.tensor([True, True])
    metadata = _metadata(target, proposals, active, generation=1)
    target, output, committed, count, anchor, enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active)
    ticket = ring.acquire(
        round_generation=1,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        max_committed=[6, 6],
        active=active,
    )
    record = pack_record_batch(
        metadata, committed, count, output, torch.zeros_like(proposals), enabled, status)
    ticket.record.copy_(record)
    ring.stage(ticket)
    assert ring.copy_count == ring.event_record_count == 1
    with pytest.raises(RuntimeError, match="already staged"):
        ring.stage(ticket)
    commit = ring.consume(ticket)
    assert commit.committed == (6, 2)
    assert ring.completion_sync_count == 1
    with pytest.raises(RuntimeError, match="already consumed"):
        ring.consume(ticket)
    ring.release(ticket)

    metadata = _metadata(target, proposals, active, generation=2)
    ticket = ring.acquire(
        round_generation=2,
        request_slots=[0, 1],
        slot_generations=[4, 10],
        max_committed=[6, 6],
        active=active,
    )
    record = pack_record_batch(metadata, committed, count, output, torch.zeros_like(proposals), enabled, status)
    record[1, 0] += 2
    ticket.record.copy_(record)
    ring.stage(ticket)
    with pytest.raises(RuntimeError, match="Invalid DSpark batch commit"):
        ring.consume(ticket)
    ring.close()


def test_native_batch_readback_is_staged_and_synchronized_once(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace(
        record=lambda stream: None,
        synchronize=lambda: None,
    ))
    calls = []
    syncs = []

    class Completion:
        def synchronize(self):
            syncs.append(1)

    target = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0]], dtype=torch.int64)
    proposals = torch.tensor([[1, 2, 3, 4, 5], [-1, -1, -1, -1, -1]], dtype=torch.int64)
    active = torch.tensor([True, False])
    metadata = batch_control_metadata(
        round_generation=4,
        request_slots=[0, -1],
        slot_generations=[7, 0],
        target_counts=[6, 1],
        proposal_counts=[5, 0],
        remaining_output_budget=[64, 1],
        target_start_positions=[10, 0],
        max_model_len=100,
        need_sample=[1, 1],
        active=active,
    )
    _, output, committed, output_count, _, enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active)
    record = pack_record_batch(metadata, committed, output_count, output, torch.zeros_like(proposals), enabled, status)

    def readback(_):
        calls.append(1)
        return record.clone(), Completion()

    ring = BatchVerifyRing("cpu", batch_capacity=2, native_readback=readback)
    ticket = ring.acquire(
        round_generation=4,
        request_slots=[0, -1],
        slot_generations=[7, 0],
        max_committed=[6, 6],
        active=active,
    )
    ticket.record.copy_(record)
    ring.stage(ticket)
    commit = ring.consume(ticket)
    ring.release(ticket)
    assert commit.committed == (6, 0)
    assert calls == [1]
    assert syncs == [1]
    assert ring.copy_count == 1
    assert ring.event_record_count == 0
    assert ring.completion_sync_count == 1


def test_binding_includes_slot_and_input_dtypes_are_strict():
    word = pack_binding_word(7, 9, 3)
    assert unpack_binding_word(word) == (7, 9, 3)
    assert batch_binding_words(7, [3, -1], [9, 0], [True, False]).tolist() == [word, 0]

    with pytest.raises(ValueError, match="CPU integer"):
        batch_binding_words(7, torch.tensor([3.0]), torch.tensor([9]), [True])
    with pytest.raises(ValueError, match="CPU integer"):
        batch_binding_words(7, torch.tensor([3]), torch.tensor([9.0]), [True])
    with pytest.raises(ValueError, match=r"\(-1, 0\) sentinel"):
        batch_binding_words(7, [3, 4], [9, 0], [True, False])

    target = torch.tensor([[10, 0, 0, 0, 0, 0]], dtype=torch.float32)
    proposals = torch.full((1, 5), -1)
    metadata = batch_control_metadata(
        round_generation=1,
        request_slots=[0],
        slot_generations=[1],
        target_counts=[1],
        proposal_counts=[0],
        remaining_output_budget=[4],
        target_start_positions=[0],
        max_model_len=8,
        need_sample=[1],
        active=[True],
    )
    with pytest.raises(TypeError, match="target ids"):
        verify_control_from_target_batch(target, proposals, metadata, torch.tensor([True]))
    with pytest.raises(ValueError, match="max_model_len"):
        batch_control_metadata(
            round_generation=1,
            request_slots=[0],
            slot_generations=[1],
            target_counts=[1],
            proposal_counts=[0],
            remaining_output_budget=[4],
            target_start_positions=[0],
            max_model_len=torch.tensor([8.0]),
            need_sample=[1],
            active=[True],
        )


_BINDING_BOUNDARY_CASES = (
    ("negative", -1, None),
    ("zero", 0, None),
    ("round_zero_low_full", (1 << 32) - 1, None),
    ("round_one_generation_zero", 1 << 32, None),
    ("round_one", (1 << 32) | 1, (1, 1, 0)),
    ("round_two", (2 << 32) | 1, (2, 1, 0)),
    ("slot_255", (7 << 32) | (255 << 24) | 1, (7, 1, 255)),
    (
        "generation_max",
        (7 << 32) | (255 << 24) | ((1 << 24) - 1),
        (7, (1 << 24) - 1, 255),
    ),
    ("signed_max", (1 << 63) - 1, ((1 << 31) - 1, (1 << 24) - 1, 255)),
)


@pytest.mark.parametrize(("name", "binding", "identity"), _BINDING_BOUNDARY_CASES)
def test_binding_sidecar_boundaries_use_real_verify_pack_validate(name, binding, identity):
    target = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int64)
    proposals = target[:, :5].clone()
    active = torch.tensor([True])
    metadata = batch_control_metadata(
        round_generation=1,
        request_slots=[0],
        slot_generations=[1],
        target_counts=[6],
        proposal_counts=[5],
        remaining_output_budget=[64],
        target_start_positions=[0],
        max_model_len=128,
        need_sample=[1],
        active=active,
    ).clone()
    metadata[0, 0] = binding
    binding_sidecar = batch_binding_sidecar(metadata)

    _, output, committed, output_count, _, draft_enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active, binding_sidecar)
    record = pack_record_batch(
        metadata, committed, output_count, output, proposals, draft_enabled, status, binding_sidecar)

    if identity is None:
        assert status.tolist() == [1], name
        assert record.tolist() == [[0, 0, 0, 0, *([-1] * 11), 1]], name
        with pytest.raises(RuntimeError, match="Invalid DSpark batch commit"):
            validate_batch_records(
                record,
                round_generation=1,
                request_slots=[0],
                slot_generations=[1],
                max_committed=[6],
                active=active,
            )
        return

    round_generation, slot_generation, slot = identity
    assert status.tolist() == [0], name
    assert record[0, 0].item() == binding
    commit = validate_batch_records(
        record,
        round_generation=round_generation,
        request_slots=[slot],
        slot_generations=[slot_generation],
        max_committed=[6],
        active=active,
    )
    assert commit.committed == (6, ), name


def test_binding_sidecar_uses_low_high_order_and_rejects_bad_geometry():
    control = torch.tensor([[1 << 32 | 1, 6, 5, 64, 0, 128, 1]], dtype=torch.int64)
    sidecar = batch_binding_sidecar(control)
    assert sidecar.dtype == torch.int32
    assert sidecar.tolist() == [[1, 1]]
    assert BINDING_SIDECAR_LOW == 0 and BINDING_SIDECAR_HIGH == 1

    signed_max = control.clone()
    signed_max[0, 0] = (1 << 63) - 1
    assert batch_binding_sidecar(signed_max).tolist() == [[-1, (1 << 31) - 1]]

    target = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int64)
    proposals = target[:, :5].clone()
    active = torch.tensor([True])
    with pytest.raises(ValueError, match=r"int32 \[B, 2\]"):
        verify_control_from_target_batch(target, proposals, control, active, sidecar.to(torch.int64))
    with pytest.raises(ValueError, match=r"int32 \[B, 2\]"):
        verify_control_from_target_batch(target, proposals, control, active, sidecar[:, :1])
    with pytest.raises(ValueError, match="verification device"):
        verify_control_from_target_batch(
            target, proposals, control, active, torch.empty((1, 2), dtype=torch.int32, device="meta"))
    with pytest.raises(RuntimeError, match="requires an explicit I32 binding sidecar"):
        _validate_binding_sidecar(None, torch.device("hpu"), 1)
    with pytest.raises(ValueError, match="authoritative CPU I64 control"):
        batch_binding_sidecar(control.to(torch.int32))


@pytest.mark.parametrize(
    ("round_generation", "slot_generation", "slot"),
    [(2, 1, 0), (1, 1, 1), (1, 2, 0)],
)
def test_sidecar_round_slot_generation_mismatch_is_rejected_by_host(round_generation, slot_generation, slot):
    target = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int64)
    proposals = target[:, :5].clone()
    active = torch.tensor([True])
    control = batch_control_metadata(
        round_generation=1,
        request_slots=[0],
        slot_generations=[1],
        target_counts=[6],
        proposal_counts=[5],
        remaining_output_budget=[64],
        target_start_positions=[0],
        max_model_len=128,
        need_sample=[1],
        active=active,
    )
    wrong_control = batch_control_metadata(
        round_generation=round_generation,
        request_slots=[slot],
        slot_generations=[slot_generation],
        target_counts=[6],
        proposal_counts=[5],
        remaining_output_budget=[64],
        target_start_positions=[0],
        max_model_len=128,
        need_sample=[1],
        active=active,
    )
    sidecar = batch_binding_sidecar(wrong_control)
    _, output, committed, output_count, _, draft_enabled, status = verify_control_from_target_batch(
        target, proposals, control, active, sidecar)
    assert status.tolist() == [0]
    record = pack_record_batch(
        control, committed, output_count, output, proposals, draft_enabled, status, sidecar)
    with pytest.raises(RuntimeError, match="Invalid DSpark batch commit"):
        validate_batch_records(
            record,
            round_generation=1,
            request_slots=[0],
            slot_generations=[1],
            max_committed=[6],
            active=active,
        )


def test_inactive_binding_uses_real_verify_pack_validate():
    target = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int64)
    proposals = torch.full((1, 5), -1, dtype=torch.int64)
    active = torch.tensor([False])
    metadata = batch_control_metadata(
        round_generation=1,
        request_slots=[-1],
        slot_generations=[0],
        target_counts=[1],
        proposal_counts=[0],
        remaining_output_budget=[0],
        target_start_positions=[0],
        max_model_len=128,
        need_sample=[0],
        active=active,
    )
    assert metadata[0, 0].item() == 0
    binding_sidecar = batch_binding_sidecar(metadata)

    _, output, committed, output_count, _, draft_enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active, binding_sidecar)
    record = pack_record_batch(
        metadata, committed, output_count, output, proposals, draft_enabled, status, binding_sidecar)
    assert status.tolist() == [BATCH_INACTIVE]
    assert record.tolist() == [[0, 0, 0, 0, *([-1] * 11), BATCH_INACTIVE]]
    commit = validate_batch_records(
        record,
        round_generation=1,
        request_slots=[-1],
        slot_generations=[0],
        max_committed=[0],
        active=active,
    )
    assert commit.committed == (0, )
    assert commit.output == commit.draft == ((), )


def test_budget_context_and_counts_are_per_request_and_consistent():
    target = torch.tensor([[10, 11, 12, 13, 14, 15], [20, 0, 0, 0, 0, 0]], dtype=torch.int64)
    proposals = torch.tensor([[10, 11, 12, 13, 14], [-1, -1, -1, -1, -1]], dtype=torch.int64)
    active = torch.tensor([True, True])
    metadata = batch_control_metadata(
        round_generation=3,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        target_counts=[6, 1],
        proposal_counts=[5, 0],
        remaining_output_budget=[2, 64],
        target_start_positions=[10, 90],
        max_model_len=[100, 100],
        need_sample=[1, 1],
        active=active,
    )
    _, output, committed, output_count, _, draft_enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active)

    assert status.tolist() == [0, 0]
    assert committed.tolist() == [2, 1]
    assert output_count.tolist() == [2, 1]
    assert output.tolist() == [[10, 11, -1, -1, -1, -1], [20, -1, -1, -1, -1, -1]]
    assert draft_enabled.tolist() == [False, True]

    record = pack_record_batch(metadata, committed, output_count, output, torch.full_like(proposals, 77),
                               draft_enabled, status)
    commit = validate_batch_records(
        record,
        round_generation=3,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        max_committed=[6, 6],
        active=active,
    )
    assert commit.committed == (2, 1)
    assert commit.output == ((10, 11), (20, ))
    assert commit.draft == ((), (77, 77, 77, 77, 77))

    with pytest.raises(ValueError, match="C1 target_count"):
        batch_control_metadata(
            round_generation=1,
            request_slots=[0],
            slot_generations=[1],
            target_counts=[6],
            proposal_counts=[0],
            remaining_output_budget=[8],
            target_start_positions=[0],
            max_model_len=8,
            need_sample=[1],
            active=[True],
        )
    with pytest.raises(ValueError, match="exceed the context"):
        batch_control_metadata(
            round_generation=1,
            request_slots=[0],
            slot_generations=[1],
            target_counts=[6],
            proposal_counts=[5],
            remaining_output_budget=[8],
            target_start_positions=[3],
            max_model_len=8,
            need_sample=[1],
            active=[True],
        )
    with pytest.raises(ValueError, match="need_sample"):
        batch_control_metadata(
            round_generation=1,
            request_slots=[0],
            slot_generations=[1],
            target_counts=[1],
            proposal_counts=[0],
            remaining_output_budget=[8],
            target_start_positions=[0],
            max_model_len=8,
            need_sample=[0],
            active=[True],
        )


def test_device_invalid_and_host_record_mismatches_are_zeroed_or_rejected():
    target = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int64)
    proposals = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.int64)
    active = torch.tensor([True])
    metadata = batch_control_metadata(
        round_generation=1,
        request_slots=[0],
        slot_generations=[1],
        target_counts=[6],
        proposal_counts=[5],
        remaining_output_budget=[64],
        target_start_positions=[10],
        max_model_len=16,
        need_sample=[1],
        active=active,
    )
    invalid_metadata = metadata.clone()
    invalid_metadata[0, 4] = 11
    binding_sidecar = batch_binding_sidecar(invalid_metadata)
    _, output, committed, output_count, _, draft_enabled, status = verify_control_from_target_batch(
        target, proposals, invalid_metadata, active, binding_sidecar)
    assert status.tolist() == [1]
    assert committed.tolist() == [0]
    assert output_count.tolist() == [0]
    assert output.tolist() == [[-1] * 6]
    assert draft_enabled.tolist() == [False]
    invalid_record = pack_record_batch(
        invalid_metadata, committed, output_count, output, torch.zeros_like(proposals), draft_enabled, status,
        binding_sidecar)
    assert invalid_record[0].tolist() == [0, 0, 0, 0, *([-1] * 11), 1]

    _, output, committed, output_count, _, draft_enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active, batch_binding_sidecar(metadata))
    record = pack_record_batch(
        metadata,
        committed,
        output_count,
        output,
        torch.zeros_like(proposals),
        draft_enabled,
        status,
        batch_binding_sidecar(metadata),
    )
    bad_count = record.clone()
    bad_count[0, 2] = 1
    with pytest.raises(RuntimeError, match="Invalid DSpark batch commit"):
        validate_batch_records(
            bad_count,
            round_generation=1,
            request_slots=[0],
            slot_generations=[1],
            max_committed=[6],
            active=active,
        )
    bad_draft = record.clone()
    bad_draft[0, 3] = 2
    with pytest.raises(RuntimeError, match="Invalid DSpark batch commit"):
        validate_batch_records(
            bad_draft,
            round_generation=1,
            request_slots=[0],
            slot_generations=[1],
            max_committed=[6],
            active=active,
        )


def test_validate_rejects_same_generation_slot_swap_and_mixed_padding(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace(
        record=lambda stream: None,
        synchronize=lambda: None,
    ))
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: None)

    target = torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]])
    proposals = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
    active = torch.tensor([True, True])
    metadata = batch_control_metadata(
        round_generation=1,
        request_slots=[0, 1],
        slot_generations=[1, 1],
        target_counts=[6, 6],
        proposal_counts=[5, 5],
        remaining_output_budget=[64, 64],
        target_start_positions=[10, 20],
        max_model_len=100,
        need_sample=[1, 1],
        active=active,
    )
    binding_sidecar = batch_binding_sidecar(metadata)
    _, output, committed, output_count, _, enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active, binding_sidecar)
    record = pack_record_batch(
        metadata, committed, output_count, output, torch.zeros_like(proposals), enabled, status, binding_sidecar)
    with pytest.raises(RuntimeError, match="Invalid DSpark batch commit"):
        validate_batch_records(
            record,
            round_generation=1,
            request_slots=[1, 0],
            slot_generations=[1, 1],
            max_committed=[6, 6],
            active=active,
        )

    ring = BatchVerifyRing("cpu", batch_capacity=2)
    padding_ticket = ring.acquire(
        round_generation=1,
        request_slots=[0, -1],
        slot_generations=[1, 0],
        max_committed=[6, 6],
        active=[True, False],
    )
    assert padding_ticket.request_slots == (0, -1)
    with pytest.raises(ValueError, match=r"\(-1, 0\) sentinel"):
        ring.acquire(
            round_generation=2,
            request_slots=[0, 1],
            slot_generations=[1, 0],
            max_committed=[6, 6],
            active=[True, False],
        )
    ring.close()


def test_ring_acquire_is_atomic_and_reuses_a_free_physical_slot(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace(
        record=lambda stream: None,
        synchronize=lambda: None,
    ))
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: None)

    ring = BatchVerifyRing("cpu", batch_capacity=2)
    target = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    proposals = torch.tensor([[1, 2, 3, 4, 5], [7, 99, -1, -1, -1]])
    active = torch.tensor([True, True])

    def prepare(ticket, generation):
        metadata = batch_control_metadata(
            round_generation=generation,
            request_slots=[0, 1],
            slot_generations=[4, 9],
            target_counts=[6, 2],
            proposal_counts=[5, 1],
            remaining_output_budget=[64, 64],
            target_start_positions=[10, 20],
            max_model_len=100,
            need_sample=[1, 1],
            active=active,
        )
        _, output, committed, output_count, _, enabled, status = verify_control_from_target_batch(
            target, proposals, metadata, active)
        ticket.record.copy_(
            pack_record_batch(metadata, committed, output_count, output, torch.zeros_like(proposals), enabled, status))

    first = ring.acquire(
        round_generation=1,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        max_committed=[6, 2],
        active=active,
    )
    second = ring.acquire(
        round_generation=2,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        max_committed=[6, 2],
        active=active,
    )
    with pytest.raises(RuntimeError, match="no free physical slot"):
        ring.acquire(
            round_generation=3,
            request_slots=[0, 1],
            slot_generations=[4, 9],
            max_committed=[6, 2],
            active=active,
        )
    assert ring.generation == 2

    prepare(second, 2)
    ring.stage(second)
    ring.consume(second)
    with pytest.raises(RuntimeError, match="consumed before release"):
        ring.release(first)
    ring.release(second)

    third = ring.acquire(
        round_generation=3,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        max_committed=[6, 2],
        active=active,
    )
    assert third.slot == second.slot
    with pytest.raises(RuntimeError, match="Stale"):
        ring.stage(second)

    prepare(first, 1)
    ring.stage(first)
    ring.consume(first)
    ring.release(first)
    prepare(third, 3)
    ring.stage(third)
    ring.consume(third)
    ring.release(third)
    ring.close()


def test_owner_validation_rejects_reused_slots_at_consume(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace(
        record=lambda stream: None,
        synchronize=lambda: None,
    ))
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: None)

    owner_generation = 4
    calls = []

    def owner_validator(slots, generations, active):
        calls.append((slots, generations, active))
        return not active[0] or generations[0] == owner_generation

    rejected = BatchVerifyRing("cpu", batch_capacity=2, owner_validator=lambda *_args: False)
    with pytest.raises(RuntimeError, match="no longer owns"):
        rejected.acquire(
            round_generation=1,
            request_slots=[0, -1],
            slot_generations=[4, 0],
            max_committed=[6, 6],
            active=torch.tensor([True, False]),
        )
    assert rejected.generation == 0

    ring = BatchVerifyRing("cpu", batch_capacity=2, owner_validator=owner_validator)
    target = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0]])
    proposals = torch.tensor([[1, 2, 3, 4, 5], [-1, -1, -1, -1, -1]])
    active = torch.tensor([True, False])
    ticket = ring.acquire(
        round_generation=1,
        request_slots=[0, -1],
        slot_generations=[4, 0],
        max_committed=[6, 6],
        active=active,
    )
    metadata = batch_control_metadata(
        round_generation=1,
        request_slots=[0, -1],
        slot_generations=[4, 0],
        target_counts=[6, 1],
        proposal_counts=[5, 0],
        remaining_output_budget=[64, 1],
        target_start_positions=[10, 0],
        max_model_len=100,
        need_sample=[1, 1],
        active=active,
    )
    _, output, committed, output_count, _, enabled, status = verify_control_from_target_batch(
        target, proposals, metadata, active)
    ticket.record.copy_(
        pack_record_batch(metadata, committed, output_count, output, torch.zeros_like(proposals), enabled, status))
    ring.stage(ticket)
    owner_generation = 5
    with pytest.raises(RuntimeError, match="no longer owns"):
        ring.consume(ticket)
    assert ring.completion_sync_count == 0
    assert len(calls) == 2
    ring.close()


def test_request_slots_batch_identity_requires_live_slot_and_generation():
    slots = RequestSlots(2)
    first, second = slots.acquire("a"), slots.acquire("b")
    assert slots.validate_batch_identities(
        [second.index, first.index], [second.generation, first.generation], [True, True])
    assert slots.validate_batch_identities([second.index, -1], [second.generation, 0], [True, False])
    assert not slots.validate_batch_identities(
        [second.index, first.index], [second.generation + 1, first.generation], [True, True])
    assert not slots.validate_batch_identities([second.index, -1], [second.generation, 1], [True, False])

    slots.release(first)
    reused = slots.acquire("c")
    assert reused.index == first.index and reused.generation == first.generation + 1
    assert not slots.validate_batch_identities(
        [second.index, first.index], [second.generation, first.generation], [True, True])
    assert slots.validate_batch_identities(
        [second.index, reused.index], [second.generation, reused.generation], [True, True])


def test_model_initializer_requires_and_injects_request_slot_owner(monkeypatch):
    from vllm_gaudi.models.deepseek_v41 import HpuDeepseekV41ForCausalLM
    from vllm_gaudi.ops import deepseek_v41_batch_draft
    from vllm_gaudi.v1.worker import deepseek_v41_dspark_batch_runner as adapter

    target, draft = object(), object()
    request_slots = RequestSlots(2)
    batch_state = SimpleNamespace(capacity=2, slots=request_slots)
    _patch_cpu_ring_runtime(monkeypatch)
    def target_factory(program, state, *, device, engram_host, batch_capacity):
        assert batch_capacity == state.capacity == 2
        return target

    monkeypatch.setattr(adapter, "DSparkPackedTargetStages", target_factory)
    monkeypatch.setattr(deepseek_v41_batch_draft, "DraftBatchState", lambda draft, capacity: object())
    monkeypatch.setattr(deepseek_v41_batch_draft, "DSparkBatchDraftStage", lambda *args, **kwargs: draft)

    model = object.__new__(HpuDeepseekV41ForCausalLM)
    model.native = True
    model.device = "cpu"
    model.max_model_len = 8192
    model.engram_host = None
    model.program = SimpleNamespace(dspark=True, draft=object())
    model.initialize_request_state = lambda capacity: setattr(model, "batch_state", batch_state)

    assert model.initialize_dspark_batch(2) == (target, draft)
    validator = model.dspark_verify_ring.owner_validator
    assert validator is not None and validator.__self__ is request_slots
    assert validator((0, -1), (1, 0), (True, False)) is False
    first = request_slots.acquire("request-a")
    assert validator((first.index, -1), (first.generation, 0), (True, False)) is True

    model.batch_state = SimpleNamespace(capacity=2)
    model.initialize_request_state = lambda capacity: None
    with pytest.raises(RuntimeError, match="authoritative RequestSlots owner validation"):
        model.initialize_dspark_batch(2)


@pytest.mark.parametrize("native", [False, True])
def test_abort_drains_staged_readback_and_reuses_slot(monkeypatch, native):
    syncs = _patch_cpu_ring_runtime(monkeypatch)

    class Completion:
        def synchronize(self):
            syncs.append(2)

    record = _record_for_stage()

    def readback(_record):
        return record.clone(), Completion()

    ring = BatchVerifyRing(
        "cpu", batch_capacity=2, native_readback=readback if native else None)
    ticket = ring.acquire(
        round_generation=1,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        max_committed=[6, 6],
        active=[True, True],
    )
    ticket.record.copy_(record)
    ring.stage(ticket)
    ring.abort(ticket)
    assert ring.completion_sync_count == 1
    assert len(syncs) == 1
    assert ring.released[ticket.slot] == 1

    reused = ring.acquire(
        round_generation=2,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        max_committed=[6, 6],
        active=[True, True],
    )
    assert reused.slot == ticket.slot
    ring.abort(reused)
    assert ring.completion_sync_count == 1


def test_abort_synchronizes_unstaged_hpu_error_path_before_release(monkeypatch):
    syncs = []
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: syncs.append(1))
    ring = BatchVerifyRing.__new__(BatchVerifyRing)
    ring.device = torch.device("hpu")
    ring.generations = [1, 0]
    ring.staged = [None, None]
    ring.synchronized = [0, 0]
    ring.consumed = [0, 0]
    ring.released = [0, 0]
    ring.closed = False
    ring.native_readback = None
    ring.readbacks = [None, None]
    ring.completion_sync_count = 0
    ticket = SimpleNamespace(slot=0, round_generation=1)

    ring.abort(ticket)

    assert syncs == [1]
    assert ring.completion_sync_count == 1
    assert ring.released[0] == 1


def test_stage_slot_permutation_uses_metadata_owner_order(monkeypatch):
    _patch_cpu_ring_runtime(monkeypatch)
    metadata = _stage_metadata(request_slots=(1, 0), slot_generations=(9, 4))
    target = _stage_target()
    ring = BatchVerifyRing("cpu", batch_capacity=2)
    builder_calls = []
    original_builder = draft_batch.batch_control_metadata

    def builder(**kwargs):
        builder_calls.append((tuple(kwargs["request_slots"]), tuple(kwargs["slot_generations"])))
        return original_builder(**kwargs)

    monkeypatch.setattr(draft_batch, "batch_control_metadata", builder)
    inserted_slots = []
    monkeypatch.setattr(
        draft_batch,
        "insert_context_batch",
        lambda draft, bank, states, positions, slots, committed, active: inserted_slots.append(
            tuple(int(value) for value in slots.tolist())),
    )

    class Backbone:
        def __call__(self, anchors, positions, slots, active):
            return torch.zeros((2, 5, 4)), torch.zeros((2, 5, 6))

    class Markov:
        def __call__(self, anchors, hidden, logits, active):
            return torch.full((2, 5), 77, dtype=torch.int64), torch.zeros((2, 5))

    stage = draft_batch.DSparkBatchDraftStage(
        draft=SimpleNamespace(),
        batch_state=object(),
        verify_ring=ring,
        backbone=Backbone(),
        markov=Markov(),
        max_model_len=100,
        target_ids=lambda target, batch_size: target.token_ids,
        target_context=lambda target, positions: target.states,
    )
    result = stage.verify_and_propose(metadata, target)

    assert builder_calls == [((1, 0), (9, 4))]
    assert inserted_slots == [(1, 0)]
    assert result.accepted.tolist() == [6, 6]
    assert result.draft_token_ids == ((77, 77, 77, 77, 77), (77, 77, 77, 77, 77))
    assert ring.generation == 1 and ring.released == [1, 0]


def test_owner_revocation_rejects_old_consume_and_abort_reuses_slot(monkeypatch):
    _patch_cpu_ring_runtime(monkeypatch)
    slots = RequestSlots(2)
    first, second = slots.acquire("a"), slots.acquire("b")
    ring = BatchVerifyRing("cpu", batch_capacity=2,
                           owner_validator=slots.validate_batch_identities)
    record = _record_for_stage(
        request_slots=(second.index, first.index),
        slot_generations=(second.generation, first.generation),
    )
    ticket = ring.acquire(
        round_generation=1,
        request_slots=[second.index, first.index],
        slot_generations=[second.generation, first.generation],
        max_committed=[6, 6],
        active=[True, True],
    )
    ticket.record.copy_(record)
    ring.stage(ticket)

    slots.release(first)
    reused = slots.acquire("c")
    assert reused.index == first.index and reused.generation == first.generation + 1
    with pytest.raises(RuntimeError, match="no longer owns"):
        ring.consume(ticket)
    assert ring.completion_sync_count == 0
    ring.abort(ticket)
    assert ring.completion_sync_count == 1

    next_ticket = ring.acquire(
        round_generation=2,
        request_slots=[second.index, reused.index],
        slot_generations=[second.generation, reused.generation],
        max_committed=[6, 6],
        active=[True, True],
    )
    assert next_ticket.slot == ticket.slot
    ring.abort(next_ticket)


def test_stage_preflight_failures_do_not_change_ring_occupancy(monkeypatch):
    _patch_cpu_ring_runtime(monkeypatch)
    from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata

    one = DSparkBatchMetadata.build(
        requests=("request-a", ),
        target_query_lengths=[1],
        draft_token_ids=[[]],
        request_slots=[0],
        slot_generations=[1],
        remaining_output_budget=[8],
        target_start_positions=[0],
        round_generation=1,
    )
    ring = BatchVerifyRing("cpu", batch_capacity=2)
    stage = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(), verify_ring=ring, transaction=lambda *args: None, max_model_len=100)
    with pytest.raises(ValueError, match="exactly two request lanes"):
        stage.verify_and_propose(one, _stage_target())
    assert ring.generation == 0 and ring.generations == [0, 0] and ring.released == [0, 0]

    metadata = _stage_metadata()
    target = _stage_target()
    invalid_control = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(),
        verify_ring=ring,
        transaction=lambda *args: None,
        max_model_len=15,
        target_ids=lambda target, batch_size: target.token_ids,
    )
    with pytest.raises(ValueError, match="context limit"):
        invalid_control.verify_and_propose(metadata, target)
    assert ring.generation == 0 and ring.generations == [0, 0] and ring.released == [0, 0]

    def fail_context(target, positions):
        raise RuntimeError("target context unavailable")

    invalid_context = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(),
        verify_ring=ring,
        transaction=lambda *args: None,
        max_model_len=100,
        target_ids=lambda target, batch_size: target.token_ids,
        target_context=fail_context,
    )
    with pytest.raises(RuntimeError, match="target context unavailable"):
        invalid_context.verify_and_propose(metadata, target)
    assert ring.generation == 0 and ring.generations == [0, 0] and ring.released == [0, 0]


def test_stage_graph_failure_aborts_ticket_and_reuses_slot(monkeypatch):
    _patch_cpu_ring_runtime(monkeypatch)
    ring = BatchVerifyRing("cpu", batch_capacity=2)

    def fail_graph(*args):
        raise RuntimeError("graph failed")

    stage = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(),
        verify_ring=ring,
        transaction=fail_graph,
        max_model_len=100,
        target_ids=lambda target, batch_size: target.token_ids,
        target_context=lambda target, positions: target.states,
    )
    with pytest.raises(RuntimeError, match="graph failed"):
        stage.verify_and_propose(_stage_metadata(), _stage_target())
    assert ring.generation == 1 and ring.released == [1, 0]
    assert ring.completion_sync_count == 0

    reused = ring.acquire(
        round_generation=2,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        max_committed=[6, 6],
        active=[True, True],
    )
    assert reused.slot == 0
    ring.abort(reused)


def test_stage_preserves_original_error_when_abort_also_fails(monkeypatch):
    _patch_cpu_ring_runtime(monkeypatch)
    ring = BatchVerifyRing("cpu", batch_capacity=2)

    def fail_graph(*args):
        raise RuntimeError("graph failed")

    def fail_abort(ticket):
        raise RuntimeError("abort failed")

    monkeypatch.setattr(ring, "abort", fail_abort)
    stage = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(),
        verify_ring=ring,
        transaction=fail_graph,
        max_model_len=100,
        target_ids=lambda target, batch_size: target.token_ids,
        target_context=lambda target, positions: target.states,
    )
    with pytest.raises(RuntimeError, match="graph failed") as caught:
        stage.verify_and_propose(_stage_metadata(), _stage_target())
    assert any("abort failed" in note for note in getattr(caught.value, "__notes__", ()))


def test_stage_validation_failure_aborts_staged_ticket_and_reuses_slot(monkeypatch):
    _patch_cpu_ring_runtime(monkeypatch)
    ring = BatchVerifyRing("cpu", batch_capacity=2)
    stage = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(),
        verify_ring=ring,
        transaction=lambda *args: torch.zeros((2, 16), dtype=torch.int64),
        max_model_len=100,
        target_ids=lambda target, batch_size: target.token_ids,
        target_context=lambda target, positions: target.states,
    )
    with pytest.raises(RuntimeError, match="Invalid DSpark batch commit"):
        stage.verify_and_propose(_stage_metadata(), _stage_target())
    assert ring.generation == 1 and ring.released == [1, 0]
    assert ring.completion_sync_count == 1

    reused = ring.acquire(
        round_generation=2,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        max_committed=[6, 6],
        active=[True, True],
    )
    assert reused.slot == 0
    ring.abort(reused)


def test_stage_factory_failure_releases_consumed_ticket_without_extra_sync(monkeypatch):
    _patch_cpu_ring_runtime(monkeypatch)
    ring = BatchVerifyRing("cpu", batch_capacity=2)

    def fail_factory(**kwargs):
        raise RuntimeError("factory failed")

    stage = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(),
        verify_ring=ring,
        transaction=lambda *args: _record_for_stage(),
        result_factory=fail_factory,
        max_model_len=100,
        target_ids=lambda target, batch_size: target.token_ids,
        target_context=lambda target, positions: target.states,
    )
    with pytest.raises(RuntimeError, match="factory failed"):
        stage.verify_and_propose(_stage_metadata(), _stage_target())
    assert ring.generation == 1 and ring.released == [1, 0]
    assert ring.completion_sync_count == 1

    reused = ring.acquire(
        round_generation=2,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        max_committed=[6, 6],
        active=[True, True],
    )
    assert reused.slot == 0
    ring.abort(reused)


def test_draft_attention_rows_isolate_requests_and_keep_tentative_scratch():
    positions = torch.tensor([[260, 261, 262, 263, 264], [1000, 1001, 1002, 1003, 1004]], dtype=torch.int32)
    rows = draft_batch.draft_attention_rows(
        positions,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([True, True]),
        ring_rows=512,
    )
    assert rows.row_ids.shape == (2 * (128 + 5), )
    assert rows.indices.shape == (10, 192)
    assert rows.lengths.tolist() == [133] * 10
    # Request 0 and request 1 never share accepted-context or tentative rows.
    request0 = set(rows.row_ids[:133].tolist())
    request1 = set(rows.row_ids[133:].tolist())
    assert request0.isdisjoint(request1)
    assert set(rows.row_ids[128:133].tolist()) == set(range(512, 517))
    assert set(rows.row_ids[133 + 128:133 + 133].tolist()) == set(range(517, 522))
    assert (rows.indices[:, 133:] == -1).all()


def test_partial_accept_commits_only_accepted_draft_context(monkeypatch):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("swa", torch.zeros(256, 8, dtype=torch.uint8))
            self.weights = SimpleNamespace(wkv=object(), kv_norm=SimpleNamespace(weight=torch.ones(8)))
            self.eps = 1e-6
            self.linear = lambda value, weight: value

        def _rope(self, value, positions, request_batch=False):
            assert request_batch
            return value

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer(), Layer()])
            first = SimpleNamespace(main_proj=object(), main_norm=SimpleNamespace(weight=torch.ones(8)))
            self.weights = SimpleNamespace(get_submodule=lambda name: first)
            self.eps = 1e-6

    monkeypatch.setattr(draft_batch, "_linear", lambda value, owner: value)
    monkeypatch.setattr(draft_batch, "rms_norm", lambda value, weight, eps: value)
    monkeypatch.setattr(draft_batch, "pack_swa", lambda value: value.to(torch.uint8))
    draft = Draft()
    bank = draft_batch.DraftBatchState(draft, capacity=2)
    target_states = torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8).bfloat16()
    positions = torch.tensor([[100, 101, 102, 103, 104, 105], [300, 301, 302, 303, 304, 305]], dtype=torch.int32)
    draft_batch.insert_context_batch(
        draft,
        bank,
        target_states,
        positions,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([2, 1], dtype=torch.int32),
        torch.tensor([True, True]),
    )
    ring = bank.ring(0)
    assert torch.equal(ring[100:102], target_states[0, :2].to(torch.uint8))
    assert torch.equal(ring[300:301], target_states[1, :1].to(torch.uint8))
    assert torch.count_nonzero(ring[102:106]) == 0
    assert torch.count_nonzero(ring[301:306]) == 0


class _MarkovDraft(torch.nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.hidden = hidden
        last = SimpleNamespace(
            markov_head=SimpleNamespace(
                embed=object(),
                head=SimpleNamespace(weight=torch.eye(hidden)),
            ),
            confidence_head=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(1, hidden * 2))),
        )
        self.weights = SimpleNamespace(get_submodule=lambda name: last)

    def _embed(self, previous, embedding):
        value = previous.reshape(-1).long().remainder(self.hidden)
        return F.one_hot(value, self.hidden).float() * 2

    def _global_argmax(self, scores):
        return scores.argmax(-1)


def _serial_markov(draft, first, hidden, logits, active):
    previous = first.clone()
    ids, confidence = [], []
    for index in range(5):
        markov = draft._embed(previous, None)
        scores = logits[:, index] + F.linear(markov, draft.weights.get_submodule("2").markov_head.head.weight)
        sampled = scores.argmax(-1)
        ids.append(torch.where(active, sampled, -1))
        confidence.append(torch.zeros_like(sampled, dtype=torch.float32))
        previous = torch.where(active, sampled, 0)
    return torch.stack(ids, 1), torch.stack(confidence, 1)


def test_markov_steps_are_sequential_but_cross_request_independent(monkeypatch):
    monkeypatch.setattr(draft_batch, "_linear", lambda value, owner: torch.zeros(
        (value.shape[0], 1), dtype=torch.float32))
    draft = _MarkovDraft()
    anchors = torch.tensor([3, 1], dtype=torch.int64)
    hidden = torch.zeros((2, 5, 4), dtype=torch.float32)
    logits = torch.full((2, 5, 4), 0.1, dtype=torch.float32)
    active = torch.tensor([True, True])
    actual, confidence = draft_batch.sample_greedy_markov_batch(draft, anchors, hidden, logits, active)
    expected, _ = _serial_markov(draft, anchors, hidden, logits, active)
    assert torch.equal(actual, expected)
    assert actual.tolist() == [[3, 3, 3, 3, 3], [1, 1, 1, 1, 1]]
    assert confidence.shape == (2, 5)

    changed_anchor = anchors.clone()
    changed_anchor[0] = 0
    changed, _ = draft_batch.sample_greedy_markov_batch(draft, changed_anchor, hidden, logits, active)
    assert torch.equal(changed[1], actual[1])
    assert torch.equal(changed[0], torch.zeros(5, dtype=torch.int64))


def test_draft_handoff_uses_each_request_accepted_prefix():
    target_positions = torch.tensor([[100, 101, 102, 103, 104, 105], [500, 501, 502, 503, 504, 505]],
                                    dtype=torch.int32)
    output = torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]], dtype=torch.int64)
    anchors, positions = draft_batch.draft_inputs_after_commit(
        target_positions,
        output,
        torch.tensor([6, 2]),
        torch.tensor([6, 2]),
        torch.tensor([True, False]),
    )
    assert anchors.tolist() == [15, 0]
    assert positions[0].tolist() == [106, 107, 108, 109, 110]
    assert positions[1].tolist() == [-1] * 5


def test_runner_facing_draft_stage_returns_one_commit_and_next_proposals(monkeypatch):
    from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata

    metadata = DSparkBatchMetadata.build(
        requests=("request-a", "request-b"),
        target_query_lengths=[6, 6],
            draft_token_ids=[[1, 2, 3, 4, 5], [7, 99, 100, 101, 102]],
        request_slots=[0, 1],
        slot_generations=[4, 9],
        remaining_output_budget=[64, 64],
        target_start_positions=[10, 12],
        round_generation=1,
    )
    target = SimpleNamespace(
        token_ids=torch.tensor([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ], dtype=torch.int64),
        payload=(torch.zeros((2, 6, 8), dtype=torch.bfloat16),
                 torch.zeros((12, 8), dtype=torch.bfloat16)),
    )
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", lambda: SimpleNamespace(
        record=lambda stream: None,
        synchronize=lambda: None,
        query=lambda: True,
    ))
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: None)
    committed_seen = []
    monkeypatch.setattr(draft_batch, "insert_context_batch",
                        lambda draft, bank, states, positions, slots, committed, active:
                        committed_seen.append(committed.tolist()))

    class Backbone:
        def __call__(self, anchors, positions, slots, active):
            assert anchors.shape == (2, 1)
            assert positions.shape == (2, 5)
            assert active.tolist() == [True, True]
            return (torch.zeros((2, 5, 4)), torch.zeros((2, 5, 6)))

    class Markov:
        def __call__(self, anchors, hidden, logits, active):
            return torch.tensor([[20, 21, 22, 23, 24], [30, 31, 32, 33, 34]]), torch.zeros((2, 5))

    stage = draft_batch.DSparkBatchDraftStage(
        draft=SimpleNamespace(),
        verify_ring=BatchVerifyRing("cpu", batch_capacity=2),
        backbone=Backbone(),
        markov=Markov(),
        max_model_len=128,
    )
    result = stage.verify_and_propose(metadata, target)
    assert result.accepted.tolist() == [6, 2]
    assert result.output_token_ids == ((1, 2, 3, 4, 5, 6), (7, 8))
    assert result.draft_token_ids == ((20, 21, 22, 23, 24), (30, 31, 32, 33, 34))
    assert committed_seen == [[6, 2]]
    assert result.completion_event.query()
