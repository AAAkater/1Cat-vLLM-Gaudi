# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for DSpark active compact metadata."""

from dataclasses import fields, replace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_active_batch import (
    DSparkActiveBatchMetadata,
    DSparkActiveGenerationTracker,
    DSparkActiveIdentity,
    compact_dspark_metadata,
    restore_fixed_dspark_metadata,
)
from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata


def _fixed_metadata(
    capacity,
    active,
    *,
    lengths=None,
    drafts=None,
    budgets=None,
    starts=None,
    generations=None,
    target_padding=None,
    draft_padding=None,
    round_generation=17,
):
    lengths = lengths or {}
    drafts = drafts or {}
    budgets = budgets or {}
    starts = starts or {}
    generations = generations or {}
    target_padding = target_padding or {}
    draft_padding = draft_padding or {}

    active_mask = [lane in active for lane in range(capacity)]
    request_ids = [f"request-{lane}" if active_mask[lane] else None for lane in range(capacity)]
    target_lengths = [lengths.get(lane, 0) for lane in range(capacity)]
    draft_rows = [list(drafts.get(lane, ())) for lane in range(capacity)]
    slots = [lane if active_mask[lane] else -1 for lane in range(capacity)]
    slot_generations = [generations.get(lane, 0) for lane in range(capacity)]
    remaining = [budgets.get(lane, 0) for lane in range(capacity)]
    positions = [starts.get(lane, 0) for lane in range(capacity)]
    padding = []
    for lane in range(capacity):
        if active_mask[lane]:
            row = list(target_padding.get(lane, [False] * target_lengths[lane]))
            assert len(row) == target_lengths[lane]
            padding.extend(row)

    metadata = DSparkBatchMetadata.build(
        requests=request_ids,
        active=active_mask,
        target_query_lengths=target_lengths,
        draft_token_ids=draft_rows,
        request_slots=slots,
        slot_generations=slot_generations,
        remaining_output_budget=remaining,
        target_padding_mask=padding,
        target_start_positions=positions,
        round_generation=round_generation,
    )
    if draft_padding is not None:
        flat = []
        for lane in range(capacity):
            count = len(draft_rows[lane])
            flat.extend(draft_padding.get(lane, [False] * count))
        metadata = replace(metadata, draft_padding_mask=torch.tensor(flat, dtype=torch.bool))
    return metadata


def _assert_fixed_equal(actual, expected):
    assert actual.request_ids == expected.request_ids
    for field in fields(DSparkBatchMetadata):
        left = getattr(actual, field.name)
        right = getattr(expected, field.name)
        if isinstance(right, torch.Tensor):
            assert isinstance(left, torch.Tensor)
            assert left.device.type == right.device.type == "cpu"
            assert left.dtype == right.dtype, field.name
            assert left.shape == right.shape, field.name
            assert torch.equal(left, right), field.name
        else:
            assert left == right, field.name


def test_n32_a1_compact_row_is_decoupled_from_logical_slot():
    fixed = _fixed_metadata(
        32,
        {17},
        lengths={17: 6},
        drafts={17: [101, 102, 103, 104, 105]},
        budgets={17: 9},
        starts={17: 321},
        generations={17: 5},
        round_generation=13,
    )

    compact = compact_dspark_metadata(fixed, scheduler_order=("request-17", ))

    assert isinstance(compact, DSparkActiveBatchMetadata)
    assert compact.capacity == 32
    assert compact.active_count == compact.num_active == compact.compact_capacity == 1
    assert compact.request_ids == ("request-17", )
    assert compact.request_active.tolist() == [True]
    assert compact.scheduler_slots == (17, )
    assert compact.compact_to_logical_slots == (17, )
    assert compact.compact_to_fixed_lanes == (17, )
    assert compact.logical_slot_to_compact_rows[17] == 0
    assert sum(row >= 0 for row in compact.logical_slot_to_compact_rows) == 1
    assert compact.fixed_lane_to_compact_rows[17] == 0
    assert compact.inactive_fixed_lanes == tuple(lane for lane in range(32) if lane != 17)
    assert compact.remaining_output_budget.tolist() == [9]
    assert compact.identities == (DSparkActiveIdentity("request-17", 17, 5, 13), )
    assert compact.identity_for_logical_slot(17).request_id == "request-17"

    no_write = compact.no_write_mask
    assert no_write.shape == (32, 6)
    assert not no_write[17].any()
    assert all(no_write[lane].all() for lane in range(32) if lane != 17)

    restored = restore_fixed_dspark_metadata(compact)
    _assert_fixed_equal(restored, fixed)


def test_interleaved_slots_and_scheduler_order_round_trip_every_field():
    fixed = _fixed_metadata(
        8,
        {1, 4, 6},
        lengths={1: 1, 4: 6, 6: 6},
        drafts={1: [], 4: [11, 12, 13, 14, 15], 6: [61, 62]},
        budgets={1: 2, 4: 4, 6: 9},
        starts={1: 5, 4: 40, 6: 900},
        generations={1: 3, 4: 7, 6: 9},
        target_padding={6: [False, True, False, False, False, False]},
        draft_padding={6: [False, True]},
        round_generation=23,
    )

    compact = compact_dspark_metadata(
        fixed,
        scheduler_order=("request-6", "request-1", "request-4"),
    )

    assert compact.request_ids == ("request-6", "request-1", "request-4")
    assert compact.scheduler_slots == (6, 1, 4)
    assert compact.canonical_slots == (1, 4, 6)
    assert compact.canonical_request_ids == ("request-1", "request-4", "request-6")
    assert compact.canonical_compact_rows == (1, 2, 0)
    assert compact.compact_to_fixed_lanes == (6, 1, 4)
    assert compact.fixed_lane_to_compact_rows == (-1, 1, -1, -1, 2, -1, 0, -1)
    assert compact.target_query_lengths == (6, 1, 6)
    assert compact.num_draft_tokens.tolist() == [2, 0, 5]
    assert compact.remaining_output_budget.tolist() == [9, 2, 4]
    assert compact.target_padding_mask.tolist() == [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert compact.draft_padding_mask.tolist() == [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert compact.logits_indices.tolist() == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    no_write = compact.no_write_mask
    assert not no_write[1, 0]
    assert no_write[1, 1:].all()
    assert no_write[6, 1]
    assert no_write[0].all() and no_write[2].all() and no_write[3].all()
    assert no_write[4].shape == (6, ) and not no_write[4].any()

    restored = restore_fixed_dspark_metadata(compact)
    _assert_fixed_equal(restored, fixed)


@pytest.mark.parametrize(("capacity", "active"), ((3, (0, 2)), (11, (2, 7, 10))))
def test_custom_capacities_n3_n11_support_c1_c6(capacity, active):
    lengths = {lane: (1 if index % 2 == 0 else 6) for index, lane in enumerate(active)}
    drafts = {
        lane: ([] if lengths[lane] == 1 else list(range(100 + lane * 10, 105 + lane * 10)))
        for lane in active
    }
    fixed = _fixed_metadata(
        capacity,
        set(active),
        lengths=lengths,
        drafts=drafts,
        budgets={lane: lane + 2 for lane in active},
        starts={lane: 1000 + lane * 100 for lane in active},
        generations={lane: lane + 4 for lane in active},
        round_generation=31,
    )
    scheduler_order = tuple(f"request-{lane}" for lane in reversed(active))

    compact = compact_dspark_metadata(fixed, scheduler_order=scheduler_order)

    assert compact.capacity == capacity
    assert compact.active_count == len(active)
    assert compact.request_ids == scheduler_order
    assert compact.scheduler_slots == tuple(reversed(active))
    assert compact.canonical_slots == tuple(active)
    assert set(compact.target_query_lengths) == {1, 6}
    assert compact.remaining_output_budget.tolist() == [lane + 2 for lane in reversed(active)]
    _assert_fixed_equal(restore_fixed_dspark_metadata(compact), fixed)


def test_arbitrary_logical_slots_restore_fixed_lane_order_without_http_indexes():
    fixed = DSparkBatchMetadata.build(
        requests=("http-9", "request-z", "10", "2"),
        target_query_lengths=[1, 6, 2, 6],
        draft_token_ids=[[], [11, 12, 13, 14, 15], [21], [31, 32, 33, 34, 35]],
        request_slots=[3, 1, 2, 0],
        slot_generations=[5, 2, 8, 3],
        remaining_output_budget=[1, 7, 4, 6],
        target_start_positions=[90, 10, 500, 70],
        target_padding_mask=[False] * 15,
        round_generation=19,
    )

    compact = compact_dspark_metadata(
        fixed,
        scheduler_order=("10", "http-9", "2", "request-z"),
    )

    assert compact.request_ids == ("10", "http-9", "2", "request-z")
    assert compact.scheduler_slots == (2, 3, 0, 1)
    assert compact.compact_to_fixed_lanes == (2, 0, 3, 1)
    assert compact.canonical_slots == (0, 1, 2, 3)
    assert compact.canonical_request_ids == ("2", "request-z", "10", "http-9")
    assert compact.identity_for_request("http-9") == DSparkActiveIdentity("http-9", 3, 5, 19)
    _assert_fixed_equal(restore_fixed_dspark_metadata(compact), fixed)


def test_generation_reuse_and_expired_round_are_rejected():
    fixed = _fixed_metadata(
        4,
        {2},
        lengths={2: 1},
        budgets={2: 5},
        starts={2: 44},
        generations={2: 5},
        round_generation=7,
    )
    compact = compact_dspark_metadata(
        fixed,
        owners={"request-2": (2, 5)},
        expected_round_generation=7,
        minimum_round_generation=7,
    )

    with pytest.raises(RuntimeError, match="slot generation is stale"):
        compact_dspark_metadata(fixed, owners={"request-2": (2, 4)})
    with pytest.raises(RuntimeError, match="expired"):
        compact_dspark_metadata(fixed, minimum_round_generation=8)
    with pytest.raises(RuntimeError, match="stale"):
        compact_dspark_metadata(fixed, expected_round_generation=8)

    idle_tracker = DSparkActiveGenerationTracker({"idle": (0, 4)}, round_generation=6)
    idle_tracker.publish(compact)
    assert idle_tracker.owners["idle"] == (0, 4)
    with pytest.raises(RuntimeError, match="already used"):
        idle_tracker.reuse_slot("idle-replacement", 0, 4)

    tracker = DSparkActiveGenerationTracker()
    tracker.publish(compact)
    assert tracker.validate(compact)
    with pytest.raises(RuntimeError, match="must advance"):
        tracker.publish(compact)

    expired = replace(fixed, round_generation=torch.tensor(6, dtype=torch.int64))
    expired_compact = compact_dspark_metadata(expired)
    with pytest.raises(RuntimeError, match="expired"):
        tracker.publish(expired_compact)

    with pytest.raises(RuntimeError, match="already used"):
        tracker.reuse_slot("replacement", 2, 5)
    tracker.reuse_slot("replacement", 2, 6)
    with pytest.raises(RuntimeError, match="no longer owns|slot generation is stale"):
        tracker.validate(compact)

    replacement_fixed = _fixed_metadata(
        4,
        {2},
        lengths={2: 6},
        drafts={2: [71, 72, 73, 74, 75]},
        budgets={2: 3},
        starts={2: 80},
        generations={2: 6},
        round_generation=8,
    )
    replacement_fixed = replace(
        replacement_fixed,
        request_ids=(None, None, "replacement", None),
    )
    replacement = compact_dspark_metadata(
        replacement_fixed,
        owners=tracker.owners,
        expected_round_generation=8,
    )
    tracker.publish(replacement)
    assert tracker.validate(replacement)
    with pytest.raises(RuntimeError, match="stale"):
        tracker.validate(compact)
    with pytest.raises(RuntimeError, match="already used"):
        tracker.reuse_slot("replacement-2", 2, 6)

    legacy_zero = _fixed_metadata(
        2,
        {0},
        lengths={0: 1},
        budgets={0: 1},
        starts={0: 0},
        generations={0: 1},
        round_generation=0,
    )
    legacy_tracker = DSparkActiveGenerationTracker()
    legacy_compact = compact_dspark_metadata(legacy_zero)
    legacy_tracker.publish(legacy_compact)
    assert legacy_tracker.validate(legacy_compact)
    with pytest.raises(RuntimeError, match="must advance"):
        legacy_tracker.publish(legacy_compact)


def test_restored_fixed_metadata_keeps_inactive_rows_no_write():
    fixed = _fixed_metadata(
        5,
        {1, 4},
        lengths={1: 1, 4: 6},
        drafts={4: [81, 82, 83, 84, 85]},
        budgets={1: 1, 4: 8},
        starts={1: 3, 4: 200},
        generations={1: 2, 4: 7},
        round_generation=41,
    )
    compact = compact_dspark_metadata(
        fixed,
        scheduler_order=("request-4", "request-1"),
    )
    restored = compact.to_fixed_metadata()
    source = torch.arange(14, dtype=torch.int32).reshape(7, 2)

    packed = restored.pack_target_queries(source)

    assert packed.queries.shape == (5, 6, 2)
    assert not packed.queries[0].any()
    assert not packed.queries[2].any()
    assert not packed.queries[3].any()
    assert torch.equal(packed.queries[1, 0], source[0])
    assert torch.equal(packed.queries[4], source[1:])
    assert packed.padding_mask.tolist() == compact.no_write_mask.tolist()
    assert torch.equal(restored.unpack_target_queries(packed), source)
    _assert_fixed_equal(restored, fixed)


def test_invalid_scheduler_orders_duplicate_slots_and_non_cpu_metadata_are_rejected():
    fixed = DSparkBatchMetadata.build(
        requests=("a", "b", "c"),
        target_query_lengths=[1, 6, 1],
        draft_token_ids=[[], [1, 2, 3, 4, 5], []],
        request_slots=[0, 1, 2],
        slot_generations=[1, 2, 3],
        remaining_output_budget=[1, 1, 1],
        round_generation=5,
    )

    with pytest.raises(ValueError, match="not active"):
        compact_dspark_metadata(fixed, scheduler_order=("a", "missing"))
    with pytest.raises(ValueError, match="duplicate"):
        compact_dspark_metadata(fixed, scheduler_order=("a", "a", "b", "c"))
    with pytest.raises(ValueError, match="missing"):
        compact_dspark_metadata(fixed, scheduler_order=("a", "b"))
    with pytest.raises(TypeError, match="request ID strings"):
        compact_dspark_metadata(fixed, scheduler_order=(0, 1, 2))
    with pytest.raises(ValueError, match="logical slots must be unique"):
        compact_dspark_metadata(replace(fixed, request_slots=torch.tensor([0, 0, 2], dtype=torch.int32)))

    meta = DSparkBatchMetadata.build(
        requests=["a", "b"],
        target_query_lengths=[1, 1],
        draft_token_ids=[[], []],
        request_slots=[0, 1],
        slot_generations=[1, 1],
        remaining_output_budget=[1, 1],
        device="meta",
    )
    with pytest.raises(ValueError, match="CPU host"):
        compact_dspark_metadata(meta)


def test_restore_rejects_tampered_mappings_and_round_only_validation_works():
    fixed = _fixed_metadata(
        4,
        {1},
        lengths={1: 1},
        budgets={1: 2},
        starts={1: 9},
        generations={1: 3},
        round_generation=11,
    )
    compact = compact_dspark_metadata(fixed, expected_round_generation=11)

    tampered = replace(
        compact,
        logical_slot_to_compact_rows=(0, -1, -1, -1),
    )
    with pytest.raises(ValueError, match="inverse"):
        restore_fixed_dspark_metadata(tampered)


def test_host_adapter_remains_on_cpu_under_another_default_device():
    fixed = _fixed_metadata(
        32, {31}, lengths={31: 1}, budgets={31: 16}, starts={31: 0}, generations={31: 1})
    with torch.device("meta"):
        compact = compact_dspark_metadata(fixed)
        restored = compact.to_fixed_metadata()
    for metadata in (compact, restored):
        for item in fields(metadata):
            value = getattr(metadata, item.name)
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu", item.name
