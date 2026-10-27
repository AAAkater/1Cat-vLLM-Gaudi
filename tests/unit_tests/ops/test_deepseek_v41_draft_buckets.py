# SPDX-License-Identifier: Apache-2.0
"""Host-only draft bucket selection keeps graph keys independent of lane data."""

from dataclasses import FrozenInstanceError

import pytest

from vllm_gaudi.ops.deepseek_v41_draft_buckets import (
    DraftBucketPlan,
    draft_execution_buckets,
    select_draft_bucket,
)


def _active_at(capacity, *lanes):
    selected = set(lanes)
    return tuple(lane in selected for lane in range(capacity))


def _slots_at(capacity, slots):
    return tuple(slots.get(lane, -1) for lane in range(capacity))


class _ReadbackTrap:
    def tolist(self):
        raise AssertionError("device readback is forbidden")


@pytest.mark.parametrize(
    ("capacity", "buckets", "expected"),
    [
        (1, [1], (1,)),
        (3, [1, 2, 4], (1, 2, 3)),
        (6, [1, 2, 4, 8], (1, 2, 4, 6)),
        (11, [1, 2, 4, 8, 16], (1, 2, 4, 8, 11)),
        (32, [1, 2, 4], (1, 2, 4, 32)),
        (64, [1, 2, 4, 8, 16, 32], (1, 2, 4, 8, 16, 32, 64)),
    ],
)
def test_draft_buckets_cover_supported_capacities_and_add_custom_n(capacity, buckets, expected):
    assert draft_execution_buckets(capacity, buckets) == expected


def test_draft_buckets_are_sorted_and_deduplicated():
    assert draft_execution_buckets(11, [8, 1, 4, 2, 8]) == (1, 2, 4, 8, 11)


@pytest.mark.parametrize("capacity", [0, 65, True, 3.0, None])
def test_draft_buckets_reject_invalid_state_capacity(capacity):
    with pytest.raises(ValueError, match="draft state capacity"):
        draft_execution_buckets(capacity, [1])


@pytest.mark.parametrize("buckets", [None, (), [], [True], [0], [3], [65], [1.0], [1, 128]])
def test_draft_buckets_reject_invalid_input_buckets(buckets):
    with pytest.raises(ValueError, match="draft execution buckets"):
        draft_execution_buckets(32, buckets)


def test_n1_selects_identity_and_exposes_five_rows_per_lane():
    plan = select_draft_bucket((True,), (0,), draft_execution_buckets(1, [1]))
    assert plan == DraftBucketPlan(1, 1, (0,), (0,))
    assert plan.graph_key == (1, 1)
    assert plan.execution_rows == 5


def test_n3_selects_one_noncontiguous_active_lane():
    plan = select_draft_bucket(
        _active_at(3, 1),
        _slots_at(3, {1: 2}),
        draft_execution_buckets(3, [1, 2, 4]),
    )
    assert plan.state_capacity == 3
    assert plan.execution_capacity == 1
    assert plan.source_lanes == (1,)
    assert plan.logical_slots == (2,)


def test_n6_fills_custom_bucket_with_distinct_inactive_lanes():
    plan = select_draft_bucket(
        _active_at(6, 1, 4),
        _slots_at(6, {1: 5, 4: 1}),
        draft_execution_buckets(6, [1, 4]),
    )
    assert plan.execution_capacity == 4
    assert plan.source_lanes == (1, 4, 0, 2)
    assert plan.logical_slots == (5, 1, -1, -1)
    assert plan.graph_key == (6, 4)
    assert plan.execution_rows == 20


def test_n11_keeps_active_lane_order_then_identity_inactive_fillers():
    plan = select_draft_bucket(
        _active_at(11, 0, 2, 5, 8, 10),
        _slots_at(11, {0: 10, 2: 0, 5: 4, 8: 7, 10: 2}),
        draft_execution_buckets(11, [1, 2, 4, 8, 16]),
    )
    assert plan.execution_capacity == 8
    assert plan.source_lanes == (0, 2, 5, 8, 10, 1, 3, 4)
    assert plan.logical_slots == (10, 0, 4, 7, 2, -1, -1, -1)


def test_n32_preserves_noncontiguous_lanes_and_slots():
    plan = select_draft_bucket(
        _active_at(32, 4, 19, 31),
        _slots_at(32, {4: 0, 19: 17, 31: 31}),
        draft_execution_buckets(32, [1, 2, 4]),
    )
    assert plan.execution_capacity == 4
    assert plan.source_lanes == (4, 19, 31, 0)
    assert plan.logical_slots == (0, 17, 31, -1)
    assert plan.graph_key == (32, 4)
    assert plan.execution_rows == 20


def test_n64_selects_partial_bucket_and_fills_unique_inactive_lanes():
    plan = select_draft_bucket(
        _active_at(64, 0, 9, 33, 48, 63),
        _slots_at(64, {0: 63, 9: 1, 33: 17, 48: 7, 63: 48}),
        draft_execution_buckets(64, [1, 2, 4, 8, 16, 32]),
    )
    assert plan.execution_capacity == 8
    assert plan.source_lanes == (0, 9, 33, 48, 63, 1, 2, 3)
    assert plan.logical_slots == (63, 1, 17, 7, 48, -1, -1, -1)
    assert plan.execution_rows == 40


def test_full_n_fallback_retains_identity_order_not_active_first():
    slots = _slots_at(6, {0: 5, 1: 1, 3: 0, 4: 3, 5: 2})
    plan = select_draft_bucket(
        _active_at(6, 0, 1, 3, 4, 5),
        slots,
        draft_execution_buckets(6, [1, 2, 4, 8]),
    )
    assert plan.execution_capacity == 6
    assert plan.source_lanes == (0, 1, 2, 3, 4, 5)
    assert plan.logical_slots == slots
    assert plan.graph_key == (6, 6)
    assert plan.execution_rows == 30


def test_empty_active_set_uses_full_n_identity_fallback():
    plan = select_draft_bucket(
        (False,) * 32,
        (-1,) * 32,
        draft_execution_buckets(32, [1, 2, 4, 8, 16]),
    )
    assert plan.execution_capacity == 32
    assert plan.source_lanes == tuple(range(32))
    assert plan.logical_slots == (-1,) * 32
    assert plan.graph_key == (32, 32)
    assert plan.execution_rows == 160


def test_graph_key_depends_only_on_n_and_b():
    buckets = draft_execution_buckets(11, [1, 2, 4, 8, 16])
    first = select_draft_bucket(
        _active_at(11, 1, 3, 7),
        _slots_at(11, {1: 9, 3: 3, 7: 0}),
        buckets,
    )
    second = select_draft_bucket(
        _active_at(11, 0, 2, 5),
        _slots_at(11, {0: 10, 2: 4, 5: 1}),
        buckets,
    )
    assert first.source_lanes != second.source_lanes
    assert first.logical_slots != second.logical_slots
    assert first.graph_key == second.graph_key == (11, 4)
    with pytest.raises(FrozenInstanceError):
        first.graph_key = (11, 8)


@pytest.mark.parametrize(
    ("active", "slots", "buckets", "match"),
    [
        ((True,), (), (1,), "same length"),
        ((1,), (0,), (1,), "booleans"),
        ((True,), (0.0,), (1,), "integers"),
        ((True,), (-1,), (1,), "active draft request slots"),
        (_active_at(3, 1), _slots_at(3, {1: 3}), (1, 2, 3), "active draft request slots"),
        ((True, True, False), (1, 1, -1), (1, 2, 3), "unique"),
        ((False,), (0,), (1,), "inactive draft request slots"),
        ((), (), (1,), "draft state capacity"),
    ],
)
def test_select_rejects_invalid_host_activity_and_slots(active, slots, buckets, match):
    with pytest.raises(ValueError, match=match):
        select_draft_bucket(active, slots, buckets)


@pytest.mark.parametrize(
    ("capacity", "buckets", "match"),
    [
        (4, (1, 2), "include the draft state capacity"),
        (4, (2, 1, 4), "strictly ascending"),
        (4, (1, 1, 4), "strictly ascending"),
        (4, (1, 3, 4), "powers of two or the custom state capacity"),
        (4, (1, 2, 4, 8), "must not exceed"),
        (4, (True, 4), "integers"),
        (6, (1, 2, 4, 5, 6), "powers of two or the custom state capacity"),
        (4, (), "nonempty"),
    ],
)
def test_select_rejects_invalid_execution_bucket_sets(capacity, buckets, match):
    with pytest.raises(ValueError, match=match):
        select_draft_bucket((False,) * capacity, (-1,) * capacity, buckets)


def test_select_rejects_tensor_like_inputs_without_readback():
    with pytest.raises(ValueError, match="CPU tuple or list"):
        select_draft_bucket(_ReadbackTrap(), (0,), (1,))
    with pytest.raises(ValueError, match="CPU tuple or list"):
        select_draft_bucket((True,), _ReadbackTrap(), (1,))


def test_plan_rejects_invalid_identity_layouts():
    with pytest.raises(ValueError, match="source lanes"):
        DraftBucketPlan(6, 4, (1, 1, 2, 3), (-1, -1, -1, -1))
    with pytest.raises(ValueError, match="source lanes"):
        DraftBucketPlan(6, 4, (1, 2, 3), (-1, -1, -1))
    with pytest.raises(ValueError, match="logical slots"):
        DraftBucketPlan(6, 4, (1, 2, 3, 4), (1, 1, -1, -1))
    with pytest.raises(ValueError, match="logical slots"):
        DraftBucketPlan(6, 4, (1, 2, 3, 4), (1, 6, -1, -1))
