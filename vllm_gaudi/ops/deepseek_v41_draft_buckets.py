# SPDX-License-Identifier: Apache-2.0
"""Host-only execution bucket selection for the fixed five-row draft stage.

The draft graph is keyed by the state capacity and execution capacity only.
Runtime lane and logical-slot identities remain ordinary host data, so selecting
a bucket never reads a device tensor back to the host.
"""

from dataclasses import dataclass

from vllm_gaudi.ops.deepseek_v41_capacity import MAX_DSV41_DSPARK_CAPACITY, validate_dspark_capacity

_DRAFT_ROWS_PER_LANE = 5


def _is_power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def _host_tuple(values, name):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a CPU tuple or list")
    return tuple(values)


def _validate_execution_buckets(execution_buckets, state_capacity):
    buckets = _host_tuple(execution_buckets, "draft execution buckets")
    if not buckets:
        raise ValueError("draft execution buckets must be nonempty")
    if any(type(bucket) is not int for bucket in buckets):
        raise ValueError("draft execution buckets must contain only integers")
    if any(bucket < 1 or bucket > MAX_DSV41_DSPARK_CAPACITY for bucket in buckets):
        raise ValueError("draft execution buckets must be in 1..64")
    if any(left >= right for left, right in zip(buckets, buckets[1:])):
        raise ValueError("draft execution buckets must be strictly ascending and unique")
    if any(bucket != state_capacity and not _is_power_of_two(bucket) for bucket in buckets):
        raise ValueError("draft execution buckets must be powers of two or the custom state capacity")
    if any(bucket > state_capacity for bucket in buckets):
        raise ValueError("draft execution buckets must not exceed the draft state capacity")
    if state_capacity not in buckets:
        raise ValueError("draft execution buckets must include the draft state capacity")
    return buckets


@dataclass(frozen=True)
class DraftBucketPlan:
    """Resolved host-side draft execution layout.

    ``source_lanes`` always has exactly ``execution_capacity`` distinct entries.
    ``logical_slots`` uses ``-1`` for each inactive filler.  The graph key is
    deliberately independent of the selected lanes and their logical slots.
    """

    state_capacity: int
    execution_capacity: int
    source_lanes: tuple[int, ...]
    logical_slots: tuple[int, ...]

    def __post_init__(self):
        validate_dspark_capacity(self.state_capacity, allow_serial=True, name="draft state capacity")
        if (type(self.execution_capacity) is not int or not 1 <= self.execution_capacity <= self.state_capacity):
            raise ValueError("draft execution capacity must be an integer in 1..state capacity")

        source_lanes = _host_tuple(self.source_lanes, "draft source lanes")
        logical_slots = _host_tuple(self.logical_slots, "draft logical slots")
        if any(type(lane) is not int for lane in source_lanes):
            raise ValueError("draft source lanes must contain only integers")
        if any(type(slot) is not int for slot in logical_slots):
            raise ValueError("draft logical slots must contain only integers")
        if len(source_lanes) != self.execution_capacity:
            raise ValueError("draft source lanes must have exactly execution capacity entries")
        if len(logical_slots) != self.execution_capacity:
            raise ValueError("draft logical slots must have exactly execution capacity entries")
        if (len(set(source_lanes)) != len(source_lanes)
                or any(lane < 0 or lane >= self.state_capacity for lane in source_lanes)):
            raise ValueError("draft source lanes must be unique lanes in 0..state capacity-1")
        active_slots = tuple(slot for slot in logical_slots if slot >= 0)
        if (any(slot < -1 or slot >= self.state_capacity for slot in logical_slots)
                or len(set(active_slots)) != len(active_slots)):
            raise ValueError("draft logical slots must be -1 or unique slots in 0..state capacity-1")

        object.__setattr__(self, "source_lanes", source_lanes)
        object.__setattr__(self, "logical_slots", logical_slots)

    @property
    def graph_key(self):
        """Return the only values allowed to key the compiled draft graph."""
        return (self.state_capacity, self.execution_capacity)

    @property
    def execution_rows(self):
        """Return the fixed five draft rows per execution lane."""
        return _DRAFT_ROWS_PER_LANE * self.execution_capacity


def draft_execution_buckets(capacity, buckets):
    """Return sorted draft execution buckets for a 1..64 state capacity.

    Input buckets must be powers of two in 1..64.  The custom state capacity is
    appended when it is not itself a power of two, matching the target-stage
    bounded-bucket contract while using draft-specific validation messages.
    """
    capacity = validate_dspark_capacity(capacity, allow_serial=True, name="draft state capacity")
    if not isinstance(buckets, (list, tuple)) or not buckets:
        raise ValueError("draft execution buckets must be a nonempty list or tuple of powers of two")
    for bucket in buckets:
        if type(bucket) is not int or bucket < 1 or bucket > MAX_DSV41_DSPARK_CAPACITY or not _is_power_of_two(bucket):
            raise ValueError("draft execution buckets must be powers of two in 1..64")
    return tuple(sorted({bucket for bucket in buckets if bucket < capacity} | {capacity}))


def select_draft_bucket(active, request_slots, execution_buckets):
    """Select a draft execution bucket without any device readback.

    ``active`` is a host sequence of booleans and ``request_slots`` is the
    matching host sequence of integers.  Active logical slots must be unique and
    in ``0..N-1``; inactive lanes must use ``-1``.  Active lanes are packed
    first for a partial bucket, then distinct inactive lanes fill the remainder.
    A full bucket retains original identity-lane order, including the empty
    active case.
    """
    active = _host_tuple(active, "draft active lanes")
    request_slots = _host_tuple(request_slots, "draft request slots")
    state_capacity = len(active)
    validate_dspark_capacity(state_capacity, allow_serial=True, name="draft state capacity")
    if len(request_slots) != state_capacity:
        raise ValueError("draft active lanes and request slots must have the same length")
    if any(type(value) is not bool for value in active):
        raise ValueError("draft active lanes must contain only booleans")
    if any(type(slot) is not int for slot in request_slots):
        raise ValueError("draft request slots must contain only integers")

    active_lanes = []
    active_slots = set()
    for lane, (is_active, slot) in enumerate(zip(active, request_slots)):
        if is_active:
            if slot < 0 or slot >= state_capacity:
                raise ValueError("active draft request slots must be in 0..state capacity-1")
            if slot in active_slots:
                raise ValueError("active draft request slots must be unique")
            active_slots.add(slot)
            active_lanes.append(lane)
        elif slot != -1:
            raise ValueError("inactive draft request slots must be -1")

    buckets = _validate_execution_buckets(execution_buckets, state_capacity)
    active_count = len(active_lanes)
    if active_count == 0:
        execution_capacity = state_capacity
    else:
        execution_capacity = next(bucket for bucket in buckets if bucket >= active_count)
    if execution_capacity < active_count:
        raise ValueError("draft execution capacity must fit every active lane")

    if execution_capacity == state_capacity:
        source_lanes = tuple(range(state_capacity))
    else:
        inactive_lanes = tuple(lane for lane in range(state_capacity) if not active[lane])
        fill_count = execution_capacity - active_count
        source_lanes = tuple(active_lanes) + inactive_lanes[:fill_count]
    logical_slots = tuple(request_slots[lane] for lane in source_lanes)
    return DraftBucketPlan(state_capacity, execution_capacity, source_lanes, logical_slots)


__all__ = ["DraftBucketPlan", "draft_execution_buckets", "select_draft_bucket"]
