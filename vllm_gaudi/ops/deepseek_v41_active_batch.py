# SPDX-License-Identifier: Apache-2.0
"""Pure-host active metadata for the DSpark TP8 batch path.

This module separates two concepts that must not be conflated:

* a compact row is only this round's execution position; and
* a logical slot is the stable owner of model state.

Every active compact row therefore carries the identity
``(request_id, logical_slot, slot_generation, round_generation)``.  The
adapter keeps scheduler order separate from canonical logical-slot order and
can explicitly restore the legacy fixed-capacity
:class:`DSparkBatchMetadata` layout.  It does not execute a model, upload
tensors, or initialize an HPU runtime.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import operator

import torch

from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata
from vllm_gaudi.ops.deepseek_v41_capacity import validate_dspark_capacity

_TARGET_QUERY_BUCKET = 6
_MAX_DRAFT_TOKENS = 5
_INTEGER_DTYPES = (torch.int32, torch.int64)


def _as_int(value, name: str) -> int:
    if type(value) is bool:
        raise TypeError(f"{name} must contain integers")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must contain integers") from error


def _scalar_int(value, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU host tensor")
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must be an integer scalar")
        value = value.item()
    if type(value) is bool:
        raise TypeError(f"{name} must be an integer scalar")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer scalar") from error


def _sequence(value, name: str) -> list:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    try:
        return list(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence") from error


def _integer_values(value, name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU host tensor")
        if value.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"{name} must contain integers")
        values = value.tolist()
    else:
        values = _sequence(value, name)
    return [_as_int(item, name) for item in values]


def _bool_values(value, name: str) -> list[bool]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU host tensor")
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must contain boolean values")
        return value.tolist()
    values = _sequence(value, name)
    if any(type(item) is not bool for item in values):
        raise TypeError(f"{name} must contain boolean values")
    return values


def _tensor_values(value, name: str, *, dtype: torch.dtype | None = None) -> list:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU host tensor")
    if dtype is not None and value.dtype != dtype:
        expected = "boolean" if dtype == torch.bool else "integer"
        raise TypeError(f"{name} must contain {expected} values")
    return value.tolist()


def _active_values(metadata: DSparkBatchMetadata, batch_size: int) -> list[bool]:
    raw = getattr(metadata, "request_active", None)
    if raw is None:
        return [True] * batch_size
    if isinstance(raw, torch.Tensor):
        if raw.numel() == 0:
            return [True] * batch_size
        active = _bool_values(raw, "request_active")
    else:
        active = _bool_values(raw, "request_active")
        if not active:
            return [True] * batch_size
    if len(active) != batch_size:
        raise ValueError(f"request_active length {len(active)} does not match batch size {batch_size}")
    return active


def _starts(lengths: Sequence[int]) -> list[int]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def _cumulative(values: Sequence[int]) -> list[int]:
    result = []
    total = 0
    for value in values:
        total += value
        result.append(total)
    return result


def _int_tensor(values: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(list(values), dtype=dtype, device="cpu")


@dataclass(frozen=True)
class DSparkActiveIdentity:
    """Stable ownership identity for one active compact row."""

    request_id: str
    logical_slot: int
    slot_generation: int
    round_generation: int

    def as_tuple(self) -> tuple[str, int, int, int]:
        return (self.request_id, self.logical_slot, self.slot_generation, self.round_generation)


@dataclass(frozen=True)
class _FixedLayout:
    request_ids: tuple[str | None, ...]
    active: tuple[bool, ...]
    slots: tuple[int, ...]
    generations: tuple[int, ...]
    target_lengths: tuple[int, ...]
    target_starts: tuple[int, ...]
    draft_counts: tuple[int, ...]
    draft_starts: tuple[int, ...]
    budgets: tuple[int, ...]
    target_positions: tuple[int, ...]
    target_padding: tuple[bool, ...]
    draft_padding: tuple[bool, ...]
    draft_ids: tuple[int, ...]
    round_generation: int

    @property
    def capacity(self) -> int:
        return len(self.request_ids)

    @property
    def active_lanes(self) -> tuple[int, ...]:
        return tuple(lane for lane, enabled in enumerate(self.active) if enabled)


def _read_fixed_layout(metadata: DSparkBatchMetadata) -> _FixedLayout:
    if not isinstance(metadata, DSparkBatchMetadata):
        raise TypeError("metadata must be a DSparkBatchMetadata instance")
    request_ids = tuple(getattr(metadata, "request_ids", ()))
    capacity = len(request_ids)
    validate_dspark_capacity(capacity, allow_serial=True, name="DSpark active metadata capacity")
    active = tuple(_active_values(metadata, capacity))

    starts = _integer_values(metadata.target_query_start_loc, "target_query_start_loc")
    if len(starts) != capacity + 1 or starts[0] != 0:
        raise ValueError("target_query_start_loc must contain capacity + 1 entries starting at zero")
    target_lengths = tuple(right - left for left, right in zip(starts[:-1], starts[1:], strict=True))
    if any(length < 0 for length in target_lengths):
        raise ValueError("target query lengths must be nonnegative")
    for lane, (enabled, length) in enumerate(zip(active, target_lengths, strict=True)):
        if enabled and not 1 <= length <= _TARGET_QUERY_BUCKET:
            raise ValueError(f"active target query length for lane {lane} must be in 1..6")
        if not enabled and length != 0:
            raise ValueError(f"inactive target query length for lane {lane} must be zero")

    draft_counts = tuple(_integer_values(metadata.num_draft_tokens, "num_draft_tokens"))
    if len(draft_counts) != capacity or any(not 0 <= count <= _MAX_DRAFT_TOKENS for count in draft_counts):
        raise ValueError("num_draft_tokens must contain capacity values in 0..5")
    draft_starts = tuple(_integer_values(metadata.draft_query_start_loc, "draft_query_start_loc"))
    if draft_starts != tuple(_starts(draft_counts)):
        raise ValueError("draft_query_start_loc is inconsistent with num_draft_tokens")
    if _integer_values(metadata.cu_num_draft_tokens, "cu_num_draft_tokens") != _cumulative(draft_counts):
        raise ValueError("cu_num_draft_tokens is inconsistent with num_draft_tokens")

    sampled_cumulative = []
    target_logits_expected = []
    bonus_logits_expected = []
    sampled_total = 0
    for count, enabled in zip(draft_counts, active, strict=True):
        if enabled:
            target_logits_expected.extend(range(sampled_total, sampled_total + count))
            bonus_logits_expected.append(sampled_total + count)
            sampled_total += count + 1
        sampled_cumulative.append(sampled_total)
    if _integer_values(metadata.cu_num_sampled_tokens, "cu_num_sampled_tokens") != sampled_cumulative:
        raise ValueError("cu_num_sampled_tokens is inconsistent with active draft counts")
    if _integer_values(metadata.target_logits_indices, "target_logits_indices") != target_logits_expected:
        raise ValueError("target_logits_indices is inconsistent with active draft counts")
    if _integer_values(metadata.bonus_logits_indices, "bonus_logits_indices") != bonus_logits_expected:
        raise ValueError("bonus_logits_indices is inconsistent with active draft counts")

    logits_expected = []
    for lane, (enabled, length, count) in enumerate(
            zip(active, target_lengths, draft_counts, strict=True)):
        if enabled:
            selected_start = starts[lane] + length - (count + 1)
            if selected_start < starts[lane]:
                raise ValueError(f"lane {lane} has fewer target rows than its selected tail")
            logits_expected.extend(range(selected_start, starts[lane] + length))
        if not enabled and count != 0:
            raise ValueError(f"inactive lane {lane} must not contain draft tokens")
        if enabled and length < count + 1:
            raise ValueError(f"lane {lane} target rows do not cover its selected tail")
    if _integer_values(metadata.logits_indices, "logits_indices") != logits_expected:
        raise ValueError("logits_indices is inconsistent with target lengths and draft counts")

    target_padding = tuple(_tensor_values(metadata.target_padding_mask, "target_padding_mask", dtype=torch.bool))
    if len(target_padding) != starts[-1]:
        raise ValueError("target_padding_mask length must match packed target rows")
    if any(target_padding[index] for index in logits_expected):
        raise ValueError("selected target logits rows must not be padding")
    draft_padding = tuple(_tensor_values(metadata.draft_padding_mask, "draft_padding_mask", dtype=torch.bool))
    if len(draft_padding) != draft_starts[-1]:
        raise ValueError("draft_padding_mask length must match packed draft rows")
    draft_ids = tuple(_integer_values(metadata.draft_token_ids, "draft_token_ids"))
    if len(draft_ids) != draft_starts[-1] or any(token < 0 for token in draft_ids):
        raise ValueError("draft_token_ids must contain nonnegative values matching draft counts")

    slots = tuple(_integer_values(metadata.request_slots, "request_slots"))
    generations = tuple(_integer_values(metadata.slot_generations, "slot_generations"))
    budgets = tuple(_integer_values(metadata.remaining_output_budget, "remaining_output_budget"))
    positions = tuple(_integer_values(metadata.target_start_positions, "target_start_positions"))
    for name, values in (
        ("request_slots", slots),
        ("slot_generations", generations),
        ("remaining_output_budget", budgets),
        ("target_start_positions", positions),
    ):
        if len(values) != capacity:
            raise ValueError(f"{name} length {len(values)} does not match capacity {capacity}")

    active_ids = []
    active_slots = []
    for lane, (request_id, enabled) in enumerate(zip(request_ids, active, strict=True)):
        if enabled:
            if not isinstance(request_id, str):
                raise TypeError(f"active request_ids[{lane}] must be a string")
            active_ids.append(request_id)
            if not 0 <= slots[lane] < capacity:
                raise ValueError(f"active logical slot for lane {lane} must be in 0..{capacity - 1}")
            active_slots.append(slots[lane])
            if generations[lane] < 0:
                raise ValueError(f"active slot generation for lane {lane} must be nonnegative")
            if budgets[lane] < 0:
                raise ValueError(f"active remaining output budget for lane {lane} must be nonnegative")
            if positions[lane] < 0:
                raise ValueError(f"active target start position for lane {lane} must be nonnegative")
        else:
            if request_id is not None:
                raise TypeError(f"inactive request_ids[{lane}] must be None")
            if slots[lane] != -1 or generations[lane] != 0:
                raise ValueError(f"inactive lane {lane} must use the (-1, 0) slot sentinel")
            if budgets[lane] != 0 or positions[lane] != 0:
                raise ValueError(f"inactive lane {lane} must use zero budget and target start")
    if len(set(active_ids)) != len(active_ids):
        raise ValueError("active request IDs must be unique")
    if len(set(active_slots)) != len(active_slots):
        raise ValueError("active logical slots must be unique")

    round_generation = _scalar_int(metadata.round_generation, "round_generation")
    if round_generation < 0:
        raise ValueError("round_generation must be nonnegative")

    return _FixedLayout(
        request_ids=request_ids,
        active=active,
        slots=slots,
        generations=generations,
        target_lengths=target_lengths,
        target_starts=starts,
        draft_counts=draft_counts,
        draft_starts=draft_starts,
        budgets=budgets,
        target_positions=positions,
        target_padding=target_padding,
        draft_padding=draft_padding,
        draft_ids=draft_ids,
        round_generation=round_generation,
    )


def _scheduler_lanes(layout: _FixedLayout, scheduler_order) -> tuple[int, ...]:
    if scheduler_order is None:
        return tuple(sorted(layout.active_lanes, key=lambda lane: layout.slots[lane]))
    if isinstance(scheduler_order, Mapping):
        scheduler_order = tuple(scheduler_order.keys())
    order = tuple(_sequence(scheduler_order, "scheduler_order"))
    if not order:
        raise ValueError("scheduler_order must contain every active request")

    lane_by_request = {
        layout.request_ids[lane]: lane
        for lane in layout.active_lanes
    }
    lanes = []
    seen = set()
    for item in order:
        if isinstance(item, DSparkActiveIdentity):
            request_id = item.request_id
        elif isinstance(item, str):
            request_id = item
        else:
            raise TypeError("scheduler_order entries must be request ID strings")
        if request_id in seen:
            raise ValueError(f"scheduler_order contains duplicate request {request_id!r}")
        lane = lane_by_request.get(request_id)
        if lane is None:
            raise ValueError(f"scheduler_order request {request_id!r} is not active in metadata")
        if isinstance(item, DSparkActiveIdentity):
            expected = DSparkActiveIdentity(
                request_id,
                layout.slots[lane],
                layout.generations[lane],
                layout.round_generation,
            )
            if item != expected:
                raise RuntimeError(f"scheduler identity for request {request_id!r} is stale")
        lanes.append(lane)
        seen.add(request_id)
    if len(lanes) != len(lane_by_request):
        missing = sorted(set(lane_by_request) - seen)
        raise ValueError(f"scheduler_order is missing active requests: {missing}")
    return tuple(lanes)


@dataclass(frozen=True)
class DSparkActiveBatchMetadata:
    """Compact host metadata for ``A`` active rows within capacity ``N``.

    ``request_ids`` and all request-axis tensors are in scheduler order.  The
    explicit maps distinguish compact rows, fixed/canonical lanes, and stable
    logical slots.  No inactive lane is represented as an active row.
    """

    capacity: int
    request_ids: tuple[str, ...]
    request_active: torch.Tensor
    request_slots: torch.Tensor
    slot_generations: torch.Tensor
    target_query_start_loc: torch.Tensor
    draft_query_start_loc: torch.Tensor
    num_draft_tokens: torch.Tensor
    cu_num_draft_tokens: torch.Tensor
    cu_num_sampled_tokens: torch.Tensor
    draft_token_ids: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor
    logits_indices: torch.Tensor
    remaining_output_budget: torch.Tensor
    target_padding_mask: torch.Tensor
    draft_padding_mask: torch.Tensor
    target_start_positions: torch.Tensor
    round_generation: torch.Tensor
    compact_to_logical_slots: tuple[int, ...]
    compact_to_fixed_lanes: tuple[int, ...]
    logical_slot_to_compact_rows: tuple[int, ...]
    fixed_lane_to_compact_rows: tuple[int, ...]
    canonical_compact_rows: tuple[int, ...]

    @property
    def active_count(self) -> int:
        return len(self.request_ids)

    @property
    def num_active(self) -> int:
        return self.active_count

    @property
    def compact_capacity(self) -> int:
        return self.active_count

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return self.request_ids

    @property
    def scheduler_request_ids(self) -> tuple[str, ...]:
        return self.request_ids

    @property
    def scheduler_order(self) -> tuple[str, ...]:
        return self.request_ids

    @property
    def compact_row_to_logical_slot(self) -> tuple[int, ...]:
        return self.compact_to_logical_slots

    @property
    def logical_slot_to_compact_row(self) -> tuple[int, ...]:
        return self.logical_slot_to_compact_rows

    @property
    def compact_row_to_fixed_lane(self) -> tuple[int, ...]:
        return self.compact_to_fixed_lanes

    @property
    def fixed_lane_to_compact_row(self) -> tuple[int, ...]:
        return self.fixed_lane_to_compact_rows

    @property
    def scheduler_slots(self) -> tuple[int, ...]:
        return tuple(self.request_slots.tolist())

    @property
    def canonical_slots(self) -> tuple[int, ...]:
        return tuple(self.compact_to_logical_slots[row] for row in self.canonical_compact_rows)

    @property
    def canonical_request_ids(self) -> tuple[str, ...]:
        return tuple(self.request_ids[row] for row in self.canonical_compact_rows)

    @property
    def canonical_order(self) -> tuple[str, ...]:
        return self.canonical_request_ids

    @property
    def active_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.capacity, dtype=torch.bool, device="cpu")
        for lane in self.compact_to_fixed_lanes:
            mask[lane] = True
        return mask

    @property
    def inactive_mask(self) -> torch.Tensor:
        return ~self.active_mask

    @property
    def active_fixed_lanes(self) -> tuple[int, ...]:
        return tuple(self.compact_to_fixed_lanes)

    @property
    def inactive_fixed_lanes(self) -> tuple[int, ...]:
        return tuple(lane for lane, row in enumerate(self.fixed_lane_to_compact_rows) if row < 0)

    @property
    def round_generation_value(self) -> int:
        return int(self.round_generation.item())

    @property
    def identities(self) -> tuple[DSparkActiveIdentity, ...]:
        round_generation = self.round_generation_value
        return tuple(
            DSparkActiveIdentity(request_id, slot, generation, round_generation)
            for request_id, slot, generation in zip(
                self.request_ids,
                self.scheduler_slots,
                self.slot_generations.tolist(),
                strict=True,
            )
        )

    @property
    def target_query_lengths(self) -> tuple[int, ...]:
        starts = self.target_query_start_loc.tolist()
        return tuple(right - left for left, right in zip(starts[:-1], starts[1:], strict=True))

    @property
    def fixed_request_ids(self) -> tuple[str | None, ...]:
        result: list[str | None] = [None] * self.capacity
        for row, lane in enumerate(self.compact_to_fixed_lanes):
            result[lane] = self.request_ids[row]
        return tuple(result)

    @property
    def no_write_mask(self) -> torch.Tensor:
        """Return the fixed ``[N, 6]`` target mask, ``True`` for no-write."""
        mask = torch.ones((self.capacity, _TARGET_QUERY_BUCKET), dtype=torch.bool, device="cpu")
        target_starts = self.target_query_start_loc.tolist()
        for row, lane in enumerate(self.compact_to_fixed_lanes):
            start, end = target_starts[row], target_starts[row + 1]
            length = end - start
            mask[lane, :length] = self.target_padding_mask[start:end]
        return mask

    def identity_for_compact_row(self, compact_row: int) -> DSparkActiveIdentity:
        row = _as_int(compact_row, "compact_row")
        if not 0 <= row < self.active_count:
            raise IndexError("compact row is outside the active batch")
        return self.identities[row]

    def identity_for_request(self, request_id: str) -> DSparkActiveIdentity:
        try:
            row = self.request_ids.index(request_id)
        except ValueError as error:
            raise KeyError(f"request {request_id!r} is not active") from error
        return self.identities[row]

    def identity_for_logical_slot(self, logical_slot: int) -> DSparkActiveIdentity:
        slot = _as_int(logical_slot, "logical_slot")
        if not 0 <= slot < self.capacity or self.logical_slot_to_compact_rows[slot] < 0:
            raise KeyError(f"logical slot {slot} is not active")
        return self.identities[self.logical_slot_to_compact_rows[slot]]

    def compact_row_for_request(self, request_id: str) -> int:
        try:
            return self.request_ids.index(request_id)
        except ValueError as error:
            raise KeyError(f"request {request_id!r} is not active") from error

    def compact_row_for_logical_slot(self, logical_slot: int) -> int:
        slot = _as_int(logical_slot, "logical_slot")
        if not 0 <= slot < self.capacity or self.logical_slot_to_compact_rows[slot] < 0:
            raise KeyError(f"logical slot {slot} is not active")
        return self.logical_slot_to_compact_rows[slot]

    def validate_owners(
        self,
        owners,
        *,
        expected_round_generation: int | None = None,
        minimum_round_generation: int | None = None,
    ) -> bool:
        return validate_dspark_active_identities(
            self,
            owners,
            expected_round_generation=expected_round_generation,
            minimum_round_generation=minimum_round_generation,
        )

    def to_fixed_metadata(self) -> DSparkBatchMetadata:
        return restore_fixed_dspark_metadata(self)

    def to_fixed(self) -> DSparkBatchMetadata:
        return restore_fixed_dspark_metadata(self)

    def restore_fixed_metadata(self) -> DSparkBatchMetadata:
        return restore_fixed_dspark_metadata(self)

    def as_fixed_metadata(self) -> DSparkBatchMetadata:
        return restore_fixed_dspark_metadata(self)

    @classmethod
    def from_fixed_metadata(cls, metadata: DSparkBatchMetadata, **kwargs) -> "DSparkActiveBatchMetadata":
        return compact_dspark_metadata(metadata, **kwargs)

    @classmethod
    def from_fixed(cls, metadata: DSparkBatchMetadata, **kwargs) -> "DSparkActiveBatchMetadata":
        return compact_dspark_metadata(metadata, **kwargs)


def _validate_compact_layout(metadata: DSparkActiveBatchMetadata) -> None:
    if not isinstance(metadata, DSparkActiveBatchMetadata):
        raise TypeError("metadata must be a DSparkActiveBatchMetadata instance")
    validate_dspark_capacity(metadata.capacity, allow_serial=True, name="DSpark active metadata capacity")
    active_count = len(metadata.request_ids)
    if active_count < 1:
        raise ValueError("active metadata must contain at least one compact row")
    if len(set(metadata.request_ids)) != active_count or any(
            not isinstance(request_id, str) for request_id in metadata.request_ids):
        raise ValueError("active request IDs must be unique strings")

    active = _tensor_values(metadata.request_active, "request_active", dtype=torch.bool)
    if active != [True] * active_count:
        raise ValueError("compact active metadata must not contain inactive rows")
    slots = tuple(_integer_values(metadata.request_slots, "request_slots"))
    if slots != tuple(metadata.compact_to_logical_slots):
        raise ValueError("compact_to_logical_slots disagrees with request_slots")
    if len(slots) != active_count or len(set(slots)) != active_count or any(
            not 0 <= slot < metadata.capacity for slot in slots):
        raise ValueError("active logical slots must be unique and inside fixed capacity")

    generations = tuple(_integer_values(metadata.slot_generations, "slot_generations"))
    budgets = tuple(_integer_values(metadata.remaining_output_budget, "remaining_output_budget"))
    positions = tuple(_integer_values(metadata.target_start_positions, "target_start_positions"))
    for name, values in (
        ("slot_generations", generations),
        ("remaining_output_budget", budgets),
        ("target_start_positions", positions),
    ):
        if len(values) != active_count:
            raise ValueError(f"{name} length {len(values)} does not match active count {active_count}")
    if any(value < 0 for value in generations):
        raise ValueError("active slot generations must be nonnegative")
    if any(value < 0 for value in budgets):
        raise ValueError("remaining output budgets must be nonnegative")
    if any(value < 0 for value in positions):
        raise ValueError("target start positions must be nonnegative")

    target_starts = tuple(_integer_values(metadata.target_query_start_loc, "target_query_start_loc"))
    draft_starts = tuple(_integer_values(metadata.draft_query_start_loc, "draft_query_start_loc"))
    if len(target_starts) != active_count + 1 or target_starts[0] != 0:
        raise ValueError("target_query_start_loc must contain active_count + 1 entries starting at zero")
    if len(draft_starts) != active_count + 1 or draft_starts[0] != 0:
        raise ValueError("draft_query_start_loc must contain active_count + 1 entries starting at zero")
    target_lengths = tuple(right - left for left, right in zip(target_starts[:-1], target_starts[1:], strict=True))
    if any(not 1 <= length <= _TARGET_QUERY_BUCKET for length in target_lengths):
        raise ValueError("active target query lengths must be in 1..6")

    draft_counts = tuple(_integer_values(metadata.num_draft_tokens, "num_draft_tokens"))
    if len(draft_counts) != active_count or any(not 0 <= count <= _MAX_DRAFT_TOKENS for count in draft_counts):
        raise ValueError("num_draft_tokens must contain active_count values in 0..5")
    if draft_starts != tuple(_starts(draft_counts)):
        raise ValueError("draft_query_start_loc is inconsistent with num_draft_tokens")
    if any(length < count + 1 for length, count in zip(target_lengths, draft_counts, strict=True)):
        raise ValueError("each target row span must cover its selected draft tail")
    if tuple(_integer_values(metadata.cu_num_draft_tokens, "cu_num_draft_tokens")) != tuple(
            _cumulative(draft_counts)):
        raise ValueError("cu_num_draft_tokens is inconsistent with num_draft_tokens")

    sampled_cumulative = []
    target_logits = []
    bonus_logits = []
    sampled_total = 0
    for count in draft_counts:
        target_logits.extend(range(sampled_total, sampled_total + count))
        bonus_logits.append(sampled_total + count)
        sampled_total += count + 1
        sampled_cumulative.append(sampled_total)
    if tuple(_integer_values(metadata.cu_num_sampled_tokens, "cu_num_sampled_tokens")) != tuple(
            sampled_cumulative):
        raise ValueError("cu_num_sampled_tokens is inconsistent with draft counts")
    if tuple(_integer_values(metadata.target_logits_indices, "target_logits_indices")) != tuple(target_logits):
        raise ValueError("target_logits_indices is inconsistent with draft counts")
    if tuple(_integer_values(metadata.bonus_logits_indices, "bonus_logits_indices")) != tuple(bonus_logits):
        raise ValueError("bonus_logits_indices is inconsistent with draft counts")

    logits = []
    for row, (start, end, count) in enumerate(zip(target_starts[:-1], target_starts[1:], draft_counts, strict=True)):
        selected_start = end - (count + 1)
        if selected_start < start:
            raise ValueError(f"compact row {row} has fewer target rows than its selected tail")
        logits.extend(range(selected_start, end))
    if tuple(_integer_values(metadata.logits_indices, "logits_indices")) != tuple(logits):
        raise ValueError("logits_indices is inconsistent with compact target spans")

    target_padding = tuple(_tensor_values(metadata.target_padding_mask, "target_padding_mask", dtype=torch.bool))
    if len(target_padding) != target_starts[-1]:
        raise ValueError("target_padding_mask length must match compact target rows")
    if any(target_padding[index] for index in logits):
        raise ValueError("selected compact target logits rows must not be padding")
    draft_padding = tuple(_tensor_values(metadata.draft_padding_mask, "draft_padding_mask", dtype=torch.bool))
    if len(draft_padding) != draft_starts[-1]:
        raise ValueError("draft_padding_mask length must match compact draft rows")
    draft_ids = tuple(_integer_values(metadata.draft_token_ids, "draft_token_ids"))
    if len(draft_ids) != draft_starts[-1] or any(token < 0 for token in draft_ids):
        raise ValueError("draft_token_ids must contain nonnegative values matching compact draft counts")

    compact_to_fixed = tuple(metadata.compact_to_fixed_lanes)
    fixed_to_compact = tuple(metadata.fixed_lane_to_compact_rows)
    if len(compact_to_fixed) != active_count or len(set(compact_to_fixed)) != active_count:
        raise ValueError("compact_to_fixed_lanes must map each compact row to one fixed lane")
    if any(not 0 <= lane < metadata.capacity for lane in compact_to_fixed):
        raise ValueError("compact_to_fixed_lanes contains an out-of-range lane")
    if len(fixed_to_compact) != metadata.capacity:
        raise ValueError("fixed_lane_to_compact_rows must contain one entry per fixed lane")
    expected_fixed = [-1] * metadata.capacity
    for row, lane in enumerate(compact_to_fixed):
        expected_fixed[lane] = row
    if fixed_to_compact != tuple(expected_fixed):
        raise ValueError("fixed_lane_to_compact_rows is not the inverse compact-to-fixed map")

    logical_to_compact = tuple(metadata.logical_slot_to_compact_rows)
    if len(logical_to_compact) != metadata.capacity:
        raise ValueError("logical_slot_to_compact_rows must contain one entry per logical slot")
    expected_logical = [-1] * metadata.capacity
    for row, slot in enumerate(slots):
        expected_logical[slot] = row
    if logical_to_compact != tuple(expected_logical):
        raise ValueError("logical_slot_to_compact_rows is not the inverse compact-to-logical map")
    if tuple(metadata.canonical_compact_rows) != tuple(sorted(range(active_count), key=lambda row: slots[row])):
        raise ValueError("canonical_compact_rows must follow ascending logical-slot order")

    round_generation = _scalar_int(metadata.round_generation, "round_generation")
    if round_generation < 0:
        raise ValueError("round_generation must be nonnegative")
    tensors = (
        metadata.request_active,
        metadata.request_slots,
        metadata.slot_generations,
        metadata.target_query_start_loc,
        metadata.draft_query_start_loc,
        metadata.num_draft_tokens,
        metadata.cu_num_draft_tokens,
        metadata.cu_num_sampled_tokens,
        metadata.draft_token_ids,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        metadata.logits_indices,
        metadata.remaining_output_budget,
        metadata.target_padding_mask,
        metadata.draft_padding_mask,
        metadata.target_start_positions,
        metadata.round_generation,
    )
    if any(not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("all active metadata tensors must be CPU host tensors")


def compact_dspark_metadata(
    metadata: DSparkBatchMetadata,
    *,
    scheduler_order=None,
    owners=None,
    expected_round_generation: int | None = None,
    minimum_round_generation: int | None = None,
) -> DSparkActiveBatchMetadata:
    """Create compact active metadata from legacy fixed-capacity metadata.

    ``scheduler_order`` contains request ID strings in this round's scheduling
    order.  If omitted, rows use canonical ascending logical-slot order.  The
    returned object retains both maps needed for an exact fixed-layout
    round-trip: compact row to logical slot and compact row to fixed lane.
    """
    layout = _read_fixed_layout(metadata)
    compact_lanes = _scheduler_lanes(layout, scheduler_order)
    active_count = len(compact_lanes)

    compact_target_lengths = [layout.target_lengths[lane] for lane in compact_lanes]
    compact_draft_counts = [layout.draft_counts[lane] for lane in compact_lanes]
    compact_target_starts = _starts(compact_target_lengths)
    compact_draft_starts = _starts(compact_draft_counts)

    target_padding = []
    draft_padding = []
    draft_ids = []
    logits = []
    target_logits = []
    bonus_logits = []
    sampled_total = 0
    for row, lane in enumerate(compact_lanes):
        target_start = layout.target_starts[lane]
        target_end = layout.target_starts[lane + 1]
        target_padding.extend(layout.target_padding[target_start:target_end])
        draft_start = layout.draft_starts[lane]
        draft_end = layout.draft_starts[lane + 1]
        draft_padding.extend(layout.draft_padding[draft_start:draft_end])
        draft_ids.extend(layout.draft_ids[draft_start:draft_end])

        selected_start = compact_target_starts[row + 1] - (compact_draft_counts[row] + 1)
        logits.extend(range(selected_start, compact_target_starts[row + 1]))
        target_logits.extend(range(sampled_total, sampled_total + compact_draft_counts[row]))
        bonus_logits.append(sampled_total + compact_draft_counts[row])
        sampled_total += compact_draft_counts[row] + 1

    compact_to_slots = tuple(layout.slots[lane] for lane in compact_lanes)
    fixed_to_compact = [-1] * layout.capacity
    for row, lane in enumerate(compact_lanes):
        fixed_to_compact[lane] = row
    logical_to_compact = [-1] * layout.capacity
    for row, slot in enumerate(compact_to_slots):
        logical_to_compact[slot] = row
    canonical_rows = tuple(sorted(range(active_count), key=lambda row: compact_to_slots[row]))

    result = DSparkActiveBatchMetadata(
        capacity=layout.capacity,
        request_ids=tuple(layout.request_ids[lane] for lane in compact_lanes),
        request_active=torch.ones(active_count, dtype=torch.bool, device="cpu"),
        request_slots=_int_tensor(compact_to_slots, torch.int32),
        slot_generations=_int_tensor([layout.generations[lane] for lane in compact_lanes], torch.int64),
        target_query_start_loc=_int_tensor(compact_target_starts, torch.int32),
        draft_query_start_loc=_int_tensor(compact_draft_starts, torch.int32),
        num_draft_tokens=_int_tensor(compact_draft_counts, torch.int32),
        cu_num_draft_tokens=_int_tensor(_cumulative(compact_draft_counts), torch.int32),
        cu_num_sampled_tokens=_int_tensor(_cumulative([count + 1 for count in compact_draft_counts]),
                                          torch.int32),
        draft_token_ids=_int_tensor(draft_ids, torch.int32),
        target_logits_indices=_int_tensor(target_logits, torch.int32),
        bonus_logits_indices=_int_tensor(bonus_logits, torch.int32),
        logits_indices=_int_tensor(logits, torch.int32),
        remaining_output_budget=_int_tensor([layout.budgets[lane] for lane in compact_lanes], torch.int32),
        target_padding_mask=_int_tensor(target_padding, torch.bool),
        draft_padding_mask=_int_tensor(draft_padding, torch.bool),
        target_start_positions=_int_tensor([layout.target_positions[lane] for lane in compact_lanes], torch.int64),
        round_generation=torch.tensor(layout.round_generation, dtype=torch.int64, device="cpu"),
        compact_to_logical_slots=compact_to_slots,
        compact_to_fixed_lanes=tuple(compact_lanes),
        logical_slot_to_compact_rows=tuple(logical_to_compact),
        fixed_lane_to_compact_rows=tuple(fixed_to_compact),
        canonical_compact_rows=canonical_rows,
    )
    _validate_compact_layout(result)
    if owners is not None:
        result.validate_owners(
            owners,
            expected_round_generation=expected_round_generation,
            minimum_round_generation=minimum_round_generation,
        )
    elif expected_round_generation is not None or minimum_round_generation is not None:
        round_generation = result.round_generation_value
        if expected_round_generation is not None:
            expected = _as_int(expected_round_generation, "expected_round_generation")
            if round_generation != expected:
                raise RuntimeError(
                    f"DSpark active metadata round {round_generation} is stale; expected {expected}")
        if minimum_round_generation is not None:
            minimum = _as_int(minimum_round_generation, "minimum_round_generation")
            if round_generation < minimum:
                raise RuntimeError(
                    f"DSpark active metadata round {round_generation} is expired; minimum is {minimum}")
    return result


def _copy_slice(values: Sequence, start: int, end: int) -> list:
    return list(values[start:end])


def restore_fixed_dspark_metadata(
    metadata: DSparkActiveBatchMetadata,
    *,
    capacity: int | None = None,
) -> DSparkBatchMetadata:
    """Explicitly restore the legacy fixed-capacity ``DSparkBatchMetadata``.

    Inactive lanes are restored as ``(None, -1, 0)`` with zero target rows,
    zero drafts, zero budget, and target start zero.  Existing fixed-lane
    ordering and every tensor-valued metadata field are reconstructed.
    """
    _validate_compact_layout(metadata)
    fixed_capacity = metadata.capacity if capacity is None else _as_int(capacity, "capacity")
    if fixed_capacity != metadata.capacity:
        raise ValueError(
            f"requested fixed capacity {fixed_capacity} does not match active metadata capacity {metadata.capacity}"
        )

    target_lengths = metadata.target_query_lengths
    draft_counts = tuple(metadata.num_draft_tokens.tolist())
    target_starts = tuple(metadata.target_query_start_loc.tolist())
    draft_starts = tuple(metadata.draft_query_start_loc.tolist())
    target_padding = tuple(metadata.target_padding_mask.tolist())
    draft_padding = tuple(metadata.draft_padding_mask.tolist())
    draft_ids = tuple(metadata.draft_token_ids.tolist())
    budgets = tuple(metadata.remaining_output_budget.tolist())
    positions = tuple(metadata.target_start_positions.tolist())
    generations = tuple(metadata.slot_generations.tolist())

    fixed_request_ids: list[str | None] = [None] * fixed_capacity
    fixed_active = [False] * fixed_capacity
    fixed_slots = [-1] * fixed_capacity
    fixed_generations = [0] * fixed_capacity
    fixed_budgets = [0] * fixed_capacity
    fixed_positions = [0] * fixed_capacity
    fixed_target_lengths = [0] * fixed_capacity
    fixed_draft_counts = [0] * fixed_capacity
    target_padding_by_lane: list[list[bool] | None] = [None] * fixed_capacity
    draft_padding_by_lane: list[list[bool] | None] = [None] * fixed_capacity
    draft_ids_by_lane: list[list[int] | None] = [None] * fixed_capacity

    for row, lane in enumerate(metadata.compact_to_fixed_lanes):
        fixed_request_ids[lane] = metadata.request_ids[row]
        fixed_active[lane] = True
        fixed_slots[lane] = metadata.compact_to_logical_slots[row]
        fixed_generations[lane] = generations[row]
        fixed_budgets[lane] = budgets[row]
        fixed_positions[lane] = positions[row]
        fixed_target_lengths[lane] = target_lengths[row]
        fixed_draft_counts[lane] = draft_counts[row]
        target_padding_by_lane[lane] = _copy_slice(
            target_padding, target_starts[row], target_starts[row + 1])
        draft_padding_by_lane[lane] = _copy_slice(
            draft_padding, draft_starts[row], draft_starts[row + 1])
        draft_ids_by_lane[lane] = _copy_slice(draft_ids, draft_starts[row], draft_starts[row + 1])

    fixed_target_padding = []
    fixed_draft_padding = []
    fixed_draft_ids = []
    for lane in range(fixed_capacity):
        if target_padding_by_lane[lane] is not None:
            fixed_target_padding.extend(target_padding_by_lane[lane])
        if draft_padding_by_lane[lane] is not None:
            fixed_draft_padding.extend(draft_padding_by_lane[lane])
        if draft_ids_by_lane[lane] is not None:
            fixed_draft_ids.extend(draft_ids_by_lane[lane])

    fixed_target_starts = _starts(fixed_target_lengths)
    fixed_draft_starts = _starts(fixed_draft_counts)
    fixed_logits = []
    fixed_target_logits = []
    fixed_bonus_logits = []
    sampled_total = 0
    for lane in range(fixed_capacity):
        if not fixed_active[lane]:
            continue
        target_start = fixed_target_starts[lane]
        target_end = fixed_target_starts[lane + 1]
        selected_start = target_end - (fixed_draft_counts[lane] + 1)
        fixed_logits.extend(range(selected_start, target_end))
        fixed_target_logits.extend(range(sampled_total, sampled_total + fixed_draft_counts[lane]))
        fixed_bonus_logits.append(sampled_total + fixed_draft_counts[lane])
        sampled_total += fixed_draft_counts[lane] + 1

    return DSparkBatchMetadata(
        request_ids=tuple(fixed_request_ids),
        request_active=_int_tensor(fixed_active, torch.bool),
        request_slots=_int_tensor(fixed_slots, torch.int32),
        slot_generations=_int_tensor(fixed_generations, torch.int64),
        target_query_start_loc=_int_tensor(fixed_target_starts, torch.int32),
        draft_query_start_loc=_int_tensor(fixed_draft_starts, torch.int32),
        num_draft_tokens=_int_tensor(fixed_draft_counts, torch.int32),
        cu_num_draft_tokens=_int_tensor(_cumulative(fixed_draft_counts), torch.int32),
        cu_num_sampled_tokens=_int_tensor(
            [sum(count + 1 for count, enabled in zip(fixed_draft_counts[:lane + 1],
                                                       fixed_active[:lane + 1], strict=True) if enabled)
             for lane in range(fixed_capacity)],
            torch.int32,
        ),
        draft_token_ids=_int_tensor(fixed_draft_ids, torch.int32),
        target_logits_indices=_int_tensor(fixed_target_logits, torch.int32),
        bonus_logits_indices=_int_tensor(fixed_bonus_logits, torch.int32),
        logits_indices=_int_tensor(fixed_logits, torch.int32),
        remaining_output_budget=_int_tensor(fixed_budgets, torch.int32),
        target_padding_mask=_int_tensor(fixed_target_padding, torch.bool),
        draft_padding_mask=_int_tensor(fixed_draft_padding, torch.bool),
        target_start_positions=_int_tensor(fixed_positions, torch.int64),
        round_generation=metadata.round_generation.clone(),
    )


def _normalize_owners(owners) -> dict[str, tuple[int, int]]:
    if owners is None:
        raise TypeError("owners must name every active request")
    if isinstance(owners, DSparkActiveGenerationTracker):
        owners = owners.owners
    if isinstance(owners, Mapping):
        result = {}
        for request_id, owner in owners.items():
            if not isinstance(request_id, str):
                raise TypeError("owner request IDs must be strings")
            if isinstance(owner, DSparkActiveIdentity):
                result[request_id] = (owner.logical_slot, owner.slot_generation)
            elif isinstance(owner, Sequence) and not isinstance(owner, (str, bytes)):
                values = tuple(owner)
                if len(values) != 2:
                    raise ValueError("owner tuples must contain (logical_slot, slot_generation)")
                result[request_id] = (
                    _as_int(values[0], "owner logical_slot"),
                    _as_int(values[1], "owner slot_generation"),
                )
            else:
                slot = getattr(owner, "index", getattr(owner, "logical_slot", None))
                generation = getattr(owner, "generation", getattr(owner, "slot_generation", None))
                if slot is None or generation is None:
                    raise TypeError("owners must map request IDs to identities or slot owners")
                result[request_id] = (
                    _as_int(slot, "owner logical_slot"),
                    _as_int(generation, "owner slot_generation"),
                )
        return result

    result = {}
    for owner in _sequence(owners, "owners"):
        if not isinstance(owner, DSparkActiveIdentity):
            raise TypeError("owner sequences must contain DSparkActiveIdentity values")
        result[owner.request_id] = (owner.logical_slot, owner.slot_generation)
    return result


def validate_dspark_active_identities(
    metadata: DSparkActiveBatchMetadata,
    owners,
    *,
    expected_round_generation: int | None = None,
    minimum_round_generation: int | None = None,
) -> bool:
    """Reject stale slot ownership or an expired compact round."""
    _validate_compact_layout(metadata)
    round_generation = metadata.round_generation_value
    if expected_round_generation is not None:
        expected = _as_int(expected_round_generation, "expected_round_generation")
        if round_generation != expected:
            raise RuntimeError(
                f"DSpark active metadata round {round_generation} is stale; expected {expected}")
    if minimum_round_generation is not None:
        minimum = _as_int(minimum_round_generation, "minimum_round_generation")
        if round_generation < minimum:
            raise RuntimeError(
                f"DSpark active metadata round {round_generation} is expired; minimum is {minimum}")

    current = _normalize_owners(owners)
    for identity in metadata.identities:
        owner = current.get(identity.request_id)
        if owner is None:
            raise RuntimeError(
                f"DSpark request {identity.request_id!r} no longer owns logical slot {identity.logical_slot}")
        if owner != (identity.logical_slot, identity.slot_generation):
            raise RuntimeError(
                f"DSpark request {identity.request_id!r} slot generation is stale: "
                f"metadata={(identity.logical_slot, identity.slot_generation)} current={owner}")
    return True


def is_dspark_active_metadata_valid(
    metadata: DSparkActiveBatchMetadata,
    owners,
    *,
    expected_round_generation: int | None = None,
    minimum_round_generation: int | None = None,
) -> bool:
    try:
        return validate_dspark_active_identities(
            metadata,
            owners,
            expected_round_generation=expected_round_generation,
            minimum_round_generation=minimum_round_generation,
        )
    except (RuntimeError, TypeError, ValueError):
        return False


class DSparkActiveGenerationTracker:
    """Host-only monotonic round and slot-generation guard.

    ``publish`` accepts an advancing compact round; the first round may be
    zero for legacy compatibility.  A live slot may only change owner through
    :meth:`reuse_slot`, which requires a strictly larger slot generation.
    Re-publishing a round or validating an older compact object is rejected
    as stale.
    """

    def __init__(self, owners=None, *, round_generation: int = 0):
        self.round_generation = _as_int(round_generation, "round_generation")
        if self.round_generation < 0:
            raise ValueError("round_generation must be nonnegative")
        self._owners: dict[str, tuple[int, int]] = {} if owners is None else _normalize_owners(owners)
        self._published = bool(self._owners) or self.round_generation > 0
        self._slot_generations: dict[int, int] = {}
        for request_id, (slot, generation) in self._owners.items():
            if slot < 0 or generation < 0:
                raise ValueError("owner slots and generations must be nonnegative")
            prior = self._slot_generations.get(slot, -1)
            if generation <= prior:
                raise ValueError(f"owners contain duplicate/reused logical slot {slot}")
            self._slot_generations[slot] = generation
            if not isinstance(request_id, str):
                raise TypeError("owner request IDs must be strings")

    @property
    def owners(self) -> dict[str, tuple[int, int]]:
        return dict(self._owners)

    def publish(self, metadata: DSparkActiveBatchMetadata, *, require_advance: bool = True) -> None:
        _validate_compact_layout(metadata)
        round_generation = metadata.round_generation_value
        if round_generation < self.round_generation:
            raise RuntimeError(
                f"DSpark active metadata round {round_generation} is expired; "
                f"last published round is {self.round_generation}")
        if self._published and require_advance and round_generation <= self.round_generation:
            raise RuntimeError(
                f"DSpark active metadata round {round_generation} must advance beyond "
                f"{self.round_generation}")

        owner_by_slot = {slot: request_id for request_id, (slot, _generation) in self._owners.items()}
        for identity in metadata.identities:
            current_request_owner = self._owners.get(identity.request_id)
            if current_request_owner is not None and current_request_owner != (
                    identity.logical_slot, identity.slot_generation):
                raise RuntimeError(f"DSpark request {identity.request_id!r} changed logical slot or generation")
            slot_owner = owner_by_slot.get(identity.logical_slot)
            if slot_owner is not None and slot_owner != identity.request_id:
                raise RuntimeError(
                    f"DSpark logical slot {identity.logical_slot} is still owned by {slot_owner!r}; "
                    "call reuse_slot explicitly")
            if slot_owner is None:
                last_generation = self._slot_generations.get(identity.logical_slot, -1)
                if identity.slot_generation <= last_generation:
                    raise RuntimeError(
                        f"DSpark logical slot {identity.logical_slot} generation must advance beyond "
                        f"{last_generation}")

        for identity in metadata.identities:
            self._owners[identity.request_id] = (identity.logical_slot, identity.slot_generation)
            self._slot_generations[identity.logical_slot] = identity.slot_generation
        self.round_generation = round_generation
        self._published = True

    def validate(self, metadata: DSparkActiveBatchMetadata) -> bool:
        return validate_dspark_active_identities(
            metadata,
            self._owners,
            expected_round_generation=self.round_generation,
        )

    def reuse_slot(self, request_id: str, logical_slot: int, slot_generation: int) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        slot = _as_int(logical_slot, "logical_slot")
        generation = _as_int(slot_generation, "slot_generation")
        if slot < 0 or generation < 1:
            raise ValueError("logical_slot must be nonnegative and slot_generation must be positive")
        if request_id in self._owners:
            raise RuntimeError(f"DSpark request {request_id!r} still has a live logical slot")
        last_generation = self._slot_generations.get(slot, 0)
        if generation <= last_generation:
            raise RuntimeError(
                f"DSpark logical slot {slot} generation {generation} was already used; "
                f"next generation must exceed {last_generation}")
        for old_request_id, (old_slot, _old_generation) in tuple(self._owners.items()):
            if old_slot == slot:
                del self._owners[old_request_id]
        self._owners[request_id] = (slot, generation)
        self._slot_generations[slot] = generation

    def release(self, request_id: str) -> None:
        if request_id not in self._owners:
            raise RuntimeError(f"DSpark request {request_id!r} has no live logical slot")
        del self._owners[request_id]


# Compatibility aliases keep call sites descriptive without changing the
# legacy DSparkBatchMetadata interface.
DSparkActiveMetadata = DSparkActiveBatchMetadata
DSparkRowIdentity = DSparkActiveIdentity
DSparkActiveMetadataTracker = DSparkActiveGenerationTracker
compact_from_fixed = compact_dspark_metadata
compact_from_fixed_metadata = compact_dspark_metadata
compact_metadata_from_fixed = compact_dspark_metadata
restore_fixed = restore_fixed_dspark_metadata
restore_fixed_metadata = restore_fixed_dspark_metadata


__all__ = [
    "DSparkActiveBatchMetadata",
    "DSparkActiveGenerationTracker",
    "DSparkActiveIdentity",
    "DSparkActiveMetadata",
    "DSparkActiveMetadataTracker",
    "DSparkRowIdentity",
    "compact_dspark_metadata",
    "compact_from_fixed",
    "compact_from_fixed_metadata",
    "compact_metadata_from_fixed",
    "is_dspark_active_metadata_valid",
    "restore_fixed",
    "restore_fixed_dspark_metadata",
    "restore_fixed_metadata",
    "validate_dspark_active_identities",
]
