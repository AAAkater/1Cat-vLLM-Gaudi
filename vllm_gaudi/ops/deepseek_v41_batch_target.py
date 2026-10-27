# SPDX-License-Identifier: Apache-2.0
"""Packed target-round ownership for a fixed-capacity DSpark batch.

The public tensor contract is request-major: lane ``i`` owns rows
``6*i:6*(i+1)``.  Any active lane may have one through six valid rows; remaining
rows are no-write sentinels.  This keeps one fixed ``6*N``-row model program
while allowing mixed C1/C6 requests and inactive lanes.

This module owns host metadata adaptation and speculative-state snapshots.  It
does not execute a model or mutate scheduler state.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import operator
from types import SimpleNamespace

import torch

from vllm_gaudi.ops.deepseek_v41_capacity import MIN_DSV41_DSPARK_CAPACITY, validate_dspark_capacity


TARGET_REQUESTS = MIN_DSV41_DSPARK_CAPACITY
TARGET_ROWS_PER_REQUEST = 6
TARGET_ROWS = TARGET_REQUESTS * TARGET_ROWS_PER_REQUEST
TARGET_REQUEST_OFFSETS = (0, TARGET_ROWS_PER_REQUEST, TARGET_ROWS)
TARGET_QUERY_LENGTHS = frozenset(range(1, TARGET_ROWS_PER_REQUEST + 1))
_PAGE_TOKENS = 128
_SWA_ROWS = 256
_HISTORY_ROWS = 8


def _cpu_tensor(value, name: str, *, ndim: int | None = None, allow_bool: bool = False) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU host tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if (value.dtype == torch.bool and not allow_bool) or value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must contain integers")
    return value.detach()


def _cpu_values(value, name: str, *, length: int | None = None) -> list[int]:
    tensor = _cpu_tensor(value, name, ndim=1)
    values = [operator.index(value) for value in tensor.tolist()]
    if length is not None and len(values) != length:
        raise ValueError(f"{name} length {len(values)} does not match {length}")
    return values


def _scalar(value, name: str) -> int:
    tensor = _cpu_tensor(value, name)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must contain one integer")
    return operator.index(tensor.reshape(-1)[0].item())


def _unique_host_indices(values: list[int]) -> list[int]:
    # The snapshot is constructed outside the captured graph.  Host
    # deduplication keeps one index_copy per destination and avoids duplicate
    # scatter destinations in native replay.
    return list(dict.fromkeys(value for value in values if value >= 0))


def target_rows_for_capacity(capacity: int) -> int:
    """Return the fixed physical target-row count for one capacity."""
    validate_dspark_capacity(capacity, allow_serial=True, name="target batch capacity")
    return capacity * TARGET_ROWS_PER_REQUEST


def target_request_offsets(capacity: int) -> tuple[int, ...]:
    """Return request-major offsets ``(0, 6, ..., 6*N)``."""
    rows = target_rows_for_capacity(capacity)
    return tuple(range(0, rows + 1, TARGET_ROWS_PER_REQUEST))


def _sequence_length(value, name: str, *, allow_empty: bool = False) -> int:
    try:
        length = len(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence") from error
    if length == 0 and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return length


def _host_bool_values(value, name: str, *, length: int | None = None) -> list[bool]:
    tensor = _cpu_tensor(value, name, ndim=1, allow_bool=True)
    values = [bool(item) for item in tensor.tolist()]
    if length is not None and len(values) != length:
        raise ValueError(f"{name} length {len(values)} does not match {length}")
    return values


@dataclass(frozen=True)
class DSparkTargetBatch:
    """Validated fixed ``[batch_capacity, 6]`` target inputs.

    ``target_query_start_loc`` is the logical packed layout used by verify.
    It is deliberately different from :meth:`request_offsets`, which is the
    physical fixed-row layout consumed by the target program.
    """

    batch_capacity: int
    request_ids: tuple[str, ...]
    request_active: torch.Tensor
    request_lengths: tuple[int, ...]
    request_slots: torch.Tensor
    slot_generations: torch.Tensor
    target_start_positions: torch.Tensor
    target_query_start_loc: torch.Tensor
    round_generation: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    row_slots: torch.Tensor
    row_request: torch.Tensor
    row_valid: torch.Tensor
    pages: torch.Tensor
    row_pages: torch.Tensor
    real_indices: torch.Tensor
    padding_indices: torch.Tensor
    host_positions: tuple[int, ...]
    host_request_active: tuple[bool, ...]
    host_request_slots: tuple[int, ...]
    host_row_slots: tuple[int, ...]
    host_row_request: tuple[int, ...]
    host_row_valid: tuple[bool, ...]
    host_pages: tuple[tuple[int, ...], ...]
    host_slot_generations: tuple[int, ...]
    host_round_generation: int
    state_capacity: int | None = None
    query_bucket_rows: int | None = None
    query_indices: torch.Tensor | None = None
    query_valid: torch.Tensor | None = None

    @property
    def request_offsets(self) -> tuple[int, ...]:
        return target_request_offsets(self.batch_capacity)

    @property
    def target_rows(self) -> int:
        return target_rows_for_capacity(self.batch_capacity)

    @property
    def host_active_lanes(self) -> tuple[int, ...]:
        """Return active physical lanes from the validated host mask only."""
        return tuple(lane for lane, active in enumerate(self.host_request_active) if active)

    @property
    def actual_query_rows(self):
        return sum(self.host_row_valid)

    def with_query_buckets(self, buckets=(1, 8, 32, 128)):
        from vllm_gaudi.ops.deepseek_v41_target_queries import pack_target_query_rows

        plan = pack_target_query_rows(self.host_row_valid, buckets)
        if plan.bucket_rows == self.target_rows:
            return replace(self, query_bucket_rows=plan.bucket_rows)
        return replace(
            self, query_bucket_rows=plan.bucket_rows,
            query_indices=torch.tensor(plan.source_rows, dtype=torch.int64, device=self.input_ids.device),
            query_valid=torch.tensor(plan.valid, dtype=torch.bool, device=self.input_ids.device))

    def to(self, device) -> "DSparkTargetBatch":
        """Move graph inputs while retaining the host-owned snapshot map."""
        device = torch.device(device)
        return DSparkTargetBatch(
            batch_capacity=self.batch_capacity,
            request_ids=self.request_ids,
            request_active=self.request_active.to(device),
            request_lengths=self.request_lengths,
            request_slots=self.request_slots.to(device),
            slot_generations=self.slot_generations.to(device),
            target_start_positions=self.target_start_positions.to(device),
            target_query_start_loc=self.target_query_start_loc.to(device),
            round_generation=self.round_generation.to(device),
            input_ids=self.input_ids.to(device),
            positions=self.positions.to(device),
            row_slots=self.row_slots.to(device),
            row_request=self.row_request.to(device),
            row_valid=self.row_valid.to(device),
            pages=self.pages.to(device),
            row_pages=self.row_pages.to(device),
            real_indices=self.real_indices.to(device),
            padding_indices=self.padding_indices.to(device),
            host_positions=self.host_positions,
            host_request_active=self.host_request_active,
            host_request_slots=self.host_request_slots,
            host_row_slots=self.host_row_slots,
            host_row_request=self.host_row_request,
            host_row_valid=self.host_row_valid,
            host_pages=self.host_pages,
            host_slot_generations=self.host_slot_generations,
            host_round_generation=self.host_round_generation,
            state_capacity=self.state_capacity,
            query_bucket_rows=self.query_bucket_rows,
            query_indices=None if self.query_indices is None else self.query_indices.to(device),
            query_valid=None if self.query_valid is None else self.query_valid.to(device),
        )

    @classmethod
    def from_metadata(cls,
                      metadata,
                      queries,
                      pages,
                      *,
                      batch_capacity: int | None = None,
                      state_capacity: int | None = None,
                      device="cpu") -> "DSparkTargetBatch":
        """Build the target contract from CPU ``DSparkBatchMetadata`` output.

        ``queries`` contains ``[anchor, proposal_0, ..., proposal_4]`` per
        lane, matching ``pack_target_queries``.  The fixed bucket is
        ``[batch_capacity, 6]``; explicit capacity makes the canonical
        lane->slot mapping authoritative.  Without an explicit capacity the
        legacy B2 metadata's host slot permutation is retained.

        ``metadata.request_active`` is optional.  Missing or empty means all
        lanes active.  Inactive lanes contribute six no-write rows even though
        their metadata query length and draft list are empty.
        """
        request_ids = tuple(getattr(metadata, "request_ids", ()))
        if not request_ids:
            raise ValueError("target batching requires at least one request lane")

        if isinstance(queries, torch.Tensor):
            query_tensor = _cpu_tensor(queries, "queries", ndim=2)
            query_padding = None
        else:
            query_tensor = _cpu_tensor(getattr(queries, "queries"), "queries", ndim=2)
            query_padding = getattr(queries, "padding_mask", None)
        if query_tensor.ndim != 2 or query_tensor.shape[1] != TARGET_ROWS_PER_REQUEST:
            raise ValueError("queries must have shape [batch_capacity, 6]")
        if query_padding is not None:
            query_padding = _cpu_tensor(query_padding, "queries.padding_mask", ndim=2, allow_bool=True)
            if query_padding.shape[1] != TARGET_ROWS_PER_REQUEST:
                raise ValueError("queries.padding_mask must have shape [batch_capacity, 6]")

        page_tensor = _cpu_tensor(pages, "pages", ndim=2)
        if page_tensor.shape[1] < 1:
            raise ValueError("pages must have shape [batch_capacity, P] with P >= 1")
        if bool((page_tensor <= 0).any()):
            raise ValueError("pages must contain positive scheduler block ids")

        raw_active = getattr(metadata, "request_active", None)
        if isinstance(raw_active, torch.Tensor):
            active_length = raw_active.numel()
        elif raw_active is None:
            active_length = 0
        else:
            active_length = _sequence_length(raw_active, "request_active", allow_empty=True)

        starts = _cpu_values(getattr(metadata, "target_start_positions"), "target_start_positions")
        raw_query_loc = _cpu_values(getattr(metadata, "target_query_start_loc"), "target_query_start_loc")
        if len(raw_query_loc) < 2:
            raise ValueError("target_query_start_loc must have at least two entries")
        request_slots = _cpu_values(getattr(metadata, "request_slots"), "request_slots")
        generations = _cpu_values(getattr(metadata, "slot_generations"), "slot_generations")
        draft_counts = None
        raw_draft_counts = getattr(metadata, "num_draft_tokens", None)
        if raw_draft_counts is not None:
            draft_counts = _cpu_values(raw_draft_counts, "num_draft_tokens")

        observed = [("queries", query_tensor.shape[0]), ("pages", page_tensor.shape[0]),
                    ("request_ids", len(request_ids)), ("target_start_positions", len(starts)),
                    ("target_query_start_loc", len(raw_query_loc) - 1), ("request_slots", len(request_slots)),
                    ("slot_generations", len(generations))]
        if query_padding is not None:
            observed.append(("queries.padding_mask", query_padding.shape[0]))
        if draft_counts is not None:
            observed.append(("num_draft_tokens", len(draft_counts)))
        if active_length:
            observed.append(("request_active", active_length))

        if batch_capacity is None:
            inferred = {length for _name, length in observed if length >= 0}
            if len(inferred) != 1:
                mismatch = ", ".join(f"{name}={length}" for name, length in observed)
                raise ValueError(f"cannot infer one target batch capacity from metadata: {mismatch}")
            capacity = inferred.pop()
        else:
            capacity = target_rows_for_capacity(batch_capacity) // TARGET_ROWS_PER_REQUEST
        if capacity < 1:
            raise ValueError("target batch capacity must be positive")
        for name, length in observed:
            if length != capacity:
                raise ValueError(f"{name} length {length} does not match target batch capacity {capacity}")
        if raw_query_loc and len(raw_query_loc) != capacity + 1:
            raise ValueError("target_query_start_loc must have batch_capacity + 1 entries")

        query_shape = (capacity, TARGET_ROWS_PER_REQUEST)
        if tuple(query_tensor.shape) != query_shape:
            raise ValueError(f"queries must have shape [{capacity}, 6]")
        if query_padding is not None and tuple(query_padding.shape) != query_shape:
            raise ValueError(f"queries.padding_mask must have shape [{capacity}, 6]")

        if active_length == 0:
            request_active = [True] * capacity
        else:
            request_active = _host_bool_values(raw_active, "request_active", length=capacity)
        if not any(request_active):
            raise ValueError("target batching requires at least one active request")
        for lane, (request_id, is_active) in enumerate(zip(request_ids, request_active, strict=True)):
            if is_active and not isinstance(request_id, str):
                raise TypeError(f"active request_ids[{lane}] must be a string")
            if not is_active and request_id is not None and not isinstance(request_id, str):
                raise TypeError(f"inactive request_ids[{lane}] must be None")

        if any(start < 0 for start in starts):
            raise ValueError("target_start_positions must be nonnegative")
        if raw_query_loc[0] != 0 or any(left > right for left, right in zip(raw_query_loc, raw_query_loc[1:])):
            raise ValueError("target_query_start_loc must be monotonic and start at zero")
        # Curie's metadata keeps logical packed spans.  A caller may instead
        # pass the physical [0,6,12] bucket; derive the logical spans from
        # draft counts when available and accept both representations.
        if draft_counts is not None:
            if any(count < 0 or count > TARGET_ROWS_PER_REQUEST - 1 for count in draft_counts):
                raise ValueError("num_draft_tokens must be in 0..5")
            lengths = tuple(int(count) + 1 if request_active[index] else 0
                            for index, count in enumerate(draft_counts))
        else:
            lengths = tuple(int(right - left) for left, right in zip(raw_query_loc, raw_query_loc[1:]))
        for index, (active, length) in enumerate(zip(request_active, lengths, strict=True)):
            if active and length not in TARGET_QUERY_LENGTHS:
                raise ValueError(f"active target request {index} must contain one through six rows")
            if not active and length != 0:
                raise ValueError(f"inactive target request {index} must have zero query rows")
        running_length = 0
        logical_query_loc = [0]
        for length in lengths:
            running_length += length
            logical_query_loc.append(running_length)
        if raw_query_loc not in (logical_query_loc, list(target_request_offsets(capacity))):
            raise ValueError("target_query_start_loc disagrees with draft counts")
        if any(not active and start != 0 for active, start in zip(request_active, starts, strict=True)):
            raise ValueError("inactive target requests must use start position zero")
        if draft_counts is not None:
            if any(not request_active[index] and count != 0 for index, count in enumerate(draft_counts)):
                raise ValueError("inactive target requests must have no draft tokens")

        metadata_padding = getattr(metadata, "target_padding_mask", None)
        if metadata_padding is not None:
            padding = _cpu_tensor(metadata_padding, "target_padding_mask", allow_bool=True)
            if padding.ndim == 1:
                if padding.numel() == logical_query_loc[-1]:
                    padding_rows = padding
                    flat_logical_padding = True
                elif padding.numel() == capacity * TARGET_ROWS_PER_REQUEST:
                    padding_rows = padding.reshape(query_shape)
                    flat_logical_padding = False
                    if query_padding is None:
                        query_padding = padding_rows
                else:
                    raise ValueError("flat target_padding_mask must match logical or fixed target rows")
                for request_index, length in enumerate(lengths):
                    if flat_logical_padding:
                        start = logical_query_loc[request_index]
                        selected = padding_rows[start:start + length]
                    else:
                        selected = padding_rows[request_index, :length]
                    if any(bool(value) for value in selected.tolist()):
                        raise ValueError("selected target rows must not be padding")
            elif padding.ndim == 2:
                if tuple(padding.shape) != query_shape:
                    raise ValueError(f"fixed target_padding_mask must have shape [{capacity}, 6]")
                if query_padding is None:
                    query_padding = padding
                for request_index, length in enumerate(lengths):
                    if any(bool(value) for value in padding[request_index, :length].tolist()):
                        raise ValueError("selected target rows must not be padding")
            else:
                raise ValueError("target_padding_mask must be flat or [batch_capacity, 6]")

        owner_capacity = capacity if state_capacity is None else state_capacity
        validate_dspark_capacity(owner_capacity, allow_serial=True, name="target state capacity")
        active_slots = [slot for active, slot in zip(request_active, request_slots, strict=True) if active]
        if (len(set(active_slots)) != len(active_slots)
                or any(slot < 0 or slot >= owner_capacity for slot in active_slots)):
            raise ValueError("active request_slots must be distinct nonnegative owners")
        for index, (active, slot, generation) in enumerate(
                zip(request_active, request_slots, generations, strict=True)):
            if not active and (slot != -1 or generation != 0):
                raise ValueError(f"inactive target request {index} must use slot -1 and generation zero")
            if active and generation < 0:
                raise ValueError("active slot_generations must be nonnegative")
        canonical_slots = (batch_capacity is not None or capacity != TARGET_REQUESTS
                           or (active_length > 0 and not all(request_active)))
        if canonical_slots and state_capacity is None:
            for index, (active, slot) in enumerate(zip(request_active, request_slots, strict=True)):
                if active and slot != index:
                    raise ValueError("canonical target lanes require active device slot == lane index")
            host_request_slots = tuple(range(capacity))
        else:
            host_request_slots = tuple(request_slots)

        budget_values = getattr(metadata, "remaining_output_budget", None)
        if budget_values is not None:
            budgets = _cpu_values(budget_values, "remaining_output_budget", length=capacity)
            if any(budget < 0 for budget in budgets):
                raise ValueError("remaining_output_budget must be nonnegative")
            if any(not active and budget != 0 for active, budget in zip(request_active, budgets, strict=True)):
                raise ValueError("inactive target requests must have remaining output budget zero")
        round_generation = _scalar(getattr(metadata, "round_generation", 0), "round_generation")
        if round_generation < 0:
            raise ValueError("round_generation must be nonnegative")

        host_pages = tuple(tuple(int(value) for value in row) for row in page_tensor.tolist())

        row_request_host, row_slots_host, row_valid_host = [], [], []
        positions_host, ids_host = [], []
        for request_index in range(capacity):
            length = lengths[request_index]
            for row_index in range(TARGET_ROWS_PER_REQUEST):
                row_request_host.append(request_index)
                is_valid = request_active[request_index] and row_index < length
                if query_padding is not None:
                    is_valid = is_valid and not bool(query_padding[request_index, row_index])
                if is_valid:
                    row_valid_host.append(True)
                    row_slots_host.append(request_slots[request_index])
                    positions_host.append(starts[request_index] + row_index)
                    ids_host.append(int(query_tensor[request_index, row_index]))
                else:
                    row_valid_host.append(False)
                    row_slots_host.append(-1)
                    # The masked compressor treats -1 as a no-write row;
                    # model math clamps it to a safe RoPE address later.
                    positions_host.append(-1)
                    ids_host.append(0)

        if sum(row_valid_host) != sum(lengths):
            raise ValueError("query padding masks must leave every selected target row valid")

        row_request = torch.tensor(row_request_host, dtype=torch.int32)
        row_valid = torch.tensor(row_valid_host, dtype=torch.bool)
        row_pages_cpu = page_tensor.index_select(0, row_request.long())
        real_indices = torch.tensor([i for i, valid in enumerate(row_valid_host) if valid], dtype=torch.int64)
        padding_indices = torch.tensor([i for i, valid in enumerate(row_valid_host) if not valid], dtype=torch.int64)

        return cls(
            batch_capacity=capacity,
            request_ids=request_ids,
            request_active=torch.tensor(request_active, dtype=torch.bool, device=device),
            request_lengths=tuple(lengths),
            request_slots=torch.tensor(request_slots, dtype=torch.int32, device=device),
            slot_generations=torch.tensor(generations, dtype=torch.int64, device=device),
            target_start_positions=torch.tensor(starts, dtype=torch.int64, device=device),
            target_query_start_loc=torch.tensor(logical_query_loc, dtype=torch.int32, device=device),
            round_generation=torch.tensor([round_generation], dtype=torch.int64, device=device),
            input_ids=torch.tensor(ids_host, dtype=torch.int32, device=device),
            positions=torch.tensor(positions_host, dtype=torch.int32, device=device),
            row_slots=torch.tensor(row_slots_host, dtype=torch.int32, device=device),
            row_request=row_request.to(device),
            row_valid=row_valid.to(device),
            pages=page_tensor.to(device),
            row_pages=row_pages_cpu.to(device),
            real_indices=real_indices.to(device),
            padding_indices=padding_indices.to(device),
            host_positions=tuple(positions_host),
            host_request_active=tuple(request_active),
            host_request_slots=host_request_slots,
            host_row_slots=tuple(row_slots_host),
            host_row_request=tuple(row_request_host),
            host_row_valid=tuple(row_valid_host),
            host_pages=host_pages,
            host_slot_generations=tuple(generations),
            host_round_generation=round_generation,
            state_capacity=state_capacity,
        )

    def stage_inputs(self):
        """Return the fixed tensor arguments consumed by ``DSparkTargetStage``."""
        return SimpleNamespace(
            input_ids=self.input_ids,
            positions=self.positions,
            row_slots=self.row_slots,
            row_pages=self.row_pages,
            row_valid=self.row_valid,
        )


@dataclass
class _RowSnapshot:
    destination: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor
    # Each index maps to (request_index, row_inside_request) owners.  A row is
    # retained when any owner is inside the verifier's accepted prefix.
    owners: tuple[tuple[tuple[int, int], ...], ...]
    # One combined I64 ``[2, rows, width]`` tensor: plane 0 is the owner
    # request lane, plane 1 the row inside that request.  Padding owners use
    # query ``TARGET_ROWS_PER_REQUEST``, which no count in 0..6 can retain, so
    # no separate validity mask is needed.  Identical owner layouts share one
    # tensor (see :meth:`DSparkTargetSnapshot.prepare_device_restore`).
    device_owners: torch.Tensor | None = None

    @staticmethod
    def build_device_owners(owners, device) -> torch.Tensor:
        width = max(1, max((len(row) for row in owners), default=0))
        requests, queries = [], []
        for row in owners:
            padding = width - len(row)
            requests.extend([request for request, _query in row] + [0] * padding)
            queries.extend([query for _request, query in row] + [TARGET_ROWS_PER_REQUEST] * padding)
        return torch.tensor(requests + queries, dtype=torch.int64, device=device).reshape(2, len(owners), width)

    def prepare_device_restore(self, cache: dict | None = None):
        device = self.indices.device
        if cache is None:
            self.device_owners = self.build_device_owners(self.owners, device)
            return
        key = (self.owners, device)
        tensor = cache.get(key)
        if tensor is None:
            tensor = cache[key] = self.build_device_owners(self.owners, device)
        self.device_owners = tensor

    def restore_device(self, committed):
        if self.device_owners is None:
            raise RuntimeError("target snapshot device ownership must be prepared before execution")
        requests, queries = self.device_owners[0], self.device_owners[1]
        limits = committed.index_select(0, requests.reshape(-1)).reshape(requests.shape)
        retained = (queries < limits).any(dim=1)
        current = self.destination.index_select(0, self.indices)
        shape = (retained.shape[0], ) + (1, ) * (self.values.ndim - 1)
        restored = torch.where(retained.reshape(shape), current, self.values)
        self.destination.index_copy_(0, self.indices, restored)

    def restore(self, committed=None) -> None:
        keep = self.indices
        values = self.values
        if committed is not None:
            committed = tuple(int(value) for value in committed)
            restore = [
                index for index, owners in enumerate(self.owners)
                if not any(row_index < committed[request_index] for request_index, row_index in owners)
            ]
            if not restore:
                return
            selector = torch.tensor(restore, dtype=torch.int64, device=self.indices.device)
            keep = self.indices.index_select(0, selector)
            values = self.values.index_select(0, selector)
        self.destination.index_copy_(0, keep, values)


@dataclass
class DSparkTargetSnapshot:
    """Speculative target writes that must be restored before state is reused."""

    rows: tuple[_RowSnapshot, ...]
    bytes: int
    capacity: int = TARGET_REQUESTS

    def restore(self, committed=None) -> None:
        if committed is not None:
            committed = tuple(int(value) for value in committed)
            if len(committed) != self.capacity:
                raise ValueError(f"partial target restore requires committed[B={self.capacity}]")
            if any(value < 0 or value > TARGET_ROWS_PER_REQUEST for value in committed):
                raise ValueError("committed lengths must be in 0..6")
        for row in self.rows:
            row.restore(committed)

    def prepare_device_restore(self):
        """Upload owner layouts once per distinct ``(owners, device)`` layout.

        SWA/history/page rows of every layer usually share one owner layout,
        so a round performs one small H2D copy per layout rather than per row.
        """
        cache: dict = {}
        for row in self.rows:
            row.prepare_device_restore(cache)

    def restore_device(self, committed):
        if (not isinstance(committed, torch.Tensor) or committed.shape != (self.capacity, )
                or committed.dtype not in (torch.int32, torch.int64)):
            raise ValueError("device target restore requires integer accepted counts for every target lane")
        if any(committed.device != row.indices.device for row in self.rows):
            raise ValueError("device target restore counts must share the snapshot device")
        # A malformed count cannot authorize tentative state retention, so any
        # out-of-range count rolls back every lane (safe fallback).  This path
        # cannot raise without a host readback; the host finish
        # (DSparkPackedTargetStages.finish) is the authority that detects the
        # same counts, read back from the [N,16] record, and fails the round.
        valid = ((committed >= 0) & (committed <= TARGET_ROWS_PER_REQUEST)).all()
        safe_counts = torch.where(valid, committed, torch.zeros_like(committed))
        for row in self.rows:
            row.restore_device(safe_counts)

    @classmethod
    def capture(cls, program, batch: DSparkTargetBatch) -> "DSparkTargetSnapshot":
        if batch.row_valid.numel() != target_rows_for_capacity(batch.batch_capacity):
            raise ValueError("target snapshot rows disagree with batch capacity")

        snapshots: list[_RowSnapshot] = []
        seen: dict[int, tuple[torch.Tensor, dict[int, list[tuple[int, int]]]]] = {}

        def add(destination, entries: list[tuple[int, int, int]]) -> None:
            """Capture ``(row_index, request_index, row_inside_request)`` entries."""
            if not destination.ndim:
                return
            owners_by_index: dict[int, list[tuple[int, int]]] = {}
            for index, request_index, row_inside_request in entries:
                if 0 <= index < destination.shape[0]:
                    owners_by_index.setdefault(index, []).append((request_index, row_inside_request))
            if not owners_by_index:
                return
            existing = seen.get(id(destination))
            if existing is None:
                indices = torch.tensor(sorted(owners_by_index), dtype=torch.int64, device=destination.device)
                owners = tuple(tuple(owners_by_index[index]) for index in sorted(owners_by_index))
                snapshot = _RowSnapshot(destination, indices, destination.index_select(0, indices).clone(), owners)
                snapshots.append(snapshot)
                seen[id(destination)] = (indices, owners_by_index)
                return
            indices, existing_owners = existing
            # A destination is normally captured once. Merge any repeated
            # indices defensively without creating duplicate scatter targets.
            current_indices = set(existing_owners)
            for index, owners in owners_by_index.items():
                existing_owners.setdefault(index, []).extend(owners)
                if index not in current_indices:
                    current_indices.add(index)
                    indices = torch.cat((indices, torch.tensor([index], dtype=torch.int64,
                                                               device=destination.device)))
            ordered = sorted(current_indices)
            ordered_tensor = torch.tensor(ordered, dtype=torch.int64, device=destination.device)
            ordered_owners = tuple(tuple(existing_owners[index]) for index in ordered)
            snapshot = _RowSnapshot(destination, ordered_tensor,
                                    destination.index_select(0, ordered_tensor).clone(), ordered_owners)
            snapshots[:] = [row for row in snapshots if row.destination is not destination]
            snapshots.append(snapshot)
            seen[id(destination)] = (ordered_tensor, existing_owners)

        valid_rows = [index for index, valid in enumerate(batch.host_row_valid) if valid]
        for layer in getattr(program, "layers", ()):
            attention = getattr(layer, "attention", None)
            state = getattr(attention, "batch_state", None)
            if state is None:
                continue
            swa_entries = []
            history_entries = []
            for row in valid_rows:
                position = batch.host_positions[row]
                slot = batch.host_row_slots[row]
                request_index = batch.host_row_request[row]
                row_inside_request = row - request_index * TARGET_ROWS_PER_REQUEST
                swa_entries.append((slot * _SWA_ROWS + position % _SWA_ROWS, request_index, row_inside_request))
                history_entries.append((slot * _HISTORY_ROWS + position % _HISTORY_ROWS,
                                        request_index, row_inside_request))
            add(state.swa, swa_entries)
            if hasattr(state, "kv_history"):
                add(state.kv_history, history_entries)
                add(state.score_history, history_entries)

        shared = getattr(program, "shared", None)
        if shared is not None:
            # The target branch uses packed pages and does not consume or
            # mutate derived mirrors.  Only actual canonical page writes are
            # snapshotted here.
            for cache in getattr(shared, "sources", {}).values():
                ratio = int(getattr(cache, "ratio", 1))
                if ratio < 1:
                    continue
                width = _PAGE_TOKENS // ratio
                page_entries = []
                for row in valid_rows:
                    position = batch.host_positions[row]
                    # This is the exact publication predicate in
                    # PagedCSA2Attention._compress_batch_target.
                    if (position + 1) % ratio:
                        continue
                    request_index = batch.host_row_request[row]
                    row_inside_request = row - request_index * TARGET_ROWS_PER_REQUEST
                    logical = position // ratio
                    page_slot = logical // width
                    pages = batch.host_pages[request_index]
                    if page_slot >= len(pages):
                        continue
                    physical = int(pages[page_slot]) * width + logical % width
                    page_entries.append((physical, request_index, row_inside_request))
                add(cache.main, page_entries)
                add(cache.index, page_entries)

        total_bytes = sum(row.values.numel() * row.values.element_size() for row in snapshots)
        return cls(tuple(snapshots), total_bytes, batch.batch_capacity)


# Stable (N, B) snapshot layout.  Entry ``e`` of every layout group is the
# physical target row ``e`` (owner lane ``e // 6``, query ``e % 6``), so every
# shape depends only on the state capacity N and the execution bucket B.
_STABLE_PADDING_QUERY = TARGET_ROWS_PER_REQUEST


def _stable_target_destinations(program):
    """Return the target write set as ``((destination, (kind, parameter)), ...)``.

    The order and predicates mirror :meth:`DSparkTargetSnapshot.capture`.
    """
    entries = []
    for layer in getattr(program, "layers", ()):
        attention = getattr(layer, "attention", None)
        state = getattr(attention, "batch_state", None)
        if state is None:
            continue
        entries.append((state.swa, ("swa", _SWA_ROWS)))
        if hasattr(state, "kv_history"):
            entries.append((state.kv_history, ("history", _HISTORY_ROWS)))
            entries.append((state.score_history, ("history", _HISTORY_ROWS)))
    shared = getattr(program, "shared", None)
    if shared is not None:
        for cache in getattr(shared, "sources", {}).values():
            ratio = int(getattr(cache, "ratio", 1))
            if ratio < 1:
                continue
            if ratio > _PAGE_TOKENS:
                raise ValueError("stable target snapshot requires compressed ratios in 1..128")
            entries.append((cache.main, ("page", ratio)))
            entries.append((cache.index, ("page", ratio)))
    return tuple((destination, kind) for destination, kind in entries if destination.ndim)


def stable_target_restore_layout(positions, row_slots, row_valid, row_pages, groups):
    """Return fixed ``[G, 6B]`` destination rows and owner queries on device.

    Pure tensor function of the target batch's existing device metadata.  Real
    entries are exactly the rows the target writes (the predicates of
    :meth:`DSparkTargetSnapshot.capture` and the batch attention publication).
    Every other entry is padding: destination row 0 and owner query 6, which
    no accepted count in 0..6 can retain.
    """
    rows = positions.shape[0]
    device = positions.device
    query = torch.arange(rows, dtype=torch.int64, device=device).remainder(TARGET_ROWS_PER_REQUEST)
    position = positions.to(torch.int64)
    slot = row_slots.to(torch.int64)
    live = row_valid.to(torch.bool) & (position >= 0) & (slot >= 0)
    safe_position = position.clamp_min(0)
    safe_slot = slot.clamp_min(0)
    pages = row_pages.to(torch.int32)
    page_count = pages.shape[1]
    padding_query = torch.full_like(query, _STABLE_PADDING_QUERY)
    indices, queries = [], []
    for kind, parameter, limit in groups:
        if kind == "page":
            width = _PAGE_TOKENS // parameter
            logical = torch.div(safe_position, parameter, rounding_mode="floor")
            page_slot = torch.div(logical, width, rounding_mode="floor")
            # Gather the I32 page table and widen afterwards: an eager HPU
            # gather over I64 data fails graph compilation (synStatus 26).
            page = pages.gather(1, page_slot.clamp_max(page_count - 1).unsqueeze(1)).squeeze(1).to(torch.int64)
            index = page * width + logical.remainder(width)
            real = live & ((safe_position + 1).remainder(parameter) == 0) & (page_slot < page_count)
        else:
            index = safe_slot * parameter + safe_position.remainder(parameter)
            real = live
        real = real & (index >= 0) & (index < limit)
        indices.append(torch.where(real, index, torch.zeros_like(index)))
        queries.append(torch.where(real, query, padding_query))
    if not indices:
        empty = torch.zeros((0, rows), dtype=torch.int64, device=device)
        return empty, empty.clone()
    return torch.stack(indices), torch.stack(queries)


def capture_stable_target_rows(positions, row_slots, row_valid, row_pages, request_slots, groups, indices,
                               owners, lane_map, destinations, saved, destination_groups):
    """Write one round's layout and saved rows into persistent buffers in place.

    Buffers may be larger than the round's ``6*B`` rows (the fused restore
    always uses the ``6*N`` storage); the static tail becomes padding: row 0,
    owner query 6, lane -1, and saved bytes read from row 0 like any padding.
    """
    layout_indices, layout_queries = stable_target_restore_layout(positions, row_slots, row_valid, row_pages,
                                                                  groups)
    padding = indices.shape[1] - layout_indices.shape[1]
    if padding:
        layout_indices = torch.cat((layout_indices, layout_indices.new_zeros((layout_indices.shape[0], padding))), 1)
        layout_queries = torch.cat(
            (layout_queries, layout_queries.new_full((layout_queries.shape[0], padding), _STABLE_PADDING_QUERY)), 1)
    indices.copy_(layout_indices)
    owners[1].copy_(layout_queries)
    slots = request_slots.to(torch.int64)
    lanes = torch.where(slots >= 0, slots, torch.full_like(slots, -1))
    if lane_map.shape[0] != lanes.shape[0]:
        lanes = torch.cat((lanes, lanes.new_full((lane_map.shape[0] - lanes.shape[0], ), -1)))
    lane_map.copy_(lanes)
    for destination, values, group in zip(destinations, saved, destination_groups):
        values.copy_(destination.index_select(0, indices[group]))


def _restore_stable_target_lanes(limits, indices, owners, destinations, saved, destination_groups, *,
                                 native_writes=False):
    """Apply accepted-prefix rollback for B-lane limits over the fixed layout.

    A destination row is rolled back iff some real entry writes it and no real
    entry for that row is inside its owner's accepted prefix; this is the
    "any owner accepted keeps the row" rule of :class:`_RowSnapshot`.  The
    decision is reduced over equal destination rows, so repeated rows (shared
    owners or padding at row 0) all write identical bytes: padding not sharing
    a real row writes back its current value.

    ``native_writes`` (compiled HPU graphs) uses the qualified native state
    writer instead of ``index_copy_`` on a graph input: only the first real
    entry of each rolled-back row writes its saved bytes, every other entry is
    a ``-1`` no-write row, so there are no duplicate destinations at all.
    Both variants leave bit-identical state.  Returns a completion tensor.
    """
    requests, queries = owners[0], owners[1]
    real = queries < _STABLE_PADDING_QUERY
    accepted = queries < limits.index_select(0, requests.reshape(-1)).reshape(requests.shape)
    plans = {}
    completions = []
    for destination, values, group in zip(destinations, saved, destination_groups):
        plan = plans.get(group)
        if plan is None:
            index = indices[group]
            real_rows = real[group]
            same = index.unsqueeze(1) == index.unsqueeze(0)
            owned = (same & real_rows.unsqueeze(0)).any(1)
            kept = (same & accepted[group].unsqueeze(0)).any(1)
            rollback = owned & ~kept
            rows = None
            if native_writes:
                order = torch.arange(index.shape[0], device=index.device)
                earlier = (same & real_rows.unsqueeze(0) & (order.unsqueeze(0) < order.unsqueeze(1))).any(1)
                rows = torch.where(rollback & real_rows & ~earlier, index, torch.full_like(index, -1))
                rows = rows.to(torch.int32)
            plan = plans[group] = (index, rollback, rows)
        index, rollback, rows = plan
        if rows is not None:
            from vllm_gaudi.ops.deepseek_v41_batch_attention import write_state_rows

            completions.append(write_state_rows(destination, values, rows).to(torch.int32).amax())
            continue
        current = destination.index_select(0, index)
        shape = (index.shape[0], ) + (1, ) * (destination.ndim - 1)
        destination.index_copy_(0, index, torch.where(rollback.reshape(shape), values, current))
    if completions:
        return torch.stack(completions).amax()
    return torch.zeros((), dtype=torch.int32, device=limits.device)


def restore_stable_target_rows(counts, lane_map, indices, owners, destinations, saved, destination_groups, *,
                               native_writes=False):
    """Map full state-capacity counts to B lanes and restore target rows.

    Pure tensor entry for the device target finish: no host reads, no Python
    scalars derived from tensors and no data-dependent Python branch.  Shapes
    depend only on ``(N, B)`` and the program's fixed destination set.  Any
    out-of-range count rolls back every lane (the host finish rejects the
    same counts after readback).  Returns a completion tensor.
    """
    in_range = ((counts >= 0) & (counts <= TARGET_ROWS_PER_REQUEST)).all()
    counts = counts.to(torch.int64)
    safe_counts = torch.where(in_range, counts, torch.zeros_like(counts))
    limits = torch.where(lane_map >= 0, safe_counts.index_select(0, lane_map.clamp_min(0)),
                         torch.zeros_like(lane_map))
    return _restore_stable_target_lanes(limits, indices, owners, destinations, saved, destination_groups,
                                        native_writes=native_writes)


class DSparkStableTargetLayout:
    """Persistent fixed-shape target snapshot storage for one state capacity N.

    All buffers are allocated once for ``6*N`` entries; an execution bucket B
    uses ``[:6*B]`` views (see :meth:`view`).  Only one round may own the
    storage at a time, matching the runner's single in-flight transaction.
    """

    def __init__(self, program, state_capacity: int, *, device=None):
        validate_dspark_capacity(state_capacity, allow_serial=True, name="target state capacity")
        destinations = _stable_target_destinations(program)
        if len({id(destination) for destination, _kind in destinations}) != len(destinations):
            raise ValueError("stable target snapshot requires each state tensor in exactly one layout")
        devices = {destination.device for destination, _kind in destinations}
        if len(devices) > 1:
            raise ValueError("stable target snapshot state tensors must share one device")
        device = devices.pop() if devices else torch.device("cpu" if device is None else device)
        groups: list[tuple[str, int, int]] = []
        destination_groups = []
        for destination, (kind, parameter) in destinations:
            key = (kind, parameter, int(destination.shape[0]))
            if key not in groups:
                groups.append(key)
            destination_groups.append(groups.index(key))
        entries = state_capacity * TARGET_ROWS_PER_REQUEST
        self.program = program
        self.state_capacity = state_capacity
        self.device = device
        self.destinations = tuple(destination for destination, _kind in destinations)
        self.groups = tuple(groups)
        self.destination_groups = tuple(destination_groups)
        self.signature = self._signature(destinations)
        self.indices = torch.zeros((len(groups), entries), dtype=torch.int64, device=device)
        self.owners = torch.empty((2, len(groups), entries), dtype=torch.int64, device=device)
        lanes = torch.div(torch.arange(entries, dtype=torch.int64), TARGET_ROWS_PER_REQUEST, rounding_mode="floor")
        self.owners[0].copy_(lanes.expand(len(groups), entries))
        self.owners[1].fill_(_STABLE_PADDING_QUERY)
        self.lane_map = torch.full((state_capacity, ), -1, dtype=torch.int64, device=device)
        self.zero_limits = torch.zeros((state_capacity, ), dtype=torch.int64, device=device)
        self.saved = tuple(
            torch.empty((entries, *destination.shape[1:]), dtype=destination.dtype, device=destination.device)
            for destination in self.destinations)
        self.owner = None
        self.views: dict[tuple[int, bool], DSparkStableTargetSnapshot] = {}
        self._compiled: dict[str, Callable] = {}

    def _run(self, name, function, *args, **kwargs):
        """Run a pure-tensor layout step; HPU uses one compiled graph per view.

        Each eager HPU launch costs about as much as the whole step's device
        work (dozens of small index ops per round), so HPU compiles the step
        once per ``(B, padded)`` view shape.  All variants share one code
        object, so the recompile budget covers every cached view.
        """
        if self.device.type != "hpu":
            return function(*args, **kwargs)
        compiled = self._compiled.get(name)
        if compiled is None:
            compiled = self._compiled[name] = torch.compile(function, backend="hpu_backend", fullgraph=True,
                                                            dynamic=False)
        limit = max(torch._dynamo.config.recompile_limit, len(self.views) + 1)
        with torch._dynamo.config.patch(recompile_limit=limit):
            return compiled(*args, **kwargs)

    @staticmethod
    def _signature(destinations):
        return tuple((id(destination), tuple(destination.shape), destination.dtype, kind)
                     for destination, kind in destinations)

    def matches(self, program) -> bool:
        """Return whether ``program`` still binds the same state tensors."""
        return program is self.program and self._signature(_stable_target_destinations(program)) == self.signature

    @property
    def bytes(self) -> int:
        return sum(values.numel() * values.element_size() for values in self.saved)

    def view(self, batch_capacity: int, *, padded: bool = False) -> "DSparkStableTargetSnapshot":
        """Return the cached view for bucket B.

        ``padded`` views span the full ``6*N`` storage for every B, so a
        consumer compiled once per N (the fused draft transaction) sees the
        same tensor arguments whichever target bucket ran.
        """
        key = (batch_capacity, bool(padded))
        view = self.views.get(key)
        if view is None:
            view = self.views[key] = DSparkStableTargetSnapshot(self, batch_capacity, padded=bool(padded))
        return view


class DSparkStableTargetSnapshot:
    """Fixed ``(N, B)`` snapshot view with :class:`DSparkTargetSnapshot` semantics.

    ``restore_device`` consumes the full state-capacity counts and maps them
    to B lanes inside :func:`restore_stable_target_rows`; host ``restore``
    (abort/fallback) uses B-lane counts like the legacy snapshot.
    """

    def __init__(self, layout: DSparkStableTargetLayout, batch_capacity: int, *, padded: bool = False):
        validate_dspark_capacity(batch_capacity, allow_serial=True, name="target execution capacity")
        if batch_capacity > layout.state_capacity:
            raise ValueError("target execution capacity must fit the state capacity")
        lanes = layout.state_capacity if padded else batch_capacity
        entries = lanes * TARGET_ROWS_PER_REQUEST
        self.layout = layout
        self.capacity = batch_capacity
        self.padded = padded
        self.state_capacity = layout.state_capacity
        self.indices = layout.indices[:, :entries]
        self.owners = layout.owners[:, :, :entries]
        self.lane_map = layout.lane_map[:lanes]
        self.zero_limits = layout.zero_limits[:lanes]
        self.saved = tuple(values[:entries] for values in layout.saved)
        self.bytes = sum(values.numel() * values.element_size() for values in self.saved)
        self.captured = False

    def restore_arguments(self):
        """Tensor arguments of :func:`restore_stable_target_rows` after ``counts``."""
        layout = self.layout
        return (self.lane_map, self.indices, self.owners, layout.destinations, self.saved,
                layout.destination_groups)

    def capture(self, batch: DSparkTargetBatch, source_lanes) -> None:
        layout = self.layout
        if layout.owner is not None:
            raise RuntimeError("stable target snapshot is still owned by an unfinished round")
        if batch.batch_capacity != self.capacity or batch.row_valid.numel() != target_rows_for_capacity(self.capacity):
            raise ValueError("stable target snapshot rows disagree with the execution bucket")
        if batch.positions.device != layout.device:
            raise ValueError("stable target snapshot requires batch metadata on the state device")
        source_lanes = tuple(source_lanes)
        if len(source_lanes) != self.capacity:
            raise ValueError("stable target snapshot requires one source lane per execution lane")
        # The device lane map is the target's own request-slot tensor.  It is
        # authoritative only while state lane == logical slot, which the
        # runner's canonical plan guarantees; verify that on host tuples.
        for lane, (active, slot) in enumerate(zip(batch.host_request_active, batch.host_request_slots,
                                                  strict=True)):
            if active and source_lanes[lane] != slot:
                raise ValueError("stable target snapshot requires state lane == logical slot for active lanes")
        layout._run("capture", capture_stable_target_rows, batch.positions, batch.row_slots, batch.row_valid,
                    batch.row_pages, batch.request_slots, layout.groups, self.indices, self.owners, self.lane_map,
                    layout.destinations, self.saved, layout.destination_groups)
        layout.owner = self
        self.captured = True

    def _require_captured(self):
        if not self.captured or self.layout.owner is not self:
            raise RuntimeError("stable target snapshot has no captured round")

    def restore(self, committed=None) -> None:
        self._require_captured()
        if committed is None:
            committed = (0, ) * self.capacity
        committed = tuple(int(value) for value in committed)
        if len(committed) != self.capacity:
            raise ValueError(f"partial target restore requires committed[B={self.capacity}]")
        if any(value < 0 or value > TARGET_ROWS_PER_REQUEST for value in committed):
            raise ValueError("committed lengths must be in 0..6")
        padding = (0, ) * (self.zero_limits.shape[0] - self.capacity)
        limits = (self.zero_limits if not any(committed) else
                  torch.tensor(committed + padding, dtype=torch.int64, device=self.layout.device))
        _restore_stable_target_lanes(limits, self.indices, self.owners, self.layout.destinations, self.saved,
                                     self.layout.destination_groups)

    def restore_device(self, committed) -> None:
        self._require_captured()
        if (not isinstance(committed, torch.Tensor) or committed.shape != (self.state_capacity, )
                or committed.dtype not in (torch.int32, torch.int64)):
            raise ValueError("stable device target restore requires integer counts for every state lane")
        if committed.device != self.layout.device:
            raise ValueError("device target restore counts must share the snapshot device")
        # HPU compiled graphs write through the native state writer, like the
        # fused transaction; both variants leave bit-identical state.
        native = self.layout.device.type == "hpu"
        self.layout._run("restore", restore_stable_target_rows, committed, *self.restore_arguments(),
                         native_writes=native)

    def release(self) -> None:
        if self.layout.owner is self:
            self.layout.owner = None
        self.captured = False


@dataclass(frozen=True)
class DSparkTargetTicket:
    """Round identity plus the state snapshot owned by the target transaction."""

    batch: DSparkTargetBatch
    snapshot: DSparkTargetSnapshot | DSparkStableTargetSnapshot

    @property
    def round_generation(self) -> int:
        return self.batch.host_round_generation

    @property
    def slot_generations(self) -> tuple[int, ...]:
        return self.batch.host_slot_generations

    def restore(self, committed=None) -> None:
        self.snapshot.restore(committed)


_MASKED_SEQUENCE_NAME = "custom_deepseek_v41_compressor_sequence_masked_bf16_gaudi2"


def _resolve_masked_sequence_op():
    op = getattr(torch.ops.custom_op, _MASKED_SEQUENCE_NAME, None)
    if op is None:
        raise RuntimeError(
            "The unified runtime does not register the required "
            f"{_MASKED_SEQUENCE_NAME}; falling back to the unmasked sequence op is forbidden"
        )
    return op


def compressor_sequence_slot_publish_read(
    history_bank: torch.Tensor,
    score_bank: torch.Tensor,
    slot: int,
    kv: torch.Tensor,
    score: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish fixed-six rows and read one request-local history bank.

    Invalid rows use row id ``-1`` and cannot write.  The returned tensors are
    the completion-dependent eight-row scratch consumed by either compressor
    arithmetic arm.
    """
    if not isinstance(slot, int) or slot < 0:
        raise ValueError("compressor slot must be a host-static nonnegative integer")
    if history_bank.ndim != 2 or history_bank.shape[1] != 512 or score_bank.shape != history_bank.shape:
        raise ValueError("compressor slot bank must be [capacity*8, 512]")
    if history_bank.shape[0] % 8:
        raise ValueError("compressor slot bank row count must be a multiple of eight")
    if slot >= history_bank.shape[0] // 8:
        raise ValueError("compressor slot is outside the request bank")
    if kv.ndim != 2 or kv.shape != (6, 512) or score.shape != kv.shape or positions.shape != (6,):
        raise ValueError("masked compressor rows must be fixed [6,512] with positions [6]")
    from vllm_gaudi.ops.deepseek_v41_batch_attention import read_state_rows, write_state_rows

    positions = positions.to(torch.int32).contiguous()
    rows = torch.where(
        positions >= 0,
        slot * _HISTORY_ROWS + positions.remainder(_HISTORY_ROWS),
        torch.full_like(positions, -1),
    ).contiguous()
    done_history = write_state_rows(history_bank, kv.contiguous(), rows)
    done_score = write_state_rows(score_bank, score.contiguous(), rows)
    completion_history = done_history.amax().expand(_HISTORY_ROWS).contiguous()
    completion_score = done_score.amax().expand(_HISTORY_ROWS).contiguous()
    slot_rows = torch.arange(
        slot * _HISTORY_ROWS,
        (slot + 1) * _HISTORY_ROWS,
        dtype=torch.int32,
        device=history_bank.device,
    )
    history = read_state_rows(history_bank, slot_rows, completion_history)
    score_history = read_state_rows(score_bank, slot_rows, completion_score)
    return history, score_history



def compressor_sequence_runtime_slot_publish_read(
    history_bank: torch.Tensor,
    score_bank: torch.Tensor,
    slot: torch.Tensor,
    kv: torch.Tensor,
    score: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish a C6 execution lane to a runtime-selected persistent slot.

    The host metadata owns slot/generation validation. Invalid tensor slots
    produce no writes and zero reads, so padded execution lanes cannot touch
    another request. Slot values stay on device and do not specialize graphs.
    """
    if (not isinstance(slot, torch.Tensor) or slot.ndim != 0
            or slot.dtype not in (torch.int32, torch.int64) or slot.device != history_bank.device):
        raise ValueError("runtime compressor slot must be a device integer scalar")
    if (history_bank.ndim != 2 or history_bank.shape[1] != 512 or score_bank.shape != history_bank.shape
            or history_bank.shape[0] % _HISTORY_ROWS):
        raise ValueError("compressor slot bank must be [capacity*8, 512]")
    if kv.shape != (6, 512) or score.shape != kv.shape or positions.shape != (6,):
        raise ValueError("runtime compressor requires C6 values and positions")
    from vllm_gaudi.ops.deepseek_v41_batch_attention import read_state_rows, write_state_rows

    positions = positions.to(torch.int32).contiguous()
    slot = slot.to(torch.int32)
    valid_slot = (slot >= 0) & (slot < history_bank.shape[0] // _HISTORY_ROWS)
    rows = torch.where(
        valid_slot & (positions >= 0),
        slot * _HISTORY_ROWS + positions.remainder(_HISTORY_ROWS),
        torch.full_like(positions, -1),
    ).contiguous()
    done_history = write_state_rows(history_bank, kv.contiguous(), rows)
    done_score = write_state_rows(score_bank, score.contiguous(), rows)
    offsets = torch.arange(_HISTORY_ROWS, dtype=torch.int32, device=history_bank.device)
    slot_rows = torch.where(valid_slot, slot * _HISTORY_ROWS + offsets, -1).contiguous()
    history = read_state_rows(history_bank, slot_rows, done_history.amax().expand(_HISTORY_ROWS).contiguous())
    scores = read_state_rows(score_bank, slot_rows, done_score.amax().expand(_HISTORY_ROWS).contiguous())
    return history, scores


def compressor_sequence_slot_torch_mix(
    history: torch.Tensor,
    score_history: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Preserve the old C2-C6 compiled mix boundary on published scratch rows."""
    if history.shape != (8, 512) or score_history.shape != history.shape or positions.shape != (6,):
        raise ValueError("compiled compressor mix requires [8,512] history and six positions")
    first = torch.bitwise_and(positions, -2)
    a = torch.bitwise_and(first, _HISTORY_ROWS - 1).long()
    b = torch.bitwise_and(first + 1, _HISTORY_ROWS - 1).long()
    gates = torch.stack((score_history[a], score_history[b]), 1).softmax(1)
    return (history[a] * gates[:, 0] + history[b] * gates[:, 1]).to(torch.bfloat16)


def compressor_sequence_slot_native_mix(
    history: torch.Tensor,
    score_history: torch.Tensor,
    kv: torch.Tensor,
    score: torch.Tensor,
    positions: torch.Tensor,
    *,
    sequence_op=None,
) -> torch.Tensor:
    """Run the qualified native C1 latent boundary on published scratch rows."""
    if history.shape != (8, 512) or score_history.shape != history.shape:
        raise ValueError("native compressor mix requires [8,512] history scratch")
    if kv.shape != (6, 512) or score.shape != kv.shape or positions.shape != (6,):
        raise ValueError("native compressor mix requires fixed [6,512] rows")
    if sequence_op is None:
        sequence_op = _resolve_masked_sequence_op()
    result = sequence_op(history, score_history, kv.contiguous(), score.contiguous(), positions)
    if result.shape != (6, 512):
        raise ValueError("masked sequence op must return [6,512]")
    return result


def select_compressor_norm_by_request_length(
    native_norm: torch.Tensor,
    compiled_norm: torch.Tensor,
    row_valid: torch.Tensor,
) -> torch.Tensor:
    """Select C1 native and C2-C6 compiled results after normalization."""
    if native_norm.ndim != 2 or native_norm.shape[1] != 512 or compiled_norm.shape != native_norm.shape:
        raise ValueError("compressor norm selection requires matching [6*N,512] post-norm tensors")
    rows = native_norm.shape[0]
    if rows % TARGET_ROWS_PER_REQUEST or row_valid.shape != (rows,):
        raise ValueError("compressor norm selection requires one [6*N] row-valid vector")
    capacity = rows // TARGET_ROWS_PER_REQUEST
    logical_lengths = row_valid.to(torch.bool).reshape(capacity, TARGET_ROWS_PER_REQUEST).sum(1)
    use_native = (logical_lengths == 1)[:, None].expand(capacity, TARGET_ROWS_PER_REQUEST).reshape(rows)
    return torch.where(use_native[:, None], native_norm, compiled_norm)


def compressor_sequence_slot_masked(
    history_bank: torch.Tensor,
    score_bank: torch.Tensor,
    slot: int,
    kv: torch.Tensor,
    score: torch.Tensor,
    positions: torch.Tensor,
    *,
    sequence_op=None,
) -> torch.Tensor:
    """Run the qualified native C1 latent boundary on published scratch rows."""
    history, score_history = compressor_sequence_slot_publish_read(
        history_bank, score_bank, slot, kv, score, positions)
    return compressor_sequence_slot_native_mix(
        history, score_history, kv, score, positions, sequence_op=sequence_op)


def make_target_ticket(program, batch: DSparkTargetBatch) -> DSparkTargetTicket:
    return DSparkTargetTicket(batch=batch, snapshot=DSparkTargetSnapshot.capture(program, batch))


def initialize_target_batch_state(program, capacity: int | None = None):
    """Install the existing request-slot bank without changing model policy gates."""
    if capacity is None:
        capacity = TARGET_REQUESTS
    target_rows_for_capacity(capacity)
    state = getattr(program, "batch_state", None)
    if state is None:
        from vllm_gaudi.ops.deepseek_v41_batch_state import BatchStageState

        state = BatchStageState(program, capacity)
        program.batch_state = state
    if getattr(state, "capacity", None) != capacity:
        raise ValueError("existing batch state capacity differs from the target contract")
    return state


__all__ = [
    "DSparkStableTargetLayout",
    "DSparkStableTargetSnapshot",
    "DSparkTargetBatch",
    "DSparkTargetSnapshot",
    "DSparkTargetTicket",
    "TARGET_REQUESTS",
    "TARGET_REQUEST_OFFSETS",
    "TARGET_ROWS",
    "TARGET_ROWS_PER_REQUEST",
    "capture_stable_target_rows",
    "compressor_sequence_slot_masked",
    "compressor_sequence_slot_native_mix",
    "compressor_sequence_slot_publish_read",
    "compressor_sequence_runtime_slot_publish_read",
    "compressor_sequence_slot_torch_mix",
    "initialize_target_batch_state",
    "make_target_ticket",
    "restore_stable_target_rows",
    "select_compressor_norm_by_request_length",
    "stable_target_restore_layout",
    "target_request_offsets",
    "target_rows_for_capacity",
]
