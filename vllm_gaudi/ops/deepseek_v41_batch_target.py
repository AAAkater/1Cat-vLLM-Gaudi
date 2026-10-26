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

from dataclasses import dataclass
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
        )

    @classmethod
    def from_metadata(cls,
                      metadata,
                      queries,
                      pages,
                      *,
                      batch_capacity: int | None = None,
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

        active_slots = [slot for active, slot in zip(request_active, request_slots, strict=True) if active]
        if (len(set(active_slots)) != len(active_slots)
                or any(slot < 0 or slot >= capacity for slot in active_slots)):
            raise ValueError("active request_slots must be distinct nonnegative owners")
        for index, (active, slot, generation) in enumerate(
                zip(request_active, request_slots, generations, strict=True)):
            if not active and (slot != -1 or generation != 0):
                raise ValueError(f"inactive target request {index} must use slot -1 and generation zero")
            if active and generation < 0:
                raise ValueError("active slot_generations must be nonnegative")
        canonical_slots = (batch_capacity is not None or capacity != TARGET_REQUESTS
                           or (active_length > 0 and not all(request_active)))
        if canonical_slots:
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


@dataclass(frozen=True)
class DSparkTargetTicket:
    """Round identity plus the state snapshot owned by the target transaction."""

    batch: DSparkTargetBatch
    snapshot: DSparkTargetSnapshot

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
    "DSparkTargetBatch",
    "DSparkTargetSnapshot",
    "DSparkTargetTicket",
    "TARGET_REQUESTS",
    "TARGET_REQUEST_OFFSETS",
    "TARGET_ROWS",
    "TARGET_ROWS_PER_REQUEST",
    "compressor_sequence_slot_masked",
    "compressor_sequence_slot_native_mix",
    "compressor_sequence_slot_publish_read",
    "compressor_sequence_slot_torch_mix",
    "initialize_target_batch_state",
    "make_target_ticket",
    "select_compressor_norm_by_request_length",
    "target_request_offsets",
    "target_rows_for_capacity",
]
