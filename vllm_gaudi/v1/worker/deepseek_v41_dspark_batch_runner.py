# SPDX-License-Identifier: Apache-2.0
"""Scheduler-facing fixed-capacity DSpark transaction adapter for greedy PP1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import torch

from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm_gaudi.ops.deepseek_v41_capacity import validate_dspark_capacity
from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata
from vllm_gaudi.ops.deepseek_v41_active_batch import DSparkActiveBatchMetadata, compact_dspark_metadata
from vllm_gaudi.ops.deepseek_v41_verify import vocab_parallel_argmax


_MAX_DRAFT_TOKENS = 5
_TARGET_QUERY_BUCKET = 6
_SCHEDULER_PAGE_TOKENS = 128
_DSPARK_TILE_WARMUP_CAPACITY = 32
_TRACE_SCHEMA = "dsv41.dspark.round.v2"

logger = logging.getLogger(__name__)


class _CompletedEvent:
    @staticmethod
    def query() -> bool:
        return True


def _host_integer(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _batch_capacity(value: Any, name: str = "batch capacity") -> int:
    return validate_dspark_capacity(value, allow_serial=True, name=name)


def _fixed_page_table_width(max_model_len: int) -> int:
    """Return the fixed scheduler page-table width for one model context."""
    if type(max_model_len) is not int or max_model_len < 1:
        raise ValueError("max_model_len must be a positive integer")
    return (max_model_len + _SCHEDULER_PAGE_TOKENS - 1) // _SCHEDULER_PAGE_TOKENS


def warmup_scenarios(capacity: int) -> tuple[tuple[str, dict[int, int]], ...]:
    """Return bounded host-lane patterns exercised before serving.

    The original three patterns remain the complete contract for capacities
    other than the N32 packed-MLA profile.  N32 additionally needs one sparse
    round for every packed-MLA tile-plan bucket.  The added lane sets use the
    minimum active lanes that reach each bucket; ``reuse_c1`` intentionally
    repeats bucket one to retain the physical-slot generation/reuse check.
    """
    if type(capacity) is not int or capacity < 1:
        raise ValueError("DSpark warmup capacity must be a positive integer")

    full_lengths = {lane: (_TARGET_QUERY_BUCKET if lane % 2 == 0 else 1) for lane in range(capacity)}
    gap_lane = capacity - 1
    scenarios = (
        ("full_mixed_c1_c6", full_lengths),
        ("partial_gap", {gap_lane: _TARGET_QUERY_BUCKET}),
        ("reuse_c1", {gap_lane: 1}),
    )
    if capacity != _DSPARK_TILE_WARMUP_CAPACITY:
        return scenarios

    # Exact tile coverage at N32 (24 original eight-row tiles):
    # lane 1 -> {0,1}; lanes 1,5 -> {0,1,3,4};
    # lanes 1,2,5 -> {0..4}; lanes 0,2,5,9,13 -> 9 tiles.
    # Those select buckets 2, 4, 8 and 16 respectively.  full_mixed selects
    # bucket 24, while partial_gap/reuse_c1 select bucket 1.
    return scenarios + (
        ("tile_bucket_2", {1: _TARGET_QUERY_BUCKET}),
        ("tile_bucket_4", {1: _TARGET_QUERY_BUCKET, 5: _TARGET_QUERY_BUCKET}),
        ("tile_bucket_8", {
            1: _TARGET_QUERY_BUCKET,
            2: _TARGET_QUERY_BUCKET,
            5: _TARGET_QUERY_BUCKET,
        }),
        ("tile_bucket_16", {
            0: _TARGET_QUERY_BUCKET,
            2: _TARGET_QUERY_BUCKET,
            5: _TARGET_QUERY_BUCKET,
            9: _TARGET_QUERY_BUCKET,
            13: _TARGET_QUERY_BUCKET,
        }),
    )


def _component_capacity(value: Any) -> int | None:
    """Read one fixed batch capacity from a runner-owned component."""
    if value is None:
        return None
    for name in ("batch_capacity", "capacity"):
        observed = getattr(value, name, None)
        if observed is not None:
            return _batch_capacity(observed, f"{type(value).__name__}.{name}")
    slots = getattr(value, "slots", None)
    observed = getattr(slots, "capacity", None)
    return None if observed is None else _batch_capacity(observed, f"{type(value).__name__}.slots.capacity")


def _resolve_capacity(explicit: Any, *components: Any) -> int:
    """Resolve one explicit capacity or infer the legacy state capacity."""
    observed = [capacity for capacity in (_component_capacity(value) for value in components)
                if capacity is not None]
    if explicit is None:
        # Standalone two-lane callers predate the capacity argument.  A state-owned
        # bank is authoritative; the old two-lane default is the last resort.
        capacity = observed[0] if observed else 2
    else:
        capacity = _batch_capacity(explicit)
    if any(value != capacity for value in observed):
        raise ValueError(f"DSpark batch capacity {capacity} differs from an owned component: {observed}")
    return capacity


def _host_rows(value: Any, batch_size: int, name: str) -> list[list[int]]:
    """Normalize a batch output to Python rows without touching an accelerator."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU before scheduler publication")
        if value.ndim == 2:
            rows = value.tolist()
        elif value.ndim == 1:
            rows = value.tolist()
        else:
            raise ValueError(f"{name} must be one- or two-dimensional")
    else:
        rows = list(value)

    if rows and isinstance(rows[0], int):
        rows = [rows]
    if len(rows) != batch_size:
        raise ValueError(f"{name} must contain one row per request")

    normalized: list[list[int]] = []
    for row in rows:
        values = row.tolist() if isinstance(row, torch.Tensor) else list(row)
        row_values = []
        for item in values:
            item = _host_integer(item, name)
            row_values.append(item)
        normalized.append(row_values)
    return normalized


def _logical_rows(value: Any, lengths: Sequence[int], name: str) -> list[list[int]]:
    """Accept either fixed ``[B, 6]`` rows or flat logical packed rows."""
    batch_size = len(lengths)
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU before scheduler publication")
        if value.ndim == 2:
            if value.shape[0] != batch_size or value.shape[1] < max(lengths, default=0):
                raise ValueError(f"{name} has an incompatible packed shape")
            rows = value.tolist()
            return [[_host_integer(item, name) for item in row[:length]]
                    for row, length in zip(rows, lengths, strict=True)]
        if value.ndim == 1:
            if value.numel() != sum(lengths):
                raise ValueError(f"{name} flat row count differs from metadata")
            flat = value.tolist()
            rows, offset = [], 0
            for length in lengths:
                rows.append([_host_integer(item, name) for item in flat[offset:offset + length]])
                offset += length
            return rows
        raise ValueError(f"{name} must be one- or two-dimensional")
    return _host_rows(value, batch_size, name)


def _ragged_rows(value: Any, lengths: Sequence[int], name: str) -> list[list[int]]:
    """Accept exact ragged rows or fixed rows with trailing ``-1`` padding."""
    rows = _logical_rows(value, lengths, name)
    result = []
    for length, row in zip(lengths, rows, strict=True):
        trailing = len(row) - length
        if trailing < 0:
            raise ValueError(f"{name} row is shorter than its expected length")
        padded = row[length:]
        if any(token != -1 for token in padded):
            raise ValueError(f"{name} may only use trailing -1 padding")
        result.append(row[:length])
    return result


def _next_draft_rows(value: Any, batch_size: int, name: str) -> list[list[int]]:
    """Normalize fixed ``[B, 5]`` or ragged next-round draft rows."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.ndim != 2 or value.shape[0] != batch_size:
            raise ValueError(f"{name} must be a CPU [B, 5] tensor")
        rows = value.tolist()
    else:
        rows = list(value)
    if len(rows) != batch_size:
        raise ValueError(f"{name} must contain one row per request")
    result = []
    for row in rows:
        values = [_host_integer(item, name) for item in row]
        if len(values) > _MAX_DRAFT_TOKENS:
            raise ValueError(f"{name} rows may contain at most five ids")
        first_pad = next((index for index, token in enumerate(values) if token == -1), len(values))
        if any(token != -1 for token in values[first_pad:]):
            raise ValueError(f"{name} may only use trailing -1 padding")
        if any(token < 0 for token in values[:first_pad]):
            raise ValueError(f"{name} contains a negative draft id")
        result.append(values[:first_pad])
    return result


def _target_greedy_ids(program,
                       hidden: torch.Tensor,
                       host_active_lanes: tuple[int, ...] | None = None) -> torch.Tensor:
    """Select target ids with request-local C6 head projections."""
    if hidden.ndim != 2:
        raise ValueError("target hidden states must be a two-dimensional tensor")
    rows = hidden.shape[0]
    if rows == 0 or rows % _TARGET_QUERY_BUCKET != 0:
        raise ValueError(
            f"target hidden rows must be a positive multiple of {_TARGET_QUERY_BUCKET}, got {rows}")

    capacity = rows // _TARGET_QUERY_BUCKET
    if host_active_lanes is None:
        active_lanes = tuple(range(capacity))
    else:
        if not isinstance(host_active_lanes, tuple):
            raise TypeError("host_active_lanes must be a tuple or None")
        if not host_active_lanes:
            raise ValueError("host_active_lanes must contain at least one lane")
        previous_lane = -1
        for lane in host_active_lanes:
            if isinstance(lane, bool) or not isinstance(lane, int):
                raise TypeError("host_active_lanes must contain only integers")
            if lane < 0 or lane >= capacity:
                raise ValueError(
                    f"host_active_lanes entries must be in [0, {capacity}), got {lane}")
            if lane <= previous_lane:
                raise ValueError("host_active_lanes must be strictly increasing and unique")
            previous_lane = lane
        active_lanes = host_active_lanes

    target_ids = torch.zeros((rows, ), dtype=torch.int64, device=hidden.device)
    for lane in active_lanes:
        start = lane * _TARGET_QUERY_BUCKET
        lane_hidden = hidden[start:start + _TARGET_QUERY_BUCKET]
        lane_ids = vocab_parallel_argmax(
            program._head_projection(lane_hidden),
            program.tp_rank,
            program.all_gather,
        )
        target_ids[start:start + _TARGET_QUERY_BUCKET] = lane_ids
    return target_ids


@dataclass(frozen=True)
class DSparkBatchRequest:
    req_id: str
    request: Any
    scheduler_count: int
    scheduler_draft_count: int
    draft_token_ids: tuple[int, ...]
    target_query_length: int
    target_tokens: tuple[int, ...]
    target_positions: tuple[int, ...]
    remaining_output_budget: int
    request_slot: int
    slot_generation: int
    placeholder: bool

    @property
    def scheduler_num_rejected(self) -> int:
        return self.scheduler_draft_count - (self.target_query_length - 1)

    @property
    def lane(self) -> int:
        """Physical request lane and canonical persistent slot."""
        return self.request_slot


@dataclass(frozen=True)
class DSparkBatchPlan:
    """One all-request packed target/draft transaction."""

    requests: tuple[DSparkBatchRequest, ...]
    metadata: DSparkBatchMetadata
    round_generation: int
    packed_input_ids: torch.Tensor
    packed_positions: torch.Tensor
    packed_padding_mask: torch.Tensor
    pages: torch.Tensor
    active_metadata: DSparkActiveBatchMetadata | None = None

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(item.req_id for item in self.requests)

    @property
    def capacity(self) -> int:
        return self.metadata.capacity

    @property
    def active_mask(self) -> tuple[bool, ...]:
        return tuple(bool(value) for value in self.metadata.request_active.tolist())

    @property
    def active_lanes(self) -> tuple[int, ...]:
        return tuple(item.request_slot for item in self.requests)

    @property
    def lane_request_ids(self) -> tuple[str | None, ...]:
        return tuple(self.metadata.request_ids)

    @property
    def entry_by_lane(self) -> dict[int, DSparkBatchRequest]:
        return {item.request_slot: item for item in self.requests}

    @property
    def target_lengths(self) -> tuple[int, ...]:
        return tuple(item.target_query_length for item in self.requests)

    @property
    def flat_input_ids(self) -> torch.Tensor:
        lengths = self.target_lengths
        return torch.cat(
            [self.packed_input_ids[item.lane, :length]
             for item, length in zip(self.requests, lengths, strict=True)]
        ).to(torch.int64)

    @property
    def flat_positions(self) -> torch.Tensor:
        lengths = self.target_lengths
        return torch.cat(
            [self.packed_positions[item.lane, :length]
             for item, length in zip(self.requests, lengths, strict=True)]
        ).to(torch.int64)

    @classmethod
    def from_scheduler(
        cls,
        scheduled: Any,
        requests: Mapping[str, Any],
        slot_owners: Mapping[str, Any],
        *,
        max_model_len: int,
        round_generation: int,
        batch_capacity: int | None = None,
        use_active_metadata: bool = True,
    ) -> "DSparkBatchPlan":
        if type(use_active_metadata) is not bool:
            raise TypeError("use_active_metadata must be a boolean")
        max_model_len = _host_integer(max_model_len, "max_model_len")
        fixed_page_width = _fixed_page_table_width(max_model_len)
        scheduled_order = tuple(scheduled.num_scheduled_tokens)
        if not scheduled_order:
            raise ValueError("DSpark batch planning requires at least one active request")
        if len(set(scheduled_order)) != len(scheduled_order):
            raise ValueError("DSpark request IDs must be unique")
        if int(round_generation) <= 0:
            raise ValueError("DSpark requires a positive round generation")

        active_requests: list[tuple[str, Any, Any, int, int]] = []
        for req_id in scheduled_order:
            request = requests.get(req_id)
            if request is None:
                raise ValueError(f"DSpark request {req_id!r} is absent from runner state")
            owner = slot_owners.get(req_id)
            if owner is None:
                raise ValueError(f"DSpark request {req_id!r} has no request-slot owner")
            slot = _host_integer(owner.index, "request slot")
            generation = _host_integer(owner.generation, "slot generation")
            if slot < 0 or generation < 1:
                raise ValueError("DSpark request slot and generation must be nonnegative")
            active_requests.append((req_id, request, owner, slot, generation))

        occupied_lanes = [slot for _req_id, _request, _owner, slot, _generation in active_requests]
        if len(set(occupied_lanes)) != len(occupied_lanes):
            raise ValueError("DSpark request slots must be distinct")
        if batch_capacity is None:
            # Legacy standalone callers have no separate slot bank.  Derive
            # the smallest fixed capacity that contains every physical owner.
            capacity = _batch_capacity(max(occupied_lanes) + 1)
        else:
            capacity = _batch_capacity(batch_capacity)
        if any(slot >= capacity for slot in occupied_lanes):
            raise ValueError("DSpark request slot is outside the fixed batch capacity")

        active_requests.sort(key=lambda item: item[3])
        active_mask = [False] * capacity
        lane_ids: list[str | None] = [None] * capacity
        target_lengths = [0] * capacity
        draft_rows: list[list[int]] = [[] for _ in range(capacity)]
        slots = [-1] * capacity
        generations = [0] * capacity
        budgets = [0] * capacity
        starts = [0] * capacity
        request_pages: list[list[int] | None] = [None] * capacity
        entries: list[DSparkBatchRequest] = []
        raw_packed_ids = torch.zeros((capacity, _TARGET_QUERY_BUCKET), dtype=torch.int64)
        raw_packed_positions = torch.full((capacity, _TARGET_QUERY_BUCKET), -1, dtype=torch.int32)
        raw_packed_mask = torch.ones((capacity, _TARGET_QUERY_BUCKET), dtype=torch.bool)

        for req_id, request, _owner, lane, generation in active_requests:
            if lane >= capacity:
                raise ValueError("DSpark request lane is outside the fixed batch capacity")
            active_mask[lane] = True
            lane_ids[lane] = req_id
            slots[lane] = lane
            generations[lane] = generation

            count = int(scheduled.num_scheduled_tokens[req_id])
            if not 1 <= count <= _TARGET_QUERY_BUCKET:
                raise ValueError(f"DSpark scheduled count for {req_id!r} must be in 1..6")
            if getattr(request, "mm_features", None):
                raise ValueError("DSpark greedy PP1 excludes multimodal requests")
            if request.num_computed_tokens < request.decode_start:
                raise ValueError(f"DSpark request {req_id!r} is not in steady decode")

            sampling_params = getattr(request, "sampling_params", None)
            if sampling_params is None or getattr(sampling_params, "temperature", None) != 0:
                raise ValueError("DSpark is greedy-only; temperature must be 0")

            raw_drafts = list(scheduled.scheduled_spec_decode_tokens.get(req_id, ()))
            placeholder = bool(raw_drafts) and all(token < 0 for token in raw_drafts)
            if placeholder:
                if any(token >= 0 for token in raw_drafts) or len(raw_drafts) > _MAX_DRAFT_TOKENS:
                    raise ValueError(f"DSpark placeholder row for {req_id!r} is malformed")
                if count != len(raw_drafts) + 1:
                    raise ValueError(f"DSpark placeholder count for {req_id!r} is inconsistent")
                drafts: list[int] = []
            else:
                if any(token < 0 for token in raw_drafts):
                    raise ValueError(f"DSpark draft row for {req_id!r} mixes real and padding ids")
                if len(raw_drafts) > _MAX_DRAFT_TOKENS:
                    raise ValueError(f"DSpark draft row for {req_id!r} exceeds five proposals")
                if count != len(raw_drafts) + 1:
                    raise ValueError(f"DSpark scheduled count for {req_id!r} must equal drafts + 1")
                drafts = [_host_integer(token, "scheduled draft token") for token in raw_drafts]

            start = int(request.num_computed_tokens)
            target_length = len(drafts) + 1
            if start < 0 or target_length > _TARGET_QUERY_BUCKET:
                raise ValueError(f"DSpark target range for {req_id!r} is invalid")
            if start + target_length > max_model_len:
                raise ValueError(f"DSpark target range for {req_id!r} exceeds max_model_len")
            anchor = _host_integer(request.token_at(start), "target anchor token")
            target_tokens = [anchor, *drafts]
            target_positions = list(range(start, start + target_length))

            max_tokens = getattr(sampling_params, "max_tokens", None)
            if max_tokens is None:
                raise ValueError("DSpark requires an output token budget")
            remaining = int(max_tokens) - len(request.output)
            if remaining <= 0:
                raise ValueError(f"DSpark request {req_id!r} has no remaining output budget")

            raw_packed_ids[lane, :target_length] = torch.tensor(target_tokens, dtype=torch.int64)
            raw_packed_positions[lane, :target_length] = torch.tensor(target_positions, dtype=torch.int32)
            raw_packed_mask[lane, :target_length] = False
            block_groups = getattr(request, "block_ids", ())
            if len(block_groups) != 1:
                raise ValueError("DSpark expects one scheduler page group")
            blocks = [int(block) for block in block_groups[0]]
            required_pages = (start + target_length + _SCHEDULER_PAGE_TOKENS - 1) // _SCHEDULER_PAGE_TOKENS
            if len(blocks) < required_pages or any(block <= 0 for block in blocks):
                raise ValueError("DSpark target range is not backed by scheduler pages")
            if len(blocks) > fixed_page_width:
                raise ValueError(
                    "DSpark scheduler page count exceeds fixed context width: "
                    f"request={req_id!r} pages={len(blocks)} fixed_width={fixed_page_width}"
                )
            target_lengths[lane] = target_length
            draft_rows[lane] = drafts
            budgets[lane] = remaining
            starts[lane] = start
            request_pages[lane] = blocks
            entries.append(
                DSparkBatchRequest(
                    req_id=req_id,
                    request=request,
                    scheduler_count=count,
                    scheduler_draft_count=len(raw_drafts),
                    draft_token_ids=tuple(drafts),
                    target_query_length=target_length,
                    target_tokens=tuple(target_tokens),
                    target_positions=tuple(target_positions),
                    remaining_output_budget=remaining,
                    request_slot=lane,
                    slot_generation=generation,
                    placeholder=placeholder,
                )
            )

        active_pages = [blocks for blocks in request_pages if blocks is not None]
        if not active_pages:
            raise ValueError("DSpark batch planning requires at least one active request")
        # The target graph has a fixed context capacity, while a request's
        # scheduler block list only grows with its current prefix.  Publish a
        # stable table width so the native index ABI sees fixed coverage.  The
        # per-request required-page check above remains authoritative; filler
        # repeats an already validated positive scheduler page and is never
        # visible outside the current target range.
        page_width = fixed_page_width
        dummy_page = active_pages[0][0]
        padded_pages: list[list[int]] = []
        for blocks in request_pages:
            source = blocks if blocks is not None else [dummy_page]
            padded_pages.append(source + [source[-1]] * (page_width - len(source)))
        target_padding = [False] * sum(target_lengths)
        metadata = DSparkBatchMetadata.build(
            requests=lane_ids,
            target_query_lengths=target_lengths,
            draft_token_ids=draft_rows,
            request_slots=slots,
            slot_generations=generations,
            remaining_output_budget=budgets,
            target_padding_mask=target_padding,
            target_start_positions=starts,
            round_generation=round_generation,
            active=active_mask,
        )
        active_metadata = None
        if use_active_metadata:
            active_metadata = compact_dspark_metadata(
                metadata,
                scheduler_order=scheduled_order,
                owners=slot_owners,
                expected_round_generation=round_generation,
            )
        return cls(
            requests=tuple(entries),
            metadata=metadata,
            round_generation=int(round_generation),
            packed_input_ids=raw_packed_ids,
            packed_positions=raw_packed_positions,
            packed_padding_mask=raw_packed_mask,
            pages=torch.tensor(padded_pages, dtype=torch.int32),
            active_metadata=active_metadata,
        )


@dataclass(frozen=True)
class DSparkTargetBatch:
    """Opaque actual target-module result plus logical target token ids."""

    token_ids: Any
    payload: Any = None


class DSparkPackedTargetStages:
    """Own one fixed-capacity target graph for every active-lane pattern."""

    def __init__(self,
                 program,
                 batch_state,
                 *,
                 device,
                 engram_host=None,
                 backend="hpu_backend",
                 batch_capacity: int | None = None):
        from vllm_gaudi.models.deepseek_v41_batch_target_program import DSparkTargetStage

        self.program = program
        self.batch_state = batch_state
        self.device = torch.device(device)
        self.engram_host = engram_host
        self.batch_capacity = _resolve_capacity(batch_capacity, batch_state)
        rows = self.batch_capacity * _TARGET_QUERY_BUCKET
        if engram_host is not None:
            max_tokens = getattr(engram_host, "max_tokens", None)
            if max_tokens is not None and max_tokens < rows:
                raise ValueError(
                    f"DSpark Engram staging capacity {max_tokens} is smaller than fixed target rows {rows}")
        self.stage_key = tuple(range(self.batch_capacity))
        self.stage = DSparkTargetStage(
            program,
            (6, ) * self.batch_capacity,
            self.stage_key,
            backend=backend,
        )
        self._bug004_profile = None
        self._backend = backend
        self.execution_buckets = (self.batch_capacity,)
        # One mapping entry is retained for older introspection callers.  The
        # key is physical lane order, not request order or an active-lane set.
        self.stages = {self.stage_key: self.stage}
        hidden_size = int(program.config["text_config"]["hidden_size"])
        self.hidden_seed = torch.zeros((rows, 4, hidden_size), dtype=torch.bfloat16, device=self.device)
        self.pre_mix_seed = torch.zeros((rows, 4), dtype=torch.float32, device=self.device)

    def scoped_recipe_status(self):
        return self.stage.scoped_recipe_status()

    def configure_execution_buckets(self, enabled: bool, *, buckets=(1, 2, 4, 8, 16, 32, 64)):
        from vllm_gaudi.ops.deepseek_v41_target_buckets import target_execution_buckets

        if type(enabled) is not bool:
            raise TypeError("dsv41_dspark_target_buckets must be a boolean")
        selected = target_execution_buckets(self.batch_capacity, buckets)
        self.execution_buckets = selected if enabled else (self.batch_capacity,)
        self.stage.compiled.execution_bucket_count = len(self.execution_buckets)
        # Buckets remove active-pattern specialization. Each bucket uses its
        # dense packed MLA call, including a supported partial tail tile.
        self.stage.compiled.use_sparse_mla_plan = not enabled

    def stage_for(self, batch):
        from vllm_gaudi.models.deepseek_v41_batch_target_program import DSparkTargetStage

        capacity = batch.batch_capacity
        if capacity not in getattr(self, "execution_buckets", (self.batch_capacity,)):
            raise ValueError("DSpark target execution bucket was not selected for warmup")
        if capacity == self.batch_capacity:
            if batch.host_request_slots != self.stage_key:
                raise ValueError("DSpark canonical target host slots must be range(batch_capacity)")
            return self.stage
        key = tuple(range(capacity))
        if key not in self.stages:
            stage = DSparkTargetStage(
                self.program, (6,) * capacity, key, backend=self._backend,
                state_capacity=self.batch_capacity,
            )
            stage.compiled.execution_bucket_count = len(self.execution_buckets)
            stage.compiled.use_sparse_mla_plan = False
            self.stages[key] = stage
        return self.stages[key]

    def enter_batched(self):
        """Retire the selected native B1 owner before packed target reads state."""
        self.batch_state.leave_single()
        self.batch_state.restore_single_bindings()
        self.batch_state._invalidate_index_mirror()

    def forward_batch(self, metadata, input_ids, positions, padding_mask, pages):
        del positions, padding_mask
        from vllm_gaudi.ops.deepseek_v41_batch_target import DSparkTargetBatch as TargetBatch

        profile = self._bug004_profile_for_round()
        from vllm_gaudi.ops.deepseek_v41_target_buckets import pack_target_bucket

        active_count = int(metadata.request_active.sum())
        execution_capacity = next(bucket for bucket in getattr(
            self, "execution_buckets", (self.batch_capacity,)) if bucket >= active_count)
        packed = pack_target_bucket(metadata, input_ids, pages, execution_capacity)
        compact = execution_capacity != self.batch_capacity
        input_ids = packed.queries
        batch = TargetBatch.from_metadata(
            packed.metadata,
            input_ids,
            packed.pages,
            batch_capacity=execution_capacity,
            state_capacity=self.batch_capacity if compact else None,
            device=self.device,
        )
        source_indices = (torch.tensor(packed.source_lanes[:active_count], dtype=torch.int64, device=self.device)
                          if compact else None)
        stage = self.stage_for(batch)
        ticket = None
        engram_owner = None
        try:
            if profile is not None:
                profile.mark("target.snapshot")
            ticket = stage.snapshot(batch)
            if profile is not None:
                profile.mark("target.engram")
            engram_rows, engram_owner = self._engram_rows(batch, input_ids)
            if profile is not None:
                profile.mark("target.forward")
            rows = batch.target_rows
            hidden, aux = stage(self.hidden_seed[:rows], self.pre_mix_seed[:rows], batch, engram_rows)
            if profile is not None:
                profile.mark("target.head_projection")
            target_ids = _target_greedy_ids(self.program, hidden, batch.host_active_lanes)
            if compact:
                expanded_ids = torch.zeros((self.batch_capacity, 6), dtype=target_ids.dtype, device=self.device)
                expanded_ids.index_copy_(0, source_indices, target_ids.reshape(execution_capacity, 6)[:active_count])
                target_ids = expanded_ids.reshape(-1)
                if not isinstance(aux, torch.Tensor) or aux.shape[0] != rows:
                    raise ValueError("compact target must return one aux state per execution row")
                expanded_aux = torch.zeros((self.batch_capacity, 6, *aux.shape[1:]),
                                           dtype=aux.dtype, device=aux.device)
                expanded_aux.index_copy_(0, source_indices, aux.reshape(execution_capacity, 6, *aux.shape[1:])[
                    :active_count])
                aux = expanded_aux.reshape(self.batch_capacity * 6, *aux.shape[1:])
            payload = {
                "hidden": hidden,
                "aux": aux,
                "batch": batch,
                "ticket": ticket,
                "engram_owner": engram_owner,
                "finished": False,
                "source_lanes": packed.source_lanes,
                "target_execution_capacity": execution_capacity,
                "target_execution_rows": rows,
                "target_execution_slots": batch.host_request_slots,
            }
            return DSparkTargetBatch(token_ids=target_ids, payload=payload)
        except BaseException as error:
            if ticket is not None:
                try:
                    ticket.restore((0, ) * batch.batch_capacity)
                except BaseException as cleanup_error:
                    if hasattr(error, "add_note"):
                        error.add_note(
                            f"DSpark target rollback failed: {type(cleanup_error).__name__}: {cleanup_error}")
            if engram_owner is not None:
                owner, engram_ticket = engram_owner
                try:
                    self._complete_engram_zero(owner, engram_ticket, sum(batch.host_request_active))
                except BaseException as cleanup_error:
                    if hasattr(error, "add_note"):
                        error.add_note(
                            f"DSpark Engram rollback failed: {type(cleanup_error).__name__}: {cleanup_error}")
            raise

    def _bug004_profile_for_round(self):
        return getattr(self, "_bug004_profile", None)

    def _engram_rows(self, batch, input_ids):
        owner = self.engram_host
        if owner is None:
            return (), None
        if not isinstance(input_ids, torch.Tensor) or input_ids.device.type != "cpu":
            raise ValueError("DSpark Engram spans require original CPU token rows")
        spans = []
        for lane, request_id in enumerate(batch.request_ids):
            if not batch.host_request_active[lane]:
                continue
            length = batch.request_lengths[lane]
            tokens = tuple(int(value) for value in input_ids[lane, :length].tolist())
            start = batch.host_positions[lane * _TARGET_QUERY_BUCKET]
            if start < 0:
                raise RuntimeError("Active DSpark Engram span has no target start position")
            spans.append((request_id, start, tokens, [False] * length))
        if not spans:
            return (), None
        capacity = batch.target_rows
        real_indices = batch.real_indices.to(self.device)
        real_count = real_indices.numel()
        ticket = None
        try:
            ticket = owner.prepare_batch(spans, capacity=capacity, defer_wait=True)
            contract = getattr(ticket, "batch", None)
            if contract is not None:
                ticket_capacity = getattr(contract, "capacity", None)
                if ticket_capacity is not None and ticket_capacity != capacity:
                    raise ValueError(
                        f"DSpark Engram ticket capacity {ticket_capacity} differs from fixed target rows {capacity}")
                members = getattr(contract, "members", None)
                if members is not None and len(members) != len(spans):
                    raise ValueError("DSpark Engram ticket member count differs from active spans")
                hash_ids = getattr(contract, "hash_ids", None)
                hash_shape = getattr(hash_ids, "shape", None)
                if hash_shape is not None and hash_shape[0] != real_count:
                    raise ValueError("DSpark Engram hash rows differ from active target rows")
            packed = owner.wait(ticket)
            physical = []
            for layer_rows in packed:
                if not isinstance(layer_rows, torch.Tensor) or layer_rows.ndim < 1:
                    raise TypeError("DSpark Engram wait must return layer-row tensors")
                if layer_rows.shape[0] != capacity:
                    raise ValueError(
                        f"DSpark Engram layer rows {layer_rows.shape[0]} differ from fixed target rows {capacity}")
                fixed = torch.zeros(
                    (capacity, *layer_rows.shape[1:]),
                    dtype=layer_rows.dtype,
                    device=layer_rows.device,
                )
                fixed.index_copy_(0, real_indices, layer_rows[:real_count])
                physical.append(fixed)
            return tuple(physical), (owner, ticket)
        except BaseException as error:
            if ticket is not None:
                try:
                    self._complete_engram_zero(owner, ticket, len(spans))
                except BaseException as cleanup_error:
                    if hasattr(error, "add_note"):
                        error.add_note(
                            f"DSpark Engram input rollback failed: {type(cleanup_error).__name__}: {cleanup_error}")
            raise

    @staticmethod
    def _complete_engram_zero(owner, ticket, active_count: int) -> None:
        """Retire one failed input ticket while preserving its original error."""
        pending = getattr(owner, "pending", ticket)
        if pending is not ticket:
            return
        if getattr(owner, "ready_ticket", ticket) is not ticket:
            owner.wait(ticket)
        owner.complete_batch(ticket, (0, ) * active_count)

    def finish(self, target, accepted):
        payload = getattr(target, "payload", None)
        if not isinstance(payload, dict):
            raise TypeError("DSpark target result lacks state ownership payload")
        if payload.get("finished"):
            raise RuntimeError("DSpark target transaction was already completed")
        accepted = tuple(_host_integer(value, "accepted") for value in
                         (accepted.tolist() if isinstance(accepted, torch.Tensor) else list(accepted)))
        if len(accepted) != self.batch_capacity:
            raise ValueError("DSpark target completion requires one accepted value per lane")
        batch = payload["batch"]
        lanes = payload.get("source_lanes", tuple(range(self.batch_capacity)))
        accepted = tuple(accepted[lane] if lane >= 0 else 0 for lane in lanes)
        payload["ticket"].restore(accepted)
        owner_info = payload.get("engram_owner")
        if owner_info is not None:
            owner, ticket = owner_info
            active_accepted = tuple(value for value, active in
                                    zip(accepted, batch.host_request_active, strict=True) if active)
            owner.complete_batch(ticket, active_accepted)
        payload["finished"] = True

    def abort(self, target) -> None:
        """Roll back an uncommitted warmup/failure transaction exactly once."""
        payload = getattr(target, "payload", None)
        if not isinstance(payload, dict) or payload.get("finished"):
            return
        batch = payload["batch"]
        payload["ticket"].restore((0, ) * batch.batch_capacity)
        owner_info = payload.get("engram_owner")
        if owner_info is not None:
            owner, ticket = owner_info
            active_count = sum(batch.host_request_active)
            self._complete_engram_zero(owner, ticket, active_count)
        payload["finished"] = True


@dataclass(frozen=True)
class DSparkDraftBatch:
    """Per-request greedy commit and next-round draft output."""

    accepted: Any
    output_token_ids: Any
    draft_token_ids: Any
    completed: Any = None
    completion_event: Any = None


@dataclass(frozen=True)
class _PreparedPublication:
    """Fully validated lane-major result; no request state has been mutated."""

    accepted: tuple[int, ...]
    output_rows: tuple[tuple[int, ...], ...]
    draft_rows: tuple[tuple[int, ...], ...]
    completed: tuple[bool, ...]
    active_lanes: tuple[int, ...]
    model_output: ModelRunnerOutput
    draft_output: DraftTokenIds


class DSparkBatchTrace:
    """One full main-rank record and compact consistency digests elsewhere."""

    def __init__(self, *, rank: int = 0, tp_rank: int = 0, pp_rank: int = 0):
        self.rank = int(rank)
        self.tp_rank = int(tp_rank)
        self.pp_rank = int(pp_rank)
        self.path = None
        run_dir = os.getenv("DSV41_RUN_EVIDENCE")
        if run_dir:
            directory = Path(run_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"dspark_batch_rounds.rank{self.rank}.jsonl"

    @staticmethod
    def _digest(payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def prepare(self, payload: Mapping[str, Any]) -> str:
        """Build and serialize one semantic record before request mutation."""
        req_ids = list(payload["req_ids"])
        request_slots = list(payload.get("active_lanes", payload["request_slots"]))
        capacity = int(payload.get("batch_capacity", payload.get("capacity", len(request_slots))))
        active_count = int(payload.get("active_count", len(req_ids)))
        digest_payload = {
            "schema": payload["schema"],
            "batch_round": payload["batch_round"],
            "batch_capacity": capacity,
            "active_count": active_count,
            "req_ids": req_ids,
            "active_lanes": request_slots,
            "request_slots": list(payload["request_slots"]),
            "slot_generations": payload["slot_generations"],
            "logical_target_queries": payload["logical_target_queries"],
            "draft_counts": payload["draft_counts"],
            "metadata_digest": payload["metadata_digest"],
            "committed_counts": payload["committed_counts"],
            "output_counts": payload["output_counts"],
            "output_ids_digest": payload["output_ids_digest"],
            "next_draft_digest": payload["next_draft_digest"],
            "physical_target_rows": int(payload.get("physical_target_rows",
                                                     payload["actual_packed_target_rows"])),
            "physical_draft_rows": int(payload.get("physical_draft_rows",
                                                    payload["actual_packed_draft_rows"])),
        }
        if "active_metadata" in payload:
            digest_payload["active_metadata"] = payload["active_metadata"]
        if "target_execution_slots" in payload:
            digest_payload["target_execution_slots"] = payload["target_execution_slots"]
            digest_payload["target_execution_capacity"] = payload["target_execution_capacity"]
        digest = self._digest(digest_payload)
        record = {
            **payload,
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "pp_rank": self.pp_rank,
            "consistency_digest": digest,
        }
        if self.rank != 0:
            record = {
                "schema": _TRACE_SCHEMA,
                "batch_round": payload["batch_round"],
                "batch_capacity": capacity,
                "active_count": active_count,
                "rank": self.rank,
                "tp_rank": self.tp_rank,
                "pp_rank": self.pp_rank,
                "consistency_digest": digest,
            }
        return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def emit(self, line: str) -> None:
        """Perform only the fallible evidence I/O for a prepared record."""
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        else:
            logger.info("DSV41_DSPARK_TRACE %s", line)

    def write(self, payload: Mapping[str, Any]) -> None:
        self.emit(self.prepare(payload))


class DSparkBatchRunner:
    """Invoke one fixed-capacity packed target and draft round."""

    def __init__(
        self,
        target_stage: Any,
        draft_stage: Any,
        *,
        trace: DSparkBatchTrace | None = None,
        slots: Any = None,
        batch_capacity: int | None = None,
        use_active_metadata: bool = True,
    ):
        if type(use_active_metadata) is not bool:
            raise TypeError("use_active_metadata must be a boolean")
        self.use_active_metadata = use_active_metadata
        if target_stage is None or draft_stage is None:
            raise RuntimeError("DSpark requires actual packed target and draft stages")
        self.target_stage = target_stage
        self.draft_stage = draft_stage
        self.trace = trace or DSparkBatchTrace()
        trace_owner = self.trace
        self.slots = slots
        self.batch_capacity = _resolve_capacity(batch_capacity, slots, target_stage, draft_stage)
        self.round_generation = 0
        self.pending = False
        self.single_draft_owner = None
        self.draft_bank_generations = {}
        self.trace_failures = 0
        self.trace_available = True
        # BUG-004 diagnostics are constructed only for an enabled round.
        self._bug004_profile = None
        self._bug004_rank = getattr(trace_owner, "rank", 0)
        self._bug004_tp_rank = getattr(trace_owner, "tp_rank", 0)
        self._bug004_pp_rank = getattr(trace_owner, "pp_rank", 0)

    def build_plan(
        self,
        scheduled: Any,
        requests: Mapping[str, Any],
        slot_owners: Mapping[str, Any],
        *,
        max_model_len: int,
    ) -> DSparkBatchPlan | None:
        if self.pending:
            raise RuntimeError("Previous DSpark transaction is still in flight")
        if not scheduled.num_scheduled_tokens:
            # Idle is not a device round: no generation, graph, ticket or slot
            # ownership may advance.
            return None
        round_generation = self.round_generation + 1
        batch_capacity = getattr(self, "batch_capacity", None)
        plan = DSparkBatchPlan.from_scheduler(
            scheduled,
            requests,
            slot_owners,
            max_model_len=max_model_len,
            round_generation=round_generation,
            batch_capacity=batch_capacity,
            use_active_metadata=getattr(self, "use_active_metadata", True),
        )
        self.round_generation = round_generation
        return plan

    def warmup(self, *, pool_blocks: int, max_model_len: int, request_factory: Any) -> None:
        """Exercise bounded lane patterns without enumerating request orders."""
        state = getattr(self.target_stage, "batch_state", None)
        draft_state = self._draft_state()
        engram = getattr(self.target_stage, "engram_host", None)
        if state is None or self.slots is None or getattr(state, "slots", None) is not self.slots:
            raise RuntimeError("DSpark warmup requires the runner-owned request slot bank")
        if (self.pending or self.slots.owners or getattr(state, "pending", None) is not None
                or getattr(state, "single_owner", None) is not None
                or getattr(state, "prefill_owner", None) is not None or self.single_draft_owner is not None):
            raise RuntimeError("DSpark warmup requires idle target and draft bindings")
        if pool_blocks <= self.batch_capacity:
            raise ValueError(f"DSpark warmup requires more than {self.batch_capacity} non-null scheduler pages")
        if draft_state is not None and _component_capacity(draft_state) not in (None, self.batch_capacity):
            raise ValueError("DSpark warmup draft state capacity differs from the runner")

        started_round = self.round_generation
        observed: list[dict[str, Any]] = []
        try:
            buckets = getattr(self.target_stage, "execution_buckets", (self.batch_capacity,))
            scenarios = warmup_scenarios(self.batch_capacity)
            if len(buckets) > 1:
                scenarios = scenarios[:3] + tuple(
                    (f"target_bucket_{bucket}_mixed", {lane: (6 if lane % 2 == 0 else 1)
                                                       for lane in range(bucket)})
                    for bucket in buckets if 1 < bucket < self.batch_capacity
                )
            for round_index, (name, lengths_by_lane) in enumerate(scenarios):
                observed.append(
                    self._warmup_round(
                        round_index=round_index,
                        name=name,
                        lengths_by_lane=lengths_by_lane,
                        pool_blocks=pool_blocks,
                        max_model_len=max_model_len,
                        request_factory=request_factory,
                        state=state,
                        draft_state=draft_state,
                        engram=engram,
                    )
                )

            stage_key = tuple(range(self.batch_capacity))
            stages = getattr(self.target_stage, "stages", None)
            fixed_stage = getattr(self.target_stage, "stage", None)
            if fixed_stage is None and isinstance(stages, Mapping):
                fixed_stage = stages.get(stage_key)
            if getattr(self.target_stage, "stage_key", stage_key) != stage_key:
                raise RuntimeError("DSpark warmup observed a capacity-dependent target key")
            expected_keys = {tuple(range(bucket)) for bucket in buckets}
            if isinstance(stages, Mapping) and set(stages) != expected_keys:
                raise RuntimeError("DSpark warmup target keys differ from its bounded execution buckets")
            transaction = getattr(self.draft_stage, "transaction", None)
            verify_ring = getattr(self.draft_stage, "verify_ring", None)
            if getattr(fixed_stage, "compiled", None) is None:
                raise RuntimeError("DSpark warmup did not build the fixed target graph")
            if transaction is None or getattr(transaction, "compiled", None) is None:
                raise RuntimeError("DSpark warmup did not build the packed draft transaction")
            if verify_ring is None or self.round_generation != verify_ring.generation:
                raise RuntimeError("DSpark warmup round and verify generations diverged")
            if self.round_generation != started_round + len(scenarios):
                raise RuntimeError("DSpark warmup round count is not bounded by its lane scenarios")
            if observed[0]["active_count"] != self.batch_capacity:
                raise RuntimeError("DSpark warmup did not exercise a full fixed-capacity lane set")
            if self.batch_capacity > 1 and observed[1]["active_lanes"] == stage_key:
                raise RuntimeError("DSpark warmup did not exercise a partial physical lane gap")
            # The first three scenarios deliberately reuse the same tail lane.
            # New N32 tile scenarios activate different lanes, so comparing
            # their final entries would compare unrelated slot histories.
            reuse_lane = self.batch_capacity - 1
            reuse_observations = observed[:3]
            if any(item["active_lanes"][-1] != reuse_lane for item in reuse_observations):
                raise RuntimeError("DSpark warmup reuse scenarios do not share the canonical tail lane")
            generations = [item["slot_generations"][-1] for item in reuse_observations]
            if not all(second > first for first, second in zip(generations[:-1], generations[1:], strict=True)):
                raise RuntimeError("DSpark warmup did not advance a reused physical slot generation")
            if self.pending or self.slots.owners or getattr(state, "pending", None) is not None:
                raise RuntimeError("DSpark warmup left a fake request or driver transaction pending")
        finally:
            self.single_draft_owner = None
            self.draft_bank_generations.clear()
            state.restore_single_bindings()

    def _warmup_round(self, *, round_index: int, name: str, lengths_by_lane: Mapping[int, int], pool_blocks: int,
                      max_model_len: int, request_factory: Any, state, draft_state, engram) -> dict[str, Any]:
        from types import SimpleNamespace

        entries = []
        try:
            for lane in range(self.batch_capacity):
                req_id = f"__v41_dspark_{name}_{round_index}_{lane}__"
                request = request_factory(req_id, lane + 1, 4096 + round_index * 16 + lane)
                slot = self.slots.acquire(req_id)
                if slot.index != lane:
                    raise RuntimeError("DSpark warmup could not acquire its canonical physical lane")
                entries.append((request, slot))
                state.publish_pages(slot, request.block_ids[0], pool_blocks)
                if engram is not None:
                    engram.activate(req_id, reset=True)

            by_lane = {slot.index: request for request, slot in entries}
            requests_by_id = {request.req_id: request for request, _slot in entries}
            active_lanes = tuple(sorted(lengths_by_lane))
            if not active_lanes:
                raise ValueError("DSpark warmup scenario needs at least one active lane")
            counts = {}
            proposals = {}
            for lane in reversed(active_lanes):
                request = by_lane[lane]
                count = lengths_by_lane[lane]
                if count not in (1, _TARGET_QUERY_BUCKET):
                    raise ValueError("DSpark warmup target lengths must be C1 or C6")
                counts[request.req_id] = count
                if count == _TARGET_QUERY_BUCKET:
                    first = 1000 + round_index * 128 + lane * 8
                    proposals[request.req_id] = tuple(range(first, first + _MAX_DRAFT_TOKENS))
            scheduled = SimpleNamespace(
                num_scheduled_tokens=counts,
                scheduled_spec_decode_tokens=proposals,
            )
            plan = self.build_plan(scheduled, requests_by_id, self.slots.owners, max_model_len=max_model_len)
            if plan is None or plan.capacity != self.batch_capacity:
                raise RuntimeError("DSpark warmup did not build one fixed-capacity plan")
            if plan.active_lanes != active_lanes:
                raise RuntimeError("DSpark warmup scheduler order changed the physical lane mapping")
            active_mask = [lane in lengths_by_lane for lane in range(self.batch_capacity)]
            if plan.active_mask != tuple(active_mask):
                raise RuntimeError("DSpark warmup activity mask disagrees with its physical lanes")
            if not self.slots.validate_batch_identities(
                    tuple(entry.request_slot for entry in plan.requests),
                    tuple(entry.slot_generation for entry in plan.requests),
                    (True, ) * len(plan.requests)):
                raise RuntimeError("DSpark warmup plan does not own its request slots")
            self.execute(plan)
            return {
                "name": name,
                "active_count": len(active_lanes),
                "active_lanes": active_lanes,
                "target_lengths": tuple(plan.target_lengths),
                "slot_generations": tuple(entry.slot_generation for entry in plan.requests),
            }
        finally:
            self._clear_warmup_entries(entries, state, draft_state, engram)

    def _clear_warmup_entries(self, entries, state, draft_state, engram) -> None:
        if not entries:
            return
        if state.pages.device.type == "hpu":
            torch.hpu.synchronize()
        for request, slot in reversed(entries):
            if draft_state is not None:
                prepared_generation = self.draft_bank_generations.pop(slot.index, None)
                if prepared_generation != slot.generation:
                    draft_state.clear_slot(slot.index)
            for layer_state in getattr(state, "layers", {}).values():
                layer_state.clear_slot(slot.index)
            state.pages[slot.index].zero_()
            state.page_host[slot.index].zero_()
            state.page_versions.pop(slot.index, None)
            if self.slots.owners.get(request.req_id) == slot:
                self.slots.release(slot)
            if engram is not None:
                engram.release_request(request.req_id)
        self.single_draft_owner = None
        self.draft_bank_generations.clear()

    def _draft_state(self):
        return getattr(self.draft_stage, "batch_state", None)

    def _ensure_draft_generation(self, slot) -> None:
        state = self._draft_state()
        if state is None:
            return
        if self.draft_bank_generations.get(slot.index) != slot.generation:
            state.clear_slot(slot.index)
            self.draft_bank_generations[slot.index] = slot.generation

    def _copy_draft_ring(self, slot, *, to_bank: bool) -> None:
        state = self._draft_state()
        draft = getattr(self.draft_stage, "draft", None)
        if state is None or draft is None:
            return
        self._ensure_draft_generation(slot)
        for index, layer in enumerate(draft.layers):
            working = layer.attention.swa
            stored = state.rings[index][slot.index * 256:(slot.index + 1) * 256]
            if working.shape != stored.shape:
                raise RuntimeError("DSpark draft single/batched state geometries differ")
            if to_bank:
                stored.copy_(working)
            else:
                working.copy_(stored)

    def bind_single(self, slot) -> None:
        """Load the request's packed draft ring before a native B1 round."""
        if self.single_draft_owner == slot:
            return
        previous = self.single_draft_owner
        if previous is not None and self._slot_is_live(previous):
            self._copy_draft_ring(previous, to_bank=True)
        self._copy_draft_ring(slot, to_bank=False)
        self.single_draft_owner = slot

    def _slot_is_live(self, slot) -> bool:
        return self.slots is not None and self.slots.owners.get(slot.request_id) == slot

    def release(self, request_id: str) -> None:
        """Forget a completed/cancelled owner without flushing it to a reused slot."""
        if self.single_draft_owner is not None and self.single_draft_owner.request_id == request_id:
            self.single_draft_owner = None

    def _leave_single(self) -> None:
        owner = self.single_draft_owner
        if owner is not None:
            if self._slot_is_live(owner):
                self._copy_draft_ring(owner, to_bank=True)
            self.single_draft_owner = None
        enter = getattr(self.target_stage, "enter_batched", None)
        if callable(enter):
            enter()

    def _prepare_batched_generations(self, plan: DSparkBatchPlan) -> None:
        if self.slots is None:
            return
        for entry in plan.requests:
            slot = self.slots.owners.get(entry.req_id)
            if slot is None or slot.index != entry.request_slot or slot.generation != entry.slot_generation:
                raise RuntimeError("DSpark request slot changed before pack preparation")
            self._ensure_draft_generation(slot)

    def execute(self, plan: DSparkBatchPlan) -> tuple[ModelRunnerOutput, DraftTokenIds]:
        if plan.round_generation != self.round_generation:
            raise RuntimeError("DSpark plan generation is stale")
        if self.pending:
            raise RuntimeError("Previous DSpark transaction is still in flight")
        if plan.active_metadata is not None and self.slots is not None:
            plan.active_metadata.validate_owners(
                self.slots.owners, expected_round_generation=self.round_generation)
        target = None
        finished = False
        self.pending = True
        profile = None
        profile_closed = False
        try:
            profile = self.begin_bug004_profile()
            if profile is not None:
                profile.mark("target.input_prepare")
            self._leave_single()
            self._prepare_batched_generations(plan)
            target = self.target_stage.forward_batch(
                plan.metadata,
                plan.packed_input_ids,
                plan.packed_positions,
                plan.packed_padding_mask,
                plan.pages,
            )
            if not isinstance(target, DSparkTargetBatch):
                raise TypeError("DSpark target stage must return DSparkTargetBatch")
            result = self.draft_stage.verify_and_propose(plan.metadata, target)
            if not isinstance(result, DSparkDraftBatch):
                raise TypeError("DSpark draft stage must return DSparkDraftBatch")
            if profile is not None:
                profile.mark("verify.commit")
            prepared = self._prepare_publication(plan, result)
            readback_observed_ns = self._observe_completion(result.completion_event)
            self._validate_final_owners(plan)
            trace_payload = self._prepare_trace_payload(
                plan,
                prepared,
                readback_observed_ns=readback_observed_ns,
                final_observed_ns=0,
                target=target,
            )
            if profile is not None:
                profile.mark("round.finish")
            finish = getattr(self.target_stage, "finish", None)
            if callable(finish):
                finish(target, prepared.accepted)
            finished = True
            if profile is not None:
                profile.mark("round.final_event")
            _final_event, final_observed_ns = self._record_final_event(plan)
            trace_payload["final_event_observed_ns"] = final_observed_ns
            if profile is not None:
                profile.mark("round.commit")
            self._emit_trace(trace_payload)
            output, drafts = self._commit_publication(plan, prepared)
            self.close_bug004_profile(profile)
            profile_closed = True
            return output, drafts
        except BaseException as error:
            if profile is not None and not profile_closed:
                try:
                    self.close_bug004_profile(
                        profile, status="failed", error=f"{type(error).__name__}: {error}")
                    profile_closed = True
                except BaseException as profile_error:
                    if hasattr(error, "add_note"):
                        error.add_note(
                            f"BUG004 profile finalization failed: {type(profile_error).__name__}: {profile_error}")
            abort = getattr(self.target_stage, "abort", None)
            if target is not None and not finished and callable(abort):
                try:
                    abort(target)
                except BaseException as abort_error:
                    if hasattr(error, "add_note"):
                        error.add_note(
                            f"DSpark target abort failed: {type(abort_error).__name__}: {abort_error}")
            raise
        finally:
            self.pending = False

    def begin_bug004_profile(self, *, rank: int | None = None, tp_rank: int | None = None,
                             pp_rank: int | None = None):
        """Bind one diagnostic observer to a round without importing it by default."""
        from vllm_gaudi.ops.deepseek_v41_bug004_profile import begin_round

        device = getattr(self.target_stage, "device", None)
        profile = begin_round(
            rank=self._bug004_rank if rank is None else rank,
            tp_rank=self._bug004_tp_rank if tp_rank is None else tp_rank,
            pp_rank=self._bug004_pp_rank if pp_rank is None else pp_rank,
            device=device,
        )
        self._bug004_profile = profile
        if profile is not None:
            self.target_stage._bug004_profile = profile
            self.draft_stage._bug004_profile = profile
        return profile

    def close_bug004_profile(self, profile=None, *, status: str = "complete",
                             error: str | None = None):
        """Finalize the round observer and clear all temporary bindings."""
        profile = getattr(self, "_bug004_profile", None) if profile is None else profile
        self.target_stage._bug004_profile = None
        if hasattr(self.draft_stage, "_bug004_profile"):
            self.draft_stage._bug004_profile = None
        self._bug004_profile = None
        return None if profile is None else profile.close(status=status, error=error)

    @staticmethod
    def _observe_completion(event: Any) -> int:
        if event is None or isinstance(event, bool):
            raise RuntimeError("DSpark verify must expose its actual completion event")
        query = getattr(event, "query", None)
        if callable(query):
            if not query():
                synchronize = getattr(event, "synchronize", None)
                if not callable(synchronize):
                    raise RuntimeError("DSpark completion event has not completed")
                synchronize()
                if not query():
                    raise RuntimeError("DSpark completion event remained incomplete after wait")
        else:
            synchronize = getattr(event, "synchronize", None)
            if not callable(synchronize):
                raise TypeError("DSpark completion evidence must expose query() or synchronize()")
            synchronize()
        return time.monotonic_ns()

    def _record_final_event(self, plan: DSparkBatchPlan):
        self._validate_final_owners(plan)
        device = getattr(self.target_stage, "device", None)
        event = None
        if device is not None and torch.device(device).type == "hpu":
            event = torch.hpu.Event()
            event.record(torch.hpu.current_stream())
            event.synchronize()
            if not event.query():
                raise RuntimeError("DSpark final state event remained incomplete")
        if self.slots is not None:
            completion = event if event is not None else _CompletedEvent()
            for entry in plan.requests:
                slot = self.slots.owners[entry.req_id]
                self.slots.submitted(slot, completion)
        return event, time.monotonic_ns()

    def _validate_final_owners(self, plan: DSparkBatchPlan) -> None:
        if self.slots is None:
            return
        if plan.active_metadata is not None:
            plan.active_metadata.validate_owners(
                self.slots.owners, expected_round_generation=self.round_generation)
        for entry in plan.requests:
            slot = self.slots.owners.get(entry.req_id)
            if slot is None or slot.index != entry.request_slot or slot.generation != entry.slot_generation:
                raise RuntimeError("DSpark final state owner changed before publication")

    def _prepare_trace_payload(
        self,
        plan: DSparkBatchPlan,
        prepared: _PreparedPublication,
        *,
        readback_observed_ns: int,
        final_observed_ns: int,
        target: DSparkTargetBatch | None = None,
    ) -> dict[str, Any]:
        output = prepared.model_output
        drafts = prepared.draft_output
        active_lanes = list(prepared.active_lanes)
        active_accepted = [prepared.accepted[lane] for lane in active_lanes]
        active_output_counts = [len(prepared.output_rows[lane]) for lane in active_lanes]
        active_draft_counts = [len(prepared.draft_rows[lane]) for lane in active_lanes]
        metadata_digest = self._metadata_digest(plan)
        output_ids_digest = self._values_digest(output.sampled_token_ids)
        next_draft_digest = self._values_digest(drafts.draft_token_ids)
        target_payload = getattr(target, "payload", None)
        target_payload = target_payload if isinstance(target_payload, dict) else {}
        physical_target_rows = target_payload.get("target_execution_rows", plan.capacity * _TARGET_QUERY_BUCKET)
        physical_draft_rows = plan.capacity * _MAX_DRAFT_TOKENS
        payload = {
            "schema": _TRACE_SCHEMA,
            "batch_round": plan.round_generation,
            "batch_capacity": plan.capacity,
            "capacity": plan.capacity,
            "active_count": len(active_lanes),
            "active_lanes": active_lanes,
            "lanes": active_lanes,
            "active_mask": list(plan.active_mask),
            "lane_request_ids": list(plan.lane_request_ids),
            "req_ids": list(plan.request_ids),
            "reqids": list(plan.request_ids),
            "request_slots": [entry.request_slot for entry in plan.requests],
            "slot_generations": [entry.slot_generation for entry in plan.requests],
            "lane_slot_generations": list(plan.metadata.slot_generations.tolist()),
            "logical_target_queries": list(plan.target_lengths),
            "draft_counts": [len(entry.draft_token_ids) for entry in plan.requests],
            "scheduler_draft_counts": [entry.scheduler_draft_count for entry in plan.requests],
            "physical_target_rows": physical_target_rows,
            "target_execution_capacity": target_payload.get("target_execution_capacity", plan.capacity),
            "target_execution_slots": list(target_payload.get("target_execution_slots", ())),
            "physical_draft_rows": physical_draft_rows,
            "actual_packed_target_rows": physical_target_rows,
            "actual_packed_draft_rows": physical_draft_rows,
            "targetrows": physical_target_rows,
            "draftrows": physical_draft_rows,
            "batch_width": len(active_lanes),
            "committed_counts": active_accepted,
            "lane_committed_counts": list(prepared.accepted),
            "output_counts": active_output_counts,
            "lane_output_counts": [len(row) for row in prepared.output_rows],
            "next_draft_counts": active_draft_counts,
            "lane_next_draft_counts": [len(row) for row in prepared.draft_rows],
            "metadata_digest": metadata_digest,
            "output_ids_digest": output_ids_digest,
            "next_draft_digest": next_draft_digest,
            "completion_event_observed": True,
            "completion_observed_ns": readback_observed_ns,
            "final_event_observed": True,
            "final_event_observed_ns": final_observed_ns,
            "target_stage": f"{type(self.target_stage).__module__}.{type(self.target_stage).__qualname__}",
            "draft_stage": f"{type(self.draft_stage).__module__}.{type(self.draft_stage).__qualname__}",
            "target_stage_key": list(getattr(self.target_stage, "stage_key", ())),
        }
        if plan.active_metadata is not None:
            active = plan.active_metadata
            payload["active_metadata"] = {
                "request_ids": list(active.request_ids),
                "compact_to_logical_slots": list(active.compact_to_logical_slots),
                "compact_to_fixed_lanes": list(active.compact_to_fixed_lanes),
                "canonical_compact_rows": list(active.canonical_compact_rows),
                "identities": [list(identity.as_tuple()) for identity in active.identities],
                "logical_target_rows": int(active.target_query_start_loc[-1]),
                "logical_draft_rows": int(active.draft_query_start_loc[-1]),
            }
        scoped_status = getattr(self.target_stage, "scoped_recipe_status", None)
        if callable(scoped_status):
            payload["c6_scoped_recipe"] = scoped_status()
        return payload

    def _emit_trace(self, payload: Mapping[str, Any]) -> None:
        prepare = getattr(self.trace, "prepare", None)
        emit = getattr(self.trace, "emit", None)
        try:
            if callable(prepare) and callable(emit):
                emit(prepare(payload))
            else:
                write = getattr(self.trace, "write", None)
                if not callable(write):
                    raise TypeError("DSpark trace must expose write() or prepare()/emit()")
                write(payload)
        except OSError as error:
            self.trace_failures += 1
            self.trace_available = False
            logger.warning("DSV41_DSPARK trace evidence unavailable: %s", error)

    @staticmethod
    def _values_digest(values) -> str:
        encoded = json.dumps([[int(value) for value in row] for row in values],
                             sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _metadata_digest(plan: DSparkBatchPlan) -> str:
        metadata = plan.metadata
        values = {
            "batch_capacity": metadata.capacity,
            "request_ids": list(metadata.request_ids),
            "request_active": metadata.request_active.tolist(),
            "target_start_positions": metadata.target_start_positions.tolist(),
            "target_query_start_loc": metadata.target_query_start_loc.tolist(),
            "num_draft_tokens": metadata.num_draft_tokens.tolist(),
            "draft_token_ids": metadata.draft_token_ids.tolist(),
            "request_slots": metadata.request_slots.tolist(),
            "slot_generations": metadata.slot_generations.tolist(),
            "round_generation": int(metadata.round_generation.item()),
        }
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _prepare_publication(plan: DSparkBatchPlan, result: DSparkDraftBatch) -> _PreparedPublication:
        accepted = tuple(
            _host_integer(value, "accepted")
            for value in (result.accepted.tolist() if isinstance(result.accepted, torch.Tensor)
                          else list(result.accepted))
        )
        if len(accepted) != plan.capacity:
            raise ValueError("DSpark accepted lengths must contain one value per fixed lane")

        output_rows = tuple(_ragged_rows(result.output_token_ids, accepted, "output_token_ids"))
        draft_rows = tuple(_next_draft_rows(result.draft_token_ids, plan.capacity, "draft_token_ids"))
        if result.completed is None:
            completed = (False, ) * plan.capacity
        else:
            completed_values = (result.completed.tolist()
                                if isinstance(result.completed, torch.Tensor) else list(result.completed))
            completed = tuple(bool(value) for value in completed_values)
            if len(completed) != plan.capacity:
                raise ValueError("DSpark completed flags must contain one value per fixed lane")

        active_mask = plan.active_mask
        active_lanes = tuple(lane for lane, active in enumerate(active_mask) if active)
        entries = plan.entry_by_lane
        for lane, (active, accepted_count, output, draft, was_completed) in enumerate(
                zip(active_mask, accepted, output_rows, draft_rows, completed, strict=True)):
            if not active:
                if accepted_count != 0 or output or draft or was_completed:
                    raise ValueError(f"Inactive DSpark lane {lane} published an output or completion record")
                continue
            entry = entries.get(lane)
            if entry is None:
                raise RuntimeError(f"Active DSpark lane {lane} has no physical request entry")
            if accepted_count != len(output):
                raise ValueError(f"DSpark accepted prefix for {entry.req_id!r} differs from output length")
            if not 1 <= accepted_count <= entry.target_query_length:
                raise ValueError(f"DSpark accepted prefix for {entry.req_id!r} is outside its target range")
            if len(draft) > _MAX_DRAFT_TOKENS:
                raise ValueError(f"DSpark next-round draft count for {entry.req_id!r} exceeds five")
            if entry.scheduler_num_rejected < 0:
                raise ValueError(f"DSpark output for {entry.req_id!r} exceeds scheduled speculation")
            if len(entry.request.output) + len(output) > entry.request.sampling_params.max_tokens:
                raise ValueError(f"DSpark output for {entry.req_id!r} exceeds its output budget")
            if was_completed and draft:
                raise ValueError(f"Completed DSpark request {entry.req_id!r} cannot publish next-round drafts")

        req_ids = [plan.lane_request_ids[lane] for lane in active_lanes]
        if any(req_id is None for req_id in req_ids):
            raise RuntimeError("DSpark publication would leak an inactive request ID")
        active_outputs = [output_rows[lane] for lane in active_lanes]
        active_drafts = [draft_rows[lane] for lane in active_lanes]
        model_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=active_outputs,
            execution_rounds=[
                (entries[lane].req_id, plan.round_generation, entries[lane].target_query_length)
                for lane in active_lanes
            ],
        )
        draft_output = DraftTokenIds(req_ids, active_drafts)
        return _PreparedPublication(
            accepted,
            output_rows,
            draft_rows,
            completed,
            active_lanes,
            model_output,
            draft_output,
        )

    @staticmethod
    def _commit_publication(
        plan: DSparkBatchPlan,
        prepared: _PreparedPublication,
    ) -> tuple[ModelRunnerOutput, DraftTokenIds]:
        entries = plan.entry_by_lane
        for lane in prepared.active_lanes:
            entries[lane].request.output.extend(prepared.output_rows[lane])
        return prepared.model_output, prepared.draft_output

    @staticmethod
    def publish(
        plan: DSparkBatchPlan, result: DSparkDraftBatch
    ) -> tuple[ModelRunnerOutput, DraftTokenIds]:
        prepared = DSparkBatchRunner._prepare_publication(plan, result)
        return DSparkBatchRunner._commit_publication(plan, prepared)
