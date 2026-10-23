# SPDX-License-Identifier: Apache-2.0
"""Capacity-bounded packed DSpark greedy verification, records and readback.

This module is intentionally independent of target replay. A target producer
supplies fixed ``[B, 6]`` token ids and fixed ``[B, 5]`` proposals; this owner
computes one independent commit per request, packs one fixed-width record per
request, and exposes one batch readback ticket. The ring accepts an injected
authoritative slot-owner validator without owning the request-state bank.
PP1 only: there is no per-request graph and no per-request device readback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

DEFAULT_BATCH_CAPACITY = 2
MAX_BATCH_REQUESTS = 8
MAX_TARGET_QUERIES = 6
MAX_DRAFT_TOKENS = 5
BATCH_RECORD_SIZE = 16
BATCH_OUTPUT_START = 4
BATCH_DRAFT_START = 10
BATCH_STATUS = 15
BATCH_INACTIVE = 3
BINDING_SIDECAR_SIZE = 2
BINDING_SIDECAR_LOW = 0
BINDING_SIDECAR_HIGH = 1
_MAX_ROUND_GENERATION = (1 << 31) - 1
_MAX_SLOT_GENERATION = (1 << 24) - 1
_MAX_SLOT_INDEX = (1 << 8) - 1
_SLOT_INDEX_SHIFT = 24
_SLOT_GENERATION_MASK = (1 << 24) - 1
_U32_MASK = (1 << 32) - 1


def _host_ints(value: Sequence[int] | torch.Tensor, name: str, length: int) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if (value.ndim != 1 or value.device.type != "cpu" or value.numel() != length
                or value.dtype not in (torch.int32, torch.int64)):
            raise ValueError(f"{name} must be a CPU integer [{length}] tensor")
        values = tuple(value.tolist())
    else:
        values = tuple(value)
    if len(values) != length or any(type(value) is not int for value in values):
        raise ValueError(f"{name} must contain exactly {length} integers")
    return values


def _host_bools(value: Sequence[bool] | torch.Tensor, name: str, length: int) -> tuple[bool, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.device.type != "cpu" or value.dtype != torch.bool or value.numel() != length:
            raise ValueError(f"{name} must be a CPU bool [ {length} ] tensor")
        values = tuple(value.tolist())
    else:
        values = tuple(value)
    if len(values) != length or any(type(value) is not bool for value in values):
        raise ValueError(f"{name} must contain exactly {length} bools")
    return values


def _scalar_generation(
    value: int | torch.Tensor,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_ROUND_GENERATION,
) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.device.type != "cpu" or value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be a CPU integer scalar")
        value = value.item()
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _slot_identity(
    request_slots: Sequence[int] | torch.Tensor,
    slot_generations: Sequence[int] | torch.Tensor,
    active: Sequence[bool] | torch.Tensor,
    batch_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[bool, ...]]:
    slots = _host_ints(request_slots, "request_slots", batch_size)
    generations = _host_ints(slot_generations, "slot_generations", batch_size)
    active_values = _host_bools(active, "active", batch_size)
    active_slots = set()
    for index, enabled in enumerate(active_values):
        if not enabled:
            if slots[index] != -1 or generations[index] != 0:
                raise ValueError("inactive request slots must use the (-1, 0) sentinel")
            continue
        if not 0 <= slots[index] <= _MAX_SLOT_INDEX:
            raise ValueError(f"active request slot must be in 0..{_MAX_SLOT_INDEX}")
        _scalar_generation(generations[index], "slot_generation", minimum=1, maximum=_MAX_SLOT_GENERATION)
        if slots[index] in active_slots:
            raise ValueError("active request slots must be unique")
        active_slots.add(slots[index])
    return slots, generations, active_values


def pack_binding_word(round_generation: int, slot_generation: int, slot_index: int) -> int:
    """Bind a record to round, request slot index and slot generation."""
    round_generation = _scalar_generation(round_generation, "round_generation", minimum=1)
    slot_generation = _scalar_generation(
        slot_generation,
        "slot_generation",
        minimum=1,
        maximum=_MAX_SLOT_GENERATION,
    )
    slot_index = _scalar_generation(slot_index, "slot_index", minimum=0, maximum=_MAX_SLOT_INDEX)
    return (round_generation << 32) | (slot_index << _SLOT_INDEX_SHIFT) | slot_generation


def unpack_binding_word(word: int) -> tuple[int, int, int]:
    if type(word) is not int or word < 0:
        raise ValueError("binding word must be a nonnegative integer")
    round_generation = word >> 32
    slot_index = (word >> _SLOT_INDEX_SHIFT) & _MAX_SLOT_INDEX
    slot_generation = word & _SLOT_GENERATION_MASK
    if not (1 <= round_generation <= _MAX_ROUND_GENERATION
            and 0 <= slot_index <= _MAX_SLOT_INDEX
            and 1 <= slot_generation <= _MAX_SLOT_GENERATION):
        raise ValueError("binding word has an invalid round, slot or generation")
    return round_generation, slot_generation, slot_index


def batch_binding_words(
    round_generation: int,
    request_slots: Sequence[int] | torch.Tensor,
    slot_generations: Sequence[int] | torch.Tensor,
    active: Sequence[bool] | torch.Tensor,
) -> torch.Tensor:
    """Return CPU I64 binding words; inactive lanes use word zero."""
    round_generation = _scalar_generation(round_generation, "round_generation", minimum=1)
    if isinstance(request_slots, (torch.Tensor, list, tuple)):
        batch_size = request_slots.numel() if isinstance(request_slots, torch.Tensor) else len(request_slots)
    else:
        try:
            batch_size = len(request_slots)
        except TypeError as error:
            raise ValueError("request_slots must be a CPU vector") from error
    if not 1 <= batch_size <= MAX_BATCH_REQUESTS:
        raise ValueError(f"slot_generations must contain 1..{MAX_BATCH_REQUESTS} values")
    slots, generations, active_values = _slot_identity(request_slots, slot_generations, active, batch_size)
    return torch.tensor([
        pack_binding_word(round_generation, generations[index], slots[index]) if active_values[index] else 0
        for index in range(batch_size)
    ], dtype=torch.int64)


def _signed_i32(value: int) -> int:
    value &= _U32_MASK
    return value - (1 << 32) if value & (1 << 31) else value


def batch_binding_sidecar(control: torch.Tensor) -> torch.Tensor:
    """Split authoritative CPU I64 control bindings into I32 ``[low, high]`` rows.

    The device consumes this sidecar for binding validation and reconstructs the
    I64 record binding from the masked I32 words. The original I64 binding in
    ``control`` remains record/provenance data and is never inspected on HPU.
    """
    if (not isinstance(control, torch.Tensor) or control.device.type != "cpu" or control.dtype != torch.int64
            or control.ndim != 2 or control.shape[1] != 7 or not 1 <= control.shape[0] <= MAX_BATCH_REQUESTS):
        raise ValueError("binding sidecar requires authoritative CPU I64 control [B, 7]")
    rows = []
    for binding in control[:, 0].tolist():
        bits = int(binding) & ((1 << 64) - 1)
        rows.append((_signed_i32(bits), _signed_i32(bits >> 32)))
    return torch.tensor(rows, dtype=torch.int32)


def _validate_binding_sidecar(
    sidecar: torch.Tensor | None,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor | None:
    if sidecar is None:
        if device.type == "hpu":
            raise RuntimeError("compiled HPU batch verification requires an explicit I32 binding sidecar")
        return None
    if (not isinstance(sidecar, torch.Tensor) or sidecar.device != device or sidecar.dtype != torch.int32
            or sidecar.shape != (batch_size, BINDING_SIDECAR_SIZE)):
        raise ValueError("binding sidecar must be int32 [B, 2] [low, high] on the verification device")
    return sidecar


def _validate_max_committed(limits: tuple[int, ...], active: tuple[bool, ...]) -> None:
    for limit, enabled in zip(limits, active, strict=True):
        if enabled:
            if not 1 <= limit <= MAX_TARGET_QUERIES:
                raise ValueError("max_committed must be in 1..6 for active lanes")
        elif not 0 <= limit <= MAX_TARGET_QUERIES:
            raise ValueError("max_committed must be in 0..6 for inactive lanes")


def batch_control_metadata(
    *,
    round_generation: int,
    request_slots: Sequence[int] | torch.Tensor,
    slot_generations: Sequence[int] | torch.Tensor,
    target_counts: Sequence[int] | torch.Tensor,
    proposal_counts: Sequence[int] | torch.Tensor,
    remaining_output_budget: Sequence[int] | torch.Tensor,
    target_start_positions: Sequence[int] | torch.Tensor,
    max_model_len: int | Sequence[int] | torch.Tensor,
    need_sample: Sequence[int] | torch.Tensor,
    active: Sequence[bool] | torch.Tensor,
) -> torch.Tensor:
    """Build CPU I64 ``[B, 7]`` control rows consumed by the fixed batch graph.

    Columns are ``[binding, target_count, proposal_count, remaining,
    target_start, context_limit, need_sample]``.  Inactive lanes are zeroed;
    they must use the ``(-1, 0)`` slot sentinel and their packed record carries
    ``BATCH_INACTIVE``.  This entry is for steady decode, so an active lane
    always samples and a no-proposal request is the C1 ``target_count=1`` case.
    """
    bindings = batch_binding_words(round_generation, request_slots, slot_generations, active)
    if not isinstance(bindings, torch.Tensor):
        raise TypeError("internal binding construction changed type")
    batch_size = bindings.numel()
    target_values = _host_ints(target_counts, "target_counts", batch_size)
    proposal_values = _host_ints(proposal_counts, "proposal_counts", batch_size)
    budget_values = _host_ints(remaining_output_budget, "remaining_output_budget", batch_size)
    start_values = _host_ints(target_start_positions, "target_start_positions", batch_size)
    sample_values = _host_ints(need_sample, "need_sample", batch_size)
    active_values = _host_bools(active, "active", batch_size)

    if isinstance(max_model_len, torch.Tensor):
        if (max_model_len.ndim != 1 or max_model_len.device.type != "cpu"
                or max_model_len.dtype not in (torch.int32, torch.int64)):
            raise ValueError("max_model_len must be a CPU integer vector")
        limits = tuple(max_model_len.tolist())
    elif isinstance(max_model_len, int) and not isinstance(max_model_len, bool):
        limits = (max_model_len,) * batch_size
    else:
        try:
            limits = tuple(max_model_len)
        except TypeError as error:
            raise ValueError("max_model_len must be scalar or match batch size") from error
    if len(limits) != batch_size or any(type(value) is not int for value in limits):
        raise ValueError("max_model_len must contain integer values matching batch size")

    result = torch.zeros((batch_size, 7), dtype=torch.int64)
    for index in range(batch_size):
        if not active_values[index]:
            continue
        if not 1 <= target_values[index] <= MAX_TARGET_QUERIES:
            raise ValueError("active target_count must be in 1..6")
        if not 0 <= proposal_values[index] <= MAX_DRAFT_TOKENS:
            raise ValueError("active proposal_count must be in 0..5")
        if proposal_values[index] == 0:
            if target_values[index] != 1:
                raise ValueError("active no-proposal request must use C1 target_count == 1")
        elif proposal_values[index] + 1 != target_values[index]:
            raise ValueError("active proposals require target_count == proposal_count + 1")
        if budget_values[index] < 1:
            raise ValueError("active steady-decode request needs a positive output budget")
        if start_values[index] < 0 or limits[index] <= start_values[index]:
            raise ValueError("active request has an invalid start/context interval")
        if start_values[index] + target_values[index] > limits[index]:
            raise ValueError("active target rows exceed the context limit")
        if sample_values[index] != 1:
            raise ValueError("active steady-decode request must set need_sample == 1")
        result[index] = torch.tensor([
            int(bindings[index]),
            target_values[index],
            proposal_values[index],
            budget_values[index],
            start_values[index],
            limits[index],
            sample_values[index],
        ], dtype=torch.int64)
    return result


def verify_control_from_target_batch(
    target: torch.Tensor,
    proposed: torch.Tensor,
    metadata: torch.Tensor,
    active: torch.Tensor,
    binding_sidecar: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the greedy C6/C5 contract independently on the request axis.

    ``target`` is the already-reduced global target id tensor ``[B, 6]``.
    ``proposed`` is ``[B, 5]``.  ``metadata`` is ``[B, 7]`` from
    :func:`batch_control_metadata`. HPU callers must provide the CPU-derived
    I32 ``[B, 2]`` ``[low, high]`` sidecar; CPU-only callers may omit it for
    legacy compatibility. No request is sliced or launched alone.
    """
    if target.ndim != 2 or target.shape[1] != MAX_TARGET_QUERIES:
        raise ValueError("batched target verification requires [B, 6] ids")
    if target.dtype not in (torch.int32, torch.int64):
        raise TypeError("batched target ids must be int32 or int64")
    batch_size = target.shape[0]
    if not 1 <= batch_size <= MAX_BATCH_REQUESTS:
        raise ValueError(f"batch size must be in 1..{MAX_BATCH_REQUESTS}")
    if proposed.shape != (batch_size, MAX_DRAFT_TOKENS) or proposed.device != target.device:
        raise ValueError("batched verification requires matching [B, 5] proposals")
    if proposed.dtype not in (torch.int32, torch.int64):
        raise TypeError("batched proposals must be int32 or int64")
    if metadata.shape != (batch_size, 7) or metadata.device != target.device or metadata.dtype != torch.int64:
        raise ValueError("batched verification requires matching I64 [B, 7] metadata")
    if active.shape != (batch_size, ) or active.device != target.device or active.dtype != torch.bool:
        raise ValueError("batched verification requires matching bool [B] activity")
    binding_sidecar = _validate_binding_sidecar(binding_sidecar, target.device, batch_size)

    count = metadata[:, 1]
    proposal_count = metadata[:, 2]
    remaining = metadata[:, 3]
    start = metadata[:, 4]
    limit = metadata[:, 5]
    sample = metadata[:, 6]
    lane = torch.arange(MAX_DRAFT_TOKENS, dtype=torch.int64, device=target.device).unsqueeze(0)
    mismatch = (lane < proposal_count.unsqueeze(1)) & (target[:, :MAX_DRAFT_TOKENS] != proposed)
    accepted = torch.where(mismatch, lane, proposal_count.unsqueeze(1)).amin(1)
    has_proposals = proposal_count > 0
    proposal_committed = torch.where(has_proposals, accepted, count - 1) + 1
    budgeted_committed = torch.minimum(proposal_committed, remaining.clamp_min(0))
    output_count = torch.minimum(budgeted_committed, (limit - start).clamp_min(0))

    out_lane = torch.arange(MAX_TARGET_QUERIES, dtype=torch.int64, device=target.device).unsqueeze(0)
    last_index = (count - 1).clamp(0, MAX_TARGET_QUERIES - 1).unsqueeze(1).expand(-1, MAX_TARGET_QUERIES)
    candidates = torch.where(has_proposals.unsqueeze(1), target, target.gather(1, last_index))
    target_lane_valid = ((target >= 0) | (out_lane >= count.unsqueeze(1))).all(1)
    proposal_lane_valid = ((proposed >= 0) | (lane >= proposal_count.unsqueeze(1))).all(1)
    if binding_sidecar is None:
        binding_round_valid = (metadata[:, 0] >> 32) >= 1
        binding_generation = metadata[:, 0] & _SLOT_GENERATION_MASK
    else:
        binding_round_valid = binding_sidecar[:, BINDING_SIDECAR_HIGH] >= 1
        binding_generation = binding_sidecar[:, BINDING_SIDECAR_LOW] & _SLOT_GENERATION_MASK
    valid = (binding_round_valid & (binding_generation >= 1)
             & (count >= 1) & (count <= MAX_TARGET_QUERIES)
             & (proposal_count >= 0) & (proposal_count <= MAX_DRAFT_TOKENS)
             & (((proposal_count == 0) & (count == 1)) | (proposal_count + 1 == count))
             & (remaining >= 1) & (start >= 0) & (start + count <= limit)
             & (sample == 1) & (proposal_committed >= 1) & (proposal_committed <= count)
             & target_lane_valid & proposal_lane_valid)
    if binding_sidecar is None:
        status = torch.where(active, (~valid).to(torch.int64), torch.full_like(count, BATCH_INACTIVE))
    else:
        status = torch.where(
            active,
            (~valid).to(torch.int32),
            torch.full_like(binding_sidecar[:, BINDING_SIDECAR_LOW], BATCH_INACTIVE),
        )
    committed_valid = valid & active
    safe_output_count = torch.where(committed_valid, output_count, torch.zeros_like(output_count))
    safe_committed = safe_output_count
    output = torch.where((out_lane < safe_output_count.unsqueeze(1)) & committed_valid.unsqueeze(1), candidates, -1)
    anchor = output.gather(
        1,
        (safe_output_count - 1).clamp(0, MAX_TARGET_QUERIES - 1).unsqueeze(1),
    ).squeeze(1).clamp_min(0)
    draft_enabled = (committed_valid & (output_count == proposal_committed)
                     & (remaining - output_count >= 6) & (start + output_count + 6 <= limit))
    return target, output, safe_committed, safe_output_count, anchor, draft_enabled, status


def verify_control_from_logits_batch(
    logits: torch.Tensor,
    proposed: torch.Tensor,
    metadata: torch.Tensor,
    active: torch.Tensor,
    binding_sidecar: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience wrapper for callers that already own full target logits."""
    if logits.ndim != 3 or logits.shape[1] != MAX_TARGET_QUERIES:
        raise ValueError("batched target logits require [B, 6, V]")
    return verify_control_from_target_batch(logits.argmax(-1), proposed, metadata, active, binding_sidecar)


def pack_record_batch(
    metadata: torch.Tensor,
    committed: torch.Tensor,
    output_count: torch.Tensor,
    output: torch.Tensor,
    draft_ids: torch.Tensor,
    draft_enabled: torch.Tensor,
    status: torch.Tensor,
    binding_sidecar: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack one immutable 16-word commit record per request into ``[B, 16]``.

    With an I32 ``[low, high]`` sidecar, failed/inactive words are zeroed as I32
    before being packed into the I64 binding column. The I64 metadata binding is
    not read in this path.
    """
    if metadata.ndim != 2 or metadata.shape[1] != 7 or metadata.dtype != torch.int64:
        raise ValueError("batch commit metadata must be I64 [B, 7]")
    batch_size = metadata.shape[0]
    if not 1 <= batch_size <= MAX_BATCH_REQUESTS:
        raise ValueError(f"batch size must be in 1..{MAX_BATCH_REQUESTS}")
    if committed.shape != (batch_size, ) or output_count.shape != (batch_size, ):
        raise ValueError("batch commit counts must be [B]")
    if committed.dtype not in (torch.int32, torch.int64) or output_count.dtype not in (torch.int32, torch.int64):
        raise TypeError("batch commit counts must be integer tensors")
    if output.shape != (batch_size, MAX_TARGET_QUERIES) or draft_ids.shape != (batch_size, MAX_DRAFT_TOKENS):
        raise ValueError("batch commit ids must be fixed C6/C5 tensors")
    if output.dtype not in (torch.int32, torch.int64) or draft_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("batch commit ids must be integer tensors")
    if draft_enabled.shape != (batch_size, ) or draft_enabled.dtype != torch.bool:
        raise ValueError("batch commit draft_enabled must be bool [B]")
    if status.shape != (batch_size, ) or status.dtype not in (torch.int32, torch.int64):
        raise ValueError("batch commit status must be integer [B]")
    binding_sidecar = _validate_binding_sidecar(binding_sidecar, metadata.device, batch_size)
    valid = status == 0
    if binding_sidecar is None:
        binding_column = torch.where(valid.unsqueeze(1), metadata[:, :1], torch.zeros_like(metadata[:, :1]))
    else:
        zeros_i32 = torch.zeros_like(binding_sidecar[:, BINDING_SIDECAR_LOW])
        low = torch.where(valid, binding_sidecar[:, BINDING_SIDECAR_LOW], zeros_i32)
        high = torch.where(valid, binding_sidecar[:, BINDING_SIDECAR_HIGH], zeros_i32)
        binding_i64 = torch.stack((low, high), dim=1).flatten().view(torch.int64)
        binding_column = binding_i64.unsqueeze(1)
    draft_count = torch.where(draft_enabled & valid, MAX_DRAFT_TOKENS, 0).to(torch.int64)
    out_lane = torch.arange(MAX_TARGET_QUERIES, device=output.device).unsqueeze(0)
    output = torch.where((out_lane < output_count.unsqueeze(1)) & valid.unsqueeze(1), output.to(torch.int64), -1)
    draft_ids = torch.where((draft_enabled & valid).unsqueeze(1), draft_ids.to(torch.int64), -1)
    return torch.cat((
        binding_column,
        torch.where(valid, committed.to(torch.int64), torch.zeros_like(committed, dtype=torch.int64)).unsqueeze(1),
        torch.where(
            valid,
            output_count.to(torch.int64),
            torch.zeros_like(output_count, dtype=torch.int64),
        ).unsqueeze(1),
        draft_count.unsqueeze(1),
        output,
        draft_ids,
        status.to(torch.int64).unsqueeze(1),
    ), dim=1)


@dataclass(frozen=True)
class BatchVerifyCommit:
    committed: tuple[int, ...]
    output: tuple[tuple[int, ...], ...]
    draft: tuple[tuple[int, ...], ...]


def validate_batch_records(
    records: torch.Tensor,
    *,
    round_generation: int,
    request_slots: Sequence[int] | torch.Tensor,
    slot_generations: Sequence[int] | torch.Tensor,
    max_committed: Sequence[int] | torch.Tensor,
    active: Sequence[bool] | torch.Tensor,
) -> BatchVerifyCommit:
    """Validate one pinned host record matrix and decode per-request commits."""
    if not isinstance(records, torch.Tensor) or records.device.type != "cpu":
        raise ValueError("batch verify validation requires a CPU host record")
    if records.dtype not in (torch.int32, torch.int64) or records.ndim != 2:
        raise ValueError("batch verify record must be an integer [B, 16] tensor")
    batch_size = records.shape[0]
    if records.shape[1] != BATCH_RECORD_SIZE or not 1 <= batch_size <= MAX_BATCH_REQUESTS:
        raise ValueError("batch verify record has an invalid [B, 16] shape")
    slots, generations, active_values = _slot_identity(request_slots, slot_generations, active, batch_size)
    limits = _host_ints(max_committed, "max_committed", batch_size)
    _validate_max_committed(limits, active_values)

    committed_rows, output_rows, draft_rows = [], [], []
    values = records.tolist()
    for index, row in enumerate(values):
        if not active_values[index]:
            if (row[0] != 0 or row[1:4] != [0, 0, 0] or any(token != -1 for token in row[4:BATCH_STATUS])
                    or row[BATCH_STATUS] != BATCH_INACTIVE):
                raise RuntimeError(f"inactive batch lane {index} published an active record")
            committed_rows.append(0)
            output_rows.append(())
            draft_rows.append(())
            continue
        committed, output_count, draft_count = row[1:4]
        expected = pack_binding_word(round_generation, generations[index], slots[index])
        status = row[BATCH_STATUS]
        valid = (
            row[0] == expected
            and status == 0
            and 1 <= committed <= limits[index]
            and output_count == committed
            and output_count <= MAX_TARGET_QUERIES
            and draft_count in (0, MAX_DRAFT_TOKENS)
        )
        if not valid:
            raise RuntimeError(
                "Invalid DSpark batch commit record: "
                f"lane={index} slot={slots[index]} expected_binding={expected} record={row}"
            )
        output = tuple(row[BATCH_OUTPUT_START:BATCH_OUTPUT_START + output_count])
        draft = tuple(row[BATCH_DRAFT_START:BATCH_DRAFT_START + draft_count])
        if len(output) != output_count or len(draft) != draft_count or any(token < 0 for token in (*output, *draft)):
            raise RuntimeError(f"Invalid DSpark batch output/draft ids: lane={index} record={row}")
        committed_rows.append(committed)
        output_rows.append(output)
        draft_rows.append(draft)
    return BatchVerifyCommit(tuple(committed_rows), tuple(output_rows), tuple(draft_rows))


@dataclass(frozen=True)
class BatchVerifyTicket:
    slot: int
    round_generation: int
    request_slots: tuple[int, ...]
    slot_generations: tuple[int, ...]
    max_committed: tuple[int, ...]
    active: tuple[bool, ...]
    record: torch.Tensor
    host: torch.Tensor
    completion: object


class BatchVerifyRing:
    """Capacity-bounded PP1 ticket ring with one host copy per staged batch.

    Tickets follow ``acquire -> stage -> consume -> release``. Logical round
    generations are independent of which currently free physical ring slot is
    selected. An optional owner callback is invoked at acquire and consume so
    the runtime can reject a ticket after authoritative request-slot reuse.
    """

    def __init__(
        self,
        device,
        *,
        batch_capacity: int = DEFAULT_BATCH_CAPACITY,
        size: int = 2,
        native_readback: Callable[[torch.Tensor], tuple[torch.Tensor, object]] | None = None,
        owner_validator: Callable[[tuple[int, ...], tuple[int, ...], tuple[bool, ...]], bool] | None = None,
    ):
        if type(batch_capacity) is not int or not 1 <= batch_capacity <= MAX_BATCH_REQUESTS:
            raise ValueError(f"batch capacity must be in 1..{MAX_BATCH_REQUESTS}")
        if size < 2:
            raise ValueError("batch verify ring needs at least two retained tickets")
        self.device = torch.device(device)
        self.batch_capacity = batch_capacity
        self.records = [
            torch.empty((batch_capacity, BATCH_RECORD_SIZE), dtype=torch.int64, device=device) for _ in range(size)
        ]
        self.host = [
            torch.empty((batch_capacity, BATCH_RECORD_SIZE), dtype=torch.int64, device="cpu").pin_memory("hpu")
            for _ in range(size)
        ]
        self.events = [torch.hpu.Event() for _ in range(size)]
        self.generations = [0] * size
        self.staged = [None] * size
        self.synchronized = [0] * size
        self.consumed = [0] * size
        self.released = [0] * size
        self.generation, self.closed = 0, False
        self.native_readback = native_readback
        self.owner_validator = owner_validator
        self.readbacks = [None] * size
        self.copy_count = 0
        self.event_record_count = 0
        self.completion_sync_count = 0

    def _validate_owner(self, slots, generations, active_values):
        if self.owner_validator is None:
            return
        valid = self.owner_validator(slots, generations, active_values)
        if type(valid) is not bool:
            raise TypeError("owner_validator must return bool")
        if not valid:
            raise RuntimeError("DSpark batch verify ticket no longer owns its request slots")

    def acquire(
        self,
        *,
        round_generation: int,
        request_slots: Sequence[int] | torch.Tensor,
        slot_generations: Sequence[int] | torch.Tensor,
        max_committed: Sequence[int] | torch.Tensor,
        active: Sequence[bool] | torch.Tensor,
    ) -> BatchVerifyTicket:
        if self.closed:
            raise RuntimeError("DSpark batch verify ring is closed")
        round_generation = _scalar_generation(round_generation, "round_generation", minimum=1)
        slots, generations, active_values = _slot_identity(
            request_slots,
            slot_generations,
            active,
            self.batch_capacity,
        )
        limits = _host_ints(max_committed, "max_committed", self.batch_capacity)
        _validate_max_committed(limits, active_values)
        if round_generation <= self.generation:
            raise RuntimeError("DSpark batch verify round generation must advance")
        free_slots = [
            index for index in range(len(self.records))
            if self.generations[index] == self.released[index]
        ]
        if not free_slots:
            raise RuntimeError("DSpark batch verify ring has no free physical slot")
        self._validate_owner(slots, generations, active_values)
        slot = free_slots[0]
        self.generation = round_generation
        self.generations[slot] = round_generation
        self.staged[slot] = None
        self.synchronized[slot] = 0
        self.readbacks[slot] = None
        return BatchVerifyTicket(
            slot,
            round_generation,
            slots,
            generations,
            limits,
            active_values,
            self.records[slot],
            self.host[slot],
            self.events[slot],
        )

    def _check(self, ticket: BatchVerifyTicket):
        if (self.closed or self.generations[ticket.slot] != ticket.round_generation
                or self.released[ticket.slot] == ticket.round_generation):
            raise RuntimeError("Stale or already consumed DSpark batch verify result")

    def stage(self, ticket: BatchVerifyTicket):
        self._check(ticket)
        if ticket.record.shape != (self.batch_capacity, BATCH_RECORD_SIZE):
            raise RuntimeError("Batch verify ticket record shape changed")
        if self.staged[ticket.slot] == ticket.round_generation or self.consumed[ticket.slot] == ticket.round_generation:
            raise RuntimeError("DSpark batch verify ticket was already staged or consumed")
        if self.native_readback is not None:
            host, completion = self.native_readback(ticket.record)
            if not isinstance(host, torch.Tensor) or host.device.type != "cpu":
                raise RuntimeError("Batch native readback did not return a CPU host tensor")
            expected_shapes = (
                (self.batch_capacity, BATCH_RECORD_SIZE),
                (1, self.batch_capacity * BATCH_RECORD_SIZE),
            )
            if host.shape not in expected_shapes:
                raise RuntimeError("Batch native readback returned the wrong record geometry")
            self.readbacks[ticket.slot] = (host.reshape(self.batch_capacity, BATCH_RECORD_SIZE), completion)
            self.copy_count += 1
            self.staged[ticket.slot] = ticket.round_generation
            return
        ticket.host.copy_(ticket.record, non_blocking=True)
        ticket.completion.record(torch.hpu.current_stream())
        self.copy_count += 1
        self.event_record_count += 1
        self.staged[ticket.slot] = ticket.round_generation

    def _await(self, ticket: BatchVerifyTicket) -> torch.Tensor:
        self._check(ticket)
        if self.staged[ticket.slot] != ticket.round_generation:
            raise RuntimeError("Batch verify record was not staged")
        if self.consumed[ticket.slot] == ticket.round_generation:
            raise RuntimeError("DSpark batch verify result was already consumed")
        self._validate_owner(ticket.request_slots, ticket.slot_generations, ticket.active)
        return self._drain(ticket)

    def _drain(self, ticket: BatchVerifyTicket) -> torch.Tensor:
        """Complete the one staged host readback exactly once for this ticket."""
        if self.synchronized[ticket.slot] == ticket.round_generation:
            host = self.readbacks[ticket.slot]
            if self.native_readback is not None:
                if host is None:
                    raise RuntimeError("Batch verify record was not staged")
                return host[0]
            return ticket.host
        if self.native_readback is not None:
            readback = self.readbacks[ticket.slot]
            if readback is None:
                raise RuntimeError("Batch verify record was not staged")
            host, completion = readback
            completion.synchronize()
        else:
            ticket.completion.synchronize()
            host = ticket.host
        self.synchronized[ticket.slot] = ticket.round_generation
        self.completion_sync_count += 1
        return host

    def consume(self, ticket: BatchVerifyTicket) -> BatchVerifyCommit:
        host = self._await(ticket)
        result = validate_batch_records(
            host,
            round_generation=ticket.round_generation,
            request_slots=ticket.request_slots,
            slot_generations=ticket.slot_generations,
            max_committed=ticket.max_committed,
            active=ticket.active,
        )
        self.consumed[ticket.slot] = ticket.round_generation
        return result

    def release(self, ticket: BatchVerifyTicket):
        self._check(ticket)
        if self.consumed[ticket.slot] != ticket.round_generation:
            raise RuntimeError("DSpark batch verify ticket must be consumed before release")
        self.released[ticket.slot] = ticket.round_generation
        self.readbacks[ticket.slot] = None

    def abort(self, ticket: BatchVerifyTicket):
        """Drain any submitted work, discard the ticket and free its ring slot.

        This never publishes a commit and never attempts to roll back device
        model-state writes.  A staged copy is synchronized here because the
        next generation may immediately reuse the pinned host buffer.  If an
        error occurred before ``stage`` recorded its completion event, the HPU
        stream is synchronized globally before the slot is released.
        """
        self._check(ticket)
        if self.consumed[ticket.slot] != ticket.round_generation:
            if self.staged[ticket.slot] == ticket.round_generation:
                self._drain(ticket)
            elif self.device.type == "hpu":
                torch.hpu.synchronize()
                self.synchronized[ticket.slot] = ticket.round_generation
                self.completion_sync_count += 1
        self.released[ticket.slot] = ticket.round_generation
        self.readbacks[ticket.slot] = None

    def close(self):
        if self.closed:
            return
        for slot, generation in enumerate(self.generations):
            if (generation == self.released[slot] or self.consumed[slot] == generation
                    or self.staged[slot] != generation):
                continue
            readback = self.readbacks[slot]
            if readback is not None:
                readback[1].synchronize()
            else:
                self.events[slot].synchronize()
            self.completion_sync_count += 1
        self.closed = True


__all__ = [
    "BINDING_SIDECAR_HIGH",
    "BINDING_SIDECAR_LOW",
    "BINDING_SIDECAR_SIZE",
    "BATCH_DRAFT_START",
    "BATCH_INACTIVE",
    "BATCH_OUTPUT_START",
    "BATCH_RECORD_SIZE",
    "BATCH_STATUS",
    "DEFAULT_BATCH_CAPACITY",
    "MAX_BATCH_REQUESTS",
    "BatchVerifyCommit",
    "BatchVerifyRing",
    "BatchVerifyTicket",
    "batch_binding_sidecar",
    "batch_binding_words",
    "batch_control_metadata",
    "pack_binding_word",
    "pack_record_batch",
    "unpack_binding_word",
    "validate_batch_records",
    "verify_control_from_logits_batch",
    "verify_control_from_target_batch",
]
