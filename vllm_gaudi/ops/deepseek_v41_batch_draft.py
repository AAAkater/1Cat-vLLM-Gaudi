# SPDX-License-Identifier: Apache-2.0
"""Capacity-bounded packed DSpark draft backbone, Markov and context commit.

The module shares every original MTP weight and performs no per-request model
loop.  The five draft queries of every request are packed on one query axis;
only the five Markov steps are sequential, and each step acts on all requests
at once.  Program/paged-attention integration remains a thin wrapper owned by
the target worker.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable

import torch
import torch.nn.functional as F

from vllm_gaudi.ops.deepseek_v41_batch_attention import write_state_rows
from vllm_gaudi.ops.deepseek_v41_capacity import validate_dspark_capacity
from vllm_gaudi.ops.deepseek_v41_batch_verify import (
    BINDING_SIDECAR_SIZE,
    DEFAULT_BATCH_CAPACITY,
    MAX_BATCH_REQUESTS,
    BatchVerifyRing,
    batch_binding_sidecar,
    batch_control_metadata,
    pack_record_batch,
    verify_control_from_target_batch,
)
from vllm_gaudi.ops.deepseek_v41_math import hc_post, hc_pre, pack_swa, rms_norm

MAX_DRAFT_TOKENS = 5
TARGET_QUERY_BUCKET = 6
SWA_ROWS = 256
DRAFT_WINDOW = 128
DRAFT_ATTENTION_WIDTH = 192
DRAFT_DENSE_ROW_CHUNK = 40

# Runner-facing call signature:
#
#   backbone = PackedDraftBackbone(draft, batch_state)
#   hidden, logits = backbone(anchors[B,1], positions[B,5], slots[B], active[B])
#
#   markov = PackedDraftMarkovSampler(draft, batch_capacity=2)
#   proposals, confidence = markov(anchors[B], hidden[B,5,H], logits[B,5,V], active[B])


def _linear(value: torch.Tensor, owner) -> torch.Tensor:
    # Preserve the B1 prepared-dense branch when packing B*5 rows. program.linear
    # gates that branch at six rows; full five-query request lanes use the same
    # row-independent prepared matrix without changing ordinary serial dispatch.
    # The 40-row chunks retain the qualified N8 geometry while allowing any N up
    # to the shared contract bound.
    prepared = getattr(owner, "dspark_dense_kn", None)
    packed_rows = (value.ndim == 2 and value.shape[0] > TARGET_QUERY_BUCKET
                   and value.shape[0] % MAX_DRAFT_TOKENS == 0)
    if prepared is not None and packed_rows:
        rows = value.shape[0]
        products = tuple(
            torch.mm(value[start:min(start + DRAFT_DENSE_ROW_CHUNK, rows)], prepared)
            for start in range(0, rows, DRAFT_DENSE_ROW_CHUNK)
        )
        product = products[0] if len(products) == 1 else torch.cat(products, dim=0)
        bias = getattr(owner, "bias", None)
        return product if bias is None else product + bias
    # Import lazily so this owner can be CPU-tested without importing the model
    # package's device initialization path first.
    from vllm_gaudi.models.deepseek_v41_program import linear

    return linear(value, owner)


def _validate_batch_tensor(value: torch.Tensor, shape: tuple, name: str, dtype: torch.dtype | None = None):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")


def _call_with_optional_context_done(
    component,
    *args,
    context_done: tuple[torch.Tensor, ...] | None,
    name: str,
):
    """Pass the commit dependency to compatible custom stage components.

    CPU-only legacy components may keep their old signature.  On HPU an
    incompatible component is rejected because dropping the dependency can
    make the context write disappear from the compiled graph.
    """
    target = component.forward if isinstance(component, torch.nn.Module) else component
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        parameters = None
    if parameters is not None:
        parameter = parameters.get("context_done")
        if parameter is not None:
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                return component(*args, context_done)
            return component(*args, context_done=context_done)
        if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
            return component(*args, context_done=context_done)
    if context_done is not None and args and isinstance(args[0], torch.Tensor) and args[0].device.type == "hpu":
        raise RuntimeError(
            f"custom DSpark {name} must accept context_done on HPU so the packed context write "
            "keeps its completion dependency")
    return component(*args)


def write_state_rows_masked(cache: torch.Tensor, values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Write selected packed rows, treating negative row ids as no-write.

    HPU uses the existing dependency-carrying native writer.  The CPU branch is
    only for contract tests and keeps the same indexed-write semantics.
    """
    if cache.device.type == "cpu":
        if cache.ndim != 2 or values.ndim != 2 or rows.ndim != 1 or values.shape[0] != rows.numel():
            raise ValueError("masked state rows have incompatible shapes")
        if values.shape[1] != cache.shape[1]:
            raise ValueError("masked state row width differs from its cache")
        rows = rows.to(torch.int64)
        valid = rows >= 0
        safe_rows = rows.clamp_min(0)
        previous = cache.index_select(0, safe_rows)
        update = torch.where(valid.unsqueeze(1), values, previous)
        cache.scatter_(0, safe_rows.unsqueeze(1).expand_as(values), update)
        return torch.zeros(rows.shape, dtype=torch.int32, device=rows.device)
    return write_state_rows(cache, values.contiguous(), rows.contiguous())


@dataclass(frozen=True)
class DraftAttentionRows:
    row_ids: torch.Tensor
    indices: torch.Tensor
    lengths: torch.Tensor
    sparse_indices: torch.Tensor
    context_row_ids: torch.Tensor


@dataclass(frozen=True)
class DraftContextRows:
    target_states: torch.Tensor
    positions: torch.Tensor
    request_slots: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class BatchDraftResult:
    """Duck-compatible with the scheduler worker's DSparkDraftBatch."""

    accepted: torch.Tensor
    output_token_ids: tuple[tuple[int, ...], ...]
    draft_token_ids: tuple[tuple[int, ...], ...]
    completed: tuple[bool, ...] | None = None
    completion_event: object | None = None


class DraftBatchState(torch.nn.Module):
    """Authoritative packed draft SWA bank per MTP layer.

    Ring row ownership is ``slot * 256 + absolute_position.remainder(256)``.
    Tentative draft KV exists only in the current packed transaction and is
    never stored in this persistent bank.
    """

    def __init__(self, draft, *, capacity: int = DEFAULT_BATCH_CAPACITY, backing_rings=None):
        super().__init__()
        validate_dspark_capacity(capacity, allow_serial=True, name="draft batch capacity")
        if len(draft.layers) != 3:
            raise ValueError("DSpark batch draft requires exactly three MTP layers")
        if backing_rings is not None and len(backing_rings) != len(draft.layers):
            raise ValueError("draft backing bank must expose one ring per MTP layer")
        self.capacity = capacity
        self.batch_capacity = capacity
        self.owns_rings = backing_rings is None
        self.rings = []
        for index, layer in enumerate(draft.layers):
            swa = layer.attention.swa
            if swa.ndim != 2 or swa.shape[0] != SWA_ROWS:
                raise ValueError(f"draft layer {index} must expose a 256-row packed SWA ring")
            ring = (torch.zeros((capacity * SWA_ROWS, swa.shape[1]), dtype=swa.dtype, device=swa.device)
                    if backing_rings is None else backing_rings[index])
            if ring.shape != (capacity * SWA_ROWS, swa.shape[1]) or ring.dtype != swa.dtype:
                raise ValueError("draft backing ring geometry differs from the prepared MTP ring")
            self.register_buffer(f"ring{index}", ring, False)
            self.rings.append(getattr(self, f"ring{index}"))

    def state_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.rings)

    def ring(self, layer_index: int) -> torch.Tensor:
        return self.rings[layer_index]

    def clear_slot(self, slot: int):
        if not 0 <= slot < self.capacity:
            raise ValueError("draft state slot is outside capacity")
        for ring in self.rings:
            ring[slot * SWA_ROWS:(slot + 1) * SWA_ROWS].zero_()

    def slot_rings(self, slot: int) -> tuple[torch.Tensor, ...]:
        """Return fixed views of the authoritative rings for one request slot."""
        if not 0 <= slot < self.capacity:
            raise ValueError("draft state slot is outside capacity")
        start = slot * SWA_ROWS
        return tuple(ring[start:start + SWA_ROWS] for ring in self.rings)

    def adopt_single_ring(self, slot: int, draft):
        """Promote a completed B1/C6-seed draft ring into one request slot."""
        if not 0 <= slot < self.capacity:
            raise ValueError("draft state slot is outside capacity")
        for destination, layer in zip(self.slot_rings(slot), draft.layers, strict=True):
            destination.copy_(layer.attention.swa)

    def adopt_packed_slot(self, slot: int, source_rings):
        """Copy an existing authoritative per-request bank into one slot."""
        target = self.slot_rings(slot)
        if len(source_rings) != len(target):
            raise ValueError("source draft bank must expose one ring per MTP layer")
        for destination, source in zip(target, source_rings, strict=True):
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise ValueError("source draft slot differs from the prepared ring geometry")
            destination.copy_(source)


def draft_attention_rows(
    positions: torch.Tensor,
    slots: torch.Tensor,
    active: torch.Tensor,
    *,
    ring_rows: int,
    window: int = DRAFT_WINDOW,
) -> DraftAttentionRows:
    """Build request-isolated packed MLA indices for Bx5 draft queries.

    All five queries of a request see the same 128 accepted-context rows and
    all five tentative rows, preserving the original bidirectional draft block.
    """
    if positions.ndim != 2 or positions.shape[1] != MAX_DRAFT_TOKENS:
        raise ValueError("draft positions must have shape [B, 5]")
    batch_size = positions.shape[0]
    validate_dspark_capacity(batch_size, allow_serial=True, name="draft batch size")
    if slots.shape != (batch_size, ) or active.shape != (batch_size, ) or active.dtype != torch.bool:
        raise ValueError("draft slots/activity must have shapes [B] and [B]")
    if slots.device != positions.device or active.device != positions.device:
        raise ValueError("draft row metadata must share one device")
    if ring_rows < SWA_ROWS or ring_rows % SWA_ROWS:
        raise ValueError("draft ring must contain complete 256-row request slots")

    device = positions.device
    offsets = torch.arange(window, dtype=torch.int32, device=device)
    first_position = positions[:, :1]
    context_positions = first_position - window + offsets.unsqueeze(0)
    context_rows = slots.unsqueeze(1) * SWA_ROWS + context_positions.remainder(SWA_ROWS)
    context_rows = torch.where((context_positions >= 0) & active.unsqueeze(1), context_rows, -1)

    tentative_lane = torch.arange(MAX_DRAFT_TOKENS, dtype=torch.int32, device=device)
    tentative_rows = ring_rows + torch.arange(batch_size, device=device, dtype=torch.int32).unsqueeze(1) * (
        MAX_DRAFT_TOKENS) + tentative_lane.unsqueeze(0)
    tentative_rows = torch.where(active.unsqueeze(1), tentative_rows, -1)

    row_ids = torch.cat((context_rows, tentative_rows), dim=1).reshape(-1).to(torch.int32)
    local = torch.cat((
        torch.arange(window, dtype=torch.int32, device=device),
        window + tentative_lane,
    ))
    local = F.pad(local, (0, DRAFT_ATTENTION_WIDTH - local.numel()), value=-1)
    request_base = (torch.arange(batch_size, device=device, dtype=torch.int32) * local.numel()).reshape(
        batch_size, 1, 1)
    indices = local.reshape(1, 1, -1).expand(batch_size, MAX_DRAFT_TOKENS, -1)
    indices = torch.where(indices >= 0, indices + request_base, -1).reshape(batch_size * MAX_DRAFT_TOKENS, -1)
    lengths = torch.where(active, torch.full_like(active, window + MAX_DRAFT_TOKENS, dtype=torch.int32),
                          torch.zeros_like(active, dtype=torch.int32))
    lengths = lengths.unsqueeze(1).expand(batch_size, MAX_DRAFT_TOKENS).reshape(-1)
    context_row_ids = context_rows.reshape(-1).contiguous()
    context_per_request = window
    selected_context = (
        torch.arange(context_per_request, device=device, dtype=torch.int32).unsqueeze(0)
        + torch.arange(batch_size, device=device, dtype=torch.int32).unsqueeze(1) * context_per_request)
    selected_context = torch.where((context_positions >= 0) & active.unsqueeze(1), selected_context, -1)
    tentative_selected = (
        batch_size * context_per_request
        + torch.arange(batch_size, device=device, dtype=torch.int32).unsqueeze(1) * MAX_DRAFT_TOKENS
        + tentative_lane.unsqueeze(0))
    sparse_page = F.pad(
        torch.cat((selected_context, tentative_selected), dim=1),
        (0, DRAFT_ATTENTION_WIDTH - context_per_request - MAX_DRAFT_TOKENS),
        value=-1,
    )
    sparse_indices = sparse_page.unsqueeze(1).expand(batch_size, MAX_DRAFT_TOKENS, -1)
    sparse_indices = torch.where(active.unsqueeze(1).unsqueeze(2), sparse_indices, -1)
    sparse_indices = sparse_indices.reshape(batch_size * MAX_DRAFT_TOKENS, -1).contiguous()
    return DraftAttentionRows(
        row_ids.contiguous(), indices.contiguous(), lengths.contiguous(), sparse_indices,
        context_row_ids)


def draft_context_rows(
    target_states: torch.Tensor,
    positions: torch.Tensor,
    request_slots: torch.Tensor,
    committed: torch.Tensor,
    active: torch.Tensor,
) -> DraftContextRows:
    """Select only the accepted target prefix; never expose a rejected suffix."""
    if target_states.ndim != 3 or target_states.shape[1:] != (TARGET_QUERY_BUCKET, target_states.shape[2]):
        raise ValueError("target states must have shape [B, 6, D]")
    batch_size = target_states.shape[0]
    if positions.shape != (batch_size, TARGET_QUERY_BUCKET):
        raise ValueError("target positions must have shape [B, 6]")
    if request_slots.shape != (batch_size, ) or committed.shape != (batch_size, ) or active.shape != (batch_size, ):
        raise ValueError("context slots/committed/active must have shape [B]")
    if active.dtype != torch.bool:
        raise ValueError("context activity must be boolean")
    if request_slots.dtype != torch.int32 or committed.dtype not in (torch.int32, torch.int64):
        raise ValueError("context slots must be int32 and committed counts must be integer")
    # Host metadata validates committed bounds before upload. Runtime decisions
    # stay tensor-only so fullgraph capture never performs a device readback.
    query_lane = torch.arange(TARGET_QUERY_BUCKET, device=target_states.device)
    valid = (query_lane.unsqueeze(0) < committed.unsqueeze(1)) & active.unsqueeze(1)
    return DraftContextRows(
        target_states=target_states,
        positions=positions.to(torch.int32),
        request_slots=request_slots.to(torch.int32),
        valid=valid,
    )


def insert_context_batch(
    draft,
    batch_state: DraftBatchState,
    target_states: torch.Tensor,
    positions: torch.Tensor,
    request_slots: torch.Tensor,
    committed: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Commit accepted draft context directly into each request slot."""
    selected = draft_context_rows(target_states, positions, request_slots, committed, active)
    first = draft.weights.get_submodule("0")
    target_rows = selected.target_states.reshape(-1, selected.target_states.shape[-1])
    positions = selected.positions.reshape(-1)
    valid_rows = selected.valid.reshape(-1)
    safe_positions = torch.where(valid_rows, positions, torch.zeros_like(positions))
    slot_rows = selected.request_slots.unsqueeze(1).expand(-1, TARGET_QUERY_BUCKET).reshape(-1)
    main_value = rms_norm(_linear(target_rows, first.main_proj), first.main_norm.weight, draft.eps)
    outputs = []
    for index, layer in enumerate(draft.layers):
        attention = layer.attention
        kv = attention.linear(main_value, attention.weights.wkv)
        kv = rms_norm(kv, attention.weights.kv_norm.weight, attention.eps)
        kv = attention._rope(kv, safe_positions, request_batch=True)
        packed = pack_swa(kv)
        rows = slot_rows * SWA_ROWS + safe_positions.remainder(SWA_ROWS)
        rows = torch.where(valid_rows, rows, -1)
        outputs.append(write_state_rows_masked(batch_state.ring(index), packed, rows.to(torch.int32)))
    return tuple(outputs)


def decode_draft_swa_rows(packed: torch.Tensor, unused_main: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Decode a capacity-sized SWA bank using the existing native limits.

    Each native call sees at most 512 source rows and 1024 selected rows.
    Source blocks retain the original selected-row order through masked
    placement; invalid IDs stay zero and no persistent storage is modified.
    All loop bounds are static tensor shapes, independent of active requests.
    """
    if packed.ndim != 2 or packed.shape[1] != 528 or packed.shape[0] < 1 or packed.dtype != torch.uint8:
        raise ValueError("draft decoder requires a nonempty packed SWA528 bank")
    if indices.ndim != 2 or indices.shape[0] != 1 or indices.shape[1] < 1 or indices.dtype != torch.int32:
        raise ValueError("draft decoder requires nonempty int32 selected rows [1,K]")
    if indices.device != packed.device:
        raise ValueError("draft decoder indices and bank must share a device")
    operation = torch.ops.custom_op.custom_deepseek_v41_selected_kv_bf16_gaudi2
    if packed.shape[0] <= 512 and indices.shape[1] <= 1024:
        decoded, _ = operation(packed.contiguous(), unused_main, indices.contiguous(), False)
        return decoded
    outputs = []
    for selected_start in range(0, indices.shape[1], 1024):
        selected = indices[:, selected_start:selected_start + 1024].contiguous()
        result = torch.zeros((selected.shape[1], 512), dtype=torch.bfloat16, device=packed.device)
        for source_start in range(0, packed.shape[0], 512):
            source = packed[source_start:source_start + 512].contiguous()
            belongs = (selected >= source_start) & (selected < source_start + source.shape[0])
            local = torch.where(belongs, selected - source_start, -1).contiguous()
            decoded, _ = operation(source, unused_main, local, False)
            result = torch.where(belongs.reshape(-1, 1), decoded, result)
        outputs.append(result)
    return torch.cat(outputs, dim=0)


def draft_attention_batch(
    draft,
    layer_index: int,
    value: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    active: torch.Tensor,
    batch_state: DraftBatchState,
    context_done: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one MTP layer's five-query draft attention for all requests."""
    batch_size = positions.shape[0]
    if value.shape[0] != batch_size * MAX_DRAFT_TOKENS:
        raise ValueError("packed draft attention value must contain B*5 rows")
    attention = draft.layers[layer_index].attention
    if getattr(attention, "draft_dense_fp8", False):
        query_input, kv_input = attention.draft_input_projection(value)
    else:
        query_input, kv_input = attention._project_qkv_input(value)
    positions_flat = positions.reshape(-1).to(torch.int32)
    active_rows = active.unsqueeze(1).expand(batch_size, MAX_DRAFT_TOKENS).reshape(-1)
    safe_positions_flat = torch.where(active_rows, positions_flat, torch.zeros_like(positions_flat))
    query = rms_norm(query_input.contiguous(), attention.weights.q_norm.weight, attention.eps)
    query = attention.project_query(query, safe_positions_flat, request_batch=True)
    kv = rms_norm(kv_input.contiguous(), attention.weights.kv_norm.weight, attention.eps)
    kv = attention._rope(kv, safe_positions_flat, request_batch=True)
    ring = batch_state.ring(layer_index)
    rows = draft_attention_rows(positions, slots, active, ring_rows=ring.shape[0])
    # Serial draft attention also encodes/decodes tentative KV through the
    # SWA codec. Keep that numerical boundary while using only scratch storage.
    # Cache rows are all accepted contexts followed by all tentative rows,
    # matching sparse_indices rather than the request-interleaved row_ids.
    # ``unused_main`` does not participate in selected-KV addressing, but the
    # native write completion must reach it so fullgraph cannot discard the
    # preceding context mutation as dead.
    context_main = attention.swa_only_main
    if context_done is not None:
        if not isinstance(context_done, torch.Tensor) or context_done.numel() == 0:
            raise ValueError("context completion must be a nonempty tensor")
        if context_done.device != ring.device:
            raise ValueError("context completion and packed draft ring must share a device")
        context_main = context_done.amax().to(torch.uint8).expand(1, ring.shape[1]).contiguous()
    context_cache = decode_draft_swa_rows(
        ring, context_main, rows.context_row_ids.reshape(1, -1))
    tentative_rows = torch.arange(
        batch_size * MAX_DRAFT_TOKENS, dtype=torch.int32, device=ring.device)
    tentative_rows = torch.where(active_rows, tentative_rows, -1).reshape(1, -1)
    tentative_cache = decode_draft_swa_rows(pack_swa(kv), attention.swa_only_main, tentative_rows)
    cache = torch.cat((context_cache, tentative_cache), dim=0)
    output, _, _ = torch.ops.custom_op.custom_deepseek_v4_sparse_attn_bf16_gaudi2(
        query.contiguous(), cache.contiguous(), rows.sparse_indices.contiguous(),
        attention.weights.attn_sink, attention.scale)
    return attention._finish_output(output, safe_positions_flat, request_batch=True)


class PackedDraftRound(torch.nn.Module):
    """One fixed capacity/C5 body plus five-step Markov HPU graph.

    The graph accepts fixed ``[N,1]`` anchors, ``[N,5]`` positions, ``[N]``
    request slots and a ``[N]`` activity mask.  Inactive lanes never write a
    draft ring entry and return ``-1`` proposals.  ``N`` defaults to two and
    must match the authoritative batch state.
    """

    def __init__(
        self,
        draft,
        batch_state: DraftBatchState,
        *,
        batch_capacity: int = DEFAULT_BATCH_CAPACITY,
        backend: str = "hpu_backend",
        compile_graph: bool = True,
    ):
        super().__init__()
        validate_dspark_capacity(batch_capacity, allow_serial=True, name="draft round capacity")
        if batch_state.capacity != batch_capacity:
            raise ValueError("draft round capacity must match the batch state")
        self.batch_capacity = batch_capacity
        self.backbone = PackedDraftBackbone(draft, batch_state)
        self.markov = PackedDraftMarkovSampler(draft, batch_capacity=batch_capacity)
        self.backend = backend
        self.compile_graph = compile_graph
        self.compiled = None

    def _forward(self, anchors, positions, slots, active, context_done=None):
        hidden, logits = self.backbone(anchors, positions, slots, active, context_done=context_done)
        return self.markov(anchors.reshape(-1), hidden, logits, active)

    def forward(self, anchors, positions, slots, active, context_done=None):
        if anchors.shape != (self.batch_capacity, 1):
            raise ValueError(f"PackedDraftRound requires fixed anchors [{self.batch_capacity},1]")
        if positions.shape != (self.batch_capacity, MAX_DRAFT_TOKENS):
            raise ValueError(f"PackedDraftRound requires fixed positions [{self.batch_capacity},{MAX_DRAFT_TOKENS}]")
        if slots.shape != (self.batch_capacity, ) or active.shape != (self.batch_capacity, ):
            raise ValueError(f"PackedDraftRound requires fixed slots/activity [{self.batch_capacity}]")
        if self.compile_graph and anchors.device.type == "hpu":
            if self.compiled is None:
                self.compiled = torch.compile(
                    self._forward, backend=self.backend, fullgraph=True, dynamic=False)
            return self.compiled(anchors, positions, slots, active, context_done)
        return self._forward(anchors, positions, slots, active, context_done)


class PackedDraftTransaction(torch.nn.Module):
    """Unified fixed-capacity verify/debug/draft graph.

    One compiled entry consumes fixed target ids/context, verify control and
    proposals, writes only the accepted target prefix into draft request rings,
    executes one 5*N-row MTP body, performs five Markov steps, and returns one
    ``[N,16]`` commit record.  The caller owns ticket acquisition and the single
    ``[N,16]`` host readback.
    """

    def __init__(
        self,
        draft,
        batch_state: DraftBatchState,
        *,
        batch_capacity: int = DEFAULT_BATCH_CAPACITY,
        backend: str = "hpu_backend",
        compile_graph: bool = True,
    ):
        super().__init__()
        validate_dspark_capacity(batch_capacity, allow_serial=True, name="draft transaction capacity")
        if batch_state.capacity != batch_capacity:
            raise ValueError("draft transaction capacity must match the batch state")
        self.draft = draft
        self.batch_state = batch_state
        self.batch_capacity = batch_capacity
        self.round = PackedDraftRound(
            draft, batch_state, batch_capacity=batch_capacity, backend=backend, compile_graph=False)
        self.backend = backend
        self.compile_graph = compile_graph
        self.compiled = None

    def _forward(
        self,
        target_ids,
        target_states,
        positions,
        request_slots,
        control,
        proposals,
        active,
        binding_sidecar=None,
    ):
        _, output, committed, output_count, _, draft_enabled, status = verify_control_from_target_batch(
            target_ids, proposals, control, active, binding_sidecar)
        valid_context = active & (status == 0)
        context_done = insert_context_batch(
            self.draft,
            self.batch_state,
            target_states,
            positions,
            request_slots,
            committed,
            valid_context,
        )
        draft_active = draft_enabled & valid_context
        anchors, draft_positions = draft_inputs_after_commit(
            positions, output, output_count, committed, draft_active)
        proposal_ids, _confidence = self.round(
            anchors.reshape(self.batch_capacity, 1),
            draft_positions,
            request_slots,
            draft_active,
            context_done=context_done,
        )
        record = pack_record_batch(
            control, committed, output_count, output, proposal_ids, draft_active, status, binding_sidecar)
        return record, committed, output, output_count, proposal_ids, status

    def forward(
        self,
        target_ids,
        target_states,
        positions,
        request_slots,
        control,
        proposals,
        active,
        binding_sidecar=None,
    ):
        if target_ids.shape != (self.batch_capacity, TARGET_QUERY_BUCKET):
            raise ValueError(
                f"PackedDraftTransaction requires fixed target ids [{self.batch_capacity},{TARGET_QUERY_BUCKET}]")
        if target_states.shape[:2] != (self.batch_capacity, TARGET_QUERY_BUCKET):
            raise ValueError("PackedDraftTransaction target states must be fixed [N,6,D]")
        if positions.shape != (self.batch_capacity, TARGET_QUERY_BUCKET):
            raise ValueError(
                f"PackedDraftTransaction requires fixed positions [{self.batch_capacity},{TARGET_QUERY_BUCKET}]")
        if request_slots.shape != (self.batch_capacity, ) or active.shape != (self.batch_capacity, ):
            raise ValueError(f"PackedDraftTransaction requires fixed slots/activity [{self.batch_capacity}]")
        if control.shape != (self.batch_capacity,
                             7) or proposals.shape != (self.batch_capacity, MAX_DRAFT_TOKENS):
            raise ValueError("PackedDraftTransaction requires fixed verify control/proposals")
        if binding_sidecar is None:
            if target_ids.device.type == "hpu":
                raise RuntimeError("compiled HPU transaction requires an explicit I32 binding sidecar")
        elif (not isinstance(binding_sidecar, torch.Tensor) or binding_sidecar.device != target_ids.device
              or binding_sidecar.dtype != torch.int32
              or binding_sidecar.shape != (self.batch_capacity, BINDING_SIDECAR_SIZE)):
            raise ValueError(
                "PackedDraftTransaction requires int32 [N,2] [low, high] binding sidecar on the input device")
        if self.compile_graph and target_ids.device.type == "hpu":
            if self.compiled is None:
                self.compiled = torch.compile(
                    self._forward, backend=self.backend, fullgraph=True, dynamic=False)
            return self.compiled(
                target_ids, target_states, positions, request_slots, control, proposals, active, binding_sidecar)
        return self._forward(
            target_ids, target_states, positions, request_slots, control, proposals, active, binding_sidecar)

    def prepare(
        self,
        target_ids,
        target_states,
        positions,
        request_slots,
        control,
        proposals,
        active,
        binding_sidecar=None,
    ):
        """Compile the fixed transaction before serving and restore draft state."""
        if target_ids.device.type != "hpu":
            return
        snapshots = tuple(ring.clone() for ring in self.batch_state.rings)
        try:
            for _ in range(2):
                self(
                    target_ids,
                    target_states,
                    positions,
                    request_slots,
                    control,
                    proposals,
                    active,
                    binding_sidecar,
                )
                torch.hpu.synchronize()
            if self.compiled is None:
                raise RuntimeError("PackedDraftTransaction did not build its fixed compiled graph")
        finally:
            for destination, source in zip(self.batch_state.rings, snapshots, strict=True):
                destination.copy_(source)


class PackedDraftBackbone(torch.nn.Module):
    """Original three MTP layers over packed Bx5 queries, without B request loops."""

    def __init__(self, draft, batch_state: DraftBatchState):
        super().__init__()
        if getattr(draft.layers[0], "draft_mhc", False):
            raise ValueError("Packed baseline draft does not enable the optional draft mHC path")
        self.draft = draft
        self.batch_state = batch_state
        self.batch_capacity = batch_state.capacity

    def _validate(self, anchors, positions, slots, active, context_done):
        batch_size = anchors.shape[0]
        _validate_batch_tensor(anchors, (batch_size, 1), "anchors", torch.int64)
        _validate_batch_tensor(positions, (batch_size, MAX_DRAFT_TOKENS), "positions", torch.int32)
        _validate_batch_tensor(slots, (batch_size, ), "slots", torch.int32)
        _validate_batch_tensor(active, (batch_size, ), "active", torch.bool)
        if not 1 <= batch_size <= self.batch_capacity:
            raise ValueError(f"draft batch size must be in 1..{self.batch_capacity}")
        if anchors.device != positions.device or anchors.device != slots.device or anchors.device != active.device:
            raise ValueError("packed draft inputs must share one device")
        if context_done is not None:
            if not isinstance(context_done, (tuple, list)):
                raise ValueError("context completions must be a per-layer sequence")
            if len(context_done) != len(self.draft.layers):
                raise ValueError("context completions must contain one entry per draft layer")
            for completion in context_done:
                if (not isinstance(completion, torch.Tensor) or completion.numel() == 0
                        or completion.device != anchors.device):
                    raise ValueError(
                        "context completions must be nonempty tensors on the packed input device")

    def forward(self, anchors, positions, slots, active, context_done=None):
        self._validate(anchors, positions, slots, active, context_done)
        batch_size = anchors.shape[0]
        device = anchors.device
        offsets = torch.arange(MAX_DRAFT_TOKENS, dtype=torch.int64, device=device)
        ids = torch.where(offsets.unsqueeze(0) == 0, anchors.reshape(batch_size, 1),
                          torch.full((batch_size, MAX_DRAFT_TOKENS), self.draft.noise, dtype=torch.int64,
                                     device=device))
        ids = torch.where(active.unsqueeze(1), ids, 0)
        value = self.draft.embed_input_ids(ids.reshape(-1))
        residual = value.unsqueeze(1).expand(-1, 4, -1).contiguous()
        pre = F.one_hot(torch.zeros_like(ids.reshape(-1), dtype=torch.int64), 4).float()
        image_mask = torch.zeros_like(ids.reshape(-1), dtype=torch.bool)

        for index, layer in enumerate(self.draft.layers):
            w = layer.weights
            value, new_pre, post, comb = hc_pre(
                residual, pre, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base,
                layer.eps, layer.hc_eps, layer.iterations)
            value = rms_norm(value, w.attn_norm.weight, layer.eps)
            value = draft_attention_batch(
                self.draft,
                index,
                value,
                positions,
                slots,
                active,
                self.batch_state,
                None if context_done is None else context_done[index],
            )
            residual = hc_post(value, residual, post, comb)

            value, pre, post, comb = hc_pre(
                residual, new_pre, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base,
                layer.eps, layer.hc_eps, layer.iterations)
            value = rms_norm(value, w.ffn_norm.weight, layer.eps)
            value = layer.moe(value, image_mask, decode=False)
            residual = hc_post(value, residual, post, comb)

        hidden = (residual.float() * pre.unsqueeze(-1)).sum(1).to(residual.dtype)
        logits = self.draft.compute_logits_local(hidden)
        return hidden.reshape(batch_size, MAX_DRAFT_TOKENS, -1), logits.reshape(
            batch_size, MAX_DRAFT_TOKENS, -1)


def sample_greedy_markov_batch(
    draft,
    first_token: torch.Tensor,
    hidden: torch.Tensor,
    logits: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Five sequential Markov steps, with all requests advanced in parallel."""
    if first_token.ndim != 1:
        raise ValueError("Markov anchors must be [B]")
    batch_size = first_token.shape[0]
    if hidden.ndim != 3 or hidden.shape[:2] != (batch_size, MAX_DRAFT_TOKENS):
        raise ValueError("Markov hidden must be [B, 5, H]")
    if logits.ndim != 3 or logits.shape[:2] != (batch_size, MAX_DRAFT_TOKENS):
        raise ValueError("Markov logits must be [B, 5, V_local]")
    if active.shape != (batch_size, ) or active.dtype != torch.bool:
        raise ValueError("Markov activity must be bool [B]")
    if first_token.device != hidden.device or logits.device != hidden.device or active.device != hidden.device:
        raise ValueError("Markov tensors must share one device")

    last = draft.weights.get_submodule("2")
    previous = torch.where(active, first_token.long(), 0)
    token_ids, confidence = [], []
    for index in range(MAX_DRAFT_TOKENS):
        markov = draft._embed(previous, last.markov_head.embed)
        bias = F.linear(markov.float(), last.markov_head.head.weight)
        scores = logits[:, index] + bias
        sampled = draft._global_argmax(scores).reshape(batch_size)
        token_ids.append(torch.where(active, sampled, -1))
        confidence_input = torch.cat((hidden[:, index], markov), dim=-1).float()
        confidence.append(_linear(confidence_input, last.confidence_head.proj).reshape(batch_size))
        previous = torch.where(active, sampled, 0)
    return torch.stack(token_ids, dim=1), torch.stack(confidence, dim=1)


class PackedDraftMarkovSampler(torch.nn.Module):
    """Runner-facing wrapper preserving the original draft Markov math."""

    def __init__(self, draft, *, batch_capacity: int = DEFAULT_BATCH_CAPACITY):
        super().__init__()
        validate_dspark_capacity(batch_capacity, allow_serial=True, name="Markov batch capacity")
        self.draft = draft
        self.batch_capacity = batch_capacity

    def forward(self, first_token, hidden, logits, active):
        if first_token.shape != (self.batch_capacity, ):
            raise ValueError(f"Markov batch must contain exactly {self.batch_capacity} lanes")
        return sample_greedy_markov_batch(self.draft, first_token, hidden, logits, active)


def draft_inputs_after_commit(
    target_positions: torch.Tensor,
    output: torch.Tensor,
    output_count: torch.Tensor,
    committed: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the next request-specific anchor and five draft positions."""
    batch_size = output.shape[0]
    if target_positions.shape != (batch_size, TARGET_QUERY_BUCKET) or output.shape != (
            batch_size, TARGET_QUERY_BUCKET):
        raise ValueError("draft handoff requires C6 positions and output")
    if output_count.shape != (batch_size, ) or committed.shape != (batch_size, ) or active.shape != (batch_size, ):
        raise ValueError("draft handoff counts must be [B]")
    query_index = (output_count - 1).clamp(0, TARGET_QUERY_BUCKET - 1).unsqueeze(1)
    anchors = output.gather(1, query_index).squeeze(1).clamp_min(0)
    anchors = torch.where(active, anchors, 0).to(torch.int64)
    start = target_positions[:, 0].to(torch.int32) + committed.to(torch.int32)
    positions = start.unsqueeze(1) + torch.arange(MAX_DRAFT_TOKENS, dtype=torch.int32,
                                                    device=target_positions.device).unsqueeze(0)
    positions = torch.where(active.unsqueeze(1), positions, -1)
    return anchors, positions


def _host_metadata_values(metadata):
    """Read immutable CPU plan metadata once before any device upload."""
    fields = {}
    for name in ("num_draft_tokens", "draft_token_ids", "request_slots", "slot_generations",
                 "target_start_positions", "remaining_output_budget", "target_query_start_loc",
                 "round_generation"):
        value = getattr(metadata, name)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"DSpark batch metadata {name} must be a CPU host tensor")
        fields[name] = value
    request_ids = tuple(getattr(metadata, "request_ids", ()))
    batch_size = len(request_ids)
    validate_dspark_capacity(batch_size, allow_serial=True, name="DSpark batch draft")

    host = {}
    for name in ("request_slots", "slot_generations", "target_start_positions", "remaining_output_budget"):
        value = fields[name]
        if value.ndim != 1 or value.numel() != batch_size or value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"DSpark batch metadata {name} must be an integer [{batch_size}] tensor")
        values = tuple(value.tolist())
        if any(type(item) is not int for item in values):
            raise ValueError(f"DSpark batch metadata {name} must contain integers")
        host[name] = values
    starts = fields["target_query_start_loc"].tolist()
    if len(starts) != batch_size + 1 or starts[0] != 0:
        raise ValueError("target_query_start_loc must start at zero and contain B + 1 entries")
    if any(type(value) is not int for value in starts):
        raise ValueError("target_query_start_loc must contain integers")
    target_counts = tuple(starts[index + 1] - starts[index] for index in range(batch_size))
    draft_counts = tuple(int(value) for value in fields["num_draft_tokens"].tolist())
    if len(draft_counts) != batch_size or any(not 0 <= value <= MAX_DRAFT_TOKENS for value in draft_counts):
        raise ValueError("num_draft_tokens must contain B values in 0..5")
    draft_ids = fields["draft_token_ids"]
    if draft_ids.ndim != 1 or draft_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("draft_token_ids must be an integer host vector")
    draft_ids = tuple(draft_ids.tolist())
    if len(draft_ids) != sum(draft_counts) or any(type(value) is not int or value < 0 for value in draft_ids):
        raise ValueError("draft_token_ids must contain nonnegative integers matching num_draft_tokens")
    host["draft_token_ids"] = draft_ids
    proposal_counts = tuple(draft_counts)
    request_active = getattr(metadata, "request_active", None)
    legacy_active = request_active is None
    if isinstance(request_active, torch.Tensor):
        legacy_active = request_active.numel() == 0
        if not legacy_active and (request_active.ndim != 1 or request_active.device.type != "cpu"
                                  or request_active.dtype != torch.bool or request_active.numel() != batch_size):
            raise ValueError(f"request_active must be an optional CPU bool [{batch_size}] tensor")
        active = () if legacy_active else tuple(request_active.tolist())
    elif isinstance(request_active, (list, tuple)):
        legacy_active = len(request_active) == 0
        active = () if legacy_active else tuple(request_active)
    elif request_active is None:
        active = ()
    else:
        raise ValueError(f"request_active must be an optional bool [{batch_size}] vector")
    if legacy_active:
        active = (True, ) * batch_size
    if len(active) != batch_size or any(type(value) is not bool for value in active):
        raise ValueError(f"request_active must contain exactly {batch_size} bools")
    for index, enabled in enumerate(active):
        if enabled:
            if not 1 <= target_counts[index] <= TARGET_QUERY_BUCKET:
                raise ValueError("active C6 draft targets must contain 1..6 target rows")
            if target_counts[index] != proposal_counts[index] + 1:
                raise ValueError("C6 draft targets must contain proposal_count + 1 rows")
        else:
            if not 0 <= target_counts[index] <= TARGET_QUERY_BUCKET:
                raise ValueError("inactive lanes must use zero or a legacy target count in 1..6")
            if proposal_counts[index] != 0 or draft_counts[index] != 0:
                raise ValueError("inactive lanes must not carry draft proposals")
            if host["request_slots"][index] != -1 or host["slot_generations"][index] != 0:
                raise ValueError("inactive lanes must use the (-1, 0) slot sentinel")
            if host["remaining_output_budget"][index] != 0 or host["target_start_positions"][index] != 0:
                raise ValueError("inactive lanes must use zero budget and target start")
    round_generation = int(fields["round_generation"].item())
    return host, request_ids, target_counts, proposal_counts, active, round_generation


def _fixed_proposals(counts: tuple[int, ...], flat: tuple[int, ...], active: tuple[bool, ...]) -> torch.Tensor:
    proposals = torch.full((len(counts), MAX_DRAFT_TOKENS), -1, dtype=torch.int64)
    offset = 0
    for index, count in enumerate(counts):
        if count:
            proposals[index, :count] = torch.tensor(flat[offset:offset + count], dtype=torch.int64)
        if not active[index]:
            proposals[index].fill_(-1)
        offset += count
    return proposals


def _default_target_ids(target, batch_size: int) -> torch.Tensor:
    token_ids = getattr(target, "token_ids", None)
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("DSpark target must expose integer token_ids before batch verification")
    token_ids = token_ids.to(dtype=torch.int64)
    if token_ids.ndim == 1 and token_ids.numel() == batch_size * TARGET_QUERY_BUCKET:
        token_ids = token_ids.reshape(batch_size, TARGET_QUERY_BUCKET)
    if token_ids.shape != (batch_size, TARGET_QUERY_BUCKET):
        raise ValueError("DSpark target token_ids must be [B,6] or fixed [B*6]")
    return token_ids


def _default_target_context(target, positions: torch.Tensor):
    payload = getattr(target, "payload", None)
    if isinstance(payload, dict):
        aux = payload.get("aux", payload.get("target_aux"))
    elif isinstance(payload, (tuple, list)) and len(payload) >= 2:
        aux = payload[1]
    else:
        aux = payload
    if not isinstance(aux, torch.Tensor):
        raise TypeError("DSpark target payload must expose target aux states for draft context")
    aux = aux.to(device=positions.device)
    if aux.ndim == 2:
        if aux.shape[0] != positions.shape[0] * TARGET_QUERY_BUCKET:
            raise ValueError("target aux rows must equal B*6")
        aux = aux.reshape(positions.shape[0], TARGET_QUERY_BUCKET, -1)
    if aux.ndim != 3 or aux.shape[:2] != positions.shape:
        raise ValueError("target aux must be [B,6,D] or fixed [B*6,D]")
    return aux


class DSparkBatchDraftStage:
    """Runner-facing packed verify + draft stage for greedy DPark PP1.

    The scheduler worker may pass ``result_factory=DSparkDraftBatch`` to retain
    its concrete dataclass.  Otherwise a duck-compatible ``BatchDraftResult`` is
    returned.  ``target_ids`` and ``target_context`` are optional adapters for
    the target module's opaque payload.
    """

    def __init__(
        self,
        draft,
        *,
        batch_state: DraftBatchState | None = None,
        verify_ring: BatchVerifyRing | None = None,
        backbone: Callable | None = None,
        markov: Callable | None = None,
        draft_round: Callable | None = None,
        transaction: Callable | None = None,
        result_factory: Callable | None = None,
        target_ids: Callable | None = None,
        target_context: Callable | None = None,
        batch_capacity: int = DEFAULT_BATCH_CAPACITY,
        max_model_len: int | None = None,
    ):
        validate_dspark_capacity(batch_capacity, allow_serial=True, name="batch capacity")
        self.draft = draft
        self.batch_capacity = batch_capacity
        self.batch_state = batch_state
        self.verify_ring = verify_ring
        self.backbone = backbone
        self.markov = markov
        self.draft_round = draft_round
        self.transaction = transaction
        self.result_factory = result_factory or BatchDraftResult
        self.target_ids = target_ids
        self.target_context = target_context
        self.max_model_len = max_model_len

    def _validate_capacity_contract(self):
        components = (
            ("batch state", self.batch_state, "capacity"),
            ("verify ring", self.verify_ring, "batch_capacity"),
            ("transaction", self.transaction, "batch_capacity"),
            ("draft round", self.draft_round, "batch_capacity"),
            ("backbone", self.backbone, "batch_capacity"),
            ("Markov sampler", self.markov, "batch_capacity"),
        )
        for name, component, attribute in components:
            observed = getattr(component, attribute, None)
            if observed is not None and observed != self.batch_capacity:
                raise ValueError(
                    f"DSpark batch draft {name} capacity {observed} does not match stage capacity "
                    f"{self.batch_capacity}")

    def _ensure_runtime(self, device):
        default_path = self.transaction is None and self.draft_round is None and self.backbone is None
        if default_path and self.batch_state is None:
            raise RuntimeError(
                "DSpark batch draft needs the runner-owned DraftBatchState; "
                "seed each slot from the authoritative B1/C6 backing ring first")
        self._validate_capacity_contract()
        if self.verify_ring is None:
            self.verify_ring = BatchVerifyRing(device, batch_capacity=self.batch_capacity)
        if default_path:
            self.transaction = PackedDraftTransaction(
                self.draft, self.batch_state, batch_capacity=self.batch_capacity)
        elif self.draft_round is None and self.backbone is None and self.markov is None:
            self.markov = PackedDraftMarkovSampler(self.draft, batch_capacity=self.batch_capacity)
        return self.batch_state, self.verify_ring, self.backbone, self.markov

    def verify_and_propose(self, metadata, target):
        host, request_ids, target_counts, proposal_counts, active_tuple, round_generation = (
            _host_metadata_values(metadata))
        batch_size = len(request_ids)
        if batch_size != self.batch_capacity:
            if self.batch_capacity == DEFAULT_BATCH_CAPACITY:
                raise ValueError("the fixed packed draft transaction requires exactly two request lanes")
            raise ValueError(
                "the fixed packed draft transaction requires exactly "
                f"{self.batch_capacity} request lanes")
        self._validate_capacity_contract()
        if self.max_model_len is None:
            raise RuntimeError("DSpark batch draft stage requires max_model_len for speculative admission")
        target_ids = (self.target_ids(target, batch_size) if self.target_ids is not None
                      else _default_target_ids(target, batch_size))
        if not isinstance(target_ids, torch.Tensor) or target_ids.shape != (batch_size, TARGET_QUERY_BUCKET):
            raise ValueError(f"DSpark target token_ids must be [{batch_size},{TARGET_QUERY_BUCKET}]")
        if target_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("DSpark target token_ids must be int32 or int64")
        device = target_ids.device
        batch_state, verify_ring, backbone, markov = self._ensure_runtime(device)
        active = torch.tensor(active_tuple, dtype=torch.bool, device=device)
        proposals = _fixed_proposals(proposal_counts, host["draft_token_ids"], active_tuple).to(device)
        control_host = batch_control_metadata(
            round_generation=round_generation,
            request_slots=host["request_slots"],
            slot_generations=host["slot_generations"],
            target_counts=target_counts,
            proposal_counts=proposal_counts,
            remaining_output_budget=host["remaining_output_budget"],
            target_start_positions=host["target_start_positions"],
            max_model_len=self.max_model_len,
            need_sample=tuple(1 if enabled else 0 for enabled in active_tuple),
            active=active_tuple,
        )
        binding_sidecar = batch_binding_sidecar(control_host).to(device)
        control = control_host.to(device)
        positions = (
            torch.tensor(host["target_start_positions"], dtype=torch.int64, device=device).unsqueeze(1)
            + torch.arange(TARGET_QUERY_BUCKET, dtype=torch.int32, device=device).unsqueeze(0)
        ).to(torch.int32)
        positions = torch.where(active.unsqueeze(1), positions, torch.zeros_like(positions))
        context_fn = self.target_context
        target_states = (context_fn(target, positions) if context_fn is not None
                         else _default_target_context(target, positions))
        if not isinstance(target_states, torch.Tensor) or target_states.shape[:2] != (
                batch_size, TARGET_QUERY_BUCKET) or target_states.device != device:
            raise ValueError(f"DSpark target context must be [{batch_size},{TARGET_QUERY_BUCKET},D]")
        slots = torch.tensor(host["request_slots"], dtype=torch.int32, device=device)

        ticket = verify_ring.acquire(
            round_generation=round_generation,
            request_slots=host["request_slots"],
            slot_generations=host["slot_generations"],
            max_committed=target_counts,
            active=active_tuple,
        )
        try:
            if self.transaction is not None:
                values = self.transaction(
                    target_ids, target_states, positions, slots, control, proposals, active, binding_sidecar)
                record = values[0] if isinstance(values, (tuple, list)) else values
                if not isinstance(record, torch.Tensor) or record.shape != (self.batch_capacity, 16):
                    raise RuntimeError(
                        f"PackedDraftTransaction did not return a [{self.batch_capacity},16] commit record")
            else:
                _, output, committed, output_count, _, draft_enabled, status = verify_control_from_target_batch(
                    target_ids, proposals, control, active, binding_sidecar)
                valid_context = active & (status == 0)
                context_done = insert_context_batch(
                    self.draft, batch_state, target_states, positions, slots, committed, valid_context)
                anchors, draft_positions = draft_inputs_after_commit(
                    positions, output, output_count, committed, draft_enabled & valid_context)
                if self.draft_round is not None:
                    proposal_ids, _confidence = _call_with_optional_context_done(
                        self.draft_round,
                        anchors.reshape(batch_size, 1),
                        draft_positions,
                        slots,
                        draft_enabled & valid_context,
                        context_done=context_done,
                        name="draft_round",
                    )
                else:
                    hidden, logits = _call_with_optional_context_done(
                        backbone,
                        anchors.reshape(batch_size, 1),
                        draft_positions,
                        slots,
                        draft_enabled & valid_context,
                        context_done=context_done,
                        name="backbone",
                    )
                    proposal_ids, _confidence = markov(
                        anchors, hidden, logits, draft_enabled & valid_context)
                record = pack_record_batch(
                    control,
                    committed,
                    output_count,
                    output,
                    proposal_ids,
                    draft_enabled & valid_context,
                    status,
                    binding_sidecar,
                )

            ticket.record.copy_(record)
            verify_ring.stage(ticket)
            commit = verify_ring.consume(ticket)

            result = self.result_factory(
                accepted=torch.tensor(commit.committed, dtype=torch.int64),
                output_token_ids=commit.output,
                draft_token_ids=commit.draft,
                completed=None,
                completion_event=ticket.completion,
            )
            verify_ring.release(ticket)
            ticket = None
            return result
        except BaseException as error:
            if ticket is not None:
                try:
                    verify_ring.abort(ticket)
                except BaseException as abort_error:
                    if hasattr(error, "add_note"):
                        error.add_note(f"DSpark batch verify abort failed: {type(abort_error).__name__}: {abort_error}")
            raise


__all__ = [
    "DEFAULT_BATCH_CAPACITY",
    "MAX_BATCH_REQUESTS",
    "decode_draft_swa_rows",
    "DRAFT_ATTENTION_WIDTH",
    "DraftAttentionRows",
    "DraftBatchState",
    "BatchDraftResult",
    "DraftContextRows",
    "DSparkBatchDraftStage",
    "PackedDraftBackbone",
    "PackedDraftMarkovSampler",
    "PackedDraftRound",
    "PackedDraftTransaction",
    "draft_attention_batch",
    "draft_attention_rows",
    "draft_context_rows",
    "draft_inputs_after_commit",
    "insert_context_batch",
    "sample_greedy_markov_batch",
    "write_state_rows_masked",
]
