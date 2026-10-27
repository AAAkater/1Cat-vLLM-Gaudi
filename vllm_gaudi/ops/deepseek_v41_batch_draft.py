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
from vllm_gaudi.ops.deepseek_v41_batch_target import restore_stable_target_rows
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
from vllm_gaudi.ops.deepseek_v41_capacity import validate_dspark_capacity
from vllm_gaudi.ops.deepseek_v41_draft_buckets import draft_execution_buckets, select_draft_bucket
from vllm_gaudi.ops.deepseek_v41_math import hc_post, hc_pre, pack_swa, rms_norm
from vllm_gaudi.v1.spec_decode.deepseek_v41_dspark import HPUDSparkSpeculator

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

    ``forward`` routes through :class:`HPUDSparkSpeculator.propose_packed`, so
    the real packed backbone is called from the speculator's ``_run_model``
    rather than from a legacy precompute-then-cache path.  ``backbone`` and
    ``markov`` remain public for standalone diagnostics, but production
    ``forward`` does not use the Markov wrapper as a second execution path.
    """

    def __init__(
        self,
        draft,
        batch_state: DraftBatchState,
        *,
        batch_capacity: int = DEFAULT_BATCH_CAPACITY,
        backend: str = "hpu_backend",
        compile_graph: bool = True,
        backbone: Callable | None = None,
        markov: Callable | None = None,
        speculator: HPUDSparkSpeculator | None = None,
        context_writer: Callable | None = None,
        inputs_builder: Callable | None = None,
    ):
        super().__init__()
        validate_dspark_capacity(batch_capacity, allow_serial=True, name="draft round capacity")
        if batch_state.capacity < batch_capacity:
            raise ValueError("draft execution capacity must fit the batch state")
        self.batch_capacity = batch_capacity
        self.state_capacity = batch_state.capacity
        # Construction-only capacity tests use headless draft placeholders.
        # Real transactions always expose the three MTP layers and bind the
        # actual PackedDraftBackbone here.
        self.backbone = (backbone if backbone is not None else
                         PackedDraftBackbone(draft, batch_state) if hasattr(draft, "layers") else None)
        # Compatibility-only accessor used by standalone transaction tools.
        self.markov = markov or PackedDraftMarkovSampler(draft, batch_capacity=batch_capacity)
        if speculator is None and self.backbone is not None and hasattr(draft, "weights"):
            speculator = HPUDSparkSpeculator(
                draft,
                max_num_reqs=batch_capacity,
                linear=_linear,
                backbone=self.backbone,
                context_writer=context_writer or insert_context_batch,
                inputs_builder=inputs_builder or draft_inputs_after_commit,
                batch_state=batch_state,
            )
        self.speculator = speculator
        self.backend = backend
        self.compile_graph = compile_graph
        self.compiled = None

    def _forward(self, anchors, positions, slots, active, context_done=None):
        if self.speculator is None:
            raise RuntimeError("PackedDraftRound requires a bound HPUDSparkSpeculator")
        return self.speculator.propose_packed(
            anchors,
            positions,
            slots,
            active,
            context_done,
        )

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


def _validate_target_restore(target_restore, batch_capacity: int, device) -> None:
    """Check the fused restore arguments of a padded stable target snapshot.

    Every tensor spans the full state capacity, so the transaction graph key
    stays ``(N, B)`` whichever target execution bucket captured the round.
    """
    if not isinstance(target_restore, tuple) or len(target_restore) != 6:
        raise ValueError("fused target restore requires stable snapshot restore_arguments()")
    lane_map, indices, owners, destinations, saved, groups = target_restore
    entries = batch_capacity * TARGET_QUERY_BUCKET
    tensors = (lane_map, indices, owners)
    if (any(not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.device != device
            for value in tensors) or lane_map.shape != (batch_capacity, ) or indices.ndim != 2
            or indices.shape[1] != entries or owners.shape != (2, *indices.shape)):
        raise ValueError("fused target restore requires int64 full state-capacity layout tensors on the input device")
    if (not isinstance(destinations, tuple) or not isinstance(saved, tuple) or not isinstance(groups, tuple)
            or not len(destinations) == len(saved) == len(groups)):
        raise ValueError("fused target restore destinations, saved rows and groups must be matching tuples")
    for destination, values, group in zip(destinations, saved, groups, strict=True):
        if (type(group) is not int or not 0 <= group < indices.shape[0] or destination.device != device
                or values.shape != (entries, *destination.shape[1:]) or values.dtype != destination.dtype):
            raise ValueError("fused target restore saved rows must match their destination and full layout")


class PackedDraftTransaction(torch.nn.Module):
    """Unified fixed-capacity verify/speculator/commit graph.

    The transaction owns only the target verify call, the speculator's packed
    proposal call and record packing.  Accepted-context writes, post-commit
    anchors and the real backbone are orchestrated inside
    :class:`HPUDSparkSpeculator`.  The caller owns ticket acquisition and the
    single ``[N,16]`` host readback.
    """

    def __init__(
        self,
        draft,
        batch_state: DraftBatchState,
        *,
        batch_capacity: int = DEFAULT_BATCH_CAPACITY,
        execution_capacity: int | None = None,
        backend: str = "hpu_backend",
        compile_graph: bool = True,
        speculator: HPUDSparkSpeculator | None = None,
        context_writer: Callable | None = None,
        inputs_builder: Callable | None = None,
    ):
        super().__init__()
        validate_dspark_capacity(batch_capacity, allow_serial=True, name="draft transaction capacity")
        if batch_state.capacity != batch_capacity:
            raise ValueError("draft transaction capacity must match the batch state")
        self.draft = draft
        self.batch_state = batch_state
        self.batch_capacity = batch_capacity
        self.execution_capacity = batch_capacity if execution_capacity is None else execution_capacity
        validate_dspark_capacity(self.execution_capacity, allow_serial=True, name="draft execution capacity")
        if self.execution_capacity > batch_capacity:
            raise ValueError("draft execution capacity must fit the state capacity")
        self.graph_key = (batch_capacity, self.execution_capacity)
        self.round = PackedDraftRound(
            draft,
            batch_state,
            batch_capacity=self.execution_capacity,
            backend=backend,
            compile_graph=False,
            speculator=speculator,
            context_writer=context_writer,
            inputs_builder=inputs_builder,
        )
        self.speculator = speculator or self.round.speculator
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
        execution_lanes=None,
        target_restore=None,
    ):
        _, output, committed, output_count, _, draft_enabled, status = verify_control_from_target_batch(
            target_ids, proposals, control, active, binding_sidecar)
        if self.speculator is None:
            raise RuntimeError("PackedDraftTransaction requires a bound HPUDSparkSpeculator")
        draft_active = draft_enabled & active & (status == 0)
        propose_kwargs = {} if execution_lanes is None else {"execution_lanes": execution_lanes}
        proposal_ids, _confidence = self.speculator.propose_after_verify(
            target_states,
            positions,
            request_slots,
            committed,
            output,
            output_count,
            draft_enabled,
            active,
            status,
            **propose_kwargs,
        )
        record = pack_record_batch(
            control, committed, output_count, output, proposal_ids, draft_active, status, binding_sidecar)
        if target_restore is None:
            return record, committed, output, output_count, proposal_ids, status
        # Fused target finish: the same ``committed`` that column 1 of the
        # record carries restores the target's accepted prefix inside this
        # graph, after the record and before any host readback (the eager
        # device finish order).  HPU writes go through the native state writer.
        restore_done = restore_stable_target_rows(
            committed, *target_restore, native_writes=committed.device.type == "hpu")
        return record, committed, output, output_count, proposal_ids, status, restore_done

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
        execution_lanes=None,
        target_restore=None,
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
        if self.execution_capacity < self.batch_capacity:
            if (not isinstance(execution_lanes, torch.Tensor)
                    or execution_lanes.shape != (self.execution_capacity, )
                    or execution_lanes.dtype != torch.int64 or execution_lanes.device != target_ids.device):
                raise ValueError("compact draft transaction requires int64 [B] execution lanes on the input device")
        elif execution_lanes is not None:
            raise ValueError("full-capacity draft transaction uses the identity lane layout")
        if target_restore is not None:
            _validate_target_restore(target_restore, self.batch_capacity, target_ids.device)
        restore_args = () if target_restore is None else (target_restore, )
        if self.compile_graph and target_ids.device.type == "hpu":
            if self.compiled is None:
                self.compiled = torch.compile(
                    self._forward, backend=self.backend, fullgraph=True, dynamic=False)
            return self.compiled(
                target_ids, target_states, positions, request_slots, control, proposals, active, binding_sidecar,
                execution_lanes, *restore_args)
        return self._forward(
            target_ids, target_states, positions, request_slots, control, proposals, active, binding_sidecar,
            execution_lanes, *restore_args)

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
        execution_lanes=None,
        target_restore=None,
    ):
        """Compile the fixed transaction before serving and restore draft state.

        A ``target_restore`` argument must come from a captured stable target
        snapshot whose owner aborts it afterwards; only draft rings are saved
        and restored here.
        """
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
                    execution_lanes,
                    target_restore=target_restore,
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
    """Run the real upstream five-step loop over the packed HPU tensors."""
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

    # The adapter is intentionally per-call here so no request tensor is kept
    # beyond this sampler invocation.  The packed transaction constructs one
    # sampler-backed adapter ahead of time for graph capture.
    speculator = HPUDSparkSpeculator(
        draft,
        device=hidden.device,
        max_num_reqs=batch_size,
        linear=_linear,
    )
    return speculator.propose_cached(first_token, hidden, logits, active)


class PackedDraftMarkovSampler(torch.nn.Module):
    """Runner-facing wrapper preserving the original draft Markov math."""

    def __init__(self, draft, *, batch_capacity: int = DEFAULT_BATCH_CAPACITY):
        super().__init__()
        validate_dspark_capacity(batch_capacity, allow_serial=True, name="Markov batch capacity")
        self.draft = draft
        self.batch_capacity = batch_capacity
        # Construction-only transaction tests use headless draft placeholders.
        # Real drafts bind eagerly so their compiled graph state stays stable.
        self.speculator = (HPUDSparkSpeculator(
            draft,
            max_num_reqs=batch_capacity,
            linear=_linear,
        ) if hasattr(draft, "weights") else None)

    def forward(self, first_token, hidden, logits, active):
        if first_token.shape != (self.batch_capacity, ):
            raise ValueError(f"Markov batch must contain exactly {self.batch_capacity} lanes")
        if self.speculator is None:
            self.speculator = HPUDSparkSpeculator(
                self.draft,
                max_num_reqs=self.batch_capacity,
                linear=_linear,
            )
        return self.speculator.propose_cached(first_token, hidden, logits, active)


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


# I64: control[7] | proposals[5] | target start per lane, then the B execution lanes.
_PACKED_I64_COLUMNS = 7 + MAX_DRAFT_TOKENS + 1
# I32: request slot | sidecar[2] per lane.  The bool vector carries the activity.
_PACKED_I32_COLUMNS = 1 + BINDING_SIDECAR_SIZE
_PACKED_INPUT_COLUMNS = _PACKED_I64_COLUMNS + _PACKED_I32_COLUMNS + 1


class DraftRoundInputStaging:
    """Fixed typed staging vectors: three small H2D copies per round, no device casts.

    ``I64[13*N + B]``: ``control[N,7] | proposals[N,5] | starts[N] |
    execution_lanes[B]`` (B is 0 for the identity layout); ``I32[3*N]``:
    ``slots[N] | sidecar[N,2]``; ``bool[N]``: ``active``.  Device tensors are views
    of persistent device copies and already carry the exact dtypes of the
    separate uploads they replace, so no device-side cast runs: on HPU each
    eager cast kernel costs more than one of these small copies.  Host vectors
    are pinned only on HPU and the copies are blocking, so the next round may
    reuse them immediately.
    """

    def __init__(self, capacity: int, lanes: int, device):
        validate_dspark_capacity(capacity, allow_serial=True, name="draft input staging capacity")
        if not 0 <= lanes <= capacity:
            raise ValueError("draft input staging lanes must fit the state capacity")
        device = torch.device(device)
        self.capacity = capacity
        self.lanes = lanes
        self.device = device
        self.size = capacity * _PACKED_INPUT_COLUMNS + lanes
        hosts = (torch.zeros(capacity * _PACKED_I64_COLUMNS + lanes, dtype=torch.int64),
                 torch.zeros(capacity * _PACKED_I32_COLUMNS, dtype=torch.int32),
                 torch.zeros(capacity, dtype=torch.bool))
        self.hosts = tuple(host.pin_memory("hpu") if device.type == "hpu" else host for host in hosts)
        self.device_buffers = tuple(torch.zeros(host.shape, dtype=host.dtype, device=device) for host in hosts)

    def _views(self, buffers):
        capacity = self.capacity
        wide, narrow, active = buffers
        proposals_start = 7 * capacity
        starts_start = proposals_start + MAX_DRAFT_TOKENS * capacity
        lanes_start = starts_start + capacity
        return (wide[:proposals_start].view(capacity, 7),
                wide[proposals_start:starts_start].view(capacity, MAX_DRAFT_TOKENS),
                wide[starts_start:lanes_start],
                narrow[:capacity],
                active,
                narrow[capacity:].view(capacity, BINDING_SIDECAR_SIZE),
                wide[lanes_start:lanes_start + self.lanes])

    def upload(self, control, proposals, starts, slots, active, sidecar, lanes=None):
        """Fill the host vectors, copy each once and return device views.

        Returns ``(control I64[N,7], proposals I64[N,5], starts I64[N],
        slots I32[N], active bool[N], sidecar I32[N,2], lanes I64[B] | None)``.
        """
        capacity = self.capacity
        if lanes is None and self.lanes or lanes is not None and len(lanes) != self.lanes:
            raise ValueError("draft input staging lanes do not match the staged execution bucket")
        if (control.shape != (capacity, 7) or proposals.shape != (capacity, MAX_DRAFT_TOKENS)
                or sidecar.shape != (capacity, BINDING_SIDECAR_SIZE)):
            raise ValueError("draft input staging requires fixed [N] control, proposal and sidecar rows")
        host = self._views(self.hosts)
        host[0].copy_(control)
        host[1].copy_(proposals)
        host[2].copy_(torch.tensor(starts, dtype=torch.int64))
        host[3].copy_(torch.tensor(slots, dtype=torch.int32))
        host[4].copy_(torch.tensor(active, dtype=torch.bool))
        host[5].copy_(sidecar)
        if self.lanes:
            host[6].copy_(torch.tensor(lanes, dtype=torch.int64))
        for device_buffer, host_buffer in zip(self.device_buffers, self.hosts, strict=True):
            device_buffer.copy_(host_buffer)
        device = self._views(self.device_buffers)
        return (*device[:6], device[6] if self.lanes else None)


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
        self._owns_transactions = transaction is None and draft_round is None and backbone is None
        self.execution_buckets = (batch_capacity, )
        self.transactions = {} if transaction is None else {batch_capacity: transaction}
        self.last_execution_plan = None
        self.result_factory = result_factory or BatchDraftResult
        self.target_ids = target_ids
        self.target_context = target_context
        self.max_model_len = max_model_len
        self._bug004_profile = None
        # Opt-in single-H2D round inputs (dsv41_dspark_draft_packed_inputs).
        self.packed_inputs = False
        self.input_staging: dict[int, DraftRoundInputStaging] = {}

    def configure_execution_buckets(self, enabled: bool, *, buckets=(1, 2, 4, 8, 16, 32, 64)):
        if type(enabled) is not bool:
            raise TypeError("dsv41_dspark_draft_buckets must be a boolean")
        selected = draft_execution_buckets(self.batch_capacity, buckets)
        if enabled and not self._owns_transactions and selected != (self.batch_capacity, ):
            raise ValueError("draft execution buckets require the default packed transaction")
        if (self.transactions and (set(self.transactions) != {self.batch_capacity}
                                   or any(getattr(value, "compiled", None) is not None
                                          for value in self.transactions.values()))):
            raise RuntimeError("draft execution buckets must be configured before warmup")
        self.execution_buckets = selected if enabled else (self.batch_capacity, )

    def configure_packed_inputs(self, enabled):
        """Upload the per-round draft inputs as one fixed staging vector (default off)."""
        if type(enabled) is not bool:
            raise TypeError("dsv41_dspark_draft_packed_inputs must be a boolean")
        if any(getattr(value, "compiled", None) is not None for value in self.transactions.values()):
            raise RuntimeError("draft packed inputs must be configured before warmup")
        self.packed_inputs = enabled
        self.input_staging = {}

    def _round_input_staging(self, lanes: int, device) -> DraftRoundInputStaging:
        staging = self.input_staging.get(lanes)
        if staging is None or staging.device != torch.device(device):
            staging = self.input_staging[lanes] = DraftRoundInputStaging(self.batch_capacity, lanes, device)
        return staging

    def _bug004_mark(self, name):
        profile = getattr(self, "_bug004_profile", None)
        if profile is not None:
            profile.mark(name)

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
            self.transactions[self.batch_capacity] = self.transaction
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
        execution_plan = select_draft_bucket(active_tuple, host["request_slots"], self.execution_buckets)
        transaction = self.transaction
        execution_lanes = None
        packed_inputs = getattr(self, "packed_inputs", False)
        staged_lanes = None
        if self._owns_transactions:
            bucket = execution_plan.execution_capacity
            if bucket not in self.transactions:
                self.transactions[bucket] = PackedDraftTransaction(
                    self.draft, batch_state, batch_capacity=self.batch_capacity, execution_capacity=bucket)
            transaction = self.transactions[bucket]
            if bucket < self.batch_capacity:
                if packed_inputs:
                    staged_lanes = tuple(execution_plan.source_lanes)
                else:
                    execution_lanes = torch.tensor(execution_plan.source_lanes, dtype=torch.int64, device=device)
        profile = getattr(self, "_bug004_profile", None)
        self._bug004_mark("draft.control_prepare")
        if not packed_inputs:
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
        if packed_inputs:
            control, proposals, starts, slots, active, binding_sidecar, execution_lanes = (
                self._round_input_staging(0 if staged_lanes is None else len(staged_lanes), device).upload(
                    control_host,
                    _fixed_proposals(proposal_counts, host["draft_token_ids"], active_tuple),
                    host["target_start_positions"],
                    host["request_slots"],
                    active_tuple,
                    batch_binding_sidecar(control_host),
                    staged_lanes,
                ))
        else:
            binding_sidecar = batch_binding_sidecar(control_host).to(device)
            control = control_host.to(device)
            starts = torch.tensor(host["target_start_positions"], dtype=torch.int64, device=device)
        positions = (
            starts.unsqueeze(1)
            + torch.arange(TARGET_QUERY_BUCKET, dtype=torch.int32, device=device).unsqueeze(0)
        ).to(torch.int32)
        positions = torch.where(active.unsqueeze(1), positions, torch.zeros_like(positions))
        context_fn = self.target_context
        target_states = (context_fn(target, positions) if context_fn is not None
                         else _default_target_context(target, positions))
        if not isinstance(target_states, torch.Tensor) or target_states.shape[:2] != (
                batch_size, TARGET_QUERY_BUCKET) or target_states.device != device:
            raise ValueError(f"DSpark target context must be [{batch_size},{TARGET_QUERY_BUCKET},D]")
        if not packed_inputs:
            slots = torch.tensor(host["request_slots"], dtype=torch.int32, device=device)
        target_payload = getattr(target, "payload", None)
        # A fused target restore runs inside the packed transaction graph.
        fused_restore = target_payload.get("fused_target_restore") if isinstance(target_payload, dict) else None
        if fused_restore is not None and transaction is None:
            raise RuntimeError("fused target restore requires the packed draft transaction")

        ticket = verify_ring.acquire(
            round_generation=round_generation,
            request_slots=host["request_slots"],
            slot_generations=host["slot_generations"],
            max_committed=target_counts,
            active=active_tuple,
        )
        try:
            self._bug004_mark("draft.verify_and_propose")
            if transaction is not None:
                transaction_args = (target_ids, target_states, positions, slots, control, proposals, active,
                                    binding_sidecar)
                transaction_kwargs = {} if execution_lanes is None else {"execution_lanes": execution_lanes}
                if fused_restore is not None:
                    transaction_kwargs["target_restore"] = fused_restore
                values = transaction(*transaction_args, **transaction_kwargs)
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

            if fused_restore is not None:
                # The graph already restored target state; only mark the host
                # payload, which keeps the same finish/abort authority.
                fused_finish = target_payload.get("fused_finish_callback")
                if not isinstance(values, (tuple, list)) or len(values) < 7 or not callable(fused_finish):
                    raise RuntimeError("fused target restore requires the transaction's in-graph restore result")
                fused_finish(target, values[1])
            else:
                device_finish = (target_payload.get("device_finish_callback")
                                 if isinstance(target_payload, dict) else None)
                if device_finish is not None:
                    if transaction is None or not isinstance(values, (tuple, list)) or len(values) < 2:
                        raise RuntimeError("device target finish requires the packed transaction's device counts")
                    device_finish(target, values[1])
            ticket.record.copy_(record)
            self._bug004_mark("draft.record_stage")
            verify_ring.stage(ticket)
            commit = verify_ring.consume(ticket)
            self._bug004_mark("draft.consume")

            result = self.result_factory(
                accepted=torch.tensor(commit.committed, dtype=torch.int64),
                output_token_ids=commit.output,
                draft_token_ids=commit.draft,
                completed=None,
                completion_event=ticket.completion,
            )
            verify_ring.release(ticket)
            ticket = None
            self.last_execution_plan = execution_plan
            self._bug004_mark("draft.publish")
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
