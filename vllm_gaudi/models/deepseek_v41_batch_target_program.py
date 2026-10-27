# SPDX-License-Identifier: Apache-2.0
"""Packed target-stage program for a fixed-capacity DSpark round.

The program is deliberately separate from the ordinary request-batch program.
Its fixed tensor shape is ``6*N`` target rows, while each active lane owns one
through six valid rows.  A single call runs every decoder layer over all rows;
only the ratio-two compressor makes one fixed-shape, request-local T6 native
call per fixed lane inside a layer.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch._dynamo
from torch import nn

from vllm_gaudi.models.deepseek_v41_program import (
    PreparedInput,
    decode_bf16_identity,
    final_collapse_rms_norm,
)
from vllm_gaudi.ops.deepseek_v41_batch_target import (
    TARGET_ROWS_PER_REQUEST,
    DSparkTargetBatch,
    initialize_target_batch_state,
    make_target_ticket,
    target_request_offsets,
    target_rows_for_capacity,
)

# Kept in sync with the optional batch-MLA helper's fixed complete-tile
# buckets.  N32 is the only target layout with a complete 192-row tile tail;
# 24 tiles is equivalent to the full-row path.
_TARGET_MLA_TILE_PLAN_BUCKETS = (1, 2, 4, 8, 16, 24)
_AUTO_TARGET_MLA_TILE_PLAN = object()


def _build_target_mla_tile_plan(active_lanes, *, capacity: int, rows: int, device):
    """Build the N32 sparse tile plan once for a target round.

    ``None`` is the full-row path.  Only N32 with a strict subset of active
    lanes uses the compact path, and a plan covering all 24 tiles is equivalent
    to that full path.
    """
    if rows != 192 or len(active_lanes) == capacity:
        return None
    from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_packed_mla_tile_plan

    plan = batch_packed_mla_tile_plan(active_lanes, total_rows=rows, device=device)
    if plan.numel() == _TARGET_MLA_TILE_PLAN_BUCKETS[-1]:
        return None
    return plan


def _validated_active_lanes(active_lanes, capacity: int) -> tuple[int, ...]:
    """Validate host lane ownership without reading a device mask."""
    if active_lanes is None:
        return tuple(range(capacity))
    if not isinstance(active_lanes, tuple):
        raise TypeError("DSpark host active lanes must be a tuple")
    if not active_lanes:
        raise ValueError("DSpark target stage requires at least one active host lane")
    if any(type(lane) is not int for lane in active_lanes):
        raise TypeError("DSpark host active lanes must contain integers")
    if any(lane < 0 or lane >= capacity for lane in active_lanes):
        raise ValueError("DSpark host active lane is outside the fixed capacity")
    if tuple(sorted(set(active_lanes))) != active_lanes:
        raise ValueError("DSpark host active lanes must be unique and strictly increasing")
    return active_lanes


class _TargetAttentionCall:
    """Plain callable binding one target layer to its fixed-capacity attention.

    This deliberately is not an ``nn.Module``.  The target layer already owns
    the attention module, and registering this holder would duplicate its
    parameters and buffers in ``state_dict``.  ``torch.compile`` compiles the
    callable itself while keeping its fixed request ownership as static
    attributes.
    """

    def __init__(self, attention, *, request_lengths: tuple[int, ...], request_slots: tuple[int, ...]):
        self.attention = attention
        self.request_lengths = tuple(request_lengths)
        self.request_slots = tuple(request_slots)

    def __call__(
        self,
        value,
        positions,
        slots,
        pages,
        selected,
        candidates,
        main_done,
        index_done,
        row_valid,
        ready_outputs,
        tile_indices=None,
    ):
        return self.attention.forward_batch_target(
            value,
            positions,
            slots,
            pages,
            selected,
            candidates,
            main_done,
            index_done,
            row_valid=row_valid,
            request_lengths=self.request_lengths,
            request_slots=self.request_slots,
            ready_outputs=ready_outputs,
            defer_output_finish=True,
            tile_indices=tile_indices,
        )


class _TargetOutputFinishCall:
    """Plain callable binding one target layer's C6 output-finish sequence."""

    def __init__(self, attention):
        self.attention = attention

    def __call__(self, output, positions, active, ready_outputs):
        return self.attention.finish_output_target(output, positions, active, ready_outputs)


class _TargetAttentionFinishAdapter:
    """Finish only host-active lanes after one full packed attention call.

    This adapter is intentionally eager.  The raw attention wrapper and the
    per-lane output-finish wrapper are the compiled boundaries; the adapter
    supplies the host active-lane order without putting that tuple into either
    compiled callable.
    """

    def __init__(self, raw_attention, finish_output, active_lanes, tile_indices=None):
        self.raw_attention = raw_attention
        self.finish_output = finish_output
        self.active_lanes = tuple(active_lanes)
        self.tile_indices = tile_indices

    def __call__(
        self,
        value,
        positions,
        slots,
        pages,
        selected,
        candidates,
        main_done,
        index_done,
        row_valid,
        ready_outputs,
    ):
        raw_output, selected, candidates, main_done, index_done = self.raw_attention(
            value,
            positions,
            slots,
            pages,
            selected,
            candidates,
            main_done,
            index_done,
            row_valid,
            ready_outputs,
            tile_indices=self.tile_indices,
        )
        if raw_output.shape[0] % TARGET_ROWS_PER_REQUEST:
            raise ValueError("packed target attention returned a non-lane-aligned result")
        if not self.active_lanes:
            raise ValueError("target attention finish requires at least one active lane")
        capacity = raw_output.shape[0] // TARGET_ROWS_PER_REQUEST
        active = row_valid.to(torch.bool) & (slots >= 0)
        safe_positions = positions.clamp_min(0)
        finished = [None] * capacity
        for lane in self.active_lanes:
            start = lane * TARGET_ROWS_PER_REQUEST
            stop = start + TARGET_ROWS_PER_REQUEST
            lane_ready_outputs = tuple(ready[start:stop] for ready in ready_outputs)
            finished[lane] = self.finish_output(
                raw_output[start:stop],
                safe_positions[start:stop],
                active[start:stop],
                lane_ready_outputs,
            )
        zero = torch.zeros_like(finished[self.active_lanes[0]])
        for lane in range(capacity):
            if finished[lane] is None:
                finished[lane] = zero
        return torch.cat(finished, dim=0), selected, candidates, main_done, index_done


class PreparedTargetBatchLayerGroup(nn.Module):
    """Decoder layers sharing one fixed ``6*N`` target bucket."""

    def __init__(self,
                 stage,
                 start: int,
                 stop: int,
                 *,
                 request_lengths: tuple[int, ...],
                 request_slots: tuple[int, ...],
                 text_input: bool = False):
        super().__init__()
        if not getattr(stage, "dspark", False):
            raise ValueError("DSpark target batch requires a DSpark prepared stage")
        if len(request_lengths) != len(request_slots) or any(
                not 1 <= length <= TARGET_ROWS_PER_REQUEST for length in request_lengths):
            raise ValueError("DSpark target query lengths must provide one value in 1..6 per lane")
        self.batch_capacity = len(request_slots)
        self.rows = target_rows_for_capacity(self.batch_capacity)
        # The target graph is always the fixed physical six-row layout for
        # each lane.  Logical valid counts live in row_valid, not in the
        # compiled signature.
        self.logical_request_lengths = tuple(request_lengths)
        self.request_lengths = (TARGET_ROWS_PER_REQUEST, ) * self.batch_capacity
        self.request_slots = tuple(int(slot) for slot in request_slots)
        if (len(set(self.request_slots)) != self.batch_capacity
                or any(slot < 0 or slot >= self.batch_capacity for slot in self.request_slots)):
            raise ValueError("target request slots must be distinct nonnegative owners")
        self.layers = nn.ModuleList(list(stage.layers[start:stop]))
        self.preserve_layer_rounding = getattr(stage, "tensor_parallel_size", 2) in (4, 8)
        self.slice_target_layers = all(getattr(layer, "_target_c6_slices", False) for layer in self.layers)
        self.attention_calls = (tuple(
            _TargetAttentionCall(
                layer.attention,
                request_lengths=self.request_lengths,
                request_slots=self.request_slots,
            )
            for layer in self.layers
        ) if self.slice_target_layers else ())
        # Direct group users exercise the same callable path eagerly.  The
        # production stage replaces these with fullgraph wrappers below.
        self.output_finish_calls = (tuple(_TargetOutputFinishCall(layer.attention) for layer in self.layers)
                                    if self.slice_target_layers else ())
        self.attention_wrappers = self.attention_calls
        self.output_finish_wrappers = self.output_finish_calls
        self.attention_wrappers_compiled = False
        self.input = PreparedInput(stage.weights.embed, stage.tp_rank, stage.reduce) if text_input else None
        self.final = bool(getattr(stage, "is_last_stage", stage.pp_rank == 1)) and stop == len(stage.layers)
        self.norm = getattr(stage.weights, "norm", None) if self.final else None
        self.eps = stage.config["text_config"]["rms_norm_eps"]
        self.index_sources = tuple(int(key) for key in stage.shared.topk)
        self.kv_sources = tuple(int(key) for key in stage.shared.sources)
        self.mapping = tuple(
            (self.index_sources.index(max(source for source in self.index_sources if source <= layer.layer)),
             self.kv_sources.index(max(source for source in self.kv_sources if source <= layer.layer)))
            if layer.attention.ratio else (-1, -1)
            for layer in self.layers
        )

    def compile_attention_wrappers(self, backend: str):
        """Compile each fixed-capacity attention call as its own graph."""
        if not self.slice_target_layers:
            return self.attention_wrappers
        if self.attention_wrappers_compiled:
            return self.attention_wrappers
        self.attention_wrappers = tuple(
            torch.compile(wrapper, backend=backend, fullgraph=True, dynamic=False)
            for wrapper in self.attention_calls
        )
        self.output_finish_wrappers = tuple(
            torch.compile(wrapper, backend=backend, fullgraph=True, dynamic=False)
            for wrapper in self.output_finish_calls
        )
        self.attention_wrappers_compiled = True
        return self.attention_wrappers

    def _target_mla_tile_indices(self, active_lanes, device):
        """Auto-plan one N32-only tile plan for a standalone group call."""
        if not self.slice_target_layers:
            return None
        return _build_target_mla_tile_plan(
            active_lanes,
            capacity=self.batch_capacity,
            rows=self.rows,
            device=device,
        )

    def forward(
        self,
        residual,
        pre_mix,
        positions,
        input_ids,
        engram_rows,
        slots,
        pages,
        row_valid,
        selections,
        candidates,
        main_ready,
        index_ready,
        *,
        active_lanes=None,
        tile_indices=_AUTO_TARGET_MLA_TILE_PLAN,
    ):
        if residual.shape[0] != self.rows or positions.numel() != self.rows:
            raise ValueError(f"DSpark target layer group requires exactly {self.rows} rows")
        lane_indices = (_validated_active_lanes(active_lanes, self.batch_capacity)
                        if self.slice_target_layers else tuple(range(self.batch_capacity)))
        active_lane_set = set(lane_indices)
        if tile_indices is _AUTO_TARGET_MLA_TILE_PLAN:
            # Standalone group callers retain the convenient auto-plan path.
            # The production stage supplies either one shared tensor or an
            # explicit None, so it never re-enters this branch per group.
            tile_indices = self._target_mla_tile_indices(lane_indices, positions.device)
        if self.input is not None:
            residual, pre_mix = self.input(input_ids)
        image_mask = (input_ids == 129264) | (input_ids == 129265)
        selections = list(selections)
        main_ready = list(main_ready)
        index_ready = list(index_ready)
        target_states = []
        collapse_handoffs = (tuple({} for _ in range(self.batch_capacity))
                             if self.slice_target_layers and self.preserve_layer_rounding else None)
        for layer_index, (layer, (index_source, kv_source)) in enumerate(
                zip(self.layers, self.mapping, strict=True)):
            rows = engram_rows[0 if layer.layer == 1 else 1] if layer.layer in (1, 14) and engram_rows else None
            selection = selections[index_source] if index_source >= 0 else selections[0]
            main_done = main_ready[kv_source] if kv_source >= 0 else main_ready[0]
            index_done = index_ready[kv_source] if kv_source >= 0 else index_ready[0]
            publish_collapse = False
            if collapse_handoffs is not None and getattr(layer, "mhc_interlayer_collapse", False):
                following = self.layers[layer_index + 1] if layer_index + 1 < len(self.layers) else None
                publish_collapse = (
                    following is not None
                    and following.layer == layer.layer + 1
                    and getattr(following, "mhc_interlayer_collapse", False)
                    and not getattr(layer, "draft", False)
                    and not getattr(following, "draft", False)
                    and not hasattr(following.weights, "engram")
                )
            target_args = (
                residual,
                pre_mix,
                positions,
                image_mask,
                rows,
                slots,
                pages,
                selection,
                candidates,
                main_done,
                index_done,
                row_valid,
                self.request_lengths,
                self.request_slots,
            )
            if self.slice_target_layers:
                target_attention = _TargetAttentionFinishAdapter(
                    self.attention_wrappers[layer_index],
                    self.output_finish_wrappers[layer_index],
                    lane_indices,
                    tile_indices=tile_indices,
                )
                result = layer.forward_batch_target(
                    *target_args,
                    target_attention=target_attention,
                    collapse_handoffs=collapse_handoffs,
                    publish_collapse=publish_collapse,
                    active_lanes=lane_indices,
                )
            else:
                # Preserve compatibility for synthetic target layers that do
                # not implement the fixed C6 slice contract.
                result = layer.forward_batch_target(*target_args)
            residual, pre_mix, selection, candidates, main_done, index_done, target = result
            if index_source >= 0:
                selections[index_source] = selection
                main_ready[kv_source], index_ready[kv_source] = main_done, index_done
            if target is not None:
                target_states.append(target)
            if collapse_handoffs is not None and layer_index + 1 < len(self.layers):
                residual_lanes = []
                for lane in range(self.batch_capacity):
                    start = lane * 6
                    lane_residual = residual[start:start + 6]
                    if lane in active_lane_set and layer.layer + 1 not in collapse_handoffs[lane]:
                        lane_residual = decode_bf16_identity(lane_residual)
                    residual_lanes.append(lane_residual)
                residual = torch.cat(residual_lanes)
        if not self.final:
            return residual, pre_mix, None, tuple(selections), candidates, tuple(main_ready), tuple(index_ready)
        value_lanes = [None] * self.batch_capacity
        for lane in lane_indices:
            start = lane * 6
            value_lanes[lane] = final_collapse_rms_norm(
                residual[start:start + 6], pre_mix[start:start + 6], self.norm.weight, self.eps)
        if self.slice_target_layers and len(lane_indices) != self.batch_capacity:
            zero_value = torch.zeros_like(value_lanes[lane_indices[0]])
            for lane in range(self.batch_capacity):
                if value_lanes[lane] is None:
                    value_lanes[lane] = zero_value
        value = torch.cat(value_lanes)
        aux = torch.cat(target_states, -1) if target_states else None
        # A sentinel row may contain arbitrary tile-local arithmetic; it must
        # never escape into verify or a draft producer.
        mask = row_valid[:, None]
        value = torch.where(mask, value, torch.zeros_like(value))
        if aux is not None:
            aux = torch.where(mask, aux, torch.zeros_like(aux))
        return value, pre_mix, aux, tuple(selections), candidates, tuple(main_ready), tuple(index_ready)


class CompiledTargetBatchStage:
    """Compiled native/eager group sequence for one request-length pattern."""

    def __init__(self, stage, request_lengths: tuple[int, ...], request_slots: tuple[int, ...], *,
                 backend: str = "hpu_backend", group_size: int = 4):
        if not getattr(stage, "dspark", False):
            raise ValueError("DSpark target batch requires a DSpark prepared stage")
        if len(stage.layers) % group_size:
            raise ValueError("DSpark target stage requires complete four-layer groups")
        if len(request_lengths) != len(request_slots):
            raise ValueError("target request lengths and slots must describe the same capacity")
        self.stage = stage
        # The target graph is always the fixed physical six-row layout for
        # each lane.  Logical valid counts live in row_valid, not in the
        # compiled signature.
        self.batch_capacity = len(request_slots)
        self.rows = target_rows_for_capacity(self.batch_capacity)
        self.logical_request_lengths = tuple(request_lengths)
        self.request_lengths = (TARGET_ROWS_PER_REQUEST, ) * self.batch_capacity
        self.request_slots = tuple(int(slot) for slot in request_slots)
        self.groups = nn.ModuleList([
            PreparedTargetBatchLayerGroup(
                stage,
                start,
                start + group_size,
                request_lengths=self.request_lengths,
                request_slots=self.request_slots,
                text_input=start == 0,
            )
            for start in range(0, len(stage.layers), group_size)
        ])
        self.slice_target_layers = all(group.slice_target_layers for group in self.groups)
        if self.slice_target_layers:
            # Per-lane C6 mHC/MoE stays eager.  Only attention is a fullgraph
            # boundary so batch_packed_mla can bind into recipe-owned SRAM.
            for group in self.groups:
                group.compile_attention_wrappers(backend)
            self.chunks = tuple(self.groups)
        else:
            self.chunks = tuple(
                torch.compile(group, backend=backend, fullgraph=True, dynamic=False) for group in self.groups)

    def _target_mla_tile_indices(self, active_lanes, device):
        if not self.slice_target_layers:
            return None
        return _build_target_mla_tile_plan(
            active_lanes,
            capacity=self.batch_capacity,
            rows=self.rows,
            device=device,
        )

    def __call__(
        self,
        hidden,
        pre_mix,
        input_ids,
        positions,
        row_slots,
        row_pages,
        row_valid,
        engram_rows=(),
        active_lanes=None,
    ):
        if hidden.shape[0] != self.rows or positions.numel() != self.rows:
            raise ValueError(f"DSpark target batch requires fixed {self.rows}-row inputs")
        if self.slice_target_layers:
            lane_indices = _validated_active_lanes(active_lanes, self.batch_capacity)
            # This is the sole tile-plan H2D for the round.  Every layer in
            # every group receives the same runtime tensor (or explicit None).
            tile_indices = self._target_mla_tile_indices(lane_indices, positions.device)
        else:
            lane_indices = active_lanes
            tile_indices = None
        selections = tuple(
            torch.full((self.rows, 512), -1, dtype=torch.int32, device=positions.device)
            for _ in self.stage.shared.topk
        )
        candidates = torch.full((self.rows, 2048), -1, dtype=torch.int32, device=positions.device)
        main_ready = tuple(torch.zeros(self.rows, dtype=torch.int32, device=positions.device)
                           for _ in self.stage.shared.sources)
        index_ready = tuple(torch.zeros(self.rows, dtype=torch.int32, device=positions.device)
                            for _ in self.stage.shared.sources)
        aux = None
        if self.slice_target_layers:
            # The compiled attention wrappers lazily compile on their first
            # invocation.  They share one callable code object, so the budget
            # must cover every target layer and every optional MLA tile-plan
            # bucket.  None is the full-row variant; the fixed sparse tile
            # shapes are 1/2/4/8/16 tiles.
            variant_budget = (len(_TARGET_MLA_TILE_PLAN_BUCKETS) + 1) * (
                2 * len(self.stage.layers) + 1)
            recompile_limit = max(torch._dynamo.config.recompile_limit, variant_budget)
        else:
            recompile_limit = max(torch._dynamo.config.recompile_limit, 2 * len(self.chunks) + 1)
        config_patch = {"recompile_limit": recompile_limit}
        if self.slice_target_layers:
            # The wrappers keep ``attention.layer`` as an integer attribute of
            # an nn.Module.  Specializing it per layer creates one Dynamo
            # variant per layer and exhausts the budget during the first call.
            config_patch["allow_unspec_int_on_nn_module"] = True
        with torch._dynamo.config.patch(**config_patch):
            for chunk in self.chunks:
                chunk_args = (
                    hidden,
                    pre_mix,
                    positions,
                    input_ids,
                    engram_rows,
                    row_slots,
                    row_pages,
                    row_valid,
                    selections,
                    candidates,
                    main_ready,
                    index_ready,
                )
                if self.slice_target_layers:
                    result = chunk(*chunk_args, active_lanes=lane_indices, tile_indices=tile_indices)
                else:
                    result = chunk(*chunk_args)
                hidden, pre_mix, aux, selections, candidates, main_ready, index_ready = result
        return hidden, aux


class DSparkTargetStage(nn.Module):
    """Runner-facing target stage for one fixed-capacity target graph.

    The runner supplies one logical query length per fixed lane, then calls::

        stage(hidden_seed, pre_mix, batch, engram_rows)

    where ``batch`` is a :class:`DSparkTargetBatch`.  The returned pair is
    ``(target_hidden[6*N,H], target_aux[6*N,3H])``.
    """

    def __init__(self, program, request_lengths: tuple[int, ...], request_slots: tuple[int, ...], *,
                 backend: str = "hpu_backend"):
        super().__init__()
        self.program = program
        if len(request_lengths) != len(request_slots):
            raise ValueError("target request lengths and slots must describe the same capacity")
        self.batch_capacity = len(request_slots)
        self.rows = target_rows_for_capacity(self.batch_capacity)
        # The target graph is always the fixed physical six-row layout for
        # each lane.  Logical valid counts live in row_valid, not in the
        # compiled signature.
        self.logical_request_lengths = tuple(request_lengths)
        self.request_lengths = (TARGET_ROWS_PER_REQUEST, ) * self.batch_capacity
        self.request_slots = tuple(int(slot) for slot in request_slots)
        configure_recipes = getattr(program, "set_c6_scoped_recipe_backend", None)
        if configure_recipes is not None:
            configure_recipes(backend)
        self.state = initialize_target_batch_state(program, self.batch_capacity)
        self.compiled = CompiledTargetBatchStage(program, self.request_lengths, self.request_slots, backend=backend)

    @classmethod
    def from_batch(cls, program, batch: DSparkTargetBatch, *, backend: str = "hpu_backend"):
        return cls(program, batch.request_lengths, batch.host_request_slots, backend=backend)

    def snapshot(self, batch: DSparkTargetBatch):
        return make_target_ticket(self.program, batch)

    def forward(self, hidden, pre_mix, batch: DSparkTargetBatch, engram_rows=()):
        if batch.batch_capacity != self.batch_capacity:
            raise ValueError("target batch capacity differs from the compiled stage")
        if len(batch.request_lengths) != self.batch_capacity:
            raise ValueError("target stage request lengths differ from the compiled capacity")
        if hidden.shape[0] != self.rows or pre_mix.shape[0] != self.rows:
            raise ValueError("target stage hidden rows differ from the compiled capacity")
        host_active = batch.host_request_active
        if not isinstance(host_active, tuple) or len(host_active) != self.batch_capacity:
            raise ValueError("target stage host_request_active must cover the fixed capacity")
        if any(type(value) is not bool for value in host_active):
            raise TypeError("target stage host_request_active must contain bools")
        # This is the scheduler-owned host plan shared by every TP rank; never
        # derive lane filtering from a rank-local device mask.
        active_lanes = _validated_active_lanes(batch.host_active_lanes, self.batch_capacity)
        return self.compiled(
            hidden,
            pre_mix,
            batch.input_ids,
            batch.positions,
            batch.row_slots,
            batch.row_pages,
            batch.row_valid,
            engram_rows,
            active_lanes,
        )

    def stage_interface(self) -> SimpleNamespace:
        """Small introspection object for runner integration and tests."""
        return SimpleNamespace(
            request_lengths=self.request_lengths,
            rows=self.rows,
            request_offsets=target_request_offsets(self.batch_capacity),
            requests=self.batch_capacity,
        )

    def scoped_recipe_status(self):
        """Expose the shared C6 recipe cache state to warmup/probe callers."""
        status = getattr(self.program, "scoped_recipe_status", None)
        if status is not None:
            return status()
        cache = getattr(self.program, "_c6_scoped_recipe_cache", None)
        if cache is None:
            return {"state": "unavailable", "disabled_reason": None, "ineligible_reasons": {}}
        return cache.status()


__all__ = [
    "CompiledTargetBatchStage",
    "DSparkTargetStage",
    "PreparedTargetBatchLayerGroup",
]
