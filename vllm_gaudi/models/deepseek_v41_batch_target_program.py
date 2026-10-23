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

from vllm_gaudi.models.deepseek_v41_program import PreparedInput, final_collapse_rms_norm
from vllm_gaudi.ops.deepseek_v41_batch_target import (
    TARGET_ROWS_PER_REQUEST,
    DSparkTargetBatch,
    initialize_target_batch_state,
    make_target_ticket,
    target_request_offsets,
    target_rows_for_capacity,
)


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
    ):
        if residual.shape[0] != self.rows or positions.numel() != self.rows:
            raise ValueError(f"DSpark target layer group requires exactly {self.rows} rows")
        if self.input is not None:
            residual, pre_mix = self.input(input_ids)
        image_mask = (input_ids == 129264) | (input_ids == 129265)
        selections = list(selections)
        main_ready = list(main_ready)
        index_ready = list(index_ready)
        target_states = []
        for layer, (index_source, kv_source) in zip(self.layers, self.mapping, strict=True):
            rows = engram_rows[0 if layer.layer == 1 else 1] if layer.layer in (1, 14) and engram_rows else None
            selection = selections[index_source] if index_source >= 0 else selections[0]
            main_done = main_ready[kv_source] if kv_source >= 0 else main_ready[0]
            index_done = index_ready[kv_source] if kv_source >= 0 else index_ready[0]
            result = layer.forward_batch_target(
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
            residual, pre_mix, selection, candidates, main_done, index_done, target = result
            if index_source >= 0:
                selections[index_source] = selection
                main_ready[kv_source], index_ready[kv_source] = main_done, index_done
            if target is not None:
                target_states.append(target)
        if not self.final:
            return residual, pre_mix, None, tuple(selections), candidates, tuple(main_ready), tuple(index_ready)
        value = final_collapse_rms_norm(residual, pre_mix, self.norm.weight, self.eps)
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
        self.chunks = tuple(
            torch.compile(group, backend=backend, fullgraph=True, dynamic=False) for group in self.groups)

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
    ):
        if hidden.shape[0] != self.rows or positions.numel() != self.rows:
            raise ValueError(f"DSpark target batch requires fixed {self.rows}-row inputs")
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
        recompile_limit = max(torch._dynamo.config.recompile_limit, 2 * len(self.chunks) + 1)
        with torch._dynamo.config.patch(recompile_limit=recompile_limit):
            for chunk in self.chunks:
                hidden, pre_mix, aux, selections, candidates, main_ready, index_ready = chunk(
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
        return self.compiled(
            hidden,
            pre_mix,
            batch.input_ids,
            batch.positions,
            batch.row_slots,
            batch.row_pages,
            batch.row_valid,
            engram_rows,
        )

    def stage_interface(self) -> SimpleNamespace:
        """Small introspection object for runner integration and tests."""
        return SimpleNamespace(
            request_lengths=self.request_lengths,
            rows=self.rows,
            request_offsets=target_request_offsets(self.batch_capacity),
            requests=self.batch_capacity,
        )


__all__ = [
    "CompiledTargetBatchStage",
    "DSparkTargetStage",
    "PreparedTargetBatchLayerGroup",
]
