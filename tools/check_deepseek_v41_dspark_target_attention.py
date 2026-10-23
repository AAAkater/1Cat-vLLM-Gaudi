# SPDX-License-Identifier: Apache-2.0
"""Real-weight rank-local DSpark target attention qualification.
Loads only layers 0/2 attention weights/sidecars; compares per-active-lane serial
forward(decode=True) calls with one fixed ``6*N``-row forward_batch_target graph.
``--batch-capacity`` supports 2 and 4 and defaults to the historical B2 profile.
Both sides use hpu_backend compiled component graphs. TP collectives are
synthetic identity/replicated shims: component contract only, not TP8 service.
Masked compressor positions < 0 are no-write. Results report exact/max/mismatch;
tolerances are never relaxed.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path

import torch

MASKED_SEQUENCE = "custom_deepseek_v41_compressor_sequence_masked_bf16_gaudi2"
COMPONENT_SEARCH_LENGTH, _CACHE_PAGE_TOKENS = 1024, 128
DEFAULT_BATCH_CAPACITY = 2
SUPPORTED_BATCH_CAPACITIES = (2, 4)
TARGET_ROWS_PER_REQUEST = 6
_SWA_ROWS = 256
_HISTORY_ROWS = 8


CAPACITY_SCENARIOS = {
    2: (
        {
            "name": "c6_c6_ringwrap",
            "lengths": (6, 6),
            "starts": (252, 508),
            "slots": (0, 1),
            "active": (True, True),
            "committed": (3, 2),
        },
        {
            "name": "mixed_c1_c6",
            "lengths": (1, 6),
            "starts": (252, 508),
            "slots": (0, 1),
            "active": (True, True),
            "committed": (1, 2),
        },
        {
            "name": "mixed_c2_c5",
            "lengths": (2, 5),
            "starts": (252, 508),
            "slots": (0, 1),
            "active": (True, True),
            "committed": (2, 2),
        },
    ),
    4: (
        {
            "name": "capacity4_full_a4_lane_slots",
            "lengths": (6, 6, 1, 2),
            "starts": (252, 508, 764, 1000),
            "slots": (0, 1, 2, 3),
            "active": (True, True, True, True),
            "committed": (3, 2, 1, 0),
        },
        {
            "name": "capacity4_partial_a2_lane_slots",
            "lengths": (6, 0, 1, 0),
            "starts": (252, 0, 764, 0),
            "slots": (0, -1, 2, -1),
            "active": (True, False, True, False),
            "committed": (3, 0, 0, 0),
        },
        {
            "name": "capacity4_partial_a1_lane_slots",
            "lengths": (0, 6, 0, 0),
            "starts": (0, 508, 0, 0),
            "slots": (-1, 1, -1, -1),
            "active": (False, True, False, False),
            "committed": (0, 3, 0, 0),
        },
    ),
}


def scenarios_for_capacity(capacity):
    """Return the fixed-capacity scenario matrix without inferring another N."""
    if capacity not in SUPPORTED_BATCH_CAPACITIES:
        raise ValueError(f"unsupported target attention capacity: {capacity}")
    scenarios = CAPACITY_SCENARIOS[capacity]
    for scenario in scenarios:
        lengths = tuple(scenario["lengths"])
        starts = tuple(scenario["starts"])
        slots = tuple(scenario["slots"])
        active = tuple(scenario["active"])
        committed = tuple(scenario["committed"])
        if not all(len(values) == capacity for values in (lengths, starts, slots, active, committed)):
            raise ValueError("scenario fields must describe exactly one capacity")
        if not any(active):
            raise ValueError("each target attention scenario needs an active lane")
        for lane in range(capacity):
            if active[lane]:
                if not 1 <= lengths[lane] <= TARGET_ROWS_PER_REQUEST:
                    raise ValueError("active target query lengths must be in 1..6")
                if slots[lane] != lane or starts[lane] < 0:
                    raise ValueError("active target lanes require lane slots and nonnegative starts")
            elif lengths[lane] != 0 or starts[lane] != 0 or slots[lane] != -1:
                raise ValueError("inactive target lanes require zero length/start and slot -1")
            if not 0 <= committed[lane] <= TARGET_ROWS_PER_REQUEST:
                raise ValueError("committed lengths must be in 0..6")
            if not active[lane] and committed[lane] != 0:
                raise ValueError("inactive target lanes must commit zero rows")
    return scenarios


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path, help="Prepared TP8 rank0 shard directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence path")
    parser.add_argument("--native-dir", type=Path, help="Explicit V4.1 native library directory")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-size", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument(
        "--batch-capacity",
        type=int,
        choices=SUPPORTED_BATCH_CAPACITIES,
        default=DEFAULT_BATCH_CAPACITY,
        help="Fixed target request capacity N (default: historical B2)",
    )
    parser.add_argument("--layer-ratio0", type=int, default=0)
    parser.add_argument("--layer-ratio2", type=int, default=2)
    return parser.parse_args(argv)


def make_metadata(capacity, lengths, starts, *, active=None, slots=None, pages=None):
    """Build explicit-activity target metadata for exactly one fixed capacity."""
    from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata

    if capacity not in SUPPORTED_BATCH_CAPACITIES:
        raise ValueError(f"unsupported target attention capacity: {capacity}")
    lengths = tuple(int(value) for value in lengths)
    starts = tuple(int(value) for value in starts)
    active = (True, ) * capacity if active is None else tuple(bool(value) for value in active)
    slots = (tuple(range(capacity)) if slots is None else tuple(int(value) for value in slots))
    if not all(len(values) == capacity for values in (lengths, starts, active, slots)):
        raise ValueError("target metadata fields must describe exactly one capacity")
    if not any(active):
        raise ValueError("target metadata requires at least one active lane")
    for lane, (is_active, length, start, slot) in enumerate(
            zip(active, lengths, starts, slots, strict=True)):
        if is_active:
            if not 1 <= length <= TARGET_ROWS_PER_REQUEST or start < 0 or slot != lane:
                raise ValueError(f"active lane {lane} has invalid length/start/slot")
        elif length != 0 or start != 0 or slot != -1:
            raise ValueError(f"inactive lane {lane} must use length=0, start=0 and slot=-1")
    if pages is None:
        pages = torch.arange(1, capacity * 8 + 1, dtype=torch.int32).reshape(capacity, 8)
    pages = torch.as_tensor(pages, dtype=torch.int32)
    if pages.shape != (capacity, 8) or bool((pages <= 0).any()):
        raise ValueError("target pages must have shape [capacity, 8] with positive block ids")

    metadata = DSparkBatchMetadata.build(
        requests=tuple(f"target-attention-lane-{lane}" if enabled else None
                       for lane, enabled in enumerate(active)),
        active=active,
        target_query_lengths=lengths,
        draft_token_ids=[
            [lane * 17 + token for token in range(length - 1)] if enabled else []
            for lane, (enabled, length) in enumerate(zip(active, lengths, strict=True))
        ],
        request_slots=slots,
        slot_generations=[lane + 3 if enabled else 0 for lane, enabled in enumerate(active)],
        remaining_output_budget=[64 if enabled else 0 for enabled in active],
        target_start_positions=starts,
        round_generation=7,
        device="cpu",
    )
    source_rows = []
    for lane, (enabled, length, start) in enumerate(zip(active, lengths, starts, strict=True)):
        if enabled:
            first = start % 97 + lane * 17
            source_rows.extend(range(first, first + length))
    packed = metadata.pack_target_queries(torch.tensor(source_rows, dtype=torch.int32))
    return metadata, packed.queries, pages


def make_target_batch(capacity, lengths, starts, *, active=None, slots=None, pages=None, device="hpu"):
    from vllm_gaudi.ops.deepseek_v41_batch_target import DSparkTargetBatch

    metadata, queries, page_tensor = make_metadata(
        capacity, lengths, starts, active=active, slots=slots, pages=pages)
    return DSparkTargetBatch.from_metadata(
        metadata, queries, page_tensor, batch_capacity=capacity, device=device)


def compare_tensors(name, reference, actual):
    if reference.shape != actual.shape:
        return {"name": name, "shape_reference": list(reference.shape), "shape_actual": list(actual.shape),
                "exact": False, "mismatch": None, "max_abs": None}
    delta = (reference.float() - actual.float()).abs()
    return {"name": name, "shape": list(reference.shape), "exact": bool(torch.equal(reference, actual)),
            "mismatch": int((delta != 0).sum().item()),
            "max_abs": float(delta.max().item()) if delta.numel() else 0.0}


def _state_buffers(attention, shared, ratio):
    state = attention.batch_state
    buffers = [("swa", state.swa)]
    if ratio:
        buffers.extend((("kv_history", state.kv_history), ("score_history", state.score_history)))
        cache = shared.sources[str(attention.layer)]
        buffers.extend((("main", cache.main), ("index", cache.index)))
    return buffers


def _clone_state(buffers):
    return {name: value.clone() for name, value in buffers}


def _restore_state(buffers, saved):
    for name, value in buffers:
        value.copy_(saved[name])


def _compare_state(buffers, saved, ratio):
    checks = []
    for name, value in buffers:
        # Incomplete groups use the unowned page-0 cache as scratch in the
        # serial oracle. The packed target intentionally does not write that
        # shared scratch; compare request-owned cache rows only.
        current = value[_CACHE_PAGE_TOKENS // ratio:] if ratio and name in ("main", "index") else value
        reference = saved[name][_CACHE_PAGE_TOKENS // ratio:] if ratio and name in ("main", "index") else saved[name]
        checks.append(compare_tensors(name, reference, current))
    return checks


def _make_program(attention, shared):
    program = torch.nn.Module()
    layer = torch.nn.Module()
    layer.attention, program.layers, program.shared = attention, torch.nn.ModuleList([layer]), shared
    return program


def _build_attention(args, layer, weights, config, shared, linear, reduce, gather, *, capacity, woa_layers,
                     woa_output_roundtrip):
    from vllm_gaudi.ops.deepseek_v41_batch_state import BatchLayerState
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention

    ratio = int(config["compress_ratios"][layer])
    attention = PagedCSA2Attention(
        weights.layers.get_submodule(str(layer)).attn, config, layer, shared, linear, reduce, gather,
        "hpu", tensor_parallel_size=args.tp_size,
    )
    attention.woa_fp8 = layer in woa_layers
    attention.woa_output_roundtrip = bool(attention.woa_fp8 and woa_output_roundtrip)
    attention.set_search_length(COMPONENT_SEARCH_LENGTH)
    attention.prepare_qkv_input_weight()
    attention.prepare_compressor_input_weight()
    attention.prepare_index_gain_weight()
    attention.prepare_output_weight()
    attention.prepare_local_index_query_weights(release_local=False)
    attention.batch_state = BatchLayerState(capacity, "hpu", compressor=ratio == 2)
    if not ratio:
        attention.register_buffer("batch_main_unused", torch.zeros(1, 288, dtype=torch.uint8, device="hpu"), False)
    return attention, ratio


def _make_inputs(config, *, capacity, seed, device="hpu"):
    torch.manual_seed(seed)
    width = int(config["hidden_size"])
    rows = TARGET_ROWS_PER_REQUEST * capacity
    value = torch.randn((rows, width), dtype=torch.bfloat16, device=device)
    selected = torch.full((rows, 512), -1, dtype=torch.int32, device=device)
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
    done = torch.zeros(rows, dtype=torch.int32, device=device)
    return value, selected, candidates, done


def _make_reference(attention, config, shared, capacity, ratio):
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention

    state = attention.batch_state
    refs = []
    for slot in range(capacity):
        module = PagedCSA2Attention(attention.weights, config, attention.layer, shared, attention.linear,
                                    attention.reduce, attention.gather, "hpu",
                                    tensor_parallel_size=attention.tensor_parallel_size)
        module.woa_fp8, module.woa_output_roundtrip = attention.woa_fp8, attention.woa_output_roundtrip
        module.set_search_length(COMPONENT_SEARCH_LENGTH)
        module.prepare_index_gain_weight()
        if attention.tp4_local_index_queries:
            module.prepare_local_index_query_weights(release_local=False)
        module.swa = state.swa[slot * _SWA_ROWS:(slot + 1) * _SWA_ROWS]
        if ratio:
            module.kv_history = state.kv_history[slot * _HISTORY_ROWS:(slot + 1) * _HISTORY_ROWS]
            module.score_history = state.score_history[slot * _HISTORY_ROWS:(slot + 1) * _HISTORY_ROWS]
        module._fused_qkv_weight, module._fused_qkv_quantized = (
            attention._fused_qkv_weight, attention._fused_qkv_quantized)
        module._fused_compressor_weight, module._fused_compressor_kv_width = (
            attention._fused_compressor_weight, attention._fused_compressor_kv_width)
        refs.append(module)
    return refs


def _run_serial_reference(refs, counters, batch, value, ratio):
    capacity = len(refs)
    rows = TARGET_ROWS_PER_REQUEST * capacity
    padded = torch.zeros((rows, value.shape[-1]), dtype=value.dtype, device=value.device)
    padded_selected = (torch.full((rows, 512), -1, dtype=torch.int32, device=value.device)
                       if ratio else None)
    for lane, (module, counter, length) in enumerate(zip(refs, counters, batch.request_lengths, strict=True)):
        if length == 0:
            continue
        start = lane * TARGET_ROWS_PER_REQUEST
        module.shared.block_table = batch.pages[lane]
        output = counter(value[start:start + length], batch.positions[start:start + length])
        padded[start:start + length].copy_(output)
        if ratio:
            padded_selected[start:start + length].copy_(module.selection.indices[:length].clone())
    return padded, padded_selected


class _SerialTargetEntry(torch.nn.Module):

    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, value, positions):
        return self.attention(value, positions, decode=True)


class _PackedTargetEntry(torch.nn.Module):

    def __init__(self, attention, capacity):
        super().__init__()
        self.attention = attention
        self.request_lengths = (TARGET_ROWS_PER_REQUEST, ) * capacity
        self.request_slots = tuple(range(capacity))

    def forward(self, value, positions, slots, pages, selected, candidates, main_done, index_done, row_valid):
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
        )


def _lookup_backend(name):
    from torch._dynamo.backends.registry import lookup_backend

    return lookup_backend(name)


class _CountingBackend:
    """Count actual hpu_backend callback invocations, including recompiles."""

    def __init__(self, backend):
        self.backend = backend
        self.compiles = 0

    def __call__(self, graph, inputs):
        self.compiles += 1
        return self.backend(graph, inputs)


@lru_cache(maxsize=None)
def _shared_counting_backend(name):
    # Independent oracle entries share backend identity so the counter itself
    # does not create Dynamo BACKEND_MATCH recompilations across layers.
    return _CountingBackend(_lookup_backend(name))


def _compile(module, backend):
    return torch.compile(module, backend=backend, fullgraph=True, dynamic=False)


class _GraphCallCounter:
    """Separate compiled-entry calls/replays from real backend compilations."""

    def __init__(self, module, *, backend=None, backend_name="hpu_backend"):
        self.module = module
        if backend is None:
            self.backend = _shared_counting_backend(backend_name)
        else:
            self.backend = backend if isinstance(backend, _CountingBackend) else _CountingBackend(backend)
        self.entry = _compile(module, self.backend)
        self.successful_calls = 0
        self.failed_calls = 0
        self.backend_compiles = 0

    def __call__(self, *args, **kwargs):
        before = self.backend.compiles
        try:
            result = self.entry(*args, **kwargs)
        except Exception:
            self.failed_calls += 1
            raise
        finally:
            # Entries are invoked sequentially, so the shared backend delta
            # belongs only to the entry currently being called.
            self.backend_compiles += self.backend.compiles - before
        self.successful_calls += 1
        return result

    def snapshot(self):
        backend_compiles = self.backend_compiles
        return {
            "entry_calls": self.successful_calls,
            "failed_entry_calls": self.failed_calls,
            "backend_compiles": backend_compiles,
            "entry_replays": max(0, self.successful_calls - backend_compiles),
        }


def _graph_usage(counters, before=None):
    snapshots = [counter.snapshot() for counter in counters]
    if before is not None:
        if len(before) != len(snapshots):
            raise ValueError("graph snapshots do not match their counters")
        snapshots = [{
            key: after[key] - prior[key]
            for key in ("entry_calls", "failed_entry_calls", "backend_compiles", "entry_replays")
        } for prior, after in zip(before, snapshots, strict=True)]
    return {
        "entries": len(counters),
        "entries_invoked": sum(item["entry_calls"] > 0 for item in snapshots),
        "entry_calls": sum(item["entry_calls"] for item in snapshots),
        "failed_entry_calls": sum(item["failed_entry_calls"] for item in snapshots),
        "backend_compiles": sum(item["backend_compiles"] for item in snapshots),
        "entry_replays": sum(item["entry_replays"] for item in snapshots),
        "per_entry": snapshots,
        "counter_semantics": (
            "backend_compiles counts actual hpu_backend callback invocations; "
            "entry_replays is successful compiled-entry calls minus backend compilations"
        ),
    }


def _packed_graph_gate(usage, *, expected_calls, expected_compiles):
    """Require actual callback-derived call/compile counts, not elapsed-call guesses."""
    exact = (
        usage["entry_calls"] == expected_calls
        and usage["backend_compiles"] == expected_compiles
        and usage["failed_entry_calls"] == 0
        and usage["entry_replays"] == expected_calls - expected_compiles
    )
    return {
        "expected_calls": expected_calls,
        "expected_backend_compiles": expected_compiles,
        "failed_entry_calls_zero": usage["failed_entry_calls"] == 0,
        "entry_calls_exact": usage["entry_calls"] == expected_calls,
        "backend_compiles_exact": usage["backend_compiles"] == expected_compiles,
        "entry_replays_exact": usage["entry_replays"] == expected_calls - expected_compiles,
        "exact": exact,
    }


def _buffer_names_by_identity(buffers):
    names = {}
    for name, value in buffers:
        identity = id(value)
        if identity in names:
            raise ValueError(f"duplicate buffer identity for {names[identity]!r} and {name!r}")
        names[identity] = name
    return names


def _restore_expected(snapshot, buffers, baseline, target, committed):
    """Model snapshot restore using the live destination identities."""
    names = _buffer_names_by_identity(buffers)
    expected = {name: value.clone() for name, value in target.items()}
    for row in snapshot.rows:
        name = names.get(id(row.destination))
        if name is None:
            raise ValueError("snapshot destination is not one of the tracked buffers")
        if name not in baseline or name not in expected:
            raise ValueError(f"snapshot destination {name!r} is missing from baseline or target state")
        accepted = [any(row_index < committed[request] for request, row_index in owners) for owners in row.owners]
        for index, is_accepted in enumerate(accepted):
            if not is_accepted:
                expected[name][int(row.indices[index])] = baseline[name][int(row.indices[index])]
    return expected


def _inactive_rows(name, lane, batch, ratio):
    if name == "swa":
        return torch.arange(lane * _SWA_ROWS, (lane + 1) * _SWA_ROWS, dtype=torch.int64)
    if name in ("kv_history", "score_history"):
        return torch.arange(lane * _HISTORY_ROWS, (lane + 1) * _HISTORY_ROWS, dtype=torch.int64)
    if ratio and name in ("main", "index"):
        width = _CACHE_PAGE_TOKENS // ratio
        rows = []
        for page in batch.host_pages[lane]:
            rows.extend(range(page * width, (page + 1) * width))
        return torch.tensor(rows, dtype=torch.int64)
    raise ValueError(f"unsupported target state buffer {name!r}")


def _inactive_state_checks(buffers, expected, batch, ratio, *, prefix):
    checks = []
    for lane, active in enumerate(batch.host_request_active):
        if active:
            continue
        for name, value in buffers:
            rows = _inactive_rows(name, lane, batch, ratio)
            if rows.numel() == 0 or int(rows.max()) >= value.shape[0]:
                raise ValueError(f"inactive lane {lane} {name} rows exceed state capacity")
            checks.append(compare_tensors(
                f"{prefix}_{name}_lane{lane}",
                expected[name].index_select(0, rows.to(expected[name].device)),
                value.index_select(0, rows.to(value.device)),
            ))
    return checks


def _inactive_output_checks(output, batch, *, prefix):
    checks = []
    for lane, active in enumerate(batch.host_request_active):
        if active:
            continue
        start = lane * TARGET_ROWS_PER_REQUEST
        block = output[start:start + TARGET_ROWS_PER_REQUEST]
        checks.append(compare_tensors(f"{prefix}_lane{lane}", torch.zeros_like(block), block))
    return checks


def _make_alternate_inactive_state(buffers, baseline, batch, ratio, *, seed):
    from vllm_gaudi.ops.deepseek_v41_math import pack_swa

    alternate = {name: value.clone() for name, value in baseline.items()}
    for lane, active in enumerate(batch.host_request_active):
        if active:
            continue
        generator = torch.Generator().manual_seed(seed + lane * 1009)
        swa = torch.randn((_SWA_ROWS, 512), generator=generator, dtype=torch.bfloat16)
        alternate["swa"][lane * _SWA_ROWS:(lane + 1) * _SWA_ROWS].copy_(pack_swa(swa))
        if ratio:
            alternate["kv_history"][lane * _HISTORY_ROWS:(lane + 1) * _HISTORY_ROWS].copy_(
                torch.randn((_HISTORY_ROWS, 512), generator=generator, dtype=torch.float32))
            alternate["score_history"][lane * _HISTORY_ROWS:(lane + 1) * _HISTORY_ROWS].copy_(
                torch.randn((_HISTORY_ROWS, 512), generator=generator, dtype=torch.float32))
            for name in ("main", "index"):
                rows = _inactive_rows(name, lane, batch, ratio)
                width = buffers[[item[0] for item in buffers].index(name)][1].shape[1]
                alternate[name].index_copy_(
                    0,
                    rows.to(alternate[name].device),
                    torch.randint(
                        0, 256, (rows.numel(), width), generator=generator, dtype=torch.uint8
                    ).to(alternate[name].device),
                )
    return alternate


def _qualify_layer(args, layer, weights, config, shared, linear, reduce, gather, *, capacity, woa_layers,
                   woa_output_roundtrip):
    from vllm_gaudi.ops.deepseek_v41_batch_target import make_target_ticket
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa

    attention, ratio = _build_attention(
        args, layer, weights, config, shared, linear, reduce, gather,
        capacity=capacity, woa_layers=woa_layers, woa_output_roundtrip=woa_output_roundtrip,
    )
    buffers = _state_buffers(attention, shared, ratio)
    state = attention.batch_state
    device = state.swa.device
    torch.manual_seed(7000 + layer)
    state.swa.copy_(pack_swa(torch.randn((state.swa.shape[0], 512), dtype=torch.bfloat16, device=device)))
    if ratio:
        state.kv_history.copy_(torch.randn(state.kv_history.shape, dtype=torch.float32, device=device))
        state.score_history.copy_(torch.randn(state.score_history.shape, dtype=torch.float32, device=device))
        cache = shared.sources[str(layer)]
        cache.main.copy_(pack_fp4(
            torch.randn((cache.main.shape[0], 512), dtype=torch.bfloat16, device=device), 16))
        cache.index.copy_(pack_fp4(
            torch.randn((cache.index.shape[0], 128), dtype=torch.bfloat16, device=device), 32))
    baseline = _clone_state(buffers)
    program = _make_program(attention, shared)
    scenarios = scenarios_for_capacity(capacity)
    backend = _shared_counting_backend("hpu_backend")
    refs = _make_reference(attention, config, shared, capacity, ratio)
    serial_counters = [
        _GraphCallCounter(_SerialTargetEntry(module), backend=backend) for module in refs
    ]
    packed_counter = _GraphCallCounter(_PackedTargetEntry(attention, capacity), backend=backend)
    serial_initial = [counter.snapshot() for counter in serial_counters]
    packed_initial = packed_counter.snapshot()
    checks = []
    for scenario_index, scenario in enumerate(scenarios):
        lengths = tuple(scenario["lengths"])
        starts = tuple(scenario["starts"])
        slots = tuple(scenario["slots"])
        active = tuple(scenario["active"])
        committed = tuple(scenario["committed"])
        batch = make_target_batch(
            capacity, lengths, starts, active=active, slots=slots, device=device)
        value, selected, candidates, done = _make_inputs(
            config, capacity=capacity, seed=9100 + layer + sum(lengths), device=device)
        serial_before = [counter.snapshot() for counter in serial_counters]
        packed_before = packed_counter.snapshot()
        _restore_state(buffers, baseline)
        reference, reference_selected = _run_serial_reference(refs, serial_counters, batch, value, ratio)
        reference_state = _clone_state(buffers)
        _restore_state(buffers, baseline)
        torch.hpu.synchronize()
        ticket = make_target_ticket(program, batch)
        target, selected_out, blocks, _, _ = packed_counter(
            value, batch.positions, batch.row_slots, batch.row_pages, selected, candidates, done, done,
            batch.row_valid)
        torch.hpu.synchronize()
        target_state = _clone_state(buffers)
        output_check = compare_tensors("output", reference, target)
        selection_check = (compare_tensors("selection", reference_selected, selected_out)
                           if ratio else None)
        state_checks = _compare_state(buffers, reference_state, ratio)
        inactive_lanes = [lane for lane, enabled in enumerate(active) if not enabled]
        inactive_outputs = _inactive_output_checks(target, batch, prefix="inactive_output")
        primary_no_write = _inactive_state_checks(
            buffers, baseline, batch, ratio, prefix="primary_no_write")
        expected = _restore_expected(ticket.snapshot, buffers, baseline, target_state, committed)
        ticket.restore(committed)
        restored = _clone_state(buffers)
        restore_checks = [compare_tensors(name, expected[name], value) for name, value in buffers]
        inactive_invariance = None
        if inactive_lanes:
            _restore_state(buffers, baseline)
            alternate = _make_alternate_inactive_state(
                buffers, baseline, batch, ratio, seed=12000 + layer * 100 + scenario_index)
            _restore_state(buffers, alternate)
            alternate_before = packed_counter.snapshot()
            alternate_target, _, _, _, _ = packed_counter(
                value, batch.positions, batch.row_slots, batch.row_pages, selected, candidates, done, done,
                batch.row_valid)
            torch.hpu.synchronize()
            active_output_checks = []
            for lane, enabled in enumerate(active):
                if not enabled:
                    continue
                start = lane * TARGET_ROWS_PER_REQUEST
                length = lengths[lane]
                active_output_checks.append(compare_tensors(
                    f"lane{lane}_active_output_after_inactive_state_change",
                    target[start:start + length],
                    alternate_target[start:start + length],
                ))
            alternate_outputs = _inactive_output_checks(
                alternate_target, batch, prefix="alternate_inactive_output")
            alternate_no_write = _inactive_state_checks(
                buffers, alternate, batch, ratio, prefix="alternate_no_write")
            alternate_graph = _graph_usage([packed_counter], before=[alternate_before])
            inactive_invariance = {
                "changed_inactive_lanes": inactive_lanes,
                "active_outputs": active_output_checks,
                "inactive_outputs": alternate_outputs,
                "inactive_no_write": alternate_no_write,
                "graph_runtime": alternate_graph,
                "exact": (
                    all(item["exact"] for item in active_output_checks)
                    and all(item["exact"] for item in alternate_outputs)
                    and all(item["exact"] for item in alternate_no_write)
                ),
            }
            _restore_state(buffers, baseline)
        lane_slot_contract = {
            "active_device_lane_equals_slot": all(slots[lane] == lane
                                                  for lane, enabled in enumerate(active) if enabled),
            "inactive_device_slot_is_minus_one": all(slots[lane] == -1
                                                    for lane, enabled in enumerate(active) if not enabled),
            "persistent_state_slot_equals_host_lane": tuple(batch.host_request_slots) == tuple(range(capacity)),
        }
        scenario_graph_runtime = {
            "serial": _graph_usage(serial_counters, before=serial_before),
            "packed": _graph_usage([packed_counter], before=[packed_before]),
        }
        packed_graph_gate = _packed_graph_gate(
            scenario_graph_runtime["packed"],
            expected_calls=1 + int(bool(inactive_lanes)),
            expected_compiles=1 if scenario_index == 0 else 0,
        )
        scenario_numeric_exact = (
            output_check["exact"]
            and (selection_check is None or selection_check["exact"])
            and all(item["exact"] for item in state_checks)
            and selected_out.dtype == torch.int32
            and blocks.dtype == torch.int32
            and all(item["exact"] for item in restore_checks)
            and all(item["exact"] for item in inactive_outputs)
            and all(item["exact"] for item in primary_no_write)
            and all(lane_slot_contract.values())
            and (inactive_invariance is None or inactive_invariance["exact"])
        )
        scenario_exact = scenario_numeric_exact and packed_graph_gate["exact"]
        checks.append({
            "scenario": scenario["name"],
            "batch_capacity": capacity,
            "active_requests": sum(active),
            "active_lanes": [lane for lane, enabled in enumerate(active) if enabled],
            "inactive_lanes": inactive_lanes,
            "slots": list(slots),
            "lane_slot_contract": lane_slot_contract,
            "lengths": list(lengths),
            "starts": list(starts),
            "ratio": ratio,
            "selected_int32": selected_out.dtype == torch.int32,
            "candidates_int32": blocks.dtype == torch.int32,
            "shapes": {
                "value": list(value.shape),
                "positions": list(batch.positions.shape),
                "slots": list(batch.row_slots.shape),
                "page_table": list(batch.pages.shape),
                "physical_target_rows": capacity * TARGET_ROWS_PER_REQUEST,
                "active_target_rows": sum(lengths),
                "swa_state": list(state.swa.shape),
                "kv_history": list(state.kv_history.shape) if ratio else None,
                "score_history": list(state.score_history.shape) if ratio else None,
            },
            "output": output_check,
            "selection": selection_check,
            "state": state_checks,
            "state_scope": "request-owned rows; shared incomplete-page scratch excluded",
            "snapshot_restore": restore_checks,
            "committed": list(committed),
            "snapshot_bytes": ticket.snapshot.bytes,
            "native_masked_required": ratio == 2,
            "inactive_outputs": inactive_outputs,
            "inactive_no_write": primary_no_write,
            "inactive_state_invariance": inactive_invariance,
            "graph_runtime": scenario_graph_runtime,
            "packed_graph_gate": packed_graph_gate,
            "numeric_and_state_exact": scenario_numeric_exact,
            "exact": scenario_exact,
        })
        _restore_state(buffers, baseline)
    graph_runtime = {
        "serial": _graph_usage(serial_counters, before=serial_initial),
        "packed": _graph_usage([packed_counter], before=[packed_initial]),
    }
    expected_packed_calls = 5 if capacity == 4 else 3
    packed_graph_gate = _packed_graph_gate(
        graph_runtime["packed"], expected_calls=expected_packed_calls, expected_compiles=1)
    graph_qualified = packed_graph_gate["exact"] and all(
        check["packed_graph_gate"]["exact"] for check in checks)
    qualification = {
        "numerics_and_state_exact": all(check["numeric_and_state_exact"] for check in checks),
        "packed_graph_exact": graph_qualified,
        "expected_packed_calls": expected_packed_calls,
        "expected_packed_compiles": 1,
        "later_a_or_inactive_perturbation_added_compile": any(
            check["packed_graph_gate"]["expected_backend_compiles"] == 0
            and check["packed_graph_gate"]["backend_compiles_exact"] is False
            for check in checks
        ),
        "exact": bool(checks) and all(check["exact"] for check in checks) and graph_qualified,
    }
    exact = qualification["exact"]
    return {
        "layer": layer,
        "ratio": ratio,
        "batch_capacity": capacity,
        "exact": exact,
        "graph_runtime": graph_runtime,
        "packed_graph_gate": packed_graph_gate,
        "qualification": qualification,
        "checks": checks,
    }


def main(argv=None):
    args = parse_args(argv)
    native_dir = args.native_dir.resolve() if args.native_dir else None
    if native_dir:
        os.environ["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"] = str(native_dir)
        bridge = native_dir / "tp2_fused_ar_norm_bridge.so"
        if bridge.is_file():
            os.environ["VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE"] = str(bridge)
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators, prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=args.tp_size, pipeline_parallel_size=1)
    from vllm_gaudi import envs
    if not envs.VLLM_HPU_DSV41_DSPARK:
        raise RuntimeError("set VLLM_HPU_DSV41_DSPARK=1 for the DSpark target component profile")
    load_native_operators()
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, linear, load_weight_tree
    from vllm_gaudi.ops.deepseek_v41_dense_fp8 import DenseFP8Sidecar, precision_config
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import WoaFP8Sidecar, layer_selection

    torch.set_num_threads(1)
    torch.hpu.set_device(args.rank)
    shard = PreparedV41Shard(args.prepared, 0, args.rank)
    if (shard.tensor_parallel_size, shard.pipeline_parallel_size) != (args.tp_size, 1):
        raise ValueError("prepared shard topology differs from --tp-size and PP1")
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    layers = (args.layer_ratio0, args.layer_ratio2)
    if config["compress_ratios"][layers[0]] != 0 or config["compress_ratios"][layers[1]] != 2:
        raise ValueError("--layer-ratio0/--layer-ratio2 must name ratio 0 and ratio 2 attention layers")
    specs = {
        name: spec for name, spec in shard.specs.items()
        if any(name.startswith(f"layers.{layer}.attn.") for layer in layers)
    }
    weights = _weight_tree(specs)
    dense_config = None
    # Use the same sidecar controls as the existing rank-local batch gate.
    if envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8:
        dense_config = precision_config(envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_CONFIG)
    woa_config = (layer_selection(envs.VLLM_HPU_DSV41_WO_A_FP8_CONFIG)
                  if envs.VLLM_HPU_DSV41_WO_A_FP8 else {"layers": []})
    dense_sidecar = (DenseFP8Sidecar(envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8_SIDECAR, shard)
                     if dense_config else None)
    woa_sidecar = (WoaFP8Sidecar(envs.VLLM_HPU_DSV41_WO_A_FP8_SIDECAR, shard)
                   if woa_config["layers"] else None)
    load_weight_tree(
        shard, weights, "hpu", specs, woa_sidecar=woa_sidecar, woa_layers=woa_config["layers"],
        dense_sidecar=dense_sidecar, dense_config=dense_config,
    )

    def reduce(value, **kwargs):
        return value

    def gather(value, dim):
        return torch.cat((value,) * args.tp_size, dim=dim)
    report = {
        "scope": __doc__,
        "status": "running",
        "rank": args.rank,
        "tp_size": args.tp_size,
        "batch_capacity": args.batch_capacity,
        "supported_batch_capacities": list(SUPPORTED_BATCH_CAPACITIES),
        "scenario_plan": [{
            "name": scenario["name"],
            "batch_capacity": args.batch_capacity,
            "active_requests": sum(scenario["active"]),
            "active_lanes": [lane for lane, enabled in enumerate(scenario["active"]) if enabled],
            "inactive_lanes": [lane for lane, enabled in enumerate(scenario["active"]) if not enabled],
            "slots": list(scenario["slots"]),
        } for scenario in scenarios_for_capacity(args.batch_capacity)],
        "capacity_evidence": (
            f"this report is evidence only for --batch-capacity {args.batch_capacity}; "
            "other capacities are not inferred"
        ),
        "prepared": str(args.prepared),
        "native_dir": str(native_dir) if native_dir else os.environ.get("VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"),
        "search_length": COMPONENT_SEARCH_LENGTH,
        "masked_sequence_op": MASKED_SEQUENCE,
        "masked_sequence_available": hasattr(torch.ops.custom_op, MASKED_SEQUENCE),
        "reference_path": "PagedCSA2Attention.forward(decode=True), one serial call per active lane",
        "target_path": (
            "PagedCSA2Attention.forward_batch_target, one packed "
            f"{TARGET_ROWS_PER_REQUEST * args.batch_capacity}-row graph"
        ),
        "compiled_backend": "hpu_backend",
        "compile_contract": {
            "backend": "hpu_backend",
            "fullgraph": True,
            "dynamic": False,
            "backend_compiles": "actual hpu_backend callback invocations, including recompiles",
            "entry_replays": "successful compiled-entry calls minus backend compilations",
        },
        "comparison_contract": {
            "criterion": "bitwise_equal_after_host_copy",
            "rtol": 0,
            "atol": 0,
            "tolerance_relaxed": False,
            "active_output": "serial request-local attention and packed active lane are BF16-bitwise equal",
            "inactive_output": "bitwise zero in every inactive lane",
            "inactive_state": "inactive persistent state is not written and does not affect active outputs",
            "partial_restore": "production target snapshot restores only rejected active rows",
        },
        "math_path_note": "serial request-local attention vs one fixed 6*N-row target call; TP is synthetic",
        "aux_note": "attention-only component has no aux output; SWA/history/cache/selection state is compared",
        "checks": [],
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not report["masked_sequence_available"]:
            report["status"] = "blocked"
            report["blocker"] = f"required native op {MASKED_SEQUENCE} is not registered"
            output.write_text(json.dumps(report, indent=2) + "\n")
            raise RuntimeError(report["blocker"])
        for layer in layers:
            shared = PagedCSA2SharedState(config, layer, layer + 1, "hpu", COMPONENT_SEARCH_LENGTH,
                                          tensor_parallel_size=args.tp_size)
            cache = shared.sources[str(layer)] if str(layer) in shared.sources else None
            if cache is not None:
                # Every request owns eight positive page ids; page 0 is the
                # unowned scratch page used by the serial incomplete-group path.
                cache_rows = (args.batch_capacity * 8 + 1) * (_CACHE_PAGE_TOKENS // cache.ratio)
                cache.main = torch.zeros(cache_rows, 288, dtype=torch.uint8, device="hpu")
                cache.index = torch.zeros(cache_rows, 68, dtype=torch.uint8, device="hpu")
            report["checks"].append(_qualify_layer(
                args, layer, weights, config, shared, linear, reduce, gather, capacity=args.batch_capacity,
                woa_layers=woa_config["layers"],
                woa_output_roundtrip=envs.VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP))
            output.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = ("passed" if all(item.get("exact", False) for item in report["checks"])
                            else "mismatch")
    except Exception as error:
        report["status"] = "failed" if report["status"] != "blocked" else "blocked"
        report["error"] = repr(error)
        output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {str(item["layer"]): item["exact"] for item in report["checks"]}
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "batch_capacity": args.batch_capacity,
        "layers": summary,
    }), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
