# SPDX-License-Identifier: Apache-2.0
"""Rank-local real-weight C5 serial versus packed NxC5 MTP draft attention.

This component tool loads only ``mtp.{0,1,2}.attn.*`` from a prepared TP8 PP1
rank-0 file.  It compares the production serial single-request path
``PagedCSA2Attention.draft(value[C5], positions)`` with the packed production
entry ``draft_attention_batch(..., N, C5)``.  ``--batch-capacity`` supports
1, 2 and 4 and defaults to the historical B2 contract.  No target oracle,
target tool, model body, MoE, head, distributed process group, or HCCL
collective is used.

The rank-0 attention weights are real, but reductions/gathers are synthetic
identity/replicated callables.  This is a single-rank component contract and
is not a TP8 service result.  Both arms use fixed-shape ``hpu_backend``
``fullgraph=True`` and ``dynamic=False`` graphs.  Outputs and persistent rings
are compared bit-for-bit after host copy; no floating-point tolerance is used.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

DEFAULT_BATCH_CAPACITY = 2
SUPPORTED_BATCH_CAPACITIES = (1, 2, 4)
DRAFT_TOKENS = 5
DRAFT_INPUT_WIDTH = 5120
WINDOW = 128
SWA_ROWS = 256
ATTENTION_WIDTH = 192
PACKED_SWA_WIDTH = 528
SELECTED_KV_OP = "custom_deepseek_v41_selected_kv_bf16_gaudi2"
SPARSE_ATTN_OP = "custom_deepseek_v4_sparse_attn_bf16_gaudi2"
ROPE_OP = "custom_deepseek_v41_rope_bf16_gaudi2"
ROPE_INVERSE_OP = "custom_deepseek_v41_rope_inverse_bf16_gaudi2"

SOURCE_FILES = (
    "tools/check_deepseek_v41_dspark_draft_attention.py",
    "tests/standalone/deepseek_v41/test_dspark_draft_attention_tool.py",
    "vllm_gaudi/ops/deepseek_v41_batch_draft.py",
    "vllm_gaudi/ops/deepseek_v41_batch_speculation.py",
    "vllm_gaudi/ops/deepseek_v41_paged_attention.py",
    "vllm_gaudi/ops/deepseek_v41_qkv.py",
    "vllm_gaudi/ops/deepseek_v41_math.py",
    "vllm_gaudi/ops/deepseek_v41_shard_loader.py",
    "vllm_gaudi/models/deepseek_v41_program.py",
    "vllm_gaudi/entrypoints/deepseek_v41.py",
)

FLAG_ENV_NAMES = (
    "VLLM_HPU_DSV41_DSPARK",
    "VLLM_HPU_DSV41_NATIVE_ROPE",
    "VLLM_HPU_DSV41_DSPARK_DRAFT_KV_DECODE",
    "VLLM_HPU_DSV41_DSPARK_DRAFT_PACKED_MLA",
    "VLLM_HPU_DSV41_DSPARK_DRAFT_QUERY_FP8",
    "VLLM_HPU_DSV41_DSPARK_DRAFT_DENSE_FP8",
    "VLLM_HPU_DSV41_DSPARK_WO_HANDOFF",
    "VLLM_HPU_DSV41_DSPARK_ROPE_COHERENT",
    "VLLM_HPU_DSV41_PRETRANSPOSE_ATTN",
    "VLLM_HPU_DSV41_ATTN_DENSE_FP8",
    "VLLM_HPU_DSV41_WO_A_FP8",
    "VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP",
    "VLLM_HPU_DSV4_TPC_OP_LIBRARY",
    "VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR",
    "GC_KERNEL_PATH",
)

SCENARIOS = (
    {
        "name": "distinct_slots_and_different_positions",
        "slots": (1, 0),
        "positions": ((300, 301, 302, 303, 304), (700, 701, 702, 703, 704)),
        "active": (True, True),
    },
    {
        "name": "inactive_lane_padding",
        # Keep the inactive RoPE lookups in-range; activity, not a sentinel
        # position, is the padding contract under test here.
        "slots": (0, 1),
        "positions": ((0, 0, 0, 0, 0), (900, 901, 902, 903, 904)),
        "active": (False, True),
    },
)

CAPACITY_SCENARIOS = {
    1: (
        {
            "name": "capacity1_full_a1",
            "slots": (0, ),
            "positions": ((300, 301, 302, 303, 304), ),
            "active": (True, ),
        },
    ),
    # Keep the historical B2 scenarios byte-for-byte as the default regression.
    2: SCENARIOS,
    4: (
        {
            "name": "capacity4_full_a4_lane_slots",
            "slots": (0, 1, 2, 3),
            "positions": (
                (300, 301, 302, 303, 304),
                (500, 501, 502, 503, 504),
                (700, 701, 702, 703, 704),
                (900, 901, 902, 903, 904),
            ),
            "active": (True, True, True, True),
        },
        {
            "name": "capacity4_partial_a2_lane_slots",
            "slots": (0, -1, 2, -1),
            "positions": (
                (300, 301, 302, 303, 304),
                (500, 501, 502, 503, 504),
                (700, 701, 702, 703, 704),
                (900, 901, 902, 903, 904),
            ),
            # A hole in the middle proves activity, not array position, owns
            # numeric work while lane==slot remains fixed.
            "active": (True, False, True, False),
        },
        {
            "name": "capacity4_partial_a1_lane_slots",
            "slots": (-1, 1, -1, -1),
            "positions": (
                (300, 301, 302, 303, 304),
                (500, 501, 502, 503, 504),
                (700, 701, 702, 703, 704),
                (900, 901, 902, 903, 904),
            ),
            "active": (False, True, False, False),
        },
    ),
}


def scenarios_for_capacity(capacity):
    """Return the fixed-capacity scenario plan without inferring another N."""
    try:
        scenarios = CAPACITY_SCENARIOS[capacity]
    except KeyError as error:
        raise ValueError(f"unsupported draft batch capacity: {capacity}") from error
    for scenario in scenarios:
        if len(scenario["slots"]) != capacity:
            raise ValueError("scenario slot count differs from capacity")
        if (len(scenario["positions"]) != capacity
                or any(len(row) != DRAFT_TOKENS for row in scenario["positions"])):
            raise ValueError("scenario position shape differs from [N, 5]")
        if len(scenario["active"]) != capacity:
            raise ValueError("scenario activity count differs from capacity")
        if capacity == 4:
            expected_slots = tuple(lane if enabled else -1
                                   for lane, enabled in enumerate(scenario["active"]))
            if tuple(scenario["slots"]) != expected_slots:
                raise ValueError("N=4 scenarios must use lane slots for active requests and -1 for inactive")
    return scenarios


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path, help="Prepared TP8 PP1 rank directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence path")
    parser.add_argument("--rank", type=int, choices=(0, ), default=0, help="Prepared TP rank (rank 0 only)")
    parser.add_argument("--tp-size", type=int, choices=(8, ), default=8, help="Prepared TP geometry")
    parser.add_argument("--layers", nargs="+", type=int, choices=(0, 1, 2), default=(0, 1, 2))
    parser.add_argument(
        "--batch-capacity",
        type=int,
        choices=SUPPORTED_BATCH_CAPACITIES,
        default=DEFAULT_BATCH_CAPACITY,
        help="Fixed draft request capacity N (default: historical B2)",
    )
    parser.add_argument("--max-length", type=int, default=2048, help="Synthetic RoPE/shared-state length")
    parser.add_argument("--seed", type=int, default=4105)
    parser.add_argument("--native-dir", type=Path, help="Explicit V4.1 native library directory")
    args = parser.parse_args(argv)
    args.layers = tuple(dict.fromkeys(args.layers))
    if args.max_length < 1024:
        parser.error("--max-length must cover the 900..904 scenario")
    return args


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path, *, include_sha=True):
    path = Path(path)
    stat = path.stat()
    record = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }
    if include_sha:
        record["sha256"] = _sha256_file(path)
    return record


def source_fingerprints(root=None):
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    records = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source artifact is missing: {path}")
        records[relative] = _file_record(path)
    return records


def make_request_metadata(capacity, active, positions):
    """Build the canonical explicit-activity metadata for one capacity run.

    The returned slots are scheduler lane slots.  Legacy B2 attention scenarios
    can still exercise swapped physical attention slots; that input is reported
    separately as ``attention_slots`` and is not conflated with request metadata.
    """
    from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata

    if isinstance(active, torch.Tensor):
        active_values = active.detach().cpu().tolist()
    else:
        active_values = list(active)
    if isinstance(positions, torch.Tensor):
        position_values = positions.detach().cpu().tolist()
    else:
        position_values = [list(row) for row in positions]
    if capacity not in SUPPORTED_BATCH_CAPACITIES:
        raise ValueError(f"unsupported draft batch capacity: {capacity}")
    if (len(active_values) != capacity or len(position_values) != capacity
            or any(len(row) != DRAFT_TOKENS for row in position_values)):
        raise ValueError("metadata inputs must have request_active[N] and positions[N,5]")
    if not any(active_values):
        raise ValueError("at least one request lane must be active")

    metadata = DSparkBatchMetadata.build(
        requests=tuple(f"draft-attention-lane-{lane}" if enabled else None
                       for lane, enabled in enumerate(active_values)),
        active=active_values,
        target_query_lengths=[DRAFT_TOKENS + 1 if enabled else 0 for enabled in active_values],
        draft_token_ids=[
            [lane * 17 + token for token in range(DRAFT_TOKENS)] if enabled else []
            for lane, enabled in enumerate(active_values)
        ],
        request_slots=[lane if enabled else -1 for lane, enabled in enumerate(active_values)],
        slot_generations=[lane + 1 if enabled else 0 for lane, enabled in enumerate(active_values)],
        remaining_output_budget=[64 if enabled else 0 for enabled in active_values],
        target_start_positions=[int(row[0]) if enabled else 0
                                for row, enabled in zip(position_values, active_values, strict=True)],
        round_generation=1,
        device="cpu",
    )
    target_query_lengths = (
        metadata.target_query_start_loc[1:] - metadata.target_query_start_loc[:-1]).tolist()
    return {
        "capacity": metadata.capacity,
        "request_active": metadata.request_active.tolist(),
        "request_ids": list(metadata.request_ids),
        "request_slots": metadata.request_slots.tolist(),
        "slot_generations": metadata.slot_generations.tolist(),
        "target_start_positions": metadata.target_start_positions.tolist(),
        "remaining_output_budget": metadata.remaining_output_budget.tolist(),
        "target_query_lengths": target_query_lengths,
        "round_generation": int(metadata.round_generation.item()),
        "shapes": {
            "request_active": list(metadata.request_active.shape),
            "request_slots": list(metadata.request_slots.shape),
            "slot_generations": list(metadata.slot_generations.shape),
            "target_start_positions": list(metadata.target_start_positions.shape),
            "remaining_output_budget": list(metadata.remaining_output_budget.shape),
        },
        "inactive_sentinel_contract": {
            "request_id": None,
            "request_slot": -1,
            "slot_generation": 0,
            "target_start_position": 0,
            "remaining_output_budget": 0,
            "target_query_length": 0,
        },
        "slot_semantics": "canonical scheduler lane slots; physical attention_slots are reported separately",
    }


def validate_compatible_flags(flags):
    """Return blockers where serial C5 and packed NxC5 do not share one path."""
    checks = (
        ("dspark", True, "DSpark must be enabled"),
        ("draft_kv_decode", True, "both arms must use the shared selected-KV decoder"),
        ("draft_packed_mla", False, "serial C5 uses sparse attention, not packed MLA"),
        ("draft_query_fp8", False, "the serial-only request_batch=False Q fusion changes the path"),
        ("draft_dense_fp8", False, "C5 output handoff is unavailable at packed 5*N row counts"),
        ("wo_handoff", False, "the C2-C6 output handoff is unavailable at packed 5*N row counts"),
        ("pretranspose_attn", False, "MTP attention consumes the checkpoint wo_a layout"),
    )
    blockers = []
    for name, expected, reason in checks:
        actual = bool(flags[name])
        if actual != expected:
            blockers.append({
                "flag": name,
                "expected": expected,
                "actual": actual,
                "reason": reason,
            })
    if flags.get("native_rope", False) and flags["rope_coherent"]:
        blockers.append({
            "flag": "rope_coherent",
            "expected": False,
            "actual": True,
            "reason": "coherent RoPE is serial-only for request_batch=False",
        })
    return blockers


def collect_flags(gaudi_envs):
    return {
        "dspark": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK),
        "native_rope": bool(gaudi_envs.VLLM_HPU_DSV41_NATIVE_ROPE),
        "draft_kv_decode": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_DRAFT_KV_DECODE),
        "draft_packed_mla": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_DRAFT_PACKED_MLA),
        "draft_query_fp8": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_DRAFT_QUERY_FP8),
        "draft_dense_fp8": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_DRAFT_DENSE_FP8),
        "wo_handoff": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_WO_HANDOFF),
        "rope_coherent": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_ROPE_COHERENT),
        "pretranspose_attn": bool(gaudi_envs.VLLM_HPU_DSV41_PRETRANSPOSE_ATTN),
        "attention_dense_fp8": bool(gaudi_envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8),
        "wo_a_fp8": bool(gaudi_envs.VLLM_HPU_DSV41_WO_A_FP8),
        "woa_output_roundtrip": bool(gaudi_envs.VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP),
    }


def _tensor_sha256(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_exact(name, reference, actual):
    """Bitwise comparison after host copy; there is deliberately no tolerance."""
    reference = reference.detach().cpu()
    actual = actual.detach().cpu()
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        return {
            "name": name,
            "shape_reference": list(reference.shape),
            "shape_actual": list(actual.shape),
            "dtype_reference": str(reference.dtype),
            "dtype_actual": str(actual.dtype),
            "reference_sha256": _tensor_sha256(reference),
            "actual_sha256": _tensor_sha256(actual),
            "exact": False,
            "numeric_equal": False,
            "bitwise_equal": False,
            "mismatch": None,
            "max_abs": None,
        }
    numeric_equal = torch.equal(reference, actual)
    reference_bits = reference.contiguous().view(torch.uint8)
    actual_bits = actual.contiguous().view(torch.uint8)
    row_bytes = reference.element_size()
    changed = (reference_bits != actual_bits).reshape(reference.numel(), row_bytes).any(1)
    delta = (reference.float() - actual.float()).abs()
    return {
        "name": name,
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "reference_sha256": _tensor_sha256(reference_bits),
        "actual_sha256": _tensor_sha256(actual_bits),
        "exact": bool(torch.equal(reference_bits, actual_bits)),
        "numeric_equal": bool(numeric_equal),
        "bitwise_equal": bool(torch.equal(reference_bits, actual_bits)),
        "mismatch": int(changed.sum().item()),
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "criterion": "bitwise_equal_after_host_copy",
        "rtol": 0,
        "atol": 0,
    }


def audit_attention_rows(rows, positions, slots, active, *, ring_rows):
    """Audit the real production row builder's slot/window/scratch ownership."""
    positions = positions.detach().cpu().to(torch.int32)
    slots = slots.detach().cpu().to(torch.int32)
    active = active.detach().cpu().to(torch.bool)
    batch = int(positions.shape[0])
    lanes = torch.arange(DRAFT_TOKENS, dtype=torch.int32)
    window_offsets = torch.arange(WINDOW, dtype=torch.int32)
    context_positions = positions[:, :1] - WINDOW + window_offsets.unsqueeze(0)
    expected_context = slots.unsqueeze(1) * SWA_ROWS + context_positions.remainder(SWA_ROWS)
    expected_context = torch.where(context_positions >= 0, expected_context, -1)
    expected_context = torch.where(active.unsqueeze(1), expected_context, -1)
    expected_tentative = ring_rows + torch.arange(batch, dtype=torch.int32).unsqueeze(1) * DRAFT_TOKENS
    expected_tentative = expected_tentative + lanes.unsqueeze(0)
    expected_tentative = torch.where(active.unsqueeze(1), expected_tentative, -1)
    expected_row_ids = torch.cat((expected_context, expected_tentative), dim=1).reshape(-1)

    expected_lengths = torch.where(
        active,
        torch.full((batch, ), WINDOW + DRAFT_TOKENS, dtype=torch.int32),
        torch.zeros((batch, ), dtype=torch.int32),
    ).unsqueeze(1).expand(batch, DRAFT_TOKENS).reshape(-1)

    expected_sparse_context = (
        torch.arange(WINDOW, dtype=torch.int32).unsqueeze(0)
        + torch.arange(batch, dtype=torch.int32).unsqueeze(1) * WINDOW
    )
    expected_sparse_context = torch.where(
        active.unsqueeze(1) & (context_positions >= 0), expected_sparse_context, -1)
    expected_sparse_tentative = (
        batch * WINDOW
        + torch.arange(batch, dtype=torch.int32).unsqueeze(1) * DRAFT_TOKENS
        + lanes.unsqueeze(0)
    )
    sparse_page = torch.cat((expected_sparse_context, expected_sparse_tentative), dim=1)
    sparse_page = torch.nn.functional.pad(
        sparse_page, (0, ATTENTION_WIDTH - sparse_page.shape[1]), value=-1)
    expected_sparse = sparse_page.unsqueeze(1).expand(batch, DRAFT_TOKENS, -1)
    expected_sparse = torch.where(active.unsqueeze(1).unsqueeze(2), expected_sparse, -1).reshape(
        batch * DRAFT_TOKENS, -1)

    local = torch.cat((torch.arange(WINDOW, dtype=torch.int32), WINDOW + lanes))
    local = torch.nn.functional.pad(local, (0, ATTENTION_WIDTH - local.numel()), value=-1)
    request_base = (torch.arange(batch, dtype=torch.int32) * local.numel()).reshape(batch, 1, 1)
    expected_indices = local.reshape(1, 1, -1).expand(batch, DRAFT_TOKENS, -1)
    expected_indices = torch.where(
        expected_indices >= 0, expected_indices + request_base, -1).reshape(batch * DRAFT_TOKENS, -1)

    actual_row_ids = rows.row_ids.detach().cpu().to(torch.int32)
    actual_lengths = rows.lengths.detach().cpu().to(torch.int32)
    actual_indices = rows.indices.detach().cpu().to(torch.int32)
    actual_sparse = rows.sparse_indices.detach().cpu().to(torch.int32)
    actual_context = actual_row_ids.reshape(batch, WINDOW + DRAFT_TOKENS)[:, :WINDOW]
    actual_tentative = actual_row_ids.reshape(batch, WINDOW + DRAFT_TOKENS)[:, WINDOW:]
    checks = {
        "row_ids": compare_exact("row_ids", expected_row_ids, actual_row_ids),
        "lengths": compare_exact("lengths", expected_lengths, actual_lengths),
        "indices": compare_exact("indices", expected_indices, actual_indices),
        "sparse_indices": compare_exact("sparse_indices", expected_sparse, actual_sparse),
        "context_rows": compare_exact("context_rows", expected_context, actual_context),
        "tentative_rows": compare_exact("tentative_rows", expected_tentative, actual_tentative),
    }
    active_context = actual_context[active]
    active_tentative = actual_tentative[active]
    unique_context = len(set(active_context.reshape(-1).tolist()))
    expected_context_count = int(active.sum().item()) * WINDOW
    tentative_outside_ring = bool((active_tentative >= ring_rows).all().item()) if active_tentative.numel() else True
    inactive_rows_masked = bool((actual_tentative[~active] == -1).all().item()) if (~active).any() else True
    return {
        "exact": all(check["exact"] for check in checks.values()),
        "checks": checks,
        "active_requests": int(active.sum().item()),
        "context_rows_per_active_request": WINDOW,
        "tentative_rows_per_active_request": DRAFT_TOKENS,
        "unique_active_context_rows": unique_context,
        "expected_unique_active_context_rows": expected_context_count,
        "unique_context_complete": unique_context == expected_context_count,
        "active_tentative_rows_outside_persistent_ring": tentative_outside_ring,
        "inactive_tentative_rows_masked": inactive_rows_masked,
        "tentative_scratch_row_ids": actual_tentative.tolist(),
    }


class _AttentionOwner(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention


class _AttentionOnlyDraft(torch.nn.Module):
    def __init__(self, attentions):
        super().__init__()
        self.layers = torch.nn.ModuleList([_AttentionOwner(attention) for attention in attentions])


class _SerialAttentionEntry(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, value, positions):
        return self.attention.draft(value, positions)


class _PackedAttentionEntry(torch.nn.Module):
    def __init__(self, draft, layer_index, batch_state):
        super().__init__()
        self.draft = draft
        self.layer_index = layer_index
        self.batch_state = batch_state

    def forward(self, value, positions, slots, active):
        from vllm_gaudi.ops.deepseek_v41_batch_draft import draft_attention_batch

        return draft_attention_batch(
            self.draft, self.layer_index, value, positions, slots, active, self.batch_state)


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
    # Independent oracle modules share backend identity so the counter itself
    # does not create Dynamo BACKEND_MATCH recompilations across MTP layers.
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
            # This qualification tool invokes entries sequentially, so the
            # shared backend delta belongs only to the entry being called.
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


def _make_ring_seed(seed, device, capacity):
    rows = []
    for slot in range(capacity):
        generator = torch.Generator().manual_seed(seed + slot * 1009)
        values = torch.randn((SWA_ROWS, 512), generator=generator, dtype=torch.bfloat16)
        from vllm_gaudi.ops.deepseek_v41_math import pack_swa

        rows.append(pack_swa(values))
    return torch.cat(rows, dim=0).to(device)


def _make_value(seed, device, capacity):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        (capacity * DRAFT_TOKENS, DRAFT_INPUT_WIDTH), generator=generator, dtype=torch.bfloat16).to(device)


def _state_checks(before, after_serial, after_packed, *, capacity, active):
    checks = []
    active = active.detach().cpu().to(torch.bool)
    for index, (baseline, serial, packed) in enumerate(zip(before, after_serial, after_packed, strict=True)):
        if baseline.shape[0] != capacity * SWA_ROWS:
            raise ValueError("persistent ring does not contain exactly N fixed slots")
        slot_checks = []
        for slot in range(capacity):
            start = slot * SWA_ROWS
            stop = (slot + 1) * SWA_ROWS
            slot_checks.append({
                "slot": slot,
                "active": bool(active[slot].item()),
                "after_serial": compare_exact(
                    f"ring{index}_slot{slot}_after_serial", baseline[start:stop], serial[start:stop]),
                "after_packed": compare_exact(
                    f"ring{index}_slot{slot}_after_packed", baseline[start:stop], packed[start:stop]),
            })
        checks.append({
            "ring": index,
            "after_serial": compare_exact(f"ring{index}_after_serial", baseline, serial),
            "after_packed": compare_exact(f"ring{index}_after_packed", baseline, packed),
            "slots": slot_checks,
        })
    return checks


def audit_scenario_outputs(*, capacity, active, serial_outputs, packed_output, ring_before,
                           ring_after_serial, ring_after_packed):
    """Compare every active C5 block and require every inactive block to be zero."""
    active = active.detach().cpu().to(torch.bool)
    if active.shape != (capacity, ):
        raise ValueError("activity mask must have shape [N]")
    expected_rows = capacity * DRAFT_TOKENS
    if packed_output.shape[0] != expected_rows:
        raise ValueError(f"packed output must have {expected_rows} rows")
    output_checks = []
    inactive_zero_checks = []
    serial_dtypes = []
    for lane in range(capacity):
        start = lane * DRAFT_TOKENS
        stop = start + DRAFT_TOKENS
        if bool(active[lane].item()):
            if lane not in serial_outputs:
                raise ValueError(f"missing serial C5 oracle for active lane {lane}")
            reference = serial_outputs[lane]
            serial_dtypes.append(str(reference.dtype))
            output_checks.append(compare_exact(
                f"lane{lane}_output", reference, packed_output[start:stop]))
        else:
            inactive_zero_checks.append(compare_exact(
                f"lane{lane}_inactive_zero", torch.zeros_like(packed_output[start:stop]),
                packed_output[start:stop]))
    state_checks = _state_checks(
        ring_before,
        ring_after_serial,
        ring_after_packed,
        capacity=capacity,
        active=active,
    )
    serial_state_exact = all(item["after_serial"]["exact"] for item in state_checks)
    packed_state_exact = all(item["after_packed"]["exact"] for item in state_checks)
    dtype_contract = {
        "expected": "torch.bfloat16",
        "packed_actual": str(packed_output.dtype),
        "serial_actuals": serial_dtypes,
        "exact": packed_output.dtype == torch.bfloat16 and all(item == "torch.bfloat16" for item in serial_dtypes),
    }
    return {
        "active_outputs": output_checks,
        "inactive_zero_outputs": inactive_zero_checks,
        "dtype_contract": dtype_contract,
        "persistent_state": {
            "criterion": "whole packed ring and every fixed slot, bitwise_equal_after_host_copy",
            "serial_unchanged": serial_state_exact,
            "packed_unchanged": packed_state_exact,
            "checks": state_checks,
        },
        "exact": (
            dtype_contract["exact"]
            and all(check["exact"] for check in output_checks)
            and all(check["exact"] for check in inactive_zero_checks)
            and serial_state_exact
            and packed_state_exact
        ),
    }


def _synth_reduce(value, **kwargs):
    del kwargs
    return value


def _synth_gather(value, dim):
    # Replication makes ABI shapes explicit; it does not implement TP8.
    return torch.cat(tuple(value for _ in range(8)), dim=dim)


def _build_attention(weights, config, shared, layer, linear, device):
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention

    attention = PagedCSA2Attention(
        weights,
        config,
        layer,
        shared,
        linear,
        _synth_reduce,
        _synth_gather,
        device,
        tensor_parallel_size=8,
    )
    attention.set_search_length(attention.length)
    return attention


def _tensor_specs(specs):
    dtype_bytes = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "F64": 8,
        "I8": 1,
        "U8": 1,
        "I16": 2,
        "I32": 4,
        "I64": 8,
        "BOOL": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
    }
    records = []
    for name, spec in sorted(specs.items()):
        elements = 1
        for extent in spec["shape"]:
            elements *= int(extent)
        records.append({
            "name": name,
            "dtype": spec["dtype"],
            "shape": list(spec["shape"]),
            "bytes": elements * dtype_bytes[spec["dtype"]],
        })
    return records


def _native_artifacts():
    records = {}
    for name in ("VLLM_HPU_DSV4_TPC_OP_LIBRARY", "GC_KERNEL_PATH"):
        value = os.environ.get(name)
        if not value or not Path(value).is_file():
            continue
        # The kernel database can be very large.  Its locking manifest is hashed
        # below; retain an inode/size/mtime identity for the live database.
        records[name] = _file_record(value, include_sha=name != "GC_KERNEL_PATH")
    library = os.environ.get("VLLM_HPU_DSV4_TPC_OP_LIBRARY")
    if library:
        manifest = Path(library).parent / "deepseek_v4_build.json"
        if manifest.is_file():
            records["deepseek_v4_build.json"] = _file_record(manifest)
    return records


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def _qualify_layer(report, *, layer, weights, config, shared, draft, batch_state, linear, device, seed,
                   capacity):
    from vllm_gaudi.ops.deepseek_v41_batch_draft import draft_attention_rows

    attention_weights = weights.mtp.get_submodule(str(layer)).attn
    packed_attention = draft.layers[layer].attention
    reference_attentions = [
        _build_attention(attention_weights, config, shared, layer + 40, linear, device)
        for _ in range(capacity)
    ]
    reference_banks = [
        torch.empty((SWA_ROWS, PACKED_SWA_WIDTH), dtype=torch.uint8, device=device) for _ in range(capacity)
    ]
    for attention, bank in zip(reference_attentions, reference_banks, strict=True):
        attention.swa = bank
    serial_entries = [
        _GraphCallCounter(_SerialAttentionEntry(attention)) for attention in reference_attentions
    ]
    packed_entry = _GraphCallCounter(_PackedAttentionEntry(draft, layer, batch_state))
    serial_initial = [entry.snapshot() for entry in serial_entries]
    packed_initial = packed_entry.snapshot()
    layer_report = {
        "layer": layer,
        "module_layer": layer + 40,
        "batch_capacity": capacity,
        "real_weight_prefix": f"mtp.{layer}.attn.",
        "reference_path": "PagedCSA2Attention.draft(C5), one serial call per active lane",
        "packed_path": f"deepseek_v41_batch_draft.draft_attention_batch(N={capacity}xC5)",
        "compile": {
            "backend": "hpu_backend",
            "fullgraph": True,
            "dynamic": False,
            "serial_graphs": len(serial_entries),
            "packed_graphs": 1,
            "one_packed_graph_per_layer": True,
        },
        "attention_effective_flags": {
            "draft_kv_decode": bool(packed_attention.draft_kv_decode),
            "draft_packed_mla": bool(packed_attention.draft_packed_mla),
            "draft_query_fp8": bool(packed_attention.draft_query_fp8),
            "draft_dense_fp8": bool(packed_attention.draft_dense_fp8),
            "native_rope": bool(packed_attention.native_rope),
            "woa_fp8": bool(packed_attention.woa_fp8),
            "woa_output_roundtrip": bool(packed_attention.woa_output_roundtrip),
            "length": int(packed_attention.length),
        },
        "scenarios": [],
        "exact": True,
    }
    for scenario_index, scenario in enumerate(scenarios_for_capacity(capacity)):
        positions_host = torch.tensor(scenario["positions"], dtype=torch.int32)
        slots_host = torch.tensor(scenario["slots"], dtype=torch.int32)
        active_host = torch.tensor(scenario["active"], dtype=torch.bool)
        positions = positions_host.to(device)
        slots = slots_host.to(device)
        active = active_host.to(device)
        active_count = int(active_host.sum().item())
        active_lanes = [lane for lane, enabled in enumerate(scenario["active"]) if enabled]
        inactive_lanes = [lane for lane, enabled in enumerate(scenario["active"]) if not enabled]
        persistent_state_slots = [
            int(slots_host[lane]) if enabled or int(slots_host[lane]) >= 0 else lane
            for lane, enabled in enumerate(scenario["active"])
        ]
        if any(slot < 0 or slot >= capacity for slot in persistent_state_slots):
            raise ValueError("persistent state slot is outside capacity")
        ring_seed = _make_ring_seed(
            seed + layer * 10000 + scenario_index * 1000, device, capacity)
        batch_state.ring(layer).copy_(ring_seed)
        value = _make_value(
            seed + layer * 10000 + scenario_index * 1000 + 700, device, capacity)
        torch.hpu.synchronize()

        rows = draft_attention_rows(positions, slots, active, ring_rows=batch_state.ring(layer).shape[0])
        row_audit = audit_attention_rows(
            rows, positions, slots, active, ring_rows=batch_state.ring(layer).shape[0])
        before = tuple(ring.clone() for ring in batch_state.state_tensors())

        serial_before = [entry.snapshot() for entry in serial_entries]
        serial_outputs = {}
        for lane in active_lanes:
            slot = int(slots_host[lane])
            reference_banks[lane].copy_(
                ring_seed[slot * SWA_ROWS:(slot + 1) * SWA_ROWS])
            torch.hpu.synchronize()
            start = lane * DRAFT_TOKENS
            serial_outputs[lane] = serial_entries[lane](
                value[start:start + DRAFT_TOKENS], positions[lane])
        torch.hpu.synchronize()
        after_serial = tuple(ring.clone() for ring in batch_state.state_tensors())
        serial_graph_runtime = _graph_usage(serial_entries, before=serial_before)

        packed_before = packed_entry.snapshot()
        packed_output = packed_entry(value, positions, slots, active)
        torch.hpu.synchronize()
        after_packed = tuple(ring.clone() for ring in batch_state.state_tensors())
        packed_graph_runtime = _graph_usage([packed_entry], before=[packed_before])

        output_audit = audit_scenario_outputs(
            capacity=capacity,
            active=active_host,
            serial_outputs=serial_outputs,
            packed_output=packed_output,
            ring_before=before,
            ring_after_serial=after_serial,
            ring_after_packed=after_packed,
        )

        inactive_invariance = None
        if inactive_lanes:
            alternate_source = _make_ring_seed(
                seed + layer * 10000 + scenario_index * 1000 + 500003, device, capacity)
            alternate_ring = ring_seed.clone()
            for lane in inactive_lanes:
                slot = persistent_state_slots[lane]
                rows = slice(slot * SWA_ROWS, (slot + 1) * SWA_ROWS)
                alternate_ring[rows].copy_(alternate_source[rows])
            batch_state.ring(layer).copy_(alternate_ring)
            torch.hpu.synchronize()
            alternate_before = tuple(ring.clone() for ring in batch_state.state_tensors())
            alternate_before_graph = packed_entry.snapshot()
            alternate_output = packed_entry(value, positions, slots, active)
            torch.hpu.synchronize()
            alternate_after = tuple(ring.clone() for ring in batch_state.state_tensors())
            alternate_graph_runtime = _graph_usage(
                [packed_entry], before=[alternate_before_graph])
            alternate_audit = audit_scenario_outputs(
                capacity=capacity,
                active=active_host,
                serial_outputs=serial_outputs,
                packed_output=alternate_output,
                ring_before=alternate_before,
                ring_after_serial=alternate_before,
                ring_after_packed=alternate_after,
            )
            alternate_output_checks = []
            for lane in range(capacity):
                start = lane * DRAFT_TOKENS
                alternate_output_checks.append(compare_exact(
                    f"lane{lane}_output_after_inactive_ring_change",
                    packed_output[start:start + DRAFT_TOKENS],
                    alternate_output[start:start + DRAFT_TOKENS],
                ))
            inactive_invariance = {
                "changed_inactive_slots": [
                    persistent_state_slots[lane] for lane in inactive_lanes
                ],
                "inactive_device_attention_slots": [int(slots_host[lane]) for lane in inactive_lanes],
                "alternate_ring_sha256": _tensor_sha256(alternate_ring.detach().cpu()),
                "outputs_compared": alternate_output_checks,
                "alternate_output_audit": alternate_audit,
                "persistent_state_unchanged": alternate_audit["persistent_state"]["packed_unchanged"],
                "state_checks": alternate_audit["persistent_state"]["checks"],
                "graph_runtime": alternate_graph_runtime,
                "exact": (
                    all(check["exact"] for check in alternate_output_checks)
                    and alternate_audit["exact"]
                ),
                "oracle_note": (
                    "Active C5 blocks retain their serial oracles, inactive blocks remain bitwise zero, "
                    "and persistent slots do not change when inactive-slot ring rows change."
                ),
            }

        serial_output = serial_outputs[active_lanes[0]]
        request_metadata = make_request_metadata(capacity, active_host, positions_host)
        active_device_slots_exact = all(
            int(slots_host[lane]) == lane for lane in active_lanes)
        inactive_device_sentinel_exact = all(
            int(slots_host[lane]) == -1 for lane in inactive_lanes)
        persistent_state_lane_exact = all(
            slot == lane for lane, slot in enumerate(persistent_state_slots))
        capacity_lane_contract_exact = (
            capacity == 2
            or (active_device_slots_exact and inactive_device_sentinel_exact and persistent_state_lane_exact)
        )
        scenario_exact = (
            row_audit["exact"]
            and row_audit["unique_context_complete"]
            and row_audit["active_tentative_rows_outside_persistent_ring"]
            and row_audit["inactive_tentative_rows_masked"]
            and capacity_lane_contract_exact
            and output_audit["exact"]
            and (inactive_invariance is None or inactive_invariance["exact"])
        )
        layer_report["scenarios"].append({
            "scenario": scenario["name"],
            "batch_capacity": capacity,
            "active_requests": active_count,
            "active_lanes": active_lanes,
            "inactive_lanes": inactive_lanes,
            "slots": list(scenario["slots"]),
            "lane_slot_contract": {
                "capacity_lane_contract_exact": capacity_lane_contract_exact,
                "active_device_lane_equals_slot": active_device_slots_exact,
                "inactive_device_slot_is_minus_one": inactive_device_sentinel_exact,
                "persistent_state_slot_equals_host_lane": persistent_state_lane_exact,
                "device_attention_slots": list(scenario["slots"]),
                "persistent_state_slots": persistent_state_slots,
                "host_lane_request_slots": request_metadata["request_slots"],
            },
            "positions": [list(item) for item in scenario["positions"]],
            "active": list(scenario["active"]),
            "request_metadata": request_metadata,
            "shapes": {
                "packed_value": list(value.shape),
                "serial_value": [DRAFT_TOKENS, value.shape[1]],
                "positions": list(positions.shape),
                "attention_slots": list(slots.shape),
                "request_active": list(active.shape),
                "packed_output": list(packed_output.shape),
                "serial_output": list(serial_output.shape),
                "persistent_ring": list(batch_state.ring(layer).shape),
                "physical_draft_rows": capacity * DRAFT_TOKENS,
                "active_draft_rows": active_count * DRAFT_TOKENS,
                "physical_context_rows": capacity * WINDOW,
                "active_context_rows": active_count * WINDOW,
                "sparse_indices": [capacity * DRAFT_TOKENS, ATTENTION_WIDTH],
            },
            "graph_runtime": {
                "serial": serial_graph_runtime,
                "packed": packed_graph_runtime,
            },
            "input_fingerprints": {
                "packed_value_sha256": _tensor_sha256(value.detach().cpu()),
                "capacity_ring_sha256": _tensor_sha256(ring_seed.detach().cpu()),
            },
            "row_audit": row_audit,
            "outputs": output_audit["active_outputs"],
            "inactive_outputs": output_audit["inactive_zero_outputs"],
            "output_dtype": output_audit["dtype_contract"],
            "persistent_state": output_audit["persistent_state"],
            "inactive_ring_invariance": inactive_invariance,
            "inactive_output_numeric_reference": (
                "bitwise_zero" if inactive_lanes else "not_applicable"
            ),
            "inactive_output_note": (
                "Inactive C5 blocks must be bitwise-zero BF16, remain zero when their physical slot "
                "rings change, and leave every persistent ring slot unchanged."
                if inactive_lanes else None
            ),
            "exact": scenario_exact,
        })
        layer_report["exact"] = layer_report["exact"] and scenario_exact
        report["status"] = (
            "running" if report["status"] != "mismatch" and layer_report["exact"] else "mismatch")
        _write_report(report["output_path"], report)
    layer_report["graph_runtime"] = {
        "serial": _graph_usage(serial_entries, before=serial_initial),
        "packed": _graph_usage([packed_entry], before=[packed_initial]),
    }
    return layer_report


def main(argv=None):
    args = parse_args(argv)
    report_path = args.output.resolve()
    scenario_plan = [{
        "name": scenario["name"],
        "batch_capacity": args.batch_capacity,
        "active_requests": sum(scenario["active"]),
        "active_lanes": [lane for lane, enabled in enumerate(scenario["active"]) if enabled],
        "attention_slots": list(scenario["slots"]),
    } for scenario in scenarios_for_capacity(args.batch_capacity)]
    report = {
        "scope": f"DSpark MTP draft attention serial C5 versus packed N={args.batch_capacity}xC5",
        "status": "initializing",
        "output_path": str(report_path),
        "prepared": str(args.prepared.resolve()),
        "rank": args.rank,
        "tp_size": args.tp_size,
        "batch_capacity": args.batch_capacity,
        "supported_batch_capacities": list(SUPPORTED_BATCH_CAPACITIES),
        "capacity_evidence": (
            f"this report is evidence only for --batch-capacity {args.batch_capacity}; "
            "other capacity results are not inferred"
        ),
        "scenario_plan": scenario_plan,
        "layers": list(args.layers),
        "source_fingerprints": source_fingerprints(),
        "shared_state_max_length": args.max_length,
        "context_source": "deterministic finite random BF16 encoded by the production pack_swa codec",
        "tp_scope": "single-rank rank0 real prepared shard with synthetic identity/replicated collectives",
        "not_tp8": True,
        "distributed_process_group": False,
        "hccl_collectives": False,
        "reference_path": "PagedCSA2Attention.draft, serial per request, C5",
        "packed_path": f"draft_attention_batch, N={args.batch_capacity} x C5",
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
            "active_output": "serial C5 reference and packed N=capacity block are BF16-bitwise equal",
            "inactive_output": "bitwise zero in every inactive C5 block",
            "persistent_ring": "every ring and fixed slot remains unchanged",
            "inactive_ring_invariance": "outputs do not depend on changed inactive-slot ring rows",
        },
        "command": {
            "argv": list(sys.argv if argv is None else [sys.argv[0], *argv]),
            "cwd": str(Path.cwd()),
        },
        "sidecars": {
            "loaded": [],
            "reason": (
                "Only mtp.{0,1,2}.attn.* tensors are loaded. Target dense/wo_a sidecars own "
                "backbone layers 0..39 and do not contain MTP attention tensors."
            ),
        },
        "checks": [],
    }
    _write_report(report_path, report)
    try:
        if args.native_dir:
            native_dir = args.native_dir.resolve()
            os.environ["VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR"] = str(native_dir)
            bridge = native_dir / "tp2_fused_ar_norm_bridge.so"
            if bridge.is_file():
                os.environ["VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE"] = str(bridge)
        visible = os.environ.get("HABANA_VISIBLE_MODULES")
        if visible:
            os.environ.setdefault("HLS_MODULE_ID", visible.split(",")[0])
        if "PT_HPU_RECIPE_CACHE_CONFIG" in os.environ:
            os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = os.environ[
                "PT_HPU_RECIPE_CACHE_CONFIG"].replace("{rank}", "0")

        from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators, prepare_environment

        prepare_environment(args.prepared, tensor_parallel_size=args.tp_size, pipeline_parallel_size=1)
        from vllm_gaudi import envs as gaudi_envs

        report["native_artifacts"] = _native_artifacts()
        flags = collect_flags(gaudi_envs)
        report["effective_flags"] = flags
        report["raw_env"] = {name: os.environ.get(name) for name in FLAG_ENV_NAMES}
        blockers = validate_compatible_flags(flags)
        if blockers:
            report["status"] = "blocked"
            report["blockers"] = blockers
            _write_report(report_path, report)
            raise RuntimeError(
                f"serial C5 and packed N={args.batch_capacity}xC5 do not share the same attention path: "
                + ", ".join(item["flag"] for item in blockers))

        required_operators = [SELECTED_KV_OP, SPARSE_ATTN_OP]
        if flags["native_rope"]:
            required_operators.extend((ROPE_OP, ROPE_INVERSE_OP))
        load_native_operators(required=tuple(required_operators))
        import habana_frameworks.torch.core  # noqa: F401

        from vllm_gaudi.models.deepseek_v41_program import _weight_tree, linear, load_weight_tree
        from vllm_gaudi.ops.deepseek_v41_batch_draft import DraftBatchState
        from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState
        from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

        torch.set_num_threads(1)
        torch.hpu.set_device(args.rank)
        shard = PreparedV41Shard(args.prepared, 0, args.rank)
        if (shard.tensor_parallel_size, shard.pipeline_parallel_size) != (args.tp_size, 1):
            raise ValueError("prepared shard topology is not TP8 PP1")
        text_config = json.loads((args.prepared / "config.json").read_text())["text_config"]
        prefixes = tuple(f"mtp.{layer}.attn." for layer in range(3))
        specs = {name: spec for name, spec in shard.specs.items() if name.startswith(prefixes)}
        loaded_layers = {int(name.split(".")[1]) for name in specs}
        if not specs or loaded_layers != {0, 1, 2}:
            raise ValueError("attention-only workload must contain mtp.0/1/2.attn tensors")
        weight_tree = _weight_tree(specs)
        load_weight_tree(shard, weight_tree, "hpu", specs)
        shard.check_identity()
        attention_specs = _tensor_specs(specs)
        report["weights"] = {
            "loaded_scope": "mtp.{0,1,2}.attn.* only",
            "full_mtp_loaded": False,
            "tensor_count": len(attention_specs),
            "source_bytes": sum(item["bytes"] for item in attention_specs),
            "tensors": attention_specs,
            "preparation": "checkpoint F8_E4M3 block32 decoded to BF16 by PreparedV41Shard.dense",
        }
        rank_record = shard.manifest["rank_files"][f"pp0-tp{args.rank}"]
        report["prepared_identity"] = {
            "manifest_sha256": _sha256_file(args.prepared / "manifest.json"),
            "config_sha256": _sha256_file(args.prepared / "config.json"),
            "plan_fingerprint": shard.manifest["plan_fingerprint"],
            "rank_file": str(shard.path.resolve()),
            "rank_file_bytes": rank_record["bytes"],
            "rank_file_sha256_from_manifest": rank_record["sha256"],
            "rank_source_identity": list(shard.source_identity),
        }
        report["native_operators"] = {
            name: hasattr(torch.ops.custom_op, name) for name in required_operators
        }
        if not all(report["native_operators"].values()):
            raise RuntimeError("one or more required native operators are unavailable")

        shared = PagedCSA2SharedState(
            text_config, 40, 43, "hpu", args.max_length, tensor_parallel_size=args.tp_size)
        attentions = [
            _build_attention(
                weight_tree.mtp.get_submodule(str(layer)).attn,
                text_config,
                shared,
                layer + 40,
                linear,
                "hpu",
            )
            for layer in range(3)
        ]
        draft = _AttentionOnlyDraft(attentions)
        batch_state = DraftBatchState(draft, capacity=args.batch_capacity)
        with torch.inference_mode():
            for layer in args.layers:
                layer_report = _qualify_layer(
                    report,
                    layer=layer,
                    weights=weight_tree,
                    config=text_config,
                    shared=shared,
                    draft=draft,
                    batch_state=batch_state,
                    linear=linear,
                    device="hpu",
                    seed=args.seed,
                    capacity=args.batch_capacity,
                )
                report["checks"].append(layer_report)
        closing_source_fingerprints = source_fingerprints()
        report["source_fingerprints_end"] = closing_source_fingerprints
        report["source_stable"] = all(
            record["sha256"] == report["source_fingerprints"][relative]["sha256"]
            for relative, record in closing_source_fingerprints.items()
        )
        if not report["source_stable"]:
            raise RuntimeError("source artifacts changed during draft-attention qualification")
        report["status"] = (
            "passed" if report["checks"] and all(item["exact"] for item in report["checks"])
            else "mismatch"
        )
        _write_report(report_path, report)
        print(json.dumps({
            "status": report["status"],
            "output": str(report_path),
            "batch_capacity": args.batch_capacity,
            "layers": {str(item["layer"]): item["exact"] for item in report["checks"]},
            "not_tp8": True,
        }), flush=True)
        return 0 if report["status"] == "passed" else 1
    except Exception as error:
        if report["status"] not in ("blocked", "mismatch"):
            report["status"] = "failed"
        report["error"] = repr(error)
        _write_report(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
