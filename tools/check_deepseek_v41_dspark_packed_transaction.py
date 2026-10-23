# SPDX-License-Identifier: Apache-2.0
"""Qualify the packed DSpark verify/context/MTP/Markov transaction.

The HPU path loads only ``mtp.*`` and ``head.weight`` from one prepared TP8 PP1
rank, builds the real ``PreparedDraft``, ``DraftBatchState``,
``PackedDraftTransaction`` and ``BatchVerifyRing``, then compares the fixed-N
packed graph with one independent serial oracle per active lane.

This is a rank-local component check. Embedding reduction, MoE reduction and
the vocabulary argmax/all-gather use synthetic identity or replicated
callables, so passing here does not qualify a real TP8 process group.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Iterable, Sequence

import torch

from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
from vllm_gaudi.ops.deepseek_v41_batch_draft import (
    DraftBatchState,
    PackedDraftTransaction,
    draft_inputs_after_commit,
    insert_context_batch,
)
from vllm_gaudi.ops.deepseek_v41_batch_verify import (
    BATCH_INACTIVE,
    BATCH_RECORD_SIZE,
    BatchVerifyRing,
    batch_binding_sidecar,
    batch_control_metadata,
    pack_record_batch,
    validate_batch_records,
    verify_control_from_target_batch,
)
from vllm_gaudi.ops.deepseek_v41_verify import pack_record, verify_control_from_target

DEFAULT_BATCH_CAPACITY = 2
SUPPORTED_BATCH_CAPACITIES = (1, 2, 4)
TP_SIZE = 8
TARGET_QUERY_BUCKET = 6
DRAFT_TOKENS = 5
HIDDEN_SIZE = 5120
TARGET_CONTEXT_WIDTH = 3 * HIDDEN_SIZE
SWA_ROWS = 256
MAX_TARGET_QUERIES = 6
MAX_MODEL_LENGTH = 2048

SELECTED_KV_OP = "custom_deepseek_v41_selected_kv_bf16_gaudi2"
SPARSE_ATTN_OP = "custom_deepseek_v4_sparse_attn_bf16_gaudi2"
ROPE_OP = "custom_deepseek_v41_rope_bf16_gaudi2"
ROPE_INVERSE_OP = "custom_deepseek_v41_rope_inverse_bf16_gaudi2"
STATE_ROWS_WRITE_OP = "custom_deepseek_v41_state_rows_write_gaudi2"
GENERIC_MTP_MOE_OP = "custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2"
BF16_LINEAR_F32_OP = "custom_deepseek_v41_bf16_linear_f32_gaudi2"
BF16_IDENTITY_OP = "custom_deepseek_v41_bf16_identity_gaudi2"
ATTENTION_NORM_BF16_OP = "custom_deepseek_v41_attention_norm_bf16_gaudi2"

SOURCE_FILES = (
    "tools/check_deepseek_v41_dspark_packed_transaction.py",
    "tests/standalone/deepseek_v41/test_dspark_packed_transaction_tool.py",
    "vllm_gaudi/ops/deepseek_v41_batch.py",
    "vllm_gaudi/ops/deepseek_v41_batch_attention.py",
    "vllm_gaudi/ops/deepseek_v41_batch_draft.py",
    "vllm_gaudi/ops/deepseek_v41_batch_speculation.py",
    "vllm_gaudi/ops/deepseek_v41_batch_verify.py",
    "vllm_gaudi/ops/deepseek_v41_math.py",
    "vllm_gaudi/ops/deepseek_v41_paged_attention.py",
    "vllm_gaudi/ops/deepseek_v41_shard_loader.py",
    "vllm_gaudi/ops/deepseek_v41_verify.py",
    "vllm_gaudi/entrypoints/deepseek_v41.py",
    "vllm_gaudi/models/deepseek_v41_program.py",
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
    "VLLM_HPU_DSV41_DSPARK_HEAD_INPUT_BOUNDARY",
    "VLLM_HPU_DSV41_DSPARK_MTP_CACHE",
    "VLLM_HPU_DSV41_DSPARK_MTP_SAT",
    "VLLM_HPU_DSV41_DSPARK_MTP_FP8",
    "VLLM_HPU_DSV41_DSPARK_MTP_K128",
    "VLLM_HPU_DSV41_DSPARK_DRAFT_MHC",
    "VLLM_HPU_DSV41_DSPARK_SHARED_FP8",
    "VLLM_HPU_DSV41_BF16_LM_HEAD",
    "VLLM_HPU_DSV4_TPC_OP_LIBRARY",
    "VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR",
    "GC_KERNEL_PATH",
)


@dataclass(frozen=True)
class LanePlan:
    """Expected greedy result for one fixed lane."""

    active: bool
    target_count: int = 0
    proposal_count: int = 0
    accepted_prefix: int = 0

    def validate(self):
        if not self.active:
            if (self.target_count, self.proposal_count, self.accepted_prefix) != (0, 0, 0):
                raise ValueError("inactive lanes must have zero target/proposal/accept counts")
            return
        if not 1 <= self.target_count <= MAX_TARGET_QUERIES:
            raise ValueError("active target_count must be in 1..6")
        if not 0 <= self.proposal_count <= DRAFT_TOKENS:
            raise ValueError("active proposal_count must be in 0..5")
        if self.proposal_count == 0:
            if self.target_count != 1 or self.accepted_prefix != 0:
                raise ValueError("C1 lanes require target_count=1 and accepted_prefix=0")
            return
        if self.target_count != self.proposal_count + 1:
            raise ValueError("C6 lanes require target_count == proposal_count + 1")
        if not 0 <= self.accepted_prefix <= self.proposal_count:
            raise ValueError("accepted_prefix must be within the proposal count")


@dataclass(frozen=True)
class Scenario:
    name: str
    lanes: tuple[LanePlan, ...]

    def validate(self, capacity: int):
        if len(self.lanes) != capacity:
            raise ValueError(f"scenario {self.name} does not contain {capacity} lanes")
        for lane in self.lanes:
            lane.validate()
        if not any(lane.active for lane in self.lanes):
            raise ValueError("a scenario must contain at least one active lane")


@dataclass(frozen=True)
class RoundInputs:
    target_ids: torch.Tensor
    target_states: torch.Tensor
    positions: torch.Tensor
    request_slots: torch.Tensor
    slot_generations: torch.Tensor
    control: torch.Tensor
    binding_sidecar: torch.Tensor
    proposals: torch.Tensor
    active: torch.Tensor
    target_counts: tuple[int, ...]
    proposal_counts: tuple[int, ...]
    round_generation: int
    host_request_slots: tuple[int, ...]
    host_slot_generations: tuple[int, ...]
    host_active: tuple[bool, ...]

    def args(self):
        return (
            self.target_ids,
            self.target_states,
            self.positions,
            self.request_slots,
            self.control,
            self.binding_sidecar,
            self.proposals,
            self.active,
        )


@dataclass(frozen=True)
class PackedProbe:
    record: torch.Tensor
    output: torch.Tensor
    committed: torch.Tensor
    output_count: torch.Tensor
    proposals: torch.Tensor
    confidence: torch.Tensor
    status: torch.Tensor
    draft_active: torch.Tensor
    anchors: torch.Tensor
    positions: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor


def _lanes(capacity: int, active: dict[int, LanePlan]) -> tuple[LanePlan, ...]:
    return tuple(active.get(lane, LanePlan(False)) for lane in range(capacity))


def scenarios_for_capacity(capacity: int) -> tuple[Scenario, ...]:
    """Return the non-duplicated fixed-lane matrix for N=1, N=2 and N=4."""
    if capacity not in SUPPORTED_BATCH_CAPACITIES:
        raise ValueError(f"unsupported transaction capacity: {capacity}")
    if capacity == 1:
        scenarios = (
            Scenario("n1_full_c6", _lanes(1, {0: LanePlan(True, 6, 5, 5)})),
            Scenario("n1_partial_c3", _lanes(1, {0: LanePlan(True, 6, 5, 2)})),
        )
    elif capacity == 2:
        scenarios = (
            Scenario(
                "n2_full_mixed_partial",
                _lanes(2, {
                    0: LanePlan(True, 6, 5, 5),
                    1: LanePlan(True, 6, 5, 2),
                }),
            ),
            Scenario(
                "n2_partial_a1_lane1",
                _lanes(2, {1: LanePlan(True, 6, 5, 3)}),
            ),
        )
    else:
        scenarios = (
            Scenario(
                "n4_full_mixed_c1_c6_partial",
                _lanes(4, {
                    0: LanePlan(True, 6, 5, 5),
                    1: LanePlan(True, 1, 0, 0),
                    2: LanePlan(True, 6, 5, 2),
                    3: LanePlan(True, 6, 5, 5),
                }),
            ),
            Scenario(
                "n4_partial_a2_lane_slots",
                _lanes(4, {
                    0: LanePlan(True, 6, 5, 5),
                    2: LanePlan(True, 6, 5, 1),
                }),
            ),
            Scenario(
                "n4_partial_a1_lane1",
                _lanes(4, {1: LanePlan(True, 6, 5, 2)}),
            ),
        )
    for scenario in scenarios:
        scenario.validate(capacity)
    return scenarios


def bind_all_lanes(capacity: int, prefix: str) -> tuple[RequestSlots, list[object]]:
    """Occupy every physical lane so a released owner must return to its lane."""
    slots = RequestSlots(capacity)
    owners = [slots.acquire(f"{prefix}-lane-{lane}") for lane in range(capacity)]
    if [owner.index for owner in owners] != list(range(capacity)):
        raise RuntimeError("request ownership did not bind lane i to slot i")
    return slots, owners


def make_round_inputs(
    capacity: int,
    scenario: Scenario,
    owners: Sequence[object],
    *,
    round_generation: int,
    seed: int,
    max_model_len: int = MAX_MODEL_LENGTH,
    device="cpu",
) -> RoundInputs:
    """Build fixed-N transaction inputs without reading a target model."""
    scenario.validate(capacity)
    if len(owners) != capacity:
        raise ValueError("one live request owner is required for every physical lane")
    if not 1 <= round_generation <= (1 << 31) - 1:
        raise ValueError("round generation must be positive int32")
    if max_model_len < 1024:
        raise ValueError("max_model_len must cover the fixed scenario positions")

    generator = torch.Generator().manual_seed(seed + round_generation * 1009)
    states = torch.randn(
        (capacity, TARGET_QUERY_BUCKET, TARGET_CONTEXT_WIDTH), generator=generator, dtype=torch.bfloat16)
    target_rows = torch.zeros((capacity, TARGET_QUERY_BUCKET), dtype=torch.int64)
    proposal_rows = torch.full((capacity, DRAFT_TOKENS), -1, dtype=torch.int64)
    position_rows = torch.zeros((capacity, TARGET_QUERY_BUCKET), dtype=torch.int32)
    starts = [0] * capacity
    slots = [-1] * capacity
    generations = [0] * capacity
    target_counts = [0] * capacity
    proposal_counts = [0] * capacity
    budgets = [0] * capacity
    need_sample = [0] * capacity
    active_values = [lane.active for lane in scenario.lanes]

    for lane, plan in enumerate(scenario.lanes):
        if not plan.active:
            continue
        start = 128 + 256 * lane
        starts[lane] = start
        slots[lane] = owners[lane].index
        generations[lane] = owners[lane].generation
        target_counts[lane] = plan.target_count
        proposal_counts[lane] = plan.proposal_count
        budgets[lane] = 64
        need_sample[lane] = 1
        tokens = torch.arange(start, start + TARGET_QUERY_BUCKET, dtype=torch.int64)
        target_rows[lane].copy_(tokens)
        position_rows[lane].copy_(torch.arange(start, start + TARGET_QUERY_BUCKET, dtype=torch.int32))
        for index in range(plan.proposal_count):
            proposal_rows[lane, index] = tokens[index] if index < plan.accepted_prefix else tokens[index] + 10007

    states = torch.where(torch.tensor(active_values).reshape(capacity, 1, 1), states, torch.zeros_like(states))
    active = torch.tensor(active_values, dtype=torch.bool, device=device)
    control_host = batch_control_metadata(
        round_generation=round_generation,
        request_slots=slots,
        slot_generations=generations,
        target_counts=target_counts,
        proposal_counts=proposal_counts,
        remaining_output_budget=budgets,
        target_start_positions=starts,
        max_model_len=max_model_len,
        need_sample=need_sample,
        active=active_values,
    )
    binding_sidecar = batch_binding_sidecar(control_host).to(device)
    control = control_host.to(device)
    return RoundInputs(
        target_ids=target_rows.to(device),
        target_states=states.to(device),
        positions=position_rows.to(device),
        request_slots=torch.tensor(slots, dtype=torch.int32, device=device),
        slot_generations=torch.tensor(generations, dtype=torch.int64, device=device),
        control=control,
        binding_sidecar=binding_sidecar,
        proposals=proposal_rows.to(device),
        active=active,
        target_counts=tuple(target_counts),
        proposal_counts=tuple(proposal_counts),
        round_generation=round_generation,
        host_request_slots=tuple(slots),
        host_slot_generations=tuple(generations),
        host_active=tuple(active_values),
    )


class _PackedEntry(torch.nn.Module):
    def __init__(self, transaction: PackedDraftTransaction):
        super().__init__()
        self.transaction = transaction

    def forward(
        self,
        target_ids,
        target_states,
        positions,
        request_slots,
        control,
        binding_sidecar,
        proposals,
        active,
    ):
        return self.transaction(
            target_ids, target_states, positions, request_slots, control, proposals, active, binding_sidecar)


class _SerialVerifyContextEntry(torch.nn.Module):
    """Production serial C6 verify plus accepted-prefix context commit."""

    def __init__(self, draft):
        super().__init__()
        self.draft = draft

    def forward(self, target_ids, target_states, positions, control, proposals):
        target_ids = target_ids.reshape(-1)
        target_states = target_states.reshape(-1, target_states.shape[-1])
        positions = positions.reshape(-1)
        proposals = proposals.reshape(-1)
        _, output, committed, output_count, anchor, draft_enabled, status = verify_control_from_target(
            target_ids, proposals, control)
        self.draft.insert_context(target_states, positions, committed)
        draft_positions = positions[0].to(torch.int32) + committed.to(torch.int32) + self.draft.offsets
        return output, committed, output_count, anchor, draft_enabled, status, draft_positions


class _SerialMTPEntry(torch.nn.Module):
    """Serial C5 MTP body and five-step Markov sampler for one request."""

    def __init__(self, draft):
        super().__init__()
        self.draft = draft

    def forward(self, anchor, draft_positions):
        hidden, logits = self.draft.forward_local(anchor.reshape(1), draft_positions)
        token_ids, confidence = self.draft.sample_greedy_local(anchor.reshape(1), hidden, logits)
        return hidden, logits, token_ids, confidence


class _PackedStageProbeEntry(torch.nn.Module):
    """Fixed-shape compiled decomposition of the production packed stages."""

    def __init__(self, transaction: PackedDraftTransaction):
        super().__init__()
        self.transaction = transaction

    def forward(
        self,
        target_ids,
        target_states,
        positions,
        request_slots,
        control,
        binding_sidecar,
        proposals,
        active,
    ):
        _, output, committed, output_count, _, draft_enabled, status = verify_control_from_target_batch(
            target_ids, proposals, control, active, binding_sidecar)
        valid_context = active & (status == 0)
        context_done = insert_context_batch(
            self.transaction.draft,
            self.transaction.batch_state,
            target_states,
            positions,
            request_slots,
            committed,
            valid_context,
        )
        draft_active = draft_enabled & valid_context
        anchors, draft_positions = draft_inputs_after_commit(
            positions, output, output_count, committed, draft_active)
        hidden, logits = self.transaction.round.backbone(
            anchors.reshape(self.transaction.batch_capacity, 1),
            draft_positions,
            request_slots,
            draft_active,
            context_done=context_done,
        )
        sampled, confidence = self.transaction.round.markov(anchors, hidden, logits, draft_active)
        record = pack_record_batch(
            control, committed, output_count, output, sampled, draft_active, status, binding_sidecar)
        return (
            record,
            output,
            committed,
            output_count,
            sampled,
            confidence,
            status,
            draft_active,
            anchors,
            draft_positions,
            hidden,
            logits,
        )


def packed_stage_probe(entry: _GraphCallCounter, inputs: RoundInputs) -> PackedProbe:
    """Run the fixed probe through the same compiled entry as qualification."""
    return PackedProbe(*entry(*inputs.args()))


def _lookup_backend(name):
    from torch._dynamo.backends.registry import lookup_backend

    return lookup_backend(name)


class _CountingBackend:
    """Count actual backend callbacks, including recompilations."""

    def __init__(self, backend):
        self.backend = backend
        self.compiles = 0

    def __call__(self, graph, inputs):
        self.compiles += 1
        return self.backend(graph, inputs)


@lru_cache(maxsize=None)
def _shared_counting_backend(name):
    # A single wrapper identity prevents BACKEND_MATCH churn between the packed
    # graph and the independent serial oracles.
    return _CountingBackend(_lookup_backend(name))


def _compile(module, backend):
    return torch.compile(module, backend=backend, fullgraph=True, dynamic=False)


class _GraphCallCounter:
    def __init__(self, module, *, backend=None, backend_name="hpu_backend"):
        self.module = module
        self.backend = backend if backend is not None else _shared_counting_backend(backend_name)
        if not isinstance(self.backend, _CountingBackend):
            self.backend = _CountingBackend(self.backend)
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
            self.backend_compiles += self.backend.compiles - before
        self.successful_calls += 1
        return result

    def snapshot(self):
        return {
            "entry_calls": self.successful_calls,
            "failed_entry_calls": self.failed_calls,
            "backend_compiles": self.backend_compiles,
            "entry_replays": max(0, self.successful_calls - self.backend_compiles),
        }


def _graph_usage(counters: Iterable[_GraphCallCounter], before=None):
    snapshots = [counter.snapshot() for counter in counters]
    if before is not None:
        if len(before) != len(snapshots):
            raise ValueError("graph snapshots do not match their counters")
        snapshots = [{
            key: after[key] - prior[key]
            for key in ("entry_calls", "failed_entry_calls", "backend_compiles", "entry_replays")
        } for prior, after in zip(before, snapshots, strict=True)]
    return {
        "entries": len(snapshots),
        "entry_calls": sum(item["entry_calls"] for item in snapshots),
        "failed_entry_calls": sum(item["failed_entry_calls"] for item in snapshots),
        "backend_compiles": sum(item["backend_compiles"] for item in snapshots),
        "entry_replays": sum(item["entry_replays"] for item in snapshots),
        "per_entry": snapshots,
        "counter_semantics": (
            "backend_compiles counts actual hpu_backend callback invocations; "
            "entry_replays is successful compiled-entry calls minus compilations"
        ),
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    bits = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(bits.numpy().tobytes()).hexdigest()


def compare_exact(name: str, reference: torch.Tensor, actual: torch.Tensor):
    """Bitwise comparison after host copy; no tolerance exists."""
    reference = reference.detach().cpu()
    actual = actual.detach().cpu()
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        return {
            "name": name,
            "shape_reference": list(reference.shape),
            "shape_actual": list(actual.shape),
            "dtype_reference": str(reference.dtype),
            "dtype_actual": str(actual.dtype),
            "exact": False,
            "mismatch": None,
            "max_abs": None,
            "criterion": "bitwise_equal_after_host_copy",
            "rtol": 0,
            "atol": 0,
        }
    reference_bits = reference.contiguous().reshape(-1).view(torch.uint8)
    actual_bits = actual.contiguous().reshape(-1).view(torch.uint8)
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
        "numeric_equal": bool(torch.equal(reference, actual)),
        "mismatch": int(changed.sum().item()),
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "criterion": "bitwise_equal_after_host_copy",
        "rtol": 0,
        "atol": 0,
    }


def _sync(device):
    if torch.device(device).type == "hpu":
        torch.hpu.synchronize()


def _copy_rings(destinations, sources):
    if len(destinations) != len(sources):
        raise ValueError("ring sets differ in length")
    for destination, source in zip(destinations, sources, strict=True):
        if destination.shape != source.shape or destination.dtype != source.dtype:
            raise ValueError("ring sets differ in geometry")
        destination.copy_(source)


def _make_ring_seed(seed: int, device, capacity: int):
    from vllm_gaudi.ops.deepseek_v41_math import pack_swa

    rows = []
    for slot in range(capacity):
        generator = torch.Generator().manual_seed(seed + slot * 1009)
        values = torch.randn((SWA_ROWS, 512), generator=generator, dtype=torch.bfloat16)
        rows.append(pack_swa(values))
    return torch.cat(rows, dim=0).to(device)


def _seed_packed_rings(batch_state: DraftBatchState, seed: int, device, capacity: int):
    for layer, ring in enumerate(batch_state.state_tensors()):
        ring.copy_(_make_ring_seed(seed + layer * 10007, device, capacity))


def _seed_serial_lane(draft, ring_before, lane: int):
    for layer, ring in enumerate(ring_before):
        start = lane * SWA_ROWS
        draft.layers[layer].attention.swa.copy_(ring[start:start + SWA_ROWS])


def _readback_round(entry, ring: BatchVerifyRing, inputs: RoundInputs):
    ticket = ring.acquire(
        round_generation=inputs.round_generation,
        request_slots=inputs.host_request_slots,
        slot_generations=inputs.host_slot_generations,
        max_committed=inputs.target_counts,
        active=inputs.host_active,
    )
    try:
        values = entry(*inputs.args())
        record = values[0]
        if not isinstance(record, torch.Tensor) or record.shape != (ring.batch_capacity, BATCH_RECORD_SIZE):
            raise RuntimeError("packed transaction did not return a fixed [N,16] record")
        physical_ticket_slot = ticket.slot
        before_copy = ring.copy_count
        before_sync = ring.completion_sync_count
        ticket.record.copy_(record)
        ring.stage(ticket)
        commit = ring.consume(ticket)
        ring.release(ticket)
        ticket = None
        return values, commit, {
            "record_copy_count": ring.copy_count - before_copy,
            "completion_sync_count": ring.completion_sync_count - before_sync,
            "physical_ticket_slot": physical_ticket_slot,
        }
    except BaseException:
        if ticket is not None:
            try:
                ring.abort(ticket)
            except BaseException as abort_error:
                traceback.print_exception(abort_error)
        raise


def _inactive_ring_checks(before, after, active, capacity: int):
    checks = []
    active_host = active.detach().cpu().bool()
    for layer, (prior, current) in enumerate(zip(before, after, strict=True)):
        for lane in range(capacity):
            if bool(active_host[lane].item()):
                continue
            start = lane * SWA_ROWS
            checks.append(compare_exact(
                f"ring{layer}_inactive_lane{lane}",
                prior[start:start + SWA_ROWS],
                current[start:start + SWA_ROWS],
            ))
    return checks


def _poison_inactive_slots(rings, active, capacity: int, seed: int, device):
    active_host = active.detach().cpu().bool()
    poisoned = tuple(ring.clone() for ring in rings)
    alternate = _make_ring_seed(seed, device, capacity)
    for lane in range(capacity):
        if bool(active_host[lane].item()):
            continue
        start = lane * SWA_ROWS
        for ring in poisoned:
            ring[start:start + SWA_ROWS].copy_(alternate[start:start + SWA_ROWS])
    return poisoned


def _replay_input_checks(reference: RoundInputs, replay: RoundInputs, prefix: str):
    """Prove a poison replay changed only inactive persisted state, not inputs."""
    checks = {}
    tensor_fields = (
        "target_ids",
        "target_states",
        "positions",
        "request_slots",
        "slot_generations",
        "control",
        "binding_sidecar",
        "proposals",
        "active",
    )
    for name in tensor_fields:
        checks[name] = compare_exact(
            f"{prefix}_{name}", getattr(reference, name), getattr(replay, name))
    metadata_fields = {
        "target_counts": (reference.target_counts, replay.target_counts, torch.int64),
        "proposal_counts": (reference.proposal_counts, replay.proposal_counts, torch.int64),
        "round_generation": ((reference.round_generation, ), (replay.round_generation, ), torch.int64),
        "host_request_slots": (reference.host_request_slots, replay.host_request_slots, torch.int64),
        "host_slot_generations": (reference.host_slot_generations, replay.host_slot_generations, torch.int64),
        "host_active": (reference.host_active, replay.host_active, torch.bool),
    }
    for name, (prior, current, dtype) in metadata_fields.items():
        checks[name] = compare_exact(
            f"{prefix}_{name}", torch.tensor(prior, dtype=dtype), torch.tensor(current, dtype=dtype))
    return checks


def _record_fields(result) -> dict:
    return {
        "record": result[0],
        "committed": result[1],
        "output": result[2],
        "output_count": result[3],
        "proposals": result[4],
        "status": result[5],
    }


def _normalize_status_for_exact(value: torch.Tensor) -> torch.Tensor:
    """Canonicalize the known I32/I64 status representations without changing values."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("status must be a tensor")
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"status must be int32 or int64, got {value.dtype}")
    return value.detach().cpu().to(torch.int64)


def _singleton_for_exact(
    name: str,
    value: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Normalize only singleton shape representation; never values or element bits."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.numel() != 1:
        raise ValueError(f"{name} must contain exactly one element, got shape {tuple(value.shape)}")
    if dtype is not None:
        value = value.to(dtype)
    return value.reshape(1)


def _compare_singleton_exact(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
):
    return compare_exact(
        name,
        _singleton_for_exact(f"{name}_reference", reference, dtype=dtype),
        _singleton_for_exact(f"{name}_actual", actual, dtype=dtype),
    )


def _compare_result_fields(prefix: str, reference, actual, *, lane: int | None = None):
    """Compare complete packed results, or one serial oracle against its packed lane."""
    reference_fields = _record_fields(reference)
    actual_fields = _record_fields(actual)
    if lane is not None:
        if isinstance(lane, bool) or not isinstance(lane, int) or lane < 0:
            raise ValueError("result lane must be a non-negative integer")
        actual_fields = {
            name: value[lane] for name, value in actual_fields.items()
        }
    checks = {}
    for name in ("record", "committed", "output", "output_count", "proposals", "status"):
        reference_value = reference_fields[name]
        actual_value = actual_fields[name]
        if name == "status":
            reference_value = _normalize_status_for_exact(reference_value)
            actual_value = _normalize_status_for_exact(actual_value)
        if lane is not None and name in ("committed", "output_count", "status"):
            checks[name] = compare_exact(
                f"{prefix}_{name}",
                _singleton_for_exact(f"{prefix}_{name}_reference", reference_value),
                _singleton_for_exact(f"{prefix}_{name}_actual", actual_value),
            )
        else:
            checks[name] = compare_exact(
                f"{prefix}_{name}", reference_value, actual_value)
    return checks


def _serial_lane_result(serial_draft, verify_entry, mtp_entry, inputs: RoundInputs, lane: int, ring_before):
    _seed_serial_lane(serial_draft, ring_before, lane)
    _sync(inputs.active.device)
    target_ids = inputs.target_ids[lane]
    target_states = inputs.target_states[lane]
    positions = inputs.positions[lane]
    control = inputs.control[lane]
    proposals = inputs.proposals[lane]
    verified = verify_entry(target_ids, target_states, positions, control, proposals)
    output, committed, output_count, anchor, draft_enabled, status, draft_positions = verified
    _sync(inputs.active.device)
    if bool(draft_enabled.detach().cpu().item()):
        hidden, logits, token_ids, confidence = mtp_entry(anchor, draft_positions)
        _sync(inputs.active.device)
    else:
        hidden = logits = confidence = None
        token_ids = torch.full((DRAFT_TOKENS, ), -1, dtype=torch.int64, device=inputs.active.device)
    record = pack_record(
        control,
        committed.reshape(1),
        output_count.reshape(1),
        output,
        token_ids,
        draft_enabled.reshape(1),
        status.reshape(1),
    )
    ring_after = tuple(layer.attention.swa.clone() for layer in serial_draft.layers)
    return {
        "output": output,
        "committed": committed,
        "output_count": output_count,
        "anchor": anchor,
        "draft_enabled": draft_enabled,
        "status": status,
        "draft_positions": draft_positions,
        "hidden": hidden,
        "logits": logits,
        "token_ids": token_ids,
        "confidence": confidence,
        "record": record,
        "ring_after": ring_after,
    }


def _record_probe_prefix(probe: PackedProbe):
    return (
        probe.record,
        probe.committed,
        probe.output,
        probe.output_count,
        probe.proposals,
        probe.status,
    )


def _qualify_scenario(
    *,
    scenario: Scenario,
    capacity: int,
    packed_entry: _GraphCallCounter,
    packed_probe_entry: _GraphCallCounter,
    transaction: PackedDraftTransaction,
    serial_drafts,
    serial_verify_entries,
    serial_mtp_entries,
    batch_state: DraftBatchState,
    ring: BatchVerifyRing,
    owners,
    seed: int,
    device,
    max_model_len: int,
):
    ring_before = tuple(item.clone() for item in batch_state.state_tensors())
    _seed_packed_rings(batch_state, seed, device, capacity)
    _sync(device)
    ring_before = tuple(item.clone() for item in batch_state.state_tensors())
    inputs = make_round_inputs(
        capacity,
        scenario,
        owners,
        round_generation=1,
        seed=seed,
        max_model_len=max_model_len,
        device=device,
    )

    graph_before = packed_entry.snapshot()
    actual, commit, readback = _readback_round(packed_entry, ring, inputs)
    ring_after = tuple(item.clone() for item in batch_state.state_tensors())
    graph_first = _graph_usage([packed_entry], before=[graph_before])

    _copy_rings(batch_state.state_tensors(), ring_before)
    _sync(device)
    probe_before = packed_probe_entry.snapshot()
    probe = packed_stage_probe(packed_probe_entry, inputs)
    _sync(device)
    ring_probe = tuple(item.clone() for item in batch_state.state_tensors())
    probe_usage = _graph_usage([packed_probe_entry], before=[probe_before])
    _copy_rings(batch_state.state_tensors(), ring_after)
    _sync(device)

    result_checks = _compare_result_fields("probe_vs_packed", _record_probe_prefix(probe), actual)
    context_checks = [
        compare_exact(f"ring{layer}_probe_vs_packed", prior, current)
        for layer, (prior, current) in enumerate(zip(ring_probe, ring_after, strict=True))
    ]

    lane_reports = []
    actual_values = {
        "record": actual[0],
        "committed": actual[1],
        "output": actual[2],
        "output_count": actual[3],
        "proposals": actual[4],
        "status": actual[5],
    }
    for lane, plan in enumerate(scenario.lanes):
        lane_start = lane * SWA_ROWS
        lane_end = lane_start + SWA_ROWS
        lane_report = {
            "lane": lane,
            "active": plan.active,
            "request_slot": int(inputs.request_slots[lane].detach().cpu().item()),
            "slot_generation": int(inputs.slot_generations[lane].detach().cpu().item()),
            "expected_target_count": plan.target_count,
            "expected_proposal_count": plan.proposal_count,
            "expected_accepted_prefix": plan.accepted_prefix,
            "checks": [],
            "exact": True,
        }
        if not plan.active:
            inactive_checks = [
                compare_exact(
                    f"lane{lane}_inactive_record",
                    torch.tensor([0, 0, 0, 0, *([-1] * 11), BATCH_INACTIVE], dtype=torch.int64),
                    actual_values["record"][lane],
                ),
                compare_exact(
                    f"lane{lane}_inactive_proposals",
                    torch.full((DRAFT_TOKENS, ), -1, dtype=torch.int64),
                    actual_values["proposals"][lane],
                ),
            ]
            for layer, (before, after) in enumerate(zip(ring_before, ring_after, strict=True)):
                inactive_checks.append(compare_exact(
                    f"lane{lane}_ring{layer}_unchanged", before[lane_start:lane_end], after[lane_start:lane_end]))
            lane_report["checks"] = inactive_checks
            lane_report["readback"] = {
                "committed": commit.committed[lane],
                "output": list(commit.output[lane]),
                "draft": list(commit.draft[lane]),
            }
            lane_report["exact"] = all(check["exact"] for check in inactive_checks)
            lane_reports.append(lane_report)
            continue

        serial = _serial_lane_result(
            serial_drafts[lane],
            serial_verify_entries[lane],
            serial_mtp_entries[lane],
            inputs,
            lane,
            ring_before,
        )
        serial_fields = (
            serial["record"],
            serial["committed"],
            serial["output"],
            serial["output_count"],
            serial["token_ids"],
            serial["status"],
        )
        lane_checks = _compare_result_fields(
            f"lane{lane}_serial_vs_packed", serial_fields, actual, lane=lane)
        lane_checks["lane_serial_vs_probe"] = compare_exact(
            f"lane{lane}_serial_vs_probe_record", serial["record"], probe.record[lane])
        lane_checks["committed_expected"] = compare_exact(
            f"lane{lane}_committed_expected",
            torch.tensor(plan.accepted_prefix + 1, dtype=torch.int64, device=actual_values["committed"].device),
            actual_values["committed"][lane],
        )
        lane_checks["anchor"] = _compare_singleton_exact(
            f"lane{lane}_anchor", serial["anchor"], probe.anchors[lane])
        lane_checks["draft_positions"] = compare_exact(
            f"lane{lane}_draft_positions", serial["draft_positions"], probe.positions[lane])
        lane_checks["draft_enabled"] = compare_exact(
            f"lane{lane}_draft_enabled",
            serial["draft_enabled"].reshape(1),
            probe.draft_active[lane].reshape(1),
        )
        for layer, serial_ring in enumerate(serial["ring_after"]):
            lane_checks[f"ring{layer}_serial_vs_packed"] = compare_exact(
                f"lane{lane}_ring{layer}_serial_vs_packed",
                serial_ring,
                ring_after[layer][lane_start:lane_end],
            )
        if serial["token_ids"] is not None:
            lane_checks["packed_probe_proposals"] = compare_exact(
                f"lane{lane}_packed_probe_proposals", serial["token_ids"], probe.proposals[lane])
            lane_checks["hidden"] = compare_exact(
                f"lane{lane}_hidden", serial["hidden"], probe.hidden[lane])
            lane_checks["logits"] = compare_exact(
                f"lane{lane}_logits", serial["logits"], probe.logits[lane])
            lane_checks["confidence"] = compare_exact(
                f"lane{lane}_confidence", serial["confidence"], probe.confidence[lane])
        lane_report["readback"] = {
            "committed": commit.committed[lane],
            "output": list(commit.output[lane]),
            "draft": list(commit.draft[lane]),
        }
        lane_report["shapes"] = {
            "target_ids": list(inputs.target_ids[lane].shape),
            "target_states": list(inputs.target_states[lane].shape),
            "positions": list(inputs.positions[lane].shape),
            "control": list(inputs.control[lane].shape),
            "proposals": list(inputs.proposals[lane].shape),
            "hidden": None if serial["hidden"] is None else list(serial["hidden"].shape),
            "logits": None if serial["logits"] is None else list(serial["logits"].shape),
        }
        lane_report["checks"] = list(lane_checks.values())
        lane_report["exact"] = all(check["exact"] for check in lane_checks.values())
        lane_reports.append(lane_report)

    inactive_checks = _inactive_ring_checks(ring_before, ring_after, inputs.active, capacity)
    poison_replay = None
    if inactive_checks:
        poisoned = _poison_inactive_slots(ring_after, inputs.active, capacity, seed + 500003, device)
        _copy_rings(batch_state.state_tensors(), poisoned)
        _sync(device)
        # The poison replay must isolate inactive persisted state. Reuse the
        # first-round inputs exactly, including round_generation=1; a fresh
        # BatchVerifyRing is required because a ring rejects a repeated logical
        # generation even after its previous ticket was released.
        replay_inputs = inputs
        replay_input_checks = _replay_input_checks(
            inputs, replay_inputs, "inactive_poison_replay_input")
        replay_ring = BatchVerifyRing(
            ring.device,
            batch_capacity=capacity,
            size=len(ring.records),
            native_readback=ring.native_readback,
            owner_validator=ring.owner_validator,
        )
        replay_before = packed_entry.snapshot()
        try:
            replay, replay_commit, replay_readback = _readback_round(packed_entry, replay_ring, replay_inputs)
        finally:
            replay_ring.close()
        ring_replay = tuple(item.clone() for item in batch_state.state_tensors())
        replay_usage = _graph_usage([packed_entry], before=[replay_before])
        poison_checks = dict(replay_input_checks)
        for field_index, field_name in ((1, "committed"), (2, "output"), (3, "output_count"), (4, "proposals"),
                                        (5, "status")):
            poison_checks[field_name] = compare_exact(
                f"inactive_poison_replay_{field_name}",
                actual[field_index],
                replay[field_index],
            )
        poison_checks["record"] = compare_exact(
            "inactive_poison_replay_record",
            actual[0],
            replay[0],
        )
        for layer, (first, second) in enumerate(zip(ring_after, ring_replay, strict=True)):
            for lane, plan in enumerate(scenario.lanes):
                start = lane * SWA_ROWS
                stop = start + SWA_ROWS
                if plan.active:
                    poison_checks[f"ring{layer}_active_lane{lane}"] = compare_exact(
                        f"ring{layer}_active_lane{lane}_replay", first[start:stop], second[start:stop])
                else:
                    poison_checks[f"ring{layer}_inactive_lane{lane}"] = compare_exact(
                        f"ring{layer}_inactive_lane{lane}_poison", poisoned[layer][start:stop], second[start:stop])
        poison_replay = {
            "inactive_lanes": [lane for lane, plan in enumerate(scenario.lanes) if not plan.active],
            "input_checks": list(replay_input_checks.values()),
            "checks": list(poison_checks.values()),
            "graph_runtime": replay_usage,
            "readback": replay_readback,
            "exact": all(check["exact"] for check in poison_checks.values()),
        }

    scenario_checks = [
        *result_checks.values(),
        *context_checks,
        *inactive_checks,
        *[check for lane in lane_reports for check in lane["checks"]],
    ]
    if poison_replay is not None:
        scenario_checks.extend(poison_replay["checks"])
    report = {
        "scenario": scenario.name,
        "batch_capacity": capacity,
        "active_requests": sum(plan.active for plan in scenario.lanes),
        "active_lanes": [lane for lane, plan in enumerate(scenario.lanes) if plan.active],
        "inactive_lanes": [lane for lane, plan in enumerate(scenario.lanes) if not plan.active],
        "slots": inputs.request_slots.detach().cpu().tolist(),
        "slot_generations": inputs.slot_generations.detach().cpu().tolist(),
        "round_generation": inputs.round_generation,
        "physical_rows": {
            "target": capacity * TARGET_QUERY_BUCKET,
            "draft": capacity * DRAFT_TOKENS,
            "active_target": sum(plan.target_count for plan in scenario.lanes),
            "active_draft": sum(DRAFT_TOKENS if plan.active else 0 for plan in scenario.lanes),
        },
        "readback": {
            **readback,
            "committed": list(commit.committed),
            "output": [list(row) for row in commit.output],
            "draft": [list(row) for row in commit.draft],
        },
        "graph_runtime": {
            "packed": graph_first,
            "packed_probe": probe_usage,
        },
        "probe_vs_packed": list(result_checks.values()),
        "context_probe_vs_packed": context_checks,
        "lanes": lane_reports,
        "inactive_ring_checks": inactive_checks,
        "inactive_poison_replay": poison_replay,
        "exact": all(check["exact"] for check in scenario_checks),
    }
    return report


def _qualify_slot_reuse(
    *,
    capacity: int,
    packed_entry: _GraphCallCounter,
    batch_state: DraftBatchState,
    seed: int,
    device,
    max_model_len: int,
):
    scenario = Scenario(
        f"n{capacity}_slot_reuse",
        _lanes(capacity, {capacity - 1: LanePlan(True, 6, 5, 3)}),
    )
    scenario.validate(capacity)
    slots, owners = bind_all_lanes(capacity, "transaction-reuse")
    ring = BatchVerifyRing(device, batch_capacity=capacity, owner_validator=slots.validate_batch_identities)
    lane = capacity - 1
    try:
        _seed_packed_rings(batch_state, seed, device, capacity)
        _sync(device)
        first_inputs = make_round_inputs(
            capacity, scenario, owners, round_generation=1, seed=seed,
            max_model_len=max_model_len, device=device)
        ticket = ring.acquire(
            round_generation=1,
            request_slots=first_inputs.host_request_slots,
            slot_generations=first_inputs.host_slot_generations,
            max_committed=first_inputs.target_counts,
            active=first_inputs.host_active,
        )
        values = packed_entry(*first_inputs.args())
        ticket.record.copy_(values[0])
        ring.stage(ticket)
        old_owner = owners[lane]
        slots.release(old_owner)
        replacement = slots.acquire("transaction-reuse-replacement")
        if replacement.index != lane:
            raise RuntimeError("replacement owner did not reuse the fixed lane")
        owners[lane] = replacement
        stale_rejected = False
        stale_error = None
        try:
            ring.consume(ticket)
        except RuntimeError as error:
            stale_rejected = "no longer owns" in str(error)
            stale_error = repr(error)
        if not stale_rejected:
            raise RuntimeError("old ticket was not rejected after slot reuse")
        syncs_before_abort = ring.completion_sync_count
        ring.abort(ticket)
        abort_syncs = ring.completion_sync_count - syncs_before_abort
        ticket = None

        second_inputs = make_round_inputs(
            capacity, scenario, owners, round_generation=2, seed=seed + 1,
            max_model_len=max_model_len, device=device)
        before = packed_entry.snapshot()
        second_values, second_commit, second_readback = _readback_round(packed_entry, ring, second_inputs)
        graph_runtime = _graph_usage([packed_entry], before=[before])
        record_row = second_values[0][lane]
        binding_words = first_inputs.control[lane, 0]
        report = {
            "scenario": scenario.name,
            "batch_capacity": capacity,
            "lane": lane,
            "old_slot_generation": int(old_owner.generation),
            "new_slot_generation": int(replacement.generation),
            "old_ticket_rejected": stale_rejected,
            "old_ticket_error": stale_error,
            "abort_sync_count": abort_syncs,
            "old_record_binding": int(binding_words.detach().cpu().item()),
            "new_record_binding": int(record_row[0].detach().cpu().item()),
            "new_readback": {
                **second_readback,
                "committed": list(second_commit.committed),
                "output": [list(row) for row in second_commit.output],
                "draft": [list(row) for row in second_commit.draft],
            },
            "graph_runtime": graph_runtime,
        }
        report["exact"] = (
            stale_rejected
            and abort_syncs == 1
            and replacement.index == old_owner.index
            and replacement.generation == old_owner.generation + 1
            and second_commit.committed[lane] == 4
            and graph_runtime["backend_compiles"] == 0
        )
        return report
    finally:
        ring.close()
        for owner in owners:
            try:
                slots.release(owner)
            except RuntimeError:
                pass


def _sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, include_sha=True):
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


def _native_artifacts():
    records = {}
    for name in ("VLLM_HPU_DSV4_TPC_OP_LIBRARY", "GC_KERNEL_PATH"):
        value = os.environ.get(name)
        if not value or not Path(value).is_file():
            continue
        records[name] = _file_record(Path(value), include_sha=name != "GC_KERNEL_PATH")
    library = os.environ.get("VLLM_HPU_DSV4_TPC_OP_LIBRARY")
    if library:
        manifest = Path(library).parent / "deepseek_v4_build.json"
        if manifest.is_file():
            records["deepseek_v4_build.json"] = _file_record(manifest)
    return records


def _write_report(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


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
        elements = math.prod(spec["shape"])
        records.append({
            "name": name,
            "dtype": spec["dtype"],
            "shape": list(spec["shape"]),
            "source_bytes": elements * dtype_bytes[spec["dtype"]],
            "runtime_bytes": elements * (2 if spec["dtype"] == "F8_E4M3" else dtype_bytes[spec["dtype"]]),
        })
    return records


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
        "head_input_boundary": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_HEAD_INPUT_BOUNDARY),
        "mtp_cache": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_MTP_CACHE),
        "mtp_sat": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_MTP_SAT),
        "mtp_fp8": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_MTP_FP8),
        "mtp_k128": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_MTP_K128),
        "draft_mhc": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_DRAFT_MHC),
        "shared_fp8": bool(gaudi_envs.VLLM_HPU_DSV41_DSPARK_SHARED_FP8),
        "bf16_head": bool(gaudi_envs.VLLM_HPU_DSV41_BF16_LM_HEAD),
    }


def validate_transaction_flags(flags, *, mtp_k128_eligible: bool):
    checks = (
        ("dspark", True, "DSpark must be enabled"),
        ("draft_kv_decode", True, "packed and serial draft attention must share the selected-KV decoder"),
        ("draft_packed_mla", False, "serial C5 and packed NxC5 do not share packed MLA"),
        ("draft_query_fp8", False, "serial-only query FP8 changes the attention path"),
        ("draft_dense_fp8", False, "dense FP8 handoff is not qualified for 5*N rows"),
        ("wo_handoff", False, "WO handoff is serial-only at the packed row count"),
        ("pretranspose_attn", False, "MTP attention consumes the checkpoint WO layout"),
        ("mtp_cache", False, "cached MTP branches require C1-C6 and bypass packed 5*N rows"),
        ("mtp_sat", False, "SAT MTP branches require C1-C6 and bypass packed 5*N rows"),
        ("mtp_fp8", False, "FP8 MTP branches require C1-C6 and bypass packed 5*N rows"),
        ("draft_mhc", False, "the packed baseline rejects optional draft mHC"),
        ("shared_fp8", False, "shared FP8 is not part of the packed transaction qualification"),
    )
    blockers = []
    for name, expected, reason in checks:
        actual = bool(flags[name])
        if actual != expected:
            blockers.append({"flag": name, "expected": expected, "actual": actual, "reason": reason})
    if flags["native_rope"] and flags["rope_coherent"]:
        blockers.append({
            "flag": "rope_coherent",
            "expected": False,
            "actual": True,
            "reason": "coherent RoPE is serial-only for request_batch=False",
        })
    if flags["mtp_k128"] and mtp_k128_eligible:
        blockers.append({
            "flag": "mtp_k128",
            "expected": False,
            "actual": True,
            "reason": "an eligible K128 MTP checkpoint would use different serial and packed row branches",
        })
    return blockers


def _synth_reduce(value, **kwargs):
    del kwargs
    return value


def _synth_gather(value, dim):
    return torch.cat(tuple(value for _ in range(TP_SIZE)), dim=dim)


def _memory_snapshot(device):
    if torch.device(device).type != "hpu":
        return None
    return {
        "allocated_bytes": torch.hpu.memory_allocated(),
        "reserved_bytes": torch.hpu.memory_reserved(),
        "peak_allocated_bytes": torch.hpu.max_memory_allocated(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path, help="Prepared TP8 PP1 rank directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence path")
    parser.add_argument("--rank", type=int, choices=(0, ), default=0, help="Prepared TP rank (rank 0 only)")
    parser.add_argument("--tp-size", type=int, choices=(8, ), default=8, help="Prepared TP geometry")
    parser.add_argument(
        "--batch-capacity",
        type=int,
        choices=SUPPORTED_BATCH_CAPACITIES,
        default=DEFAULT_BATCH_CAPACITY,
        help="Fixed transaction lane capacity N",
    )
    parser.add_argument("--max-length", type=int, default=MAX_MODEL_LENGTH)
    parser.add_argument("--seed", type=int, default=5107)
    parser.add_argument("--native-dir", type=Path)
    args = parser.parse_args(argv)
    if args.max_length < 1024:
        parser.error("--max-length must cover the fixed 128..1023 scenario positions")
    return args


def main(argv=None):
    args = parse_args(argv)
    report_path = args.output.resolve()
    report = {
        "scope": f"DSpark packed transaction component qualification N={args.batch_capacity}",
        "status": "initializing",
        "output_path": str(report_path),
        "prepared": str(args.prepared.resolve()),
        "rank": args.rank,
        "tp_size": args.tp_size,
        "batch_capacity": args.batch_capacity,
        "synthetic_tp_boundary": {
            "rank_local": True,
            "real_prepared_rank0_weights": True,
            "distributed_process_group": False,
            "hccl_collectives": False,
            "embedding_reduce": "synthetic_identity",
            "moe_reduce": "synthetic_identity",
            "vocab_gather": "synthetic_replicated",
            "not_tp8_service_evidence": True,
        },
        "comparison_contract": {
            "criterion": "bitwise_equal_after_host_copy",
            "rtol": 0,
            "atol": 0,
            "tolerance_relaxed": False,
        },
        "command": {
            "argv": list(sys.argv if argv is None else [sys.argv[0], *argv]),
            "cwd": str(Path.cwd()),
        },
        "checks": [],
        "blockers": [],
        "risks": [],
    }
    _write_report(report_path, report)
    try:
        report["source_fingerprints"] = source_fingerprints()

        from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

        shard = PreparedV41Shard(args.prepared, 0, args.rank)
        if (shard.tensor_parallel_size, shard.pipeline_parallel_size) != (args.tp_size, 1):
            raise ValueError("prepared shard topology is not TP8 PP1")
        specs = {
            name: spec for name, spec in shard.specs.items()
            if name.startswith("mtp.") or name == "head.weight"
        }
        required_prefixes = tuple(f"mtp.{layer}." for layer in range(3))
        loaded_layers = {
            int(name.split(".")[1]) for name in specs
            if name.startswith("mtp.") and name.split(".")[1].isdigit()
        }
        shared_weights = {"mtp.embed.weight", "head.weight"}
        if loaded_layers != {0, 1, 2} or not shared_weights.issubset(specs):
            raise ValueError("transaction workload requires mtp.0/1/2, mtp.embed.weight and head.weight")
        if any(not any(name.startswith(prefix) for prefix in required_prefixes)
               and name not in shared_weights for name in specs):
            raise ValueError("weight filter admitted a tensor outside mtp/head scope")
        context_width = int(specs["mtp.0.main_proj.weight"]["shape"][1])
        if context_width != TARGET_CONTEXT_WIDTH:
            raise ValueError(f"MTP main projection requires context width {context_width}, "
                             f"but this probe builds {TARGET_CONTEXT_WIDTH}")
        report["target_context_width"] = context_width
        tensor_specs = _tensor_specs(specs)
        report["weights"] = {
            "loaded_scope": "mtp.{0,1,2}.*, mtp.embed.weight and head.weight only",
            "full_target_model_loaded": False,
            "tensor_count": len(tensor_specs),
            "source_bytes": sum(item["source_bytes"] for item in tensor_specs),
            "runtime_materialized_bytes": sum(item["runtime_bytes"] for item in tensor_specs),
            "tensors": tensor_specs,
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
        w2 = specs["mtp.0.ffn.experts.w2_q16"]
        mtp_k128_eligible = (
            len(w2["shape"]) == 3 and int(w2["shape"][2]) == 640 * 32
            and int(w2["shape"][0]) == 128
        )
        report["mtp_moe_path"] = {
            "w2_q16_shape": list(w2["shape"]),
            "k128_eligible": mtp_k128_eligible,
            "serial_rows": DRAFT_TOKENS,
            "packed_rows_by_capacity": {
                str(capacity): capacity * DRAFT_TOKENS for capacity in SUPPORTED_BATCH_CAPACITIES
            },
            "known_risk": (
                "N>1 uses the generic prepared MXFP4 MoE for 5*N rows; row-count-independent "
                "numerics against the serial M=5 path were not previously qualified"
            ),
        }
        report["risks"].append(report["mtp_moe_path"]["known_risk"])

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
        blockers = validate_transaction_flags(flags, mtp_k128_eligible=mtp_k128_eligible)
        if blockers:
            report["status"] = "blocked"
            report["blockers"] = blockers
            _write_report(report_path, report)
            return 1

        required_operators = [
            SELECTED_KV_OP,
            SPARSE_ATTN_OP,
            STATE_ROWS_WRITE_OP,
            GENERIC_MTP_MOE_OP,
            BF16_LINEAR_F32_OP,
            BF16_IDENTITY_OP,
        ]
        if flags["native_rope"]:
            required_operators.extend((ROPE_OP, ROPE_INVERSE_OP))
        if flags["head_input_boundary"]:
            required_operators.append(ATTENTION_NORM_BF16_OP)
        load_native_operators(required=tuple(required_operators))
        import habana_frameworks.torch.core  # noqa: F401

        text_config = json.loads((args.prepared / "config.json").read_text())["text_config"]
        from vllm_gaudi.models.deepseek_v41_program import (
            PreparedDraft,
            _weight_tree,
            load_weight_tree,
        )
        from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
        from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2SharedState

        torch.set_num_threads(1)
        torch.hpu.set_device(args.rank)
        weight_tree = _weight_tree(specs)
        load_weight_tree(shard, weight_tree, "hpu", specs)
        shard.check_identity()
        report["native_operators"] = {
            name: hasattr(torch.ops.custom_op, name) for name in required_operators
        }
        if not all(report["native_operators"].values()):
            raise RuntimeError("one or more required native operators are unavailable")

        device = "hpu"
        report["memory"] = {"initial": _memory_snapshot(device)}
        shared = PagedCSA2SharedState(
            text_config, 40, 43, device, args.max_length, tensor_parallel_size=args.tp_size)
        lookup = mxfp4_bf16_lut(torch.device(device))
        stage = SimpleNamespace(
            shard=shard,
            weights=weight_tree,
            bf16_head=flags["bf16_head"],
            tp_rank=args.rank,
            reduce=_synth_reduce,
            all_gather=_synth_gather,
            tensor_parallel_size=args.tp_size,
            config={"text_config": text_config},
            shared=shared,
        )
        packed_draft = PreparedDraft(stage, lookup, device)
        serial_drafts = [PreparedDraft(stage, lookup, device) for _ in range(args.batch_capacity)]
        batch_state = DraftBatchState(packed_draft, capacity=args.batch_capacity)
        transaction = PackedDraftTransaction(
            packed_draft,
            batch_state,
            batch_capacity=args.batch_capacity,
            backend="hpu_backend",
            compile_graph=False,
        )
        packed_entry = _GraphCallCounter(_PackedEntry(transaction))
        packed_probe_entry = _GraphCallCounter(_PackedStageProbeEntry(transaction))
        serial_verify_entries = [
            _GraphCallCounter(_SerialVerifyContextEntry(draft)) for draft in serial_drafts
        ]
        serial_mtp_entries = [
            _GraphCallCounter(_SerialMTPEntry(draft)) for draft in serial_drafts
        ]
        all_counters = [
            packed_entry,
            packed_probe_entry,
            *serial_verify_entries,
            *serial_mtp_entries,
        ]
        initial_snapshots = [counter.snapshot() for counter in all_counters]
        report["memory"]["after_build"] = _memory_snapshot(device)

        for index, scenario in enumerate(scenarios_for_capacity(args.batch_capacity)):
            slots, owners = bind_all_lanes(args.batch_capacity, f"transaction-{scenario.name}")
            ring = BatchVerifyRing(
                device,
                batch_capacity=args.batch_capacity,
                owner_validator=slots.validate_batch_identities,
            )
            try:
                scenario_report = _qualify_scenario(
                    scenario=scenario,
                    capacity=args.batch_capacity,
                    packed_entry=packed_entry,
                    packed_probe_entry=packed_probe_entry,
                    transaction=transaction,
                    serial_drafts=serial_drafts,
                    serial_verify_entries=serial_verify_entries,
                    serial_mtp_entries=serial_mtp_entries,
                    batch_state=batch_state,
                    ring=ring,
                    owners=owners,
                    seed=args.seed + index * 100003,
                    device=device,
                    max_model_len=args.max_length,
                )
                report["checks"].append(scenario_report)
            finally:
                ring.close()
                for owner in owners:
                    try:
                        slots.release(owner)
                    except RuntimeError:
                        pass

        report["slot_reuse"] = _qualify_slot_reuse(
            capacity=args.batch_capacity,
            packed_entry=packed_entry,
            batch_state=batch_state,
            seed=args.seed + 700001,
            device=device,
            max_model_len=args.max_length,
        )
        report["graph_runtime"] = _graph_usage(all_counters, before=initial_snapshots)
        report["memory"]["final"] = _memory_snapshot(device)

        closing_source_fingerprints = source_fingerprints()
        report["source_fingerprints_end"] = closing_source_fingerprints
        report["source_stable"] = all(
            record["sha256"] == report["source_fingerprints"][relative]["sha256"]
            for relative, record in closing_source_fingerprints.items()
        )
        if not report["source_stable"]:
            raise RuntimeError("source artifacts changed during packed transaction qualification")
        all_exact = (
            report["checks"]
            and all(item["exact"] for item in report["checks"])
            and report["slot_reuse"]["exact"]
            and report["graph_runtime"]["failed_entry_calls"] == 0
            and all(item["backend_compiles"] <= 1 for item in report["graph_runtime"]["per_entry"])
        )
        report["graph_runtime"]["single_compile_per_entry"] = all(
            item["backend_compiles"] <= 1 for item in report["graph_runtime"]["per_entry"])
        report["status"] = "passed" if all_exact else "mismatch"
        _write_report(report_path, report)
        print(json.dumps({
            "status": report["status"],
            "output": str(report_path),
            "batch_capacity": args.batch_capacity,
            "scenarios": {item["scenario"]: item["exact"] for item in report["checks"]},
            "slot_reuse": report["slot_reuse"]["exact"],
            "not_tp8": True,
        }), flush=True)
        return 0 if report["status"] == "passed" else 1
    except Exception as error:
        if report["status"] not in ("blocked", "mismatch"):
            report["status"] = "failed"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        _write_report(report_path, report)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
