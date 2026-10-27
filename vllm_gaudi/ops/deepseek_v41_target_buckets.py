# SPDX-License-Identifier: Apache-2.0
"""Host packing for bounded target execution independent of state capacity."""
from dataclasses import dataclass, replace

import torch

from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata
from vllm_gaudi.ops.deepseek_v41_capacity import validate_dspark_capacity


@dataclass(frozen=True)
class TargetBucketInputs:
    metadata: DSparkBatchMetadata
    queries: torch.Tensor
    pages: torch.Tensor
    source_lanes: tuple[int, ...]
    logical_slots: tuple[int, ...]


def target_execution_buckets(capacity, buckets):
    validate_dspark_capacity(capacity, allow_serial=True, name="state capacity")
    if not isinstance(buckets, (list, tuple)) or not buckets:
        raise ValueError("target warmup buckets must be a nonempty list of powers of two")
    for bucket in buckets:
        if type(bucket) is not int or bucket < 1 or bucket > 64 or bucket & (bucket - 1):
            raise ValueError("target warmup buckets must be powers of two in 1..64")
    return tuple(sorted({bucket for bucket in buckets if bucket < capacity} | {capacity}))


def pack_target_bucket(metadata, queries, pages, execution_capacity):
    capacity = metadata.capacity
    validate_dspark_capacity(execution_capacity, allow_serial=True, name="target execution capacity")
    if queries.device.type != "cpu" or pages.device.type != "cpu":
        raise ValueError("target bucket packing requires host inputs")
    if queries.shape != (capacity, 6) or pages.ndim != 2 or pages.shape[0] != capacity:
        raise ValueError("target bucket inputs must cover the fixed state capacity")
    lanes = tuple(i for i, active in enumerate(metadata.request_active.tolist()) if active)
    if not lanes or len(lanes) > execution_capacity:
        raise ValueError("active requests do not fit the target execution bucket")
    if execution_capacity == capacity:
        return TargetBucketInputs(metadata, queries, pages, tuple(range(capacity)),
                                  tuple(metadata.request_slots.tolist()))
    count = len(lanes)
    pad = execution_capacity - count
    starts = metadata.target_query_start_loc.tolist()
    draft_starts = metadata.draft_query_start_loc.tolist()
    draft_ids = metadata.draft_token_ids.tolist()
    logical_slots = tuple(int(metadata.request_slots[lane]) for lane in lanes) + (-1,) * pad
    compact = DSparkBatchMetadata.build(
        requests=[metadata.request_ids[lane] for lane in lanes] + [None] * pad,
        target_query_lengths=[starts[lane + 1] - starts[lane] for lane in lanes] + [0] * pad,
        draft_token_ids=[draft_ids[draft_starts[lane]:draft_starts[lane + 1]] for lane in lanes] + [[]] * pad,
        request_slots=list(range(count)) + [-1] * pad,
        slot_generations=[int(metadata.slot_generations[lane]) for lane in lanes] + [0] * pad,
        remaining_output_budget=[int(metadata.remaining_output_budget[lane]) for lane in lanes] + [0] * pad,
        target_start_positions=[int(metadata.target_start_positions[lane]) for lane in lanes] + [0] * pad,
        round_generation=int(metadata.round_generation),
        active=[True] * count + [False] * pad,
    )
    compact = replace(compact, request_slots=torch.tensor(logical_slots, dtype=torch.int32, device="cpu"))
    packed_queries = torch.zeros((execution_capacity, 6), dtype=queries.dtype, device="cpu")
    packed_pages = pages[lanes[0]].expand(execution_capacity, -1).clone()
    indices = torch.tensor(lanes, dtype=torch.int64, device="cpu")
    packed_queries[:count].copy_(queries.index_select(0, indices))
    packed_pages[:count].copy_(pages.index_select(0, indices))
    return TargetBucketInputs(compact, packed_queries, packed_pages, lanes + (-1,) * pad, logical_slots)
