# SPDX-License-Identifier: Apache-2.0
"""Actual-query layouts for target attention, independent of C6 body rows."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TargetQueryPlan:
    body_rows: int
    query_rows: int
    bucket_rows: int
    source_rows: tuple[int, ...]
    valid: tuple[bool, ...]


def target_query_buckets(body_rows, buckets=(1, 8, 32, 128)):
    if type(body_rows) is not int or not 1 <= body_rows <= 384:
        raise ValueError("target body rows must be in 1..384")
    if (not isinstance(buckets, (list, tuple)) or not buckets
            or any(type(value) is not int or not 1 <= value <= 384 for value in buckets)
            or len(set(buckets)) != len(buckets)):
        raise ValueError("target query buckets must contain distinct integers in 1..384")
    return tuple(sorted({value for value in buckets if value < body_rows} | {body_rows}))


def pack_target_query_rows(row_valid, buckets=(1, 8, 32, 128)):
    if not isinstance(row_valid, (list, tuple)) or any(type(value) is not bool for value in row_valid):
        raise ValueError("target query packing requires host boolean row ownership")
    available = target_query_buckets(len(row_valid), buckets)
    real = tuple(index for index, valid in enumerate(row_valid) if valid)
    if not real:
        raise ValueError("target query packing requires at least one real query")
    bucket = next(value for value in available if value >= len(real))
    if bucket == len(row_valid):
        source = tuple(range(len(row_valid)))
    else:
        padding = tuple(index for index, valid in enumerate(row_valid) if not valid)
        source = real + padding[:bucket - len(real)]
    return TargetQueryPlan(len(row_valid), len(real), bucket, source, tuple(row_valid[index] for index in source))


def restore_target_query_rows(values, source_rows, valid, body_rows, *, fill=0):
    """Scatter unique execution rows back to the C6 body without padded writes."""
    shape = (values.shape[0], ) + (1, ) * (values.ndim - 1)
    masked = torch.where(valid.reshape(shape), values, torch.full_like(values, fill))
    destination = torch.full((body_rows, *values.shape[1:]), fill, dtype=values.dtype, device=values.device)
    return destination.index_copy(0, source_rows, masked)


def target_query_warmup_cases(request_buckets, query_buckets=(1, 8, 32, 128)):
    """Enumerate only reachable request/query bucket pairs using C1..C6 counts."""
    cases = []
    previous_capacity = 0
    for capacity in request_buckets:
        minimum_active = previous_capacity + 1
        previous_query = 0
        for query_bucket in target_query_buckets(6 * capacity, query_buckets):
            for active in range(minimum_active, capacity + 1):
                total = max(active, previous_query + 1)
                if total <= min(query_bucket, 6 * active):
                    lengths = [1] * active
                    extra = total - active
                    for lane in range(active):
                        added = min(5, extra)
                        lengths[lane] += added
                        extra -= added
                    cases.append((capacity, query_bucket, {lane: count for lane, count in enumerate(lengths)}))
                    break
            previous_query = query_bucket
        previous_capacity = capacity
    return tuple(cases)
