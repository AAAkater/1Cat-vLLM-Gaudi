# SPDX-License-Identifier: Apache-2.0
"""Contracts for compact target attention and reachable query warmup pairs."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_target_queries import (
    pack_target_query_rows,
    restore_target_query_rows,
    target_query_buckets,
    target_query_warmup_cases,
)


@pytest.mark.parametrize("capacity", [1, 3, 6, 11, 32, 64])
def test_query_layout_keeps_real_rows_and_unique_padding(capacity):
    valid = tuple(index % 6 == 0 for index in range(6 * capacity))
    plan = pack_target_query_rows(valid)
    assert plan.query_rows == capacity
    assert plan.query_rows <= plan.bucket_rows <= plan.body_rows
    assert len(set(plan.source_rows)) == len(plan.source_rows) == plan.bucket_rows
    assert {i for i, value in enumerate(valid) if value}.issubset(plan.source_rows)
    assert plan.valid == tuple(valid[i] for i in plan.source_rows)


def test_mixed_c1_c6_pack_has_seven_queries_in_eight_row_bucket():
    valid = tuple(i in (*range(6, 12), 60) for i in range(66))
    plan = pack_target_query_rows(valid)
    assert plan.source_rows == (*range(6, 12), 60, 0)
    assert plan.query_rows == 7
    assert plan.bucket_rows == 8
    values = torch.arange(32).reshape(8, 4)
    restored = restore_target_query_rows(values, torch.tensor(plan.source_rows), torch.tensor(plan.valid), 66)
    assert torch.equal(restored[6:12], values[:6])
    assert torch.equal(restored[60], values[6])
    assert torch.count_nonzero(restored[0]) == 0
    assert torch.count_nonzero(restored[12:60]) == 0


@pytest.mark.parametrize("request_buckets", [(1,), (3,), (1, 2, 4, 8, 11), (32,), (1, 8, 32, 64)])
def test_warmup_cases_cover_exactly_the_reachable_bucket_pairs(request_buckets):
    observed = set()
    for active in range(1, request_buckets[-1] + 1):
        capacity = next(b for b in request_buckets if b >= active)
        for total in range(active, 6 * active + 1):
            query_bucket = next(b for b in target_query_buckets(6 * capacity) if b >= total)
            observed.add((capacity, query_bucket))
    cases = target_query_warmup_cases(request_buckets)
    assert {(capacity, query) for capacity, query, lengths in cases} == observed
    for capacity, query, lengths in cases:
        assert next(b for b in request_buckets if b >= len(lengths)) == capacity
        assert all(1 <= length <= 6 for length in lengths.values())
        assert next(b for b in target_query_buckets(6 * capacity) if b >= sum(lengths.values())) == query


def test_full_body_fallback_keeps_original_row_order():
    valid = (True, False, True, False, False, True)
    plan = pack_target_query_rows(valid, (1, ))
    assert plan.bucket_rows == 6
    assert plan.source_rows == tuple(range(6))
    assert plan.valid == valid


def test_query_inputs_reject_empty_and_device_like_metadata():
    with pytest.raises(ValueError):
        pack_target_query_rows((False, ) * 6)
    with pytest.raises(ValueError):
        pack_target_query_rows(torch.ones(6, dtype=torch.bool))
    with pytest.raises(ValueError):
        target_query_buckets(66, (1, 1, 8))
