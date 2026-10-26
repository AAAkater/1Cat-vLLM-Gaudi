# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for fixed-capacity DSpark scheduler metadata."""

from dataclasses import replace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata


def _capacity_metadata():
    return DSparkBatchMetadata.build(
        requests=[None, "request-b", None, "request-d"],
        active=[False, True, False, True],
        target_query_lengths=[0, 3, 0, 6],
        draft_token_ids=[[], [11, 12], [], [21, 22, 23, 24, 25]],
        request_slots=[-1, 1, -1, 3],
        slot_generations=[0, 7, 0, 9],
        remaining_output_budget=[0, 4, 0, 6],
        target_padding_mask=[False] * 9,
        target_start_positions=[0, 50, 0, 100],
        round_generation=23,
    )


@pytest.mark.parametrize(
    ("requests", "active", "target_lengths", "drafts", "slots", "generations", "budgets"),
    (
        (["request-a"], [True], [1], [[]], [0], [1], [1]),
        ([None, "request-b"], [False, True], [0, 1], [[], []], [-1, 1], [0, 1], [0, 1]),
        (
            [None, "request-b", None, "request-d"],
            [False, True, False, True],
            [0, 3, 0, 6],
            [[], [11, 12], [], [21, 22, 23, 24, 25]],
            [-1, 1, -1, 3],
            [0, 7, 0, 9],
            [0, 4, 0, 6],
        ),
    ),
)
def test_explicit_active_preserves_small_capacity_buckets(
        requests, active, target_lengths, drafts, slots, generations, budgets):
    metadata = DSparkBatchMetadata.build(
        requests=requests,
        active=active,
        target_query_lengths=target_lengths,
        draft_token_ids=drafts,
        request_slots=slots,
        slot_generations=generations,
        remaining_output_budget=budgets,
        target_start_positions=[0] * len(active),
    )

    assert metadata.capacity == len(active)
    assert metadata.request_active.tolist() == active
    assert len(metadata.active_request_ids) == sum(active)


def test_explicit_active_uses_lane_major_capacity_and_active_aware_indices():
    metadata = _capacity_metadata()

    assert metadata.capacity == 4
    assert metadata.request_ids == (None, "request-b", None, "request-d")
    assert metadata.active_request_ids == ("request-b", "request-d")
    assert metadata.request_active.tolist() == [False, True, False, True]
    assert metadata.request_active.dtype == torch.bool
    assert metadata.request_slots.tolist() == [-1, 1, -1, 3]
    assert metadata.slot_generations.tolist() == [0, 7, 0, 9]
    assert metadata.remaining_output_budget.tolist() == [0, 4, 0, 6]
    assert metadata.target_start_positions.tolist() == [0, 50, 0, 100]
    assert metadata.target_query_start_loc.tolist() == [0, 0, 3, 3, 9]
    assert metadata.draft_query_start_loc.tolist() == [0, 0, 2, 2, 7]
    assert metadata.num_draft_tokens.tolist() == [0, 2, 0, 5]
    assert metadata.cu_num_draft_tokens.tolist() == [0, 2, 2, 7]
    assert metadata.cu_num_sampled_tokens.tolist() == [0, 3, 3, 9]
    assert metadata.draft_token_ids.tolist() == [11, 12, 21, 22, 23, 24, 25]
    assert metadata.logits_indices.tolist() == list(range(9))
    assert metadata.target_logits_indices.tolist() == [0, 1, 3, 4, 5, 6, 7]
    assert metadata.bonus_logits_indices.tolist() == [2, 8]
    assert metadata.target_padding_mask.tolist() == [False] * 9
    assert metadata.draft_padding_mask.tolist() == [False] * 7


def test_explicit_active_accepts_cpu_boolean_tensor_mask():
    metadata = DSparkBatchMetadata.build(
        requests=[None, "request-b"],
        active=torch.tensor([False, True], dtype=torch.bool),
        target_query_lengths=[0, 1],
        draft_token_ids=[[], []],
        request_slots=[-1, 1],
        slot_generations=[0, 3],
        remaining_output_budget=[0, 2],
    )

    assert metadata.request_active.tolist() == [False, True]
    assert metadata.active_request_ids == ("request-b",)


@pytest.mark.parametrize("capacity", (3, 6, 8, 11, 12, 16, 64))
def test_explicit_capacity_keeps_six_n_rows_and_metadata_dynamic(capacity):
    active = [index % 2 == 0 for index in range(capacity)]
    lengths = [1 if enabled and index % 4 == 0 else 6 if enabled else 0
               for index, enabled in enumerate(active)]
    drafts = [
        [] if length <= 1 else list(range(index * 10, index * 10 + length - 1))
        for index, length in enumerate(lengths)
    ]
    metadata = DSparkBatchMetadata.build(
        requests=[f"request-{index}" if enabled else None for index, enabled in enumerate(active)],
        active=active,
        target_query_lengths=lengths,
        draft_token_ids=drafts,
        request_slots=[index if enabled else -1 for index, enabled in enumerate(active)],
        slot_generations=[index + 1 if enabled else 0 for index, enabled in enumerate(active)],
        remaining_output_budget=[32 if enabled else 0 for enabled in active],
        target_start_positions=[10 + index * 20 if enabled else 0
                                for index, enabled in enumerate(active)],
        round_generation=11,
    )
    query_rows = sum(lengths)
    source = torch.arange(query_rows * 3, dtype=torch.int32).reshape(query_rows, 3)

    packed = metadata.pack_target_queries(source)

    assert metadata.capacity == capacity
    assert metadata.target_query_start_loc.tolist()[-1] == query_rows
    assert len(metadata.request_active) == len(metadata.target_start_positions) == capacity
    assert packed.queries.shape == (capacity, 6, 3)
    assert torch.equal(metadata.unpack_target_queries(packed), source)


def test_explicit_active_pack_and_unpack_keep_inactive_lanes_no_write():
    metadata = _capacity_metadata()
    source = torch.arange(18, dtype=torch.int32).reshape(9, 2)

    packed = metadata.pack_target_queries(source)

    assert packed.queries.shape == (4, 6, 2)
    assert not packed.queries[0].any() and not packed.queries[2].any()
    assert torch.equal(packed.queries[1, :3], source[:3])
    assert torch.equal(packed.queries[3], source[3:])
    assert packed.padding_mask.tolist() == [
        [True, True, True, True, True, True],
        [False, False, False, True, True, True],
        [True, True, True, True, True, True],
        [False, False, False, False, False, False],
    ]
    assert packed.real_indices.tolist() == [6, 7, 8, 18, 19, 20, 21, 22, 23]
    assert packed.padding_indices.tolist() == [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    assert packed.logits_indices.tolist() == [6, 7, 8, 18, 19, 20, 21, 22, 23]
    assert packed.target_logits_indices.tolist() == [0, 1, 3, 4, 5, 6, 7]
    assert packed.bonus_logits_indices.tolist() == [2, 8]
    assert torch.equal(metadata.unpack_target_queries(packed), source)

    exposed_inactive = packed.padding_mask.clone()
    exposed_inactive[0, 0] = False
    with pytest.raises(ValueError, match="inactive packed lane"):
        metadata.unpack_target_queries(replace(packed, padding_mask=exposed_inactive))


@pytest.mark.parametrize(
    "overrides",
    (
        {"requests": ["request-a", "request-b", None, "request-d"]},
        {"requests": [None, None, None, "request-d"]},
        {"target_query_lengths": [1, 3, 0, 6]},
        {"draft_token_ids": [[1], [11, 12], [], [21, 22, 23, 24, 25]]},
        {"request_slots": [0, 1, -1, 3]},
        {"slot_generations": [1, 7, 0, 9]},
        {"remaining_output_budget": [1, 4, 0, 6]},
        {"target_start_positions": [1, 50, 0, 100]},
        {"request_slots": [-1, 0, -1, 3]},
        {"slot_generations": [0, 0, 0, 9]},
        {"remaining_output_budget": [0, 0, 0, 6]},
        {"active": [False, False, False, False]},
        {"active": [False, True, False]},
    ),
)
def test_explicit_active_rejects_noncanonical_or_non_sentinel_lanes(overrides):
    values = {
        "requests": [None, "request-b", None, "request-d"],
        "active": [False, True, False, True],
        "target_query_lengths": [0, 3, 0, 6],
        "draft_token_ids": [[], [11, 12], [], [21, 22, 23, 24, 25]],
        "request_slots": [-1, 1, -1, 3],
        "slot_generations": [0, 7, 0, 9],
        "remaining_output_budget": [0, 4, 0, 6],
        "target_padding_mask": [False] * 9,
        "target_start_positions": [0, 50, 0, 100],
        "round_generation": 23,
    }
    with pytest.raises((TypeError, ValueError)):
        DSparkBatchMetadata.build(**{**values, **overrides})


def test_explicit_active_rejects_more_than_contract_lanes_and_non_boolean_mask():
    values = {
        "target_query_lengths": [0] * 65,
        "draft_token_ids": [[] for _ in range(65)],
        "request_slots": [-1] * 65,
        "slot_generations": [0] * 65,
        "remaining_output_budget": [0] * 65,
        "active": [False] * 65,
    }
    with pytest.raises(ValueError, match="batch size"):
        DSparkBatchMetadata.build(requests=[None] * 65, **values)

    values["active"] = torch.tensor([True, False], dtype=torch.int64)
    with pytest.raises(TypeError, match="boolean"):
        DSparkBatchMetadata.build(
            requests=["request-a", None],
            target_query_lengths=[1, 0],
            draft_token_ids=[[], []],
            request_slots=[0, -1],
            slot_generations=[1, 0],
            remaining_output_budget=[1, 0],
            active=values["active"],
        )


def test_legacy_default_keeps_all_active_and_old_zero_generation_rules():
    metadata = DSparkBatchMetadata.build(
        requests=["request-a", "request-b"],
        target_query_lengths=[1, 6],
        draft_token_ids=[[], [11, 12, 13, 14, 15]],
        request_slots=[7, 0],
        slot_generations=[0, 4],
        remaining_output_budget=[0, 2],
        target_start_positions=[4, 20],
    )

    assert metadata.capacity == 2
    assert metadata.request_active.tolist() == [True, True]
    assert metadata.active_request_ids == ("request-a", "request-b")
    assert metadata.request_slots.tolist() == [7, 0]
    assert metadata.slot_generations.tolist() == [0, 4]
    assert metadata.remaining_output_budget.tolist() == [0, 2]
    assert metadata.cu_num_sampled_tokens.tolist() == [1, 7]
    assert metadata.bonus_logits_indices.tolist() == [0, 6]


def test_pack_treats_missing_or_empty_request_active_as_legacy_all_active():
    metadata = DSparkBatchMetadata.build(
        requests=["request-a", "request-b"],
        target_query_lengths=[1, 6],
        draft_token_ids=[[], [11, 12, 13, 14, 15]],
        request_slots=[7, 0],
        slot_generations=[0, 4],
        remaining_output_budget=[0, 2],
    )
    source = torch.arange(14, dtype=torch.int32).reshape(7, 2)

    for legacy_active in (None, torch.empty(0, dtype=torch.bool)):
        legacy = replace(metadata, request_active=legacy_active)
        assert torch.equal(legacy.unpack_target_queries(legacy.pack_target_queries(source)), source)
