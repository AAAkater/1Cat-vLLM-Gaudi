# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for DSpark B2 packed batch metadata."""

import torch

from vllm_gaudi.ops.deepseek_v41_batch_speculation import (
    DSparkBatchMetadata,
    accepted_prefix_lengths,
    num_rejected,
    padded_draft_tokens,
)


def assert_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def assert_type_error(call):
    try:
        call()
    except TypeError:
        return
    raise AssertionError("expected TypeError")


def test_b2_packed_metadata_matches_request_order():
    metadata = DSparkBatchMetadata.build(
        requests=("request-a", "request-b"),
        target_query_lengths=[6, 6],
        draft_token_ids=[[11, 12, 13, 14, 15], [21, 22]],
        request_slots=[0, 1],
        slot_generations=[7, 9],
        remaining_output_budget=[3, 2],
    )

    assert metadata.request_ids == ("request-a", "request-b")
    assert metadata.request_slots.tolist() == [0, 1]
    assert metadata.slot_generations.tolist() == [7, 9]
    assert metadata.target_query_start_loc.tolist() == [0, 6, 12]
    assert metadata.draft_query_start_loc.tolist() == [0, 5, 7]
    assert metadata.num_draft_tokens.tolist() == [5, 2]
    assert metadata.cu_num_draft_tokens.tolist() == [5, 7]
    assert metadata.cu_num_sampled_tokens.tolist() == [6, 9]
    assert metadata.draft_token_ids.tolist() == [11, 12, 13, 14, 15, 21, 22]
    assert metadata.target_logits_indices.tolist() == [0, 1, 2, 3, 4, 6, 7]
    assert metadata.bonus_logits_indices.tolist() == [5, 8]
    assert metadata.logits_indices.tolist() == [0, 1, 2, 3, 4, 5, 9, 10, 11]
    assert metadata.logits_indices[metadata.target_logits_indices].tolist() == [0, 1, 2, 3, 4, 9, 10]
    assert metadata.logits_indices[metadata.bonus_logits_indices].tolist() == [5, 11]
    assert metadata.remaining_output_budget.tolist() == [3, 2]
    assert metadata.target_padding_mask.tolist() == [False] * 12
    assert metadata.draft_padding_mask.tolist() == [False] * 7

    tensors = (
        metadata.request_slots,
        metadata.slot_generations,
        metadata.target_query_start_loc,
        metadata.draft_query_start_loc,
        metadata.num_draft_tokens,
        metadata.cu_num_draft_tokens,
        metadata.cu_num_sampled_tokens,
        metadata.draft_token_ids,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        metadata.logits_indices,
        metadata.remaining_output_budget,
        metadata.target_padding_mask,
        metadata.draft_padding_mask,
    )
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert all(tensor.dtype == expected for tensor, expected in zip(
        tensors,
        (
            torch.int32,
            torch.int64,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.bool,
            torch.bool,
        ),
        strict=True,
    ))


def test_padding_and_greedy_acceptance():
    draft = [[11, 12, 13, 14, 15], [21, 22]]
    padded, valid = padded_draft_tokens(draft)
    target = torch.tensor(
        [
            [11, 12, 13, 14, 15, 99],
            [21, 98, 97, 96, 95, 94],
        ],
        dtype=torch.int32,
    )

    assert padded.tolist() == [
        [11, 12, 13, 14, 15],
        [21, 22, -1, -1, -1],
    ]
    assert valid.tolist() == [
        [True, True, True, True, True],
        [True, True, False, False, False],
    ]
    assert accepted_prefix_lengths(target, padded).tolist() == [6, 2]
    assert num_rejected(torch.tensor([5, 2]), torch.tensor([6, 2])).tolist() == [0, 1]
    assert DSparkBatchMetadata.accepted_prefix_lengths(target, padded).tolist() == [6, 2]
    assert DSparkBatchMetadata.num_rejected([5, 2], [6, 2]).tolist() == [0, 1]

    mismatch_first = target.clone()
    mismatch_first[1, 0] = 77
    assert accepted_prefix_lengths(mismatch_first, padded).tolist() == [6, 1]
    assert num_rejected([5, 2], [6, 1]).tolist() == [0, 2]
    assert_type_error(lambda: accepted_prefix_lengths([[11.5, 12.0]], [[11]]))


def test_no_draft_b1_keeps_bonus_row_only():
    metadata = DSparkBatchMetadata.build(
        requests=["request"],
        target_query_lengths=[6],
        draft_token_ids=[[]],
        request_slots=[2],
        slot_generations=[4],
        remaining_output_budget=[1],
    )

    assert metadata.target_query_start_loc.tolist() == [0, 6]
    assert metadata.draft_query_start_loc.tolist() == [0, 0]
    assert metadata.num_draft_tokens.tolist() == [0]
    assert metadata.cu_num_draft_tokens.tolist() == [0]
    assert metadata.cu_num_sampled_tokens.tolist() == [1]
    assert metadata.draft_token_ids.tolist() == []
    assert metadata.target_logits_indices.tolist() == []
    assert metadata.bonus_logits_indices.tolist() == [0]
    assert metadata.logits_indices.tolist() == [5]
    assert metadata.target_padding_mask.tolist() == [False] * 6
    assert metadata.draft_padding_mask.tolist() == []
    assert accepted_prefix_lengths(torch.tensor([[10, 11, 12, 13, 14, 15]]), torch.empty((1, 0))).tolist() == [1]
    assert num_rejected([0], [1]).tolist() == [0]


def test_optional_target_padding_mask_and_device_are_preserved():
    mask = [True, True, True, False, False, False]
    metadata = DSparkBatchMetadata.build(
        requests=["request"],
        target_query_lengths=[6],
        draft_token_ids=[[1, 2]],
        request_slots=torch.tensor([3], dtype=torch.int64),
        slot_generations=torch.tensor([5], dtype=torch.int32),
        remaining_output_budget=torch.tensor([4], dtype=torch.int64),
        target_padding_mask=mask,
        device=torch.device("cpu"),
    )
    assert metadata.target_padding_mask.tolist() == mask
    assert metadata.request_slots.dtype == torch.int32
    assert metadata.slot_generations.dtype == torch.int64
    assert metadata.remaining_output_budget.dtype == torch.int32

    meta = DSparkBatchMetadata.build(
        requests=["request"],
        target_query_lengths=[6],
        draft_token_ids=[[1, 2]],
        request_slots=[3],
        slot_generations=[5],
        remaining_output_budget=[4],
        device="meta",
    )
    assert meta.request_slots.device.type == "meta"
    assert meta.logits_indices.device.type == "meta"
    assert meta.target_padding_mask.device.type == "meta"


def test_invalid_batch_shapes_and_values_are_rejected():
    valid = dict(
        requests=["request"],
        target_query_lengths=[6],
        draft_token_ids=[[1]],
        request_slots=[0],
        slot_generations=[0],
        remaining_output_budget=[1],
    )

    assert_value_error(lambda: DSparkBatchMetadata.build(**{**valid, "requests": []}) )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(**{
            **valid,
            "requests": [f"request-{index}" for index in range(9)],
            "target_query_lengths": [6] * 9,
            "draft_token_ids": [[1]] * 9,
            "request_slots": list(range(9)),
            "slot_generations": [0] * 9,
            "remaining_output_budget": [1] * 9,
        })
    )
    for field in ("target_query_lengths", "draft_token_ids", "request_slots", "slot_generations",
                  "remaining_output_budget"):
        assert_value_error(lambda field=field: DSparkBatchMetadata.build(**{**valid, field: []}))
    assert_value_error(lambda: DSparkBatchMetadata.build(**{**valid, "target_query_lengths": [0]}))
    assert_value_error(lambda: DSparkBatchMetadata.build(**{**valid, "draft_token_ids": [[1, 2, 3, 4, 5, 6]]}))
    assert_value_error(lambda: DSparkBatchMetadata.build(**{**valid, "request_slots": [-1]}))
    assert_value_error(lambda: DSparkBatchMetadata.build(**{**valid, "slot_generations": [-1]}))
    assert_value_error(lambda: DSparkBatchMetadata.build(**{**valid, "remaining_output_budget": [-1]}))
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            **{**valid, "request_slots": torch.tensor([0], device="meta")}
        )
    )
    assert_type_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "target_query_lengths": [6.0]})
    )
    assert_type_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "draft_token_ids": [[1.5]]})
    )
    assert_type_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "request_slots": [0.5]})
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "draft_token_ids": [[-1]]})
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            **{
                **valid,
                "requests": ["request", "request"],
                "target_query_lengths": [6, 6],
                "draft_token_ids": [[1], [2]],
                "request_slots": [0, 1],
                "slot_generations": [0, 0],
                "remaining_output_budget": [1, 1],
            }
        )
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            **{
                **valid,
                "requests": ["request-a", "request-b"],
                "target_query_lengths": [6, 6],
                "draft_token_ids": [[1], [2]],
                "request_slots": [0, 0],
                "slot_generations": [0, 0],
                "remaining_output_budget": [1, 1],
            }
        )
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "target_padding_mask": [False] * 5})
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            **{**valid, "target_padding_mask": [False, False, False, False, False, True]}
        )
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            **{**valid, "target_padding_mask": [False, False, False, False, True, False]}
        )
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            **{
                **valid,
                "target_query_lengths": [1],
                "target_padding_mask": [False],
            }
        )
    )


def test_rejected_count_bounds_are_checked():
    assert_value_error(lambda: num_rejected([2], [0]))
    assert_value_error(lambda: num_rejected([2], [4]))
    assert_value_error(lambda: num_rejected([-1], [1]))
    assert_value_error(lambda: num_rejected([2, 1], [3]))
    assert_type_error(lambda: num_rejected([2.0], [2]))
