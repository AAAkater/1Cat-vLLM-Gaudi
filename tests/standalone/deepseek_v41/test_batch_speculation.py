# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for DSpark B2 packed batch metadata."""

from dataclasses import replace

import torch

from vllm_gaudi.ops.deepseek_v41_batch_speculation import (
    DSparkBatchMetadata,
    accepted_prefix_lengths,
    num_rejected,
    pack_target_queries,
    padded_draft_tokens,
    unpack_target_queries,
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
        request_slots=[3, 1],
        slot_generations=[7, 9],
        remaining_output_budget=[3, 2],
        target_start_positions=[128, 7],
        round_generation=11,
    )

    assert metadata.request_ids == ("request-a", "request-b")
    assert metadata.capacity == 2
    assert metadata.active_request_ids == ("request-a", "request-b")
    assert metadata.request_active.tolist() == [True, True]
    assert metadata.request_slots.tolist() == [3, 1]
    assert metadata.slot_generations.tolist() == [7, 9]
    assert metadata.target_start_positions.tolist() == [128, 7]
    assert metadata.round_generation.tolist() == 11
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
        metadata.request_active,
        metadata.request_slots,
        metadata.slot_generations,
        metadata.target_start_positions,
        metadata.round_generation,
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
            torch.bool,
            torch.int32,
            torch.int64,
            torch.int64,
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
    assert metadata.target_start_positions.tolist() == [0]
    assert metadata.round_generation.tolist() == 0
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
        target_start_positions=torch.tensor([39], dtype=torch.int32),
        round_generation=torch.tensor(17, dtype=torch.int32),
    )
    assert metadata.target_padding_mask.tolist() == mask
    assert metadata.request_slots.dtype == torch.int32
    assert metadata.slot_generations.dtype == torch.int64
    assert metadata.target_start_positions.tolist() == [39]
    assert metadata.target_start_positions.dtype == torch.int64
    assert metadata.round_generation.tolist() == 17
    assert metadata.round_generation.dtype == torch.int64
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
    assert meta.request_active.device.type == "meta"
    assert meta.logits_indices.device.type == "meta"
    assert meta.target_padding_mask.device.type == "meta"


def test_b2_each_draft_count_keeps_request_positions_and_tail_indices():
    for draft_count in range(6):
        metadata = DSparkBatchMetadata.build(
            requests=["request-a", "request-b"],
            target_query_lengths=[draft_count + 1, 6],
            draft_token_ids=[list(range(1, draft_count + 1)), [11, 12, 13, 14, 15]],
            request_slots=[5, 2],
            slot_generations=[8, 3],
            remaining_output_budget=[4, 2],
            target_start_positions=[100 + draft_count, 7],
            round_generation=23,
        )

        assert metadata.target_start_positions.tolist() == [100 + draft_count, 7]
        assert metadata.round_generation.tolist() == 23
        assert metadata.target_query_start_loc.tolist() == [0, draft_count + 1, draft_count + 7]
        assert metadata.num_draft_tokens.tolist() == [draft_count, 5]
        assert metadata.logits_indices.tolist() == list(range(draft_count + 7))
        assert metadata.target_logits_indices.tolist() == list(range(draft_count)) + list(
            range(draft_count + 1, draft_count + 6)
        )
        assert metadata.bonus_logits_indices.tolist() == [draft_count, draft_count + 6]


def test_fixed_c1_c6_pack_preserves_tail_and_two_logits_spaces():
    metadata = DSparkBatchMetadata.build(
        requests=["request-a", "request-b"],
        target_query_lengths=[1, 6],
        draft_token_ids=[[], [11, 12, 13, 14, 15]],
        request_slots=[5, 1],
        slot_generations=[8, 4],
        remaining_output_budget=[2, 3],
        target_start_positions=[91, 4],
        round_generation=31,
    )
    source = torch.arange(14, dtype=torch.int32).reshape(7, 2)

    packed = metadata.pack_target_queries(source)

    assert packed.queries.tolist() == [
        [[0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]],
    ]
    assert packed.padding_mask.tolist() == [
        [False, True, True, True, True, True],
        [False, False, False, False, False, False],
    ]
    assert packed.real_indices.tolist() == [0, 6, 7, 8, 9, 10, 11]
    assert packed.padding_indices.tolist() == [1, 2, 3, 4, 5]
    assert metadata.logits_indices.tolist() == list(range(7))
    assert packed.logits_indices.tolist() == [0, 6, 7, 8, 9, 10, 11]
    assert packed.target_logits_indices.tolist() == [1, 2, 3, 4, 5]
    assert packed.bonus_logits_indices.tolist() == [0, 6]
    assert torch.equal(
        packed.queries.reshape(12, 2).index_select(0, packed.logits_indices),
        source.index_select(0, metadata.logits_indices.long()),
    )
    assert torch.equal(unpack_target_queries(packed, metadata), source)
    assert torch.equal(metadata.unpack_target_queries(packed), source)

    helper_result = pack_target_queries(source, metadata)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (
                helper_result.queries,
                helper_result.padding_mask,
                helper_result.real_indices,
                helper_result.padding_indices,
                helper_result.logits_indices,
                helper_result.target_logits_indices,
                helper_result.bonus_logits_indices,
            ),
            (
                packed.queries,
                packed.padding_mask,
                packed.real_indices,
                packed.padding_indices,
                packed.logits_indices,
                packed.target_logits_indices,
                packed.bonus_logits_indices,
            ),
            strict=True,
        )
    )


def test_fixed_pack_marks_logical_and_bucket_padding_no_write():
    mask = [False, True, False, False, False, False, False]
    metadata = DSparkBatchMetadata.build(
        requests=["request-a", "request-b"],
        target_query_lengths=[6, 1],
        draft_token_ids=[[1, 2], []],
        request_slots=[4, 0],
        slot_generations=[6, 2],
        remaining_output_budget=[3, 1],
        target_padding_mask=mask,
        target_start_positions=[256, 18],
    )
    source = torch.arange(14, dtype=torch.int32).reshape(7, 2)
    source[1] = 999

    packed = metadata.pack_target_queries(source)

    assert packed.padding_mask.tolist() == [
        [False, True, False, False, False, False],
        [False, True, True, True, True, True],
    ]
    assert packed.real_indices.tolist() == [0, 2, 3, 4, 5, 6]
    assert packed.padding_indices.tolist() == [1, 7, 8, 9, 10, 11]
    assert packed.logits_indices.tolist() == [3, 4, 5, 6]
    assert torch.equal(unpack_target_queries(packed, metadata), source)


def test_b2_reordering_keeps_slots_generations_and_query_rows_together():
    first = DSparkBatchMetadata.build(
        requests=["request-a", "request-b"],
        target_query_lengths=[2, 6],
        draft_token_ids=[[1], [11, 12, 13, 14, 15]],
        request_slots=[7, 2],
        slot_generations=[4, 9],
        remaining_output_budget=[1, 5],
        target_start_positions=[50, 10],
        round_generation=41,
    )
    reordered = DSparkBatchMetadata.build(
        requests=["request-b", "request-a"],
        target_query_lengths=[6, 2],
        draft_token_ids=[[11, 12, 13, 14, 15], [1]],
        request_slots=[2, 7],
        slot_generations=[9, 4],
        remaining_output_budget=[5, 1],
        target_start_positions=[10, 50],
        round_generation=41,
    )
    source = torch.arange(16, dtype=torch.int32).reshape(8, 2)
    reordered_source = torch.cat((source[2:], source[:2]))

    first_packed = first.pack_target_queries(source)
    reordered_packed = reordered.pack_target_queries(reordered_source)

    assert reordered.request_slots.tolist() == [2, 7]
    assert reordered.slot_generations.tolist() == [9, 4]
    assert reordered.target_start_positions.tolist() == [10, 50]
    assert torch.equal(reordered_packed.queries[0], first_packed.queries[1])
    assert torch.equal(reordered_packed.queries[1, :2], first_packed.queries[0, :2])
    assert not reordered_packed.padding_mask[0].any()
    assert torch.equal(unpack_target_queries(reordered_packed, reordered), reordered_source)


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
        lambda: DSparkBatchMetadata.build(**{**valid, "target_start_positions": []})
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "target_start_positions": [-1]})
    )
    assert_type_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "target_start_positions": [0.5]})
    )
    assert_type_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "target_start_positions": [True]})
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            **{**valid, "target_start_positions": torch.tensor([[0]], dtype=torch.int64)}
        )
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            **{**valid, "target_start_positions": torch.tensor([0], device="meta")}
        )
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "round_generation": -1})
    )
    assert_type_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "round_generation": 1.5})
    )
    assert_type_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "round_generation": True})
    )
    assert_type_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "round_generation": [1]})
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            **{**valid, "round_generation": torch.tensor([1], dtype=torch.int64)}
        )
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            **{**valid, "round_generation": torch.tensor(1, device="meta")}
        )
    )
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
    assert_type_error(
        lambda: DSparkBatchMetadata.build(**{**valid, "slot_generations": [0.5]})
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


def test_fixed_pack_and_unpack_reject_invalid_shapes_and_metadata():
    metadata = DSparkBatchMetadata.build(
        requests=["request-a", "request-b"],
        target_query_lengths=[2, 2],
        draft_token_ids=[[1], [2]],
        request_slots=[3, 1],
        slot_generations=[5, 2],
        remaining_output_budget=[2, 2],
        target_start_positions=[20, 4],
        round_generation=7,
    )
    source = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    packed = metadata.pack_target_queries(source)

    assert_type_error(lambda: pack_target_queries(source.tolist(), metadata))
    assert_value_error(lambda: pack_target_queries(source[:3], metadata))
    assert_value_error(lambda: pack_target_queries(torch.tensor(1), metadata))
    assert_value_error(lambda: pack_target_queries(source.to("meta"), metadata))
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            requests=["request"],
            target_query_lengths=[7],
            draft_token_ids=[[1, 2, 3, 4, 5]],
            request_slots=[0],
            slot_generations=[1],
            remaining_output_budget=[1],
        ).pack_target_queries(torch.zeros((7, 2)))
    )
    assert_value_error(
        lambda: DSparkBatchMetadata.build(
            requests=["request"],
            target_query_lengths=[2],
            draft_token_ids=[[1]],
            request_slots=[0],
            slot_generations=[1],
            remaining_output_budget=[1],
            device="meta",
        ).pack_target_queries(torch.zeros((2, 2)))
    )

    assert_value_error(
        lambda: pack_target_queries(
            source, replace(metadata, logits_indices=metadata.logits_indices[:-1])
        )
    )
    assert_value_error(
        lambda: pack_target_queries(
            source,
            replace(
                metadata,
                target_padding_mask=torch.tensor([False, True, False, False], dtype=torch.bool),
            ),
        )
    )
    assert_value_error(
        lambda: pack_target_queries(
            source, replace(metadata, round_generation=torch.tensor(-1, dtype=torch.int64))
        )
    )

    assert_value_error(
        lambda: unpack_target_queries(replace(packed, padding_mask=torch.zeros((1, 6), dtype=torch.bool)), metadata)
    )
    assert_type_error(
        lambda: unpack_target_queries(replace(packed, padding_mask=torch.zeros((2, 6), dtype=torch.int32)), metadata)
    )
    assert_value_error(
        lambda: unpack_target_queries(replace(packed, real_indices=torch.tensor([0], dtype=torch.int64)), metadata)
    )
    assert_value_error(
        lambda: unpack_target_queries(replace(packed, queries=packed.queries[:, :5]), metadata)
    )
    other = DSparkBatchMetadata.build(
        requests=["other"],
        target_query_lengths=[1],
        draft_token_ids=[[]],
        request_slots=[0],
        slot_generations=[1],
        remaining_output_budget=[1],
    )
    assert_value_error(lambda: unpack_target_queries(packed, other))


def test_rejected_count_bounds_are_checked():
    assert_value_error(lambda: num_rejected([2], [0]))
    assert_value_error(lambda: num_rejected([2], [4]))
    assert_value_error(lambda: num_rejected([-1], [1]))
    assert_value_error(lambda: num_rejected([2, 1], [3]))
    assert_type_error(lambda: num_rejected([2.0], [2]))
