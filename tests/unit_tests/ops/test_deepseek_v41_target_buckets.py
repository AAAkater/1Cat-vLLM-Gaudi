# SPDX-License-Identifier: Apache-2.0
"""Compact target rows keep global state ownership and fixed draft indexing."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata
from vllm_gaudi.ops.deepseek_v41_batch_target import DSparkTargetBatch, DSparkTargetSnapshot
from vllm_gaudi.ops.deepseek_v41_target_buckets import pack_target_bucket, target_execution_buckets


def inputs(capacity, lengths):
    lanes = tuple(sorted(lengths))
    metadata = DSparkBatchMetadata.build(
        requests=[f"request-{i}" if i in lengths else None for i in range(capacity)],
        target_query_lengths=[lengths.get(i, 0) for i in range(capacity)],
        draft_token_ids=[list(range(lengths[i] - 1)) if i in lengths else [] for i in range(capacity)],
        request_slots=[i if i in lengths else -1 for i in range(capacity)],
        slot_generations=[13 + i if i in lengths else 0 for i in range(capacity)],
        remaining_output_budget=[64 if i in lengths else 0 for i in range(capacity)],
        target_start_positions=[100 if i in lengths else 0 for i in range(capacity)],
        round_generation=9,
        active=[i in lengths for i in range(capacity)],
    )
    queries = torch.arange(capacity * 6, dtype=torch.int64).reshape(capacity, 6)
    pages = torch.arange(1, capacity + 1, dtype=torch.int32).reshape(capacity, 1)
    return metadata, queries, pages, lanes


def test_bucket_set_is_finite_and_keeps_custom_capacity_fallback():
    assert target_execution_buckets(32, [1, 2, 4]) == (1, 2, 4, 32)
    assert target_execution_buckets(11, [1, 2, 4, 8, 16]) == (1, 2, 4, 8, 11)
    assert target_execution_buckets(3, [1]) == (1, 3)
    for bad in ([3], [True], [0], [], [128]):
        with pytest.raises(ValueError):
            target_execution_buckets(32, bad)


def test_n32_slot31_uses_one_c6_execution_lane():
    metadata, queries, pages, _ = inputs(32, {31: 6})
    packed = pack_target_bucket(metadata, queries, pages, 1)
    batch = DSparkTargetBatch.from_metadata(packed.metadata, packed.queries, packed.pages,
                                           batch_capacity=1, state_capacity=32)
    assert batch.target_rows == 6
    assert batch.state_capacity == 32
    assert batch.host_request_slots == (31,)
    assert batch.host_row_slots == (31,) * 6
    assert batch.host_pages == ((32,),)
    assert packed.source_lanes == (31,)
    assert torch.equal(packed.queries[0], queries[31])
    assert metadata.capacity == 32


def test_c1_c6_padding_preserves_global_owners_and_no_write_rows():
    metadata, queries, pages, _ = inputs(11, {2: 6, 10: 1})
    packed = pack_target_bucket(metadata, queries, pages, 4)
    batch = DSparkTargetBatch.from_metadata(packed.metadata, packed.queries, packed.pages,
                                           batch_capacity=4, state_capacity=11)
    assert packed.source_lanes == (2, 10, -1, -1)
    assert batch.host_request_slots == (2, 10, -1, -1)
    assert batch.host_request_active == (True, True, False, False)
    assert batch.host_row_slots[:7] == (2,) * 6 + (10,)
    assert batch.host_row_slots[7:] == (-1,) * 17
    assert batch.real_indices.numel() == 7
    assert not batch.row_valid[12:].any()


def test_fixed_bucket_preserves_original_metadata_and_buffers():
    metadata, queries, pages, _ = inputs(3, {2: 1})
    packed = pack_target_bucket(metadata, queries, pages, 3)
    assert packed.metadata is metadata
    assert packed.queries is queries
    assert packed.pages is pages


def test_global_slot_bounds_are_checked_against_the_state_bank():
    metadata, queries, pages, _ = inputs(32, {31: 1})
    packed = pack_target_bucket(metadata, queries, pages, 1)
    invalid = replace(packed.metadata, request_slots=torch.tensor([32], dtype=torch.int32))
    with pytest.raises(ValueError, match="owners"):
        DSparkTargetBatch.from_metadata(invalid, packed.queries, packed.pages,
                                       batch_capacity=1, state_capacity=32)


def test_compact_snapshot_restores_rejected_suffix_in_global_slot31():
    metadata, queries, pages, _ = inputs(32, {31: 6})
    packed = pack_target_bucket(metadata, queries, pages, 1)
    batch = DSparkTargetBatch.from_metadata(packed.metadata, packed.queries, packed.pages,
                                           batch_capacity=1, state_capacity=32)
    state = SimpleNamespace(swa=torch.zeros(32 * 256, 2), kv_history=torch.zeros(32 * 8, 512),
                            score_history=torch.zeros(32 * 8, 512))
    program = SimpleNamespace(layers=[SimpleNamespace(attention=SimpleNamespace(batch_state=state))],
                              shared=SimpleNamespace(sources={}))
    ticket = DSparkTargetSnapshot.capture(program, batch)
    for position in range(100, 106):
        state.swa[31 * 256 + position].fill_(7)
        state.kv_history[31 * 8 + position % 8].fill_(7)
        state.score_history[31 * 8 + position % 8].fill_(7)
    ticket.restore((1,))
    assert state.swa[31 * 256 + 100].eq(7).all()
    assert not state.swa[31 * 256 + 101:31 * 256 + 106].any()
    assert state.kv_history[31 * 8 + 100 % 8].eq(7).all()
    assert not state.swa[:31 * 256].any()
    assert not state.kv_history[:31 * 8].any()


def test_target_stage_reuses_n32_state_bank_for_b1_execution():
    from vllm_gaudi.models.deepseek_v41_batch_target_program import DSparkTargetStage

    metadata, queries, pages, _ = inputs(32, {31: 6})
    packed = pack_target_bucket(metadata, queries, pages, 1)
    batch = DSparkTargetBatch.from_metadata(packed.metadata, packed.queries, packed.pages,
                                           batch_capacity=1, state_capacity=32)
    bank = SimpleNamespace(capacity=32)
    program = SimpleNamespace(dspark=True, layers=[], batch_state=bank)
    stage = DSparkTargetStage.from_batch(program, batch, backend="eager")
    assert stage.state is bank
    assert stage.rows == 6
    assert stage.request_slots == (0,)


def test_target_scatter_and_completion_keep_the_full_draft_layout(monkeypatch):
    import vllm_gaudi.models.deepseek_v41_batch_target_program as target_program
    import vllm_gaudi.v1.worker.deepseek_v41_dspark_batch_runner as runner_module

    restored = []

    class Stage:
        def __init__(self, program, lengths, slots, **kwargs):
            self.compiled = SimpleNamespace()

        def snapshot(self, batch):
            return SimpleNamespace(restore=lambda counts: restored.append(tuple(counts)))

        def __call__(self, hidden, pre_mix, batch, engram):
            assert hidden.shape[0] == 6
            return torch.ones(6, 2), torch.full((6, 3), 9.0)

    monkeypatch.setattr(target_program, "DSparkTargetStage", Stage)
    monkeypatch.setattr(runner_module, "_target_greedy_ids",
                        lambda program, hidden, lanes: torch.arange(100, 106, dtype=torch.int64))
    program = SimpleNamespace(config={"text_config": {"hidden_size": 2}})
    owner = runner_module.DSparkPackedTargetStages(program, SimpleNamespace(capacity=32), device="cpu")
    owner.configure_execution_buckets(True, buckets=[1])
    for slot in (31, 3):
        metadata, queries, pages, _ = inputs(32, {slot: 6})
        result = owner.forward_batch(metadata, queries, None, None, pages)
        assert result.payload["target_execution_rows"] == 6
        assert result.payload["target_execution_slots"] == (slot,)
        assert result.token_ids.shape == (192,)
        assert result.token_ids.reshape(32, 6)[slot].tolist() == list(range(100, 106))
        assert result.token_ids.count_nonzero() == 6
        assert result.payload["aux"].shape == (192, 3)
        assert result.payload["aux"].reshape(32, 6, 3)[slot].eq(9).all()
        assert result.payload["aux"].count_nonzero() == 18
        accepted = [0] * 32
        accepted[slot] = 2
        owner.finish(result, accepted)
        assert restored[-1] == (2,)
    assert set(owner.stages) == {tuple(range(32)), (0,)}
    metadata, queries, pages, _ = inputs(32, {9: 1})
    result = owner.forward_batch(metadata, queries, None, None, pages)
    owner.abort(result)
    assert restored[-1] == (0,)
