# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the packed B2 DSpark target stage."""
from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_gaudi.models import deepseek_v41_program as program_module
from vllm_gaudi.models.deepseek_v41_batch_target_program import CompiledTargetBatchStage
from vllm_gaudi.models.deepseek_v41_program import PreparedDecoderLayer
from vllm_gaudi.ops import deepseek_v41_batch_attention as batch_attention
from vllm_gaudi.ops.deepseek_v41_batch_target import (
    TARGET_ROWS,
    DSparkTargetBatch,
    DSparkTargetSnapshot,
    compressor_sequence_slot_masked,
    compressor_sequence_slot_torch_mix,
    initialize_target_batch_state,
    select_compressor_norm_by_request_length,
)


def _metadata(lengths, starts, slots=(0, 1), generations=(3, 4), round_generation=7):
    query_loc = [0, lengths[0], lengths[0] + lengths[1]]
    padding = torch.ones((2, 6), dtype=torch.bool)
    for request, length in enumerate(lengths):
        padding[request, :length] = False
    return SimpleNamespace(
        request_ids=("request-a", "request-b"),
        target_start_positions=torch.tensor(starts, dtype=torch.int64),
        target_query_start_loc=torch.tensor(query_loc, dtype=torch.int32),
        target_padding_mask=padding,
        request_slots=torch.tensor(slots, dtype=torch.int32),
        slot_generations=torch.tensor(generations, dtype=torch.int64),
        round_generation=torch.tensor([round_generation], dtype=torch.int64),
    )


def _batch(lengths=(1, 6), starts=(100, 200), *, slots=(0, 1), pages=None):
    if pages is None:
        pages = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
    queries = torch.zeros((2, 6), dtype=torch.int32)
    for request, length in enumerate(lengths):
        queries[request, :length] = torch.arange(10 + request * 10, 10 + request * 10 + length)
    return DSparkTargetBatch.from_metadata(_metadata(lengths, starts, slots), queries, pages)


def test_mixed_c1_c6_keeps_request_major_fixed_rows():
    batch = _batch()
    assert batch.request_lengths == (1, 6)
    assert batch.row_request.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert batch.row_valid.tolist() == [True, False, False, False, False, False, True, True, True, True, True, True]
    assert batch.positions.tolist()[:6] == [100, -1, -1, -1, -1, -1]
    assert batch.positions.tolist()[6:] == [200, 201, 202, 203, 204, 205]
    assert batch.row_slots.tolist() == [0, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1]
    assert batch.target_query_start_loc.tolist() == [0, 1, 7]
    assert batch.real_indices.tolist() == [0, 6, 7, 8, 9, 10, 11]
    assert batch.padding_indices.tolist() == [1, 2, 3, 4, 5]


def test_real_metadata_and_fixed_query_pack_contract():
    from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata

    metadata = DSparkBatchMetadata.build(
        requests=("request-a", "request-b"),
        target_query_lengths=[1, 6],
        draft_token_ids=[[], [21, 22, 23, 24, 25]],
        request_slots=[0, 1],
        slot_generations=[3, 4],
        remaining_output_budget=[8, 8],
        target_start_positions=[100, 200],
        round_generation=7,
    )
    logical_queries = torch.tensor([10, 20, 21, 22, 23, 24, 25], dtype=torch.int32)
    packed = metadata.pack_target_queries(logical_queries)
    batch = DSparkTargetBatch.from_metadata(
        metadata, packed, torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
    )
    assert batch.request_lengths == (1, 6)
    assert batch.target_query_start_loc.tolist() == [0, 1, 7]
    assert batch.row_valid.tolist() == [True, False, False, False, False, False, True, True, True, True, True, True]
    assert batch.input_ids.tolist()[:7] == [10, 0, 0, 0, 0, 0, 20]
    assert batch.positions.tolist()[:7] == [100, -1, -1, -1, -1, -1, 200]
    assert batch.input_ids.tolist()[6:] == [20, 21, 22, 23, 24, 25]


def test_real_metadata_accepts_two_through_five_target_counts():
    from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata

    metadata = DSparkBatchMetadata.build(
        requests=("request-a", "request-b"),
        target_query_lengths=[2, 5],
        draft_token_ids=[[11], [21, 22, 23, 24]],
        request_slots=[0, 1],
        slot_generations=[3, 4],
        remaining_output_budget=[8, 8],
        target_start_positions=[100, 200],
        round_generation=7,
    )
    packed = metadata.pack_target_queries(torch.tensor([10, 11, 20, 21, 22, 23, 24], dtype=torch.int32))
    batch = DSparkTargetBatch.from_metadata(
        metadata, packed, torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
    )
    assert batch.request_lengths == (2, 5)
    assert batch.target_query_start_loc.tolist() == [0, 2, 7]
    assert batch.row_valid.tolist() == [
        True, True, False, False, False, False,
        True, True, True, True, True, False,
    ]
    assert batch.positions.tolist() == [
        100, 101, -1, -1, -1, -1,
        200, 201, 202, 203, 204, -1,
    ]


@pytest.mark.parametrize(
    "lengths",
    [(1, 1), (1, 6), (6, 1), (6, 6), (2, 3), (3, 4), (4, 5), (5, 2)],
)
def test_target_layout_accepts_each_valid_count(lengths):
    batch = _batch(lengths=lengths)
    assert tuple(batch.request_offsets) == (0, 6, 12)
    assert batch.row_valid.sum().item() == sum(lengths)
    assert batch.row_pages.shape == (TARGET_ROWS, 4)


def test_c1c6_snapshot_keeps_padding_untouched_and_pages_independent():
    program = _snapshot_program()
    _fill_program(program)
    batch = _batch(
        lengths=(1, 6),
        starts=(255, 511),
        pages=torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]], dtype=torch.int32
        ),
    )
    snapshot = DSparkTargetSnapshot.capture(program, batch)
    state = program.layers[0].attention.batch_state
    saved = (state.swa.clone(), state.kv_history.clone(), state.score_history.clone())
    cache = program.shared.sources["2"]
    saved_main, saved_index = cache.main.clone(), cache.index.clone()
    _mutate(program)
    snapshot.restore(committed=(1, 3))

    # Only the C1 anchor is accepted for request 0.  Its five padding rows
    # have no snapshot owner and remain untouched sentinel state.
    assert torch.equal(state.swa[255], torch.full((528,), 99, dtype=torch.uint8))
    for row in (0, 1, 2, 3, 4):
        assert torch.equal(state.swa[row], torch.full((528,), 99, dtype=torch.uint8))
    # The C6 request accepts positions 511..513; 514..516 restore.
    assert torch.equal(state.swa[256 + 255], torch.full((528,), 99, dtype=torch.uint8))
    assert torch.equal(state.swa[256 + 0], torch.full((528,), 99, dtype=torch.uint8))
    assert torch.equal(state.swa[256 + 1], torch.full((528,), 99, dtype=torch.uint8))
    for row in (256 + 2, 256 + 3, 256 + 4):
        assert torch.equal(state.swa[row], saved[0][row])

    width = 64
    def physical(request, position):
        logical = position // 2
        return batch.host_pages[request][logical // width] * width + logical % width
    # Request 0 only publishes position 255.  Request 1 publishes 511 and 513
    # as accepted, and restores 515 as rejected.
    assert torch.equal(cache.main[physical(0, 255)], torch.full((288,), 99, dtype=torch.uint8))
    assert torch.equal(cache.main[physical(1, 511)], torch.full((288,), 99, dtype=torch.uint8))
    assert torch.equal(cache.main[physical(1, 513)], torch.full((288,), 99, dtype=torch.uint8))
    assert torch.equal(cache.main[physical(1, 515)], saved_main[physical(1, 515)])
    assert torch.equal(cache.index[physical(1, 515)], saved_index[physical(1, 515)])
    assert physical(0, 255) != physical(1, 511)


def test_zero_committed_restores_all_valid_rows_and_padding_stays_inactive():
    program = _snapshot_program()
    _fill_program(program)
    batch = _batch(lengths=(2, 5), starts=(250, 500))
    snapshot = DSparkTargetSnapshot.capture(program, batch)
    state = program.layers[0].attention.batch_state
    saved_swa = state.swa.clone()
    saved_kv = state.kv_history.clone()
    # Every snapshot owner must be a valid physical row.
    assert all(
        batch.host_row_valid[request_index * 6 + row_inside_request]
        for row in snapshot.rows
        for owners in row.owners
        for request_index, row_inside_request in owners
    )
    _mutate(program)
    snapshot.restore(committed=(0, 0))
    for row_index, valid in enumerate(batch.host_row_valid):
        if not valid:
            continue
        position = batch.host_positions[row_index]
        slot = batch.host_row_slots[row_index]
        assert torch.equal(state.swa[slot * 256 + position % 256],
                          saved_swa[slot * 256 + position % 256])
        assert torch.equal(state.kv_history[slot * 8 + position % 8],
                          saved_kv[slot * 8 + position % 8])
    # Inactive padding rows were never captured and remain sentinel state.
    assert torch.equal(state.swa[0], torch.full((528,), 99, dtype=torch.uint8))


def test_target_state_reuses_existing_program_bank():
    existing = SimpleNamespace(capacity=2)
    program = SimpleNamespace(batch_state=existing)
    assert initialize_target_batch_state(program) is existing


class _LaneState(nn.Module):
    def __init__(self, capacity=2):
        super().__init__()
        self.register_buffer("swa", torch.zeros((capacity * 256, 528), dtype=torch.uint8))
        self.register_buffer("kv_history", torch.zeros((capacity * 8, 512), dtype=torch.float32))
        self.register_buffer("score_history", torch.zeros((capacity * 8, 512), dtype=torch.float32))


class _Layer(nn.Module):
    def __init__(self, index, *, ratio=0):
        super().__init__()
        self.layer = index
        self.attention = nn.Module()
        self.attention.ratio = ratio
        self.attention.batch_state = _LaneState()


class _Cache(nn.Module):
    def __init__(self, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.register_buffer("main", torch.zeros((1024, 288), dtype=torch.uint8))
        self.register_buffer("index", torch.zeros((1024, 68), dtype=torch.uint8))


def _snapshot_program():
    layers = nn.ModuleList([_Layer(1), _Layer(14)])
    shared = nn.Module()
    shared.sources = nn.ModuleDict({"2": _Cache(ratio=2)})
    program = nn.Module()
    program.layers = layers
    program.shared = shared
    return program


def _fill_program(program):
    with torch.no_grad():
        for layer in program.layers:
            state = layer.attention.batch_state
            state.swa.copy_(torch.arange(state.swa.numel()).reshape_as(state.swa).remainder(251).to(torch.uint8))
            state.kv_history.copy_(torch.arange(state.kv_history.numel()).reshape_as(state.kv_history).float())
            state.score_history.copy_(torch.arange(state.score_history.numel()).reshape_as(state.score_history).float() + 1000)
        cache = program.shared.sources["2"]
        cache.main.copy_(torch.arange(cache.main.numel()).reshape_as(cache.main).remainder(251).to(torch.uint8))
        cache.index.copy_(torch.arange(cache.index.numel()).reshape_as(cache.index).remainder(251).to(torch.uint8))


def _mutate(program):
    with torch.no_grad():
        for layer in program.layers:
            state = layer.attention.batch_state
            state.swa.fill_(99)
            state.kv_history.fill_(99)
            state.score_history.fill_(99)
        program.shared.sources["2"].main.fill_(99)
        program.shared.sources["2"].index.fill_(99)


def test_snapshot_restores_only_rejected_suffix_by_slot_and_position():
    program = _snapshot_program()
    _fill_program(program)
    batch = _batch(lengths=(6, 6), starts=(255, 511), pages=torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]], dtype=torch.int32))
    snapshot = DSparkTargetSnapshot.capture(program, batch)
    assert snapshot.bytes > 0
    state = program.layers[0].attention.batch_state
    saved_swa = state.swa.clone()
    saved_kv = state.kv_history.clone()
    saved_main = program.shared.sources["2"].main.clone()
    saved_index = program.shared.sources["2"].index.clone()
    _mutate(program)
    # Keep three rows of request 0 and one row of request 1.
    snapshot.restore(committed=(3, 1))

    # Accepted rows keep the speculative values.  Only the rejected suffix
    # is rolled back to the pre-round snapshot.
    for row in (255, 0, 1):
        assert torch.equal(state.swa[row], torch.full((528,), 99, dtype=torch.uint8))
    for row in (2, 3, 4):
        assert torch.equal(state.swa[row], saved_swa[row])
    # slot 1: ring rows are offset by 256.
    assert torch.equal(state.swa[256 + 255], torch.full((528,), 99, dtype=torch.uint8))
    for row in (256, 257, 258, 259):
        assert torch.equal(state.swa[row], saved_swa[row])

    # History ring rows are independent for the two slots and retain the
    # accepted prefix only.  Position 255 maps to ring 7.
    assert torch.equal(state.kv_history[7], torch.full((512,), 99.0))
    assert torch.equal(state.kv_history[0], torch.full((512,), 99.0))
    assert torch.equal(state.kv_history[1], torch.full((512,), 99.0))
    for row in (2, 3, 4):
        assert torch.equal(state.kv_history[row], saved_kv[row])
    assert torch.equal(state.kv_history[8 + 7], torch.full((512,), 99.0))
    for row in (8, 9, 10, 11, 12):
        assert torch.equal(state.kv_history[row], saved_kv[row])

    # Ratio-two page rows are published only for odd positions.  Request 0
    # accepted positions 255 and 257; request 1 accepted only 511.
    cache = program.shared.sources["2"]
    width = 64
    def physical(request, position):
        logical = position // 2
        return batch.host_pages[request][logical // width] * width + logical % width

    physical_255 = physical(0, 255)
    physical_257 = physical(0, 257)
    physical_259 = physical(0, 259)
    physical_511 = physical(1, 511)
    physical_513 = physical(1, 513)
    assert torch.equal(cache.main[physical_255], torch.full((288,), 99, dtype=torch.uint8))
    assert torch.equal(cache.index[physical_255], torch.full((68,), 99, dtype=torch.uint8))
    assert torch.equal(cache.main[physical_259], saved_main[physical_259])
    assert torch.equal(cache.main[physical_511], torch.full((288,), 99, dtype=torch.uint8))
    assert torch.equal(cache.main[physical_513], saved_main[physical_513])
    # An odd/even position pair is not both published.
    even_256 = physical(0, 256)
    assert torch.equal(cache.main[even_256], torch.full((288,), 99, dtype=torch.uint8))
    assert physical_255 != 0 and physical_257 != 0 and physical_259 != 0


def test_snapshot_ring_wrap_is_distinct_for_both_slots():
    program = _snapshot_program()
    _fill_program(program)
    for start in (255, 511):
        batch = _batch(lengths=(6, 6), starts=(start, start + 256))
        snapshot = DSparkTargetSnapshot.capture(program, batch)
        touched = {int(value) for row in snapshot.rows if row.destination is program.layers[0].attention.batch_state.swa
                   for value in row.indices.tolist()}
        # The first request wraps its SWA ring exactly once.
        assert 255 in touched or 511 in touched


def test_masked_compressor_publishes_writes_without_sequence_mutation(monkeypatch):
    history = torch.zeros((16, 512), dtype=torch.float32)
    score = torch.ones((16, 512), dtype=torch.float32)
    original_history = history.clone()
    original_score = score.clone()
    kv = torch.arange(6 * 512, dtype=torch.float32).reshape(6, 512)
    score_rows = kv + 1000
    positions = torch.tensor([255, 256, 257, -1, -1, -1], dtype=torch.int32)
    events = []

    def fake_write(cache, values, rows):
        events.append(("write", cache.shape[0]))
        assert values.shape == (6, 512)
        completion = torch.full((6,), -1, dtype=torch.int32)
        for index, row in enumerate(rows.tolist()):
            if row >= 0:
                cache[row].copy_(values[index])
                completion[index] = row
        return completion

    def fake_read(cache, rows, completion):
        events.append(("read", cache.shape[0]))
        assert rows.tolist() == list(range(8, 16))
        assert completion.tolist() == [15] * 8
        return cache[rows.long()].clone()

    def pure_latent_sequence(history_scratch, score_scratch, kv_rows, score_arg, positions_arg):
        events.append(("sequence", None))
        assert history_scratch.shape == (8, 512)
        assert score_scratch.shape == (8, 512)
        assert torch.equal(history_scratch[7], kv_rows[0])
        assert torch.equal(history_scratch[0], kv_rows[1])
        assert torch.equal(history_scratch[1], kv_rows[2])
        assert torch.equal(score_scratch[7], score_arg[0])
        assert positions_arg.tolist() == positions.tolist()
        # Deliberately do not mutate either scratch history.  Publication must
        # come only from state_rows_write, never from an implicit sequence write.
        return kv_rows.clone()

    monkeypatch.setattr(batch_attention, "write_state_rows", fake_write)
    monkeypatch.setattr(batch_attention, "read_state_rows", fake_read)
    out = compressor_sequence_slot_masked(
        history, score, 1, kv, score_rows, positions, sequence_op=pure_latent_sequence
    )

    assert [event[0] for event in events] == ["write", "write", "read", "read", "sequence"]
    assert torch.equal(out, kv)
    assert torch.equal(history[:8], original_history[:8])
    assert torch.equal(score[:8], original_score[:8])
    assert torch.equal(history[8 + 7], kv[0])
    assert torch.equal(history[8 + 0], kv[1])
    assert torch.equal(history[8 + 1], kv[2])
    assert torch.equal(score[8 + 7], score_rows[0])
    assert torch.equal(score[8 + 0], score_rows[1])
    assert torch.equal(score[8 + 1], score_rows[2])
    # Negative positions have row id -1 and cannot reach either bank.
    assert torch.equal(history[8 + 2], original_history[8 + 2])
    assert torch.equal(score[8 + 2], original_score[8 + 2])


def test_masked_compressor_all_padding_reads_zero_and_writes_nothing(monkeypatch):
    history = torch.arange(16 * 512, dtype=torch.float32).reshape(16, 512)
    score = -history
    original_history = history.clone()
    original_score = score.clone()
    positions = torch.full((6,), -1, dtype=torch.int32)
    events = []

    def fake_write(cache, values, rows):
        events.append(("write", cache.shape[0]))
        assert rows.tolist() == [-1] * 6
        return rows.clone()

    def fake_read(cache, rows, completion):
        events.append(("read", cache.shape[0]))
        assert rows.tolist() == list(range(8, 16))
        assert completion.tolist() == [-1] * 8
        return torch.zeros((8, 512), dtype=cache.dtype)

    def pure_latent_sequence(history_scratch, score_scratch, kv_rows, score_arg, positions_arg):
        events.append(("sequence", None))
        assert torch.count_nonzero(history_scratch).item() == 0
        assert torch.count_nonzero(score_scratch).item() == 0
        return torch.zeros((6, 512), dtype=torch.bfloat16)

    monkeypatch.setattr(batch_attention, "write_state_rows", fake_write)
    monkeypatch.setattr(batch_attention, "read_state_rows", fake_read)
    out = compressor_sequence_slot_masked(
        history,
        score,
        1,
        torch.zeros((6, 512), dtype=torch.float32),
        torch.zeros((6, 512), dtype=torch.float32),
        positions,
        sequence_op=pure_latent_sequence,
    )

    assert [event[0] for event in events] == ["write", "write", "read", "read", "sequence"]
    assert torch.equal(out, torch.zeros((6, 512), dtype=torch.bfloat16))
    assert torch.equal(history, original_history)
    assert torch.equal(score, original_score)


def test_compiled_compressor_mix_matches_old_post_publish_formula():
    torch.manual_seed(11)
    history = torch.randn((8, 512), dtype=torch.float32)
    score_history = torch.randn((8, 512), dtype=torch.float32)
    positions = torch.tensor([255, 256, 258, 260, 262, 264], dtype=torch.int32)
    expected_history, expected_score = history.clone(), score_history.clone()

    first = torch.bitwise_and(positions, -2)
    a = torch.bitwise_and(first, 7).long()
    b = torch.bitwise_and(first + 1, 7).long()
    gates = torch.stack((score_history[a], score_history[b]), 1).softmax(1)
    expected = (history[a] * gates[:, 0] + history[b] * gates[:, 1]).to(torch.bfloat16)

    result = compressor_sequence_slot_torch_mix(history, score_history, positions)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, expected)
    assert torch.equal(history, expected_history)
    assert torch.equal(score_history, expected_score)


@pytest.mark.parametrize("lengths", [(1, 1), (1, 6), (6, 1), (2, 5), (5, 2), (6, 6)])
def test_compressor_norm_selection_uses_per_request_device_valid_length(lengths):
    native_norm = torch.full((TARGET_ROWS, 512), 7.0, dtype=torch.bfloat16)
    compiled_norm = torch.full((TARGET_ROWS, 512), -3.0, dtype=torch.bfloat16)
    row_valid = torch.zeros((2, 6), dtype=torch.bool)
    for request, length in enumerate(lengths):
        row_valid[request, :length] = True

    result = select_compressor_norm_by_request_length(
        native_norm, compiled_norm, row_valid.reshape(TARGET_ROWS))
    for request, length in enumerate(lengths):
        expected = native_norm if length == 1 else compiled_norm
        actual = result[request * 6:(request + 1) * 6]
        assert torch.equal(actual, expected[request * 6:(request + 1) * 6])


class _FakeTargetAttention(nn.Module):
    def __init__(self, ratio=0):
        super().__init__()
        self.ratio = ratio
        self.batch_state = _LaneState()


class _FakeTargetLayer(nn.Module):
    def __init__(self, index, calls):
        super().__init__()
        self.layer = index
        self.attention = _FakeTargetAttention()
        self.calls = calls

    def forward_batch_target(
        self,
        residual,
        pre_mix,
        positions,
        image_mask,
        engram_rows,
        slots,
        pages,
        selected,
        candidates,
        main_done,
        index_done,
        row_valid,
        request_lengths,
        request_slots,
    ):
        assert selected.dtype == torch.int32
        assert candidates.dtype == torch.int32
        self.calls.append((self.layer, tuple(request_lengths), tuple(request_slots),
                           tuple(selected.shape), tuple(candidates.shape)))
        target = (residual.mean(1) + float(self.layer + 1)) if self.layer else None
        return residual + 1, pre_mix, selected, candidates, main_done, index_done, target


def _fake_stage(lengths=(1, 6), slots=(0, 1)):
    stage = nn.Module()
    hidden = 8
    stage.dspark = True
    stage.pp_rank = 0
    stage.tp_rank = 0
    stage.is_last_stage = True
    stage.reduce = lambda value: value
    stage.config = {"text_config": {"rms_norm_eps": 1e-6}}
    stage.weights = nn.Module()
    stage.weights.embed = nn.Embedding(32, hidden)
    stage.weights.norm = nn.Module()
    stage.weights.norm.register_buffer("weight", torch.ones(hidden))
    stage.shared = nn.Module()
    stage.shared.topk = nn.ModuleDict({"1": nn.Module()})
    stage.shared.sources = nn.ModuleDict({"1": nn.Module()})
    calls = []
    stage.layers = nn.ModuleList([_FakeTargetLayer(index, calls) for index in range(4)])
    stage.calls = calls
    return stage, calls


@pytest.mark.parametrize("capacity", [1, 2, 4])
def test_prepared_decoder_target_method_forwards_fixed_args_and_aux(monkeypatch, capacity):
    from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata

    attention_calls = []
    target_attention_calls = []
    hc_pre_calls = []
    hc_post_calls = []
    norm_calls = []
    quant_calls = []
    moe_calls = []

    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.ratio = 0
            self.batch_state = _LaneState()
            self.swa = self.batch_state.swa
            self._fused_qkv_quantized = False
            self._fused_qkv_weight = None
            self.shared_decode_metadata = False

        def forward_batch_target(self, value, positions, slots, pages, selected, candidates,
                                 main_done, index_done, **kwargs):
            attention_calls.append({
                "value_shape": tuple(value.shape),
                "positions_shape": tuple(positions.shape),
                "slots": slots.clone(),
                "pages_shape": tuple(pages.shape),
                "selected_shape": tuple(selected.shape),
                "candidates_shape": tuple(candidates.shape),
                "row_valid": kwargs["row_valid"].clone(),
                "request_lengths": kwargs["request_lengths"],
                "request_slots": kwargs["request_slots"],
            })
            return value, selected, candidates, main_done, index_done

    class FakeMoE(nn.Module):
        n256_fp8 = True
        n256_fused = True
        shared_gate_up_channel = torch.ones(1)
        concurrent_moe_rows = 0

        def forward(self, value, image_mask, **kwargs):
            moe_calls.append({
                "rows": value.shape[0],
                "decode": kwargs.get("decode"),
                "ordinary_decode": kwargs.get("ordinary_decode", False),
            })
            return value

    def fake_hc_pre(residual, previous_pre, *args, **kwargs):
        hc_pre_calls.append({
            "rows": residual.shape[0],
            "decode": kwargs.get("decode"),
            "request_batch": kwargs.get("request_batch"),
        })
        return residual, previous_pre, residual, residual

    def fake_hc_post(value, residual, post, comb):
        hc_post_calls.append(value.shape[0])
        return value + residual

    def fake_rms_norm(value, *args, **kwargs):
        norm_calls.append(value.shape[0])
        return value

    def fake_ffn_dual_quant(value, weight, eps):
        quant_calls.append(value.shape[0])
        scale = torch.ones((value.shape[0],), dtype=torch.float32)
        return value, value, scale, value, scale

    monkeypatch.setattr(program_module, "hc_pre", fake_hc_pre)
    monkeypatch.setattr(program_module, "hc_post", fake_hc_post)
    monkeypatch.setattr(program_module, "rms_norm", fake_rms_norm)
    monkeypatch.setattr(program_module, "decode_ffn_dual_quant", fake_ffn_dual_quant)
    monkeypatch.setattr(program_module, "draft_context_state", lambda residual, *args, **kwargs: residual.mean(1))
    monkeypatch.setattr(program_module.gaudi_envs, "VLLM_HPU_DSV41_MHC_SCHEDULE", False)

    layer = SimpleNamespace(
        layer=1,
        draft=False,
        draft_mhc=False,
        collect_target_state=True,
        dspark_mhc_deferred=False,
        dspark_peer_post_collapse=False,
        dspark_moe_peer_post=False,
        dspark_peer_post_norm=False,
        mhc_post_collapse=True,
        mhc_interlayer_collapse=True,
        mhc_interlayer_bf16=True,
        ffn_dual_quant=True,
        ffn_bf16_quant=True,
        mhc_control_mme=False,
        decode_attention_norm_quant=False,
        dspark_input_norm=False,
        hc_attn_fn_packed=torch.ones(1),
        hc_ffn_fn_packed=torch.ones(1),
        hc_attn_fn_mme=None,
        hc_ffn_fn_mme=None,
        hc_attn_fn_fp8=None,
        hc_attn_fn_fp8_channel=None,
        hc_ffn_fn_fp8=None,
        hc_ffn_fn_fp8_channel=None,
        weights=SimpleNamespace(
            hc_attn_fn=torch.empty(1),
            hc_attn_scale=torch.empty(1),
            hc_attn_base=torch.empty(1),
            hc_ffn_fn=torch.empty(1),
            hc_ffn_scale=torch.empty(1),
            hc_ffn_base=torch.empty(1),
            attn_norm=SimpleNamespace(weight=torch.empty(1)),
            ffn_norm=SimpleNamespace(weight=torch.empty(1)),
        ),
        attention=FakeAttention(),
        all_gather=lambda value, dim: value,
        eps=1e-6,
        hc_eps=1e-6,
        iterations=20,
        batch_control_reuse=False,
        batch_control_prefetch=False,
        batch_mhc_fusion=False,
        batch_ffn_fusion=False,
        moe=FakeMoE(),
        _c6_scoped_recipe_cache=None,
    )
    for method in (
        "_c6_slice_path_error",
        "_c6_scoped_prefix_math",
        "_c6_scoped_post_pre_moe_math",
        "_forward_c6_attention_prefix",
        "_forward_c6_post_attention",
        "_target_c6_attention_prefix",
        "_target_c6_post_attention",
    ):
        setattr(layer, method, MethodType(getattr(PreparedDecoderLayer, method), layer))

    def target_attention(value, positions, slots, pages, selected, candidates,
                         main_done, index_done, row_valid, ready_outputs):
        target_attention_calls.append((value.shape[0], tuple(row_valid.shape)))
        return layer.attention.forward_batch_target(
            value,
            positions,
            slots,
            pages,
            selected,
            candidates,
            main_done,
            index_done,
            row_valid=row_valid,
            request_lengths=(6, ) * capacity,
            request_slots=tuple(range(capacity)),
            ready_outputs=ready_outputs,
        )

    lengths = tuple(1 if lane % 2 == 0 else 6 for lane in range(capacity))
    starts = tuple(100 + lane * 100 for lane in range(capacity))
    metadata = DSparkBatchMetadata.build(
        requests=tuple(f"request-{lane}" for lane in range(capacity)),
        target_query_lengths=lengths,
        draft_token_ids=[list(range(length - 1)) for length in lengths],
        request_slots=tuple(range(capacity)),
        slot_generations=(1, ) * capacity,
        remaining_output_budget=(16, ) * capacity,
        target_start_positions=starts,
        round_generation=1,
    )
    batch = DSparkTargetBatch.from_metadata(
        metadata, torch.zeros((capacity, 6), dtype=torch.int64),
        torch.ones((capacity, 4), dtype=torch.int32), batch_capacity=capacity)
    rows = 6 * capacity
    hidden = torch.zeros((rows, 4, 4), dtype=torch.bfloat16)
    pre = torch.zeros((rows, 4), dtype=torch.float32)
    selected = torch.full((rows, 512), -1, dtype=torch.int32)
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32)
    main_done = torch.zeros(rows, dtype=torch.int32)
    index_done = torch.zeros(rows, dtype=torch.int32)
    result = PreparedDecoderLayer.forward_batch_target(
        layer,
        hidden,
        pre,
        batch.positions,
        torch.zeros(rows, dtype=torch.bool),
        None,
        batch.row_slots,
        batch.row_pages,
        selected,
        candidates,
        main_done,
        index_done,
        batch.row_valid,
        (6, ) * capacity,
        batch.host_request_slots,
        target_attention=target_attention,
    )
    residual, returned_pre, returned_selected, returned_candidates, _, _, aux = result
    assert residual.shape == (rows, 4, 4)
    assert returned_pre.shape == (rows, 4)
    assert torch.equal(returned_selected, selected)
    assert torch.equal(returned_candidates, candidates)
    assert aux.shape == (rows, 4)
    assert torch.equal(residual[~batch.row_valid], torch.zeros_like(residual[~batch.row_valid]))

    # Attention remains one packed 6*N call.  The serial C6 prefix/suffix
    # math and MoE consume exactly one six-row lane at a time.
    assert target_attention_calls == [(rows, (rows, ))]
    assert len(attention_calls) == 1
    assert attention_calls[0]["value_shape"] == (rows, 4, 4)
    assert attention_calls[0]["positions_shape"] == (rows, )
    assert attention_calls[0]["pages_shape"] == (rows, 4)
    assert attention_calls[0]["request_lengths"] == (6, ) * capacity
    assert attention_calls[0]["request_slots"] == tuple(range(capacity))
    assert torch.equal(attention_calls[0]["row_valid"], batch.row_valid)

    assert [call["rows"] for call in hc_pre_calls] == [6] * (2 * capacity)
    assert all(call["decode"] is True for call in hc_pre_calls)
    assert all(call["request_batch"] is None for call in hc_pre_calls)
    assert hc_post_calls == [6] * (2 * capacity)
    assert norm_calls == [6] * capacity
    assert quant_calls == [6] * capacity
    assert [call["rows"] for call in moe_calls] == [6] * capacity
    assert all(call["decode"] is True and call["ordinary_decode"] is False for call in moe_calls)


def test_compiled_target_group_uses_one_12row_loop_and_mixed_mask(monkeypatch):
    # Dynamo's stream tracker must not acquire a visible-but-busy HPU in CPU tests.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    stage, calls = _fake_stage()
    batch = _batch()
    compiled = CompiledTargetBatchStage(stage, batch.request_lengths, batch.host_request_slots, backend="eager")
    hidden = torch.zeros((TARGET_ROWS, 4, 8), dtype=torch.bfloat16)
    pre = torch.zeros((TARGET_ROWS, 4), dtype=torch.float32)
    output, aux = compiled(
        hidden,
        pre,
        batch.input_ids,
        batch.positions,
        batch.row_slots,
        batch.row_pages,
        batch.row_valid,
    )
    assert output.shape == (TARGET_ROWS, 8)
    assert aux.shape == (TARGET_ROWS, 24)
    assert output[~batch.row_valid].abs().sum().item() == 0
    assert aux[~batch.row_valid].abs().sum().item() == 0
    assert {call[0] for call in calls} == {0, 1, 2, 3}
    assert all(call[1] == (6, 6) and call[2] == (0, 1) for call in calls)
    # Scratch is row-owned, not a capacity-two request bank.
    assert all(call[3] == (12, 512) and call[4] == (12, 2048) for call in calls)
    # The target program allocates I32 selection/candidate scratch, not a
    # state_rows cache and not a capacity-two request bank.
    assert all(call[3][0] == TARGET_ROWS and call[4][0] == TARGET_ROWS for call in calls)
