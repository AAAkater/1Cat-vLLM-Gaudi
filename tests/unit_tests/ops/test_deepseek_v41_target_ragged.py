# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the optional compact-row target compressor and row C6 math.

The compact compressor must publish exactly the same lane-local history and
compressed cache rows as the fixed C6 compressor.  These tests use CPU facades
for the native state-row ops and a CPU reference of the masked native sequence
kernel; they do not exercise HPU kernels or HPU numerics.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_gaudi.models import deepseek_v41_batch_target_program as target_program
from vllm_gaudi.models.deepseek_v41_batch_target_program import DSparkTargetStage, _TargetAttentionCall
from vllm_gaudi.models.deepseek_v41_program import PreparedDecoderLayer, attention_input_rows, post_attention_rows
from vllm_gaudi.ops import deepseek_v41_batch_attention as batch_attention
from vllm_gaudi.ops import deepseek_v41_paged_attention as paged_module
from vllm_gaudi.ops import deepseek_v41_target_ragged as ragged_module
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention
from vllm_gaudi.ops.deepseek_v41_qkv import FusedCompressorInput
from vllm_gaudi.ops.deepseek_v41_target_queries import pack_target_query_rows
from vllm_gaudi.ops.deepseek_v41_target_ragged import (
    compress_target_query_rows,
    compressor_query_ownership,
    validate_target_ragged_compressor,
)

ROWS = 6
MASKED_SEQUENCE = "custom_deepseek_v41_compressor_sequence_masked_bf16_gaudi2"
FFN_NORM_QUANT = "custom_deepseek_v41_ffn_norm_quant_gaudi2"


def _masked_sequence(history, score_history, kv, score, positions):
    """CPU reference of the fixed-six masked native compressor sequence."""
    rows = kv.shape[0]
    logical_positions = [int(value) for value in positions.tolist()]
    base = logical_positions[0]
    result = torch.zeros((rows, kv.shape[1]), dtype=torch.bfloat16)
    for token, logical in enumerate(logical_positions):
        if logical < 0:
            continue
        ring = logical & 7
        current_is_first = (logical & 1) == 0
        other = (ring + 1) & 7 if current_is_first else (ring - 1) & 7
        partner_position = logical + (1 if current_is_first else -1)
        partner = partner_position - base
        present = 0 <= partner < rows and logical_positions[partner] == partner_position
        other_value = kv[partner] if present else history[other]
        other_score = score[partner] if present else score_history[other]
        if current_is_first:
            first_value, second_value, first_score, second_score = kv[token], other_value, score[token], other_score
        else:
            first_value, second_value, first_score, second_score = other_value, kv[token], other_score, score[token]
        maximum = torch.maximum(first_score, second_score)
        first_exp, second_exp = (first_score - maximum).exp(), (second_score - maximum).exp()
        inverse = (first_exp + second_exp).reciprocal()
        result[token] = (first_value * (first_exp * inverse) + second_value * (second_exp * inverse)).to(torch.bfloat16)
    return result


@pytest.fixture
def cpu_state_ops(monkeypatch):
    def write(cache, values, rows):
        assert values.shape[0] == rows.shape[0]
        valid = rows >= 0
        target = rows[valid].long()
        assert bool((target < cache.shape[0]).all())
        assert target.unique().numel() == target.numel()
        cache[target] = values[valid].to(cache.dtype)
        return torch.where(valid, rows, torch.full_like(rows, -1)).to(torch.int32)

    def read(cache, rows, completion):
        assert completion.shape == rows.shape
        values = cache.index_select(0, rows.clamp_min(0).long())
        return torch.where((rows >= 0)[:, None], values, torch.zeros_like(values))

    monkeypatch.setattr(batch_attention, "write_state_rows", write)
    monkeypatch.setattr(batch_attention, "read_state_rows", read)
    monkeypatch.setattr(torch.ops.custom_op, MASKED_SEQUENCE, _masked_sequence, raising=False)


class _Compressor:
    """Attention-shaped compressor owner using the production RoPE/projection code."""

    eps = 1e-6
    native_rope = False
    _fused_compressor_weight = None
    _rope = PagedCSA2Attention._rope
    _rotary_table = PagedCSA2Attention._rotary_table

    def __init__(self, ratio, capacity, *, state_capacity=None, hidden=64, row_invariant=True, exact=False, seed=0):
        generator = torch.Generator().manual_seed(seed)
        state_capacity = capacity if state_capacity is None else state_capacity
        self.ratio = ratio
        self.row_invariant = row_invariant
        self.linear_rows = []

        def dense(*shape):
            if exact:
                return torch.randint(-2, 3, shape, generator=generator).float() / 16
            return torch.randn(shape, generator=generator) / 8

        def single_term(outputs, inputs):
            # One power-of-two term per output keeps any GEMM order exact.
            weight = torch.zeros(outputs, inputs)
            columns = torch.randint(0, inputs, (outputs, ), generator=generator)
            weight[torch.arange(outputs), columns] = 2.0**-torch.randint(0, 3, (outputs, ), generator=generator).float()
            return weight

        self.weights = SimpleNamespace(
            compressor=SimpleNamespace(
                wkv=SimpleNamespace(weight=dense(512, hidden)),
                wgate=SimpleNamespace(weight=dense(512, hidden)),
                norm=SimpleNamespace(weight=torch.rand(512, generator=generator) + 0.5),
            ),
            indexer=SimpleNamespace(
                wk=SimpleNamespace(weight=single_term(128, 512) if exact else dense(128, 512)),
                k_norm=SimpleNamespace(weight=torch.rand(128, generator=generator) + 0.5),
            ),
        )
        self.batch_state = SimpleNamespace(
            kv_history=torch.randn((state_capacity * 8, 512), generator=generator),
            score_history=torch.randn((state_capacity * 8, 512), generator=generator),
        )
        cache_rows = (4 * capacity + 1) * (128 // ratio)
        self.cache = SimpleNamespace(
            main=torch.randint(0, 255, (cache_rows, 288), dtype=torch.uint8, generator=generator),
            index=torch.randint(0, 255, (cache_rows, 68), dtype=torch.uint8, generator=generator),
        )
        angles = torch.rand((1024, 32), generator=generator) * 6.0
        self.rotary = torch.stack((angles.cos(), angles.sin()), -1)

    def linear(self, value, weight):
        self.linear_rows.append(value.shape[0])
        weight = weight.weight.float()
        if self.row_invariant:
            result = torch.cat([F.linear(value[row:row + 1].float(), weight) for row in range(value.shape[0])])
        else:
            result = F.linear(value.float(), weight)
        return result.to(value.dtype)

    def _project_compressor_input(self, value):
        return FusedCompressorInput._project_compressor_input(self, value)

    def snapshot(self):
        return tuple(tensor.clone() for tensor in (self.batch_state.kv_history, self.batch_state.score_history,
                                                   self.cache.main, self.cache.index))


def _start(lane):
    # Mixed parity and 128-boundary crossings exercise ratio 2/4/128 publication.
    return ((lane % 3) + 1) * 128 - 4 + lane // 3


def _round(lengths, *, hidden=64, exact=False, seed=1, slot_map=None):
    capacity = len(lengths)
    rows = capacity * ROWS
    generator = torch.Generator().manual_seed(seed)
    if exact:
        value = (torch.randint(-4, 5, (rows, hidden), generator=generator).float() / 8).to(torch.bfloat16)
    else:
        value = torch.randn((rows, hidden), generator=generator).to(torch.bfloat16)
    positions = torch.full((rows, ), -1, dtype=torch.int32)
    slots = torch.full((rows, ), -1, dtype=torch.int32)
    valid = torch.zeros(rows, dtype=torch.bool)
    for lane, length in enumerate(lengths):
        start = lane * ROWS
        positions[start:start + length] = torch.arange(_start(lane), _start(lane) + length, dtype=torch.int32)
        slots[start:start + length] = lane if slot_map is None else slot_map[lane]
        valid[start:start + length] = True
    pages = torch.arange(1, 4 * capacity + 1, dtype=torch.int32).reshape(capacity, 4).repeat_interleave(ROWS, 0)
    return SimpleNamespace(value=value, positions=positions, slots=slots, valid=valid, pages=pages, capacity=capacity)


def _plan(round_, buckets=(1, 8, 32, 128)):
    plan = pack_target_query_rows(round_.valid.tolist(), buckets)
    return plan, torch.tensor(plan.source_rows, dtype=torch.int64), torch.tensor(plan.valid, dtype=torch.bool)


def _run_fixed(owner, round_, *, runtime=False):
    return PagedCSA2Attention._compress_batch_target(
        owner, round_.value, round_.positions, round_.slots, round_.pages, round_.valid & (round_.slots >= 0),
        (ROWS, ) * round_.capacity, tuple(range(round_.capacity)), runtime_request_slots=runtime)


def _run_ragged(owner, round_, indices, valid, *, runtime=False):
    return compress_target_query_rows(
        owner, round_.value, round_.positions, round_.slots, round_.pages, round_.valid & (round_.slots >= 0),
        (ROWS, ) * round_.capacity, tuple(range(round_.capacity)), indices, valid, runtime_request_slots=runtime)


def _assert_same_state(fixed, ragged, fixed_done, ragged_done):
    names = ("kv_history", "score_history", "cache.main", "cache.index")
    for name, expected, actual in zip(names, fixed.snapshot(), ragged.snapshot(), strict=True):
        assert torch.equal(expected, actual), name
    for expected, actual in zip(fixed_done, ragged_done, strict=True):
        assert expected.shape == actual.shape
        assert torch.equal(expected, actual)


LAYOUTS = {
    "c1_c6": (1, 6),
    "partial_lanes": (3, 0, 2, 0),
    "n11_mixed": (0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    "c1_to_c6": (1, 2, 3, 4, 5, 6),
    "all_c1": (1, ) * 8,
}


@pytest.mark.parametrize("ratio", (2, 4, 128))
@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_compact_compressor_matches_fixed_c6_bitwise(cpu_state_ops, ratio, layout):
    lengths = LAYOUTS[layout]
    round_ = _round(lengths)
    plan, indices, valid = _plan(round_)
    assert plan.bucket_rows < ROWS * round_.capacity
    fixed = _Compressor(ratio, round_.capacity)
    ragged = _Compressor(ratio, round_.capacity)
    initial = fixed.snapshot()

    fixed_done = _run_fixed(fixed, round_)
    ragged_done = _run_ragged(ragged, round_, indices, valid)

    _assert_same_state(fixed, ragged, fixed_done, ragged_done)
    # The ratio-two input projections keep the fixed 6N body; every other
    # row-local projection ran on the Q bucket.
    body = ROWS * round_.capacity
    assert set(fixed.linear_rows) == {body}
    expected_rows = [body, body, plan.bucket_rows] if ratio == 2 else [plan.bucket_rows] * 2
    assert ragged.linear_rows == expected_rows
    # Guard against a vacuous comparison: the round really published state.
    changed = [not torch.equal(before, after) for before, after in zip(initial, ragged.snapshot(), strict=True)]
    visible = bool((round_.valid & ((round_.positions + 1) % ratio == 0)).any())
    assert changed[2] == changed[3] == visible
    if ratio == 2:
        assert changed[0] and changed[1]
    else:
        assert not changed[0] and not changed[1]


@pytest.mark.parametrize("lengths", ((1, 0, 6, 2), (0, 0, 0, 1), (5, 4, 3, 1)))
def test_compact_compressor_runtime_slots_match_fixed(cpu_state_ops, lengths):
    state_capacity = 32
    slot_map = (31, 3, 17, 9)
    round_ = _round(lengths, slot_map=slot_map)
    _plan_value, indices, valid = _plan(round_)
    fixed = _Compressor(2, 4, state_capacity=state_capacity)
    ragged = _Compressor(2, 4, state_capacity=state_capacity)
    initial = fixed.snapshot()

    fixed_done = _run_fixed(fixed, round_, runtime=True)
    ragged_done = _run_ragged(ragged, round_, indices, valid, runtime=True)

    _assert_same_state(fixed, ragged, fixed_done, ragged_done)
    touched = {slot for slot, length in zip(slot_map, lengths, strict=True) if length}
    history = ragged.batch_state.kv_history.reshape(state_capacity, 8, 512)
    before = initial[0].reshape(state_capacity, 8, 512)
    for slot in range(state_capacity):
        assert torch.equal(history[slot], before[slot]) == (slot not in touched)


@pytest.mark.parametrize("ratio", (2, 4))
def test_dense_batched_gemm_matches_fixed_when_projection_is_exact(cpu_state_ops, ratio):
    # Real multi-row F.linear (not row-invariant in general); dyadic data makes
    # every accumulation order exact, so the comparison isolates the layout.
    round_ = _round((2, 0, 6, 1, 0, 3), exact=True)
    _plan_value, indices, valid = _plan(round_)
    fixed = _Compressor(ratio, round_.capacity, row_invariant=False, exact=True)
    ragged = _Compressor(ratio, round_.capacity, row_invariant=False, exact=True)
    _assert_same_state(fixed, ragged, _run_fixed(fixed, round_), _run_ragged(ragged, round_, indices, valid))


def test_full_body_identity_plan_matches_fixed_with_random_dense_gemm(cpu_state_ops):
    round_ = _round((1, 6, 0, 4))
    plan = pack_target_query_rows(round_.valid.tolist(), (1, ))
    assert plan.source_rows == tuple(range(ROWS * round_.capacity))
    fixed = _Compressor(2, round_.capacity, row_invariant=False)
    ragged = _Compressor(2, round_.capacity, row_invariant=False)
    _assert_same_state(fixed, ragged, _run_fixed(fixed, round_),
                       _run_ragged(ragged, round_, torch.tensor(plan.source_rows), torch.tensor(plan.valid)))


@pytest.mark.parametrize("ratio", (2, 4))
def test_padding_rows_never_publish_and_their_values_are_ignored(cpu_state_ops, ratio):
    lengths = (2, 0, 1, 5)
    round_ = _round(lengths)
    _plan_value, indices, valid = _plan(round_)
    poisoned = SimpleNamespace(**vars(round_))
    poisoned.value = round_.value.clone()
    poisoned.value[~round_.valid] = float("nan")
    clean_owner = _Compressor(ratio, round_.capacity)
    poisoned_owner = _Compressor(ratio, round_.capacity)
    initial = poisoned_owner.snapshot()

    clean_done = _run_ragged(clean_owner, round_, indices, valid)
    poisoned_done = _run_ragged(poisoned_owner, poisoned, indices, valid)

    _assert_same_state(clean_owner, poisoned_owner, clean_done, poisoned_done)
    owned_rows = torch.nonzero(round_.valid).flatten().tolist()
    history_rows = {int(round_.slots[row]) * 8 + int(round_.positions[row]) % 8 for row in owned_rows}
    after = poisoned_owner.snapshot()
    for row in range(initial[0].shape[0]):
        unchanged = torch.equal(initial[0][row], after[0][row]) and torch.equal(initial[1][row], after[1][row])
        assert unchanged == (ratio != 2 or row not in history_rows)
    visible = [row for row in owned_rows if (int(round_.positions[row]) + 1) % ratio == 0]
    assert visible
    for done in poisoned_done:
        assert torch.equal(torch.nonzero(done >= 0).flatten(), torch.tensor(visible))
        assert bool((done[~round_.valid] == -1).all())
    changed_cache_rows = {
        row
        for row in range(initial[2].shape[0])
        if not torch.equal(initial[2][row], after[2][row]) or not torch.equal(initial[3][row], after[3][row])
    }
    assert changed_cache_rows <= {int(poisoned_done[0][row]) for row in visible}


def test_uncovered_owned_row_is_treated_as_padding(cpu_state_ops):
    round_ = _round((3, 1))
    _plan_value, indices, valid = _plan(round_)
    dropped = 2  # lane 0, row 2: owned but removed from the malformed plan
    assert bool(round_.valid[dropped])
    malformed = torch.where(indices == dropped, torch.full_like(indices, 11), indices)
    owned, owned_query, owned_positions = compressor_query_ownership(
        round_.valid, round_.slots, round_.positions, malformed, valid & (indices != dropped))
    assert not bool(owned[dropped]) and int(owned_positions[dropped]) == -1
    assert int(owned.sum()) == int(round_.valid.sum()) - 1
    assert owned_query.shape == malformed.shape
    owner = _Compressor(2, round_.capacity)
    initial = owner.snapshot()
    _run_ragged(owner, round_, malformed, valid & (indices != dropped))
    ring_row = int(round_.positions[dropped]) % 8
    assert torch.equal(owner.batch_state.kv_history[ring_row], initial[0][ring_row])


def test_ragged_compressor_rejects_non_c6_ownership_and_bad_query_dtype(cpu_state_ops):
    round_ = _round((1, 2))
    _plan_value, indices, valid = _plan(round_)
    owner = _Compressor(2, round_.capacity)
    args = (owner, round_.value, round_.positions, round_.slots, round_.pages, round_.valid)
    with pytest.raises(ValueError, match="six physical state rows"):
        compress_target_query_rows(*args, (6, 3), (0, 1), indices, valid)
    with pytest.raises(ValueError, match="distinct lane owners"):
        compress_target_query_rows(*args, (6, 6), (0, 0), indices, valid)
    with pytest.raises(ValueError, match="int64 query indices"):
        compress_target_query_rows(*args, (6, 6), (0, 1), indices.int(), valid)


def test_validate_flag_requires_bool_and_query_buckets():
    assert validate_target_ragged_compressor(False, query_bucketing=False) is False
    assert validate_target_ragged_compressor(True, query_bucketing=True) is True
    for value in ("true", 1, None):
        with pytest.raises(TypeError, match="must be a boolean"):
            validate_target_ragged_compressor(value, query_bucketing=True)
    with pytest.raises(ValueError, match="dsv41_dspark_target_query_buckets"):
        validate_target_ragged_compressor(True, query_bucketing=False)


class _ForwardHarness:
    ratio = 2
    owns_kv = True
    owns_index = False
    fused_norm = False
    batch_c1_numerics = False
    q_scale_rope = False
    native_rope = False
    scale = 0.125
    window = 128
    batch_packed_mla = False
    batch_packed_mla_sram = False
    batch_packed_mla_vector = False
    batch_reindex_group = False
    eps = 1e-6

    def __init__(self, capacity, events):
        self.events = events
        self.batch_state = SimpleNamespace(swa=torch.zeros((capacity * 256, 528), dtype=torch.uint8))
        self.batch_main_unused = torch.zeros((1, 288), dtype=torch.uint8)
        self.cache = SimpleNamespace(main=torch.zeros((1, 288), dtype=torch.uint8))
        self.weights = SimpleNamespace(
            attn_sink=torch.zeros(512),
            q_norm=SimpleNamespace(weight=torch.ones(512)),
            kv_norm=SimpleNamespace(weight=torch.ones(512)),
        )

    def _project_qkv_input(self, value):
        self.events.append(("qkv", value.shape[0]))
        return value, value

    def project_query(self, value, _positions, *, request_batch):
        return value

    def _rope(self, value, _positions, *, request_batch, inverse=False):
        return value

    def _finish_output(self, output, _positions, _ready_outputs, *, request_batch, decode):
        return output

    def _compress_batch_target(self, value, positions, slots, pages, active, request_lengths, request_slots,
                               runtime_request_slots=False):
        self.events.append(("fixed", value.shape[0]))
        done = torch.ones(value.shape[0], dtype=torch.int32)
        return done, done


def _forward(monkeypatch, *, ragged, compact=True):
    capacity, body_rows = 11, 66
    events = []
    harness = _ForwardHarness(capacity, events)

    def ragged_spy(owner, value, positions, slots, pages, active, request_lengths, request_slots, indices, valid,
                   *, runtime_request_slots=False):
        assert owner is harness and value.shape[0] == body_rows and indices.shape == valid.shape
        events.append(("ragged", value.shape[0], indices.shape[0], runtime_request_slots))
        done = torch.full((body_rows, ), 2, dtype=torch.int32)
        return done, done

    monkeypatch.setattr(ragged_module, "compress_target_query_rows", ragged_spy)
    monkeypatch.setattr(batch_attention, "write_state_rows", lambda _cache, _values, rows: torch.zeros_like(rows))
    monkeypatch.setattr(batch_attention, "batch_packed_mla", lambda query, *_args, **_kwargs: query[:, :4].clone())
    monkeypatch.setattr(paged_module, "rms_norm", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(paged_module, "pack_swa", lambda value: value)
    valid = torch.zeros(body_rows, dtype=torch.bool)
    valid[6:12] = True
    valid[60] = True
    slots = torch.where(valid, torch.arange(body_rows, dtype=torch.int32) // 6, -1)
    positions = torch.where(valid, torch.arange(body_rows, dtype=torch.int32) + 100, -1)
    value = torch.arange(body_rows * 512, dtype=torch.float32).reshape(body_rows, 512)
    plan = pack_target_query_rows(valid.tolist())
    query = ({"query_indices": torch.tensor(plan.source_rows), "query_valid": torch.tensor(plan.valid)}
             if compact else {})
    flag = {"ragged_compressor": True} if ragged else {}
    result = PagedCSA2Attention.forward_batch_target(
        harness, value, positions, slots, torch.ones((body_rows, 4), dtype=torch.int32),
        torch.full((body_rows, 512), -1, dtype=torch.int32), torch.full((body_rows, 2048), -1, dtype=torch.int32),
        torch.zeros(body_rows, dtype=torch.int32), torch.zeros(body_rows, dtype=torch.int32), row_valid=valid,
        request_lengths=(6, ) * capacity, request_slots=tuple(range(capacity)), runtime_request_slots=True,
        **query, **flag)
    return events, result


def test_forward_flag_off_keeps_fixed_c6_compressor(monkeypatch):
    events, result = _forward(monkeypatch, ragged=False)
    assert events == [("fixed", 66), ("qkv", 8)]
    assert int(result[3][6]) == 1


def test_forward_flag_on_routes_compact_compressor(monkeypatch):
    events, result = _forward(monkeypatch, ragged=True)
    assert events == [("ragged", 66, 8, True), ("qkv", 8)]
    assert result[3].shape == result[4].shape == (66, )
    assert int(result[3][6]) == 2


def test_forward_flag_requires_compact_queries(monkeypatch):
    with pytest.raises(ValueError, match="requires compact query indices"):
        _forward(monkeypatch, ragged=True, compact=False)


class _RecordingAttention:

    def __init__(self):
        self.calls = []

    def forward_batch_target(self, value, *args, **kwargs):
        self.calls.append(kwargs)
        return value, None, None, None, None


@pytest.mark.parametrize(("ragged", "compact", "expected"), (
    (False, True, None),
    (True, True, True),
    (True, False, None),
    (False, False, None),
))
def test_attention_call_passes_flag_only_for_compact_queries(ragged, compact, expected):
    attention = _RecordingAttention()
    call = _TargetAttentionCall(attention, request_lengths=(6, 6), request_slots=(0, 1), runtime_request_slots=True,
                                ragged_compressor=ragged)
    rows = 12
    query = {"query_indices": torch.arange(8), "query_valid": torch.ones(8, dtype=torch.bool)} if compact else {}
    call(torch.zeros(rows, 4), torch.zeros(rows, dtype=torch.int32), torch.zeros(rows, dtype=torch.int32),
         torch.ones((rows, 1), dtype=torch.int32), None, None, None, None, torch.ones(rows, dtype=torch.bool), (),
         **query)
    kwargs = attention.calls[0]
    assert kwargs.get("ragged_compressor") is expected
    assert ("ragged_compressor" in kwargs) == (expected is not None)
    assert kwargs["runtime_request_slots"] is True


class _StageLayer(nn.Module):
    _target_c6_slices = True

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.attention = nn.Module()
        self.attention.ratio = 0


def _program(capacity):
    program = nn.Module()
    program.dspark = True
    program.pp_rank = 0
    program.tp_rank = 0
    program.is_last_stage = True
    program.tensor_parallel_size = 8
    program.reduce = lambda value: value
    program.config = {"text_config": {"rms_norm_eps": 1e-6}}
    program.weights = nn.Module()
    program.weights.embed = nn.Embedding(32, 4)
    program.weights.norm = nn.Module()
    program.weights.norm.register_buffer("weight", torch.ones(4))
    program.shared = nn.Module()
    program.shared.topk = nn.ModuleDict({"0": nn.Module()})
    program.shared.sources = nn.ModuleDict({"0": nn.Module()})
    program.layers = nn.ModuleList([_StageLayer(layer) for layer in range(4)])
    program.batch_state = SimpleNamespace(capacity=capacity)
    return program


@pytest.mark.parametrize("ragged", (False, True))
def test_target_stage_plumbs_static_flag_to_every_attention_call(monkeypatch, ragged):
    monkeypatch.setattr(target_program.torch, "compile", lambda call, **_kwargs: call)
    kwargs = {"ragged_compressor": True} if ragged else {}
    stage = DSparkTargetStage(_program(8), (6, ) * 4, tuple(range(4)), backend="eager", state_capacity=8, **kwargs)
    calls = [call for group in stage.compiled.groups for call in group.attention_calls]
    assert len(calls) == 4
    assert all(call.ragged_compressor is ragged and call.runtime_request_slots for call in calls)
    with pytest.raises(TypeError, match="must be a boolean"):
        DSparkTargetStage(_program(8), (6, ) * 4, tuple(range(4)), backend="eager", state_capacity=8,
                          ragged_compressor="true")


def _packed_stages(*, query_bucketing=True):
    from vllm_gaudi.v1.worker.deepseek_v41_dspark_batch_runner import DSparkPackedTargetStages

    stages = object.__new__(DSparkPackedTargetStages)
    stages.batch_capacity = 8
    stages.execution_buckets = (4, 8)
    stages.query_bucketing = query_bucketing
    stages.query_buckets = (1, 8, 32, 128)
    stages.query_stages = {}
    stages.observed_query_keys = set()
    stages.stages = {}
    stages.program = object()
    stages._backend = "eager"
    stages.ragged_compressor = False
    return stages


def test_packed_target_stages_configure_and_build_ragged_query_stages(monkeypatch):
    created = []

    class _Stage:

        def __init__(self, program, lengths, slots, **kwargs):
            created.append(kwargs)
            self.compiled = SimpleNamespace()

    monkeypatch.setattr(target_program, "DSparkTargetStage", _Stage)
    batch = SimpleNamespace(batch_capacity=4, query_bucket_rows=8)

    default = _packed_stages()
    default.stage_for(batch)
    assert "ragged_compressor" not in created[-1]

    stages = _packed_stages()
    with pytest.raises(TypeError, match="must be a boolean"):
        stages.configure_ragged_compressor("true")
    stages.configure_ragged_compressor(True)
    stages.stage_for(batch)
    assert created[-1]["ragged_compressor"] is True
    assert created[-1]["state_capacity"] == 8
    with pytest.raises(RuntimeError, match="before warmup"):
        stages.configure_ragged_compressor(False)

    with pytest.raises(ValueError, match="dsv41_dspark_target_query_buckets"):
        _packed_stages(query_bucketing=False).configure_ragged_compressor(True)


@pytest.mark.parametrize(("additional", "stage_supports", "expected", "error"), (
    ({}, True, False, None),
    ({"dsv41_dspark_target_ragged_compressor": True}, True, True, None),
    ({}, False, None, None),
    ({"dsv41_dspark_target_ragged_compressor": "true"}, True, None, TypeError),
    ({"dsv41_dspark_target_ragged_compressor": True}, False, None, RuntimeError),
))
def test_runner_parses_ragged_compressor_flag(monkeypatch, additional, stage_supports, expected, error):
    from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
    from vllm_gaudi.v1.worker import deepseek_v41_runner
    from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner

    calls = []

    class _Stage:

        def configure_execution_buckets(self, enabled, *, buckets):
            pass

    class _RaggedStage(_Stage):

        def configure_ragged_compressor(self, enabled):
            calls.append(enabled)

    class _Adapter:

        def __init__(self, target_stage, draft_stage, **kwargs):
            self.target_stage = target_stage
            self.draft_stage = draft_stage
            self.batch_capacity = kwargs["batch_capacity"]

    target = _RaggedStage() if stage_supports else _Stage()
    state = SimpleNamespace(capacity=4, slots=RequestSlots(4), pending=None, single_owner=None, prefill_owner=None)
    runner = object.__new__(V41ModelRunner)
    runner.vllm_config = SimpleNamespace(additional_config=additional)
    runner.model = SimpleNamespace(batch_state=state, tp_rank=0, pp_rank=0,
                                   initialize_dspark_batch=lambda capacity: (target, _Stage()))
    monkeypatch.setattr(deepseek_v41_runner, "DSparkBatchRunner", _Adapter)
    monkeypatch.setattr(deepseek_v41_runner, "DSparkBatchTrace", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(deepseek_v41_runner.dist, "is_initialized", lambda: False)
    if error is not None:
        with pytest.raises(error, match="dsv41_dspark_target_ragged_compressor"):
            runner._initialize_dspark_batch_runner(4)
        assert calls == []
        return
    runner._initialize_dspark_batch_runner(4)
    assert calls == ([] if expected is None else [expected])


class _RowMoE(nn.Module):

    def forward(self, value, image_mask, *, prequant, shared_prequant, **_kwargs):
        assert shared_prequant is None and value.shape[0] == image_mask.shape[0]
        quantized, scale = prequant
        return (value.float().tanh() * 3 + quantized.float() * scale[:, None]).to(torch.bfloat16)


class _MathLayer(PreparedDecoderLayer):
    """Production C6/row math methods with small CPU weights."""

    def __init__(self, hidden=16, seed=3):
        nn.Module.__init__(self)
        generator = torch.Generator().manual_seed(seed)

        def control(outputs, inputs):
            # One power-of-two term per mHC control output keeps F.linear exact.
            weight = torch.zeros(outputs, inputs)
            columns = torch.randint(0, inputs, (outputs, ), generator=generator)
            weight[torch.arange(outputs), columns] = 2.0**-torch.randint(0, 3, (outputs, ), generator=generator).float()
            return weight

        self.layer = 5
        self.draft = False
        self.draft_mhc = False
        self.collect_target_state = False
        self.eps, self.hc_eps, self.iterations = 1e-6, 1e-6, 20
        self.batch_control_reuse = False
        self.mhc_control_mme = False
        self.mhc_post_collapse = True
        self.mhc_interlayer_collapse = True
        self.ffn_dual_quant = False
        self.ffn_bf16_quant = False
        self.decode_attention_norm_quant = False
        self.dspark_input_norm = False
        for name in ("hc_attn_fn_packed", "hc_ffn_fn_packed", "hc_attn_fn_mme", "hc_ffn_fn_mme", "hc_attn_fn_fp8",
                     "hc_attn_fn_fp8_channel", "hc_ffn_fn_fp8", "hc_ffn_fn_fp8_channel"):
            setattr(self, name, None)
        self.weights = SimpleNamespace(
            hc_attn_fn=control(24, 4 * hidden),
            hc_attn_scale=torch.tensor([0.5, 0.25, 0.125]),
            hc_attn_base=torch.randn(24, generator=generator) * 0.1,
            hc_ffn_fn=control(24, 4 * hidden),
            hc_ffn_scale=torch.tensor([0.25, 0.5, 0.25]),
            hc_ffn_base=torch.randn(24, generator=generator) * 0.1,
            attn_norm=SimpleNamespace(weight=torch.rand(hidden, generator=generator) + 0.5),
            ffn_norm=SimpleNamespace(weight=torch.rand(hidden, generator=generator) + 0.5),
        )
        self.attention = SimpleNamespace(_fused_qkv_quantized=False, _fused_qkv_weight=None, _buffers={})
        self.moe = _RowMoE()


@pytest.fixture
def cpu_ffn_quant(monkeypatch):
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm

    def norm_quant(value, weight, eps):
        normalized = rms_norm(value, weight, eps)
        return normalized, normalized.float() * 0.5, torch.full((value.shape[0], ), 2.0)

    monkeypatch.setattr(torch.ops.custom_op, FFN_NORM_QUANT, norm_quant, raising=False)


def _math_inputs(lengths, hidden=16, seed=5):
    generator = torch.Generator().manual_seed(seed)
    rows = ROWS * len(lengths)
    valid = torch.zeros(rows, dtype=torch.bool)
    for lane, length in enumerate(lengths):
        valid[lane * ROWS:lane * ROWS + length] = True
    residual = torch.randn((rows, 4, hidden), generator=generator).to(torch.bfloat16)
    pre_mix = torch.rand((rows, 4), generator=generator) + 0.1
    attention = torch.randn((rows, hidden), generator=generator).to(torch.bfloat16)
    image_mask = torch.zeros(rows, dtype=torch.bool)
    return valid, residual, pre_mix, attention, image_mask


def _assert_row_count_close(actual, expected):
    # Row math is row independent, but CPU sigmoid rounds its SIMD tail
    # differently, so a compact row count may move mHC gates by FP32 ulps and
    # the following BF16 values by one ulp.  Bitwise C6 equality is asserted
    # separately for the six-row call.
    if actual.dtype == torch.bfloat16:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=2**-7, atol=1e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("lengths", ((1, 6), (3, 0, 2, 0), (1, 2, 3, 4, 5, 6)))
def test_row_math_on_compact_rows_matches_c6_lanes(cpu_ffn_quant, lengths):
    layer = _MathLayer()
    valid, residual, pre_mix, attention, image_mask = _math_inputs(lengths)
    positions = torch.zeros(valid.shape[0], dtype=torch.int32)
    lane_prefix, lane_post = [], []
    for lane in range(len(lengths)):
        rows = slice(lane * ROWS, (lane + 1) * ROWS)
        prefix = layer._forward_c6_attention_prefix(residual[rows], pre_mix[rows], positions[rows], image_mask[rows],
                                                    collapse_handoff={})
        value, new_pre, post, comb, _target, prequant, roundtrip, _schedule = prefix
        assert prequant is None and roundtrip is None
        lane_prefix.append((value, new_pre, post, comb))
        lane_post.append(layer._forward_c6_post_attention(
            attention[rows], residual[rows], new_pre, post, comb, image_mask[rows], schedule=False,
            fp8_decode=False, collapse_handoff={}, publish_collapse=False))
    expected_prefix = [torch.cat(parts) for parts in zip(*lane_prefix, strict=True)]
    expected_post = [torch.cat(parts) for parts in zip(*lane_post, strict=True)]

    _plan_value, indices, query_valid = _plan(SimpleNamespace(valid=valid))
    value, new_pre, post, comb, prequant, roundtrip = attention_input_rows(
        layer, residual[indices], pre_mix[indices])
    assert prequant is None and roundtrip is None
    for actual, expected in zip((value, new_pre, post, comb), expected_prefix, strict=True):
        assert actual.shape[0] == indices.shape[0]
        _assert_row_count_close(actual[query_valid], expected[indices][query_valid])
    compact_post = post_attention_rows(
        layer, attention[indices], residual[indices], new_pre, post, comb, image_mask[indices], schedule=False,
        fp8_decode=False)
    for actual, expected in zip(compact_post, expected_post, strict=True):
        assert actual.shape[0] == indices.shape[0]
        _assert_row_count_close(actual[query_valid], expected[indices][query_valid])


def test_row_math_six_rows_is_the_eager_c6_suffix_and_c6_stays_fixed(cpu_ffn_quant):
    layer = _MathLayer()
    _valid, residual, pre_mix, attention, image_mask = _math_inputs((6, ))
    _value, new_pre, post, comb, *_rest = attention_input_rows(layer, residual, pre_mix)
    expected = layer._forward_c6_post_attention(
        attention, residual, new_pre, post, comb, image_mask, schedule=False, fp8_decode=False, collapse_handoff={},
        publish_collapse=False)
    actual = post_attention_rows(layer, attention, residual, new_pre, post, comb, image_mask, schedule=False,
                                 fp8_decode=False)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    for rows in (1, 7, 13):
        _valid, residual, pre_mix, attention, image_mask = _math_inputs((6, 6, 6))
        _value, new_pre, post, comb, *_rest = attention_input_rows(layer, residual[:rows], pre_mix[:rows])
        result = post_attention_rows(layer, attention[:rows], residual[:rows], new_pre, post, comb,
                                     image_mask[:rows], schedule=False, fp8_decode=False)
        assert result[0].shape == residual[:rows].shape
        with pytest.raises(ValueError, match="exactly six rows"):
            layer._forward_c6_post_attention(attention[:rows], residual[:rows], new_pre, post, comb,
                                             image_mask[:rows], schedule=False, fp8_decode=False,
                                             collapse_handoff={}, publish_collapse=False)


@pytest.mark.parametrize("ratio", (2, 4))
def test_compact_compressor_is_one_fullgraph_across_active_patterns(monkeypatch, ratio):
    from vllm_gaudi.ops import deepseek_v41_batch_target as target_ops

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)

    def write(bank, values, rows):
        # Traceable facade; no valid row in these rounds targets bank row 0.
        valid = rows >= 0
        safe = torch.where(valid, rows, 0).long()
        current = bank.index_select(0, safe)
        bank.index_copy_(0, safe, torch.where(valid[:, None], values.to(bank.dtype), current))
        return torch.where(valid, rows, -1).to(torch.int32)

    def read(bank, rows, completion):
        values = bank.index_select(0, rows.clamp_min(0).long())
        return torch.where((rows >= 0)[:, None], values, torch.zeros_like(values))

    def native_mix(history, score_history, kv, score, positions, **_kwargs):
        return torch.where((positions >= 0)[:, None], kv + history[:6], 0).to(torch.bfloat16)

    monkeypatch.setattr(batch_attention, "write_state_rows", write)
    monkeypatch.setattr(batch_attention, "read_state_rows", read)
    monkeypatch.setattr(target_ops, "compressor_sequence_slot_native_mix", native_mix)
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    torch._dynamo.reset()
    compiled = torch.compile(compress_target_query_rows, backend=backend, fullgraph=True, dynamic=False)
    try:
        for lengths in ((2, 0, 3, 1), (0, 3, 1, 2), (1, 1, 1, 1)):
            round_ = _round(lengths)
            plan, indices, valid = _plan(round_)
            assert plan.bucket_rows == 8
            eager_owner, compiled_owner = _Compressor(ratio, 4), _Compressor(ratio, 4)
            expected = _run_ragged(eager_owner, round_, indices, valid)
            actual = compiled(compiled_owner, round_.value, round_.positions, round_.slots, round_.pages,
                              round_.valid, (ROWS, ) * 4, (0, 1, 2, 3), indices, valid)
            _assert_same_state(eager_owner, compiled_owner, expected, actual)
    finally:
        torch._dynamo.reset()
    assert len(graphs) == 1
