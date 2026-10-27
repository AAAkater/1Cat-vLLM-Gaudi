# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for request-local C6 math inside the packed target stage."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_gaudi.models import deepseek_v41_batch_target_program as target_module
from vllm_gaudi.models import deepseek_v41_program as program_module
from vllm_gaudi.models.deepseek_v41_batch_target_program import (
    CompiledTargetBatchStage,
    DSparkTargetStage,
    PreparedTargetBatchLayerGroup,
)
from vllm_gaudi.models.deepseek_v41_program import PreparedDecoderLayer
from vllm_gaudi.ops import deepseek_v41_batch_attention as batch_attention
from vllm_gaudi.ops import deepseek_v41_paged_attention as paged_module
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention


ROWS_PER_LANE = 6


@pytest.fixture(autouse=True)
def restore_lazy_environment_attributes():
    # monkeypatch restores values obtained through module __getattr__ as real
    # attributes. Remove those shadows so later configuration tests stay lazy.
    from vllm_gaudi import envs

    original = {name: envs.__dict__[name] for name in envs.environment_variables if name in envs.__dict__}
    yield
    for name in envs.environment_variables:
        if name in original:
            envs.__dict__[name] = original[name]
        else:
            envs.__dict__.pop(name, None)


class _PackedAttention(nn.Module):
    def __init__(self, capacity, calls):
        super().__init__()
        self.ratio = 0
        self.batch_state = SimpleNamespace(swa=torch.empty(64, 8, 8))
        self.swa = self.batch_state.swa
        self._buffers = {}
        self._fused_qkv_quantized = False
        self._fused_qkv_weight = None
        self.shared_decode_metadata = False
        self.calls = calls

    def forward_batch_target(self, value, positions, slots, pages, selected, candidates,
                             main_ready, index_ready, **kwargs):
        tile_indices = kwargs.get("tile_indices")
        if self.calls is not None:
            self.calls.append(("attention", value.shape[0], tuple(kwargs["row_valid"].shape)))
            self.calls.append((
                "tile_plan",
                None if tile_indices is None else id(tile_indices),
                None if tile_indices is None else tuple(tile_indices.shape),
                None if tile_indices is None else tuple(tile_indices.tolist()),
            ))
        if tile_indices is not None:
            # Make the runtime tile plan an observable graph input without
            # changing the fake attention values.
            value = value + tile_indices.sum().to(value.dtype) * 0
        if kwargs.get("defer_output_finish", False):
            return value, selected, candidates, main_ready, index_ready
        return value + 1, selected, candidates, main_ready, index_ready

    def finish_output_target(self, output, positions, active, ready_outputs=()):
        if self.calls is not None:
            self.calls.append(("finish", output.shape[0], tuple(active.shape)))
        return torch.where(active[:, None], output + 1, torch.zeros_like(output))

    def forward(self, value, positions, ready_outputs=(), **kwargs):
        if self.calls is not None:
            self.calls.append(("serial_attention", value.shape[0], kwargs.get("decode")))
        return value + 3


class _DeterministicMoE(nn.Module):
    n256_fp8 = True
    n256_fused = True
    shared_gate_up_channel = torch.ones(1)

    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    def forward(self, value, image_mask, **kwargs):
        if self.calls is not None:
            self.calls.append((
                "moe",
                value.shape[0],
                kwargs.get("decode"),
                kwargs.get("ordinary_decode", False),
                kwargs.get("prequant"),
                kwargs.get("shared_prequant"),
            ))
        return value + 2


class _FakePreparedLayer(PreparedDecoderLayer):
    def __init__(self, layer, capacity, calls, *, collect_target_state=False, simulate_publish=False):
        # Deliberately bypass the checkpoint constructor while retaining the
        # production PreparedDecoderLayer methods under test.
        nn.Module.__init__(self)
        self.layer = layer
        self.calls = calls
        self.simulate_publish = simulate_publish
        self.draft = False
        self.draft_mhc = False
        self.collect_target_state = collect_target_state
        self.dspark_mhc_deferred = False
        self.dspark_peer_post_collapse = False
        self.dspark_moe_peer_post = False
        self.dspark_peer_post_norm = False
        self.mhc_post_collapse = True
        self.mhc_interlayer_collapse = True
        self.mhc_interlayer_bf16 = True
        self.ffn_dual_quant = True
        self.ffn_bf16_quant = True
        self.mhc_control_mme = False
        self.decode_attention_norm_quant = False
        self.dspark_input_norm = False
        self.batch_control_reuse = False
        self.hc_eps = 1e-6
        self.iterations = 20
        self.eps = 1e-6
        self.hc_attn_fn_packed = torch.ones(1)
        self.hc_ffn_fn_packed = torch.ones(1)
        self.hc_attn_fn_mme = None
        self.hc_ffn_fn_mme = None
        self.hc_attn_fn_fp8 = None
        self.hc_attn_fn_fp8_channel = None
        self.hc_ffn_fn_fp8 = None
        self.hc_ffn_fn_fp8_channel = None
        self.weights = SimpleNamespace(
            hc_attn_fn=torch.empty(1),
            hc_attn_scale=torch.empty(1),
            hc_attn_base=torch.empty(1),
            hc_ffn_fn=torch.empty(1),
            hc_ffn_scale=torch.empty(1),
            hc_ffn_base=torch.empty(1),
            attn_norm=SimpleNamespace(weight=torch.empty(1)),
            ffn_norm=SimpleNamespace(weight=torch.empty(1)),
        )
        self.attention = _PackedAttention(capacity, calls)
        self.moe = _DeterministicMoE(calls)
        self.all_gather = lambda value, dim: value

    def forward_batch_target(self, *args, **kwargs):
        if self.calls is not None:
            self.calls.append((
                "target_layer",
                self.layer,
                kwargs.get("publish_collapse", False),
                len(kwargs.get("collapse_handoffs", ())),
            ))
        result = super().forward_batch_target(*args, **kwargs)
        if self.simulate_publish and kwargs.get("publish_collapse", False):
            residual = result[0]
            for lane, handoff in enumerate(kwargs["collapse_handoffs"]):
                start = lane * ROWS_PER_LANE
                handoff[self.layer + 1] = residual[start:start + ROWS_PER_LANE]
        return result


def _disable_optional_target_branches(monkeypatch):
    env = program_module.gaudi_envs
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_MHC_SCHEDULE", True)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_MHC_GATE_PACKET", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_MHC_POST_STATS", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_PEER_POST_COLLAPSE", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_MOE_PEER_POST", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_PEER_POST_NORM", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_PEER_POST_COLLAPSE", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_PEER_POST_NORM", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_SCHEDULED_PEER", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_LAYER_MAIN_SPLIT", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_LAYER_MAIN_REUSE", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_NORM_ROUNDTRIP", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_INPUT_QUANT_REMAT", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_WO_HANDOFF", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_Q_PROLOGUE", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_C1_DENSE_CHAIN", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_FP8_QKV_PROLOGUE", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_FP8_QKV_PROLOGUE_SPLIT", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_INPUT_FP8", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_OUTPUT_FP8", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_DSPARK_SHARED_FP8", False)
    monkeypatch.setattr(env, "VLLM_HPU_DSV41_SHARED_GATE_UP", False)


def _install_deterministic_math(monkeypatch, calls):
    def hc_pre(residual, previous_pre, *args, **kwargs):
        if calls is not None:
            calls.append(("hc_pre", residual.shape[0], kwargs.get("decode")))
        value = residual.mean(1)
        return value, previous_pre, value, value

    def hc_post(value, residual, post, comb):
        if calls is not None:
            calls.append(("hc_post", value.shape[0]))
        return residual + value.unsqueeze(1)

    def rms_norm(value, *args, **kwargs):
        if calls is not None:
            calls.append(("norm", value.shape[0]))
        return value

    def dual_quant(value, weight, eps):
        if calls is not None:
            calls.append(("quant", value.shape[0]))
        zero = torch.zeros_like(value)
        one = torch.ones_like(value)
        return value, zero, one, zero, one

    def final_collapse(residual, pre_mix, weight, eps):
        if calls is not None:
            calls.append(("final_collapse", residual.shape[0]))
        return residual.mean(1) * 0.5

    def bf16_identity(value):
        if calls is not None:
            calls.append(("bf16_identity", value.shape[0]))
        return value

    def context_state(residual, capacity, *, grouped_prefill):
        if calls is not None:
            calls.append(("context_state", residual.shape[0], capacity, grouped_prefill))
        return residual.mean(1)

    monkeypatch.setattr(program_module, "hc_pre", hc_pre)
    monkeypatch.setattr(program_module, "hc_post", hc_post)
    monkeypatch.setattr(program_module, "rms_norm", rms_norm)
    monkeypatch.setattr(program_module, "decode_ffn_dual_quant", dual_quant)
    monkeypatch.setattr(program_module, "draft_context_state", context_state)
    monkeypatch.setattr(target_module, "decode_bf16_identity", bf16_identity)
    monkeypatch.setattr(target_module, "final_collapse_rms_norm", final_collapse)


def _install_target_helper_counters(monkeypatch, calls):
    for name, label in (
        ("_target_c6_attention_prefix", "target_prefix"),
        ("_target_c6_post_attention", "target_post"),
    ):
        helper = getattr(PreparedDecoderLayer, name)

        def counted(layer, *args, _helper=helper, _label=label, **kwargs):
            calls.append((_label, args[0].shape[0]))
            return _helper(layer, *args, **kwargs)

        monkeypatch.setattr(PreparedDecoderLayer, name, counted)


def _make_stage(capacity, calls, *, simulate_publish_layers=(), layer_count=4):
    stage = nn.Module()
    stage.dspark = True
    stage.pp_rank = 0
    stage.tp_rank = 0
    stage.is_last_stage = True
    stage.tensor_parallel_size = 8
    stage.reduce = lambda value: value
    stage.config = {"text_config": {"rms_norm_eps": 1e-6}}
    stage.weights = nn.Module()
    stage.weights.embed = nn.Embedding(32, 4)
    stage.weights.norm = nn.Module()
    stage.weights.norm.register_buffer("weight", torch.ones(4))
    stage.shared = nn.Module()
    stage.shared.topk = nn.ModuleDict({"0": nn.Module()})
    stage.shared.sources = nn.ModuleDict({"0": nn.Module()})
    stage.layers = nn.ModuleList([
        _FakePreparedLayer(
            layer,
            capacity,
            calls,
            collect_target_state=(layer == 2),
            simulate_publish=(layer in simulate_publish_layers),
        )
        for layer in range(layer_count)
    ])
    return stage


def _make_group(capacity, calls, *, simulate_publish_layers=()):
    stage = _make_stage(capacity, calls, simulate_publish_layers=simulate_publish_layers)
    return PreparedTargetBatchLayerGroup(
        stage,
        0,
        4,
        request_lengths=(ROWS_PER_LANE,) * capacity,
        request_slots=tuple(range(capacity)),
        text_input=False,
    )


def _inputs(capacity, *, bases, valid_lengths, positions_offset=0):
    rows = capacity * ROWS_PER_LANE
    residual = torch.zeros((rows, 4, 4), dtype=torch.bfloat16)
    positions = torch.full((rows,), -1, dtype=torch.int32)
    slots = torch.full((rows,), -1, dtype=torch.int32)
    row_valid = torch.zeros(rows, dtype=torch.bool)
    for lane, (base, length) in enumerate(zip(bases, valid_lengths, strict=True)):
        start = lane * ROWS_PER_LANE
        stop = start + ROWS_PER_LANE
        residual[start:stop] = float(base)
        positions[start:start + length] = torch.arange(
            positions_offset + lane * 100,
            positions_offset + lane * 100 + length,
            dtype=torch.int32,
        )
        slots[start:start + length] = lane
        row_valid[start:start + length] = True
    selected = torch.full((rows, 512), -1, dtype=torch.int32)
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32)
    main_ready = torch.zeros(rows, dtype=torch.int32)
    index_ready = torch.zeros(rows, dtype=torch.int32)
    return {
        "residual": residual,
        "pre_mix": torch.zeros((rows, 4), dtype=torch.float32),
        "positions": positions,
        "input_ids": torch.zeros(rows, dtype=torch.long),
        "engram_rows": (),
        "slots": slots,
        "pages": torch.ones((rows, 4), dtype=torch.int32),
        "row_valid": row_valid,
        "selections": (selected,),
        "candidates": candidates,
        "main_ready": (main_ready,),
        "index_ready": (index_ready,),
    }


def _run(group, inputs, active_lanes=None, *, tile_indices=target_module._AUTO_TARGET_MLA_TILE_PLAN):
    kwargs = {} if active_lanes is None else {"active_lanes": active_lanes}
    if tile_indices is not target_module._AUTO_TARGET_MLA_TILE_PLAN:
        kwargs["tile_indices"] = tile_indices
    return group(
        inputs["residual"],
        inputs["pre_mix"],
        inputs["positions"],
        inputs["input_ids"],
        inputs["engram_rows"],
        inputs["slots"],
        inputs["pages"],
        inputs["row_valid"],
        inputs["selections"],
        inputs["candidates"],
        inputs["main_ready"],
        inputs["index_ready"],
        **kwargs,
    )


def _count(calls, name):
    return [call for call in calls if call[0] == name]


def test_target_c6_graph_break_helpers_are_dynamo_disabled():
    for name in ("_target_c6_attention_prefix", "_target_c6_post_attention"):
        assert getattr(getattr(PreparedDecoderLayer, name), "_torchdynamo_disable", False)


def test_target_c6_slices_keep_one_packed_attention_and_static_lane_calls(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    _install_target_helper_counters(monkeypatch, calls)
    capacity = 3
    group = _make_group(capacity, calls, simulate_publish_layers=(0, 1))

    output, _, aux, _, _, _, _ = _run(
        group,
        _inputs(capacity, bases=(10, 20, 30), valid_lengths=(6, 3, 0)),
    )

    attention = _count(calls, "attention")
    assert len(attention) == 4
    assert all(row_count == capacity * ROWS_PER_LANE for _, row_count, _ in attention)
    assert all(shape == (capacity * ROWS_PER_LANE,) for _, _, shape in attention)

    assert len(_count(calls, "target_layer")) == 4
    assert [call[2] for call in _count(calls, "target_layer")] == [True, True, True, False]
    assert all(call[3] == capacity for call in _count(calls, "target_layer"))

    # Only the target path enters the Dynamo-disabled wrappers.
    assert len(_count(calls, "target_prefix")) == 4 * capacity
    assert len(_count(calls, "target_post")) == 4 * capacity
    assert all(call[1] == ROWS_PER_LANE for call in _count(calls, "target_prefix"))
    assert all(call[1] == ROWS_PER_LANE for call in _count(calls, "target_post"))

    # One prefix and one suffix hc_pre per lane, per layer.
    assert len(_count(calls, "hc_pre")) == 4 * capacity * 2
    assert all(call[1] == ROWS_PER_LANE and call[2] is True for call in _count(calls, "hc_pre"))
    assert len(_count(calls, "norm")) == 4 * capacity
    assert all(call[1] == ROWS_PER_LANE for call in _count(calls, "norm"))
    assert len(_count(calls, "quant")) == 4 * capacity
    assert len(_count(calls, "moe")) == 4 * capacity
    assert all(call[1] == ROWS_PER_LANE and call[2] is True and call[3] is False
               for call in _count(calls, "moe"))
    assert len(_count(calls, "final_collapse")) == capacity
    assert all(call[1] == ROWS_PER_LANE for call in _count(calls, "final_collapse"))

    # Native publication is simulated on layers 0/1.  Their BF16 identity is
    # skipped; layer 2 has no handoff and must cross the explicit boundary.
    assert len(_count(calls, "bf16_identity")) == capacity
    assert len(_count(calls, "context_state")) == capacity
    assert aux.shape == (capacity * ROWS_PER_LANE, 4)

    active = _inputs(capacity, bases=(10, 20, 30), valid_lengths=(6, 3, 0))["row_valid"]
    assert torch.count_nonzero(output[~active]).item() == 0
    assert torch.count_nonzero(aux[~active]).item() == 0


def test_target_c6_slices_do_not_cross_lane_sentinels_or_change_calls_with_active_count(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    capacity = 3
    group = _make_group(capacity, calls)

    first = _inputs(capacity, bases=(10, 20, 30), valid_lengths=(6, 6, 6))
    first_output, _, _, _, _, _, _ = _run(group, first)
    first_counts = [call[0] for call in calls]
    calls.clear()

    changed_neighbor = _inputs(capacity, bases=(10, 70, 30), valid_lengths=(6, 6, 6))
    changed_output, _, _, _, _, _, _ = _run(group, changed_neighbor)
    changed_counts = [call[0] for call in calls]

    lane = slice(0, ROWS_PER_LANE)
    other_lane = slice(2 * ROWS_PER_LANE, 3 * ROWS_PER_LANE)
    assert torch.equal(first_output[lane], changed_output[lane])
    assert torch.equal(first_output[other_lane], changed_output[other_lane])
    assert not torch.equal(
        first_output[ROWS_PER_LANE:2 * ROWS_PER_LANE],
        changed_output[ROWS_PER_LANE:2 * ROWS_PER_LANE],
    )

    calls.clear()
    partial = _inputs(capacity, bases=(10, 20, 30), valid_lengths=(6, 0, 0))
    _run(group, partial)
    partial_counts = [call[0] for call in calls]
    assert partial_counts == first_counts == changed_counts


def test_target_c6_slice_active_lanes_match_unfiltered_baseline(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    _install_target_helper_counters(monkeypatch, calls)
    capacity = 4
    active_lanes = (1, 3)
    group = _make_group(capacity, calls, simulate_publish_layers=(0, 1))
    inputs = _inputs(capacity, bases=(10, 20, 30, 40), valid_lengths=(0, 6, 0, 3))

    baseline = _run(group, inputs)
    calls.clear()

    def forbidden_device_read(*_args, **_kwargs):
        raise AssertionError("active lanes must not be derived from a device mask")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden_device_read)
        patch.setattr(torch.Tensor, "cpu", forbidden_device_read)
        filtered = _run(group, inputs, active_lanes=active_lanes)

    for lane in active_lanes:
        lane_slice = slice(lane * ROWS_PER_LANE, (lane + 1) * ROWS_PER_LANE)
        assert torch.equal(filtered[0][lane_slice], baseline[0][lane_slice])
        assert torch.equal(filtered[1][lane_slice], baseline[1][lane_slice])
        assert torch.equal(filtered[2][lane_slice], baseline[2][lane_slice])

    lane_active = inputs["row_valid"].reshape(capacity, ROWS_PER_LANE).any(1)
    inactive_rows = (~lane_active).repeat_interleave(ROWS_PER_LANE)
    assert torch.equal(filtered[0][inactive_rows], torch.zeros_like(filtered[0][inactive_rows]))
    assert torch.equal(filtered[1][inactive_rows], torch.zeros_like(filtered[1][inactive_rows]))
    assert torch.equal(filtered[2][inactive_rows], torch.zeros_like(filtered[2][inactive_rows]))

    assert len(_count(calls, "target_prefix")) == 4 * len(active_lanes)
    assert len(_count(calls, "target_post")) == 4 * len(active_lanes)
    assert len(_count(calls, "hc_pre")) == 4 * len(active_lanes) * 2
    assert len(_count(calls, "norm")) == 4 * len(active_lanes)
    assert len(_count(calls, "quant")) == 4 * len(active_lanes)
    assert len(_count(calls, "moe")) == 4 * len(active_lanes)
    assert len(_count(calls, "final_collapse")) == len(active_lanes)
    assert len(_count(calls, "bf16_identity")) == len(active_lanes)
    assert len(_count(calls, "context_state")) == len(active_lanes)
    attention = _count(calls, "attention")
    assert len(attention) == 4
    assert all(row_count == capacity * ROWS_PER_LANE for _, row_count, _ in attention)
    finish = _count(calls, "finish")
    assert len(finish) == 4 * len(active_lanes)
    assert all(row_count == ROWS_PER_LANE for _, row_count, _ in finish)


@pytest.mark.parametrize(
    ("capacity", "active_lanes"),
    [
        (1, (0, )),
        (2, (1, )),
        (3, (0, 2)),
        (4, (1, 3)),
        (8, (0, 7)),
        (16, (1, 15)),
        (32, (0, 31)),
        (32, tuple(range(32))),
    ],
)
def test_target_c6_slice_active_lanes_cover_generic_capacity(monkeypatch, capacity, active_lanes):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    _install_target_helper_counters(monkeypatch, calls)
    valid_lengths = [0] * capacity
    for lane in active_lanes:
        valid_lengths[lane] = 6 if lane % 2 == 0 else 1
    bases = tuple(10 + lane for lane in range(capacity))
    group = _make_group(capacity, calls, simulate_publish_layers=(0, 1))
    inputs = _inputs(capacity, bases=bases, valid_lengths=tuple(valid_lengths))

    baseline = _run(group, inputs)
    calls.clear()
    filtered = _run(group, inputs, active_lanes=active_lanes)

    for lane in active_lanes:
        lane_slice = slice(lane * ROWS_PER_LANE, (lane + 1) * ROWS_PER_LANE)
        assert torch.equal(filtered[0][lane_slice], baseline[0][lane_slice])
        assert torch.equal(filtered[1][lane_slice], baseline[1][lane_slice])
        assert torch.equal(filtered[2][lane_slice], baseline[2][lane_slice])

    lane_active = inputs["row_valid"].reshape(capacity, ROWS_PER_LANE).any(1)
    inactive_rows = (~lane_active).repeat_interleave(ROWS_PER_LANE)
    assert torch.equal(filtered[0][inactive_rows], torch.zeros_like(filtered[0][inactive_rows]))
    assert torch.equal(filtered[1][inactive_rows], torch.zeros_like(filtered[1][inactive_rows]))
    assert torch.equal(filtered[2][inactive_rows], torch.zeros_like(filtered[2][inactive_rows]))
    assert len(_count(calls, "target_prefix")) == 4 * len(active_lanes)
    assert len(_count(calls, "target_post")) == 4 * len(active_lanes)
    assert len(_count(calls, "moe")) == 4 * len(active_lanes)
    assert len(_count(calls, "final_collapse")) == len(active_lanes)
    assert len(_count(calls, "bf16_identity")) == len(active_lanes)
    attention = _count(calls, "attention")
    assert len(attention) == 4
    assert all(row_count == capacity * ROWS_PER_LANE for _, row_count, _ in attention)
    finish = _count(calls, "finish")
    assert len(finish) == 4 * len(active_lanes)
    assert all(row_count == ROWS_PER_LANE for _, row_count, _ in finish)


def test_target_c6_group_auto_plans_n32_sparse_but_explicit_none_skips_planner(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    planner_calls = []
    real_plan = batch_attention.batch_packed_mla_tile_plan

    def counting_plan(active_lanes, **kwargs):
        planner_calls.append((active_lanes, kwargs))
        return real_plan(active_lanes, **kwargs)

    monkeypatch.setattr(batch_attention, "batch_packed_mla_tile_plan", counting_plan)
    capacity = 32
    group = _make_group(capacity, calls)
    bases = tuple(10 + lane for lane in range(capacity))
    partial_lengths = [0] * capacity
    partial_lengths[0] = partial_lengths[31] = 6
    partial = _inputs(capacity, bases=bases, valid_lengths=tuple(partial_lengths))

    _run(group, partial, active_lanes=(0, 31))
    plans = _count(calls, "tile_plan")
    assert len(plans) == 4
    assert all(call[1] is not None and call[2] == (2, ) and call[3] == (0, 23) for call in plans)
    assert len({call[1] for call in plans}) == 1
    assert len(planner_calls) == 1

    calls.clear()
    planner_calls.clear()
    all_active = _inputs(capacity, bases=bases, valid_lengths=(6, ) * capacity)
    _run(group, all_active, active_lanes=tuple(range(capacity)))
    plans = _count(calls, "tile_plan")
    assert len(plans) == 4
    assert all(call[1] is None for call in plans)
    assert planner_calls == []

    calls.clear()
    planner_calls.clear()
    all_but_last = tuple(range(capacity - 1))
    covered_lengths = [0] * capacity
    for lane in all_but_last:
        covered_lengths[lane] = 6
    covered = _inputs(capacity, bases=bases, valid_lengths=tuple(covered_lengths))
    _run(group, covered, active_lanes=all_but_last)
    plans = _count(calls, "tile_plan")
    assert len(plans) == 4
    assert all(call[1] is None for call in plans)
    assert len(planner_calls) == 1

    calls.clear()
    planner_calls.clear()
    _run(group, partial, active_lanes=(0, 31), tile_indices=None)
    plans = _count(calls, "tile_plan")
    assert len(plans) == 4
    assert all(call[1] is None for call in plans)
    assert planner_calls == []


@pytest.mark.parametrize("capacity", (3, 6, 11, 64))
def test_target_c6_non_n32_keeps_full_mla_path(monkeypatch, capacity):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    group = _make_group(capacity, calls)
    active_lanes = (0, capacity - 1)
    valid_lengths = [0] * capacity
    for lane in active_lanes:
        valid_lengths[lane] = 6
    inputs = _inputs(
        capacity,
        bases=tuple(10 + lane for lane in range(capacity)),
        valid_lengths=tuple(valid_lengths),
    )

    _run(group, inputs, active_lanes=active_lanes)

    plans = _count(calls, "tile_plan")
    assert len(plans) == 4
    assert all(call[1] is None for call in plans)


def test_target_c6_stage_uploads_one_n32_tile_plan_for_ten_groups(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    capacity = 32
    stage = _make_stage(capacity, calls, simulate_publish_layers=(0, 1), layer_count=40)
    planner_calls = []
    real_plan = batch_attention.batch_packed_mla_tile_plan

    def counting_plan(active_lanes, **kwargs):
        planner_calls.append((active_lanes, kwargs))
        return real_plan(active_lanes, **kwargs)

    monkeypatch.setattr(batch_attention, "batch_packed_mla_tile_plan", counting_plan)
    monkeypatch.setattr(target_module.torch, "compile", lambda call, **_kwargs: call)
    compiled = CompiledTargetBatchStage(
        stage,
        (ROWS_PER_LANE, ) * capacity,
        tuple(range(capacity)),
        backend="eager",
    )
    assert len(compiled.groups) == 10
    valid_lengths = [0] * capacity
    valid_lengths[0] = valid_lengths[31] = 6
    inputs = _inputs(
        capacity,
        bases=tuple(10 + lane for lane in range(capacity)),
        valid_lengths=tuple(valid_lengths),
    )

    compiled(
        inputs["residual"],
        inputs["pre_mix"],
        inputs["input_ids"],
        inputs["positions"],
        inputs["slots"],
        inputs["pages"],
        inputs["row_valid"],
        inputs["engram_rows"],
        active_lanes=(0, 31),
    )

    assert len(planner_calls) == 1
    assert planner_calls[0][0] == (0, 31)
    assert planner_calls[0][1]["total_rows"] == capacity * ROWS_PER_LANE
    plans = _count(calls, "tile_plan")
    assert len(plans) == 40
    assert len({call[1] for call in plans}) == 1
    assert all(call[2] == (2, ) and call[3] == (0, 23) for call in plans)
    assert len(_count(calls, "attention")) == 40


@pytest.mark.parametrize(
    ("capacity", "active_lanes"),
    (
        (32, tuple(range(32))),
        (3, (0, 2)),
    ),
)
def test_target_c6_stage_skips_tile_plan_for_full_or_non_n32(monkeypatch, capacity, active_lanes):
    _disable_optional_target_branches(monkeypatch)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    stage = _make_stage(capacity, calls)

    def unexpected_plan(*_args, **_kwargs):
        raise AssertionError("full and non-N32 rounds must not upload a tile plan")

    monkeypatch.setattr(batch_attention, "batch_packed_mla_tile_plan", unexpected_plan)
    monkeypatch.setattr(target_module.torch, "compile", lambda call, **_kwargs: call)
    compiled = CompiledTargetBatchStage(
        stage,
        (ROWS_PER_LANE, ) * capacity,
        tuple(range(capacity)),
        backend="eager",
    )
    valid_lengths = [0] * capacity
    for lane in active_lanes:
        valid_lengths[lane] = 6
    inputs = _inputs(
        capacity,
        bases=tuple(10 + lane for lane in range(capacity)),
        valid_lengths=tuple(valid_lengths),
    )

    compiled(
        inputs["residual"],
        inputs["pre_mix"],
        inputs["input_ids"],
        inputs["positions"],
        inputs["slots"],
        inputs["pages"],
        inputs["row_valid"],
        inputs["engram_rows"],
        active_lanes=active_lanes,
    )

    plans = _count(calls, "tile_plan")
    assert len(plans) == 4
    assert all(call[1] is None for call in plans)


def test_target_c6_slice_active_lanes_reuse_sparse_patterns(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    _install_target_helper_counters(monkeypatch, calls)
    capacity = 8
    group = _make_group(capacity, calls, simulate_publish_layers=(0, 1))
    patterns = ((0, ), (7, ), (1, 6), (0, 7))
    for active_lanes in patterns:
        valid_lengths = [0] * capacity
        for lane in active_lanes:
            valid_lengths[lane] = 6 if lane % 2 == 0 else 1
        inputs = _inputs(
            capacity,
            bases=tuple(10 + lane for lane in range(capacity)),
            valid_lengths=tuple(valid_lengths),
        )
        baseline = _run(group, inputs)
        calls.clear()
        filtered = _run(group, inputs, active_lanes=active_lanes)
        for lane in active_lanes:
            lane_slice = slice(lane * ROWS_PER_LANE, (lane + 1) * ROWS_PER_LANE)
            assert torch.equal(filtered[0][lane_slice], baseline[0][lane_slice])
            assert torch.equal(filtered[1][lane_slice], baseline[1][lane_slice])
            assert torch.equal(filtered[2][lane_slice], baseline[2][lane_slice])
        assert len(_count(calls, "target_prefix")) == 4 * len(active_lanes)
        assert len(_count(calls, "target_post")) == 4 * len(active_lanes)
        assert len(_count(calls, "moe")) == 4 * len(active_lanes)


class _HostActiveBatch(SimpleNamespace):
    @property
    def host_active_lanes(self):
        return tuple(lane for lane, active in enumerate(self.host_request_active) if active)


def test_target_c6_stage_validates_host_active_lanes_without_device_reads(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    _install_target_helper_counters(monkeypatch, calls)
    capacity = 4
    stage = _make_stage(capacity, calls, simulate_publish_layers=(0, 1))
    stage.batch_state = SimpleNamespace(capacity=capacity)
    monkeypatch.setattr(target_module.torch, "compile", lambda call, **_kwargs: call)
    target_stage = DSparkTargetStage(stage, (ROWS_PER_LANE, ) * capacity, tuple(range(capacity)), backend="eager")
    inputs = _inputs(capacity, bases=(10, 20, 30, 40), valid_lengths=(0, 6, 0, 3))
    batch = _HostActiveBatch(
        batch_capacity=capacity,
        host_request_active=(False, True, False, True),
        request_lengths=(ROWS_PER_LANE, ) * capacity,
        input_ids=inputs["input_ids"],
        positions=inputs["positions"],
        row_slots=inputs["slots"],
        row_pages=inputs["pages"],
        row_valid=inputs["row_valid"],
    )
    rows = capacity * ROWS_PER_LANE
    hidden = torch.zeros((rows, 4, 4), dtype=torch.bfloat16)
    pre_mix = torch.zeros((rows, 4), dtype=torch.float32)

    def forbidden_device_read(*_args, **_kwargs):
        raise AssertionError("stage boundary must use host_request_active only")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden_device_read)
        patch.setattr(torch.Tensor, "cpu", forbidden_device_read)
        output, aux = target_stage(hidden, pre_mix, batch)

    lane_active = inputs["row_valid"].reshape(capacity, ROWS_PER_LANE).any(1)
    inactive_rows = (~lane_active).repeat_interleave(ROWS_PER_LANE)
    assert torch.equal(output[inactive_rows], torch.zeros_like(output[inactive_rows]))
    assert torch.equal(aux[inactive_rows], torch.zeros_like(aux[inactive_rows]))
    assert len(_count(calls, "target_prefix")) == 4 * 2
    assert len(_count(calls, "target_post")) == 4 * 2
    assert len(_count(calls, "attention")) == 4

    batch.host_request_active = (False, ) * capacity
    with pytest.raises(ValueError, match="at least one active host lane"):
        target_stage(hidden, pre_mix, batch)


def test_target_c6_non_slice_stage_ignores_host_filter(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    capacity = 3
    stage = _make_stage(capacity, calls)

    class _NonSliceLayer(nn.Module):
        _target_c6_slices = False

        def __init__(self, index):
            super().__init__()
            self.layer = index
            self.attention = SimpleNamespace(ratio=0)

        def forward_batch_target(self, residual, pre_mix, positions, image_mask, engram_rows, slots, pages,
                                 selected, candidates, main_ready, index_ready, row_valid, request_lengths,
                                 request_slots):
            calls.append(("target_layer", self.layer, residual.shape[0]))
            return residual, pre_mix, selected, candidates, main_ready, index_ready, None

    stage.layers = nn.ModuleList([_NonSliceLayer(index) for index in range(4)])
    monkeypatch.setattr(target_module.torch, "compile", lambda call, **_kwargs: call)
    compiled = CompiledTargetBatchStage(
        stage,
        (ROWS_PER_LANE, ) * capacity,
        tuple(range(capacity)),
        backend="eager",
    )
    inputs = _inputs(capacity, bases=(10, 20, 30), valid_lengths=(6, 0, 0))

    compiled(
        inputs["residual"],
        inputs["pre_mix"],
        inputs["input_ids"],
        inputs["positions"],
        inputs["slots"],
        inputs["pages"],
        inputs["row_valid"],
        inputs["engram_rows"],
        active_lanes=(1, ),
    )

    target_layers = _count(calls, "target_layer")
    assert len(target_layers) == 4
    assert all(row_count == capacity * ROWS_PER_LANE for _, _, row_count in target_layers)


def test_target_c6_n2_fallback_quantizes_each_lane_without_shared_prequant(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)

    def ordinary_norm_quant(value, weight, eps):
        del weight, eps
        calls.append(("norm_quant", value.shape[0]))
        scale = torch.ones((value.shape[0], ), dtype=torch.float32)
        return value, value, scale

    monkeypatch.setattr(
        torch.ops.custom_op,
        "custom_deepseek_v41_ffn_norm_quant_gaudi2",
        ordinary_norm_quant,
        raising=False,
    )
    capacity = 2
    group = _make_group(capacity, calls)
    for layer in group.layers:
        layer.moe.shared_gate_up_channel = None
        assert layer.ffn_dual_quant is True
        assert layer._c6_slice_path_error() is None

    _run(group, _inputs(capacity, bases=(10, 20), valid_lengths=(6, 3)))

    norm_quant = _count(calls, "norm_quant")
    assert len(norm_quant) == 4 * capacity
    assert all(call[1] == ROWS_PER_LANE for call in norm_quant)
    assert _count(calls, "quant") == []

    moe = _count(calls, "moe")
    assert len(moe) == 4 * capacity
    assert all(call[1] == ROWS_PER_LANE and call[4] is not None and call[5] is None for call in moe)


def test_target_c6_n2_fallback_runs_eager_without_dynamo(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)

    def ordinary_norm_quant(value, weight, eps):
        del weight, eps
        scale = torch.ones((value.shape[0], ), dtype=torch.float32)
        return value, value, scale

    monkeypatch.setattr(
        torch.ops.custom_op,
        "custom_deepseek_v41_ffn_norm_quant_gaudi2",
        ordinary_norm_quant,
        raising=False,
    )
    capacity = 2
    group = _make_group(capacity, calls)
    for layer in group.layers:
        layer.moe.shared_gate_up_channel = None
    inputs = _inputs(capacity, bases=(10, 20), valid_lengths=(6, 2))

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("C6 slice groups must not enter Dynamo")

    monkeypatch.setattr(target_module.torch, "compile", unexpected_compile)
    output = _run(group, inputs)[0]

    assert output.shape == (capacity * ROWS_PER_LANE, 4)
    moe = _count(calls, "moe")
    assert len(moe) == 4 * capacity
    assert all(call[1] == ROWS_PER_LANE and call[4] is not None and call[5] is None for call in moe)


def test_target_c6_output_finish_clamps_positions_and_masks_rows(monkeypatch):
    calls = []

    def finish(output, positions, ready_outputs=(), *, request_batch=False, decode=False):
        calls.append((positions.clone(), ready_outputs, request_batch, decode))
        return output + positions[:, None].float()

    attention = SimpleNamespace(_finish_output=finish)
    output = torch.arange(12, dtype=torch.float32).reshape(ROWS_PER_LANE, 2)
    positions = torch.tensor([-1, -1, 3, 4, 5, 6], dtype=torch.int32)
    active = torch.tensor([False, False, True, True, False, True])
    ready_outputs = (torch.arange(ROWS_PER_LANE), torch.arange(ROWS_PER_LANE, 2 * ROWS_PER_LANE))

    result = PagedCSA2Attention.finish_output_target(attention, output, positions, active, ready_outputs)

    assert len(calls) == 1
    observed_positions, observed_ready, request_batch, decode = calls[0]
    assert torch.equal(observed_positions, torch.tensor([0, 0, 3, 4, 5, 6]))
    assert observed_ready is ready_outputs
    assert request_batch is True
    assert decode is True
    expected = torch.where(
        active[:, None],
        output + torch.tensor([0, 0, 3, 4, 5, 6], dtype=torch.float32)[:, None],
        torch.zeros_like(output),
    )
    assert torch.equal(result, expected)


def test_target_c6_paged_attention_passes_runtime_tile_plan_to_mla(monkeypatch):
    capacity = 4
    rows = capacity * ROWS_PER_LANE
    tile_indices = torch.tensor([1], dtype=torch.int32)
    observed = []

    class Harness:
        ratio = 0
        fused_norm = False
        owns_index = False
        owns_kv = False
        batch_c1_numerics = False
        q_scale_rope = False
        native_rope = False
        batch_packed_mla = False
        batch_packed_mla_sram = False
        batch_packed_mla_vector = False
        batch_reindex_group = False
        eps = 1e-6
        scale = 0.125
        window = 128

        def __init__(self):
            self.batch_state = SimpleNamespace(swa=torch.zeros((capacity * 256, 528), dtype=torch.uint8))
            self.batch_main_unused = torch.zeros((1, 288), dtype=torch.uint8)
            self.weights = SimpleNamespace(
                attn_sink=torch.zeros(512),
                q_norm=SimpleNamespace(weight=torch.ones(512)),
                kv_norm=SimpleNamespace(weight=torch.ones(512)),
            )

        def _project_qkv_input(self, value):
            return value, value

        def project_query(self, value, _positions, *, request_batch):
            assert request_batch
            return value

        def _rope(self, value, _positions, *, request_batch, inverse=False):
            del inverse
            assert request_batch
            return value

    def fake_mla(query, _swa, _main, _selected, _pages, _positions, _slots, _sink, _scale,
                 _swa_done, _main_done, **kwargs):
        observed.append((query.dtype, kwargs.get("tile_indices")))
        return torch.zeros((query.shape[0], 1, 512), dtype=torch.bfloat16)

    monkeypatch.setattr(batch_attention, "batch_packed_mla", fake_mla, raising=False)
    monkeypatch.setattr(batch_attention, "write_state_rows",
                        lambda _cache, _values, rows: torch.zeros(rows.shape, dtype=torch.int32))
    monkeypatch.setattr(paged_module, "rms_norm", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(paged_module, "pack_swa", lambda value: value)

    harness = Harness()
    positions = torch.arange(rows, dtype=torch.int32)
    slots = torch.repeat_interleave(torch.arange(capacity, dtype=torch.int32), ROWS_PER_LANE)
    row_valid = torch.ones((rows, ), dtype=torch.bool)
    value = torch.randn((rows, 512), dtype=torch.bfloat16)
    pages = torch.ones((rows, 4), dtype=torch.int32)
    selected = torch.full((rows, 512), -1, dtype=torch.int32)
    candidates = torch.full((rows, 2048), -1, dtype=torch.int32)
    done = torch.zeros((rows, ), dtype=torch.int32)

    result = PagedCSA2Attention.forward_batch_target(
        harness,
        value,
        positions,
        slots,
        pages,
        selected,
        candidates,
        done,
        done,
        row_valid=row_valid,
        request_lengths=(ROWS_PER_LANE, ) * capacity,
        request_slots=tuple(range(capacity)),
        defer_output_finish=True,
        tile_indices=tile_indices,
    )

    assert len(observed) == 1
    assert observed[0][0] is torch.bfloat16
    assert observed[0][1] is tile_indices
    assert result[0].shape == (rows, 1, 512)


def test_target_c6_stage_compiles_one_fullgraph_attention_wrapper_per_layer(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    _install_target_helper_counters(monkeypatch, calls)
    capacity = 2
    stage = _make_stage(capacity, calls, simulate_publish_layers=(0, 1))
    stage.batch_state = SimpleNamespace(capacity=capacity)
    inputs = _inputs(capacity, bases=(10, 20), valid_lengths=(6, 2))
    slots = tuple(range(capacity))
    compile_calls = []

    class _CompiledExecutor:
        def __init__(self, call):
            self.call = call
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append(args[0].shape[0])
            return self.call(*args, **kwargs)

    def recording_compile(call, **kwargs):
        compile_calls.append((call, kwargs))
        return _CompiledExecutor(call)

    monkeypatch.setattr(target_module.torch, "compile", recording_compile)

    target_stage = DSparkTargetStage(stage, (ROWS_PER_LANE, ) * capacity, slots, backend="eager")
    compiled = target_stage.compiled
    assert compiled is not None
    assert compiled.slice_target_layers
    group = compiled.groups[0]
    assert len(group.attention_calls) == len(group.output_finish_calls) == len(stage.layers)
    assert len(compile_calls) == 2 * len(stage.layers)
    attention_compile_calls = compile_calls[:len(group.attention_calls)]
    finish_compile_calls = compile_calls[len(group.attention_calls):]
    assert all(call is expected
               for (call, _), expected in zip(attention_compile_calls, group.attention_calls, strict=True))
    assert all(call is expected
               for (call, _), expected in zip(finish_compile_calls, group.output_finish_calls, strict=True))
    assert all(chunk is expected for chunk, expected in zip(compiled.chunks, compiled.groups, strict=True))
    assert all(kwargs == {"backend": "eager", "fullgraph": True, "dynamic": False}
               for _, kwargs in compile_calls)
    assert all(call.request_lengths == (ROWS_PER_LANE, ) * capacity for call in group.attention_calls)
    assert all(call.request_slots == slots for call in group.attention_calls)
    assert all(isinstance(wrapper, _CompiledExecutor) for wrapper in group.attention_wrappers)
    assert all(isinstance(wrapper, _CompiledExecutor) for wrapper in group.output_finish_wrappers)
    assert all(not isinstance(call, nn.Module) for call in group.attention_calls)
    assert all(not isinstance(call, nn.Module) for call in group.output_finish_calls)
    assert all("attention_wrapper" not in key and "attention_call" not in key for key in group.state_dict())
    assert all("output_finish_wrapper" not in key and "output_finish_call" not in key for key in group.state_dict())

    output, aux = compiled(
        inputs["residual"],
        inputs["pre_mix"],
        inputs["input_ids"],
        inputs["positions"],
        inputs["slots"],
        inputs["pages"],
        inputs["row_valid"],
        inputs["engram_rows"],
    )

    rows = capacity * ROWS_PER_LANE
    assert output.shape == (rows, 4)
    assert aux.shape == (rows, 4)
    active = inputs["row_valid"]
    assert torch.count_nonzero(output[~active]).item() == 0
    assert torch.count_nonzero(aux[~active]).item() == 0

    attention = _count(calls, "attention")
    assert len(attention) == 4
    assert all(row_count == rows for _, row_count, _ in attention)
    finish = _count(calls, "finish")
    assert len(finish) == 4 * capacity
    assert all(row_count == ROWS_PER_LANE for _, row_count, _ in finish)
    assert all(len(wrapper.calls) == 1 for wrapper in group.attention_wrappers)
    assert all(len(wrapper.calls) == capacity for wrapper in group.output_finish_wrappers)
    assert all(call == rows for wrapper in group.attention_wrappers for call in wrapper.calls)
    assert all(call == ROWS_PER_LANE for wrapper in group.output_finish_wrappers for call in wrapper.calls)
    assert len(_count(calls, "hc_pre")) == 4 * capacity * 2
    assert len(_count(calls, "moe")) == 4 * capacity
    assert all(call[1] == ROWS_PER_LANE for call in _count(calls, "moe"))
    assert len(_count(calls, "target_prefix")) == 4 * capacity
    assert len(_count(calls, "target_post")) == 4 * capacity


def test_target_c6_finish_wrappers_compile_once_across_active_patterns(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    _install_deterministic_math(monkeypatch, None)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    capacity = 4
    stage = _make_stage(capacity, None, simulate_publish_layers=(0, 1))
    group = PreparedTargetBatchLayerGroup(
        stage,
        0,
        4,
        request_lengths=(ROWS_PER_LANE, ) * capacity,
        request_slots=tuple(range(capacity)),
        text_input=False,
    )
    compiled_graphs = []
    real_torch_compile = torch.compile

    def recording_backend(graph_module, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    def real_compile(call, **kwargs):
        assert kwargs == {"backend": "eager", "fullgraph": True, "dynamic": False}
        return real_torch_compile(call, backend=recording_backend, fullgraph=True, dynamic=False)

    monkeypatch.setattr(target_module.torch, "compile", real_compile)
    torch._dynamo.reset()
    try:
        group.compile_attention_wrappers("eager")
        inputs = _inputs(capacity, bases=(10, 20, 30, 40), valid_lengths=(6, 6, 6, 6))

        _run(group, inputs, active_lanes=(0, 3))
        first_compile_count = len(compiled_graphs)
        assert 0 < first_compile_count <= 2 * len(group.layers)

        _run(group, inputs, active_lanes=(1, 2))
        _run(group, inputs, active_lanes=(0, 3))
        assert len(compiled_graphs) == first_compile_count
    finally:
        torch._dynamo.reset()


def test_target_c6_tile_plan_bucket_reuses_dynamo_graph(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    _install_deterministic_math(monkeypatch, None)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    capacity = 32
    stage = _make_stage(capacity, None, simulate_publish_layers=(0, 1))
    group = PreparedTargetBatchLayerGroup(
        stage,
        0,
        4,
        request_lengths=(ROWS_PER_LANE, ) * capacity,
        request_slots=tuple(range(capacity)),
        text_input=False,
    )
    compiled_graphs = []
    real_torch_compile = torch.compile

    def recording_backend(graph_module, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    def real_compile(call, **kwargs):
        assert kwargs == {"backend": "eager", "fullgraph": True, "dynamic": False}
        return real_torch_compile(call, backend=recording_backend, fullgraph=True, dynamic=False)

    monkeypatch.setattr(target_module.torch, "compile", real_compile)
    torch._dynamo.reset()
    try:
        group.compile_attention_wrappers("eager")
        bases = tuple(10 + lane for lane in range(capacity))

        def inputs_for(active_lanes):
            valid_lengths = [0] * capacity
            for lane in active_lanes:
                valid_lengths[lane] = 6
            return _inputs(capacity, bases=bases, valid_lengths=tuple(valid_lengths))

        # Lanes 1, 6, and the pair (0, 31) all produce the same two-tile plan
        # shape, with different runtime tile ids.
        _run(group, inputs_for((1, )), active_lanes=(1, ))
        first_compile_count = len(compiled_graphs)
        assert 0 < first_compile_count <= 2 * len(group.layers)

        _run(group, inputs_for((6, )), active_lanes=(6, ))
        _run(group, inputs_for((0, 31)), active_lanes=(0, 31))
        assert len(compiled_graphs) == first_compile_count

        # A one-tile bucket is a new finite shape and may compile a new graph.
        _run(group, inputs_for((0, )), active_lanes=(0, ))
        assert len(compiled_graphs) > first_compile_count
    finally:
        torch._dynamo.reset()


def test_target_c6_stage_sets_dynamo_configs_during_first_attention_wrapper_call(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    capacity = 2
    stage = _make_stage(capacity, calls)
    stage.batch_state = SimpleNamespace(capacity=capacity)
    inputs = _inputs(capacity, bases=(10, 20), valid_lengths=(6, 2))
    observed_configs = []

    class _CompiledAttentionExecutor:
        def __init__(self, call):
            self.call = call
            self.observed = False

        def __call__(self, *args, **kwargs):
            # torch.compile performs its lazy first compilation here rather
            # than when the wrapper is constructed in CompiledTargetBatchStage.
            if not self.observed:
                observed_configs.append((
                    torch._dynamo.config.recompile_limit,
                    torch._dynamo.config.allow_unspec_int_on_nn_module,
                ))
                self.observed = True
            return self.call(*args, **kwargs)

    monkeypatch.setattr(
        target_module.torch,
        "compile",
        lambda call, **_kwargs: _CompiledAttentionExecutor(call),
    )
    compiled = target_module.CompiledTargetBatchStage(
        stage,
        (ROWS_PER_LANE, ) * capacity,
        tuple(range(capacity)),
        backend="eager",
    )
    assert compiled.slice_target_layers

    original_limit = torch._dynamo.config.recompile_limit
    original_allow_unspec_int = torch._dynamo.config.allow_unspec_int_on_nn_module
    expected_limit = max(
        2,
        (len(target_module._TARGET_MLA_TILE_PLAN_BUCKETS) + 1) * (2 * len(stage.layers) + 1),
    )
    with torch._dynamo.config.patch(
            recompile_limit=2,
            allow_unspec_int_on_nn_module=False,
    ):
        compiled(
            inputs["residual"],
            inputs["pre_mix"],
            inputs["input_ids"],
            inputs["positions"],
            inputs["slots"],
            inputs["pages"],
            inputs["row_valid"],
            inputs["engram_rows"],
        )
        assert torch._dynamo.config.recompile_limit == 2
        assert torch._dynamo.config.allow_unspec_int_on_nn_module is False

    assert torch._dynamo.config.recompile_limit == original_limit
    assert torch._dynamo.config.allow_unspec_int_on_nn_module is original_allow_unspec_int
    assert len(observed_configs) == 2 * len(stage.layers)
    assert all(limit >= expected_limit for limit, _allow in observed_configs)
    assert observed_configs == [(expected_limit, True)] * (2 * len(stage.layers))


def test_target_c6_stage_restores_dynamo_configs_when_attention_wrapper_raises(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    capacity = 1
    stage = _make_stage(capacity, calls)
    stage.batch_state = SimpleNamespace(capacity=capacity)
    inputs = _inputs(capacity, bases=(10, ), valid_lengths=(6, ))
    observed_configs = []

    class _FailingAttentionExecutor:
        def __call__(self, *_args, **_kwargs):
            observed_configs.append((
                torch._dynamo.config.recompile_limit,
                torch._dynamo.config.allow_unspec_int_on_nn_module,
            ))
            raise RuntimeError("attention compile failed")

    monkeypatch.setattr(
        target_module.torch,
        "compile",
        lambda _call, **_kwargs: _FailingAttentionExecutor(),
    )
    compiled = target_module.CompiledTargetBatchStage(
        stage,
        (ROWS_PER_LANE, ) * capacity,
        tuple(range(capacity)),
        backend="eager",
    )

    original_limit = torch._dynamo.config.recompile_limit
    original_allow_unspec_int = torch._dynamo.config.allow_unspec_int_on_nn_module
    with torch._dynamo.config.patch(
            recompile_limit=2,
            allow_unspec_int_on_nn_module=False,
    ):
        with pytest.raises(RuntimeError, match="attention compile failed"):
            compiled(
                inputs["residual"],
                inputs["pre_mix"],
                inputs["input_ids"],
                inputs["positions"],
                inputs["slots"],
                inputs["pages"],
                inputs["row_valid"],
                inputs["engram_rows"],
            )
        assert torch._dynamo.config.recompile_limit == 2
        assert torch._dynamo.config.allow_unspec_int_on_nn_module is False

    assert torch._dynamo.config.recompile_limit == original_limit
    assert torch._dynamo.config.allow_unspec_int_on_nn_module is original_allow_unspec_int
    expected_limit = (len(target_module._TARGET_MLA_TILE_PLAN_BUCKETS) + 1) * (2 * len(stage.layers) + 1)
    assert observed_configs == [(expected_limit, True)]


def test_target_c6_serial_forward_bypasses_target_attention_wrapper(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    calls = []
    _install_deterministic_math(monkeypatch, calls)
    group = _make_group(1, calls)
    layer = group.layers[0]
    inputs = _inputs(1, bases=(10, ), valid_lengths=(2, ))

    def unexpected_target_attention(*args, **kwargs):
        raise AssertionError("serial C6 must not enter the packed target attention wrapper")

    group.attention_wrappers = (unexpected_target_attention, )
    result = layer.forward(
        inputs["residual"],
        inputs["pre_mix"],
        inputs["positions"],
        torch.zeros(ROWS_PER_LANE, dtype=torch.bool),
        None,
        decode=True,
    )

    assert len(result) == 3
    assert _count(calls, "serial_attention") == [("serial_attention", ROWS_PER_LANE, True)]
    assert _count(calls, "attention") == []


def test_target_c6_non_slice_stage_keeps_fullgraph_compile_contract(monkeypatch):
    _disable_optional_target_branches(monkeypatch)
    capacity = 2
    stage = _make_stage(capacity, None)
    for layer in stage.layers:
        layer._target_c6_slices = False
    compile_calls = []

    def recording_compile(group, **kwargs):
        compile_calls.append((group, kwargs))
        return object()

    monkeypatch.setattr(target_module.torch, "compile", recording_compile)

    compiled = target_module.CompiledTargetBatchStage(
        stage,
        (ROWS_PER_LANE, ) * capacity,
        tuple(range(capacity)),
        backend="eager",
    )

    assert not compiled.slice_target_layers
    assert all(group.attention_calls == () and group.attention_wrappers == ()
               and group.output_finish_calls == () and group.output_finish_wrappers == ()
               for group in compiled.groups)
    assert len(compile_calls) == len(compiled.groups)
    assert all(group is expected for (group, _), expected in zip(compile_calls, compiled.groups, strict=True))
    assert all(kwargs == {"backend": "eager", "fullgraph": True, "dynamic": False}
               for _, kwargs in compile_calls)


@pytest.mark.parametrize(
    ("branch", "message"),
    (
        ("gate_packet", "mHC gate packet"),
        ("deferred", "deferred mHC"),
        ("peer", "DSpark peer/post/collapse"),
    ),
)
def test_target_c6_slices_fail_closed_for_optional_branches(monkeypatch, branch, message):
    _disable_optional_target_branches(monkeypatch)
    _install_deterministic_math(monkeypatch, None)
    group = _make_group(1, None)
    if branch == "gate_packet":
        monkeypatch.setattr(
            program_module.gaudi_envs,
            "VLLM_HPU_DSV41_DSPARK_MHC_GATE_PACKET",
            True,
        )
    elif branch == "deferred":
        group.layers[0].dspark_mhc_deferred = True
    else:
        group.layers[0].dspark_peer_post_collapse = True
    with pytest.raises(RuntimeError, match=message):
        _run(group, _inputs(1, bases=(10,), valid_lengths=(6,)))


@pytest.mark.parametrize(
    ("flag", "message"),
    (
        ("VLLM_HPU_DSV41_DSPARK_WO_HANDOFF", "DSpark WO handoff"),
        ("VLLM_HPU_DSV41_DSPARK_Q_PROLOGUE", "DSpark query prologue"),
        ("VLLM_HPU_DSV41_DSPARK_C1_DENSE_CHAIN", "DSpark C1 dense chain"),
        ("VLLM_HPU_DSV41_DSPARK_FP8_QKV_PROLOGUE", "DSpark FP8 QKV prologue"),
        ("VLLM_HPU_DSV41_DSPARK_FP8_QKV_PROLOGUE_SPLIT", "split DSpark FP8 QKV prologue"),
        ("VLLM_HPU_DSV41_DSPARK_INPUT_FP8", "DSpark input FP8"),
        ("VLLM_HPU_DSV41_DSPARK_OUTPUT_FP8", "DSpark output FP8"),
    ),
)
def test_target_c6_slices_fail_closed_for_unsupported_target_chains(monkeypatch, flag, message):
    _disable_optional_target_branches(monkeypatch)
    _install_deterministic_math(monkeypatch, None)
    group = _make_group(1, None)
    monkeypatch.setattr(program_module.gaudi_envs, flag, True)
    with pytest.raises(RuntimeError, match=message):
        _run(group, _inputs(1, bases=(10, ), valid_lengths=(6, )))
