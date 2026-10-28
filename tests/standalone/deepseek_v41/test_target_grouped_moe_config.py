# SPDX-License-Identifier: Apache-2.0
"""Fail-closed CPU contracts for the grouped DSpark target MoE option."""

from types import SimpleNamespace

import pytest

from vllm_gaudi.models import deepseek_v41_batch_target_program as target_module
from vllm_gaudi.models.deepseek_v41_batch_target_program import DSparkTargetStage


def _program(*, normal_scales=True, channel13=None):
    experts = SimpleNamespace(
        w13_q16=object(),
        w2_q16=object(),
        w13_s16=object(),
        w2_s16=object(),
        w13_fp8_channel=channel13,
    )
    moe = SimpleNamespace(
        n256_fp8=True,
        n256_fused=True,
        router_top6=True,
        topk=6,
        lookup=object(),
        normal_scales=normal_scales,
        weights=SimpleNamespace(experts=experts),
    )
    return SimpleNamespace(layers=[SimpleNamespace(layer=7, moe=moe)])


def _install_stage_doubles(monkeypatch):
    state_calls = []
    compiled_calls = []

    def initialize(program, capacity):
        state_calls.append((program, capacity))
        return SimpleNamespace(capacity=capacity)

    def compile_stage(*args, **kwargs):
        compiled_calls.append((args, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(target_module, "initialize_target_batch_state", initialize)
    monkeypatch.setattr(target_module, "CompiledTargetBatchStage", compile_stage)
    return state_calls, compiled_calls


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("n256_fp8", "moe.n256_fp8=true"),
        ("n256_fused", "moe.n256_fused=true"),
        ("router_top6", "moe.router_top6=true"),
        ("topk", "moe.topk=6"),
        ("w13_q16", "moe.weights.experts.w13_q16"),
        ("w2_q16", "moe.weights.experts.w2_q16"),
        ("w13_s16", "moe.weights.experts.w13_s16"),
        ("w2_s16", "moe.weights.experts.w2_s16"),
        ("w13_q16_none", "moe.weights.experts.w13_q16"),
        ("w2_q16_none", "moe.weights.experts.w2_q16"),
        ("w13_s16_none", "moe.weights.experts.w13_s16"),
        ("w2_s16_none", "moe.weights.experts.w2_s16"),
        ("lookup", "moe.lookup"),
        ("lookup_none", "moe.lookup"),
        ("experts", "moe.weights.experts"),
        ("normal_scales_missing", "moe.normal_scales"),
        ("channel_missing", "moe.weights.experts.w13_fp8_channel"),
        ("moe", "layer 7: moe is missing"),
        ("layers", "program.layers"),
        ("normal_scales_with_channel", "moe.normal_scales=true for experts.w13_fp8_channel"),
        ("column_interleave_with_channel", "VLLM_HPU_DSV41_PREFILL_COLUMN_INTERLEAVE=true"),
    ],
)
def test_each_missing_grouped_moe_capability_fails_before_state_init(monkeypatch, missing, expected):
    program = _program()
    layer = program.layers[0]
    moe = layer.moe
    experts = moe.weights.experts
    if missing in {"n256_fp8", "n256_fused", "router_top6"}:
        setattr(moe, missing, False)
    elif missing == "topk":
        moe.topk = 3
    elif missing in {"w13_q16", "w2_q16", "w13_s16", "w2_s16"}:
        delattr(experts, missing)
    elif missing in {"w13_q16_none", "w2_q16_none", "w13_s16_none", "w2_s16_none"}:
        setattr(experts, missing.removesuffix("_none"), None)
    elif missing == "lookup":
        delattr(moe, "lookup")
    elif missing == "lookup_none":
        moe.lookup = None
    elif missing == "experts":
        delattr(moe.weights, "experts")
    elif missing == "normal_scales_missing":
        delattr(moe, "normal_scales")
    elif missing == "channel_missing":
        delattr(experts, "w13_fp8_channel")
    elif missing == "moe":
        delattr(layer, "moe")
    elif missing == "layers":
        program.layers = []
    elif missing == "normal_scales_with_channel":
        monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_COLUMN_INTERLEAVE", "1")
        moe.normal_scales = False
        experts.w13_fp8_channel = object()
    elif missing == "column_interleave_with_channel":
        monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_COLUMN_INTERLEAVE", "0")
        experts.w13_fp8_channel = object()

    state_calls, compiled_calls = _install_stage_doubles(monkeypatch)
    with pytest.raises(ValueError) as error:
        DSparkTargetStage(program, (6, ), (0, ), backend="eager", packed_lane_math=True,
                          packed_grouped_moe=True)

    message = str(error.value)
    assert expected in message
    if missing != "layers":
        assert "layer 7" in message
    assert state_calls == []
    assert compiled_calls == []


@pytest.mark.parametrize(
    ("request_lengths", "request_slots", "stage_kwargs"),
    [
        ((6, 6), (0, 1), {}),
        ((6, ), (0, ), {"state_capacity": 2}),
        ((6, ), (0, ), {"state_capacity": 2, "ragged_compressor": True}),
    ],
    ids=["canonical", "execution", "query"],
)
def test_all_target_stage_modes_validate_before_state_init(monkeypatch, request_lengths, request_slots, stage_kwargs):
    program = _program()
    program.layers[0].moe.router_top6 = False
    state_calls, compiled_calls = _install_stage_doubles(monkeypatch)

    with pytest.raises(ValueError) as error:
        DSparkTargetStage(program, request_lengths, request_slots, backend="eager", packed_lane_math=True,
                          packed_grouped_moe=True, **stage_kwargs)

    message = str(error.value)
    assert "layer 7" in message
    assert "moe.router_top6=true" in message
    assert state_calls == []
    assert compiled_calls == []


@pytest.mark.parametrize(
    ("normal_scales", "channel13", "column_interleave", "grouped"),
    [
        (False, None, "0", True),
        (None, None, "0", True),
        (True, None, "0", True),
        (True, object(), "1", True),
        (False, None, "0", False),
    ],
    ids=["nonchannel", "nonchannel-normal-scales-none", "nonchannel-normal-scales", "channel", "default-off"],
)
def test_supported_grouped_paths_and_default_off_reach_state_init(monkeypatch, normal_scales, channel13,
                                                                  column_interleave, grouped):
    program = _program(normal_scales=normal_scales, channel13=channel13)
    if not grouped:
        del program.layers
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_COLUMN_INTERLEAVE", column_interleave)
    state_calls, compiled_calls = _install_stage_doubles(monkeypatch)

    DSparkTargetStage(program, (6, ), (0, ), backend="eager", packed_lane_math=grouped,
                      packed_grouped_moe=grouped)

    assert len(state_calls) == 1
    assert len(compiled_calls) == 1
    kwargs = compiled_calls[0][1]
    assert kwargs.get("packed_grouped_moe", False) is grouped
