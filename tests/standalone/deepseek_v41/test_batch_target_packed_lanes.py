# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the opt-in packed target lane math (BUG-004).

The packed path must compute the same row-independent arithmetic as the
per-lane C6 path, once over all fixed rows, with a launch/collective count
that does not depend on the number of active lanes.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.models import deepseek_v41_batch_target_program as target_module
from vllm_gaudi.models import deepseek_v41_program as program_module
from vllm_gaudi.models.deepseek_v41_batch_target_program import (
    CompiledTargetBatchStage,
    DSparkTargetStage,
    PreparedTargetBatchLayerGroup,
)
from vllm_gaudi.v1.worker import deepseek_v41_dspark_batch_runner as runner_module

from tests.standalone.deepseek_v41 import test_batch_target_c6_slices as c6
from tests.standalone.deepseek_v41.test_batch_target_c6_slices import (  # noqa: F401
    restore_lazy_environment_attributes,
)

ROWS = c6.ROWS_PER_LANE


class _PackedCapableAttention(c6._PackedAttention):
    """Fake attention whose packed finish equals the six-row lane finish."""

    def project_output_target_rows(self, output, positions):
        if self.calls is not None:
            self.calls.append(("project_rows", output.shape[0]))
        return output + 1

    def reduce_output_target_rows(self, partial, active):
        if self.calls is not None:
            self.calls.append(("reduce_rows", partial.shape[0]))
        return torch.where(active[:, None], partial, torch.zeros_like(partial))


def _make_group(capacity, calls, *, packed):
    stage = c6._make_stage(capacity, calls)
    for layer in stage.layers:
        layer.attention = _PackedCapableAttention(capacity, calls)
    return PreparedTargetBatchLayerGroup(
        stage,
        0,
        4,
        request_lengths=(ROWS, ) * capacity,
        request_slots=tuple(range(capacity)),
        text_input=False,
        packed_lane_math=packed,
    )


def _setup(monkeypatch, calls=None):
    c6._disable_optional_target_branches(monkeypatch)
    c6._install_deterministic_math(monkeypatch, calls)


@pytest.mark.parametrize(
    ("capacity", "bases", "lengths", "active_lanes"),
    [
        (3, (10, 20, 30), (6, 3, 1), None),
        (4, (10, 20, 30, 40), (0, 6, 0, 3), (1, 3)),
        (8, tuple(range(5, 85, 10)), (1, 0, 6, 2, 0, 0, 5, 1), (0, 2, 3, 6, 7)),
    ],
)
def test_packed_lane_math_matches_lane_path(monkeypatch, capacity, bases, lengths, active_lanes):
    _setup(monkeypatch)
    inputs = c6._inputs(capacity, bases=bases, valid_lengths=lengths)
    lane = c6._run(_make_group(capacity, None, packed=False), inputs, active_lanes)
    packed = c6._run(_make_group(capacity, None, packed=True), inputs, active_lanes)
    for name, expected, actual in zip(("value", "pre_mix", "aux"), lane[:3], packed[:3], strict=True):
        assert torch.equal(expected, actual), name
    assert torch.count_nonzero(packed[0][~inputs["row_valid"]]).item() == 0
    assert torch.count_nonzero(packed[2][~inputs["row_valid"]]).item() == 0


def test_packed_lane_math_work_does_not_depend_on_active_lanes(monkeypatch):
    calls = []
    _setup(monkeypatch, calls)
    capacity = 4
    group = _make_group(capacity, calls, packed=True)

    c6._run(group, c6._inputs(capacity, bases=(1, 2, 3, 4), valid_lengths=(6, 6, 6, 6)))
    full = list(calls)
    calls.clear()
    c6._run(group, c6._inputs(capacity, bases=(1, 2, 3, 4), valid_lengths=(0, 2, 0, 0)), active_lanes=(1, ))
    assert [call[0] for call in calls] == [call[0] for call in full]

    rows = capacity * ROWS
    # One packed attention, finish, two mHC inputs, one MoE per layer.
    assert len(c6._count(full, "attention")) == 4
    assert [call[1] for call in c6._count(full, "project_rows")] == [rows] * 4
    assert [call[1] for call in c6._count(full, "reduce_rows")] == [rows] * 4
    assert [call[1] for call in c6._count(full, "hc_pre")] == [rows] * 8
    assert [call[1] for call in c6._count(full, "moe")] == [rows] * 4
    assert all(call[2] is True for call in c6._count(full, "moe"))
    assert [call[1] for call in c6._count(full, "final_collapse")] == [rows]
    # No six-row lane finish, inter-layer lane identity or per-lane prefix.
    assert not c6._count(full, "finish")
    assert not c6._count(full, "bf16_identity")
    assert not c6._count(full, "quant")


def test_packed_lane_math_requires_c6_slice_layers(monkeypatch):
    _setup(monkeypatch)
    stage = c6._make_stage(2, None)
    for layer in stage.layers:
        layer._target_c6_slices = False
    with pytest.raises(ValueError, match="fixed C6 slice contract"):
        PreparedTargetBatchLayerGroup(stage, 0, 4, request_lengths=(ROWS, ) * 2, request_slots=(0, 1),
                                      packed_lane_math=True)


def test_packed_stage_plumbs_option_and_skips_lane_finish_compile(monkeypatch):
    _setup(monkeypatch)
    compiled = []

    def recording_compile(call, **kwargs):
        compiled.append(type(call).__name__)
        return call

    monkeypatch.setattr(torch, "compile", recording_compile)
    stage = c6._make_stage(2, None, layer_count=8)
    packed = CompiledTargetBatchStage(stage, (ROWS, ) * 2, (0, 1), backend="eager", packed_lane_math=True)
    assert packed.packed_lane_math is True
    assert all(group.packed_lane_math for group in packed.groups)
    assert all(group.packed_recipes is packed.packed_recipes for group in packed.groups)
    assert compiled.count("_TargetAttentionCall") == 8
    assert "_TargetOutputFinishCall" not in compiled

    compiled.clear()
    lane = CompiledTargetBatchStage(stage, (ROWS, ) * 2, (0, 1), backend="eager")
    assert not any(group.packed_lane_math for group in lane.groups)
    assert compiled.count("_TargetOutputFinishCall") == 8
    with pytest.raises(TypeError, match="packed lane math must be a boolean"):
        DSparkTargetStage(stage, (ROWS, ) * 2, (0, 1), backend="eager", packed_lane_math=1)


def test_packed_recipes_run_eagerly_off_hpu():
    recipes = program_module.PackedTargetRecipes()
    value = torch.ones(12, 4, 4, dtype=torch.bfloat16)
    pre = torch.ones(12, 4)
    active = torch.tensor([True, False] * 6)
    residual, pre_mix = recipes.run("ffn_output", value[:, 0], value, pre, torch.zeros(12, 4, 4), pre, active)
    assert recipes.status() == {"compiled": [], "calls": 0, "disabled_reason": None, "grouped_moe": False}
    assert torch.count_nonzero(residual[~active]).item() == 0
    assert torch.count_nonzero(pre_mix[~active]).item() == 0


class _HpuRecipeValue:

    def __init__(self, value):
        self.value = value
        self.device = SimpleNamespace(type="hpu")


def _recipe_harness(eager):
    recipes = program_module.PackedTargetRecipes()
    recipes._FUNCTIONS = {"test": eager}
    return recipes


def test_packed_recipes_fall_back_if_compile_construction_fails(monkeypatch):
    eager_calls = []

    def eager(value):
        eager_calls.append(value.value)
        return value.value * 3

    compile_calls = []

    def unavailable(function, **kwargs):
        compile_calls.append((function, kwargs))
        raise RuntimeError("hpu_backend unavailable")

    monkeypatch.setattr(program_module.torch, "compile", unavailable)
    recipes = _recipe_harness(eager)
    value = _HpuRecipeValue(4)

    assert recipes.run("test", value) == 12
    assert not recipes._compiled
    assert len(compile_calls) == 1
    assert compile_calls[0] == (eager, {"backend": "hpu_backend", "fullgraph": True, "dynamic": False})
    assert recipes.status() == {
        "compiled": [],
        "calls": 0,
        "disabled_reason": "test: RuntimeError: hpu_backend unavailable",
        "grouped_moe": False,
    }

    assert recipes.run("test", value) == 12
    assert eager_calls == [4, 4]
    assert len(compile_calls) == 1


def test_packed_recipes_fall_back_if_first_compiled_call_fails(monkeypatch):
    eager_calls = []

    def eager(value):
        eager_calls.append(value.value)
        return value.value * 3

    compile_calls = []
    compiled_calls = []

    def failing_compile(function, **kwargs):
        compile_calls.append((function, kwargs))

        def fail(value):
            compiled_calls.append(value.value)
            raise RuntimeError("first graph execution failed")

        return fail

    monkeypatch.setattr(program_module.torch, "compile", failing_compile)
    recipes = _recipe_harness(eager)
    value = _HpuRecipeValue(5)

    assert recipes.run("test", value) == 15
    assert len(compile_calls) == 1
    assert compiled_calls == [5]
    assert recipes.status() == {
        "compiled": ["test"],
        "calls": 0,
        "disabled_reason": "test: RuntimeError: first graph execution failed",
        "grouped_moe": False,
    }

    assert recipes.run("test", value) == 15
    assert eager_calls == [5, 5]
    assert len(compile_calls) == 1
    assert compiled_calls == [5]


def test_packed_recipes_cache_successful_compile(monkeypatch):
    eager_calls = []

    def eager(value):
        eager_calls.append(value.value)
        return value.value * 3

    compile_calls = []
    compiled_calls = []

    def recording_compile(function, **kwargs):
        compile_calls.append((function, kwargs))

        def run(value):
            compiled_calls.append(value.value)
            return function(value)

        return run

    monkeypatch.setattr(program_module.torch, "compile", recording_compile)
    recipes = _recipe_harness(eager)

    assert recipes.run("test", _HpuRecipeValue(6)) == 18
    assert recipes.run("test", _HpuRecipeValue(7)) == 21
    assert len(compile_calls) == 1
    assert compile_calls[0] == (eager, {"backend": "hpu_backend", "fullgraph": True, "dynamic": False})
    assert compiled_calls == [6, 7]
    assert eager_calls == [6, 7]
    assert recipes.status() == {"compiled": ["test"], "calls": 2, "disabled_reason": None, "grouped_moe": False}


def test_packed_moe_falls_back_off_hpu_without_grouped():
    seen = []

    class _MoE:
        n256_fp8 = True
        n256_fused = True
        router_top6 = True
        topk = 6

        def __call__(self, value, image_mask, **kwargs):
            seen.append(kwargs)
            return value * 2

    value = torch.ones(12, 4)
    image_mask = torch.zeros(12, dtype=torch.bool)
    assert torch.equal(program_module.packed_target_moe(_MoE(), value, image_mask, grouped=False), value * 2)
    assert seen == [{"decode": True}]


class _GroupedCapabilityMoE:

    def __init__(self):
        self.n256_fp8 = True
        self.n256_fused = True
        self.router_top6 = True
        self.topk = 6
        self.calls = []

    def __call__(self, value, image_mask, **kwargs):
        self.calls.append((value, image_mask, kwargs))
        return value


@pytest.mark.parametrize("attribute", ["n256_fp8", "n256_fused", "router_top6", "topk"])
def test_grouped_packed_moe_rejects_missing_capability_without_ordinary_fallback(attribute):
    moe = _GroupedCapabilityMoE()
    delattr(moe, attribute)
    value = SimpleNamespace(device=SimpleNamespace(type="hpu"))

    with pytest.raises(RuntimeError, match="Grouped packed target MoE requires HPU"):
        program_module.packed_target_moe(moe, value, None, grouped=True)
    assert moe.calls == []


def test_grouped_packed_moe_rejects_non_hpu_without_ordinary_fallback():
    moe = _GroupedCapabilityMoE()
    value = torch.ones(12, 4)
    image_mask = torch.zeros(12, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="Grouped packed target MoE requires HPU"):
        program_module.packed_target_moe(moe, value, image_mask, grouped=True)
    assert moe.calls == []


class _HeadProgram:
    tp_rank = 0

    def __init__(self, calls):
        self.calls = calls

    def _head_projection(self, hidden):
        self.calls.append(hidden.shape[0])
        return hidden.float() * torch.arange(1, hidden.shape[-1] + 1).float()

    @staticmethod
    def all_gather(value, dim):
        return value


@pytest.mark.parametrize("active_lanes", [None, (1, ), (0, 2)])
def test_packed_target_greedy_ids_match_lane_path(active_lanes):
    torch.manual_seed(3)
    hidden = torch.randn(3 * ROWS, 16)
    lane_calls, packed_calls = [], []
    expected = runner_module._target_greedy_ids(_HeadProgram(lane_calls), hidden, active_lanes)
    actual = runner_module._target_greedy_ids(_HeadProgram(packed_calls), hidden, active_lanes, packed=True)
    assert torch.equal(expected, actual)
    assert packed_calls == [3 * ROWS]
    assert lane_calls == [ROWS] * (3 if active_lanes is None else len(active_lanes))


def test_runner_configures_packed_lane_math_before_warmup(monkeypatch):
    built = []

    class _Stage:
        def __init__(self, program, lengths, slots, *, backend, packed_lane_math=False, **kwargs):
            built.append((lengths, slots, backend, packed_lane_math))
            self.compiled = SimpleNamespace()

    monkeypatch.setattr(target_module, "DSparkTargetStage", _Stage)
    stages = runner_module.DSparkPackedTargetStages.__new__(runner_module.DSparkPackedTargetStages)
    stages.program = object()
    stages.batch_capacity = 2
    stages.stage_key = (0, 1)
    stages._backend = "eager"
    stages.packed_lane_math = False
    stages.observed_query_keys = set()
    stages.query_stages = {}
    stages.stage = SimpleNamespace(compiled=SimpleNamespace(execution_bucket_count=3, use_sparse_mla_plan=False))
    stages.configure_packed_lane_math(True)
    assert stages.packed_lane_math is True
    assert built == [((6, 6), (0, 1), "eager", True)]
    assert stages.stage.compiled.execution_bucket_count == 3
    assert stages.stage.compiled.use_sparse_mla_plan is False
    assert stages.stages == {(0, 1): stages.stage}
    with pytest.raises(TypeError):
        stages.configure_packed_lane_math("yes")
    stages.observed_query_keys.add((2, 12))
    with pytest.raises(RuntimeError, match="before warmup"):
        stages.configure_packed_lane_math(False)


def _rank_partials(ranks, lanes, width, *, replicate):
    generator = torch.Generator().manual_seed(17)
    partials = []
    for rank in range(ranks):
        lane = (torch.randn(ROWS, width, generator=generator) * (0.5 + rank)).to(torch.bfloat16)
        rows = (torch.cat((lane, ) * lanes) if replicate else
                (torch.randn(lanes * ROWS, width, generator=generator) * (0.5 + rank)).to(torch.bfloat16))
        partials.append(rows)
    return partials


@pytest.mark.parametrize("lanes", [1, 8, 32])
def test_ordered_gather_sum_matches_six_row_native_reduction(lanes):
    from vllm_gaudi.ops.deepseek_v41_ordered_peer_sum import ordered_gather_sum, ordered_peer_sum_reference

    ranks = 8
    partials = _rank_partials(ranks, lanes, 64, replicate=False)
    seen = []

    def gather(value, dim):
        seen.append((tuple(value.shape), dim))
        return torch.cat(tuple(partial.reshape(1, -1) for partial in partials), dim=0)

    actual = ordered_gather_sum(partials[0], gather, ranks)
    assert seen == [((1, lanes * ROWS * 64), 0)]
    for lane in range(lanes):
        rows = slice(lane * ROWS, lane * ROWS + ROWS)
        shards = torch.stack(tuple(partial[rows].reshape(-1) for partial in partials))
        expected = ordered_peer_sum_reference(shards).reshape(ROWS, 64)
        assert torch.equal(actual[rows], expected)


def test_ordered_gather_sum_is_lane_position_invariant():
    from vllm_gaudi.ops.deepseek_v41_ordered_peer_sum import ordered_gather_sum

    partials = _rank_partials(8, 32, 64, replicate=True)
    actual = ordered_gather_sum(
        partials[0], lambda value, dim: torch.cat(tuple(p.reshape(1, -1) for p in partials), dim=0), 8)
    lanes = actual.reshape(32, ROWS, 64)
    assert all(torch.equal(lanes[lane], lanes[0]) for lane in range(32))


def test_ordered_rows_reduce_keeps_native_reduce_for_small_or_host_rows():
    from vllm_gaudi.ops.deepseek_v41_ordered_peer_sum import ordered_rows_reduce

    value = torch.ones(8 * ROWS, 5120, dtype=torch.bfloat16)
    calls = []

    def reduce(partial):
        calls.append(partial.shape)
        return partial

    def gather(partial, dim):  # pragma: no cover - must not be used off HPU
        raise AssertionError("host rows keep the stage reduction")

    assert ordered_rows_reduce(value, reduce, gather, 8) is value
    assert ordered_rows_reduce(value[:ROWS], reduce, gather, 8) is not None
    assert calls == [value.shape, value[:ROWS].shape]


@pytest.mark.parametrize("tokens", [48, 192])
@pytest.mark.parametrize("concentrated", [False, True])
def test_static_target_occupancy_bounds_every_route(tokens, concentrated):
    from vllm_gaudi.ops.deepseek_v41_prefill_buckets import (_FP8_ROW_BUCKETS, device_bucketed_route_blocks,
                                                             static_occupied_blocks)

    generator = torch.Generator().manual_seed(tokens + concentrated)
    if concentrated:
        # Every token selects the same six experts: the largest reachable bucket.
        ids = torch.arange(6, dtype=torch.int32).repeat(tokens, 1)
    else:
        ids = torch.stack([torch.randperm(384, generator=generator)[:6] for _ in range(tokens)]).int()
    descriptors = device_bucketed_route_blocks(ids, 384, 32)
    occupied = [int(count) for count in descriptors[-1].tolist()]
    static = static_occupied_blocks(descriptors, _FP8_ROW_BUCKETS, 32, tokens)
    assert all(count <= bound for count, bound in zip(occupied, static, strict=True))
    unreachable = [rows for rows, bound in zip(_FP8_ROW_BUCKETS, static, strict=True) if not bound]
    assert unreachable == [rows for rows in _FP8_ROW_BUCKETS if rows - 31 > tokens]
    # Static blocks beyond the occupied prefix carry no expert and no route.
    for index, count in enumerate(occupied):
        experts, slots = descriptors[index * 2], descriptors[index * 2 + 1]
        assert bool((experts[0, count:] == -1).all()) and bool((slots[count:] == -1).all())


def test_grouped_moe_option_requires_packed_lane_math_and_reaches_recipes(monkeypatch):
    with pytest.raises(TypeError, match="grouped MoE"):
        program_module.PackedTargetRecipes(grouped_moe=1)
    assert program_module.PackedTargetRecipes(grouped_moe=True).status()["grouped_moe"] is True
    with pytest.raises(TypeError, match="grouped MoE"):
        DSparkTargetStage(object(), (6, ), (0, ), packed_lane_math=True, packed_grouped_moe="yes")
    with pytest.raises(ValueError, match="requires packed lane math"):
        DSparkTargetStage(object(), (6, ), (0, ), packed_grouped_moe=True)


def test_runner_configures_grouped_moe_only_with_packed_lane_math(monkeypatch):
    built = []

    class _Stage:
        def __init__(self, program, lengths, slots, *, backend, packed_lane_math=False, **kwargs):
            built.append((packed_lane_math, kwargs))
            self.compiled = SimpleNamespace()

    monkeypatch.setattr(target_module, "DSparkTargetStage", _Stage)
    stages = runner_module.DSparkPackedTargetStages.__new__(runner_module.DSparkPackedTargetStages)
    stages.program = object()
    stages.batch_capacity = 1
    stages.stage_key = (0, )
    stages._backend = "eager"
    stages.observed_query_keys = set()
    stages.query_stages = {}
    stages.stage = SimpleNamespace(compiled=SimpleNamespace())
    with pytest.raises(ValueError, match="requires dsv41_dspark_target_packed_lane_math"):
        stages.configure_packed_lane_math(False, grouped_moe=True)
    with pytest.raises(TypeError, match="grouped_moe"):
        stages.configure_packed_lane_math(True, grouped_moe=1)
    stages.configure_packed_lane_math(True, grouped_moe=True)
    assert built == [(True, {"packed_grouped_moe": True})]
    assert stages.packed_grouped_moe is True and stages.packed_lane_math is True
    stages.configure_packed_lane_math(True, grouped_moe=True)
    assert len(built) == 1
