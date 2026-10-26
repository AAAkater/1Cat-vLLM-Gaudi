# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the bounded packed-C6 scoped recipes."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_gaudi.models import deepseek_v41_c6_scoped_recipe as recipe_module
from vllm_gaudi.models import deepseek_v41_program as program_module
from vllm_gaudi.models.deepseek_v41_program import PreparedDecoderLayer


def _fake_hc_pre(residual, previous_pre, *args, **kwargs):
    collapsed = kwargs.get("collapsed_input")
    value = residual.float().mean(1).to(residual.dtype) if collapsed is None else collapsed
    new_pre = previous_pre + 1.0
    post = torch.ones_like(previous_pre)
    comb = torch.ones((residual.shape[0], 4, 4), dtype=previous_pre.dtype, device=previous_pre.device)
    return value, new_pre, post, comb


def _fake_attention_norm_quant(value, weight, eps):
    del weight, eps
    scale = torch.ones((value.shape[0], 1), dtype=torch.float32, device=value.device)
    return value, value, scale


def _fake_mhc_post(value, residual, post, comb, new_pre):
    del post, comb, new_pre
    return residual + value.unsqueeze(1), residual.mean(1)


def _fake_ffn_dual_quant(value, weight, eps):
    del weight, eps
    scale = torch.ones((value.shape[0], 1), dtype=torch.float32, device=value.device)
    return value, value, scale, value, scale


@pytest.fixture(autouse=True)
def _install_fake_math(monkeypatch):
    # These CPU contracts must not acquire an HPU owned by a serving worker.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(recipe_module, "hc_pre", _fake_hc_pre)
    monkeypatch.setattr(recipe_module, "_bf16_boundary", lambda value: value.clone())
    monkeypatch.setattr(recipe_module, "_attention_norm_quant", _fake_attention_norm_quant)
    monkeypatch.setattr(recipe_module, "decode_mhc_post_collapse", _fake_mhc_post)
    monkeypatch.setattr(recipe_module, "decode_ffn_dual_quant", _fake_ffn_dual_quant)
    monkeypatch.setattr(recipe_module, "_ffn_norm_quant", lambda *args: _fake_ffn_dual_quant(*args)[:3])


def _prefix_args(collapsed=True, *, residual_offset=0):
    storage = torch.zeros((6 * 32, 4, 4), dtype=torch.bfloat16)
    residual = storage[residual_offset:residual_offset + 6]
    previous_pre = torch.zeros((6, 4), dtype=torch.float32)
    collapsed_value = (torch.ones((6, 4), dtype=torch.bfloat16)
                       if collapsed else None)
    hc_fn = torch.zeros((24, 16), dtype=torch.float32)
    hc_scale = torch.zeros(3, dtype=torch.float32)
    hc_base = torch.zeros(24, dtype=torch.float32)
    hc_packed = torch.zeros((24, 16), dtype=torch.float32)
    norm = torch.ones(4, dtype=torch.bfloat16)
    return residual, previous_pre, collapsed_value, hc_fn, hc_scale, hc_base, hc_packed, norm


def _post_args(*, offset=0):
    storage = torch.zeros((6 * 32, 4, 4), dtype=torch.bfloat16)
    attention = torch.ones((6, 4), dtype=torch.bfloat16)
    residual = storage[offset:offset + 6]
    new_pre = torch.ones((6, 4), dtype=torch.float32)
    post = torch.ones((6, 4), dtype=torch.float32)
    comb = torch.ones((6, 4, 4), dtype=torch.float32)
    hc_fn = torch.zeros((24, 16), dtype=torch.float32)
    hc_scale = torch.zeros(3, dtype=torch.float32)
    hc_base = torch.zeros(24, dtype=torch.float32)
    hc_packed = torch.zeros((24, 16), dtype=torch.float32)
    norm = torch.ones(4, dtype=torch.bfloat16)
    return attention, residual, new_pre, post, comb, hc_fn, hc_scale, hc_base, hc_packed, norm


def _assert_tensor_tuple_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if expected_value is None:
            assert actual_value is None
        else:
            assert torch.equal(actual_value, expected_value)


def test_prefix_wrappers_preserve_collapsed_and_none_branches():
    no_collapse_args = _prefix_args(collapsed=False)
    with_collapse_args = _prefix_args(collapsed=True, residual_offset=6 * 31)

    no_collapse = recipe_module.c6_prefix_no_collapse(
        no_collapse_args[0], no_collapse_args[1], *no_collapse_args[3:], 1e-6, 1e-6, 20
    )
    with_collapse = recipe_module.c6_prefix_with_collapse(
        with_collapse_args[0],
        with_collapse_args[1],
        with_collapse_args[2],
        *with_collapse_args[3:],
        1e-6,
        1e-6,
        20,
    )

    expected_no_collapse = recipe_module._c6_prefix_math(
        no_collapse_args[0], no_collapse_args[1], None, *no_collapse_args[3:], 1e-6, 1e-6, 20
    )
    expected_with_collapse = recipe_module._c6_prefix_math(
        with_collapse_args[0],
        with_collapse_args[1],
        with_collapse_args[2],
        *with_collapse_args[3:],
        1e-6,
        1e-6,
        20,
    )
    _assert_tensor_tuple_equal(no_collapse, expected_no_collapse)
    _assert_tensor_tuple_equal(with_collapse, expected_with_collapse)
    assert not torch.equal(no_collapse[0], with_collapse[0])


def test_post_pre_moe_math_preserves_operator_order_and_outputs():
    args = _post_args(offset=6 * 31)
    actual = recipe_module._c6_post_pre_moe_math(*args, 1e-6, 1e-6, 20)

    attention, residual, new_pre, post, comb, hc_fn, hc_scale, hc_base, hc_packed, norm = args
    expected_residual, collapsed = _fake_mhc_post(attention, residual, post, comb, new_pre)
    expected_value, expected_pre, expected_post, expected_comb = _fake_hc_pre(
        expected_residual,
        new_pre,
        hc_fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        20,
        packed_fn=hc_packed,
        decode=True,
        collapsed_input=collapsed,
        batch_control_reuse=False,
        control_mme_weight=None,
        control_epilogue=False,
        control_fp8_weight=None,
    )
    expected_normalized, expected_quantized, expected_scale, expected_shared_q, expected_shared_scale = (
        _fake_ffn_dual_quant(expected_value, norm, 1e-6)
    )
    _assert_tensor_tuple_equal(
        actual,
        (
            expected_residual,
            expected_normalized,
            expected_quantized,
            expected_scale,
            expected_shared_q,
            expected_shared_scale,
            expected_pre,
            expected_post,
            expected_comb,
        ),
    )


class _RecordingCompiled:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def test_cache_uses_three_variants_across_layers_lanes_and_weight_rebinding(monkeypatch):
    compile_calls = []

    def recording_compile(function, **kwargs):
        compile_calls.append((function, kwargs))
        return _RecordingCompiled(function)

    monkeypatch.setattr(recipe_module.torch, "compile", recording_compile)
    owner = SimpleNamespace(generation=3, precision_fingerprint="precision")
    cache = recipe_module.C6ScopedRecipeCache(owner, backend="eager")

    for offset in (0, 6 * 31, 0, 6 * 31):
        prefix = _prefix_args(collapsed=False, residual_offset=offset)
        cache.run_prefix(*prefix, 1e-6, 1e-6, 20)
        prefix = _prefix_args(collapsed=True, residual_offset=offset)
        cache.run_prefix(*prefix, 1e-6, 1e-6, 20)
        post = _post_args(offset=offset)
        cache.run_post_pre_moe(*post, 1e-6, 1e-6, 20)

    assert len(compile_calls) == 3
    assert all(kwargs == {"backend": "eager", "fullgraph": True, "dynamic": False}
               for _, kwargs in compile_calls)
    status = cache.status()
    assert status["state"] == "active"
    assert status["compiled_variants"] == ("post_pre_moe", "prefix_no_collapse", "prefix_with_collapse")
    assert status["hits"] == 9
    assert status["misses"] == 3
    assert status["ineligible_calls"] == 0


def test_cache_invalidates_on_generation_and_weight_layout(monkeypatch):
    compile_calls = []

    def recording_compile(function, **kwargs):
        compile_calls.append((function, kwargs))
        return _RecordingCompiled(function)

    monkeypatch.setattr(recipe_module.torch, "compile", recording_compile)
    owner = SimpleNamespace(generation=1, precision_fingerprint="precision")
    cache = recipe_module.C6ScopedRecipeCache(owner, backend="eager")
    prefix = _prefix_args(collapsed=False)
    cache.run_prefix(*prefix, 1e-6, 1e-6, 20)
    assert len(compile_calls) == 1

    owner.generation += 1
    cache.run_prefix(*prefix, 1e-6, 1e-6, 20)
    assert len(compile_calls) == 2
    assert cache.status()["compiled_variants"] == ("prefix_no_collapse", )

    changed_dtype = list(prefix)
    changed_dtype[3] = changed_dtype[3].to(torch.bfloat16)
    cache.run_prefix(*changed_dtype, 1e-6, 1e-6, 20)
    assert len(compile_calls) == 3
    assert cache.status()["misses"] == 3


def test_cache_disables_after_first_failure_and_warns(monkeypatch, caplog):
    calls = []

    def failing_compile(function, **kwargs):
        calls.append((function, kwargs))
        def fail(*_args, **_kwargs):
            raise RuntimeError("trace failed")
        return fail

    monkeypatch.setattr(recipe_module.torch, "compile", failing_compile)
    owner = SimpleNamespace(generation=1, precision_fingerprint="precision")
    cache = recipe_module.C6ScopedRecipeCache(owner, backend="eager")
    args = _prefix_args(collapsed=False)
    assert cache.run_prefix(*args, 1e-6, 1e-6, 20) is None
    assert cache.run_prefix(*args, 1e-6, 1e-6, 20) is None
    assert len(calls) == 1
    status = cache.status()
    assert status["state"] == "disabled"
    assert status["failed_variant"] == "prefix_no_collapse"
    assert "falling back to eager" in status["disabled_reason"]
    assert any("Disabling C6 scoped recipe" in record.message for record in caplog.records)


def test_cache_records_explicit_eligibility_reason():
    owner = SimpleNamespace(generation=1, precision_fingerprint="precision")
    cache = recipe_module.C6ScopedRecipeCache(owner, backend="eager")
    cache.record_ineligible("C6 scoped recipe requires HPU tensors")
    status = cache.status()
    assert status["state"] == "not_enabled"
    assert status["ineligible_calls"] == 1
    assert status["ineligible_reasons"] == {"C6 scoped recipe requires HPU tensors": 1}


def test_scoped_recipes_trace_as_single_fullgraphs(monkeypatch):
    graphs = []

    def capture_backend(graph, inputs, **kwargs):
        del inputs, kwargs
        graphs.append(graph)
        return graph.forward

    def compile_and_run(function, args):
        torch._dynamo.reset()
        compiled = torch.compile(function, backend=capture_backend, fullgraph=True, dynamic=False)
        return compiled(*args, 1e-6, 1e-6, 20)

    no_collapse_args = _prefix_args(collapsed=False)
    with_collapse_args = _prefix_args(collapsed=True)
    post_args = _post_args()

    no_collapse_compile_args = (no_collapse_args[0], no_collapse_args[1], *no_collapse_args[3:])
    expected_prefix = recipe_module.c6_prefix_no_collapse(*no_collapse_compile_args, 1e-6, 1e-6, 20)
    actual_prefix = compile_and_run(recipe_module.c6_prefix_no_collapse, no_collapse_compile_args)
    _assert_tensor_tuple_equal(actual_prefix, expected_prefix)

    expected_collapsed = recipe_module.c6_prefix_with_collapse(*with_collapse_args, 1e-6, 1e-6, 20)
    actual_collapsed = compile_and_run(recipe_module.c6_prefix_with_collapse, with_collapse_args)
    _assert_tensor_tuple_equal(actual_collapsed, expected_collapsed)

    expected_post = recipe_module._c6_post_pre_moe_math(*post_args, 1e-6, 1e-6, 20)
    actual_post = compile_and_run(recipe_module._c6_post_pre_moe_math, post_args)
    _assert_tensor_tuple_equal(actual_post, expected_post)
    assert len(graphs) == 3


class _FakeAttention:
    def __init__(self):
        self._buffers = {}
        self._fused_qkv_quantized = False
        self._fused_qkv_weight = None

    def __call__(self, value, positions, ready_outputs=(), **kwargs):
        del positions, ready_outputs, kwargs
        return value + 1


class _FakeMoE(nn.Module):
    n256_fp8 = True
    n256_fused = True
    shared_gate_up_channel = torch.ones(1)

    def forward(self, value, image_mask, **kwargs):
        del image_mask, kwargs
        return value + 2


class _RecipeLayer(PreparedDecoderLayer):
    def __init__(self, cache):
        nn.Module.__init__(self)
        self.layer = 2
        self.draft = False
        self.draft_mhc = False
        self.collect_target_state = False
        self.dspark_mhc_deferred = False
        self.dspark_peer_post_collapse = False
        self.dspark_moe_peer_post = False
        self.dspark_peer_post_norm = False
        self.mhc_post_collapse = True
        self.mhc_interlayer_collapse = True
        self.mhc_interlayer_bf16 = True
        self.mhc_control_rrms = True
        self.mhc_control_mme = False
        self.ffn_dual_quant = True
        self.ffn_bf16_quant = False
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
            hc_attn_fn=torch.ones(1),
            hc_attn_scale=torch.ones(1),
            hc_attn_base=torch.ones(1),
            hc_ffn_fn=torch.ones(1),
            hc_ffn_scale=torch.ones(1),
            hc_ffn_base=torch.ones(1),
            attn_norm=SimpleNamespace(weight=torch.ones(1)),
            ffn_norm=SimpleNamespace(weight=torch.ones(1)),
        )
        self.attention = _FakeAttention()
        self.moe = _FakeMoE()
        self.all_gather = lambda value, dim: value
        self._c6_scoped_recipe_cache = cache


def _serial_inputs():
    residual = torch.zeros((6, 4, 4), dtype=torch.bfloat16)
    return {
        "residual": residual,
        "pre_mix": torch.zeros((6, 4), dtype=torch.float32),
        "positions": torch.zeros(6, dtype=torch.int32),
        "image_mask": torch.zeros(6, dtype=torch.bool),
    }


def test_serial_c6_never_touches_scoped_cache(monkeypatch):
    calls = []
    cache = SimpleNamespace(
        run_prefix=lambda *args, **kwargs: calls.append("prefix"),
        run_post_pre_moe=lambda *args, **kwargs: calls.append("post"),
        record_ineligible=lambda reason: calls.append(reason),
    )
    layer = _RecipeLayer(cache)
    inputs = _serial_inputs()

    monkeypatch.setattr(program_module, "hc_pre", _fake_hc_pre)
    monkeypatch.setattr(program_module, "rms_norm", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(program_module, "hc_post", lambda value, residual, post, comb: residual + value.unsqueeze(1))
    monkeypatch.setattr(program_module, "decode_ffn_dual_quant", _fake_ffn_dual_quant)

    layer._forward_c6_slice(
        inputs["residual"],
        inputs["pre_mix"],
        inputs["positions"],
        inputs["image_mask"],
        None,
        fp8_decode=False,
        selected_main=None,
        decode_metadata=None,
        collapse_handoff={},
        publish_collapse=False,
    )
    assert calls == []


def test_packed_wrappers_hit_scoped_cache(monkeypatch):
    compile_calls = []

    def recording_compile(function, **kwargs):
        compile_calls.append((function, kwargs))
        return _RecordingCompiled(function)

    monkeypatch.setattr(recipe_module.torch, "compile", recording_compile)
    layer = _RecipeLayer(recipe_module.C6ScopedRecipeCache(
        SimpleNamespace(generation=1, precision_fingerprint="precision"), backend="eager"
    ))
    monkeypatch.setattr(layer, "_c6_scoped_recipe_error", lambda **kwargs: None)
    inputs = _serial_inputs()

    layer._target_c6_attention_prefix(
        inputs["residual"],
        inputs["pre_mix"],
        inputs["positions"],
        inputs["image_mask"],
        None,
        fp8_decode=False,
    )
    layer._target_c6_attention_prefix(
        inputs["residual"],
        inputs["pre_mix"],
        inputs["positions"],
        inputs["image_mask"],
        None,
        fp8_decode=False,
    )
    status = layer._c6_scoped_recipe_cache.status()
    assert status["hits"] == 1
    assert status["misses"] == 1
    assert status["compiled_variants"] == ("prefix_no_collapse", )
    assert len(compile_calls) == 1

    attention = torch.ones((6, 4), dtype=torch.bfloat16)
    layer._target_c6_post_attention(
        attention,
        inputs["residual"],
        torch.ones((6, 4), dtype=torch.float32),
        torch.ones((6, 4), dtype=torch.float32),
        torch.ones((6, 4, 4), dtype=torch.float32),
        image_mask=inputs["image_mask"],
        schedule=False,
        fp8_decode=False,
        collapse_handoff={},
        publish_collapse=False,
    )
    layer._target_c6_post_attention(
        attention,
        inputs["residual"],
        torch.ones((6, 4), dtype=torch.float32),
        torch.ones((6, 4), dtype=torch.float32),
        torch.ones((6, 4, 4), dtype=torch.float32),
        image_mask=inputs["image_mask"],
        schedule=False,
        fp8_decode=False,
        collapse_handoff={},
        publish_collapse=False,
    )
    status = layer._c6_scoped_recipe_cache.status()
    assert status["hits"] == 2
    assert status["misses"] == 2
    assert status["compiled_variants"] == ("post_pre_moe", "prefix_no_collapse")
    assert len(compile_calls) == 2


def test_draft_construction_does_not_require_target_scoped_cache(monkeypatch):
    """MTP construction must not read the target-only cache from PreparedDraft."""
    for name, value in {
        "VLLM_HPU_DSV41_DSPARK_VOCAB_HEAD_FP8": False,
        "VLLM_HPU_DSV41_DSPARK_MTP_FP8": False,
        "VLLM_HPU_DSV41_DSPARK_SAMPLING_WIDTH": 128,
    }.items():
        monkeypatch.setattr(program_module.gaudi_envs, name, value)
    received_caches = []

    class DraftLayer(nn.Module):
        def __init__(self, *args, recipe_cache=None, **kwargs):
            super().__init__()
            received_caches.append(recipe_cache)
            self.moe = SimpleNamespace(mtp_cache=False, mtp_sat=False, draft_shared_fp8=False)
            self.attention = SimpleNamespace()
            self.draft_mhc = False

    monkeypatch.setattr(program_module, "PreparedDecoderLayer", DraftLayer)
    mtp = nn.Module()
    for index in range(3):
        mtp.add_module(str(index), nn.Module())
    stage = SimpleNamespace(
        weights=SimpleNamespace(mtp=mtp, head=nn.Module()), bf16_head=True,
        tp_rank=0, reduce=lambda value: value, all_gather=lambda value, **kwargs: value,
        tensor_parallel_size=8, shared=None,
        config={"text_config": {
            "dspark_num_experts_per_tok": 3, "rms_norm_eps": 1e-20, "dspark_noise_token_id": 0,
        }},
        shard=SimpleNamespace(manifest={
            "normal_scales": {f"mtp.{index}.ffn.experts": [None] for index in range(3)},
        }),
    )
    draft = program_module.PreparedDraft(stage, lookup=None, device="cpu")
    assert len(draft.layers) == 3
    assert received_caches == [None, None, None]


def test_cache_bound_when_adding_variant_to_existing_key(monkeypatch):
    monkeypatch.setattr(recipe_module.torch, "compile", lambda function, **kwargs: function)
    cache = recipe_module.C6ScopedRecipeCache(SimpleNamespace(generation=1), max_variants=3)
    value = torch.ones(1)

    def invoke(kind, contract):
        return cache._invoke(kind, lambda value: value, (value,), (), (value,), (contract,))

    invoke("prefix_no_collapse", 0)
    invoke("prefix_no_collapse", 1)
    invoke("post_pre_moe", 1)
    assert cache.status()["compiled_callable_count"] == 3
    invoke("post_pre_moe", 0)
    assert cache.status()["compiled_callable_count"] <= 3
    assert cache.status()["variant_kinds"] == ("post_pre_moe", "prefix_no_collapse")


@pytest.mark.parametrize("norm_mode", ["bf16", "bf16_native"])
def test_bf16_prefix_and_single_quant_trace_without_fp8_channels(monkeypatch, norm_mode):
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    original_compile = torch.compile
    monkeypatch.setattr(recipe_module.torch, "compile", lambda function, **kwargs: original_compile(
        function, backend=backend, fullgraph=True, dynamic=False))
    torch._dynamo.reset()
    cache = recipe_module.C6ScopedRecipeCache(SimpleNamespace(generation=1))
    for collapsed in (False, True):
        args = _prefix_args(collapsed=collapsed)
        actual = cache.run_prefix(*args, 1e-6, 1e-6, 20, norm_mode=norm_mode)
        expected = recipe_module._c6_prefix_math(*args, 1e-6, 1e-6, 20, norm_mode)
        _assert_tensor_tuple_equal(actual, expected)
        assert actual[-2:] == (None, None)
    args = _post_args()
    actual = cache.run_post_pre_moe(*args, 1e-6, 1e-6, 20, dual_quant=False)
    expected = recipe_module._c6_post_pre_moe_math(*args, 1e-6, 1e-6, 20, False)
    _assert_tensor_tuple_equal(actual, expected)
    assert actual[4:6] == (None, None)
    assert len(graphs) == 3
    assert cache.status()["compiled_callable_count"] == 3


def test_bf16_qkv_and_shared_inputs_do_not_block_scoped_eligibility(monkeypatch):
    layer = _RecipeLayer(None)
    layer.moe.shared_gate_up_channel = None
    monkeypatch.setattr(layer, "_c6_slice_path_error", lambda: None)
    monkeypatch.setattr(program_module.gaudi_envs, "VLLM_HPU_DSV41_MHC_GATES_FUSED", True)
    monkeypatch.setattr(program_module.gaudi_envs, "VLLM_HPU_DSV41_MHC_POSITIVE_GATES", True)
    residual = SimpleNamespace(dtype=torch.bfloat16, device=SimpleNamespace(type="hpu"))
    assert "fused_qkv_channel" not in layer.attention._buffers
    assert layer._c6_scoped_recipe_error(fp8_decode=False, residual=residual) is None


def test_mme_serving_control_is_eligible_and_passes_prepared_weights(monkeypatch):
    layer = _RecipeLayer(SimpleNamespace(
        run_prefix=lambda *args, **kwargs: kwargs,
        run_post_pre_moe=lambda *args, **kwargs: kwargs,
    ))
    layer.mhc_control_mme = True
    layer.hc_attn_fn_mme = torch.ones((24, 16), dtype=torch.bfloat16)
    layer.hc_ffn_fn_mme = torch.ones((24, 16), dtype=torch.bfloat16)
    layer.moe.shared_gate_up_channel = None
    monkeypatch.setattr(layer, "_c6_slice_path_error", lambda: None)
    monkeypatch.setattr(program_module.gaudi_envs, "VLLM_HPU_DSV41_MHC_GATES_FUSED", True)
    monkeypatch.setattr(program_module.gaudi_envs, "VLLM_HPU_DSV41_MHC_POSITIVE_GATES", True)
    residual = SimpleNamespace(dtype=torch.bfloat16, device=SimpleNamespace(type="hpu"))
    assert layer._c6_scoped_recipe_error(fp8_decode=False, residual=residual) is None
    prefix = layer._c6_scoped_prefix_math(residual, None, None, fp8_decode=False)
    post = layer._c6_scoped_post_pre_moe_math(None, residual, None, None, None, fp8_decode=False)
    assert prefix["control_mme_weight"] is layer.hc_attn_fn_mme
    assert prefix["norm_mode"] == "bf16"
    assert post["control_mme_weight"] is layer.hc_ffn_fn_mme
    assert post["dual_quant"] is False


def test_packed_target_adapter_exposes_scoped_recipe_status():
    from vllm_gaudi.v1.worker.deepseek_v41_dspark_batch_runner import DSparkPackedTargetStages

    expected = {"state": "active", "compiled_callable_count": 3}
    stages = DSparkPackedTargetStages.__new__(DSparkPackedTargetStages)
    stages.stage = SimpleNamespace(scoped_recipe_status=lambda: expected)
    assert stages.scoped_recipe_status() is expected
