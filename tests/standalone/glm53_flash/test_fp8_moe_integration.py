# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle and dispatch tests for the eager raw-block FP8 candidate.

Hardware/runtime and the upstream backend selector are patched explicitly;
these tests do not qualify HPU numerics or full-model support.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops import hpu_fp8


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def _layer(*, loaded=True, ep_size=1, ep_rank=0, local=2):
    total = local * ep_size
    expert_map = torch.full((total,), -1, dtype=torch.int32)
    expert_map[ep_rank * local:(ep_rank + 1) * local] = torch.arange(local, dtype=torch.int32)
    layer = SimpleNamespace(
        activation="silu", swiglu_limit=10.0, swiglu_alpha=None, swiglu_beta=None,
        local_num_experts=local, global_num_experts=total,
        expert_placement_strategy="linear", expert_map=expert_map,
        use_grouped_topk=True, custom_routing_function=None, top_k=1,
        weight_block_size=[128, 128], w13_input_scale=None, w2_input_scale=None,
        moe_config=SimpleNamespace(
            is_act_and_mul=True, ep_rank=ep_rank, ep_size=ep_size,
            hidden_dim=128, intermediate_size_per_partition=128,
            num_logical_experts=total, dp_size=1, is_sequence_parallel=False,
            moe_parallel_config=SimpleNamespace(enable_eplb=False),
        ),
    )
    if loaded:
        layer.w13_weight = torch.ones(local, 256, 128).to(torch.float8_e4m3fn)
        layer.w2_weight = torch.ones(local, 128, 128).to(torch.float8_e4m3fn)
        layer.w13_weight_scale_inv = torch.full((local, 2, 1), 0.125)
        layer.w2_weight_scale_inv = torch.full((local, 1, 1), 0.25)
    return layer


@pytest.fixture
def runtime(monkeypatch):
    flags = SimpleNamespace(scale_adjustment=True, use_dispatch_fn=False)
    model = SimpleNamespace(enforce_eager=True, hf_text_config=SimpleNamespace(model_type="glm5_next_text"))
    base_calls = []

    def base_init(self, quant_config, layer):
        base_calls.append(layer)
        self.quant_config = quant_config
        self.block_quant = quant_config.weight_block_size is not None

    monkeypatch.setattr(hpu_fp8.Fp8MoEMethod, "__init__", base_init)
    monkeypatch.setattr(hpu_fp8, "get_current_vllm_config_or_none", lambda: SimpleNamespace(model_config=model))
    monkeypatch.setattr(hpu_fp8, "get_config", lambda: flags)
    monkeypatch.setattr(hpu_fp8.hpu_ops, "get_config", lambda: flags)
    monkeypatch.setattr(hpu_fp8.hpu_ops, "is_hpu_gaudi2", True)
    monkeypatch.setattr(hpu_fp8.envs, "VLLM_HPU_FORCE_CHANNEL_FP8", False)
    monkeypatch.setattr(hpu_fp8, "model_has_quant_config", lambda: True)
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)

    def forbidden(*args, **kwargs):
        pytest.fail("candidate bound native MoE or converted raw weights")

    for name in ("VllmMixtureOfExpertsOpFP8", "VllmMixtureOfExpertsOpFP8PerChannel"):
        monkeypatch.setattr(hpu_fp8, name, forbidden)
    for name in ("fp8_block_moe_prepare_weights", "fp8_channel_moe_prepare_weights"):
        monkeypatch.setattr(hpu_fp8.hpu_ops, name, forbidden)
    quant = SimpleNamespace(
        is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128])
    return SimpleNamespace(flags=flags, model=model, quant=quant, base_calls=base_calls)


def _method(runtime, layer):
    return hpu_fp8.HPUFp8MoEMethod(runtime.quant, layer)


def test_constructor_only_reads_configuration(runtime):
    class UnloadedLayer(SimpleNamespace):
        def __getattribute__(self, name):
            if name in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
                pytest.fail("constructor inspected unloaded expert tensors")
            return super().__getattribute__(name)

    layer = UnloadedLayer(**vars(_layer(loaded=False)))
    method = _method(runtime, layer)
    assert method._glm53_fp8_spec.limit == 10.0
    assert runtime.base_calls == [layer]
    assert not hasattr(layer, "_hpu_glm53_fp8_prepared")


@pytest.mark.parametrize("case,diagnostic", [
    ("serialized", "serialized FP8"), ("gaudi", "Gaudi2"), ("adjustment", "scale adjustment"),
    ("activation_scheme", "dynamic activation"), ("block", r"raw \[128,128\]"),
    ("channel", "force-channel"), ("eager", "enforce_eager"), ("model", "GLM5Next"),
    ("activation", "ordinary gated SiLU"), ("limit", "swiglu_limit=10"),
    ("alpha", "alpha=1, beta=0"), ("beta", "alpha=1, beta=0"), ("gated", "gated"),
    ("eplb", "EPLB"), ("redundancy", "redundant"), ("physical", "logical"),
    ("placement", "linear expert placement"), ("unequal", "equal contiguous"),
    ("unaligned", "128-aligned"), ("bias", "bias-free"), ("input_route", "after W2"),
])
def test_constructor_rejects_unsupported_before_base(runtime, monkeypatch, case, diagnostic):
    layer = _layer(loaded=False)
    if case == "serialized":
        runtime.quant.is_checkpoint_fp8_serialized = False
    elif case == "gaudi":
        monkeypatch.setattr(hpu_fp8.hpu_ops, "is_hpu_gaudi2", False)
    elif case == "adjustment":
        runtime.flags.scale_adjustment = False
    elif case == "activation_scheme":
        runtime.quant.activation_scheme = "static"
    elif case == "block":
        runtime.quant.weight_block_size = [64, 128]
    elif case == "channel":
        monkeypatch.setattr(hpu_fp8.envs, "VLLM_HPU_FORCE_CHANNEL_FP8", True)
    elif case == "eager":
        runtime.model.enforce_eager = False
    elif case == "model":
        runtime.model.hf_text_config.model_type = "other_moe"
    elif case == "activation":
        layer.activation = "gelu"
    elif case == "limit":
        layer.swiglu_limit = 7.0
    elif case in ("alpha", "beta"):
        setattr(layer, f"swiglu_{case}", 2.0)
    elif case == "gated":
        layer.moe_config.is_act_and_mul = False
    elif case == "eplb":
        layer.moe_config.moe_parallel_config.enable_eplb = True
    elif case == "redundancy":
        layer.num_redundant_experts = 1
    elif case == "physical":
        layer.moe_config.num_logical_experts -= 1
    elif case == "placement":
        layer.expert_placement_strategy = "round_robin"
    elif case == "unequal":
        layer.global_num_experts += 1
    elif case == "unaligned":
        layer.moe_config.intermediate_size_per_partition = 96
    elif case == "bias":
        layer.moe_config.has_bias = True
    elif case == "input_route":
        layer.apply_router_weight_on_input = True
    with pytest.raises(NotImplementedError, match=diagnostic):
        _method(runtime, layer)
    assert runtime.base_calls == []


def test_create_weights_converts_each_checkpoint_tensor_once(runtime, monkeypatch):
    layer = _layer(loaded=False)
    method = _method(runtime, layer)
    captured = []
    wrappers = []
    original_wrapper = hpu_fp8.hpu_ops.gaudi_weight_wrapper

    def counted_wrapper(loader):
        wrappers.append(loader)
        return original_wrapper(loader)

    monkeypatch.setattr(hpu_fp8.hpu_ops, "gaudi_weight_wrapper", counted_wrapper)
    monkeypatch.setattr(hpu_fp8.Fp8MoEMethod, "create_weights", lambda self, *args, **kw: captured.append(kw))
    loaded = []

    def loader(param, loaded_weight, *args, **kwargs):
        loaded.append((param, loaded_weight.clone()))

    # A fresh create call must wrap the caller's loader once, never a cached wrapper.
    for _ in range(2):
        method.create_weights(layer=layer, weight_loader=loader)
        wrapped = captured[-1]["weight_loader"]
        for name in ("gate", "up", "down"):
            weight = torch.tensor([448.0, -448.0, 240.0, 0.0]).to(torch.float8_e4m3fn)
            scale = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
            wrapped(name, weight)
            wrapped(param=name, loaded_weight=scale)
            torch.testing.assert_close(loaded[-2][1].float(), weight.float() * 0.5, rtol=0, atol=0)
            torch.testing.assert_close(loaded[-1][1], scale * 2, rtol=0, atol=0)
            torch.testing.assert_close(loaded[-2][1].float() * loaded[-1][1][0, 0],
                                       weight.float() * scale[0, 0], rtol=0, atol=0)
    assert wrappers == [loader, loader]
    assert len(loaded) == 12


def test_process_keeps_raw_blocks_and_never_binds_native(runtime):
    layer = _layer(ep_size=2, ep_rank=1)
    method = _method(runtime, layer)
    original = hpu_fp8._glm53_fp8_tensors(layer)
    contents = [tensor.view(torch.uint8).clone() for tensor in original]
    method.process_weights_after_loading(layer)
    assert layer._hpu_glm53_fp8_prepared.spec.ep_rank == 1
    assert not hasattr(layer, "moe_op")
    for old, new, content in zip(original, hpu_fp8._glm53_fp8_tensors(layer), contents):
        assert old is new
        torch.testing.assert_close(new.view(torch.uint8), content, rtol=0, atol=0)


@pytest.mark.parametrize("case,diagnostic", [
    ("shape", "w13_weight_scale_inv"), ("weight_dtype", "w2_weight"), ("scale_dtype", "w2_weight_scale_inv"),
    ("nan", "finite positive"), ("zero", "finite positive"), ("layout", "contiguous"),
    ("map", "global-to-local"), ("map_missing", "explicit contiguous"),
    ("block", "loaded raw"), ("input_scale", "dynamic activation"), ("adjustment", "scale adjustment"),
])
def test_process_rejects_invalid_loaded_state_and_invalidates_prepared(runtime, case, diagnostic):
    layer = _layer(ep_size=2, ep_rank=1)
    method = _method(runtime, layer)
    method.process_weights_after_loading(layer)
    if case == "shape":
        layer.w13_weight_scale_inv = torch.ones(2, 256)
    elif case == "weight_dtype":
        layer.w2_weight = layer.w2_weight.float()
    elif case == "scale_dtype":
        layer.w2_weight_scale_inv = layer.w2_weight_scale_inv.bfloat16()
    elif case in ("nan", "zero"):
        layer.w2_weight_scale_inv.fill_(float("nan") if case == "nan" else 0)
    elif case == "layout":
        layer.w13_weight = layer.w13_weight.transpose(1, 2).contiguous().transpose(1, 2)
    elif case == "map":
        layer.expert_map[-2:] = torch.tensor([1, 0], dtype=torch.int32)
    elif case == "map_missing":
        layer.expert_map = None
    elif case == "block":
        layer.weight_block_size = [64, 128]
    elif case == "input_scale":
        layer.w13_input_scale = torch.ones(2)
    elif case == "adjustment":
        runtime.flags.scale_adjustment = False
    with pytest.raises((ValueError, NotImplementedError), match=diagnostic):
        method.process_weights_after_loading(layer)
    assert layer._hpu_glm53_fp8_prepared is None
    assert not hasattr(layer, "moe_op")


@pytest.mark.parametrize("routing", ["grouped", "custom", "regular"])
@pytest.mark.parametrize("dp,sp,deferred", [(1, False, False), (1, True, False), (2, False, False),
                                            (2, True, False), (2, False, True), (2, True, True)])
def test_forward_candidate_preserves_routing_dispatch_and_rank_partial(runtime, monkeypatch, routing, dp, sp, deferred):
    layer = _layer(ep_size=2, ep_rank=1)
    layer.moe_config.dp_size = dp
    layer.moe_config.is_sequence_parallel = sp
    layer.use_grouped_topk = routing == "grouped"
    layer.custom_routing_function = (lambda: None) if routing == "custom" else None
    runtime.flags.use_dispatch_fn = deferred
    method = _method(runtime, layer)
    method.process_weights_after_loading(layer)
    events = []
    x = torch.arange(256, dtype=torch.float32).reshape(2, 128).bfloat16()
    logits = torch.tensor([[0., 1., 2., 3.], [3., 2., 1., 0.]])
    ids = torch.tensor([[2], [0]], dtype=torch.int32)
    weights = torch.tensor([[0.75], [0.25]], dtype=torch.float32)

    def select(actual_layer, actual_x, actual_logits):
        assert actual_layer is layer
        assert actual_x.data_ptr() == x.data_ptr()
        assert actual_logits is logits
        events.append("select")
        return weights, ids

    metadata = SimpleNamespace(hidden_states_across_dp=object(), topk_ids_across_dp=object(),
                               topk_weights_across_dp=object())

    def dispatch(tensor, output=None, is_sequence_parallel=False):
        assert is_sequence_parallel == sp
        if tensor.shape[-1] == 128:
            kind, buffer = "hidden", metadata.hidden_states_across_dp
        elif tensor.dtype == torch.int64:
            kind, buffer = "ids", metadata.topk_ids_across_dp
        else:
            kind, buffer = "weights", metadata.topk_weights_across_dp
        assert output is (buffer if dp > 1 else None)
        events.append(kind)
        return torch.cat([tensor, tensor])

    def deferred_dispatch(tensor, *, is_sequence_parallel):
        assert deferred and dp > 1 and is_sequence_parallel == sp
        events.append("deferred_hidden")
        return torch.cat([tensor, tensor])

    def candidate(actual_x, actual_ids, actual_weights, *tensors, **kwargs):
        events.append("candidate")
        expanded = dp > 1 or sp
        expected_x = torch.cat([x, x]) if expanded else x
        torch.testing.assert_close(actual_x, expected_x)
        assert actual_ids.shape == actual_weights.shape == (expected_x.shape[0], 1)
        assert actual_ids.dtype == torch.int64 and actual_weights.dtype == x.dtype
        expected_ids = ids if routing != "regular" else torch.topk(logits.softmax(-1), 1).indices
        expected_weights = weights if routing != "regular" else torch.ones_like(weights)
        if expanded:
            expected_ids = torch.cat([expected_ids, expected_ids])
            expected_weights = torch.cat([expected_weights, expected_weights])
        torch.testing.assert_close(actual_ids, expected_ids.long())
        torch.testing.assert_close(actual_weights, expected_weights.to(x.dtype))
        assert all(old is new for old, new in zip(tensors, hpu_fp8._glm53_fp8_tensors(layer)))
        assert kwargs == dict(ep_rank=1, global_num_experts=4, swiglu_limit=10.0,
                              block_size=(128, 128), fp8_max=240.0)
        # Caller receives this partial unchanged; no combine/reduce occurs here.
        return torch.full_like(actual_x, 7)

    monkeypatch.setattr(hpu_fp8, "select_experts_from_routed", select)
    monkeypatch.setattr(hpu_fp8, "get_hpu_dp_metadata", lambda: metadata)
    monkeypatch.setattr(hpu_fp8, "dispatch_tensor", dispatch)
    monkeypatch.setattr(hpu_fp8, "dispatch_hidden_states", deferred_dispatch)
    monkeypatch.setattr(hpu_fp8, "glm53_fp8_moe", candidate)
    layer.moe_op = lambda *args, **kwargs: pytest.fail("candidate invoked native MoE")
    output = method.apply_monolithic(layer, x, logits)
    expected_events = [] if routing == "regular" else ["select"]
    if dp > 1 or sp:
        expected_events += (["ids", "weights", "deferred_hidden"] if deferred else ["hidden", "ids", "weights"])
    assert events == expected_events + ["candidate"]
    assert output.shape[0] == (4 if dp > 1 or sp else 2)
    assert torch.all(output == 7)


@pytest.mark.parametrize("case", ["unprepared", "rebound", "changed_map", "compile", "limit"])
def test_forward_fails_closed_before_route_selection(runtime, monkeypatch, case):
    layer = _layer()
    method = _method(runtime, layer)
    if case != "unprepared":
        method.process_weights_after_loading(layer)
    if case == "rebound":
        layer.w13_weight = layer.w13_weight.clone()
    elif case == "changed_map":
        layer.expert_map.fill_(-1)
    elif case == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif case == "limit":
        layer.swiglu_limit = 7.0
    monkeypatch.setattr(hpu_fp8, "select_experts_from_routed", lambda *a: pytest.fail("routed invalid candidate"))
    monkeypatch.setattr(hpu_fp8, "glm53_fp8_moe", lambda *a, **kw: pytest.fail("executed invalid candidate"))
    with pytest.raises((RuntimeError, NotImplementedError), match="GLM53 FP8"):
        method.apply_monolithic(layer, torch.ones(1, 128), torch.ones(1, 2))


def test_unclamped_non_glm_still_prepares_and_calls_native(runtime, monkeypatch):
    layer = _layer()
    layer.swiglu_limit = None
    runtime.model.enforce_eager = False
    runtime.model.hf_text_config.model_type = "other_moe"
    calls = []

    def native(x, ids, weights, **kwargs):
        calls.append(("native", kwargs))
        return x + 1

    monkeypatch.setattr(hpu_fp8, "VllmMixtureOfExpertsOpFP8", lambda *args: native)
    monkeypatch.setattr(hpu_fp8.hpu_ops, "fp8_block_moe_prepare_weights", lambda *args: calls.append(("prepare", args)))
    monkeypatch.setattr(hpu_fp8, "glm53_fp8_moe", lambda *a, **kw: pytest.fail("non-GLM called candidate"))
    monkeypatch.setattr(hpu_fp8, "select_experts_from_routed", lambda *a: (torch.ones(1, 1), torch.zeros(1, 1).int()))
    method = _method(runtime, layer)
    assert method._glm53_fp8_spec is None
    method.process_weights_after_loading(layer)
    x = torch.ones(1, 128)
    output = method.apply_monolithic(layer, x, torch.ones(1, 2))
    torch.testing.assert_close(output, x + 1)
    assert calls[0][0] == "prepare"
    assert calls[1] == ("native", dict(permuted_weights=True, activation="silu"))


def test_forward_uses_real_cpu_core_with_only_local_routes(runtime, monkeypatch):
    layer = _layer(ep_size=2, ep_rank=1)
    method = _method(runtime, layer)
    method.process_weights_after_loading(layer)
    monkeypatch.setattr(hpu_fp8, "select_experts_from_routed", lambda *args: (
        torch.tensor([[0.75], [1.0]]), torch.tensor([[2], [0]], dtype=torch.int32)))
    x = torch.full((2, 128), 100.0, dtype=torch.bfloat16)
    output = method.apply_monolithic(layer, x, torch.ones(2, 4))
    assert output.shape == x.shape and output.dtype == x.dtype
    # Both W13 halves clamp at 10; intermediate rounds to BF16 100. W2 is
    # 0.25 per element, so the local route is 128*100*0.25*0.75 = 2400.
    torch.testing.assert_close(output[0].float(), torch.full((128,), 2400.0), rtol=0, atol=0)
    assert torch.count_nonzero(output[1]) == 0


def test_prepared_expert_map_guard_supports_inference_tensors(runtime, monkeypatch):
    with torch.inference_mode():
        layer = _layer(ep_size=2, ep_rank=1)
        method = _method(runtime, layer)
        method.process_weights_after_loading(layer)
        monkeypatch.setattr(hpu_fp8, "select_experts_from_routed", lambda *a: (
            torch.ones(1, 1), torch.full((1, 1), 2, dtype=torch.int32)))
        monkeypatch.setattr(hpu_fp8, "glm53_fp8_moe", lambda x, *a, **kw: x)
        x = torch.ones(1, 128)
        torch.testing.assert_close(method.apply_monolithic(layer, x, torch.ones(1, 2)), x)
        layer.expert_map.fill_(-1)
        with pytest.raises(RuntimeError, match="after loading/rebinding"):
            method.apply_monolithic(layer, x, torch.ones(1, 2))
