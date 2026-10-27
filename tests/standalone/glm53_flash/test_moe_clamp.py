# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for GLM routed-expert SiLU clamping on HPU MoE paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_gaudi.ops.hpu_fused_moe as hpu_fused_moe
from vllm_gaudi.ops.hpu_fused_moe import _gather_swigluoai_moe, _unfused_swigluoai_moe
from vllm_gaudi.ops.hpu_glm5next_moe import (
    require_fp8_clamped_moe_supported,
    resolve_glm_clamp_config,
)


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def _glm_layer(*, limit: float | None = 10.0, ep_rank: int = 0, local_experts: int = 1):
    return SimpleNamespace(
        swiglu_limit=limit,
        swiglu_alpha=None,
        swiglu_beta=None,
        local_num_experts=local_experts,
        moe_config=SimpleNamespace(
            swiglu_limit=limit,
            swiglu_alpha=None,
            swiglu_beta=None,
            is_act_and_mul=True,
            ep_rank=ep_rank,
        ),
    )


def _clamped_glm_act(gate: torch.Tensor, up: torch.Tensor, limit: float = 10.0) -> torch.Tensor:
    gate_c = torch.clamp(gate, max=limit)
    up_c = torch.clamp(up, min=-limit, max=limit)
    return gate_c * torch.sigmoid(gate_c) * up_c


@torch.no_grad()
def _handwritten_rank_local_oracle(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    limit: float = 10.0,
) -> torch.Tensor:
    """Scalar/matrix oracle for one EP rank's routed-expert partial."""
    num_local = layer.local_num_experts
    experts_min = layer.moe_config.ep_rank * num_local
    out = torch.zeros_like(x, dtype=torch.float32)
    for token in range(x.shape[0]):
        for route in range(topk_ids.shape[1]):
            global_expert = int(topk_ids[token, route])
            if global_expert < experts_min or global_expert >= experts_min + num_local:
                # Non-local experts contribute exactly zero to this rank's
                # partial; the outer MoERunner owns the EP reduction.
                continue
            local_expert = global_expert - experts_min
            projected = x[token].float() @ layer.w13_weight[local_expert].float().T
            intermediate = projected.shape[-1] // 2
            gate = projected[:intermediate]
            up = projected[intermediate:]
            activation = _clamped_glm_act(gate, up, limit)
            token_out = activation @ layer.w2_weight[local_expert].float().T
            out[token] += topk_weights[token, route].float() * token_out
    return out


def test_resolve_glm_clamp_from_layer_moe_config_and_wrapper():
    direct = _glm_layer(limit=10.0)
    config = resolve_glm_clamp_config(direct, "silu")
    assert config is not None
    assert (config.limit, config.alpha, config.beta) == (10.0, 1.0, 0.0)

    # Current RoutedExperts exposes both attributes, but older wrappers may only
    # retain the value in moe_config.  A None top-level value must not mask it.
    layer_without_top_level = SimpleNamespace(
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        moe_config=SimpleNamespace(swiglu_limit=10.0, is_act_and_mul=True),
    )
    hidden = resolve_glm_clamp_config(layer_without_top_level, "silu")
    assert hidden is not None and hidden.limit == 10.0

    routed_experts = _glm_layer(limit=10.0)
    runner = SimpleNamespace(routed_experts=routed_experts)
    nested = resolve_glm_clamp_config(runner, "silu")
    assert nested is not None and nested.limit == 10.0

    assert resolve_glm_clamp_config(_glm_layer(limit=None), "silu") is None
    # MiniMax keeps its existing swigluoai_uninterleave branch and defaults.
    assert resolve_glm_clamp_config(_glm_layer(limit=7.0), "swigluoai_uninterleave") is None


def test_non_glm_alpha_beta_and_non_gated_are_fail_closed():
    layer = _glm_layer(limit=10.0)
    layer.swiglu_alpha = 1.702
    layer.swiglu_beta = 1.0
    with pytest.raises(NotImplementedError, match="alpha=1, beta=0"):
        resolve_glm_clamp_config(layer, "silu")

    non_gated = _glm_layer(limit=10.0)
    non_gated.moe_config.is_act_and_mul = False
    with pytest.raises(NotImplementedError, match="gated"):
        resolve_glm_clamp_config(non_gated, "silu")


def test_activation_path_dispatch_glm_unclamped_and_minimax(monkeypatch):
    calls: list[tuple[str, tuple]] = []

    def fake_swigluoai(layer, x, topk_ids, topk_weights, *, alpha, beta, limit):
        calls.append(("swigluoai", (alpha, beta, limit)))
        return x

    class FakeMoeOp:

        def __call__(self, x, topk_ids, topk_weights, **kwargs):
            calls.append(("native", (kwargs["activation"], kwargs["permuted_weights"])))
            return x

    monkeypatch.setattr(hpu_fused_moe, "_swigluoai_moe", fake_swigluoai)
    x = torch.ones(2, 1, dtype=torch.bfloat16)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    topk_weights = torch.ones(2, 1, dtype=torch.bfloat16)

    glm = _glm_layer(limit=10.0)
    glm.w13_weight = torch.zeros(1, 2, 1, dtype=torch.bfloat16)
    glm.w2_weight = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
    glm.moe_op = FakeMoeOp()
    hpu_fused_moe._apply_moe_activation_path(
        glm,
        x,
        topk_ids,
        topk_weights,
        "silu",
        minimax_alpha=1.702,
        minimax_beta=1.0,
        minimax_limit=7.0,
    )
    assert calls[-1] == ("swigluoai", (1.0, 0.0, 10.0))

    unclamped = _glm_layer(limit=None)
    unclamped.moe_op = FakeMoeOp()
    hpu_fused_moe._apply_moe_activation_path(
        unclamped,
        x,
        topk_ids,
        topk_weights,
        "silu",
        minimax_alpha=1.702,
        minimax_beta=1.0,
        minimax_limit=7.0,
    )
    assert calls[-1] == ("native", ("silu", True))

    minimax = _glm_layer(limit=7.0)
    minimax.moe_op = FakeMoeOp()
    hpu_fused_moe._apply_moe_activation_path(
        minimax,
        x,
        topk_ids,
        topk_weights,
        "swigluoai_uninterleave",
        minimax_alpha=1.702,
        minimax_beta=1.0,
        minimax_limit=7.0,
    )
    assert calls[-1] == ("swigluoai", (1.702, 1.0, 7.0))


def test_bf16_glm_gather_matches_handwritten_clamped_rank_local_oracle():
    layer = _glm_layer(limit=10.0, ep_rank=1, local_experts=3)
    layer.w13_weight = torch.tensor(
        [
            [[12.0], [12.0]],    # local expert 0: positive gate saturates, up=+12
            [[-12.0], [-12.0]],  # local expert 1: negative gate is not lower-clamped
            [[5.0], [-12.0]],    # local expert 2: up=-12 saturates
        ],
        dtype=torch.bfloat16,
    )
    layer.w2_weight = torch.tensor([[[1.0]], [[1.0]], [[1.0]]], dtype=torch.bfloat16)
    x = torch.ones(2, 1, dtype=torch.bfloat16)
    # EP rank 1 owns global experts 3..5; global expert 0 is non-local.
    topk_ids = torch.tensor([[3, 4], [5, 0]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.bfloat16)

    expected = _handwritten_rank_local_oracle(layer, x, topk_ids, topk_weights, limit=10.0)
    gathered = _gather_swigluoai_moe(
        layer,
        x,
        topk_ids,
        topk_weights,
        alpha=1.0,
        beta=0.0,
        limit=10.0,
    )
    dense = _unfused_swigluoai_moe(
        layer,
        x,
        topk_ids,
        topk_weights,
        alpha=1.0,
        beta=0.0,
        limit=10.0,
    )

    torch.testing.assert_close(gathered.float(), expected, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(dense.float(), expected, atol=3e-2, rtol=2e-2)

    # Explicitly distinguish GLM semantics from the two common wrong variants:
    # no clamp, and symmetrically lower-clamping the gate.
    unclamped = _handwritten_rank_local_oracle(layer, x, topk_ids, topk_weights, limit=1e6)
    gate_lower_clamped = x.new_zeros(2, 1, dtype=torch.float32)
    for token in range(2):
        for route in range(2):
            global_expert = int(topk_ids[token, route])
            if not 3 <= global_expert < 6:
                continue
            local_expert = global_expert - 3
            projected = x[token].float() @ layer.w13_weight[local_expert].float().T
            gate = torch.clamp(projected[:1], min=-10.0, max=10.0)
            up = torch.clamp(projected[1:], min=-10.0, max=10.0)
            act = gate * torch.sigmoid(gate) * up
            token_out = act @ layer.w2_weight[local_expert].float().T
            gate_lower_clamped[token] += topk_weights[token, route].float() * token_out
    assert not torch.allclose(expected, unclamped, atol=1e-3, rtol=1e-3)
    assert not torch.allclose(expected, gate_lower_clamped, atol=1e-6, rtol=0.0)

    # The non-local route contributes exactly zero on token 1.
    local_only = _handwritten_rank_local_oracle(
        layer,
        x[1:],
        topk_ids[1:, :1],
        topk_weights[1:, :1],
        limit=10.0,
    )
    torch.testing.assert_close(expected[1:], local_only, atol=0.0, rtol=0.0)


def test_fp8_clamp_without_candidate_capability_never_dispatches_native(monkeypatch):
    from vllm_gaudi.ops import hpu_fp8

    layer = _glm_layer(limit=10.0, ep_rank=0, local_experts=1)
    layer.moe_op = lambda *args, **kwargs: pytest.fail("clamped FP8 reached native MoE")
    method = object.__new__(hpu_fp8.HPUFp8MoEMethod)
    monkeypatch.setattr(hpu_fp8, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(hpu_fp8, "get_config", lambda: SimpleNamespace(scale_adjustment=True))
    # Missing model/config eligibility fails before base construction or loading.
    with pytest.raises(NotImplementedError, match="GLM5Next model configuration"):
        method.__init__(quant_config=None, layer=layer)
    with pytest.raises(NotImplementedError, match="refusing an unclamped native MoE or W8A16 fallback"):
        method.process_weights_after_loading(layer)
    with pytest.raises(NotImplementedError, match="refusing an unclamped native MoE or W8A16 fallback"):
        method.apply_monolithic(layer, torch.empty(1, 2), torch.empty(1, 1))
    assert require_fp8_clamped_moe_supported(_glm_layer(limit=None)) is None
