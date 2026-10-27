# SPDX-License-Identifier: Apache-2.0
"""CPU correctness tests for the GLM-5.3 block-FP8 routed-MoE core.

The oracle intentionally dequantizes the complete E4M3FN matrix with the
checkpoint's multiplication scales and performs route-by-route FP32 matmuls.
It does not share the implementation's K-tile loop or expert grouping.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm_gaudi.ops.hpu_glm5next_fp8_moe import (
    CpuBlockFP8Backend,
    HPUBlockFP8Backend,
    block_fp8_gemm,
    glm53_fp8_moe,
    glm_swiglu_clamp,
    validate_hpu_fp8_schemas,
)

FP8 = torch.float8_e4m3fn
FP8_MAX = 240.0
QUANT_EPS = 1e-8


def _oracle_quantize_activation(x: torch.Tensor, block_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    quantized_blocks: list[torch.Tensor] = []
    scale_blocks: list[torch.Tensor] = []
    for start in range(0, x.shape[1], block_k):
        block = x[:, start:start + block_k].to(torch.float32)
        scale = (block.abs().amax(dim=1, keepdim=True) + QUANT_EPS) / FP8_MAX
        quantized = (block / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        quantized_blocks.append(quantized)
        scale_blocks.append(scale)
    return torch.cat(quantized_blocks, dim=1), torch.cat(scale_blocks, dim=1)


def _oracle_block_gemm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    block_n: int,
    block_k: int,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Full-matrix dequantization oracle; deliberately ignores K tiling."""
    quantized, activation_scale = _oracle_quantize_activation(activation, block_k)
    activation_dequant = quantized.to(torch.float32)
    activation_dequant *= activation_scale.repeat_interleave(block_k, dim=1)[:, :activation.shape[1]]

    expanded_weight_scale = weight_scale.repeat_interleave(block_n, dim=0)
    expanded_weight_scale = expanded_weight_scale.repeat_interleave(block_k, dim=1)
    weight_dequant = weight.to(torch.float32) * expanded_weight_scale[:weight.shape[0], :weight.shape[1]]
    return (activation_dequant @ weight_dequant.T).to(output_dtype)


def _oracle_rank_local_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    *,
    ep_rank: int,
    num_local_experts: int,
    block_n: int,
    block_k: int,
    limit: float,
) -> torch.Tensor:
    output = torch.zeros(hidden_states.shape, dtype=torch.float32)
    intermediate_size = w13_weight.shape[1] // 2
    expert_base = ep_rank * num_local_experts
    for token in range(hidden_states.shape[0]):
        for route in range(topk_ids.shape[1]):
            global_expert = int(topk_ids[token, route])
            local_expert = global_expert - expert_base
            if local_expert < 0 or local_expert >= num_local_experts:
                continue
            projection = _oracle_block_gemm(
                hidden_states[token:token + 1],
                w13_weight[local_expert],
                w13_weight_scale[local_expert],
                block_n=block_n,
                block_k=block_k,
            )
            gate = projection[:, :intermediate_size]
            up = projection[:, intermediate_size:]
            gate_clamped = gate.clamp(max=limit)
            up_clamped = up.clamp(min=-limit, max=limit)
            middle = (torch.nn.functional.silu(gate_clamped) * up_clamped).to(hidden_states.dtype)
            down = _oracle_block_gemm(
                middle,
                w2_weight[local_expert],
                w2_weight_scale[local_expert],
                block_n=block_n,
                block_k=block_k,
            )
            output[token] += down[0] * topk_weights[token, route].to(torch.float32)
    return output.to(hidden_states.dtype)


def _make_moe_weights(
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    block_n: int,
    block_k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    w13 = (torch.randn(num_local_experts, 2 * intermediate_size, hidden_size, generator=generator) * 2.5).to(FP8)
    w2 = (torch.randn(num_local_experts, hidden_size, intermediate_size, generator=generator) * 2.5).to(FP8)
    w13_shape = (
        num_local_experts,
        math.ceil(2 * intermediate_size / block_n),
        math.ceil(hidden_size / block_k),
    )
    w2_shape = (
        num_local_experts,
        math.ceil(hidden_size / block_n),
        math.ceil(intermediate_size / block_k),
    )
    w13_scale = torch.rand(w13_shape, generator=generator, dtype=torch.float32) * 0.03 + 0.01
    w2_scale = torch.rand(w2_shape, generator=generator, dtype=torch.float32) * 0.03 + 0.01
    return w13, w13_scale, w2, w2_scale


class _RecordingBackend:
    def __init__(self) -> None:
        self.name = "recording-cpu"
        self.delegate = CpuBlockFP8Backend()
        self.gemm_calls: list[tuple[int, int]] = []

    def quantize_activation(self, x: torch.Tensor, *, fp8_max: float, eps: float):
        return self.delegate.quantize_activation(x, fp8_max=fp8_max, eps=eps)

    def gemm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        self.gemm_calls.append(tuple(a.shape))
        return self.delegate.gemm(a, b, a_scale=a_scale, b_scale=b_scale, out_dtype=out_dtype)


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def test_block_gemm_matches_full_dequant_oracle_with_tails_and_nonuniform_scales():
    generator = torch.Generator().manual_seed(7)
    activation = torch.randn(5, 7, generator=generator)
    weight = (torch.randint(-24, 25, (9, 7), generator=generator, dtype=torch.int16).float() / 16).to(FP8)
    weight_scale = torch.tensor(
        [
            [0.25, 1.50, 0.75],
            [1.25, 0.50, 2.00],
            [0.875, 1.75, 0.375],
        ],
        dtype=torch.float32,
    )
    backend = _RecordingBackend()

    actual = block_fp8_gemm(
        activation,
        weight,
        weight_scale,
        block_size=(4, 3),
        backend=backend,
    )
    expected = _oracle_block_gemm(
        activation,
        weight,
        weight_scale,
        block_n=4,
        block_k=3,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    # Three K blocks force the implementation through the injected GEMM once
    # per K tile, including the tail block.
    assert backend.gemm_calls == [(5, 3), (5, 3), (5, 1)]
    assert weight.dtype == FP8
    assert weight_scale.dtype == torch.float32


def test_dynamic_activation_quantization_changes_the_result():
    activation = torch.linspace(0.123, 1.987, 64, dtype=torch.float32).reshape(1, 64)
    weight = torch.ones((1, 64), dtype=torch.float32).to(FP8)
    weight_scale = torch.ones((1, 4), dtype=torch.float32)

    actual = block_fp8_gemm(
        activation,
        weight,
        weight_scale,
        block_size=(16, 16),
    )
    expected = _oracle_block_gemm(
        activation,
        weight,
        weight_scale,
        block_n=16,
        block_k=16,
    )
    dequant_weight_scale = weight_scale.repeat_interleave(16, dim=0)
    dequant_weight_scale = dequant_weight_scale.repeat_interleave(16, dim=1)
    dequant_weight = weight.to(torch.float32) * dequant_weight_scale[:, :64]
    unquantized_activation = activation @ dequant_weight.T

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(actual, unquantized_activation, rtol=1e-5, atol=1e-5)


def test_glm_swiglu_clamps_positive_gate_and_symmetric_up_only():
    limit = 10.0
    gate = torch.tensor([[-20.0, -2.0, 2.0, 20.0]], dtype=torch.float32)
    up = torch.tensor([[-20.0, 3.0, -20.0, 20.0]], dtype=torch.float32)

    actual = glm_swiglu_clamp(gate, up, limit)
    expected = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    gate_lower_clamped = torch.nn.functional.silu(gate.clamp(-limit, limit)) * up.clamp(-limit, limit)
    no_up_clamp = torch.nn.functional.silu(gate.clamp(max=limit)) * up
    assert not torch.allclose(actual, gate_lower_clamped, rtol=0.0, atol=0.0)
    assert not torch.allclose(actual, no_up_clamp, rtol=0.0, atol=0.0)


def test_half_weight_double_scale_loader_pair_is_unchanged_and_equivalent():
    generator = torch.Generator().manual_seed(11)
    activation = torch.randn(3, 6, generator=generator)
    weight = torch.tensor(
        [
            [0.5, 1.0, 2.0, 4.0, -0.5, -1.0],
            [1.0, 2.0, 4.0, 8.0, -1.0, -2.0],
            [0.25, 0.5, 1.0, 2.0, -0.25, -0.5],
            [2.0, 4.0, 8.0, 16.0, -2.0, -4.0],
        ],
        dtype=torch.float32,
    ).to(FP8)
    weight_scale = torch.tensor(
        [[0.25, 0.5], [1.0, 2.0]],
        dtype=torch.float32,
    )
    halved_weight = (weight.to(torch.float32) * 0.5).to(FP8)
    doubled_scale = weight_scale * 2.0

    baseline = block_fp8_gemm(activation, weight, weight_scale, block_size=(2, 3))
    loader_pair = block_fp8_gemm(activation, halved_weight, doubled_scale, block_size=(2, 3))

    torch.testing.assert_close(loader_pair, baseline, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        weight.to(torch.float32) * weight_scale.repeat_interleave(2, dim=0).repeat_interleave(3, dim=1),
        halved_weight.to(torch.float32) * doubled_scale.repeat_interleave(2, dim=0).repeat_interleave(3, dim=1),
        rtol=0.0,
        atol=0.0,
    )


def test_rank_local_moe_matches_route_oracle_with_duplicate_padding_and_ep_partial():
    torch.manual_seed(17)
    num_local_experts = 2
    hidden_size = 4
    intermediate_size = 3
    block_n, block_k = 2, 2
    w13, w13_scale, w2, w2_scale = _make_moe_weights(
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        block_n=block_n,
        block_k=block_k,
        seed=23,
    )
    hidden_states = torch.randn(4, hidden_size)
    # EP rank 1 owns global experts 2 and 3.
    topk_ids = torch.tensor(
        [
            [2, 3, 0],
            [3, 3, -1],
            [4, 5, 2],
            [0, 1, -1],
        ],
        dtype=torch.int64,
    )
    topk_weights = torch.tensor(
        [
            [0.50, 0.25, 0.0],
            [0.75, 0.125, 0.0],
            [0.20, 0.30, 0.40],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    w13_before = w13.clone()
    w13_scale_before = w13_scale.clone()

    actual = glm53_fp8_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        w2,
        w2_scale,
        ep_rank=1,
        global_num_experts=6,
        swiglu_limit=10.0,
        block_size=(block_n, block_k),
        token_chunk=2,
    )
    expected = _oracle_rank_local_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        w2,
        w2_scale,
        ep_rank=1,
        num_local_experts=num_local_experts,
        block_n=block_n,
        block_k=block_k,
        limit=10.0,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert torch.count_nonzero(actual[3]) == 0
    assert torch.count_nonzero(actual[1]) > 0
    torch.testing.assert_close(w13, w13_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(w13_scale, w13_scale_before, rtol=0.0, atol=0.0)
    assert w13.dtype == FP8 and w13_scale.dtype == torch.float32


def test_bfloat16_activation_moe_matches_route_oracle():
    w13, w13_scale, w2, w2_scale = _make_moe_weights(
        num_local_experts=2,
        hidden_size=4,
        intermediate_size=3,
        block_n=2,
        block_k=2,
        seed=31,
    )
    hidden_states = torch.tensor(
        [
            [0.25, -0.50, 1.00, -1.25],
            [-0.75, 0.125, 0.375, 0.625],
        ],
        dtype=torch.bfloat16,
    )
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.bfloat16)

    actual = glm53_fp8_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        w2,
        w2_scale,
        ep_rank=0,
        global_num_experts=2,
        swiglu_limit=10.0,
        block_size=(2, 2),
        token_chunk=1,
    )
    expected = _oracle_rank_local_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        w2,
        w2_scale,
        ep_rank=0,
        num_local_experts=2,
        block_n=2,
        block_k=2,
        limit=10.0,
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)


def test_hpu_quantizer_promotes_bf16_cast_input_without_changing_fp32_scale(monkeypatch):
    """Regression for BF16 pre-cast rounding seen on the pinned HPU.

    Native cast with the original BF16 input moved normalized values such as
    -232.5 across an FP8 midpoint even though the FP32 input stayed on the
    midpoint.  The backend must promote only the cast values; the dynamic scale
    remains the FP32 row-block scale.
    """

    observed: dict[str, torch.Tensor | torch.dtype] = {}

    def fake_cast_to_fp8_v2(*, input, scale, stochastic_rounding, is_amax, dtype):
        observed["input"] = input
        observed["scale"] = scale
        observed["dtype"] = dtype
        observed["stochastic_rounding"] = stochastic_rounding
        observed["is_amax"] = is_amax
        return (input * scale).to(dtype), None

    monkeypatch.setattr(torch.ops.hpu, "cast_to_fp8_v2", fake_cast_to_fp8_v2)
    backend = object.__new__(HPUBlockFP8Backend)
    activation = torch.tensor(
        [[-1.0, -0.96875, -0.90625, 0.90625, 1.0]],
        dtype=torch.bfloat16,
    )

    quantized, scale = backend.quantize_activation(activation, fp8_max=FP8_MAX, eps=QUANT_EPS)
    expected_scale = (activation.float().abs().amax(dim=-1, keepdim=True) + QUANT_EPS) / FP8_MAX

    assert observed["input"].dtype == torch.float32
    assert torch.equal(observed["input"], activation.float())
    assert observed["scale"].dtype == torch.float32
    assert torch.equal(observed["scale"], expected_scale.reciprocal())
    assert observed["dtype"] == FP8
    assert observed["stochastic_rounding"] is False
    assert observed["is_amax"] is False
    assert scale.dtype == torch.float32
    assert torch.equal(scale, expected_scale)
    assert quantized.dtype == FP8
    torch.testing.assert_close(
        quantized.float(),
        (activation.float() * expected_scale.reciprocal()).to(FP8).float(),
        rtol=0.0,
        atol=0.0,
    )


def test_real_glm53_local_block_layout_h4096_i256():
    hidden_size = 4096
    intermediate_size = 256
    block_n, block_k = 128, 128
    generator = torch.Generator().manual_seed(41)
    w13_values = torch.randint(
        -8,
        9,
        (1, 2 * intermediate_size, hidden_size),
        generator=generator,
        dtype=torch.int8,
    )
    w2_values = torch.randint(
        -8,
        9,
        (1, hidden_size, intermediate_size),
        generator=generator,
        dtype=torch.int8,
    )
    w13 = w13_values.float().to(FP8)
    w2 = w2_values.float().to(FP8)
    w13_scale = torch.linspace(0.01, 0.05, 4 * 32, dtype=torch.float32).reshape(1, 4, 32)
    w2_scale = torch.linspace(0.02, 0.06, 32 * 2, dtype=torch.float32).reshape(1, 32, 2)
    hidden_states = torch.linspace(-1.0, 1.0, hidden_size, dtype=torch.bfloat16).reshape(1, hidden_size)
    topk_ids = torch.zeros((1, 1), dtype=torch.int64)
    topk_weights = torch.ones((1, 1), dtype=torch.bfloat16)

    actual = glm53_fp8_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        w2,
        w2_scale,
        ep_rank=0,
        global_num_experts=1,
        swiglu_limit=10.0,
        block_size=(block_n, block_k),
        token_chunk=1,
    )
    expected = _oracle_rank_local_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        w2,
        w2_scale,
        ep_rank=0,
        num_local_experts=1,
        block_n=block_n,
        block_k=block_k,
        limit=10.0,
    )

    assert actual.shape == (1, hidden_size)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_empty_batch_and_zero_activations_are_exact():
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32).to(FP8)
    weight_scale = torch.ones((1, 2), dtype=torch.float32)

    empty = block_fp8_gemm(
        torch.empty((0, 4), dtype=torch.bfloat16),
        weight,
        weight_scale,
        block_size=(2, 2),
    )
    zeros = block_fp8_gemm(
        torch.zeros((2, 4), dtype=torch.bfloat16),
        weight,
        weight_scale,
        block_size=(2, 2),
        output_dtype=torch.bfloat16,
    )
    assert empty.shape == (0, 1) and empty.dtype == torch.float32
    assert zeros.shape == (2, 1) and zeros.dtype == torch.bfloat16
    assert torch.count_nonzero(zeros) == 0

    w13 = torch.ones((1, 4, 4), dtype=torch.float32).to(FP8)
    w13_scale = torch.ones((1, 2, 2), dtype=torch.float32)
    w2 = torch.ones((1, 4, 2), dtype=torch.float32).to(FP8)
    w2_scale = torch.ones((1, 2, 1), dtype=torch.float32)
    empty_moe = glm53_fp8_moe(
        torch.empty((0, 4), dtype=torch.float32),
        torch.empty((0, 2), dtype=torch.int64),
        torch.empty((0, 2), dtype=torch.float32),
        w13,
        w13_scale,
        w2,
        w2_scale,
        ep_rank=0,
        global_num_experts=1,
        swiglu_limit=10.0,
        block_size=(2, 2),
    )
    zero_moe = glm53_fp8_moe(
        torch.zeros((2, 4), dtype=torch.bfloat16),
        torch.zeros((2, 1), dtype=torch.int64),
        torch.ones((2, 1), dtype=torch.bfloat16),
        w13,
        w13_scale,
        w2,
        w2_scale,
        ep_rank=0,
        global_num_experts=1,
        swiglu_limit=10.0,
        block_size=(2, 2),
    )
    assert empty_moe.shape == (0, 4) and empty_moe.dtype == torch.float32
    assert zero_moe.shape == (2, 4) and zero_moe.dtype == torch.bfloat16
    assert torch.count_nonzero(zero_moe) == 0


def test_rejects_unsupported_dtype_layout_scales_and_invalid_routes():
    activation = torch.ones((2, 4), dtype=torch.float32)
    weight = torch.ones((4, 4), dtype=torch.float32).to(FP8)
    scale = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(TypeError, match="float16 is intentionally unsupported"):
        block_fp8_gemm(activation.to(torch.float16), weight, scale, block_size=(2, 2))
    with pytest.raises(TypeError, match="must remain E4M3FN"):
        block_fp8_gemm(activation, weight.to(torch.float32), scale, block_size=(2, 2))
    with pytest.raises(ValueError, match="contiguous"):
        block_fp8_gemm(activation, weight.T, scale, block_size=(2, 2))
    with pytest.raises(TypeError, match="scale must be float32"):
        block_fp8_gemm(activation, weight, scale.double(), block_size=(2, 2))
    with pytest.raises(ValueError, match="scale shape"):
        block_fp8_gemm(activation, weight, scale[:1], block_size=(2, 2))
    nonfinite_scale = scale.clone()
    nonfinite_scale[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        block_fp8_gemm(activation, weight, nonfinite_scale, block_size=(2, 2))
    negative_scale = scale.clone()
    negative_scale[0, 0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        block_fp8_gemm(activation, weight, negative_scale, block_size=(2, 2))

    w13 = torch.ones((1, 4, 4), dtype=torch.float32).to(FP8)
    w13_scale = torch.ones((1, 2, 2), dtype=torch.float32)
    w2 = torch.ones((1, 4, 2), dtype=torch.float32).to(FP8)
    w2_scale = torch.ones((1, 2, 1), dtype=torch.float32)
    hidden = torch.ones((1, 4), dtype=torch.float32)
    valid_ids = torch.zeros((1, 1), dtype=torch.int64)
    valid_weights = torch.ones((1, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="invalid global expert id 1"):
        glm53_fp8_moe(
            hidden,
            valid_ids + 1,
            valid_weights,
            w13,
            w13_scale,
            w2,
            w2_scale,
            ep_rank=0,
            global_num_experts=1,
            swiglu_limit=10.0,
            block_size=(2, 2),
        )
    with pytest.raises(ValueError, match="padding routes must have exactly zero"):
        glm53_fp8_moe(
            hidden,
            valid_ids - 1,
            valid_weights,
            w13,
            w13_scale,
            w2,
            w2_scale,
            ep_rank=0,
            global_num_experts=1,
            swiglu_limit=10.0,
            block_size=(2, 2),
        )
    nonfinite_route = valid_weights.clone()
    nonfinite_route[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        glm53_fp8_moe(
            hidden,
            valid_ids,
            nonfinite_route,
            w13,
            w13_scale,
            w2,
            w2_scale,
            ep_rank=0,
            global_num_experts=1,
            swiglu_limit=10.0,
            block_size=(2, 2),
        )


def test_hpu_schema_validation_checks_real_B_scale_shape_keyword():
    cast_schema = (
        "hpu::cast_to_fp8_v2(Tensor input, Tensor? scale=None, bool stochastic_rounding=False, "
        "bool is_amax=False, ScalarType dtype=None, int[]? scale_shape=None) -> (Tensor, Tensor)"
    )
    gemm_schema = (
        "hpu::fp8_gemm_v2(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor? D, "
        "ScalarType out_dtype, Tensor? A_scale_inv=None, Tensor? B_scale_inv=None, "
        "Tensor? bias=None, bool accumulate=False, int[]? B_scale_shape=None) -> Tensor"
    )
    validated = validate_hpu_fp8_schemas(
        {"cast_to_fp8_v2": cast_schema, "fp8_gemm_v2": gemm_schema}
    )
    assert "B_scale_shape=None" in validated["fp8_gemm_v2"]

    with pytest.raises(RuntimeError, match="unsupported fp8_gemm_v2 schema"):
        validate_hpu_fp8_schemas(
            {
                "cast_to_fp8_v2": cast_schema,
                "fp8_gemm_v2": gemm_schema.replace("B_scale_shape=None", "scale_shape=None"),
            }
        )


def test_actual_hpu_dispatcher_schema_is_readable_without_executing_hpu_ops():
    schemas = validate_hpu_fp8_schemas()
    assert "B_scale_shape" in schemas["fp8_gemm_v2"]
    assert "scale_shape" in schemas["cast_to_fp8_v2"]
