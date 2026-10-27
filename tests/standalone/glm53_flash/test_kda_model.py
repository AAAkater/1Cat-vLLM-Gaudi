# SPDX-License-Identifier: Apache-2.0
"""CPU tests for HPU GLM-5.3 KDA model wiring helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention as UpstreamGlm5NextLinearAttention
from vllm_gaudi.models.glm5next_kda import (
    HPUGlm5NextLinearAttention,
    _build_token_mask,
    _input_state,
    _merge_conv_weights,
    _resolve_state_indices,
    _scatter_active_states,
    _validate_state_indices,
)
from vllm_gaudi.ops.hpu_glm5next_norm import rmsnorm_sigmoid_gate


def test_state_indices_select_exact_cache_group_row():
    metadata = SimpleNamespace(
        load_indices_tensor=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32),
        store_indices_tensor=torch.tensor([[30, 31, 32], [40, 41, 42]], dtype=torch.int32),
    )
    owner = SimpleNamespace(cache_group_idx=torch.tensor(1))

    load = _resolve_state_indices(owner, metadata, "load_indices_tensor")
    store = _resolve_state_indices(owner, metadata, "store_indices_tensor")

    torch.testing.assert_close(load, torch.tensor([20, 21, 22], dtype=torch.int32))
    torch.testing.assert_close(store, torch.tensor([40, 41, 42], dtype=torch.int32))

    owner.cache_group_idx = None
    with pytest.raises(RuntimeError, match="no cache_group_idx"):
        _resolve_state_indices(owner, metadata, "load_indices_tensor")


def test_validation_allows_repeated_padding_even_when_slots_are_fewer_than_lanes():
    _validate_state_indices(
        torch.tensor([2, 0, -1, -1]),
        num_slots=3,
        name="load_indices",
    )

    with pytest.raises(ValueError, match="duplicate live state slots"):
        _validate_state_indices(
            torch.tensor([2, 2, -1, -1]),
            num_slots=3,
            name="load_indices",
        )
    with pytest.raises(IndexError, match="outside state pool"):
        _validate_state_indices(
            torch.tensor([3, 0, -1, -1]),
            num_slots=3,
            name="load_indices",
        )
    with pytest.raises(IndexError, match="only use -1"):
        _validate_state_indices(
            torch.tensor([2, -2, -1, -1]),
            num_slots=3,
            name="load_indices",
        )


def test_rectangular_token_mask_marks_only_real_tokens():
    lengths = torch.tensor([2, 1, 0])
    mask = _build_token_mask(None, lengths, seq_len=3, num_tokens=9, device=torch.device("cpu"))
    assert mask.tolist() == [True, True, False, True, False, False, False, False, False]

    provided = torch.tensor([[1], [0], [1], [0], [0], [0]], dtype=torch.int32)
    torch.testing.assert_close(
        _build_token_mask(provided, torch.tensor([2, 0]), 3, 6, torch.device("cpu")),
        provided.reshape(-1).bool(),
    )

    with pytest.raises(ValueError, match="must contain"):
        _build_token_mask(torch.ones(5), lengths, 3, 9, torch.device("cpu"))


def test_initial_state_uses_zero_for_padding_and_missing_history():
    ssm_state = torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(4, 1, 2, 2)
    load_indices = torch.tensor([3, 1, -1])
    active_lanes = torch.tensor([True, True, False])
    has_initial_state = torch.tensor([True, False, False])

    initial = _input_state(ssm_state, load_indices, active_lanes, has_initial_state)

    torch.testing.assert_close(initial[0], ssm_state[3])
    torch.testing.assert_close(initial[1], torch.zeros_like(initial[1]))
    torch.testing.assert_close(initial[2], torch.zeros_like(initial[2]))


def test_final_state_scatter_never_writes_padding_slots():
    ssm_state = torch.zeros(4, 1, 2, 2)
    final_state = torch.arange(4 * 4, dtype=torch.float32).reshape(4, 1, 2, 2)
    store_indices = torch.tensor([2, 0, -1, -1])
    active_lanes = torch.tensor([True, True, False, False])

    passthrough = torch.tensor([7.0])
    returned = _scatter_active_states(passthrough, final_state, ssm_state, store_indices, active_lanes)

    assert returned is passthrough
    torch.testing.assert_close(ssm_state[2], final_state[0])
    torch.testing.assert_close(ssm_state[0], final_state[1])
    torch.testing.assert_close(ssm_state[1], torch.zeros_like(ssm_state[1]))
    torch.testing.assert_close(ssm_state[3], torch.zeros_like(ssm_state[3]))


def test_rmsnorm_sigmoid_gate_matches_fp32_reference_and_zero_padding():
    x = torch.randn(2, 4, 5, dtype=torch.bfloat16)
    gate = torch.randn(2, 4, 5, dtype=torch.bfloat16)
    weight = torch.randn(5)
    x[0, 0].zero_()
    gate[0, 0].zero_()

    actual = rmsnorm_sigmoid_gate(x, gate, weight, eps=1e-5)
    xf = x.float()
    expected = (
        xf
        * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + 1e-5)
        * weight.float()
        * torch.sigmoid(gate.float())
    ).to(x.dtype)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual[0, 0], torch.zeros_like(actual[0, 0]))


def test_merged_conv_weight_preserves_q_then_k_then_v_channel_order():
    q = torch.arange(2 * 1 * 3, dtype=torch.float32).reshape(2, 1, 3)
    k = q + 100
    v = q + 200

    merged = _merge_conv_weights(q, k, v)

    assert merged.shape == (6, 3)
    torch.testing.assert_close(merged[:2], q[:, 0])
    torch.testing.assert_close(merged[2:4], k[:, 0])
    torch.testing.assert_close(merged[4:], v[:, 0])


def test_prefill_metadata_assembles_padding_and_per_layer_state_contract():
    owner = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        cache_group_idx=torch.tensor(1),
        local_num_heads=1,
        head_dim=2,
        conv_size=3,
        local_projection_size=2,
        kv_cache=(
            torch.arange(5 * 2 * 6, dtype=torch.bfloat16).reshape(5, 2, 6),
            torch.arange(5 * 1 * 2 * 2, dtype=torch.float32).reshape(5, 1, 2, 2),
        ),
    )
    metadata = SimpleNamespace(
        is_prompt=True,
        num_prefills=2,
        load_indices_tensor=torch.tensor([[0, 0, 0, 0], [4, 1, -1, -1]], dtype=torch.int32),
        store_indices_tensor=torch.tensor([[0, 0, 0, 0], [4, 1, -1, -1]], dtype=torch.int32),
        query_start_loc_p=torch.tensor([0, 2, 3, 3, 3], dtype=torch.int32),
        has_initial_states_p=torch.tensor([1, 0, 0, 0], dtype=torch.int32),
        padding_mask_flat=torch.tensor(
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=torch.bfloat16,
        ).view(-1, 1),
    )
    context = SimpleNamespace(attn_metadata=metadata)

    with patch("vllm_gaudi.models.glm5next_kda.get_forward_context", return_value=context):
        extracted = HPUGlm5NextLinearAttention._extract_metadata(owner, num_tokens=12)

    assert extracted is not None
    assert extracted.prefill_num_seqs == 4
    assert extracted.prefill_seq_len == 3
    assert extracted.active_lanes.tolist() == [True, True, False, False]
    assert extracted.query_start_loc.tolist() == [0, 2, 3, 3, 3]
    torch.testing.assert_close(extracted.initial_state[0], owner.kv_cache[1][4])
    torch.testing.assert_close(extracted.initial_state[1], torch.zeros_like(extracted.initial_state[1]))
    torch.testing.assert_close(extracted.initial_state[2], torch.zeros_like(extracted.initial_state[2]))
    assert extracted.conv_scratch_state is not None
    torch.testing.assert_close(extracted.conv_scratch_state[0], owner.kv_cache[0][4])
    torch.testing.assert_close(
        extracted.conv_scratch_state[1],
        torch.zeros_like(extracted.conv_scratch_state[1]),
    )
    torch.testing.assert_close(
        extracted.conv_scratch_state[2],
        torch.zeros_like(extracted.conv_scratch_state[2]),
    )


def test_decode_metadata_allows_fewer_state_slots_than_padded_lanes():
    owner = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        cache_group_idx=torch.tensor(0),
        local_num_heads=1,
        head_dim=2,
        conv_size=3,
        local_projection_size=2,
        kv_cache=(
            torch.zeros(3, 2, 6, dtype=torch.bfloat16),
            torch.zeros(3, 1, 2, 2, dtype=torch.float32),
        ),
    )
    indices = torch.tensor([-1, 1, 2, -1], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prompt=False,
        load_indices_tensor=indices.unsqueeze(0),
        store_indices_tensor=indices.unsqueeze(0),
        query_start_loc_p=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
    )
    context = SimpleNamespace(attn_metadata=metadata)

    with patch("vllm_gaudi.models.glm5next_kda.get_forward_context", return_value=context):
        extracted = HPUGlm5NextLinearAttention._extract_metadata(owner, num_tokens=4)

    assert extracted is not None
    assert extracted.token_mask.tolist() == [False, True, True, False]
    assert extracted.active_lanes.tolist() == [False, True, True, False]
    assert extracted.query_start_loc.tolist() == [0, 1, 2, 3, 4]


def test_glm53_tp8_state_shape_matches_sd_wiring_and_ds_is_unsupported():
    """Use upstream allocation for GLM-5.3 geometry; HPU KDA wiring is SD-only."""
    tp_size, num_heads, head_dim, conv_size = 8, 64, 128, 4
    local_heads = num_heads // tp_size
    conv_dim = 3 * num_heads * head_dim // tp_size

    layer = object.__new__(UpstreamGlm5NextLinearAttention)
    nn.Module.__init__(layer)
    layer.tp_size = tp_size
    layer.num_heads = num_heads
    layer.head_dim = head_dim
    layer.conv_size = conv_size
    layer.num_spec = 0

    with patch("vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first", return_value=False):
        upstream_shape = layer.get_state_shape()
        calculator_shape = MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            num_heads,
            head_dim,
            conv_kernel_size=conv_size,
        )

    assert upstream_shape == calculator_shape == ((conv_size - 1, conv_dim), (local_heads, head_dim, head_dim))

    slots = 2
    conv_pool = torch.zeros((slots, *upstream_shape[0]), dtype=torch.bfloat16)
    ssm_pool = torch.zeros((slots, *upstream_shape[1]), dtype=torch.float32)
    owner = SimpleNamespace(
        prefix="model.layers.0.linear_attn",
        local_projection_size=local_heads * head_dim,
        local_num_heads=local_heads,
        head_dim=head_dim,
        conv_size=conv_size,
        kv_cache=(conv_pool, ssm_pool),
    )
    metadata = SimpleNamespace(
        is_prompt=False,
        num_prefills=0,
        num_decodes=1,
        load_indices_tensor=torch.tensor([0], dtype=torch.int32),
        store_indices_tensor=torch.tensor([0], dtype=torch.int32),
        query_start_loc_p=torch.tensor([0, 1], dtype=torch.int32),
    )
    context = SimpleNamespace(attn_metadata=metadata)
    with patch("vllm_gaudi.models.glm5next_kda.get_forward_context", return_value=context):
        extracted = HPUGlm5NextLinearAttention._extract_metadata(owner, num_tokens=1)

    assert extracted is not None
    assert tuple(extracted.conv_state.shape) == (slots, conv_size - 1, conv_dim)
    assert tuple(extracted.ssm_state.shape) == (slots, local_heads, head_dim, head_dim)

    with patch("vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first", return_value=True):
        ds_shape = layer.get_state_shape()
    assert ds_shape == ((conv_dim, conv_size - 1), (local_heads, head_dim, head_dim))

    owner.kv_cache = (
        torch.zeros((slots, *ds_shape[0]), dtype=torch.bfloat16),
        ssm_pool,
    )
    with patch("vllm_gaudi.models.glm5next_kda.get_forward_context", return_value=context):
        with pytest.raises(ValueError, match="HPU KDA conv state must use"):
            HPUGlm5NextLinearAttention._extract_metadata(owner, num_tokens=1)


class _TupleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, generator: torch.Generator):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, generator=generator, dtype=torch.float32) * 0.2
        )

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight), None


class _ConvParameter(nn.Module):
    def __init__(self, channels: int, width: int, generator: torch.Generator):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels, 1, width, generator=generator, dtype=torch.float32) * 0.3)
        self.bias = None


def _make_tiny_kda_model():
    generator = torch.Generator().manual_seed(101)
    model = object.__new__(HPUGlm5NextLinearAttention)
    nn.Module.__init__(model)
    model.prefix = "model.layers.0.self_attn"
    model.cache_group_idx = torch.tensor(1)
    model.local_num_heads = 1
    model.head_dim = 2
    model.local_projection_size = 2
    model.conv_size = 3
    model.kda_lower_bound = -5.0

    model.in_proj_qkvbfg_a = _TupleLinear(3, 11, generator)
    model.f_b_proj = _TupleLinear(2, 2, generator)
    model.g_b_proj = _TupleLinear(2, 2, generator)
    model.o_proj = _TupleLinear(2, 3, generator)
    model.q_conv1d = _ConvParameter(2, 3, generator)
    model.k_conv1d = _ConvParameter(2, 3, generator)
    model.v_conv1d = _ConvParameter(2, 3, generator)
    model.A_log = nn.Parameter(torch.tensor([[[[0.2]]]], dtype=torch.float32))
    model.dt_bias = nn.Parameter(torch.tensor([0.1, -0.2], dtype=torch.float32))
    model.o_norm = SimpleNamespace(
        weight=nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.float32)),
        eps=1e-5,
    )
    model._merged_conv_weight = None

    conv_before = torch.arange(5 * 2 * 6, dtype=torch.float32).reshape(5, 2, 6) / 20
    ssm_before = torch.arange(5 * 1 * 2 * 2, dtype=torch.float32).reshape(5, 1, 2, 2) / 17
    model.kv_cache = (conv_before.clone(), ssm_before.clone())
    return model, conv_before, ssm_before


def _project_kda(model, hidden_states: torch.Tensor):
    projected = F.linear(hidden_states, model.in_proj_qkvbfg_a.weight)
    qkv, beta_raw, f_a, g_a = projected.split([6, 1, 2, 2], dim=-1)
    g1 = F.linear(f_a, model.f_b_proj.weight).reshape(-1, 1, 2)
    g2 = F.linear(g_a, model.g_b_proj.weight).reshape(-1, 1, 2)
    head_scale = torch.exp(model.A_log.reshape(1).float()).reshape(1, 1)
    gate = -5.0 * torch.sigmoid(head_scale * (g1.float() + model.dt_bias.reshape(1, 2).float()))
    beta = torch.sigmoid(beta_raw.float())
    return qkv, gate, beta, g2


def _conv_kda_oracle(qkv, gate, beta, initial_conv, initial_ssm, conv_weight):
    seq_len = qkv.shape[0]
    state_len = conv_weight.shape[1] - 1
    conv_input = torch.cat([initial_conv, qkv], dim=0)
    mixed = torch.zeros(seq_len, conv_weight.shape[0], dtype=qkv.dtype)
    for tap in range(conv_weight.shape[1]):
        mixed += conv_input[tap:tap + seq_len] * conv_weight[:, tap]
    mixed = F.silu(mixed)
    q, k, v = mixed.split(2, dim=-1)
    q = q.reshape(seq_len, 1, 2)
    k = k.reshape(seq_len, 1, 2)
    v = v.reshape(seq_len, 1, 2)

    q = q.float() / torch.sqrt(q.float().square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k.float() / torch.sqrt(k.float().square().sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (2**-0.5)
    state = initial_ssm.float().clone()
    output = torch.empty(seq_len, 1, 2, dtype=torch.float32)
    for token in range(seq_len):
        decayed = state * torch.exp(gate[token].float()).unsqueeze(-2)
        prediction = decayed.matmul(k[token].unsqueeze(-1)).squeeze(-1)
        residual = (v[token].float() - prediction) * beta[token].float()
        state = decayed + torch.outer(residual.reshape(-1), k[token].reshape(-1)).reshape(1, 2, 2)
        output[token] = state.matmul(q[token].unsqueeze(-1)).squeeze(-1)
    return output, conv_input[-state_len:], state, mixed


def _norm_gate_oracle(core, gate, weight):
    core_fp32 = core.float()
    normalized = core_fp32 * torch.rsqrt(core_fp32.square().mean(dim=-1, keepdim=True) + 1e-5)
    return (normalized * weight.float() * torch.sigmoid(gate.float())).to(core.dtype)


def test_prefill_and_decode_share_conv_and_ssm_store_state_with_full_oracle():
    model, conv_before, ssm_before = _make_tiny_kda_model()
    conv_before = conv_before.clone()
    ssm_before = ssm_before.clone()
    conv_pool, ssm_pool = model.kv_cache
    model._build_merged_conv_weight()
    conv_weight = model._merged_conv_weight

    prefill_hidden = torch.tensor(
        [
            [0.2, -0.3, 0.5],
            [0.1, 0.4, -0.2],
            [-0.5, 0.7, 0.3],
            [0.6, -0.1, 0.2],
            [float("nan"), float("nan"), float("nan")],
            [float("nan"), float("nan"), float("nan")],
            [float("nan"), float("nan"), float("nan")],
            [float("nan"), float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    prefill_metadata = SimpleNamespace(
        is_prompt=True,
        num_prefills=2,
        load_indices_tensor=torch.tensor([[0, 0, 0, 0], [-1, 2, -1, -1]], dtype=torch.int32),
        store_indices_tensor=torch.tensor([[0, 0, 0, 0], [3, 1, -1, -1]], dtype=torch.int32),
        query_start_loc_p=torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
        has_initial_states_p=torch.tensor([0, 1, 0, 0], dtype=torch.int32),
        padding_mask_flat=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]).view(-1, 1),
    )
    context = SimpleNamespace(attn_metadata=prefill_metadata)

    with patch("vllm_gaudi.models.glm5next_kda.get_forward_context", return_value=context):
        prefill_output = model(prefill_hidden, positions=None)

    qkv, gate, beta, g2 = _project_kda(model, prefill_hidden)
    lane_outputs = []
    lane_conv_states = []
    lane_ssm_states = []
    for lane, start in enumerate((0, 2)):
        load_slot = prefill_metadata.load_indices_tensor[1, lane].item()
        initial_conv = conv_before[load_slot] if load_slot >= 0 else torch.zeros_like(conv_before[0])
        initial_ssm = ssm_before[load_slot] if load_slot >= 0 else torch.zeros_like(ssm_before[0])
        lane_out, lane_conv, lane_ssm, _ = _conv_kda_oracle(
            qkv[start:start + 2],
            gate[start:start + 2],
            beta[start:start + 2],
            initial_conv,
            initial_ssm,
            conv_weight,
        )
        lane_outputs.append(lane_out)
        lane_conv_states.append(lane_conv)
        lane_ssm_states.append(lane_ssm)

    active_core = torch.cat([_norm_gate_oracle(out, g2[start:start + 2], model.o_norm.weight)
                             for out, start in zip(lane_outputs, (0, 2))])
    expected_prefill = torch.zeros(8, 3, dtype=torch.float32)
    expected_prefill[:4] = F.linear(active_core.reshape(4, 2), model.o_proj.weight)
    torch.testing.assert_close(prefill_output, expected_prefill, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(conv_pool[3], lane_conv_states[0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(conv_pool[1], lane_conv_states[1], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(ssm_pool[3], lane_ssm_states[0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(ssm_pool[1], lane_ssm_states[1], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(conv_pool[0], conv_before[0], atol=0, rtol=0)
    torch.testing.assert_close(conv_pool[2], conv_before[2], atol=0, rtol=0)
    torch.testing.assert_close(conv_pool[-1], conv_before[-1], atol=0, rtol=0)
    torch.testing.assert_close(ssm_pool[0], ssm_before[0], atol=0, rtol=0)
    torch.testing.assert_close(ssm_pool[2], ssm_before[2], atol=0, rtol=0)
    torch.testing.assert_close(ssm_pool[-1], ssm_before[-1], atol=0, rtol=0)

    conv_after_prefill = conv_pool.clone()
    ssm_after_prefill = ssm_pool.clone()
    decode_hidden = torch.tensor(
        [
            [0.3, 0.2, -0.4],
            [-0.2, 0.5, 0.1],
            [float("nan"), float("nan"), float("nan")],
            [float("nan"), float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )
    context.attn_metadata = SimpleNamespace(
        is_prompt=False,
        load_indices_tensor=torch.tensor([[0, 0, 0, 0], [3, 1, -1, -1]], dtype=torch.int32),
        store_indices_tensor=torch.tensor([[0, 0, 0, 0], [3, 1, -1, -1]], dtype=torch.int32),
        query_start_loc_p=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
    )
    with patch("vllm_gaudi.models.glm5next_kda.get_forward_context", return_value=context):
        decode_output = model(decode_hidden, positions=None)

    qkv_decode, gate_decode, beta_decode, g2_decode = _project_kda(model, decode_hidden[:2])
    decode_cores = []
    decode_conv_states = []
    decode_ssm_states = []
    for lane, slot in enumerate((3, 1)):
        lane_out, lane_conv, lane_ssm, _ = _conv_kda_oracle(
            qkv_decode[lane:lane + 1],
            gate_decode[lane:lane + 1],
            beta_decode[lane:lane + 1],
            conv_after_prefill[slot],
            ssm_after_prefill[slot],
            conv_weight,
        )
        decode_cores.append(_norm_gate_oracle(lane_out, g2_decode[lane:lane + 1], model.o_norm.weight))
        decode_conv_states.append(lane_conv)
        decode_ssm_states.append(lane_ssm)

    expected_decode = torch.zeros(4, 3, dtype=torch.float32)
    expected_decode[:2] = F.linear(torch.cat(decode_cores).reshape(2, 2), model.o_proj.weight)
    torch.testing.assert_close(decode_output, expected_decode, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(conv_pool[3], decode_conv_states[0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(conv_pool[1], decode_conv_states[1], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(ssm_pool[3], decode_ssm_states[0], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(ssm_pool[1], decode_ssm_states[1], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(conv_pool[0], conv_after_prefill[0], atol=0, rtol=0)
    torch.testing.assert_close(conv_pool[2], conv_after_prefill[2], atol=0, rtol=0)
    torch.testing.assert_close(ssm_pool[0], ssm_after_prefill[0], atol=0, rtol=0)
    torch.testing.assert_close(ssm_pool[2], ssm_after_prefill[2], atol=0, rtol=0)
