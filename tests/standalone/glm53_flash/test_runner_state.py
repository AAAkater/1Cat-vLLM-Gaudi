# SPDX-License-Identifier: Apache-2.0
"""CPU-only structural tests for the GLM-5.3 HPU runner wiring."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MLAAttentionSpec, MambaSpec
from vllm_gaudi.models.glm5next_kda import (
    HPUGlm5NextLinearAttention,
    _resolve_state_indices,
)
from vllm_gaudi.v1.worker.hpu_model_runner import (
    _needs_compile_validation_warmup,
    _zero_compact_gdn_slot,
    apply_model_specific_patches,
    maybe_set_mamba_kv_cache_groups_ids,
)


class _FakeKDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache_group_idx = None
        self.compact_state_group_offset = None
        self.compact_state_group_count = None
        self.anchor = nn.Parameter(torch.zeros(1))


class _FakeMLA(nn.Module):
    pass


class _FakeDecoderLayer(nn.Module):
    def __init__(self, *, kda: bool):
        super().__init__()
        self.self_attn = _FakeKDA() if kda else _FakeMLA()


def _gdn_spec() -> MambaSpec:
    return MambaSpec(
        shapes=((4, 3), (2, 2)),
        dtypes=(torch.float32, torch.float32),
        block_size=16,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


def _mla_spec(*, tokens_per_state: int = 1) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        tokens_per_state=tokens_per_state,
    )


def test_glm_kda_cache_groups_use_self_attn_name_and_group_rows():
    layers = [_FakeDecoderLayer(kda=False) for _ in range(5)]
    for layer_idx in (0, 2, 4):
        layers[layer_idx] = _FakeDecoderLayer(kda=True)
    model = nn.Module()
    model.config = SimpleNamespace(
        model_type="glm5_next",
        architectures=["Glm5NextForConditionalGeneration"],
    )
    model.language_model = SimpleNamespace(model=SimpleNamespace(layers=layers))

    groups = [
        SimpleNamespace(layer_names=["language_model.model.layers.1.self_attn.attn"],
                        kv_cache_spec=_mla_spec()),
        SimpleNamespace(layer_names=["language_model.model.layers.0.self_attn"], kv_cache_spec=_gdn_spec()),
        SimpleNamespace(layer_names=["language_model.model.layers.3.self_attn.attn"],
                        kv_cache_spec=_mla_spec()),
        SimpleNamespace(layer_names=["language_model.model.layers.2.self_attn"], kv_cache_spec=_gdn_spec()),
        SimpleNamespace(layer_names=["language_model.model.layers.4.self_attn"], kv_cache_spec=_gdn_spec()),
    ]

    maybe_set_mamba_kv_cache_groups_ids(model, SimpleNamespace(kv_cache_groups=groups))

    for layer_idx, expected_group, expected_offset in ((0, 1, 0), (2, 3, 1), (4, 4, 2)):
        attn = layers[layer_idx].self_attn
        assert attn.cache_group_idx.tolist() == expected_group
        assert attn.cache_group_idx.device.type == "cpu"
        assert attn.compact_state_group_offset == expected_offset
        assert attn.compact_state_group_count == 3
    assert not hasattr(layers[1].self_attn, "cache_group_idx")
    assert not hasattr(layers[3].self_attn, "cache_group_idx")


def test_kda_state_row_resolution_never_falls_back_to_row_zero():
    metadata = SimpleNamespace(load_indices_tensor=torch.tensor([[10, 11], [20, 21]]))
    owner = SimpleNamespace(cache_group_idx=None)

    with pytest.raises(RuntimeError, match="first-row"):
        _resolve_state_indices(owner, metadata, "load_indices_tensor")

    owner.cache_group_idx = torch.tensor(1)
    resolved = _resolve_state_indices(owner, metadata, "load_indices_tensor")
    assert resolved.tolist() == [20, 21]


def test_glm_post_load_initializes_merged_conv_before_warmup():
    layer = object.__new__(HPUGlm5NextLinearAttention)
    nn.Module.__init__(layer)
    layer._merged_conv_weight = None
    for name, start in (("q_conv1d", 0), ("k_conv1d", 12), ("v_conv1d", 24)):
        holder = nn.Module()
        holder.weight = nn.Parameter(torch.arange(start, start + 12, dtype=torch.float32).reshape(4, 1, 3))
        setattr(layer, name, holder)

    model = nn.Module()
    model.add_module("kda", layer)
    hf_config = SimpleNamespace(
        model_type="glm5_next_text",
        architectures=["Glm5NextForCausalLM"],
    )
    model.config = hf_config
    runner = SimpleNamespace(
        model=model,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config)),
    )

    apply_model_specific_patches(runner)

    assert layer._merged_conv_weight is not None
    assert layer._merged_conv_weight.shape == (12, 3)
    assert torch.equal(layer._merged_conv_weight[:4], layer.q_conv1d.weight[:, 0])
    assert runner._has_heterogeneous_layers
    assert _needs_compile_validation_warmup("glm5_next")
    assert _needs_compile_validation_warmup("glm5_next_text")


def test_zero_compact_gdn_slot_clears_only_target_rows_and_sentinels():
    groups, max_reqs, slots_per_req = 2, 4, 2
    num_rows = groups * max_reqs * slots_per_req + 2
    states = [
        torch.arange(num_rows * width, dtype=torch.float32).reshape(num_rows, width) + 1
        for width in (2, 3)
    ]
    before = [state.clone() for state in states]
    target_base_slot = 2

    _zero_compact_gdn_slot(states, target_base_slot, groups, max_reqs, slots_per_req)

    cleared = {0, num_rows - 1}
    for group_offset in range(groups):
        start = (group_offset * max_reqs + target_base_slot) * slots_per_req + 1
        cleared.update(range(start, start + slots_per_req))
    for state, original in zip(states, before):
        assert torch.count_nonzero(state[list(cleared)]) == 0
        keep = torch.ones(num_rows, dtype=torch.bool)
        keep[list(cleared)] = False
        assert torch.equal(state[keep], original[keep])
