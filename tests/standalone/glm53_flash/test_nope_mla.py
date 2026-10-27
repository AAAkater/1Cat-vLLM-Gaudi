# SPDX-License-Identifier: Apache-2.0
"""CPU-only NoPE MLA compatibility tests for GLM-5.3 on HPU.

The standalone location avoids the HPU test fixtures.  These tests exercise the
real shape/projection paths with small CPU tensors and an independent attention
oracle; they do not initialize the HPU platform or acquire a device.
"""

from types import SimpleNamespace

import torch
from torch import nn

from vllm_gaudi.attention import oot_mla
from vllm_gaudi.attention.backends import hpu_attn


class _TupleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.last_input = None

    def forward(self, x: torch.Tensor):
        self.last_input = x
        return self.linear(x), None


class _ScaledNorm(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x * self.scale


class _FakeLatentCache:
    def __init__(self, past: torch.Tensor):
        self.past = past
        self.cache_shape = None

    def fetch_from_cache(self, cache: torch.Tensor, block_list: torch.Tensor) -> torch.Tensor:
        del block_list
        self.cache_shape = cache.shape
        return self.past


class _CaptureMLAAttn(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.q = None
        self.kv_c = None
        self.k_pe = None

    def forward(self, q, kv_c, k_pe, output_shape=None, **kwargs):
        del kwargs
        self.q = q
        self.kv_c = kv_c
        self.k_pe = k_pe
        return q.new_zeros(output_shape)


class _Rotary(nn.Module):
    def __init__(self, fail_if_called: bool = False):
        super().__init__()
        self.fail_if_called = fail_if_called
        self.calls = 0

    def forward(self, positions, q_pe, k_pe):
        del positions
        self.calls += 1
        if self.fail_if_called:
            raise AssertionError("NoPE must not invoke rotary embeddings")
        return q_pe + 1, k_pe + 2


def _causal_attention_oracle(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    valid_seq_lengths: torch.Tensor | None,
    **_kwargs,
) -> torch.Tensor:
    query = query.float()
    key = key.float()
    value = value.float()
    output = torch.zeros_like(value)
    for batch_idx in range(query.shape[0]):
        length = int(valid_seq_lengths[batch_idx]) if valid_seq_lengths is not None else query.shape[1]
        logits = torch.einsum("shd,thd->hst", query[batch_idx, :length], key[batch_idx, :length]) * scale
        if is_causal:
            causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
            logits = logits.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        output[batch_idx, :length] = torch.einsum("hst,thd->shd", weights, value[batch_idx, :length])
    return output.to(query.dtype)


def _make_impl(
    *,
    qk_nope_head_dim: int = 4,
    qk_rope_head_dim: int = 0,
    v_head_dim: int = 4,
    kv_lora_rank: int = 3,
    num_heads: int = 2,
) -> hpu_attn.HPUMLAImpl:
    impl = hpu_attn.HPUMLAImpl.__new__(hpu_attn.HPUMLAImpl)
    nn.Module.__init__(impl)
    impl.num_heads = num_heads
    impl.head_size = qk_nope_head_dim + qk_rope_head_dim
    impl.scale = 0.5
    impl.num_kv_heads = 1
    impl.qk_nope_head_dim = qk_nope_head_dim
    impl.qk_rope_head_dim = qk_rope_head_dim
    impl.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    impl.v_head_dim = v_head_dim
    impl.kv_lora_rank = kv_lora_rank
    impl.kv_b_proj = _TupleLinear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
    impl.use_merged_prefill = False
    impl.prefill_impl = "oracle"
    impl.matmul_qk = None
    impl.softmax = None
    impl.matmul_av = None
    impl.batch2block_matmul = None
    impl.block2batch_matmul = None
    impl.latent_cache_k = SimpleNamespace(fetch_from_cache=lambda *args, **kwargs: None)
    impl.fused_scaled_dot_product_attention = None
    return impl


def _prefill_metadata(
    seq_lens: list[int],
    *,
    block_list: torch.Tensor | None = None,
    block_size: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        block_list=block_list,
        block_size=block_size,
        seq_lens_tensor=torch.tensor(seq_lens),
        causal=True,
        attn_bias=None,
    )


def test_nope_prefill_with_prefix_cache_matches_attention_oracle(monkeypatch):
    impl = _make_impl()
    heads, nope_dim, v_dim, kv_lora = impl.num_heads, impl.qk_nope_head_dim, impl.v_head_dim, impl.kv_lora_rank
    q = torch.randn(6, heads * nope_dim)
    current_latent = torch.randn(4, kv_lora)
    past_latent = torch.randn(2, kv_lora)
    cache = _FakeLatentCache(past_latent)
    impl.latent_cache_k = cache
    kv_cache = torch.empty(4, kv_lora)
    metadata = _prefill_metadata([3, 3], block_list=torch.tensor([0]), block_size=2)
    q_before = q.clone()
    current_before = current_latent.clone()
    calls = []

    def oracle(**kwargs):
        calls.append(kwargs)
        return _causal_attention_oracle(**kwargs)

    monkeypatch.setattr(hpu_attn.ops, "prompt_attention", oracle)

    output = impl.forward_mha(q, current_latent, kv_cache, metadata)

    combined_latent = torch.cat((past_latent, current_latent))
    kv = impl.kv_b_proj(combined_latent)[0].view(-1, heads, nope_dim + v_dim)
    k_nope, v = kv.split([nope_dim, v_dim], dim=-1)
    expected = _causal_attention_oracle(
        query=q.view(2, 3, heads, nope_dim),
        key=k_nope.view(2, 3, heads, nope_dim),
        value=v.view(2, 3, heads, v_dim),
        scale=impl.scale,
        is_causal=True,
        valid_seq_lengths=torch.tensor([3, 3]),
    ).reshape(-1, heads * v_dim)

    torch.testing.assert_close(output, expected)
    assert cache.cache_shape == (2, 2, kv_lora)
    assert calls[0]["query"].shape == (2, 3, heads, nope_dim)
    assert calls[0]["key"].shape[-1] == nope_dim
    torch.testing.assert_close(q, q_before)
    torch.testing.assert_close(current_latent, current_before)


def test_rope_prefill_regression_matches_attention_oracle(monkeypatch):
    impl = _make_impl(qk_rope_head_dim=2, v_head_dim=4)
    heads = impl.num_heads
    nope_dim, rope_dim, v_dim, kv_lora = (
        impl.qk_nope_head_dim,
        impl.qk_rope_head_dim,
        impl.v_head_dim,
        impl.kv_lora_rank,
    )
    q = torch.randn(4, heads * impl.qk_head_dim)
    latent = torch.randn(4, kv_lora + rope_dim)
    metadata = _prefill_metadata([2, 2])
    calls = []

    def oracle(**kwargs):
        calls.append(kwargs)
        return _causal_attention_oracle(**kwargs)

    monkeypatch.setattr(hpu_attn.ops, "prompt_attention", oracle)

    output = impl.forward_mha(q, latent, None, metadata)

    k_c, k_pe = latent.split([kv_lora, rope_dim], dim=-1)
    kv = impl.kv_b_proj(k_c)[0].view(-1, heads, nope_dim + v_dim)
    k_nope, v = kv.split([nope_dim, v_dim], dim=-1)
    key = torch.cat((k_nope, k_pe.view(-1, 1, rope_dim).expand(-1, heads, -1)), dim=-1)
    expected = _causal_attention_oracle(
        query=q.view(2, 2, heads, impl.qk_head_dim),
        key=key.view(2, 2, heads, impl.qk_head_dim),
        value=v.view(2, 2, heads, v_dim),
        scale=impl.scale,
        is_causal=True,
        valid_seq_lengths=torch.tensor([2, 2]),
    ).reshape(-1, heads * v_dim)

    torch.testing.assert_close(output, expected)
    assert calls[0]["query"].shape[-1] == impl.qk_head_dim
    assert calls[0]["key"].shape[-1] == impl.qk_head_dim


def test_nope_decode_uses_nope_query_and_expected_cache_shape(monkeypatch):
    impl = _make_impl(v_head_dim=4)
    heads, nope_dim, kv_lora = impl.num_heads, impl.qk_nope_head_dim, impl.kv_lora_rank
    q_nope = torch.randn(5, heads, nope_dim)
    q_pe = torch.empty(5, heads, 0)
    k_cache = torch.randn(7, kv_lora)
    k_cache_before = k_cache.clone()
    metadata = SimpleNamespace(
        block_list=None,
        block_mapping=None,
        block_groups=None,
        attn_bias=None,
        block_size=2,
    )
    captured = {}

    def fake_forward_decode(**kwargs):
        captured.update(kwargs)
        return kwargs["query"]

    monkeypatch.setattr(hpu_attn.HPUPagedAttention, "forward_decode", fake_forward_decode)
    output = impl.forward_mqa(q_nope, q_pe, k_cache, metadata)

    assert captured["query"].shape == (5, heads, nope_dim)
    assert captured["key_cache"].shape == (7, 1, kv_lora)
    torch.testing.assert_close(output, q_nope)
    torch.testing.assert_close(k_cache, k_cache_before)


def test_attention_forward_impl_nope_avoids_empty_split_and_concat():
    attention = oot_mla.HPUMLAAttention.__new__(oot_mla.HPUMLAAttention)
    nn.Module.__init__(attention)
    attention.qk_nope_head_dim = 4
    attention.qk_rope_head_dim = 0
    attention.kv_lora_rank = 3
    attention.num_heads = 2
    attention.v_head_dim = 2
    attention.W_UK_T = torch.randn(attention.num_heads, 4, 3)
    attention.W_UV = torch.randn(attention.num_heads, 3, 2)
    captured = {}

    class _Impl:
        def __init__(self):
            self.latent_cache_k = lambda latent, cache, slots: captured.setdefault("cache_latent", latent.clone())

        def forward_mha(self, q, latent, kv_cache, attn_metadata):
            del kv_cache, attn_metadata
            captured["prefill_q"] = q.clone()
            captured["prefill_latent"] = latent.clone()
            return latent.new_ones(latent.shape[0], attention.num_heads * attention.v_head_dim)

        def forward_mqa(self, q_nope, q_pe, kv_cache, attn_metadata):
            del kv_cache, attn_metadata
            captured["decode_q_nope"] = q_nope.clone()
            captured["decode_q_pe"] = q_pe.clone()
            return q_nope.new_ones(q_nope.shape[0], attention.num_heads, attention.kv_lora_rank)

    attention.impl = _Impl()
    q = torch.randn(4, attention.num_heads * attention.qk_nope_head_dim)
    k_c = torch.randn(4, attention.kv_lora_rank)
    k_pe = torch.empty(4, 0)
    kv_cache = (torch.empty(8, attention.kv_lora_rank), torch.empty(0))

    prefill_metadata = SimpleNamespace(is_prompt=True, slot_mapping=torch.arange(4))
    prefill_output = attention.forward_impl(q, k_c, k_pe, kv_cache, prefill_metadata)
    assert captured["cache_latent"].shape == (4, attention.kv_lora_rank)
    assert captured["prefill_latent"].shape == (4, attention.kv_lora_rank)
    assert prefill_output.shape == (4, attention.num_heads * attention.v_head_dim)

    decode_metadata = SimpleNamespace(is_prompt=False, slot_mapping=torch.arange(4))
    decode_q = q.view(4, attention.num_heads, attention.qk_nope_head_dim)
    decode_output = attention.forward_impl(decode_q, k_c, k_pe, kv_cache, decode_metadata)
    assert captured["decode_q_pe"].shape == (4, attention.num_heads, 0)
    assert captured["cache_latent"].shape == (4, attention.kv_lora_rank)
    assert decode_output.shape == (4, attention.num_heads * attention.v_head_dim)


def _make_wrapper(monkeypatch, *, qk_rope_head_dim: int, fuse_qkv_rmsnorm: bool):
    heads, nope_dim, v_dim, q_lora, kv_lora, hidden_size = 2, 4, 2, 3, 5, 4
    qk_head_dim = nope_dim + qk_rope_head_dim
    q_a_norm = _ScaledNorm(2.0)
    kv_a_norm = _ScaledNorm(3.0)
    fused_qkv_a_proj = _TupleLinear(hidden_size, q_lora + kv_lora + qk_rope_head_dim)
    q_b_proj = _TupleLinear(q_lora, heads * qk_head_dim)
    o_proj = _TupleLinear(heads * v_dim, hidden_size)
    rotary = _Rotary(fail_if_called=qk_rope_head_dim == 0)
    modules = SimpleNamespace(
        fused_qkv_a_proj=fused_qkv_a_proj,
        kv_a_proj_with_mqa=None,
        q_a_layernorm=q_a_norm,
        q_b_proj=q_b_proj,
        q_proj=None,
        kv_a_layernorm=kv_a_norm,
        kv_b_proj=None,
        rotary_emb=rotary,
        o_proj=o_proj,
        indexer=None,
        indexer_rotary_emb=None,
        is_sparse=False,
        g_proj=None,
    )
    monkeypatch.setattr(oot_mla, "HPUMLAAttention", _CaptureMLAAttn)
    monkeypatch.setattr(
        oot_mla,
        "get_current_vllm_config",
        lambda: SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={})),
    )
    wrapper = oot_mla.HPUMultiHeadLatentAttentionWrapper(
        hidden_size,
        heads,
        0.5,
        nope_dim,
        qk_rope_head_dim,
        v_dim,
        q_lora,
        kv_lora,
        modules,
        prefix="test",
        fuse_qkv_rmsnorm=fuse_qkv_rmsnorm,
    )
    return wrapper, q_a_norm, kv_a_norm, q_b_proj, rotary


def test_wrapper_nope_accepts_fused_flag_without_double_norm(monkeypatch):
    wrapper, q_a_norm, kv_a_norm, q_b_proj, rotary = _make_wrapper(
        monkeypatch,
        qk_rope_head_dim=0,
        fuse_qkv_rmsnorm=True,
    )
    hidden_states = torch.randn(5, wrapper.hidden_size)

    output = wrapper(torch.arange(5), hidden_states)
    projected = wrapper.fused_qkv_a_proj(hidden_states)[0]
    q_c, kv_c = projected.split([wrapper.q_lora_rank, wrapper.kv_lora_rank], dim=-1)

    assert wrapper._requested_fuse_qkv_rmsnorm is True
    assert wrapper.fuse_qkv_rmsnorm is False
    assert q_a_norm.calls == 1
    assert kv_a_norm.calls == 1
    assert rotary.calls == 0
    torch.testing.assert_close(q_b_proj.last_input, q_c * q_a_norm.scale)
    assert wrapper.mla_attn.q.shape == (5, wrapper.num_heads, wrapper.qk_head_dim)
    assert wrapper.mla_attn.kv_c.shape == (5, wrapper.kv_lora_rank)
    assert wrapper.mla_attn.k_pe.shape == (5, 0)
    assert output.shape == hidden_states.shape


def test_wrapper_rope_regression_uses_upstream_path(monkeypatch):
    wrapper, q_a_norm, kv_a_norm, _q_b_proj, rotary = _make_wrapper(
        monkeypatch,
        qk_rope_head_dim=2,
        fuse_qkv_rmsnorm=True,
    )
    hidden_states = torch.randn(5, wrapper.hidden_size)

    output = wrapper(torch.arange(5), hidden_states)

    assert q_a_norm.calls == 1
    assert kv_a_norm.calls == 1
    assert rotary.calls == 1
    assert wrapper.mla_attn.q.shape == (5, wrapper.num_heads, wrapper.qk_head_dim)
    assert wrapper.mla_attn.k_pe.shape == (5, 1, wrapper.qk_rope_head_dim)
    assert output.shape == hidden_states.shape
