# SPDX-License-Identifier: Apache-2.0
"""Compiled attention traces must not specialize on the diagnostic layer."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from vllm_gaudi.ops import deepseek_v41_paged_attention as attention_module


def test_trace_layer_returns_real_layer_outside_compilation(monkeypatch):
    monkeypatch.setattr(attention_module.torch.compiler, "is_compiling", lambda: False)

    assert attention_module._trace_layer(SimpleNamespace(layer=37)) == 37


def test_trace_layer_returns_none_while_compiling(monkeypatch):
    monkeypatch.setattr(attention_module.torch.compiler, "is_compiling", lambda: True)

    assert attention_module._trace_layer(SimpleNamespace(layer=37)) is None


def test_project_query_trace_omits_static_layer_while_compiling(monkeypatch):
    monkeypatch.setattr(attention_module.gaudi_envs, "VLLM_HPU_DSV41_PREFILL_Q_PROJECTION", True)
    monkeypatch.setattr(attention_module.torch.compiler, "is_compiling", lambda: True)

    spans = []
    monkeypatch.setattr(
        attention_module,
        "prefill_event_span",
        lambda *args: spans.append(args) or nullcontext(),
    )
    marker = torch.ones((7, 8, 512))
    monkeypatch.setattr(attention_module, "prefill_q_projection", lambda *_args: marker)
    weight = SimpleNamespace(
        weight=torch.empty((4096, 4), dtype=torch.bfloat16),
        channel_scale=torch.ones(4096),
        dense_fp8=True,
    )
    attention = SimpleNamespace(
        layer=37,
        heads=8,
        draft_query_fp8=False,
        dspark_q_bf16_rope=False,
        q_scale_rope=False,
        c6_query_fusion=False,
        weights=SimpleNamespace(wq_b=weight),
        _rotary_native_table=lambda: None,
    )

    actual = attention_module.PagedCSA2Attention.project_query(
        attention,
        torch.zeros((7, 4), dtype=torch.bfloat16),
        torch.arange(7, dtype=torch.int32),
        request_batch=True,
    )

    assert actual is marker
    assert spans == [
        ("attention_query_quantization", None, 7),
        ("attention_query_projection_rope", None, 7),
    ]
