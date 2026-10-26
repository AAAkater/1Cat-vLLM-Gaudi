# SPDX-License-Identifier: Apache-2.0
"""The retained recipe must use current routed inputs and weight storage."""

import os

import torch

from vllm_gaudi.ops import deepseek_v41_c6_recipe as recipe


def test_recipe_reuses_graph_with_new_values_weights_and_lane_offsets(monkeypatch):
    # This CPU contract must not acquire an HPU when devices are serving.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    graphs = []
    original_compile = torch.compile

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def compile_for_cpu(function, **kwargs):
        assert kwargs == dict(backend="hpu_backend", fullgraph=True, dynamic=False)
        return original_compile(function, backend=backend, fullgraph=True, dynamic=False)

    def stand_in(value, weights, routing):
        return value @ weights + routing

    monkeypatch.setattr(torch, "compile", compile_for_cpu)
    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_expert_n256_moe_prequant_fused_fp8_gaudi2",
                        stand_in, raising=False)
    monkeypatch.setenv("PT_HPU_ENABLE_EAGER_CACHE", "0")
    @torch.compiler.disable
    def target_boundary(value, weights, routing):
        return recipe.c6_prequant_moe(value, weights, routing)

    recipe._compiled_prequant_moe.cache_clear()
    try:
        generator = torch.Generator().manual_seed(1947)
        lanes = torch.randn(18, 4, generator=generator)
        weights = torch.randn(4, 3, generator=generator)
        routing = torch.randn(6, 3, generator=generator)
        with torch.no_grad():
            for start in (0, 6, 12):
                weights.add_(0.125)
                routing.add_(0.25)
                value = lanes[start:start + 6]
                actual = target_boundary(value, weights, routing)
                assert torch.equal(actual, stand_in(value, weights, routing))
        assert len(graphs) == 1
        assert recipe._compiled_prequant_moe.cache_info().hits == 2
        assert os.environ["PT_HPU_ENABLE_EAGER_CACHE"] == "0"
    finally:
        recipe._compiled_prequant_moe.cache_clear()
