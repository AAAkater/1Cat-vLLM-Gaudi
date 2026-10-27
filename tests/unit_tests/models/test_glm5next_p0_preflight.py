# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the GLM-5.3-Flash P0 preflight."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KpoolTailSpec,
    MambaAttentionBackendEnum,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_gaudi.models.glm5next_preflight import (
    _validate_checkpoint_index,
    _validate_quantized_header,
    _validate_text_config,
    _validate_weight_layers,
)
from vllm_gaudi.patches import _hpu_get_num_layers_by_block_type


def _layer_types() -> list[str]:
    return ["deepseek_sparse_attention" if index % 4 == 3 else "linear_attention" for index in range(45)]


def _text_config() -> dict:
    return {
        "model_type": "glm5_next_text",
        "num_hidden_layers": 45,
        "num_nextn_predict_layers": 1,
        "layer_types": _layer_types(),
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
        "linear_attn_config": {
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": [index for index in range(45) if index % 4 != 3],
            "full_attn_layers": [index for index in range(45) if index % 4 == 3],
        },
        "indexer_types": ["full"] * 45,
        "index_topk": 2048,
        "index_kpool": 4,
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "n_routed_experts": 288,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "swiglu_limit": 10.0,
    }


class _FakeModelConfig:
    is_hybrid = True
    has_noops = False
    is_attention_free = False
    model_arch_config = SimpleNamespace(text_model_type="glm5_next")

    def __init__(self, text_config):
        self.hf_text_config = SimpleNamespace(**text_config)
        self.hf_config = SimpleNamespace()

    def get_layers_start_end_indices(self, _parallel_config):
        return 0, 45

    def get_num_layers(self, _parallel_config):
        return 45


def test_sparse_attention_layer_types_use_attention_alias():
    raw = {"layer_types": _layer_types()}
    normalized = {
        "layers_block_type": [
            "linear_attention" if value == "linear_attention" else "attention" for value in _layer_types()
        ]
    }

    for text_config, expected in (
        (raw, {"attention": 11, "linear_attention": 34, "mamba": 34}),
        (normalized, {"attention": 11, "linear_attention": 34, "mamba": 34}),
    ):
        model_config = _FakeModelConfig(text_config)
        actual = {
            block_type: _hpu_get_num_layers_by_block_type(model_config, None, block_type)
            for block_type in expected
        }
        assert actual == expected


def test_real_glm53_config_shape_passes():
    errors = []
    layout = _validate_text_config(_text_config(), errors)

    assert errors == []
    assert layout["layer_counts"] == {"linear_attention": 34, "attention": 11}
    assert layout["mlp_layer_counts"] == {"dense": 3, "sparse": 42}
    assert layout["dsa_layers"] == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
    assert layout["linear_attention_layers"][-1] == 44


def test_layer_44_mismatch_is_reported_at_the_exact_index():
    text_config = _text_config()
    text_config["layer_types"][44] = "deepseek_sparse_attention"
    errors = []

    _validate_text_config(text_config, errors)

    assert any(
        issue["code"] == "config.layer_types.value"
        and "layer 44" in issue["message"]
        and "linear_attention" in issue["message"]
        for issue in errors
    )


def test_checkpoint_requires_base_layer_44_and_separate_mtp_layer_45():
    weight_map = {}
    mtp_markers = ("eh_proj.weight", "enorm.weight", "hnorm.weight")
    for layer in range(45):
        if layer != 44:
            weight_map[f"model.language_model.layers.{layer}.input_layernorm.weight"] = "shard.safetensors"
    for marker in mtp_markers:
        weight_map[f"model.language_model.layers.45.{marker}"] = "shard.safetensors"
    errors = []

    summary = _validate_weight_layers(weight_map, mtp_layers=1, errors=errors)

    assert summary["mtp_layer_indices"] == [45]
    assert summary["missing_base_layers"] == [44]
    assert any(
        issue["code"] == "checkpoint.base_layer_missing" and "layer 44" in issue["message"] for issue in errors
    )


def test_dtype_and_scale_mismatches_fail_clearly():
    key = "model.language_model.layers.45.mlp.experts.0.down_proj.weight"
    scale_key = f"{key}_scale_inv"
    weight_map = {key: "shard.safetensors", scale_key: "shard.safetensors"}
    header = {
        key: {"dtype": "BF16", "shape": [4096, 2048], "data_offsets": [0, 16777216]},
        scale_key: {"dtype": "BF16", "shape": [16, 32], "data_offsets": [16777216, 16778240]},
    }
    errors = []

    _validate_quantized_header(
        "shard.safetensors",
        header,
        weight_map,
        {"fmt": "e4m3", "weight_block_size": [128, 128]},
        errors,
    )

    codes = {issue["code"] for issue in errors}
    assert "checkpoint.dtype_mismatch" in codes
    assert "checkpoint.scale_dtype_mismatch" in codes
    assert "checkpoint.scale_shape_mismatch" in codes


def test_index_rejects_scale_without_base_weight(tmp_path):
    index = {
        "metadata": {"total_size": 1},
        "weight_map": {"model.language_model.layers.0.proj.weight_scale_inv": "shard.safetensors"},
    }
    errors = []
    warnings = []

    _validate_checkpoint_index(index, 1, tmp_path, errors, warnings)

    assert any(issue["code"] == "checkpoint.scale_base_missing" for issue in errors)
    assert any(issue["code"] == "checkpoint.base_layer_missing" for issue in errors)


def test_minimal_glm_cache_contract_is_recognized():
    specs = {}
    for layer in range(45):
        if layer % 4 == 3:
            specs[f"layers.{layer}.attn"] = MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=384,
                dtype=torch.bfloat16,
            )
            specs[f"layers.{layer}.indexer"] = MLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=132,
                dtype=torch.uint8,
                tokens_per_state=4,
            )
            specs[f"layers.{layer}.tail"] = KpoolTailSpec(
                block_size=4,
                num_kv_heads=2,
                head_size=128,
                head_size_v=0,
                dtype=torch.bfloat16,
                sliding_window=4,
            )
        else:
            specs[f"layers.{layer}.linear_attn"] = MambaSpec(
                block_size=128,
                shapes=((8, 128), (8, 16, 8)),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            )

    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=None),
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 45),
        speculative_config=None,
    )
    groups = get_kv_cache_groups(vllm_config, specs)
    mamba_layers = sum(len(group.layer_names) for group in groups if isinstance(group.kv_cache_spec, MambaSpec))
    tail_layers = sum(len([name for name in group.layer_names if name.endswith(".tail")]) for group in groups)
    assert mamba_layers == 34
    assert tail_layers == 11
    assert isinstance(specs["layers.3.attn"], FullAttentionSpec)

    cache_config = get_kv_cache_config_from_groups(vllm_config, groups, available_memory=100_000_000)
    tensors = {tensor.layers[0]: tensor for tensor in cache_config.kv_cache_tensors}
    mla_page = specs["layers.3.attn"].page_size_bytes
    indexer_page = specs["layers.3.indexer"].page_size_bytes
    expected_bytes_per_block = 11 * mla_page + 11 * indexer_page
    assert len(cache_config.kv_cache_tensors) == 67
    assert {tensor.size for tensor in cache_config.kv_cache_tensors} == {
        expected_bytes_per_block * cache_config.num_blocks
    }
    assert tensors["layers.3.tail"].offset == tensors["layers.3.indexer"].offset
    assert tensors["layers.0.linear_attn"].offset == tensors["layers.3.attn"].offset


def test_cli_reports_tiny_model_failures_as_json(tmp_path):
    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next"}),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts" / "preflight.json"
    repo_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "check_glm53_flash.py"),
            str(model_dir),
            "--quiet",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    assert result.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    codes = {issue["code"] for issue in report["errors"]}
    assert report["ok"] is False
    assert "config.text_config" in codes
    assert "checkpoint.index_missing" in codes
    assert "model_dir_missing" not in codes
