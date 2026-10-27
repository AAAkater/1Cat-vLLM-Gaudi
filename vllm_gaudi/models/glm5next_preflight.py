# SPDX-License-Identifier: Apache-2.0
"""CPU-only preflight checks for the GLM-5.3-Flash checkpoint.

The module deliberately keeps every vLLM import lazy.  The CLI disables vLLM
plugins and selects the CPU target before calling ``run_preflight`` so the
preflight never initializes Habana runtimes or allocates accelerator memory.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import struct
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "glm53-flash-preflight-v1"
EXPECTED_ARCHITECTURE = "Glm5NextForConditionalGeneration"
EXPECTED_MODEL_TYPE = "glm5_next"
EXPECTED_LAYERS = 45
EXPECTED_ATTENTION_LAYERS = 11
EXPECTED_LINEAR_LAYERS = 34
EXPECTED_SHARDS = 62
EXPECTED_TENSORS = 76108
EXPECTED_TOTAL_BYTES = 328326771576
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_LAYER_ALIASES = {
    "full_attention": "attention",
    "deepseek_sparse_attention": "attention",
    "linear_attention": "mamba",
}


def _add_error(errors: list[dict[str, str]], code: str, message: str) -> None:
    errors.append({"code": code, "message": message})


def _add_warning(warnings: list[dict[str, str]], code: str, message: str) -> None:
    warnings.append({"code": code, "message": message})


def _cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_head(root: Path | None) -> str | None:
    if root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_dirty(root: Path | None) -> bool | None:
    if root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def _source_file_hashes(root: Path | None, relative_paths: tuple[str, ...]) -> dict[str, str | None]:
    if root is None:
        return {relative_path: None for relative_path in relative_paths}
    return {
        relative_path: _sha256_file(root / relative_path) if (root / relative_path).is_file() else None
        for relative_path in relative_paths
    }


def _exception_text(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}"[:2000]


def _derive_layer_layout(text_config: dict[str, Any]) -> dict[str, Any]:
    layer_types = text_config.get("layer_types")
    mlp_layer_types = text_config.get("mlp_layer_types")
    linear_config = text_config.get("linear_attn_config") or {}
    indexer_types = text_config.get("indexer_types")

    normalized_layer_types = None
    if isinstance(layer_types, list):
        normalized_layer_types = [
            "attention" if value in ("full_attention", "deepseek_sparse_attention") else value for value in layer_types
        ]

    dsa_layers = None
    linear_layers = None
    if isinstance(layer_types, list):
        dsa_layers = [index for index, value in enumerate(layer_types) if value == "deepseek_sparse_attention"]
        linear_layers = [index for index, value in enumerate(layer_types) if value == "linear_attention"]

    return {
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_nextn_predict_layers": text_config.get("num_nextn_predict_layers"),
        "layer_types": layer_types,
        "normalized_layer_types": normalized_layer_types,
        "layer_counts": dict(Counter(normalized_layer_types)) if normalized_layer_types is not None else None,
        "dsa_layers": dsa_layers,
        "linear_attention_layers": linear_layers,
        "mlp_layer_types": mlp_layer_types,
        "mlp_layer_counts": dict(Counter(mlp_layer_types)) if isinstance(mlp_layer_types, list) else None,
        "linear_attn_config_kda_layers": linear_config.get("kda_layers"),
        "linear_attn_config_full_attn_layers": linear_config.get("full_attn_layers"),
        "indexer_types": indexer_types,
        "index_topk": text_config.get("index_topk"),
        "index_kpool": text_config.get("index_kpool"),
        "hc_mult": text_config.get("hc_mult"),
        "moe": {
            "n_routed_experts": text_config.get("n_routed_experts"),
            "num_experts_per_tok": text_config.get("num_experts_per_tok"),
            "n_shared_experts": text_config.get("n_shared_experts"),
            "swiglu_limit": text_config.get("swiglu_limit"),
        },
    }


def _validate_text_config(text_config: dict[str, Any], errors: list[dict[str, str]]) -> dict[str, Any]:
    layout = _derive_layer_layout(text_config)
    num_layers = layout["num_hidden_layers"]
    if num_layers != EXPECTED_LAYERS:
        _add_error(
            errors,
            "config.num_hidden_layers",
            f"expected {EXPECTED_LAYERS}, got {num_layers!r}",
        )

    layer_types = layout["layer_types"]
    if not isinstance(layer_types, list):
        _add_error(errors, "config.layer_types", f"expected a list, got {type(layer_types).__name__}")
    elif len(layer_types) != EXPECTED_LAYERS:
        _add_error(
            errors,
            "config.layer_types.length",
            f"expected {EXPECTED_LAYERS} entries, got {len(layer_types)}",
        )
    else:
        expected_layer_types = [
            "deepseek_sparse_attention" if index % 4 == 3 else "linear_attention" for index in range(EXPECTED_LAYERS)
        ]
        mismatches = [
            (index, expected, actual)
            for index, (expected, actual) in enumerate(zip(expected_layer_types, layer_types))
            if expected != actual
        ]
        for index, expected, actual in mismatches[:10]:
            _add_error(
                errors,
                "config.layer_types.value",
                f"layer {index}: expected {expected!r}, got {actual!r}",
            )
        if len(mismatches) > 10:
            _add_error(
                errors,
                "config.layer_types.more_mismatches",
                f"{len(mismatches) - 10} additional layer-type mismatches were not printed",
            )

    mlp_layer_types = layout["mlp_layer_types"]
    expected_mlp = ["dense"] * 3 + ["sparse"] * (EXPECTED_LAYERS - 3)
    if not isinstance(mlp_layer_types, list):
        _add_error(errors, "config.mlp_layer_types", f"expected a list, got {type(mlp_layer_types).__name__}")
    elif mlp_layer_types != expected_mlp:
        _add_error(
            errors,
            "config.mlp_layer_types.value",
            "expected three dense layers followed by 42 sparse layers",
        )

    linear_config = text_config.get("linear_attn_config") or {}
    expected_kda = [index for index in range(EXPECTED_LAYERS) if index % 4 != 3]
    expected_dsa = [index for index in range(EXPECTED_LAYERS) if index % 4 == 3]
    if linear_config.get("kda_layers") != expected_kda:
        _add_error(errors, "config.linear_attn_config.kda_layers", f"expected layer indices {expected_kda}")
    if linear_config.get("full_attn_layers") != expected_dsa:
        _add_error(
            errors,
            "config.linear_attn_config.full_attn_layers",
            f"expected layer indices {expected_dsa}",
        )
    if linear_config.get("num_heads") != 64:
        _add_error(errors, "config.linear_attn_config.num_heads", "expected 64")
    if linear_config.get("head_dim") != 128:
        _add_error(errors, "config.linear_attn_config.head_dim", "expected 128")
    if linear_config.get("short_conv_kernel_size") != 4:
        _add_error(errors, "config.linear_attn_config.short_conv_kernel_size", "expected 4")
    if linear_config.get("gate_lower_bound") != -5.0:
        _add_error(errors, "config.linear_attn_config.gate_lower_bound", "expected -5.0")

    if layout["index_topk"] != 2048:
        _add_error(errors, "config.index_topk", f"expected 2048, got {layout['index_topk']!r}")
    if layout["index_kpool"] != 4:
        _add_error(errors, "config.index_kpool", f"expected 4, got {layout['index_kpool']!r}")
    if text_config.get("index_kpool_compress") is not True:
        _add_error(errors, "config.index_kpool_compress", "expected True")
    if text_config.get("index_kpool_always_select_tail") is not True:
        _add_error(errors, "config.index_kpool_always_select_tail", "expected True")
    if layout["hc_mult"] != 4:
        _add_error(errors, "config.hc_mult", f"expected 4, got {layout['hc_mult']!r}")
    if text_config.get("hc_sinkhorn_iters") != 20:
        _add_error(errors, "config.hc_sinkhorn_iters", "expected 20")

    moe = layout["moe"]
    if moe["n_routed_experts"] != 288:
        _add_error(errors, "config.n_routed_experts", f"expected 288, got {moe['n_routed_experts']!r}")
    if moe["num_experts_per_tok"] != 8:
        _add_error(errors, "config.num_experts_per_tok", f"expected 8, got {moe['num_experts_per_tok']!r}")
    if moe["n_shared_experts"] != 1:
        _add_error(errors, "config.n_shared_experts", f"expected 1, got {moe['n_shared_experts']!r}")
    if moe["swiglu_limit"] != 10.0:
        _add_error(errors, "config.swiglu_limit", f"expected 10.0, got {moe['swiglu_limit']!r}")

    indexer_types = layout["indexer_types"]
    if not isinstance(indexer_types, list) or len(indexer_types) != EXPECTED_LAYERS:
        _add_error(errors, "config.indexer_types.length", f"expected {EXPECTED_LAYERS} entries")
    elif set(indexer_types) != {"full"}:
        _add_error(errors, "config.indexer_types.value", f"expected only 'full', got {sorted(set(indexer_types))}")

    if layout["num_nextn_predict_layers"] != 1:
        _add_error(
            errors,
            "config.num_nextn_predict_layers",
            f"expected one checkpoint MTP layer, got {layout['num_nextn_predict_layers']!r}",
        )

    return layout


def _validate_quant_config(config: dict[str, Any], errors: list[dict[str, str]]) -> dict[str, Any]:
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        _add_error(errors, "config.quantization_config", "expected an FP8 quantization_config object")
        return {"present": False}

    fmt = quant.get("fmt")
    activation_scheme = quant.get("activation_scheme")
    block_size = quant.get("weight_block_size")
    modules_to_not_convert = quant.get("modules_to_not_convert")
    if fmt != "e4m3":
        _add_error(errors, "config.quantization_config.fmt", f"expected 'e4m3', got {fmt!r}")
    if activation_scheme != "dynamic":
        _add_error(
            errors,
            "config.quantization_config.activation_scheme",
            f"expected 'dynamic', got {activation_scheme!r}",
        )
    if block_size != [128, 128]:
        _add_error(
            errors,
            "config.quantization_config.weight_block_size",
            f"expected [128, 128], got {block_size!r}",
        )
    if not isinstance(modules_to_not_convert, list):
        _add_error(
            errors,
            "config.quantization_config.modules_to_not_convert",
            "expected a list",
        )

    return {
        "present": True,
        "fmt": fmt,
        "activation_scheme": activation_scheme,
        "weight_block_size": block_size,
        "modules_to_not_convert_count": (
            len(modules_to_not_convert) if isinstance(modules_to_not_convert, list) else None
        ),
    }


def _validate_weight_layers(
    weight_map: dict[str, str],
    mtp_layers: int,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    by_layer: dict[int, list[str]] = defaultdict(list)
    for key in weight_map:
        match = _LAYER_RE.search(key)
        if match is not None:
            by_layer[int(match.group(1))].append(key)

    missing_base_layers = [index for index in range(EXPECTED_LAYERS) if index not in by_layer]
    for index in missing_base_layers:
        _add_error(errors, "checkpoint.base_layer_missing", f"layer {index} has no tensors in the weight index")

    mtp_layer_indices = sorted(index for index in by_layer if index >= EXPECTED_LAYERS)
    expected_mtp = list(range(EXPECTED_LAYERS, EXPECTED_LAYERS + mtp_layers))
    if mtp_layer_indices != expected_mtp:
        _add_error(
            errors,
            "checkpoint.mtp_layers",
            f"expected MTP layer indices {expected_mtp}, got {mtp_layer_indices}",
        )

    for layer_index in expected_mtp:
        suffix_markers = ("eh_proj.weight", "enorm.weight", "hnorm.weight")
        layer_keys = by_layer.get(layer_index, [])
        for marker in suffix_markers:
            if not any(key.endswith(marker) for key in layer_keys):
                _add_error(
                    errors,
                    "checkpoint.mtp_signature_missing",
                    f"layer {layer_index} is missing {marker}; it must be classified as MTP data",
                )

    return {
        "base_layers_present": [index for index in range(EXPECTED_LAYERS) if index in by_layer],
        "missing_base_layers": missing_base_layers,
        "mtp_layer_indices": mtp_layer_indices,
        "tensor_counts_by_layer": {str(index): len(by_layer[index]) for index in sorted(by_layer)},
    }


def _validate_checkpoint_index(
    index: dict[str, Any],
    mtp_layers: int,
    model_dir: Path,
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict):
        _add_error(errors, "checkpoint.metadata", "model.safetensors.index.json has no metadata object")
        metadata = {}
    if not isinstance(weight_map, dict) or not all(isinstance(key, str) and isinstance(value, str)
                                                   for key, value in weight_map.items()):
        _add_error(errors, "checkpoint.weight_map", "expected a string-to-string weight_map")
        weight_map = {}

    total_size = metadata.get("total_size")
    if total_size != EXPECTED_TOTAL_BYTES:
        _add_error(
            errors,
            "checkpoint.total_size",
            f"expected {EXPECTED_TOTAL_BYTES}, got {total_size!r}",
        )
    if len(weight_map) != EXPECTED_TENSORS:
        _add_error(
            errors,
            "checkpoint.tensor_count",
            f"expected {EXPECTED_TENSORS} tensors, got {len(weight_map)}",
        )

    shards = sorted(set(weight_map.values()))
    if len(shards) != EXPECTED_SHARDS:
        _add_error(errors, "checkpoint.shard_count", f"expected {EXPECTED_SHARDS} shards, got {len(shards)}")

    missing_shards: list[str] = []
    shard_sizes: dict[str, int] = {}
    for shard in shards:
        shard_path = model_dir / shard
        if Path(shard).is_absolute() or ".." in Path(shard).parts:
            _add_error(errors, "checkpoint.invalid_shard_path", f"unsafe shard path {shard!r}")
            continue
        if not shard_path.is_file():
            missing_shards.append(shard)
            continue
        shard_sizes[shard] = shard_path.stat().st_size
    for shard in missing_shards:
        _add_error(errors, "checkpoint.shard_missing", f"missing shard {shard}")

    scale_keys = [key for key in weight_map if key.endswith("_scale_inv")]
    missing_scale_bases = [key for key in scale_keys if key[: -len("_scale_inv")] not in weight_map]
    for key in missing_scale_bases[:10]:
        _add_error(errors, "checkpoint.scale_base_missing", f"{key} has no matching base weight")
    if len(missing_scale_bases) > 10:
        _add_error(
            errors,
            "checkpoint.scale_base_missing_more",
            f"{len(missing_scale_bases) - 10} additional scale tensors have no base weight",
        )

    split_scale_pairs = 0
    for scale_key in scale_keys:
        weight_key = scale_key[: -len("_scale_inv")]
        if weight_key in weight_map and weight_map[weight_key] != weight_map[scale_key]:
            split_scale_pairs += 1
    if split_scale_pairs:
        _add_warning(
            warnings,
            "checkpoint.scale_pair_split_shard",
            f"{split_scale_pairs} weight/scale pairs are stored in different shards",
        )

    layer_summary = _validate_weight_layers(weight_map, mtp_layers, errors)
    if "lm_head.weight" not in weight_map:
        _add_error(errors, "checkpoint.lm_head_missing", "lm_head.weight is absent from the index")
    lm_head_shard = weight_map.get("lm_head.weight")

    visual_keys = [key for key in weight_map if key.startswith("model.visual.") or key.startswith("visual.")]
    return {
        "total_size": total_size,
        "tensor_count": len(weight_map),
        "shard_count": len(shards),
        "shards": shards,
        "shard_sizes": shard_sizes,
        "missing_shards": missing_shards,
        "scale_tensor_count": len(scale_keys),
        "scale_pairs_missing_base": len(missing_scale_bases),
        "scale_pairs_split_across_shards": split_scale_pairs,
        "lm_head_shard": lm_head_shard,
        "visual_tensor_count": len(visual_keys),
        "layers": layer_summary,
    }


def _read_safetensors_header(
    path: Path,
    errors: list[dict[str, str]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    file_size = path.stat().st_size
    info: dict[str, Any] = {"path": str(path), "file_size": file_size}
    try:
        with path.open("rb") as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise ValueError("truncated eight-byte header length")
            header_length = struct.unpack("<Q", length_bytes)[0]
            if header_length <= 0 or header_length > MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError(f"invalid header length {header_length}")
            header_bytes = handle.read(header_length)
    except (OSError, ValueError) as exc:
        _add_error(errors, "checkpoint.header_read", f"{path.name}: {_exception_text(exc)}")
        return None, info

    if len(header_bytes) != header_length:
        _add_error(errors, "checkpoint.header_truncated", f"{path.name}: truncated safetensors header")
        return None, info
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        _add_error(errors, "checkpoint.header_json", f"{path.name}: {_exception_text(exc)}")
        return None, info
    if not isinstance(header, dict):
        _add_error(errors, "checkpoint.header_shape", f"{path.name}: header is not a JSON object")
        return None, info

    payload_bytes = file_size - 8 - header_length
    max_end = 0
    tensor_count = 0
    dtype_counts: Counter[str] = Counter()
    for key, descriptor in header.items():
        if key == "__metadata__":
            continue
        tensor_count += 1
        if not isinstance(descriptor, dict):
            _add_error(errors, "checkpoint.header_tensor", f"{path.name}:{key}: descriptor is not an object")
            continue
        dtype_counts[str(descriptor.get("dtype"))] += 1
        offsets = descriptor.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(item, int) for item in offsets):
            _add_error(errors, "checkpoint.header_offsets", f"{path.name}:{key}: invalid data_offsets")
            continue
        start, end = offsets
        if start < 0 or end < start:
            _add_error(errors, "checkpoint.header_offsets", f"{path.name}:{key}: invalid offsets {offsets}")
        max_end = max(max_end, end)

    if max_end > payload_bytes:
        _add_error(
            errors,
            "checkpoint.truncated_payload",
            f"{path.name}: header references {max_end} payload bytes, file has {payload_bytes}",
        )
    info.update(
        header_bytes=header_length,
        payload_bytes=payload_bytes,
        tensor_count=tensor_count,
        dtypes=dict(sorted(dtype_counts.items())),
    )
    return header, info


def _sample_header_shards(weight_map: dict[str, str], limit: int) -> list[str]:
    shards = sorted(set(weight_map.values()))
    if not shards or limit <= 0:
        return []

    priority: list[str] = []
    for layer_index in (44, 45, 3):
        layer_keys = sorted(
            key for key in weight_map if _LAYER_RE.search(key) and int(_LAYER_RE.search(key).group(1)) == layer_index
        )
        quantized_keys = [key for key in layer_keys if key.endswith(".weight") and f"{key}_scale_inv" in weight_map]
        for key in quantized_keys[:1]:
            priority.append(weight_map[key])
            priority.append(weight_map[f"{key}_scale_inv"])
    if "lm_head.weight" in weight_map:
        priority.append(weight_map["lm_head.weight"])
    priority.extend((shards[0], shards[len(shards) // 2], shards[-1]))

    selected: list[str] = []
    for shard in priority + shards:
        if shard not in selected:
            selected.append(shard)
        if len(selected) >= limit:
            break
    return selected


def _validate_quantized_header(
    shard: str,
    header: dict[str, Any],
    weight_map: dict[str, str],
    quant_config: dict[str, Any],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    fmt = str(quant_config.get("fmt") or "").lower()
    expected_dtype = {"e4m3": "F8_E4M3"}.get(fmt)
    block = quant_config.get("weight_block_size")
    block_shape = tuple(block) if isinstance(block, list) and len(block) == 2 else None
    fp8_count = 0
    checked_pairs = 0
    for key, descriptor in header.items():
        if key == "__metadata__" or not key.endswith(".weight") or not isinstance(descriptor, dict):
            continue
        scale_key = f"{key}_scale_inv"
        dtype = descriptor.get("dtype")
        has_scale = scale_key in weight_map
        if dtype == "F8_E4M3":
            fp8_count += 1
            if not has_scale:
                _add_error(errors, "checkpoint.scale_missing", f"{shard}:{key}: FP8 weight has no weight_scale_inv")
                continue
        elif has_scale:
            _add_error(
                errors,
                "checkpoint.dtype_mismatch",
                f"{shard}:{key}: expected {expected_dtype or 'FP8'} because a scale exists, got {dtype}",
            )

        if not has_scale or scale_key not in header:
            continue
        scale_descriptor = header[scale_key]
        if not isinstance(scale_descriptor, dict):
            continue
        checked_pairs += 1
        if scale_descriptor.get("dtype") != "F32":
            _add_error(
                errors,
                "checkpoint.scale_dtype_mismatch",
                f"{shard}:{scale_key}: expected F32, got {scale_descriptor.get('dtype')}",
            )
        shape = descriptor.get("shape")
        scale_shape = scale_descriptor.get("shape")
        if block_shape is not None and isinstance(shape, list) and len(shape) == 2 and isinstance(scale_shape, list):
            expected_shape = [_cdiv(int(shape[0]), block_shape[0]), _cdiv(int(shape[1]), block_shape[1])]
            if scale_shape != expected_shape:
                _add_error(
                    errors,
                    "checkpoint.scale_shape_mismatch",
                    f"{shard}:{scale_key}: expected shape {expected_shape}, got {scale_shape}",
                )
    return {"fp8_weights": fp8_count, "checked_scale_pairs": checked_pairs}


def _inspect_header_samples(
    model_dir: Path,
    weight_map: dict[str, str],
    quant_config: dict[str, Any],
    sample_limit: int,
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for shard in _sample_header_shards(weight_map, sample_limit):
        header, info = _read_safetensors_header(model_dir / shard, errors)
        if header is None:
            continue
        index_keys = {key for key, value in weight_map.items() if value == shard}
        header_keys = set(header) - {"__metadata__"}
        missing_from_header = sorted(index_keys - header_keys)
        unexpected_in_header = sorted(header_keys - index_keys)
        if missing_from_header or unexpected_in_header:
            _add_error(
                errors,
                "checkpoint.header_index_mismatch",
                f"{shard}: index/header differ; "
                f"missing={missing_from_header[:5]}, unexpected={unexpected_in_header[:5]}",
            )
        quant_result = _validate_quantized_header(shard, header, weight_map, quant_config, errors)
        if shard == weight_map.get("lm_head.weight"):
            lm_head = header.get("lm_head.weight")
            if isinstance(lm_head, dict) and lm_head.get("dtype") != "BF16":
                _add_error(
                    errors,
                    "checkpoint.lm_head_dtype_mismatch",
                    f"{shard}: expected lm_head.weight dtype BF16, got {lm_head.get('dtype')}",
                )
        layer_markers = sorted(
            {
                int(match.group(1))
                for key in header_keys
                if (match := _LAYER_RE.search(key)) is not None and int(match.group(1)) >= EXPECTED_LAYERS
            }
        )
        samples.append(
            {
                **info,
                "shard": shard,
                "index_tensor_count": len(index_keys),
                "mtp_layer_indices": layer_markers,
                "quantization": quant_result,
            }
        )

    if quant_config.get("present", True) and not any(sample["quantization"]["fp8_weights"] for sample in samples):
        _add_warning(
            warnings,
            "checkpoint.no_sampled_fp8",
            "sampled headers contained no FP8 weights; expand --header-sample-limit for stronger coverage",
        )
    return samples


def _load_model_contract(
    model_path: Path,
    max_model_len: int,
    errors: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract: dict[str, Any] = {}
    versions: dict[str, Any] = {}
    try:
        import vllm
        from vllm.config import ModelConfig, ParallelConfig
        from vllm.model_executor.models.registry import ModelRegistry

        from vllm_gaudi.patches import _patch_get_num_layers_by_block_type

        model_config = ModelConfig(
            model=str(model_path),
            tokenizer=str(model_path),
            max_model_len=max_model_len,
            enforce_eager=True,
        )
        model_cls, architecture = ModelRegistry.resolve_model_cls(model_config.architectures, model_config)
        _patch_get_num_layers_by_block_type()
        parallel_config = ParallelConfig()

        attention_layers = model_config.get_num_layers_by_block_type(parallel_config, "attention")
        linear_layers = model_config.get_num_layers_by_block_type(parallel_config, "linear_attention")
        mamba_layers = model_config.get_num_layers_by_block_type(parallel_config, "mamba")
        layers_block_type = getattr(model_config.hf_text_config, "layers_block_type", None)
        layers_block_type_counts = dict(Counter(layers_block_type)) if isinstance(layers_block_type, list) else None

        if architecture != EXPECTED_ARCHITECTURE:
            _add_error(
                errors,
                "registry.architecture",
                f"expected {EXPECTED_ARCHITECTURE}, resolved {architecture!r}",
            )
        if model_config.is_hybrid is not True:
            _add_error(errors, "registry.is_hybrid", "resolved model is not classified as hybrid")
        if model_config.has_inner_state is not True:
            _add_error(errors, "registry.has_inner_state", "resolved model does not declare inner state")
        if model_config.use_mla is not True:
            _add_error(errors, "registry.use_mla", "resolved model is not configured for MLA")
        if attention_layers != EXPECTED_ATTENTION_LAYERS:
            _add_error(
                errors,
                "layer_count.attention",
                f"expected {EXPECTED_ATTENTION_LAYERS}, got {attention_layers}",
            )
        if linear_layers != EXPECTED_LINEAR_LAYERS:
            _add_error(
                errors,
                "layer_count.linear_attention",
                f"expected {EXPECTED_LINEAR_LAYERS}, got {linear_layers}",
            )
        if mamba_layers != EXPECTED_LINEAR_LAYERS:
            _add_error(
                errors,
                "layer_count.mamba_alias",
                f"expected {EXPECTED_LINEAR_LAYERS}, got {mamba_layers}",
            )

        contract = {
            "architecture": architecture,
            "class_module": model_cls.__module__,
            "class_name": model_cls.__name__,
            "class_is_hybrid": bool(getattr(model_cls, "is_hybrid", False)),
            "class_has_inner_state": bool(getattr(model_cls, "has_inner_state", False)),
            "model_config_is_hybrid": bool(model_config.is_hybrid),
            "model_config_has_inner_state": bool(model_config.has_inner_state),
            "model_config_use_mla": bool(model_config.use_mla),
            "layers_block_type_counts": layers_block_type_counts,
            "layer_counts_after_hpu_patch": {
                "attention": attention_layers,
                "linear_attention": linear_layers,
                "mamba_alias": mamba_layers,
            },
        }

        vllm_root = Path(vllm.__file__).resolve().parents[1]
        plugin_root = Path(__file__).resolve().parents[2]
        versions = {
            "python": platform.python_version(),
            "note": "package versions are installed metadata; repository heads identify the loaded source roots",
            "packages": {
                "vllm": _package_version("vllm"),
                "vllm_gaudi": _package_version("vllm_gaudi"),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "safetensors": _package_version("safetensors"),
            },
            "engine_root": str(vllm_root),
            "engine_module_file": str(Path(vllm.__file__).resolve()),
            "engine_git_head": _git_head(vllm_root),
            "engine_git_dirty": _git_dirty(vllm_root),
            "plugin_root": str(plugin_root),
            "plugin_module_file": str(Path(__file__).resolve()),
            "plugin_git_head": _git_head(plugin_root),
            "plugin_git_dirty": _git_dirty(plugin_root),
            "provenance_note": (
                "dirty flags cover each whole worktree; core_file_sha256 identifies the exact files used by preflight"
            ),
            "core_file_sha256": {
                "engine": _source_file_hashes(
                    vllm_root,
                    (
                        "vllm/transformers_utils/configs/glm5_next.py",
                        "vllm/transformers_utils/config.py",
                        "vllm/model_executor/models/registry.py",
                        "vllm/v1/core/kv_cache_utils.py",
                    ),
                ),
                "plugin": _source_file_hashes(
                    plugin_root,
                    (
                        "vllm_gaudi/patches.py",
                        "vllm_gaudi/models/glm5next_preflight.py",
                        "tools/check_glm53_flash.py",
                        "tests/unit_tests/models/test_glm5next_p0_preflight.py",
                    ),
                ),
            },
        }
    except Exception as exc:
        _add_error(errors, "config.model_contract_failed", _exception_text(exc))

    return contract, versions


def _support_boundaries() -> dict[str, Any]:
    return {
        "scope": "offline static preflight only",
        "model_weights_loaded": False,
        "hpu_runtime_initialized": False,
        "network_access_allowed": False,
        "full_model_support_claimed": False,
        "dsa_sparse": "not validated; short <=2048 dense diagnostics still require kpool semantics validation",
        "mtp": "layer 45 weights are detected but MTP execution is not validated or enabled",
        "vision": "visual weights are detected but the vision tower is not validated",
        "prefix_caching": "not validated; keep disabled for the initial bring-up",
        "hpu_cache_layout": (
            "not validated by this preflight; mixed legacy/framework cache support remains a runtime blocker"
        ),
    }


def _interface_corrections() -> dict[str, Any]:
    return {
        "model_config_layers_block_type": "Glm5NextTextConfig already maps deepseek_sparse_attention to attention",
        "mla_spec_hierarchy": "MLAAttentionSpec inherits FullAttentionSpec",
        "compact_gdn_default": "GLM/model-runner auto-enables compact GDN when GDN layers are detected",
        "cache_contract": {
            "attention_layers": 11,
            "indexer_layers": 11,
            "linear_attention_layers": 34,
            "tail_layers": 11,
            "bytes_per_block_formula": "11 * MLA_page + 11 * indexer_page; Mamba and tail storage alias it",
        },
    }


def run_preflight(
    model_path: str | Path,
    *,
    tensor_parallel_size: int = 8,
    max_model_len: int = 2048,
    header_sample_limit: int = 7,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest without loading checkpoint payloads."""

    started = datetime.now(timezone.utc)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    path = Path(model_path).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": started.isoformat().replace("+00:00", "Z"),
        "ok": False,
        "model": {
            "path": str(path),
            "config_path": str(path / "config.json"),
            "index_path": str(path / "model.safetensors.index.json"),
        },
        "support_boundaries": _support_boundaries(),
        "interface_corrections": _interface_corrections(),
        "warnings": warnings,
        "errors": errors,
    }

    if tensor_parallel_size < 1:
        _add_error(errors, "args.tensor_parallel_size", "must be at least 1")
    if max_model_len < 1:
        _add_error(errors, "args.max_model_len", "must be at least 1")
    if header_sample_limit < 1:
        _add_error(errors, "args.header_sample_limit", "must be at least 1")
    if not path.is_dir():
        _add_error(errors, "model_dir_missing", f"{path} is not a directory")
        report["duration_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
        return report

    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    raw_config: dict[str, Any] | None = None
    index: dict[str, Any] | None = None
    if not config_path.is_file():
        _add_error(errors, "config.missing", f"missing {config_path}")
    else:
        try:
            loaded_config = _read_json(config_path)
            if not isinstance(loaded_config, dict):
                raise ValueError("config.json root is not an object")
            raw_config = loaded_config
            report["model"]["config_sha256"] = _sha256_file(config_path)
        except Exception as exc:
            _add_error(errors, "config.read_failed", _exception_text(exc))

    if not index_path.is_file():
        _add_error(errors, "checkpoint.index_missing", f"missing {index_path}")
    else:
        try:
            loaded_index = _read_json(index_path)
            if not isinstance(loaded_index, dict):
                raise ValueError("model.safetensors.index.json root is not an object")
            index = loaded_index
            report["model"]["index_sha256"] = _sha256_file(index_path)
        except Exception as exc:
            _add_error(errors, "checkpoint.index_read_failed", _exception_text(exc))

    layout: dict[str, Any] | None = None
    quant_config: dict[str, Any] = {"present": False}
    if raw_config is not None:
        model_type = raw_config.get("model_type")
        architectures = raw_config.get("architectures")
        if model_type != EXPECTED_MODEL_TYPE:
            _add_error(errors, "config.model_type", f"expected {EXPECTED_MODEL_TYPE!r}, got {model_type!r}")
        if architectures != [EXPECTED_ARCHITECTURE]:
            _add_error(
                errors,
                "config.architectures",
                f"expected [{EXPECTED_ARCHITECTURE!r}], got {architectures!r}",
            )
        text_config = raw_config.get("text_config")
        if not isinstance(text_config, dict):
            _add_error(errors, "config.text_config", "expected a text_config object")
        else:
            report["model"]["text_config"] = {
                "model_type": text_config.get("model_type"),
                "num_hidden_layers": text_config.get("num_hidden_layers"),
                "hidden_size": text_config.get("hidden_size"),
                "q_lora_rank": text_config.get("q_lora_rank"),
                "kv_lora_rank": text_config.get("kv_lora_rank"),
                "qk_nope_head_dim": text_config.get("qk_nope_head_dim"),
                "qk_rope_head_dim": text_config.get("qk_rope_head_dim"),
                "v_head_dim": text_config.get("v_head_dim"),
            }
            layout = _validate_text_config(text_config, errors)
            report["layer_layout"] = layout
        quant_config = _validate_quant_config(raw_config, errors)
        report["quantization"] = quant_config

    checkpoint_summary: dict[str, Any] | None = None
    if index is not None:
        mtp_layers = 1
        if isinstance(raw_config, dict):
            text_config = raw_config.get("text_config")
            if isinstance(text_config, dict) and isinstance(text_config.get("num_nextn_predict_layers"), int):
                mtp_layers = text_config["num_nextn_predict_layers"]
        checkpoint_summary = _validate_checkpoint_index(index, mtp_layers, path, errors, warnings)
        report["checkpoint"] = checkpoint_summary
        weight_map = index.get("weight_map")
        if isinstance(weight_map, dict):
            checkpoint_summary["header_samples"] = _inspect_header_samples(
                path,
                weight_map,
                quant_config,
                header_sample_limit,
                errors,
                warnings,
            )
        total_size = checkpoint_summary.get("total_size")
        if isinstance(total_size, int) and tensor_parallel_size > 0:
            bytes_per_rank = _cdiv(total_size, tensor_parallel_size)
            report["capacity"] = {
                "weight_bytes_total": total_size,
                "weight_gib_total": total_size / (1024**3),
                "tensor_parallel_size": tensor_parallel_size,
                "bytes_per_rank_ceil": bytes_per_rank,
                "gib_per_rank_ceil": bytes_per_rank / (1024**3),
                "is_static_lower_bound": True,
                "excludes": [
                    "runtime workspace",
                    "KV cache",
                    "KDA state",
                    "mHC temporaries",
                    "collectives",
                    "graph capture",
                    "allocator fragmentation and safety margin",
                ],
            }

    if raw_config is not None:
        contract, versions = _load_model_contract(path, max_model_len, errors)
        report["registry"] = contract
        report["source_versions"] = versions
    else:
        report["source_versions"] = {
            "packages": {
                "vllm_gaudi": _package_version("vllm_gaudi"),
                "torch": _package_version("torch"),
            },
            "plugin_root": str(Path(__file__).resolve().parents[2]),
            "plugin_git_head": _git_head(Path(__file__).resolve().parents[2]),
        }

    report["ok"] = not errors
    report["duration_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
    return report
