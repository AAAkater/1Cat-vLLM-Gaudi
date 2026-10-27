#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Single-HPU smoke test for GLM-5.3 block-FP8 routed MoE.

This tool deliberately uses no model loader, vLLM runtime, or CPU fallback for
device cases.  Every device case injects ``HPUBlockFP8Backend`` and compares
the native result with an independent full-dequantization CPU oracle.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import traceback
from typing import Any

import torch

from vllm_gaudi.ops.hpu_glm5next_fp8_moe import (
    FP8_E4M3FN,
    HPUBlockFP8Backend,
    block_fp8_gemm,
    glm53_fp8_moe,
    validate_hpu_fp8_schemas,
)

FP8_MAX = 240.0
QUANT_EPS = 1e-8
DEFAULT_OUTPUT_DIR = Path("artifacts/glm53-flash/fp8-moe-hpu-20261024")
EXPECTED_MODULE_ID = "3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().float().cpu()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "finite": bool(finite.all().item()),
    }
    if finite_values.numel():
        summary.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "abs_mean": float(finite_values.abs().mean().item()),
                "sample": finite_values.flatten()[:8].tolist(),
            }
        )
    return summary


def _oracle_quantize_activation(x: torch.Tensor, block_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    quantized_blocks: list[torch.Tensor] = []
    scale_blocks: list[torch.Tensor] = []
    for start in range(0, x.shape[1], block_k):
        block = x[:, start:start + block_k].to(torch.float32)
        scale = (block.abs().amax(dim=1, keepdim=True) + QUANT_EPS) / FP8_MAX
        quantized = (block / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_E4M3FN)
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
    output_dtype: torch.dtype,
) -> torch.Tensor:
    quantized, activation_scale = _oracle_quantize_activation(activation, block_k)
    activation_dequant = quantized.to(torch.float32)
    activation_dequant *= activation_scale.repeat_interleave(block_k, dim=1)[:, :activation.shape[1]]

    expanded_weight_scale = weight_scale.repeat_interleave(block_n, dim=0)
    expanded_weight_scale = expanded_weight_scale.repeat_interleave(block_k, dim=1)
    weight_dequant = weight.to(torch.float32) * expanded_weight_scale[:weight.shape[0], :weight.shape[1]]
    return (activation_dequant @ weight_dequant.T).to(output_dtype)


def _oracle_moe(
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
) -> tuple[torch.Tensor, dict[str, int | float]]:
    output = torch.zeros(hidden_states.shape, dtype=torch.float32)
    intermediate_size = w13_weight.shape[1] // 2
    expert_base = ep_rank * num_local_experts
    stats: dict[str, int | float] = {
        "local_routes": 0,
        "gate_upper_clamped": 0,
        "gate_negative": 0,
        "up_upper_clamped": 0,
        "up_lower_clamped": 0,
        "gate_min": math.inf,
        "gate_max": -math.inf,
        "up_min": math.inf,
        "up_max": -math.inf,
    }
    for token in range(hidden_states.shape[0]):
        for route in range(topk_ids.shape[1]):
            local_expert = int(topk_ids[token, route]) - expert_base
            if local_expert < 0 or local_expert >= num_local_experts:
                continue
            projection = _oracle_block_gemm(
                hidden_states[token:token + 1],
                w13_weight[local_expert],
                w13_weight_scale[local_expert],
                block_n=block_n,
                block_k=block_k,
                output_dtype=torch.float32,
            )
            gate = projection[:, :intermediate_size]
            up = projection[:, intermediate_size:]
            stats["local_routes"] += 1
            stats["gate_upper_clamped"] += int((gate > limit).sum().item())
            stats["gate_negative"] += int((gate < 0).sum().item())
            stats["up_upper_clamped"] += int((up > limit).sum().item())
            stats["up_lower_clamped"] += int((up < -limit).sum().item())
            stats["gate_min"] = min(float(stats["gate_min"]), float(gate.min().item()))
            stats["gate_max"] = max(float(stats["gate_max"]), float(gate.max().item()))
            stats["up_min"] = min(float(stats["up_min"]), float(up.min().item()))
            stats["up_max"] = max(float(stats["up_max"]), float(up.max().item()))

            gate_clamped = gate.clamp(max=limit)
            up_clamped = up.clamp(min=-limit, max=limit)
            middle = (torch.nn.functional.silu(gate_clamped) * up_clamped).to(hidden_states.dtype)
            down = _oracle_block_gemm(
                middle,
                w2_weight[local_expert],
                w2_weight_scale[local_expert],
                block_n=block_n,
                block_k=block_k,
                output_dtype=torch.float32,
            )
            output[token] += down[0] * topk_weights[token, route].to(torch.float32)
    return output.to(hidden_states.dtype), stats


def _compare(
    name: str,
    cpu: torch.Tensor,
    hpu: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    cpu_fp32 = cpu.detach().float().cpu()
    hpu_fp32 = hpu.detach().float().cpu()
    if cpu_fp32.shape != hpu_fp32.shape:
        raise RuntimeError(f"{name}: shape mismatch CPU {cpu_fp32.shape} vs HPU {hpu_fp32.shape}")
    if not bool(torch.isfinite(hpu_fp32).all().item()):
        raise RuntimeError(f"{name}: HPU output contains non-finite values")
    delta = (hpu_fp32 - cpu_fp32).abs()
    scale = cpu_fp32.abs().clamp_min(1e-12)
    passed = bool(torch.isclose(hpu_fp32, cpu_fp32, rtol=rtol, atol=atol).all().item())
    record = {
        "status": "pass" if passed else "mismatch",
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_error": float(delta.mean().item()) if delta.numel() else 0.0,
        "max_rel_error": float((delta / scale).max().item()) if delta.numel() else 0.0,
        "cpu_summary": _tensor_summary(cpu_fp32),
        "hpu_summary": _tensor_summary(hpu_fp32),
    }
    if not passed:
        raise RuntimeError(f"{name}: HPU/CPU oracle mismatch: {json.dumps(record, sort_keys=True)}")
    return record


def _nonuniform_scale(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return 0.01 + 0.04 * torch.rand(shape, generator=generator, dtype=torch.float32)


def _random_fp8(shape: tuple[int, ...], generator: torch.Generator, amplitude: int = 8) -> torch.Tensor:
    values = torch.randint(
        -amplitude,
        amplitude + 1,
        shape,
        generator=generator,
        dtype=torch.int16,
    )
    return values.float().to(FP8_E4M3FN)


def _make_random_weights(
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    block_n: int,
    block_k: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    w13 = _random_fp8((num_local_experts, 2 * intermediate_size, hidden_size), generator)
    w2 = _random_fp8((num_local_experts, hidden_size, intermediate_size), generator)
    w13_scale = _nonuniform_scale(
        (
            num_local_experts,
            math.ceil(2 * intermediate_size / block_n),
            math.ceil(hidden_size / block_k),
        ),
        generator,
    )
    w2_scale = _nonuniform_scale(
        (
            num_local_experts,
            math.ceil(hidden_size / block_n),
            math.ceil(intermediate_size / block_k),
        ),
        generator,
    )
    return w13, w13_scale, w2, w2_scale


def _make_clamp_weights(
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    block_n: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w13 = torch.zeros((num_local_experts, 2 * intermediate_size, hidden_size), dtype=torch.float32)
    w13[0, :intermediate_size] = 120.0
    w13[0, intermediate_size:] = -120.0
    if num_local_experts > 1:
        w13[1, :intermediate_size] = -120.0
        w13[1, intermediate_size:] = 120.0
    w13 = w13.to(FP8_E4M3FN)

    w2 = torch.ones((num_local_experts, hidden_size, intermediate_size), dtype=torch.float32).to(FP8_E4M3FN)
    w13_scale = torch.empty(
        (
            num_local_experts,
            math.ceil(2 * intermediate_size / block_n),
            math.ceil(hidden_size / block_k),
        ),
        dtype=torch.float32,
    )
    w13_scale[0].fill_(0.25)
    w13_scale[0, :, -1] = 0.5
    if num_local_experts > 1:
        w13_scale[1].fill_(0.125)
        w13_scale[1, :, -1] = 0.375
    w2_scale = torch.empty(
        (
            num_local_experts,
            math.ceil(hidden_size / block_n),
            math.ceil(intermediate_size / block_k),
        ),
        dtype=torch.float32,
    )
    w2_scale[0].fill_(1e-4)
    w2_scale[0, :, -1] = 2e-4
    if num_local_experts > 1:
        w2_scale[1].fill_(1.5e-4)
        w2_scale[1, :, -1] = 2.5e-4
    return w13, w13_scale, w2, w2_scale


def _collect_hpu_schemas() -> dict[str, Any]:
    validated = validate_hpu_fp8_schemas()
    all_schemas: dict[str, dict[str, str]] = {}
    for name in ("cast_to_fp8_v2", "fp8_gemm_v2"):
        op = getattr(torch.ops.hpu, name)
        all_schemas[name] = {key: str(value) for key, value in op._schemas.items()}
    return {
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "validated_default": validated,
        "all_overloads": all_schemas,
    }


def _device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "habana_visible_modules": os.environ.get("HABANA_VISIBLE_MODULES"),
        "hls_module_id": os.environ.get("HLS_MODULE_ID"),
        "pt_hpu_lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"),
        "is_available": bool(torch.hpu.is_available()),
    }
    if info["is_available"]:
        info["device_count"] = int(torch.hpu.device_count())
        try:
            info["device_name"] = torch.hpu.get_device_name(0)
        except Exception as exc:  # pragma: no cover - diagnostic only
            info["device_name_error"] = f"{type(exc).__name__}: {exc}"
        try:
            info["device_properties"] = str(torch.hpu.get_device_properties(0))
        except Exception as exc:  # pragma: no cover - diagnostic only
            info["device_properties_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _move_hpu(value: torch.Tensor) -> torch.Tensor:
    return value.to("hpu")


def _case_checkpoint(stage: str, op: str) -> str:
    return (
        f"artifacts=glm53-flash/fp8-moe-hpu-20261024 "
        f"stage={stage} op={op} module_id={os.environ.get('HABANA_VISIBLE_MODULES')} "
        f"pid={os.getpid()}"
    )


def run_smoke(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    oracle_stats: dict[str, Any] = {}
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    native_errors: list[dict[str, Any]] = []
    active_stage = "startup"
    active_op = "environment"

    try:
        if os.environ.get("HABANA_VISIBLE_MODULES") != EXPECTED_MODULE_ID:
            raise RuntimeError(
                "HABANA_VISIBLE_MODULES must pin the physical card-1 module id "
                f"{EXPECTED_MODULE_ID}, got {os.environ.get('HABANA_VISIBLE_MODULES')!r}"
            )
        if os.environ.get("HLS_MODULE_ID") != EXPECTED_MODULE_ID:
            raise RuntimeError(
                "HLS_MODULE_ID must pin physical card-1 module id "
                f"{EXPECTED_MODULE_ID}, got {os.environ.get('HLS_MODULE_ID')!r}"
            )
        if os.environ.get("PT_HPU_LAZY_MODE") != "0":
            raise RuntimeError("PT_HPU_LAZY_MODE must be 0 for eager smoke validation")

        active_stage = "schema"
        active_op = "dispatcher_metadata"
        schema = _collect_hpu_schemas()
        _write_json(output_dir / "schema.json", schema)

        active_stage = "device"
        active_op = "device_discovery"
        device = _device_info()
        if not device["is_available"] or device.get("device_count", 0) < 1:
            raise RuntimeError(f"no usable pinned HPU, got {device!r}")
        torch.hpu.set_device(0)
        device["effective_hls_module_id"] = os.environ.get("HLS_MODULE_ID")
        device["logical_device"] = int(torch.hpu.current_device())
        _write_json(output_dir / "device.json", device)
        if device["effective_hls_module_id"] != EXPECTED_MODULE_ID:
            raise RuntimeError(f"HPU pin changed during device initialization: {device!r}")

        active_stage = "backend_init"
        active_op = "validate_schema"
        hpu_backend = HPUBlockFP8Backend()

        # Native cast + FP32-output fp8_gemm + K-tile block GEMM, BF16 and FP32.
        active_stage = "block_gemm"
        block_generator = torch.Generator().manual_seed(101)
        block_h, block_n, block_k = 256, 256, 128
        block_weight = _random_fp8((block_n, block_h), block_generator, amplitude=12)
        block_weight_scale = _nonuniform_scale((2, 2), block_generator)
        block_activation = torch.randn(3, block_h, generator=block_generator)
        for dtype in (torch.bfloat16, torch.float32):
            active_op = f"cast+gemm+accumulate [{dtype}]"
            x_cpu = block_activation.to(dtype)
            cpu = _oracle_block_gemm(
                x_cpu,
                block_weight,
                block_weight_scale,
                block_n=block_k,
                block_k=block_k,
                output_dtype=dtype,
            )
            hpu = block_fp8_gemm(
                _move_hpu(x_cpu),
                _move_hpu(block_weight),
                _move_hpu(block_weight_scale),
                block_size=(block_k, block_k),
                output_dtype=dtype,
                backend=hpu_backend,
            )
            torch.hpu.synchronize()
            name = f"block_gemm_{str(dtype).split('.')[-1]}"
            results[name] = _compare(name, cpu, hpu, rtol=2e-3, atol=2e-3)
            oracle_stats[name] = _tensor_summary(cpu)
            outputs[name] = {"cpu": cpu, "hpu": hpu.detach().float().cpu()}
            print(f"PASS {name}: {json.dumps(results[name]['hpu_summary'], sort_keys=True)}")

        # Tiny clamped MoE: two K blocks, nonuniform scales, EP rank-local
        # partial, duplicate routes, zero-weight padding, and saturating gates.
        active_stage = "tiny_clamped_moe"
        tiny_h, tiny_i = 256, 128
        tiny_e = 2
        w13, w13_scale, w2, w2_scale = _make_clamp_weights(
            num_local_experts=tiny_e,
            hidden_size=tiny_h,
            intermediate_size=tiny_i,
            block_n=block_k,
            block_k=block_k,
        )
        topk_ids = torch.tensor([[2, 3], [3, 3], [0, 1]], dtype=torch.int64)
        topk_weights_cpu = torch.tensor([[0.60, 0.40], [0.70, 0.30], [0.0, 0.0]], dtype=torch.float32)
        for dtype in (torch.bfloat16, torch.float32):
            active_op = f"custom_rank_local_moe [{dtype}]"
            x_cpu = torch.ones((3, tiny_h), dtype=dtype)
            route_weights = topk_weights_cpu.to(dtype)
            cpu, clamp_stats = _oracle_moe(
                x_cpu,
                topk_ids,
                route_weights,
                w13,
                w13_scale,
                w2,
                w2_scale,
                ep_rank=1,
                num_local_experts=tiny_e,
                block_n=block_k,
                block_k=block_k,
                limit=10.0,
            )
            if not (
                clamp_stats["gate_upper_clamped"] > 0
                and clamp_stats["gate_negative"] > 0
                and clamp_stats["up_upper_clamped"] > 0
                and clamp_stats["up_lower_clamped"] > 0
            ):
                raise RuntimeError(f"clamp fixture did not exercise all clamp directions: {clamp_stats}")
            hpu = glm53_fp8_moe(
                _move_hpu(x_cpu),
                _move_hpu(topk_ids),
                _move_hpu(route_weights),
                _move_hpu(w13),
                _move_hpu(w13_scale),
                _move_hpu(w2),
                _move_hpu(w2_scale),
                ep_rank=1,
                global_num_experts=4,
                swiglu_limit=10.0,
                block_size=(block_k, block_k),
                token_chunk=2,
                backend=hpu_backend,
            )
            torch.hpu.synchronize()
            name = f"tiny_clamped_moe_{str(dtype).split('.')[-1]}"
            results[name] = _compare(name, cpu, hpu, rtol=4e-2, atol=3e-2)
            results[name]["clamp_oracle_stats"] = clamp_stats
            oracle_stats[name] = {"output": _tensor_summary(cpu), "clamp": clamp_stats}
            outputs[name] = {"cpu": cpu, "hpu": hpu.detach().float().cpu()}
            print(f"PASS {name}: clamp={clamp_stats}")

        # Real TP8 local shape: H=4096, intermediate-per-rank=256, two small
        # local experts.  One local route is enough to validate all 128x128
        # block boundaries without loading a checkpoint on the device.
        active_stage = "real_tp8_shape"
        active_op = "custom_rank_local_moe [h4096,i256,bf16]"
        real_h, real_i, real_e = 4096, 256, 2
        real_w13, real_w13_scale, real_w2, real_w2_scale = _make_random_weights(
            num_local_experts=real_e,
            hidden_size=real_h,
            intermediate_size=real_i,
            block_n=block_k,
            block_k=block_k,
            seed=211,
        )
        real_x = torch.linspace(-1.0, 1.0, real_h, dtype=torch.bfloat16).reshape(1, real_h)
        real_ids = torch.tensor([[2]], dtype=torch.int64)
        real_weights = torch.ones((1, 1), dtype=torch.bfloat16)
        real_cpu, real_clamp_stats = _oracle_moe(
            real_x,
            real_ids,
            real_weights,
            real_w13,
            real_w13_scale,
            real_w2,
            real_w2_scale,
            ep_rank=1,
            num_local_experts=real_e,
            block_n=block_k,
            block_k=block_k,
            limit=10.0,
        )
        real_hpu = glm53_fp8_moe(
            _move_hpu(real_x),
            _move_hpu(real_ids),
            _move_hpu(real_weights),
            _move_hpu(real_w13),
            _move_hpu(real_w13_scale),
            _move_hpu(real_w2),
            _move_hpu(real_w2_scale),
            ep_rank=1,
            global_num_experts=4,
            swiglu_limit=10.0,
            block_size=(block_k, block_k),
            token_chunk=1,
            backend=hpu_backend,
        )
        torch.hpu.synchronize()
        name = "real_tp8_h4096_i256_bf16"
        results[name] = _compare(name, real_cpu, real_hpu, rtol=4e-2, atol=3e-2)
        results[name]["clamp_oracle_stats"] = real_clamp_stats
        oracle_stats[name] = {"output": _tensor_summary(real_cpu), "clamp": real_clamp_stats}
        outputs[name] = {"cpu": real_cpu, "hpu": real_hpu.detach().float().cpu()}
        print(f"PASS {name}: {json.dumps(results[name]['hpu_summary'], sort_keys=True)}")

        _write_json(output_dir / "cpu_oracle.json", oracle_stats)
        _write_json(output_dir / "results.json", results)
        _write_json(output_dir / "native_errors.json", [])
        torch.save(outputs, output_dir / "outputs.pt")
        _write_json(
            output_dir / "summary.json",
            {
                "status": "pass",
                "timestamp_utc": _utc_now(),
                "checkpoint": _case_checkpoint("complete", "all_native_cases"),
                "cases": sorted(results),
                "fallback": "none; every device call injected HPUBlockFP8Backend",
            },
        )
        print(f"SMOKE_PASS checkpoint={_case_checkpoint('complete', 'all_native_cases')}")
        return 0
    except Exception as exc:
        error = {
            "status": "fail",
            "timestamp_utc": _utc_now(),
            "checkpoint": _case_checkpoint(active_stage, active_op),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "fallback": "none; HPU failure is not converted to a W8A16 or CPU device path",
        }
        native_errors.append(error)
        _write_json(output_dir / "native_errors.json", native_errors)
        _write_json(output_dir / "cpu_oracle.json", oracle_stats)
        _write_json(output_dir / "results.json", results)
        torch.save(outputs, output_dir / "outputs.pt")
        _write_json(output_dir / "summary.json", {"status": "fail", "error": error})
        print(f"SMOKE_FAILED checkpoint={error['checkpoint']}")
        print(json.dumps(error, indent=2, sort_keys=True))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    return run_smoke(args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
