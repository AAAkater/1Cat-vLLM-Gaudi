# SPDX-License-Identifier: Apache-2.0
"""Portable block-FP8 W8A8 GEMM and GLM-5.3 rank-local routed MoE.

The checkpoint contract is deliberately narrow:

* expert weights remain E4M3FN matrices shaped ``[N, K]``;
* their multiplication scales remain FP32 matrices shaped
  ``[ceil(N / block_n), ceil(K / block_k)]``;
* activations are dynamically quantized once per token and K block;
* partial dot products are accumulated in FP32.

Gaudi2's checkpoint loader has already transformed serialized E4M3FN values
and scales as a pair (half the FP8 values and double the FP32 scales).  This
module never performs that conversion again.

No vLLM or Habana module is imported here.  The default CPU backend is a
round-to-``float8_e4m3fn`` plus FP32 matmul reference.  The HPU backend reads
the real dispatcher schemas and then uses ``cast_to_fp8_v2`` and
``fp8_gemm_v2`` with ``B_scale_shape``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

FP8_E4M3FN = torch.float8_e4m3fn
_FP8_MAX_FN = float(torch.finfo(FP8_E4M3FN).max)
_SUPPORTED_ACTIVATION_DTYPES = (torch.float32, torch.bfloat16)
_SUPPORTED_OUTPUT_DTYPES = (torch.float32, torch.bfloat16)
_DEFAULT_BLOCK_SIZE = (128, 128)
_DEFAULT_TOKEN_CHUNK = 128
_DEFAULT_QUANT_EPS = 1e-8


class BlockFP8Backend(Protocol):
    """Injected activation quantizer and FP8 GEMM primitive."""

    name: str

    def quantize_activation(
        self,
        x: Tensor,
        *,
        fp8_max: float,
        eps: float,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(fp8_x, dequant_scale)`` with scale shaped ``[T, 1]``."""

    def gemm(
        self,
        a: Tensor,
        b: Tensor,
        *,
        a_scale: Tensor,
        b_scale: Tensor,
        out_dtype: torch.dtype,
    ) -> Tensor:
        """Compute ``(a * a_scale) @ (b * b_scale).T``.

        ``a`` is ``[T, Ktile]``, ``b`` is ``[N, Ktile]``, and ``b_scale`` is
        shaped ``[N]`` with one multiplication scale per output channel.
        """


def _row_dynamic_scale(x: Tensor, fp8_max: float, eps: float) -> Tensor:
    x_fp32 = x.to(torch.float32)
    return (x_fp32.abs().amax(dim=-1, keepdim=True) + eps) / fp8_max


@dataclass(frozen=True)
class CpuBlockFP8Backend:
    """CPU test backend: FP8 round-trip and an FP32 matrix multiply."""

    name: str = "cpu"

    def quantize_activation(
        self,
        x: Tensor,
        *,
        fp8_max: float,
        eps: float,
    ) -> tuple[Tensor, Tensor]:
        scale = _row_dynamic_scale(x, fp8_max, eps)
        scaled = (x.to(torch.float32) / scale).clamp(-fp8_max, fp8_max)
        return scaled.to(FP8_E4M3FN), scale

    def gemm(
        self,
        a: Tensor,
        b: Tensor,
        *,
        a_scale: Tensor,
        b_scale: Tensor,
        out_dtype: torch.dtype,
    ) -> Tensor:
        partial = torch.matmul(a.to(torch.float32), b.to(torch.float32).transpose(0, 1))
        partial = partial * a_scale.to(torch.float32)
        partial = partial * b_scale.to(torch.float32).reshape(1, -1)
        return partial.to(out_dtype)


_CPU_BACKEND = CpuBlockFP8Backend()
_HPU_BACKEND: HPUBlockFP8Backend | None = None



def _default_schema(op_name: str) -> str:
    """Read an operator's default dispatcher schema without executing it."""

    ops = getattr(torch.ops, "hpu", None)
    if ops is None:
        raise RuntimeError("torch.ops.hpu is unavailable; HPU FP8 backend cannot run")
    try:
        op = getattr(ops, op_name)
        schemas = op._schemas
        schema = schemas.get("", None)
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"torch.ops.hpu.{op_name} has no readable default schema") from exc
    if schema is None:
        raise RuntimeError(f"torch.ops.hpu.{op_name} has no default dispatcher schema")
    return str(schema)


def _normalize_schema(schema: str) -> str:
    return " ".join(schema.split())


def validate_hpu_fp8_schemas(schemas: Mapping[str, str] | None = None) -> dict[str, str]:
    """Validate the exact HPU FP8 dispatcher contract.

    This only reads dispatcher metadata.  It does not import Habana, allocate a
    device tensor, or initialize an HPU.
    """

    actual = dict(schemas) if schemas is not None else {
        "cast_to_fp8_v2": _default_schema("cast_to_fp8_v2"),
        "fp8_gemm_v2": _default_schema("fp8_gemm_v2"),
    }
    try:
        cast_schema = _normalize_schema(actual["cast_to_fp8_v2"])
        gemm_schema = _normalize_schema(actual["fp8_gemm_v2"])
    except KeyError as exc:
        raise RuntimeError(f"missing HPU dispatcher schema: {exc.args[0]}") from exc

    cast_parts = (
        "Tensor input",
        "Tensor? scale=None",
        "bool stochastic_rounding=False",
        "bool is_amax=False",
        "ScalarType dtype=None",
        "int[]? scale_shape=None",
    )
    missing_cast = [part for part in cast_parts if part not in cast_schema]
    if missing_cast:
        raise RuntimeError(f"unsupported cast_to_fp8_v2 schema, missing {missing_cast}: {cast_schema}")

    gemm_parts = (
        "Tensor A",
        "bool trans_A",
        "Tensor B",
        "bool trans_B",
        "Tensor? D",
        "ScalarType out_dtype",
        "Tensor? A_scale_inv=None",
        "Tensor? B_scale_inv=None",
        "Tensor? bias=None",
        "bool accumulate=False",
        "int[]? B_scale_shape=None",
    )
    missing_gemm = [part for part in gemm_parts if part not in gemm_schema]
    if missing_gemm:
        raise RuntimeError(f"unsupported fp8_gemm_v2 schema, missing {missing_gemm}: {gemm_schema}")
    if "scale_shape=None" in gemm_schema.replace("B_scale_shape=None", ""):
        raise RuntimeError(f"fp8_gemm_v2 exposes an unverified scale_shape keyword: {gemm_schema}")

    return actual


@dataclass(frozen=True)
class HPUBlockFP8Backend:
    """Default HPU backend using native FP8 cast and FP32-output GEMM."""

    name: str = "hpu"

    def __post_init__(self) -> None:
        validate_hpu_fp8_schemas()

    def quantize_activation(
        self,
        x: Tensor,
        *,
        fp8_max: float,
        eps: float,
    ) -> tuple[Tensor, Tensor]:
        scale = _row_dynamic_scale(x, fp8_max, eps)
        # The native cast evaluates its normalized values in the input dtype.
        # Feeding BF16 here rounds those values before FP8 rounding and can move
        # them across an FP8 midpoint by far more than one FP32 ULP.  Promote the
        # values only; the dynamic scale remains the original FP32 definition.
        cast_input = x.to(torch.float32)
        quantized = torch.ops.hpu.cast_to_fp8_v2(
            input=cast_input,
            scale=scale.reciprocal(),
            stochastic_rounding=False,
            is_amax=False,
            dtype=FP8_E4M3FN,
        )[0]
        return quantized, scale

    def gemm(
        self,
        a: Tensor,
        b: Tensor,
        *,
        a_scale: Tensor,
        b_scale: Tensor,
        out_dtype: torch.dtype,
    ) -> Tensor:
        # The verified reference uses a [K, N] B operand with
        # trans_B=False, a 1-D per-channel scale, and B_scale_shape=[N].
        # The transient FP8 transpose preserves the serialized E4M3FN codes.
        b_columns = b.transpose(0, 1).contiguous()
        b_scale = b_scale.reshape(-1)
        output = torch.ops.hpu.fp8_gemm_v2(
            A=a,
            trans_A=False,
            B=b_columns,
            trans_B=False,
            D=None,
            out_dtype=out_dtype,
            A_scale_inv=a_scale,
            B_scale_inv=b_scale,
            bias=None,
            accumulate=False,
            B_scale_shape=[b_columns.shape[1]],
        )
        if output.dtype != out_dtype:
            raise RuntimeError(
                "fp8_gemm_v2 did not preserve the requested output dtype: "
                f"requested {out_dtype}, got {output.dtype}"
            )
        return output


def _resolve_backend(x: Tensor, backend: BlockFP8Backend | None) -> BlockFP8Backend:
    if backend is not None:
        return backend
    if x.device.type == "cpu":
        return _CPU_BACKEND
    if x.device.type == "hpu":
        global _HPU_BACKEND
        if _HPU_BACKEND is None:
            _HPU_BACKEND = HPUBlockFP8Backend()
        return _HPU_BACKEND
    raise ValueError(f"unsupported block-FP8 device type: {x.device.type!r}")


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_block_size(block_size: tuple[int, int] | list[int]) -> tuple[int, int]:
    if not isinstance(block_size, (tuple, list)) or len(block_size) != 2:
        raise ValueError(f"block_size must be a two-element tuple, got {block_size!r}")
    block_n = _validate_positive_int(block_size[0], "block_n")
    block_k = _validate_positive_int(block_size[1], "block_k")
    return block_n, block_k


def _validate_activation(x: Tensor, name: str) -> None:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(x.shape)}")
    if x.dtype not in _SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            f"{name} must have dtype float32 or bfloat16, got {x.dtype}; "
            "float16 is intentionally unsupported"
        )


def _cdiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_fp8_weight_and_scale(
    weight: Tensor,
    weight_scale: Tensor,
    block_size: tuple[int, int],
    *,
    weight_name: str,
) -> None:
    if not isinstance(weight, Tensor) or not isinstance(weight_scale, Tensor):
        raise TypeError(f"{weight_name} and its scale must be torch.Tensor objects")
    if weight.ndim != 2:
        raise ValueError(f"{weight_name} must be [N, K], got shape {tuple(weight.shape)}")
    if weight.dtype != FP8_E4M3FN:
        raise TypeError(f"{weight_name} must remain E4M3FN, got {weight.dtype}")
    if not weight.is_contiguous():
        raise ValueError(f"{weight_name} must use a contiguous [N, K] layout")
    if weight_scale.dtype != torch.float32:
        raise TypeError(f"{weight_name} scale must be float32, got {weight_scale.dtype}")
    if weight_scale.ndim != 2:
        raise ValueError(
            f"{weight_name} scale must be [ceil(N/block_n), ceil(K/block_k)], "
            f"got shape {tuple(weight_scale.shape)}"
        )
    if not weight_scale.is_contiguous():
        raise ValueError(f"{weight_name} scale must be contiguous")
    if weight.shape[0] == 0 or weight.shape[1] == 0:
        raise ValueError(f"{weight_name} must have non-empty N and K dimensions")

    block_n, block_k = block_size
    expected = (_cdiv(weight.shape[0], block_n), _cdiv(weight.shape[1], block_k))
    if tuple(weight_scale.shape) != expected:
        raise ValueError(f"{weight_name} scale shape must be {expected}, got {tuple(weight_scale.shape)}")
    if not bool(torch.isfinite(weight_scale).all().item()):
        raise ValueError(f"{weight_name} scale contains a non-finite value")
    if not bool((weight_scale > 0).all().item()):
        raise ValueError(f"{weight_name} scale must be strictly positive")


def _validate_moe_tensor_shapes(
    hidden_states: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    w13_weight: Tensor,
    w13_weight_scale: Tensor,
    w2_weight: Tensor,
    w2_weight_scale: Tensor,
    block_size: tuple[int, int],
) -> tuple[int, int, int]:
    _validate_activation(hidden_states, "hidden_states")
    if hidden_states.shape[0] == 0:
        tokens = 0
    else:
        tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]

    if topk_ids.ndim != 2 or topk_ids.shape[0] != tokens:
        raise ValueError(f"topk_ids must be [T, K] with T={tokens}, got {tuple(topk_ids.shape)}")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_ids must be int32 or int64, got {topk_ids.dtype}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            f"topk_weights must match topk_ids shape {tuple(topk_ids.shape)}, got {tuple(topk_weights.shape)}"
        )
    if topk_weights.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"topk_weights must be float32 or bfloat16, got {topk_weights.dtype}")
    if not bool(torch.isfinite(topk_weights).all().item()):
        raise ValueError("topk_weights contains a non-finite value")

    if w13_weight.ndim != 3 or w13_weight.shape[0] == 0:
        raise ValueError(f"w13_weight must be [E_local, 2I, H], got {tuple(w13_weight.shape)}")
    num_local_experts = w13_weight.shape[0]
    if w13_weight.shape[1] % 2 != 0:
        raise ValueError(f"w13_weight output width must be even, got {w13_weight.shape[1]}")
    intermediate_size = w13_weight.shape[1] // 2
    if intermediate_size == 0 or w13_weight.shape[2] != hidden_size:
        raise ValueError(
            f"w13_weight must be [E_local, 2I, {hidden_size}] with I > 0, got {tuple(w13_weight.shape)}"
        )
    if w2_weight.shape != (num_local_experts, hidden_size, intermediate_size):
        raise ValueError(
            "w2_weight shape must match the w13-derived local expert layout "
            f"{(num_local_experts, hidden_size, intermediate_size)}, got {tuple(w2_weight.shape)}"
        )
    if w13_weight_scale.ndim != 3 or w13_weight_scale.shape[0] != num_local_experts:
        raise ValueError(
            "w13_weight_scale must have one [ceil(2I/bN), ceil(H/bK)] matrix per local expert, "
            f"got {tuple(w13_weight_scale.shape)}"
        )
    if w2_weight_scale.ndim != 3 or w2_weight_scale.shape[0] != num_local_experts:
        raise ValueError(
            "w2_weight_scale must have one [ceil(H/bN), ceil(I/bK)] matrix per local expert, "
            f"got {tuple(w2_weight_scale.shape)}"
        )
    return tokens, hidden_size, intermediate_size


def _validate_rank_and_routes(
    topk_ids: Tensor,
    topk_weights: Tensor,
    *,
    num_local_experts: int,
    ep_rank: int,
    global_num_experts: int,
    padding_route_id: int | None,
) -> None:
    if isinstance(ep_rank, bool) or not isinstance(ep_rank, int) or ep_rank < 0:
        raise ValueError(f"ep_rank must be a non-negative integer, got {ep_rank!r}")
    if isinstance(global_num_experts, bool) or not isinstance(global_num_experts, int) or global_num_experts <= 0:
        raise ValueError(f"global_num_experts must be a positive integer, got {global_num_experts!r}")
    if global_num_experts % num_local_experts != 0:
        raise ValueError(
            "rank-local contiguous expert mapping requires global_num_experts divisible by "
            f"num_local_experts, got {global_num_experts} and {num_local_experts}"
        )
    ep_size = global_num_experts // num_local_experts
    if ep_rank >= ep_size:
        raise ValueError(f"ep_rank {ep_rank} is outside EP size {ep_size}")

    padding = torch.zeros_like(topk_ids, dtype=torch.bool)
    if padding_route_id is not None:
        if isinstance(padding_route_id, bool) or not isinstance(padding_route_id, int):
            raise ValueError(f"padding_route_id must be an integer or None, got {padding_route_id!r}")
        padding = topk_ids == padding_route_id
    invalid = ((topk_ids < 0) & ~padding) | (topk_ids >= global_num_experts)
    if bool(invalid.any().item()):
        bad = topk_ids[invalid][0].item()
        raise ValueError(f"topk_ids contains invalid global expert id {bad}")
    if bool((padding & (topk_weights != 0)).any().item()):
        raise ValueError("padding routes must have exactly zero routing weight")


def _scale_per_output_channel(
    weight_scale: Tensor,
    *,
    n_columns: int,
    k_block: int,
    block_n: int,
) -> Tensor:
    column_scales = weight_scale[:, k_block]
    expanded = column_scales.repeat_interleave(block_n)[:n_columns]
    return expanded.contiguous().to(torch.float32)


def _validate_quantized_tile(
    quantized: Tensor,
    scale: Tensor,
    reference: Tensor,
    name: str,
) -> None:
    if quantized.shape != reference.shape or quantized.dtype != FP8_E4M3FN:
        raise RuntimeError(
            f"{name} quantizer returned {quantized.dtype}{tuple(quantized.shape)}, "
            f"expected {FP8_E4M3FN}{tuple(reference.shape)}"
        )
    expected_scale_shape = (reference.shape[0], 1)
    if scale.shape != expected_scale_shape or scale.dtype != torch.float32:
        raise RuntimeError(
            f"{name} quantizer returned scale {scale.dtype}{tuple(scale.shape)}, "
            f"expected float32{expected_scale_shape}"
        )
    if not bool(torch.isfinite(scale).all().item()) or not bool((scale > 0).all().item()):
        raise RuntimeError(f"{name} quantizer produced a non-finite or non-positive scale")


def block_fp8_gemm(
    activation: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    *,
    block_size: tuple[int, int] | list[int] = _DEFAULT_BLOCK_SIZE,
    k_tile: int | None = None,
    fp8_max: float = 240.0,
    quant_eps: float = _DEFAULT_QUANT_EPS,
    output_dtype: torch.dtype = torch.float32,
    backend: BlockFP8Backend | None = None,
) -> Tensor:
    """Block-FP8 W8A8 GEMM with FP32 K-tile accumulation.

    Args:
        activation: ``[T, K]`` float32 or bfloat16 input.
        weight: Serialized/prepared ``[N, K]`` E4M3FN weight.  It is never
            quantized or dequantized persistently.
        weight_scale: Positive FP32 multiplication scales shaped
            ``[ceil(N / block_n), ceil(K / block_k)]``.
        block_size: Checkpoint scale block ``(block_n, block_k)``.
        k_tile: GEMM K tile, defaulting to ``block_k``.  The current
            kFp8Dynamic128 contract requires it to equal ``block_k``.
        fp8_max: Dynamic activation range.  Gaudi2 uses 240 rather than the
            software E4M3FN maximum of 448.
        quant_eps: Positive bias added to the row maximum before scaling.
        output_dtype: FP32 or BF16 result dtype.  Accumulation is always FP32.
        backend: Optional injected quantize/GEMM backend.  CPU is the default;
            HPU uses the native FP8 ops.

    Returns:
        ``[T, N]`` tensor in ``output_dtype``.

    Raises:
        TypeError: For unsupported activation, weight, or scale dtypes.
        ValueError: For malformed layouts, block sizes, or non-finite scales.
        RuntimeError: If an injected/native backend violates the dtype contract.
    """

    block_n, block_k = _validate_block_size(block_size)
    _validate_activation(activation, "activation")
    _validate_fp8_weight_and_scale(
        weight,
        weight_scale,
        (block_n, block_k),
        weight_name="weight",
    )
    if activation.shape[1] != weight.shape[1]:
        raise ValueError(
            f"activation K {activation.shape[1]} does not match weight K {weight.shape[1]}"
        )
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise TypeError(f"output_dtype must be float32 or bfloat16, got {output_dtype}")
    if not isinstance(fp8_max, (int, float)) or not math.isfinite(float(fp8_max)):
        raise ValueError(f"fp8_max must be finite, got {fp8_max!r}")
    fp8_max = float(fp8_max)
    if fp8_max <= 0 or fp8_max > _FP8_MAX_FN:
        raise ValueError(f"fp8_max must be in (0, {_FP8_MAX_FN}], got {fp8_max!r}")
    if not isinstance(quant_eps, (int, float)) or not math.isfinite(float(quant_eps)) or quant_eps <= 0:
        raise ValueError(f"quant_eps must be finite and positive, got {quant_eps!r}")
    quant_eps = float(quant_eps)

    if k_tile is None:
        k_tile = block_k
    else:
        _validate_positive_int(k_tile, "k_tile")
    if k_tile != block_k:
        raise ValueError(
            "k_tile must equal block_k because activation quantization is per "
            f"K block; got k_tile={k_tile}, block_k={block_k}"
        )

    tokens = activation.shape[0]
    n_columns = weight.shape[0]
    if tokens == 0:
        return torch.empty((0, n_columns), dtype=output_dtype, device=activation.device)

    activation = activation.contiguous()
    kernel = _resolve_backend(activation, backend)
    accumulator = torch.zeros(
        (tokens, n_columns),
        dtype=torch.float32,
        device=activation.device,
    )
    k_size = activation.shape[1]
    for k_start in range(0, k_size, k_tile):
        k_end = min(k_start + k_tile, k_size)
        activation_tile = activation[:, k_start:k_end]
        weight_tile = weight[:, k_start:k_end].contiguous()
        quantized, activation_scale = kernel.quantize_activation(
            activation_tile,
            fp8_max=fp8_max,
            eps=quant_eps,
        )
        _validate_quantized_tile(quantized, activation_scale, activation_tile, "activation")

        k_block = k_start // block_k
        b_scale = _scale_per_output_channel(
            weight_scale,
            n_columns=n_columns,
            k_block=k_block,
            block_n=block_n,
        )
        partial = kernel.gemm(
            quantized,
            weight_tile,
            a_scale=activation_scale,
            b_scale=b_scale,
            out_dtype=torch.float32,
        )
        if partial.shape != accumulator.shape:
            raise RuntimeError(
                f"FP8 GEMM returned shape {tuple(partial.shape)}, expected {tuple(accumulator.shape)}"
            )
        if partial.dtype != torch.float32:
            raise RuntimeError(
                f"FP8 GEMM must return FP32 for K-tile accumulation, got {partial.dtype}"
            )
        accumulator.add_(partial)

    return accumulator.to(output_dtype)


def glm_swiglu_clamp(gate: Tensor, up: Tensor, limit: float) -> Tensor:
    """GLM SwiGLU: ``silu(min(gate, limit)) * clamp(up, -limit, limit)``.

    Gate clamping is intentionally one-sided.  Both inputs are evaluated in
    FP32 and the result remains FP32 so the caller can choose the activation
    boundary explicitly.
    """

    if not isinstance(limit, (int, float)) or not math.isfinite(float(limit)) or limit <= 0:
        raise ValueError(f"swiglu_limit must be finite and positive, got {limit!r}")
    if gate.shape != up.shape:
        raise ValueError(f"gate and up shapes differ: {tuple(gate.shape)} vs {tuple(up.shape)}")
    gate_fp32 = gate.to(torch.float32).clamp(max=float(limit))
    up_fp32 = up.to(torch.float32).clamp(min=-float(limit), max=float(limit))
    return F.silu(gate_fp32) * up_fp32


def glm53_fp8_moe(
    hidden_states: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    w13_weight: Tensor,
    w13_weight_scale: Tensor,
    w2_weight: Tensor,
    w2_weight_scale: Tensor,
    *,
    ep_rank: int,
    global_num_experts: int,
    swiglu_limit: float,
    block_size: tuple[int, int] | list[int] = _DEFAULT_BLOCK_SIZE,
    token_chunk: int = _DEFAULT_TOKEN_CHUNK,
    fp8_max: float = 240.0,
    quant_eps: float = _DEFAULT_QUANT_EPS,
    padding_route_id: int | None = -1,
    backend: BlockFP8Backend | None = None,
) -> Tensor:
    """Run one eager, rank-local GLM-5.3-Flash FP8 routed-expert partial.

    Global expert ids are mapped to this rank's contiguous local range
    ``[ep_rank * E_local, (ep_rank + 1) * E_local)``.  Only selected routes for
    active local experts gather token rows and expert weights.  Non-local and
    zero-weighted padding routes contribute exact zeros; this function performs
    no collective.

    Weights stay E4M3FN and scales stay FP32 for the full call.  ``w13`` and
    ``w2`` each accumulate K tiles in FP32.  The GLM clamp is applied in FP32,
    the intermediate is rounded to ``hidden_states.dtype`` before its dynamic
    FP8 quantization, and the routing weight is applied only after ``w2``.

    Args:
        hidden_states: ``[T, H]`` float32 or bfloat16 routed activations.
        topk_ids: ``[T, K]`` global expert ids.  ``padding_route_id`` is
            accepted only with an exactly zero route weight.
        topk_weights: ``[T, K]`` router weights.
        w13_weight: Local ``[E_local, 2I, H]`` E4M3FN gate/up weights.
        w13_weight_scale: Local ``[E_local, ceil(2I/bN), ceil(H/bK)]`` FP32
            multiplication scales.
        w2_weight: Local ``[E_local, H, I]`` E4M3FN down weights.
        w2_weight_scale: Local ``[E_local, ceil(H/bN), ceil(I/bK)]`` FP32
            multiplication scales.
        ep_rank: Zero-based EP rank owning this contiguous expert range.
        global_num_experts: Total number of routed experts.
        swiglu_limit: Positive GLM clamp limit.
        block_size: Checkpoint scale block ``(block_n, block_k)``.
        token_chunk: Maximum selected-route rows processed at once.
        fp8_max: Dynamic activation range (240 on Gaudi2).
        quant_eps: Positive dynamic-scale bias.
        padding_route_id: Optional sentinel expert id for zero-weight padding.
        backend: Optional injected backend; CPU and HPU are selected
            automatically from the activation device.

    Returns:
        ``[T, H]`` FP32 rank-local partial converted to
        ``hidden_states.dtype``.
    """

    block_n, block_k = _validate_block_size(block_size)
    _validate_positive_int(token_chunk, "token_chunk")
    if not isinstance(swiglu_limit, (int, float)) or not math.isfinite(float(swiglu_limit)):
        raise ValueError(f"swiglu_limit must be finite, got {swiglu_limit!r}")
    if swiglu_limit <= 0:
        raise ValueError(f"swiglu_limit must be positive, got {swiglu_limit!r}")

    tokens, hidden_size, intermediate_size = _validate_moe_tensor_shapes(
        hidden_states,
        topk_ids,
        topk_weights,
        w13_weight,
        w13_weight_scale,
        w2_weight,
        w2_weight_scale,
        (block_n, block_k),
    )
    num_local_experts = w13_weight.shape[0]
    _validate_rank_and_routes(
        topk_ids,
        topk_weights,
        num_local_experts=num_local_experts,
        ep_rank=ep_rank,
        global_num_experts=global_num_experts,
        padding_route_id=padding_route_id,
    )

    for expert in range(num_local_experts):
        _validate_fp8_weight_and_scale(
            w13_weight[expert],
            w13_weight_scale[expert],
            (block_n, block_k),
            weight_name=f"w13_weight[{expert}]",
        )
        _validate_fp8_weight_and_scale(
            w2_weight[expert],
            w2_weight_scale[expert],
            (block_n, block_k),
            weight_name=f"w2_weight[{expert}]",
        )

    if tokens == 0 or topk_ids.shape[1] == 0:
        return torch.zeros(
            (tokens, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    kernel = _resolve_backend(hidden_states, backend)
    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1).to(torch.float32)
    flat_tokens = torch.arange(tokens, device=hidden_states.device).repeat_interleave(topk_ids.shape[1])
    expert_base = ep_rank * num_local_experts
    local_ids = flat_ids - expert_base
    local_mask = (local_ids >= 0) & (local_ids < num_local_experts) & (flat_weights != 0)
    active_routes = torch.nonzero(local_mask, as_tuple=False).reshape(-1)
    output = torch.zeros(
        (tokens, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    if active_routes.numel() == 0:
        return output.to(hidden_states.dtype)

    active_local_ids = local_ids.index_select(0, active_routes)
    active_experts = torch.unique(active_local_ids, sorted=True)
    for expert_tensor in active_experts:
        expert = int(expert_tensor.item())
        expert_routes = active_routes.index_select(
            0,
            torch.nonzero(active_local_ids == expert_tensor, as_tuple=False).reshape(-1),
        )
        route_tokens = flat_tokens.index_select(0, expert_routes)
        route_scales = flat_weights.index_select(0, expert_routes)

        for chunk_start in range(0, expert_routes.numel(), token_chunk):
            chunk_end = min(chunk_start + token_chunk, expert_routes.numel())
            selected_tokens = route_tokens[chunk_start:chunk_end]
            selected_weights = route_scales[chunk_start:chunk_end]
            selected_x = hidden_states.index_select(0, selected_tokens)

            gate_up = block_fp8_gemm(
                selected_x,
                w13_weight[expert],
                w13_weight_scale[expert],
                block_size=(block_n, block_k),
                fp8_max=fp8_max,
                quant_eps=quant_eps,
                output_dtype=torch.float32,
                backend=kernel,
            )
            gate = gate_up[:, :intermediate_size]
            up = gate_up[:, intermediate_size:]
            intermediate = glm_swiglu_clamp(gate, up, float(swiglu_limit))
            intermediate = intermediate.to(hidden_states.dtype)

            down = block_fp8_gemm(
                intermediate,
                w2_weight[expert],
                w2_weight_scale[expert],
                block_size=(block_n, block_k),
                fp8_max=fp8_max,
                quant_eps=quant_eps,
                output_dtype=torch.float32,
                backend=kernel,
            )
            contribution = down * selected_weights.unsqueeze(-1)
            output.index_add_(0, selected_tokens, contribution)

    return output.to(hidden_states.dtype)


def reset_hpu_backend_cache() -> None:
    """Clear the lazily created native backend (useful for process-level tests)."""

    global _HPU_BACKEND
    _HPU_BACKEND = None


__all__ = [
    "BlockFP8Backend",
    "CpuBlockFP8Backend",
    "FP8_E4M3FN",
    "HPUBlockFP8Backend",
    "block_fp8_gemm",
    "glm53_fp8_moe",
    "glm_swiglu_clamp",
    "reset_hpu_backend_cache",
    "validate_hpu_fp8_schemas",
]
