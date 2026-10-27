# SPDX-License-Identifier: Apache-2.0
"""Portable GLM-5.3-Flash MoE configuration helpers.

These helpers deliberately avoid importing vLLM or Habana runtime modules so
the dispatch decisions can be tested on CPU without initializing a device.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

_MISSING = object()


@dataclass(frozen=True)
class GlmSwiGLUClamp:
    """Validated GLM routed-expert activation parameters."""

    limit: float
    alpha: float = 1.0
    beta: float = 0.0


def _option_holders(layer: Any):
    """Yield objects that may hold RoutedExperts/FusedMoE options.

    Upstream has moved these fields more than once: current RoutedExperts keeps
    them both directly and in ``moe_config``.  A wrapper MoERunner may hold the
    RoutedExperts as ``routed_experts``.  Check all known holders, preferring the
    nearest non-None value.
    """
    seen: set[int] = set()
    pending = [layer, getattr(layer, "routed_experts", None)]
    while pending:
        holder = pending.pop(0)
        if holder is None or id(holder) in seen:
            continue
        seen.add(id(holder))
        yield holder
        moe_config = getattr(holder, "moe_config", None)
        if moe_config is not None and id(moe_config) not in seen:
            seen.add(id(moe_config))
            yield moe_config


def _resolve_option(layer: Any, name: str, default: Any = None) -> Any:
    for holder in _option_holders(layer):
        value = getattr(holder, name, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    return default


def resolve_glm_clamp_config(layer: Any, activation: Any) -> GlmSwiGLUClamp | None:
    """Resolve the clamp-aware path for GLM's ordinary SiLU experts.

    GLM passes ``swiglu_limit`` to FusedMoE without changing the default
    activation from ``"silu"``.  vLLM's generic kernels interpret that as
    ``silu_with_clamp``; the HPU monolithic path must make the same decision.

    MiniMax-style ``swigluoai_uninterleave`` is intentionally excluded here so
    its existing alpha/beta defaults remain unchanged.
    """
    act = getattr(activation, "value", activation)
    if act != "silu":
        return None

    limit = _resolve_option(layer, "swiglu_limit")
    if limit is None:
        return None
    limit = float(limit)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(f"swiglu_limit must be a finite positive value, got {limit!r}")

    alpha_value = _resolve_option(layer, "swiglu_alpha")
    beta_value = _resolve_option(layer, "swiglu_beta")
    alpha = 1.0 if alpha_value is None else float(alpha_value)
    beta = 0.0 if beta_value is None else float(beta_value)
    if alpha != 1.0 or beta != 0.0:
        raise NotImplementedError(
            "HPU's ordinary silu + swiglu_limit path is validated only for GLM's "
            f"alpha=1, beta=0 semantics, got alpha={alpha}, beta={beta}"
        )

    is_act_and_mul = _resolve_option(layer, "is_act_and_mul", True)
    if not is_act_and_mul:
        raise NotImplementedError("swiglu_limit on HPU requires a gated expert activation")
    return GlmSwiGLUClamp(limit=limit, alpha=alpha, beta=beta)


@dataclass(frozen=True)
class Glm53Fp8MoeSpec:
    """Configuration eligibility for the eager candidate, not device qualification."""

    limit: float
    local_experts: int
    global_experts: int
    ep_rank: int
    ep_size: int
    hidden_size: int
    intermediate_size: int


def require_fp8_clamped_moe_supported(
    layer: Any,
    quant_config: Any = None,
    *,
    is_gaudi2: bool = False,
    scale_adjustment: bool = False,
    enforce_eager: bool = False,
    force_channel_fp8: bool = False,
    model_type: str | None = None,
) -> Glm53Fp8MoeSpec | None:
    """Check configuration only; never inspect unloaded expert weight payloads.

    No clamp means the existing native path. A clamp must satisfy every
    candidate constraint; a failed check must never fall back to native MoE.
    Passing this check does not assert HPU numerical or full-model validation.
    """
    if _resolve_option(layer, "swiglu_limit") is None:
        return None

    def require(condition: bool, reason: str) -> None:
        if not condition:
            raise NotImplementedError(
                f"HPU GLM53 FP8 W8A8 clamp candidate requires {reason}; "
                "refusing an unclamped native MoE or W8A16 fallback"
            )

    require(model_type in ("glm5_next", "glm5_next_text"), "a GLM5Next model configuration")
    clamp = resolve_glm_clamp_config(layer, _resolve_option(layer, "activation"))
    require(clamp is not None, "ordinary gated SiLU (alpha=1, beta=0)")
    require(clamp.limit == 10.0, "swiglu_limit=10")
    require(getattr(quant_config, "is_checkpoint_fp8_serialized", False), "serialized FP8 weights")
    require(is_gaudi2, "Gaudi2")
    require(scale_adjustment, "scale adjustment enabled for paired half-weight/double-scale loading")
    require(getattr(quant_config, "activation_scheme", None) == "dynamic", "dynamic activation quantization")
    require(getattr(quant_config, "weight_block_size", None) in ([128, 128], (128, 128)), "raw [128,128] blocks")
    require(not force_channel_fp8, "force-channel FP8 disabled to preserve checkpoint blocks")
    require(enforce_eager, "enforce_eager=True (dynamic route selection cannot be captured or compiled)")

    config = getattr(layer, "moe_config", None)
    parallel = getattr(config, "moe_parallel_config", None)
    require(not any(getattr(holder, "enable_eplb", False) for holder in (layer, config, parallel)), "EPLB disabled")
    require(not any(getattr(holder, "num_redundant_experts", 0) for holder in (layer, config, parallel)),
            "no redundant experts")
    require(not _resolve_option(layer, "has_bias", False), "bias-free experts")
    require(not _resolve_option(layer, "is_lora_enabled", False), "LoRA disabled")
    require(not _resolve_option(layer, "apply_router_weight_on_input", False), "routing weights applied after W2")
    require(getattr(layer, "expert_placement_strategy", "linear") == "linear", "linear expert placement")

    local = getattr(layer, "local_num_experts", None)
    total = getattr(layer, "global_num_experts", None)
    ep_rank = getattr(config, "ep_rank", None)
    ep_size = getattr(config, "ep_size", None)
    hidden = getattr(config, "hidden_dim", None)
    intermediate = getattr(config, "intermediate_size_per_partition", None)
    for name, value in (("local experts", local), ("global experts", total), ("EP size", ep_size),
                        ("hidden size", hidden), ("intermediate partition size", intermediate)):
        require(type(value) is int and value > 0, f"a positive {name}")
    require(type(ep_rank) is int and 0 <= ep_rank < ep_size, "a valid EP rank")
    require(total == local * ep_size, "equal contiguous expert partitions")
    require(getattr(config, "num_logical_experts", None) == total, "physical experts equal logical experts")
    require(not _resolve_option(layer, "num_fused_shared_experts", 0), "no fused shared experts")
    require(hidden % 128 == 0 and intermediate % 128 == 0, "128-aligned hidden and intermediate partitions")
    return Glm53Fp8MoeSpec(clamp.limit, local, total, ep_rank, ep_size, hidden, intermediate)


__all__ = [
    "GlmSwiGLUClamp",
    "Glm53Fp8MoeSpec",
    "require_fp8_clamped_moe_supported",
    "resolve_glm_clamp_config",
]
