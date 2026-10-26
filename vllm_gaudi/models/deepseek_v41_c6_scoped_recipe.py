# SPDX-License-Identifier: Apache-2.0
"""Bounded C6 scoped recipes for request-local target math.

The recipes in this module deliberately contain only the six-row mHC and
normalization/quantization chains around attention.  Attention, Engram,
router/MoE, target-state collection, and final collapse stay in their existing
eager callers.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import logging

import torch

from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.extension.logger import logger as runtime_logger
from vllm_gaudi.ops.deepseek_v41_math import hc_pre, rms_norm

logger = logging.getLogger(__name__)

RECIPE_VARIANTS = (
    "prefix_no_collapse",
    "prefix_with_collapse",
    "post_pre_moe",
)

_ENV_FLAGS = (
    "VLLM_HPU_DSV41_DSPARK",
    "VLLM_HPU_DSV41_DSPARK_INPUT_NORM",
    "VLLM_HPU_DSV41_DSPARK_MHC_WEIGHT_REUSE",
    "VLLM_HPU_DSV41_DSPARK_MHC_CONTROL_TILES",
    "VLLM_HPU_DSV41_DSPARK_MHC_MME_EPILOGUE",
    "VLLM_HPU_DSV41_DSPARK_MHC_HIGH_PLANE",
    "VLLM_HPU_DSV41_MHC_CONTROL_RRMS",
    "VLLM_HPU_DSV41_MHC_PARALLEL_CONTROL",
    "VLLM_HPU_DSV41_MHC_LINEAR_LOAD",
    "VLLM_HPU_DSV41_TPC_MHC",
    "VLLM_HPU_DSV41_MHC_GATES_FUSED",
    "VLLM_HPU_DSV41_MHC_POSITIVE_GATES",
    "VLLM_HPU_DSV41_FFN_DUAL_QUANT",
    "VLLM_HPU_DSV41_FFN_BF16_QUANT",
    "VLLM_HPU_DSV41_DSPARK_NORM_ROUNDTRIP",
    "VLLM_HPU_DSV41_DSPARK_INPUT_QUANT_REMAT",
    "VLLM_HPU_DSV41_DSPARK_CONTROL_FP8",
    "VLLM_HPU_DSV41_DSPARK_CONTROL_FP8_PAIR",
    "VLLM_HPU_DSV41_DSPARK_MHC_DEFERRED",
    "VLLM_HPU_DSV41_MHC_SCHEDULE",
    "VLLM_HPU_DSV41_MHC_BATCH_REUSE",
    "VLLM_HPU_DSV41_MHC_CONTROL_MME",
)


def decode_mhc_post_collapse(value, residual, post, comb, new_pre):
    """Fuse the C6 attention post/collapse while preserving its BF16 boundary."""
    return torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
        value.contiguous(), residual.contiguous(), post.contiguous(), comb.contiguous(), new_pre.contiguous()
    )


def decode_ffn_dual_quant(value, weight, eps):
    """Produce the C6 routed and shared N256 FP8 operands together."""
    return torch.ops.custom_op.custom_deepseek_v41_ffn_norm_dual_quant_gaudi2(
        value.contiguous(), weight, eps
    )


def _ffn_norm_quant(value, weight, eps):
    return torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
        value.contiguous(), weight, eps
    )


def _attention_norm_quant(value, weight, eps):
    return torch.ops.custom_op.custom_deepseek_v41_attention_norm_quant_gaudi2(
        value.contiguous(), weight, eps
    )


def _bf16_boundary(value):
    # Preserve the existing collapse-to-BF16 rounding before RMSNorm widens
    # back to FP32. Otherwise the HPU compiler can fold that cast pair.
    shape = value.shape
    return torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(
        value.reshape(1, -1).contiguous()).reshape(shape)


def _c6_prefix_math(
    residual6,
    previous_pre6,
    collapsed_attention6,
    hc_fn,
    hc_scale,
    hc_base,
    hc_packed,
    attn_norm_weight,
    eps,
    hc_eps,
    iterations,
    norm_mode="fp8",
    control_mme_weight=None,
):
    """Run the exact six-row prefix math with all weights passed explicitly."""
    value, new_pre, post, comb = hc_pre(
        residual6,
        previous_pre6,
        hc_fn,
        hc_scale,
        hc_base,
        eps,
        hc_eps,
        iterations,
        packed_fn=hc_packed,
        decode=True,
        collapsed_input=collapsed_attention6,
        batch_control_reuse=False,
        control_mme_weight=control_mme_weight,
        control_epilogue=False,
        control_fp8_weight=None,
    )
    if norm_mode == "fp8":
        value, quantized, activation_scale = _attention_norm_quant(value.contiguous(), attn_norm_weight, eps)
    else:
        if collapsed_attention6 is None:
            value = _bf16_boundary(value)
        value = rms_norm(value, attn_norm_weight, eps, native_decode=norm_mode == "bf16_native")
        quantized, activation_scale = None, None
    return value, new_pre, post, comb, quantized, activation_scale


def c6_prefix_no_collapse(
    residual6,
    previous_pre6,
    hc_fn,
    hc_scale,
    hc_base,
    hc_packed,
    attn_norm_weight,
    eps,
    hc_eps,
    iterations,
    norm_mode="fp8",
    control_mme_weight=None,
):
    """Prefix recipe for a layer that has no preceding collapsed residual."""
    return _c6_prefix_math(
        residual6,
        previous_pre6,
        None,
        hc_fn,
        hc_scale,
        hc_base,
        hc_packed,
        attn_norm_weight,
        eps,
        hc_eps,
        iterations,
        norm_mode,
        control_mme_weight,
    )


def c6_prefix_with_collapse(
    residual6,
    previous_pre6,
    collapsed_attention6,
    hc_fn,
    hc_scale,
    hc_base,
    hc_packed,
    attn_norm_weight,
    eps,
    hc_eps,
    iterations,
    norm_mode="fp8",
    control_mme_weight=None,
):
    """Prefix recipe for a layer consuming a resolved collapsed residual."""
    return _c6_prefix_math(
        residual6,
        previous_pre6,
        collapsed_attention6,
        hc_fn,
        hc_scale,
        hc_base,
        hc_packed,
        attn_norm_weight,
        eps,
        hc_eps,
        iterations,
        norm_mode,
        control_mme_weight,
    )


def _c6_post_pre_moe_math(
    attention_value6,
    residual6,
    new_pre6,
    post6,
    comb6,
    hc_fn,
    hc_scale,
    hc_base,
    hc_packed,
    ffn_norm_weight,
    eps,
    hc_eps,
    iterations,
    dual_quant=True,
    control_mme_weight=None,
):
    """Run attention post/collapse, FFN mHC input, and FFN dual quant."""
    residual6, collapsed_ffn = decode_mhc_post_collapse(
        attention_value6, residual6, post6, comb6, new_pre6
    )
    value, pre_mix6, post2, comb2 = hc_pre(
        residual6,
        new_pre6,
        hc_fn,
        hc_scale,
        hc_base,
        eps,
        hc_eps,
        iterations,
        packed_fn=hc_packed,
        decode=True,
        collapsed_input=collapsed_ffn,
        batch_control_reuse=False,
        control_mme_weight=control_mme_weight,
        control_epilogue=False,
        control_fp8_weight=None,
    )
    if dual_quant:
        normalized, quantized, activation_scale, shared_q, shared_scale = decode_ffn_dual_quant(
            value, ffn_norm_weight, eps
        )
    else:
        normalized, quantized, activation_scale = _ffn_norm_quant(value, ffn_norm_weight, eps)
        shared_q, shared_scale = None, None
    return (
        residual6,
        normalized,
        quantized,
        activation_scale,
        shared_q,
        shared_scale,
        pre_mix6,
        post2,
        comb2,
    )


def _tensor_metadata(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"C6 scoped recipe input must be a tensor, got {type(tensor).__name__}")
    if tensor.layout != torch.strided:
        raise ValueError("C6 scoped recipes require strided tensors")
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device.type, tensor.device.index)


def _weight_contract(weights):
    """Describe weight layout and aliasing without pinning tensor addresses."""
    aliases = {}
    contract = []
    for tensor in weights:
        metadata = _tensor_metadata(tensor)
        storage = tensor.untyped_storage()._cdata
        alias = aliases.setdefault(storage, len(aliases))
        contract.append((*metadata, alias))
    return tuple(contract)


def _lane_contract(lanes):
    """Describe lane views without storage identity or lane storage offset."""
    return tuple(_tensor_metadata(tensor) for tensor in lanes)


def _env_signature():
    return tuple((name, bool(getattr(gaudi_envs, name))) for name in _ENV_FLAGS)


class C6ScopedRecipeCache:
    """Share at most eight compiled C6 math variants across target layers.

    The owner generation and precision fingerprint invalidate graphs.  Lane
    views contribute shape/stride/dtype/device only, so lane zero, lane 31,
    and all-lane rounds reuse the same callable.  A compilation or execution
    failure disables the candidate for the rest of the process and the caller
    explicitly falls back to its eager path.
    """

    def __init__(self, owner, *, backend: str = "hpu_backend", max_variants: int = 8):
        if max_variants < len(RECIPE_VARIANTS):
            raise ValueError("C6 scoped recipe cache must hold all three variants")
        self.owner = owner
        self.backend = backend
        self.max_variants = max_variants
        self._entries = OrderedDict()
        self._generation_marker = self._owner_marker()
        self._hits = 0
        self._misses = 0
        self._variant_hits = Counter()
        self._variant_misses = Counter()
        self._ineligible_calls = 0
        self._ineligible_reasons = Counter()
        self._disabled_reason = None
        self._failed_variant = None

    def _owner_marker(self):
        return (
            getattr(self.owner, "generation", None),
            getattr(self.owner, "precision_fingerprint", None),
        )

    def _sync_generation(self):
        marker = self._owner_marker()
        if marker != self._generation_marker:
            self._entries.clear()
            self._generation_marker = marker

    def set_backend(self, backend: str):
        if backend != self.backend:
            self.backend = backend
            self._entries.clear()

    def record_ineligible(self, reason: str):
        self._ineligible_calls += 1
        self._ineligible_reasons[str(reason)] += 1

    def status(self):
        variant_names = sorted(
            variant for entry in self._entries.values() for variant in entry
        )
        if self._disabled_reason is not None:
            state = "disabled"
        elif variant_names:
            state = "active"
        else:
            state = "not_enabled"
        return {
            "state": state,
            "backend": self.backend,
            "generation": self._generation_marker[0],
            "precision_fingerprint": self._generation_marker[1],
            "hits": self._hits,
            "misses": self._misses,
            "compiled_variants": tuple(variant_names),
            "compiled_callable_count": len(variant_names),
            "variant_kinds": tuple(sorted(set(variant_names))),
            "variant_hits": dict(sorted(self._variant_hits.items())),
            "variant_misses": dict(sorted(self._variant_misses.items())),
            "ineligible_calls": self._ineligible_calls,
            "ineligible_reasons": dict(sorted(self._ineligible_reasons.items())),
            "disabled_reason": self._disabled_reason,
            "failed_variant": self._failed_variant,
        }

    def _disable(self, variant: str, error: Exception):
        self._disabled_reason = f"{variant}: {type(error).__name__}: {error}; falling back to eager"
        self._failed_variant = variant
        logger.warning(
            "Disabling C6 scoped recipe after %s compilation/execution failure; "
            "falling back to eager for the rest of the process: %s",
            variant,
            error,
            exc_info=True,
        )

    def _make_room(self, current_key):
        while sum(len(entry) for entry in self._entries.values()) >= self.max_variants:
            oldest = next(key for key in self._entries if key != current_key)
            del self._entries[oldest]

    def _invoke(self, variant, function, arguments, weight_arguments, lane_arguments, constants):
        if self._disabled_reason is not None:
            return None
        self._sync_generation()
        key = (
            self._generation_marker,
            self.backend,
            _weight_contract(weight_arguments),
            _lane_contract(lane_arguments),
            _env_signature(),
            tuple(constants),
        )
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        else:
            entry = {}
            self._entries[key] = entry
        compiled = entry.get(variant)
        if compiled is not None:
            self._hits += 1
            self._variant_hits[variant] += 1
            try:
                return compiled(*arguments)
            except Exception as error:  # pragma: no cover - HPU failure policy
                self._disable(variant, error)
                return None
        self._make_room(key)
        self._misses += 1
        self._variant_misses[variant] += 1
        try:
            compiled = torch.compile(function, backend=self.backend, fullgraph=True, dynamic=False)
            result = compiled(*arguments)
        except Exception as error:
            self._disable(variant, error)
            return None
        entry[variant] = compiled
        runtime_logger().info("Compiled C6 scoped recipe variant %s with backend %s; control=%s",
                              variant, self.backend, "mme" if len(weight_arguments) == 6 else "tpc_rrms")
        return result

    def run_prefix(
        self,
        residual6,
        previous_pre6,
        collapsed_attention6,
        hc_fn,
        hc_scale,
        hc_base,
        hc_packed,
        attn_norm_weight,
        eps,
        hc_eps,
        iterations,
        *,
        norm_mode="fp8",
        control_mme_weight=None,
    ):
        if norm_mode not in ("fp8", "bf16", "bf16_native"):
            raise ValueError(f"Unsupported C6 attention norm mode: {norm_mode}")
        weights = (hc_fn, hc_scale, hc_base, hc_packed, attn_norm_weight)
        lanes = (residual6, previous_pre6)
        if collapsed_attention6 is None:
            variant = "prefix_no_collapse"
            function = c6_prefix_no_collapse
            arguments = (residual6, previous_pre6, *weights, eps, hc_eps, iterations, norm_mode, control_mme_weight)
        else:
            variant = "prefix_with_collapse"
            function = c6_prefix_with_collapse
            lanes = (*lanes, collapsed_attention6)
            arguments = (
                residual6, previous_pre6, collapsed_attention6, *weights,
                eps, hc_eps, iterations, norm_mode, control_mme_weight,
            )
        cache_weights = weights if control_mme_weight is None else (*weights, control_mme_weight)
        return self._invoke(variant, function, arguments, cache_weights, lanes, (eps, hc_eps, iterations, norm_mode))

    def run_post_pre_moe(
        self,
        attention_value6,
        residual6,
        new_pre6,
        post6,
        comb6,
        hc_fn,
        hc_scale,
        hc_base,
        hc_packed,
        ffn_norm_weight,
        eps,
        hc_eps,
        iterations,
        *,
        dual_quant=True,
        control_mme_weight=None,
    ):
        weights = (hc_fn, hc_scale, hc_base, hc_packed, ffn_norm_weight)
        lanes = (attention_value6, residual6, new_pre6, post6, comb6)
        arguments = (*lanes, *weights, eps, hc_eps, iterations, dual_quant, control_mme_weight)
        cache_weights = weights if control_mme_weight is None else (*weights, control_mme_weight)
        return self._invoke(
            "post_pre_moe",
            _c6_post_pre_moe_math,
            arguments,
            cache_weights,
            lanes,
            (eps, hc_eps, iterations, dual_quant),
        )
