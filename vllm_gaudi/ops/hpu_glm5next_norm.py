# SPDX-License-Identifier: Apache-2.0
"""Pure PyTorch normalization used by HPU GLM-5.3 KDA attention."""

from __future__ import annotations

import torch


def rmsnorm_sigmoid_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply RMSNorm followed by the KDA sigmoid output gate.

    The output dtype follows ``x``. Arithmetic is performed in FP32, matching
    ``FusedRMSNormGated.forward_native`` for ``activation="sigmoid"``.
    """
    if x.shape != gate.shape:
        raise ValueError(f"x and gate must have identical shapes, got {tuple(x.shape)} and {tuple(gate.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, got {x.dtype}")
    if not gate.is_floating_point():
        raise TypeError(f"gate must be floating point, got {gate.dtype}")
    if x.device != gate.device:
        raise ValueError(f"x and gate must be on the same device, got {x.device} and {gate.device}")
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError(f"eps must be positive, got {eps!r}")

    x_fp32 = x.float()
    variance = x_fp32.square().mean(dim=-1, keepdim=True)
    normalized = x_fp32 * torch.rsqrt(variance + float(eps))
    if weight is not None:
        if weight.numel() != x.shape[-1]:
            raise ValueError(f"weight must contain {x.shape[-1]} values, got shape {tuple(weight.shape)}")
        normalized = normalized * weight.reshape(-1).float()
    return (normalized * torch.sigmoid(gate.float())).to(x.dtype)


__all__ = ["rmsnorm_sigmoid_gate"]
