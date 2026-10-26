# SPDX-License-Identifier: Apache-2.0
"""Retain the pure C6 expert compound recipe without changing eager settings."""

from functools import lru_cache

import torch


def _prequant_moe(*args):
    return torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_prequant_fused_fp8_gaudi2(*args)


@lru_cache(maxsize=1)
def _compiled_prequant_moe():
    # Keep all weights and routing values as inputs. Only the compound op is
    # compiled; router arithmetic and surrounding C6 rounding stay unchanged.
    return torch.compile(_prequant_moe, backend="hpu_backend", fullgraph=True, dynamic=False)


def c6_prequant_moe(*args):
    return _compiled_prequant_moe()(*args)
