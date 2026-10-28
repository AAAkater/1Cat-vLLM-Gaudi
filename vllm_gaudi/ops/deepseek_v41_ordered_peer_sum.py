# SPDX-License-Identifier: Apache-2.0
"""Fixed-rank BF16 peer reduction with a single FP32 accumulation boundary."""
import torch


def ordered_peer_sum_reference(shards):
    if shards.dtype != torch.bfloat16 or shards.ndim != 2 or not 2 <= shards.shape[0] <= 8:
        raise ValueError('Ordered peer sum requires BF16 [ranks, width]')
    total = shards[0].float()
    for rank in range(1, shards.shape[0]):
        total = total + shards[rank].float()
    return total.to(torch.bfloat16).reshape(1, -1)


def ordered_peer_sum(shards):
    return torch.ops.custom_op.custom_deepseek_v41_ordered_peer_sum_gaudi2(shards)


# Largest BF16 message of the native small-message peer path without batch
# decode buffers (see ``stage_collectives``).  One six-row lane is 30720.
NATIVE_PEER_ELEMENTS = 32768


def ordered_rows_reduce(value, reduce, gather, tensor_parallel_size):
    """TP sum of any row count with the native small-message arithmetic.

    The native peer path sums rank-ordered BF16 shards in FP32 and rounds to
    BF16 once.  Messages above its bound fall back to the generic HCCL
    all-reduce, which accumulates in BF16 in an order that depends on each
    element's offset in the buffer: identical rows at different offsets round
    differently.  Gather the exact BF16 partials once (one collective) and
    repeat the rank-ordered FP32 sum so every row matches its six-row result.
    """
    if (tensor_parallel_size <= 1 or value.device.type != "hpu" or value.dtype != torch.bfloat16
            or value.numel() <= NATIVE_PEER_ELEMENTS):
        return reduce(value)
    return ordered_gather_sum(value, gather, tensor_parallel_size, compiled=True)


def ordered_rank_sum(shards):
    """Rank-ordered FP32 sum of BF16 shards ``[ranks, width]`` with one BF16 rounding."""
    total = shards[0].float()
    for rank in range(1, shards.shape[0]):
        total = total + shards[rank].float()
    return total.to(torch.bfloat16)


_rank_sum_graph = {"compiled": None, "disabled_reason": None}


def _compiled_rank_sum(shards):
    """One compiled graph per width instead of 2*ranks eager launches.

    Falls back to the identical eager expression if compilation fails.
    """
    state = _rank_sum_graph
    if state["disabled_reason"] is None and shards.device.type == "hpu":
        try:
            if state["compiled"] is None:
                compiled = torch.compile(ordered_rank_sum, backend="hpu_backend", fullgraph=True, dynamic=False)
                state["compiled"] = compiled
            return state["compiled"](shards)
        except Exception as error:  # pragma: no cover - HPU failure policy
            state["disabled_reason"] = f"{type(error).__name__}: {error}"
    return ordered_rank_sum(shards)


def ordered_gather_sum(value, gather, tensor_parallel_size, *, compiled=False):
    """Gather exact BF16 partials along a new rank axis and sum them in rank order."""
    if value.dtype != torch.bfloat16:
        raise ValueError("Ordered gather sum requires BF16 partials")
    shards = gather(value.reshape(1, -1).contiguous(), 0).reshape(tensor_parallel_size, -1)
    total = _compiled_rank_sum(shards) if compiled else ordered_rank_sum(shards)
    return total.reshape(value.shape)
