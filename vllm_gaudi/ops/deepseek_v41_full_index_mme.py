# SPDX-License-Identifier: Apache-2.0
"""Bounded concurrent Full scoring on MME, with the canonical long scorer.

Both branches have fixed addresses and shapes. Device positions suppress the
packed TPC dot products for requests in the bounded prefix. MME tiles are still
executed for the entire batch: this does not claim conditional MME execution.
The packed page pool remains canonical and no persistent decoded mirror is used.
"""

import torch

from vllm_gaudi.ops.deepseek_v41_indexer import ordered_index_ids, runtime_index_select
from vllm_gaudi.ops.deepseek_v41_reindex_mme import _bf16_boundary, _ordered_reindex_scores

FULL_MME_PREFIX_TOKENS = 4096
# TP8 adaptation branch: the 64K bounded-prefix candidate stays opt-out.
FULL_MME_PREFIX_CHOICES = (FULL_MME_PREFIX_TOKENS,)
FULL_MME_TILE_ROWS = 2048


def full_index_mme_scores(query, weights, cache, pages, positions, *, ratio,
                          prefix_tokens=FULL_MME_PREFIX_TOKENS, tiled_keys=False, local_heads=16):
    """Score one bounded prefix with full K128 and explicit TP shard rounding.

    Positions must already suppress requests outside this prefix. Every tile
    executes even for an inactive row. This intentionally does not unroll a
    capacity-sized (up to 1M) loop or allocate a persistent decoded key mirror.
    """
    if ratio not in (1, 2) or prefix_tokens not in FULL_MME_PREFIX_CHOICES:
        raise ValueError("Full MME requires ratio 1/2 and the supported 4K token prefix")
    if local_heads not in (4, 8, 16):
        raise ValueError("Full MME requires TP2/TP4/TP8 head groups")
    if query.ndim != 3 or query.shape[1:] != (32, 128):
        raise ValueError("Full MME queries must have shape [requests,32,128]")
    ops = torch.ops.custom_op
    visible = ((positions + 1) // ratio)[:, None]
    scores = []
    for begin in range(0, prefix_tokens // ratio, FULL_MME_TILE_ROWS):
        rows = torch.arange(begin, begin + FULL_MME_TILE_ROWS, device=query.device, dtype=torch.int32)[None]
        valid = (rows < visible) & (visible > 512)
        rows = torch.where(valid, rows, -1).contiguous()
        gather = (ops.custom_deepseek_v41_index_keys_tiled_gaudi2
                  if tiled_keys else ops.custom_deepseek_v41_index_keys_gaudi2)
        keys = gather(cache, pages, rows, ratio)
        dots = _bf16_boundary(torch.bmm(query, keys.transpose(1, 2)))
        score = _ordered_reindex_scores(dots, weights, positions, rows, ratio, local_heads)
        scores.append(score.masked_fill(~valid, -torch.inf))
    scores = torch.cat(scores, -1) if len(scores) > 1 else scores[0]
    # This is the maintained bounded-MME publication contract: each block is
    # the max of its eight masked scores; the newest visible block must survive.
    block_scores = scores.reshape(scores.shape[0], -1, 8).amax(-1)
    block_ids = torch.arange(block_scores.shape[-1], dtype=torch.int32, device=query.device)
    newest = ((visible - 1) // 8)
    block_scores = block_scores.masked_fill((visible > 0) & (block_ids == newest), torch.inf)
    return scores, block_scores


def full_index_mme_select(query,
                          weights,
                          cache,
                          pages,
                          positions,
                          candidates,
                          *,
                          ratio,
                          capacity,
                          publish_candidates=False,
                          key_ready=None,
                          tiled_keys=False,
                          local_heads=16,
                          prefix_tokens=FULL_MME_PREFIX_TOKENS):
    if local_heads not in (4, 8, 16):
        raise ValueError("Full MME requires TP2/TP4/TP8 head groups")
    if (ratio not in (1, 2) or prefix_tokens not in FULL_MME_PREFIX_CHOICES
            or capacity < prefix_tokens // ratio):
        raise ValueError("Full MME scoring requires ratio 1/2 and the complete bounded prefix")
    ops = torch.ops.custom_op
    if key_ready is not None:
        # Cache mutation must precede key gather itself, not only the query
        # consumer. -1 denotes a completed compression pair with no write.
        positions = torch.where(key_ready >= -1, positions, -1).contiguous()
    hot = (positions >= 0) & (positions < prefix_tokens)
    hot_positions = torch.where(hot, positions, -1).contiguous()
    long_positions = torch.where(hot, -1, positions).contiguous()
    scores, _ = full_index_mme_scores(
        query, weights, cache, pages, hot_positions, ratio=ratio,
        prefix_tokens=prefix_tokens, tiled_keys=tiled_keys, local_heads=local_heads)
    hot_selected = ordered_index_ids(scores, hot_positions, candidates, ratio)
    selected, blocks = runtime_index_select(query,
                                            weights,
                                            cache,
                                            pages,
                                            long_positions,
                                            candidates,
                                            ratio=ratio,
                                            capacity=capacity,
                                            publish_candidates=publish_candidates)
    selected = torch.where(hot[:, None], hot_selected, selected)
    if publish_candidates:
        # At most 512 visible blocks; the emitter's direct prefix branch
        # does not inspect scores or metadata. Reuse initialized tensors
        # rather than introducing unread, uninitialized graph inputs.
        hot_blocks = ordered_index_ids(scores, hot_positions, candidates, ratio, blocks=True)
        blocks = torch.where(hot[:, None], hot_blocks, blocks)
    return selected, blocks
