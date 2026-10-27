# SPDX-License-Identifier: Apache-2.0
"""Compact-row target compressor for the optional ragged target path.

The fixed target body keeps ``[N,6]`` lane ownership: lane ``i`` owns rows
``6*i:6*(i+1)`` and every row without an owner has ``position=-1``.  The
compressor in that layout mixes two kinds of work:

* row-local arithmetic: the ratio-two KV/score input projection, the non-two
  ``wkv`` projection and norm, the indexer projection/norm, RoPE and the FP4
  cache codec.  Each output row depends only on its own input row.  The
  ratio-two projection nevertheless stays on the fixed body (see Numerics).
* lane-local state: the ratio-two history publish/read, the masked native
  sequence mix, the compiled C2-C6 mix and the C1/C2-C6 selection.  These
  depend on the row order inside one lane and on ``position=-1`` no-write
  sentinels, so they stay on the fixed ``[N,6]`` layout.

:func:`compress_target_query_rows` runs the row-local work only on the compact
query rows described by :class:`TargetQueryPlan` (``source_rows`` + ``valid``)
and scatters to the fixed layout exactly where lane-local state or the final
cache row-id ABI is consumed.  Rows that are not valid compact queries
(padding inside a lane, inactive lanes and padded query slots) are always
no-write, and their projected values never reach state.

Numerics: every row-local step is mathematically row independent, but a GEMM
over ``Q`` rows is not guaranteed to round like the same rows inside a
``6*N``-row GEMM.  HPU qualification (2026-10-28) found the FP32
``[1024, 5120]`` ratio-two projection not invariant in the row count or the
graph context, while the K=512 indexer projection and the norms were invariant
and the BF16 K=5120 ``wkv`` differed only in rare single-ulp cases.  The
ratio-two projection feeds persistent history and its per-lane state work
dominates the cost, so it keeps the fixed ``6*N`` input exactly as the C6
compressor does; unowned rows carry position ``-1`` and are neither sequence
partners nor published.  The remaining compact-row GEMMs may still differ from
C6 like a change of the target execution bucket.
"""
from __future__ import annotations

import torch

TARGET_ROWS_PER_LANE = 6
TARGET_RAGGED_COMPRESSOR_KEY = "dsv41_dspark_target_ragged_compressor"


def validate_target_ragged_compressor(enabled, *, query_bucketing):
    """Validate the additional-config value for the ragged compressor."""
    if type(enabled) is not bool:
        raise TypeError(f"{TARGET_RAGGED_COMPRESSOR_KEY} must be a boolean")
    if enabled and not query_bucketing:
        raise ValueError(f"{TARGET_RAGGED_COMPRESSOR_KEY} requires dsv41_dspark_target_query_buckets")
    return enabled


def _validate_layout(value, positions, slots, pages, active, request_lengths, request_slots, query_indices,
                     query_valid):
    request_slots = tuple(int(slot) for slot in request_slots)
    capacity = len(request_slots)
    body_rows = TARGET_ROWS_PER_LANE * capacity
    if tuple(request_lengths) != (TARGET_ROWS_PER_LANE, ) * capacity:
        raise ValueError("ragged target compressor keeps six physical state rows per lane")
    if (capacity < 1 or len(set(request_slots)) != capacity
            or any(slot < 0 or slot >= capacity for slot in request_slots)):
        raise ValueError("ragged target compressor slots must be fixed distinct lane owners")
    if (value.shape[0] != body_rows or positions.shape != (body_rows, ) or slots.shape != (body_rows, )
            or active.shape != (body_rows, ) or pages.shape[0] != body_rows):
        raise ValueError("ragged target compressor inputs must match the fixed 6*N body")
    if (not isinstance(query_indices, torch.Tensor) or query_indices.ndim != 1
            or query_indices.dtype != torch.int64 or query_indices.device != value.device
            or not 1 <= query_indices.numel() <= body_rows or not isinstance(query_valid, torch.Tensor)
            or query_valid.dtype != torch.bool or query_valid.shape != query_indices.shape
            or query_valid.device != value.device):
        raise ValueError("ragged target compressor requires int64 query indices and bool validity on the input device")
    return request_slots, body_rows


def compressor_query_ownership(active, slots, positions, query_indices, query_valid):
    """Return ``(owned_body, owned_query, owned_positions)`` for one round.

    A body row may publish compressor state only when it is an active owned
    row *and* a valid compact query.  Under the ``DSparkTargetBatch`` contract
    every owned row is a valid query and every other row already has
    ``position=-1``, so ``owned_positions`` equals the fixed C6 positions.  The
    coverage term keeps a malformed plan from publishing a row whose projection
    was never computed.
    """
    covered = torch.zeros(positions.shape, dtype=torch.bool, device=positions.device)
    covered = covered.index_copy(0, query_indices, query_valid)
    owned = active.to(torch.bool) & (slots >= 0) & covered
    owned_positions = torch.where(owned, positions, torch.full_like(positions, -1))
    return owned, owned.index_select(0, query_indices), owned_positions


def compress_target_query_rows(attention, value, positions, slots, pages, active, request_lengths, request_slots,
                               query_indices, query_valid, *, runtime_request_slots=False):
    """Compact-row equivalent of ``PagedCSA2Attention._compress_batch_target``.

    ``value``/``positions``/``slots``/``pages``/``active`` use the fixed
    ``6*N`` body.  ``query_indices``/``query_valid`` are the device form of a
    :class:`TargetQueryPlan`.  Row-local projections run on ``Q`` rows except
    the ratio-two input projection, which keeps the ``6*N`` body; the
    ratio-two history publish/read and both mix arms run once per fixed
    six-row lane; the cache writes use the fixed ``6*N`` row-id vector so the
    completion tensors keep the C6 shape consumed by later layers.

    Returns ``(main_done, index_done)`` with shape ``[6*N]``.
    """
    from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_physical_rows, write_state_rows
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, rms_norm
    from vllm_gaudi.ops.deepseek_v41_target_queries import restore_target_query_rows

    request_slots, body_rows = _validate_layout(value, positions, slots, pages, active, request_lengths,
                                                request_slots, query_indices, query_valid)
    owned, query_owned, owned_positions = compressor_query_ownership(active, slots, positions, query_indices,
                                                                     query_valid)
    query_positions = owned_positions.index_select(0, query_indices).clamp_min(0)
    compressor = attention.weights.compressor
    if attention.ratio == 2:
        from vllm_gaudi.ops.deepseek_v41_batch_target import (
            compressor_sequence_runtime_slot_publish_read,
            compressor_sequence_slot_native_mix,
            compressor_sequence_slot_publish_read,
            compressor_sequence_slot_torch_mix,
            select_compressor_norm_by_request_length,
        )

        # Same 6N projection as the C6 compressor: its FP32 GEMM bits depend on
        # the row count on HPU and would otherwise enter persistent history.
        # Unowned rows keep position -1, so they neither publish nor act as a
        # native sequence partner whatever their projected values are.
        kv, score = attention._project_compressor_input(value)
        publish_read = (compressor_sequence_runtime_slot_publish_read
                        if runtime_request_slots else compressor_sequence_slot_publish_read)
        native_norms, compiled_norms = [], []
        for request_index, slot in enumerate(request_slots):
            start = request_index * TARGET_ROWS_PER_LANE
            stop = start + TARGET_ROWS_PER_LANE
            lane_positions = owned_positions[start:stop]
            history, score_history = publish_read(
                attention.batch_state.kv_history,
                attention.batch_state.score_history,
                slots[start] if runtime_request_slots else slot,
                kv[start:stop],
                score[start:stop],
                lane_positions,
            )
            native_norms.append(rms_norm(
                compressor_sequence_slot_native_mix(history, score_history, kv[start:stop], score[start:stop],
                                                    lane_positions),
                compressor.norm.weight,
                attention.eps,
            ))
            compiled_norms.append(rms_norm(
                compressor_sequence_slot_torch_mix(history, score_history, lane_positions),
                compressor.norm.weight,
                attention.eps,
            ))
        # Same post-norm C1/C2-C6 selection as the fixed path, then gather.
        latent = select_compressor_norm_by_request_length(torch.cat(native_norms, 0), torch.cat(compiled_norms, 0),
                                                          owned)
        query_latent = latent.index_select(0, query_indices)
        query_first = query_positions - query_positions.remainder(2)
    else:
        query_first = query_positions
        query_value = value.index_select(0, query_indices)
        query_latent = rms_norm(attention.linear(query_value, compressor.wkv), compressor.norm.weight, attention.eps)
    indexer = attention.weights.indexer
    query_index = rms_norm(attention.linear(query_latent, indexer.wk), indexer.k_norm.weight, attention.eps)
    query_index = attention._rope(query_index, query_first, request_batch=True)
    query_latent = attention._rope(query_latent, query_first, request_batch=True)
    main_rows = restore_target_query_rows(pack_fp4(query_latent, 16), query_indices, query_owned, body_rows)
    index_rows = restore_target_query_rows(pack_fp4(query_index, 32), query_indices, query_owned, body_rows)
    safe_positions = owned_positions.clamp_min(0)
    visible = owned & ((safe_positions + 1).remainder(attention.ratio) == 0)
    physical = batch_physical_rows((safe_positions // attention.ratio).reshape(-1, 1), pages, attention.ratio).flatten()
    rows = torch.where(visible, physical, -1).int()
    main_done = write_state_rows(attention.cache.main, main_rows, rows)
    index_done = write_state_rows(attention.cache.index, index_rows, rows)
    return main_done, index_done


__all__ = [
    "TARGET_RAGGED_COMPRESSOR_KEY",
    "compress_target_query_rows",
    "compressor_query_ownership",
    "validate_target_ragged_compressor",
]
