# SPDX-License-Identifier: Apache-2.0
"""Bounded multi-request packed-KV consumption for ordinary C1 decode.

Each query row is an independent request, with its own absolute position,
SWA ring slot and page table. No historical-length-dependent graph shapes.
"""

import torch

from vllm_gaudi.ops.deepseek_v41_capacity import MAX_DSV41_NATIVE_STATE_ROWS

_MLA_TILE = 8
_MLA_ROWS_PER_LANE = 6
_MLA_TILE_PLAN_BUCKETS = (1, 2, 4, 8, 16, 24)


def write_state_rows(cache, values, rows):
    """Write selected state rows through the native 64-row ABI.

    Target rounds own ``6*N`` rows, so a capacity above ten must be split at
    the Python call boundary. Completion values are concatenated in row order
    and retain their per-chunk producer dependency.
    """
    if (not isinstance(rows, torch.Tensor) or rows.ndim != 1 or rows.numel() < 1
            or not isinstance(values, torch.Tensor) or values.ndim < 1
            or values.shape[0] != rows.numel()):
        raise ValueError("state row writes require matching nonempty values and row ids")
    outputs = []
    for start in range(0, rows.numel(), MAX_DSV41_NATIVE_STATE_ROWS):
        stop = min(start + MAX_DSV41_NATIVE_STATE_ROWS, rows.numel())
        outputs.append(
            torch.ops.custom_op.custom_deepseek_v41_state_rows_write_gaudi2(
                cache, values[start:stop].contiguous(), rows[start:stop].contiguous()
            )
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


def read_state_rows(cache, rows, completion):
    """Read selected state rows through the native 64-row ABI."""
    if (not isinstance(rows, torch.Tensor) or rows.ndim != 1 or rows.numel() < 1
            or not isinstance(completion, torch.Tensor) or completion.shape != rows.shape):
        raise ValueError("state row reads require matching nonempty row ids and completions")
    outputs = []
    for start in range(0, rows.numel(), MAX_DSV41_NATIVE_STATE_ROWS):
        stop = min(start + MAX_DSV41_NATIVE_STATE_ROWS, rows.numel())
        outputs.append(
            torch.ops.custom_op.custom_deepseek_v41_state_rows_read_gaudi2(
                cache, rows[start:stop].contiguous(), completion[start:stop].contiguous()
            )
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


def batch_physical_rows(rows, pages, ratio):
    width = 128 // ratio
    logical = rows.clamp_min(0)
    page = pages.gather(1, (logical // width).long().clamp_max(pages.shape[1] - 1))
    valid = (rows >= 0) & (logical < pages.shape[1] * width) & (page > 0)
    return torch.where(valid, page * width + logical.remainder(width), -1).int()


def batch_packed_mla_tile_plan(active_lanes, *, total_rows, tile=_MLA_TILE, device=None):
    """Build a fixed-bucket int32 tile-id vector from host active lanes.

    The retained packed-MLA path partitions the original request-major rows
    into fixed eight-row tiles.  A six-row lane can cross one tile boundary, so
    the plan is computed from row coverage, not from lane ids divided by eight.
    The returned vector always has one of the fixed bucket lengths
    ``1/2/4/8/16/24`` (subject to the total tile count).  Missing slots are
    filled with unique inactive tile ids, which lets the compiled graph depend
    only on the vector length rather than on a particular active-lane set.

    This helper is host-side by design.  ``batch_packed_mla`` consumes the
    returned device tensor as runtime data and never converts it back to host.
    It intentionally supports the N32-compatible complete-tile contract only;
    callers with a partial tail tile should keep the default full-row path.
    """
    if tile != _MLA_TILE:
        raise ValueError("packed MLA tile plan requires the retained tile size of eight")
    if type(total_rows) is not int or total_rows < 1 or total_rows % _MLA_ROWS_PER_LANE:
        raise ValueError("packed MLA tile plan total_rows must be a positive multiple of six")
    if total_rows % tile:
        raise ValueError("packed MLA tile plan requires complete eight-row tiles")

    total_tiles = total_rows // tile
    if total_tiles not in _MLA_TILE_PLAN_BUCKETS:
        raise ValueError(
            "packed MLA tile plan requires a supported complete-tile bucket, "
            f"got total_tiles={total_tiles}"
        )

    try:
        lanes = tuple(active_lanes)
    except TypeError as error:
        raise TypeError("packed MLA active_lanes must be a sequence of lane ids") from error
    if not lanes:
        raise ValueError("packed MLA tile plan requires at least one active lane")
    if any(type(lane) is not int for lane in lanes):
        raise TypeError("packed MLA active_lanes must contain integers")
    capacity = total_rows // _MLA_ROWS_PER_LANE
    if any(lane < 0 or lane >= capacity for lane in lanes):
        raise ValueError("packed MLA active lane is outside the fixed row capacity")
    if tuple(sorted(set(lanes))) != lanes:
        raise ValueError("packed MLA active_lanes must be unique and strictly increasing")

    occupied = set()
    for lane in lanes:
        first = lane * _MLA_ROWS_PER_LANE
        last = first + _MLA_ROWS_PER_LANE - 1
        occupied.update(range(first // tile, last // tile + 1))

    bucket = next(
        (size for size in _MLA_TILE_PLAN_BUCKETS if size >= len(occupied) and size <= total_tiles),
        None,
    )
    if bucket is None:
        raise ValueError(
            "packed MLA active lanes do not fit a supported complete-tile bucket"
        )
    fillers = [tile_id for tile_id in range(total_tiles) if tile_id not in occupied][:bucket - len(occupied)]
    if len(occupied) + len(fillers) != bucket:
        raise ValueError("packed MLA tile plan cannot fill a unique inactive-tile bucket")
    selected = sorted((*occupied, *fillers))
    return torch.tensor(selected, dtype=torch.int32, device=device)


def batch_packed_mla(query,
                     swa,
                     main,
                     selected,
                     pages,
                     positions,
                     slots,
                     sink,
                     scale,
                     swa_done,
                     main_done,
                     *,
                     ratio,
                     window=128,
                     tile=8,
                     fused_gather=False,
                     sram_gather=False,
                     vector_gather=False,
                     tile_indices=None):
    """Packed KV -> shared BF16 rows -> QK/PV MME for independent requests.

    Completion values participate in row eligibility, connecting mutations to
    the first device consumer. -1 is a completed no-write (e.g. an unfinished
    compression pair), while padding is independently masked by ownership.
    The internal query tile bounds SRAM; it is not the scheduler batch size.
    ``tile_indices`` is an optional runtime-data plan produced by
    :func:`batch_packed_mla_tile_plan`.  When supplied, only the listed
    original eight-row tiles are gathered and executed, then scattered back to
    the fixed query shape.  Omitted rows remain zero.  The default ``None``
    path is unchanged.
    """
    if (ratio not in (0, 1, 2) or tile not in (4, 8, 16) or window != 128 or query.ndim != 3
            or query.shape[0] < 1 or query.shape[-1] != 512):
        raise ValueError("Invalid batched C1 MLA contract")
    batch = query.shape[0]
    if sram_gather and (not fused_gather or tile > 8):
        raise ValueError("SRAM MLA requires fused decoding and a tile of at most eight requests")
    if vector_gather and not sram_gather:
        raise ValueError("Vector MLA decoding requires the packed SRAM consumer")
    if tile_indices is not None:
        if tile != _MLA_TILE:
            raise ValueError("tile_indices requires the retained packed MLA tile size of eight")
        if batch % tile:
            raise ValueError("tile_indices requires a complete eight-row query tail")
        if (not isinstance(tile_indices, torch.Tensor) or tile_indices.ndim != 1
                or tile_indices.numel() < 1 or tile_indices.dtype != torch.int32
                or tile_indices.device != query.device or not tile_indices.is_contiguous()):
            raise ValueError("tile_indices must be a nonempty contiguous int32 vector on the query device")
        if tile_indices.numel() not in _MLA_TILE_PLAN_BUCKETS:
            raise ValueError("tile_indices must use one of the fixed tile-plan bucket lengths")
        if tile_indices.numel() > batch // tile:
            raise ValueError("tile_indices contains more tiles than the packed query")
        # CPU/eager validation catches a malformed caller-provided plan without
        # introducing a tensor-value guard into the compiled HPU graph.  The
        # production contract is the host-side helper above, which emits unique
        # strictly-increasing ids; the compiled branch consumes those ids as
        # runtime data and never inspects their values in Python.
        if tile_indices.device.type == "cpu" and not torch.compiler.is_compiling():
            values = tile_indices.tolist()
            if any(value < 0 or value >= batch // tile for value in values):
                raise ValueError("tile_indices contains an out-of-range tile id")
            if any(left >= right for left, right in zip(values, values[1:])):
                raise ValueError("tile_indices must be unique and strictly increasing")
    if (positions.shape != (batch, ) or slots.shape != (batch, ) or pages.ndim != 2 or pages.shape[0] != batch
            or swa_done.shape != (batch, ) or main_done.shape != (batch, ) or selected.shape != (batch, 512)
            or swa.shape[1] != 528 or swa.shape[0] % 256):
        raise ValueError("Batch MLA request metadata does not match query rows")
    active = (positions >= 0) & (slots >= 0) & (slots < swa.shape[0] // 256)
    ready = (swa_done >= -1) & (main_done >= -1) & active
    window_rows = positions[:, None] - window + 1 + torch.arange(window, device=query.device, dtype=torch.int32)
    swa_rows = slots[:, None] * 256 + window_rows.remainder(256)
    swa_rows = torch.where((window_rows >= 0) & ready[:, None], swa_rows, -1)
    width = window
    if ratio:
        physical = batch_physical_rows(selected, pages, ratio)
        visible = selected < (positions[:, None] + 1) // ratio
        physical = torch.where((physical >= 0) & visible & ready[:, None], physical + swa.shape[0], -1)
        rows = torch.cat((swa_rows, physical), -1).int()
        width += 512
    else:
        rows = swa_rows.int()
    outputs = []
    op = (torch.ops.custom_op.custom_deepseek_v41_batch_packed_vector_mla_mme_gaudi2
          if vector_gather else torch.ops.custom_op.custom_deepseek_v41_batch_packed_sram_mla_mme_gaudi2
          if sram_gather else torch.ops.custom_op.custom_deepseek_v41_batch_packed_mla_mme_gaudi2
          if fused_gather else torch.ops.custom_op.custom_deepseek_v41_batch_paged_mla_mme_gaudi2)
    if tile_indices is not None:
        # The native batch MLA node has no cross-tile state: it reads the
        # shared SWA/main caches and produces one output for the current tile.
        # Keep the original tile boundaries and within-tile row order so a
        # compact subset is numerically identical to the same tile in the full
        # path.  tile_indices is runtime device data; the loop extent alone
        # determines the graph bucket and is never derived from active_lanes.
        output = torch.zeros_like(query)
        offsets = torch.arange(tile, dtype=torch.int64, device=query.device)
        for index in range(tile_indices.numel()):
            tile_index = tile_indices[index].to(torch.int64)
            row_indices = tile_index * tile + offsets
            current = rows.index_select(0, row_indices).contiguous()
            local = torch.arange(tile * width, device=query.device,
                                 dtype=torch.int32).reshape(tile, width)
            indices = torch.where(current >= 0, local, -1).contiguous()
            lengths = torch.where(active.index_select(0, row_indices), width, 0).int().contiguous()
            tile_output = op(
                query.index_select(0, row_indices).contiguous(),
                swa,
                main,
                current.reshape(1, -1),
                indices,
                sink,
                scale,
                lengths,
            )
            output.index_copy_(0, row_indices, tile_output)
        return torch.where(active[:, None, None], output, torch.zeros_like(output))

    for start in range(0, batch, tile):
        stop = min(batch, start + tile)
        current = rows[start:stop].contiguous()
        local = torch.arange((stop - start) * width, device=query.device,
                             dtype=torch.int32).reshape(stop - start, width)
        indices = torch.where(current >= 0, local, -1).contiguous()
        lengths = torch.where(active[start:stop], width, 0).int().contiguous()
        output = op(query[start:stop].contiguous(), swa, main, current.reshape(1, -1), indices, sink, scale, lengths)
        outputs.append(output)
    return torch.cat(outputs, 0)
