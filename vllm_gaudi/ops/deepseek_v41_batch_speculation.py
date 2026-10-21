# SPDX-License-Identifier: Apache-2.0
"""CPU-side packed metadata for the DSpark batch verification stage.

This module only owns request ordering and index construction. It does not
allocate KV state, execute a model, or depend on the vLLM engine runtime.
Padding masks use ``True`` for a padding/no-write position.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
import operator

import torch

_MAX_BATCH_SIZE = 8
_MAX_DRAFT_TOKENS = 5
_DRAFT_PAD = -1
_DEFAULT_DEVICE = torch.device("cpu")


def _as_int(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must contain integers")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must contain integers") from error


def _host_values(value, name):
    """Materialize a one-dimensional host sequence without requiring NumPy."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU host tensor")
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must contain integers")
        return value.detach().tolist()
    try:
        return list(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence") from error


def _integer_values(value, name):
    return [_as_int(item, name) for item in _host_values(value, name)]


def _draft_rows(value, batch_size: int):
    """Return exact draft lengths and their non-padding token values."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 2:
            raise ValueError("draft_token_ids must be a sequence of sequences")
        if value.device.type != "cpu":
            raise ValueError("draft_token_ids must be a CPU host tensor")
        if value.numel() and (value.dtype == torch.bool or value.is_floating_point() or value.is_complex()):
            raise TypeError("draft_token_ids must contain integers")
        rows = value.detach().tolist()
    else:
        try:
            rows = list(value)
        except TypeError as error:
            raise TypeError("draft_token_ids must be a sequence of sequences") from error

    if len(rows) != batch_size:
        raise ValueError(f"draft_token_ids length {len(rows)} does not match batch size {batch_size}")

    result = []
    for request_index, row in enumerate(rows):
        if isinstance(row, torch.Tensor):
            if row.ndim != 1:
                raise ValueError(f"draft_token_ids[{request_index}] must be one-dimensional")
            if row.device.type != "cpu":
                raise ValueError(f"draft_token_ids[{request_index}] must be a CPU host tensor")
            if row.dtype == torch.bool or row.is_floating_point() or row.is_complex():
                raise TypeError(f"draft_token_ids[{request_index}] must contain integers")
            row = row.detach().tolist()
        elif isinstance(row, (str, bytes)):
            raise TypeError(f"draft_token_ids[{request_index}] must be a sequence of integers")
        else:
            try:
                row = list(row)
            except TypeError as error:
                raise TypeError(f"draft_token_ids[{request_index}] must be a sequence") from error

        tokens = [_as_int(token, f"draft_token_ids[{request_index}]") for token in row]
        if len(tokens) > _MAX_DRAFT_TOKENS:
            raise ValueError(f"draft_token_ids[{request_index}] has more than {_MAX_DRAFT_TOKENS} tokens")
        if any(token < 0 for token in tokens):
            raise ValueError(f"draft_token_ids[{request_index}] must be nonnegative")
        result.append(tokens)
    return result


def _starts(lengths: Sequence[int]) -> list[int]:
    return [0, *accumulate(lengths)]


def _host_padding_mask(value, expected_length: int) -> list[bool]:
    if value is None:
        return [False] * expected_length
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError("target_padding_mask must be one-dimensional")
        if value.device.type != "cpu":
            raise ValueError("target_padding_mask must be a CPU host tensor")
        values = value.detach().tolist()
    else:
        try:
            values = list(value)
        except TypeError as error:
            raise TypeError("target_padding_mask must be a sequence") from error
    if len(values) != expected_length:
        raise ValueError(
            f"target_padding_mask length {len(values)} does not match packed target length {expected_length}"
        )
    return [bool(item) for item in values]


def _integer_tensor(value, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU host tensor")
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must contain integers")
        return value.to(device=device, dtype=dtype)
    try:
        values = [_as_int(item, name) for item in list(value)]
        return torch.tensor(values, dtype=dtype, device=device)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence") from error


def padded_draft_tokens(draft_token_ids, pad: int = _DRAFT_PAD, *, device=None):
    """Pack ragged draft IDs into ``int32[B, max_drafts]`` and a valid mask.

    The returned mask is ``True`` for a real draft token and ``False`` for a
    padded lane. Draft counts must be in ``0..5`` and the batch must be
    non-empty.
    """
    pad = _as_int(pad, "pad")
    if device is None:
        device = draft_token_ids.device if isinstance(draft_token_ids, torch.Tensor) else _DEFAULT_DEVICE
    device = torch.device(device)

    if isinstance(draft_token_ids, torch.Tensor):
        batch_size = draft_token_ids.shape[0] if draft_token_ids.ndim > 0 else 0
    else:
        try:
            batch_size = len(draft_token_ids)
        except TypeError as error:
            raise TypeError("draft_token_ids must be a sequence of sequences") from error
    if batch_size < 1:
        raise ValueError("draft_token_ids must contain at least one request")

    rows = _draft_rows(draft_token_ids, batch_size)
    max_drafts = max((len(row) for row in rows), default=0)
    tokens = torch.full((batch_size, max_drafts), pad, dtype=torch.int32, device=device)
    valid = torch.zeros((batch_size, max_drafts), dtype=torch.bool, device=device)
    for request_index, row in enumerate(rows):
        if row:
            tokens[request_index, :len(row)] = torch.tensor(row, dtype=torch.int32, device=device)
            valid[request_index, :len(row)] = True
    return tokens, valid


def accepted_prefix_lengths(target_token_ids, draft_token_ids) -> torch.Tensor:
    """CPU reference oracle for the accepted token count, including the anchor.

    ``target_token_ids`` is ``[B, max_draft + 1]`` or wider and
    ``draft_token_ids`` is ``[B, <=5]``. Draft lanes padded with ``-1`` are
    ignored. The result is ``int32[B]`` in ``1..draft_count + 1``; the first
    mismatch ends acceptance for that request. This helper may synchronize and
    must not be used in a captured graph or hot path.
    """
    if isinstance(target_token_ids, torch.Tensor):
        if target_token_ids.ndim != 2:
            raise ValueError("target_token_ids must be two-dimensional")
        if target_token_ids.device.type != "cpu":
            raise ValueError("target_token_ids must be a CPU host tensor")
        if (
            target_token_ids.dtype == torch.bool
            or target_token_ids.is_floating_point()
            or target_token_ids.is_complex()
        ):
            raise TypeError("target_token_ids must contain integers")
        target = target_token_ids
    else:
        try:
            rows = list(target_token_ids)
        except TypeError as error:
            raise TypeError("target_token_ids must be a sequence of sequences") from error
        target_values = []
        for row in rows:
            if isinstance(row, (str, bytes)):
                raise TypeError("target_token_ids must be a sequence of sequences")
            try:
                row = list(row)
            except TypeError as error:
                raise TypeError("target_token_ids must be a sequence of sequences") from error
            target_values.append([_as_int(token, "target_token_ids") for token in row])
        try:
            target = torch.tensor(target_values, dtype=torch.int64)
        except (RuntimeError, ValueError) as error:
            raise ValueError("target_token_ids must be a rectangular two-dimensional sequence") from error
        if target.ndim != 2:
            raise ValueError("target_token_ids must be two-dimensional")

    if isinstance(draft_token_ids, torch.Tensor):
        if draft_token_ids.ndim != 2:
            raise ValueError("draft_token_ids must be two-dimensional")
        if draft_token_ids.device.type != "cpu":
            raise ValueError("draft_token_ids must be a CPU host tensor")
        if draft_token_ids.numel() and (
            draft_token_ids.dtype == torch.bool
            or draft_token_ids.is_floating_point()
            or draft_token_ids.is_complex()
        ):
            raise TypeError("draft_token_ids must contain integers")
        draft_dtype = torch.int64 if draft_token_ids.numel() == 0 else draft_token_ids.dtype
        draft = draft_token_ids.to(dtype=draft_dtype)
    else:
        draft, _ = padded_draft_tokens(draft_token_ids)

    if target.shape[0] != draft.shape[0]:
        raise ValueError("target_token_ids and draft_token_ids must have the same batch size")
    batch_size = target.shape[0]
    if batch_size < 1:
        raise ValueError("token tensors must contain at least one request")
    if draft.shape[1] > _MAX_DRAFT_TOKENS:
        raise ValueError(f"draft_token_ids may contain at most {_MAX_DRAFT_TOKENS} drafts per request")

    valid = draft != _DRAFT_PAD
    draft_counts = valid.sum(dim=1)
    if draft.shape[1]:
        expected_valid = torch.arange(draft.shape[1]).unsqueeze(0) < draft_counts.unsqueeze(1)
        if not torch.equal(valid, expected_valid):
            raise ValueError("draft_token_ids must use only trailing -1 padding")
    max_drafts = int(draft_counts.max().item()) if batch_size else 0
    if target.shape[1] < max_drafts + 1:
        raise ValueError("target_token_ids must contain at least draft_count + 1 rows per request")

    accepted = torch.ones(batch_size, dtype=torch.int32)
    if max_drafts:
        matches = target[:, :max_drafts].eq(draft[:, :max_drafts]) & valid[:, :max_drafts]
        accepted += torch.cumprod(matches.to(torch.int32), dim=1).sum(dim=1)
    return accepted


def num_rejected(draft_counts, accepted) -> torch.Tensor:
    """Return CPU ``draft_count - (accepted - 1)`` as ``int32``."""
    counts = _integer_tensor(draft_counts, "draft_counts", _DEFAULT_DEVICE, torch.int64)
    accepted = _integer_tensor(accepted, "accepted", _DEFAULT_DEVICE, torch.int64)
    if counts.numel() != accepted.numel():
        raise ValueError("draft_counts and accepted must have the same length")
    if torch.any(counts < 0).item():
        raise ValueError("draft_counts must be nonnegative")
    if torch.any(accepted < 1).item() or torch.any(accepted > counts + 1).item():
        raise ValueError("accepted must be in 1..draft_count + 1")
    return (counts - accepted + 1).to(torch.int32)


@dataclass
class DSparkBatchMetadata:
    """Packed request metadata for one DSpark target/draft batch.

    Tensors remain on the device selected in :meth:`build`. The padding masks
    are flattened with their corresponding packed token tensors and use
    ``True`` for padding/no-write positions.
    """

    request_ids: tuple[str, ...]
    request_slots: torch.Tensor
    slot_generations: torch.Tensor
    target_query_start_loc: torch.Tensor
    draft_query_start_loc: torch.Tensor
    num_draft_tokens: torch.Tensor
    cu_num_draft_tokens: torch.Tensor
    cu_num_sampled_tokens: torch.Tensor
    draft_token_ids: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor
    logits_indices: torch.Tensor
    remaining_output_budget: torch.Tensor
    target_padding_mask: torch.Tensor
    draft_padding_mask: torch.Tensor

    @classmethod
    def build(
        cls,
        requests,
        target_query_lengths,
        draft_token_ids,
        request_slots,
        slot_generations,
        remaining_output_budget,
        target_padding_mask=None,
        device=_DEFAULT_DEVICE,
    ) -> "DSparkBatchMetadata":
        """Build packed B1..B8 metadata without engine or HPU dependencies.

        All metadata inputs must be Python sequences or CPU host tensors;
        ``device`` only selects where the returned tensors are constructed.
        Draft IDs are flattened in request order. ``logits_indices`` selects
        the raw target-query tail of ``draft_count + 1`` rows for each request.
        ``target_logits_indices`` and ``bonus_logits_indices`` are positions in
        that selected-logits space, matching the upstream metadata contract.
        A selected tail may not contain a row marked as padding.
        """
        device = torch.device(device)
        try:
            request_ids = tuple(requests)
        except TypeError as error:
            raise TypeError("requests must be a sequence") from error
        batch_size = len(request_ids)
        if not 1 <= batch_size <= _MAX_BATCH_SIZE:
            raise ValueError(f"batch size must be in 1..{_MAX_BATCH_SIZE}")
        if any(not isinstance(request_id, str) for request_id in request_ids):
            raise TypeError("request IDs must be strings")
        if len(set(request_ids)) != batch_size:
            raise ValueError("request IDs must be unique within a batch")

        target_lengths = _integer_values(target_query_lengths, "target_query_lengths")
        if len(target_lengths) != batch_size:
            raise ValueError(
                f"target_query_lengths length {len(target_lengths)} does not match batch size {batch_size}"
            )
        if any(length < 1 for length in target_lengths):
            raise ValueError("target query lengths must be at least one")

        drafts = _draft_rows(draft_token_ids, batch_size)
        draft_counts = [len(row) for row in drafts]
        for request_index, (target_length, draft_count) in enumerate(zip(target_lengths, draft_counts, strict=True)):
            if target_length < draft_count + 1:
                raise ValueError(
                    f"target query length for request {request_index} must be at least draft_count + 1"
                )

        slots = _integer_values(request_slots, "request_slots")
        generations = _integer_values(slot_generations, "slot_generations")
        budgets = _integer_values(remaining_output_budget, "remaining_output_budget")
        for name, values in (
            ("request_slots", slots),
            ("slot_generations", generations),
            ("remaining_output_budget", budgets),
        ):
            if len(values) != batch_size:
                raise ValueError(f"{name} length {len(values)} does not match batch size {batch_size}")
            if any(value < 0 for value in values):
                raise ValueError(f"{name} must be nonnegative")
        if len(set(slots)) != batch_size:
            raise ValueError("request_slots must be unique within a batch")

        target_starts = _starts(target_lengths)
        draft_starts = _starts(draft_counts)
        cu_draft = list(accumulate(draft_counts))
        cu_sampled = list(accumulate(draft_count + 1 for draft_count in draft_counts))
        bonus_indices = [total - 1 for total in cu_sampled]
        padding_values = _host_padding_mask(target_padding_mask, target_starts[-1])

        target_indices = []
        logits_indices = []
        sampled_start = 0
        for request_index, draft_count in enumerate(draft_counts):
            selected_start = target_starts[request_index + 1] - (draft_count + 1)
            selected_end = target_starts[request_index + 1]
            if any(padding_values[selected_start:selected_end]):
                raise ValueError(
                    f"selected target tail for request {request_index} contains padding; "
                    "target_query_lengths and target_padding_mask must provide a valid selected span"
                )
            logits_indices.extend(range(selected_start, selected_end))
            target_indices.extend(range(sampled_start, sampled_start + draft_count))
            sampled_start += draft_count + 1

        total_draft = draft_starts[-1]
        flattened_drafts = [token for row in drafts for token in row]
        return cls(
            request_ids=request_ids,
            request_slots=torch.tensor(slots, dtype=torch.int32, device=device),
            slot_generations=torch.tensor(generations, dtype=torch.int64, device=device),
            target_query_start_loc=torch.tensor(target_starts, dtype=torch.int32, device=device),
            draft_query_start_loc=torch.tensor(draft_starts, dtype=torch.int32, device=device),
            num_draft_tokens=torch.tensor(draft_counts, dtype=torch.int32, device=device),
            cu_num_draft_tokens=torch.tensor(cu_draft, dtype=torch.int32, device=device),
            cu_num_sampled_tokens=torch.tensor(cu_sampled, dtype=torch.int32, device=device),
            draft_token_ids=torch.tensor(flattened_drafts, dtype=torch.int32, device=device),
            target_logits_indices=torch.tensor(target_indices, dtype=torch.int32, device=device),
            bonus_logits_indices=torch.tensor(bonus_indices, dtype=torch.int32, device=device),
            logits_indices=torch.tensor(logits_indices, dtype=torch.int32, device=device),
            remaining_output_budget=torch.tensor(budgets, dtype=torch.int32, device=device),
            target_padding_mask=torch.tensor(padding_values, dtype=torch.bool, device=device),
            draft_padding_mask=torch.zeros(total_draft, dtype=torch.bool, device=device),
        )

    @staticmethod
    def accepted_prefix_lengths(target_token_ids, draft_token_ids) -> torch.Tensor:
        return accepted_prefix_lengths(target_token_ids, draft_token_ids)

    @staticmethod
    def num_rejected(draft_counts, accepted) -> torch.Tensor:
        return num_rejected(draft_counts, accepted)

    @staticmethod
    def padded_draft_tokens(draft_token_ids, pad: int = _DRAFT_PAD, *, device=None):
        return padded_draft_tokens(draft_token_ids, pad, device=device)


__all__ = [
    "DSparkBatchMetadata",
    "accepted_prefix_lengths",
    "num_rejected",
    "padded_draft_tokens",
]
