# SPDX-License-Identifier: Apache-2.0
"""CPU-side packed metadata for the DSpark batch verification stage.

This module only owns request ordering and index construction. It does not
allocate KV state, execute a model, or depend on the vLLM engine runtime.
Padding masks use ``True`` for a padding/no-write position.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import accumulate
import operator

import torch

_MAX_BATCH_SIZE = 8
_MAX_DRAFT_TOKENS = 5
_TARGET_QUERY_BUCKET = 6
_DRAFT_PAD = -1
_DEFAULT_DEVICE = torch.device("cpu")


def _as_int(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must contain integers")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must contain integers") from error


def _as_scalar_int(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer scalar")
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU host tensor")
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must be an integer scalar")
        value = value.item()
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer scalar") from error


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


def _bool_values(value, name):
    """Materialize a one-dimensional CPU boolean mask without coercing ints."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU host tensor")
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must contain boolean values")
        return value.detach().tolist()
    try:
        values = list(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence") from error
    if any(type(item) is not bool for item in values):
        raise TypeError(f"{name} must contain boolean values")
    return values


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


def _host_tensor_values(value, name: str, *, dtype=None) -> list:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU host tensor")
    if dtype is not None and value.dtype != dtype:
        expected = "boolean" if dtype == torch.bool else "integer"
        raise TypeError(f"{name} must contain {expected} values")
    return value.tolist()


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


@dataclass(frozen=True)
class FixedTargetQueries:
    """Host-side view of packed target queries in a fixed ``[B, 6, ...]`` bucket.

    ``padding_mask`` is ``True`` for both unused bucket tails and logical
    padding rows; device consumers must treat those positions as no-write.
    ``logits_indices`` indexes the fixed bucket, while
    ``target_logits_indices`` and ``bonus_logits_indices`` remain positions in
    the selected-logits space. All tensors are CPU host tensors.
    """

    queries: torch.Tensor
    padding_mask: torch.Tensor
    real_indices: torch.Tensor
    padding_indices: torch.Tensor
    logits_indices: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor


@dataclass
class DSparkBatchMetadata:
    """Packed request metadata for one DSpark target/draft batch.

    Tensors remain on the device selected in :meth:`build`. The padding masks
    are flattened with their corresponding packed token tensors and use
    ``True`` for padding/no-write positions.

    ``request_ids`` and every request-axis tensor are lane-major. In explicit
    capacity mode, inactive lanes have ``request_ids[i] is None`` and
    ``request_active[i] is False``. The legacy builder mode keeps all lanes
    active and requires string request IDs. Explicit inactive lanes also use
    ``target_query_length=0``, no drafts, ``slot=-1``, ``generation=0``,
    ``remaining_output_budget=0`` and ``target_start_position=0``.
    """

    request_ids: tuple[str | None, ...]
    request_active: torch.Tensor
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
    target_start_positions: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int64),
    )
    round_generation: torch.Tensor = field(
        default_factory=lambda: torch.tensor(0, dtype=torch.int64),
    )

    @property
    def capacity(self) -> int:
        """Return the fixed physical lane count, including inactive lanes."""
        return len(self.request_ids)

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        """Return active request IDs in canonical lane order."""
        return tuple(request_id for request_id in self.request_ids if request_id is not None)

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
        *,
        target_start_positions=None,
        round_generation=0,
        active=None,
    ) -> "DSparkBatchMetadata":
        """Build packed B1..B8 metadata without engine or HPU dependencies.

        All metadata inputs must be Python sequences or CPU host tensors;
        ``device`` only selects where the returned tensors are constructed.
        ``target_start_positions`` supplies each request's absolute first
        target-query position and defaults to zero for compatibility.
        ``round_generation`` is one batch-scratch generation shared by the
        requests and defaults to zero.
        Draft IDs are flattened in request order. ``logits_indices`` selects
        the raw target-query tail of ``draft_count + 1`` rows for each request.
        ``target_logits_indices`` and ``bonus_logits_indices`` are positions in
        that selected-logits space, matching the upstream metadata contract.
        A selected tail may not contain a row marked as padding.

        With ``active=None`` the historical contract is preserved: every lane
        is active, target lengths are at least one, request slots may be any
        distinct nonnegative integers, and generations and budgets may be zero.
        With an explicit ``active`` mask, arrays are fixed-capacity and
        lane-major. Active lanes use ``slot == lane`` and a positive generation
        and budget. Inactive lanes use the exact sentinel values documented by
        :class:`DSparkBatchMetadata`.
        """
        device = torch.device(device)
        try:
            request_ids = tuple(requests)
        except TypeError as error:
            raise TypeError("requests must be a sequence") from error
        batch_size = len(request_ids)
        if not 1 <= batch_size <= _MAX_BATCH_SIZE:
            raise ValueError(f"batch size must be in 1..{_MAX_BATCH_SIZE}")

        explicit_active = active is not None
        if explicit_active:
            active_values = _bool_values(active, "active")
            if len(active_values) != batch_size:
                raise ValueError(f"active length {len(active_values)} does not match batch size {batch_size}")
            if not any(active_values):
                raise ValueError("active must contain at least one request")
            for request_id, enabled in zip(request_ids, active_values, strict=True):
                if enabled and not isinstance(request_id, str):
                    raise TypeError("active request IDs must be strings")
                if not enabled and request_id is not None:
                    raise TypeError("inactive request IDs must be None")
            active_ids = [request_id for request_id, enabled in zip(request_ids, active_values, strict=True) if enabled]
            if len(set(active_ids)) != len(active_ids):
                raise ValueError("active request IDs must be unique within a batch")
        else:
            active_values = [True] * batch_size
            if any(not isinstance(request_id, str) for request_id in request_ids):
                raise TypeError("request IDs must be strings")
            if len(set(request_ids)) != batch_size:
                raise ValueError("request IDs must be unique within a batch")

        target_lengths = _integer_values(target_query_lengths, "target_query_lengths")
        if len(target_lengths) != batch_size:
            raise ValueError(
                f"target_query_lengths length {len(target_lengths)} does not match batch size {batch_size}"
            )
        for lane, (length, enabled) in enumerate(zip(target_lengths, active_values, strict=True)):
            if enabled and length < 1:
                raise ValueError(f"active target query length for lane {lane} must be at least one")
            if explicit_active and enabled and length > _TARGET_QUERY_BUCKET:
                raise ValueError(
                    f"active target query length for lane {lane} must be at most {_TARGET_QUERY_BUCKET}"
                )
            if not enabled and length != 0:
                raise ValueError(f"inactive target query length for lane {lane} must be zero")

        drafts = _draft_rows(draft_token_ids, batch_size)
        draft_counts = [len(row) for row in drafts]
        for request_index, (target_length, draft_count, enabled) in enumerate(
                zip(target_lengths, draft_counts, active_values, strict=True)):
            if not enabled and draft_count:
                raise ValueError(f"inactive lane {request_index} must not contain draft tokens")
            if enabled and target_length < draft_count + 1:
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
            if not explicit_active and any(value < 0 for value in values):
                raise ValueError(f"{name} must be nonnegative")
        if explicit_active:
            for lane, enabled in enumerate(active_values):
                if enabled:
                    if slots[lane] != lane:
                        raise ValueError(f"active request slot for lane {lane} must equal its lane")
                    if generations[lane] < 1:
                        raise ValueError(f"active slot generation for lane {lane} must be positive")
                    if budgets[lane] < 1:
                        raise ValueError(f"active output budget for lane {lane} must be positive")
                else:
                    if slots[lane] != -1 or generations[lane] != 0 or budgets[lane] != 0:
                        raise ValueError(
                            f"inactive lane {lane} must use slot=-1, generation=0 and budget=0")
        elif len(set(slots)) != batch_size:
            raise ValueError("request_slots must be unique within a batch")

        if target_start_positions is None:
            target_positions = [0] * batch_size
        else:
            target_positions = _integer_values(target_start_positions, "target_start_positions")
            if len(target_positions) != batch_size:
                raise ValueError(
                    f"target_start_positions length {len(target_positions)} does not match batch size {batch_size}"
                )
            if any(position < 0 for position in target_positions):
                raise ValueError("target_start_positions must be nonnegative")
        if explicit_active:
            for lane, enabled in enumerate(active_values):
                if not enabled and target_positions[lane] != 0:
                    raise ValueError(f"inactive target start position for lane {lane} must be zero")

        round_generation_value = _as_scalar_int(round_generation, "round_generation")
        if round_generation_value < 0:
            raise ValueError("round_generation must be nonnegative")

        target_starts = _starts(target_lengths)
        draft_starts = _starts(draft_counts)
        cu_draft = list(accumulate(draft_counts))
        cu_sampled = []
        bonus_indices = []
        sampled_total = 0
        for draft_count, enabled in zip(draft_counts, active_values, strict=True):
            if enabled:
                sampled_total += draft_count + 1
                bonus_indices.append(sampled_total - 1)
            cu_sampled.append(sampled_total)
        padding_values = _host_padding_mask(target_padding_mask, target_starts[-1])

        target_indices = []
        logits_indices = []
        sampled_start = 0
        for request_index, (draft_count, enabled) in enumerate(zip(draft_counts, active_values, strict=True)):
            if not enabled:
                continue
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
            request_active=torch.tensor(active_values, dtype=torch.bool, device=device),
            request_slots=torch.tensor(slots, dtype=torch.int32, device=device),
            slot_generations=torch.tensor(generations, dtype=torch.int64, device=device),
            target_start_positions=torch.tensor(target_positions, dtype=torch.int64, device=device),
            round_generation=torch.tensor(round_generation_value, dtype=torch.int64, device=device),
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

    def pack_target_queries(self, target_queries) -> FixedTargetQueries:
        return pack_target_queries(target_queries, self)

    def unpack_target_queries(self, packed: FixedTargetQueries) -> torch.Tensor:
        return unpack_target_queries(packed, self)


def _request_active_values(metadata: DSparkBatchMetadata, batch_size: int) -> list[bool]:
    """Read the optional activity mask, treating missing or empty as legacy."""
    value = getattr(metadata, "request_active", None)
    if value is None:
        return [True] * batch_size
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return [True] * batch_size
        if value.ndim != 1 or value.device.type != "cpu" or value.dtype != torch.bool:
            raise ValueError("request_active must be a CPU boolean [N] tensor")
        active = value.tolist()
    else:
        active = _bool_values(value, "request_active")
        if not active:
            return [True] * batch_size
    if len(active) != batch_size:
        raise ValueError(f"request_active length {len(active)} does not match batch size {batch_size}")
    return active


def _fixed_metadata_layout(metadata: DSparkBatchMetadata):
    if not isinstance(metadata, DSparkBatchMetadata):
        raise TypeError("metadata must be a DSparkBatchMetadata instance")

    batch_size = len(metadata.request_ids)
    if not 1 <= batch_size <= _MAX_BATCH_SIZE:
        raise ValueError(f"metadata batch size must be in 1..{_MAX_BATCH_SIZE}")
    active = _request_active_values(metadata, batch_size)

    starts = _integer_values(metadata.target_query_start_loc, "target_query_start_loc")
    if len(starts) != batch_size + 1 or starts[0] != 0:
        raise ValueError("target_query_start_loc must have B + 1 entries starting at zero")
    lengths = [end - start for start, end in zip(starts[:-1], starts[1:], strict=True)]
    if any(length < 0 for length in lengths):
        raise ValueError("target query lengths must be nonnegative")
    if any(enabled and length < 1 for enabled, length in zip(active, lengths, strict=True)):
        raise ValueError("active target query lengths must be at least one")
    if any(not enabled and length != 0 for enabled, length in zip(active, lengths, strict=True)):
        raise ValueError("inactive target query lengths must be zero")
    if any(length > _TARGET_QUERY_BUCKET for length in lengths):
        raise ValueError(
            f"fixed target query bucket supports at most {_TARGET_QUERY_BUCKET} queries per request"
        )

    positions = _integer_values(metadata.target_start_positions, "target_start_positions")
    if len(positions) != batch_size or any(position < 0 for position in positions):
        raise ValueError("target_start_positions must be nonnegative and match the batch size")
    round_generation = _as_scalar_int(metadata.round_generation, "round_generation")
    if round_generation < 0:
        raise ValueError("round_generation must be nonnegative")

    padding = _host_tensor_values(metadata.target_padding_mask, "target_padding_mask", dtype=torch.bool)
    if len(padding) != starts[-1]:
        raise ValueError("target_padding_mask length must match packed target queries")

    draft_counts = _integer_values(metadata.num_draft_tokens, "num_draft_tokens")
    if len(draft_counts) != batch_size or any(count < 0 or count > _MAX_DRAFT_TOKENS for count in draft_counts):
        raise ValueError(f"num_draft_tokens must contain B values in 0..{_MAX_DRAFT_TOKENS}")
    if any(not enabled and count != 0 for enabled, count in zip(active, draft_counts, strict=True)):
        raise ValueError("inactive lanes must not contain draft tokens")
    if _integer_values(metadata.cu_num_draft_tokens, "cu_num_draft_tokens") != list(accumulate(draft_counts)):
        raise ValueError("cu_num_draft_tokens is inconsistent with num_draft_tokens")
    cumulative_sampled = []
    bonus_logits_indices = []
    sampled_total = 0
    for draft_count, enabled in zip(draft_counts, active, strict=True):
        if enabled:
            sampled_total += draft_count + 1
            bonus_logits_indices.append(sampled_total - 1)
        cumulative_sampled.append(sampled_total)
    if _integer_values(metadata.cu_num_sampled_tokens, "cu_num_sampled_tokens") != cumulative_sampled:
        raise ValueError("cu_num_sampled_tokens is inconsistent with num_draft_tokens")

    source_logits_indices = []
    target_logits_indices = []
    sampled_start = 0
    for request_index, (draft_count, enabled) in enumerate(zip(draft_counts, active, strict=True)):
        if not enabled:
            continue
        selected_start = starts[request_index + 1] - (draft_count + 1)
        if selected_start < starts[request_index]:
            raise ValueError(f"request {request_index} has fewer target queries than its selected tail")
        source_logits_indices.extend(range(selected_start, starts[request_index + 1]))
        target_logits_indices.extend(range(sampled_start, sampled_start + draft_count))
        sampled_start += draft_count + 1

    if _integer_values(metadata.logits_indices, "logits_indices") != source_logits_indices:
        raise ValueError("logits_indices is inconsistent with target query lengths and draft counts")
    if _integer_values(metadata.target_logits_indices, "target_logits_indices") != target_logits_indices:
        raise ValueError("target_logits_indices is inconsistent with draft counts")
    if _integer_values(metadata.bonus_logits_indices, "bonus_logits_indices") != bonus_logits_indices:
        raise ValueError("bonus_logits_indices is inconsistent with draft counts")
    if any(padding[index] for index in source_logits_indices):
        raise ValueError("selected target logits rows must not be padding")

    return lengths, padding, source_logits_indices, target_logits_indices, bonus_logits_indices


def pack_target_queries(target_queries, metadata: DSparkBatchMetadata) -> FixedTargetQueries:
    """Pack CPU ``[sum target query, ...]`` rows into ``[B, 6, ...]``.

    The result also carries flat bucket indices for real and no-write rows and
    both logits index spaces. Inputs and metadata must already be on the host;
    this helper never copies from an accelerator and never synchronizes HPU.
    """
    if not isinstance(target_queries, torch.Tensor):
        raise TypeError("target_queries must be a tensor")
    if target_queries.ndim < 1:
        raise ValueError("target_queries must have a query dimension")
    if target_queries.device.type != "cpu":
        raise ValueError("target_queries must be a CPU host tensor")

    lengths, source_padding, source_logits, target_logits, bonus_logits = _fixed_metadata_layout(metadata)
    batch_size = len(lengths)
    starts = _starts(lengths)
    if target_queries.shape[0] != starts[-1]:
        raise ValueError(
            f"target_queries has {target_queries.shape[0]} rows but metadata describes {starts[-1]}"
        )

    packed = target_queries.new_zeros((batch_size, _TARGET_QUERY_BUCKET, *target_queries.shape[1:]))
    padding_mask = torch.ones((batch_size, _TARGET_QUERY_BUCKET), dtype=torch.bool)
    source_to_fixed = {}
    for request_index, length in enumerate(lengths):
        source_start = starts[request_index]
        source_end = starts[request_index + 1]
        packed[request_index, :length] = target_queries[source_start:source_end]
        padding_mask[request_index, :length] = torch.tensor(
            source_padding[source_start:source_end], dtype=torch.bool
        )
        for offset in range(length):
            source_to_fixed[source_start + offset] = request_index * _TARGET_QUERY_BUCKET + offset

    flat_padding = padding_mask.reshape(-1).tolist()
    real_indices = [index for index, is_padding in enumerate(flat_padding) if not is_padding]
    padding_indices = [index for index, is_padding in enumerate(flat_padding) if is_padding]
    fixed_logits = [source_to_fixed[index] for index in source_logits]

    return FixedTargetQueries(
        queries=packed,
        padding_mask=padding_mask,
        real_indices=torch.tensor(real_indices, dtype=torch.int64),
        padding_indices=torch.tensor(padding_indices, dtype=torch.int64),
        logits_indices=torch.tensor(fixed_logits, dtype=torch.int64),
        target_logits_indices=torch.tensor(target_logits, dtype=torch.int64),
        bonus_logits_indices=torch.tensor(bonus_logits, dtype=torch.int64),
    )


def unpack_target_queries(packed: FixedTargetQueries, metadata: DSparkBatchMetadata) -> torch.Tensor:
    """Undo :func:`pack_target_queries` into the logical packed-query layout."""
    if not isinstance(packed, FixedTargetQueries):
        raise TypeError("packed must be a FixedTargetQueries instance")
    if not isinstance(packed.queries, torch.Tensor):
        raise TypeError("packed queries must be a tensor")
    if packed.queries.ndim < 2 or packed.queries.shape[1] != _TARGET_QUERY_BUCKET:
        raise ValueError(f"packed queries must have shape [B, {_TARGET_QUERY_BUCKET}, ...]")
    if packed.queries.device.type != "cpu":
        raise ValueError("packed queries must be a CPU host tensor")

    lengths, _source_padding, source_logits, target_logits, bonus_logits = _fixed_metadata_layout(metadata)
    batch_size = len(lengths)
    if packed.queries.shape[0] != batch_size:
        raise ValueError("packed queries and metadata batch sizes differ")

    if not isinstance(packed.padding_mask, torch.Tensor):
        raise TypeError("packed padding_mask must be a tensor")
    if packed.padding_mask.ndim != 2 or packed.padding_mask.shape != (batch_size, _TARGET_QUERY_BUCKET):
        raise ValueError(f"packed padding_mask must have shape [B, {_TARGET_QUERY_BUCKET}]")
    if packed.padding_mask.device.type != "cpu" or packed.padding_mask.dtype != torch.bool:
        raise TypeError("packed padding_mask must be a CPU boolean tensor")
    for lane, length in enumerate(lengths):
        if length == 0 and not bool(packed.padding_mask[lane].all()):
            raise ValueError(f"inactive packed lane {lane} must be fully padding/no-write")
    flat_padding = packed.padding_mask.reshape(-1).tolist()
    if len(flat_padding) != batch_size * _TARGET_QUERY_BUCKET:
        raise ValueError("packed padding_mask must have shape [B, 6]")
    expected_real = [index for index, is_padding in enumerate(flat_padding) if not is_padding]
    expected_padding = [index for index, is_padding in enumerate(flat_padding) if is_padding]
    if _integer_values(packed.real_indices, "real_indices") != expected_real:
        raise ValueError("real_indices is inconsistent with packed padding_mask")
    if _integer_values(packed.padding_indices, "padding_indices") != expected_padding:
        raise ValueError("padding_indices is inconsistent with packed padding_mask")

    starts = _starts(lengths)
    source_to_fixed = {}
    for request_index, length in enumerate(lengths):
        for offset in range(length):
            source_to_fixed[starts[request_index] + offset] = request_index * _TARGET_QUERY_BUCKET + offset
    if _integer_values(packed.logits_indices, "packed logits_indices") != [
            source_to_fixed[index] for index in source_logits
    ]:
        raise ValueError("packed logits_indices is inconsistent with metadata")
    if _integer_values(packed.target_logits_indices, "packed target_logits_indices") != target_logits:
        raise ValueError("packed target_logits_indices is inconsistent with metadata")
    if _integer_values(packed.bonus_logits_indices, "packed bonus_logits_indices") != bonus_logits:
        raise ValueError("packed bonus_logits_indices is inconsistent with metadata")

    rows = [packed.queries[index, :length] for index, length in enumerate(lengths) if length]
    if not rows:
        return packed.queries.new_empty((0, *packed.queries.shape[2:]))
    return torch.cat(rows, dim=0)


__all__ = [
    "DSparkBatchMetadata",
    "FixedTargetQueries",
    "accepted_prefix_lengths",
    "num_rejected",
    "pack_target_queries",
    "padded_draft_tokens",
    "unpack_target_queries",
]
