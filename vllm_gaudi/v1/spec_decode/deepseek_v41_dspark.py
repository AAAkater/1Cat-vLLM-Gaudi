# SPDX-License-Identifier: Apache-2.0
"""HPU bridge from the packed DSpark transaction to the upstream speculator.

The upstream :class:`DSparkSpeculator` owns the fixed five-step Markov loop.
Gaudi computes the packed ``[B, 5, H]`` draft hidden state and the
vocabulary-sharded ``[B, 5, V_local]`` logits with its existing
``PackedDraftBackbone``.  This module wires that real backbone callback into
the inherited upstream ``_generate_draft`` and ``_sample_sequential`` methods
without importing the plugin transaction module (and creating an import
cycle).

The upstream ``enable_adaptive_verification`` flag is enabled solely because
the HPU engine consumes the confidence tensor returned by the upstream loop.
It does not introduce an adaptive acceptance policy here.
"""
from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

DSPARK_DRAFT_TOKENS = 5
DSPARK_TARGET_QUERY_BUCKET = 6
MAX_DSPARK_REQUESTS = 64


def _default_linear(value: torch.Tensor, owner: Any) -> torch.Tensor:
    """Fallback for CPU/standalone use; HPU passes the prepared-dense helper."""
    return F.linear(value, owner.weight, getattr(owner, "bias", None))


class HPUDSparkModelHooks:
    """Thin upstream-model API over the packed backbone's local draft tensors.

    ``compute_draft_logits`` only reshapes the logits produced once by the
    packed backbone.  It deliberately has no vocabulary gather.
    """

    def __init__(self, draft: torch.nn.Module, *, linear: Callable | None = None):
        self.draft = draft
        last = draft.weights.get_submodule("2")
        self._markov_embedding = last.markov_head.embed
        self._markov_weight = last.markov_head.head.weight
        self._confidence_proj = last.confidence_head.proj
        self._linear = linear or _default_linear
        self._cached_hidden: torch.Tensor | None = None
        self._cached_logits: torch.Tensor | None = None

    def set_cached(self, hidden: torch.Tensor, logits: torch.Tensor) -> None:
        if hidden.ndim != 3 or hidden.shape[1] != DSPARK_DRAFT_TOKENS:
            raise ValueError("cached DSpark hidden must be [B, 5, H]")
        if logits.ndim != 3 or logits.shape[:2] != hidden.shape[:2]:
            raise ValueError("cached DSpark logits must be [B, 5, V_local]")
        if hidden.device != logits.device:
            raise ValueError("cached DSpark hidden and logits must share one device")
        self._cached_hidden = hidden
        self._cached_logits = logits

    def clear_cached(self) -> None:
        self._cached_hidden = None
        self._cached_logits = None

    def compute_draft_logits(self, sample_hidden: torch.Tensor) -> torch.Tensor:
        del sample_hidden  # The packed backbone already computed these logits.
        if self._cached_logits is None:
            raise RuntimeError("HPU DSpark proposal requires packed local draft logits")
        return self._cached_logits.reshape(-1, self._cached_logits.shape[-1])

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Preserve the original embedding exactly, including a nonzero row 0.
        return self.draft._embed(token_ids, self._markov_embedding)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return F.linear(markov_embed.float(), self._markov_weight)

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Match the old B-row-per-step confidence projection."""
        if head_hidden.shape[0] % DSPARK_DRAFT_TOKENS:
            raise ValueError("confidence rows must be divisible by five draft steps")
        batch_size = head_hidden.shape[0] // DSPARK_DRAFT_TOKENS
        hidden = head_hidden.reshape(batch_size, DSPARK_DRAFT_TOKENS, -1)
        markov = markov_embed.reshape(batch_size, DSPARK_DRAFT_TOKENS, -1)
        values = []
        for step in range(DSPARK_DRAFT_TOKENS):
            confidence_input = torch.cat((hidden[:, step], markov[:, step]), dim=-1).float()
            values.append(self._linear(confidence_input, self._confidence_proj).reshape(batch_size))
        return torch.stack(values, dim=1).reshape(batch_size * DSPARK_DRAFT_TOKENS)


class HPUDSparkSpeculator(DSparkSpeculator):
    """HPU DSpark speculator whose real backbone is an injected callback.

    The plugin transaction supplies :class:`PackedDraftBackbone` as ``backbone``
    and can also inject the packed context writer and post-commit input builder.
    The speculator owns proposal orchestration while the transaction continues
    to own target verification, records, rings and B1 handoff.
    """

    _speculator_name = "HPU DSpark"

    def __init__(
        self,
        draft: torch.nn.Module,
        *,
        device: torch.device | None = None,
        max_num_reqs: int = MAX_DSPARK_REQUESTS,
        num_speculative_steps: int = DSPARK_DRAFT_TOKENS,
        linear: Callable | None = None,
        draft_sample_method: str = "greedy",
        draft_topk: int | None = None,
        sample_from_anchor: bool = True,
        backbone: Callable | None = None,
        context_writer: Callable | None = None,
        inputs_builder: Callable | None = None,
        batch_state: Any | None = None,
    ):
        if num_speculative_steps != DSPARK_DRAFT_TOKENS:
            raise ValueError("HPU DSpark supports the fixed five-token upstream loop only")
        if type(max_num_reqs) is not int or not 1 <= max_num_reqs <= MAX_DSPARK_REQUESTS:
            raise ValueError(f"max_num_reqs must be an integer in 1..{MAX_DSPARK_REQUESTS}")
        if draft_sample_method != "greedy":
            raise ValueError("HPU DSpark currently supports greedy draft sampling only")
        if draft_topk is not None:
            raise ValueError("HPU DSpark does not support the upstream top-k sampling variant")
        if not sample_from_anchor:
            raise ValueError("HPU DSpark requires the anchor-as-first-prediction layout")

        last = draft.weights.get_submodule("2")
        if not hasattr(last, "markov_head") or not hasattr(last, "confidence_head"):
            raise ValueError("HPU DSpark draft must expose Markov and confidence heads")
        if device is None:
            device = last.markov_head.head.weight.device
        device = torch.device(device)

        # These are the only fields read by the inherited upstream loop.
        self.draft = draft
        self.device = device
        self.max_num_reqs = max_num_reqs
        self.state_capacity = getattr(batch_state, "capacity", max_num_reqs)
        if type(self.state_capacity) is not int or not max_num_reqs <= self.state_capacity <= MAX_DSPARK_REQUESTS:
            raise ValueError("DSpark state capacity must cover the execution capacity in 1..64")
        self.num_speculative_steps = DSPARK_DRAFT_TOKENS
        self.num_query_per_req = DSPARK_DRAFT_TOKENS
        self.sample_from_anchor = True
        self.draft_sample_method = draft_sample_method
        self._draft_topk = None
        self.draft_logits = None

        max_num_sampled = max_num_reqs * DSPARK_DRAFT_TOKENS
        self.sample_indices = torch.arange(max_num_sampled, dtype=torch.int64, device=device)
        self.sample_pos = torch.arange(max_num_sampled, dtype=torch.int64, device=device)
        self.sample_idx_mapping = torch.arange(max_num_sampled, dtype=torch.int32, device=device)
        self._anchor_idx = (torch.arange(max_num_reqs, dtype=torch.int64, device=device) *
                            DSPARK_DRAFT_TOKENS)

        self._anchor_input_ids = torch.zeros(max_num_sampled, dtype=torch.int64, device=device)
        self.input_buffers = SimpleNamespace(input_ids=self._anchor_input_ids)
        self.draft_tokens = torch.zeros((max_num_reqs, DSPARK_DRAFT_TOKENS),
                                        dtype=torch.int64,
                                        device=device)
        self.draft_token_confidence_probs = torch.empty((max_num_reqs, DSPARK_DRAFT_TOKENS),
                                                       dtype=torch.float32,
                                                       device=device)
        self.model = HPUDSparkModelHooks(draft, linear=linear)

        # Injected dependencies keep this module independent from the plugin
        # transaction module while still making the real packed backbone and
        # accepted-context commit part of one traced proposal graph.
        self._backbone = backbone
        self._context_writer = context_writer
        self._inputs_builder = inputs_builder
        self._batch_state = batch_state
        self._packed_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                   tuple[torch.Tensor, ...] | None] | None = None

        # Compatibility switches only: the pinned upstream loop collects the
        # old HPU confidence output through enable_adaptive_verification, while
        # newer upstream revisions may gate it with use_confidence_head.  No
        # adaptive acceptance policy is introduced by either field.
        self.enable_adaptive_verification = True
        self.use_confidence_head = True
        self._active: torch.Tensor | None = None

    def _validate_cached_inputs(
        self,
        anchors: torch.Tensor,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        active: torch.Tensor,
    ) -> int:
        if anchors.ndim != 1:
            raise ValueError("DSpark anchors must be [B]")
        batch_size = anchors.shape[0]
        if not 1 <= batch_size <= self.max_num_reqs:
            raise ValueError(f"DSpark batch size must be in 1..{self.max_num_reqs}")
        if hidden.ndim != 3 or hidden.shape[:2] != (batch_size, DSPARK_DRAFT_TOKENS):
            raise ValueError("DSpark hidden must be [B, 5, H]")
        if logits.ndim != 3 or logits.shape[:2] != (batch_size, DSPARK_DRAFT_TOKENS):
            raise ValueError("DSpark logits must be [B, 5, V_local]")
        if active.shape != (batch_size, ) or active.dtype != torch.bool:
            raise ValueError("DSpark activity must be bool [B]")
        if anchors.dtype not in (torch.int32, torch.int64):
            raise ValueError("DSpark anchors must be int32 or int64")
        if anchors.device != hidden.device or logits.device != hidden.device or active.device != hidden.device:
            raise ValueError("DSpark anchors, hidden, logits and activity must share one device")
        if anchors.device != self.device:
            raise ValueError("DSpark input tensors must be on the speculator device")
        return batch_size

    def _validate_context_done(self, context_done, device: torch.device) -> None:
        if context_done is None:
            return
        if not isinstance(context_done, (tuple, list)):
            raise ValueError("DSpark context_done must be a per-layer sequence")
        if len(context_done) != len(self.draft.layers):
            raise ValueError("DSpark context_done must contain one completion per draft layer")
        for completion in context_done:
            if (not isinstance(completion, torch.Tensor) or completion.numel() == 0
                    or completion.device != device):
                raise ValueError("DSpark context_done entries must be nonempty tensors on the input device")

    def _validate_packed_inputs(
        self,
        anchors: torch.Tensor,
        positions: torch.Tensor,
        request_slots: torch.Tensor,
        active: torch.Tensor,
        context_done,
    ) -> int:
        if not isinstance(anchors, torch.Tensor) or anchors.ndim != 2 or anchors.shape[1] != 1:
            raise ValueError("packed DSpark anchors must be [B, 1]")
        batch_size = anchors.shape[0]
        if not 1 <= batch_size <= self.max_num_reqs:
            raise ValueError(f"DSpark batch size must be in 1..{self.max_num_reqs}")
        if anchors.dtype not in (torch.int32, torch.int64):
            raise ValueError("packed DSpark anchors must be int32 or int64")
        if not isinstance(positions, torch.Tensor) or positions.shape != (batch_size, DSPARK_DRAFT_TOKENS):
            raise ValueError("packed DSpark positions must be [B, 5]")
        if positions.dtype != torch.int32:
            raise ValueError("packed DSpark positions must be int32")
        if not isinstance(request_slots, torch.Tensor) or request_slots.shape != (batch_size, ):
            raise ValueError("packed DSpark request slots must be [B]")
        if request_slots.dtype != torch.int32:
            raise ValueError("packed DSpark request slots must be int32")
        if not isinstance(active, torch.Tensor) or active.shape != (batch_size, ) or active.dtype != torch.bool:
            raise ValueError("packed DSpark activity must be bool [B]")
        if (anchors.device != positions.device or anchors.device != request_slots.device
                or anchors.device != active.device):
            raise ValueError("packed DSpark inputs must share one device")
        if anchors.device != self.device:
            raise ValueError("packed DSpark inputs must be on the speculator device")
        self._validate_context_done(context_done, anchors.device)
        return batch_size

    def _validate_after_verify_inputs(
        self,
        target_states: torch.Tensor,
        positions: torch.Tensor,
        request_slots: torch.Tensor,
        committed: torch.Tensor,
        output: torch.Tensor,
        output_count: torch.Tensor,
        draft_enabled: torch.Tensor,
        active: torch.Tensor,
        status: torch.Tensor,
    ) -> int:
        if (not isinstance(target_states, torch.Tensor) or target_states.ndim != 3
                or target_states.shape[1] != DSPARK_TARGET_QUERY_BUCKET):
            raise ValueError("DSpark target states must be [B, 6, D]")
        batch_size = target_states.shape[0]
        if not 1 <= batch_size <= self.state_capacity:
            raise ValueError(f"DSpark batch size must be in 1..{self.state_capacity}")
        if (not isinstance(positions, torch.Tensor)
                or positions.shape != (batch_size, DSPARK_TARGET_QUERY_BUCKET)):
            raise ValueError("DSpark target positions must be [B, 6]")
        if (not isinstance(output, torch.Tensor)
                or output.shape != (batch_size, DSPARK_TARGET_QUERY_BUCKET)):
            raise ValueError("DSpark verified output must be [B, 6]")
        for name, value in (
            ("request_slots", request_slots),
            ("committed", committed),
            ("output_count", output_count),
            ("draft_enabled", draft_enabled),
            ("active", active),
            ("status", status),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != (batch_size, ):
                raise ValueError(f"DSpark {name} must be [B]")
        if request_slots.dtype != torch.int32:
            raise ValueError("DSpark request slots must be int32")
        if committed.dtype not in (torch.int32, torch.int64) or output_count.dtype not in (
                torch.int32, torch.int64):
            raise ValueError("DSpark committed/output counts must be integer")
        if status.dtype not in (torch.int32, torch.int64):
            raise ValueError("DSpark verify status must be integer")
        if active.dtype != torch.bool or draft_enabled.dtype != torch.bool:
            raise ValueError("DSpark active/draft_enabled flags must be bool")
        device = target_states.device
        for value in (positions, request_slots, committed, output, output_count, draft_enabled, active, status):
            if value.device != device:
                raise ValueError("DSpark post-verify inputs must share one device")
        if device != self.device:
            raise ValueError("DSpark post-verify inputs must be on the speculator device")
        return batch_size

    def _set_anchor_feedback(self, anchors: torch.Tensor, active: torch.Tensor) -> None:
        # Inactive requests feed token 0 into the Markov embedding, exactly as
        # the legacy sampler did.  Embedding row 0 is intentionally not masked.
        batch_size = anchors.shape[0]
        anchor_feedback = torch.where(active, anchors.to(torch.int64),
                                      torch.zeros_like(anchors, dtype=torch.int64))
        self._anchor_input_ids[self._anchor_idx[:batch_size]] = anchor_feedback

    def _execute_upstream(
        self,
        anchors: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = anchors.shape[0]
        self._set_anchor_feedback(anchors, active)
        self._generate_draft(
            batch_size,
            batch_size * DSPARK_DRAFT_TOKENS,
            attn_metadata=None,
            slot_mappings=None,
            num_tokens_across_dp=None,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        internal_proposals = self.draft_tokens[:batch_size]
        proposals = torch.where(active.unsqueeze(1), internal_proposals,
                                torch.full_like(internal_proposals, -1))
        confidence = self.draft_token_confidence_probs[:batch_size].clone()
        return proposals, confidence

    def propose_cached(
        self,
        anchors: torch.Tensor,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility path for tools that already precomputed packed tensors."""
        self._validate_cached_inputs(anchors, hidden, logits, active)
        self._active = active
        self._packed_inputs = None
        try:
            self.model.set_cached(hidden, logits)
            return self._execute_upstream(anchors, active)
        finally:
            self._active = None
            self._packed_inputs = None
            self.model.clear_cached()

    def propose_packed(
        self,
        anchors: torch.Tensor,
        positions: torch.Tensor,
        request_slots: torch.Tensor,
        active: torch.Tensor,
        context_done,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one real packed backbone and the inherited upstream Markov loop.

        ``context_done`` is passed unchanged to the injected backbone so the
        accepted-context writes remain a graph dependency of draft attention.
        """
        self._validate_packed_inputs(anchors, positions, request_slots, active, context_done)
        if self._backbone is None:
            raise RuntimeError("HPU DSpark propose_packed requires an injected packed backbone")
        if anchors.dtype != torch.int64:
            anchors = anchors.to(torch.int64)
        self._active = active
        self._packed_inputs = (anchors, positions, request_slots, active, context_done)
        try:
            self.model.clear_cached()
            return self._execute_upstream(anchors.reshape(-1), active)
        finally:
            self._active = None
            self._packed_inputs = None
            self.model.clear_cached()

    def propose_after_verify(
        self,
        target_states: torch.Tensor,
        positions: torch.Tensor,
        request_slots: torch.Tensor,
        committed: torch.Tensor,
        output: torch.Tensor,
        output_count: torch.Tensor,
        draft_enabled: torch.Tensor,
        active: torch.Tensor,
        status: torch.Tensor,
        execution_lanes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Commit accepted context, build draft inputs and run packed proposal.

        The injected ``context_writer`` contract is
        ``writer(draft, batch_state, target_states, positions, request_slots,
        committed, valid_context)`` and must return the per-layer completion
        tuple consumed by the packed backbone.  The injected ``inputs_builder``
        contract is ``builder(positions, output, output_count, committed,
        draft_active)`` and must return ``anchors[N], positions[N, 5]``.
        Context is committed at state capacity N. Optional host-validated
        ``execution_lanes[B]`` gather the B-row model input and scatter the
        candidates back to N rows without changing logical ring owners.
        """
        self._validate_after_verify_inputs(
            target_states,
            positions,
            request_slots,
            committed,
            output,
            output_count,
            draft_enabled,
            active,
            status,
        )
        state_capacity = target_states.shape[0]
        if execution_lanes is None:
            if state_capacity > self.max_num_reqs:
                raise ValueError("compact DSpark proposal requires explicit execution lanes")
        elif (not isinstance(execution_lanes, torch.Tensor) or execution_lanes.ndim != 1
              or execution_lanes.shape[0] != self.max_num_reqs or execution_lanes.dtype != torch.int64
              or execution_lanes.device != self.device):
            raise ValueError("DSpark execution lanes must be int64 [B] on the proposal device")
        if self._context_writer is None or self._inputs_builder is None or self._batch_state is None:
            raise RuntimeError(
                "HPU DSpark propose_after_verify requires injected context_writer, "
                "inputs_builder and batch_state")
        valid_context = active & (status == 0)
        context_done = self._context_writer(
            self.draft,
            self._batch_state,
            target_states,
            positions,
            request_slots,
            committed,
            valid_context,
        )
        draft_active = draft_enabled & valid_context
        anchors, draft_positions = self._inputs_builder(
            positions,
            output,
            output_count,
            committed,
            draft_active,
        )
        if not isinstance(anchors, torch.Tensor) or anchors.shape not in ((draft_active.shape[0], ),
                                                                          (draft_active.shape[0], 1)):
            raise ValueError("DSpark draft inputs builder must return anchors [B] or [B, 1]")
        if anchors.ndim == 1:
            anchors = anchors.unsqueeze(1)
        if execution_lanes is not None:
            # Values come from the validated host plan.  Keep the global logical
            # slots when compacting; execution rows never identify ring owners.
            anchors = anchors.index_select(0, execution_lanes)
            draft_positions = draft_positions.index_select(0, execution_lanes)
            request_slots = request_slots.index_select(0, execution_lanes)
            draft_active = draft_active.index_select(0, execution_lanes)
        proposals, confidence = self.propose_packed(
            anchors,
            draft_positions,
            request_slots,
            draft_active,
            context_done,
        )
        if execution_lanes is None:
            return proposals, confidence
        restored_proposals = torch.full(
            (state_capacity, DSPARK_DRAFT_TOKENS), -1, dtype=proposals.dtype, device=self.device)
        restored_confidence = torch.zeros(
            (state_capacity, DSPARK_DRAFT_TOKENS), dtype=confidence.dtype, device=self.device)
        return (restored_proposals.index_copy(0, execution_lanes, proposals),
                restored_confidence.index_copy(0, execution_lanes, confidence))

    def propose(
        self,
        anchors: torch.Tensor,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        active: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit compatibility bridge; the full GPU DFlash path is unsupported."""
        if args or kwargs:
            raise NotImplementedError("HPU DSpark uses the explicit packed proposal bridge")
        return self.propose_cached(anchors, hidden, logits, active)

    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        """Run the injected real packed backbone for ``propose_packed``."""
        del attn_metadata, slot_mappings, num_tokens_across_dp, cudagraph_runtime_mode
        packed_inputs = self._packed_inputs
        if packed_inputs is not None:
            if self._backbone is None:
                raise RuntimeError("HPU DSpark _run_model requires an injected packed backbone")
            anchors, positions, request_slots, active, context_done = packed_inputs
            batch_size = anchors.shape[0]
            expected_tokens = batch_size * DSPARK_DRAFT_TOKENS
            if num_tokens != expected_tokens:
                raise ValueError(f"packed DSpark inputs have {expected_tokens} rows, expected {num_tokens}")
            hidden, logits = self._backbone(
                anchors,
                positions,
                request_slots,
                active,
                context_done=context_done,
            )
            if not isinstance(hidden, torch.Tensor) or not isinstance(logits, torch.Tensor):
                raise TypeError("packed DSpark backbone must return hidden and local-logit tensors")
            if hidden.ndim != 3 or hidden.shape[:2] != (batch_size, DSPARK_DRAFT_TOKENS):
                raise ValueError("packed DSpark backbone hidden must be [B, 5, H]")
            if logits.ndim != 3 or logits.shape[:2] != (batch_size, DSPARK_DRAFT_TOKENS):
                raise ValueError("packed DSpark backbone logits must be [B, 5, V_local]")
            if hidden.device != self.device or logits.device != self.device:
                raise ValueError("packed DSpark backbone outputs must be on the speculator device")
            self.model.set_cached(hidden, logits)
            return hidden.reshape(-1, hidden.shape[-1])

        # Standalone tools may still inject already-computed packed tensors.
        cached_hidden = self.model._cached_hidden
        if cached_hidden is None:
            raise RuntimeError("HPU DSpark _run_model requires packed inputs or cached tensors")
        expected_tokens = cached_hidden.shape[0] * DSPARK_DRAFT_TOKENS
        if num_tokens != expected_tokens:
            raise ValueError(f"HPU DSpark cached hidden has {expected_tokens} rows, expected {num_tokens}")
        return cached_hidden.reshape(-1, cached_hidden.shape[-1])

    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Greedy local-shard argmax with the existing TP global reduction."""
        del idx_map, sample_pos
        if self.draft_logits is not None:
            raise NotImplementedError("HPU DSpark does not support probabilistic upstream sampling")
        if self._active is None or self._active.shape != (logits.shape[0], ):
            raise RuntimeError("HPU DSpark _sample_logits requires an active proposal batch")
        if not 0 <= step < DSPARK_DRAFT_TOKENS:
            raise ValueError("DSpark Markov step is outside the fixed five-token loop")
        sampled = self.draft._global_argmax(logits).reshape(-1).to(torch.int64)
        if sampled.shape != (logits.shape[0], ):
            raise ValueError("DSpark global argmax must return one token per request")
        # Keep the upstream loop's internal inactive feedback at zero.  The
        # public bridge masks these rows to -1 after the loop.
        return torch.where(self._active, sampled, torch.zeros_like(sampled))

    def load_draft_model(self, target_model: torch.nn.Module, target_attn_layer_names: set[str]):
        del target_model, target_attn_layer_names
        raise NotImplementedError("HPU DSpark receives its prepared draft model from the packed transaction")

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode
        raise NotImplementedError("HPU DSpark uses the plugin's packed graph, not the CUDA graph manager")

    def capture(self) -> None:
        raise NotImplementedError("HPU DSpark uses the plugin's packed graph, not CUDA graph capture")


__all__ = [
    "DSPARK_DRAFT_TOKENS",
    "DSPARK_TARGET_QUERY_BUCKET",
    "HPUDSparkModelHooks",
    "HPUDSparkSpeculator",
    "MAX_DSPARK_REQUESTS",
]
