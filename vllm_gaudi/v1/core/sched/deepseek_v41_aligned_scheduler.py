# SPDX-License-Identifier: Apache-2.0
"""Opt-in scheduler for V4.1 TP4/TP8 ordinary request batches.

Select this class explicitly with --scheduler-cls; defaults are unchanged.
The pinned base provides a hook after every admission cap and before both
allocation paths. No worker-side shortening and no monkey-patching are used.
"""
import hashlib
import inspect
import os
from pathlib import Path

from vllm.v1.core.sched.scheduler import Scheduler
from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler
from vllm_gaudi.v1.core.sched.deepseek_v41_prefill_alignment import (
    aligned_prefill_count, write_prefill_trace, write_scheduler_init,
)

PINNED_SCHEDULER_SHA256 = "03284cd1377b79f9897f0700880bf64b756474913dcf368a38b962687a5d3c4b"


class DSV41AlignedScheduler(HPUAsyncScheduler):
    def __init__(self, *args, **kwargs):
        filename = inspect.getsourcefile(Scheduler)
        if filename is None or hashlib.sha256(Path(filename).read_bytes()).hexdigest() != PINNED_SCHEDULER_SHA256:
            raise RuntimeError("V4.1 alignment requires its qualified pinned scheduler source")
        super().__init__(*args, **kwargs)
        from vllm_gaudi import envs
        from vllm_gaudi.ops.deepseek_v41_config import is_v41, uses_v2
        config = self.vllm_config
        if not (is_v41(config) and uses_v2(config)
                and config.parallel_config.tensor_parallel_size in (4, 8)
                and config.parallel_config.pipeline_parallel_size == 1
                and config.scheduler_config.async_scheduling
                and config.scheduler_config.enable_chunked_prefill
                and envs.VLLM_HPU_DSV41_BATCH_DECODE
                and config.speculative_config is None):
            raise ValueError("Aligned prefill candidate requires ordinary TP4/TP8 PP1 V2 request batches")
        self._v41_prefill_quantum = (config.additional_config or {}).get("dsv41_prefill_alignment", 128)
        if type(self._v41_prefill_quantum) is not int or self._v41_prefill_quantum not in (0, 128):
            raise ValueError("dsv41_prefill_alignment must be 0 (trace baseline) or 128")
        threshold = self.scheduler_config.long_prefill_token_threshold
        if self._v41_prefill_quantum and (self.max_num_scheduled_tokens < 128
                or self.scheduler_config.max_num_batched_tokens < 128
                or 0 < threshold < 128):
            raise ValueError("Aligned partial prefill requires at least a 128-token admission budget")
        from vllm_gaudi.ops.deepseek_v41_prefill_capacity import prefill_capacity, prefill_compute_buckets

        capacity = prefill_capacity(self.scheduler_config.max_num_batched_tokens,
                                    config.parallel_config.tensor_parallel_size)
        write_scheduler_init(dict(
            scheduler_class=f"{type(self).__module__}.{type(self).__qualname__}",
            pinned_scheduler_sha256=PINNED_SCHEDULER_SHA256,
            scheduler_source_path=str(Path(__file__).resolve()),
            scheduler_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            pinned_scheduler_source_path=str(Path(filename).resolve()),
            hpu_parent_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(HPUAsyncScheduler)).read_bytes()).hexdigest(),
            max_model_len=config.model_config.max_model_len,
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_num_scheduled_tokens=self.max_num_scheduled_tokens,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            long_prefill_token_threshold=threshold,
            alignment_quantum=self._v41_prefill_quantum,
            derived_prefill_storage_capacity=capacity,
            derived_prefill_compute_buckets=list(prefill_compute_buckets(capacity)),
            capacity_evidence="Derived by the same pure helpers as worker construction; worker allocation must be audited separately",
            trace_scope="Host-only allocation/chunk evidence; per-step JSON writes add timing overhead",
        ))

    def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
        count = super()._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)
        # Preserve encoder scheduling unchanged; this candidate qualifies text.
        if request.has_encoder_inputs:
            return count
        return aligned_prefill_count(
            start=num_computed_tokens, count=count,
            prompt_tokens=request.num_prompt_tokens, total_tokens=request.num_tokens,
            quantum=self._v41_prefill_quantum,
        )

    def schedule(self, throttle_prefills=False):
        output = super().schedule(throttle_prefills)
        if os.environ.get("DSV41_PREFILL_ALLOCATION_TRACE") != "1":
            return output
        rows = []
        for request_id, count in output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            # The pinned _update_after_schedule advances computed tokens once.
            start = request.num_computed_tokens - count
            rows.append(dict(request_id=request_id, start=start, count=count,
                             prompt_tokens=request.num_prompt_tokens,
                             total_tokens=request.num_tokens,
                             prefill=start < max(request.num_prompt_tokens, request.num_tokens - 1)))
        write_prefill_trace("scheduler_allocation", dict(
            step=self.current_step, quantum=self._v41_prefill_quantum,
            max_num_scheduled_tokens=self.max_num_scheduled_tokens,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            total_num_scheduled_tokens=output.total_num_scheduled_tokens,
            requests=rows,
        ))
        return output
