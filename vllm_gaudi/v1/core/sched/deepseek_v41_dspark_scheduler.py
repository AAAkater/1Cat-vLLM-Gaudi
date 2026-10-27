# SPDX-License-Identifier: Apache-2.0
"""Synchronous DSpark scheduler with independent draft query accounting."""

from vllm.v1.core.sched.scheduler import Scheduler
from vllm_gaudi.ops.deepseek_v41_scheduling import SeparateQuerySchedulerConfig, validate_separate_query_profile


class DSparkSeparateQueryScheduler(Scheduler):
    """Reuse upstream admission and KV allocation with a local budget view.

    Target query accounting and all lookahead allocations remain upstream's
    responsibility. Only the redundant input-row reservation for the separate
    draft pass is excluded. Async scheduling is rejected by the profile gate.
    """

    def __init__(self, *args, **kwargs):
        config = kwargs.get("vllm_config", args[0] if args else None)
        if config is None or not validate_separate_query_profile(config):
            raise ValueError("DSparkSeparateQueryScheduler requires the enabled separate-query profile")
        if "max_num_new_slots_for_drafting" not in Scheduler.schedule.__code__.co_names:
            raise RuntimeError("the installed scheduler does not expose the qualified draft admission budget")
        super().__init__(*args, **kwargs)
        self._dspark_config = self.vllm_config
        self._dspark_budget_config = SeparateQuerySchedulerConfig(self._dspark_config)
        self._dspark_scheduling = False
        if self.num_lookahead_tokens != config.num_lookahead_tokens:
            raise RuntimeError("separate-query scheduling changed the KV lookahead reservation")

    def schedule(self, throttle_prefills=False):
        if self._dspark_scheduling or self.vllm_config is not self._dspark_config:
            raise RuntimeError("separate-query scheduler cannot enter a concurrent or foreign configuration")
        self._dspark_scheduling = True
        self.vllm_config = self._dspark_budget_config
        try:
            return super().schedule(throttle_prefills)
        finally:
            self.vllm_config = self._dspark_config
            self._dspark_scheduling = False
