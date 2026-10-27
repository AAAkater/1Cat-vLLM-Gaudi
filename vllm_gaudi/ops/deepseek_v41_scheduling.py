# SPDX-License-Identifier: Apache-2.0
"""Scheduler-local budget policy for a separate HPU DSpark query pass."""

SEPARATE_QUERY_SCHEDULER = (
    "vllm_gaudi.v1.core.sched.deepseek_v41_dspark_scheduler.DSparkSeparateQueryScheduler"
)


class SeparateDraftBudget:
    """Delegate the speculative configuration while excluding draft input rows.

    This view is local to a synchronous scheduler call. It does not change the
    engine configuration, speculative token count, or KV lookahead reservation.
    """

    def __init__(self, speculative_config):
        self._speculative_config = speculative_config

    @property
    def max_num_new_slots_for_drafting(self):
        return 0

    def __getattr__(self, name):
        return getattr(self._speculative_config, name)


class SeparateQuerySchedulerConfig:
    """Read-only scheduler view; all non-budget fields retain their owners."""

    def __init__(self, config):
        self._config = config
        self._speculative_config = SeparateDraftBudget(config.speculative_config)

    @property
    def speculative_config(self):
        return self._speculative_config

    def __getattr__(self, name):
        return getattr(self._config, name)


def validate_separate_query_profile(config):
    from vllm_gaudi.ops.deepseek_v41_config import is_dspark_batch_tp8_profile

    enabled = (getattr(config, "additional_config", None) or {}).get("dsv41_dspark_separate_query_pass", False)
    if type(enabled) is not bool:
        raise TypeError("dsv41_dspark_separate_query_pass must be a boolean")
    if not enabled:
        return False
    if not is_dspark_batch_tp8_profile(config):
        raise ValueError("separate DSpark queries require the synchronous, prefix-off TP8 PP1 batch profile")
    spec = getattr(config, "speculative_config", None)
    if (spec is None or not callable(getattr(spec, "use_dspark", None)) or not spec.use_dspark()
            or getattr(spec, "num_speculative_tokens", None) != 5
            or getattr(spec, "parallel_drafting", None) is not True):
        raise ValueError("separate DSpark queries require the fixed five-token parallel DSpark profile")
    if getattr(config, "num_lookahead_tokens", None) != spec.num_speculative_tokens:
        raise ValueError("separate DSpark queries must preserve all five KV lookahead slots")
    return True


def configure_separate_query_scheduler(config):
    """Select the plugin scheduler without modifying upstream config classes."""
    enabled = validate_separate_query_profile(config)
    scheduler = config.scheduler_config
    selected = getattr(scheduler, "scheduler_cls", None)
    if isinstance(selected, type):
        selected = f"{selected.__module__}.{selected.__qualname__}"
    if not enabled:
        if selected == SEPARATE_QUERY_SCHEDULER:
            raise ValueError("the separate-query scheduler requires dsv41_dspark_separate_query_pass=true")
        return
    if selected not in (None, SEPARATE_QUERY_SCHEDULER):
        raise ValueError("separate DSpark queries cannot replace an explicitly configured scheduler")
    scheduler.scheduler_cls = SEPARATE_QUERY_SCHEDULER
