# SPDX-License-Identifier: Apache-2.0
"""Budget isolation and rollback contracts for a separate DSpark query pass."""
from types import SimpleNamespace

import pytest

from vllm_gaudi.ops.deepseek_v41_scheduling import (
    SEPARATE_QUERY_SCHEDULER,
    SeparateQuerySchedulerConfig,
    configure_separate_query_scheduler,
)
from vllm_gaudi.v1.core.sched.deepseek_v41_dspark_scheduler import DSparkSeparateQueryScheduler


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
        parallel_config=SimpleNamespace(tensor_parallel_size=8, pipeline_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=11, async_scheduling=False, scheduler_cls=None),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=SimpleNamespace(
            use_dspark=lambda: True, num_speculative_tokens=5, parallel_drafting=True,
            max_num_new_slots_for_drafting=4),
        additional_config={"dsv41_dspark_separate_query_pass": True},
        num_lookahead_tokens=5,
    )


def test_budget_view_does_not_change_engine_spec_or_kv_lookahead():
    config = _config()
    view = SeparateQuerySchedulerConfig(config)
    assert view.speculative_config.max_num_new_slots_for_drafting == 0
    assert config.speculative_config.max_num_new_slots_for_drafting == 4
    assert view.speculative_config.num_speculative_tokens == 5
    assert view.num_lookahead_tokens == 5
    assert view.scheduler_config is config.scheduler_config
    assert view.cache_config is config.cache_config
    assert view.speculative_config.use_dspark()


def test_profile_routes_only_the_explicit_separate_query_flag(monkeypatch):
    # vllm_gaudi.envs resolves flags lazily via module __getattr__; setattr would leave
    # a stale module attribute after undo, so patch the environment instead.
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    config = _config()
    configure_separate_query_scheduler(config)
    assert config.scheduler_config.scheduler_cls == SEPARATE_QUERY_SCHEDULER
    config.scheduler_config.scheduler_cls = None
    config.additional_config.clear()
    configure_separate_query_scheduler(config)
    assert config.scheduler_config.scheduler_cls is None


@pytest.mark.parametrize("change", [
    lambda c: setattr(c.scheduler_config, "async_scheduling", True),
    lambda c: setattr(c.cache_config, "enable_prefix_caching", True),
    lambda c: setattr(c.parallel_config, "tensor_parallel_size", 4),
    lambda c: setattr(c.speculative_config, "num_speculative_tokens", 4),
    lambda c: setattr(c, "num_lookahead_tokens", 0),
    lambda c: setattr(c.scheduler_config, "scheduler_cls", "custom.scheduler"),
])
def test_unsupported_profile_fails_before_scheduler_selection(monkeypatch, change):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    config = _config()
    change(config)
    before = config.scheduler_config.scheduler_cls
    with pytest.raises(ValueError):
        configure_separate_query_scheduler(config)
    assert config.scheduler_config.scheduler_cls == before
    assert config.speculative_config.max_num_new_slots_for_drafting == 4


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_flag_is_a_boolean(flag):
    config = _config()
    config.additional_config["dsv41_dspark_separate_query_pass"] = flag
    with pytest.raises(TypeError):
        configure_separate_query_scheduler(config)


@pytest.mark.parametrize("fail", [False, True])
def test_schedule_restores_original_config_after_success_or_exception(monkeypatch, fail):
    from vllm.v1.core.sched.scheduler import Scheduler

    config = _config()
    scheduler = object.__new__(DSparkSeparateQueryScheduler)
    scheduler._dspark_config = config
    scheduler._dspark_budget_config = SeparateQuerySchedulerConfig(config)
    scheduler._dspark_scheduling = False
    scheduler.vllm_config = config
    result = object()

    def schedule(self, throttle_prefills):
        assert throttle_prefills is True
        assert self.vllm_config.speculative_config.max_num_new_slots_for_drafting == 0
        assert self.vllm_config.num_lookahead_tokens == 5
        assert config.speculative_config.max_num_new_slots_for_drafting == 4
        if fail:
            raise RuntimeError("admission failed")
        return result

    monkeypatch.setattr(Scheduler, "schedule", schedule)
    if fail:
        with pytest.raises(RuntimeError, match="admission failed"):
            scheduler.schedule(True)
    else:
        assert scheduler.schedule(True) is result
    assert scheduler.vllm_config is config
    assert not scheduler._dspark_scheduling
