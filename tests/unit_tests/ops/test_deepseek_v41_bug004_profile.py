# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the opt-in BUG-004 packed-round diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from vllm_gaudi.ops.deepseek_v41_bug004_profile import (
    ENV,
    SCHEMA,
    RoundProfile,
    begin_round,
    enabled,
)


def _cache_metric():
    return {"available": False, "reason": "test"}


def _enable_hpu_trace(monkeypatch, tmp_path):
    from vllm_gaudi.ops import deepseek_v41_bug004_profile as module

    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(module.HPU_TRACE_ENV, "1")
    monkeypatch.delenv("VLLM_HPU_DSV41_BUG004_PROFILE_TRIGGER", raising=False)
    monkeypatch.setenv("DSV41_RUN_EVIDENCE", str(tmp_path))
    monkeypatch.setattr(module, "_HPU_TRACE_CLAIMED", False)
    monkeypatch.setattr(module, "_is_hpu", lambda device: True)
    monkeypatch.setattr(module, "_event", lambda device: None)
    monkeypatch.setattr(module, "_record", lambda event: None)
    monkeypatch.setattr(module, "_sync", lambda device: None)
    return module


class _DummyRecord:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeActivity:
    CPU = "CPU"
    HPU = "HPU"


class _FakeProfiler:
    instances = []
    exports = []

    def __init__(self, *, activities, **_kwargs):
        self.activities = list(activities)
        self.started = False
        self.stopped = False
        type(self).instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def export_chrome_trace(self, path):
        type(self).exports.append(path)
        Path(path).write_text("{}", encoding="utf-8")

    def events(self):
        return []


def test_disabled_observer_keeps_default_path(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("VLLM_HPU_DSV41_BUG004_PROFILE_TRIGGER", raising=False)
    monkeypatch.setenv("DSV41_RUN_EVIDENCE", str(tmp_path))
    assert begin_round(rank=0, tp_rank=0, pp_rank=0, device="cpu") is None
    assert not any(tmp_path.iterdir())


def test_trigger_file_enables_only_while_present(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    trigger = tmp_path / "profile.enable"
    monkeypatch.setenv("VLLM_HPU_DSV41_BUG004_PROFILE_TRIGGER", str(trigger))

    assert not enabled()
    trigger.write_text("")
    assert enabled()
    trigger.unlink()
    assert not enabled()


def test_profile_env_true_remains_compatible_without_trigger(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, "true")
    monkeypatch.setenv("VLLM_HPU_DSV41_BUG004_PROFILE_TRIGGER", str(tmp_path / "missing"))

    assert enabled()


def test_enabled_observer_requires_evidence(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.delenv("DSV41_RUN_EVIDENCE", raising=False)
    try:
        begin_round(rank=0, tp_rank=0, pp_rank=0, device="cpu")
    except RuntimeError as error:
        assert "DSV41_RUN_EVIDENCE" in str(error)
    else:
        raise AssertionError("enabled diagnostics must require an evidence directory")


def test_round_record_has_stable_segment_and_op_attribution(tmp_path):
    profile = RoundProfile(
        tmp_path,
        rank=3,
        tp_rank=2,
        pp_rank=1,
        device="cpu",
        cache_metric=_cache_metric,
    )
    torch.ones(1)
    profile.mark("target.forward")
    torch.zeros(1)
    profile.mark("draft.control_prepare")
    torch.tensor([1])
    record = profile.close()

    assert record["schema"] == SCHEMA
    assert record["rank"] == 3 and record["tp_rank"] == 2 and record["pp_rank"] == 1
    assert record["status"] == "complete"
    assert record["op_count"] == sum(segment["ops"] or 0 for segment in record["segments"])
    assert all(
        segment["ops"] is None or segment["ops"] == sum(segment["op_counts"].values())
        for segment in record["segments"]
    )
    assert record["profiler"]["available"] is False
    assert record["profiler"]["reason"] == "device is not HPU"

    lines = (tmp_path / "bug004-rounds.rank3.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["schema"] == SCHEMA


def test_enabled_observer_writes_one_jsonl_line_per_round(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, "true")
    monkeypatch.setenv("DSV41_RUN_EVIDENCE", str(tmp_path))
    for _ in range(2):
        profile = begin_round(rank=0, tp_rank=0, pp_rank=0, device="cpu")
        profile.mark("target.forward")
        torch.ones(1)
        profile.close()
    lines = (tmp_path / "bug004-rounds.rank0.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["status"] == "complete" for line in lines)


def test_segment_op_scopes_are_paired_and_named(monkeypatch, tmp_path):
    from vllm_gaudi.ops import deepseek_v41_bug004_profile as module

    scopes = []
    opened = []

    class Observer:
        def __init__(self, device, **_kwargs):
            pass

        def enter(self, label):
            assert not opened
            opened.append(label)
            scopes.append(("enter", label))

        def exit(self):
            scopes.append(("exit", opened.pop()))

        def result(self):
            assert not opened
            return {"available": False, "reason": "test observer"}

    monkeypatch.setattr(module, "_Profiler", Observer)
    profile = module.RoundProfile(tmp_path, rank=0, tp_rank=0, pp_rank=0,
                                  device="cpu", cache_metric=_cache_metric)
    profile.mark("target.forward")
    profile.close()
    assert scopes == [
        ("enter", f"{module.PREFIX}.round.start"),
        ("exit", f"{module.PREFIX}.round.start"),
        ("enter", f"{module.PREFIX}.target.forward"),
        ("exit", f"{module.PREFIX}.target.forward"),
    ]


def test_hpu_trace_uses_cpu_hpu_activities_and_exports_rank0(monkeypatch, tmp_path):
    module = _enable_hpu_trace(monkeypatch, tmp_path)
    _FakeProfiler.instances = []
    _FakeProfiler.exports = []
    monkeypatch.setattr(torch.profiler, "ProfilerActivity", _FakeActivity)
    monkeypatch.setattr(torch.profiler, "profile", _FakeProfiler)
    monkeypatch.setattr(torch.profiler, "record_function", lambda _label: _DummyRecord())

    profile = module.RoundProfile(tmp_path, rank=0, tp_rank=0, pp_rank=0,
                                  device="hpu", cache_metric=_cache_metric)
    profile.mark("target.forward")
    record = profile.close()

    assert _FakeProfiler.instances[0].activities == [_FakeActivity.CPU, _FakeActivity.HPU]
    assert record["profiler"]["available"] is True
    assert record["hpu_trace"]["status"] == "complete"
    assert record["hpu_trace"]["activities"] == ["CPU", "HPU"]
    trace_path = Path(record["hpu_trace"]["path"])
    assert trace_path == tmp_path / module.HPU_TRACE_FILE
    assert trace_path.is_file()


def test_hpu_trace_is_limited_to_one_rank0_round(monkeypatch, tmp_path):
    module = _enable_hpu_trace(monkeypatch, tmp_path)
    _FakeProfiler.instances = []
    _FakeProfiler.exports = []
    monkeypatch.setattr(torch.profiler, "ProfilerActivity", _FakeActivity)
    monkeypatch.setattr(torch.profiler, "profile", _FakeProfiler)
    monkeypatch.setattr(torch.profiler, "record_function", lambda _label: _DummyRecord())

    first = module.RoundProfile(tmp_path, rank=0, tp_rank=0, pp_rank=0,
                                device="hpu", cache_metric=_cache_metric)
    first_record = first.close()
    second = module.RoundProfile(tmp_path, rank=0, tp_rank=0, pp_rank=0,
                                 device="hpu", cache_metric=_cache_metric)
    second_record = second.close()

    assert first_record["hpu_trace"]["status"] == "complete"
    assert second_record["hpu_trace"]["status"] == "disabled"
    assert "one-round" in second_record["hpu_trace"]["reason"]
    assert len(_FakeProfiler.exports) == 1


def test_hpu_trace_does_not_export_on_rank1(monkeypatch, tmp_path):
    module = _enable_hpu_trace(monkeypatch, tmp_path)
    _FakeProfiler.instances = []
    _FakeProfiler.exports = []
    monkeypatch.setattr(torch.profiler, "ProfilerActivity", _FakeActivity)
    monkeypatch.setattr(torch.profiler, "profile", _FakeProfiler)
    monkeypatch.setattr(torch.profiler, "record_function", lambda _label: _DummyRecord())

    profile = module.RoundProfile(tmp_path, rank=1, tp_rank=1, pp_rank=0,
                                  device="hpu", cache_metric=_cache_metric)
    record = profile.close()

    assert record["hpu_trace"]["status"] == "disabled"
    assert "rank is not 0" in record["hpu_trace"]["reason"]
    assert not _FakeProfiler.exports


def test_hpu_trace_start_failure_is_reported_and_cpu_fallback_runs(monkeypatch, tmp_path):
    module = _enable_hpu_trace(monkeypatch, tmp_path)

    class FailingHpuProfiler(_FakeProfiler):
        def __init__(self, *, activities, **kwargs):
            if _FakeActivity.HPU in activities:
                raise RuntimeError("HPU activity unavailable")
            super().__init__(activities=activities, **kwargs)

    monkeypatch.setattr(torch.profiler, "ProfilerActivity", _FakeActivity)
    monkeypatch.setattr(torch.profiler, "profile", FailingHpuProfiler)
    monkeypatch.setattr(torch.profiler, "record_function", lambda _label: _DummyRecord())

    profile = module.RoundProfile(tmp_path, rank=0, tp_rank=0, pp_rank=0,
                                  device="hpu", cache_metric=_cache_metric)
    record = profile.close()

    assert record["status"] == "complete"
    assert record["profiler"]["available"] is True
    assert record["hpu_trace"]["status"] == "failed"
    assert "HPU activity unavailable" in record["hpu_trace"]["reason"]


def test_hpu_trace_requires_profile_gate(monkeypatch, tmp_path):
    from vllm_gaudi.ops import deepseek_v41_bug004_profile as module

    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(module.HPU_TRACE_ENV, "1")
    monkeypatch.delenv("VLLM_HPU_DSV41_BUG004_PROFILE_TRIGGER", raising=False)
    monkeypatch.setenv("DSV41_RUN_EVIDENCE", str(tmp_path))
    monkeypatch.setattr(module, "_HPU_TRACE_CLAIMED", False)

    assert module.hpu_trace_enabled() is False
    assert module.begin_round(rank=0, tp_rank=0, pp_rank=0, device="cpu") is None
    assert not (tmp_path / module.HPU_TRACE_FILE).exists()


def test_hpu_trace_follows_trigger_gate(monkeypatch, tmp_path):
    from vllm_gaudi.ops import deepseek_v41_bug004_profile as module

    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(module.HPU_TRACE_ENV, "1")
    trigger = tmp_path / "profile.enable"
    monkeypatch.setenv("VLLM_HPU_DSV41_BUG004_PROFILE_TRIGGER", str(trigger))
    monkeypatch.setattr(module, "_HPU_TRACE_CLAIMED", False)

    assert module.hpu_trace_enabled() is False
    trigger.write_text("", encoding="utf-8")
    assert module.hpu_trace_enabled() is True
    trigger.unlink()
    assert module.hpu_trace_enabled() is False


def test_hpu_trace_defaults_off_with_profile_enabled(monkeypatch, tmp_path):
    from vllm_gaudi.ops import deepseek_v41_bug004_profile as module

    monkeypatch.setenv(ENV, "1")
    monkeypatch.delenv(module.HPU_TRACE_ENV, raising=False)
    monkeypatch.setattr(module, "_HPU_TRACE_CLAIMED", False)

    profile = module.RoundProfile(tmp_path, rank=0, tp_rank=0, pp_rank=0,
                                  device="cpu", cache_metric=_cache_metric)
    record = profile.close()

    assert record["hpu_trace"]["status"] == "disabled"
    assert record["hpu_trace"]["enabled"] is False
    assert not (tmp_path / module.HPU_TRACE_FILE).exists()


def test_hpu_trace_export_failure_is_reported(monkeypatch, tmp_path):
    module = _enable_hpu_trace(monkeypatch, tmp_path)

    class ExportFailingProfiler(_FakeProfiler):
        def export_chrome_trace(self, path):
            raise RuntimeError("trace export failed")

    monkeypatch.setattr(torch.profiler, "ProfilerActivity", _FakeActivity)
    monkeypatch.setattr(torch.profiler, "profile", ExportFailingProfiler)
    monkeypatch.setattr(torch.profiler, "record_function", lambda _label: _DummyRecord())

    profile = module.RoundProfile(tmp_path, rank=0, tp_rank=0, pp_rank=0,
                                  device="hpu", cache_metric=_cache_metric)
    record = profile.close()

    assert record["status"] == "complete"
    assert record["profiler"]["available"] is True
    assert record["hpu_trace"]["status"] == "failed"
    assert "trace export failed" in record["hpu_trace"]["reason"]
