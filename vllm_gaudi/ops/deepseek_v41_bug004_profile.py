# SPDX-License-Identifier: Apache-2.0
"""Opt-in BUG-004 packed-round diagnostics.

Default-off.  Each named segment has a host timestamp and current-stream HPU
event pair; one final synchronize drains the round before intervals are read.
A CPU profiler window supplies dispatcher counts by record_function parentage.
An optional rank0 CPU+HPU Chrome trace covers only the first diagnostic round.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

import torch

ENV = "VLLM_HPU_DSV41_BUG004_PROFILE"
HPU_TRACE_ENV = "VLLM_HPU_DSV41_BUG004_HPU_TRACE"
HPU_TRACE_FILE = "bug004-hpu-rank0.trace.json"
SCHEMA = "dsv41.bug004.profile.v3"
PREFIX = "vllm_gaudi.dsv41.bug004.segment"
_HPU_TRACE_LOCK = threading.Lock()
_HPU_TRACE_CLAIMED = False


def enabled() -> bool:
    if os.environ.get(ENV, "0").lower() in ("1", "true", "yes", "on"):
        return True
    trigger = os.environ.get("VLLM_HPU_DSV41_BUG004_PROFILE_TRIGGER")
    return bool(trigger and Path(trigger).is_file())


def hpu_trace_enabled() -> bool:
    """Return true only when the profile gate and the HPU trace switch are both on."""
    return os.environ.get(HPU_TRACE_ENV, "0").lower() in ("1", "true", "yes", "on") and enabled()


def _is_hpu(device: Any) -> bool:
    return device is not None and torch.device(device).type == "hpu"


def _hpu_trace_plan(rank: int, directory: str | Path) -> dict:
    """Claim the one process-local rank0 trace without enabling any global profiler."""
    path = Path(directory) / HPU_TRACE_FILE
    try:
        if int(rank) != 0:
            return {"enabled": False, "status": "disabled", "reason": "rank is not 0", "path": str(path)}
        if os.environ.get(HPU_TRACE_ENV, "0").lower() not in ("1", "true", "yes", "on"):
            return {
                "enabled": False,
                "status": "disabled",
                "reason": f"{HPU_TRACE_ENV} is not enabled",
                "path": str(path),
            }
        if not enabled():
            return {
                "enabled": False,
                "status": "disabled",
                "reason": "BUG004 profile gate is not enabled",
                "path": str(path),
            }
        global _HPU_TRACE_CLAIMED
        with _HPU_TRACE_LOCK:
            if _HPU_TRACE_CLAIMED:
                return {"enabled": False, "status": "disabled", "reason": "one-round HPU trace cap already claimed",
                        "path": str(path)}
            _HPU_TRACE_CLAIMED = True
        return {"enabled": True, "status": "starting", "reason": None, "path": str(path)}
    except Exception as error:
        return {"enabled": False, "status": "failed", "reason": f"{type(error).__name__}: {error}", "path": str(path)}


def _profiler_activities(*, with_hpu: bool) -> list:
    activity = torch.profiler.ProfilerActivity
    if not with_hpu:
        return [activity.CPU]
    hpu = getattr(activity, "HPU", None)
    if hpu is None:
        raise RuntimeError("torch.profiler.ProfilerActivity.HPU is unavailable")
    return [activity.CPU, hpu]


def _activity_names(activities: list) -> list[str]:
    return [str(getattr(activity, "name", activity)) for activity in activities]


def _event(device: Any):
    if device is None or torch.device(device).type != "hpu":
        return None
    return torch.hpu.Event(enable_timing=True)


def _record(event) -> None:
    if event is not None:
        event.record(torch.hpu.current_stream())


def _sync(device: Any) -> None:
    if device is not None and torch.device(device).type == "hpu":
        torch.hpu.synchronize()


def _cache_metric() -> dict:
    """Read process-global bridge recipe counters without changing config."""
    if os.environ.get("PT_HPU_ENABLE_CACHE_METRICS", "0").lower() not in ("1", "true", "yes", "on"):
        return {"available": False, "reason": "PT_HPU_ENABLE_CACHE_METRICS is not enabled"}
    try:
        from habana_frameworks.torch.hpu.metrics import metric_global

        metric = metric_global("recipe_cache")
        if metric is None:
            return {"available": False, "reason": "recipe_cache was not registered at process start"}
        stats = dict(metric.stats())
        return {
            "available": True,
            "scope": "process-global cumulative bridge recipe cache events",
            "hit": int(stats.get("TotalHit", 0)),
            "miss": int(stats.get("TotalMiss", 0)),
            "recipe_hit": dict(stats.get("RecipeHit", {})),
            "recipe_miss": dict(stats.get("RecipeMiss", {})),
        }
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


class _Profiler:
    def __init__(self, device: Any, *, trace_plan: dict | None = None):
        self.prof = None
        self.reason = None
        self.enabled = _is_hpu(device)
        self.trace_plan = trace_plan or {
            "enabled": False,
            "status": "disabled",
            "reason": "no HPU trace plan",
            "path": None,
        }
        self.trace_status = self.trace_plan.get("status", "disabled")
        self.trace_reason = self.trace_plan.get("reason")
        self.trace_activities = None
        self.trace_bytes = None
        self._scopes = []
        if self.enabled:
            self._start()
        else:
            self.reason = "device is not HPU"
            if self.trace_plan.get("enabled"):
                self.trace_status = "failed"
                self.trace_reason = "device is not HPU"

    def _start_profiler(self, activities: list) -> None:
        self.prof = torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            with_stack=False,
        )
        self.prof.start()

    def _start(self):
        if self.trace_plan.get("enabled"):
            try:
                activities = _profiler_activities(with_hpu=True)
                self.trace_activities = _activity_names(activities)
                self._start_profiler(activities)
                self.trace_status = "running"
                self.trace_reason = None
                return
            except Exception as error:
                self.trace_status = "failed"
                self.trace_reason = f"HPU trace start failed: {type(error).__name__}: {error}"
                self.prof = None
                # Preserve the original CPU-only observer if the HPU activity is unavailable.
                try:
                    self._start_profiler(_profiler_activities(with_hpu=False))
                    self.reason = None
                except Exception as fallback_error:
                    self.prof = None
                    self.reason = f"{type(fallback_error).__name__}: {fallback_error}"
                return
        try:
            self._start_profiler(_profiler_activities(with_hpu=False))
            self.reason = None
        except Exception as error:
            self.prof = None
            self.reason = f"{type(error).__name__}: {error}"

    def _trace_result(self) -> dict:
        return {
            "enabled": bool(self.trace_plan.get("enabled")),
            "status": self.trace_status,
            "path": self.trace_plan.get("path"),
            "reason": self.trace_reason,
            "activities": self.trace_activities,
            "format": "chrome_trace",
            "bytes": self.trace_bytes,
        }

    def enter(self, label: str) -> None:
        if self.enabled and self.prof is not None:
            # A profiler record_function is process-scoped, not thread-local:
            # one paired scope may nest, but never overlap arbitrary spans.
            record = torch.profiler.record_function(label)
            record.__enter__()
            self._scopes.append(record)

    def exit(self) -> None:
        if self._scopes:
            self._scopes.pop().__exit__(None, None, None)

    def result(self) -> dict:
        if self.prof is None:
            return {"available": False, "reason": self.reason, "trace": self._trace_result()}
        try:
            self.prof.stop()
            if self.trace_status == "running":
                try:
                    trace_path = Path(self.trace_plan["path"])
                    self.prof.export_chrome_trace(str(trace_path))
                    self.trace_bytes = trace_path.stat().st_size if trace_path.is_file() else None
                    self.trace_status = "complete"
                    self.trace_reason = None
                except Exception as error:
                    self.trace_status = "failed"
                    self.trace_reason = f"HPU trace export failed: {type(error).__name__}: {error}"
            events = list(self.prof.events())
        except Exception as error:
            if self.trace_status == "running":
                self.trace_status = "failed"
                self.trace_reason = f"profiler finish failed: {type(error).__name__}: {error}"
            return {
                "available": False,
                "reason": f"profiler finish failed: {type(error).__name__}: {error}",
                "trace": self._trace_result(),
            }
        rows = []
        seen = set()
        for event in events:
            if not event.name.startswith(PREFIX):
                continue
            name = event.name[len(PREFIX) + 1:]
            if name == "__end__" or name in seen:
                continue
            seen.add(name)
            counts = Counter()
            for child in getattr(event, "cpu_children", ()) or ():
                if not child.name.startswith(PREFIX):
                    counts[child.name] += 1
            rows.append({
                "name": name,
                "host_ms": max(0.0, event.time_range.end - event.time_range.start) / 1000.0,
                "ops": sum(counts.values()),
                "op_counts": dict(sorted(counts.items())),
            })
        return {
            "available": True,
            "scope": "CPU profiler events under each segment record_function; top-level parent events only",
            "segments": rows,
            "event_count": len(events),
            "trace": self._trace_result(),
        }


class RoundProfile:
    """One rank-local round; one JSONL line is appended at close."""

    def __init__(self, directory: str | Path, *, rank: int, tp_rank: int, pp_rank: int,
                 device: Any, cache_metric: Callable[[], dict] = _cache_metric):
        self.rank, self.tp_rank, self.pp_rank = int(rank), int(tp_rank), int(pp_rank)
        self.device = None if device is None else torch.device(device)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"bug004-rounds.rank{self.rank}.jsonl"
        if not self.path.exists():
            self.path.write_text("")
        self.cache_metric = cache_metric
        self.cache_before = cache_metric()
        self._profiler = _Profiler(self.device, trace_plan=_hpu_trace_plan(self.rank, self.directory))
        self.started_ns = time.perf_counter_ns()
        self.segments: list[_Segment] = []
        self._profiler_segments: list[dict] = []
        self._open: _Segment | None = None
        self._gap_start_ns = None
        self._closed = False
        self.target_layers = []
        self.control_ops = None
        self.control_op_counts = {}
        self.start_event = _event(self.device)
        _record(self.start_event)
        self.mark("round.start")

    def mark(self, name: str) -> None:
        """Open one stable segment; callers must not emit it twice in a round."""
        if self._closed:
            raise RuntimeError("BUG004 profile is already closed")
        if any(segment.name == name for segment in self.segments):
            raise RuntimeError(f"BUG004 segment {name!r} was emitted twice")
        if self._open is not None:
            self._open.close()
            self._open = None
            self._gap_start_ns = time.perf_counter_ns()
        self._open = _Segment(name, self._profiler, self.device)
        self.segments.append(self._open)

    @property
    def top_level_ops(self):
        """Eager dispatch counting is profiler-based; signal that to old hooks."""
        return None

    def close(self, *, status: str = "complete", error: str | None = None) -> dict:
        if self._closed:
            raise RuntimeError("BUG004 profile is already closed")
        if self._open is not None:
            self._open.close()
            self._open = None
        end_event = _event(self.device)
        _record(end_event)
        cache_after = self.cache_metric()
        cache_delta = None
        if self.cache_before.get("available") and cache_after.get("available"):
            cache_delta = {
                "hit": cache_after["hit"] - self.cache_before["hit"],
                "miss": cache_after["miss"] - self.cache_before["miss"],
            }
            total = cache_delta["hit"] + cache_delta["miss"]
            cache_delta["rate"] = None if total == 0 else cache_delta["hit"] / total
        sync_started = time.perf_counter_ns()
        _sync(self.device)
        sync_ended = time.perf_counter_ns()
        profiler = self._profiler.result()
        hpu_trace = profiler.get("trace", {})
        self._closed = True
        profiler_rows = list(profiler.get("segments", [])) if profiler.get("available") else []
        profiler_index = 0
        rows = []
        previous_event = self.start_event
        for segment in self.segments:
            start_event, end_event = segment.start_event, segment.end_event
            device_ms = None
            if previous_event is not None and end_event is not None:
                device_ms = previous_event.elapsed_time(end_event)
            measure = None
            if profiler_rows and profiler_index < len(profiler_rows):
                candidate = profiler_rows[profiler_index]
                if candidate["name"] == segment.name:
                    measure = candidate
                    profiler_index += 1
            gap_ms = None
            if previous_event is not None and start_event is not None:
                gap_ms = previous_event.elapsed_time(start_event)
            rows.append({
                "name": segment.name,
                "host_gap_ms": gap_ms,
                "host_submit_ms": segment.host_ms,
                "device_ms": device_ms,
                "ops": None if measure is None else measure["ops"],
                "op_counts": {} if measure is None else measure["op_counts"],
            })
            if end_event is not None:
                previous_event = end_event
        record = {
            "schema": SCHEMA,
            "rank": self.rank,
            "tp_rank": self.tp_rank,
            "pp_rank": self.pp_rank,
            "status": status,
            "error": error,
            "host_total_ms": (sync_ended - self.started_ns) / 1e6,
            "sync_wait_ms": (sync_ended - sync_started) / 1e6,
            "op_count": sum(int(row["ops"] or 0) for row in rows),
            "segments": rows,
            "target_layers": self.target_layers,
            "control_ops": self.control_ops,
            "control_op_counts": self.control_op_counts,
            "cache": {"before": self.cache_before, "after": cache_after, "delta": cache_delta},
            "profiler": profiler,
            "hpu_trace": hpu_trace,
            "clock": "CLOCK_MONOTONIC host; current-stream HPU event offsets",
            "measurement": "one final synchronization; dispatcher counts are host-side profiler events",
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        return record


class _Segment:
    __slots__ = ("name", "record", "start_event", "end_event", "host_start_ns", "host_end_ns", "device")

    def __init__(self, name: str, record, device: Any):
        self.device = device
        self.name = name
        self.record = record
        self.record.enter(f"{PREFIX}.{name}")
        self.host_start_ns = time.perf_counter_ns()
        self.host_end_ns = None
        self.start_event = _event(device)
        self.end_event = None
        _record(self.start_event)

    @property
    def host_ms(self) -> float:
        return ((self.host_end_ns if self.host_end_ns is not None else time.perf_counter_ns()) -
                self.host_start_ns) / 1e6

    def close(self) -> None:
        self.end_event = _event(self.device)
        _record(self.end_event)
        self.host_end_ns = time.perf_counter_ns()
        self.record.exit()


def begin_round(*, rank: int, tp_rank: int, pp_rank: int, device: Any,
                directory: str | Path | None = None) -> RoundProfile | None:
    if not enabled():
        return None
    root = directory or os.environ.get("DSV41_RUN_EVIDENCE")
    if not root:
        raise RuntimeError(f"{ENV}=1 requires DSV41_RUN_EVIDENCE")
    return RoundProfile(root, rank=rank, tp_rank=tp_rank, pp_rank=pp_rank, device=device)


__all__ = [
    "ENV",
    "HPU_TRACE_ENV",
    "HPU_TRACE_FILE",
    "PREFIX",
    "SCHEMA",
    "RoundProfile",
    "begin_round",
    "enabled",
    "hpu_trace_enabled",
]
