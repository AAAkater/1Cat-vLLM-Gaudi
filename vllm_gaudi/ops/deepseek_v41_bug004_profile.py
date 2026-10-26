# SPDX-License-Identifier: Apache-2.0
"""Opt-in BUG-004 packed-round diagnostics.

Default-off.  Each named segment has a host timestamp and current-stream HPU
event pair; one final synchronize drains the round before intervals are read.
A CPU profiler window supplies dispatcher counts by record_function parentage.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import torch

ENV = "VLLM_HPU_DSV41_BUG004_PROFILE"
SCHEMA = "dsv41.bug004.profile.v3"
PREFIX = "vllm_gaudi.dsv41.bug004.segment"


def enabled() -> bool:
    return os.environ.get(ENV, "0").lower() in ("1", "true", "yes", "on")


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
    def __init__(self, device: Any):
        self.prof = None
        self.reason = None
        self.enabled = device is not None and torch.device(device).type == "hpu"
        self._scopes = []
        if self.enabled:
            self._start()
        else:
            self.reason = "device is not HPU"

    def _start(self):
        try:
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=False,
                with_stack=False,
            )
            self.prof.start()
            self.reason = None
        except Exception as error:
            self.prof = None
            self.reason = f"{type(error).__name__}: {error}"

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
            return {"available": False, "reason": self.reason}
        try:
            self.prof.stop()
            events = list(self.prof.events())
        except Exception as error:
            return {"available": False, "reason": f"profiler finish failed: {type(error).__name__}: {error}"}
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
        self._profiler = _Profiler(self.device)
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


__all__ = ["ENV", "PREFIX", "SCHEMA", "RoundProfile", "begin_round", "enabled"]
