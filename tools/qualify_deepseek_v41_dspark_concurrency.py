# SPDX-License-Identifier: Apache-2.0
"""Compact fixed-capacity DSpark concurrency and runner-evidence qualifier.

Two serial greedy references are collected for every client payload. All clients
then start behind one barrier; token IDs, timings, health, and metrics are
retained. Optional cancellation runs a survivor alongside the cancelled stream,
then reuses the cancelled prompt with a new request ID while the survivor is
still live. HTTP overlap is deliberately reported as HTTP evidence only.

A separate actual-packed JSONL trace may correlate one or more active clients to
one fixed-capacity batch. Correlation uses the explicit payload ``request_id``
and actual streamed completion response IDs (engine suffixes included); runner
code does not need client prompt hashes. Capacity ``N`` is checked separately
from active count ``A``. The strongest status emitted is
``runner_evidence_correlated``: the raw JSONL hash and packed fields are
retained, but this is not full serving/N qualification and the main agent must
inspect the trace before publishing a conclusion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_MODEL = "deepseek-v4.1-flash-dspark"
DEFAULT_URL = "http://127.0.0.1:8001"
REFERENCE_REPEATS = 2
MAX_OUTPUT_TOKENS = 32
MAX_PROMPT_TOKENS = 2048
MIN_CANCEL_TOKENS = 7
DEFAULT_CANCEL_AFTER = 7
CANCEL_BUDGET_HEADROOM = 8
DEFAULT_BUDGETS = (8, 16, 24, 32)
DEFAULT_BATCH_CAPACITY = 2
MIN_BATCH_CAPACITY = 2
MAX_BATCH_CAPACITY = 4
TARGET_ROWS_PER_LANE = 6
DRAFT_ROWS_PER_LANE = 5
TRACE_SCHEMA_V2 = "dsv41.dspark.round.v2"
TRACE_SCHEMA_LEGACY_B2 = "dsv41.dspark.b2_round.v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def payload(model: str, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "request_id": request["id"],
        "prompt": request["prompt_token_ids"],
        "temperature": 0,
        "n": 1,
        "max_tokens": request["max_tokens"],
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }


def hashes(model: str, request: dict[str, Any]) -> dict[str, str]:
    prompt = {"prompt_token_ids": request["prompt_token_ids"]}
    return {
        "case_id": request["case_id"],
        "prompt_sha256": digest(dump(prompt).encode()),
        "request_sha256": digest(dump(payload(model, request)).encode()),
    }


def build_requests(cases: int, clients: int, budgets: list[int] | None = None,
                   prompt_lengths: list[int] | None = None) -> list[dict[str, Any]]:
    if not 1 <= cases <= 8:
        raise ValueError("--cases must be in 1..8")
    if not 1 <= clients <= 12:
        raise ValueError("--clients must be in 1..12")
    values = list(budgets or DEFAULT_BUDGETS[:clients])[:clients]
    if len(values) != clients or len(set(values)) != clients:
        raise ValueError("provide one distinct output budget per client")
    if any(type(value) is not int or not 1 <= value <= MAX_OUTPUT_TOKENS for value in values):
        raise ValueError(f"client budgets must be integers in 1..{MAX_OUTPUT_TOKENS}")
    lengths = [7 + 4 * index for index in range(cases)]
    if prompt_lengths is not None:
        if len(prompt_lengths) != cases:
            raise ValueError("provide one prompt length per case")
        if any(type(value) is not int or not 1 <= value <= MAX_PROMPT_TOKENS for value in prompt_lengths):
            raise ValueError(f"prompt lengths must be integers in 1..{MAX_PROMPT_TOKENS}")
        lengths = list(prompt_lengths)
    case_rows = []
    for index in range(cases):
        length = lengths[index]
        start = 1000 * (index + 1)
        case_rows.append({"id": f"case-{index}", "prompt_token_ids": list(range(start, start + length))})
    return [{
        "id": f"{case_rows[index % cases]['id']}-b{budget:02d}-client-{index}",
        "case_id": case_rows[index % cases]["id"],
        "prompt_token_ids": list(case_rows[index % cases]["prompt_token_ids"]),
        "max_tokens": budget,
    } for index, budget in enumerate(values)]


def cancellation_requests(requests: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return distinct cancel, survivor, and reuse payloads without changing their refs."""
    if len(requests) < 2:
        raise ValueError("cancellation probe requires at least two client payloads")
    cancelled = {**requests[0], "id": requests[0]["id"] + "-cancel"}
    survivor = {**requests[1], "id": requests[1]["id"] + "-survivor"}
    reuse = {**requests[0], "id": requests[0]["id"] + "-reuse"}
    return cancelled, survivor, reuse


def http_get(base_url: str, path: str, timeout: float, api_key: str | None) -> dict[str, Any]:
    headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = Request(base_url.rstrip("/") + path, headers=headers)
    started = time.perf_counter_ns()
    try:
        with urlopen(request, timeout=timeout) as response:
            return {"status": response.status, "body": response.read(), "error": None,
                    "duration_ms": (time.perf_counter_ns() - started) / 1e6}
    except HTTPError as exc:
        return {"status": exc.code, "body": exc.read(), "error": None,
                "duration_ms": (time.perf_counter_ns() - started) / 1e6}
    except (OSError, URLError) as exc:
        return {"status": None, "body": b"", "error": f"{type(exc).__name__}: {exc}",
                "duration_ms": (time.perf_counter_ns() - started) / 1e6}


def save_health(base_url: str, destination: Path, timeout: float, api_key: str | None) -> dict[str, Any]:
    result = http_get(base_url, "/health", timeout, api_key)
    result["body"] = result["body"].decode("utf-8", "replace")
    write_json(destination, result)
    return result


def metric_value(document: str, name: str, model: str) -> float | None:
    pattern = rf'^{re.escape(name)}\{{[^}}]*model_name="{re.escape(model)}"[^}}]*\}}\s+([-+0-9.eE]+)'
    match = re.search(pattern, document, re.MULTILINE)
    return float(match.group(1)) if match else None


def save_metrics(base_url: str, model: str, destination: Path, timeout: float,
                 api_key: str | None) -> dict[str, Any]:
    result = http_get(base_url, "/metrics", timeout, api_key)
    body = result.pop("body")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    document = body.decode("utf-8", "replace")
    result.update({
        "path": str(destination), "body_sha256": digest(body),
        "num_requests_running": metric_value(document, "vllm:num_requests_running", model),
        "num_requests_waiting": metric_value(document, "vllm:num_requests_waiting", model),
    })
    return result


def connection(base_url: str, timeout: float) -> http.client.HTTPConnection:
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("service URL must be http(s)")
    factory = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    return factory(parsed.hostname, parsed.port, timeout=timeout)


def completions_path(base_url: str) -> str:
    return urlsplit(base_url).path.rstrip("/") + "/v1/completions"


def parse_sse(line: bytes, record: dict[str, Any], received_ns: int) -> None:
    line = line.strip()
    if not line.startswith(b"data:"):
        return
    body = line[5:].strip()
    if body == b"[DONE]":
        record["done"] = True
        return
    try:
        event = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid SSE JSON: {exc}") from exc
    if event.get("error") is not None:
        raise ValueError(f"API error: {event['error']}")
    if event.get("id") is not None and event["id"] not in record["response_ids"]:
        record["response_ids"].append(event["id"])
    if event.get("usage") is not None:
        record["usage"] = event["usage"]
    for choice in event.get("choices", []):
        ids = choice.get("token_ids") or []
        if ids:
            if record["first_token_ns"] is None:
                record["first_token_ns"] = received_ns
            record["last_token_ns"] = received_ns
            record["token_ids"].extend(ids)
            record["token_arrivals_ms"].append({
                "arrival_ms": (received_ns - record["started_ns"]) / 1e6,
                "count": len(ids),
            })
        if choice.get("finish_reason") is not None:
            record["finish_reason"] = choice["finish_reason"]


def run_request(base_url: str, model: str, request: dict[str, Any], directory: Path, timeout: float,
                api_key: str | None, phase: str, barrier: threading.Barrier | None = None,
                cancel_after: int | None = None) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    body = payload(model, request)
    write_json(directory / "request.json", {**request, **hashes(model, request), "phase": phase, "payload": body})
    record: dict[str, Any] = {
        "phase": phase, "request_id": request["id"], "max_tokens": request["max_tokens"],
        "started_ns": None, "first_token_ns": None, "last_token_ns": None, "finished_ns": None,
        "status": None, "response_ids": [], "token_ids": [], "token_arrivals_ms": [], "usage": None,
        "finish_reason": None, "done": False, "client_cancelled": False, "error": None,
        "cancel_requested_ns": None, "cancelled_after_tokens": None,
    }
    raw_path = directory / "response.sse"
    record.update({"directory": str(directory), "result_path": str(directory / "result.json"),
                   "raw_sse_path": str(raw_path)})
    client = connection(base_url, timeout)
    try:
        client.connect()
        if barrier is not None:
            barrier.wait(timeout=max(10.0, min(timeout, 30.0)))
        record["started_ns"] = time.perf_counter_ns()
        headers = {"Accept": "text/event-stream", "Accept-Encoding": "identity", "Connection": "close",
                   "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        client.request("POST", completions_path(base_url), body=dump(body).encode(), headers=headers)
        response = client.getresponse()
        record["status"] = response.status
        with raw_path.open("xb") as raw:
            for line in response:
                received_ns = time.perf_counter_ns()
                raw.write(line)
                raw.flush()
                parse_sse(line, record, received_ns)
                if cancel_after is not None and len(record["token_ids"]) >= cancel_after:
                    record["client_cancelled"] = True
                    record["cancel_requested_ns"] = received_ns
                    record["cancelled_after_tokens"] = len(record["token_ids"])
                    break
        response.close()
        if not record["client_cancelled"] and not record["done"]:
            record["error"] = "stream ended without [DONE]"
        if record["status"] is None or not 200 <= record["status"] < 300:
            record["error"] = f"HTTP {record['status']}"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        client.close()
        raw_path.touch(exist_ok=True)
        record["finished_ns"] = time.perf_counter_ns()
        if record["started_ns"] is not None:
            record["total_ms"] = (record["finished_ns"] - record["started_ns"]) / 1e6
        if record["first_token_ns"] is not None:
            record["ttft_ms"] = (record["first_token_ns"] - record["started_ns"]) / 1e6
        if record["last_token_ns"] is not None:
            record["last_token_ms"] = (record["last_token_ns"] - record["started_ns"]) / 1e6
        write_json(directory / "result.json", record)
    return record


def complete(record: dict[str, Any]) -> bool:
    usage = record.get("usage") or {}
    return bool(record.get("error") is None and record.get("status") == 200 and record.get("done")
                and record.get("finish_reason") == "length"
                and len(record.get("token_ids", [])) == record.get("max_tokens")
                and usage.get("completion_tokens") == len(record["token_ids"]))


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, pair in enumerate(zip(left, right)):
        if pair[0] != pair[1]:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def overlap(records: list[dict[str, Any]]) -> dict[str, Any]:
    spans = [(row["started_ns"], row["finished_ns"], row["request_id"]) for row in records
             if row.get("started_ns") is not None and row.get("finished_ns") is not None]
    points = sorted([(start, 1, rid) for start, _, rid in spans] + [(end, -1, rid) for _, end, rid in spans])
    active: set[str] = set()
    maximum = overlap_ns = 0
    previous = None
    for at, change, request_id in points:
        if previous is not None and len(active) >= 2:
            overlap_ns += at - previous
        (active.add if change == 1 else active.discard)(request_id)
        maximum = max(maximum, len(active))
        previous = at
    starts = [row[0] for row in spans]
    return {"client_overlap": len(spans) >= 2 and maximum >= 2 and overlap_ns > 0,
            "max_client_active": maximum,
            "start_spread_ms": (max(starts) - min(starts)) / 1e6 if starts else None,
            "overlap_ms": overlap_ns / 1e6,
            "scope": "client HTTP intervals only; not device-batch proof"}


def poll_metrics(stop: threading.Event, base_url: str, model: str, timeout: float, api_key: str | None,
                 directory: Path, interval: float, snapshots: list[dict[str, Any]]) -> None:
    index = 0
    while True:
        snapshots.append(save_metrics(base_url, model, directory / f"metrics-during-{index:03d}.txt",
                                      timeout, api_key))
        index += 1
        if stop.wait(interval):
            return


def wait_idle(base_url: str, model: str, timeout: float, api_key: str | None, directory: Path,
              drain_timeout: float, interval: float) -> dict[str, Any]:
    started, consecutive, count = time.monotonic(), 0, 0
    while time.monotonic() - started < drain_timeout:
        snap = save_metrics(base_url, model, directory / f"drain-{count:03d}.txt", timeout, api_key)
        count += 1
        if snap.get("num_requests_running") == 0 and snap.get("num_requests_waiting") == 0:
            consecutive += 1
            if consecutive == 2:
                return {"idle": True, "elapsed_s": time.monotonic() - started, "snapshots": count}
        else:
            consecutive = 0
        time.sleep(interval)
    return {"idle": False, "elapsed_s": time.monotonic() - started, "snapshots": count}


def evidence_field(row: dict[str, Any], *names: str) -> Any:
    nested = row.get("actual_packed") or row.get("actualpacked") or row.get("packed") or {}
    if isinstance(nested, dict):
        row = {**row, **nested}
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def validate_batch_capacity(value: Any, name: str = "batch capacity") -> int:
    if type(value) is not int or not MIN_BATCH_CAPACITY <= value <= MAX_BATCH_CAPACITY:
        raise ValueError(f"{name} must be an integer in {MIN_BATCH_CAPACITY}..{MAX_BATCH_CAPACITY}")
    return value


def request_activity_mask(value: Any, capacity: int) -> tuple[list[int] | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, list) or len(value) != capacity:
        return None, f"active mask must contain {capacity} booleans"
    mask = []
    for index, active in enumerate(value):
        if isinstance(active, bool):
            mask.append(int(active))
        elif type(active) is int and active in (0, 1):
            mask.append(active)
        else:
            return None, f"active mask[{index}] must be boolean or 0/1"
    return mask, None


def request_axis_values(value: Any, active_lanes: list[int], capacity: int, name: str, inactive_value: int,
                        *, require_fixed: bool = False) -> tuple[list[Any] | None, bool | None, str | None]:
    """Accept active-only ``[A]`` or fixed ``[N]`` fields and validate every inactive slot."""
    if value is None:
        return None, None, f"{name} is missing"
    if not isinstance(value, list):
        return None, None, f"{name} must be a list"
    if len(value) == capacity:
        for lane in range(capacity):
            if lane not in active_lanes and value[lane] != inactive_value:
                return None, None, f"inactive {name}[{lane}] must be {inactive_value}"
        return [value[lane] for lane in active_lanes], True, None
    if require_fixed:
        return None, None, f"{name} must contain {capacity} fixed lanes"
    if len(value) == len(active_lanes):
        return list(value), False, None
    return None, None, f"{name} length must be capacity N or active count A"


def event_observed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for name in ("observed", "completed", "complete", "ok", "passed"):
            if name in value:
                return event_observed(value[name])
        return True
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "observed", "completed", "complete", "ok", "passed")
    return bool(value)


def trace_event_flags(row: dict[str, Any]) -> dict[str, bool]:
    """Normalize legacy booleans and unnamed adapter ``events`` structures."""
    completion = evidence_field(row, "completion_event_observed", "completion_observed", "completion_event")
    final = evidence_field(row, "final_event_observed", "final_observed", "final_event")
    legacy_event = evidence_field(row, "eventobserved", "event_observed")
    completion = legacy_event if completion is None else completion
    final = legacy_event if final is None else final
    flags = {"completion": event_observed(completion) if completion is not None else False,
             "final": event_observed(final) if final is not None else False}
    events = evidence_field(row, "events", "event_flags", "trace_events")
    if isinstance(events, dict):
        entries = [(str(key), value) for key, value in events.items()]
    elif isinstance(events, list):
        entries = []
        for item in events:
            if isinstance(item, str):
                entries.append((item, True))
            elif isinstance(item, dict):
                name = item.get("name", item.get("event", item.get("type", "")))
                entries.append((str(name), item.get("observed", item.get("completed", item))))
    else:
        entries = []
    for key, value in entries:
        name = key.lower()
        observed = event_observed(value)
        if "completion" in name or "readback" in name:
            flags["completion"] = flags["completion"] or observed
        if "final" in name or "state_event" in name:
            flags["final"] = flags["final"] or observed
    return flags


def trace_aliases(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for record in records:
        request_id = record["request_id"]
        values = {request_id, f"cmpl-{request_id}"}
        for response_id in record.get("response_ids", []):
            values.add(response_id)
            values.add(response_id.removeprefix("cmpl-"))
        for value in tuple(values):
            for suffix in range(10):
                values.add(f"{value}-{suffix}")
        aliases[request_id] = values
    return aliases


def canonical_active_lanes(value: Any, reqids: Any, capacity: int, legacy: bool,
                           mask: list[int] | None = None) -> tuple[list[int] | None, str | None]:
    """Return canonical ascending physical lanes, or explain why the row is unusable."""
    mask_lanes = [lane for lane, active in enumerate(mask) if active] if mask is not None else None
    if value is None:
        if mask_lanes is not None:
            if not mask_lanes:
                return None, "active mask selects no request lanes"
            return mask_lanes, None
        if isinstance(reqids, list) and len(reqids) == capacity:
            lanes = [lane for lane, reqid in enumerate(reqids) if reqid is not None]
            return (lanes, None) if lanes else (None, "req_ids selects no request lanes")
        if legacy and isinstance(reqids, list) and 1 <= len(reqids) <= capacity:
            return list(range(len(reqids))), None
        return None, "missing active_lanes for active-only req_ids"
    if not isinstance(value, list) or not value:
        return None, "active_lanes must be a non-empty list"
    if any(type(lane) is not int for lane in value):
        return None, "active_lanes must contain integers"
    if value != sorted(set(value)):
        return None, "active_lanes must be unique and ascending"
    if any(lane < 0 or lane >= capacity for lane in value):
        return None, f"active_lanes must be in 0..{capacity - 1}"
    if mask_lanes is not None and value != mask_lanes:
        return None, "active_lanes differs from active mask"
    return list(value), None


def canonical_request_ids(reqids: Any, active_lanes: list[int], capacity: int) -> tuple[list[str] | None, str | None]:
    """Canonicalize active-only IDs or fixed-N IDs with ``None`` in inactive lanes."""
    if not isinstance(reqids, list):
        return None, "req_ids must be a list"
    if len(reqids) == capacity:
        active_ids = []
        for lane, reqid in enumerate(reqids):
            if lane in active_lanes:
                if not isinstance(reqid, str) or not reqid:
                    return None, f"active lane {lane} has no request ID"
                active_ids.append(reqid)
            elif reqid is not None:
                return None, f"inactive lane {lane} carries a request ID"
    elif len(reqids) == len(active_lanes):
        if any(not isinstance(reqid, str) or not reqid for reqid in reqids):
            return None, "active-only req_ids must contain non-empty strings"
        active_ids = list(reqids)
    else:
        return None, "req_ids length must be capacity N or active count A"
    if len(set(active_ids)) != len(active_ids):
        return None, "req_ids contains duplicate request IDs"
    return active_ids, None


def normalize_trace_event(row: dict[str, Any], expected_capacity: int,
                          aliases: dict[str, set[str]]) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one trace row and expose capacity ``N`` separately from activity ``A``."""
    schema = row.get("schema")
    if schema == TRACE_SCHEMA_V2:
        legacy = False
    elif schema in (None, TRACE_SCHEMA_LEGACY_B2):
        legacy = True
    else:
        return None, f"unknown trace schema {schema!r}"
    observed_capacity = evidence_field(row, "batch_capacity", "batchcapacity", "capacity")
    if observed_capacity is None:
        reqids_hint = evidence_field(row, "req_ids", "reqids", "request_ids")
        if legacy and expected_capacity == DEFAULT_BATCH_CAPACITY and isinstance(reqids_hint, list) and len(
                reqids_hint) <= expected_capacity:
            observed_capacity = expected_capacity
        else:
            return None, "missing batch_capacity (capacity N is not inferred from req_ids)"
    try:
        observed_capacity = validate_batch_capacity(observed_capacity, "trace batch_capacity")
    except ValueError as exc:
        return None, str(exc)
    if observed_capacity != expected_capacity:
        return None, f"trace capacity {observed_capacity} differs from configured {expected_capacity}"

    reqids = evidence_field(row, "req_ids", "reqids", "request_ids")
    raw_mask = evidence_field(row, "active_mask", "request_active", "activemask", "requestactive")
    active_mask, error = request_activity_mask(raw_mask, observed_capacity)
    if error is not None:
        return None, error
    raw_active_lanes = evidence_field(row, "active_lanes", "activelanes", "active_request_lanes")
    active_lanes, error = canonical_active_lanes(raw_active_lanes, reqids, observed_capacity, legacy, active_mask)
    if error is not None:
        return None, error
    active_ids, error = canonical_request_ids(reqids, active_lanes, observed_capacity)
    if error is not None:
        return None, error

    lane_reqids = evidence_field(row, "lane_request_ids")
    if lane_reqids is not None:
        expected = [None] * observed_capacity
        for lane, reqid in zip(active_lanes, active_ids, strict=True):
            expected[lane] = reqid
        if not isinstance(lane_reqids, list) or lane_reqids != expected:
            return None, "lane_request_ids must match active req_ids and be None in inactive lanes"

    slots, _, error = request_axis_values(evidence_field(row, "request_slots", "slots", "slot_ids"),
                                          active_lanes, observed_capacity, "request_slots", -1)
    if error is not None:
        return None, error
    gens, _, error = request_axis_values(
        evidence_field(row, "slot_generations", "gens", "generation_ids", "gen_ids"),
        active_lanes, observed_capacity, "slot_generations", 0)
    if error is not None:
        return None, error
    query_counts, _, error = request_axis_values(
        evidence_field(row, "logical_target_queries", "querycounts", "query_counts", "query_count", "target_counts"),
        active_lanes, observed_capacity, "logical_target_queries", 0)
    if error is not None:
        return None, error
    commit_counts, _, error = request_axis_values(
        evidence_field(row, "committed_counts", "commitcounts", "commit_counts", "commits"),
        active_lanes, observed_capacity, "committed_counts", 0)
    if error is not None:
        return None, error

    if any(type(value) is not int or value < 0 for value in slots):
        return None, "active request_slots must be nonnegative integers"
    if not legacy and slots != active_lanes:
        return None, "v2 active request_slots must equal active_lanes"
    if any(type(value) is not int or value < 1 for value in gens):
        return None, "active slot_generations must be positive integers"
    if any(type(value) is not int or not 1 <= value <= TARGET_ROWS_PER_LANE for value in query_counts):
        return None, f"active logical_target_queries must be integers in 1..{TARGET_ROWS_PER_LANE}"
    if any(type(value) is not int or value < 1 or value > query for value, query in zip(
            commit_counts, query_counts, strict=True)):
        return None, "active committed_counts must be integers in 1..logical_target_queries"

    output_counts = None
    raw_output_counts = evidence_field(row, "output_counts", "outputcounts", "output_count")
    if raw_output_counts is not None:
        output_counts, _, error = request_axis_values(raw_output_counts, active_lanes, observed_capacity,
                                                      "output_counts", 0)
        if error is not None:
            return None, error
        if any(type(value) is not int or value < 0 for value in output_counts):
            return None, "active output_counts must be nonnegative integers"

    fixed_prefix_fields = (
        ("lane_slot_generations", gens, "slot_generations"),
        ("lane_committed_counts", commit_counts, "committed_counts"),
        ("lane_output_counts", output_counts, "output_counts"),
    )
    for field, active_field, active_name in fixed_prefix_fields:
        raw_fixed = evidence_field(row, field)
        if raw_fixed is None:
            continue
        fixed_values, _, error = request_axis_values(raw_fixed, active_lanes, observed_capacity, field, 0,
                                                     require_fixed=True)
        if error is not None:
            return None, error
        if any(type(value) is not int or value < 0 for value in fixed_values):
            return None, f"active {field} must be nonnegative integers"
        if active_field is not None and fixed_values != active_field:
            return None, f"{field} active values differ from {active_name}"

    next_draft_counts = None
    raw_next_draft_counts = evidence_field(row, "next_draft_counts", "nextdraftcounts")
    if raw_next_draft_counts is not None:
        next_draft_counts, _, error = request_axis_values(raw_next_draft_counts, active_lanes, observed_capacity,
                                                          "next_draft_counts", 0)
        if error is not None:
            return None, error
        if any(type(value) is not int or value < 0 for value in next_draft_counts):
            return None, "active next_draft_counts must be nonnegative integers"
    raw_lane_next = evidence_field(row, "lane_next_draft_counts")
    if raw_lane_next is not None:
        lane_next, _, error = request_axis_values(raw_lane_next, active_lanes, observed_capacity,
                                                  "lane_next_draft_counts", 0, require_fixed=True)
        if error is not None:
            return None, error
        if any(type(value) is not int or value < 0 for value in lane_next):
            return None, "active lane_next_draft_counts must be nonnegative integers"
        if next_draft_counts is not None and lane_next != next_draft_counts:
            return None, "lane_next_draft_counts active values differ from next_draft_counts"

    target_rows = evidence_field(row, "actual_packed_target_rows", "targetrows", "target_rows", "targetrows12")
    draft_rows = evidence_field(row, "actual_packed_draft_rows", "draftrows", "draft_rows", "draftrows10")
    if target_rows != observed_capacity * TARGET_ROWS_PER_LANE:
        expected = observed_capacity * TARGET_ROWS_PER_LANE
        return None, f"actual_packed_target_rows must be {expected} for N={observed_capacity}"
    if draft_rows != observed_capacity * DRAFT_ROWS_PER_LANE:
        expected = observed_capacity * DRAFT_ROWS_PER_LANE
        return None, f"actual_packed_draft_rows must be {expected} for N={observed_capacity}"

    events = trace_event_flags(row)
    if not events["completion"] or not events["final"]:
        return None, "completion and final events were not both observed"
    batch_round = evidence_field(row, "batch_round", "batchround", "round")
    if type(batch_round) is not int or batch_round <= 0:
        return None, "batch_round must be a positive integer"

    assignments: dict[int, str] = {}
    for lane, reqid in zip(active_lanes, active_ids, strict=True):
        matches = [request_id for request_id, candidates in aliases.items() if reqid in candidates]
        if len(matches) != 1:
            return None, f"reqid {reqid!r} matched {len(matches)} HTTP request IDs"
        assignments[lane] = matches[0]
    request_evidence = {
        assignments[lane]: {"reqid": reqid, "lane": lane, "slot": slot, "generation": generation,
                            "query_count": query, "commit_count": commit,
                            "output_count": output_counts[index] if output_counts is not None else None,
                            "next_draft_count": next_draft_counts[index] if next_draft_counts is not None else None}
        for index, (lane, reqid, slot, generation, query, commit) in enumerate(zip(
            active_lanes, active_ids, slots, gens, query_counts, commit_counts, strict=True))
    }
    return {
        "schema": schema,
        "line": None,
        "batch_round": batch_round,
        "batch_capacity": observed_capacity,
        "active_lanes": active_lanes,
        "active_count": len(active_lanes),
        "capacity_fully_active": len(active_lanes) == observed_capacity,
        "reqids": active_ids,
        "request_ids": sorted(request_evidence),
        "slots": slots,
        "gens": gens,
        "query_counts": query_counts,
        "output_counts": output_counts,
        "next_draft_counts": next_draft_counts,
        "target_rows": target_rows,
        "draft_rows": draft_rows,
        "commit_counts": commit_counts,
        "request_evidence": request_evidence,
        "completion_event_observed": True,
        "final_event_observed": True,
        "events": events,
    }, None


def correlate_lifecycle(events: list[dict[str, Any]], lifecycle: dict[str, str]) -> dict[str, Any]:
    cancelled_id, survivor_id, reuse_id = (lifecycle[key] for key in ("cancelled", "survivor", "reuse"))
    result: dict[str, Any] = {
        "status": "runner_lifecycle_not_correlated", "cancel_line": None, "reuse_line": None,
        "slot_generation_advanced": None, "survivor_slot_generation_consistent": None,
    }
    for cancel_event in events:
        if not {cancelled_id, survivor_id}.issubset(cancel_event["request_ids"]):
            continue
        for reuse_event in events:
            if reuse_event["line"] <= cancel_event["line"] or not {reuse_id, survivor_id}.issubset(
                    reuse_event["request_ids"]):
                continue
            cancel_info = cancel_event["request_evidence"][cancelled_id]
            reuse_info = reuse_event["request_evidence"][reuse_id]
            survivor_a = cancel_event["request_evidence"][survivor_id]
            survivor_b = reuse_event["request_evidence"][survivor_id]
            if all(value is not None for value in (
                    cancel_info["slot"], cancel_info["generation"], reuse_info["slot"], reuse_info["generation"])):
                result["slot_generation_advanced"] = (
                    cancel_info["slot"] == reuse_info["slot"]
                    and reuse_info["generation"] > cancel_info["generation"])
            if all(value is not None for value in (
                    survivor_a["slot"], survivor_a["generation"], survivor_b["slot"], survivor_b["generation"])):
                result["survivor_slot_generation_consistent"] = (
                    survivor_a["slot"] == survivor_b["slot"]
                    and survivor_a["generation"] == survivor_b["generation"])
            if result["slot_generation_advanced"] is False or result[
                    "survivor_slot_generation_consistent"] is False:
                continue
            result.update({
                "status": "runner_lifecycle_correlated", "cancel_line": cancel_event["line"],
                "reuse_line": reuse_event["line"], "cancelled_request_evidence": cancel_info,
                "survivor_evidence": survivor_a, "reuse_request_evidence": reuse_info,
            })
            return result
    return result


def assess_evidence(path: Path | None, records: list[dict[str, Any]],
                    lifecycle: dict[str, str] | None = None,
                    batch_capacity: int = DEFAULT_BATCH_CAPACITY) -> dict[str, Any]:
    expected_capacity = validate_batch_capacity(batch_capacity, "configured batch capacity")
    result: dict[str, Any] = {
        "format": "actual-packed-jsonl-v2-or-legacy-b2",
        "status": "not_provided" if path is None else "runner_evidence_not_correlated",
        "inspection_required": True,
        "configured_batch_capacity": expected_capacity,
        "capacity_scope": "single trace file only; all-rank semantic digests are not assessed",
        "all_rank_digest_checked": False,
        "conclusion": ("HTTP and raw single-file trace correlation only; this is not full capacity, "
                       "serving, cancellation, graph-reuse, or TP8 qualification."),
        "matched_events": [], "raw_refs": [], "lifecycle": None, "observed_capacities": [],
        "max_active_count": None, "capacity_fully_active_observed": False,
    }
    if path is None:
        return result
    raw_ref: dict[str, Any] = {"path": str(path), "errors": []}
    try:
        raw = path.read_bytes()
        raw_ref["sha256"] = digest(raw)
        aliases = trace_aliases(records)
        rejected = []
        observed_capacities = set()
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raw_ref["errors"].append(f"line {line_number}: {exc}")
                continue
            if not isinstance(row, dict):
                continue
            capacity_value = evidence_field(row, "batch_capacity", "batchcapacity", "capacity")
            if type(capacity_value) is int:
                observed_capacities.add(capacity_value)
            event, error = normalize_trace_event(row, expected_capacity, aliases)
            if error is not None:
                if len(rejected) < 20:
                    rejected.append({"line": line_number, "reason": error})
                continue
            event["line"] = line_number
            result["matched_events"].append(event)
        if lifecycle is not None:
            result["lifecycle"] = correlate_lifecycle(result["matched_events"], lifecycle)
            if result["lifecycle"]["status"] != "runner_lifecycle_correlated":
                raw_ref["errors"].append("no later reuse+survivor packed line followed a cancel+survivor line")
        if result["matched_events"] and (
                lifecycle is None or result["lifecycle"]["status"] == "runner_lifecycle_correlated"):
            result["status"] = "runner_evidence_correlated"
        else:
            raw_ref["errors"].append(
                f"no packed line matched HTTP request IDs at capacity N={expected_capacity} with canonical "
                "active_lanes, inactive-no-ID, 6N/5N rows, and observed completion/final events")
        raw_ref["rejected_row_count"] = len(rejected)
        raw_ref["rejected_rows"] = rejected
        result["observed_capacities"] = sorted(observed_capacities)
        result["max_active_count"] = max((event["active_count"] for event in result["matched_events"]
                                          if isinstance(event, dict)), default=None)
        result["capacity_fully_active_observed"] = any(
            event["capacity_fully_active"] for event in result["matched_events"] if isinstance(event, dict))
    except (OSError, UnicodeDecodeError) as exc:
        raw_ref["errors"].append(f"{type(exc).__name__}: {exc}")
    result["raw_refs"] = [raw_ref]
    return result


def execute(config: dict[str, Any]) -> dict[str, Any]:
    capacity = validate_batch_capacity(config.get("batch_capacity", DEFAULT_BATCH_CAPACITY))
    if config["cancel_probe"] and config["clients"] < 2:
        raise ValueError("cancellation probe requires at least two clients")
    requests = build_requests(config["cases"], config["clients"], config["budgets"], config.get("prompt_lengths"))
    cancel_after = int(config.get("cancel_after", DEFAULT_CANCEL_AFTER))
    if config["cancel_probe"] and cancel_after < MIN_CANCEL_TOKENS:
        raise ValueError(f"cancel_after must be at least {MIN_CANCEL_TOKENS}")
    if config["cancel_probe"] and (
            requests[0]["max_tokens"] < cancel_after + CANCEL_BUDGET_HEADROOM
            or requests[1]["max_tokens"] < requests[0]["max_tokens"] + CANCEL_BUDGET_HEADROOM):
        raise ValueError("cancel probe needs budget headroom; use --budgets 16 24 for --cancel-after 7")
    output = config["output"].resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    api_key = config["api_key_file"].read_text().strip() if config["api_key_file"] else None
    report: dict[str, Any] = {
        "schema": "dsv41.dspark.concurrency_report.v2", "started_at": now(),
        "service": {"url": config["url"], "model": config["model"]},
        "cases": config["cases"], "clients": config["clients"], "batch_capacity": capacity,
        "prompt_lengths": [len(request["prompt_token_ids"]) for request in requests],
        "reference_repeats": REFERENCE_REPEATS,
        "request_matrix": [{**request, **hashes(config["model"], request)} for request in requests],
        "health": {}, "metrics": {}, "serial": {}, "overlap": {}, "cancellation": None, "runner_evidence": {},
    }
    summary: dict[str, Any] = {
        "schema": "dsv41.dspark.concurrency_summary.v2", "started_at": report["started_at"],
        "batch_capacity": capacity, "qualification_scope": (
            "HTTP checks plus optional single-file runner-trace correlation; not full capacity, serving, "
            "graph-reuse, cancellation, or TP8 qualification"),
        "all_rank_semantic_digest_checked": False,
    }
    try:
        health_before = save_health(config["url"], output / "health-before.json", config["timeout"], api_key)
        reference_rows: dict[str, list[dict[str, Any]]] = {request["id"]: [] for request in requests}
        for repeat in range(REFERENCE_REPEATS):
            for request in requests if repeat == 0 else reversed(requests):
                reference_rows[request["id"]].append(run_request(
                    config["url"], config["model"], request, output / "serial" / f"round-{repeat}" / request["id"],
                    config["timeout"], api_key, f"serial-round-{repeat}"))
        stability = {}
        for request in requests:
            rows = reference_rows[request["id"]]
            stable = (len(rows) == REFERENCE_REPEATS
                      and all(row["token_ids"] == rows[0]["token_ids"] for row in rows[1:]))
            stability[request["id"]] = {"stable": stable, "token_ids": rows[0]["token_ids"]}
        report["serial"] = {"stability": stability, "directory": str(output / "serial")}
        overlap_root = output / "overlap"
        metrics_before = save_metrics(config["url"], config["model"], overlap_root / "metrics-before.txt",
                                      config["timeout"], api_key)
        stop, during = threading.Event(), []
        poller = threading.Thread(target=poll_metrics, args=(
            stop, config["url"], config["model"], config["timeout"], api_key, overlap_root,
            config["metrics_interval"], during), daemon=True)
        poller.start()
        barrier = threading.Barrier(len(requests))
        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            futures = [pool.submit(run_request, config["url"], config["model"], request,
                                   overlap_root / request["id"], config["timeout"], api_key,
                                   "simultaneous", barrier) for request in requests]
            overlap_rows = [future.result() for future in futures]
        stop.set()
        poller.join(timeout=config["timeout"])
        metrics_after = save_metrics(config["url"], config["model"], overlap_root / "metrics-after.txt",
                                     config["timeout"], api_key)
        comparisons = [{
            "request_id": row["request_id"],
            "match": row["token_ids"] == stability[row["request_id"]]["token_ids"],
            "first_different_token": first_difference(row["token_ids"], stability[row["request_id"]]["token_ids"]),
        } for row in overlap_rows]
        running = [row.get("num_requests_running") for row in during if row.get("num_requests_running") is not None]
        waiting = [row.get("num_requests_waiting") for row in during if row.get("num_requests_waiting") is not None]
        overlap_result = overlap(overlap_rows)
        report["metrics"] = {
            "before": str(overlap_root / "metrics-before.txt"), "after": str(overlap_root / "metrics-after.txt"),
            "during_snapshots": len(during), "observed_max_running": max(running) if running else None,
            "observed_max_waiting": max(waiting) if waiting else None,
            "note": "Service-wide asynchronous context, not device-batch proof.",
        }
        report["overlap"] = {"overlap": overlap_result, "comparisons": comparisons,
                             "directory": str(overlap_root), "metrics_before": metrics_before["status"],
                             "metrics_after": metrics_after["status"]}
        cancellation = None
        lifecycle_records: list[dict[str, Any]] = []
        if config["cancel_probe"]:
            cancel_root = output / "cancellation"
            cancelled_request, survivor_request, reuse_request = cancellation_requests(requests)
            cancel_before = save_metrics(config["url"], config["model"], cancel_root / "metrics-before.txt",
                                         config["timeout"], api_key)
            cancel_stop, cancel_during = threading.Event(), []
            cancel_poller = threading.Thread(target=poll_metrics, args=(
                cancel_stop, config["url"], config["model"], config["timeout"], api_key, cancel_root,
                config["metrics_interval"], cancel_during), daemon=True)
            cancel_poller.start()
            cancel_barrier = threading.Barrier(2)
            with ThreadPoolExecutor(max_workers=3) as pool:
                cancelled_future = pool.submit(
                    run_request, config["url"], config["model"], cancelled_request, cancel_root / "cancelled",
                    config["timeout"], api_key, "cancel-overlap", cancel_barrier, cancel_after)
                survivor_future = pool.submit(
                    run_request, config["url"], config["model"], survivor_request, cancel_root / "survivor",
                    config["timeout"], api_key, "survivor-overlap", cancel_barrier)
                cancelled = cancelled_future.result()
                survivor_live_at_cancel = not survivor_future.done()
                reuse_live_at_submit = not survivor_future.done()
                reuse_future = pool.submit(
                    run_request, config["url"], config["model"], reuse_request, cancel_root / "reuse",
                    config["timeout"], api_key, "reuse-overlap")
                survivor, reuse = survivor_future.result(), reuse_future.result()
            cancel_stop.set()
            cancel_poller.join(timeout=config["timeout"])
            cancel_after_metrics = save_metrics(config["url"], config["model"], cancel_root / "metrics-after.txt",
                                                config["timeout"], api_key)
            drain = wait_idle(config["url"], config["model"], config["timeout"], api_key, cancel_root,
                              config["drain_timeout"], config["metrics_interval"])
            lifecycle_records = [cancelled, survivor, reuse]
            cancel_overlap = overlap([cancelled, survivor])
            reuse_overlap = overlap([survivor, reuse])
            reference_a = stability[requests[0]["id"]]["token_ids"]
            reference_b = stability[requests[1]["id"]]["token_ids"]
            survivor_match = complete(survivor) and survivor["token_ids"] == reference_b
            reuse_match = complete(reuse) and reuse["token_ids"] == reference_a
            cancel_running = [snap.get("num_requests_running") for snap in cancel_during
                              if snap.get("num_requests_running") is not None]
            cancel_waiting = [snap.get("num_requests_waiting") for snap in cancel_during
                              if snap.get("num_requests_waiting") is not None]
            cancel_metrics_ok = bool(cancel_before["status"] == 200 and cancel_after_metrics["status"] == 200
                                     and cancel_during)
            cancellation = {
                "passed": bool(cancelled["client_cancelled"] and not cancelled["done"] and drain["idle"]
                               and survivor_live_at_cancel and reuse_live_at_submit
                               and cancel_overlap["client_overlap"] and reuse_overlap["client_overlap"]
                               and survivor_match and reuse_match and cancel_metrics_ok),
                "cancel_after": cancel_after, "client_cancelled": cancelled["client_cancelled"],
                "cancelled_after_tokens": cancelled["cancelled_after_tokens"],
                "cancelled_prefix_match": cancelled["token_ids"] == reference_a[:len(cancelled["token_ids"])],
                "survivor_live_at_cancel": survivor_live_at_cancel,
                "reuse_live_at_submit": reuse_live_at_submit, "drain": drain,
                "survivor_complete": complete(survivor), "survivor_token_match": survivor_match,
                "reuse_complete": complete(reuse), "token_match": reuse_match,
                "reference_request_ids": {"survivor": requests[1]["id"], "reuse": requests[0]["id"]},
                "http_overlap": {"cancel_survivor": cancel_overlap, "reuse_survivor": reuse_overlap},
                "records": {
                    "cancelled": cancelled, "survivor": survivor, "reuse": reuse,
                },
                "metrics": {
                    "before": str(cancel_root / "metrics-before.txt"),
                    "after": str(cancel_root / "metrics-after.txt"), "during_snapshots": len(cancel_during),
                    "observed_max_running": max(cancel_running) if cancel_running else None,
                    "observed_max_waiting": max(cancel_waiting) if cancel_waiting else None,
                    "body_sha256": {"before": cancel_before.get("body_sha256"),
                                    "after": cancel_after_metrics.get("body_sha256")},
                },
                "directory": str(cancel_root),
            }
        report["cancellation"] = cancellation
        health_after = save_health(config["url"], output / "health-after.json", config["timeout"], api_key)
        report["health"] = {"before": health_before, "after": health_after}
        lifecycle = None
        if cancellation is not None:
            lifecycle = {"cancelled": lifecycle_records[0]["request_id"],
                         "survivor": lifecycle_records[1]["request_id"],
                         "reuse": lifecycle_records[2]["request_id"]}
        evidence = assess_evidence(config["runner_evidence"], overlap_rows + lifecycle_records, lifecycle,
                                   batch_capacity=capacity)
        report["runner_evidence"] = evidence
        references_stable = all(row["stable"] for row in stability.values())
        serial_complete = all(complete(row) for rows in reference_rows.values() for row in rows)
        overlap_complete = all(complete(row) for row in overlap_rows)
        all_match = all(row["match"] for row in comparisons)
        metrics_ok = metrics_before["status"] == 200 and metrics_after["status"] == 200 and bool(during)
        health_ok = health_before["status"] == 200 and health_after["status"] == 200
        concurrency_ok = overlap_result["client_overlap"] if len(requests) > 1 else True
        http_qualified = bool(references_stable and serial_complete and overlap_complete and all_match
                              and concurrency_ok and metrics_ok and health_ok)
        cancellation_ok = cancellation is None or cancellation["passed"]
        if not http_qualified or not cancellation_ok:
            status = "failed"
        elif evidence["status"] == "runner_evidence_correlated":
            status = "runner_evidence_correlated"
        elif evidence["status"] == "not_provided":
            status = "http_only_runner_evidence_not_provided"
        else:
            status = "http_only_runner_evidence_not_correlated"
        summary.update({
            "status": status, "http_qualified": http_qualified, "all_responses_match": all_match,
            "batch_capacity": capacity, "http_active_clients": len(requests),
            "active_clients_within_capacity": len(requests) <= capacity,
            "qualification_scope": ("HTTP checks plus optional single-file runner-trace correlation; not full "
                                    "capacity, serving, graph-reuse, cancellation, or TP8 qualification"),
            "references_stable": references_stable, "overlap": overlap_result, "metrics": report["metrics"],
            "health": {"before": health_before["status"], "after": health_after["status"]},
            "cancellation": cancellation, "runner_evidence": evidence, "inspection_required": True,
            "report_path": str(output / "report.json"), "summary_path": str(output / "summary.json"),
            "requests": [{"request_id": request["id"], "case_id": request["case_id"],
                          "prompt_length": len(request["prompt_token_ids"]), "max_tokens": request["max_tokens"],
                          "reference_stable": stability[request["id"]]["stable"],
                          "match": next(row["match"] for row in comparisons if row["request_id"] == request["id"])}
                         for request in requests],
        })
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        summary.update({"status": "failed", "http_qualified": False, "all_responses_match": False,
                        "batch_capacity": capacity, "http_active_clients": len(requests),
                        "error": report["error"], "inspection_required": True,
                        "report_path": str(output / "report.json"), "summary_path": str(output / "summary.json")})
    report["summary"] = summary
    report["finished_at"] = now()
    write_json(output / "summary.json", summary)
    write_json(output / "report.json", report)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--url", default=DEFAULT_URL)
    result.add_argument("--model", default=DEFAULT_MODEL)
    result.add_argument("--output", type=Path, required=True, help="new unique artifact directory")
    result.add_argument("--cases", type=int, default=2, help="distinct prompt cases (default: 2)")
    result.add_argument("--clients", type=int, default=3,
                        help="simultaneous clients, including A=1 partial activity (default: 3)")
    result.add_argument("--batch-capacity", type=int, default=DEFAULT_BATCH_CAPACITY,
                        help=f"fixed device batch capacity N in {MIN_BATCH_CAPACITY}..{MAX_BATCH_CAPACITY} "
                             f"(default: {DEFAULT_BATCH_CAPACITY})")
    result.add_argument("--budgets", type=int, nargs="+", help="one distinct budget per client; default 8,16,24,32")
    result.add_argument(
        "--prompt-lengths", type=int, nargs="+",
        help=f"one token-ID prompt length per case, bounded to 1..{MAX_PROMPT_TOKENS}; e.g. 259 511")
    result.add_argument(
        "--runner-evidence", type=Path,
        help="actual-packed JSONL trace for one rank; raw path, capacity, active lanes and hash are retained")
    result.add_argument(
        "--cancel-probe", action="store_true",
        help="cancel beside a live survivor, then reuse the slot beside that survivor (requires two clients)")
    result.add_argument(
        "--cancel-after", type=int, default=DEFAULT_CANCEL_AFTER,
        help=f"cancel after at least {MIN_CANCEL_TOKENS} streamed tokens (default: {DEFAULT_CANCEL_AFTER})")
    result.add_argument("--timeout", type=float, default=120.0)
    result.add_argument("--metrics-interval", type=float, default=0.1)
    result.add_argument("--drain-timeout", type=float, default=30.0)
    result.add_argument("--api-key-file", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        validate_batch_capacity(args.batch_capacity, "--batch-capacity")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.timeout <= 0 or args.metrics_interval <= 0 or args.drain_timeout <= 0:
        raise SystemExit("timeouts must be positive")
    if not MIN_CANCEL_TOKENS <= args.cancel_after <= MAX_OUTPUT_TOKENS:
        raise SystemExit(f"--cancel-after must be in {MIN_CANCEL_TOKENS}..{MAX_OUTPUT_TOKENS}")
    if args.prompt_lengths is not None and len(args.prompt_lengths) != args.cases:
        raise SystemExit("--prompt-lengths requires exactly --cases values")
    try:
        summary = execute({
            "url": args.url, "model": args.model, "output": args.output, "cases": args.cases,
            "clients": args.clients, "batch_capacity": args.batch_capacity, "budgets": args.budgets,
            "prompt_lengths": args.prompt_lengths,
            "runner_evidence": args.runner_evidence,
            "cancel_probe": args.cancel_probe, "cancel_after": args.cancel_after, "timeout": args.timeout,
            "metrics_interval": args.metrics_interval, "drain_timeout": args.drain_timeout,
            "api_key_file": args.api_key_file,
        })
    except Exception as exc:
        print(f"qualification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: summary.get(key) for key in (
        "status", "http_qualified", "batch_capacity", "http_active_clients", "all_responses_match", "overlap",
        "runner_evidence")},
        indent=2, ensure_ascii=False))
    if not summary.get("http_qualified"):
        return 1
    return 1 if summary.get("status") == "http_only_runner_evidence_not_correlated" else 0


if __name__ == "__main__":
    raise SystemExit(main())
