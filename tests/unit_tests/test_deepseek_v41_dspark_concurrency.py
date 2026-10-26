# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the compact DSpark concurrency/evidence qualifier."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import qualify_deepseek_v41_dspark_concurrency as qualifier  # noqa: E402

MODEL = "mock-dspark-model"


def sse_lines(request, *, include_id=True, include_done=True):
    response_id = f"cmpl-{request['request_id']}-0"
    prompt = request["prompt"]
    max_tokens = request["max_tokens"]
    base = sum(prompt) % 100000
    for index in range(max_tokens):
        choice = {
            "index": 0,
            "text": "",
            "token_ids": [(base + index) % 100000],
            "finish_reason": "length" if index == max_tokens - 1 else None,
        }
        if index == 0:
            choice["prompt_token_ids"] = prompt
        event = {
            "object": "text_completion",
            "model": MODEL,
            "choices": [choice],
        }
        if include_id:
            event["id"] = response_id
        yield ("data: " + json.dumps(event) + "\n\n").encode()
    event = {
        "choices": [],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": max_tokens,
            "total_tokens": len(prompt) + max_tokens,
        },
    }
    if include_id:
        event["id"] = response_id
    yield ("data: " + json.dumps(event) + "\n\n").encode()
    if include_done:
        yield b"data: [DONE]\n\n"


class MockServer(ThreadingHTTPServer):
    daemon_threads = True


class MockService(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    include_response_ids = True
    include_done = True
    metrics_status = 200

    def log_message(self, format, *args):
        return

    def send_body(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path == "/health":
            self.send_body(200, b"", "application/json")
        elif self.path == "/metrics":
            body = (
                f'vllm:num_requests_running{{engine="0",model_name="{MODEL}"}} 0.0\n'
                f'vllm:num_requests_waiting{{engine="0",model_name="{MODEL}"}} 0.0\n'
            ).encode()
            self.send_body(self.metrics_status, body, "text/plain")
        else:
            self.send_body(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_body(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        self.assert_payload_has_request_id(request)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for line in self.response_lines(request):
                self.wfile.write(line)
                self.wfile.flush()
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def response_lines(self, request):
        return sse_lines(
            request, include_id=self.include_response_ids, include_done=self.include_done)

    @staticmethod
    def assert_payload_has_request_id(request):
        if not request.get("request_id"):
            raise AssertionError("payload omitted explicit request_id")
        if not request.get("ignore_eos") or not request.get("stream"):
            raise AssertionError("payload did not retain streaming and ignore_eos")


class MissingIdService(MockService):
    include_response_ids = False


class TruncatedService(MockService):
    include_done = False


class MetricsDownService(MockService):
    metrics_status = 503


class SurvivorTruncatedService(MockService):
    def response_lines(self, request):
        return sse_lines(
            request, include_done=not request["request_id"].endswith("-survivor"))


class SurvivorMissingIdService(MockService):
    def response_lines(self, request):
        return sse_lines(
            request, include_id=not request["request_id"].endswith("-survivor"))


def start_mock(service=MockService):
    server = MockServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def runtime_config(url, output, **values):
    config = {
        "mode": qualifier.RUNTIME_SMOKE_MODE, "url": url, "model": MODEL, "output": output,
        "cases": 1, "clients": 2, "batch_capacity": 3, "budgets": [8], "prompt_lengths": [16],
        "runner_evidence": None, "cancel_probe": False, "cancel_after": 7,
        "timeout": 5.0, "metrics_interval": 0.01, "drain_timeout": 2.0, "api_key_file": None,
    }
    config.update(values)
    return config


def packed_row(reqids, complete=True):
    row = {
        "batch_round": 7,
        "reqids": reqids,
        "slots": [0, 1],
        "gens": [11, 12],
        "querycounts": [1, 1],
        "targetrows": 12,
        "draftrows": 10,
        "commitcounts": [1, 1],
        "eventobserved": True,
    }
    if not complete:
        row.pop("querycounts")
    return row


def actual_trace_row(reqids, generations, batch_round):
    width = len(reqids)
    return {
        "schema": "dsv41.dspark.round.v2",
        "batch_round": batch_round,
        "batch_capacity": width,
        "capacity": width,
        "active_count": width,
        "active_lanes": list(range(width)),
        "lanes": list(range(width)),
        "req_ids": reqids,
        "lane_request_ids": reqids,
        "request_slots": list(range(width)),
        "slot_generations": generations,
        "logical_target_queries": [6] * width,
        "draft_counts": [5] * width,
        "scheduler_draft_counts": [5] * width,
        "actual_packed_target_rows": width * 6,
        "actual_packed_draft_rows": width * 5,
        "physical_target_rows": width * 6,
        "physical_draft_rows": width * 5,
        "batch_width": width,
        "committed_counts": [1] * width,
        "output_counts": [1] * width,
        "next_draft_counts": [5] * width,
        "metadata_digest": "m" * 64,
        "output_ids_digest": "o" * 64,
        "next_draft_digest": "d" * 64,
        "completion_event_observed": True,
        "completion_observed_ns": batch_round * 100,
        "final_event_observed": True,
        "final_event_observed_ns": batch_round * 100 + 1,
        "target_stage": "mock.target",
        "draft_stage": "mock.draft",
    }


def a1_trace_row_at_capacity(request_id, capacity, generation, batch_round):
    row = actual_trace_row([request_id], [generation], batch_round)
    row.update({
        "batch_capacity": capacity, "capacity": capacity, "active_count": 1,
        "active_lanes": [0], "lanes": [0], "batch_width": 1,
        "req_ids": [request_id] + [None] * (capacity - 1),
        "lane_request_ids": [request_id] + [None] * (capacity - 1),
        "request_slots": [0] + [-1] * (capacity - 1),
        "slot_generations": [generation] + [0] * (capacity - 1),
        "logical_target_queries": [6] + [0] * (capacity - 1),
        "draft_counts": [5] + [0] * (capacity - 1),
        "scheduler_draft_counts": [5] + [0] * (capacity - 1),
        "actual_packed_target_rows": capacity * 6, "physical_target_rows": capacity * 6,
        "actual_packed_draft_rows": capacity * 5, "physical_draft_rows": capacity * 5,
        "committed_counts": [1] + [0] * (capacity - 1),
        "output_counts": [1] + [0] * (capacity - 1),
        "next_draft_counts": [5] + [0] * (capacity - 1),
    })
    return row


def write_trace(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "runner-trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def append_trace(path, rows):
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
        stream.flush()


def run_request_appending_trace(path, rows):
    real_run_request = qualifier.run_request
    lock = threading.Lock()
    appended = False

    def run(*args, **kwargs):
        nonlocal appended
        with lock:
            if not appended:
                append_trace(path, rows)
                appended = True
        return real_run_request(*args, **kwargs)

    return mock.patch.object(qualifier, "run_request", side_effect=run)


def records_for(requests, pair):
    return [{"request_id": request["id"], "response_ids": [f"cmpl-{request['id']}-0"]}
            for request in requests if request["id"] in pair]


class CompactQualifierTests(unittest.TestCase):
    def test_two_cases_three_clients_and_explicit_request_id(self):
        requests = qualifier.build_requests(2, 3, None)
        self.assertEqual([row["case_id"] for row in requests], ["case-0", "case-1", "case-0"])
        self.assertEqual([row["max_tokens"] for row in requests], [8, 16, 24])
        body = qualifier.payload(MODEL, requests[0])
        self.assertEqual(body["request_id"], requests[0]["id"])
        self.assertNotIn("prompt_sha256", body)
        self.assertNotIn("request_sha256", body)

    def test_two_explicit_prompt_lengths_cross_the_256_token_swa_window(self):
        requests = qualifier.build_requests(2, 2, [8, 16], [259, 511])
        self.assertEqual([len(row["prompt_token_ids"]) for row in requests], [259, 511])
        self.assertTrue(all(len(row["prompt_token_ids"]) > 255 for row in requests))
        self.assertEqual(requests[0]["prompt_token_ids"][0], 1000)
        self.assertEqual(requests[1]["prompt_token_ids"][0], 2000)
        self.assertTrue(qualifier.payload(MODEL, requests[0])["ignore_eos"])
        with self.assertRaisesRegex(ValueError, "1..2048"):
            qualifier.build_requests(2, 2, [8, 16], [259, 2049])
        with self.assertRaisesRegex(ValueError, "budget headroom"):
            qualifier.execute({
                "cases": 2, "clients": 2, "budgets": [8, 16], "prompt_lengths": None,
                "cancel_probe": True, "cancel_after": 7,
            })

    def test_trace_aliases_normalize_engine_completion_suffix_without_enumeration(self):
        request_id = "case-0-b08-client-0"
        response_id = f"cmpl-{request_id}-17-deadbeef"
        aliases = qualifier.trace_aliases([{
            "request_id": request_id,
            "response_ids": [response_id],
        }])

        self.assertIn(response_id, aliases[request_id])
        self.assertIn(f"cmpl-{request_id}/{request_id}", aliases[request_id])
        self.assertIn(f"cmpl-{request_id}", aliases[request_id])

    def test_exact_sse_id_correlates_to_engine_suffixed_trace_owner(self):
        request_id = "custom-client"
        response_id = f"cmpl-{request_id}"
        trace_reqid = f"{response_id}-0-0123abcd"
        aliases = qualifier.trace_aliases([{"request_id": request_id, "response_ids": [response_id]}])
        self.assertIn(response_id, aliases[request_id])
        row = actual_trace_row([trace_reqid], [9], 17)
        row.update({
            "batch_capacity": 2,
            "capacity": 2,
            "active_count": 1,
            "active_lanes": [1],
            "lanes": [1],
            "req_ids": [None, trace_reqid],
            "lane_request_ids": [None, trace_reqid],
            "request_slots": [-1, 1],
            "slot_generations": [0, 9],
            "logical_target_queries": [0, 6],
            "draft_counts": [0, 5],
            "scheduler_draft_counts": [0, 5],
            "actual_packed_target_rows": 12,
            "actual_packed_draft_rows": 10,
            "physical_target_rows": 12,
            "physical_draft_rows": 10,
            "committed_counts": [0, 1],
            "output_counts": [0, 1],
            "next_draft_counts": [0, 1],
        })
        with tempfile.TemporaryDirectory() as temporary:
            trace = write_trace(Path(temporary), [row])
            result = qualifier.assess_evidence(
                trace,
                [{"request_id": request_id, "response_ids": [response_id]}],
                batch_capacity=2,
            )
        self.assertEqual(result["status"], "runner_evidence_correlated")
        self.assertEqual(result["matched_events"][0]["request_ids"], [request_id])

    def test_engine_suffix_canonicalization_is_anchored(self):
        request_id = "custom-client"
        canonical = f"cmpl-{request_id}/{request_id}"
        for value in (
                f"cmpl-{request_id}",
                f"cmpl-{request_id}-0",
                f"cmpl-{request_id}-0-0123abcd",
        ):
            with self.subTest(value=value):
                self.assertEqual(qualifier.canonical_trace_alias(value, request_id), canonical)
        for value in (
                f"cmpl-{request_id}-0-0123abcdef",
                f"cmpl-{request_id}-x-0123abcd",
                f"prefix-cmpl-{request_id}-0-0123abcd",
                f"cmpl-{request_id}-0-0123abcd-suffix",
        ):
            with self.subTest(value=value):
                self.assertIsNone(qualifier.canonical_trace_alias(value, request_id))

    def test_two_of_three_http_ids_correlate_to_one_packed_jsonl_batch(self):
        requests = qualifier.build_requests(2, 3, None)
        pair = [requests[0]["id"], requests[2]["id"]]
        records = records_for(requests, pair)
        reqids = [f"cmpl-{pair[0]}-0", f"{pair[1]}-0"]
        with tempfile.TemporaryDirectory() as temporary:
            trace = write_trace(Path(temporary), [packed_row(reqids)])
            result = qualifier.assess_evidence(trace, records)
        self.assertEqual(result["status"], "runner_evidence_correlated")
        self.assertEqual(result["matched_events"][0]["request_ids"], sorted(pair))
        self.assertEqual(result["matched_events"][0]["target_rows"], 12)
        self.assertEqual(result["matched_events"][0]["draft_rows"], 10)
        self.assertEqual(len(result["raw_refs"][0]["sha256"]), 64)
        self.assertTrue(result["inspection_required"])
        self.assertNotIn("b2_qualified", result)

    def test_json_boolean_alone_is_not_evidence(self):
        requests = qualifier.build_requests(2, 3, None)
        pair = [requests[0]["id"], requests[2]["id"]]
        records = records_for(requests, pair)
        reqids = [requests[0]["id"], requests[2]["id"]]
        with tempfile.TemporaryDirectory() as temporary:
            trace = write_trace(Path(temporary), [{"reqids": reqids, "eventobserved": True}])
            result = qualifier.assess_evidence(trace, records)
        self.assertEqual(result["status"], "runner_evidence_not_correlated")
        self.assertFalse(result["matched_events"])
        self.assertTrue(result["inspection_required"])

    def test_packed_trace_requires_all_declared_fields(self):
        requests = qualifier.build_requests(2, 3, None)
        pair = [requests[0]["id"], requests[1]["id"]]
        records = records_for(requests, pair)
        with tempfile.TemporaryDirectory() as temporary:
            trace = write_trace(Path(temporary), [packed_row(pair, complete=False)])
            result = qualifier.assess_evidence(trace, records)
        self.assertEqual(result["status"], "runner_evidence_not_correlated")

    def test_mock_cancel_with_survivor_then_new_generation_reuse_and_raw_trace(self):
        server = MockServer(("127.0.0.1", 0), MockService)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                requests = qualifier.build_requests(2, 2, [16, 24], [259, 511])
                cancelled, survivor, reuse = qualifier.cancellation_requests(requests)
                trace = write_trace(root / "evidence", [
                    actual_trace_row([f"cmpl-{cancelled['id']}-0", f"cmpl-{survivor['id']}-0"], [3, 7], 10),
                    actual_trace_row([f"cmpl-{reuse['id']}-0", f"cmpl-{survivor['id']}-0"], [11, 7], 11),
                ])
                output = root / "result"
                summary = qualifier.execute({
                    "url": f"http://127.0.0.1:{server.server_port}", "model": MODEL,
                    "output": output, "cases": 2, "clients": 2, "budgets": [16, 24],
                    "prompt_lengths": [259, 511], "runner_evidence": trace,
                    "cancel_probe": True, "cancel_after": 7,
                    "timeout": 5.0, "metrics_interval": 0.01, "drain_timeout": 2.0,
                    "api_key_file": None,
                })
                self.assertEqual(summary["status"], "runner_evidence_correlated")
                self.assertTrue(summary["http_qualified"])
                self.assertTrue(summary["all_responses_match"])
                self.assertEqual(summary["overlap"]["max_client_active"], 2)
                self.assertTrue(summary["cancellation"]["passed"])
                self.assertGreaterEqual(summary["cancellation"]["cancelled_after_tokens"], 7)
                self.assertTrue(summary["cancellation"]["cancelled_prefix_match"])
                self.assertTrue(summary["cancellation"]["survivor_token_match"])
                self.assertTrue(summary["cancellation"]["token_match"])
                self.assertTrue(summary["cancellation"]["http_overlap"]["cancel_survivor"]["client_overlap"])
                self.assertTrue(summary["cancellation"]["http_overlap"]["reuse_survivor"]["client_overlap"])
                self.assertTrue(summary["inspection_required"])
                self.assertEqual(summary["runner_evidence"]["status"], "runner_evidence_correlated")
                lifecycle = summary["runner_evidence"]["lifecycle"]
                self.assertEqual(lifecycle["status"], "runner_lifecycle_correlated")
                self.assertTrue(lifecycle["slot_generation_advanced"])
                self.assertTrue(lifecycle["survivor_slot_generation_consistent"])
                self.assertEqual(summary["requests"][0]["prompt_length"], 259)
                self.assertEqual(summary["requests"][1]["prompt_length"], 511)
                serialized = json.dumps(summary)
                self.assertNotIn("b2_qualified", serialized)
                self.assertNotIn("b2_claim_allowed", serialized)
                self.assertTrue((output / "summary.json").is_file())
                self.assertTrue((output / "report.json").is_file())
                for name in ("metrics-before.txt", "metrics-during-000.txt", "metrics-after.txt"):
                    self.assertTrue((output / "cancellation" / name).is_file())
                records = summary["cancellation"]["records"]
                self.assertEqual(records["cancelled"]["request_id"], cancelled["id"])
                self.assertEqual(records["survivor"]["request_id"], survivor["id"])
                self.assertEqual(records["reuse"]["request_id"], reuse["id"])
                for role in ("cancelled", "survivor", "reuse"):
                    self.assertTrue(Path(records[role]["raw_sse_path"]).is_file())
                    self.assertTrue(Path(records[role]["result_path"]).is_file())
                    self.assertTrue(records[role]["response_ids"])
                result = json.loads((output / "overlap" / requests[0]["id"] / "result.json").read_text())
                self.assertEqual(result["response_ids"], [f"cmpl-{requests[0]['id']}-0"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_runtime_smoke_skips_serial_reference_and_records_liveness_only(self):
        server, thread = start_mock()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "runtime-result"
                summary = qualifier.execute(runtime_config(
                    f"http://127.0.0.1:{server.server_port}", output))
                self.assertEqual(summary["status"], "runtime_smoke_passed")
                self.assertEqual(summary["runtime_status"], "passed")
                self.assertTrue(summary["runtime_qualified"])
                self.assertFalse(summary["http_qualified"])
                self.assertFalse(summary["formal_qualification"])
                self.assertFalse(summary["semantic_exact_checked"])
                self.assertIsNone(summary["all_responses_match"])
                self.assertIsNone(summary["references_stable"])
                self.assertEqual(summary["serial"], {"status": "not_run"})
                self.assertTrue(all(summary["runtime_checks"].values()))
                self.assertEqual(len(summary["requests"]), 2)
                self.assertTrue(all(row["passed"] and row["response_ids"] for row in summary["requests"]))
                self.assertFalse((output / "serial").exists())
                self.assertTrue((output / "summary.json").is_file())
                self.assertTrue((output / "report.json").is_file())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_runtime_smoke_does_not_reuse_prebaseline_rows_with_same_request_ids(self):
        server, thread = start_mock()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                requests = qualifier.build_requests(1, 2, [8], [16])
                reqids = [request["id"] for request in requests]
                trace = write_trace(root / "evidence", [actual_trace_row(reqids, [1, 2], 3)])
                baseline_size = trace.stat().st_size
                summary = qualifier.execute(runtime_config(
                    f"http://127.0.0.1:{server.server_port}", root / "result",
                    batch_capacity=2, runner_evidence=trace))
                self.assertEqual(summary["status"], "runtime_smoke_failed")
                self.assertFalse(summary["runtime_qualified"])
                self.assertFalse(summary["runtime_checks"]["runner_evidence_ok"])
                self.assertEqual(summary["runner_evidence"]["status"], "runner_evidence_not_correlated")
                self.assertEqual(summary["runner_evidence"]["matched_events"], [])
                raw_ref = summary["runner_evidence"]["raw_refs"][0]
                self.assertTrue(raw_ref["boundary_valid"])
                self.assertEqual(raw_ref["start_byte"], baseline_size)
                self.assertEqual(raw_ref["appended_bytes"], 0)
                self.assertEqual(len(raw_ref["sha256"]), 64)
                self.assertEqual(raw_ref["boundary"]["start_byte"], baseline_size)
                self.assertEqual(summary["runner_evidence_boundary"]["start_byte"], baseline_size)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_runtime_smoke_accepts_only_trace_rows_appended_after_baseline(self):
        server, thread = start_mock()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                requests = qualifier.build_requests(1, 3, [8], [16])
                reqids = [request["id"] for request in requests]
                trace = write_trace(root / "evidence", [a1_trace_row_at_capacity(reqids[0], 3, 1, 1)])
                baseline_size = trace.stat().st_size
                unbounded_probe = qualifier.assess_evidence(
                    trace, records_for(requests, [reqids[0]]), batch_capacity=3)
                self.assertEqual(unbounded_probe["status"], "runner_evidence_correlated")
                current_row = actual_trace_row(reqids, [2, 3, 4], 2)
                with run_request_appending_trace(trace, [current_row]):
                    summary = qualifier.execute(runtime_config(
                        f"http://127.0.0.1:{server.server_port}", root / "result",
                        clients=3, batch_capacity=3, runner_evidence=trace))
                self.assertEqual(summary["status"], "runtime_smoke_evidence_correlated")
                self.assertTrue(summary["runtime_qualified"])
                self.assertTrue(summary["runtime_checks"]["runner_evidence_ok"])
                raw_ref = summary["runner_evidence"]["raw_refs"][0]
                self.assertTrue(raw_ref["boundary_valid"])
                self.assertEqual(raw_ref["start_byte"], baseline_size)
                self.assertGreater(raw_ref["appended_bytes"], 0)
                self.assertEqual(raw_ref["observed_bytes"], trace.stat().st_size)
                self.assertEqual(len(raw_ref["sha256"]), 64)
                self.assertEqual(len(raw_ref["appended_sha256"]), 64)
                self.assertEqual(raw_ref["boundary"]["prefix_sha256"], raw_ref["prefix_sha256"])
                self.assertEqual(len(summary["runner_evidence"]["matched_events"]), 1)
                event = summary["runner_evidence"]["matched_events"][0]
                self.assertEqual(event["batch_capacity"], 3)
                self.assertEqual(event["active_count"], 3)
                self.assertEqual(event["line"], summary["runner_evidence_boundary"]["start_line"])
                self.assertEqual(
                    raw_ref["prefix_sha256"], summary["runner_evidence_boundary"]["prefix_sha256"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_evidence_boundary_rejects_missing_truncated_changed_and_invalid_prefix(self):
        records = [{"request_id": "request", "response_ids": ["cmpl-request-0"]}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.jsonl"
            missing_boundary = qualifier.capture_evidence_boundary(missing)
            missing_result = qualifier.assess_evidence(
                missing, records, batch_capacity=3, evidence_boundary=missing_boundary)
            self.assertEqual(missing_result["status"], "runner_evidence_not_correlated")
            self.assertFalse(missing_result["raw_refs"][0]["boundary_valid"])

            trace = write_trace(root / "trace", [actual_trace_row(["request"], [1], 1)])
            boundary = qualifier.capture_evidence_boundary(trace)
            trace.write_bytes(trace.read_bytes()[:boundary["start_byte"] - 1])
            truncated = qualifier.assess_evidence(
                trace, records, batch_capacity=3, evidence_boundary=boundary)
            self.assertEqual(truncated["status"], "runner_evidence_not_correlated")
            self.assertFalse(truncated["raw_refs"][0]["boundary_valid"])
            self.assertTrue(any("truncated" in error for error in truncated["raw_refs"][0]["errors"]))

            trace = write_trace(root / "trace", [actual_trace_row(["request"], [1], 1)])
            boundary = qualifier.capture_evidence_boundary(trace)
            trace.write_bytes(b"x" * boundary["start_byte"])
            changed = qualifier.assess_evidence(
                trace, records, batch_capacity=3, evidence_boundary=boundary)
            self.assertEqual(changed["status"], "runner_evidence_not_correlated")
            self.assertFalse(changed["raw_refs"][0]["boundary_valid"])
            self.assertTrue(any("prefix changed" in error for error in changed["raw_refs"][0]["errors"]))

            invalid = dict(boundary, start_line=0)
            invalid_result = qualifier.assess_evidence(
                trace, records, batch_capacity=3, evidence_boundary=invalid)
            self.assertEqual(invalid_result["status"], "runner_evidence_not_correlated")
            self.assertFalse(invalid_result["raw_refs"][0]["boundary_valid"])

    def test_runtime_smoke_faults_fail_independent_liveness_checks(self):
        cases = (
            (MissingIdService, "all_response_ids_present"),
            (TruncatedService, "all_sse_complete"),
            (MetricsDownService, "metrics_readable"),
        )
        for service, failed_check in cases:
            with self.subTest(service=service.__name__), tempfile.TemporaryDirectory() as temporary:
                server, thread = start_mock(service)
                try:
                    output = Path(temporary) / "result"
                    summary = qualifier.execute(runtime_config(
                        f"http://127.0.0.1:{server.server_port}", output,
                        clients=1, batch_capacity=2, budgets=[4], prompt_lengths=[8],
                        timeout=1.0, metrics_interval=0.01, drain_timeout=0.3))
                    self.assertEqual(summary["status"], "runtime_smoke_failed")
                    self.assertFalse(summary["runtime_qualified"])
                    self.assertFalse(summary["runtime_checks"][failed_check])
                    if service is MetricsDownService:
                        self.assertFalse(summary["runtime_checks"]["drained_idle"])
                        self.assertEqual(summary["report_path"], str(output / "report.json"))
                        report = json.loads((output / "report.json").read_text())
                        self.assertEqual(report["drain"]["last_status"], 503)
                        self.assertEqual(report["drain"]["valid_snapshots"], 0)
                        self.assertTrue(report["drain"]["observations"])
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

    def test_wait_idle_requires_status_200_and_both_parsed_metrics(self):
        invalid_snapshots = (
            {"status": 503, "num_requests_running": 0.0, "num_requests_waiting": 0.0},
            {"status": 200, "num_requests_running": 0.0, "num_requests_waiting": None},
            {"status": 200, "num_requests_running": None, "num_requests_waiting": 0.0},
        )
        for snapshot in invalid_snapshots:
            with self.subTest(snapshot=snapshot), tempfile.TemporaryDirectory() as temporary, \
                    mock.patch.object(qualifier, "save_metrics", return_value=snapshot):
                result = qualifier.wait_idle(
                    "http://unused", MODEL, 0.1, None, Path(temporary), 0.03, 0.005)
                self.assertFalse(result["idle"])
                self.assertEqual(result["valid_snapshots"], 0)
                self.assertTrue(result["observations"])
                self.assertTrue(all(not row["idle_sample"] for row in result["observations"]))
                self.assertEqual(result["observations"][0]["status"], snapshot["status"])

    def test_runtime_smoke_stops_metrics_poller_when_request_raises_before_run_try(self):
        server, thread = start_mock()
        stopped = threading.Event()

        def wait_for_stop(stop, *_args, **_kwargs):
            try:
                stop.wait(timeout=5.0)
            finally:
                stopped.set()

        try:
            with tempfile.TemporaryDirectory() as temporary:
                with mock.patch.object(
                        qualifier, "run_request", side_effect=OSError("pre-try failure")), \
                        mock.patch.object(qualifier, "poll_metrics", side_effect=wait_for_stop):
                    summary = qualifier.execute(runtime_config(
                        f"http://127.0.0.1:{server.server_port}", Path(temporary) / "result",
                        clients=1, batch_capacity=2, budgets=[4], prompt_lengths=[8]))
                self.assertTrue(stopped.wait(timeout=0.5))
                self.assertEqual(summary["status"], "runtime_smoke_failed")
                self.assertFalse(summary["runtime_qualified"])
                self.assertIn("OSError", summary["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_runtime_response_ids_require_nonempty_strings(self):
        base = {
            "request_id": "request", "status": 200, "token_ids": [7], "max_tokens": 1,
            "done": True, "finish_reason": "length", "usage": {"completion_tokens": 1}, "error": None,
        }
        cases = (
            (["cmpl-request-0"], True),
            ([], False),
            ([""], False),
            (["   "], False),
            ([123], False),
            (["cmpl-request-0", ""], False),
            ("cmpl-request-0", False),
        )
        for response_ids, expected in cases:
            with self.subTest(response_ids=response_ids):
                result = qualifier._runtime_request_result({**base, "response_ids": response_ids})
                self.assertEqual(result["response_ids_valid"], expected)
                self.assertEqual(result["response_id_present"], expected)
                self.assertEqual(result["passed"], expected)

    def test_runtime_queue_evidence_does_not_claim_physical_multi_batch(self):
        records = [{"request_id": f"request-{index}"} for index in range(3)]
        evidence = {"matched_events": [{"batch_round": 1}, {"batch_round": 2}]}
        observed = qualifier.runtime_queue_evidence(
            records, 2, [{"num_requests_running": 2, "num_requests_waiting": 1}], evidence)
        self.assertTrue(observed["a_exceeds_n"])
        self.assertTrue(observed["queue_observed"])
        self.assertFalse(observed["queue_observation_required_for_runtime_pass"])
        self.assertEqual(observed["correlated_packed_rounds"], [1, 2])
        self.assertTrue(observed["multiple_packed_rounds_observed"])
        self.assertFalse(observed["physical_multi_batch_proven"])

        not_observed = qualifier.runtime_queue_evidence(
            records, 2, [{"num_requests_running": 2, "num_requests_waiting": 0}])
        self.assertFalse(not_observed["queue_observed"])

    def test_runtime_cancel_probe_runs_extra_lifecycle_without_serial(self):
        server, thread = start_mock()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "result"
                summary = qualifier.execute(runtime_config(
                    f"http://127.0.0.1:{server.server_port}", output,
                    clients=2, batch_capacity=2, budgets=[16, 24], prompt_lengths=[16],
                    cancel_probe=True, cancel_after=7, drain_timeout=0.5))
                self.assertEqual(summary["status"], "runtime_smoke_passed")
                self.assertTrue(summary["runtime_qualified"])
                self.assertTrue(summary["runtime_checks"]["all_sse_complete"])
                self.assertTrue(summary["runtime_checks"]["cancel_reuse_ok"])
                self.assertEqual(len(summary["requests"]), 2)
                cancellation = summary["cancellation"]
                self.assertTrue(cancellation["cancel_reuse_ok"])
                self.assertGreaterEqual(cancellation["cancelled_after_tokens"], 7)
                self.assertTrue(cancellation["survivor_live_at_cancel"])
                self.assertTrue(cancellation["reuse_live_at_submit"])
                self.assertTrue(cancellation["http_overlap"]["cancel_survivor"]["client_overlap"])
                self.assertTrue(cancellation["http_overlap"]["reuse_survivor"]["client_overlap"])
                self.assertIn("extra two-lane lifecycle", cancellation["scope"])
                self.assertFalse((output / "serial").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_runtime_cancel_probe_fails_on_truncated_or_empty_id_survivor(self):
        cases = (
            (SurvivorTruncatedService, "survivor_complete", "survivor_response_id_valid"),
            (SurvivorMissingIdService, "survivor_response_id_valid", "survivor_complete"),
        )
        for service, failed_field, passing_field in cases:
            with self.subTest(service=service.__name__), tempfile.TemporaryDirectory() as temporary:
                server, thread = start_mock(service)
                try:
                    summary = qualifier.execute(runtime_config(
                        f"http://127.0.0.1:{server.server_port}", Path(temporary) / "result",
                        clients=2, batch_capacity=2, budgets=[16, 24], prompt_lengths=[16],
                        cancel_probe=True, cancel_after=7, drain_timeout=0.5))
                    self.assertEqual(summary["status"], "runtime_smoke_failed")
                    self.assertFalse(summary["runtime_qualified"])
                    self.assertTrue(summary["runtime_checks"]["all_sse_complete"])
                    self.assertFalse(summary["runtime_checks"]["cancel_reuse_ok"])
                    self.assertFalse(summary["cancellation"][failed_field])
                    self.assertTrue(summary["cancellation"][passing_field])
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

    def test_runtime_cancel_trace_requires_correlated_lifecycle(self):
        server, thread = start_mock()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                requests = qualifier.build_requests(1, 2, [16, 24], [16])
                cancelled, survivor, reuse = qualifier.cancellation_requests(requests)
                trace = write_trace(root / "evidence", [])
                rows = [
                    actual_trace_row([cancelled["id"], survivor["id"]], [10, 7], 11),
                    actual_trace_row([reuse["id"], survivor["id"]], [11, 7], 12),
                ]
                with run_request_appending_trace(trace, rows):
                    summary = qualifier.execute(runtime_config(
                        f"http://127.0.0.1:{server.server_port}", root / "result",
                        clients=2, batch_capacity=2, budgets=[16, 24], prompt_lengths=[16],
                        runner_evidence=trace, cancel_probe=True, cancel_after=7, drain_timeout=0.5))
                self.assertEqual(summary["status"], "runtime_smoke_evidence_correlated")
                self.assertTrue(summary["runtime_checks"]["runner_evidence_lifecycle_ok"])
                self.assertEqual(summary["cancellation"]["trace_status"], "correlated")
                self.assertEqual(
                    summary["runner_evidence"]["lifecycle"]["status"], "runner_lifecycle_correlated")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_runtime_cancel_trace_marks_missing_slot_generation_inconclusive(self):
        lifecycle = {"cancelled": "cancel", "survivor": "survivor", "reuse": "reuse"}
        evidence = {
            "status": "runner_evidence_not_correlated",
            "matched_events": [
                {"request_ids": ["cancel", "survivor"]},
                {"request_ids": ["reuse", "survivor"]},
            ],
            "lifecycle": {
                "status": "runner_lifecycle_not_correlated",
                "slot_generation_advanced": None,
                "survivor_slot_generation_consistent": None,
            },
        }
        self.assertEqual(
            qualifier.runtime_cancel_trace_status(evidence, lifecycle), "inconclusive")

    def test_runtime_cancel_probe_keeps_budget_headroom_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "budget headroom"):
                qualifier.execute(runtime_config(
                    "http://127.0.0.1:1", output,
                    clients=2, batch_capacity=2, budgets=[8, 16], prompt_lengths=[16],
                    cancel_probe=True, cancel_after=7))
            self.assertFalse(output.exists())


class MainStatusTests(unittest.TestCase):
    def test_exit_code_is_zero_only_for_successful_qualified_statuses(self):
        cases = (
            ("runner_evidence_correlated", True, None, 0),
            ("http_only_runner_evidence_not_provided", True, None, 0),
            ("failed", True, {"passed": False}, 1),
            ("http_only_runner_evidence_not_correlated", True, None, 1),
            ("runner_evidence_correlated", False, None, 1),
            ("unexpected_status", True, None, 1),
        )
        for status, http_qualified, cancellation, expected in cases:
            with self.subTest(status=status, http_qualified=http_qualified):
                summary = {
                    "status": status,
                    "http_qualified": http_qualified,
                    "cancellation": cancellation,
                }
                with mock.patch.object(qualifier, "execute", return_value=summary), \
                        mock.patch("builtins.print"):
                    self.assertEqual(qualifier.main(["--output", "/tmp/unused"]), expected)

    def test_runtime_mode_exit_code_does_not_imply_formal_qualification(self):
        summary = {
            "status": "runtime_smoke_passed", "runtime_qualified": True,
            "runtime_status": "passed", "http_qualified": False,
            "formal_qualification": False, "semantic_exact_checked": False,
            "runtime_success_is_formal_qualification": False, "all_responses_match": None,
        }
        with mock.patch.object(qualifier, "execute", return_value=summary) as execute, \
                mock.patch("builtins.print") as printed:
            self.assertEqual(
                qualifier.main(["--mode", "runtime-smoke", "--output", "/tmp/unused"]), 0)
        self.assertEqual(execute.call_args.args[0]["mode"], "runtime-smoke")
        output = json.loads(printed.call_args.args[0])
        self.assertEqual(output["status"], "runtime_smoke_passed")
        self.assertTrue(output["runtime_qualified"])
        self.assertFalse(output["formal_qualification"])
        self.assertFalse(output["runtime_success_is_formal_qualification"])
        self.assertFalse(output["http_qualified"])

        summary.update({"status": "runtime_smoke_failed", "runtime_qualified": False})
        with mock.patch.object(qualifier, "execute", return_value=summary), mock.patch("builtins.print"):
            self.assertEqual(
                qualifier.main(["--mode", "runtime-smoke", "--output", "/tmp/unused"]), 1)

    def test_parser_defaults_to_qualification_mode(self):
        self.assertEqual(
            qualifier.parser().parse_args(["--output", "/tmp/unused"]).mode, "qualification")
        self.assertEqual(
            qualifier.parser().parse_args(
                ["--mode", "runtime-smoke", "--output", "/tmp/unused"]).mode, "runtime-smoke")


if __name__ == "__main__":
    unittest.main()
