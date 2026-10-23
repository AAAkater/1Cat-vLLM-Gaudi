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

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import qualify_deepseek_v41_dspark_concurrency as qualifier  # noqa: E402

MODEL = "mock-dspark-model"


def sse_lines(request):
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
        yield ("data: " + json.dumps({
            "id": response_id,
            "object": "text_completion",
            "model": MODEL,
            "choices": [choice],
        }) + "\n\n").encode()
    yield ("data: " + json.dumps({
        "id": response_id,
        "choices": [],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": max_tokens,
            "total_tokens": len(prompt) + max_tokens,
        },
    }) + "\n\n").encode()
    yield b"data: [DONE]\n\n"


class MockServer(ThreadingHTTPServer):
    daemon_threads = True


class MockService(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
            self.send_body(200, body, "text/plain")
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
            for line in sse_lines(request):
                self.wfile.write(line)
                self.wfile.flush()
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass

    @staticmethod
    def assert_payload_has_request_id(request):
        if not request.get("request_id"):
            raise AssertionError("payload omitted explicit request_id")
        if not request.get("ignore_eos") or not request.get("stream"):
            raise AssertionError("payload did not retain streaming and ignore_eos")


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


def write_trace(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "runner-trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


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


if __name__ == "__main__":
    unittest.main()
