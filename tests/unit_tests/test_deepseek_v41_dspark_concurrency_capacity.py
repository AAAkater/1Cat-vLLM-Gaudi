# SPDX-License-Identifier: Apache-2.0
"""CPU-only capacity contracts for the DSpark launcher and HTTP qualifier."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO.parent / "scripts"
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

import qualify_deepseek_v41_dspark_concurrency as qualifier  # noqa: E402

MODEL = "mock-dspark-model"


def records_for(requests):
    return [{"request_id": request["id"], "response_ids": [f"cmpl-{request['id']}-0"]} for request in requests]


def trace_row(capacity, active_lanes, req_ids=None, *, fixed_arrays=True, events=None):
    if req_ids is None:
        req_ids = [f"request-{lane}" for lane in active_lanes]
    slots = [-1] * capacity
    generations = [0] * capacity
    commits = [0] * capacity
    queries = [0] * capacity
    outputs = [0] * capacity
    for offset, lane in enumerate(active_lanes):
        slots[lane] = lane
        generations[lane] = 3 + lane
        commits[lane] = 1
        queries[lane] = 1
        outputs[lane] = 1 + offset
    if not fixed_arrays:
        slots = [slots[lane] for lane in active_lanes]
        generations = [generations[lane] for lane in active_lanes]
        commits = [commits[lane] for lane in active_lanes]
        queries = [queries[lane] for lane in active_lanes]
        outputs = [outputs[lane] for lane in active_lanes]
    row = {
        "schema": "dsv41.dspark.round.v2",
        "batch_round": 9,
        "batch_capacity": capacity,
        "active_lanes": active_lanes,
        "req_ids": req_ids,
        "request_slots": slots,
        "slot_generations": generations,
        "logical_target_queries": queries,
        "actual_packed_target_rows": capacity * 6,
        "actual_packed_draft_rows": capacity * 5,
        "committed_counts": commits,
        "output_counts": outputs,
        "completion_event_observed": True,
        "final_event_observed": True,
    }
    if events is not None:
        for name in ("completion_event_observed", "final_event_observed"):
            row.pop(name)
        row["events"] = events
    return row


def write_trace(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "runner-trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class CapacityQualifierTests(unittest.TestCase):
    def test_batch_capacity_default_range_and_implementation_three(self):
        args = qualifier.parser().parse_args(["--output", "/tmp/unused"])
        self.assertEqual(args.batch_capacity, 2)
        for value in (2, 3, 4):
            self.assertEqual(qualifier.validate_batch_capacity(value), value)
        for value in (1, 5):
            with self.assertRaisesRegex(ValueError, "2..4"):
                qualifier.validate_batch_capacity(value)

    def test_a1_partial_and_n4_four_client_budgets_are_distinct(self):
        one = qualifier.build_requests(2, 1, None)
        four = qualifier.build_requests(2, 4, None)
        self.assertEqual(len(one), 1)
        self.assertEqual([row["max_tokens"] for row in four], [8, 16, 24, 32])
        self.assertEqual(len({row["id"] for row in four}), 4)
        self.assertEqual(len({row["max_tokens"] for row in four}), 4)
        with self.assertRaisesRegex(ValueError, "1..12"):
            qualifier.build_requests(2, 0, None)
        with self.assertRaisesRegex(ValueError, "distinct"):
            qualifier.build_requests(2, 2, [8, 8])

    def test_v2_n4_full_capacity_correlates_fixed_n_fields(self):
        requests = qualifier.build_requests(2, 4, None)
        records = records_for(requests)
        reqids = [row["id"] for row in requests]
        row = trace_row(4, [0, 1, 2, 3], reqids)
        with tempfile.TemporaryDirectory() as temporary:
            result = qualifier.assess_evidence(write_trace(Path(temporary), [row]), records, batch_capacity=4)
        self.assertEqual(result["status"], "runner_evidence_correlated")
        self.assertEqual(result["configured_batch_capacity"], 4)
        self.assertEqual(result["observed_capacities"], [4])
        self.assertEqual(result["max_active_count"], 4)
        self.assertTrue(result["capacity_fully_active_observed"])
        event = result["matched_events"][0]
        self.assertEqual(event["active_lanes"], [0, 1, 2, 3])
        self.assertEqual(event["target_rows"], 24)
        self.assertEqual(event["draft_rows"], 20)
        self.assertEqual(event["slots"], [0, 1, 2, 3])
        self.assertEqual(event["gens"], [3, 4, 5, 6])
        self.assertEqual(event["reqids"], reqids)
        self.assertIn("not full capacity", result["conclusion"])
        self.assertFalse(result["all_rank_digest_checked"])
        self.assertIn("single trace file", result["capacity_scope"])

    def test_v2_n4_a1_partial_uses_fixed_rows_but_one_active_lane(self):
        request = qualifier.build_requests(2, 1, None)[0]
        reqids = [None, None, request["id"], None]
        row = trace_row(4, [2], reqids)
        with tempfile.TemporaryDirectory() as temporary:
            result = qualifier.assess_evidence(write_trace(Path(temporary), [row]), records_for([request]),
                                               batch_capacity=4)
        self.assertEqual(result["status"], "runner_evidence_correlated")
        self.assertEqual(result["max_active_count"], 1)
        self.assertFalse(result["capacity_fully_active_observed"])
        event = result["matched_events"][0]
        self.assertEqual(event["active_lanes"], [2])
        self.assertEqual(event["reqids"], [request["id"]])
        self.assertEqual(event["target_rows"], 24)
        self.assertEqual(event["draft_rows"], 20)
        self.assertEqual(event["request_evidence"][request["id"]]["slot"], 2)

    def test_two_active_lanes_at_n4_do_not_prove_full_capacity(self):
        requests = qualifier.build_requests(2, 2, None)
        row = trace_row(4, [0, 1], [request["id"] for request in requests])
        with tempfile.TemporaryDirectory() as temporary:
            result = qualifier.assess_evidence(write_trace(Path(temporary), [row]), records_for(requests),
                                               batch_capacity=4)
        self.assertEqual(result["status"], "runner_evidence_correlated")
        self.assertEqual(result["max_active_count"], 2)
        self.assertFalse(result["capacity_fully_active_observed"])
        self.assertFalse(result["matched_events"][0]["capacity_fully_active"])

    def test_actual_v2_active_only_fields_and_fixed_prefix_correlate(self):
        requests = qualifier.build_requests(2, 2, None)
        reqids = [request["id"] for request in requests]
        row = trace_row(4, [1, 3], reqids, fixed_arrays=False)
        row.update({
            "active_mask": [False, True, False, True],
            "lane_request_ids": [None, reqids[0], None, reqids[1]],
            "lane_slot_generations": [0, 4, 0, 6],
            "lane_committed_counts": [0, 1, 0, 1],
            "lane_output_counts": [0, 1, 0, 2],
            "next_draft_counts": [2, 0],
            "lane_next_draft_counts": [0, 2, 0, 0],
        })
        with tempfile.TemporaryDirectory() as temporary:
            result = qualifier.assess_evidence(write_trace(Path(temporary), [row]), records_for(requests),
                                               batch_capacity=4)
        self.assertEqual(result["status"], "runner_evidence_correlated")
        event = result["matched_events"][0]
        self.assertEqual(event["active_lanes"], [1, 3])
        self.assertEqual(event["slots"], [1, 3])
        self.assertEqual(event["gens"], [4, 6])
        self.assertEqual(event["output_counts"], [1, 2])
        self.assertEqual(event["next_draft_counts"], [2, 0])

    def test_v2_rejects_noncanonical_slots_ranges_and_unknown_schema(self):
        request = qualifier.build_requests(2, 1, None)[0]
        base = trace_row(2, [0], [request["id"]])
        rows = []

        def malformed(change):
            row = json.loads(json.dumps(base))
            change(row)
            rows.append(row)

        malformed(lambda row: row["request_slots"].__setitem__(0, 1))
        malformed(lambda row: row["slot_generations"].__setitem__(0, 0))
        malformed(lambda row: row["logical_target_queries"].__setitem__(0, 0))
        malformed(lambda row: row["logical_target_queries"].__setitem__(0, 7))
        malformed(lambda row: row["committed_counts"].__setitem__(0, 0))
        malformed(lambda row: row["committed_counts"].__setitem__(0, 2))
        malformed(lambda row: row.__setitem__("batch_round", 0))
        malformed(lambda row: row.__setitem__("schema", "dsv41.dspark.round.unknown"))
        malformed(lambda row: row.__setitem__("active_mask", [False, True]))
        with tempfile.TemporaryDirectory() as temporary:
            result = qualifier.assess_evidence(write_trace(Path(temporary), rows), records_for([request]),
                                               batch_capacity=2)
        self.assertEqual(result["status"], "runner_evidence_not_correlated")
        reasons = [row["reason"] for row in result["raw_refs"][0]["rejected_rows"]]
        self.assertTrue(any("request_slots must equal active_lanes" in reason for reason in reasons))
        self.assertTrue(any("slot_generations must be positive" in reason for reason in reasons))
        self.assertTrue(any("logical_target_queries must be integers in 1..6" in reason for reason in reasons))
        self.assertTrue(any("committed_counts must be integers in 1..logical_target_queries" in reason
                            for reason in reasons))
        self.assertTrue(any("batch_round must be a positive integer" in reason for reason in reasons))
        self.assertTrue(any("unknown trace schema" in reason for reason in reasons))
        self.assertTrue(any("active_lanes differs from active mask" in reason for reason in reasons))

    def test_v2_rejects_nonzero_inactive_main_and_fixed_prefix_values(self):
        request = qualifier.build_requests(2, 1, None)[0]
        rows = []

        for field, lane_value in (("request_slots", 0), ("slot_generations", 1),
                                  ("logical_target_queries", 1), ("committed_counts", 1), ("output_counts", 1)):
            row = trace_row(2, [0], [request["id"]])
            row[field][1] = lane_value
            rows.append(row)

        base = trace_row(2, [0], [request["id"]], fixed_arrays=False)
        base.update({
            "active_mask": [True, False],
            "lane_request_ids": [request["id"], None],
            "lane_slot_generations": [3, 0],
            "lane_committed_counts": [1, 0],
            "lane_output_counts": [1, 0],
            "lane_next_draft_counts": [0, 0],
        })
        for field, lane_value in (("lane_request_ids", "stale-id"), ("lane_slot_generations", 1),
                                  ("lane_committed_counts", 1), ("lane_output_counts", 1),
                                  ("lane_next_draft_counts", 1)):
            row = json.loads(json.dumps(base))
            row[field][1] = lane_value
            rows.append(row)

        with tempfile.TemporaryDirectory() as temporary:
            result = qualifier.assess_evidence(write_trace(Path(temporary), rows), records_for([request]),
                                               batch_capacity=2)
        self.assertEqual(result["status"], "runner_evidence_not_correlated")
        reasons = [row["reason"] for row in result["raw_refs"][0]["rejected_rows"]]
        for field in ("request_slots", "slot_generations", "logical_target_queries", "committed_counts",
                      "output_counts", "lane_slot_generations", "lane_committed_counts", "lane_output_counts",
                      "lane_next_draft_counts"):
            self.assertTrue(any(f"inactive {field}[1] must be" in reason for reason in reasons), field)
        self.assertTrue(any("lane_request_ids must match" in reason for reason in reasons))

    def test_inactive_id_and_wrong_n4_packed_width_are_rejected(self):
        request = qualifier.build_requests(2, 1, None)[0]
        records = records_for([request])
        bad_id = trace_row(4, [1], [None, request["id"], None, "stale-inactive-id"])
        bad_rows = trace_row(4, [1], [None, request["id"], None, None])
        bad_rows["actual_packed_target_rows"] = 12
        with tempfile.TemporaryDirectory() as temporary:
            result = qualifier.assess_evidence(write_trace(Path(temporary), [bad_id, bad_rows]), records,
                                               batch_capacity=4)
        self.assertEqual(result["status"], "runner_evidence_not_correlated")
        reasons = [row["reason"] for row in result["raw_refs"][0]["rejected_rows"]]
        self.assertTrue(any("inactive lane" in reason for reason in reasons))
        self.assertTrue(any("actual_packed_target_rows must be 24" in reason for reason in reasons))

    def test_legacy_b2_trace_and_adapter_events_mapping_remain_compatible(self):
        requests = qualifier.build_requests(2, 2, None)
        legacy = {
            "batch_round": 3,
            "reqids": [request["id"] for request in requests],
            "slots": [1, 0],
            "gens": [5, 6],
            "querycounts": [6, 1],
            "targetrows": 12,
            "draftrows": 10,
            "commitcounts": [2, 1],
            "eventobserved": True,
        }
        events = {"completion": {"observed": True}, "final": {"completed": True}}
        current = trace_row(2, [0, 1], [request["id"] for request in requests], fixed_arrays=False, events=events)
        with tempfile.TemporaryDirectory() as temporary:
            result = qualifier.assess_evidence(write_trace(Path(temporary), [legacy, current]),
                                               records_for(requests), batch_capacity=2)
        self.assertEqual(result["status"], "runner_evidence_correlated")
        self.assertEqual(len(result["matched_events"]), 2)
        self.assertEqual(result["matched_events"][0]["slots"], [1, 0])
        self.assertTrue(all(event["active_count"] == 2 for event in result["matched_events"]))
        self.assertTrue(result["matched_events"][1]["events"]["completion"])
        self.assertTrue(result["matched_events"][1]["events"]["final"])

    def test_cancellation_requires_two_clients_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            with self.assertRaisesRegex(ValueError, "at least two clients"):
                qualifier.execute({"clients": 1, "cases": 2, "cancel_probe": True, "batch_capacity": 4,
                                   "output": output})
            self.assertFalse(output.exists())


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "uv.log"
        fake_uv = self.bin / "uv"
        fake_uv.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "{\n"
            "  printf 'MAX_NUM_SEQS=%s\\n' \"$MAX_NUM_SEQS\"\n"
            "  printf 'NATIVE_DIR=%s\\n' \"$NATIVE_DIR\"\n"
            "  printf 'BRIDGE=%s\\n' \"$BRIDGE\"\n"
            "  printf 'RUN_DIR=%s\\n' \"$RUN_DIR\"\n"
            "  printf 'PWD=%s\\n' \"$PWD\"\n"
            "  printf 'ARG=%s\\n' \"$@\"\n"
            "} > \"$FAKE_UV_LOG\"\n")
        fake_uv.chmod(0o755)
        (self.root / "repo").mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def environment(self, run_name, **values):
        environment = os.environ.copy()
        environment.update({
            "PATH": str(self.bin) + os.pathsep + environment["PATH"],
            "FAKE_UV_LOG": str(self.log),
            "REPO": str(self.root / "repo"),
            "RUN_DIR": str(self.root / run_name),
            "PYTHON": "/data/project/.venv/bin/python",
        })
        environment.update(values)
        return environment

    def invoke(self, script, arguments=(), **values):
        return subprocess.run([str(SCRIPTS / script), *arguments], env=self.environment(values.pop(
            "run_name", "run"), **values), text=True, capture_output=True, check=False)

    def test_bash_syntax_and_fake_uv_argv_for_n2_and_n4(self):
        syntax = subprocess.run(["bash", "-n", str(SCRIPTS / "run_tp8_dspark.sh"),
                                 str(SCRIPTS / "run_tp8_dspark_batch.sh"),
                                 str(SCRIPTS / "run_tp8_dspark_b2.sh")], capture_output=True, text=True, check=False)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        for capacity in (2, 4):
            with self.subTest(capacity=capacity):
                self.log.unlink(missing_ok=True)
                result = self.invoke("run_tp8_dspark_batch.sh", run_name=f"run-n{capacity}", MAX_NUM_SEQS=str(capacity))
                self.assertEqual(result.returncode, 0, result.stderr)
                log = self.log.read_text()
                self.assertIn(f"MAX_NUM_SEQS={capacity}", log)
                self.assertIn(f"RUN_DIR={self.root / f'run-n{capacity}'}", log)
                self.assertIn("NATIVE_DIR=/data/project/artifacts/dsv41-runtime-dspark-b2/lib", log)
                argv_lines = [line.removeprefix("ARG=") for line in log.splitlines() if line.startswith("ARG=")]
                self.assertIn("--max-num-seqs", argv_lines)
                self.assertEqual(argv_lines[argv_lines.index("--max-num-seqs") + 1], str(capacity))
                self.assertEqual(argv_lines[argv_lines.index("--tensor-parallel-size") + 1], "8")
                self.assertNotIn("vllm_gaudi.entrypoints.deepseek_v41", result.stderr)
        self.log.unlink(missing_ok=True)
        result = self.invoke("run_tp8_dspark_b2.sh", run_name="run-b2", MAX_NUM_SEQS="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        log = self.log.read_text()
        self.assertIn("MAX_NUM_SEQS=2", log)
        self.assertIn("NATIVE_DIR=/data/project/artifacts/dsv41-runtime-dspark-b2/lib", log)
        argv_lines = [line.removeprefix("ARG=") for line in log.splitlines() if line.startswith("ARG=")]
        self.assertEqual(argv_lines[argv_lines.index("--max-num-seqs") + 1], "2")

    def test_invalid_capacity_and_b2_override_fail_before_uv_or_run_dir(self):
        for script, capacity in (("run_tp8_dspark_batch.sh", "1"),
                                 ("run_tp8_dspark_batch.sh", "5"),
                                 ("run_tp8_dspark_b2.sh", "4")):
            with self.subTest(script=script, capacity=capacity):
                self.log.unlink(missing_ok=True)
                run_dir = self.root / f"must-not-exist-{script}-{capacity}"
                result = self.invoke(script, run_name=run_dir.name, MAX_NUM_SEQS=capacity)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.log.exists())
                self.assertFalse(run_dir.exists())
                self.assertIn("MAX_NUM_SEQS", result.stderr)

    def test_cli_capacity_override_is_rejected_even_when_value_matches(self):
        for script, capacity, argument in (
                ("run_tp8_dspark_batch.sh", "2", "--max-num-seqs"),
                ("run_tp8_dspark_batch.sh", "4", "--max-num-seqs=4"),
                ("run_tp8_dspark_batch.sh", "4", "--max_num_seqs=3"),
                ("run_tp8_dspark_batch.sh", "4", "--max-num-se"),
                ("run_tp8_dspark_b2.sh", "2", "--max-num-seqs=4")):
            with self.subTest(script=script, argument=argument):
                self.log.unlink(missing_ok=True)
                run_dir = self.root / "override-must-not-exist"
                result = self.invoke(script, (argument,), run_name=run_dir.name, MAX_NUM_SEQS=capacity)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.log.exists())
                self.assertFalse(run_dir.exists())
                self.assertIn("set MAX_NUM_SEQS", result.stderr)


if __name__ == "__main__":
    unittest.main()
