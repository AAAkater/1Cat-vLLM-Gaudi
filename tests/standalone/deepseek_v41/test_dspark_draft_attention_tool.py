# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the real-weight DSpark draft-attention comparison tool."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tools.check_deepseek_v41_dspark_draft_attention as draft_tool
from tools.check_deepseek_v41_dspark_draft_attention import (
    _GraphCallCounter,
    _graph_usage,
    SCENARIOS,
    SOURCE_FILES,
    _write_report,
    audit_attention_rows,
    audit_scenario_outputs,
    compare_exact,
    make_request_metadata,
    parse_args,
    scenarios_for_capacity,
    source_fingerprints,
    validate_compatible_flags,
)
from vllm_gaudi.ops.deepseek_v41_batch_draft import draft_attention_rows


def _compatible_flags():
    return {
        "dspark": True,
        "native_rope": False,
        "draft_kv_decode": True,
        "draft_packed_mla": False,
        "draft_query_fp8": False,
        "draft_dense_fp8": False,
        "wo_handoff": False,
        "rope_coherent": False,
        "pretranspose_attn": False,
    }


def test_row_audit_uses_real_builder_for_slots_window_and_tentative_scratch():
    positions = torch.tensor([[300, 301, 302, 303, 304], [900, 901, 902, 903, 904]], dtype=torch.int32)
    slots = torch.tensor([1, 0], dtype=torch.int32)
    active = torch.tensor([True, True])
    rows = draft_attention_rows(positions, slots, active, ring_rows=512)
    audit = audit_attention_rows(rows, positions, slots, active, ring_rows=512)
    assert audit["exact"]
    assert audit["unique_context_complete"]
    assert audit["unique_active_context_rows"] == 2 * 128
    assert audit["active_tentative_rows_outside_persistent_ring"]
    assert audit["inactive_tentative_rows_masked"]
    assert audit["context_rows_per_active_request"] == 128
    assert audit["tentative_rows_per_active_request"] == 5
    assert audit["tentative_scratch_row_ids"] == [[512, 513, 514, 515, 516], [517, 518, 519, 520, 521]]


def test_row_audit_records_inactive_padding_without_using_it_as_oracle():
    positions = torch.tensor([[0, 0, 0, 0, 0], [900, 901, 902, 903, 904]], dtype=torch.int32)
    slots = torch.tensor([0, 1], dtype=torch.int32)
    active = torch.tensor([False, True])
    rows = draft_attention_rows(positions, slots, active, ring_rows=512)
    audit = audit_attention_rows(rows, positions, slots, active, ring_rows=512)
    assert audit["exact"]
    assert audit["context_rows_per_active_request"] == 128
    assert audit["unique_active_context_rows"] == 128
    assert audit["active_tentative_rows_outside_persistent_ring"]
    assert audit["inactive_tentative_rows_masked"]
    assert audit["tentative_scratch_row_ids"][0] == [-1, -1, -1, -1, -1]


def test_exact_comparison_rejects_one_ulp_and_reports_no_tolerance():
    reference = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    actual = reference.clone()
    bits = actual.view(torch.int16)
    bits[1] += 1
    check = compare_exact("output", reference, actual)
    assert not check["exact"]
    assert not check["bitwise_equal"]
    assert check["mismatch"] == 1
    assert check["rtol"] == 0 and check["atol"] == 0
    assert check["criterion"] == "bitwise_equal_after_host_copy"
    assert check["reference_sha256"] != check["actual_sha256"]


def test_exact_comparison_passes_unchanged_bits():
    reference = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
    check = compare_exact("ring", reference, reference.clone())
    assert check["exact"]
    assert check["numeric_equal"]
    assert check["bitwise_equal"]
    assert check["mismatch"] == 0
    assert check["max_abs"] == 0.0
    assert check["reference_sha256"] == check["actual_sha256"]


def test_flag_validation_blocks_paths_without_one_serial_packed_mapping():
    flags = _compatible_flags()
    assert validate_compatible_flags(flags) == []
    for name in (
        "draft_packed_mla",
        "draft_query_fp8",
        "draft_dense_fp8",
        "wo_handoff",
        "pretranspose_attn",
    ):
        changed = dict(flags)
        changed[name] = True
        assert [item["flag"] for item in validate_compatible_flags(changed)] == [name]
    coherent_without_native_rope = dict(flags)
    coherent_without_native_rope["rope_coherent"] = True
    assert validate_compatible_flags(coherent_without_native_rope) == []
    coherent_with_native_rope = dict(coherent_without_native_rope)
    coherent_with_native_rope["native_rope"] = True
    assert [
        item["flag"] for item in validate_compatible_flags(coherent_with_native_rope)
    ] == ["rope_coherent"]
    changed = dict(flags)
    changed["draft_kv_decode"] = False
    assert [item["flag"] for item in validate_compatible_flags(changed)] == ["draft_kv_decode"]


def test_source_fingerprints_cover_tool_and_production_paths():
    records = source_fingerprints(Path(__file__).resolve().parents[3])
    assert set(records) == set(SOURCE_FILES)
    assert all(len(record["sha256"]) == 64 for record in records.values())
    assert records["tools/check_deepseek_v41_dspark_draft_attention.py"]["bytes"] > 0


def test_cli_deduplicates_layers_and_requires_rank_zero(tmp_path):
    args = parse_args([
        str(tmp_path / "prepared"),
        "--output",
        str(tmp_path / "evidence.json"),
        "--layers",
        "2",
        "0",
        "2",
    ])
    assert args.layers == (2, 0)
    assert args.rank == 0
    assert args.tp_size == 8
    assert args.batch_capacity == 2
    explicit = parse_args([
        str(tmp_path / "prepared"),
        "--output",
        str(tmp_path / "capacity4.json"),
        "--batch-capacity",
        "4",
    ])
    assert explicit.batch_capacity == 4
    with pytest.raises(SystemExit):
        parse_args([
            str(tmp_path / "prepared"),
            "--output",
            str(tmp_path / "unsupported.json"),
            "--batch-capacity",
            "3",
        ])


def test_report_writer_accepts_string_output_path(tmp_path):
    output = tmp_path / "evidence.json"
    _write_report(str(output), {"status": "passed"})
    assert output.read_text() == '{\n  "status": "passed"\n}\n'


def test_row_audit_rejects_wrong_slot_mapping():
    positions = torch.tensor([[300, 301, 302, 303, 304], [900, 901, 902, 903, 904]], dtype=torch.int32)
    active = torch.tensor([True, True])
    rows = draft_attention_rows(positions, torch.tensor([1, 0]), active, ring_rows=512)
    audit = audit_attention_rows(rows, positions, torch.tensor([0, 1]), active, ring_rows=512)
    assert not audit["exact"]
    assert not audit["checks"]["context_rows"]["exact"]


def test_capacity_scenarios_preserve_b2_and_define_fixed_lane_n4_matrix():
    assert scenarios_for_capacity(2) == SCENARIOS
    assert [sum(item["active"]) for item in scenarios_for_capacity(1)] == [1]

    n4 = scenarios_for_capacity(4)
    assert [sum(item["active"]) for item in n4] == [4, 2, 1]
    assert n4[0]["active"] == (True, True, True, True)
    assert n4[1]["active"] == (True, False, True, False)
    assert n4[2]["active"] == (False, True, False, False)
    assert n4[0]["slots"] == (0, 1, 2, 3)
    assert n4[1]["slots"] == (0, -1, 2, -1)
    assert n4[2]["slots"] == (-1, 1, -1, -1)


def test_n4_row_audit_keeps_lane_slots_and_masks_inactive_tentative_rows():
    positions = torch.tensor([
        [300, 301, 302, 303, 304],
        [500, 501, 502, 503, 504],
        [700, 701, 702, 703, 704],
        [900, 901, 902, 903, 904],
    ], dtype=torch.int32)
    slots = torch.tensor([0, -1, 2, -1], dtype=torch.int32)
    active = torch.tensor([True, False, True, False])
    rows = draft_attention_rows(positions, slots, active, ring_rows=4 * 256)
    audit = audit_attention_rows(rows, positions, slots, active, ring_rows=4 * 256)

    assert audit["exact"]
    assert audit["active_requests"] == 2
    assert audit["unique_active_context_rows"] == 2 * 128
    assert audit["tentative_scratch_row_ids"][0] == [1024, 1025, 1026, 1027, 1028]
    assert audit["tentative_scratch_row_ids"][1] == [-1, -1, -1, -1, -1]
    assert audit["tentative_scratch_row_ids"][2] == [1034, 1035, 1036, 1037, 1038]
    assert audit["tentative_scratch_row_ids"][3] == [-1, -1, -1, -1, -1]


def test_request_metadata_uses_explicit_n4_active_and_inactive_sentinels():
    positions = torch.tensor([
        [300, 301, 302, 303, 304],
        [500, 501, 502, 503, 504],
        [700, 701, 702, 703, 704],
        [900, 901, 902, 903, 904],
    ], dtype=torch.int32)
    metadata = make_request_metadata(4, torch.tensor([True, False, True, False]), positions)

    assert metadata["request_active"] == [True, False, True, False]
    assert metadata["request_ids"] == [
        "draft-attention-lane-0",
        None,
        "draft-attention-lane-2",
        None,
    ]
    assert metadata["request_slots"] == [0, -1, 2, -1]
    assert metadata["slot_generations"] == [1, 0, 3, 0]
    assert metadata["target_start_positions"] == [300, 0, 700, 0]
    assert metadata["remaining_output_budget"] == [64, 0, 64, 0]
    assert metadata["target_query_lengths"] == [6, 0, 6, 0]
    assert metadata["shapes"]["request_active"] == [4]


def test_output_audit_requires_active_bf16_exact_and_inactive_bitwise_zero():
    capacity = 4
    active = torch.tensor([True, False, True, False])
    serial_outputs = {
        0: torch.arange(10, dtype=torch.float32).reshape(5, 2).bfloat16(),
        2: torch.arange(20, 30, dtype=torch.float32).reshape(5, 2).bfloat16(),
    }
    packed = torch.zeros((capacity * 5, 2), dtype=torch.bfloat16)
    packed[0:5] = serial_outputs[0]
    packed[10:15] = serial_outputs[2]
    rings = (torch.zeros((capacity * 256, 3), dtype=torch.uint8), )
    audit = audit_scenario_outputs(
        capacity=capacity,
        active=active,
        serial_outputs=serial_outputs,
        packed_output=packed,
        ring_before=rings,
        ring_after_serial=rings,
        ring_after_packed=rings,
    )
    assert audit["exact"]
    assert len(audit["active_outputs"]) == 2
    assert len(audit["inactive_zero_outputs"]) == 2
    assert all(item["exact"] for item in audit["inactive_zero_outputs"])
    assert all(item["active"] is False for item in audit["persistent_state"]["checks"][0]["slots"][1::2])

    nonzero_inactive = packed.clone()
    nonzero_inactive[7, 0] = torch.tensor(1.0, dtype=torch.bfloat16)
    broken = audit_scenario_outputs(
        capacity=capacity,
        active=active,
        serial_outputs=serial_outputs,
        packed_output=nonzero_inactive,
        ring_before=rings,
        ring_after_serial=rings,
        ring_after_packed=rings,
    )
    assert not broken["exact"]
    assert not broken["inactive_zero_outputs"][0]["exact"]


def test_graph_counter_uses_backend_callbacks_to_detect_recompiles():
    torch._dynamo.reset()
    backend_calls = []

    class AddOne(torch.nn.Module):
        def forward(self, value):
            return value + 1

    def mock_backend(graph, inputs):
        backend_calls.append(tuple(item.shape for item in inputs))
        return graph.forward

    counter = _GraphCallCounter(AddOne(), backend=mock_backend)
    initial = counter.snapshot()
    counter(torch.ones(2))
    counter(torch.ones(2) * 2)
    same_shape = _graph_usage([counter], before=[initial])
    assert same_shape["entry_calls"] == 2
    assert same_shape["backend_compiles"] == 1
    assert same_shape["entry_replays"] == 1

    counter(torch.ones(3))
    recompiled = _graph_usage([counter], before=[initial])
    assert recompiled["entry_calls"] == 3
    assert recompiled["backend_compiles"] == 2
    assert recompiled["entry_replays"] == 1
    assert backend_calls == [(torch.Size([2]), ), (torch.Size([3]), )]


def test_n4_qualify_layer_mock_records_device_slots_shapes_and_backend_usage(tmp_path, monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_batch_draft as draft_batch

    class SerialAttention:
        def __init__(self):
            self.swa = None

        def draft(self, value, positions):
            del positions
            return value.clone()

    attention_fields = {
        "draft_kv_decode": True,
        "draft_packed_mla": False,
        "draft_query_fp8": False,
        "draft_dense_fp8": False,
        "native_rope": False,
        "woa_fp8": False,
        "woa_output_roundtrip": False,
        "length": 2048,
    }

    class FakeDraft:
        def __init__(self):
            self.layers = [
                SimpleNamespace(attention=SimpleNamespace(**attention_fields)) for _ in range(3)
            ]

    class FakeBatchState:
        def __init__(self):
            self.rings = [torch.zeros((4 * 256, 528), dtype=torch.uint8) for _ in range(3)]

        def ring(self, layer):
            return self.rings[layer]

        def state_tensors(self):
            return tuple(self.rings)

    class FakeWeights:
        def __init__(self):
            self.mtp = SimpleNamespace(get_submodule=lambda name: SimpleNamespace(attn=object()))

    packed_slots = []

    def packed_attention(draft, layer_index, value, positions, slots, active, batch_state):
        del draft, layer_index, positions, batch_state
        packed_slots.append(slots.tolist())
        output = torch.zeros_like(value)
        for lane in range(active.numel()):
            if bool(active[lane].item()):
                start = lane * 5
                output[start:start + 5] = value[start:start + 5]
        return output

    def mock_backend(graph, inputs):
        del inputs
        return graph.forward

    def mock_compile(module, backend):
        compiled = False

        def call(*args, **kwargs):
            nonlocal compiled
            if not compiled:
                backend(module, args)
                compiled = True
            return module(*args, **kwargs)

        return call

    monkeypatch.setattr(draft_tool, "_build_attention", lambda *args: SerialAttention())
    monkeypatch.setattr(draft_tool, "_lookup_backend", lambda name: mock_backend)
    monkeypatch.setattr(draft_tool, "_compile", mock_compile)
    monkeypatch.setattr(draft_batch, "draft_attention_batch", packed_attention)
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    report = {"output_path": str(tmp_path / "n4-mock.json"), "status": "running"}
    result = draft_tool._qualify_layer(
        report,
        layer=0,
        weights=FakeWeights(),
        config=object(),
        shared=object(),
        draft=FakeDraft(),
        batch_state=FakeBatchState(),
        linear=object(),
        device="cpu",
        seed=7,
        capacity=4,
    )

    assert result["exact"]
    assert result["batch_capacity"] == 4
    assert [item["active_requests"] for item in result["scenarios"]] == [4, 2, 1]
    contracts = [item["lane_slot_contract"] for item in result["scenarios"]]
    assert all(item["capacity_lane_contract_exact"] for item in contracts)
    assert all(item["active_device_lane_equals_slot"] for item in contracts)
    assert all(item["inactive_device_slot_is_minus_one"] for item in contracts)
    assert all(item["persistent_state_slot_equals_host_lane"] for item in contracts)
    assert all(item["request_metadata"]["request_slots"] == item["slots"] for item in result["scenarios"])
    assert all(item["lane_slot_contract"]["persistent_state_slots"] == [0, 1, 2, 3]
               for item in result["scenarios"])
    assert packed_slots == [
        [0, 1, 2, 3],
        [0, -1, 2, -1],
        [0, -1, 2, -1],
        [-1, 1, -1, -1],
        [-1, 1, -1, -1],
    ]
    assert [item["exact"] for item in result["scenarios"]] == [True, True, True]
    shapes = [item["shapes"] for item in result["scenarios"]]
    assert all(item["packed_value"] == [20, 5120] for item in shapes)
    assert all(item["positions"] == [4, 5] for item in shapes)
    assert all(item["persistent_ring"] == [1024, 528] for item in shapes)
    assert [item["active_draft_rows"] for item in shapes] == [20, 10, 5]
    assert [item["active_context_rows"] for item in shapes] == [512, 256, 128]
    assert result["graph_runtime"]["serial"]["entry_calls"] == 7
    assert result["graph_runtime"]["serial"]["backend_compiles"] == 4
    assert result["graph_runtime"]["serial"]["entry_replays"] == 3
    assert result["graph_runtime"]["packed"]["entry_calls"] == 5
    assert result["graph_runtime"]["packed"]["backend_compiles"] == 1
    # Full A4 uses one packed invocation; each partial scenario invokes the
    # same graph once for the main comparison and once for ring invariance.
    assert result["graph_runtime"]["packed"]["entry_replays"] == 4


def test_graph_counters_share_backend_identity_and_attribute_only_local_compiles():
    torch._dynamo.reset()
    calls = []

    class AddOne(torch.nn.Module):
        def forward(self, value):
            return value + 1

    def backend(graph, inputs):
        calls.append(tuple(item.shape for item in inputs))
        return graph.forward

    shared = draft_tool._CountingBackend(backend)
    first = _GraphCallCounter(AddOne(), backend=shared)
    second = _GraphCallCounter(AddOne(), backend=shared)
    assert first.backend is second.backend
    first(torch.ones(2))
    second(torch.ones(2))
    assert len(calls) == 1
    assert first.snapshot()["backend_compiles"] == 1
    assert second.snapshot()["backend_compiles"] == 0
    second(torch.ones(3))
    assert len(calls) == 2
    assert first.snapshot()["backend_compiles"] == 1
    assert second.snapshot()["backend_compiles"] == 1
