# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the packed DSpark transaction qualification tool."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tools.check_deepseek_v41_dspark_packed_transaction as transaction_tool
from tools.check_deepseek_v41_dspark_packed_transaction import (
    BATCH_INACTIVE,
    DRAFT_TOKENS,
    HIDDEN_SIZE,
    SWA_ROWS,
    TARGET_CONTEXT_WIDTH,
    LanePlan,
    Scenario,
    _GraphCallCounter,
    _compare_result_fields,
    _compare_singleton_exact,
    _graph_usage,
    _inactive_ring_checks,
    _poison_inactive_slots,
    _readback_round,
    _replay_input_checks,
    _serial_lane_result,
    bind_all_lanes,
    compare_exact,
    make_round_inputs,
    scenarios_for_capacity,
    source_fingerprints,
    validate_transaction_flags,
)
from vllm_gaudi.ops.deepseek_v41_batch_verify import (
    BatchVerifyRing,
    pack_record_batch,
    verify_control_from_target_batch,
)


def _patch_ring_runtime(monkeypatch):
    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            pass

        def query(self):
            return True

    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(torch.hpu, "Event", Event)
    monkeypatch.setattr(torch.hpu, "current_stream", lambda: None)


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
        "head_input_boundary": True,
        "mtp_cache": False,
        "mtp_sat": False,
        "mtp_fp8": False,
        "mtp_k128": True,
        "draft_mhc": False,
        "shared_fp8": False,
        "bf16_head": True,
    }


def _valid_record(inputs):
    _, output, committed, output_count, _, draft_enabled, status = verify_control_from_target_batch(
        inputs.target_ids, inputs.proposals, inputs.control, inputs.active, inputs.binding_sidecar)
    return pack_record_batch(
        inputs.control,
        committed,
        output_count,
        output,
        torch.full_like(inputs.proposals, 77),
        draft_enabled,
        status,
        inputs.binding_sidecar,
    )


def test_scenarios_define_fixed_lane_capacity_matrix():
    n1 = scenarios_for_capacity(1)
    n2 = scenarios_for_capacity(2)
    n4 = scenarios_for_capacity(4)

    assert [len(item.lanes) for item in (*n1, *n2, *n4)] == [1, 1, 2, 2, 4, 4, 4]
    assert [[plan.active for plan in item.lanes] for item in n4] == [
        [True, True, True, True],
        [True, False, True, False],
        [False, True, False, False],
    ]
    assert n4[0].lanes[1] == LanePlan(True, 1, 0, 0)
    assert n4[1].lanes[2].accepted_prefix == 1


def test_round_metadata_uses_fixed_lane_and_expected_partial_commits():
    capacity = 4
    scenario = scenarios_for_capacity(capacity)[0]
    slots, owners = bind_all_lanes(capacity, "metadata")
    try:
        inputs = make_round_inputs(capacity, scenario, owners, round_generation=1, seed=7)
        result = verify_control_from_target_batch(
            inputs.target_ids, inputs.proposals, inputs.control, inputs.active, inputs.binding_sidecar)
        _, output, committed, output_count, _, draft_enabled, status = result

        assert inputs.request_slots.tolist() == [0, 1, 2, 3]
        assert inputs.slot_generations.tolist() == [owner.generation for owner in owners]
        assert committed.tolist() == output_count.tolist() == [6, 1, 3, 6]
        assert draft_enabled.tolist() == [True, True, True, True]
        assert status.tolist() == [0, 0, 0, 0]
        assert output[2].tolist() == [640, 641, 642, -1, -1, -1]
        assert inputs.target_states.shape == (capacity, 6, TARGET_CONTEXT_WIDTH)
        assert TARGET_CONTEXT_WIDTH == 3 * HIDDEN_SIZE
    finally:
        for owner in owners:
            slots.release(owner)


def test_inactive_lanes_use_sentinels_and_do_not_change_rings():
    capacity = 4
    scenario = scenarios_for_capacity(capacity)[1]
    slots, owners = bind_all_lanes(capacity, "inactive")
    try:
        inputs = make_round_inputs(capacity, scenario, owners, round_generation=1, seed=11)
        assert inputs.request_slots.tolist() == [0, -1, 2, -1]
        assert inputs.slot_generations[1].item() == inputs.slot_generations[3].item() == 0
        prior = (torch.zeros((capacity * 256, 8), dtype=torch.uint8), )
        unchanged = tuple(item.clone() for item in prior)
        checks = _inactive_ring_checks(prior, unchanged, inputs.active, capacity)
        assert checks and all(check["exact"] for check in checks)

        changed = (unchanged[0].clone(), )
        changed[0][256:512, 0] = 1
        broken = _inactive_ring_checks(prior, changed, inputs.active, capacity)
        assert not all(check["exact"] for check in broken)
    finally:
        for owner in owners:
            slots.release(owner)


def test_poison_replay_changes_only_inactive_ring_and_rejects_changed_active_inputs():
    capacity = 2
    scenario = scenarios_for_capacity(capacity)[1]
    slots, owners = bind_all_lanes(capacity, "poison-replay")
    try:
        inputs = make_round_inputs(
            capacity, scenario, owners, round_generation=1, seed=37)
        assert inputs.active.tolist() == [False, True]
        assert inputs.round_generation == 1

        rings = tuple(
            torch.full((capacity * SWA_ROWS, 528), layer + 1, dtype=torch.uint8)
            for layer in range(3)
        )
        poisoned = _poison_inactive_slots(rings, inputs.active, capacity, seed=41, device="cpu")
        for layer in range(3):
            inactive = slice(0, SWA_ROWS)
            active = slice(SWA_ROWS, 2 * SWA_ROWS)
            assert not torch.equal(poisoned[layer][inactive], rings[layer][inactive])
            assert torch.equal(poisoned[layer][active], rings[layer][active])

        unchanged_checks = _replay_input_checks(inputs, inputs, "poison_replay")
        assert len(unchanged_checks) == 15
        assert all(check["exact"] for check in unchanged_checks.values())

        changed_inputs = make_round_inputs(
            capacity, scenario, owners, round_generation=2, seed=37)
        changed_checks = _replay_input_checks(inputs, changed_inputs, "wrong_replay")
        assert not all(check["exact"] for check in changed_checks.values())
        assert changed_checks["target_states"]["exact"] is False
        assert changed_checks["round_generation"]["exact"] is False
        assert not torch.equal(inputs.target_states[1], changed_inputs.target_states[1])
    finally:
        for owner in owners:
            slots.release(owner)


def test_poison_replay_reuses_round_generation_on_a_fresh_verify_ring(monkeypatch):
    _patch_ring_runtime(monkeypatch)
    capacity = 2
    scenario = scenarios_for_capacity(capacity)[1]
    slots, owners = bind_all_lanes(capacity, "poison-replay-ring")
    first_ring = BatchVerifyRing(
        "cpu", batch_capacity=capacity, owner_validator=slots.validate_batch_identities)
    replay_ring = BatchVerifyRing(
        "cpu", batch_capacity=capacity, owner_validator=slots.validate_batch_identities)
    try:
        inputs = make_round_inputs(
            capacity, scenario, owners, round_generation=1, seed=43)
        record = _valid_record(inputs)

        def entry(*args):
            del args
            return record, None, None, None, None, None

        _, first_commit, _ = _readback_round(entry, first_ring, inputs)
        with pytest.raises(RuntimeError, match="round generation must advance"):
            _readback_round(entry, first_ring, inputs)
        _, replay_commit, _ = _readback_round(entry, replay_ring, inputs)

        assert first_ring.generation == replay_ring.generation == inputs.round_generation
        assert first_commit.committed == replay_commit.committed
        assert first_commit.output == replay_commit.output
        assert first_commit.draft == replay_commit.draft
    finally:
        first_ring.close()
        replay_ring.close()
        for owner in owners:
            slots.release(owner)


def test_exact_comparison_has_no_tolerance():
    reference = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    actual = reference.clone()
    actual.view(torch.int16)[1] += 1
    check = compare_exact("output", reference, actual)
    assert not check["exact"]
    assert check["mismatch"] == 1
    assert check["rtol"] == 0 and check["atol"] == 0


@pytest.mark.parametrize(
    ("dtype", "reference_value", "actual_value"),
    (
        (torch.int64, 7, 8),
        (torch.bool, True, False),
        (torch.float32, 0.0, -0.0),
    ),
)
def test_exact_comparison_supports_0d_scalars_without_relaxing_bits(
    dtype, reference_value, actual_value
):
    reference = torch.tensor(reference_value, dtype=dtype)
    actual = torch.tensor(actual_value, dtype=dtype)

    equal = compare_exact("scalar_equal", reference, reference.clone())
    assert equal["exact"] is True
    assert equal["mismatch"] == 0
    assert equal["shape"] == []

    changed = compare_exact("scalar_changed", reference, actual)
    assert changed["exact"] is False
    assert changed["mismatch"] == 1
    assert changed["rtol"] == 0 and changed["atol"] == 0


def _packed_result_fields(capacity):
    return (
        torch.arange(capacity * 16, dtype=torch.int64).reshape(capacity, 16),
        torch.arange(1, capacity + 1, dtype=torch.int64),
        torch.arange(capacity * 6, dtype=torch.int64).reshape(capacity, 6),
        torch.arange(11, 11 + capacity, dtype=torch.int64),
        torch.arange(capacity * 5, dtype=torch.int64).reshape(capacity, 5),
        torch.arange(capacity, dtype=torch.int32),
    )


def _serial_result_fields(packed, lane):
    return (
        packed[0][lane],
        packed[1][lane].reshape(1),
        packed[2][lane],
        packed[3][lane].reshape(1),
        packed[4][lane],
        packed[5][lane].to(torch.int64).reshape(1),
    )


@pytest.mark.parametrize(("capacity", "lane"), ((2, 1), (4, 1), (4, 3)))
def test_serial_vs_packed_result_comparison_selects_exact_lane(capacity, lane):
    packed = _packed_result_fields(capacity)
    serial = _serial_result_fields(packed, lane)

    full_batch = _compare_result_fields("probe_vs_packed", packed, packed)
    assert all(check["exact"] for check in full_batch.values())
    assert full_batch["record"]["shape"] == [capacity, 16]
    assert full_batch["committed"]["shape"] == [capacity]
    assert full_batch["output"]["shape"] == [capacity, 6]
    assert full_batch["output_count"]["shape"] == [capacity]
    assert full_batch["proposals"]["shape"] == [capacity, 5]
    assert full_batch["status"]["shape"] == [capacity]

    checks = _compare_result_fields("serial_vs_packed", serial, packed, lane=lane)
    assert all(check["exact"] for check in checks.values())
    assert checks["committed"]["shape"] == [1]
    assert checks["output_count"]["shape"] == [1]
    assert checks["status"]["shape"] == [1]
    assert checks["status"]["dtype"] == "torch.int64"

    wrong_lane = 0 if lane != 0 else 1
    wrong = _compare_result_fields("serial_vs_wrong_lane", serial, packed, lane=wrong_lane)
    assert not all(check["exact"] for check in wrong.values())


def test_status_int32_int64_representation_normalization_preserves_exact_values():
    packed = (
        torch.arange(16, dtype=torch.int64).reshape(1, 16),
        torch.tensor([3], dtype=torch.int64),
        torch.arange(6, dtype=torch.int64).reshape(1, 6),
        torch.tensor([3], dtype=torch.int64),
        torch.arange(5, dtype=torch.int64).reshape(1, 5),
        torch.tensor([2], dtype=torch.int32),
    )
    serial = (
        packed[0][0],
        packed[1][0].reshape(1),
        packed[2][0],
        packed[3][0].reshape(1),
        packed[4][0],
        packed[5][0].to(torch.int64).reshape(1),
    )

    check = _compare_result_fields("status_dtype", serial, packed, lane=0)["status"]
    assert check["exact"] is True
    assert check["mismatch"] == 0
    assert check["shape"] == [1]
    assert check["dtype"] == "torch.int64"

    changed = (serial[:5] + (torch.tensor([3], dtype=torch.int64), ))
    check = _compare_result_fields("status_changed", changed, packed, lane=0)["status"]
    assert check["exact"] is False
    assert check["mismatch"] == 1
    assert check["rtol"] == 0 and check["atol"] == 0


def test_anchor_singleton_and_scalar_shapes_normalize_without_relaxing_bits():
    reference = torch.tensor([1234], dtype=torch.int64)
    actual = torch.tensor(1234, dtype=torch.int64)

    check = _compare_singleton_exact("anchor", reference, actual)
    assert check["exact"] is True
    assert check["mismatch"] == 0
    assert check["shape"] == [1]
    assert check["rtol"] == 0 and check["atol"] == 0

    changed = _compare_singleton_exact(
        "anchor_changed", reference, torch.tensor(1235, dtype=torch.int64))
    assert changed["exact"] is False
    assert changed["mismatch"] == 1


def test_serial_lane_oracle_is_independent_and_returns_exact_record():
    capacity = 1
    scenario = Scenario("serial", (LanePlan(True, 6, 5, 2), ))
    slots, owners = bind_all_lanes(capacity, "serial")
    try:
        inputs = make_round_inputs(capacity, scenario, owners, round_generation=1, seed=13)
        bank = torch.arange(256 * 8, dtype=torch.uint8).reshape(256, 8)
        draft = SimpleNamespace(
            offsets=torch.arange(DRAFT_TOKENS, dtype=torch.int32),
            layers=[
                SimpleNamespace(attention=SimpleNamespace(swa=torch.zeros_like(bank)))
                for _ in range(3)
            ],
        )

        def verify_entry(target_ids, target_states, positions, control, proposals):
            assert target_ids.shape == positions.shape == (6,)
            assert target_states.shape == (6, TARGET_CONTEXT_WIDTH)
            assert control.shape == (7,) and proposals.shape == (5,)
            return (
                torch.tensor([10, 11, 12, -1, -1, -1]),
                torch.tensor(3),
                torch.tensor(3),
                torch.tensor(12),
                torch.tensor(True),
                torch.tensor(0),
                torch.tensor([132, 133, 134, 135, 136], dtype=torch.int32),
            )

        def mtp_entry(anchor, positions):
            del anchor, positions
            return (
                torch.ones((5, 4), dtype=torch.bfloat16),
                torch.ones((5, 6), dtype=torch.float32),
                torch.tensor([20, 21, 22, 23, 24]),
                torch.ones(5, dtype=torch.float32),
            )

        result = _serial_lane_result(
            draft, verify_entry, mtp_entry, inputs, 0, (bank, bank, bank))
        assert all(torch.equal(layer.attention.swa, bank) for layer in draft.layers)
        assert result["token_ids"].tolist() == [20, 21, 22, 23, 24]
        assert result["record"].tolist() == [
            int(inputs.control[0, 0]), 3, 3, 5, 10, 11, 12, -1, -1, -1, 20, 21, 22, 23, 24, 0,
        ]
    finally:
        for owner in owners:
            slots.release(owner)


def test_shared_backend_identity_counts_compiles_and_replays(monkeypatch):
    transaction_tool._shared_counting_backend.cache_clear()
    callback_shapes = []

    def backend(graph, inputs):
        del graph
        callback_shapes.append(tuple(tuple(item.shape) for item in inputs))
        return lambda *args: args[0] + 1

    def fake_compile(module, wrapped_backend):
        compiled = False

        def call(*args):
            nonlocal compiled
            if not compiled:
                wrapped_backend(module, args)
                compiled = True
            return module(*args)

        return call

    monkeypatch.setattr(transaction_tool, "_lookup_backend", lambda name: backend)
    monkeypatch.setattr(transaction_tool, "_compile", fake_compile)
    first = _GraphCallCounter(torch.nn.Identity())
    second = _GraphCallCounter(torch.nn.Identity())
    assert first.backend is second.backend
    first(torch.ones(2))
    first(torch.ones(2))
    second(torch.ones(3))
    usage = _graph_usage([first, second])
    assert usage["backend_compiles"] == 2
    assert usage["entry_replays"] == 1
    assert callback_shapes == [(torch.Size([2]), ), (torch.Size([3]), )]


@pytest.mark.parametrize("identity_device", ("cpu", "meta"))
def test_ring_readback_validates_owner_and_copies_once(monkeypatch, identity_device):
    _patch_ring_runtime(monkeypatch)
    capacity = 4
    scenario = scenarios_for_capacity(capacity)[1]
    slots, owners = bind_all_lanes(capacity, "readback")
    ring = BatchVerifyRing("cpu", batch_capacity=capacity, owner_validator=slots.validate_batch_identities)
    try:
        inputs = make_round_inputs(capacity, scenario, owners, round_generation=1, seed=17)
        record = _valid_record(inputs)
        if identity_device == "meta":
            inputs = replace(
                inputs,
                request_slots=torch.empty((capacity,), dtype=torch.int32, device="meta"),
                slot_generations=torch.empty((capacity,), dtype=torch.int64, device="meta"),
                active=torch.empty((capacity,), dtype=torch.bool, device="meta"),
            )

        def entry(*args):
            del args
            return (record, torch.tensor([6, 0, 2, 0]), torch.zeros((capacity, 6), dtype=torch.int64),
                    torch.tensor([6, 0, 2, 0]), torch.zeros((capacity, 5), dtype=torch.int64),
                    torch.tensor([0, BATCH_INACTIVE, 0, BATCH_INACTIVE]))

        values, commit, evidence = _readback_round(entry, ring, inputs)
        assert values[0].shape == (capacity, 16)
        assert evidence["record_copy_count"] == evidence["completion_sync_count"] == 1
        assert commit.committed == (6, 0, 2, 0)
    finally:
        ring.close()
        for owner in owners:
            slots.release(owner)


def test_owner_reuse_rejects_old_ticket_and_accepts_new_generation(monkeypatch):
    _patch_ring_runtime(monkeypatch)
    capacity = 4
    lane = 3
    scenario = Scenario("reuse", tuple(
        LanePlan(True, 6, 5, 2) if index == lane else LanePlan(False) for index in range(capacity)))
    slots, owners = bind_all_lanes(capacity, "reuse")
    ring = BatchVerifyRing("cpu", batch_capacity=capacity, owner_validator=slots.validate_batch_identities)
    try:
        first = make_round_inputs(capacity, scenario, owners, round_generation=1, seed=19)
        ticket = ring.acquire(
            round_generation=1,
            request_slots=first.request_slots,
            slot_generations=first.slot_generations,
            max_committed=first.target_counts,
            active=first.active,
        )
        ticket.record.copy_(_valid_record(first))
        ring.stage(ticket)
        old = owners[lane]
        slots.release(old)
        replacement = slots.acquire("replacement")
        owners[lane] = replacement
        assert replacement.index == old.index and replacement.generation == old.generation + 1
        with pytest.raises(RuntimeError, match="no longer owns"):
            ring.consume(ticket)
        syncs = ring.completion_sync_count
        ring.abort(ticket)
        assert ring.completion_sync_count == syncs + 1

        second = make_round_inputs(capacity, scenario, owners, round_generation=2, seed=23)
        record = _valid_record(second)

        def entry(*args):
            del args
            return (record, torch.tensor([3, 0, 0, 0]), torch.zeros((capacity, 6), dtype=torch.int64),
                    torch.tensor([3, 0, 0, 0]), torch.zeros((capacity, 5), dtype=torch.int64),
                    torch.tensor([0, BATCH_INACTIVE, BATCH_INACTIVE, 0]))

        _, commit, evidence = _readback_round(entry, ring, second)
        assert commit.committed[lane] == 3
        assert evidence["completion_sync_count"] == 1
    finally:
        ring.close()
        for owner in owners:
            try:
                slots.release(owner)
            except RuntimeError:
                pass


def test_flags_block_c1_c6_specialized_mtp_paths():
    flags = _compatible_flags()
    assert validate_transaction_flags(flags, mtp_k128_eligible=False) == []
    for name in ("mtp_cache", "mtp_sat", "mtp_fp8", "draft_mhc", "shared_fp8"):
        changed = dict(flags)
        changed[name] = True
        assert [item["flag"] for item in validate_transaction_flags(changed, mtp_k128_eligible=False)] == [name]
    assert [item["flag"] for item in validate_transaction_flags(flags, mtp_k128_eligible=True)] == ["mtp_k128"]


def test_source_fingerprints_cover_new_tool_and_test():
    records = source_fingerprints(Path(__file__).resolve().parents[3])
    assert records["tools/check_deepseek_v41_dspark_packed_transaction.py"]["bytes"] > 0
    assert records["tests/standalone/deepseek_v41/test_dspark_packed_transaction_tool.py"]["bytes"] > 0


def test_missing_prepared_returns_nonzero_and_writes_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(transaction_tool.traceback, "print_exc", lambda: None)
    output = tmp_path / "failure.json"
    code = transaction_tool.main([
        str(tmp_path / "missing-prepared"),
        "--output",
        str(output),
        "--batch-capacity",
        "2",
    ])
    assert code == 1
    report = __import__("json").loads(output.read_text())
    assert report["status"] == "failed"
    assert "FileNotFoundError" in report["error"]


def test_serial_verify_context_uses_request_local_shapes():
    slots, owners = bind_all_lanes(1, "serial-shapes")
    calls = []
    try:
        inputs = make_round_inputs(
            1, Scenario("partial", (LanePlan(True, 6, 5, 2),)), owners,
            round_generation=1, seed=29)
        draft = SimpleNamespace(
            offsets=torch.arange(5, dtype=torch.int32),
            insert_context=lambda states, positions, committed: calls.append(
                (states.shape, positions.shape, committed.tolist())),
        )
        entry = transaction_tool._SerialVerifyContextEntry(draft)
        result = entry(inputs.target_ids[0], inputs.target_states[0], inputs.positions[0],
                       inputs.control[0], inputs.proposals[0])
        assert result[1].tolist() == [3]
        assert result[-1].tolist() == [131, 132, 133, 134, 135]
        assert calls == [(torch.Size([6, TARGET_CONTEXT_WIDTH]), torch.Size([6]), [3])]
    finally:
        for owner in owners:
            slots.release(owner)


def test_packed_stage_probe_entry_threads_i32_sidecar_through_production_stages(monkeypatch):
    capacity = 1
    slots, owners = bind_all_lanes(capacity, "compiled-probe")
    calls = {}
    try:
        inputs = make_round_inputs(
            capacity,
            Scenario("probe", (LanePlan(True, 6, 5, 2),)),
            owners,
            round_generation=1,
            seed=31,
        )
        transaction = SimpleNamespace(
            draft=object(),
            batch_state=object(),
            batch_capacity=capacity,
            round=SimpleNamespace(),
        )
        context_done = tuple(torch.tensor([index + 1], dtype=torch.int32) for index in range(3))

        def verify_entry(target_ids, proposals, control, active, binding_sidecar):
            calls["verify"] = (target_ids, proposals, control, active, binding_sidecar)
            return (
                torch.empty(0),
                torch.full((capacity, 6), 10, dtype=torch.int64),
                torch.tensor([3], dtype=torch.int64),
                torch.tensor([3], dtype=torch.int64),
                torch.empty(0),
                torch.tensor([True]),
                torch.tensor([0], dtype=torch.int64),
            )

        def insert_context(draft, batch_state, target_states, positions, request_slots, committed, valid_context):
            calls["insert"] = (
                draft, batch_state, target_states, positions, request_slots, committed, valid_context)
            return context_done

        def draft_inputs(positions, output, output_count, committed, draft_active):
            calls["draft_inputs"] = (positions, output, output_count, committed, draft_active)
            return (
                torch.tensor([10], dtype=torch.int64),
                torch.arange(5, dtype=torch.int32).reshape(1, 5),
            )

        def backbone(anchors, positions, request_slots, draft_active, *, context_done=None):
            calls["backbone"] = (anchors, positions, request_slots, draft_active, context_done)
            return (
                torch.ones((capacity, 5, 2), dtype=torch.bfloat16),
                torch.ones((capacity, 5, 3), dtype=torch.float32),
            )

        def markov(anchors, hidden, logits, draft_active):
            calls["markov"] = (anchors, hidden, logits, draft_active)
            return (
                torch.tensor([[20, 21, 22, 23, 24]], dtype=torch.int64),
                torch.ones((capacity, 5), dtype=torch.float32),
            )

        def pack_verify_record(control, committed, output_count, output, proposals,
                               draft_active, status, binding_sidecar):
            calls["pack"] = (
                control, committed, output_count, output, proposals, draft_active, status, binding_sidecar)
            return torch.zeros((capacity, 16), dtype=torch.int64)

        monkeypatch.setattr(transaction_tool, "verify_control_from_target_batch", verify_entry)
        monkeypatch.setattr(transaction_tool, "insert_context_batch", insert_context)
        monkeypatch.setattr(transaction_tool, "draft_inputs_after_commit", draft_inputs)
        monkeypatch.setattr(transaction_tool, "pack_record_batch", pack_verify_record)
        transaction.round.backbone = backbone
        transaction.round.markov = markov

        values = transaction_tool._PackedStageProbeEntry(transaction)(*inputs.args())
        probe = transaction_tool.PackedProbe(*values)

        assert inputs.binding_sidecar.dtype == torch.int32
        assert inputs.binding_sidecar.shape == (capacity, 2)
        assert calls["verify"][-1] is inputs.binding_sidecar
        assert calls["pack"][-1] is inputs.binding_sidecar
        assert calls["insert"][0] is transaction.draft
        assert calls["insert"][1] is transaction.batch_state
        assert calls["backbone"][0].shape == (capacity, 1)
        assert calls["backbone"][-1] is context_done
        assert probe.record.shape == (capacity, 16)
        assert probe.proposals.tolist() == [[20, 21, 22, 23, 24]]
        assert probe.positions.tolist() == [[0, 1, 2, 3, 4]]
        assert probe.status.tolist() == [0]
    finally:
        for owner in owners:
            slots.release(owner)
