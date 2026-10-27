# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for unified DSpark capacity admission and runner entry."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.entrypoints.serve.exception_handling.error_response import create_error_response
from vllm.exceptions import VLLMClientError, VLLMValidationError
from vllm.sampling_params import SamplingParams

from vllm_gaudi.models.deepseek_v41 import (
    HpuDeepseekV41ForCausalLM,
    _validate_dspark_batch_native_runtime,
)
from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
from vllm_gaudi.ops.deepseek_v41_capacity import (
    MAX_DSV41_DSPARK_CAPACITY,
    MIN_DSV41_DSPARK_CAPACITY,
    validate_dspark_capacity,
)
from vllm_gaudi.ops.deepseek_v41_config import (
    is_dspark_b2_tp8_profile,
    is_dspark_batch_tp8_profile,
    validate_dspark_batch_capacity,
    validate_dspark_tp8_profile,
)
from vllm_gaudi.platform import HpuPlatform
from vllm_gaudi.v1.worker.deepseek_v41_runner import V41ModelRunner


@pytest.fixture(autouse=True)
def _isolate_capacity_profile(monkeypatch):
    monkeypatch.setattr(HpuPlatform, "_v41_dspark_batch_profile", False)
    monkeypatch.setattr(HpuPlatform, "_v41_dspark_b2_profile", False)
    monkeypatch.setenv("VLLM_HPU_DSV41_PREPARED_SHARDS", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    for name in (
            "VLLM_HPU_DSV41_DEVICE_VERIFY",
            "VLLM_HPU_DSV4_NATIVE_DECODE_GRAPH",
            "VLLM_HPU_NATIVE_DECODE_GRAPH",
            "QUANT_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)
    from vllm_gaudi.entrypoints.deepseek_v41 import _DSPARK_TP8_DEFAULTS

    for name in _DSPARK_TP8_DEFAULTS:
        monkeypatch.setenv(name, "0")


def _config(*, max_num_seqs=4, tp=8, pp=1, prefix=False, async_scheduling=False):
    from vllm.config import CompilationMode

    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
            is_hybrid=False,
            dtype=torch.bfloat16,
            quantization=None,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            worker_cls="test.worker",
            distributed_executor_backend=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            async_scheduling=async_scheduling,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=prefix,
            user_specified_block_size=True,
            block_size=128,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            custom_ops=None,
            cudagraph_mode=None,
            cudagraph_capture_sizes=None,
        ),
        speculative_config=None,
        load_config=SimpleNamespace(device=None),
    )


def _record_platform_profile(config):
    with (
            patch.object(HpuPlatform, "set_compile_env_defaults"),
            patch("vllm_gaudi.platform._enable_hpu_v1_dflash2_validation"),
            patch("vllm_gaudi.platform.get_config", return_value=SimpleNamespace(VLLM_CONTIGUOUS_PA=False)),
            patch("vllm_gaudi.ops.deepseek_v41_config.configure"),
    ):
        HpuPlatform.check_and_update_config(config)


def _runner(config):
    runner = object.__new__(V41ModelRunner)
    runner.vllm_config = config
    return runner


def test_shared_capacity_contract_keeps_serial_and_concurrent_bounds_distinct():
    assert (MIN_DSV41_DSPARK_CAPACITY, MAX_DSV41_DSPARK_CAPACITY) == (2, 64)
    for capacity in (2, 3, 6, 8, 11, 16, 64):
        assert validate_dspark_capacity(capacity) == capacity
        assert validate_dspark_capacity(capacity, allow_serial=True) == capacity
    assert validate_dspark_capacity(1, allow_serial=True) == 1
    for capacity in (1, 65, True, 2.0):
        with pytest.raises(ValueError, match="integer in 2..64"):
            validate_dspark_capacity(capacity)
    assert RequestSlots(64).capacity == 64
    with pytest.raises(ValueError, match="1..64"):
        RequestSlots(65)


@pytest.mark.parametrize("capacity", (2, 3, 6, 8, 16, 64))
def test_generic_profile_accepts_each_concurrent_capacity(capacity):
    config = _config(max_num_seqs=capacity)

    assert validate_dspark_batch_capacity(config) == capacity
    assert is_dspark_batch_tp8_profile(config)
    assert is_dspark_b2_tp8_profile(config) is (capacity == 2)
    validate_dspark_tp8_profile(config)


@pytest.mark.parametrize("capacity", (0, -1, 65, True, False, 2.0, "2", None))
def test_capacity_validation_rejects_invalid_values_early(capacity):
    config = _config(max_num_seqs=capacity)

    with pytest.raises(ValueError, match="integer in 1..64"):
        validate_dspark_batch_capacity(config)
    with pytest.raises(ValueError, match="integer in 1..64"):
        validate_dspark_tp8_profile(config)
    assert not is_dspark_batch_tp8_profile(config)
    assert not is_dspark_b2_tp8_profile(config)


@pytest.mark.parametrize(
    "overrides",
    (
        {"tp": 4},
        {"pp": 2},
        {"prefix": True},
        {"async_scheduling": True},
    ),
)
def test_concurrent_profile_rejects_topology_prefix_and_async(overrides):
    config = _config(max_num_seqs=3, **overrides)

    assert not is_dspark_batch_tp8_profile(config)
    with pytest.raises(ValueError, match="DSpark concurrency|synchronous|prefix|TP8"):
        validate_dspark_tp8_profile(config)


def test_concurrent_profile_rejects_fixed_b2_default_switches():
    from vllm_gaudi.entrypoints.deepseek_v41 import _DSPARK_TP8_DEFAULTS

    switch = next(iter(_DSPARK_TP8_DEFAULTS))
    with patch.dict("os.environ", {switch: "1"}):
        with pytest.raises(ValueError, match="set these variables to 0"):
            validate_dspark_tp8_profile(_config(max_num_seqs=4))


@pytest.mark.parametrize(("tp", "pp"), ((2, 2), (4, 1), (8, 1)))
def test_n1_preserves_topology_and_sampled_device_verify(tp, pp, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_DEVICE_VERIFY", "1")
    config = _config(max_num_seqs=1, tp=tp, pp=pp)

    assert validate_dspark_batch_capacity(config) == 1
    assert not is_dspark_batch_tp8_profile(config)
    validate_dspark_tp8_profile(config)
    _record_platform_profile(config)
    sampled = SamplingParams(temperature=1, max_tokens=8)
    HpuPlatform.validate_request(None, sampled)
    _runner(config)._validate_sampling(sampled)


@pytest.mark.parametrize("capacity", (1, 2, 3, 6, 8, 16, 64))
def test_unsupported_sampling_is_http_400_at_every_service_capacity(capacity, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_DEVICE_VERIFY", "1")
    config = _config(max_num_seqs=capacity)
    _record_platform_profile(config)

    with pytest.raises(VLLMValidationError, match="penalties") as exc_info:
        HpuPlatform.validate_request(
            None, SamplingParams(temperature=0, presence_penalty=1.0, max_tokens=8))
    assert isinstance(exc_info.value, VLLMClientError)
    assert create_error_response(exc_info.value).error.code == 400


def test_profile_reset_from_n4_to_n1_restores_sampled_admission(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_DEVICE_VERIFY", "1")
    sampled = SamplingParams(temperature=1, max_tokens=8)
    _record_platform_profile(_config(max_num_seqs=4))
    with pytest.raises(VLLMValidationError, match="greedy-only"):
        HpuPlatform.validate_request(None, sampled)

    _record_platform_profile(_config(max_num_seqs=1, tp=4, pp=1))
    HpuPlatform.validate_request(None, sampled)


@pytest.mark.parametrize("capacity", (2, 3, 6, 8, 16, 64))
def test_native_masked_compressor_is_required_before_weight_load(capacity, monkeypatch):
    monkeypatch.setattr(torch.ops, "custom_op", SimpleNamespace(), raising=False)

    with pytest.raises(RuntimeError, match=f"concurrent capacity {capacity}.*masked-sequence"):
        _validate_dspark_batch_native_runtime(_config(max_num_seqs=capacity))


@pytest.mark.parametrize("capacity", (2, 3, 6, 8, 16, 64))
def test_native_masked_compressor_presence_satisfies_batch_runtime(capacity, monkeypatch):
    monkeypatch.setattr(
        torch.ops,
        "custom_op",
        SimpleNamespace(custom_deepseek_v41_compressor_sequence_masked_bf16_gaudi2=object()),
        raising=False,
    )

    _validate_dspark_batch_native_runtime(_config(max_num_seqs=capacity))


def test_native_masked_compressor_is_not_imposed_on_n1(monkeypatch):
    monkeypatch.setattr(torch.ops, "custom_op", SimpleNamespace(), raising=False)

    _validate_dspark_batch_native_runtime(_config(max_num_seqs=1, tp=4, pp=1))


def test_request_state_accepts_custom_capacity_and_rejects_configured_capacity_mismatch(monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_batch_state

    monkeypatch.setattr(
        deepseek_v41_batch_state,
        "BatchStageState",
        lambda program, capacity: SimpleNamespace(capacity=capacity, slots=RequestSlots(capacity)),
    )
    model = object.__new__(HpuDeepseekV41ForCausalLM)
    model.native = True
    model.program = SimpleNamespace(dspark=True)
    model.vllm_config = _config(max_num_seqs=64)

    model.initialize_request_state(64)
    assert model.batch_state.capacity == 64
    assert model.batch_state.slots.capacity == 64
    with pytest.raises(ValueError, match="does not match configured max_num_seqs 64"):
        model.initialize_request_state(3)


def _mock_model_initializer(monkeypatch, config, *, initialized_capacity=None, pending=None):
    from vllm_gaudi.ops import deepseek_v41_batch_draft, deepseek_v41_batch_verify
    from vllm_gaudi.v1.worker import deepseek_v41_dspark_batch_runner as adapter

    target, draft = object(), SimpleNamespace()
    captured = {}
    model = object.__new__(HpuDeepseekV41ForCausalLM)
    model.native = True
    model.program = SimpleNamespace(dspark=True, draft=object())
    model.vllm_config = config
    model.device = "cpu"
    model.max_model_len = 8192
    model.engram_host = None

    def initialize_state(capacity):
        actual = capacity if initialized_capacity is None else initialized_capacity
        model.batch_state = SimpleNamespace(
            capacity=actual,
            slots=RequestSlots(actual),
            pending=pending,
            single_owner=None,
            prefill_owner=None,
        )

    model.initialize_request_state = initialize_state
    def target_factory(*args, **kwargs):
        captured["target"] = (target, args, kwargs)
        return target

    monkeypatch.setattr(adapter, "DSparkPackedTargetStages", target_factory)
    monkeypatch.setattr(
        deepseek_v41_batch_draft,
        "DraftBatchState",
        lambda draft_program, capacity: SimpleNamespace(capacity=capacity),
    )
    monkeypatch.setattr(
        deepseek_v41_batch_verify,
        "BatchVerifyRing",
        lambda device, batch_capacity, owner_validator: SimpleNamespace(
            capacity=batch_capacity, owner_validator=owner_validator),
    )
    monkeypatch.setattr(
        deepseek_v41_batch_draft,
        "DSparkBatchDraftStage",
        lambda *args, **kwargs: setattr(draft, "capacity", kwargs["batch_capacity"]) or draft,
    )
    return model, target, draft, captured


def test_model_initializes_slots_target_draft_and_verify_from_one_capacity(monkeypatch):
    config = _config(max_num_seqs=4)
    model, target, draft, captured = _mock_model_initializer(monkeypatch, config)

    assert model.initialize_dspark_batch(4) == (target, draft)
    assert model.batch_state.capacity == 4
    assert model.batch_state.slots.capacity == 4
    assert model.dspark_draft_state.capacity == 4
    assert model.dspark_verify_ring.capacity == 4
    assert captured["target"][2]["batch_capacity"] == 4


def test_model_rejects_capacity_mismatch_before_allocating_state(monkeypatch):
    config = _config(max_num_seqs=4)
    model, _, _, _ = _mock_model_initializer(monkeypatch, config)

    with pytest.raises(ValueError, match="does not match configured max_num_seqs 4"):
        model.initialize_dspark_batch(3)
    assert not hasattr(model, "batch_state")


def test_model_rejects_state_capacity_mismatch(monkeypatch):
    config = _config(max_num_seqs=4)
    model, _, _, _ = _mock_model_initializer(monkeypatch, config, initialized_capacity=2)

    with pytest.raises(RuntimeError, match="state and RequestSlots.*capacity 4"):
        model.initialize_dspark_batch(4)


def test_model_requires_authoritative_request_slots_before_capacity_check(monkeypatch):
    config = _config(max_num_seqs=4)
    model, _, _, _ = _mock_model_initializer(monkeypatch, config)
    model.initialize_request_state = lambda capacity: setattr(
        model, "batch_state", SimpleNamespace(capacity=capacity))

    with pytest.raises(RuntimeError, match="authoritative RequestSlots owner validation"):
        model.initialize_dspark_batch(4)


def test_model_rejects_undrained_b1_close_loop_state(monkeypatch):
    config = _config(max_num_seqs=4)
    model, _, _, _ = _mock_model_initializer(monkeypatch, config, pending=(1, ()))

    with pytest.raises(RuntimeError, match="drained B1 close-loop state"):
        model.initialize_dspark_batch(4)


def test_main_runner_passes_deployment_capacity_to_batch_adapter(monkeypatch):
    from vllm_gaudi.v1.worker import deepseek_v41_runner

    call = {}
    state = SimpleNamespace(
        capacity=4,
        slots=RequestSlots(4),
        pending=None,
        single_owner=None,
        prefill_owner=None,
    )

    class FakeAdapter:

        def __init__(self, target, draft, **kwargs):
            call.update(target=target, draft=draft, **kwargs)
            self.batch_capacity = kwargs["batch_capacity"]

    class FakeTrace:

        def __init__(self, **kwargs):
            call["trace"] = kwargs

    runner = object.__new__(V41ModelRunner)
    runner.model = SimpleNamespace(
        batch_state=state,
        tp_rank=0,
        pp_rank=0,
    )
    runner.model.initialize_dspark_batch = lambda capacity: (
        call.__setitem__("model_capacity", capacity) or (object(), object()))
    monkeypatch.setattr(deepseek_v41_runner, "DSparkBatchRunner", FakeAdapter)
    monkeypatch.setattr(deepseek_v41_runner, "DSparkBatchTrace", FakeTrace)
    monkeypatch.setattr(deepseek_v41_runner.dist, "is_initialized", lambda: False)

    runner._initialize_dspark_batch_runner(4)
    assert call["model_capacity"] == 4
    assert call["batch_capacity"] == 4
    assert call["slots"] is state.slots
    assert runner.dspark_batch_capacity == 4


def test_main_runner_rejects_stale_capacity(monkeypatch):
    runner = object.__new__(V41ModelRunner)
    runner.vllm_config = _config(max_num_seqs=4)
    runner.dspark_batch_capacity = 2

    with pytest.raises(RuntimeError, match="capacity 2 does not match configured max_num_seqs 4"):
        runner._validate_dspark_batch_capacity()


def _scheduled(*req_ids):
    return SimpleNamespace(
        num_scheduled_tokens={req_id: 1 for req_id in req_ids},
        scheduled_spec_decode_tokens={req_id: () for req_id in req_ids},
    )


def _candidate_runner(requests):
    runner = object.__new__(V41ModelRunner)
    runner.dspark_batch_runner = object()
    runner.dspark_batch_capacity = 4
    runner.requests = requests
    return runner


def _decode_request(mm_features=False):
    return SimpleNamespace(num_computed_tokens=8, decode_start=4, mm_features=mm_features)


def test_main_candidate_accepts_one_through_n_steady_decode_requests():
    runner = _candidate_runner({f"req-{index}": _decode_request() for index in range(4)})

    assert not runner._dspark_batch_candidate(_scheduled())
    assert runner._dspark_batch_candidate(_scheduled("req-0"))
    # A single active request can occupy any canonical slot; gaps do not
    # change the packed admission decision.
    assert runner._dspark_batch_candidate(_scheduled("req-2"))
    assert runner._dspark_batch_candidate(_scheduled("req-0", "req-1", "req-2", "req-3"))
    assert runner._dspark_b2_candidate(_scheduled("req-0"))


def test_main_candidate_rejects_out_of_range_multimodal_and_prefill_rounds():
    requests = {f"req-{index}": _decode_request() for index in range(5)}
    requests["mm"] = _decode_request(mm_features=True)
    requests["prefill"] = SimpleNamespace(num_computed_tokens=1, decode_start=4, mm_features=False)
    runner = _candidate_runner(requests)

    assert not runner._dspark_batch_candidate(_scheduled("req-0", "req-1", "req-2", "req-3", "req-4"))
    assert not runner._dspark_batch_candidate(_scheduled("mm"))
    assert not runner._dspark_batch_candidate(_scheduled("req-0", "prefill"))
    assert not runner._dspark_batch_candidate(_scheduled("missing"))


def test_execute_model_dispatches_steady_a1_to_packed_batch_path():
    runner = _candidate_runner({"req-0": _decode_request()})
    runner.pending = None
    runner.prefix_checkpoints = None
    updated, dispatched = [], []
    runner._update = lambda scheduled: updated.append(scheduled)
    runner._execute_dspark_batch = lambda scheduled: dispatched.append(scheduled) or "packed"
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"req-0": 1},
        auxiliary_prefix_operations=None,
    )

    assert runner.execute_model(scheduled) == "packed"
    assert updated == [scheduled]
    assert dispatched == [scheduled]


def test_execute_model_rejects_over_capacity_round_before_mutation():
    requests = {f"req-{index}": _decode_request() for index in range(3)}
    runner = _candidate_runner(requests)
    runner.dspark_batch_capacity = 2
    runner.pending = None
    mutated = []
    runner._update = lambda scheduled: mutated.append(scheduled)
    scheduled = SimpleNamespace(
        num_scheduled_tokens={req_id: 1 for req_id in requests},
        auxiliary_prefix_operations=None,
    )

    with pytest.raises(RuntimeError, match="count 3 exceeds configured max_num_seqs 2"):
        runner.execute_model(scheduled)
    assert not mutated


def test_packed_request_preparation_does_not_select_a_b1_binding():
    owner = SimpleNamespace(index=2, generation=7, request_id="req-2")
    calls = []

    class Bank:

        def acquire(self, request_id):
            calls.append(("acquire", request_id))
            return owner

        def publish_pages(self, slot, blocks, pool_blocks):
            calls.append(("pages", slot, blocks, pool_blocks))

        def bind_single(self, *args):
            raise AssertionError("packed preparation must not select a B1 working binding")

        def bind_prefill(self, *args):
            raise AssertionError("packed preparation must not select a prefill binding")

    class Engram:

        def activate(self, request_id, *, reset):
            calls.append(("engram", request_id, reset))

    runner = object.__new__(V41ModelRunner)
    runner.model = SimpleNamespace(batch_state=Bank(), engram_host=Engram())
    runner.state = SimpleNamespace(blocks=16)
    runner.active_request = None
    request = SimpleNamespace(req_id="req-2", block_ids=((1, 2),), num_computed_tokens=8)

    assert runner._prepare_dspark_batch_request(request) is owner
    assert calls == [
        ("acquire", "req-2"),
        ("pages", owner, (1, 2), 16),
        ("engram", "req-2", False),
    ]


def test_packed_round_uses_owner_preparation_instead_of_per_request_b1_bind():
    owner = SimpleNamespace(index=0, generation=3, request_id="req-0")
    request = _decode_request()
    entry = SimpleNamespace(req_id="req-0", request=request, request_slot=0, slot_generation=3)
    plan = SimpleNamespace(requests=(entry,))
    calls = []

    class Adapter:

        def build_plan(self, scheduled, requests, owners, *, max_model_len):
            assert owners == {"req-0": owner}
            assert max_model_len == 8192
            return plan

        def execute(self, actual_plan):
            assert actual_plan is plan
            return "model-output", "draft-output"

    runner = object.__new__(V41ModelRunner)
    runner.dspark_batch_capacity = 4
    runner.dspark_batch_runner = Adapter()
    runner.model = SimpleNamespace(batch_state=SimpleNamespace(slots=SimpleNamespace(owners={"req-0": owner})))
    runner.model_config = SimpleNamespace(max_model_len=8192)
    runner.requests = {"req-0": request}
    runner._prepare_dspark_batch_request = lambda actual: calls.append(actual) or owner
    runner._bind_request = lambda actual: (_ for _ in ()).throw(
        AssertionError("packed rounds must not call the serial B1 binder"))
    runner.pending = None

    assert runner._execute_dspark_batch(_scheduled("req-0")) is None
    assert calls == [request]
    assert runner.batch_result == "model-output"
    assert runner.draft_token_ids == "draft-output"
    assert runner.pending == "batch_ready"


@pytest.mark.parametrize("capacity", [3, 11, 32, 64])
def test_optional_native_b1_backend_keeps_configured_multi_request_capacity(capacity):
    runner = _candidate_runner({"first": _decode_request(), "last": _decode_request()})
    runner.dspark_batch_capacity = capacity
    runner.model = SimpleNamespace(native=True)
    runner.dspark_compact_native_single = True
    assert not runner._dspark_batch_candidate(_scheduled("last"))
    assert runner._dspark_batch_candidate(_scheduled("first", "last"))
    assert runner.dspark_batch_capacity == capacity
    runner.dspark_compact_native_single = False
    assert runner._dspark_batch_candidate(_scheduled("last"))


@pytest.mark.parametrize("field", ["device_round_inputs", "device_round_queue", "device_round_tail"])
def test_optional_native_b1_backend_preserves_queued_round_owners(field):
    runner = _candidate_runner({"request": _decode_request()})
    runner.model = SimpleNamespace(native=True)
    runner.dspark_compact_native_single = True
    setattr(runner, field, [object()] if field == "device_round_tail" else object())
    assert runner._dspark_batch_candidate(_scheduled("request"))


def test_optional_native_b1_backend_requires_actual_native_runtime():
    runner = _candidate_runner({"request": _decode_request()})
    runner.model = SimpleNamespace(native=False)
    runner.dspark_compact_native_single = True
    assert runner._dspark_batch_candidate(_scheduled("request"))
