# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the scheduler-facing fixed-capacity DSpark adapter."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import OPTConfig
from vllm.exceptions import VLLMValidationError

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_gaudi.v1.worker.deepseek_v41_dspark_batch_runner import (
    DSparkBatchPlan,
    DSparkBatchRunner,
    DSparkBatchTrace,
    DSparkDraftBatch,
    DSparkPackedTargetStages,
    DSparkTargetBatch,
    _fixed_page_table_width,
    _target_greedy_ids,
)
from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_physical_rows
from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata
from vllm_gaudi.ops.deepseek_v41_batch_target import DSparkTargetBatch as TargetBatchContract
from vllm_gaudi.ops.deepseek_v41_config import (
    is_dspark_b2_tp8_profile,
    validate_dspark_tp8_profile,
    validate_sampling,
)
from vllm_gaudi.models.deepseek_v41_program import PreparedStage
from vllm_gaudi.models.deepseek_v41 import HpuDeepseekV41ForCausalLM
from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState


EOS_TOKEN_ID = 50256


def _scheduler(tmp_path):
    init_none_hash(sha256)
    OPTConfig(
        architectures=["OPTForCausalLM"],
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8192,
    ).save_pretrained(tmp_path)
    speculative = SpeculativeConfig(model="ngram", num_speculative_tokens=5)
    speculative.parallel_drafting = True
    config = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path), dtype="float16", max_model_len=8192),
        parallel_config=ParallelConfig(),
        cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            async_scheduling=False,
            watermark=0.0,
        ),
        speculative_config=speculative,
    )
    config.cache_config.num_gpu_blocks = 10000
    register_all_kvcache_specs(config)
    cache = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32),
            )
        ],
    )
    return Scheduler(
        vllm_config=config,
        kv_cache_config=cache,
        block_size=16,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(req_id, prompt):
    params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(request_id=req_id, prompt_token_ids=prompt, sampling_params=params, pooling_params=None)


def _prefill(scheduler, requests, anchors):
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(ids)},
            sampled_token_ids=[[anchor] for anchor in anchors],
        ),
    )


def _runner_states(scheduler, output):
    cached = output.scheduled_cached_reqs
    starts = dict(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
    result = {}
    for req_id in output.num_scheduled_tokens:
        scheduled = scheduler.requests[req_id]
        result[req_id] = RequestState(
            req_id=req_id,
            prompt=list(scheduled.prompt_token_ids),
            mm_features=[],
            sampling_params=scheduled.sampling_params,
            block_ids=scheduler.kv_cache_manager.get_block_ids(req_id),
            num_computed_tokens=starts[req_id],
            output=list(scheduled.output_token_ids),
        )
    return result


def _decode_case(tmp_path, drafts, anchors=(101, 201)):
    scheduler = _scheduler(tmp_path)
    requests = [_request("a", [1, 2]), _request("b", [3, 4])]
    _prefill(scheduler, requests, anchors)
    scheduler.update_draft_token_ids(DraftTokenIds(["a", "b"], drafts))
    # These requests have sampled their anchor and are steady decodes now.
    for request in scheduler.running:
        request.is_prefill_chunk = False
    output = scheduler.schedule()
    states = _runner_states(scheduler, output)
    owners = {req_id: SimpleNamespace(index=index, generation=7 + index)
              for index, req_id in enumerate(output.num_scheduled_tokens)}
    return scheduler, output, states, owners


def _make_plan(output, states, owners):
    # Exercise the same planner without requiring target/draft stages.
    class Planner(DSparkBatchRunner):
        def __init__(self):
            self.round_generation = 0
            self.pending = False

    return Planner().build_plan(output, states, owners, max_model_len=8192)


def _single_page_plan(capacity, max_model_len):
    page_ids = (7, 3, 11, 5)[:capacity]
    request_ids = tuple(f"short-{lane}" for lane in range(capacity))
    requests = {
        request_id: RequestState(
            req_id=request_id,
            prompt=[],
            mm_features=[],
            sampling_params=SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
            block_ids=((page_ids[lane],),),
            num_computed_tokens=0,
            output=[100 + lane],
        )
        for lane, request_id in enumerate(request_ids)
    }
    owners = {
        request_id: SimpleNamespace(index=lane, generation=lane + 1)
        for lane, request_id in enumerate(request_ids)
    }
    scheduled = SimpleNamespace(
        num_scheduled_tokens={request_id: 1 for request_id in request_ids},
        scheduled_spec_decode_tokens={},
    )
    plan = DSparkBatchPlan.from_scheduler(
        scheduled,
        requests,
        owners,
        max_model_len=max_model_len,
        round_generation=1,
        batch_capacity=capacity,
    )
    return plan, page_ids


def _multi_page_plan(capacity, max_model_len, page_count):
    page_rows = tuple(
        tuple(10000 + lane * 1000 + index for index in range(page_count))
        for lane in range(capacity)
    )
    request_ids = tuple(f"multi-{lane}" for lane in range(capacity))
    requests = {
        request_id: RequestState(
            req_id=request_id,
            prompt=[],
            mm_features=[],
            sampling_params=SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
            block_ids=(list(page_rows[lane]),),
            num_computed_tokens=0,
            output=[100 + lane],
        )
        for lane, request_id in enumerate(request_ids)
    }
    owners = {
        request_id: SimpleNamespace(index=lane, generation=lane + 1)
        for lane, request_id in enumerate(request_ids)
    }
    scheduled = SimpleNamespace(
        num_scheduled_tokens={request_id: 1 for request_id in request_ids},
        scheduled_spec_decode_tokens={},
    )
    plan = DSparkBatchPlan.from_scheduler(
        scheduled,
        requests,
        owners,
        max_model_len=max_model_len,
        round_generation=1,
        batch_capacity=capacity,
    )
    return plan, page_rows


@pytest.mark.parametrize(
    ("max_model_len", "expected_width"),
    ((127, 1), (128, 1), (129, 2), (65535, 512), (65536, 512), (65537, 513), (1048576, 8192)),
)
def test_fixed_page_table_width_uses_128_token_scheduler_pages(max_model_len, expected_width):
    assert _fixed_page_table_width(max_model_len) == expected_width


@pytest.mark.parametrize("max_model_len", (65535, 65536, 65537, 1048576))
@pytest.mark.parametrize("ratio", (1, 2))
def test_fixed_page_table_width_covers_native_index_capacity(max_model_len, ratio):
    page_width = _fixed_page_table_width(max_model_len)
    capacity = max_model_len // ratio
    page_rows = 128 // ratio
    assert capacity <= page_width * page_rows


@pytest.mark.parametrize(
    ("capacity", "max_model_len", "expected_width"),
    ((1, 65536, 512), (2, 65536, 512), (4, 65536, 512), (2, 8193, 65)),
)
def test_page_table_uses_fixed_max_context_width_for_short_requests(capacity, max_model_len, expected_width):
    plan, page_ids = _single_page_plan(capacity, max_model_len)

    assert plan.pages.shape == (capacity, expected_width)
    assert bool((plan.pages > 0).all())
    for lane, page_id in enumerate(page_ids):
        assert plan.pages[lane].tolist() == [page_id] * expected_width

    target = TargetBatchContract.from_metadata(
        plan.metadata,
        plan.packed_input_ids,
        plan.pages,
        batch_capacity=capacity,
        device="cpu",
    )
    assert target.row_pages.shape == (capacity * 6, expected_width)
    for lane, page_id in enumerate(page_ids):
        rows = target.row_pages[lane * 6:(lane + 1) * 6]
        assert torch.equal(rows, torch.full_like(rows, page_id))


def test_page_padding_starts_after_real_pages_and_preserves_visible_mapping():
    capacity, page_count, max_model_len, ratio = 2, 2, 65536, 2
    plan, real_pages = _multi_page_plan(capacity, max_model_len, page_count)

    assert plan.pages.shape == (capacity, 512)
    for lane in range(capacity):
        assert plan.pages[lane, :page_count].tolist() == list(real_pages[lane])
        assert plan.pages[lane, page_count:].tolist() == [real_pages[lane][-1]] * (512 - page_count)
        assert bool((plan.pages[lane, page_count:] > 0).all())

    # Two real scheduler pages cover 128 ratio-2 logical rows.  The fixed
    # padding starts after page_count and does not alter any visible mapping
    # inside the real-page prefix.
    real_logical_rows = torch.arange(page_count * (128 // ratio), dtype=torch.int64).reshape(1, -1)
    for lane in range(capacity):
        assert torch.equal(
            batch_physical_rows(real_logical_rows, plan.pages[lane:lane + 1], ratio),
            batch_physical_rows(real_logical_rows, plan.pages[lane:lane + 1, :page_count], ratio),
        )


@pytest.mark.parametrize("capacity", (2, 4))
def test_fixed_context_width_accepts_exactly_512_real_scheduler_pages(capacity):
    plan, real_pages = _multi_page_plan(capacity, 65536, 512)

    assert plan.pages.shape == (capacity, 512)
    for lane in range(capacity):
        assert plan.pages[lane].tolist() == list(real_pages[lane])


@pytest.mark.parametrize("capacity", (2, 4))
def test_fixed_context_width_rejects_513_real_scheduler_pages(capacity):
    with pytest.raises(ValueError, match="DSpark scheduler page count exceeds fixed context width"):
        _multi_page_plan(capacity, 65536, 513)


def test_business_target_beyond_real_scheduler_pages_is_rejected():
    request = RequestState(
        req_id="missing-second-page",
        prompt=[0] * 128,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
        block_ids=((7,),),
        num_computed_tokens=128,
        output=[50],
    )
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"missing-second-page": 1},
        scheduled_spec_decode_tokens={},
    )

    with pytest.raises(ValueError, match="not backed by scheduler pages"):
        DSparkBatchPlan.from_scheduler(
            scheduled,
            {"missing-second-page": request},
            {"missing-second-page": SimpleNamespace(index=0, generation=1)},
            max_model_len=65536,
            round_generation=1,
            batch_capacity=1,
        )


def test_ragged_outputs_rollback_each_request_once_under_real_scheduler(tmp_path):
    scheduler, output, states, owners = _decode_case(
        tmp_path, [[11, 12, 13, 14, 15], [21, 22]]
    )
    plan = _make_plan(output, states, owners)
    assert plan.target_lengths == (6, 3)
    assert [item.placeholder for item in plan.requests] == [False, False]

    before = {req_id: scheduler.requests[req_id].num_computed_tokens for req_id in plan.request_ids}
    result = DSparkDraftBatch(
        accepted=torch.tensor([4, 1], dtype=torch.int32),
        output_token_ids=[[11, 12, 13, 99], [21]],
        draft_token_ids=[[31, 32, 33, 34, 35], [41, 42]],
        completion_event=SimpleNamespace(query=lambda: True),
    )
    model_output, _ = DSparkBatchRunner.publish(plan, result)
    scheduler.update_from_output(output, model_output)

    expected_rejected = [2, 2]
    for req_id, rejected in zip(plan.request_ids, expected_rejected, strict=True):
        request = scheduler.requests[req_id]
        assert request.num_computed_tokens == before[req_id] - rejected
        assert request.num_tokens - request.num_computed_tokens == 1
    assert list(scheduler.requests["a"].output_token_ids) == [101, 11, 12, 13, 99]
    assert list(scheduler.requests["b"].output_token_ids) == [201, 21]


def test_all_placeholder_row_maps_to_c1_lane_and_rolls_back_scheduler_padding(tmp_path):
    scheduler, output, states, owners = _decode_case(
        tmp_path, [[11, 12, 13, 14, 15], []]
    )
    second = next(req_id for req_id in output.num_scheduled_tokens if req_id != "a")
    counts = dict(output.num_scheduled_tokens)
    counts[second] = 6
    specs = dict(output.scheduled_spec_decode_tokens)
    specs[second] = [-1] * 5
    output = replace(
        output,
        num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()),
        scheduled_spec_decode_tokens=specs,
    )
    scheduler.requests[second].num_computed_tokens += 5
    assert output.scheduled_spec_decode_tokens[second] == [-1] * 5
    plan = _make_plan(output, states, owners)
    placeholder = next(item for item in plan.requests if item.req_id == second)
    assert placeholder.placeholder
    assert placeholder.target_query_length == 1
    assert placeholder.scheduler_num_rejected == 5

    before = {req_id: scheduler.requests[req_id].num_computed_tokens for req_id in plan.request_ids}
    result = DSparkDraftBatch(
        accepted=torch.tensor([3, 1], dtype=torch.int32),
        output_token_ids=[[11, 12, 99], [201]],
        draft_token_ids=[[31, 32], []],
        completion_event=SimpleNamespace(query=lambda: True),
    )
    model_output, _ = DSparkBatchRunner.publish(plan, result)
    scheduler.update_from_output(output, model_output)

    for item, rejected in zip(plan.requests, [3, 5], strict=True):
        assert scheduler.requests[item.req_id].num_computed_tokens == before[item.req_id] - rejected


def test_cancelled_request_ignores_late_batch_result(tmp_path):
    scheduler, output, states, owners = _decode_case(
        tmp_path, [[11, 12, 13, 14, 15], [21, 22]]
    )
    plan = _make_plan(output, states, owners)
    result = DSparkDraftBatch(
        accepted=torch.tensor([6, 2], dtype=torch.int32),
        output_token_ids=[[11, 12, 13, 14, 15, 99], [21, 22]],
        draft_token_ids=[[], []],
        completion_event=SimpleNamespace(query=lambda: True),
    )
    model_output, _ = DSparkBatchRunner.publish(plan, result)
    scheduler.finish_requests(["b"], RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(output, model_output)

    assert "b" not in scheduler.requests
    assert "b" not in scheduler.schedule().num_scheduled_tokens


def test_execute_calls_each_packed_stage_once_then_emits_structured_trace(tmp_path, monkeypatch):
    scheduler, output, states, owners = _decode_case(
        tmp_path, [[11, 12, 13, 14, 15], []]
    )
    records = []

    class TargetStage:
        calls = 0

        def forward_batch(self, metadata, input_ids, positions, padding_mask, pages):
            assert metadata.request_ids == ("a", "b")
            assert tuple(input_ids.shape) == (2, 6)
            assert tuple(pages.shape) == (2, 64)
            self.calls += 1
            return DSparkTargetBatch(token_ids=torch.zeros((2, 6), dtype=torch.int32))

    class DraftStage:
        calls = 0

        def verify_and_propose(self, metadata, target):
            self.calls += 1
            return DSparkDraftBatch(
                accepted=torch.tensor([3, 1], dtype=torch.int32),
                output_token_ids=[[11, 12, 99], [201]],
                draft_token_ids=[[31, 32], []],
                completion_event=SimpleNamespace(query=lambda: True),
            )

    class Trace:
        def write(self, payload):
            records.append(payload)

    target, draft = TargetStage(), DraftStage()
    runner = DSparkBatchRunner(target, draft, trace=Trace())
    plan = runner.build_plan(output, states, owners, max_model_len=8192)
    model_output, drafts = runner.execute(plan)

    assert target.calls == 1
    assert draft.calls == 1
    assert model_output.sampled_token_ids == [[11, 12, 99], [201]]
    assert drafts.draft_token_ids == [[31, 32], []]
    assert len(records) == 1
    assert records[0]["req_ids"] == ["a", "b"]
    assert records[0]["actual_packed_target_rows"] == 12
    assert records[0]["actual_packed_draft_rows"] == 10
    assert records[0]["completion_event_observed"] is True


def test_disabled_bug004_observer_keeps_runner_path_and_enabled_writes_round(tmp_path, monkeypatch):
    scheduler, output, states, owners = _decode_case(
        tmp_path / "scheduler", [[11, 12, 13, 14, 15], [21, 22]]
    )
    del scheduler
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("DSV41_RUN_EVIDENCE", str(evidence))
    observed = []

    class TargetStage:
        device = torch.device("cpu")

        def forward_batch(self, metadata, input_ids, positions, padding_mask, pages):
            del metadata, input_ids, positions, padding_mask, pages
            observed.append("target")
            return DSparkTargetBatch(token_ids=torch.zeros((2, 6), dtype=torch.int32))

    class DraftStage:
        _bug004_profile = None

        def verify_and_propose(self, metadata, target):
            del metadata, target
            observed.append("draft")
            return DSparkDraftBatch(
                accepted=torch.tensor([4, 1], dtype=torch.int32),
                output_token_ids=[[11, 12, 13, 99], [21]],
                draft_token_ids=[[31, 32], []],
                completion_event=SimpleNamespace(query=lambda: True),
            )

    monkeypatch.delenv("VLLM_HPU_DSV41_BUG004_PROFILE", raising=False)
    runner = DSparkBatchRunner(TargetStage(), DraftStage(), trace=DSparkBatchTrace(rank=0))
    plan = runner.build_plan(output, states, owners, max_model_len=8192)
    runner.execute(plan)
    assert observed == ["target", "draft"]
    assert not list(evidence.glob("bug004-rounds.*.jsonl"))

    monkeypatch.setenv("VLLM_HPU_DSV41_BUG004_PROFILE", "1")
    runner = DSparkBatchRunner(TargetStage(), DraftStage(), trace=DSparkBatchTrace(rank=0))
    plan = runner.build_plan(output, states, owners, max_model_len=8192)
    runner.execute(plan)
    path = evidence / "bug004-rounds.rank0.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["schema"] == "dsv41.bug004.profile.v3"
    assert records[0]["status"] == "complete"
    assert any(segment["name"] == "target.input_prepare" for segment in records[0]["segments"])


def test_execute_writes_real_jsonl_trace_with_committed_and_output_digests(tmp_path, monkeypatch):
    scheduler, output, states, owners = _decode_case(
        tmp_path / "scheduler", [[11, 12, 13, 14, 15], [21, 22]]
    )
    del scheduler
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("DSV41_RUN_EVIDENCE", str(evidence))

    class TargetStage:
        device = torch.device("cpu")

        def forward_batch(self, metadata, input_ids, positions, padding_mask, pages):
            del metadata, input_ids, positions, padding_mask, pages
            return DSparkTargetBatch(token_ids=torch.zeros((2, 6), dtype=torch.int32))

    class DraftStage:
        def verify_and_propose(self, metadata, target):
            del metadata, target
            return DSparkDraftBatch(
                accepted=torch.tensor([4, 1], dtype=torch.int32),
                output_token_ids=[[11, 12, 13, 99], [21]],
                draft_token_ids=[[31, 32], []],
                completion_event=SimpleNamespace(query=lambda: True),
            )

    runner = DSparkBatchRunner(
        TargetStage(), DraftStage(), trace=DSparkBatchTrace(rank=0, tp_rank=0, pp_rank=0)
    )
    plan = runner.build_plan(output, states, owners, max_model_len=8192)
    runner.execute(plan)
    records = [json.loads(line) for line in (evidence / "dspark_batch_rounds.rank0.jsonl").read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["committed_counts"] == [4, 1]
    assert record["output_counts"] == [4, 1]
    assert len(record["output_ids_digest"]) == 64
    assert len(record["consistency_digest"]) == 64


def test_trace_digest_ignores_rank_and_timing_noise(tmp_path, monkeypatch):
    digests = []
    for rank in range(8):
        directory = tmp_path / f"rank{rank}"
        monkeypatch.setenv("DSV41_RUN_EVIDENCE", str(directory))
        trace = DSparkBatchTrace(rank=rank, tp_rank=rank, pp_rank=0)
        trace.write({
            "schema": "dsv41.dspark.round.v2",
            "batch_round": 4,
            "batch_capacity": 2,
            "capacity": 2,
            "active_count": 2,
            "active_lanes": [0, 1],
            "lanes": [0, 1],
            "active_mask": [True, True],
            "lane_request_ids": ["a", "b"],
            "req_ids": ["a", "b"],
            "request_slots": [0, 1],
            "slot_generations": [7, 9],
            "logical_target_queries": [6, 1],
            "draft_counts": [5, 0],
            "scheduler_draft_counts": [5, 5],
            "physical_target_rows": 12,
            "physical_draft_rows": 10,
            "actual_packed_target_rows": 12,
            "actual_packed_draft_rows": 10,
            "batch_width": 2,
            "committed_counts": [3, 1],
            "output_counts": [3, 1],
            "next_draft_counts": [2, 0],
            "metadata_digest": "m" * 64,
            "output_ids_digest": "o" * 64,
            "next_draft_digest": "d" * 64,
            "completion_event_observed": True,
            "completion_observed_ns": 100 + rank,
            "final_event_observed": True,
            "final_event_observed_ns": 200 + rank,
            "target_stage": "target",
            "draft_stage": "draft",
        })
        record = json.loads((directory / f"dspark_batch_rounds.rank{rank}.jsonl").read_text().splitlines()[0])
        digests.append(record["consistency_digest"])
    assert len(set(digests)) == 1


def test_execute_fails_closed_without_actual_target_or_draft_stage():
    with pytest.raises(RuntimeError, match="actual packed target and draft stages"):
        DSparkBatchRunner(None, object())
    with pytest.raises(RuntimeError, match="actual packed target and draft stages"):
        DSparkBatchRunner(object(), None)


class _DraftState:
    def __init__(self, capacity=2):
        self.rings = [torch.zeros((capacity * 256, 1), dtype=torch.float32) for _ in range(3)]

    def clear_slot(self, slot):
        for ring in self.rings:
            ring[slot * 256:(slot + 1) * 256].zero_()


class _DraftLayer:
    def __init__(self):
        self.attention = SimpleNamespace(swa=torch.zeros((256, 1), dtype=torch.float32))


class _TargetBank:
    def __init__(self):
        self.single_owner = None
        self.prefill_owner = None
        self.enter_calls = 0

    def leave_single(self):
        self.single_owner = None
        self.prefill_owner = None

    def restore_single_bindings(self):
        return None

    def _invalidate_index_mirror(self):
        return None


class _TargetStage:
    def __init__(self):
        self.batch_state = _TargetBank()

    def enter_batched(self):
        self.batch_state.enter_calls += 1
        self.batch_state.leave_single()


def _draft_owner_runner():
    state = _DraftState()
    draft_model = SimpleNamespace(layers=[_DraftLayer() for _ in range(3)])

    class DraftStage:
        batch_state = state
        draft = draft_model

    return DSparkBatchRunner(_TargetStage(), DraftStage(), trace=SimpleNamespace(write=lambda payload: None))


def _fill_working(runner, value):
    for layer in runner.draft_stage.draft.layers:
        layer.attention.swa.fill_(value)


def test_draft_owner_repeat_bind_preserves_new_writes():
    slots = RequestSlots(2)
    runner = _draft_owner_runner()
    runner.slots = slots
    owner = slots.acquire("a")
    runner.bind_single(owner)
    _fill_working(runner, 3)
    runner.bind_single(owner)
    assert all(layer.attention.swa.eq(3).all() for layer in runner.draft_stage.draft.layers)


def test_draft_owner_a_to_b_to_a_roundtrip():
    slots = RequestSlots(2)
    runner = _draft_owner_runner()
    runner.slots = slots
    a, b = slots.acquire("a"), slots.acquire("b")
    runner.bind_single(a)
    _fill_working(runner, 1)
    runner.bind_single(b)
    assert all(layer.attention.swa.eq(0).all() for layer in runner.draft_stage.draft.layers)
    _fill_working(runner, 2)
    runner.bind_single(a)
    assert all(layer.attention.swa.eq(1).all() for layer in runner.draft_stage.draft.layers)
    assert all(ring[b.index * 256:(b.index + 1) * 256].eq(2).all() for ring in runner.draft_stage.batch_state.rings)


def test_draft_owner_prefill_a_to_b_keeps_slots_isolated():
    slots = RequestSlots(2)
    runner = _draft_owner_runner()
    runner.slots = slots
    a, b = slots.acquire("a"), slots.acquire("b")
    runner.bind_single(a)
    _fill_working(runner, 4)
    runner.bind_single(b)
    assert all(layer.attention.swa.eq(0).all() for layer in runner.draft_stage.draft.layers)
    assert all(ring[a.index * 256:(a.index + 1) * 256].eq(4).all() for ring in runner.draft_stage.batch_state.rings)


def test_draft_owner_b1_to_b2_to_b1_roundtrip():
    slots = RequestSlots(2)
    runner = _draft_owner_runner()
    runner.slots = slots
    a, b = slots.acquire("a"), slots.acquire("b")
    runner.bind_single(a)
    _fill_working(runner, 5)
    runner.target_stage.batch_state.single_owner = a
    runner._leave_single()
    assert runner.single_draft_owner is None
    assert runner.target_stage.batch_state.enter_calls == 1
    assert all(ring[a.index * 256:(a.index + 1) * 256].eq(5).all() for ring in runner.draft_stage.batch_state.rings)

    for ring in runner.draft_stage.batch_state.rings:
        ring[a.index * 256:(a.index + 1) * 256].fill_(9)
        ring[b.index * 256:(b.index + 1) * 256].fill_(8)
    runner.draft_bank_generations[a.index] = a.generation
    runner.draft_bank_generations[b.index] = b.generation
    runner.bind_single(a)
    assert all(layer.attention.swa.eq(9).all() for layer in runner.draft_stage.draft.layers)
    runner.bind_single(b)
    assert all(layer.attention.swa.eq(8).all() for layer in runner.draft_stage.draft.layers)


def test_draft_owner_cancel_then_reused_slot_does_not_save_stale_working_state():
    slots = RequestSlots(2)
    runner = _draft_owner_runner()
    runner.slots = slots
    first = slots.acquire("a")
    runner.bind_single(first)
    _fill_working(runner, 7)
    runner.release("a")
    slots.release(first)
    reused = slots.acquire("b")
    assert reused.index == first.index and reused.generation == first.generation + 1
    runner.bind_single(reused)
    assert all(layer.attention.swa.eq(0).all() for layer in runner.draft_stage.draft.layers)
    assert all(ring[reused.index * 256:(reused.index + 1) * 256].eq(0).all()
               for ring in runner.draft_stage.batch_state.rings)


def test_mixed_engram_rows_scatter_to_fixed_physical_request_regions():
    class EngramOwner:
        def prepare_batch(self, spans, *, capacity, defer_wait):
            assert capacity == 24 and defer_wait
            assert [span[0] for span in spans] == ["b", "c"]
            assert [len(span[2]) for span in spans] == [5, 2]
            return object()

        def wait(self, ticket):
            del ticket
            layer = torch.arange(24, dtype=torch.float32).reshape(24, 1)
            return (layer, layer.clone())

    runner = object.__new__(DSparkPackedTargetStages)
    runner.engram_host = EngramOwner()
    runner.device = torch.device("cpu")
    batch = SimpleNamespace(
        request_ids=(None, "b", "c", None),
        host_request_active=(False, True, True, False),
        request_lengths=(0, 5, 2, 0),
        host_positions=((0, ) * 6 + (10, 11, 12, 13, 14, 0) + (20, 21, 0, 0, 0, 0) + (0, ) * 6),
        real_indices=torch.tensor([6, 7, 8, 9, 10, 12, 13], dtype=torch.int64),
        target_rows=24,
    )
    input_ids = torch.tensor(
        [[0, 0, 0, 0, 0, 0], [10, 11, 12, 13, 14, 0], [20, 21, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        dtype=torch.int64,
    )
    rows, _ = runner._engram_rows(batch, input_ids)
    expected = [0.0] * 24
    for physical, packed in zip((6, 7, 8, 9, 10, 12, 13), range(7), strict=True):
        expected[physical] = packed
    assert rows[0].reshape(-1).tolist() == expected


def _one_active_metadata(capacity=2, *, active_lane=1, target_length=5):
    active = [False] * capacity
    active[active_lane] = True
    lengths = [0] * capacity
    lengths[active_lane] = target_length
    drafts = [[] for _ in range(capacity)]
    drafts[active_lane] = list(range(10, 10 + target_length - 1))
    slots = [-1] * capacity
    slots[active_lane] = active_lane
    generations = [0] * capacity
    generations[active_lane] = 3
    budgets = [0] * capacity
    budgets[active_lane] = 32
    starts = [0] * capacity
    starts[active_lane] = 4
    requests = [None] * capacity
    requests[active_lane] = "active"
    return DSparkBatchMetadata.build(
        requests=requests,
        target_query_lengths=lengths,
        draft_token_ids=drafts,
        request_slots=slots,
        slot_generations=generations,
        remaining_output_budget=budgets,
        target_padding_mask=[False] * target_length,
        target_start_positions=starts,
        round_generation=5,
        active=active,
    )


class _RollbackTicket:
    def __init__(self):
        self.restore_calls = []

    def restore(self, committed=None):
        self.restore_calls.append(tuple(committed))


class _RollbackEngram:
    def __init__(self, *, fail_wait_once=False):
        self.pending = None
        self.ready_ticket = None
        self.prepared = []
        self.committed = []
        self.wait_calls = 0
        self.fail_wait_once = fail_wait_once

    def prepare_batch(self, spans, *, capacity, defer_wait):
        assert capacity == 12 and defer_wait
        self.prepared.append(tuple(spans))
        ticket = object()
        self.pending = ticket
        return ticket

    def wait(self, ticket):
        assert self.pending is ticket
        self.wait_calls += 1
        if self.fail_wait_once:
            self.fail_wait_once = False
            raise RuntimeError("engram wait failed")
        self.ready_ticket = ticket
        return (torch.arange(12, dtype=torch.float32).reshape(12, 1), )

    def complete_batch(self, ticket, committed):
        assert self.pending is ticket and self.ready_ticket is ticket
        self.committed.append(tuple(committed))
        self.pending = self.ready_ticket = None


def test_target_forward_failure_rolls_back_owned_target_and_engram_tickets():
    metadata = _one_active_metadata()
    ticket = _RollbackTicket()
    engram = _RollbackEngram()
    observed_error = RuntimeError("target body failed")

    class FailingStage:
        def snapshot(self, batch):
            assert batch.batch_capacity == 2
            return ticket

        def __call__(self, *args):
            raise observed_error

    runner = object.__new__(DSparkPackedTargetStages)
    runner.batch_capacity = 2
    runner.stage_key = (0, 1)
    runner.stage = FailingStage()
    runner.device = torch.device("cpu")
    runner.engram_host = engram
    runner.hidden_seed = torch.zeros((12, 4, 1), dtype=torch.bfloat16)
    runner.pre_mix_seed = torch.zeros((12, 4), dtype=torch.float32)
    input_ids = torch.zeros((2, 6), dtype=torch.int64)
    input_ids[1, :5] = torch.tensor([1, 10, 11, 12, 13])
    positions = torch.zeros((2, 6), dtype=torch.int32)
    padding = torch.ones((2, 6), dtype=torch.bool)
    padding[1, :5] = False
    pages = torch.tensor([[1], [2]], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="target body failed") as error:
        runner.forward_batch(metadata, input_ids, positions, padding, pages)

    assert error.value is observed_error
    assert ticket.restore_calls == [(0, 0)]
    assert engram.committed == [(0, )]
    assert engram.pending is None


def test_engram_wait_failure_retires_ticket_and_preserves_original_error():
    runner = object.__new__(DSparkPackedTargetStages)
    runner.engram_host = _RollbackEngram(fail_wait_once=True)
    runner.device = torch.device("cpu")
    batch = SimpleNamespace(
        request_ids=(None, "active"),
        host_request_active=(False, True),
        request_lengths=(0, 5),
        host_positions=((0, ) * 6 + (4, 5, 6, 7, 8, 0)),
        real_indices=torch.tensor([6, 7, 8, 9, 10], dtype=torch.int64),
        target_rows=12,
    )
    input_ids = torch.zeros((2, 6), dtype=torch.int64)
    input_ids[1, :5] = torch.tensor([1, 10, 11, 12, 13])

    with pytest.raises(RuntimeError, match="engram wait failed"):
        runner._engram_rows(batch, input_ids)

    assert runner.engram_host.wait_calls == 2
    assert runner.engram_host.committed == [(0, )]
    assert runner.engram_host.pending is None


def test_n4_sparse_engram_completion_gathers_only_active_accepted_lanes():
    class Engram:
        max_tokens = 24

        def __init__(self):
            self.pending = self.ready_ticket = None
            self.spans = None
            self.committed = None
            self.ticket = SimpleNamespace(
                batch=SimpleNamespace(
                    capacity=24,
                    members=("b", "c"),
                    hash_ids=torch.zeros((7, 2, 2), dtype=torch.int32),
                ),
            )

        def prepare_batch(self, spans, *, capacity, defer_wait):
            assert capacity == 24 and defer_wait
            self.spans = tuple(spans)
            self.pending = self.ticket
            return self.ticket

        def wait(self, ticket):
            assert ticket is self.pending
            self.ready_ticket = ticket
            return (torch.arange(24, dtype=torch.float32).reshape(24, 1), )

        def complete_batch(self, ticket, committed):
            assert ticket is self.pending and ticket is self.ready_ticket
            self.committed = tuple(committed)
            self.pending = self.ready_ticket = None

    engram = Engram()
    runner = object.__new__(DSparkPackedTargetStages)
    runner.engram_host = engram
    runner.device = torch.device("cpu")
    positions = (0, ) * 6 + tuple(range(10, 16)) + (0, ) * 6 + tuple(range(20, 26))
    batch = SimpleNamespace(
        request_ids=(None, "b", None, "c"),
        host_request_active=(False, True, False, True),
        request_lengths=(0, 1, 0, 6),
        host_positions=positions,
        real_indices=torch.tensor([6, 18, 19, 20, 21, 22, 23], dtype=torch.int64),
        target_rows=24,
    )
    input_ids = torch.zeros((4, 6), dtype=torch.int64)
    input_ids[1, 0] = 1
    input_ids[3] = torch.tensor([2, 10, 11, 12, 13, 14])

    rows, owner_info = runner._engram_rows(batch, input_ids)
    assert tuple(rows[0].shape) == (24, 1)
    target_ticket = _RollbackTicket()
    target = DSparkTargetBatch(
        token_ids=torch.zeros((4, 6), dtype=torch.int32),
        payload={"batch": batch, "ticket": target_ticket, "engram_owner": owner_info, "finished": False},
    )
    runner.batch_capacity = 4
    runner.finish(target, [0, 3, 0, 2])

    assert target_ticket.restore_calls == [(0, 3, 0, 2)]
    assert engram.committed == (3, 2)
    assert [span[0] for span in engram.spans] == ["b", "c"]


def test_engram_rejects_ticket_geometry_before_wait_and_retires_it():
    class Engram:
        def __init__(self):
            self.pending = self.ready_ticket = None
            self.committed = []
            self.ticket = SimpleNamespace(
                batch=SimpleNamespace(
                    capacity=24,
                    members=("b", "c"),
                    hash_ids=torch.zeros((6, 2, 2), dtype=torch.int32),
                ),
            )

        def prepare_batch(self, spans, *, capacity, defer_wait):
            del spans, capacity, defer_wait
            self.pending = self.ticket
            return self.ticket

        def wait(self, ticket):
            assert ticket is self.pending
            self.ready_ticket = ticket
            return (torch.zeros((24, 1)), )

        def complete_batch(self, ticket, committed):
            assert ticket is self.pending and ticket is self.ready_ticket
            self.committed.append(tuple(committed))
            self.pending = self.ready_ticket = None

    runner = object.__new__(DSparkPackedTargetStages)
    runner.engram_host = Engram()
    runner.device = torch.device("cpu")
    batch = SimpleNamespace(
        request_ids=(None, "b", None, "c"),
        host_request_active=(False, True, False, True),
        request_lengths=(0, 1, 0, 6),
        host_positions=(0, ) * 6 + (10, ) * 6 + (0, ) * 6 + (20, ) * 6,
        real_indices=torch.arange(6, 13, dtype=torch.int64),
        target_rows=24,
    )
    input_ids = torch.zeros((4, 6), dtype=torch.int64)

    with pytest.raises(ValueError, match="hash rows differ"):
        runner._engram_rows(batch, input_ids)

    assert runner.engram_host.committed == [(0, 0)]
    assert runner.engram_host.pending is None


def test_malformed_completion_event_aborts_before_target_commit_or_publication(tmp_path):
    scheduler, output, states, owners = _decode_case(tmp_path, [[11, 12, 13, 14, 15], [21, 22]])
    del scheduler
    events = []

    class TargetStage:
        device = torch.device("cpu")

        def forward_batch(self, metadata, input_ids, positions, padding_mask, pages):
            del metadata, input_ids, positions, padding_mask, pages
            return DSparkTargetBatch(token_ids=torch.zeros((2, 6), dtype=torch.int32), payload={})

        def finish(self, target, accepted):
            del target, accepted
            events.append("finish")

        def abort(self, target):
            del target
            events.append("abort")

    class DraftStage:
        def verify_and_propose(self, metadata, target):
            del metadata, target
            return DSparkDraftBatch(
                accepted=torch.tensor([1, 1], dtype=torch.int32),
                output_token_ids=[[11], [21]],
                draft_token_ids=[[], []],
                completion_event=None,
            )

    runner = DSparkBatchRunner(TargetStage(), DraftStage(), trace=SimpleNamespace(write=lambda payload: None))
    plan = runner.build_plan(output, states, owners, max_model_len=8192)
    before = tuple(states[req_id].output for req_id in plan.request_ids)

    with pytest.raises(RuntimeError, match="actual completion event"):
        runner.execute(plan)

    assert events == ["abort"]
    assert tuple(states[req_id].output for req_id in plan.request_ids) == before


@pytest.mark.parametrize("constructor_name", ("ModelRunnerOutput", "DraftTokenIds"))
def test_publication_constructor_failure_aborts_before_state_or_request_mutation(tmp_path, monkeypatch, constructor_name):
    import vllm_gaudi.v1.worker.deepseek_v41_dspark_batch_runner as adapter

    scheduler, output, states, owners = _decode_case(tmp_path, [[11, 12, 13, 14, 15], [21, 22]])
    del scheduler
    events = []

    class TargetStage:
        device = torch.device("cpu")

        def forward_batch(self, metadata, input_ids, positions, padding_mask, pages):
            del metadata, input_ids, positions, padding_mask, pages
            return DSparkTargetBatch(token_ids=torch.zeros((2, 6), dtype=torch.int32), payload={})

        def finish(self, target, accepted):
            del target, accepted
            events.append("finish")

        def abort(self, target):
            del target
            events.append("abort")

    class DraftStage:
        def verify_and_propose(self, metadata, target):
            del metadata, target
            return DSparkDraftBatch(
                accepted=torch.tensor([1, 1], dtype=torch.int32),
                output_token_ids=[[11], [21]],
                draft_token_ids=[[], []],
                completion_event=SimpleNamespace(query=lambda: True),
            )

    def fail_constructor(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(f"{constructor_name} constructor failed")

    monkeypatch.setattr(adapter, constructor_name, fail_constructor)
    runner = DSparkBatchRunner(TargetStage(), DraftStage(), trace=SimpleNamespace(write=lambda payload: None))
    plan = runner.build_plan(output, states, owners, max_model_len=8192)
    before = tuple(states[req_id].output for req_id in plan.request_ids)

    with pytest.raises(RuntimeError, match=f"{constructor_name} constructor failed"):
        runner.execute(plan)

    assert events == ["abort"]
    assert tuple(states[req_id].output for req_id in plan.request_ids) == before


def test_trace_io_failure_after_device_commit_is_nonfatal_and_exposed(tmp_path, caplog):
    scheduler, output, states, owners = _decode_case(tmp_path, [[11, 12, 13, 14, 15], [21, 22]])
    del scheduler
    events = []

    class TargetStage:
        device = torch.device("cpu")

        def forward_batch(self, metadata, input_ids, positions, padding_mask, pages):
            del metadata, input_ids, positions, padding_mask, pages
            return DSparkTargetBatch(token_ids=torch.zeros((2, 6), dtype=torch.int32), payload={})

        def finish(self, target, accepted):
            del target, accepted
            events.append("finish")

        def abort(self, target):
            del target
            events.append("abort")

    class DraftStage:
        def verify_and_propose(self, metadata, target):
            del metadata, target
            return DSparkDraftBatch(
                accepted=torch.tensor([2, 1], dtype=torch.int32),
                output_token_ids=[[11, 12], [21]],
                draft_token_ids=[[], []],
                completion_event=SimpleNamespace(query=lambda: True),
            )

    class FailingTrace:
        def write(self, payload):
            del payload
            raise OSError("evidence disk unavailable")

    runner = DSparkBatchRunner(TargetStage(), DraftStage(), trace=FailingTrace())
    plan = runner.build_plan(output, states, owners, max_model_len=8192)
    model_output, drafts = runner.execute(plan)

    assert events == ["finish"]
    assert model_output.sampled_token_ids == [[11, 12], [21]]
    assert drafts.draft_token_ids == [[], []]
    assert states["a"].output[-2:] == [11, 12]
    assert states["b"].output[-1] == 21
    assert runner.trace_failures == 1
    assert runner.trace_available is False
    assert "trace evidence unavailable" in caplog.text


def test_malformed_inactive_publication_leaves_active_request_unchanged():
    slots = RequestSlots(2)
    slots.acquire("hold")
    owner = slots.acquire("active")
    request = RequestState(
        req_id="active",
        prompt=[],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
        block_ids=((1, ), ),
        num_computed_tokens=0,
        output=[50],
    )
    runner = DSparkBatchRunner(
        object(),
        object(),
        slots=slots,
        batch_capacity=2,
        trace=SimpleNamespace(write=lambda payload: None),
    )
    scheduled = SimpleNamespace(num_scheduled_tokens={"active": 1}, scheduled_spec_decode_tokens={})
    plan = runner.build_plan(scheduled, {"active": request}, {"active": owner}, max_model_len=8192)
    result = DSparkDraftBatch(
        accepted=torch.tensor([0, 1], dtype=torch.int32),
        output_token_ids=[[99], [77]],
        draft_token_ids=[[], []],
        completion_event=SimpleNamespace(query=lambda: True),
    )

    with pytest.raises(ValueError, match="Inactive DSpark lane 0|trailing -1 padding"):
        runner.publish(plan, result)

    assert request.output == [50]


def test_target_greedy_ids_bind_real_prepared_stage_and_exchange_only_pairs(monkeypatch):
    import vllm_gaudi.models.deepseek_v41_program as program_module

    monkeypatch.setattr(program_module.gaudi_envs, "VLLM_HPU_DSV41_DSPARK_VOCAB_HEAD_FP8", False)
    monkeypatch.setattr(program_module.gaudi_envs, "VLLM_HPU_DSV41_DSPARK_BF16_PROJECTIONS", False)
    assert hasattr(PreparedStage, "_head_projection")
    assert "_global_argmax" not in PreparedStage.__dict__

    torch.manual_seed(17)
    rows_per_lane = 6
    lane_count = 2
    hidden = torch.randn(12, 6)
    full_weight = torch.randn(8, 6)
    shard_width = full_weight.shape[0] // 2

    # The reference must launch the same serial C6 math once per request lane;
    # a single 12-row projection is precisely the behaviour under repair.
    lane_logits = []
    lane_oracles = []
    expected_pairs = []
    for lane in range(lane_count):
        start = lane * rows_per_lane
        logits = hidden[start:start + rows_per_lane].float() @ full_weight.T
        lane_logits.append(logits)
        lane_oracles.append(logits.argmax(dim=-1))
        lane_pairs = []
        for rank in range(2):
            local = logits[:, rank * shard_width:(rank + 1) * shard_width]
            values, indices = local.max(dim=-1)
            lane_pairs.append(torch.stack(
                (values.float(), (indices + rank * shard_width).float()), dim=-1))
        expected_pairs.append(lane_pairs)
    oracle = torch.cat(lane_oracles, dim=0)

    observed = []
    observed_heads = []
    observed_gathers = []
    for rank in range(2):
        head_calls = []
        gather_calls = []

        def all_gather(pair, dim, *, _rank=rank):
            lane = len(gather_calls)
            assert dim == -1
            assert pair.shape == (rows_per_lane, 2)
            assert pair.dtype == torch.float32
            torch.testing.assert_close(pair, expected_pairs[lane][_rank])
            gather_calls.append(tuple(pair.shape))
            return torch.cat(expected_pairs[lane], dim=-1)

        stage = PreparedStage.__new__(PreparedStage)
        torch.nn.Module.__init__(stage)
        stage.weights = torch.nn.Module()
        stage.weights.head = torch.nn.Module()
        stage.weights.head.weight = torch.nn.Parameter(
            full_weight[rank * shard_width:(rank + 1) * shard_width].clone(), requires_grad=False)
        stage.weights.head.dspark_vocab_fp8_weight = None
        stage.weights.head.dspark_vocab_fp8_scale = None
        stage.bf16_head = False
        stage.tp_rank = rank
        stage.all_gather = all_gather
        real_head_projection = stage._head_projection

        def head_projection(value, *, _rank=rank, _head_projection=real_head_projection):
            lane = len(head_calls)
            start = lane * rows_per_lane
            torch.testing.assert_close(value, hidden[start:start + rows_per_lane])
            projected = _head_projection(value)
            torch.testing.assert_close(
                projected,
                lane_logits[lane][:, _rank * shard_width:(_rank + 1) * shard_width],
            )
            head_calls.append(tuple(value.shape))
            return projected

        stage._head_projection = head_projection

        actual = _target_greedy_ids(stage, hidden)
        assert actual.shape == (lane_count * rows_per_lane, )
        for lane, expected in enumerate(lane_oracles):
            start = lane * rows_per_lane
            assert torch.equal(actual[start:start + rows_per_lane], expected)
        observed.append(tuple(actual.shape))
        observed_heads.append(head_calls)
        observed_gathers.append(gather_calls)

    assert observed == [(12, ), (12, )]
    assert observed_heads == [[(6, 6), (6, 6)], [(6, 6), (6, 6)]]
    assert observed_gathers == [[(6, 2), (6, 2)], [(6, 2), (6, 2)]]


def test_target_greedy_ids_keep_cross_lane_sentinel_local():
    rows_per_lane = 6
    hidden = torch.tensor(
        [[100.0, -100.0]]
        + [[0.0, 1.0]] * (rows_per_lane - 1)
        + [[-100.0, 100.0]] * rows_per_lane,
        dtype=torch.float32,
    )
    head_calls = []
    gather_calls = []

    def head_projection(value):
        head_calls.append(value.clone())
        return value

    def all_gather(pair, dim):
        assert dim == -1
        gather_calls.append(tuple(pair.shape))
        return pair

    program = SimpleNamespace(
        _head_projection=head_projection,
        tp_rank=0,
        all_gather=all_gather,
    )
    actual = _target_greedy_ids(program, hidden)

    assert actual.tolist() == [0] + [1] * (2 * rows_per_lane - 1)
    assert len(head_calls) == 2
    assert torch.equal(head_calls[0], hidden[:rows_per_lane])
    assert torch.equal(head_calls[1], hidden[rows_per_lane:])
    assert gather_calls == [(rows_per_lane, 2), (rows_per_lane, 2)]


def test_target_greedy_ids_reject_non_c6_hidden_rows():
    with pytest.raises(ValueError, match="positive multiple of 6"):
        _target_greedy_ids(SimpleNamespace(), torch.empty(7, 6))


@pytest.mark.parametrize("capacity", (3, 6, 8, 11, 16, 64))
def test_packed_target_stages_create_one_capacity_sized_key(monkeypatch, capacity):
    import vllm_gaudi.models.deepseek_v41_batch_target_program as target_program

    created = []

    class FakeTargetStage:
        def __init__(self, program, request_lengths, request_slots, *, backend):
            self.program = program
            self.request_lengths = request_lengths
            self.request_slots = request_slots
            self.backend = backend
            self.compiled = object()
            created.append(self)

    monkeypatch.setattr(target_program, "DSparkTargetStage", FakeTargetStage)
    program = SimpleNamespace(config={"text_config": {"hidden_size": 16}})
    state = SimpleNamespace(capacity=capacity)
    stages = DSparkPackedTargetStages(program, state, device="cpu", batch_capacity=capacity)
    stage_key = tuple(range(capacity))
    rows = capacity * 6

    assert len(created) == 1
    assert stages.stage_key == stage_key
    assert stages.stages == {stage_key: stages.stage}
    assert created[0].request_lengths == (6, ) * capacity
    assert created[0].request_slots == stage_key
    assert tuple(stages.hidden_seed.shape) == (rows, 4, 16)
    assert tuple(stages.pre_mix_seed.shape) == (rows, 4)

    with pytest.raises(ValueError, match=f"staging capacity {rows - 1} is smaller than fixed target rows {rows}"):
        DSparkPackedTargetStages(
            program,
            state,
            device="cpu",
            engram_host=SimpleNamespace(max_tokens=rows - 1),
            batch_capacity=capacity,
        )


def _dspark_b2_config(*, max_num_seqs=2, tp=8, pp=1, prefix=False, async_scheduling=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=pp),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs, async_scheduling=async_scheduling),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix),
    )


@pytest.mark.parametrize("max_num_seqs", (2, 3, 6, 8, 16, 64))
def test_concurrency_profile_allows_contract_capacity(monkeypatch, max_num_seqs):
    from vllm_gaudi.entrypoints.deepseek_v41 import _DSPARK_TP8_DEFAULTS

    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    for name in _DSPARK_TP8_DEFAULTS:
        monkeypatch.setenv(name, "0")
    config = _dspark_b2_config(max_num_seqs=max_num_seqs)
    assert is_dspark_b2_tp8_profile(config) is (max_num_seqs == 2)
    validate_dspark_tp8_profile(config)


@pytest.mark.parametrize(
    "overrides",
    (
        {"max_num_seqs": 65},
        {"prefix": True},
        {"async_scheduling": True},
        {"tp": 4},
        {"pp": 2},
    ),
)
def test_concurrency_gate_rejects_unsupported_profiles(monkeypatch, overrides):
    from vllm_gaudi.entrypoints.deepseek_v41 import _DSPARK_TP8_DEFAULTS

    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    for name in _DSPARK_TP8_DEFAULTS:
        monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError, match="DSpark concurrency|max_num_seqs"):
        validate_dspark_tp8_profile(_dspark_b2_config(**overrides))


@pytest.mark.parametrize(("tp", "pp"), ((2, 2), (4, 1), (8, 1)))
def test_single_request_dspark_profiles_keep_their_existing_semantics(monkeypatch, tp, pp):
    from vllm_gaudi.entrypoints.deepseek_v41 import _DSPARK_TP8_DEFAULTS

    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    for name in _DSPARK_TP8_DEFAULTS:
        monkeypatch.setenv(name, "0")
    validate_dspark_tp8_profile(_dspark_b2_config(max_num_seqs=1, tp=tp, pp=pp))


def test_b2_sampling_admission_rejects_default_temperature_before_worker(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_DEVICE_VERIFY", "1")
    greedy = SamplingParams(temperature=0, max_tokens=8)
    sampled = SamplingParams(temperature=1, max_tokens=8)
    validate_sampling(greedy, batch=True)
    with pytest.raises(VLLMValidationError, match="B2 is greedy-only"):
        validate_sampling(sampled, batch=True)
    # B1 keeps its existing device-verify-gated sampled baseline.
    validate_sampling(sampled, batch=False)


def test_b2_profile_helper_requires_greedy_sync_pp1_prefix_off(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_DSPARK", "1")
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
        parallel_config=SimpleNamespace(tensor_parallel_size=8, pipeline_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=2, async_scheduling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    assert is_dspark_b2_tp8_profile(config)
    config.cache_config.enable_prefix_caching = True
    assert not is_dspark_b2_tp8_profile(config)


def test_model_wrapper_initializes_actual_batch_stage_provider(monkeypatch):
    target, draft = object(), object()
    from vllm_gaudi.ops import deepseek_v41_batch_draft, deepseek_v41_batch_verify
    from vllm_gaudi.v1.worker import deepseek_v41_dspark_batch_runner as adapter

    monkeypatch.setattr(adapter, "DSparkPackedTargetStages", lambda program, state, **kwargs: target)
    monkeypatch.setattr(deepseek_v41_batch_draft, "DraftBatchState", lambda draft, capacity: object())
    monkeypatch.setattr(
        deepseek_v41_batch_verify,
        "BatchVerifyRing",
        lambda device, batch_capacity, owner_validator: object(),
    )
    monkeypatch.setattr(deepseek_v41_batch_draft, "DSparkBatchDraftStage", lambda *args, **kwargs: draft)
    model = object.__new__(HpuDeepseekV41ForCausalLM)
    model.native = True
    object.__setattr__(model, "device", "cpu")
    model.max_model_len = 8192
    object.__setattr__(model, "engram_host", None)
    model.program = SimpleNamespace(dspark=True, draft=object())
    initialized = []
    def initialize_state(capacity):
        initialized.append(capacity)
        slots = SimpleNamespace(
            capacity=capacity,
            owners={},
            validate_batch_identities=lambda *args: True,
        )
        model.batch_state = SimpleNamespace(
            capacity=capacity,
            pending=None,
            single_owner=None,
            prefill_owner=None,
            slots=slots,
        )

    model.initialize_request_state = initialize_state
    assert model.initialize_dspark_batch(2) == (target, draft)
    assert initialized == [2]
    assert model.program.batch_state is model.batch_state

    model.native = False
    with pytest.raises(RuntimeError, match="requires native target replay"):
        model.initialize_dspark_batch(2)


def test_ordinary_batch_replay_still_rejects_dspark():
    model = object.__new__(HpuDeepseekV41ForCausalLM)
    model.program = SimpleNamespace(dspark=True)
    with pytest.raises(ValueError, match="BatchStageReplay does not support DSpark"):
        model.initialize_request_batches(2)


def test_dspark_request_state_uses_the_bounded_capacity_range(monkeypatch):
    from vllm_gaudi.ops import deepseek_v41_batch_state

    monkeypatch.setattr(deepseek_v41_batch_state, "BatchStageState", lambda program, capacity: (program, capacity))
    model = object.__new__(HpuDeepseekV41ForCausalLM)
    model.native = True
    model.program = SimpleNamespace(dspark=True)
    for capacity in (1, 2, 3, 6, 16, 64):
        model.initialize_request_state(capacity)
        assert model.batch_state == (model.program, capacity)
    with pytest.raises(ValueError, match="1\\.\\.64"):
        model.initialize_request_state(65)
