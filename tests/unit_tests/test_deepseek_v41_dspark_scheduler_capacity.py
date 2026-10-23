# SPDX-License-Identifier: Apache-2.0
"""Real-scheduler CPU contracts for fixed DSpark batch capacities."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers import OPTConfig
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_gaudi.ops.deepseek_v41_batch import RequestSlots
from vllm_gaudi.v1.worker.deepseek_v41_dspark_batch_runner import (
    DSparkBatchRunner,
    DSparkDraftBatch,
    DSparkTargetBatch,
)
from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState


EOS_TOKEN_ID = 50256


def _scheduler(tmp_path, capacity):
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
            max_num_seqs=capacity,
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
    states = {}
    for req_id in output.num_scheduled_tokens:
        request = scheduler.requests[req_id]
        states[req_id] = RequestState(
            req_id=req_id,
            prompt=list(request.prompt_token_ids),
            mm_features=[],
            sampling_params=request.sampling_params,
            block_ids=scheduler.kv_cache_manager.get_block_ids(req_id),
            num_computed_tokens=starts[req_id],
            output=list(request.output_token_ids),
        )
    return states


def _decode_output(scheduler, req_ids, anchors):
    requests = [_request(req_id, [index + 1, index + 2]) for index, req_id in enumerate(req_ids)]
    _prefill(scheduler, requests, anchors)
    drafts = [[index * 10 + 1, index * 10 + 2] for index in range(len(req_ids))]
    scheduler.update_draft_token_ids(DraftTokenIds(req_ids, drafts))
    for request in scheduler.running:
        request.is_prefill_chunk = False
    output = scheduler.schedule()
    states = _runner_states(scheduler, output)
    return output, states


def _reverse_scheduled(output):
    counts = dict(reversed(list(output.num_scheduled_tokens.items())))
    specs = dict(reversed(list(output.scheduled_spec_decode_tokens.items())))
    return replace(
        output,
        num_scheduled_tokens=counts,
        total_num_scheduled_tokens=sum(counts.values()),
        scheduled_spec_decode_tokens=specs,
    )


def _bank_with_lanes(capacity, req_ids, lanes):
    bank = RequestSlots(capacity)
    owners = {}
    req_iter = iter(req_ids)
    lane_set = set(lanes)
    for lane in range(capacity):
        request_id = next(req_iter) if lane in lane_set else f"__hold_{lane}__"
        owners[request_id] = bank.acquire(request_id)
    return bank, {req_id: owners[req_id] for req_id in req_ids}


def _lane_outputs(capacity, lanes, *, width):
    rows = [[] for _ in range(capacity)]
    accepted = [0] * capacity
    drafts = [[] for _ in range(capacity)]
    for lane in lanes:
        rows[lane] = [1000 + lane * 10 + index for index in range(width)]
        accepted[lane] = width
    return accepted, rows, drafts


class _TargetStage:
    device = torch.device("cpu")

    def __init__(self, capacity):
        self.capacity = capacity
        self.calls = 0

    def forward_batch(self, metadata, input_ids, positions, padding_mask, pages):
        assert metadata.capacity == self.capacity
        assert tuple(input_ids.shape) == (self.capacity, 6)
        assert tuple(positions.shape) == (self.capacity, 6)
        assert tuple(padding_mask.shape) == (self.capacity, 6)
        assert tuple(pages.shape)[0] == self.capacity
        starts = metadata.target_query_start_loc.tolist()
        lengths = [right - left for left, right in zip(starts[:-1], starts[1:], strict=True)]
        for lane, request_id in enumerate(metadata.request_ids):
            if request_id is None:
                assert padding_mask[lane].all()
            else:
                assert not padding_mask[lane, :lengths[lane]].any()
                assert padding_mask[lane, lengths[lane]:].all()
        self.calls += 1
        return DSparkTargetBatch(token_ids=torch.zeros((self.capacity, 6), dtype=torch.int32))


class _DraftStage:
    def __init__(self, capacity, *, width, fail_output_lane=None):
        self.capacity = capacity
        self.width = width
        self.fail_output_lane = fail_output_lane
        self.calls = 0

    def verify_and_propose(self, metadata, target):
        del target
        active = tuple(metadata.request_active.tolist())
        lanes = tuple(lane for lane, enabled in enumerate(active) if enabled)
        accepted, outputs, drafts = _lane_outputs(self.capacity, lanes, width=self.width)
        if self.fail_output_lane is not None:
            outputs[self.fail_output_lane].append(999)
        self.calls += 1
        return DSparkDraftBatch(
            accepted=torch.tensor(accepted, dtype=torch.int32),
            output_token_ids=outputs,
            draft_token_ids=drafts,
            completion_event=SimpleNamespace(query=lambda: True),
        )


class _Trace:
    def __init__(self):
        self.records = []

    def write(self, payload):
        self.records.append(dict(payload))


@pytest.mark.parametrize(
    ("capacity", "lanes"),
    ((1, (0, )), (2, (1, )), (4, (1, 3))),
)
def test_real_scheduler_maps_reversed_requests_to_fixed_physical_lanes(tmp_path, capacity, lanes):
    scheduler = _scheduler(tmp_path, capacity)
    req_ids = [f"r{index}" for index in range(len(lanes))]
    output, states = _decode_output(scheduler, req_ids, [100 + index for index in range(len(req_ids))])
    bank, owners = _bank_with_lanes(capacity, req_ids, lanes)
    scheduled = _reverse_scheduled(output)
    trace = _Trace()
    target, draft = _TargetStage(capacity), _DraftStage(capacity, width=2)
    runner = DSparkBatchRunner(target, draft, slots=bank, batch_capacity=capacity, trace=trace)
    plan = runner.build_plan(scheduled, states, owners, max_model_len=8192)

    assert plan is not None
    assert plan.capacity == capacity
    assert plan.active_lanes == lanes
    assert plan.request_ids == tuple(req_ids)
    dummy_page = plan.pages[lanes[0]].clone()
    assert bool((dummy_page > 0).all())
    for lane in range(capacity):
        if lane not in lanes:
            assert torch.equal(plan.pages[lane], dummy_page)
    assert list(scheduled.num_scheduled_tokens) == list(reversed(req_ids))
    model_output, _ = runner.execute(plan)

    expected_ids = list(req_ids)
    assert model_output.req_ids == expected_ids
    assert model_output.sampled_token_ids == [
        [1000 + lane * 10, 1001 + lane * 10] for lane in lanes
    ]
    assert target.calls == draft.calls == 1
    scheduler.update_from_output(output, model_output)
    for index, (req_id, lane) in enumerate(zip(req_ids, lanes, strict=True)):
        assert scheduler.requests[req_id].output_token_ids[-2:] == [
            1000 + lane * 10, 1001 + lane * 10
        ]

    record = trace.records[0]
    assert record["schema"] == "dsv41.dspark.round.v2"
    assert record["batch_capacity"] == record["capacity"] == capacity
    assert record["active_count"] == len(lanes)
    assert record["active_lanes"] == record["lanes"] == list(lanes)
    assert record["request_slots"] == list(lanes)
    assert record["batch_width"] == len(lanes)
    assert record["actual_packed_target_rows"] == record["physical_target_rows"] == 6 * capacity
    assert record["actual_packed_draft_rows"] == record["physical_draft_rows"] == 5 * capacity
    assert record["lane_request_ids"] == [req_ids[lanes.index(lane)] if lane in lanes else None
                                          for lane in range(capacity)]
    assert None not in record["req_ids"]


def test_a1_cancellation_reuse_rejects_stale_plan_and_publishes_new_owner(tmp_path):
    capacity = 4
    bank, owners = _bank_with_lanes(capacity, ["old"], (2, ))
    old = owners["old"]
    state = RequestState(
        req_id="old",
        prompt=[],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
        block_ids=((1, ), ),
        num_computed_tokens=0,
        output=[50],
    )
    scheduled = SimpleNamespace(num_scheduled_tokens={"old": 1}, scheduled_spec_decode_tokens={})
    target, draft = _TargetStage(capacity), _DraftStage(capacity, width=1)
    runner = DSparkBatchRunner(target, draft, slots=bank, batch_capacity=capacity,
                               trace=SimpleNamespace(write=lambda payload: None))
    stale = runner.build_plan(scheduled, {"old": state}, {"old": old}, max_model_len=8192)

    bank.release(old)
    reused = bank.acquire("new")
    assert reused.index == 2 and reused.generation == old.generation + 1
    new_state = RequestState(
        req_id="new",
        prompt=[],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
        block_ids=((2, ), ),
        num_computed_tokens=0,
        output=[60],
    )
    with pytest.raises(RuntimeError, match="slot changed before pack preparation"):
        runner.execute(stale)
    assert target.calls == draft.calls == 0
    assert new_state.output == [60]

    live_scheduled = SimpleNamespace(num_scheduled_tokens={"new": 1}, scheduled_spec_decode_tokens={})
    live = runner.build_plan(live_scheduled, {"new": new_state}, {"new": reused}, max_model_len=8192)
    model_output, _ = runner.execute(live)
    assert model_output.req_ids == ["new"]
    assert model_output.sampled_token_ids == [[1020]]
    assert new_state.output == [60, 1020]


def test_malformed_lane_output_does_not_mutate_any_request(tmp_path):
    capacity = 2
    scheduler = _scheduler(tmp_path, capacity)
    req_ids = ["a", "b"]
    output, states = _decode_output(scheduler, req_ids, [101, 201])
    bank, owners = _bank_with_lanes(capacity, req_ids, (0, 1))
    runner = DSparkBatchRunner(
        _TargetStage(capacity),
        _DraftStage(capacity, width=1, fail_output_lane=1),
        slots=bank,
        batch_capacity=capacity,
        trace=SimpleNamespace(write=lambda payload: None),
    )
    plan = runner.build_plan(output, states, owners, max_model_len=8192)
    before = {req_id: tuple(state.output) for req_id, state in states.items()}

    with pytest.raises(ValueError, match="trailing -1 padding|differs from output length"):
        runner.publish(plan, _DraftStage(capacity, width=1, fail_output_lane=1).verify_and_propose(
            plan.metadata, DSparkTargetBatch(token_ids=torch.zeros((capacity, 6), dtype=torch.int32))))

    assert {req_id: tuple(state.output) for req_id, state in states.items()} == before


def test_idle_plan_does_not_advance_round_generation():
    bank = RequestSlots(4)
    runner = DSparkBatchRunner(
        _TargetStage(4),
        _DraftStage(4, width=1),
        slots=bank,
        batch_capacity=4,
        trace=SimpleNamespace(write=lambda payload: None),
    )
    idle = SimpleNamespace(num_scheduled_tokens={}, scheduled_spec_decode_tokens={})
    assert runner.build_plan(idle, {}, {}, max_model_len=8192) is None
    assert runner.round_generation == 0
