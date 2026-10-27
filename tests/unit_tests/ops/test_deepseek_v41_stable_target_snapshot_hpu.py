# SPDX-License-Identifier: Apache-2.0
"""HPU contracts for the stable (N,B) target snapshot and typed draft input staging.

Production capture/restore (compiled per view on HPU, native state writes) must
leave every state tensor bit-identical to the CPU legacy snapshot.
"""
import copy
import os

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Requires an explicit HPU lease", allow_module_level=True)

rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402
from torch import nn  # noqa: E402

torch.hpu.set_device(rank)
torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])

from vllm_gaudi.ops import deepseek_v41_batch_draft as draft_batch  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_batch_target import (  # noqa: E402
    DSparkStableTargetLayout, DSparkTargetBatch, DSparkTargetSnapshot)
from vllm_gaudi.ops.deepseek_v41_target_buckets import pack_target_bucket  # noqa: E402

HPU = torch.device("hpu")
PAGES = 6


def _program(capacity, seed):
    generator = torch.Generator().manual_seed(seed)
    program = nn.Module()
    program.layers = nn.ModuleList()
    for index in range(4):
        layer = nn.Module()
        layer.attention = nn.Module()
        state = nn.Module()
        state.register_buffer("swa", torch.randint(0, 256, (capacity * 256, 528), generator=generator,
                                                   dtype=torch.uint8))
        if index == 1:
            state.register_buffer("kv_history", torch.randn((capacity * 8, 512), generator=generator))
            state.register_buffer("score_history", torch.randn((capacity * 8, 512), generator=generator))
        layer.attention.batch_state = state
        program.layers.append(layer)
    program.shared = nn.Module()
    program.shared.sources = nn.ModuleDict()
    for source, ratio in ((2, 2), (20, 1)):
        cache = nn.Module()
        cache.ratio = ratio
        rows = (capacity * PAGES + 1) * 128 // ratio
        cache.register_buffer("main", torch.randint(0, 256, (rows, 288), generator=generator, dtype=torch.uint8))
        cache.register_buffer("index", torch.randint(0, 256, (rows, 68), generator=generator, dtype=torch.uint8))
        program.shared.sources[str(source)] = cache
    return program


def _batch(capacity, lanes, bucket, aliases=()):
    metadata = DSparkBatchMetadata.build(
        requests=[f"request-{i}" if i in lanes else None for i in range(capacity)],
        target_query_lengths=[lanes[i][0] if i in lanes else 0 for i in range(capacity)],
        draft_token_ids=[list(range(1, lanes[i][0])) if i in lanes else [] for i in range(capacity)],
        request_slots=[i if i in lanes else -1 for i in range(capacity)],
        slot_generations=[3 + i if i in lanes else 0 for i in range(capacity)],
        remaining_output_budget=[64 if i in lanes else 0 for i in range(capacity)],
        target_start_positions=[lanes[i][1] if i in lanes else 0 for i in range(capacity)],
        round_generation=9,
        active=[i in lanes for i in range(capacity)],
    )
    queries = torch.arange(capacity * 6, dtype=torch.int64).reshape(capacity, 6)
    pages = torch.arange(capacity * PAGES, dtype=torch.int32).reshape(capacity, PAGES) + 1
    for alias, source in aliases:
        pages[alias] = pages[source]
    packed = pack_target_bucket(metadata, queries, pages, bucket)
    batch = DSparkTargetBatch.from_metadata(packed.metadata, packed.queries, packed.pages, batch_capacity=bucket,
                                            state_capacity=capacity if bucket != capacity else None)
    return batch, packed.source_lanes


def _rounds(capacity):
    generator = torch.Generator().manual_seed(capacity)
    starts = (254, 255, 127, 1, 0, PAGES * 128 - 3, 511, 64)
    every = {lane: (1 + lane % 6, starts[lane % len(starts)] + 2 * (lane // len(starts))) for lane in range(capacity)}
    plans = [(every, capacity, ()), ({1: (6, 124), 3: (6, 124), 6: (2, 30)}, max(4, capacity // 4), ((3, 1), )),
             ({capacity - 1: (6, 255)}, 1, ())]
    rounds = []
    for lanes, bucket, aliases in plans:
        full = [lanes[i][0] if i in lanes else 0 for i in range(capacity)]
        mixed = [int(torch.randint(0, lanes[i][0] + 1, (1, ), generator=generator)) if i in lanes else 0
                 for i in range(capacity)]
        invalid = list(mixed)
        invalid[min(lanes)] = 7
        for counts in ([0] * capacity, full, mixed, invalid):
            rounds.append((lanes, bucket, aliases, counts))
    return rounds


def _bits(tensor):
    return tensor.view(torch.int32) if tensor.dtype == torch.float32 else tensor


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("capacity", [8, 64])
def test_production_restore_matches_cpu_legacy_snapshot(capacity, padded):
    torch._dynamo.reset()
    generator = torch.Generator().manual_seed(17)
    reference = _program(capacity, capacity)
    program = copy.deepcopy(reference).to(HPU)
    buffers = dict(program.named_buffers())
    names = {id(buffer): name for name, buffer in reference.named_buffers()}
    layout = DSparkStableTargetLayout(program, capacity, device=HPU)
    for lanes, bucket, aliases, counts in _rounds(capacity):
        batch, source_lanes = _batch(capacity, lanes, bucket, aliases)
        legacy = DSparkTargetSnapshot.capture(reference, batch)
        legacy.prepare_device_restore()
        snapshot = layout.view(bucket, padded=padded)
        snapshot.capture(batch.to(HPU), source_lanes)
        for row in legacy.rows:
            shape = (row.indices.numel(), *row.destination.shape[1:])
            values = (torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
                      if row.destination.dtype == torch.uint8 else torch.randn(shape, generator=generator))
            row.destination.index_copy_(0, row.indices, values)
            buffers[names[id(row.destination)]].index_copy_(0, row.indices.to(HPU), values.to(HPU))
        with torch.no_grad():
            # Row zero is every padding entry's row; a later write must survive.
            for name, buffer in reference.named_buffers():
                buffer[0].add_(3)
                buffers[name][0].copy_(buffer[0].to(HPU))
        device_counts = torch.tensor(counts, dtype=torch.int64)
        lane_counts = device_counts
        if len(source_lanes) != capacity:
            indices = torch.tensor([max(0, lane) for lane in source_lanes], dtype=torch.int64)
            valid = torch.tensor([lane >= 0 for lane in source_lanes], dtype=torch.bool)
            lane_counts = torch.where(valid, device_counts.index_select(0, indices), 0)
        legacy.restore_device(lane_counts)
        snapshot.restore_device(device_counts.to(HPU))
        snapshot.release()
        for name, expected in reference.named_buffers():
            assert torch.equal(_bits(expected), _bits(buffers[name].cpu())), (name, bucket, counts)
    assert layout.owner is None


@pytest.mark.parametrize("lanes", [0, 3])
def test_typed_draft_input_staging_on_hpu(lanes):
    capacity = 8
    staging = draft_batch.DraftRoundInputStaging(capacity, lanes, HPU)
    assert all(host.is_pinned() for host in staging.hosts)
    generator = torch.Generator().manual_seed(lanes)
    for _round in range(3):
        control = torch.randint(-1000, 1000, (capacity, 7), generator=generator)
        proposals = torch.randint(-1, 1000, (capacity, 5), generator=generator)
        starts = torch.randint(0, 2**20, (capacity, ), generator=generator).tolist()
        slots = torch.randint(-1, capacity, (capacity, ), generator=generator).tolist()
        active = (torch.rand(capacity, generator=generator) > 0.5).tolist()
        sidecar = torch.randint(-2**31, 2**31 - 1, (capacity, 2), generator=generator, dtype=torch.int32)
        execution = tuple(sorted(torch.randperm(capacity, generator=generator)[:lanes].tolist())) if lanes else None
        values = staging.upload(control, proposals, starts, slots, active, sidecar, execution)
        expected = (control, proposals, torch.tensor(starts), torch.tensor(slots, dtype=torch.int32),
                    torch.tensor(active), sidecar, None if execution is None else torch.tensor(execution))
        for got, want in zip(values, expected, strict=True):
            if want is None:
                assert got is None
                continue
            assert got.device.type == "hpu" and got.dtype == want.dtype and torch.equal(got.cpu(), want)
