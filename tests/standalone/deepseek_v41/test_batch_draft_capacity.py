# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for draft-side fixed request capacities through N=64."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops import deepseek_v41_batch_attention as batch_attention
from vllm_gaudi.ops import deepseek_v41_batch_draft as draft_batch
from vllm_gaudi.ops.deepseek_v41_batch_verify import validate_batch_records


_CAPACITY_CASES = (
    (1, (True, )),
    (2, (True, False)),
    (3, (True, False, True)),
    (4, (True, False, True, False)),
    (6, (True, False, True, False, True, False)),
    (8, (True, False, True, False, True, False, True, False)),
    (16, tuple(index % 2 == 0 for index in range(16))),
    (64, tuple(index % 2 == 0 for index in range(64))),
)


def _metadata(capacity, active, *, target_counts=None, include_request_active=True, round_generation=1):
    lengths = list(target_counts) if target_counts is not None else [6 if enabled else 0 for enabled in active]
    invalid_lane = any(not enabled and length != 0 for length, enabled in zip(lengths, active, strict=True))
    if len(lengths) != capacity or invalid_lane:
        raise ValueError("test target counts must match capacity and keep inactive lanes empty")
    slots = [index if enabled else -1 for index, enabled in enumerate(active)]
    generations = [index + 1 if enabled else 0 for index, enabled in enumerate(active)]
    budgets = [64 if enabled else 0 for enabled in active]
    starts = [10 + 2 * index if enabled else 0 for index, enabled in enumerate(active)]
    drafts = [
        list(range(start, start + length - 1)) if enabled and length > 1 else []
        for start, length, enabled in zip(starts, lengths, active, strict=True)
    ]
    starts_loc = [0]
    for length in lengths:
        starts_loc.append(starts_loc[-1] + length)
    metadata = SimpleNamespace(
        request_ids=tuple(f"request-{index}" for index in range(capacity)),
        num_draft_tokens=torch.tensor([len(row) for row in drafts], dtype=torch.int32),
        draft_token_ids=torch.tensor([token for row in drafts for token in row], dtype=torch.int32),
        request_slots=torch.tensor(slots, dtype=torch.int32),
        slot_generations=torch.tensor(generations, dtype=torch.int64),
        target_start_positions=torch.tensor(starts, dtype=torch.int64),
        remaining_output_budget=torch.tensor(budgets, dtype=torch.int32),
        target_query_start_loc=torch.tensor(starts_loc, dtype=torch.int32),
        round_generation=torch.tensor(round_generation, dtype=torch.int64),
    )
    if include_request_active:
        metadata.request_active = torch.tensor(active, dtype=torch.bool)
    return metadata


def _target(capacity, active):
    rows = []
    for index, enabled in enumerate(active):
        start = 10 + 2 * index
        rows.append(torch.arange(start, start + 6) if enabled else torch.zeros(6, dtype=torch.int64))
    token_ids = torch.stack(rows)
    return SimpleNamespace(
        token_ids=token_ids,
        states=torch.zeros((capacity, 6, 8), dtype=torch.bfloat16),
    )


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("swa", torch.zeros(256, 8, dtype=torch.uint8))
        self.weights = SimpleNamespace(wkv=object(), kv_norm=SimpleNamespace(weight=torch.ones(8)))
        self.eps = 1e-6
        self.rope_positions = []

    def linear(self, value, weight):
        return value

    def _rope(self, value, positions, request_batch=False):
        assert request_batch
        self.rope_positions.append(positions.clone())
        return value


class _Layer(torch.nn.Module):
    def __init__(self, *, decode=False):
        super().__init__()
        self.attention = _Attention()
        self.draft_mhc = False
        self.eps = 1e-6
        self.hc_eps = 1e-6
        self.iterations = 20
        if decode:
            self.moe = SimpleNamespace(calls=[])
            self.weights = SimpleNamespace(
                hc_attn_fn=object(), hc_attn_scale=object(), hc_attn_base=object(),
                hc_ffn_fn=object(), hc_ffn_scale=object(), hc_ffn_base=object(),
                attn_norm=SimpleNamespace(weight=torch.ones(8)),
                ffn_norm=SimpleNamespace(weight=torch.ones(8)),
            )


class _Draft(torch.nn.Module):
    def __init__(self, *, decode=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(decode=decode) for _ in range(3)])
        self.eps = 1e-6
        if decode:
            self.noise = 99
            self.embedding = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8) / 128
        else:
            first = SimpleNamespace(main_proj=object(), main_norm=SimpleNamespace(weight=torch.ones(8)))
            self.weights = SimpleNamespace(get_submodule=lambda name: first)

    def embed_input_ids(self, ids):
        return F.embedding(ids, self.embedding)

    def compute_logits_local(self, hidden):
        return hidden[:, :1].expand(-1, 6)


@pytest.mark.parametrize(("capacity", "active"), _CAPACITY_CASES)
def test_context_commit_is_capacity_shaped_and_inactive_lanes_do_not_write(monkeypatch, capacity, active):
    monkeypatch.setattr(draft_batch, "_linear", lambda value, owner: value)
    monkeypatch.setattr(draft_batch, "rms_norm", lambda value, weight, eps: value)
    monkeypatch.setattr(draft_batch, "pack_swa", lambda value: value.to(torch.uint8))
    draft = _Draft()
    bank = draft_batch.DraftBatchState(draft, capacity=capacity)
    metadata = _metadata(capacity, active)
    states = torch.arange(capacity * 6 * 8, dtype=torch.uint8).reshape(capacity, 6, 8)
    positions = metadata.target_start_positions.to(torch.int32).unsqueeze(1) + torch.arange(6, dtype=torch.int32)
    positions = torch.where(torch.tensor(active).unsqueeze(1), positions, torch.zeros_like(positions))
    committed = torch.tensor([2 if enabled else 0 for enabled in active], dtype=torch.int32)

    draft_batch.insert_context_batch(
        draft,
        bank,
        states.bfloat16(),
        positions,
        metadata.request_slots,
        committed,
        torch.tensor(active),
    )

    for layer in draft.layers:
        for recorded in layer.attention.rope_positions:
            assert (recorded >= 0).all()
            inactive_rows = recorded.reshape(capacity, 6)[~torch.tensor(active)]
            assert torch.count_nonzero(inactive_rows) == 0
    for index, enabled in enumerate(active):
        slot_ring = bank.slot_rings(index)[0]
        if enabled:
            start = int(metadata.target_start_positions[index])
            assert torch.equal(slot_ring[start:start + 2], states[index, :2])
        else:
            assert torch.count_nonzero(slot_ring) == 0


@pytest.mark.parametrize(("capacity", "active"), _CAPACITY_CASES)
def test_backbone_runs_one_packed_five_capacity_body(monkeypatch, capacity, active):
    class Moe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, value, image_mask, decode=False):
            self.calls.append(tuple(value.shape))
            return value

    draft = _Draft(decode=True)
    for layer in draft.layers:
        layer.moe = Moe()

    def hc_pre(residual, previous, *args):
        tokens = residual.shape[0]
        return residual[:, 0], torch.zeros((tokens, 4)), torch.zeros((tokens, 4)), torch.zeros((tokens, 4, 4))

    monkeypatch.setattr(draft_batch, "hc_pre", hc_pre)
    monkeypatch.setattr(draft_batch, "hc_post", lambda value, residual, post, comb: residual)
    monkeypatch.setattr(draft_batch, "rms_norm", lambda value, weight, eps: value)
    attention_calls = []

    def attention(draft, index, value, positions, slots, active, bank, context_done=None):
        attention_calls.append((index, tuple(value.shape), context_done))
        return value

    monkeypatch.setattr(draft_batch, "draft_attention_batch", attention)
    bank = draft_batch.DraftBatchState(draft, capacity=capacity)
    backbone = draft_batch.PackedDraftBackbone(draft, bank)
    positions = torch.arange(capacity * 5, dtype=torch.int32).reshape(capacity, 5) + 100
    positions = torch.where(torch.tensor(active).unsqueeze(1), positions, torch.zeros_like(positions))
    hidden, logits = backbone(
        torch.arange(capacity, dtype=torch.int64).unsqueeze(1),
        positions,
        torch.arange(capacity, dtype=torch.int32),
        torch.tensor(active),
    )

    assert hidden.shape == (capacity, 5, 8)
    assert logits.shape == (capacity, 5, 6)
    assert attention_calls == [(index, (capacity * 5, 8), None) for index in range(3)]
    assert all(len(layer.moe.calls) == 1 for layer in draft.layers)
    assert all(layer.moe.calls[0] == (capacity * 5, 8) for layer in draft.layers)


@pytest.mark.parametrize(("capacity", "activity"), _CAPACITY_CASES)
def test_markov_is_five_sequential_steps_over_all_capacity_lanes(capacity, activity):
    class MarkovDraft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_calls = []
            self.argmax_calls = []
            last = SimpleNamespace(
                markov_head=SimpleNamespace(
                    embed=SimpleNamespace(weight=torch.eye(4)),
                    head=SimpleNamespace(weight=torch.roll(torch.eye(4), shifts=1, dims=0))),
                confidence_head=SimpleNamespace(proj=SimpleNamespace(weight=torch.arange(8.).reshape(1, 8))),
            )
            self.weights = SimpleNamespace(get_submodule=lambda name: last)

        def _embed(self, previous, embedding):
            self.embed_calls.append(tuple(previous.shape))
            return F.embedding(previous, embedding.weight)

        def _global_argmax(self, scores):
            self.argmax_calls.append(tuple(scores.shape))
            return scores.argmax(-1)

    draft = MarkovDraft()
    sampler = draft_batch.PackedDraftMarkovSampler(draft, batch_capacity=capacity)
    anchors = torch.arange(capacity - 1, -1, -1, dtype=torch.int64).remainder(4)
    hidden = torch.arange(capacity * 5 * 4, dtype=torch.float32).reshape(capacity, 5, 4)
    logits = torch.zeros((capacity, 5, 4), dtype=torch.float32)
    active = torch.tensor(activity)
    proposals, confidence = sampler(anchors, hidden, logits, active)

    assert proposals.shape == confidence.shape == (capacity, 5)
    previous = torch.where(active, anchors, 0)
    for step in range(5):
        expected = (previous + 1).remainder(4)
        assert torch.equal(proposals[:, step], torch.where(active, expected, -1))
        confidence_input = torch.cat((hidden[:, step], F.embedding(previous, torch.eye(4))), dim=-1)
        assert torch.equal(confidence[:, step], F.linear(confidence_input, torch.arange(8.).reshape(1, 8)).flatten())
        previous = torch.where(active, expected, 0)
    assert draft.embed_calls == [(capacity,)] * 5
    assert draft.argmax_calls == [(capacity, 4)] * 5


@pytest.mark.parametrize(("capacity", "active"), _CAPACITY_CASES)
def test_transaction_uses_capacity_shaped_contracts(monkeypatch, capacity, active):
    round_calls = []
    context_done = tuple(
        torch.tensor([layer, capacity], dtype=torch.int32) for layer in range(3))

    class Round:
        batch_capacity = capacity

        def __init__(self):
            self.speculator = self

        def propose_after_verify(self, states, positions, slots, committed, output, output_count,
                                 draft_enabled, active, status):
            round_calls.append((states.shape, positions.shape, slots.shape, active.shape,
                                committed.clone(), output_count.clone(), status.clone()))
            return torch.full((capacity, 5), 77, dtype=torch.int64), torch.zeros((capacity, 5))

    monkeypatch.setattr(draft_batch, "PackedDraftRound", lambda *args, **kwargs: Round())

    def verify(target, proposals, control, active, binding_sidecar):
        assert binding_sidecar.shape == (capacity, 2)
        assert binding_sidecar.dtype == torch.int32
        return (
            target,
            torch.arange(capacity * 6, dtype=torch.int64).reshape(capacity, 6),
            torch.full((capacity, ), 6, dtype=torch.int64),
            torch.full((capacity, ), 6, dtype=torch.int64),
            torch.arange(capacity, dtype=torch.int64),
            active.clone(),
            torch.zeros((capacity, ), dtype=torch.int64),
        )

    monkeypatch.setattr(draft_batch, "verify_control_from_target_batch", verify)
    monkeypatch.setattr(draft_batch, "insert_context_batch", lambda *args: context_done)
    monkeypatch.setattr(
        draft_batch,
        "draft_inputs_after_commit",
        lambda *args: (torch.arange(capacity, dtype=torch.int64), torch.arange(
            capacity * 5, dtype=torch.int32).reshape(capacity, 5)),
    )
    monkeypatch.setattr(
        draft_batch,
        "pack_record_batch",
        lambda *args: torch.zeros((capacity, 16), dtype=torch.int64),
    )
    state = SimpleNamespace(capacity=capacity)
    transaction = draft_batch.PackedDraftTransaction(
        SimpleNamespace(), state, batch_capacity=capacity, compile_graph=False)
    values = transaction(
        torch.zeros((capacity, 6), dtype=torch.int64),
        torch.zeros((capacity, 6, 8), dtype=torch.bfloat16),
        torch.zeros((capacity, 6), dtype=torch.int32),
        torch.arange(capacity, dtype=torch.int32),
        torch.zeros((capacity, 7), dtype=torch.int64),
        torch.full((capacity, 5), -1, dtype=torch.int64),
        torch.tensor(active),
        torch.zeros((capacity, 2), dtype=torch.int32),
    )

    record, committed, output, output_count, proposals, status = values
    assert record.shape == (capacity, 16)
    assert committed.shape == output_count.shape == proposals.shape[:1] == status.shape == (capacity, )
    assert output.shape == (capacity, 6)
    assert proposals.shape == (capacity, 5)
    assert len(round_calls) == 1
    assert round_calls[0][:4] == (
        (capacity, 6, 8), (capacity, 6), (capacity, ), (capacity, ))
    assert round_calls[0][4].tolist() == [6] * capacity
    assert round_calls[0][5].tolist() == [6] * capacity
    assert round_calls[0][6].tolist() == [0] * capacity


@pytest.mark.parametrize("prepared_rows", [15, 30, 40, 55, 80, 320])
def test_prepared_dense_branch_tiles_every_supported_packed_row_count(prepared_rows):
    prepared = torch.eye(4, dtype=torch.float32)
    value = torch.arange(prepared_rows * 4, dtype=torch.float32).reshape(prepared_rows, 4)
    owner = SimpleNamespace(dspark_dense_kn=prepared)
    assert torch.equal(draft_batch._linear(value, owner), value)


def test_n11_state_row_wrappers_tile_six_n_rows_at_native_boundary(monkeypatch):
    write_calls, read_calls = [], []

    def write(cache, values, rows):
        del cache
        assert values.shape[0] == rows.numel() <= 64
        write_calls.append(rows.numel())
        return rows.clone()

    def read(cache, rows, completion):
        del cache
        assert rows.shape == completion.shape and rows.numel() <= 64
        read_calls.append(rows.numel())
        return rows.reshape(-1, 1).repeat(1, 2)

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_state_rows_write_gaudi2", write, raising=False)
    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_state_rows_read_gaudi2", read, raising=False)
    rows = torch.arange(11 * 6, dtype=torch.int32)
    values = torch.arange(11 * 6 * 2, dtype=torch.float32).reshape(11 * 6, 2)

    done = batch_attention.write_state_rows(torch.zeros((128, 2)), values, rows)
    decoded = batch_attention.read_state_rows(torch.zeros((128, 2)), rows, done)

    assert write_calls == read_calls == [64, 2]
    assert torch.equal(done, rows)
    assert torch.equal(decoded, rows.reshape(-1, 1).repeat(1, 2))


def test_n11_packed_mla_tiles_66_query_rows(monkeypatch):
    calls = []

    def mla(query, swa, main, rows, indices, sink, scale, lengths):
        del swa, main, rows, indices, sink, scale, lengths
        assert query.shape[0] <= 8
        calls.append(query.shape[0])
        return query.clone()

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_batch_paged_mla_mme_gaudi2", mla, raising=False)
    rows = 11 * 6
    query = torch.zeros((rows, 1, 512), dtype=torch.bfloat16)
    output = batch_attention.batch_packed_mla(
        query,
        torch.zeros((256, 528), dtype=torch.uint8),
        torch.zeros((1, 288), dtype=torch.uint8),
        torch.zeros((rows, 512), dtype=torch.int32),
        torch.ones((rows, 1), dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros(1),
        torch.tensor([1.0]),
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        ratio=0,
    )

    assert calls == [8] * 8 + [2]
    assert torch.equal(output, query)


def test_missing_or_empty_request_active_is_legacy_all_active():
    missing = _metadata(2, (True, True), include_request_active=False)
    host, _, _, _, active, _ = draft_batch._host_metadata_values(missing)
    assert active == (True, True)
    assert host["request_slots"] == (0, 1)

    empty = _metadata(2, (True, True))
    empty.request_active = torch.empty(0, dtype=torch.bool)
    _, _, _, _, active, _ = draft_batch._host_metadata_values(empty)
    assert active == (True, True)


def test_capacity_constants_and_explicit_mismatches_fail_before_ticket(monkeypatch):
    assert draft_batch.DEFAULT_BATCH_CAPACITY == 2
    assert draft_batch.MAX_BATCH_REQUESTS == 64
    with pytest.raises(ValueError, match="must match the batch state"):
        draft_batch.PackedDraftTransaction(
            SimpleNamespace(), SimpleNamespace(capacity=2), batch_capacity=4, compile_graph=False)

    class Ring:
        batch_capacity = 2

        def acquire(self, **kwargs):
            raise AssertionError("capacity mismatch acquired a ticket")

    class Backbone:
        batch_capacity = 4

        def __call__(self, *args):
            raise AssertionError("capacity mismatch ran the backbone")

    stage = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(),
        batch_state=SimpleNamespace(capacity=4),
        verify_ring=Ring(),
        backbone=Backbone(),
        max_model_len=128,
        batch_capacity=4,
    )
    with pytest.raises(ValueError, match="verify ring capacity"):
        stage.verify_and_propose(_metadata(4, (True, False, True, False)), _target(4, (True, False, True, False)))


def test_hpu_transaction_rejects_missing_sidecar_before_dispatch():
    transaction = draft_batch.PackedDraftTransaction(
        SimpleNamespace(layers=[SimpleNamespace(draft_mhc=False)]),
        SimpleNamespace(capacity=2),
        batch_capacity=2,
        compile_graph=False,
    )
    hpu = torch.device("hpu")

    def shape_only(shape):
        return SimpleNamespace(shape=shape, device=hpu)

    with pytest.raises(RuntimeError, match="requires an explicit I32 binding sidecar"):
        transaction.forward(
            shape_only((2, 6)),
            shape_only((2, 6, 8)),
            shape_only((2, 6)),
            shape_only((2, )),
            shape_only((2, 7)),
            shape_only((2, 5)),
            shape_only((2, )),
        )


def test_stage_fixed_lanes_return_empty_outputs_for_inactive_requests(monkeypatch):
    class Event:
        def query(self):
            return True

    class Ticket:
        def __init__(self, capacity):
            self.record = torch.empty((capacity, 16), dtype=torch.int64)
            self.host = torch.empty((capacity, 16), dtype=torch.int64)
            self.completion = Event()

    class Ring:
        batch_capacity = 4

        def __init__(self):
            self.ticket = None

        def acquire(self, **kwargs):
            self.kwargs = kwargs
            self.ticket = Ticket(4)
            return self.ticket

        def stage(self, ticket):
            ticket.host.copy_(ticket.record)

        def consume(self, ticket):
            return validate_batch_records(
                ticket.host,
                round_generation=1,
                request_slots=self.kwargs["request_slots"],
                slot_generations=self.kwargs["slot_generations"],
                max_committed=self.kwargs["max_committed"],
                active=self.kwargs["active"],
            )

        def release(self, ticket):
            pass

    active = torch.tensor([True, False, True, False])
    inserted = []
    context_done = tuple(torch.tensor([index + 1], dtype=torch.int32) for index in range(3))

    def insert_context(draft, bank, states, positions, slots, committed, active_mask):
        inserted.append(active_mask.clone())
        return context_done

    monkeypatch.setattr(draft_batch, "insert_context_batch", insert_context)
    backbone_calls = []

    class Backbone:
        batch_capacity = 4

        def __call__(self, anchors, positions, slots, active_mask, *, context_done=None):
            backbone_calls.append((anchors.shape, positions.shape, active_mask.clone(), context_done))
            return torch.zeros((4, 5, 4)), torch.zeros((4, 5, 6))

    class Markov:
        batch_capacity = 4

        def __call__(self, anchors, hidden, logits, active_mask):
            return torch.tensor([[20, 21, 22, 23, 24], [-1] * 5, [30, 31, 32, 33, 34], [-1] * 5]), torch.zeros(
                (4, 5))

    stage = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(),
        batch_state=SimpleNamespace(capacity=4),
        verify_ring=Ring(),
        backbone=Backbone(),
        markov=Markov(),
        max_model_len=128,
        batch_capacity=4,
        target_ids=lambda target, batch_size: target.token_ids,
        target_context=lambda target, positions: target.states,
    )
    result = stage.verify_and_propose(
        _metadata(4, (True, False, True, False), target_counts=(1, 0, 6, 0)),
        _target(4, (True, False, True, False)),
    )

    assert result.accepted.tolist() == [1, 0, 6, 0]
    assert result.output_token_ids[1] == result.output_token_ids[3] == ()
    assert result.draft_token_ids[1] == result.draft_token_ids[3] == ()
    assert inserted[0].tolist() == active.tolist()
    assert len(backbone_calls) == 1
    assert backbone_calls[0][0] == torch.Size((4, 1))
    assert backbone_calls[0][1] == torch.Size((4, 5))
    assert backbone_calls[0][2].tolist() == active.tolist()
    assert backbone_calls[0][3] is context_done


def test_legacy_custom_stage_component_stays_compatible_on_cpu():
    calls = []

    class LegacyBackbone:
        def __call__(self, anchors, positions, slots, active):
            calls.append((anchors, positions, slots, active))
            return anchors, anchors

    anchors = torch.zeros((2, 1), dtype=torch.int64)
    positions = torch.zeros((2, 5), dtype=torch.int32)
    slots = torch.arange(2, dtype=torch.int32)
    active = torch.tensor([True, True])
    context_done = (torch.tensor([1], dtype=torch.int32), ) * 3
    hidden, logits = draft_batch._call_with_optional_context_done(
        LegacyBackbone(), anchors, positions, slots, active,
        context_done=context_done, name="backbone")

    assert hidden is anchors and logits is anchors
    assert calls == [(anchors, positions, slots, active)]


class _ValidatingCpuRing:
    """CPU ring double that keeps the real record validation.

    The production ring pins host buffers for HPU and therefore needs an
    acquirable device; this double exercises the same acquire/stage/consume
    contract with plain host tensors.
    """

    def __init__(self, capacity):
        self.batch_capacity = capacity
        self.kwargs = None

    def acquire(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            record=torch.empty((self.batch_capacity, 16), dtype=torch.int64),
            host=torch.empty((self.batch_capacity, 16), dtype=torch.int64),
            completion=SimpleNamespace(query=lambda: True),
        )

    def stage(self, ticket):
        ticket.host.copy_(ticket.record)

    def consume(self, ticket):
        return validate_batch_records(
            ticket.host,
            round_generation=self.kwargs["round_generation"],
            request_slots=self.kwargs["request_slots"],
            slot_generations=self.kwargs["slot_generations"],
            max_committed=self.kwargs["max_committed"],
            active=self.kwargs["active"],
        )

    def release(self, ticket):
        pass

    def abort(self, ticket):
        pass


@pytest.mark.parametrize("capacity,enabled,bucket", [(11, True, 4), (32, True, 4), (11, False, 11)])
def test_draft_stage_selects_bounded_transactions_and_keeps_full_commit_records(
        monkeypatch, capacity, enabled, bucket):
    calls = []

    class Transaction:
        def __init__(self, draft, state, *, batch_capacity, execution_capacity=None):
            self.batch_capacity = batch_capacity
            self.execution_capacity = execution_capacity or batch_capacity
            self.graph_key = (batch_capacity, self.execution_capacity)
            self.compiled = None

        def __call__(self, target_ids, states, positions, slots, control, proposals, active, sidecar,
                     execution_lanes=None):
            calls.append((self.execution_capacity, None if execution_lanes is None else execution_lanes.tolist()))
            _, output, committed, output_count, _, draft_enabled, status = (
                draft_batch.verify_control_from_target_batch(target_ids, proposals, control, active, sidecar))
            draft_active = draft_enabled & active & (status == 0)
            proposed = torch.where(draft_active[:, None], torch.full((capacity, 5), 77, dtype=torch.int64), -1)
            record = draft_batch.pack_record_batch(
                control, committed, output_count, output, proposed, draft_active, status, sidecar)
            assert record.shape == (capacity, 16)
            return record, committed, output, output_count, proposed, status

    monkeypatch.setattr(draft_batch, "PackedDraftTransaction", Transaction)
    active = tuple(lane in (3, capacity - 1) for lane in range(capacity))
    stage = draft_batch.DSparkBatchDraftStage(
        SimpleNamespace(), batch_state=SimpleNamespace(capacity=capacity), batch_capacity=capacity,
        verify_ring=_ValidatingCpuRing(capacity),
        max_model_len=1024, target_ids=lambda target, size: target.token_ids,
        target_context=lambda target, positions: target.states)
    stage.configure_execution_buckets(enabled, buckets=(4, ))
    result = stage.verify_and_propose(_metadata(capacity, active), _target(capacity, active))
    assert stage.transaction is stage.transactions[capacity]
    assert stage.last_execution_plan.graph_key == (capacity, bucket)
    assert calls[0][0] == bucket
    expected_sources = [3, capacity - 1, 0, 1] if enabled else None
    assert calls[0][1] == expected_sources
    assert len(result.output_token_ids) == len(result.draft_token_ids) == capacity
    assert all(result.draft_token_ids[lane] == () for lane in range(capacity) if not active[lane])
    assert all(len(result.draft_token_ids[lane]) == 5 for lane in range(capacity) if active[lane])
