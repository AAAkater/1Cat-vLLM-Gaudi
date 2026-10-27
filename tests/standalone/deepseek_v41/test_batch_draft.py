# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for packed DSpark draft body, Markov and context ownership."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm_gaudi.ops import deepseek_v41_batch_draft as draft_batch
from vllm_gaudi.v1.spec_decode.deepseek_v41_dspark import HPUDSparkSpeculator


def test_draft_attention_rows_isolate_requests_and_keep_tentative_scratch():
    positions = torch.tensor([[260, 261, 262, 263, 264], [1000, 1001, 1002, 1003, 1004]],
                             dtype=torch.int32)
    rows = draft_batch.draft_attention_rows(
        positions,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([True, True]),
        ring_rows=512,
    )
    assert rows.row_ids.shape == (2 * (128 + 5), )
    assert rows.indices.shape == (10, 192)
    assert rows.lengths.tolist() == [133] * 10
    request0 = set(rows.row_ids[:133].tolist())
    request1 = set(rows.row_ids[133:].tolist())
    assert request0.isdisjoint(request1)
    assert set(rows.row_ids[128:133].tolist()) == set(range(512, 517))
    assert set(rows.row_ids[133 + 128:133 + 133].tolist()) == set(range(517, 522))
    assert (rows.indices[:, 133:] == -1).all()
    assert all(torch.equal(rows.sparse_indices[index, :128], torch.arange(128)) for index in range(5))
    assert all(torch.equal(rows.sparse_indices[index, :128], torch.arange(128, 256)) for index in range(5, 10))
    assert rows.sparse_indices[0, 128:133].tolist() == [256, 257, 258, 259, 260]
    assert rows.sparse_indices[8, 128:133].tolist() == [261, 262, 263, 264, 265]
    assert (rows.sparse_indices[:, 133:] == -1).all()


def test_partial_accept_commits_only_accepted_draft_context(monkeypatch):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("swa", torch.zeros(256, 8, dtype=torch.uint8))
            self.weights = SimpleNamespace(wkv=object(), kv_norm=SimpleNamespace(weight=torch.ones(8)))
            self.eps = 1e-6
            self.linear = lambda value, weight: value

        def _rope(self, value, positions, request_batch=False):
            assert request_batch
            return value

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer(), Layer()])
            first = SimpleNamespace(main_proj=object(), main_norm=SimpleNamespace(weight=torch.ones(8)))
            self.weights = SimpleNamespace(get_submodule=lambda name: first)
            self.eps = 1e-6

    monkeypatch.setattr(draft_batch, "_linear", lambda value, owner: value)
    monkeypatch.setattr(draft_batch, "rms_norm", lambda value, weight, eps: value)
    monkeypatch.setattr(draft_batch, "pack_swa", lambda value: value.to(torch.uint8))
    draft = Draft()
    bank = draft_batch.DraftBatchState(draft, capacity=2)
    states = torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8).bfloat16()
    positions = torch.tensor([[100, 101, 102, 103, 104, 105], [300, 301, 302, 303, 304, 305]],
                             dtype=torch.int32)
    draft_batch.insert_context_batch(
        draft,
        bank,
        states,
        positions,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([2, 1], dtype=torch.int32),
        torch.tensor([True, True]),
    )
    ring = bank.ring(0)
    assert torch.equal(ring[100:102], states[0, :2].to(torch.uint8))
    assert torch.equal(ring[300:301], states[1, :1].to(torch.uint8))
    assert torch.count_nonzero(ring[102:106]) == 0
    assert torch.count_nonzero(ring[301:306]) == 0


def test_backing_bank_is_authoritative_and_single_slot_can_be_adopted():
    class Attention(torch.nn.Module):
        def __init__(self, fill):
            super().__init__()
            self.register_buffer("swa", torch.full((256, 8), fill, dtype=torch.uint8))

    class Layer(torch.nn.Module):
        def __init__(self, fill):
            super().__init__()
            self.attention = Attention(fill)

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(1), Layer(2), Layer(3)])

    draft = Draft()
    backing = [torch.full((2 * 256, 8), 9, dtype=torch.uint8) for _ in range(3)]
    bank = draft_batch.DraftBatchState(draft, capacity=2, backing_rings=backing)
    assert bank.ring(0) is backing[0]
    bank.adopt_single_ring(1, draft)
    for index, layer in enumerate(draft.layers):
        assert torch.equal(bank.slot_rings(1)[index], layer.attention.swa)
        assert torch.equal(bank.slot_rings(0)[index], backing[index][:256])


class _MarkovDraft(torch.nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.hidden = hidden
        last = SimpleNamespace(
            markov_head=SimpleNamespace(embed=object(), head=SimpleNamespace(weight=torch.eye(hidden))),
            confidence_head=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(1, hidden * 2))),
        )
        self.weights = SimpleNamespace(get_submodule=lambda name: last)

    def _embed(self, previous, embedding):
        return F.one_hot(previous.reshape(-1).long().remainder(self.hidden), self.hidden).float() * 2

    def _global_argmax(self, scores):
        return scores.argmax(-1)


def test_markov_steps_are_sequential_but_cross_request_independent(monkeypatch):
    monkeypatch.setattr(draft_batch, "_linear", lambda value, owner: torch.zeros(
        (value.shape[0], 1), dtype=torch.float32))
    draft = _MarkovDraft()
    anchors = torch.tensor([3, 1], dtype=torch.int64)
    hidden = torch.zeros((2, 5, 4), dtype=torch.float32)
    logits = torch.full((2, 5, 4), 0.1, dtype=torch.float32)
    active = torch.tensor([True, True])
    actual, confidence = draft_batch.sample_greedy_markov_batch(draft, anchors, hidden, logits, active)
    assert actual.tolist() == [[3, 3, 3, 3, 3], [1, 1, 1, 1, 1]]
    assert confidence.shape == (2, 5)

    changed_anchor = anchors.clone()
    changed_anchor[0] = 0
    changed, _ = draft_batch.sample_greedy_markov_batch(draft, changed_anchor, hidden, logits, active)
    assert torch.equal(changed[1], actual[1])
    assert changed[0].tolist() == [0, 0, 0, 0, 0]


class _UpstreamMarkovDraft(torch.nn.Module):
    def __init__(self, hidden=4, vocab=6, rank=3, *, track_calls=True):
        super().__init__()
        embedding = (torch.arange(vocab * rank, dtype=torch.float32).reshape(vocab, rank) + 1) / 10
        markov_weight = (torch.arange(vocab * rank, dtype=torch.float32).reshape(vocab, rank) - 7) / 13
        confidence_weight = torch.arange(hidden + rank, dtype=torch.float32).reshape(1, -1) / 17
        last = SimpleNamespace(
            markov_head=SimpleNamespace(embed=embedding, head=SimpleNamespace(weight=markov_weight)),
            confidence_head=SimpleNamespace(proj=SimpleNamespace(weight=confidence_weight)),
        )
        self.weights = SimpleNamespace(get_submodule=lambda name: last)
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
        self.embed_calls = [] if track_calls else None
        self.embed_inputs = [] if track_calls else None
        self.argmax_calls = [] if track_calls else None
        self.all_gather_calls = 0

    def _embed(self, previous, embedding):
        if self.embed_calls is not None:
            self.embed_calls.append(tuple(previous.shape))
            self.embed_inputs.append(previous.reshape(-1).tolist())
        return F.embedding(previous.long(), embedding)

    def _global_argmax(self, scores):
        if self.argmax_calls is not None:
            self.argmax_calls.append(tuple(scores.shape))
        return scores.argmax(-1)

    def all_gather(self, *args, **kwargs):
        del args, kwargs
        self.all_gather_calls += 1
        raise AssertionError("the cached local-logit sampler must not gather logits")

    def compute_logits_local(self, hidden):
        del hidden
        raise AssertionError("the adapter must reuse cached logits")


def _exact_linear(value, owner):
    return F.linear(value, owner.weight, getattr(owner, "bias", None))


def _legacy_markov_golden(draft, first_token, hidden, logits, active, linear):
    last = draft.weights.get_submodule("2")
    previous = torch.where(active, first_token.long(), torch.zeros_like(first_token, dtype=torch.long))
    token_ids, confidence = [], []
    for index in range(draft_batch.MAX_DRAFT_TOKENS):
        markov = draft._embed(previous, last.markov_head.embed)
        bias = F.linear(markov.float(), last.markov_head.head.weight)
        sampled = draft._global_argmax(logits[:, index] + bias).reshape(-1)
        token_ids.append(torch.where(active, sampled, -1))
        confidence_input = torch.cat((hidden[:, index], markov), dim=-1).float()
        confidence.append(linear(confidence_input, last.confidence_head.proj).reshape(-1))
        previous = torch.where(active, sampled, 0)
    return torch.stack(token_ids, dim=1), torch.stack(confidence, dim=1)


def test_hpu_speculator_inherits_real_upstream_methods():
    assert issubclass(HPUDSparkSpeculator, DSparkSpeculator)
    assert HPUDSparkSpeculator._sample_sequential is DSparkSpeculator._sample_sequential
    assert HPUDSparkSpeculator._generate_draft is DSparkSpeculator._generate_draft


class _PackedBackbone:
    def __init__(self, hidden, logits, *, track_calls=True, fail_first=False, events=None):
        self.hidden = hidden
        self.logits = logits
        self.calls = [] if track_calls else None
        self.fail_first = fail_first
        self.events = events

    def __call__(self, anchors, positions, slots, active, *, context_done=None):
        if self.calls is not None:
            self.calls.append((anchors.clone(), positions.clone(), slots.clone(), active.clone(), context_done))
        if self.events is not None:
            self.events.append("backbone")
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("packed backbone failure")
        batch_size = anchors.shape[0]
        return self.hidden[:batch_size].clone(), self.logits[:batch_size].clone()


def test_propose_packed_uses_real_backbone_once_and_masks_inactive_feedback(monkeypatch):
    monkeypatch.setattr(draft_batch, "_linear", _exact_linear)
    draft = _UpstreamMarkovDraft()
    anchors = torch.tensor([[5], [999]], dtype=torch.int64)
    positions = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int32)
    slots = torch.tensor([0, 1], dtype=torch.int32)
    active = torch.tensor([True, False])
    hidden = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4) / 7
    logits = torch.arange(2 * 5 * 6, dtype=torch.float32).reshape(2, 5, 6) / 11
    context_done = tuple(torch.tensor([index + 1], dtype=torch.int32) for index in range(3))
    backbone = _PackedBackbone(hidden, logits)
    linear_calls = []

    def tracked_linear(value, owner):
        linear_calls.append(tuple(value.shape))
        return _exact_linear(value, owner)

    speculator = HPUDSparkSpeculator(
        draft,
        max_num_reqs=2,
        linear=tracked_linear,
        backbone=backbone,
    )
    compute_logits_calls = []
    original_compute_logits = speculator.model.compute_draft_logits

    def tracked_compute_logits(sample_hidden):
        compute_logits_calls.append(tuple(sample_hidden.shape))
        return original_compute_logits(sample_hidden)

    speculator.model.compute_draft_logits = tracked_compute_logits
    proposals, confidence = speculator.propose_packed(
        anchors, positions, slots, active, context_done)

    assert len(backbone.calls) == 1
    call = backbone.calls[0]
    assert torch.equal(call[0], anchors)
    assert torch.equal(call[1], positions)
    assert torch.equal(call[2], slots)
    assert torch.equal(call[3], active)
    assert call[4] is context_done
    assert compute_logits_calls == [(10, 4)]
    assert draft.argmax_calls == [(2, 6)] * 5
    assert draft.embed_calls == [(2, )] * 5
    assert all(values[1] == 0 for values in draft.embed_inputs)
    assert draft.all_gather_calls == 0
    assert linear_calls == [(2, 7)] * 5
    assert proposals.shape == confidence.shape == (2, 5)
    assert torch.equal(proposals[~active], torch.full_like(proposals[~active], -1))
    assert speculator._active is None
    assert speculator._packed_inputs is None
    assert speculator.model._cached_hidden is None
    assert speculator.model._cached_logits is None


def test_propose_after_verify_orders_context_inputs_and_backbone(monkeypatch):
    monkeypatch.setattr(draft_batch, "_linear", _exact_linear)
    draft = _UpstreamMarkovDraft()
    events = []
    context_done = tuple(torch.tensor([index + 10], dtype=torch.int32) for index in range(3))
    batch_state = object()
    target_states = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    target_positions = torch.tensor([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]],
                                    dtype=torch.int32)
    request_slots = torch.tensor([0, 1], dtype=torch.int32)
    committed = torch.tensor([6, 6], dtype=torch.int64)
    output = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=torch.int64)
    output_count = torch.tensor([6, 6], dtype=torch.int64)
    draft_enabled = torch.tensor([True, True])
    active = torch.tensor([True, True])
    status = torch.tensor([0, 1], dtype=torch.int64)
    hidden = torch.zeros((2, 5, 4), dtype=torch.float32)
    logits = torch.zeros((2, 5, 6), dtype=torch.float32)
    backbone = _PackedBackbone(hidden, logits, events=events)

    def context_writer(draft_value, state, states, positions, slots, counts, valid):
        assert draft_value is draft
        assert state is batch_state
        assert states is target_states
        assert positions is target_positions
        assert slots is request_slots
        assert counts is committed
        assert valid.tolist() == [True, False]
        events.append("context")
        return context_done

    def inputs_builder(positions, verified, counts, verified_committed, draft_active):
        assert positions is target_positions
        assert verified is output
        assert counts is output_count
        assert verified_committed is committed
        assert draft_active.tolist() == [True, False]
        events.append("inputs")
        # Active row 0 uses an in-vocab anchor; inactive row 1 keeps an out-of-vocab poison that must be masked.
        return (torch.tensor([4, 999], dtype=torch.int64),
                torch.tensor([[30, 31, 32, 33, 34], [40, 41, 42, 43, 44]], dtype=torch.int32))

    speculator = HPUDSparkSpeculator(
        draft,
        max_num_reqs=2,
        linear=_exact_linear,
        backbone=backbone,
        context_writer=context_writer,
        inputs_builder=inputs_builder,
        batch_state=batch_state,
    )
    proposals, _confidence = speculator.propose_after_verify(
        target_states,
        target_positions,
        request_slots,
        committed,
        output,
        output_count,
        draft_enabled,
        active,
        status,
    )

    assert events == ["context", "inputs", "backbone"]
    assert backbone.calls[0][4] is context_done
    assert backbone.calls[0][3].tolist() == [True, False]
    assert backbone.calls[0][0].reshape(-1)[0].item() == 4
    assert backbone.calls[0][1][0].tolist() == [30, 31, 32, 33, 34]
    expected, _ = _legacy_markov_golden(draft, torch.tensor([4, 0]), hidden, logits, torch.tensor([True, False]),
                                        _exact_linear)
    assert proposals[0].tolist() == expected[0].tolist()
    assert proposals[1].tolist() == [-1] * 5


def test_propose_packed_cleans_references_and_is_reusable_after_backbone_error(monkeypatch):
    monkeypatch.setattr(draft_batch, "_linear", _exact_linear)
    draft = _UpstreamMarkovDraft()
    hidden = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4) / 9
    logits = torch.arange(2 * 5 * 6, dtype=torch.float32).reshape(2, 5, 6) / 13
    backbone = _PackedBackbone(hidden, logits, fail_first=True)
    speculator = HPUDSparkSpeculator(
        draft,
        max_num_reqs=2,
        linear=_exact_linear,
        backbone=backbone,
    )
    anchors = torch.tensor([[1], [2]], dtype=torch.int64)
    positions = torch.arange(10, dtype=torch.int32).reshape(2, 5)
    slots = torch.tensor([0, 1], dtype=torch.int32)
    active = torch.tensor([True, True])

    with pytest.raises(RuntimeError, match="packed backbone failure"):
        speculator.propose_packed(anchors, positions, slots, active, None)
    assert speculator._active is None
    assert speculator._packed_inputs is None
    assert speculator.model._cached_hidden is None
    assert speculator.model._cached_logits is None

    proposals, confidence = speculator.propose_packed(anchors, positions, slots, active, None)
    assert proposals.shape == confidence.shape == (2, 5)
    assert len(backbone.calls) == 2
    assert speculator._active is None
    assert speculator._packed_inputs is None
    assert speculator.model._cached_hidden is None
    assert speculator.model._cached_logits is None


@pytest.mark.parametrize("batch_size", [1, 3, 11, 32, 64])
def test_upstream_adapter_matches_legacy_golden_with_active_poison(monkeypatch, batch_size):
    monkeypatch.setattr(draft_batch, "_linear", _exact_linear)
    draft = _UpstreamMarkovDraft()
    active = torch.arange(batch_size).remainder(3) != 1
    anchors = torch.arange(batch_size, dtype=torch.int64).remainder(6)
    anchors = torch.where(active, anchors, torch.full_like(anchors, 99999))
    hidden = torch.arange(batch_size * 5 * 4, dtype=torch.float32).reshape(batch_size, 5, 4) / 7
    hidden = torch.where(active[:, None, None], hidden, torch.full_like(hidden, 12345.0))
    logits = torch.arange(batch_size * 5 * 6, dtype=torch.float32).reshape(batch_size, 5, 6) / 11
    logits = torch.where(active[:, None, None], logits, torch.full_like(logits, -54321.0))

    actual, confidence = draft_batch.sample_greedy_markov_batch(draft, anchors, hidden, logits, active)
    expected, expected_confidence = _legacy_markov_golden(
        draft, anchors, hidden, logits, active, _exact_linear)

    assert torch.equal(actual, expected)
    assert torch.equal(confidence, expected_confidence)
    assert torch.equal(actual[~active], torch.full_like(actual[~active], -1))
    if (~active).any():
        assert torch.count_nonzero(confidence[~active]) > 0


def test_cached_proposals_do_not_alias_speculator_buffers(monkeypatch):
    monkeypatch.setattr(draft_batch, "_linear", _exact_linear)
    draft = _UpstreamMarkovDraft()
    speculator = HPUDSparkSpeculator(draft, max_num_reqs=2, linear=_exact_linear)
    anchors = torch.tensor([5, 7], dtype=torch.int64)
    hidden = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    logits = torch.arange(2 * 5 * 6, dtype=torch.float32).reshape(2, 5, 6) / 9
    active = torch.tensor([True, False])

    first, first_confidence = speculator.propose_cached(anchors, hidden, logits, active)
    first_snapshot = first.clone()
    confidence_snapshot = first_confidence.clone()
    assert first.untyped_storage().data_ptr() != speculator.draft_tokens.untyped_storage().data_ptr()
    assert first_confidence.untyped_storage().data_ptr() != (
        speculator.draft_token_confidence_probs.untyped_storage().data_ptr())

    second, second_confidence = speculator.propose_cached(
        torch.tensor([1, 2], dtype=torch.int64),
        hidden + 3,
        logits.flip(-1),
        torch.tensor([False, True]),
    )
    assert torch.equal(first, first_snapshot)
    assert torch.equal(first_confidence, confidence_snapshot)
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    assert first_confidence.untyped_storage().data_ptr() != second_confidence.untyped_storage().data_ptr()


def test_cached_proposal_cleans_state_after_setup_and_loop_errors(monkeypatch):
    monkeypatch.setattr(draft_batch, "_linear", _exact_linear)
    draft = _UpstreamMarkovDraft()
    speculator = HPUDSparkSpeculator(draft, max_num_reqs=2, linear=_exact_linear)
    anchors = torch.tensor([5, 7], dtype=torch.int64)
    hidden = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    logits = torch.arange(2 * 5 * 6, dtype=torch.float32).reshape(2, 5, 6) / 9
    active = torch.tensor([True, False])
    original_set_cached = speculator.model.set_cached

    def fail_after_cache(hidden_value, logits_value):
        original_set_cached(hidden_value, logits_value)
        raise RuntimeError("cache setup failure")

    with monkeypatch.context() as patch:
        patch.setattr(speculator.model, "set_cached", fail_after_cache)
        with pytest.raises(RuntimeError, match="cache setup failure"):
            speculator.propose_cached(anchors, hidden, logits, active)
    assert speculator._active is None
    assert speculator.model._cached_hidden is None
    assert speculator.model._cached_logits is None
    speculator.propose_cached(anchors, hidden, logits, active)

    def fail_after_anchor_write(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("upstream loop failure")

    with monkeypatch.context() as patch:
        patch.setattr(speculator, "_generate_draft", fail_after_anchor_write)
        with pytest.raises(RuntimeError, match="upstream loop failure"):
            speculator.propose_cached(anchors, hidden, logits, active)
    assert speculator._active is None
    assert speculator.model._cached_hidden is None
    assert speculator.model._cached_logits is None
    proposals, confidence = speculator.propose_cached(anchors, hidden, logits, active)
    assert proposals.shape == confidence.shape == (2, 5)


def test_cached_inputs_reject_speculator_device_mismatch():
    draft = _UpstreamMarkovDraft()
    speculator = HPUDSparkSpeculator(draft, device=torch.device("meta"), max_num_reqs=1, linear=_exact_linear)
    with pytest.raises(ValueError, match="speculator device"):
        speculator.propose_cached(
            torch.tensor([5], dtype=torch.int64),
            torch.zeros((1, 5, 4)),
            torch.zeros((1, 5, 6)),
            torch.tensor([True]),
        )
    assert speculator._active is None
    assert speculator.model._cached_hidden is None
    assert speculator.model._cached_logits is None


def test_upstream_adapter_preserves_tp_hook_call_counts(monkeypatch):
    monkeypatch.setattr(draft_batch, "_linear", _exact_linear)
    draft = _UpstreamMarkovDraft()
    batch_size = 3
    anchors = torch.tensor([5, 7, 1], dtype=torch.int64)
    hidden = torch.arange(batch_size * 5 * 4, dtype=torch.float32).reshape(batch_size, 5, 4)
    logits = torch.arange(batch_size * 5 * 6, dtype=torch.float32).reshape(batch_size, 5, 6) / 9
    active = torch.tensor([True, False, True])
    linear_calls = []

    def tracked_linear(value, owner):
        linear_calls.append(tuple(value.shape))
        return _exact_linear(value, owner)

    monkeypatch.setattr(draft_batch, "_linear", tracked_linear)
    draft_batch.sample_greedy_markov_batch(draft, anchors, hidden, logits, active)

    assert draft.argmax_calls == [(batch_size, 6)] * 5
    assert draft.embed_calls == [(batch_size, )] * 5
    assert draft.all_gather_calls == 0
    assert linear_calls == [(batch_size, 7)] * 5


class _CountingBackend:
    def __init__(self):
        self.count = 0

    def __call__(self, graph_module, example_inputs):
        del example_inputs
        self.count += 1
        return graph_module.forward


def test_packed_markov_sampler_fullgraph_reuses_one_graph(monkeypatch):
    # Dynamo's stream tracker must not acquire a visible-but-busy HPU in CPU tests.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(draft_batch, "_linear", _exact_linear)
    draft = _UpstreamMarkovDraft(track_calls=False)
    sampler = draft_batch.PackedDraftMarkovSampler(draft, batch_capacity=3)
    backend = _CountingBackend()
    torch._dynamo.reset()
    compiled = torch.compile(sampler, backend=backend, fullgraph=True, dynamic=False)
    anchors = torch.tensor([5, 4, 1], dtype=torch.int64)
    hidden = torch.zeros((3, 5, 4))
    logits = torch.zeros((3, 5, 6))
    active = torch.tensor([True, False, True])

    first, _ = compiled(anchors, hidden, logits, active)
    anchors.copy_(torch.tensor([2, 3, 0]))
    active.copy_(torch.tensor([False, True, True]))
    second, _ = compiled(anchors, hidden, logits, active)

    assert backend.count == 1
    assert first.shape == second.shape == (3, 5)
    assert torch.equal(second[~active], torch.full_like(second[~active], -1))


def test_packed_round_fullgraph_runs_real_backbone_callback_once(monkeypatch):
    # Dynamo's stream tracker must not acquire a visible-but-busy HPU in CPU tests.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(draft_batch, "_linear", _exact_linear)
    draft = _UpstreamMarkovDraft(track_calls=False)
    hidden = torch.zeros((3, 5, 4), dtype=torch.float32)
    logits = torch.zeros((3, 5, 6), dtype=torch.float32)
    backbone = _PackedBackbone(hidden, logits, track_calls=False)
    round_module = draft_batch.PackedDraftRound(
        draft,
        SimpleNamespace(capacity=3),
        batch_capacity=3,
        backbone=backbone,
        compile_graph=False,
    )
    backend = _CountingBackend()
    torch._dynamo.reset()
    compiled = torch.compile(round_module, backend=backend, fullgraph=True, dynamic=False)
    anchors = torch.tensor([[5], [4], [1]], dtype=torch.int64)
    positions = torch.arange(15, dtype=torch.int32).reshape(3, 5)
    slots = torch.tensor([0, 1, 2], dtype=torch.int32)
    active = torch.tensor([True, False, True])

    first, _ = compiled(anchors, positions, slots, active)
    anchors.copy_(torch.tensor([[2], [3], [0]]))
    active.copy_(torch.tensor([False, True, True]))
    second, _ = compiled(anchors, positions, slots, active)

    assert backend.count == 1
    assert first.shape == second.shape == (3, 5)
    assert torch.equal(second[~active], torch.full_like(second[~active], -1))
    assert round_module.backbone is backbone
    assert round_module.markov is not None
    assert round_module.speculator._backbone is backbone


def test_real_packed_backbone_runs_three_layers_without_request_loops(monkeypatch):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("swa", torch.zeros(256, 8, dtype=torch.uint8))

    class Moe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, value, image_mask, decode=False):
            self.calls.append(tuple(value.shape))
            return value

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.moe = Moe()
            self.draft_mhc = False
            self.eps = 1e-6
            self.hc_eps = 1e-6
            self.iterations = 20
            self.weights = SimpleNamespace(
                hc_attn_fn=object(), hc_attn_scale=object(), hc_attn_base=object(),
                hc_ffn_fn=object(), hc_ffn_scale=object(), hc_ffn_base=object(),
                attn_norm=SimpleNamespace(weight=torch.ones(8)),
                ffn_norm=SimpleNamespace(weight=torch.ones(8)),
            )

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer(), Layer()])
            self.noise = 99
            self.embedding = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8) / 128
            last = SimpleNamespace(
                markov_head=SimpleNamespace(embed=object(), head=SimpleNamespace(weight=torch.zeros(6, 8))),
                confidence_head=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(1, 16))),
            )
            self.weights = SimpleNamespace(get_submodule=lambda name: last)
            self.compute_logits_calls = 0

        def embed_input_ids(self, ids):
            return F.embedding(ids, self.embedding)

        def compute_logits_local(self, hidden):
            self.compute_logits_calls += 1
            return hidden[:, :1].expand(-1, 6)

        def _embed(self, previous, embedding):
            return F.one_hot(previous.reshape(-1).long().remainder(8), 8).float()

        def _global_argmax(self, scores):
            return scores.argmax(-1)

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
    draft = Draft()
    bank = draft_batch.DraftBatchState(draft, capacity=2)
    backbone = draft_batch.PackedDraftBackbone(draft, bank)
    context_done = tuple(torch.tensor([index + 1], dtype=torch.int32) for index in range(3))
    hidden, logits = backbone(
        torch.tensor([[3], [1]], dtype=torch.int64),
        torch.tensor([[100, 101, 102, 103, 104], [200, 201, 202, 203, 204]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([True, True]),
        context_done=context_done,
    )
    assert hidden.shape == (2, 5, 8)
    assert logits.shape == (2, 5, 6)
    assert [call[0] for call in attention_calls] == [0, 1, 2]
    assert all(shape == (10, 8) for _, shape, _ in attention_calls)
    assert all(call[2] is expected for call, expected in zip(attention_calls, context_done, strict=True))

    attention_calls.clear()
    backbone(
        torch.tensor([[3], [1]], dtype=torch.int64),
        torch.tensor([[100, 101, 102, 103, 104], [200, 201, 202, 203, 204]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([True, True]),
    )
    assert [call[2] for call in attention_calls] == [None, None, None]
    assert all(len(layer.moe.calls) == 2 for layer in draft.layers)

    attention_calls.clear()
    backbone_calls = []
    original_forward = backbone.forward

    def tracked_forward(*args, **kwargs):
        backbone_calls.append(kwargs.get("context_done"))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(backbone, "forward", tracked_forward)
    speculator = HPUDSparkSpeculator(
        draft,
        max_num_reqs=2,
        linear=_exact_linear,
        backbone=backbone,
    )
    proposals, confidence = speculator.propose_packed(
        torch.tensor([[3], [1]], dtype=torch.int64),
        torch.tensor([[100, 101, 102, 103, 104], [200, 201, 202, 203, 204]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([True, True]),
        context_done,
    )
    assert len(backbone_calls) == 1
    assert backbone_calls[0] is context_done
    assert draft.compute_logits_calls == 3
    assert [call[0] for call in attention_calls] == [0, 1, 2]
    assert proposals.shape == confidence.shape == (2, 5)


@pytest.mark.parametrize("layer_index", range(3))
def test_context_completion_is_only_the_accepted_context_decode_dependency(monkeypatch, layer_index):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("swa", torch.zeros(256, 528, dtype=torch.uint8))
            self.swa_only_main = torch.full((1, 528), 7, dtype=torch.uint8)
            self.weights = SimpleNamespace(
                q_norm=SimpleNamespace(weight=torch.ones(8)),
                kv_norm=SimpleNamespace(weight=torch.ones(8)),
                attn_sink=torch.zeros(1),
            )
            self.eps = 1e-6
            self.scale = 1.0

        def _project_qkv_input(self, value):
            return value, value

        def project_query(self, value, positions, request_batch=False):
            assert request_batch
            return value

        def _rope(self, value, positions, request_batch=False):
            assert request_batch
            return value

        def _finish_output(self, value, positions, request_batch=False):
            assert request_batch
            return value

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer(), Layer()])

    decode_calls = []

    def decode(packed, unused_main, indices):
        decode_calls.append((packed, unused_main, indices))
        return torch.zeros((indices.shape[1], 512), dtype=torch.bfloat16)

    monkeypatch.setattr(draft_batch, "rms_norm", lambda value, weight, eps: value)
    monkeypatch.setattr(draft_batch, "pack_swa", lambda value: value.to(torch.uint8))
    monkeypatch.setattr(draft_batch, "decode_draft_swa_rows", decode)
    monkeypatch.setattr(
        torch.ops.custom_op,
        "custom_deepseek_v4_sparse_attn_bf16_gaudi2",
        lambda *args: (torch.ones((5, 8)), None, None),
        raising=False,
    )
    draft = Draft()
    bank = draft_batch.DraftBatchState(draft, capacity=1)
    value = torch.arange(5 * 8, dtype=torch.float32).reshape(5, 8)
    positions = torch.arange(10, 15, dtype=torch.int32).reshape(1, 5)
    slots = torch.tensor([0], dtype=torch.int32)
    active = torch.tensor([True])

    outputs = []
    for completion in (
        torch.tensor([1, 2, 9], dtype=torch.int32),
        torch.tensor([4, 5, 6], dtype=torch.int32),
    ):
        outputs.append(draft_batch.draft_attention_batch(
            draft, layer_index, value, positions, slots, active, bank, completion))

    assert torch.equal(outputs[0], outputs[1])
    assert decode_calls[0][1].tolist() == [[9] * 528]
    assert decode_calls[2][1].tolist() == [[6] * 528]
    assert decode_calls[0][1].is_contiguous() and decode_calls[2][1].is_contiguous()
    attention = draft.layers[layer_index].attention
    assert decode_calls[1][1] is attention.swa_only_main
    assert decode_calls[3][1] is attention.swa_only_main

    draft_batch.draft_attention_batch(
        draft, layer_index, value, positions, slots, active, bank)
    assert decode_calls[4][1] is attention.swa_only_main
    assert decode_calls[5][1] is attention.swa_only_main


class _Event:
    def record(self, stream):
        pass

    def synchronize(self):
        pass

    def query(self):
        return True


class _Ticket:
    def __init__(self, batch_size):
        self.record = torch.empty((batch_size, 16), dtype=torch.int64)
        self.host = torch.empty_like(self.record)
        self.completion = _Event()


class _Ring:
    def __init__(self):
        self.ticket = None
        self.staged = False

    def acquire(self, **kwargs):
        self.ticket = _Ticket(len(kwargs["active"]))
        return self.ticket

    def stage(self, ticket):
        ticket.host.copy_(ticket.record)
        self.staged = True

    def consume(self, ticket):
        rows = ticket.host.tolist()
        return SimpleNamespace(
            committed=tuple(row[1] for row in rows),
            output=tuple(tuple(token for token in row[4:4 + row[2]]) for row in rows),
            draft=tuple(tuple(token for token in row[10:10 + row[3]]) for row in rows),
        )

    def release(self, ticket):
        pass


def test_runner_facing_stage_returns_commit_and_next_proposals(monkeypatch):
    from vllm_gaudi.ops.deepseek_v41_batch_speculation import DSparkBatchMetadata

    metadata = DSparkBatchMetadata.build(
        requests=("request-a", "request-b"),
        target_query_lengths=[6, 6],
        draft_token_ids=[[1, 2, 3, 4, 5], [8, 99, 100, 101, 102]],
        request_slots=[0, 1],
        slot_generations=[4, 9],
        remaining_output_budget=[64, 64],
        target_start_positions=[10, 12],
        round_generation=1,
    )
    target = SimpleNamespace(
        token_ids=torch.tensor([[1, 2, 3, 4, 5, 6], [8, 9, 10, 11, 12, 13]], dtype=torch.int64),
        payload=(torch.zeros((2, 6, 8), dtype=torch.bfloat16), torch.zeros((12, 8), dtype=torch.bfloat16)),
    )
    committed_seen = []
    monkeypatch.setattr(draft_batch, "insert_context_batch",
                        lambda draft, bank, states, positions, slots, committed, active:
                        committed_seen.append(committed.tolist()))

    class Backbone:
        def __call__(self, anchors, positions, slots, active):
            return torch.zeros((2, 5, 4)), torch.zeros((2, 5, 6))

    class Markov:
        def __call__(self, anchors, hidden, logits, active):
            return torch.tensor([[20, 21, 22, 23, 24], [30, 31, 32, 33, 34]]), torch.zeros((2, 5))

    stage = draft_batch.DSparkBatchDraftStage(
        draft=SimpleNamespace(),
        verify_ring=_Ring(),
        backbone=Backbone(),
        markov=Markov(),
        max_model_len=128,
    )
    result = stage.verify_and_propose(metadata, target)
    assert result.accepted.tolist() == [6, 2]
    assert result.output_token_ids == ((1, 2, 3, 4, 5, 6), (8, 9))
    assert result.draft_token_ids == ((20, 21, 22, 23, 24), (30, 31, 32, 33, 34))
    assert committed_seen == [[6, 2]]
    assert result.completion_event.query()


def test_transaction_only_delegates_context_and_proposal_to_speculator(monkeypatch):
    calls = {}
    output = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64)
    committed = torch.tensor([6], dtype=torch.int64)
    output_count = torch.tensor([6], dtype=torch.int64)
    draft_enabled = torch.tensor([True])
    status = torch.tensor([0], dtype=torch.int64)
    proposal_ids = torch.tensor([[20, 21, 22, 23, 24]], dtype=torch.int64)
    confidence = torch.zeros((1, 5), dtype=torch.float32)

    def verify(target_ids, proposals, control, active, binding_sidecar):
        del proposals, control, active, binding_sidecar
        return (target_ids, output, committed, output_count, torch.empty(0), draft_enabled, status)

    class Speculator:
        def propose_after_verify(self, *args):
            calls["speculator"] = args
            return proposal_ids, confidence

    def pack(control, committed_value, output_count_value, output_value, draft_ids, draft_active,
             status_value, binding_sidecar):
        calls["pack"] = (
            control, committed_value, output_count_value, output_value, draft_ids,
            draft_active, status_value, binding_sidecar)
        return torch.zeros((1, 16), dtype=torch.int64)

    def fail_context(*args, **kwargs):
        del args, kwargs
        raise AssertionError("transaction must not write draft context directly")

    def fail_inputs(*args, **kwargs):
        del args, kwargs
        raise AssertionError("transaction must not build draft inputs directly")

    monkeypatch.setattr(draft_batch, "verify_control_from_target_batch", verify)
    monkeypatch.setattr(draft_batch, "pack_record_batch", pack)
    monkeypatch.setattr(draft_batch, "insert_context_batch", fail_context)
    monkeypatch.setattr(draft_batch, "draft_inputs_after_commit", fail_inputs)
    speculator = Speculator()
    transaction = draft_batch.PackedDraftTransaction(
        SimpleNamespace(),
        SimpleNamespace(capacity=1),
        batch_capacity=1,
        compile_graph=False,
        speculator=speculator,
    )
    target_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64)
    target_states = torch.zeros((1, 6, 4), dtype=torch.float32)
    positions = torch.arange(6, dtype=torch.int32).reshape(1, 6)
    request_slots = torch.tensor([0], dtype=torch.int32)
    control = torch.zeros((1, 7), dtype=torch.int64)
    proposals = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    active = torch.tensor([True])

    record, actual_committed, actual_output, actual_count, actual_proposals, actual_status = transaction(
        target_ids, target_states, positions, request_slots, control, proposals, active)

    assert record.shape == (1, 16)
    assert torch.equal(actual_committed, committed)
    assert torch.equal(actual_output, output)
    assert torch.equal(actual_count, output_count)
    assert torch.equal(actual_proposals, proposal_ids)
    assert torch.equal(actual_status, status)
    assert len(calls["speculator"]) == 9
    assert calls["speculator"][0] is target_states
    assert calls["speculator"][1] is positions
    assert calls["speculator"][2] is request_slots
    assert calls["speculator"][3] is committed
    assert calls["speculator"][4] is output
    assert calls["speculator"][5] is output_count
    assert calls["speculator"][6] is draft_enabled
    assert calls["speculator"][7] is active
    assert calls["speculator"][8] is status
    assert calls["pack"][4] is proposal_ids
    assert calls["pack"][5].tolist() == [True]


def test_fixed_transaction_compiles_fullgraph_and_masks_rejected_context(monkeypatch):
    # Dynamo's stream tracker must not acquire a visible-but-busy HPU in CPU tests.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    from vllm_gaudi.ops.deepseek_v41_batch_verify import batch_binding_sidecar, batch_control_metadata

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("swa", torch.zeros(256, 8, dtype=torch.uint8))
            self.weights = SimpleNamespace(wkv=object(), kv_norm=SimpleNamespace(weight=torch.ones(8)))
            self.eps = 1e-6
            self.linear = lambda value, weight: value

        def _rope(self, value, positions, request_batch=False):
            assert request_batch
            return value

    class Moe(torch.nn.Module):
        def forward(self, value, image_mask, decode=False):
            return value

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.moe = Moe()
            self.draft_mhc = False
            self.eps = 1e-6
            self.hc_eps = 1e-6
            self.iterations = 20
            self.weights = SimpleNamespace(
                hc_attn_fn=object(), hc_attn_scale=object(), hc_attn_base=object(),
                hc_ffn_fn=object(), hc_ffn_scale=object(), hc_ffn_base=object(),
                attn_norm=SimpleNamespace(weight=torch.ones(8)),
                ffn_norm=SimpleNamespace(weight=torch.ones(8)),
            )

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer(), Layer()])
            self.noise = 99
            self.eps = 1e-6
            self.embedding = torch.arange(128 * 8, dtype=torch.float32).reshape(128, 8) / 128
            first = SimpleNamespace(main_proj=object(), main_norm=SimpleNamespace(weight=torch.ones(8)))
            last = SimpleNamespace(
                markov_head=SimpleNamespace(embed=object(), head=SimpleNamespace(weight=torch.zeros(6, 8))),
                confidence_head=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(1, 16))),
            )
            self.weights = SimpleNamespace(get_submodule=lambda name: first if name == "0" else last)

        def embed_input_ids(self, ids):
            return F.embedding(ids, self.embedding)

        def compute_logits_local(self, hidden):
            return hidden[:, :1].expand(-1, 6)

        def _embed(self, previous, embedding):
            return F.one_hot(previous.reshape(-1).long().remainder(8), 8).float()

        def _global_argmax(self, scores):
            return scores.argmax(-1)

    def hc_pre(residual, previous, *args):
        tokens = residual.shape[0]
        return residual[:, 0], torch.zeros((tokens, 4)), torch.zeros((tokens, 4)), torch.zeros((tokens, 4, 4))

    monkeypatch.setattr(
        draft_batch, "_linear", lambda value, owner: value[:, :1] if value.shape[1] == 16 else value)
    monkeypatch.setattr(draft_batch, "hc_pre", hc_pre)
    monkeypatch.setattr(draft_batch, "hc_post", lambda value, residual, post, comb: residual)
    monkeypatch.setattr(draft_batch, "rms_norm", lambda value, weight, eps: value)
    monkeypatch.setattr(draft_batch, "pack_swa", lambda value: value.to(torch.uint8))
    monkeypatch.setattr(draft_batch, "draft_attention_batch", lambda draft, index, value, *args: value)
    draft = Draft()
    bank = draft_batch.DraftBatchState(draft, capacity=2)
    transaction = draft_batch.PackedDraftTransaction(draft, bank, compile_graph=False)
    control = batch_control_metadata(
        round_generation=1,
        request_slots=[0, 1],
        slot_generations=[4, 9],
        target_counts=[6, 6],
        proposal_counts=[5, 5],
        remaining_output_budget=[64, 64],
        target_start_positions=[10, 12],
        max_model_len=128,
        need_sample=[1, 1],
        active=[True, True],
    )
    binding_sidecar = batch_binding_sidecar(control)
    target_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [8, 9, 10, 11, 12, 13]], dtype=torch.int64)
    proposals = torch.tensor([[1, 2, 3, 4, 5], [8, 99, 100, 101, 102]], dtype=torch.int64)
    states = torch.arange(96, dtype=torch.uint8).reshape(2, 6, 8)
    positions = torch.tensor([[10, 11, 12, 13, 14, 15], [12, 13, 14, 15, 16, 17]], dtype=torch.int32)
    torch._dynamo.reset()
    compiled = torch.compile(transaction._forward, backend="eager", fullgraph=True, dynamic=False)
    values = compiled(
        target_ids,
        states,
        positions,
        torch.tensor([0, 1], dtype=torch.int32),
        control,
        proposals,
        torch.tensor([True, True]),
        binding_sidecar,
    )
    record, committed, output, output_count, draft_ids, status = values
    assert record.shape == (2, 16)
    assert committed.tolist() == [6, 2]
    assert output_count.tolist() == [6, 2]
    assert status.tolist() == [0, 0]
    assert draft_ids.shape == (2, 5)
    ring = bank.ring(0)
    assert torch.equal(ring[10:16], states[0])
    assert torch.equal(ring[268:270], states[1, :2])
    assert torch.count_nonzero(ring[270:274]) == 0


def test_short_context_uses_negative_sparse_indices_not_zero_rows():
    rows = draft_batch.draft_attention_rows(
        torch.tensor([[2, 3, 4, 5, 6], [200, 201, 202, 203, 204]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([True, True]),
        ring_rows=512,
    )
    assert (rows.sparse_indices[0, :126] == -1).all()
    assert rows.sparse_indices[0, 126:128].tolist() == [126, 127]
    assert rows.sparse_indices[0, 128:133].tolist() == [256, 257, 258, 259, 260]


def test_runner_stage_defaults_to_unified_transaction_and_requires_backing_state(monkeypatch):
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("swa", torch.zeros(256, 8, dtype=torch.uint8))

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.draft_mhc = False

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer(), Layer()])

    draft = Draft()
    bank = draft_batch.DraftBatchState(draft, capacity=2)
    stage = draft_batch.DSparkBatchDraftStage(
        draft,
        batch_state=bank,
        verify_ring=_Ring(),
        max_model_len=128,
    )
    stage._ensure_runtime(torch.device("cpu"))
    assert isinstance(stage.transaction, draft_batch.PackedDraftTransaction)

    no_state = draft_batch.DSparkBatchDraftStage(draft, verify_ring=_Ring(), max_model_len=128)
    with pytest.raises(RuntimeError, match="runner-owned DraftBatchState"):
        no_state._ensure_runtime(torch.device("cpu"))


@pytest.mark.parametrize("capacity,bucket,lanes", [
    (11, 1, (10, )), (11, 4, (2, 10)), (32, 4, (3, 31)), (64, 2, (7, 63)),
])
def test_compact_speculator_keeps_global_context_and_restores_candidate_layout(
        monkeypatch, capacity, bucket, lanes):
    from vllm_gaudi.ops.deepseek_v41_draft_buckets import select_draft_bucket

    monkeypatch.setattr(draft_batch, "_linear", _exact_linear)
    active = torch.zeros(capacity, dtype=torch.bool)
    active[list(lanes)] = True
    slots = torch.full((capacity, ), -1, dtype=torch.int32)
    # Logical owners deliberately differ from their fixed execution lanes.
    for index, lane in enumerate(lanes):
        slots[lane] = capacity - index - 1
    plan = select_draft_bucket(active.tolist(), slots.tolist(), (bucket, capacity))
    indices = torch.tensor(plan.source_lanes, dtype=torch.int64)
    hidden = torch.zeros((bucket, 5, 4))
    logits = torch.arange(bucket * 5 * 6, dtype=torch.float32).reshape(bucket, 5, 6)
    backbone = _PackedBackbone(hidden, logits)
    completion = tuple(torch.tensor([layer + 1]) for layer in range(3))
    context_calls = []

    def writer(draft, state, states, positions, logical_slots, committed, valid):
        context_calls.append((states.shape, logical_slots.clone(), committed.clone(), valid.clone()))
        return completion

    speculator = HPUDSparkSpeculator(
        _UpstreamMarkovDraft(), max_num_reqs=bucket, linear=_exact_linear,
        backbone=backbone, context_writer=writer, inputs_builder=draft_batch.draft_inputs_after_commit,
        batch_state=SimpleNamespace(capacity=capacity))
    positions = torch.arange(capacity * 6, dtype=torch.int32).reshape(capacity, 6)
    output = torch.ones((capacity, 6), dtype=torch.int64)
    committed = torch.where(active, 2, 0).to(torch.int64)
    states = torch.zeros((capacity, 6, 4))
    proposals, confidence = speculator.propose_after_verify(
        states, positions, slots, committed, output, committed, active, active,
        torch.zeros(capacity, dtype=torch.int64), execution_lanes=indices)

    assert len(context_calls) == len(backbone.calls) == 1
    assert context_calls[0][0] == (capacity, 6, 4)
    assert torch.equal(context_calls[0][1], slots)
    assert torch.equal(context_calls[0][3], active)
    assert backbone.calls[0][0].shape == (bucket, 1)
    assert backbone.calls[0][1].shape == (bucket, 5)
    assert backbone.calls[0][2].tolist() == list(plan.logical_slots)
    assert backbone.calls[0][4] is completion
    assert proposals.shape == confidence.shape == (capacity, 5)
    assert torch.all(proposals[~active] == -1)
    assert torch.all(proposals[active] >= 0)
    assert speculator._packed_inputs is None


def test_compact_proposal_fullgraph_reuses_bucket_when_lanes_and_owners_change(monkeypatch):
    # Dynamo's stream tracker must not acquire a visible-but-busy HPU in CPU tests.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    monkeypatch.setattr(draft_batch, "_linear", _exact_linear)
    capacity, bucket = 11, 2
    speculator = HPUDSparkSpeculator(
        _UpstreamMarkovDraft(track_calls=False), max_num_reqs=bucket, linear=_exact_linear,
        backbone=_PackedBackbone(torch.zeros((bucket, 5, 4)), torch.zeros((bucket, 5, 6)), track_calls=False),
        context_writer=lambda *args: (args[2].sum().reshape(1), ) * 3,
        inputs_builder=draft_batch.draft_inputs_after_commit, batch_state=SimpleNamespace(capacity=capacity))
    backend = _CountingBackend()
    torch._dynamo.reset()
    compiled = torch.compile(speculator.propose_after_verify, backend=backend, fullgraph=True, dynamic=False)
    states = torch.zeros((capacity, 6, 4))
    positions = torch.arange(capacity * 6, dtype=torch.int32).reshape(capacity, 6)
    slots = torch.full((capacity, ), -1, dtype=torch.int32)
    active = torch.zeros(capacity, dtype=torch.bool)
    counts = torch.zeros(capacity, dtype=torch.int64)
    indices = torch.tensor([3, 10], dtype=torch.int64)
    output = torch.ones((capacity, 6), dtype=torch.int64)
    status = torch.zeros(capacity, dtype=torch.int64)
    active[[3, 10]] = True
    slots[[3, 10]] = torch.tensor([9, 1], dtype=torch.int32)
    counts[[3, 10]] = 2
    first, _ = compiled(states, positions, slots, counts, output, counts, active, active, status, indices)
    active.zero_()
    active[[1, 7]] = True
    slots.fill_(-1)
    slots[[1, 7]] = torch.tensor([10, 4], dtype=torch.int32)
    counts.zero_()
    counts[[1, 7]] = 2
    indices.copy_(torch.tensor([1, 7]))
    second, _ = compiled(states, positions, slots, counts, output, counts, active, active, status, indices)
    assert backend.count == 1
    assert torch.all(first[[3, 10]] >= 0)
    assert torch.all(second[~active] == -1)
    assert torch.all(second[active] >= 0)
