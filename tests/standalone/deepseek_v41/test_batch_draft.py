# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for packed DSpark draft body, Markov and context ownership."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops import deepseek_v41_batch_draft as draft_batch


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

        def embed_input_ids(self, ids):
            return F.embedding(ids, self.embedding)

        def compute_logits_local(self, hidden):
            return hidden[:, :1].expand(-1, 6)

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


def test_fixed_transaction_compiles_fullgraph_and_masks_rejected_context(monkeypatch):
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
