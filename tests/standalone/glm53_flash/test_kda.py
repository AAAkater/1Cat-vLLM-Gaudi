# SPDX-License-Identifier: Apache-2.0
"""CPU numerical tests for the portable Glm5Next KDA math.

Kept under tests/standalone so running this file does not require the HPU
fixtures in tests/unit_tests/conftest.py.  No local pytest configuration is
needed.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gaudi.ops.hpu_kda_pytorch import (
    hpu_chunk_kda,
    hpu_fused_recurrent_kda,
    hpu_kda_gate,
)


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def _inputs(
    batch: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(batch, seq_len, heads, head_dim, generator=gen, dtype=torch.float32).to(dtype)
    k = torch.randn(batch, seq_len, heads, head_dim, generator=gen, dtype=torch.float32).to(dtype)
    v = torch.randn(batch, seq_len, heads, head_dim, generator=gen, dtype=torch.float32).to(dtype)
    g = (-torch.sigmoid(torch.randn(batch, seq_len, heads, head_dim, generator=gen)) * 0.25).to(dtype)
    beta = torch.sigmoid(torch.randn(batch, seq_len, heads, generator=gen)).to(dtype)
    return q, k, v, g, beta


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(torch.sum(x.square(), dim=-1, keepdim=True) + 1e-6)


@torch.no_grad()
def _matrix_oracle_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    scale: float | None,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent matrix/vector oracle for one sequence.

    q/k/v/g are [T, H, D], beta is [T, H], and state is [H, V, K].  Each head
    is updated with ordinary matrix-vector products and an explicit outer
    product; this intentionally shares no vectorized implementation with the
    operator under test.
    """
    seq_len, heads, head_dim = q.shape
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    gf = g.to(torch.float32)
    bf = beta.to(torch.float32)
    if use_qk_l2norm:
        qf = _l2norm(qf)
        kf = _l2norm(kf)
    qf = qf * (head_dim**-0.5 if scale is None else scale)

    if initial_state is None:
        state = torch.zeros(heads, head_dim, head_dim, dtype=torch.float32, device=q.device)
    else:
        state = initial_state.to(torch.float32).clone()

    out = torch.empty(seq_len, heads, head_dim, dtype=torch.float32, device=q.device)
    for token in range(seq_len):
        decayed = state * torch.exp(gf[token]).unsqueeze(1)
        next_state = torch.empty_like(state)
        for head in range(heads):
            prediction = decayed[head].matmul(kf[token, head])
            residual = (vf[token, head] - prediction) * bf[token, head]
            next_state[head] = decayed[head] + torch.outer(residual, kf[token, head])
            out[token, head] = next_state[head].matmul(qf[token, head])
        state = next_state
    return out, state


@torch.no_grad()
def _matrix_oracle_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    scale: float | None,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, heads, head_dim = q.shape
    outputs = []
    states = []
    for batch_idx in range(batch):
        state_i = None if initial_state is None else initial_state[batch_idx]
        out_i, state_i = _matrix_oracle_sequence(
            q[batch_idx],
            k[batch_idx],
            v[batch_idx],
            g[batch_idx],
            beta[batch_idx],
            state_i,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
        )
        outputs.append(out_i)
        states.append(state_i)
    return torch.stack(outputs), torch.stack(states)


def test_kda_gate_matches_per_channel_formula_and_bounds():
    torch.manual_seed(4)
    heads, head_dim = 3, 4
    raw_g = torch.randn(2, 5, heads, head_dim)
    raw_g[0, 0, 0, 0] = 80.0
    raw_g[0, 1, 1, 1] = -80.0
    A_log = torch.log(torch.tensor([0.25, 1.0, 4.0]))
    dt_bias = torch.linspace(-0.5, 0.75, heads * head_dim).reshape(heads, head_dim)
    lower_bound = -3.5

    expected = lower_bound * torch.sigmoid(
        A_log.exp().view(1, 1, heads, 1) * (raw_g.float() + dt_bias.float())
    )
    actual = hpu_kda_gate(raw_g, A_log, dt_bias, lower_bound, heads, head_dim)
    flattened = hpu_kda_gate(raw_g.flatten(-2), A_log, dt_bias, lower_bound, heads, head_dim)
    ambiguous_leading = torch.randn(heads, heads * head_dim)
    expected_ambiguous = lower_bound * torch.sigmoid(
        A_log.exp().view(1, heads, 1)
        * (ambiguous_leading.reshape(heads, heads, head_dim) + dt_bias)
    )
    actual_ambiguous = hpu_kda_gate(
        ambiguous_leading,
        A_log,
        dt_bias,
        lower_bound,
        heads,
        head_dim,
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(flattened, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_ambiguous, expected_ambiguous, atol=1e-6, rtol=1e-6)
    assert torch.all(actual >= lower_bound)
    assert torch.all(actual <= 0)

    with pytest.raises(ValueError, match="non-positive"):
        hpu_kda_gate(raw_g, A_log, dt_bias, 1.0, heads, head_dim)


def test_recurrent_matches_matrix_oracle_with_custom_scale_without_l2norm():
    q, k, v, g, beta = _inputs(2, 5, 2, 4, seed=10)
    initial_state = torch.randn(2, 2, 4, 4, generator=torch.Generator().manual_seed(11)) * 0.2
    scale = 0.37

    expected_out, expected_state = _matrix_oracle_batch(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        scale=scale,
        use_qk_l2norm=False,
    )
    initial_before = initial_state.clone()
    actual_out, actual_state = hpu_fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=False,
        scale=scale,
    )

    torch.testing.assert_close(actual_out, expected_out, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(initial_state, initial_before, atol=0, rtol=0)


def test_recurrent_and_chunk_match_matrix_oracle_for_batch():
    q, k, v, g, beta = _inputs(2, 11, 2, 5, seed=20)
    initial_state = torch.randn(2, 2, 5, 5, generator=torch.Generator().manual_seed(21)) * 0.15
    expected_out, expected_state = _matrix_oracle_batch(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        scale=None,
        use_qk_l2norm=True,
    )

    recurrent_out, recurrent_state = hpu_fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state.clone(),
        use_qk_l2norm_in_kernel=True,
    )
    chunk_out, chunk_state = hpu_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state.clone(),
        chunk_size=4,
    )

    torch.testing.assert_close(recurrent_out, expected_out, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(recurrent_state, expected_state, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(chunk_out, expected_out, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(chunk_state, expected_state, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(chunk_out, recurrent_out, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(chunk_state, recurrent_state, atol=3e-6, rtol=3e-6)


def test_chunk_continuity_clone_and_output_final_state_flag():
    q, k, v, g, beta = _inputs(1, 10, 2, 4, seed=30)
    initial_state = torch.randn(1, 2, 4, 4, generator=torch.Generator().manual_seed(31)) * 0.1
    split = 4

    full_out, full_state = hpu_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state.clone(),
        chunk_size=3,
    )
    prefix_out, checkpoint = hpu_chunk_kda(
        q[:, :split],
        k[:, :split],
        v[:, :split],
        g[:, :split],
        beta[:, :split],
        initial_state=initial_state.clone(),
        chunk_size=3,
    )
    checkpoint_before = checkpoint.clone()
    suffix_out, suffix_state = hpu_chunk_kda(
        q[:, split:],
        k[:, split:],
        v[:, split:],
        g[:, split:],
        beta[:, split:],
        initial_state=checkpoint,
        chunk_size=3,
        inplace_final_state=False,
    )

    torch.testing.assert_close(torch.cat((prefix_out, suffix_out), dim=1), full_out, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(suffix_state, full_state, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(checkpoint, checkpoint_before, atol=0, rtol=0)
    assert checkpoint is not suffix_state

    inplace_initial = initial_state.clone()
    _, inplace_state = hpu_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=inplace_initial,
        chunk_size=3,
        inplace_final_state=True,
    )
    assert inplace_state is inplace_initial
    assert not torch.equal(inplace_initial, initial_state)

    _out, no_state = hpu_chunk_kda(q, k, v, g, beta, output_final_state=False, chunk_size=3)
    assert no_state is None


def test_ragged_prefill_and_padded_bucket_keep_sequence_states_separate():
    q, k, v, g, beta = _inputs(1, 9, 2, 3, seed=40)
    offsets = torch.tensor([0, 0, 2, 5, 9], dtype=torch.int32)
    initial_state = torch.randn(4, 2, 3, 3, generator=torch.Generator().manual_seed(41)) * 0.1

    expected_out = []
    expected_states = []
    for seq_idx, (start, end) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist())):
        out_i, state_i = _matrix_oracle_sequence(
            q[0, start:end],
            k[0, start:end],
            v[0, start:end],
            g[0, start:end],
            beta[0, start:end],
            initial_state[seq_idx],
            scale=None,
            use_qk_l2norm=True,
        )
        expected_out.append(out_i)
        expected_states.append(state_i)
    expected_out_t = torch.cat(expected_out).unsqueeze(0)
    expected_state_t = torch.stack(expected_states)

    recurrent_out, recurrent_state = hpu_fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state.clone(),
        cu_seqlens=offsets,
    )
    chunk_out, chunk_state = hpu_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state.clone(),
        cu_seqlens=offsets,
        chunk_size=2,
    )
    torch.testing.assert_close(recurrent_out, expected_out_t, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(recurrent_state, expected_state_t, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(chunk_out, expected_out_t, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(chunk_state, expected_state_t, atol=3e-6, rtol=3e-6)

    # Two sequences in a [1, 2*L] prefill bucket.  The second half of the
    # second sequence is bucket padding and must leave its state/output inert.
    qb, kb, vb, gb, betab = _inputs(1, 10, 2, 3, seed=42)
    qb[0, 7:] = 0
    kb[0, 7:] = 0
    vb[0, 7:] = 0
    gb[0, 7:] = 0
    betab[0, 7:] = 0
    bucket_state = torch.randn(2, 2, 3, 3, generator=torch.Generator().manual_seed(43)) * 0.1
    bucket_out, bucket_final = hpu_chunk_kda(
        qb,
        kb,
        vb,
        gb,
        betab,
        initial_state=bucket_state.clone(),
        chunk_size=3,
        prefill_num_seqs=2,
        prefill_seq_len=5,
    )
    assert torch.count_nonzero(bucket_out[0, 7:]) == 0
    bucket_expected_out = []
    bucket_expected_states = []
    for seq_idx, (start, end) in enumerate(((0, 5), (5, 7))):
        out_i, state_i = _matrix_oracle_sequence(
            qb[0, start:end],
            kb[0, start:end],
            vb[0, start:end],
            gb[0, start:end],
            betab[0, start:end],
            bucket_state[seq_idx],
            scale=None,
            use_qk_l2norm=True,
        )
        bucket_expected_out.append(out_i)
        bucket_expected_states.append(state_i)
    torch.testing.assert_close(bucket_out[0, :5], bucket_expected_out[0], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(bucket_out[0, 5:7], bucket_expected_out[1], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(bucket_final, torch.stack(bucket_expected_states), atol=3e-6, rtol=3e-6)


def test_indexed_decode_lane_reorder_and_slot_reuse():
    heads, head_dim = 2, 4
    state = torch.randn(4, heads, head_dim, head_dim, generator=torch.Generator().manual_seed(50)) * 0.2
    indices = torch.tensor([3, 1], dtype=torch.int32)
    q, k, v, g, beta = _inputs(1, 2, heads, head_dim, seed=51)
    expected_out = []
    expected_states = []
    for lane, slot in enumerate(indices.tolist()):
        out_i, state_i = _matrix_oracle_sequence(
            q[0, lane : lane + 1],
            k[0, lane : lane + 1],
            v[0, lane : lane + 1],
            g[0, lane : lane + 1],
            beta[0, lane : lane + 1],
            state[slot],
            scale=None,
            use_qk_l2norm=True,
        )
        expected_out.append(out_i)
        expected_states.append(state_i)

    state_before = state.clone()
    out, returned_state = hpu_fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        ssm_state_indices=indices,
    )
    assert returned_state is state
    torch.testing.assert_close(out, torch.stack(expected_out, dim=1), atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(state[3], expected_states[0], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(state[1], expected_states[1], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(state[0], state_before[0], atol=0, rtol=0)
    torch.testing.assert_close(state[2], state_before[2], atol=0, rtol=0)

    # Reuse both live slots in a different lane order on the next decode step.
    q2, k2, v2, g2, beta2 = _inputs(1, 2, heads, head_dim, seed=52)
    reordered_indices = torch.tensor([1, 3], dtype=torch.long)
    expected_second = []
    for lane, slot in enumerate(reordered_indices.tolist()):
        out_i, _ = _matrix_oracle_sequence(
            q2[0, lane : lane + 1],
            k2[0, lane : lane + 1],
            v2[0, lane : lane + 1],
            g2[0, lane : lane + 1],
            beta2[0, lane : lane + 1],
            state[slot],
            scale=None,
            use_qk_l2norm=True,
        )
        expected_second.append(out_i)
    out2, _ = hpu_fused_recurrent_kda(
        q2,
        k2,
        v2,
        g2,
        beta2,
        initial_state=state,
        ssm_state_indices=reordered_indices,
    )
    torch.testing.assert_close(out2, torch.stack(expected_second, dim=1), atol=3e-6, rtol=3e-6)


def test_invalid_indices_do_not_alias_or_mutate_live_slots():
    heads, head_dim = 2, 3
    q, k, v, g, beta = _inputs(1, 3, heads, head_dim, seed=60)
    state = torch.randn(3, heads, head_dim, head_dim, generator=torch.Generator().manual_seed(61)) * 0.2
    state_before = state.clone()
    indices = torch.tensor([-1, 1, 5], dtype=torch.long)

    expected = []
    expected_valid_state = None
    for lane, slot in enumerate(indices.tolist()):
        lane_state = state[slot] if 0 <= slot < state.shape[0] else None
        out_i, state_i = _matrix_oracle_sequence(
            q[0, lane : lane + 1],
            k[0, lane : lane + 1],
            v[0, lane : lane + 1],
            g[0, lane : lane + 1],
            beta[0, lane : lane + 1],
            lane_state,
            scale=None,
            use_qk_l2norm=True,
        )
        expected.append(out_i)
        if slot == 1:
            expected_valid_state = state_i

    out, _ = hpu_fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        ssm_state_indices=indices,
    )
    torch.testing.assert_close(out, torch.stack(expected, dim=1), atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(state[1], expected_valid_state, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(state[0], state_before[0], atol=0, rtol=0)
    torch.testing.assert_close(state[2], state_before[2], atol=0, rtol=0)


def test_padding_lanes_can_exceed_mapped_state_slots_and_repeat_invalid_indices():
    heads, head_dim = 2, 3
    q, k, v, g, beta = _inputs(1, 4, heads, head_dim, seed=62)
    state = torch.randn(2, heads, head_dim, head_dim, generator=torch.Generator().manual_seed(63)) * 0.2
    state_before = state.clone()
    indices = torch.tensor([0, -1, 1, -1], dtype=torch.long)

    expected_out = []
    expected_states = []
    for lane, slot in enumerate(indices.tolist()):
        lane_state = state[slot] if 0 <= slot < state.shape[0] else None
        out_i, state_i = _matrix_oracle_sequence(
            q[0, lane : lane + 1],
            k[0, lane : lane + 1],
            v[0, lane : lane + 1],
            g[0, lane : lane + 1],
            beta[0, lane : lane + 1],
            lane_state,
            scale=None,
            use_qk_l2norm=True,
        )
        expected_out.append(out_i)
        if slot >= 0:
            expected_states.append((slot, state_i))

    out, _ = hpu_fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        ssm_state_indices=indices,
    )
    torch.testing.assert_close(out, torch.stack(expected_out, dim=1), atol=3e-6, rtol=3e-6)
    for slot, expected_state in expected_states:
        torch.testing.assert_close(state[slot], expected_state, atol=3e-6, rtol=3e-6)
    assert state[0].shape == state_before[0].shape


def test_duplicate_effective_state_indices_are_rejected_before_mutation():
    heads, head_dim = 2, 3
    q, k, v, g, beta = _inputs(1, 3, heads, head_dim, seed=64)
    state = torch.randn(2, heads, head_dim, head_dim, generator=torch.Generator().manual_seed(65)) * 0.2
    state_before = state.clone()

    with pytest.raises(ValueError, match="duplicate effective state slots"):
        hpu_fused_recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            ssm_state_indices=torch.tensor([0, 1, 0], dtype=torch.long),
        )
    torch.testing.assert_close(state, state_before, atol=0, rtol=0)


def test_qkvgb_device_and_dtype_validation_is_explicit():
    q, k, v, g, beta = _inputs(1, 2, 2, 3, seed=66)
    with pytest.raises(TypeError, match="q, k and v must share dtype"):
        hpu_fused_recurrent_kda(q, k.to(torch.bfloat16), v, g, beta)
    beta_on_meta = torch.empty_like(beta, device="meta")
    with pytest.raises(ValueError, match="beta must be on q.device"):
        hpu_fused_recurrent_kda(q, k, v, g, beta_on_meta)

    batch_q, batch_k, batch_v, batch_g, batch_beta = _inputs(2, 1, 2, 3, seed=67)
    one_slot_state = torch.zeros(1, 2, 3, 3)
    with pytest.raises(ValueError, match="identity slot mapping"):
        hpu_fused_recurrent_kda(
            batch_q,
            batch_k,
            batch_v,
            batch_g,
            batch_beta,
            initial_state=one_slot_state,
        )


def test_unallocated_state_inplace_identity_and_fresh_state_dtype():
    q, k, v, g, beta = _inputs(1, 3, 2, 4, seed=70)
    lane_outputs = []
    lane_states = []
    for lane in range(3):
        lane_out, lane_state = _matrix_oracle_sequence(
            q[0, lane : lane + 1],
            k[0, lane : lane + 1],
            v[0, lane : lane + 1],
            g[0, lane : lane + 1],
            beta[0, lane : lane + 1],
            None,
            scale=None,
            use_qk_l2norm=True,
        )
        lane_outputs.append(lane_out)
        lane_states.append(lane_state)
    expected_out = torch.stack(lane_outputs, dim=1)
    expected_state = torch.stack(lane_states)
    out, fresh_state = hpu_fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        ssm_state_indices=torch.arange(3),
    )
    assert fresh_state.dtype == torch.float32
    assert fresh_state.shape == (3, 2, 4, 4)
    torch.testing.assert_close(out, expected_out, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(fresh_state, expected_state, atol=3e-6, rtol=3e-6)

    q1, k1, v1, g1, beta1 = _inputs(1, 1, 2, 4, seed=71)
    initial_state = torch.randn(2, 2, 4, 4, generator=torch.Generator().manual_seed(72)) * 0.1
    initial_before = initial_state.clone()
    _, copied_state = hpu_fused_recurrent_kda(
        q1,
        k1,
        v1,
        g1,
        beta1,
        initial_state=initial_state,
        inplace_final_state=False,
    )
    assert copied_state is not initial_state
    torch.testing.assert_close(initial_state, initial_before, atol=0, rtol=0)

    _, inplace_state = hpu_fused_recurrent_kda(
        q1,
        k1,
        v1,
        g1,
        beta1,
        initial_state=initial_state,
        inplace_final_state=True,
    )
    assert inplace_state is initial_state
    assert not torch.equal(initial_state, initial_before)


def test_bf16_activations_match_oracle_with_fp32_state():
    q, k, v, g, beta = _inputs(2, 6, 2, 4, seed=80, dtype=torch.bfloat16)
    # KDA's safe gate intentionally returns FP32 even for BF16 activations.
    g = g.float()
    initial_state = torch.randn(2, 2, 4, 4, generator=torch.Generator().manual_seed(81)) * 0.1
    expected_out, expected_state = _matrix_oracle_batch(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        scale=None,
        use_qk_l2norm=True,
    )

    recurrent_out, recurrent_state = hpu_fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state.clone(),
    )
    chunk_out, chunk_state = hpu_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state.clone(),
        chunk_size=3,
    )
    assert recurrent_out.dtype == torch.bfloat16
    assert chunk_out.dtype == torch.bfloat16
    assert recurrent_state.dtype == torch.float32
    assert chunk_state.dtype == torch.float32
    torch.testing.assert_close(recurrent_out, expected_out.to(torch.bfloat16), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(chunk_out, expected_out.to(torch.bfloat16), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(recurrent_state, expected_state, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(chunk_state, expected_state, atol=2e-2, rtol=2e-2)


def test_bf16_fresh_state_is_fp32():
    q, k, v, g, beta = _inputs(1, 2, 2, 3, seed=90, dtype=torch.bfloat16)
    _out, state = hpu_fused_recurrent_kda(q, k, v, g, beta, initial_state=None)
    _chunk_out, chunk_state = hpu_chunk_kda(q, k, v, g, beta, initial_state=None, chunk_size=1)
    assert state.dtype == torch.float32
    assert chunk_state is not None and chunk_state.dtype == torch.float32
