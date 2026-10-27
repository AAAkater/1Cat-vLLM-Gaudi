# SPDX-License-Identifier: Apache-2.0
"""Device-agnostic PyTorch math for Glm5Next KDA attention.

KDA is a gated delta-rule with a per-key-channel decay.  The recurrent state
uses the same ``[V, K]`` orientation as the vLLM/HPU SSM cache::

    S_t[v, k] = exp(g_t[k]) * S_{t-1}[v, k]
                + beta_t * (v_t[v] - sum_j S_{t-1}[v, j]
                            exp(g_t[j]) k_t[j]) * k_t[k]
    o_t[v] = sum_k S_t[v, k] * scale * q_t[k]

The public entry points intentionally mirror the upstream KDA ops.  State
arithmetic and returned fresh states are FP32, even when activations are BF16.
No HPU-specific compilation is performed here; the chunk path is ordinary
PyTorch so it can be compiled explicitly by an HPU caller.
"""

from __future__ import annotations

import math

import torch

_KDA_STATE_DTYPE = torch.float32


def _l2norm_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


def _check_qkvgb(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    named = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, D], got {tuple(q.shape)}")
    if k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
        raise ValueError(
            "q, k, v and g must all have shape [B, T, H, D], got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, g={tuple(g.shape)}"
        )
    expected_beta = q.shape[:3]
    if beta.shape == (*expected_beta, 1):
        beta = beta.squeeze(-1)
    if beta.shape != expected_beta:
        raise ValueError(f"beta must have shape {tuple(expected_beta)}, got {tuple(beta.shape)}")
    for name, tensor in named.items():
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on q.device={q.device}, got {tensor.device}")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(
            "q, k and v must share dtype, got "
            f"q={q.dtype}, k={k.dtype}, v={v.dtype}"
        )
    return beta


def _check_state(
    state: torch.Tensor,
    *,
    name: str,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> None:
    expected_tail = (num_heads, head_dim, head_dim)
    if state.ndim != 4 or state.shape[1:] != expected_tail:
        raise ValueError(
            f"{name} must have shape [S, H, V, K] = [S, {num_heads}, {head_dim}, {head_dim}], "
            f"got {tuple(state.shape)}"
        )
    if state.device != device:
        raise ValueError(f"{name} must be on {device}, got {state.device}")
    if not state.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {state.dtype}")


def _normalize_scale(scale: float | None, head_dim: int) -> float:
    if scale is None:
        return float(head_dim**-0.5)
    if not isinstance(scale, (int, float)):
        raise TypeError(f"scale must be a number or None, got {type(scale).__name__}")
    return float(scale)


def hpu_kda_gate(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float = -5.0,
    num_heads: int | None = None,
    head_dim: int | None = None,
) -> torch.Tensor:
    """Apply KDA's bounded per-channel sigmoid gate.

    ``g = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))`` where both
    ``A_log`` and ``dt_bias`` are per-(head, channel).  This is deliberately
    different from GDN's softplus and per-head gate.

    Args:
        raw_g: ``[..., H, D]`` or a flattened ``[..., H * D]`` tensor.
        A_log: One value per head.
        dt_bias: Optional ``[H, D]`` (broadcastable) bias.
        lower_bound: Negative safe-gate floor.
        num_heads: Optional explicit head count for flattened inputs.
        head_dim: Optional explicit head size for flattened inputs.

    Returns:
        An FP32 tensor with shape ``[..., H, D]``.
    """
    if not isinstance(lower_bound, (int, float)) or not math.isfinite(float(lower_bound)):
        raise ValueError(f"lower_bound must be finite, got {lower_bound!r}")
    if lower_bound > 0:
        raise ValueError(f"lower_bound must be non-positive, got {lower_bound}")
    if A_log.device != raw_g.device:
        raise ValueError(f"A_log must be on raw_g.device={raw_g.device}, got {A_log.device}")
    if dt_bias is not None and dt_bias.device != raw_g.device:
        raise ValueError(f"dt_bias must be on raw_g.device={raw_g.device}, got {dt_bias.device}")

    a_flat = A_log.reshape(-1).to(dtype=torch.float32)
    if a_flat.numel() == 0:
        raise ValueError("A_log must contain at least one head")
    heads = int(a_flat.numel()) if num_heads is None else int(num_heads)
    if heads <= 0 or heads != a_flat.numel():
        raise ValueError(f"A_log must contain num_heads={heads} entries, got {a_flat.numel()}")

    if raw_g.ndim == 0:
        raise ValueError("raw_g must have at least one dimension")
    dim = head_dim
    shaped = raw_g.ndim >= 2 and raw_g.shape[-2] == heads and (dim is None or raw_g.shape[-1] == dim)
    if dim is not None and raw_g.shape[-1] == heads * dim and raw_g.shape[-1] != dim:
        # An explicitly-sized flattened input remains unambiguous even when its
        # leading dimension happens to equal num_heads.
        shaped = False
    if shaped:
        g_in = raw_g.to(dtype=torch.float32)
        dim = int(g_in.shape[-1])
    else:
        if dim is None:
            if raw_g.shape[-1] % heads:
                raise ValueError(
                    f"Flattened raw_g width {raw_g.shape[-1]} is not divisible by num_heads={heads}"
                )
            dim = int(raw_g.shape[-1] // heads)
        if raw_g.shape[-1] != heads * dim:
            raise ValueError(
                f"Flattened raw_g width must be num_heads*head_dim={heads * dim}, got {raw_g.shape[-1]}"
            )
        g_in = raw_g.to(dtype=torch.float32).reshape(*raw_g.shape[:-1], heads, dim)

    if dt_bias is not None:
        bias = dt_bias.to(dtype=torch.float32)
        try:
            bias = bias.reshape(heads, dim)
        except RuntimeError as exc:
            raise ValueError(f"dt_bias must contain {heads * dim} values, got shape {tuple(dt_bias.shape)}") from exc
        g_in = g_in + bias

    head_scale = torch.exp(a_flat).reshape(heads, 1)
    return float(lower_bound) * torch.sigmoid(head_scale * g_in)


def _kda_recurrent_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One KDA recurrence step; q/k are already normalized and q is scaled."""
    decayed = state * torch.exp(g).unsqueeze(-2)
    prediction = torch.einsum("...vk,...k->...v", decayed, k)
    residual = (v - prediction) * beta.unsqueeze(-1)
    next_state = decayed + residual.unsqueeze(-1) * k.unsqueeze(-2)
    output = torch.einsum("...vk,...k->...v", next_state, q)
    return output, next_state


def _cu_seqlens_list(cu_seqlens: torch.Tensor, total_tokens: int) -> list[int]:
    offsets = [int(x) for x in cu_seqlens.detach().reshape(-1).to(device="cpu", dtype=torch.long).tolist()]
    if len(offsets) < 2:
        raise ValueError("cu_seqlens must contain at least two offsets")
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError(f"cu_seqlens must start at 0 and end at {total_tokens}, got {offsets}")
    if any(end < start for start, end in zip(offsets[:-1], offsets[1:])):
        raise ValueError(f"cu_seqlens must be non-decreasing, got {offsets}")
    return offsets


def _recurrent_layout(
    q: torch.Tensor,
    ssm_state_indices: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> list[tuple[int, int]]:
    """Return flattened half-open token ranges, one per independent sequence."""
    batch, seq_len, _, _ = q.shape
    total_tokens = batch * seq_len
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("cu_seqlens requires q to have shape [1, T, H, D]")
        offsets = _cu_seqlens_list(cu_seqlens, total_tokens)
        return list(zip(offsets[:-1], offsets[1:]))

    num_indices = ssm_state_indices.numel() if ssm_state_indices is not None else 0
    if batch == 1 and seq_len > 1 and num_indices == seq_len:
        # Packed decode: T lanes, one token per independent sequence.
        return [(i, i + 1) for i in range(seq_len)]
    if batch > 1 and seq_len == 1:
        return [(i, i + 1) for i in range(batch)]
    if batch == 1 and ssm_state_indices is not None and num_indices != 1:
        raise ValueError(
            "For a single prefill sequence ssm_state_indices must contain one slot; "
            f"got {num_indices} entries"
        )
    if batch > 1 and ssm_state_indices is not None and num_indices != batch:
        raise ValueError(f"ssm_state_indices must contain one slot per sequence ({batch}), got {num_indices}")
    return [(i * seq_len, (i + 1) * seq_len) for i in range(batch)]


@torch._dynamo.disable
def _run_recurrent_sequences(
    qf: torch.Tensor,
    kf: torch.Tensor,
    vf: torch.Tensor,
    gf: torch.Tensor,
    bf: torch.Tensor,
    initial_states: torch.Tensor,
    ranges: list[tuple[int, int]],
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_tokens = qf.shape[0]
    flat_out = torch.empty(total_tokens, num_heads, head_dim, dtype=torch.float32, device=qf.device)
    updated = torch.empty_like(initial_states)

    if all(end - start == 1 for start, end in ranges):
        starts = torch.tensor([start for start, _ in ranges], dtype=torch.long, device=qf.device)
        step_out, updated = _kda_recurrent_step(
            qf.index_select(0, starts),
            kf.index_select(0, starts),
            vf.index_select(0, starts),
            gf.index_select(0, starts),
            bf.index_select(0, starts),
            initial_states,
        )
        flat_out.index_copy_(0, starts, step_out)
        return flat_out, updated

    for seq_id, (start, end) in enumerate(ranges):
        state = initial_states[seq_id].to(dtype=torch.float32)
        for token in range(start, end):
            token_out, state = _kda_recurrent_step(qf[token], kf[token], vf[token], gf[token], bf[token], state)
            flat_out[token] = token_out
        updated[seq_id] = state
    return flat_out, updated


def hpu_fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    ssm_state_indices: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate KDA recurrently on ``[B, T, H, D]`` tensors.

    ``initial_state`` has shape ``[S, H, V, K]`` and is FP32 by contract.
    ``ssm_state_indices`` maps each independent sequence/lane to one row of
    that state.  Out-of-range entries are invalid: they compute from a zero
    state and are never written back.  Invalid entries are *not* wrapped with
    a remainder, because doing so aliases padding lanes onto live cache slots.
    Only mapped valid slots count against ``S``, so padding lanes may outnumber
    live state slots.  Duplicate *valid* slots are rejected before any write;
    repeated invalid padding entries are allowed.

    Packed decode is inferred when ``B == 1 and ssm_state_indices.numel() ==
    T``; lane order is preserved in the output.  Ragged prefill requires
    ``B == 1`` and ``cu_seqlens``.
    """
    beta = _check_qkvgb(q, k, v, g, beta)
    batch, seq_len, num_heads, head_dim = q.shape
    if not torch.is_floating_point(q) or not torch.is_floating_point(v):
        raise TypeError("q, k and v must be floating-point tensors")
    ranges = _recurrent_layout(q, ssm_state_indices, cu_seqlens)
    num_seqs = len(ranges)
    if initial_state is None:
        final_state = torch.zeros(
            (num_seqs, num_heads, head_dim, head_dim),
            dtype=_KDA_STATE_DTYPE,
            device=q.device,
        )
    else:
        _check_state(
            initial_state,
            name="initial_state",
            num_heads=num_heads,
            head_dim=head_dim,
            device=q.device,
        )
        if num_seqs and initial_state.shape[0] == 0:
            raise ValueError("initial_state must contain at least one slot when sequences are supplied")
        if ssm_state_indices is None and initial_state.shape[0] < num_seqs:
            raise ValueError(
                "initial_state uses identity slot mapping and must contain at least "
                f"{num_seqs} slots, got {initial_state.shape[0]}"
            )
        final_state = initial_state if inplace_final_state else initial_state.clone()

    if ssm_state_indices is None:
        state_indices = torch.arange(num_seqs, dtype=torch.long, device=q.device)
    else:
        state_indices = ssm_state_indices.reshape(-1).to(dtype=torch.long, device=q.device)
        if state_indices.numel() != num_seqs:
            raise ValueError(f"ssm_state_indices must contain {num_seqs} entries, got {state_indices.numel()}")
    valid = (state_indices >= 0) & (state_indices < final_state.shape[0])
    valid_cpu = state_indices[valid].detach().to(device="cpu")
    if torch.unique(valid_cpu).numel() != valid_cpu.numel():
        raise ValueError("ssm_state_indices contains duplicate effective state slots")
    safe_indices = torch.where(valid, state_indices, torch.zeros_like(state_indices))

    state_work = final_state.to(dtype=_KDA_STATE_DTYPE)
    gathered = state_work.index_select(0, safe_indices)
    gathered = torch.where(valid.reshape(num_seqs, 1, 1, 1), gathered, torch.zeros_like(gathered))

    qf = q.reshape(-1, num_heads, head_dim).to(dtype=torch.float32)
    kf = k.reshape(-1, num_heads, head_dim).to(dtype=torch.float32)
    vf = v.reshape(-1, num_heads, head_dim).to(dtype=torch.float32)
    gf = g.reshape(-1, num_heads, head_dim).to(dtype=torch.float32)
    bf = beta.reshape(-1, num_heads).to(dtype=torch.float32)
    if use_qk_l2norm_in_kernel:
        qf = _l2norm_last_dim(qf)
        kf = _l2norm_last_dim(kf)
    qf = qf * _normalize_scale(scale, head_dim)

    flat_out, updated = _run_recurrent_sequences(
        qf,
        kf,
        vf,
        gf,
        bf,
        gathered,
        ranges,
        num_heads,
        head_dim,
    )
    if num_seqs:
        final_state.index_copy_(0, state_indices[valid], updated[valid].to(dtype=final_state.dtype))
    out = flat_out.reshape(batch, seq_len, num_heads, head_dim).to(dtype=q.dtype)
    return out, final_state


@torch._dynamo.disable
def _kda_chunk_single_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
    use_qk_l2norm_in_kernel: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact intra-chunk forward substitution with inter-chunk state carry."""
    seq_len, num_heads, head_dim = q.shape
    qf = q.to(dtype=torch.float32)
    kf = k.to(dtype=torch.float32)
    vf = v.to(dtype=torch.float32)
    gf = g.to(dtype=torch.float32)
    bf = beta.to(dtype=torch.float32)
    if use_qk_l2norm_in_kernel:
        qf = _l2norm_last_dim(qf)
        kf = _l2norm_last_dim(kf)
    qf = qf * scale

    state = initial_state.to(dtype=torch.float32)
    out = torch.empty(seq_len, num_heads, head_dim, dtype=torch.float32, device=q.device)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        qc = qf[start:end]
        kc = kf[start:end]
        vc = vf[start:end]
        bc = bf[start:end]
        chunk_len = end - start

        # Local cumulative gate.  At row i this is the product of decays from
        # the chunk boundary through token i, inclusive.
        gc = torch.cumsum(gf[start:end], dim=0)
        gc_exp = torch.exp(gc)

        # Contribution of the carried state to each token's prediction.
        state_pred = torch.einsum("hvk,chk->chv", state, gc_exp * kc)
        rhs = (vc - state_pred) * bc.unsqueeze(-1)

        # Strictly-lower interaction matrix B[i, j] for j < i.  Computing one
        # row at a time avoids materializing [C, C, H, K], whose size is
        # prohibitive for production head counts and dimensions.
        interaction = torch.zeros(num_heads, chunk_len, chunk_len, dtype=torch.float32, device=q.device)
        for i in range(1, chunk_len):
            delta = gc[i].unsqueeze(0) - gc[:i]
            row = torch.sum(torch.exp(delta) * kc[:i] * kc[i].unsqueeze(0), dim=-1)
            interaction[:, i, :i] = row.transpose(0, 1)

        # Solve (I + diag(beta) * B) v_new = beta * (v - S0 @ k).
        v_new = torch.empty(chunk_len, num_heads, head_dim, dtype=torch.float32, device=q.device)
        for i in range(chunk_len):
            if i == 0:
                lag = torch.zeros(num_heads, head_dim, dtype=torch.float32, device=q.device)
            else:
                lag = torch.einsum("hj,jhv->hv", interaction[:, i, :i], v_new[:i])
            v_new[i] = rhs[i] - bc[i].unsqueeze(-1) * lag

        state_out = torch.einsum("hvk,chk->chv", state, gc_exp * qc)
        for i in range(chunk_len):
            delta = gc[i].unsqueeze(0) - gc[:i + 1]
            local_qk = torch.sum(torch.exp(delta) * kc[:i + 1] * qc[i].unsqueeze(0), dim=-1)
            local_qk = local_qk.transpose(0, 1)
            out[start + i] = state_out[i] + torch.einsum("hj,jhv->hv", local_qk, v_new[:i + 1])

        end_decay = gc[-1]
        weighted_k = torch.exp(end_decay.unsqueeze(0) - gc) * kc
        state = state * torch.exp(end_decay).unsqueeze(-2)
        state = state + torch.einsum("chv,chk->hvk", v_new, weighted_k)

    return out, state


def hpu_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
    chunk_size: int = 64,
    prefill_num_seqs: int | None = None,
    prefill_seq_len: int | None = None,
    chunk_group: int = 4,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    inplace_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate KDA with exact intra-chunk algebra.

    The recurrence is carried across chunks in FP32.  Ragged inputs can be
    supplied as ``q[0, :T]`` plus ``cu_seqlens``; otherwise a flat
    ``[1, num_seqs * seq_len]`` bucket may be reshaped with
    ``prefill_num_seqs``/``prefill_seq_len``.  Padding tokens must have
    ``q=k=v=g=beta=0`` so they leave the state and output unchanged.

    ``chunk_group`` is accepted for API compatibility; the portable PyTorch
    implementation processes chunks in order.  ``inplace_final_state`` has
    the same identity contract as the recurrent op.
    """
    del chunk_group  # API compatibility; no grouped-compile cache in portable code.
    beta = _check_qkvgb(q, k, v, g, beta)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    batch, total_seq_len, num_heads, head_dim = q.shape

    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("cu_seqlens requires q to have shape [1, T, H, D]")
        offsets = _cu_seqlens_list(cu_seqlens, total_seq_len)
        ranges = list(zip(offsets[:-1], offsets[1:]))
        flat_q = q.reshape(-1, num_heads, head_dim)
        flat_k = k.reshape(-1, num_heads, head_dim)
        flat_v = v.reshape(-1, num_heads, head_dim)
        flat_g = g.reshape(-1, num_heads, head_dim)
        flat_beta = beta.reshape(-1, num_heads)
    elif prefill_num_seqs is not None and prefill_num_seqs > 0:
        if prefill_seq_len is None or prefill_seq_len <= 0:
            raise ValueError("prefill_seq_len must be positive when prefill_num_seqs is set")
        expected = int(prefill_num_seqs) * int(prefill_seq_len)
        if expected != batch * total_seq_len:
            raise ValueError(
                f"prefill_num_seqs*prefill_seq_len must equal B*T={batch * total_seq_len}, got {expected}"
            )
        ranges = [(i * prefill_seq_len, (i + 1) * prefill_seq_len) for i in range(prefill_num_seqs)]
        flat_q = q.reshape(-1, num_heads, head_dim)
        flat_k = k.reshape(-1, num_heads, head_dim)
        flat_v = v.reshape(-1, num_heads, head_dim)
        flat_g = g.reshape(-1, num_heads, head_dim)
        flat_beta = beta.reshape(-1, num_heads)
    else:
        ranges = [(i * total_seq_len, (i + 1) * total_seq_len) for i in range(batch)]
        flat_q = q.reshape(-1, num_heads, head_dim)
        flat_k = k.reshape(-1, num_heads, head_dim)
        flat_v = v.reshape(-1, num_heads, head_dim)
        flat_g = g.reshape(-1, num_heads, head_dim)
        flat_beta = beta.reshape(-1, num_heads)

    num_seqs = len(ranges)
    if initial_state is None:
        final_state = torch.zeros(
            (num_seqs, num_heads, head_dim, head_dim),
            dtype=_KDA_STATE_DTYPE,
            device=q.device,
        )
    else:
        _check_state(
            initial_state,
            name="initial_state",
            num_heads=num_heads,
            head_dim=head_dim,
            device=q.device,
        )
        if initial_state.shape[0] != num_seqs:
            raise ValueError(f"initial_state must contain exactly {num_seqs} sequence states")
        final_state = initial_state if inplace_final_state else initial_state.clone()

    state_work = final_state.to(dtype=_KDA_STATE_DTYPE)
    updated = torch.empty_like(state_work)
    flat_out = torch.empty(
        (batch * total_seq_len, num_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    resolved_scale = _normalize_scale(scale, head_dim)
    for seq_id, (start, end) in enumerate(ranges):
        seq_out, seq_state = _kda_chunk_single_sequence(
            flat_q[start:end],
            flat_k[start:end],
            flat_v[start:end],
            flat_g[start:end],
            flat_beta[start:end],
            state_work[seq_id],
            chunk_size,
            use_qk_l2norm_in_kernel,
            resolved_scale,
        )
        flat_out[start:end] = seq_out
        updated[seq_id] = seq_state

    # A single copy preserves untouched rows for in-place cache semantics.
    final_state.copy_(updated.to(dtype=final_state.dtype))
    out = flat_out.reshape(batch, total_seq_len, num_heads, head_dim).to(dtype=q.dtype)
    return out, final_state if output_final_state else None


__all__ = [
    "hpu_chunk_kda",
    "hpu_fused_recurrent_kda",
    "hpu_kda_gate",
]
