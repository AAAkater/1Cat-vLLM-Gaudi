# SPDX-License-Identifier: Apache-2.0
"""HPU wiring for the GLM-5.3 Flash KDA linear-attention layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from vllm.forward_context import get_forward_context
from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention

from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_glm5next_norm import rmsnorm_sigmoid_gate
from vllm_gaudi.ops.hpu_kda_pytorch import (
    hpu_chunk_kda,
    hpu_fused_recurrent_kda,
    hpu_kda_gate,
)

_PAD_STATE_SLOT = -1


@dataclass(frozen=True)
class _KDAMetadata:
    is_prompt: bool
    conv_state: torch.Tensor
    conv_scratch_state: torch.Tensor | None
    ssm_state: torch.Tensor
    load_indices: torch.Tensor
    store_indices: torch.Tensor
    query_start_loc: torch.Tensor
    token_mask: torch.Tensor
    active_lanes: torch.Tensor
    initial_state: torch.Tensor | None
    has_initial_state: torch.Tensor | None
    prefill_num_seqs: int
    prefill_seq_len: int
    chunk_size: int
    cu_seqlens: torch.Tensor | None


def _metadata_for_layer(raw_metadata, prefix: str):
    """Return this layer's metadata without a cross-layer fallback."""
    if raw_metadata is None:
        return None
    if isinstance(raw_metadata, Mapping):
        if not raw_metadata:
            return None
        metadata = raw_metadata.get(prefix)
        if metadata is None:
            raise RuntimeError(
                f"Missing HPU attention metadata for {prefix!r}; refusing to reuse another layer's metadata"
            )
        return metadata
    return raw_metadata


def _resolve_state_indices(owner, metadata, attribute: str) -> torch.Tensor | None:
    """Resolve one layer's row from legacy HPU [cache_group, batch] indices."""
    indices = getattr(metadata, attribute, None)
    if indices is None:
        return None
    if indices.ndim == 1:
        return indices
    if indices.ndim != 2:
        raise ValueError(
            f"{attribute} must have shape [batch] or [cache_group, batch], got {tuple(indices.shape)}"
        )

    cache_group_idx = getattr(owner, "cache_group_idx", None)
    if cache_group_idx is None:
        raise RuntimeError(
            f"{attribute} is 2-D but this layer has no cache_group_idx; "
            "an implicit first-row fallback could alias another cache group"
        )
    group = torch.as_tensor(cache_group_idx, dtype=torch.long, device=indices.device).reshape(-1)
    if group.numel() != 1:
        raise ValueError(f"cache_group_idx must be scalar, got shape {tuple(group.shape)}")
    group_id = int(group.item())
    if group_id < 0 or group_id >= indices.shape[0]:
        raise IndexError(
            f"cache_group_idx={group_id} is outside {attribute} cache-group dimension {indices.shape[0]}"
        )
    return indices.index_select(0, group).squeeze(0)


def _validate_state_indices(
    indices: torch.Tensor,
    *,
    num_slots: int,
    name: str,
    allow_padding: bool = True,
) -> None:
    """Validate state slots while permitting repeated -1 padding lanes.

    The number of allocated slots may be smaller than the padded lane count:
    only live, non-negative indices need to fit the pool. Duplicate live slots
    are rejected because two lanes would otherwise mutate one recurrent state.
    """
    if indices.ndim != 1:
        raise ValueError(f"{name} must be 1-D after cache-group selection, got {tuple(indices.shape)}")
    values = indices.to(dtype=torch.long)
    negative = values < 0
    if bool(negative.any()):
        if not allow_padding:
            raise IndexError(f"{name} contains a negative state slot")
        invalid_padding = values[negative] != _PAD_STATE_SLOT
        if bool(invalid_padding.any()):
            raise IndexError(f"{name} may only use -1 for padding lanes; got {values[negative].tolist()}")
    live = values[~negative]
    if bool((live >= num_slots).any()):
        raise IndexError(f"{name} contains a live slot outside state pool with {num_slots} slots")
    if live.numel() != torch.unique(live).numel():
        raise ValueError(f"{name} contains duplicate live state slots")


def _normalize_has_initial(has_initial, batch: int, device: torch.device) -> torch.Tensor | None:
    if has_initial is None:
        return None
    values = has_initial.reshape(-1).bool().to(device=device)
    if values.numel() == 1:
        return values.expand(batch)
    if values.numel() < batch:
        values = torch.nn.functional.pad(values, (0, batch - values.numel()), value=False)
    return values[:batch]


def _build_token_mask(
    padding_mask_flat: torch.Tensor | None,
    lengths: torch.Tensor,
    seq_len: int,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a flat [num_tokens] validity mask for a rectangular HPU bucket."""
    batch = lengths.numel()
    if seq_len <= 0 or batch * seq_len != num_tokens:
        raise ValueError(
            f"rectangular prefill requires num_tokens == batch * seq_len, got {num_tokens} != "
            f"{batch} * {seq_len}"
        )
    if padding_mask_flat is not None:
        mask = padding_mask_flat.reshape(-1).bool().to(device=device)
        if mask.numel() != num_tokens:
            raise ValueError(f"padding_mask_flat must contain {num_tokens} entries, got {mask.numel()}")
        mask_counts = mask.view(batch, seq_len).sum(dim=1)
        if not torch.equal(mask_counts, lengths.to(device=device, dtype=torch.long)):
            raise ValueError(
                f"padding_mask_flat per-sequence lengths {mask_counts.tolist()} do not match "
                f"query_start_loc lengths {lengths.tolist()}"
            )
        return mask
    positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    return (positions < lengths.to(device=device, dtype=torch.long).unsqueeze(1)).reshape(-1)


def _input_state(
    ssm_state: torch.Tensor,
    load_indices: torch.Tensor,
    active_lanes: torch.Tensor,
    has_initial_state: torch.Tensor | None,
) -> torch.Tensor:
    """Gather lane states without negative indexing or reading pad slots."""
    live_load = load_indices >= 0
    safe_load = torch.where(live_load, load_indices, torch.zeros_like(load_indices)).long()
    initial_state = ssm_state.index_select(0, safe_load).contiguous()
    keep = active_lanes.bool() & live_load
    if has_initial_state is not None:
        keep = keep & has_initial_state.bool()
    mask_shape = (-1,) + (1, ) * (initial_state.ndim - 1)
    return torch.where(
        keep.view(mask_shape),
        initial_state,
        torch.zeros_like(initial_state),
    )


@torch._dynamo.disable
def _scatter_active_states(
    passthrough: torch.Tensor,
    final_state: torch.Tensor,
    ssm_state: torch.Tensor,
    store_indices: torch.Tensor,
    active_lanes: torch.Tensor,
) -> torch.Tensor:
    """Commit final states only for live lanes; pad slots are never written."""
    positions = torch.nonzero(active_lanes.bool() & (store_indices >= 0), as_tuple=False).reshape(-1)
    if positions.numel() == 0:
        return passthrough
    state_rows = final_state.index_select(0, positions).to(device=ssm_state.device, dtype=ssm_state.dtype)
    with torch.no_grad():
        ssm_state.index_copy_(0, store_indices.index_select(0, positions).long(), state_rows)
    return passthrough


def _merge_conv_weights(*weights: torch.Tensor) -> torch.Tensor:
    if len(weights) != 3:
        raise ValueError(f"expected q, k and v convolution weights, got {len(weights)}")
    merged = []
    for index, weight in enumerate(weights):
        if weight.ndim != 3 or weight.shape[1] != 1:
            raise ValueError(f"conv weight {index} must have shape [channels, 1, width], got {tuple(weight.shape)}")
        merged.append(weight.view(weight.shape[0], weight.shape[2]))
    return torch.cat(merged, dim=0).contiguous()


def _layer_sequence_lengths(metadata, batch: int, num_tokens: int, device: torch.device) -> torch.Tensor:
    query_start_loc = getattr(metadata, "query_start_loc_p", None)
    if query_start_loc is None:
        query_start_loc = getattr(metadata, "query_start_loc", None)
    if query_start_loc is None:
        query_start_loc = getattr(metadata, "non_spec_query_start_loc", None)
    if query_start_loc is None:
        seq_lens = getattr(metadata, "seq_lens_tensor", None)
        if seq_lens is not None:
            return seq_lens.reshape(-1)[:batch].to(device=device, dtype=torch.long)
        raise ValueError("prefill metadata must provide query_start_loc or seq_lens_tensor")
    offsets = query_start_loc.reshape(-1).to(device=device, dtype=torch.long)
    if offsets.numel() != batch + 1:
        raise ValueError(f"query_start_loc must contain batch+1={batch + 1} offsets, got {offsets.numel()}")
    if bool((offsets[1:] < offsets[:-1]).any()) or bool((offsets < 0).any()):
        raise ValueError("query_start_loc must be non-negative and non-decreasing")
    return offsets[1:] - offsets[:-1]


class HPUGlm5NextLinearAttention(Glm5NextLinearAttention):
    """GLM-5.3 KDA with HPU convolution and recurrent/chunk state kernels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The runner assigns the layer's row in the hybrid cache-group metadata.
        self.cache_group_idx = None
        self._merged_conv_weight = None

    def _build_merged_conv_weight(self) -> torch.Tensor:
        merged = _merge_conv_weights(
            self.q_conv1d.weight,
            self.k_conv1d.weight,
            self.v_conv1d.weight,
        )
        self._merged_conv_weight = merged
        return merged

    @torch._dynamo.disable
    def _extract_metadata(self, num_tokens: int) -> _KDAMetadata | None:
        raw_metadata = get_forward_context().attn_metadata
        metadata = _metadata_for_layer(raw_metadata, self.prefix)
        if metadata is None:
            return None

        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is None or len(kv_cache) < 2:
            return None
        conv_state, ssm_state = kv_cache[0], kv_cache[1]
        if conv_state is None or ssm_state is None:
            return None
        expected_conv_dim = 3 * self.local_projection_size
        if conv_state.ndim != 3 or conv_state.shape[-1] != expected_conv_dim:
            raise ValueError(
                f"{self.prefix}: HPU KDA conv state must use [slots, width-1, {expected_conv_dim}] "
                f"layout, got {tuple(conv_state.shape)}"
            )
        if conv_state.shape[1] < self.conv_size - 1:
            raise ValueError(
                f"{self.prefix}: conv state width {conv_state.shape[1]} is smaller than "
                f"conv_size-1={self.conv_size - 1}"
            )
        if ssm_state.dtype != torch.float32:
            raise TypeError(f"{self.prefix}: KDA recurrent state must be FP32, got {ssm_state.dtype}")
        if conv_state.shape[0] != ssm_state.shape[0] or conv_state.device != ssm_state.device:
            raise ValueError(
                f"{self.prefix}: conv and recurrent state pools must have the same slots/device, got "
                f"{tuple(conv_state.shape)} {conv_state.device} and {tuple(ssm_state.shape)} {ssm_state.device}"
            )

        num_prefills = int(getattr(metadata, "num_prefills", 0) or 0)
        num_decodes = int(getattr(metadata, "num_decodes", 0) or 0)
        if num_prefills > 0 and num_decodes > 0:
            raise NotImplementedError(
                f"{self.prefix}: mixed prefill/decode KDA batches are not supported by this HPU path"
            )
        is_prompt = bool(getattr(metadata, "is_prompt", False) or num_prefills > 0)
        load_indices = _resolve_state_indices(self, metadata, "load_indices_tensor")
        if load_indices is None:
            load_indices = _resolve_state_indices(self, metadata, "non_spec_state_indices_tensor")
        store_indices = _resolve_state_indices(self, metadata, "store_indices_tensor")
        if store_indices is None:
            store_indices = load_indices
        if load_indices is None:
            raise RuntimeError(f"{self.prefix}: missing load_indices_tensor for KDA state")
        if store_indices is None:
            raise RuntimeError(f"{self.prefix}: missing store_indices_tensor for KDA state")

        load_indices = load_indices.reshape(-1).to(device=ssm_state.device, dtype=torch.long)
        store_indices = store_indices.reshape(-1).to(device=ssm_state.device, dtype=torch.long)
        if load_indices.numel() != store_indices.numel():
            raise ValueError(
                f"{self.prefix}: load/store state index lengths differ: "
                f"{load_indices.numel()} vs {store_indices.numel()}"
            )
        batch = load_indices.numel()
        if batch == 0:
            raise ValueError(f"{self.prefix}: KDA state metadata has no lanes")
        _validate_state_indices(load_indices, num_slots=ssm_state.shape[0], name="load_indices")
        _validate_state_indices(store_indices, num_slots=ssm_state.shape[0], name="store_indices")

        if ssm_state.ndim != 4 or ssm_state.shape[1:] != (
                self.local_num_heads,
                self.head_dim,
                self.head_dim,
        ):
            raise ValueError(
                f"{self.prefix}: KDA state must have shape [slots, {self.local_num_heads}, "
                f"{self.head_dim}, {self.head_dim}], got {tuple(ssm_state.shape)}"
            )

        query_start_loc = getattr(metadata, "query_start_loc_p", None)
        if query_start_loc is None:
            query_start_loc = getattr(metadata, "query_start_loc", None)
        if query_start_loc is None:
            query_start_loc = getattr(metadata, "non_spec_query_start_loc", None)

        has_initial_state = _normalize_has_initial(
            getattr(metadata, "has_initial_states_p", getattr(metadata, "has_initial_state", None)),
            batch,
            load_indices.device,
        )
        padding_mask_flat = getattr(metadata, "padding_mask_flat", None)

        if not is_prompt:
            if num_tokens != batch:
                raise NotImplementedError(
                    f"{self.prefix}: HPU KDA decode currently supports one token per padded lane; "
                    f"got num_tokens={num_tokens}, lanes={batch}"
                )
            load_live = load_indices >= 0
            store_live = store_indices >= 0
            mismatched_liveness = load_live != store_live
            if bool(mismatched_liveness.any()):
                raise IndexError(
                    f"{self.prefix}: decode lane has only one of load/store live; "
                    f"load={load_indices.tolist()}, store={store_indices.tolist()}"
                )
            if not torch.equal(load_indices, store_indices):
                raise NotImplementedError(
                    f"{self.prefix}: HPU KDA decode requires identical load/store state slots"
                )
            token_mask = load_live
            active_lanes = token_mask
            initial_state = None
            conv_scratch_state = None
            conv_qsl = torch.arange(batch + 1, dtype=torch.int32, device=load_indices.device)
            prefill_num_seqs = 0
            prefill_seq_len = 0
            cu_seqlens = None
        else:
            lengths = _layer_sequence_lengths(metadata, batch, num_tokens, load_indices.device)
            if lengths.numel() != batch:
                raise ValueError(f"{self.prefix}: expected {batch} sequence lengths, got {lengths.numel()}")
            active_lanes = lengths > 0
            bad_store = active_lanes & (store_indices < 0)
            if bool(bad_store.any()):
                raise IndexError(f"{self.prefix}: live prefill sequence has a padding store slot")
            if has_initial_state is not None:
                missing_initial = active_lanes & has_initial_state.bool() & (load_indices < 0)
                if bool(missing_initial.any()):
                    raise IndexError(
                        f"{self.prefix}: prefill sequence requires an initial state but its load slot is padding"
                    )

            rectangular = num_tokens % batch == 0
            seq_len = num_tokens // batch if rectangular else 0
            padded_bucket = padding_mask_flat is not None
            if rectangular:
                if seq_len <= 0:
                    raise ValueError(f"{self.prefix}: rectangular prefill bucket has zero sequence length")
                offsets_end = None
                if query_start_loc is not None:
                    offsets_end = int(query_start_loc.reshape(-1)[-1].item())
                if (not padded_bucket and offsets_end == num_tokens
                        and bool((lengths != seq_len).any())):
                    rectangular = False
            if rectangular:
                if bool((lengths > seq_len).any()):
                    raise ValueError(
                        f"{self.prefix}: sequence length exceeds rectangular bucket width {seq_len}"
                    )
                token_mask = _build_token_mask(
                    padding_mask_flat,
                    lengths,
                    seq_len,
                    num_tokens,
                    load_indices.device,
                )
                prefill_num_seqs = batch
                prefill_seq_len = seq_len
                cu_seqlens = None
                conv_qsl = torch.cumsum(lengths, dim=0, dtype=torch.int32)
                conv_qsl = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32, device=load_indices.device),
                        conv_qsl,
                    ]
                )
            else:
                if query_start_loc is None:
                    raise ValueError(f"{self.prefix}: ragged prefill requires query_start_loc")
                offsets = query_start_loc.reshape(-1).to(device=load_indices.device, dtype=torch.int32)
                if offsets.numel() != batch + 1 or int(offsets[-1].item()) != num_tokens:
                    raise ValueError(
                        f"{self.prefix}: ragged prefill offsets must cover exactly {num_tokens} tokens"
                    )
                token_mask = torch.ones(num_tokens, dtype=torch.bool, device=load_indices.device)
                prefill_num_seqs = 0
                prefill_seq_len = 0
                cu_seqlens = offsets
                conv_qsl = offsets

            initial_state = _input_state(
                ssm_state,
                load_indices,
                active_lanes,
                has_initial_state,
            )
            conv_scratch_state = _input_state(
                conv_state,
                load_indices,
                active_lanes,
                has_initial_state,
            )

        return _KDAMetadata(
            is_prompt=is_prompt,
            conv_state=conv_state,
            conv_scratch_state=conv_scratch_state,
            ssm_state=ssm_state,
            load_indices=load_indices,
            store_indices=store_indices,
            query_start_loc=conv_qsl,
            token_mask=token_mask,
            active_lanes=active_lanes,
            initial_state=initial_state,
            has_initial_state=has_initial_state,
            prefill_num_seqs=prefill_num_seqs,
            prefill_seq_len=prefill_seq_len,
            chunk_size=64,
            cu_seqlens=cu_seqlens,
        )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        num_tokens = hidden_states.size(0)

        projected = self.in_proj_qkvbfg_a(hidden_states)[0]
        qkv, beta_raw, f_a, g_a = projected.split(
            [
                3 * self.local_projection_size,
                self.local_num_heads,
                self.head_dim,
                self.head_dim,
            ],
            dim=-1,
        )
        g1 = self.f_b_proj(f_a)[0].reshape(-1, self.local_num_heads, self.head_dim)
        g2 = self.g_b_proj(g_a)[0].reshape(-1, self.local_num_heads, self.head_dim)
        gate = hpu_kda_gate(
            g1,
            self.A_log,
            self.dt_bias,
            self.kda_lower_bound,
            self.local_num_heads,
            self.head_dim,
        )
        beta = torch.sigmoid(beta_raw.float()).to(hidden_states.dtype)

        metadata = self._extract_metadata(num_tokens)
        core = torch.zeros(
            (num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if metadata is not None:
            token_mask = metadata.token_mask
            if token_mask.numel() != num_tokens:
                raise ValueError(
                    f"{self.prefix}: token mask has {token_mask.numel()} entries, expected {num_tokens}"
                )
            qkv = torch.where(token_mask.view(-1, 1), qkv, torch.zeros_like(qkv))
            gate = torch.where(token_mask.view(-1, 1, 1), gate, torch.zeros_like(gate))
            beta = torch.where(token_mask.view(-1, 1), beta, torch.zeros_like(beta))

            if self._merged_conv_weight is None:
                self._build_merged_conv_weight()
            conv_weights = self._merged_conv_weight
            conv_bias = self.q_conv1d.bias

            if metadata.is_prompt:
                if metadata.conv_scratch_state is None:
                    raise RuntimeError(f"{self.prefix}: prefill is missing per-lane conv state")
                lane_indices = torch.arange(
                    metadata.conv_scratch_state.shape[0],
                    dtype=torch.long,
                    device=metadata.conv_scratch_state.device,
                )
                mixed = hpu_causal_conv1d_fn(
                    x=qkv.transpose(0, 1),
                    weight=conv_weights,
                    bias=conv_bias,
                    conv_states=metadata.conv_scratch_state,
                    query_start_loc=metadata.query_start_loc,
                    cache_indices=lane_indices,
                    has_initial_state=metadata.has_initial_state,
                    activation="silu",
                    is_prompt=True,
                ).transpose(0, 1)
                mixed = torch.where(token_mask.view(-1, 1), mixed, torch.zeros_like(mixed))
                q, k, v = mixed.split(self.local_projection_size, dim=-1)
                q = q.reshape(1, -1, self.local_num_heads, self.head_dim)
                k = k.reshape(1, -1, self.local_num_heads, self.head_dim)
                v = v.reshape(1, -1, self.local_num_heads, self.head_dim)
                core, final_state = hpu_chunk_kda(
                    q,
                    k,
                    v,
                    gate.unsqueeze(0),
                    beta.unsqueeze(0),
                    initial_state=metadata.initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    chunk_size=metadata.chunk_size,
                    prefill_num_seqs=metadata.prefill_num_seqs,
                    prefill_seq_len=metadata.prefill_seq_len,
                    cu_seqlens=metadata.cu_seqlens,
                )
                if final_state is None:
                    raise RuntimeError(f"{self.prefix}: chunk KDA did not return a final state")
                _scatter_active_states(
                    mixed,
                    metadata.conv_scratch_state,
                    metadata.conv_state,
                    metadata.store_indices,
                    metadata.active_lanes,
                )
                core = _scatter_active_states(
                    core,
                    final_state,
                    metadata.ssm_state,
                    metadata.store_indices,
                    metadata.active_lanes,
                ).reshape(-1, self.local_num_heads, self.head_dim)
            else:
                mixed = hpu_causal_conv1d_update(
                    x=qkv,
                    conv_state=metadata.conv_state,
                    weight=conv_weights,
                    bias=conv_bias,
                    activation="silu",
                    conv_state_indices=metadata.load_indices,
                    query_start_loc=metadata.query_start_loc,
                )
                mixed = torch.where(token_mask.view(-1, 1), mixed, torch.zeros_like(mixed))
                q, k, v = mixed.split(self.local_projection_size, dim=-1)
                q = q.reshape(1, -1, self.local_num_heads, self.head_dim)
                k = k.reshape(1, -1, self.local_num_heads, self.head_dim)
                v = v.reshape(1, -1, self.local_num_heads, self.head_dim)
                core, _ = hpu_fused_recurrent_kda(
                    q,
                    k,
                    v,
                    gate.unsqueeze(0),
                    beta.unsqueeze(0),
                    initial_state=metadata.ssm_state,
                    inplace_final_state=True,
                    ssm_state_indices=metadata.load_indices,
                    use_qk_l2norm_in_kernel=True,
                )
                core = core.reshape(-1, self.local_num_heads, self.head_dim)
                core = torch.where(token_mask.view(-1, 1, 1), core, torch.zeros_like(core))

        core = rmsnorm_sigmoid_gate(core, g2, self.o_norm.weight, self.o_norm.eps)
        if metadata is not None:
            core = torch.where(metadata.token_mask.view(-1, 1, 1), core, torch.zeros_like(core))
        output = self.o_proj(core.reshape(core.size(0), -1))[0]
        return output.view(orig_shape)


def init_merged_conv_weights(model: torch.nn.Module) -> int:
    """Eagerly materialize merged q|k|v convolution weights after loading.

    This must run before HPU compile-only warmup. A lazy ``torch.cat`` executed
    inside a compile-only region can cache an uninitialized tensor.
    """
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, HPUGlm5NextLinearAttention):
                module._build_merged_conv_weight()
                count += 1
    return count


# Glm5NextDecoderLayer imported the upstream class by value, so patch both
# modules after defining the HPU implementation.
import vllm.models.glm5next.nvidia.kda as _kda_module  # noqa: E402
import vllm.models.glm5next.nvidia.model as _model_module  # noqa: E402

_kda_module.Glm5NextLinearAttention = HPUGlm5NextLinearAttention
_model_module.Glm5NextLinearAttention = HPUGlm5NextLinearAttention

__all__ = ["HPUGlm5NextLinearAttention", "init_merged_conv_weights"]
