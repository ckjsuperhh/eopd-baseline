# Pure-Torch reimplementation of `flash_attn.bert_padding` packing helpers.
# Used by verl's `use_remove_padding=True` (varlen) path when flash_attn is
# unavailable. Semantics match flash_attn.bert_padding exactly.
import torch
import torch.nn.functional as F
from einops import rearrange as einops_rearrange


def index_first_axis(tensor, indices):
    """Select `indices` along the first axis of `tensor`."""
    return tensor.index_select(0, indices)


def rearrange(tensor, pattern, **kwargs):
    """einops.rearrange passthrough (flash_attn.bert_padding.rearrange is einops)."""
    return einops_rearrange(tensor, pattern, **kwargs)


def unpad_input(hidden_states, attention_mask):
    """
    Pack a padded (batch, seqlen, ...) tensor into a ragged (total_nnz, ...) tensor.

    Args:
        hidden_states: (batch, seqlen, *d)
        attention_mask: (batch, seqlen) bool / 0-1; 1 = real token.
    Returns:
        hidden_states_rmpad: (total_nnz, *d)
        indices: (total_nnz,) int64 — positions into the flattened (b*s) array.
        cu_seqlens: (batch + 1,) int32 — cumulative token counts per sequence.
        max_seqlen_in_batch: int — longest unpadded sequence length.
    """
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    flat_mask = attention_mask.reshape(-1).bool()
    indices = torch.nonzero(flat_mask, as_tuple=False).flatten()  # (total_nnz,)
    rest_shape = hidden_states.shape[2:]
    hidden_states_flat = hidden_states.reshape(batch_size * seq_len, *rest_shape)
    hidden_states_rmpad = hidden_states_flat[indices]

    row_counts = attention_mask.reshape(batch_size, seq_len).sum(dim=1).int()  # (b,)
    cu_seqlens = F.pad(row_counts.cumsum(dim=0), (1, 0))  # (b + 1,)
    max_seqlen_in_batch = int(row_counts.max().item()) if row_counts.numel() > 0 else 0
    return hidden_states_rmpad, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Inverse of `unpad_input`: scatter a ragged (total_nnz, *d) tensor back to
    (batch, seqlen, *d), padding with zeros.
    """
    rest_shape = hidden_states.shape[1:]
    output = hidden_states.new_zeros(batch * seqlen, *rest_shape)
    output[indices] = hidden_states
    return output.reshape(batch, seqlen, *rest_shape)
