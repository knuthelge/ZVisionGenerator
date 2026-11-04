"""Exact speed-ups for YuE2's token phases on MPS, applied around the kit's calls without editing it.

Two swaps, each giving the same logits as the kit:

- :func:`folded_decode_attention`: the kit's attention copies every key and value head to match the query heads on
  MPS. For one new token with no mask, folding the query heads that share a key head into the query axis attends the
  same keys without the copy, whose cost grows with the song.
- :func:`semantic_head`: song tokens can only be codec tokens or the end of music, and the kit's sampler sets every other
  logit to ``-inf`` before it reads any of them. The head computes only those rows and fills the rest with zeros, which
  stay finite through the kit's guidance mix (``-inf`` there would turn into NaN).

The kit is patched for the length of a ``with`` block only, and both swaps are undone on exit.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = ["folded_decode_attention", "semantic_head"]


def _fold_grouped_decode(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Attend one query position per head over grouped key and value heads, without repeating them.

    The query heads that share a key head (``h // groups``, as ``repeat_interleave`` pairs them) become that head's query
    rows, so one attention call covers them all.

    Args:
        query: ``[batch, heads, 1, dim]``.
        key: ``[batch, kv_heads, keys, dim]``, where ``kv_heads`` divides ``heads``.
        value: Same shape as *key*.

    Returns:
        ``[batch, heads, 1, dim]``, as the attention over repeated key and value heads returns it.
    """
    import torch.nn.functional as F

    batch, heads, _, dim = query.shape
    kv_heads = key.shape[1]
    folded = query.reshape(batch, kv_heads, heads // kv_heads, dim)
    return F.scaled_dot_product_attention(folded, key, value).reshape(batch, heads, 1, dim)


@contextmanager
def folded_decode_attention() -> Iterator[None]:
    """Route the kit's single-token, unmasked grouped attention through :func:`_fold_grouped_decode` inside the block."""
    from yue2 import modeling_yue2

    original = modeling_yue2.sdpa

    def sdpa(query: Any, key: Any, value: Any, *, attn_mask: Any = None, is_causal: bool = False) -> Any:
        if query.shape[2] == 1 and attn_mask is None and not is_causal and query.shape[1] != key.shape[1]:
            return _fold_grouped_decode(query, key, value)
        return original(query, key, value, attn_mask=attn_mask, is_causal=is_causal)

    modeling_yue2.sdpa = sdpa
    try:
        yield
    finally:
        modeling_yue2.sdpa = original


def _subset_head(lm_head: torch.nn.Linear, rows: torch.Tensor) -> torch.nn.Module:
    """Return a head that computes *lm_head* on *rows* only and gives zero for every other row."""
    import torch
    import torch.nn.functional as F

    class SubsetHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vocab = lm_head.weight.shape[0]
            self.register_buffer("rows", rows, persistent=False)
            self.register_buffer("weight", lm_head.weight.detach().index_select(0, rows), persistent=False)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            logits = hidden.new_zeros((*hidden.shape[:-1], self.vocab))
            return logits.index_copy_(-1, self.rows, F.linear(hidden, self.weight))

    return SubsetHead()


@contextmanager
def semantic_head(model: Any) -> Iterator[None]:
    """Give *model* a head that computes only the logits song tokens can take (codec tokens and the end of music).

    Every other row comes out as zero; the kit's sampler sets those rows to ``-inf`` in the song phase before it reads them.
    The original head is put back on exit, also when the block raises.
    """
    import torch
    from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END

    original = model.lm_head
    device = original.weight.device
    rows = torch.cat([torch.arange(CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE, device=device), torch.tensor([MUSIC_END], device=device)])
    model.lm_head = _subset_head(original, rows)
    try:
        yield
    finally:
        model.lm_head = original
