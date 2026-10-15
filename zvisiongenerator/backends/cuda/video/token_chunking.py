"""Run token-wise layers over a long token sequence in chunks, to bound their activation memory.

Some transformer layers compute every token on its own: the feed-forward layer, which widens each token (four
times in LTX) before narrowing it again, and RMSNorm, which upcasts each token to float32. For a long video their
temporaries are the largest tensors of a step; splitting the sequence into chunks gives the same output with a
fraction of that memory.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["chunk_tokens"]


def chunk_tokens(modules: Iterable[Any], tokens_per_chunk: int) -> None:
    """Make each token-wise module in *modules* process its input ``(batch, tokens, dim)`` *tokens_per_chunk* tokens at a time.

    The module itself is kept (its parameters and their names are unchanged, so LoRAs still match); only its
    ``forward`` is wrapped. Inputs no longer than one chunk, or not shaped ``(batch, tokens, dim)``, run as before.

    Raises:
        ValueError: If *tokens_per_chunk* is not positive.
    """
    if tokens_per_chunk < 1:
        raise ValueError(f"tokens_per_chunk must be at least 1, got {tokens_per_chunk}.")
    for module in modules:
        module.forward = _chunked(module.forward, tokens_per_chunk)


def _chunked(forward: Any, tokens_per_chunk: int) -> Any:
    """Return *forward* applied chunk by chunk along the token axis (dimension 1)."""
    import torch

    def _forward(hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        if hidden_states.ndim != 3 or hidden_states.shape[1] <= tokens_per_chunk:
            return forward(hidden_states, *args, **kwargs)
        return torch.cat([forward(chunk, *args, **kwargs) for chunk in hidden_states.split(tokens_per_chunk, dim=1)], dim=1)

    return _forward
