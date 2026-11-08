"""Decode the prompt-enhancer LLM one token per CUDA-graph replay."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["graph_tokens"]

# transformers' default top-k; its default top-p of 1.0 does nothing.
_TOP_K = 50


def _choose_token(logits: Tensor, seen: Tensor, temperature: float, *, penalty: float) -> Tensor:
    """Return the next token id as a ``(1,)`` long tensor, using only operations a CUDA graph can capture.

    Args:
        logits: Next-token logits, shape ``(vocab,)``.
        seen: Float32 0/1 mask of the tokens generated so far.
        temperature: At or below zero picks the argmax; above zero samples from the top 50.
        penalty: Subtracted once from the logit of every seen token.
    """
    import torch

    scores = logits.float() - penalty * seen
    if temperature <= 0:
        return scores.argmax().view(1)
    top, idx = scores.topk(_TOP_K)
    uniform = torch.rand_like(top)
    gumbel = -torch.log(-torch.log(uniform))
    return idx.gather(0, (top / temperature + gumbel).argmax().view(1))


def graph_tokens(model: Any, input_ids: Tensor, *, max_tokens: int, temperature: float, penalty: float) -> Iterator[int]:
    """Yield up to *max_tokens* generated token ids, replaying one captured CUDA graph per token after the second.

    Args:
        model: A Qwen3.5 causal LM on CUDA.
        input_ids: Prompt ids, shape ``(1, prompt_len)``.
        max_tokens: Number of tokens to generate; at least 2.
        temperature: At or below zero decodes greedily.
        penalty: Presence penalty subtracted from the logit of each token already generated.
    """
    import torch
    from transformers import StaticCache

    with torch.inference_mode():
        cache = StaticCache(config=model.config, max_cache_len=input_ids.shape[1] + max_tokens)
        logits = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[0, -1]
        seen = torch.zeros(logits.shape[-1], device=logits.device)
        ids = _choose_token(logits, seen, temperature, penalty=penalty).view(1, 1)
        seen.index_fill_(0, ids.view(1), 1.0)
    yield int(ids)

    def step() -> None:
        nxt = _choose_token(model(input_ids=ids, past_key_values=cache, use_cache=True).logits[0, -1], seen, temperature, penalty=penalty)
        seen.index_fill_(0, nxt, 1.0)
        ids.copy_(nxt.view(1, 1))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    yield int(ids)

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph, stream=side, capture_error_mode="thread_local"):
        step()
    for _ in range(max_tokens - 2):
        graph.replay()
        yield int(ids)
