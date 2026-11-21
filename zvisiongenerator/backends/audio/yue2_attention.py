"""Pick the attention of YuE2's CUDA-graph token decoder, applied around the kit's calls without editing it.

The kit's ``auto`` choice takes FlashAttention whenever the ``aten`` operator exists with the variable-length signature it
needs. A PyTorch build made without FlashAttention (the Windows wheels) has that operator but no kernel behind it, so the first
token of every take fails. When the build cannot run FlashAttention the decoder is asked for public SDPA, which every CUDA
build runs; otherwise the kit keeps its own choice.

The kit is patched for the length of a ``with`` block only, and the patch is undone on exit.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["graph_attention", "graph_attention_backend"]


def graph_attention_backend() -> str:
    """Return the kit's ``attention_backend`` for the CUDA-graph decoder: ``"auto"`` when this PyTorch can run FlashAttention, else ``"sdpa"``."""
    import torch

    return "auto" if torch.backends.cuda.is_flash_attention_available() else "sdpa"


@contextmanager
def graph_attention(device: str) -> Iterator[None]:
    """Make the kit's CUDA-graph decoder use :func:`graph_attention_backend` inside the block; other devices run unchanged."""
    backend = graph_attention_backend() if device == "cuda" else "auto"
    if backend == "auto":
        yield
        return

    from yue2 import cuda_graph

    original = cuda_graph.GraphAR

    class GraphAR(original):  # type: ignore[valid-type, misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs.setdefault("attention_backend", backend)
            super().__init__(*args, **kwargs)

    cuda_graph.GraphAR = GraphAR
    try:
        yield
    finally:
        cuda_graph.GraphAR = original
