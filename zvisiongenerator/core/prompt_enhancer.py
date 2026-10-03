"""Define the prompt-enhancer protocol shared by the platform LLM adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class PromptEnhancer(Protocol):
    """A loaded local LLM that streams a chat completion."""

    repo: str
    revision: str | None

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int,
        max_tokens: int,
        temperature: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        """Yield generated text deltas for *messages*; stop early when *cancelled* returns True."""
        ...

    def close(self) -> None:
        """Drop the loaded weights and return accelerator memory."""
        ...
