"""The prompt and negative prompt an image generation sends to the backend, chosen in one place for the stages and for the prompt pre-encoding."""

from __future__ import annotations

__all__ = ["generation_negative", "generation_prompt"]


def generation_prompt(template: str, resolved: str | None, enhanced: str | None = None) -> str:
    """Return the text a generation conditions on: the planned rewrite, else the resolved prompt, else the template.

    Args:
        template: The prompt as written, with its ``{a|b}`` choices unexpanded.
        resolved: The prompt with its choices expanded, or ``None`` when none was made.
        enhanced: The planned rewrite, or ``None`` when the iteration is not enhanced.

    Returns:
        *enhanced* when set, else *resolved*, else *template*; an empty string counts as unset.
    """
    chosen = enhanced if enhanced is not None else resolved
    return chosen or template


def generation_negative(negative: str | None, *, supports_negative: bool) -> str | None:
    """Return the negative prompt a generation sends: the given one when the model takes negative prompts, else ``None``.

    Args:
        negative: The negative prompt as written, or ``None`` when there is none.
        supports_negative: Whether the model takes negative prompts.

    Returns:
        *negative* when *supports_negative*, else ``None``.
    """
    return negative if supports_negative else None
