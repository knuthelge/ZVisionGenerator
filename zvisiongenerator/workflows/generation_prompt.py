"""The prompt and negative prompt an image generation sends to the backend, chosen in one place for the stages and for the prompt pre-encoding."""

from __future__ import annotations

from zvisiongenerator.core.ideogram4 import is_json_caption

__all__ = ["BACKGROUND_SUFFIX", "conditioning_prompt", "generation_negative", "generation_prompt"]

# Asks the model for a plain backdrop, so the cut-out has an even background to separate the subject from.
BACKGROUND_SUFFIX = ", isolated on a plain light grey studio background"


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


def conditioning_prompt(text: str, *, transparent: bool) -> str:
    """Return the text a model conditions on: *text*, plus the plain-background phrase when the image will be cut out.

    Only text that reaches a model is conditioned. The saved, displayed and recorded prompt stays the one the user wrote.
    The phrase is not deduplicated, because recorded prompts never carry it.

    Args:
        text: The prompt from :func:`generation_prompt`.
        transparent: Whether the image is cut out and saved with transparency.

    Returns:
        *text* with the background phrase appended, or *text* unchanged when *transparent* is false or *text* is a JSON caption.
    """
    if not transparent or is_json_caption(text):
        return text
    return text + BACKGROUND_SUFFIX


def generation_negative(negative: str | None, *, supports_negative: bool) -> str | None:
    """Return the negative prompt a generation sends: the given one when the model takes negative prompts, else ``None``.

    Args:
        negative: The negative prompt as written, or ``None`` when there is none.
        supports_negative: Whether the model takes negative prompts.

    Returns:
        *negative* when *supports_negative*, else ``None``.
    """
    return negative if supports_negative else None
