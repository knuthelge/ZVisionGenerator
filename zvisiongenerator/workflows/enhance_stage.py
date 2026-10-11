"""Pass-through step that applies a rewrite planned during preflight; shared by the image and video ``enhance_prompt_stage``."""

from __future__ import annotations

from typing import Any


def apply_planned_enhancement(
    request: Any,
    artifacts: Any,
    *,
    resolved_prompt: str | None = None,
    extra_metadata: dict[str, str] | None = None,
) -> None:
    """Use the request's preflight rewrite as the resolved prompt; no-op when the iteration was not enhanced.

    The enhanced text is also stored as ``artifacts.metadata["enhanced_prompt"]`` (the embedded
    config records it) and reported through ``request.on_prompt_enhanced``. No LLM runs here.

    Args:
        request: The generation request; ``enhanced_prompt`` and ``on_prompt_enhanced`` are read from it.
        artifacts: The working artifacts that receive the resolved prompt and the metadata.
        resolved_prompt: The text to resolve to, print and report when it differs from the stored rewrite
            (a video's scene and action composed). Defaults to the rewrite itself.
        extra_metadata: Further entries stored in ``artifacts.metadata`` next to ``enhanced_prompt``.
    """
    text = request.enhanced_prompt
    if text is None:
        return
    shown = text if resolved_prompt is None else resolved_prompt
    artifacts.resolved_prompt = shown
    artifacts.metadata["enhanced_prompt"] = text
    if extra_metadata:
        artifacts.metadata.update(extra_metadata)
    print(f"Enhanced prompt:\n{' '.join(shown.split())}\n")
    if request.on_prompt_enhanced is not None:
        request.on_prompt_enhanced(shown)
