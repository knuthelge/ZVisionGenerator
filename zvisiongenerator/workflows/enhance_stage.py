"""Pass-through step used by the image ``enhance_prompt_stage``; the video stage applies its scene and action pair itself."""

from __future__ import annotations

from typing import Any


def apply_planned_enhancement(request: Any, artifacts: Any) -> None:
    """Use the request's preflight rewrite as the resolved prompt; no-op when the iteration was not enhanced.

    The enhanced text is also stored as ``artifacts.metadata["enhanced_prompt"]`` (the embedded
    config records it) and reported through ``request.on_prompt_enhanced``. No LLM runs here.
    """
    text = request.enhanced_prompt
    if text is None:
        return
    artifacts.resolved_prompt = text
    artifacts.metadata["enhanced_prompt"] = text
    print(f"Enhanced prompt:\n{' '.join(text.split())}\n")
    if request.on_prompt_enhanced is not None:
        request.on_prompt_enhanced(text)
