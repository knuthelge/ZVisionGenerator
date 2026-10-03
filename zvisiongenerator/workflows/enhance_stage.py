"""Shared auto-enhance step used by the image and video ``enhance_prompt_stage`` functions."""

from __future__ import annotations

import warnings
from typing import Any

from zvisiongenerator.utils.prompt_enhance import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, enhance_prompt


def apply_prompt_enhancement(request: Any, artifacts: Any, *, mode: str) -> None:
    """Rewrite ``artifacts.resolved_prompt`` with the request's enhancer; keep the prompt on any failure.

    No-op when ``request.enhance`` is unset or the request is a structured JSON caption.
    On success the enhanced text is also stored as ``artifacts.metadata["enhanced_prompt"]``
    (the embedded config records it) and reported through ``request.on_prompt_enhanced``.
    """
    settings = request.enhance
    if settings is None or getattr(request, "json_prompt", False):
        return
    if request.prompt_enhancer is None:
        warnings.warn("Prompt enhancement was requested but no enhancer is loaded; using the original prompt.", stacklevel=2)
        return
    prompt = artifacts.resolved_prompt or request.prompt
    options = request.enhance_options or {}
    skip_signal = getattr(request, "skip_signal", None)
    try:
        result = enhance_prompt(
            request.prompt_enhancer,
            prompt,
            settings,
            mode=mode,
            seed=request.seed,
            ceiling=request.enhance_ceiling,
            length_cfg=options.get("length"),
            temperature=options.get("temperature", DEFAULT_TEMPERATURE),
            max_tokens=options.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            # Next/Quit (Web controls or CLI keys) stop the LLM too; the generation stage then honours them.
            cancelled=skip_signal.check if skip_signal is not None else None,
        )
    except Exception as exc:  # noqa: BLE001 - enhancement is optional: any adapter/template failure falls back to the prompt
        warnings.warn(f"Prompt enhancement failed ({exc}); using the original prompt.", stacklevel=2)
        return
    artifacts.resolved_prompt = result.prompt
    artifacts.metadata["enhanced_prompt"] = result.prompt
    print(f"Enhanced prompt:\n{' '.join(result.prompt.split())}\n")
    if request.on_prompt_enhanced is not None:
        request.on_prompt_enhanced(result.prompt)
