"""Audio preflight — rewrite the text of a take with the prompt enhancer before the audio model loads.

The CLI counterpart of the Web UI's Enhance on a Sound prompt or a song's Style: the enhancer is loaded, rewrites the one text every take of the
job uses, is unloaded and has its accelerator memory freed. Only then does ``ziv audio`` fetch and load the audio model, so the two models are
never resident together. The rewritten take records the user's own text (``prompt_input``) and how it was written (``enhance_record``) exactly as a
Web UI take does.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends import prompt_enhancer_session
from zvisiongenerator.core.audio_types import AudioGenerationRequest
from zvisiongenerator.preflight import print_enhancer_licence, print_enhancer_phase, release_enhancer_memory
from zvisiongenerator.utils.config import model_reference
from zvisiongenerator.utils.enhance_record import build_audio_enhance_record, enhancer_tier
from zvisiongenerator.utils.prompt_enhance import EnhanceResult, enhance_options
from zvisiongenerator.utils.prompt_enhance_audio import TASK_LABELS, AudioEnhanceContext, AudioEnhanceSettings, enhance_audio_caption

if TYPE_CHECKING:
    from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
    from zvisiongenerator.utils.interactive import SkipSignal

__all__ = ["AudioPreflight", "run_audio_preflight"]


@dataclass(frozen=True)
class AudioPreflight:
    """What the preflight leaves for the job.

    Attributes:
        request: The take to make: the request given, or with the enhanced text as its ``prompt`` and the record of the rewrite.
        cancelled: The user quit during the rewrite; nothing may be made.
    """

    request: AudioGenerationRequest
    cancelled: bool = False


def run_audio_preflight(
    request: AudioGenerationRequest,
    settings: AudioEnhanceSettings,
    *,
    task: str,
    family: str,
    context: AudioEnhanceContext,
    own_text: str,
    config: dict[str, Any],
    seed: int,
    enhance_model: str | None = None,
    control: SkipSignal | None = None,
    session: PromptEnhancerSession | None = None,
    release_memory: Callable[[], None] | None = None,
) -> AudioPreflight:
    """Rewrite the text of *request* for the model of *family* and return the request that uses it.

    A rewrite that fails, or that the user skips (Next), leaves the request as it was, with a warning for a failure. Quit ends the job.

    Args:
        request: The take; its ``prompt`` is the text the rewrite works on.
        settings: The audio options of the *task*.
        task: ``sound`` or ``style``.
        family: The backend family of the audio model, which picks the caption rules.
        context: The take's facts that shape the rewrite (see ``audio_enhance_context``).
        own_text: The user's own text, recorded as the take's ``prompt_input``.
        config: Loaded config dict (enhancer model and options).
        seed: The seed of the rewrite.
        enhance_model: ``fast``, ``smart`` or ``REPO[@REVISION]`` from ``--enhance-model``.
        control: The job's control signal; Next skips the rewrite, Quit cancels.
        session: Enhancer session (defaults to the process-wide one).
        release_memory: Frees accelerator memory after the enhancer closes (defaults to ``release_enhancer_memory``).

    Returns:
        The preflight's outcome.

    Raises:
        ValueError: When no enhancer is configured or *enhance_model* is malformed.
        RuntimeError: When the enhancer is missing while offline, or fails to load.
    """
    if control is not None and control.pending() in ("quit", "skip"):
        return _stopped(request, control)
    reference = prompt_enhancer_session.plan_enhancer(config, platform_key=sys.platform, cli_model=enhance_model, budget_bytes=prompt_enhancer_session.smart_budget_bytes(enhance_model))
    print_enhancer_licence(config, reference)
    options = enhance_options(config)
    stop = _StopTracker(control)
    result: EnhanceResult | None = None
    failure = ""
    print(f"Enhancing the {TASK_LABELS[task].lower()}...")
    with prompt_enhancer_session.job_enhancer(*reference, on_phase=print_enhancer_phase, session=session) as enhancer:
        if prompt_enhancer_session.runs_on_cpu(enhancer):
            print_enhancer_phase("cpu")
        try:
            result = enhance_audio_caption(enhancer, request.prompt, settings, task=task, family=family, context=context, seed=seed, temperature=options["temperature"], cancelled=stop.cancelled)
        except Exception as exc:  # noqa: BLE001 - enhancement is optional: any adapter/template failure falls back to the prompt
            failure = str(exc) or type(exc).__name__
    # Reached on every path but a failure to load, where close() already freed the enhancer.
    (release_memory or release_enhancer_memory)()
    if result is None:
        if stop.stopped and control is not None:
            return _stopped(request, control)
        warnings.warn(f"Prompt enhancement failed ({failure}); using the original prompt.", stacklevel=2)
        return AudioPreflight(request)
    return AudioPreflight(_enhanced(request, result, settings, task=task, reference=reference, own_text=own_text, config=config))


class _StopTracker:
    """The LLM's ``cancelled`` callback for a job's control signal, which remembers that it ever stopped the rewrite."""

    def __init__(self, control: SkipSignal | None) -> None:
        self._control = control
        self.stopped = False

    def cancelled(self) -> bool:
        """Return whether Next or Quit has been pressed (and stay so)."""
        if self._control is not None and self._control.check():
            self.stopped = True
        return self.stopped


def _stopped(request: AudioGenerationRequest, control: SkipSignal) -> AudioPreflight:
    """Return the outcome of the Next or Quit the user pressed: the original text for Next, a cancelled job for Quit."""
    if control.consume() == "quit":
        return AudioPreflight(request, cancelled=True)
    print("Skipped the rewrite.")
    return AudioPreflight(request)


def _enhanced(
    request: AudioGenerationRequest,
    result: EnhanceResult,
    settings: AudioEnhanceSettings,
    *,
    task: str,
    reference: tuple[str, str | None],
    own_text: str,
    config: dict[str, Any],
) -> AudioGenerationRequest:
    """Return *request* with the enhanced text as its prompt, the user's own text as ``prompt_input`` and the record of how it was written."""
    print(f"Enhanced {TASK_LABELS[task].lower()}: {result.prompt}")
    tier = enhancer_tier(config, platform_key=sys.platform, model=model_reference(*reference))
    record = build_audio_enhance_record(settings, task=task, repo=reference[0], revision=reference[1], tier=tier)
    return replace(request, prompt=result.prompt, prompt_input=own_text, enhance_record=record)
