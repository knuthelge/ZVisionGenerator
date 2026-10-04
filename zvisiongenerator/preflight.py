"""Preflight phase — plan and enhance every prompt of a job before the generation model loads.

Preflight owns the prompt enhancer for the whole job: it loads the enhancer, rewrites every
prompt that asks for it, unloads the enhancer and frees accelerator memory. Only then does the
entry point load the generation model, so the two models are never resident together.
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends import prompt_enhancer_session, release_accelerator_memory
from zvisiongenerator.core.job_plan import EnhanceStatus, IterationPlan, JobPlan
from zvisiongenerator.core.progress_events import ProgressCallback, emit_progress
from zvisiongenerator.utils.prompt_compose import expand_random_choices
from zvisiongenerator.utils.prompt_enhance import EnhanceSettings, enhance_options, enhance_prompt, entry_enhance, resolve_enhance_ceiling, resolve_item_enhance

if TYPE_CHECKING:
    from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession
    from zvisiongenerator.core.prompt_enhancer import PromptEnhancer
    from zvisiongenerator.utils.interactive import SkipSignal

__all__ = ["plan_iterations", "run_preflight"]

_MODES = ("image", "video")

_PHASE_MESSAGES = {
    "downloading": "Downloading prompt enhancer model (first use only)...",
    "loading": "Loading prompt enhancer...",
    "cpu": "Prompt enhancer runs on the CPU (no CUDA GPU); each prompt may take a few minutes.",
}


def plan_iterations(
    prompts_data: dict[str, list[tuple[str, str | None]]],
    *,
    runs: int,
    seed: int | None,
    seed_min: int,
    seed_max: int,
    json_prompt: bool,
    disabled: bool,
    override: EnhanceSettings | None,
    enhance_by_set: dict[str, list[EnhanceSettings | None]] | None,
) -> tuple[IterationPlan, ...]:
    """Enumerate the job's iterations with seeds, resolved prompts and enhancement settings (no model I/O).

    Iterations are ordered runs → sets (in *prompts_data* order) → prompts, the order the runners loop in.

    Args:
        prompts_data: Set name → list of ``(prompt, negative_prompt)`` tuples.
        runs: Number of passes over every prompt.
        seed: Fixed seed for every iteration, or ``None`` to draw one per iteration.
        seed_min: Lower bound for drawn seeds.
        seed_max: Upper bound for drawn seeds.
        json_prompt: Prompts are structured JSON captions: kept verbatim and never enhanced.
        disabled: ``--no-enhance`` was given.
        override: Job-wide enhancement settings (``--enhance``).
        enhance_by_set: Per-entry prompt-file ``enhance:`` settings aligned with *prompts_data*.
    """
    iterations: list[IterationPlan] = []
    for run_index in range(runs):
        for set_name, prompts in prompts_data.items():
            for prompt_index, (prompt, _negative) in enumerate(prompts):
                enhance = None if json_prompt else resolve_item_enhance(disabled=disabled, override=override, entry=entry_enhance(enhance_by_set, set_name, prompt_index))
                iterations.append(
                    IterationPlan(
                        run_index=run_index,
                        set_name=set_name,
                        prompt_index=prompt_index,
                        prompt=prompt,
                        seed=seed if seed is not None else random.randint(seed_min, seed_max),
                        resolved_prompt=prompt if json_prompt else expand_random_choices(prompt),
                        enhance=enhance,
                    )
                )
    return tuple(iterations)


def run_preflight(
    prompts_data: dict[str, list[tuple[str, str | None]]],
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    mode: str,
    model_family: str | None,
    enhance_by_set: dict[str, list[EnhanceSettings | None]] | None = None,
    control: SkipSignal | None = None,
    progress_callback: ProgressCallback | None = None,
    session: PromptEnhancerSession | None = None,
    release_memory: Callable[[], None] | None = None,
) -> JobPlan:
    """Plan every iteration and rewrite its prompt before the generation model loads.

    Args:
        prompts_data: Set name → list of ``(prompt, negative_prompt)`` tuples.
        config: Loaded config dict (seed bounds, enhancer model and options, word ceilings).
        args: Job arguments; reads ``runs``, ``seed``, ``no_enhance``, ``enhance``,
            ``enhance_model`` and ``json_prompt_enabled``.
        mode: ``"image"`` or ``"video"``.
        model_family: Generation model family, for the enhanced-prompt word ceiling.
        enhance_by_set: Per-entry prompt-file ``enhance:`` settings aligned with *prompts_data*.
        control: The job's control signal; Next skips a rewrite, Pause waits, Quit cancels.
        progress_callback: Receives the preflight events.
        session: Enhancer session (defaults to the process-wide one).
        release_memory: Frees accelerator memory after the enhancer closes (defaults to
            ``release_accelerator_memory``, looked up at call time).

    Returns:
        The job plan; ``cancelled`` is set when the user quit during preflight.

    Raises:
        ValueError: For a bad *mode*, no enhancer configured, or a malformed ``--enhance-model``.
        RuntimeError: When the enhancer is missing while offline, or fails to load.
    """
    if mode not in _MODES:
        raise ValueError(f"Preflight mode must be one of {', '.join(_MODES)}, got '{mode}'.")
    generation = config.get("generation", {})
    disabled = bool(getattr(args, "no_enhance", False))
    override = getattr(args, "enhance", None)
    iterations = plan_iterations(
        prompts_data,
        runs=args.runs,
        seed=args.seed,
        seed_min=generation.get("seed_min", 4),
        seed_max=generation.get("seed_max", 2**32 - 1),
        json_prompt=bool(getattr(args, "json_prompt_enabled", False)),
        disabled=disabled,
        override=override,
        enhance_by_set=enhance_by_set,
    )
    positions = tuple(index for index, iteration in enumerate(iterations) if iteration.enhance is not None)
    total_iterations = len(iterations)
    emit_progress(progress_callback, "preflight_started", mode=mode, total_iterations=total_iterations, total_rewrites=len(positions))
    if not positions:
        # Nothing to rewrite: free an idle resident enhancer before the generation model loads.
        if session is not None:
            session.release()
        else:
            prompt_enhancer_session.release_resident_enhancer()
        return _finish(JobPlan(iterations), mode=mode, progress_callback=progress_callback)

    reference = prompt_enhancer_session.plan_job_enhancer(
        config, platform_key=sys.platform, disabled=disabled, override=override, enhance_by_set=enhance_by_set, cli_model=getattr(args, "enhance_model", None)
    )
    if reference is None:
        raise RuntimeError("Prompt enhancement was requested but no enhancer model could be planned.")
    pause = _make_pause(control, mode=mode, total_iterations=total_iterations, progress_callback=progress_callback)
    rewrite = _make_rewriter(config, mode=mode, model_family=model_family)

    def _on_phase(phase: str) -> None:
        _print_enhancer_phase(phase)
        emit_progress(progress_callback, "enhancer_loading", mode=mode, phase=phase)

    with prompt_enhancer_session.job_enhancer(*reference, on_phase=_on_phase, session=session) as enhancer:
        if prompt_enhancer_session.runs_on_cpu(enhancer):
            _on_phase("cpu")
        if control is not None:
            print("Commands: [n] skip rewrite  [q] quit  [p] pause\n")
        planned, cancelled = _rewrite_all(enhancer, iterations, positions, rewrite=rewrite, control=control, pause=pause, mode=mode, progress_callback=progress_callback)
    # Reached only on the normal paths (success or quit); on an exception close() already freed the enhancer.
    (release_memory or _release_accelerator_memory)()
    if not cancelled:
        cancelled = _boundary(control, pause) == "quit"
    if cancelled:
        print("\n⏹ Quitting batch...")
        emit_progress(progress_callback, "batch_cancelled", mode=mode, completed_iterations=0, total_iterations=total_iterations)
        return JobPlan(planned, cancelled=True)
    return _finish(JobPlan(planned), mode=mode, progress_callback=progress_callback)


def _rewrite_all(
    enhancer: PromptEnhancer,
    iterations: tuple[IterationPlan, ...],
    positions: tuple[int, ...],
    *,
    rewrite: Callable[[PromptEnhancer, IterationPlan, Callable[[], bool] | None], tuple[IterationPlan, str | None]],
    control: SkipSignal | None,
    pause: Callable[[], str | None],
    mode: str,
    progress_callback: ProgressCallback | None,
) -> tuple[tuple[IterationPlan, ...], bool]:
    """Rewrite the iterations at *positions*; return the updated iterations and whether the user quit."""
    planned = list(iterations)
    total = len(positions)
    for index, position in enumerate(positions, start=1):
        action = _boundary(control, pause)
        if action == "quit":
            return tuple(planned), True
        emit_progress(progress_callback, "prompts_enhancing", mode=mode, index=index, total=total)
        print(f"Enhancing prompt {index}/{total}...")
        if action == "skip":
            planned[position] = replace(planned[position], enhance_status=EnhanceStatus.SKIPPED)
            continue
        cancelled, stopped = _stop_tracker(control)
        result, message = rewrite(enhancer, planned[position], cancelled)
        if message is None:
            planned[position] = result
            continue
        if stopped():
            # The user stopped the LLM (Next, Quit, or Next followed by Pause): never a failure.
            planned[position] = replace(planned[position], enhance_status=EnhanceStatus.SKIPPED)
            if _after_stop(control, pause) == "quit":
                return tuple(planned), True
            continue
        planned[position] = replace(planned[position], enhance_status=EnhanceStatus.FAILED)
        warnings.warn(f"Prompt enhancement failed ({message}); using the original prompt.", stacklevel=2)
        emit_progress(progress_callback, "prompt_enhance_failed", mode=mode, index=index, total=total, message=message)
    return tuple(planned), False


def _make_rewriter(config: dict[str, Any], *, mode: str, model_family: str | None) -> Callable[[PromptEnhancer, IterationPlan, Callable[[], bool] | None], tuple[IterationPlan, str | None]]:
    """Bind the job's word ceiling and enhancer options into a single-iteration rewrite function."""
    ceiling = resolve_enhance_ceiling(config, family=model_family, mode=mode)
    options = enhance_options(config)

    def _rewrite(enhancer: PromptEnhancer, iteration: IterationPlan, cancelled: Callable[[], bool] | None) -> tuple[IterationPlan, str | None]:
        """Return the enhanced iteration, or the unchanged one plus the failure message (never the exception)."""
        try:
            result = enhance_prompt(
                enhancer,
                iteration.resolved_prompt,
                iteration.enhance,
                mode=mode,
                seed=iteration.seed,
                ceiling=ceiling,
                length_cfg=options["length"],
                temperature=options["temperature"],
                max_tokens=options["max_new_tokens"],
                cancelled=cancelled,
            )
        except Exception as exc:  # noqa: BLE001 - enhancement is optional: any adapter/template failure falls back to the prompt
            return iteration, str(exc) or type(exc).__name__
        return replace(iteration, enhanced_prompt=result.prompt, enhance_status=EnhanceStatus.ENHANCED), None

    return _rewrite


def _stop_tracker(control: SkipSignal | None) -> tuple[Callable[[], bool] | None, Callable[[], bool]]:
    """Return the LLM ``cancelled`` callback and a reader for whether it ever stopped the rewrite."""
    if control is None:
        return None, lambda: False
    stopped = False

    def _cancelled() -> bool:
        nonlocal stopped
        if control.check():
            stopped = True
        return stopped

    return _cancelled, lambda: stopped


def _boundary(control: SkipSignal | None, pause: Callable[[], str | None]) -> str | None:
    """Consume the pending control action; return ``"quit"``, ``"skip"`` or ``None`` (pause waits here)."""
    if control is None:
        return None
    action = control.consume()
    if action == "pause":
        action = pause()
    return action if action in ("quit", "skip") else None


def _after_stop(control: SkipSignal | None, pause: Callable[[], str | None]) -> str | None:
    """Consume the action that stopped a rewrite; return ``"quit"`` or ``None`` (a pause waits here)."""
    if control is None:
        return None
    action = control.consume()
    if action == "pause":
        action = pause()
    return "quit" if action == "quit" else None


def _make_pause(control: SkipSignal | None, *, mode: str, total_iterations: int, progress_callback: ProgressCallback | None) -> Callable[[], str | None]:
    """Return a function that pauses the job until resumed and returns the action queued meanwhile."""

    def _pause() -> str | None:
        if control is None:
            return None
        emit_progress(progress_callback, "job_paused", mode=mode, completed_iterations=0, total_iterations=total_iterations)
        print("\n⏸ Paused. Press any key to continue...")
        control.wait_for_key()
        action = control.consume()
        emit_progress(progress_callback, "job_resumed", mode=mode, completed_iterations=0, total_iterations=total_iterations)
        print("▶ Resumed.\n")
        return action

    return _pause


def _finish(plan: JobPlan, *, mode: str, progress_callback: ProgressCallback | None) -> JobPlan:
    """Report the rewrite summary and emit ``preflight_finished``; return *plan* unchanged."""
    counts = {status: sum(1 for iteration in plan.iterations if iteration.enhance_status is status) for status in EnhanceStatus}
    requested = sum(1 for iteration in plan.iterations if iteration.enhance is not None)
    enhanced, failed, skipped = counts[EnhanceStatus.ENHANCED], counts[EnhanceStatus.FAILED], counts[EnhanceStatus.SKIPPED]
    if requested:
        print(f"Enhanced {enhanced} of {requested} prompts ({failed} failed, {skipped} skipped).")
    emit_progress(progress_callback, "preflight_finished", mode=mode, total_iterations=len(plan.iterations), enhanced=enhanced, failed=failed, skipped=skipped)
    return plan


def _print_enhancer_phase(phase: str) -> None:
    """Print a status line while the enhancer downloads or loads."""
    print(_PHASE_MESSAGES.get(phase, phase))


def _release_accelerator_memory() -> None:
    """Free the closed enhancer's accelerator memory; a cleanup failure only warns."""
    try:
        release_accelerator_memory()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not release accelerator memory after prompt enhancement: {exc}", stacklevel=2)
