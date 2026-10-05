"""Upscale one existing image: read its recorded settings, then run the upscale workflow with progress events."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.progress_events import ProgressCallback
from zvisiongenerator.core.progress_events import emit_generation_finished as _emit_generation_finished
from zvisiongenerator.core.progress_events import emit_progress as _emit_progress
from zvisiongenerator.core.progress_events import make_step_progress_callback as _make_step_progress_callback
from zvisiongenerator.core.progress_events import run_workflow_with_progress as _run_workflow_with_progress
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.workflow import GenerationWorkflow
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.provenance import RecordedSettings, image_prompt_text, read_png_config, recorded_settings


@dataclass(frozen=True)
class UpscaleSource:
    """An image to upscale: its file, size, rendered prompt and the settings recorded in it (None when it has none)."""

    path: str
    width: int
    height: int
    # The prompt text saved with the image; it keeps the source's {a|b} choices, while the settings hold the template.
    rendered_prompt: str | None = None
    settings: RecordedSettings | None = None


def read_upscale_source(path: str | Path) -> UpscaleSource:
    """Read an image's size, rendered prompt and recorded settings.

    Raises:
        FileNotFoundError: If *path* is not a file.
        ValueError: If *path* is not a readable image.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Image to upscale not found: {source}")
    try:
        with Image.open(source) as image:
            width, height = image.size
            rendered_prompt = image_prompt_text(image)
    except OSError as exc:
        raise ValueError(f"Cannot read {source.name} as an image: {exc}") from exc
    try:
        config = read_png_config(source)
    except ValueError:
        config = None
    settings = recorded_settings(config) if isinstance(config, dict) and config else None
    return UpscaleSource(path=str(source), width=width, height=height, rendered_prompt=rendered_prompt, settings=settings)


def run_upscale(
    backend: Any,
    model: Any,
    request: ImageGenerationRequest,
    workflow: GenerationWorkflow,
    *,
    progress_callback: ProgressCallback | None = None,
    skip_signal: SkipSignal | None = None,
) -> StageOutcome:
    """Run *workflow* once for *request* and emit the same progress events as a one-image batch.

    Returns:
        The workflow outcome; ``skipped`` when the job was stopped.
    """
    skip = skip_signal or SkipSignal()
    set_name = Path(request.upscale_source or "upscale").stem
    context = {
        "mode": "image",
        "run_index": 0,
        "total_runs": 1,
        "ran_iterations": 1,
        "total_iterations": 1,
        "set_name": set_name,
        "prompt_index": 0,
        "total_prompts": 1,
    }
    prompt = request.resolved_prompt or request.prompt
    _emit_progress(progress_callback, "batch_started", mode="image", total_iterations=1, total_runs=1)
    _emit_progress(progress_callback, "prompt_started", **context, prompt=prompt, seed=request.seed)
    if skip.consume() == "quit":
        _emit_progress(progress_callback, "batch_cancelled", mode="image", completed_iterations=0, total_iterations=1)
        return StageOutcome.skipped

    event_context = {**context, "prompt": prompt, "seed": request.seed, "filename": set_name}
    _emit_progress(progress_callback, "generation_started", **event_context, retry=0)
    request = replace(
        request,
        backend=backend,
        model=model,
        skip_signal=skip,
        step_callback=_make_step_progress_callback(progress_callback, **context),
    )
    artifacts = ImageWorkingArtifacts()
    started = time.time()
    outcome = _run_workflow_with_progress(workflow, request, artifacts, progress_callback=progress_callback, event_context=event_context)
    status = {StageOutcome.success: "success", StageOutcome.skipped: "skipped"}.get(outcome, "failed")
    _emit_generation_finished(
        progress_callback,
        event_context=event_context,
        status=status,
        filename=Path(artifacts.filepath).name if artifacts.filepath else None,
        generation_time=time.time() - started,
        output_path=artifacts.filepath if status == "success" else None,
    )
    if status == "success":
        _emit_progress(progress_callback, "batch_completed", mode="image", completed_iterations=1, total_iterations=1)
    elif status == "skipped":
        _emit_progress(progress_callback, "batch_cancelled", mode="image", completed_iterations=0, total_iterations=1)
    else:
        _emit_progress(progress_callback, "batch_failed", mode="image", completed_iterations=1, total_iterations=1)
    return outcome
