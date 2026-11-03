"""Audio batch orchestration: plan, download, load once, make each take, save it as FLAC, close.

One job is one model: the backend loads once, makes every take of every run, and is closed in a ``finally`` also after a
failure or a Stop. Pause, Repeat and Next work between takes (Next and Quit also stop a take in progress).
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import audio_device, audio_platform_key, get_audio_backend, prompt_enhancer_session
from zvisiongenerator.core.audio_backend import AudioBackend, AudioCancelled
from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest
from zvisiongenerator.core.progress_events import ProgressCallback, emit_generation_finished, emit_progress
from zvisiongenerator.downloads import DETACHED, STOPPED, DownloadFlights, DownloadReport, audio_plan, ensure_downloaded
from zvisiongenerator.job_controls import announce_repeat, pause_until_resumed, seed_range
from zvisiongenerator.preflight import plan_iterations
from zvisiongenerator.utils.audio_level import limit_peak, peak_of
from zvisiongenerator.utils.audio_preset import AudioPreset
from zvisiongenerator.utils.audio_request import check_speech_request
from zvisiongenerator.utils.audio_save import save_flac
from zvisiongenerator.utils.filename import generate_filename, unique_output_path
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.provenance import build_audio_config_payload
from zvisiongenerator.utils.speech_text import split_speech_text

__all__ = ["OUT_OF_MEMORY_HINT", "run_audio_batch"]

OUT_OF_MEMORY_HINT = "Close other apps that use the GPU and try again"
"""What the out-of-memory message tells the user to do; music adds ``or shorten the lyrics``."""

_MPS_OUT_OF_MEMORY = "MPS backend out of memory"
_LEVEL_CEILING_DB = -1.0


@dataclass
class _Outcome:
    """What the take loop did: the files it saved and whether the user stopped the batch."""

    saved: list[Path] = field(default_factory=list)
    cancelled: bool = False
    skipped: int = 0
    attempts: int = 0
    seconds: float = 0.0


def run_audio_batch(
    request: AudioGenerationRequest,
    preset: AudioPreset,
    config: dict[str, Any],
    *,
    runs: int,
    seed: int | None,
    output_dir: str,
    progress_callback: ProgressCallback | None = None,
    skip_signal: SkipSignal | None = None,
    flights: DownloadFlights | None = None,
    on_download: Callable[[DownloadReport], None] | None = None,
) -> list[Path]:
    """Make *runs* takes of *request* with one load of the model and save each as a FLAC file with its settings.

    Steps: plan the takes (no prompt enhancement), check a speech request against the limits, release a resident prompt
    enhancer, make sure the model's files are downloaded, load the backend once, make each
    take, close the backend. A take whose peak is above 1.0 and that carries a waveform from before the library's
    clamp (``AudioClip.preclamp``) is scaled to -1 dBFS before saving; other takes are saved as they are.

    Events sent to *progress_callback* (all with ``mode="audio"``): ``batch_started``, ``prompt_started``,
    ``generation_started``, ``audio_progress`` (``phase``, ``step``, ``total``, ``part``, ``n_parts``),
    ``generation_finished`` (status ``success`` or ``skipped``), and ``batch_completed`` or ``batch_cancelled``.
    A take that fails raises; the web job then reports the failure.

    Args:
        request: The take to make; its ``seed`` is replaced per take.
        preset: The audio model of *request*.
        config: The loaded config (seed bounds).
        runs: How many takes to make.
        seed: The seed of every take, or ``None`` to draw one per take.
        output_dir: The folder the FLAC files are written to; created when missing.
        progress_callback: Receives the events above.
        skip_signal: The job's control signal (Pause, Repeat, Next, Quit).
        flights: Registry that shares a running download with other callers; a private one by default.
        on_download: Receives the download reports; by default the first report of each repository is printed.

    Returns:
        The saved files, in the order they were made.

    Raises:
        ValueError: For a request that does not fit the preset or its limits (text too long, voice outside its length, a
            model that does not run on this platform).
        FileNotFoundError: If a file the backend loads is missing from the cache.
        RuntimeError: When the GPU runs out of memory (the message names the model and what to do), or a backend fails.
    """
    if request.preset != preset.id or request.kind != preset.kind:
        raise ValueError(f"The request is for {request.kind} model '{request.preset}', not {preset.kind} model '{preset.id}'.")
    if runs < 1:
        raise ValueError("runs must be at least 1")
    platform_key = audio_platform_key()
    targets = audio_plan(preset, platform_key)  # raises when the model does not run on this platform
    platform = preset.platforms[platform_key]
    if preset.kind == "speech":
        check_speech_request(request, preset.limits)

    seed_min, seed_max = seed_range(config)
    iterations = plan_iterations(
        {request.kind: [(request.prompt, None)]},
        runs=runs,
        seed=seed,
        seed_min=seed_min,
        seed_max=seed_max,
        json_prompt=True,  # the text is kept as written: no {a|b} choices in lyrics or speech
        disabled=True,
        override=None,
        enhance_by_set=None,
    )
    total = len(iterations)
    skip = skip_signal or SkipSignal()
    emit_progress(progress_callback, "batch_started", mode="audio", total_iterations=total, total_runs=runs)
    started = time.time()

    prompt_enhancer_session.release_resident_enhancer()
    if not _prepare_model(targets, skip, flights or DownloadFlights(), on_download, progress_callback, total):
        return []

    os.makedirs(output_dir, exist_ok=True)
    backend = get_audio_backend(platform.family)
    outcome = _Outcome()
    try:
        print(f"Loading {preset.label}...", flush=True)
        _guarded(lambda: backend.load(preset, platform, audio_device()), preset)
        outcome = _make_takes(backend, request, preset, iterations, output_dir=output_dir, skip=skip, progress_callback=progress_callback, seed_range_=(seed_min, seed_max), runs=runs)
    finally:
        backend.close()

    if outcome.cancelled:
        print("\n⏹ Quitting batch...")
        emit_progress(progress_callback, "batch_cancelled", mode="audio", completed_iterations=len(outcome.saved) + outcome.skipped, total_iterations=total)
        return outcome.saved
    total_time = time.time() - started
    notes = f", {outcome.skipped} skipped" if outcome.skipped else ""
    print(f"\nBatch complete: {len(outcome.saved)}/{outcome.attempts} takes made{notes} in {total_time:.1f}s")
    emit_progress(progress_callback, "batch_completed", mode="audio", completed_iterations=outcome.attempts, total_iterations=outcome.attempts, total_time=total_time)
    return outcome.saved


def _prepare_model(
    targets: Sequence[Any],
    skip: SkipSignal,
    flights: DownloadFlights,
    on_download: Callable[[DownloadReport], None] | None,
    progress_callback: ProgressCallback | None,
    total: int,
) -> bool:
    """Download the model's files; return ``False`` when the user stopped the job meanwhile."""

    def _stopped() -> bool:
        return skip.pending() == "quit"

    outcome = ensure_downloaded(targets, flights=flights, report=on_download or _PrintFirstReport(), cancelled=_stopped)
    if outcome in (STOPPED, DETACHED):
        emit_progress(progress_callback, "batch_cancelled", mode="audio", completed_iterations=0, total_iterations=total, download=outcome)
        return False
    return True


class _PrintFirstReport:
    """Print one line for each repository the first time a download report names it."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def __call__(self, report: DownloadReport) -> None:
        if report.repo in self._seen:
            return
        self._seen.add(report.repo)
        print(f"Downloading {report.repo} (part {report.part} of {report.parts}, first use only)...", flush=True)


def _make_takes(
    backend: AudioBackend,
    request: AudioGenerationRequest,
    preset: AudioPreset,
    iterations: Sequence[Any],
    *,
    output_dir: str,
    skip: SkipSignal,
    progress_callback: ProgressCallback | None,
    seed_range_: tuple[int, int],
    runs: int,
) -> _Outcome:
    """Make and save every planned take, honouring Pause, Repeat, Next and Quit; return what was done."""
    outcome = _Outcome()
    total = len(iterations)
    parts = _speech_parts(request, preset)
    gen_times: list[float] = []
    batch_start = time.time()
    for ran, planned in enumerate(iterations, start=1):
        seed = planned.seed
        avg = sum(gen_times) / len(gen_times) if gen_times else None
        eta = avg * (total - ran + 1) if avg is not None else None
        context = {
            "mode": "audio",
            "run_index": planned.run_index,
            "total_runs": runs,
            "ran_iterations": ran,
            "total_iterations": total,
            "set_name": planned.set_name,
            "prompt_index": planned.prompt_index,
            "total_prompts": 1,
            "prompt": planned.prompt,
        }
        emit_progress(progress_callback, "prompt_started", **context, seed=seed, enhance_status=planned.enhance_status.value, elapsed_secs=time.time() - batch_start, avg_secs=avg, eta_secs=eta)
        print(f"\n[{ran}/{total}] {preset.label}")
        while True:
            action = skip.consume()
            if action == "pause":
                action = pause_until_resumed(skip, progress_callback=progress_callback, mode="audio", completed_iterations=len(outcome.saved) + outcome.skipped, total_iterations=total)
            if action == "quit":
                outcome.cancelled = True
                return outcome
            outcome.attempts += 1
            take = replace(request, seed=seed)
            emit_progress(progress_callback, "generation_started", **context, seed=seed, elapsed_secs=time.time() - batch_start, avg_secs=avg, eta_secs=eta)
            print(f"  seed={seed}")
            began = time.monotonic()
            try:
                clip = backend.generate(take, _progress_reporter(progress_callback, context), skip.check)
            except AudioCancelled:
                if skip.consume() == "quit":
                    outcome.cancelled = True
                    return outcome
                print("  Take skipped.")
                outcome.skipped += 1
                emit_generation_finished(progress_callback, event_context={**context, "seed": seed}, status="skipped")
                break
            except Exception as exc:
                friendly = _out_of_memory_error(exc, preset)
                if friendly is None:
                    raise
                raise friendly from exc
            elapsed = time.monotonic() - began
            path = _save(clip, take, parts, elapsed, output_dir)
            outcome.saved.append(path)
            gen_times.append(elapsed)
            print(f"  Saved {path}")
            emit_generation_finished(progress_callback, event_context={**context, "seed": seed}, status="success", filename=path.name, generation_time=elapsed, output_path=str(path))

            action = skip.consume()
            if action == "pause":
                action = pause_until_resumed(skip, progress_callback=progress_callback, mode="audio", completed_iterations=len(outcome.saved) + outcome.skipped, total_iterations=total)
                if action != "quit":
                    break
            if action == "quit":
                outcome.cancelled = True
                return outcome
            if action != "repeat":
                break
            seed = announce_repeat(progress_callback, mode="audio", completed_iterations=len(outcome.saved) + outcome.skipped, total_iterations=total, seed_min=seed_range_[0], seed_max=seed_range_[1])
    return outcome


def _progress_reporter(progress_callback: ProgressCallback | None, context: dict[str, Any]) -> Callable[[str, int, int | None, int, int], None]:
    """Return the backend's ``on_progress``: it prints each new phase (and part) and sends an ``audio_progress`` event for every call."""
    last: list[tuple[str, int]] = []

    def _report(phase: str, step: int, total: int | None, part: int, n_parts: int) -> None:
        marker = (phase, part)
        if not last or last[0] != marker:
            last[:] = [marker]
            label = f"{phase} (part {part} of {n_parts})" if n_parts > 1 else phase
            print(f"  {label}...", flush=True)
        emit_progress(progress_callback, "audio_progress", **context, phase=phase, step=step, total=total, part=part, n_parts=n_parts)

    return _report


def _save(clip: AudioClip, request: AudioGenerationRequest, parts: int | None, generation_time: float, output_dir: str) -> Path:
    """Level *clip* when it needs it and write it as FLAC with the take's settings; return the file."""
    samples = clip.samples
    if clip.preclamp and peak_of(samples) > 1.0:
        samples = limit_peak(samples, _LEVEL_CEILING_DB)
        print(f"  Peak above full scale; scaled to {_LEVEL_CEILING_DB:g} dBFS.")
    payload = build_audio_config_payload(request, clip, parts=parts, generation_time=generation_time)
    path = unique_output_path(output_dir, generate_filename(request.kind), ".flac")
    return save_flac(samples, clip.sample_rate, path, payload)


def _speech_parts(request: AudioGenerationRequest, preset: AudioPreset) -> int | None:
    """Return how many parts the speech text is read in, or ``None`` for other kinds."""
    if preset.kind != "speech":
        return None
    return len(split_speech_text(request.prompt, int(preset.limits.part_chars), max_text_chars=int(preset.limits.max_text_chars)))


def _guarded(action: Callable[[], None], preset: AudioPreset) -> None:
    """Run *action*, turning an out-of-memory error into the model's message."""
    try:
        action()
    except Exception as exc:
        friendly = _out_of_memory_error(exc, preset)
        if friendly is None:
            raise
        raise friendly from exc


def _out_of_memory_error(exc: BaseException, preset: AudioPreset) -> RuntimeError | None:
    """Return the user-facing error for a GPU out-of-memory *exc* (CUDA's, or the MPS ``RuntimeError``), else ``None``."""
    if not _is_out_of_memory(exc):
        return None
    advice = f"{OUT_OF_MEMORY_HINT} or shorten the lyrics" if preset.kind == "music" else OUT_OF_MEMORY_HINT
    return RuntimeError(f"{preset.label} ran out of GPU memory. {advice}.")


def _is_out_of_memory(exc: BaseException) -> bool:
    """Return whether *exc* is ``torch.OutOfMemoryError`` or a ``RuntimeError`` saying the MPS backend is out of memory."""
    if isinstance(exc, RuntimeError) and _MPS_OUT_OF_MEMORY in str(exc):
        return True
    if "torch" not in sys.modules:
        return False  # nothing torch-based ran, so nothing can have raised its error
    import torch

    return isinstance(exc, torch.OutOfMemoryError)
