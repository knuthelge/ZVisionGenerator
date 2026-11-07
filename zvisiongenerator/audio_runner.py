"""Audio batch orchestration: plan the takes, load once, make each take, save it as FLAC, close.

One job is one model: the backend loads once, makes every take of every run, and is closed in a ``finally`` also after a
failure or a Stop. Pause, Repeat and Next work between takes (Next and Quit also stop a take in progress). The model's
files are downloaded before the batch starts, by the caller: the Web UI's job runner, or ``ziv audio``.
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import accelerator_memory_budget_or_none, audio_platform_key, get_audio_backend, prompt_enhancer_session, release_accelerator_memory
from zvisiongenerator.core.audio_backend import AudioBackend, AudioCancelled
from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest
from zvisiongenerator.core.progress_events import ProgressCallback, emit_generation_finished, emit_progress
from zvisiongenerator.downloads import audio_plan, target_present_offline
from zvisiongenerator.job_controls import announce_repeat, pause_until_resumed, seed_range
from zvisiongenerator.utils.audio_console import ConsoleProgress
from zvisiongenerator.utils.audio_level import limit_peak, peak_of
from zvisiongenerator.utils.audio_memory import audio_memory_fit, memory_need_text
from zvisiongenerator.utils.audio_preset import AudioPreset
from zvisiongenerator.utils.audio_save import save_flac
from zvisiongenerator.utils.filename import generate_filename, unique_output_path
from zvisiongenerator.utils.interactive import SkipSignal
from zvisiongenerator.utils.provenance import build_audio_config_payload
from zvisiongenerator.utils.speech_text import request_speech_parts
from zvisiongenerator.utils.torch_flags import preserved_torch_flags

__all__ = ["QUIT_MESSAGE", "run_audio_batch"]

_OUT_OF_MEMORY_HINT = "Close other apps that use the GPU and try again"
"""What the out-of-memory message tells the user to do; music adds ``or shorten the lyrics``."""

QUIT_MESSAGE = "⏹ Quitting batch..."
"""What the console says when the user quits."""

_CUT_NOTE = "Reached the model's length limit; the take was cut short."
"""What the CLI says after saving a take that stopped at the model's length limit."""

_MPS_OUT_OF_MEMORY = "MPS backend out of memory"
_LEVEL_CEILING_DB = -1.0


@dataclass
class _Outcome:
    """What the take loop did: the files it saved and whether the user stopped the batch."""

    saved: list[Path] = field(default_factory=list)
    cancelled: bool = False
    skipped: int = 0
    attempts: int = 0


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
) -> list[Path]:
    """Make *runs* takes of *request* with one load of the model and save each as a FLAC file with its settings.

    The caller has checked the request against the preset's limits (``check_speech_request``) and *runs*, and has downloaded
    the model's files.

    Steps: release a resident prompt enhancer, check the model's files are on disk, load the backend once, make each take,
    close the backend, give back the torch flags the model code changed and the accelerator memory it held. A take whose peak
    is above 1.0 is scaled to -1 dBFS before saving; other takes are saved as they are.

    Events sent to *progress_callback* (all with ``mode="audio"``): ``model_loading`` (before the load), ``batch_started``
    (once the model is loaded), ``prompt_started``, ``generation_started``, ``audio_progress`` (``phase``, ``step``,
    ``total``, ``part``, ``n_parts``, ``elapsed_secs``, ``eta_secs``), ``generation_finished`` (status ``success`` or
    ``skipped``), and ``batch_completed`` or ``batch_cancelled``. A take that fails raises; the web job then reports the failure.

    Args:
        request: The take to make; its ``seed`` is replaced per take.
        preset: The audio model of *request*, as this machine runs it (``AudioPreset.on_platform``).
        config: The loaded config (seed bounds).
        runs: How many takes to make.
        seed: The seed of every take, or ``None`` to draw one per take.
        output_dir: The folder the FLAC files are written to; created when missing.
        progress_callback: Receives the events above.
        skip_signal: The job's control signal (Pause, Repeat, Next, Quit).

    Returns:
        The saved files, in the order they were made.

    Raises:
        FileNotFoundError: If a file of the model is missing from the cache.
        RuntimeError: When the GPU runs out of memory (the message names the model, what it needs against what the card has,
            and what to do), or a backend fails.
    """
    platform_key = audio_platform_key()
    seed_min, seed_max = seed_range(config)
    seeds = [seed if seed is not None else random.randint(seed_min, seed_max) for _ in range(runs)]
    skip = skip_signal or SkipSignal()
    started = time.time()

    prompt_enhancer_session.release_resident_enhancer()
    _require_downloaded(preset, voice=request.voice is not None)

    os.makedirs(output_dir, exist_ok=True)
    backend = get_audio_backend(preset.family)
    fit = audio_memory_fit(preset, platform_key, accelerator_memory_budget_or_none())
    with _backend_session(backend):
        if _load_unless_quit(backend, preset, request, platform_key, fit, skip=skip, progress_callback=progress_callback):
            emit_progress(progress_callback, "batch_started", mode="audio", total_iterations=runs, total_runs=runs)
            outcome = _make_takes(backend, request, preset, seeds, output_dir=output_dir, skip=skip, progress_callback=progress_callback, seed_range_=(seed_min, seed_max), memory_fit=fit)
        else:
            outcome = _Outcome(cancelled=True)

    if outcome.cancelled:
        print(f"\n{QUIT_MESSAGE}")
        emit_progress(progress_callback, "batch_cancelled", mode="audio", completed_iterations=len(outcome.saved) + outcome.skipped, total_iterations=runs)
        return outcome.saved
    total_time = time.time() - started
    notes = f", {outcome.skipped} skipped" if outcome.skipped else ""
    print(f"\nBatch complete: {len(outcome.saved)}/{outcome.attempts} takes made{notes} in {total_time:.1f}s")
    emit_progress(progress_callback, "batch_completed", mode="audio", completed_iterations=outcome.attempts, total_iterations=outcome.attempts, total_time=total_time)
    return outcome.saved


def _load_unless_quit(
    backend: AudioBackend,
    preset: AudioPreset,
    request: AudioGenerationRequest,
    platform_key: str,
    fit: dict[str, Any] | None,
    *,
    skip: SkipSignal,
    progress_callback: ProgressCallback | None,
) -> bool:
    """Load the model unless Quit is queued, before the load or while it runs; return whether it is loaded.

    The backend checks Quit (and only Quit: Skip leaves the load alone) between the tensors it reads, so a stop during a long
    load ends it early; the caller's backend session then frees what was loaded.
    """
    if skip.pending() == "quit":
        return False
    print(f"Loading {preset.label}...", flush=True)
    emit_progress(progress_callback, "model_loading", mode="audio", model=preset.label)
    try:
        with _friendly_oom(preset, request.kind, fit):
            backend.load(_preset_to_load(preset, request), platform_key, cancelled=lambda: skip.pending() == "quit")
    except AudioCancelled:
        return False
    return True


def _preset_to_load(preset: AudioPreset, request: AudioGenerationRequest) -> AudioPreset:
    """Return the preset the backend loads: for a take that follows a recording, one whose ``files`` are the preset's ``voice_files``.

    The checkpoint is fixed for the whole batch (every take is the request with its own seed), so the backend loads
    ``files[0]`` once and never chooses. A preset without ``voice_files`` follows a recording with its own files and loads as it is.
    """
    if request.voice is not None and preset.voice_files:
        return replace(preset, files=preset.voice_files)
    return preset


def _require_downloaded(preset: AudioPreset, *, voice: bool = False) -> None:
    """Refuse a model whose files are not on disk, before anything loads: the backend only reads the cache.

    A take that follows a voice recording (*voice*) needs the model's cloning files as well.

    Raises:
        FileNotFoundError: When a file is missing. A caller whose download step could not reach Hugging Face says so first
            (see :func:`~zvisiongenerator.downloads.ensure.require_checked`).
    """
    if not all(target_present_offline(target) for target in audio_plan(preset, voice=voice)):
        raise FileNotFoundError(f"{preset.label} is not downloaded: some of its files are missing from the cache.")


@contextmanager
def _backend_session(backend: AudioBackend) -> Iterator[AudioBackend]:
    """Own the backend's lifecycle for the length of the block: whatever happens, close it, then give back what model code takes.

    Model code switches global torch numerics flags when it builds a model, and the cached accelerator memory outlives it,
    so both are put back here once for every backend rather than by each.
    """
    with preserved_torch_flags():
        try:
            yield backend
        finally:
            try:
                backend.close()
            finally:
                release_accelerator_memory()


def _make_takes(
    backend: AudioBackend,
    request: AudioGenerationRequest,
    preset: AudioPreset,
    seeds: Sequence[int],
    *,
    output_dir: str,
    skip: SkipSignal,
    progress_callback: ProgressCallback | None,
    seed_range_: tuple[int, int],
    memory_fit: dict[str, Any] | None = None,
) -> _Outcome:
    """Make and save one take per seed in *seeds*, honouring Pause, Repeat, Next and Quit; return what was done."""
    outcome = _Outcome()
    total = len(seeds)
    parts = _speech_parts(request, preset)
    gen_times: list[float] = []
    batch_start = time.time()
    for ran, seed in enumerate(seeds, start=1):
        avg = sum(gen_times) / len(gen_times) if gen_times else None
        eta = avg * (total - ran + 1) if avg is not None else None
        context = {
            "mode": "audio",
            "run_index": ran - 1,
            "total_runs": total,
            "ran_iterations": ran,
            "total_iterations": total,
            "prompt": request.prompt,
        }
        emit_progress(progress_callback, "prompt_started", **context, seed=seed, elapsed_secs=time.time() - batch_start, avg_secs=avg, eta_secs=eta)
        if ran > 1:
            print()  # the first take's header follows the "Loading" line directly
        print(f"[{ran}/{total}] {preset.label}")
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
            console = ConsoleProgress()
            try:
                with _friendly_oom(preset, request.kind, memory_fit):
                    clip = backend.generate(
                        take, _progress_reporter(progress_callback, context, console, batch_start=batch_start, began=began, gen_times=gen_times, takes_left=total - ran + 1), skip.check
                    )
            except AudioCancelled:
                if skip.consume() == "quit":
                    outcome.cancelled = True
                    return outcome
                print("  Take skipped.")
                outcome.skipped += 1
                emit_generation_finished(progress_callback, event_context={**context, "seed": seed}, status="skipped")
                break
            finally:
                console.close()
            elapsed = time.monotonic() - began
            path = _save(clip, take, parts, elapsed, output_dir)
            outcome.saved.append(path)
            gen_times.append(elapsed)
            print(f"  Saved {path}")
            if clip.cut:
                print(f"  {_CUT_NOTE}")
            for note in clip.notes:
                print(f"  {note}")
            emit_generation_finished(progress_callback, event_context={**context, "seed": seed}, status="success", filename=path.name, generation_time=elapsed, output_path=str(path))

            action = skip.consume()
            if action == "pause":
                action = pause_until_resumed(skip, progress_callback=progress_callback, mode="audio", completed_iterations=len(outcome.saved) + outcome.skipped, total_iterations=total)
            if action == "quit":
                outcome.cancelled = True
                return outcome
            if action != "repeat":
                break
            seed = announce_repeat(progress_callback, mode="audio", completed_iterations=len(outcome.saved) + outcome.skipped, total_iterations=total, seed_min=seed_range_[0], seed_max=seed_range_[1])
    return outcome


def _progress_reporter(
    progress_callback: ProgressCallback | None,
    context: dict[str, Any],
    console: ConsoleProgress,
    *,
    batch_start: float,
    began: float,
    gen_times: Sequence[float],
    takes_left: int,
) -> Callable[[str, int, int | None, int, int], None]:
    """Return the backend's ``on_progress``: it shows the progress on the console and sends an ``audio_progress`` event for every call.

    Each event carries ``elapsed_secs`` (since the first take began) and ``eta_secs``: the average time of the takes made so far
    times the *takes_left* (this one included), less the time this take has already run (since *began*, on the monotonic clock),
    never below zero; ``None`` before a take has finished.
    """

    def _report(phase: str, step: int, total: int | None, part: int, n_parts: int) -> None:
        eta = max(0.0, sum(gen_times) / len(gen_times) * takes_left - (time.monotonic() - began)) if gen_times else None
        console(phase, step, total, part, n_parts, eta)
        emit_progress(progress_callback, "audio_progress", **context, phase=phase, step=step, total=total, part=part, n_parts=n_parts, elapsed_secs=time.time() - batch_start, eta_secs=eta)

    return _report


def _save(clip: AudioClip, request: AudioGenerationRequest, parts: int | None, generation_time: float, output_dir: str) -> Path:
    """Level *clip* when it needs it and write it as FLAC with the take's settings; return the file."""
    samples = clip.samples
    if peak_of(samples) > 1.0:
        samples = limit_peak(samples, _LEVEL_CEILING_DB)
        print(f"  Peak above full scale; scaled to {_LEVEL_CEILING_DB:g} dBFS.")
    payload = build_audio_config_payload(request, clip, parts=parts, generation_time=generation_time)
    path = unique_output_path(output_dir, generate_filename(request.kind), ".flac")
    return save_flac(samples, clip.sample_rate, path, payload)


def _speech_parts(request: AudioGenerationRequest, preset: AudioPreset) -> int | None:
    """Return how many parts the speech text is read in, or ``None`` for other kinds."""
    if request.kind != "speech":
        return None
    return len(request_speech_parts(request, part_chars=int(preset.limits.part_chars), max_text_chars=int(preset.limits.max_text_chars)))


@contextmanager
def _friendly_oom(preset: AudioPreset, kind: str, fit: dict[str, Any] | None) -> Iterator[None]:
    """Turn an out-of-memory error raised inside the block into the model's message."""
    try:
        yield
    except AudioCancelled:
        raise
    except Exception as exc:
        friendly = _out_of_memory_error(exc, preset, kind, fit)
        if friendly is None:
            raise
        raise friendly from exc


def _out_of_memory_error(exc: BaseException, preset: AudioPreset, kind: str, fit: dict[str, Any] | None) -> RuntimeError | None:
    """Return the user-facing error for a GPU out-of-memory *exc* (CUDA's, or the MPS ``RuntimeError``), else ``None``.

    With a memory *fit* the message says what the model needs against what the GPU has.
    """
    if not _is_out_of_memory(exc):
        return None
    advice = f"{_OUT_OF_MEMORY_HINT} or shorten the lyrics" if preset.takes_lyrics(kind) else _OUT_OF_MEMORY_HINT
    sizes = f" ({memory_need_text(fit)})" if fit else ""
    return RuntimeError(f"{preset.label} ran out of GPU memory{sizes}. {advice}.")


def _is_out_of_memory(exc: BaseException) -> bool:
    """Return whether *exc* is ``torch.OutOfMemoryError`` or a ``RuntimeError`` saying the MPS backend is out of memory."""
    if isinstance(exc, RuntimeError) and _MPS_OUT_OF_MEMORY in str(exc):
        return True
    if "torch" not in sys.modules:
        return False  # nothing torch-based ran, so nothing can have raised its error
    import torch

    return isinstance(exc, torch.OutOfMemoryError)
