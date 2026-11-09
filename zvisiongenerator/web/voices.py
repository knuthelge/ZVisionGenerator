"""The logic behind the saved-voices routes: what a voice looks like on the wire, saving a recording as a voice, and the design job's request, run and candidates."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zvisiongenerator.core.audio_types import VoiceDesignRequest
from zvisiongenerator.utils.audio_preset import AudioPreset
from zvisiongenerator.utils.audio_voice import consent_for_saving
from zvisiongenerator.utils.voice_candidates import DesignBatch, candidate_file, candidates_dir, find_candidate, mark_saved, read_batches
from zvisiongenerator.utils.voice_library import SavedVoice, read_recording, save_voice, save_voice_file, voice_names, voices_dir
from zvisiongenerator.web.audio_models import find_audio_model
from zvisiongenerator.web.path_browser import AUDIO_TRACK_EXTENSIONS

__all__ = ["design_payload", "parse_design_body", "save_candidate_voice", "save_recording_voice", "staged_upload", "voice_payload"]

_UPLOAD_LABEL = "Voice recording"
_MAX_DESCRIPTION_CHARS = 500


def voice_payload(voice: SavedVoice) -> dict[str, Any]:
    """Return *voice* as the routes send it: ``slug``, ``name``, ``text``, ``origin``, ``seconds``, ``description`` and ``created_at``.

    The consent and the file path stay on the host; the recording is played through ``GET /api/voices/{slug}/audio``.
    """
    return {
        "slug": voice.slug,
        "name": voice.name,
        "text": voice.text,
        "origin": voice.origin,
        "seconds": voice.seconds,
        "description": voice.description,
        "created_at": voice.created_at,
    }


def save_recording_voice(preset: AudioPreset, *, name: str, path: str | Path, text: str, ticked: bool) -> SavedVoice:
    """Save the recording at *path* as a voice named *name*, as a person's recording (``origin="recording"``).

    The checks run cheapest first: the model, the transcript, the speaker's consent, then the recording's length (probed before
    anything is decoded).

    Args:
        preset: The speech model the voice is for; its voice-length limits apply.
        name: The voice's name.
        path: The recording.
        text: What the recording says.
        ticked: Whether the speaker's consent was confirmed.

    Raises:
        ValueError: When the model cannot follow a voice, the text is blank, the recording needs consent and has none, cannot be
            read or is outside the model's limits, or the name is invalid.
        FileExistsError: When a voice already has that name.
    """
    if not preset.follows_voice("speech"):
        raise ValueError(f"{preset.label} cannot follow a voice.")
    if not text.strip():
        raise ValueError("Write what the recording says first.")
    consent = consent_for_saving(path, ticked)
    low, high = float(preset.limits.min_voice_seconds), float(preset.limits.max_voice_seconds)
    samples, sample_rate = read_recording(path, min_seconds=low, max_seconds=high)
    return save_voice(name, samples, sample_rate, text=text, consent=consent)


def parse_design_body(body: Any, config: dict[str, Any]) -> tuple[VoiceDesignRequest, AudioPreset]:
    """Return the design request a JSON body ``{model, description, text}`` asks for and the model that runs it.

    The description is at most 500 characters and the text at most one part of the model's speech; both are trimmed.

    Raises:
        ValueError: When the body is not an object, the model is unknown or cannot design voices, or the description or text is
            blank or too long.
    """
    if not isinstance(body, dict):
        raise ValueError("The request must be a JSON object.")
    preset = find_audio_model(config, body.get("model") if isinstance(body.get("model"), str) else None)
    if not preset.design_files:
        raise ValueError(f"{preset.label} cannot design voices.")
    description = _body_text(body, "description")
    text = _body_text(body, "text")
    if not description:
        raise ValueError("Describe the voice first.")
    if len(description) > _MAX_DESCRIPTION_CHARS:
        raise ValueError(f"The description can be at most {_MAX_DESCRIPTION_CHARS} characters.")
    if not text:
        raise ValueError("Write the line the voices will say first.")
    if preset.limits.part_chars is not None and len(text) > preset.limits.part_chars:
        raise ValueError(f"The line can be at most {preset.limits.part_chars} characters.")
    return VoiceDesignRequest(description=description, text=text), preset


def design_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return the voice design candidates as the routes send them: ``batches`` (newest first) and ``unpicked``.

    Each batch has its ``batch_id``, ``description``, ``text`` (the preview line), ``created_at`` and its ``candidates`` in the
    order they were made. Each candidate has its ``id``, ``number`` (counted from 1 in the order made, across batches), ``seconds``,
    whether it is ``saveable`` as a voice of its batch's model, the ``reason`` when it is not (``None`` otherwise) and ``saved_as``,
    the current name of the voice it was saved as (``None`` while unpicked, and again once that voice is deleted). ``unpicked`` counts
    the saveable, unsaved candidates of the batches made since the last save (of any candidate) or Clear voices. A batch of a model that
    is not available lists its candidates as not saveable.
    """
    batches = read_batches(candidates_dir())
    names = voice_names()
    last_save = max((item.saved_at for batch in batches for item in batch.candidates if item.saved_at is not None), default="")
    number = 0
    unpicked = 0
    payloads = []
    for batch in batches:
        preset = _batch_preset(config, batch)
        unavailable = _unavailable_message(batch)
        candidates = []
        for item in batch.candidates:
            number += 1
            reason = unavailable if preset is None else _unsaveable_reason(item.seconds, preset)
            saved_as = names.get(item.saved_as) if item.saved_as is not None else None
            if reason is None and saved_as is None and batch.created_at > last_save:
                unpicked += 1
            candidates.append({"id": item.id, "number": number, "seconds": item.seconds, "saveable": reason is None, "reason": reason, "saved_as": saved_as})
        payloads.append({"batch_id": batch.batch_id, "description": batch.description, "text": batch.text, "created_at": batch.created_at, "candidates": candidates})
    return {"batches": payloads[::-1], "unpicked": unpicked}


def _batch_preset(config: dict[str, Any], batch: DesignBatch) -> AudioPreset | None:
    """Return the model that designed *batch*, as this machine runs it, or ``None`` when it is not available or cannot design voices."""
    try:
        preset = find_audio_model(config, batch.model)
    except ValueError:
        return None
    return preset if preset.design_files else None


def _unavailable_message(batch: DesignBatch) -> str:
    return f"{batch.model} is not available, so these voices cannot be saved."


def save_candidate_voice(candidate_id: str, name: str, config: dict[str, Any]) -> SavedVoice:
    """Save the candidate *candidate_id* as a designed voice named *name*; its transcript is the line of its batch.

    The candidate stays in the list, marked as saved as the voice; the others are untouched, and the ready marker clears (see :func:`design_payload`).

    Args:
        candidate_id: The candidate's id.
        name: The voice's name.
        config: The loaded config, to find the model that designed the candidate and its voice-length limits.

    Raises:
        FileNotFoundError: When there is no such candidate.
        ValueError: When the candidate is not saveable (its reason is the message), its model is not available or the name is invalid.
        FileExistsError: When the candidate was already saved as a voice that still exists or a voice already has that name.
    """
    batch, candidate = find_candidate(candidates_dir(), candidate_id)
    saved_name = voice_names().get(candidate.saved_as) if candidate.saved_as is not None else None
    if saved_name is not None:
        raise FileExistsError(f"This voice is already saved as {saved_name}.")
    preset = _batch_preset(config, batch)
    if preset is None:
        raise ValueError(_unavailable_message(batch))
    reason = _unsaveable_reason(candidate.seconds, preset)
    if reason is not None:
        raise ValueError(reason)
    voice = save_voice_file(name, candidate_file(candidates_dir(), candidate_id), candidate.seconds, text=batch.text, description=batch.description)
    mark_saved(candidates_dir(), candidate_id, voice.slug, datetime.now(UTC).isoformat(timespec="seconds"))
    return voice


def _body_text(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    return value.strip() if isinstance(value, str) else ""


def _unsaveable_reason(seconds: float, preset: AudioPreset) -> str | None:
    """Return why a candidate of *seconds* cannot be saved as a voice of *preset*, or ``None`` when it can."""
    if seconds < float(preset.limits.min_voice_seconds):
        return "Too short to save as a voice; make the line longer."
    if seconds > float(preset.limits.max_voice_seconds):
        return "Too long to save as a voice; make the line shorter."
    return None


@contextmanager
def staged_upload(uploaded_file: Any) -> Iterator[Path]:
    """Copy an uploaded recording to a temporary file in the voices folder, yield its path and remove it afterwards.

    The file keeps its extension so the decoder can recognise it; it is removed whether the block succeeds or fails.

    Args:
        uploaded_file: The uploaded file part (a ``filename`` and a readable ``file``).

    Raises:
        ValueError: When the file is not a WAV, MP3, M4A, FLAC or OGG file.
    """
    suffix = Path(Path(str(uploaded_file.filename)).name).suffix.lower()
    if suffix not in AUDIO_TRACK_EXTENSIONS:
        raise ValueError(f"{_UPLOAD_LABEL} must be a WAV, MP3, M4A, FLAC, or OGG file.")
    folder = voices_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f".upload-{uuid.uuid4().hex}{suffix}"
    try:
        uploaded_file.file.seek(0)
        with path.open("wb") as target:
            shutil.copyfileobj(uploaded_file.file, target)
        yield path
    finally:
        path.unlink(missing_ok=True)
