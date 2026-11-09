"""Store the candidates of every voice design run in ``<data dir>/voice-design/`` until the user clears them.

The folder holds ``<id>.wav`` per candidate (16-bit mono PCM, no tags) and ``candidates.yaml``, whose ``batches`` list (oldest first)
has one entry per run: ``batch_id``, ``model``, ``description``, ``text``, ``created_at`` and its ``candidates`` (each with ``id``,
``seconds`` and, once it was saved as a voice, ``saved_as`` (the voice's slug) with ``saved_at`` (when)). A candidate's id is a random hex string that never changes, so a page
that names a candidate gets that candidate or nothing, whatever was added or cleared since. Runs add batches and never replace earlier
ones; batches stay across restarts until :func:`clear_candidates`. Candidates are never written to the output folder, so they never
become takes.

Every function takes the folder, so tests and the runner choose where the candidates live; :func:`candidates_dir` is the default.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from zvisiongenerator.utils.atomic_write import write_text_atomic
from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.utils.video_splice import write_pcm16_wav_atomic

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "DesignBatch",
    "DesignCandidate",
    "add_batch",
    "candidate_file",
    "candidates_dir",
    "clear_candidates",
    "find_candidate",
    "mark_saved",
    "read_batches",
    "write_candidate",
]

_RECORD_FILE = "candidates.yaml"
_record_lock = threading.Lock()


@dataclass(frozen=True)
class DesignCandidate:
    """One designed voice.

    Attributes:
        id: A random hex string, new for every candidate and stable for its life; its file is ``<id>.wav``.
        seconds: Length of the recording in seconds.
        saved_as: The slug of the voice it was saved as, or ``None`` while it is unpicked. The voice may have been deleted since.
        saved_at: When it was saved, as an ISO 8601 UTC timestamp; ``None`` exactly when ``saved_as`` is.
    """

    id: str
    seconds: float
    saved_as: str | None = None
    saved_at: str | None = None


@dataclass(frozen=True)
class DesignBatch:
    """The candidates of one voice design run.

    Attributes:
        batch_id: A new random hex string for every run.
        model: The id of the audio model that designed the voices.
        description: The voice in words that every candidate was made from.
        text: The line every candidate speaks; it becomes the transcript of a saved voice.
        created_at: When the run ended, as an ISO 8601 UTC timestamp.
        candidates: The candidates made, in order.
    """

    batch_id: str
    model: str
    description: str
    text: str
    created_at: str
    candidates: tuple[DesignCandidate, ...]


def candidates_dir() -> Path:
    """Return the folder of the candidates, ``<data dir>/voice-design``; it exists once a run has written a candidate."""
    return get_ziv_data_dir() / "voice-design"


def write_candidate(directory: Path, samples: np.ndarray, sample_rate: int) -> DesignCandidate:
    """Write *samples* as ``<id>.wav`` under a new id in *directory* (made when missing) and return the candidate.

    The candidate is not listed until its batch is added with :func:`add_batch`.

    Args:
        directory: The folder of the candidates.
        samples: Mono float32 of shape ``(samples,)`` with values in -1..1.
        sample_rate: The wave's sample rate in Hz; it is stored as it is.
    """
    directory.mkdir(parents=True, exist_ok=True)
    candidate_id = uuid.uuid4().hex
    write_pcm16_wav_atomic(samples.reshape(1, -1), directory / f"{candidate_id}.wav", sample_rate)
    return DesignCandidate(candidate_id, round(samples.shape[0] / sample_rate, 3))


def add_batch(directory: Path, batch: DesignBatch) -> None:
    """Append *batch* after the earlier ones in *directory*, replacing the record atomically; a batch without candidates is not added."""
    if not batch.candidates:
        return
    with _record_lock:
        _write_batches(directory, (*read_batches(directory), batch))


def read_batches(directory: Path) -> tuple[DesignBatch, ...]:
    """Return the batches in *directory*, oldest first; none when the record is missing or unreadable."""
    try:
        data = yaml.safe_load((directory / _RECORD_FILE).read_text(encoding="utf-8"))
        return tuple(_read_batch(item) for item in data["batches"])
    except OSError, ValueError, yaml.YAMLError, KeyError, TypeError:
        return ()


def find_candidate(directory: Path, candidate_id: str) -> tuple[DesignBatch, DesignCandidate]:
    """Return the candidate *candidate_id* and the batch it belongs to.

    Raises:
        FileNotFoundError: When there is no such candidate (it was never made, or the candidates were cleared).
    """
    for batch in read_batches(directory):
        for candidate in batch.candidates:
            if candidate.id == candidate_id:
                return batch, candidate
    raise FileNotFoundError("This voice is gone. Make new voices.")


def mark_saved(directory: Path, candidate_id: str, slug: str, saved_at: str) -> None:
    """Record that candidate *candidate_id* was saved as the voice *slug* at *saved_at* (an ISO 8601 UTC timestamp); do nothing when it is gone."""
    with _record_lock:
        batches = read_batches(directory)
        marked = tuple(replace(batch, candidates=tuple(replace(item, saved_as=slug, saved_at=saved_at) if item.id == candidate_id else item for item in batch.candidates)) for batch in batches)
        if marked != batches:
            _write_batches(directory, marked)


def clear_candidates(directory: Path) -> None:
    """Remove every candidate WAV and the record from *directory*; do nothing when it does not exist."""
    if not directory.is_dir():
        return
    with _record_lock:
        for path in [*directory.glob("*.wav"), directory / _RECORD_FILE]:
            path.unlink(missing_ok=True)


def candidate_file(directory: Path, candidate_id: str) -> Path:
    """Return the WAV of the listed candidate *candidate_id*.

    Raises:
        FileNotFoundError: When there is no such candidate or its file is missing.
    """
    find_candidate(directory, candidate_id)
    path = directory / f"{candidate_id}.wav"
    if not path.is_file():
        raise FileNotFoundError("This voice is gone. Make new voices.")
    return path


def _write_batches(directory: Path, batches: tuple[DesignBatch, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    document = {"batches": [_batch_document(batch) for batch in batches]}
    write_text_atomic(directory / _RECORD_FILE, yaml.safe_dump(document, sort_keys=False, allow_unicode=True))


def _batch_document(batch: DesignBatch) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "model": batch.model,
        "description": batch.description,
        "text": batch.text,
        "created_at": batch.created_at,
        "candidates": [_candidate_document(item) for item in batch.candidates],
    }


def _candidate_document(candidate: DesignCandidate) -> dict[str, Any]:
    document: dict[str, Any] = {"id": candidate.id, "seconds": candidate.seconds}
    if candidate.saved_as is not None:
        document["saved_as"] = candidate.saved_as
        document["saved_at"] = candidate.saved_at
    return document


def _read_batch(item: dict[str, Any]) -> DesignBatch:
    candidates = tuple(DesignCandidate(entry["id"], float(entry["seconds"]), entry.get("saved_as"), entry.get("saved_at")) for entry in item["candidates"])
    return DesignBatch(item["batch_id"], item["model"], item["description"], item["text"], item["created_at"], candidates)
