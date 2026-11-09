"""Store the user's saved voices: reference recordings with their transcripts, in ``<data dir>/voices/``.

A voice is two files with one slug: ``<slug>.wav`` (16-bit mono PCM, no tags) and ``<slug>.yaml`` with the keys ``name``,
``text``, ``origin``, ``consent``, ``seconds``, ``description`` and ``created_at``. The speaker's consent lives in the record
(``consent`` is ``given`` or ``not_needed``; a designed voice is always ``not_needed``), never in the WAV. A record that cannot be
read, or has no WAV, is invalid: it is not listed and cannot be found.

The slug is made once, when the voice is saved, from its name and a short random suffix; renaming changes only the name, so the files never move.
Missing voices raise ``FileNotFoundError`` and taken names ``FileExistsError``, so the routes can map them to 404 and 409.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from zvisiongenerator.core.audio_types import VoiceRef
from zvisiongenerator.core.video_types import AudioTrack
from zvisiongenerator.utils.atomic_write import write_bytes_atomic, write_text_atomic
from zvisiongenerator.utils.audio_request import check_voice_seconds
from zvisiongenerator.utils.audio_track import probe_track_seconds, read_track
from zvisiongenerator.utils.paths import get_ziv_data_dir
from zvisiongenerator.utils.slug import MAX_SLUG_LENGTH, is_slug, slugify
from zvisiongenerator.utils.video_extend import AUDIO_SAMPLE_RATE
from zvisiongenerator.utils.video_splice import write_pcm16_wav_atomic

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "SavedVoice",
    "delete_voice",
    "find_voice",
    "list_voices",
    "read_recording",
    "rename_voice",
    "save_voice",
    "save_voice_file",
    "voice_by_slug",
    "voice_names",
    "voice_slug_for_path",
    "voices_dir",
]

_MAX_NAME_LENGTH = 60
_FALLBACK_SLUG = "voice"
_SUFFIX_BYTES = 2

# What reading a record can raise: the file is missing or unreadable, is not YAML, or lacks a field.
_UNREADABLE = (OSError, ValueError, yaml.YAMLError, KeyError, TypeError)
_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True)
class SavedVoice:
    """A saved voice: its reference recording, transcript and the record of where it came from.

    Attributes:
        slug: The file stem, made once from the first name and never changed.
        name: The name shown to the user.
        text: What the recording says.
        origin: ``"recording"`` (a real person's recording) or ``"design"`` (made from a description).
        consent: ``"given"`` (the speaker's consent was confirmed) or ``"not_needed"``.
        seconds: Length of the recording in seconds.
        description: The description a designed voice was made from, if any.
        created_at: When the voice was saved, as an ISO 8601 UTC timestamp.
        path: The reference WAV.
    """

    slug: str
    name: str
    text: str
    origin: str
    consent: str
    seconds: float
    description: str | None
    created_at: str
    path: Path

    def voice_ref(self) -> VoiceRef:
        """Return the reference a speech request follows: the WAV with its transcript."""
        return VoiceRef(str(self.path), self.text)


def voices_dir() -> Path:
    """Return the folder of the saved voices; it exists once a voice has been saved."""
    return get_ziv_data_dir() / "voices"


def list_voices() -> list[SavedVoice]:
    """Return every valid saved voice sorted by name (case-insensitive), then slug; invalid records are skipped."""
    voices: list[SavedVoice] = []
    for path in sorted(voices_dir().glob("*.yaml")):
        try:
            voices.append(_read_voice(path))
        except _UNREADABLE:
            continue
    return sorted(voices, key=lambda voice: (voice.name.casefold(), voice.slug))


def find_voice(name_or_slug: str) -> SavedVoice | None:
    """Return the voice whose name matches *name_or_slug* ignoring case, else the one with exactly that slug, else ``None``."""
    voices = list_voices()
    wanted = name_or_slug.strip().casefold()
    return next((voice for voice in voices if voice.name.casefold() == wanted), None) or next((voice for voice in voices if voice.slug == name_or_slug), None)


def voice_by_slug(slug: str) -> SavedVoice:
    """Return the voice with *slug*.

    Raises:
        FileNotFoundError: When there is no valid saved voice with that slug.
    """
    if is_slug(slug):
        try:
            return _read_voice(voices_dir() / f"{slug}.yaml")
        except _UNREADABLE:
            pass
    raise FileNotFoundError(f"No saved voice {slug!r}.")


def voice_slug_for_path(path: str | Path) -> str | None:
    """Return the slug of the library file *path* names, or ``None`` when *path* is not a ``<slug>.wav`` directly in the voices folder.

    Only the path is looked at; the voice may since have been deleted, so no record is read.
    """
    try:
        resolved = Path(path).expanduser().resolve()
        if resolved.parent != voices_dir().resolve() or resolved.suffix != ".wav" or not is_slug(resolved.stem):
            return None
    except OSError, ValueError:
        return None
    return resolved.stem


def voice_names() -> dict[str, str]:
    """Return the name of every valid saved voice by its slug; reads the library once."""
    return {voice.slug: voice.name for voice in list_voices()}


def save_voice(
    name: str,
    samples: np.ndarray,
    sample_rate: int,
    *,
    text: str,
    consent: str,
) -> SavedVoice:
    """Save *samples* as a new voice of origin ``"recording"`` named *name* and return it.

    The caller has checked the recording's length; text, shape and sample rate are taken as given.

    Args:
        name: The voice's name: 1 to 60 characters once trimmed, unique ignoring case.
        samples: The reference recording as mono float32 of shape ``(samples,)`` with values in -1..1.
        sample_rate: The recording's sample rate in Hz; it is stored as it is.
        text: What the recording says.
        consent: ``"given"`` or ``"not_needed"``.

    Raises:
        FileExistsError: When a voice already has that name.
        ValueError: When the name is invalid.
    """

    def write_wav(path: Path) -> None:
        write_pcm16_wav_atomic(samples.reshape(1, -1), path, sample_rate)

    return _save(name, write_wav, round(samples.shape[0] / sample_rate, 3), text=text, origin="recording", consent=consent, description=None)


def save_voice_file(
    name: str,
    source: Path,
    seconds: float,
    *,
    text: str,
    description: str | None = None,
) -> SavedVoice:
    """Save a copy of the library-format WAV *source* (16-bit mono PCM) as a new voice of origin ``"design"`` named *name* and return it.

    A designed voice needs no consent: its record says ``"not_needed"``.

    Args:
        name: The voice's name: 1 to 60 characters once trimmed, unique ignoring case.
        source: The WAV to copy; it is left where it is.
        seconds: The WAV's length in seconds.
        text: What the recording says.
        description: The description the voice was made from.

    Raises:
        FileExistsError: When a voice already has that name.
        ValueError: When the name is invalid.
    """

    def write_wav(path: Path) -> None:
        write_bytes_atomic(path, source.read_bytes())

    return _save(name, write_wav, round(seconds, 3), text=text, origin="design", consent="not_needed", description=description)


def _save(name: str, write_wav: Callable[[Path], None], seconds: float, *, text: str, origin: str, consent: str, description: str | None) -> SavedVoice:
    """Write the recording with *write_wav* and the record beside it as a new voice; remove the recording if the record fails."""
    clean_name = _clean_name(name)
    with _WRITE_LOCK:
        _require_free_name(clean_name)
        slug = _free_slug(slugify(clean_name) or _FALLBACK_SLUG)
        wav_path = _wav_path(slug)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_wav(wav_path)
            voice = SavedVoice(
                slug=slug,
                name=clean_name,
                text=text.strip(),
                origin=origin,
                consent=consent,
                seconds=seconds,
                description=(description or "").strip() or None,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                path=wav_path,
            )
            _write_record(_yaml_path(slug), voice)
        except BaseException:
            wav_path.unlink(missing_ok=True)
            raise
    return voice


def rename_voice(slug: str, name: str) -> SavedVoice:
    """Give the voice *slug* a new name; its slug and files stay.

    Raises:
        FileNotFoundError: When there is no such voice.
        FileExistsError: When another voice has that name.
        ValueError: When the name is invalid.
    """
    clean_name = _clean_name(name)
    with _WRITE_LOCK:
        voice = voice_by_slug(slug)
        _require_free_name(clean_name, except_slug=slug)
        renamed = replace(voice, name=clean_name)
        _write_record(_yaml_path(slug), renamed)
    return renamed


def delete_voice(slug: str) -> None:
    """Delete the voice *slug*, its record first and then its recording.

    Raises:
        FileNotFoundError: When there is no such voice.
    """
    with _WRITE_LOCK:
        voice = voice_by_slug(slug)
        _yaml_path(slug).unlink()
        voice.path.unlink(missing_ok=True)


def read_recording(path: str | Path, *, min_seconds: float, max_seconds: float) -> tuple[np.ndarray, int]:
    """Read the recording at *path* as mono float32 at 48 kHz, ready to be saved as a voice.

    The length is probed first and a recording outside *min_seconds*..*max_seconds* is refused before anything is decoded; it is never cut to fit.

    Raises:
        ValueError: With ``utils.audio_track.TRACK_UNREADABLE`` if the file cannot be read, or with the length message if it is too short or too long.
    """
    seconds = probe_track_seconds(str(path))
    check_voice_seconds(seconds, min_seconds, max_seconds)
    stereo = read_track(AudioTrack(str(path), 0.0), AUDIO_SAMPLE_RATE, seconds)
    return stereo.mean(axis=0, dtype="float32"), AUDIO_SAMPLE_RATE


def _wav_path(slug: str) -> Path:
    return voices_dir() / f"{slug}.wav"


def _yaml_path(slug: str) -> Path:
    return voices_dir() / f"{slug}.yaml"


def _clean_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A voice name must be non-empty text.")
    clean = name.strip()
    if len(clean) > _MAX_NAME_LENGTH:
        raise ValueError(f"A voice name can be at most {_MAX_NAME_LENGTH} characters.")
    return clean


def _require_free_name(name: str, *, except_slug: str | None = None) -> None:
    wanted = name.casefold()
    if any(voice.name.casefold() == wanted for voice in list_voices() if voice.slug != except_slug):
        raise FileExistsError(f'A voice named "{name}" already exists.')


def _free_slug(base: str) -> str:
    """Return *base* with a short random hex suffix, drawn again until neither a record nor a recording has that slug.

    The suffix makes a slug unique and effectively never reused, even after the voice that held it is deleted.
    """
    stem = base[: MAX_SLUG_LENGTH - 1 - 2 * _SUFFIX_BYTES].rstrip("-")
    while True:
        candidate = f"{stem}-{_random_suffix()}"
        if not (_yaml_path(candidate).exists() or _wav_path(candidate).exists()):
            return candidate


def _random_suffix() -> str:
    return secrets.token_hex(_SUFFIX_BYTES)


def _read_voice(yaml_path: Path) -> SavedVoice:
    """Read the record at *yaml_path* and pair it with its recording; raise when the record is unreadable or malformed or the recording is missing."""
    slug = yaml_path.stem
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    wav_path = _wav_path(slug)
    if not wav_path.is_file():
        raise FileNotFoundError(f"No recording for the voice {slug!r}.")
    return SavedVoice(slug, data["name"], data["text"], data["origin"], data["consent"], float(data["seconds"]), data.get("description"), data["created_at"], wav_path)


def _write_record(yaml_path: Path, voice: SavedVoice) -> None:
    document = {
        "name": voice.name,
        "text": voice.text,
        "origin": voice.origin,
        "consent": voice.consent,
        "seconds": voice.seconds,
        "description": voice.description,
        "created_at": voice.created_at,
    }
    write_text_atomic(yaml_path, yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
