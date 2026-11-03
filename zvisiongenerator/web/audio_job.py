"""Turn the fields of an audio generate form into the request of one take, with the refusals ``ziv audio`` makes.

The form's field names are the ones Copy CLI reads (``frontend/src/lib/state/cliCommand.ts``), and the request built here is
the one the command it prints builds: the style of a song and the description of a sound are trimmed, the text of speech
is kept as typed, and a field that belongs to another kind is ignored, as the form hides it. A field the kind shows but
the model does not take is refused.
"""

from __future__ import annotations

from dataclasses import dataclass

from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
from zvisiongenerator.utils.audio_preset import AudioPreset
from zvisiongenerator.utils.audio_request import check_speech_request, check_supported_fields, resolve_sound_seconds
from zvisiongenerator.utils.audio_voice import voice_needs_consent

__all__ = ["AudioFields", "audio_request_from_fields"]

_PROMPT_REQUIRED = {
    "music": "Write a style first.",
    "speech": "Write the text to read first.",
    "sound": "Describe the sound first.",
}
_VOICE_TEXT_REQUIRED = "Write what the recording says first."
_CONSENT_REQUIRED = "Cloning a voice needs the speaker's permission: tick the box to confirm it."
_FIELD_NAMES = {"lyrics": "lyrics", "seconds": "seconds", "voice": "voice_path"}


@dataclass(frozen=True)
class AudioFields:
    """The values of an audio form, read but not yet checked.

    Attributes:
        prompt: The style, the text or the description, as submitted.
        lyrics: The lyrics, as submitted; blank for none.
        seconds: The requested length of a sound, or ``None``.
        voice_path: The recording to follow (a typed path, or the saved upload), or ``None``.
        voice_text: What the recording says, trimmed, or ``None``.
        voice_consent: Whether the consent box was ticked.
        voice_unread: Whether the recording is a stand-in that was never uploaded (a download plan), so its file is neither
            read for the speaker's consent nor measured; both are checked when the form is submitted.
    """

    prompt: str = ""
    lyrics: str = ""
    seconds: float | None = None
    voice_path: str | None = None
    voice_text: str | None = None
    voice_consent: bool = False
    voice_unread: bool = False


def audio_request_from_fields(preset: AudioPreset, fields: AudioFields) -> AudioGenerationRequest:
    """Check *fields* against *preset* and return the request of one take (its seed is set per take by the runner).

    Lyrics count for music only, the length for sound only and the voice for speech only. Line breaks are the browser's
    CRLF made LF again. A voice needs its transcript and the speaker's consent, except a speech take of this app made
    without a reference voice; speech is checked against the model's limits (text length, voice length, readable voice).

    Raises:
        ValueError: If the style, text or description is blank, a field is one the model does not take, the length is out
            of range, the voice has no transcript or no consent, or the speech request is over the model's limits.
    """
    kind = preset.kind
    prompt = _line_breaks(fields.prompt)
    if not prompt.strip():
        raise ValueError(_PROMPT_REQUIRED[kind])
    lyrics = _line_breaks(fields.lyrics) if kind == "music" and fields.lyrics.strip() else ""
    voice_path = fields.voice_path if kind == "speech" else None
    seconds = fields.seconds if kind == "sound" else None
    check_supported_fields(preset, lyrics=bool(lyrics), seconds=seconds is not None, voice=voice_path is not None, drop=_FIELD_NAMES)
    request = AudioGenerationRequest(
        preset=preset.id,
        kind=kind,
        prompt=prompt if kind == "speech" else prompt.strip(),
        lyrics=lyrics,
        seconds=resolve_sound_seconds(seconds, preset, name="seconds") if kind == "sound" else None,
        voice=_voice(fields, voice_path),
    )
    if kind == "speech":
        check_speech_request(request, preset.limits, check_voice_file=not fields.voice_unread)
    return request


def _voice(fields: AudioFields, voice_path: str | None) -> VoiceRef | None:
    """Return the voice to follow, after checking that it has a transcript and the consent it needs."""
    if voice_path is None:
        return None
    if not fields.voice_text:
        raise ValueError(_VOICE_TEXT_REQUIRED)
    if not fields.voice_consent and not fields.voice_unread and voice_needs_consent(voice_path):
        raise ValueError(_CONSENT_REQUIRED)
    return VoiceRef(voice_path, fields.voice_text)


def _line_breaks(text: str) -> str:
    """Return *text* with the CRLF line breaks of a multipart form made LF, as the shell hands them to ``ziv audio``."""
    return text.replace("\r\n", "\n")
