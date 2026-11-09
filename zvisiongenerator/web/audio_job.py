"""Turn the fields of an audio generate form into the request of one take, with the refusals ``ziv audio`` makes.

The form's field names are the ones Copy CLI reads (``frontend/src/lib/state/cliCommand.ts``), and the request built here is
the one the command it prints builds: the style of a song and the description of a sound are trimmed, the text of speech
is kept as typed, and a field that belongs to another kind is ignored, as the form hides it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
from zvisiongenerator.utils.audio_preset import AudioPreset
from zvisiongenerator.utils.audio_request import check_speech_request, resolve_audio_seconds, resolve_speaker
from zvisiongenerator.utils.audio_settings import DELIVERY_FIELDS, parse_delivery, parse_key, parse_sound_type, parse_tempo
from zvisiongenerator.utils.audio_voice import CONSENT_REQUIRED, voice_needs_consent

__all__ = ["AudioFields", "audio_request_from_fields"]

_PROMPT_REQUIRED = {
    "music": "Write a style first.",
    "speech": "Write the text to read first.",
    "sound": "Describe the sound first.",
}
_VOICE_TEXT_REQUIRED = "Write what the recording says first."


@dataclass(frozen=True)
class AudioFields:
    """The values of an audio form, read but not yet checked.

    Attributes:
        prompt: The style, the text or the description, as submitted.
        lyrics: The lyrics, as submitted; blank for none.
        seconds: The requested length of a sound or of music from a model that sets its length, or ``None`` for Auto.
        voice_path: The recording to follow (a typed path, or the saved upload), or ``None``.
        saved_voice: The saved voice to follow, with its stored transcript, or ``None``. It replaces the recording, its transcript, the consent and the built-in voice
            and delivery: the speaker's consent is in the voice's record, so there is no tick to check.
        voice_text: What the recording says, trimmed, or ``None``.
        voice_consent: Whether the consent box was ticked.
        speaker: The built-in voice the form chose (an id of the preset's ``voices``), or ``None`` for the preset's default; ignored with a recording.
        tempo: The tempo of a song in BPM (a number, or text from a form), or ``None``/blank for Auto; music only.
        key: The key of a song as an ABC id (``F#m``), or ``None``/blank for Auto; music only.
        emotion: The emotion of a speech take in a built-in voice, or ``None`` for the default; the same for ``manner``, ``pace``, ``pitch`` and ``expressiveness``.
        sound_type: The type of a sound (``auto``, ``ambience``, ``effect`` or ``one-shot``), or ``None`` for ``auto``; sound only.
        prompt_input: The user's own text when ``prompt`` is a rewrite of it (it may be ``""``); kept only with an ``enhance_record``.
        enhance_record: How ``prompt`` was written (options, model, tier, format), or ``None`` for text the user wrote.
        voice_unread: Whether the recording is a stand-in that was never uploaded (a download plan), so its file is neither
            read for the speaker's consent nor measured; both are checked when the form is submitted.
    """

    prompt: str = ""
    lyrics: str = ""
    seconds: float | None = None
    voice_path: str | None = None
    saved_voice: VoiceRef | None = None
    voice_text: str | None = None
    voice_consent: bool = False
    speaker: str | None = None
    tempo: int | str | None = None
    key: str | None = None
    emotion: str | None = None
    manner: str | None = None
    pace: str | None = None
    pitch: str | None = None
    expressiveness: str | None = None
    sound_type: str | None = None
    prompt_input: str | None = None
    enhance_record: dict[str, Any] | None = None
    voice_unread: bool = False


def audio_request_from_fields(preset: AudioPreset, kind: str, fields: AudioFields) -> AudioGenerationRequest:
    """Check *fields* against *preset* as a *kind* take and return the request of one take (its seed is set per take by the runner).

    Lyrics count for music of a model that sings them, the length for a (model, kind) that sets it, the tempo and key for music, the sound type for sound,
    and the voice for speech only. A saved voice stands for itself and needs no tick. Line breaks are the browser's
    CRLF made LF again. A voice needs its transcript and the speaker's consent, except a speech take of this app made
    without a reference voice; speech is checked against the model's limits (text length, voice length, readable voice).
    A speech take without a recording is read in the built-in voice the form chose, else the model's default; with a
    recording it has no built-in voice, and so no delivery. A length left blank is the Auto length of the kind (of the sound type for a sound).

    Raises:
        ValueError: If the model does not make *kind*, the style, text or description is blank, the length, tempo, key, delivery or sound type is out of range or
            unknown, the voice has no transcript or no consent, the built-in voice is not one of the model's, or the speech request is over the model's limits.
    """
    if not preset.has_kind(kind):
        raise ValueError(f"{preset.label} makes {' and '.join(preset.kinds)}, not {kind}.")
    prompt = _line_breaks(fields.prompt)
    if not prompt.strip():
        raise ValueError(_PROMPT_REQUIRED[kind])
    lyrics = _line_breaks(fields.lyrics) if preset.takes_lyrics(kind) and fields.lyrics.strip() else ""
    voice = _speech_voice(fields) if kind == "speech" else None
    sound_type = parse_sound_type(fields.sound_type) if kind == "sound" else None
    seconds = resolve_audio_seconds(fields.seconds, preset, kind, sound_type, name="seconds")
    enhance_record = fields.enhance_record or None
    request = AudioGenerationRequest(
        preset=preset.id,
        kind=kind,
        prompt=prompt if kind == "speech" else prompt.strip(),
        lyrics=lyrics,
        seconds=seconds,
        requested_seconds=seconds if fields.seconds is not None else None,
        voice=voice,
        speaker=resolve_speaker(preset, fields.speaker) if kind == "speech" and voice is None else None,
        tempo=parse_tempo(fields.tempo) if kind == "music" else None,
        key=parse_key(fields.key) if kind == "music" else None,
        delivery=parse_delivery({name: getattr(fields, name) for name in DELIVERY_FIELDS}) if kind == "speech" and voice is None else None,
        sound_type=sound_type,
        prompt_input=_line_breaks(fields.prompt_input or "") if enhance_record else None,
        enhance_record=enhance_record,
    )
    if kind == "speech":
        check_speech_request(request, preset.limits, check_voice_file=not fields.voice_unread)
    return request


def _speech_voice(fields: AudioFields) -> VoiceRef | None:
    """Return the voice a speech take follows: the saved voice when the form names one (the recording fields are ignored), else the checked recording, else ``None``."""
    return fields.saved_voice if fields.saved_voice is not None else _recording(fields)


def _recording(fields: AudioFields) -> VoiceRef | None:
    """Return the recording to follow, after checking that it has a transcript and the consent it needs."""
    voice_path = fields.voice_path
    if voice_path is None:
        return None
    if not fields.voice_text:
        raise ValueError(_VOICE_TEXT_REQUIRED)
    if not fields.voice_consent and not fields.voice_unread and voice_needs_consent(voice_path):
        raise ValueError(CONSENT_REQUIRED)
    return VoiceRef(voice_path, fields.voice_text)


def _line_breaks(text: str) -> str:
    """Return *text* with the CRLF line breaks of a multipart form made LF, as the shell hands them to ``ziv audio``."""
    return text.replace("\r\n", "\n")
