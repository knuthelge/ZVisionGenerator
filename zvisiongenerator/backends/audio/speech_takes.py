"""Speech takes, whatever model reads them: text in parts, one voice throughout, progress per part and Stop, and the join.

Text is split into parts (``utils/speech_text``), part *i* is seeded ``seed + i``, and the parts are joined into one take
(``utils/speech_join``). Every part is read in the same voice: the recording a take follows, or else the built-in voice the
request names (the preset's default when it names none). Delivery marks in the text split it into parts with their own Delivery;
a recording is read as recorded, so its marks are left out. The model itself is a :class:`SpeechEngine`; each backend supplies one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from zvisiongenerator.core.audio_backend import SPEECH_PHASE, AudioCancelled
from zvisiongenerator.core.audio_types import AudioClip
from zvisiongenerator.core.video_types import AudioTrack
from zvisiongenerator.utils.audio_resample import resample_audio
from zvisiongenerator.utils.audio_track import probe_track_seconds, read_track
from zvisiongenerator.utils.speech_join import join_speech_parts
from zvisiongenerator.utils.speech_text import has_marks, request_speech_parts

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.core.audio_types import AudioGenerationRequest, SpeechDelivery, VoiceRef
    from zvisiongenerator.utils.audio_preset import AudioLimits, AudioPreset

__all__ = ["MARKS_LEFT_OUT_NOTE", "REFERENCE_SILENCE_SECONDS", "SPEECH_PHASE", "SpeechEngine", "read_speech", "read_voice"]

REFERENCE_SILENCE_SECONDS = 0.5
"""Silence an engine appends to a voice recording before the model reads it, so the last sound of the recording does not bleed into the start of the take."""

MARKS_LEFT_OUT_NOTE = "A cloned voice reads as recorded: the Delivery marks were left out."
"""The note a take that follows a recording gets when its text held Delivery marks."""

_READ_RATE_FACTOR = 2
"""A recording is read at this multiple of the engine's rate and then polyphase-resampled to it, which matched a model's own resampling in level and word error rate."""


class SpeechEngine(Protocol):
    """What runs the model: a voice recording prepared once per take, then one part at a time."""

    @property
    def sample_rate(self) -> int:
        """The sample rate of every part's waveform, and of the take."""
        ...

    def prepare_voice(self, voice: VoiceRef) -> Any:
        """Prepare the voice recording to follow, once per take; the result is :meth:`read_part`'s ``reference``."""
        ...

    def read_part(
        self,
        text: str,
        seed: int,
        *,
        speaker: str | None,
        reference: Any,
        delivery: SpeechDelivery | None,
        max_new_tokens: int,
    ) -> tuple[np.ndarray, bool]:
        """Return the mono waveform of one part and whether it was cut.

        Args:
            text: The part's text.
            seed: The part's seed.
            speaker: The built-in voice to read in, or ``None`` when *reference* is given.
            reference: What :meth:`prepare_voice` returned for the recording to follow, or ``None`` with a built-in voice.
            delivery: How to say the part, or ``None`` for the model's own manner (a recording is read as recorded); each engine renders it its own way.
            max_new_tokens: The most tokens the part may take.

        Returns:
            The waveform, empty when the model made no sound for the part, and ``True`` when the part reached *max_new_tokens*.
        """
        ...


def read_voice(voice: VoiceRef, max_seconds: float, sample_rate: int) -> np.ndarray:
    """Return the voice recording as a mono waveform at *sample_rate*: at most *max_seconds* long, read at twice the rate and polyphase-resampled."""
    seconds = min(probe_track_seconds(voice.path), float(max_seconds))
    read_rate = _READ_RATE_FACTOR * sample_rate
    stereo = read_track(AudioTrack(voice.path, 0.0), read_rate, seconds)
    return resample_audio(stereo.mean(axis=0, keepdims=True), read_rate, sample_rate)[0]


def read_speech(
    engine: SpeechEngine,
    request: AudioGenerationRequest,
    preset: AudioPreset,
    limits: AudioLimits,
    settings: Mapping[str, Any],
    on_progress: ProgressCallback,
    cancelled: CancelCheck,
) -> AudioClip:
    """Read the request's text with *engine*, in parts when it is long, and return it as one take.

    Part *i* (counting from 0) is seeded ``request.seed + i``. With a voice recording the engine prepares it once and every
    part follows it, with no Delivery; otherwise every part is read in ``request.speaker``, or in the preset's default voice when
    it names none, with the Delivery of the segment of marked text it belongs to (see :func:`speech_text.speech_parts`).

    Args:
        engine: What runs the model.
        request: The take to make.
        preset: The preset of the model; its voices name the built-in speakers.
        limits: The preset's limits: ``part_chars``, ``max_text_chars`` and ``max_new_tokens``.
        settings: The preset's settings for the platform; the join reads its own.
        on_progress: Called as ``on_progress("Speech", 0, 1, i, n)`` before part *i* of *n* (counting from 1) and
            ``on_progress("Speech", 1, 1, i, n)`` after it.
        cancelled: Checked before each part and before the join; ``True`` stops the take.

    Returns:
        One mono take at the engine's rate, with ``cut`` set when some part reached ``max_new_tokens`` without finishing, and a note
        when a recording's text held Delivery marks that were left out. A model's output is not limited to -1..1, so a loud take may
        peak above 1.0; the runner levels it.

    Raises:
        AudioCancelled: If ``cancelled`` returned ``True``.
        ValueError: If the request names a speaker the preset does not have, or the preset has no built-in voice for a take
            without a recording.
        RuntimeError: If the model made no sound.
    """
    parts = request_speech_parts(request, part_chars=int(limits.part_chars), max_text_chars=int(limits.max_text_chars))
    max_new_tokens = int(limits.max_new_tokens)
    reference = None if request.voice is None else engine.prepare_voice(request.voice)
    speaker = None if request.voice is not None else _speaker(request, preset)
    waves: list[np.ndarray] = []
    cut = False
    for index, part in enumerate(parts):
        if cancelled():
            raise AudioCancelled
        on_progress(SPEECH_PHASE, 0, 1, index + 1, len(parts))
        wave, part_cut = engine.read_part(part.text, request.seed + index, speaker=speaker, reference=reference, delivery=part.delivery, max_new_tokens=max_new_tokens)
        on_progress(SPEECH_PHASE, 1, 1, index + 1, len(parts))
        cut = cut or part_cut
        waves.append(wave)
    if cancelled():
        raise AudioCancelled
    if not any(wave.size for wave in waves):
        raise RuntimeError("The model made no sound for this text. Try another seed.")
    joined = join_speech_parts(waves, engine.sample_rate, settings)
    notes = (MARKS_LEFT_OUT_NOTE,) if request.voice is not None and has_marks(request.prompt) else ()
    return AudioClip(samples=joined.reshape(1, -1), sample_rate=engine.sample_rate, cut=cut, notes=notes)


def _speaker(request: AudioGenerationRequest, preset: AudioPreset) -> str:
    """Return the built-in voice a take without a recording reads in: the request's speaker, else the preset's default.

    Raises:
        ValueError: If the preset has no built-in voices, or the request names one it does not have.
    """
    speaker = request.speaker or preset.default_voice
    if speaker is None:
        raise ValueError(f"{preset.label} has no built-in voice; follow a voice recording instead.")
    known = [voice.id for voice in preset.voices]
    if speaker not in known:
        raise ValueError(f"{preset.label} has no voice '{speaker}'; use one of {', '.join(known)}.")
    return speaker
