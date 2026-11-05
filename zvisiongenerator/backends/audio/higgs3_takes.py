"""Higgs TTS 3 takes, whatever runs the model: text read in parts, one voice throughout, frame progress and Stop, and the join.

Text is split into parts (``utils/speech_text``), part *i* is seeded ``seed + i``, and the parts are joined into one take
(``utils/speech_join``). With a recording every part follows it; without one, part 1 is read in the model's own voice and
its own waveform is the reference for the rest. The model itself is a :class:`SpeechEngine`: the torch backend on CUDA
and the MLX backend on macOS each supply one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from zvisiongenerator.core.audio_backend import DECODE_PHASE, AudioCancelled
from zvisiongenerator.core.audio_types import AudioClip
from zvisiongenerator.core.video_types import AudioTrack
from zvisiongenerator.utils.audio_resample import resample_audio
from zvisiongenerator.utils.audio_track import probe_track_seconds, read_track
from zvisiongenerator.utils.speech_join import join_speech_parts
from zvisiongenerator.utils.speech_text import split_speech_text

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
    from zvisiongenerator.utils.audio_preset import AudioLimits

__all__ = ["SAMPLE_RATE", "SpeechEngine", "estimate_part_frames", "read_speech"]

SAMPLE_RATE = 24000
"""The model's sample rate, and the rate of every take."""

_SPEECH_PHASE = "Speech tokens"

_FRAME_STRIDE = 10
"""Report every this many frames (about 2.5 a second at the codec's 25 frames a second), so a long take does not flood the progress events."""

_FRAMES_PER_SECOND = 25
"""The codec's frame rate: frames of speech codes per second of sound."""

_CHARS_PER_SECOND = 16.5
"""How many characters of text the model speaks a second, in the middle of the 16 to 18 measured over real takes."""

_FRAMES_PER_CHAR = _FRAMES_PER_SECOND / _CHARS_PER_SECOND
"""How many frames a character of text takes to read, which turns a part's length into the frames it should need."""

# A recording is read at twice the model's rate and then polyphase-resampled to it, which matched the model's own
# resampling in level and word error rate.
_READ_RATE = 2 * SAMPLE_RATE


class SpeechEngine(Protocol):
    """What runs the model: one part at a time, and the codec's encoder for a voice reference."""

    def read_part(
        self,
        text: str,
        seed: int,
        reference: tuple[Any, str | None] | None,
        max_new_tokens: int,
        on_frame: Callable[[], None],
        on_decode: Callable[[int], None],
    ) -> tuple[np.ndarray, bool]:
        """Return the 24 kHz mono waveform of one part and whether it was cut.

        Args:
            text: The part's text.
            seed: The part's seed.
            reference: The codes and transcript of the voice to follow, as :meth:`encode` returned them, or ``None``.
            max_new_tokens: The most frames the part may take.
            on_frame: Called once per generated frame, the one that ends the part included; it may raise to stop the part.
            on_decode: Called with 0 before the codes are decoded and 1 after; not called for a part without sound.
        """
        ...

    def encode(self, wave: np.ndarray) -> Any:
        """Encode a 24 kHz mono waveform into the codes :meth:`read_part` takes as a reference."""
        ...


def estimate_part_frames(text: str, max_new_tokens: int) -> int:
    """Return how many frames reading *text* should take, at least 1 and at most *max_new_tokens*.

    The estimate is the text's length at the model's usual speaking rate; a part with more pauses or a slower voice
    needs more, which :func:`_part_total` allows for.
    """
    return max(1, min(max_new_tokens, round(len(text) * _FRAMES_PER_CHAR)))


def _part_total(estimate: int, step: int, max_new_tokens: int) -> int:
    """Return the total to report at frame *step* of a part expected to take *estimate* frames.

    It is the estimate until the part runs past it; after that it stays one stride ahead, so the bar waits near the end
    instead of overflowing. It never exceeds *max_new_tokens*, which ends a part.
    """
    return min(max_new_tokens, max(estimate, step + _FRAME_STRIDE))


def _trim_reference(wave: np.ndarray, text: str, max_seconds: float) -> tuple[np.ndarray, str]:
    """Return the start of *wave* up to *max_seconds* with the same share of the words of *text*, or both as they are when *wave* is short enough.

    A voice reference is held to the length of a recording; the transcript is cut in proportion so it still matches what is heard.
    """
    keep = int(max_seconds * SAMPLE_RATE)
    if wave.shape[0] <= keep:
        return wave, text
    words = text.split()
    kept_words = max(1, int(len(words) * keep / wave.shape[0]))
    return wave[:keep], " ".join(words[:kept_words])


def _read_voice(voice: VoiceRef, max_seconds: float) -> np.ndarray:
    """Return the voice recording as a 24 kHz mono waveform: at most *max_seconds*, read at twice the rate and polyphase-resampled."""
    seconds = min(probe_track_seconds(voice.path), float(max_seconds))
    stereo = read_track(AudioTrack(voice.path, 0.0), _READ_RATE, seconds)
    return resample_audio(stereo.mean(axis=0, keepdims=True), _READ_RATE, SAMPLE_RATE)[0]


@dataclass
class _Progress:
    """The position of the part being made, which the frame and decode callbacks report from."""

    on_progress: ProgressCallback
    cancelled: CancelCheck
    max_new_tokens: int
    n_parts: int
    part: int = 1
    estimate: int = 1
    step: int = 0
    finished: bool = False

    def start(self, part: int, text: str) -> None:
        """Begin part *part* of *text*: no frames yet, and its estimated total."""
        self.part, self.step, self.finished = part, 0, False
        self.estimate = estimate_part_frames(text, self.max_new_tokens)

    def frame(self) -> None:
        """Run once per generated frame: stop if asked, else count and report the first frame and every ``_FRAME_STRIDE``-th."""
        if self.cancelled():
            raise AudioCancelled
        self.step += 1
        if self.step == 1 or self.step % _FRAME_STRIDE == 0:
            self.report()

    def finish(self) -> None:
        """Report the part's last frame, with the frames it took as the total, once; nothing for a part with no frames."""
        if self.step and not self.finished:
            self.finished = True
            self.report(finished=True)

    def report(self, *, finished: bool = False) -> None:
        """Report the frame count against the part's estimated total; a *finished* part reports the frames it took as its total, so its bar is full."""
        total = self.step if finished else _part_total(self.estimate, self.step, self.max_new_tokens)
        self.on_progress(_SPEECH_PHASE, self.step, total, self.part, self.n_parts)

    def decode(self, done: int) -> None:
        """Report the Decode phase of the part, after its last frame: *done* is 0 before the decode and 1 after."""
        self.finish()
        self.on_progress(DECODE_PHASE, done, 1, self.part, self.n_parts)


def read_speech(
    engine: SpeechEngine,
    request: AudioGenerationRequest,
    limits: AudioLimits,
    settings: Mapping[str, Any],
    on_progress: ProgressCallback,
    cancelled: CancelCheck,
) -> AudioClip:
    """Read the request's text with *engine*, in parts when it is long, and return it as one take.

    Part *i* is seeded ``request.seed + i``. With a voice recording, every part follows the recording (with its
    transcript). Without one, part 1 is read in the model's own voice; when there are more parts, part 1's own
    waveform is encoded once and is the reference for parts 2 to *n* (with part 1's text as the transcript), so one
    voice and room reads the whole text. That reference is cut to the preset's ``max_voice_seconds``, the same
    length a recording is held to, with the transcript cut to the same share of its words.

    Args:
        engine: What runs the model.
        request: The take to make.
        limits: The preset's limits: ``part_chars``, ``max_text_chars``, ``max_new_tokens`` and ``max_voice_seconds``.
        settings: The preset's settings for the platform; the join reads its own.
        on_progress: Called as ``on_progress("Speech tokens", step, total, part, n_parts)`` for the first frame of a
            part, every 10 frames after it and the last frame, and with phase ``Decode`` around each
            part's decode. ``total`` is the frames the part's text should need (:func:`estimate_part_frames`); the last
            frame reports the frames the part took as its total, so a part's bar ends full.
        cancelled: Checked before each part and on every frame; ``True`` stops the take.

    Returns:
        One mono take at 24 kHz, with ``cut`` set when some part reached ``max_new_tokens`` without finishing. The codec's
        output is not limited to -1..1, so a loud take may peak above 1.0; the runner levels it.

    Raises:
        AudioCancelled: If ``cancelled`` returned ``True``.
        RuntimeError: If the model made no sound.
    """
    parts = split_speech_text(request.prompt, int(limits.part_chars), max_text_chars=int(limits.max_text_chars))
    max_new_tokens = int(limits.max_new_tokens)
    recording = (engine.encode(_read_voice(request.voice, float(limits.max_voice_seconds))), request.voice.text) if request.voice is not None else None
    reference = recording  # a recording is every part's reference; otherwise part 1 has none and the later parts follow part 1
    progress = _Progress(on_progress, cancelled, max_new_tokens, len(parts))
    waves: list[np.ndarray] = []
    cut = False
    for index, text in enumerate(parts):
        if cancelled():
            raise AudioCancelled
        progress.start(index + 1, text)
        wave, part_cut = engine.read_part(text, request.seed + index, reference, max_new_tokens, progress.frame, progress.decode)
        progress.finish()  # a part without sound has no decode, so its last frame is reported here
        cut = cut or part_cut
        if index == 0 and len(parts) > 1 and recording is None:
            if not wave.size:
                raise RuntimeError("Higgs TTS 3 made no sound for the first part of the text, so the rest has no voice to follow. Try another seed.")
            reference_wave, reference_text = _trim_reference(wave, text, float(limits.max_voice_seconds))
            reference = (engine.encode(reference_wave), reference_text)
        waves.append(wave)
    if not any(wave.size for wave in waves):
        raise RuntimeError("Higgs TTS 3 made no sound for this text. Try another seed.")
    joined = join_speech_parts(waves, SAMPLE_RATE, settings)
    return AudioClip(samples=joined.reshape(1, -1), sample_rate=SAMPLE_RATE, cut=cut)
