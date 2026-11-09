"""AudioBackend Protocol: the contract of a platform's audio inference engine.

One backend serves one model family (``yue2``, ``qwen3_tts``, ``stable_audio_3``) on a torch device (``cuda`` or ``mps``).
``backends.get_audio_backend(family)`` returns it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest
    from zvisiongenerator.utils.audio_preset import AudioPreset

__all__ = ["DECODE_PHASE", "SPEECH_PHASE", "AudioBackend", "AudioCancelled", "CancelCheck", "ProgressCallback", "VoiceDesigner", "raise_if_cancelled"]

DECODE_PHASE = "Decode"
"""The progress phase after the last step, when the decoder turns what was made into the waveform; it has no count of its own worth showing."""

SPEECH_PHASE = "Speech"
"""The progress phase of a speech take: one step per part, reported before the part and after it."""

type ProgressCallback = Callable[[str, int, int | None, int, int], None]
"""Reports progress of a take, called as ``on_progress(phase, step, total, part, n_parts)``.

``phase`` is a short label of what the model is doing (``Plan``, ``Song tokens``, ``Synthesis``, ``Decode``,
``Speech`` or ``Sampling``). ``step`` counts the units done in that phase and ``total`` is how many there will be,
or ``None`` when the phase ends on its own and the total is unknown. ``part`` and ``n_parts`` are 1 and 1 for every take
except multi-part speech, where they name the part being made.
"""

type CancelCheck = Callable[[], bool]
"""Returns ``True`` once the user asked to stop; a backend checks it between steps and raises :class:`AudioCancelled`."""


class AudioCancelled(Exception):
    """A backend stopped a take because its ``cancelled`` check returned ``True``.

    Each backend catches its model library's own way of stopping and raises this instead, so the runner handles one type.
    """


def raise_if_cancelled(cancelled: CancelCheck | None) -> None:
    """Raise :class:`AudioCancelled` when *cancelled* is given and returns ``True``; do nothing otherwise."""
    if cancelled is not None and cancelled():
        raise AudioCancelled


@runtime_checkable
class AudioBackend(Protocol):
    """Platform-specific audio inference engine for one model family.

    Required interface:
    - load(): load the model for a preset, from the preset's files
    - generate(): make one take
    - close(): free everything the backend holds, also after a failed take
    """

    def load(self, preset: AudioPreset, device: str, cancelled: CancelCheck | None = None) -> None:
        """Load the model of *preset* onto *device* (``cuda`` or ``mps``).

        Args:
            preset: The audio preset: its files, limits and settings (``preset.settings_for(device)``).
            device: ``cuda`` or ``mps``.
            cancelled: Called between the tensors read from the weight files; ``True`` stops the load. The backend frees what it
                held before raising.

        Raises:
            AudioCancelled: If ``cancelled`` returned ``True`` and the load stopped.
        """
        ...

    def generate(self, request: AudioGenerationRequest, on_progress: ProgressCallback, cancelled: CancelCheck) -> AudioClip:
        """Make one take for *request* and return its audio.

        Args:
            request: The take to make.
            on_progress: Called as the take advances: ``on_progress(phase, step, total, part, n_parts)``.
            cancelled: Called between steps; ``True`` means stop.

        Raises:
            AudioCancelled: If ``cancelled`` returned ``True`` and the take stopped.
        """
        ...

    def close(self) -> None:
        """Free the model and the memory limits the backend set."""
        ...


@runtime_checkable
class VoiceDesigner(Protocol):
    """Optional capability: design a new voice from a written description.

    Not part of :class:`AudioBackend`: only the Qwen3-TTS backends offer it, so a caller checks ``isinstance(backend, VoiceDesigner)``.
    The backend must have been loaded with a preset whose ``files[0]`` is a voice-design checkpoint; the runner arranges this by
    handing it ``replace(preset, files=preset.design_files)``.
    """

    @property
    def sample_rate(self) -> int:
        """The sample rate of the waves :meth:`design_voice` returns."""
        ...

    def design_voice(self, description: str, text: str, seed: int) -> np.ndarray:
        """Speak *text* in a voice made from *description* and return it as one candidate.

        Args:
            description: The voice in words (age, tone, pace, accent).
            text: The line to speak; it is also what the candidate sounds like when saved.
            seed: Seeds the sampling, so the same seed gives the same candidate.

        Returns:
            A mono float32 wave at :attr:`sample_rate`; empty when the model made no sound.
        """
        ...
