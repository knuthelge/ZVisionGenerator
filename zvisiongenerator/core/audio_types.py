"""Audio core types: the request for one take, the voice it may follow, and the clip a backend returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

__all__ = ["AudioClip", "AudioGenerationRequest", "VoiceRef"]


@dataclass(frozen=True)
class VoiceRef:
    """A voice recording a speech model follows.

    Attributes:
        path: The audio file.
        text: What the recording says; the model follows the voice with its transcript.
    """

    path: str
    text: str


@dataclass(frozen=True)
class AudioGenerationRequest:
    """One take of audio.

    Attributes:
        preset: The audio model id (the key in ``audio_model_presets``).
        kind: ``music``, ``speech`` or ``sound``.
        prompt: The style of a song, the text of speech, or the description of a sound.
        lyrics: The lyrics of a song; empty for an instrumental or any other kind.
        seconds: The length the user asked for, or ``None`` when the model sets it.
        voice: The voice recording to follow, or ``None``.
        seed: The seed of this take.
    """

    preset: str
    kind: str
    prompt: str
    lyrics: str = ""
    seconds: float | None = None
    voice: VoiceRef | None = None
    seed: int = 0


@dataclass(frozen=True)
class AudioClip:
    """The audio of one take.

    Attributes:
        samples: Float32 samples shaped ``(channels, frames)``.
        sample_rate: Samples per second.
        cut: The model reached its length limit, so the take was cut short.
    """

    samples: NDArray[np.float32]
    sample_rate: int
    cut: bool = False

    @property
    def seconds(self) -> float:
        """The clip's length in seconds."""
        return self.samples.shape[1] / self.sample_rate
