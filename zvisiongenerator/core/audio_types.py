"""Audio core types: the request for one take, the voice it may follow, and the clip a backend returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

__all__ = ["AudioClip", "AudioGenerationRequest", "SpeechDelivery", "VoiceDesignRequest", "VoiceRef"]


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
class VoiceDesignRequest:
    """A request to design a voice from a description.

    Attributes:
        description: The voice in words (age, tone, pace, accent).
        text: The line every candidate speaks; it becomes the transcript of a saved voice.
        count: How many candidates to make.
    """

    description: str
    text: str
    count: int = 3


@dataclass(frozen=True)
class SpeechDelivery:
    """How a speech take is delivered; every setting starts at its neutral value.

    Attributes:
        emotion: ``neutral`` or one of the emotions (``sad``, ``proud``, ...).
        manner: ``natural``, ``whisper`` or ``shout``.
        pace: ``very-slow``, ``slow``, ``normal``, ``fast`` or ``very-fast``.
        pitch: ``low``, ``normal`` or ``high``.
        expressiveness: ``low``, ``normal`` or ``high``.
    """

    emotion: str = "neutral"
    manner: str = "natural"
    pace: str = "normal"
    pitch: str = "normal"
    expressiveness: str = "normal"


@dataclass(frozen=True)
class AudioGenerationRequest:
    """One take of audio.

    Attributes:
        preset: The audio model id (the key of its ``models`` entry in the model metadata).
        kind: ``music``, ``speech`` or ``sound``.
        prompt: The style of a song, the text of speech, or the description of a sound.
        lyrics: The lyrics of a song; empty for an instrumental or any other kind.
        seconds: The length of the take, resolved (a sound or music left on Auto has its Auto length), or ``None`` when the model sets it.
        requested_seconds: The length the user asked for, or ``None`` for Auto (and when the model sets the length); it is what Reuse sends back.
        voice: The voice recording to follow, or ``None``.
        seed: The seed of this take.
        speaker: The built-in voice a speech take reads in, or ``None`` for a recording or the preset's default voice.
        tempo: The tempo of a song in beats per minute, or ``None`` for Auto.
        key: The key of a song as an ABC id (``F#m``), or ``None`` for Auto.
        delivery: How a speech take in a built-in voice is delivered, or ``None`` (a recording, or any other kind).
        sound_type: The type of a sound (``auto``, ``ambience``, ``effect`` or ``one-shot``), or ``None`` for any other kind.
        prompt_input: The user's own text when ``prompt`` is a rewrite of it (it may be ``""``), else ``None``.
        enhance_record: How ``prompt`` was written (options, model, tier, format), or ``None`` for text the user wrote.
    """

    preset: str
    kind: str
    prompt: str
    lyrics: str = ""
    seconds: float | None = None
    requested_seconds: float | None = None
    voice: VoiceRef | None = None
    seed: int = 0
    speaker: str | None = None
    tempo: int | None = None
    key: str | None = None
    delivery: SpeechDelivery | None = None
    sound_type: str | None = None
    prompt_input: str | None = None
    enhance_record: dict[str, Any] | None = None


@dataclass(frozen=True)
class AudioClip:
    """The audio of one take.

    Attributes:
        samples: Float32 samples shaped ``(channels, frames)``.
        sample_rate: Samples per second.
        cut: The model reached its length limit, so the take was cut short.
        notes: Things the user should know about this take, one sentence each (such as a setting the model could not follow).
    """

    samples: NDArray[np.float32]
    sample_rate: int
    cut: bool = False
    notes: tuple[str, ...] = ()

    @property
    def seconds(self) -> float:
        """The clip's length in seconds."""
        return self.samples.shape[1] / self.sample_rate
