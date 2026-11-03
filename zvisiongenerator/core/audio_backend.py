"""AudioBackend Protocol: the contract of a platform's audio inference engine.

One backend serves one model family (``yue2``, ``higgs3``, ``stable_audio_3``) on a torch device (``cuda`` or ``mps``).
``backends.get_audio_backend(family)`` returns it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest
    from zvisiongenerator.utils.audio_preset import AudioPlatform, AudioPreset

__all__ = ["AudioBackend", "AudioCancelled", "CancelCheck", "ProgressCallback"]

type ProgressCallback = Callable[[str, int, int | None, int, int], None]
"""Reports progress of a take, called as ``on_progress(phase, step, total, part, n_parts)``.

``phase`` is a short label of what the model is doing (``Plan``, ``Song tokens``, ``Synthesis``, ``Decode``,
``Speech tokens`` or ``Sampling``). ``step`` counts the units done in that phase and ``total`` is how many there will be,
or ``None`` when the phase ends on its own and the total is unknown. ``part`` and ``n_parts`` are 1 and 1 for every take
except multi-part speech, where they name the part being made.
"""

type CancelCheck = Callable[[], bool]
"""Returns ``True`` once the user asked to stop; a backend checks it between steps and raises :class:`AudioCancelled`."""


class AudioCancelled(Exception):
    """A backend stopped a take because its ``cancelled`` check returned ``True``.

    Each backend catches its model library's own way of stopping and raises this instead, so the runner handles one type.
    """


@runtime_checkable
class AudioBackend(Protocol):
    """Platform-specific audio inference engine for one model family.

    Required interface:
    - load(): load the model for a preset, from the platform entry's files
    - generate(): make one take
    - close(): free everything the backend holds, also after a failed take
    """

    def load(self, preset: AudioPreset, platform_entry: AudioPlatform, device: str) -> None:
        """Load the model of *preset* onto *device* (``cuda`` or ``mps``).

        Args:
            preset: The audio preset, whose limits and shared settings apply.
            platform_entry: The preset's entry for this platform: the family, the files to load and the settings it overrides.
            device: ``cuda`` or ``mps``.
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
