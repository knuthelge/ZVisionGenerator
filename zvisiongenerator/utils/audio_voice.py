"""Which voice recordings need the speaker's consent before a model follows them."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.provenance import read_config

__all__ = ["CONSENT_REQUIRED", "consent_for_saving", "voice_needs_consent"]

# The message of the ``ValueError`` raised when a recording needs the speaker's consent and none was given; the Web UI and CLI show it as it is.
CONSENT_REQUIRED = "Cloning a voice needs the speaker's permission: tick the box to confirm it."


def _is_built_in_voice_take(config: Mapping[str, Any] | None) -> bool:
    """Return whether *config*, the settings embedded in an audio file, show a speech take read in a built-in voice.

    Such a take was made without a reference voice, so following it needs nobody's permission.
    """
    return isinstance(config, Mapping) and config.get("kind") == "speech" and not config.get("voice_path")


def voice_needs_consent(path: str | Path) -> bool:
    """Return whether using the recording at *path* as a voice needs the speaker's consent.

    Only a speech take of this app whose embedded settings show no reference voice (a take in a built-in voice) is free to use. Any
    other file, including one whose settings cannot be read, needs consent.
    """
    try:
        config = read_config(path)
    except subprocess.CalledProcessError, ValueError, OSError:
        return True
    return not _is_built_in_voice_take(config)


def consent_for_saving(path: str | Path, ticked: bool) -> str:
    """Return the consent to store with a voice saved from the recording at *path*: ``"given"`` or ``"not_needed"``.

    Args:
        path: The recording to save.
        ticked: Whether the speaker's consent was confirmed.

    Raises:
        ValueError: With ``CONSENT_REQUIRED`` if the recording needs consent and *ticked* is false.
    """
    if ticked:
        return "given"
    if voice_needs_consent(path):
        raise ValueError(CONSENT_REQUIRED)
    return "not_needed"
