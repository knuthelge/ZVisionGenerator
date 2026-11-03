"""Which voice recordings need the speaker's consent before a model follows them."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.provenance import read_config

__all__ = ["is_any_voice_take", "voice_needs_consent"]


def is_any_voice_take(config: Mapping[str, Any] | None) -> bool:
    """Return whether *config*, the settings embedded in an audio file, show a speech take read in the model's own voice.

    Such a take was made without a reference voice, so following it needs nobody's permission.
    """
    return isinstance(config, Mapping) and config.get("kind") == "speech" and not config.get("voice_path")


def voice_needs_consent(path: str | Path) -> bool:
    """Return whether using the recording at *path* as a voice needs the speaker's consent.

    Only a speech take of this app whose embedded settings show no reference voice (an *Any* take) is free to use. Any
    other file, including one whose settings cannot be read, needs consent.
    """
    try:
        config = read_config(path)
    except subprocess.CalledProcessError, ValueError, OSError:
        return True
    return not is_any_voice_take(config)
