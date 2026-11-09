"""Pin which audio backends can design a voice: both Qwen3-TTS backends do, the song and sound-effect backends do not.

The check is on the classes' instances only (no model is loaded), so it runs on every platform.
"""

from __future__ import annotations

import pytest

from zvisiongenerator.backends import get_audio_backend
from zvisiongenerator.core.audio_backend import VoiceDesigner


@pytest.mark.parametrize("family", ["qwen3_tts", "qwen3_tts_mlx"])
def test_both_qwen3_tts_backends_can_design_a_voice(family):
    assert isinstance(get_audio_backend(family), VoiceDesigner)


@pytest.mark.parametrize("family", ["yue2", "stable_audio_3"])
def test_the_song_and_sound_backends_cannot(family):
    assert not isinstance(get_audio_backend(family), VoiceDesigner)
