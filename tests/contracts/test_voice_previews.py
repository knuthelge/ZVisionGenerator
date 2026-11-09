"""Pin that every built-in speech voice ships a preview clip for the Web UI's Play button, and that no clip is left over."""

from __future__ import annotations

from tests._factories import load_packaged_config
from tests._paths import REPO_ROOT
from zvisiongenerator.utils.audio_preset import parse_audio_presets

CLIPS = REPO_ROOT / "frontend" / "public" / "voices"


def _configured_voice_ids() -> set[str]:
    return {voice.id for preset in parse_audio_presets(load_packaged_config()) for voice in preset.voices}


def test_every_configured_voice_has_a_preview_clip():
    ids = _configured_voice_ids()

    assert ids
    assert sorted(voice for voice in ids if not (CLIPS / f"{voice}.mp3").is_file()) == []


def test_every_preview_clip_belongs_to_a_configured_voice():
    assert sorted(path.stem for path in CLIPS.glob("*.mp3") if path.stem not in _configured_voice_ids()) == []
