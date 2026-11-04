"""The three audio presets as the config writes them, with their pinned sources, shared by tests in more than one area.

Import them with ``from tests._audio_presets import ...``. Each builder returns a fresh dict, so a test may change it freely.
"""

from __future__ import annotations

from typing import Any

YUE2_REPO = "m-a-p/YuE2-3B@8f24312f187763b9854adeec87e437e2b03bbff1"
YUE2_VAE_REPO = "m-a-p/YuE2-Vae@152733a19ad43aa67e367f9b5503ef8075bb5126"
HIGGS_REPO = "bosonai/higgs-tts-3-4b@239f63fb7b02b1aa085f98d9efae5e35cc5523e8"
HIGGS_CODEC_REPO = "bosonai/higgs-audio-v2-tokenizer@815a03f1b57dcb5ae918db415f22cd0c8a6f0f54"
SA3_CHECKPOINT_REPO = "Comfy-Org/stable-audio-3@75aad8836271cb5a5123fb58304fae5d3bb4b0f9"
SA3_CONFIG_REPO = "cocktailpeanut/stable-audio-3-medium@a30034d70bd58f6e6f967ef28ecaa6c52c185d6c"
SA3_TEXT_ENCODER_REPO = "mlx-community/stable-audio-3-small-sfx@ca6c38718d27cd2b446a4661c8851f1f3d0a3870"

INSTRUMENTAL_SECTIONS = "[Intro] [Instrumental] [Solo] [Instrumental] [Solo] [Bridge] [Instrumental] [Outro]"


def yue2_files() -> list[dict[str, Any]]:
    """Return the YuE2 files: the 3B model and the VAE."""
    return [
        {"repo": YUE2_REPO, "files": ["config.json", "generation_config.json", "yue2_generation_config.json", "model.safetensors", "modeling_yue2.py", "qwen.tiktoken", "weights_manifest.json"]},
        {"repo": YUE2_VAE_REPO, "files": ["config.json", "model.safetensors", "modeling_vae.py", "weights_manifest.json"]},
    ]


def higgs3_files() -> list[dict[str, Any]]:
    """Return the Higgs files: the official weights and the codec."""
    return [
        {"repo": HIGGS_REPO, "files": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]},
        {"repo": HIGGS_CODEC_REPO, "files": ["config.json", "model.safetensors", "preprocessor_config.json"]},
    ]


def sa3_files() -> list[dict[str, Any]]:
    """Return the Stable Audio 3 files: the checkpoint, its model config and the T5Gemma text encoder."""
    return [
        {"repo": SA3_CHECKPOINT_REPO, "files": ["checkpoints/stable_audio_3_medium.safetensors"]},
        {"repo": SA3_CONFIG_REPO, "files": ["model_config.json"]},
        {"repo": SA3_TEXT_ENCODER_REPO, "files": ["t5gemma-b-b-ul2/*"]},
    ]


def yue2_preset() -> dict[str, Any]:
    """Return the ``yue2`` preset (music, lyrics, 360 s, the weight swap on CUDA only)."""
    return {
        "label": "YuE2 3B",
        "kind": "music",
        "family": "yue2",
        "files": yue2_files(),
        "limits": {"max_seconds": 360},
        "settings": {
            "instrumental_sections": INSTRUMENTAL_SECTIONS,
            "abc_max_tokens": 4096,
            "cfg": 1.0,
            "ode_steps": 32,
            "memory_budget_gib": 24,
            "vae_core_frames": 512,
            "fade_seconds": 2,
        },
        "platforms": {"cuda": {"memory_gb": 6.7}, "mps": {"memory_gb": 10.5}},
    }


def higgs3_preset() -> dict[str, Any]:
    """Return the ``higgs3`` preset (speech, voice cloning, parts of 500 characters)."""
    return {
        "label": "Higgs TTS 3 4B",
        "kind": "speech",
        "family": "higgs3",
        "files": higgs3_files(),
        "limits": {"max_new_tokens": 1500, "part_chars": 500, "max_text_chars": 10000, "min_voice_seconds": 3, "max_voice_seconds": 30},
        "settings": {"temperature": 0.7, "top_p": 0.95, "join_pause_ms": 300, "join_fade_ms": 10, "part_level_match_db": 3, "edge_silence_max_ms": 150},
        "platforms": {"cuda": {"memory_gb": 8.2}, "mps": {"memory_gb": 11.0}},
    }


def sa3_preset() -> dict[str, Any]:
    """Return the ``sa3`` preset (sound, a chosen length up to 380 s)."""
    return {
        "label": "Stable Audio 3 Medium",
        "kind": "sound",
        "family": "stable_audio_3",
        "files": sa3_files(),
        "limits": {"max_seconds": 380},
        "settings": {"steps": 8, "cfg_scale": 1.0, "sample_size": 16777216},
        "platforms": {"cuda": {"memory_gb": 6.1}, "mps": {"memory_gb": 6.1}},
    }


def audio_config(**presets: dict[str, Any]) -> dict[str, Any]:
    """Return a config with the three presets (``yue2``, ``higgs3``, ``sa3``), or with only the *presets* given by keyword."""
    return {"audio_model_presets": presets or {"yue2": yue2_preset(), "higgs3": higgs3_preset(), "sa3": sa3_preset()}}
