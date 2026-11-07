"""The audio presets as the config writes them, with their pinned sources, shared by tests in more than one area.

Import them with ``from tests._audio_presets import ...``. Each builder returns a fresh dict, so a test may change it freely.
"""

from __future__ import annotations

from typing import Any

YUE2_REPO = "m-a-p/YuE2-3B@8f24312f187763b9854adeec87e437e2b03bbff1"
YUE2_VAE_REPO = "m-a-p/YuE2-Vae@152733a19ad43aa67e367f9b5503ef8075bb5126"
QWEN_CUSTOM_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice@0c0e3051f131929182e2c023b9537f8b1c68adfe"
QWEN_BASE_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base@fd4b254389122332181a7c3db7f27e918eec64e3"
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


QWEN_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
]
QWEN_VOICES = [
    ("ryan", "Ryan", "English"),
    ("aiden", "Aiden", "English"),
    ("vivian", "Vivian", "Chinese"),
    ("serena", "Serena", "Chinese"),
    ("uncle_fu", "Uncle Fu", "Chinese"),
    ("dylan", "Dylan", "Beijing Chinese"),
    ("eric", "Eric", "Sichuan Chinese"),
    ("ono_anna", "Ono Anna", "Japanese"),
    ("sohee", "Sohee", "Korean"),
]


def qwen3_tts_preset() -> dict[str, Any]:
    """Return the ``qwen3_tts`` preset (speech: CustomVoice in ``files``, Base in ``voice_files``, nine built-in voices)."""
    return {
        "label": "Qwen3-TTS 1.7B",
        "kinds": ["speech"],
        "family": "qwen3_tts",
        "files": [{"repo": QWEN_CUSTOM_REPO, "files": list(QWEN_FILES)}],
        "voice_files": [{"repo": QWEN_BASE_REPO, "files": list(QWEN_FILES)}],
        "voices": [{"id": voice, "label": label, "language": language} for voice, label, language in QWEN_VOICES],
        "limits": {"max_new_tokens": 1200, "part_chars": 500, "max_text_chars": 10000, "min_voice_seconds": 3, "max_voice_seconds": 30},
        "settings": {"language": "auto", "join_pause_ms": 300, "join_fade_ms": 10, "part_level_match_db": 3, "edge_silence_max_ms": 150},
        "platforms": {"cuda": {"memory_gb": 6.0}, "mps": {"family": "qwen3_tts_mlx", "memory_gb": 12.0}},
    }


def sa3_files() -> list[dict[str, Any]]:
    """Return the Stable Audio 3 files: the checkpoint, its model config and the T5Gemma text encoder."""
    return [
        {"repo": SA3_CHECKPOINT_REPO, "files": ["checkpoints/stable_audio_3_medium.safetensors"]},
        {"repo": SA3_CONFIG_REPO, "files": ["model_config.json"]},
        {
            "repo": SA3_TEXT_ENCODER_REPO,
            "files": [
                "t5gemma-b-b-ul2/config.json",
                "t5gemma-b-b-ul2/generation_config.json",
                "t5gemma-b-b-ul2/model.safetensors",
                "t5gemma-b-b-ul2/special_tokens_map.json",
                "t5gemma-b-b-ul2/tokenizer.json",
                "t5gemma-b-b-ul2/tokenizer_config.json",
            ],
        },
    ]


def yue2_preset() -> dict[str, Any]:
    """Return the ``yue2`` preset (music, lyrics, 360 s, the weight swap on CUDA only)."""
    return {
        "label": "YuE2 3B",
        "kinds": ["music"],
        "lyrics": True,
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


def sa3_preset() -> dict[str, Any]:
    """Return the ``sa3`` preset (sound, a chosen length up to 380 s)."""
    return {
        "label": "Stable Audio 3 Medium",
        "kinds": ["sound"],
        "family": "stable_audio_3",
        "files": sa3_files(),
        "limits": {"max_seconds": 380},
        "settings": {"steps": 8, "cfg_scale": 1.0, "sample_size": 16777216},
        "platforms": {"cuda": {"memory_gb": 6.1}, "mps": {"memory_gb": 6.1}},
    }


def sa3_two_kind_preset() -> dict[str, Any]:
    """Return ``sa3`` as a model of two kinds: sound first (the default), then music without lyrics, whose length is chosen."""
    return {**sa3_preset(), "kinds": ["sound", "music"]}


def audio_config(**presets: dict[str, Any]) -> dict[str, Any]:
    """Return a config with the three presets (``yue2``, ``qwen3_tts``, ``sa3``), or with only the *presets* given by keyword."""
    return {"audio_model_presets": presets or {"yue2": yue2_preset(), "qwen3_tts": qwen3_tts_preset(), "sa3": sa3_preset()}}
