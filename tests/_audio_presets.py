"""The audio presets as the config shipped them (facts and settings in one block), shared by tests in more than one area.

Import them with ``from tests._audio_presets import ...``. Each builder returns a fresh dict, so a test may change it freely.

The facts of an audio model (name, kinds, limits, voices, measured memory, pinned files) are packaged metadata and its settings
are config. :func:`audio_config` gives the settings of a preset as the config holds them, and :func:`use_audio_models` serves
the facts of a preset through the metadata, for a test that needs facts other than the packaged ones.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._factories import metadata_copy, use_model_metadata
from zvisiongenerator.utils.audio_preset import AudioPreset, parse_audio_presets

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


def _settings_block(preset: dict[str, Any]) -> dict[str, Any]:
    """Return what the config holds of *preset*: its ``settings`` and the ``settings`` it overrides per platform."""
    block: dict[str, Any] = {"settings": preset.get("settings", {})}
    platforms = {key: {"settings": value["settings"]} for key, value in preset.get("platforms", {}).items() if "settings" in value}
    if platforms:
        block["platforms"] = platforms
    return block


def audio_config(**presets: dict[str, Any]) -> dict[str, Any]:
    """Return a config with the settings of the three audio models (``yue2``, ``qwen3_tts``, ``sa3``); a preset given by keyword replaces its model's settings.

    Only the settings of a preset are config. A test that needs other facts than the packaged ones serves them with :func:`use_audio_models`.
    """
    shipped = {"yue2": yue2_preset(), "qwen3_tts": qwen3_tts_preset(), "sa3": sa3_two_kind_preset()}
    shipped.update(presets)
    return {"audio_model_presets": {model_id: _settings_block(preset) for model_id, preset in shipped.items()}}


def audio_entry(preset: dict[str, Any], *, licences: list[str] | None = None) -> dict[str, Any]:
    """Return the metadata ``models`` entry that declares the facts of the config-shaped *preset*."""
    entry: dict[str, Any] = {
        "kind": "audio",
        "family": preset["family"],
        "name": preset["label"],
        "kinds": list(preset["kinds"]),
        "limits": dict(preset["limits"]),
        "licences": list(licences or []),
        "downloads": [{"ref": item["repo"], "files": list(item["files"])} for item in preset["files"]]
        + [{"ref": item["repo"], "files": list(item["files"]), "role": "voice"} for item in preset.get("voice_files", ())],
    }
    if preset.get("lyrics"):
        entry["lyrics"] = True
    if "voices" in preset:
        entry["voices"] = [dict(voice) for voice in preset["voices"]]
    platforms = {key: {name: value[name] for name in ("memory_gb", "family") if name in value} for key, value in preset.get("platforms", {}).items()}
    if any(platforms.values()):
        entry["platforms"] = {key: value for key, value in platforms.items() if value}
    return entry


def audio_entries(**presets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the metadata entries (by model id) of the config-shaped *presets*; without any, of the three fixture presets (``sa3`` making sound only)."""
    presets = presets or {"yue2": yue2_preset(), "qwen3_tts": qwen3_tts_preset(), "sa3": sa3_preset()}
    return {model_id: audio_entry(preset) for model_id, preset in presets.items()}


def no_audio_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Serve metadata without any audio model; the other models stay as they are."""
    data = metadata_copy()
    data["models"] = {model_id: entry for model_id, entry in data["models"].items() if entry["kind"] != "audio"}
    use_model_metadata(monkeypatch, tmp_path, data)


def use_audio_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, **presets: dict[str, Any]) -> None:
    """Serve metadata whose audio models are the config-shaped *presets* (by model id), and no other audio model; the other models stay as packaged.

    Without *presets* the models are the three fixture presets (``yue2``, ``qwen3_tts`` and ``sa3`` making sound only). Each
    preset keeps the licences of the model of the same id. It edits the metadata a test already serves, so it stacks with the
    other ``use_*`` helpers.
    """
    presets = presets or {"yue2": yue2_preset(), "qwen3_tts": qwen3_tts_preset(), "sa3": sa3_preset()}
    data = metadata_copy()
    packaged = data["models"]
    data["models"] = {
        **{model_id: entry for model_id, entry in packaged.items() if entry["kind"] != "audio"},
        **{model_id: audio_entry(preset, licences=packaged.get(model_id, {}).get("licences")) for model_id, preset in presets.items()},
    }
    use_model_metadata(monkeypatch, tmp_path, data)


def audio_preset(model_id: str, config: dict[str, Any] | None = None) -> AudioPreset:
    """Return the parsed preset of the audio model *model_id*, with the settings of *config* (:func:`audio_config` by default)."""
    return next(preset for preset in parse_audio_presets(audio_config() if config is None else config) if preset.id == model_id)
