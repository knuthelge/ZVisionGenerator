"""Tests for reading the official Higgs TTS 3 config: the values it yields, what it refuses, and that it stays torch-free."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from zvisiongenerator.backends.audio.higgs3_config import Higgs3Config, read_higgs3_config

OFFICIAL: dict[str, Any] = {
    "model_type": "higgs_multimodal_qwen3",
    "text_config": {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-06,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "hidden_act": "silu",
        "attention_bias": False,
        "use_sliding_window": False,
        "layer_types": ["full_attention"] * 36,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "auto_map": {"AutoConfig": "somewhere.Config"},  # keys the class does not read are ignored
    },
    "audio_encoder_config": {
        "model_type": "higgs_audio_encoder",
        "encoder_type": "discrete",
        "num_codebooks": 8,
        "vocab_size": 1026,
        "use_delay_pattern": True,
        "tie_word_embeddings": True,
        "out_dim": 2560,
    },
}


def _write(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _changed(*keys_and_value: Any) -> dict[str, Any]:
    """Return a copy of the official layout with the value at the key path (all but the last argument) replaced; ``...`` deletes the key."""
    *keys, value = keys_and_value
    config = copy.deepcopy(OFFICIAL)
    parent = config
    for key in keys[:-1]:
        parent = parent[key]
    if value is ...:
        del parent[keys[-1]]
    else:
        parent[keys[-1]] = value
    return config


def test_the_official_layout_reads_to_the_expected_values(tmp_path: Path) -> None:
    config = read_higgs3_config(_write(tmp_path, OFFICIAL))

    assert config == Higgs3Config(
        hidden_size=2560,
        intermediate_size=9728,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        rms_norm_eps=1e-06,
        rope_theta=1_000_000.0,
        text_vocab_size=151936,
        max_positions=32768,
        num_codebooks=8,
        codebook_size=1026,
    )
    assert (config.boc_id, config.eoc_id) == (1024, 1025)


def test_the_config_is_a_frozen_value(tmp_path: Path) -> None:
    config = read_higgs3_config(_write(tmp_path, OFFICIAL))

    with pytest.raises(AttributeError):
        config.hidden_size = 1  # type: ignore[misc]


def test_a_missing_layer_types_list_counts_as_full_attention(tmp_path: Path) -> None:
    config = _changed("text_config", "layer_types", ...)

    assert read_higgs3_config(_write(tmp_path, config)).num_layers == 36


@pytest.mark.parametrize(
    ("config", "key"),
    [
        (_changed("model_type", "qwen3"), "model_type"),
        (_changed("text_config", "model_type", "llama"), "text_config.model_type"),
        (_changed("text_config", "rope_parameters", "rope_type", "yarn"), "rope_type"),
        (_changed("text_config", "rope_parameters", "rope_theta", ...), "rope_theta"),
        (_changed("text_config", "rope_parameters", ...), "rope_parameters"),
        (_changed("text_config", "hidden_act", "gelu"), "hidden_act"),
        (_changed("text_config", "attention_bias", True), "attention_bias"),
        (_changed("text_config", "layer_types", ["full_attention"] * 35 + ["sliding_attention"]), "layer_types"),
        (_changed("text_config", "use_sliding_window", True), "use_sliding_window"),
        (_changed("audio_encoder_config", "encoder_type", "whisper"), "encoder_type"),
        (_changed("audio_encoder_config", "use_delay_pattern", False), "use_delay_pattern"),
        (_changed("audio_encoder_config", "tie_word_embeddings", False), "tie_word_embeddings"),
        (_changed("audio_encoder_config", "out_dim", 1024), "out_dim"),
        (_changed("text_config", "hidden_size", ...), "hidden_size"),
        (_changed("audio_encoder_config", "num_codebooks", ...), "num_codebooks"),
        (_changed("audio_encoder_config", ...), "audio_encoder_config"),
        (_changed("audio_encoder_config", "num_codebooks", True), "num_codebooks"),
        (_changed("text_config", "num_hidden_layers", "36"), "num_hidden_layers"),
    ],
)
def test_a_setting_the_class_does_not_implement_is_refused_naming_its_key(tmp_path: Path, config: dict[str, Any], key: str) -> None:
    with pytest.raises(ValueError, match=key):
        read_higgs3_config(_write(tmp_path, config))


def test_a_file_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        read_higgs3_config(path)


def test_importing_the_module_leaves_torch_and_transformers_out() -> None:
    code = "import sys; import zvisiongenerator.backends.audio.higgs3_config; print(sorted(m for m in ('torch', 'transformers') if m in sys.modules))"
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[]"
