"""Read the official Higgs TTS 3 ``config.json`` into the few numbers our model class needs, and refuse the rest.

The file is plain JSON. It is read without an Auto class, ``auto_map`` or ``PretrainedConfig``, so no remote code can
be named by it. A setting this model class does not implement is refused with the key that holds it, so a different
checkpoint cannot load into the wrong network. The mapping to transformers' ``Qwen3Config`` lives in ``higgs3_model``.

This module imports neither torch nor transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Higgs3Config", "read_higgs3_config"]

_MODEL_TYPE = "higgs_multimodal_qwen3"
_TEXT_MODEL_TYPE = "qwen3"
_FULL_ATTENTION = "full_attention"


@dataclass(frozen=True)
class Higgs3Config:
    """The sizes of the Higgs TTS 3 network and its audio token layout."""

    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    text_vocab_size: int
    max_positions: int
    num_codebooks: int
    codebook_size: int
    """Values per codebook the model predicts: the codec's values plus the two stream markers."""

    @property
    def boc_id(self) -> int:
        """Return the id that begins a codebook's stream."""
        return self.codebook_size - 2

    @property
    def eoc_id(self) -> int:
        """Return the id that ends a codebook's stream."""
        return self.codebook_size - 1


def read_higgs3_config(path: Path) -> Higgs3Config:
    """Return the network sizes held by the official ``config.json`` at *path*.

    Args:
        path: The ``config.json`` of the Higgs TTS 3 repository.

    Raises:
        ValueError: If a key is missing, or holds a setting this model class does not implement. The message names the key.
        FileNotFoundError: If *path* does not exist.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a JSON object.")
    _expect(raw, "model_type", _MODEL_TYPE, "")
    text = _section(raw, "text_config", "")
    audio = _section(raw, "audio_encoder_config", "")
    _check_text(text)
    hidden_size = _integer(text, "hidden_size", "text_config.")
    _check_audio(audio, hidden_size)
    rope = _section(text, "rope_parameters", "text_config.")
    return Higgs3Config(
        hidden_size=hidden_size,
        intermediate_size=_integer(text, "intermediate_size", "text_config."),
        num_layers=_integer(text, "num_hidden_layers", "text_config."),
        num_heads=_integer(text, "num_attention_heads", "text_config."),
        num_kv_heads=_integer(text, "num_key_value_heads", "text_config."),
        head_dim=_integer(text, "head_dim", "text_config."),
        rms_norm_eps=float(_need(text, "rms_norm_eps", "text_config.")),
        rope_theta=float(_need(rope, "rope_theta", "text_config.rope_parameters.")),
        text_vocab_size=_integer(text, "vocab_size", "text_config."),
        max_positions=_integer(text, "max_position_embeddings", "text_config."),
        num_codebooks=_integer(audio, "num_codebooks", "audio_encoder_config."),
        codebook_size=_integer(audio, "vocab_size", "audio_encoder_config."),
    )


def _check_text(text: dict[str, Any]) -> None:
    """Refuse a text backbone that is not the plain full-attention Qwen3 this model class implements."""
    prefix = "text_config."
    _expect(text, "model_type", _TEXT_MODEL_TYPE, prefix)
    _expect(text, "hidden_act", "silu", prefix)
    if text.get("attention_bias"):
        raise ValueError(f"{prefix}attention_bias is true, which this model class does not implement.")
    if text.get("use_sliding_window"):
        raise ValueError(f"{prefix}use_sliding_window is true, which this model class does not implement.")
    kinds = {kind for kind in text.get("layer_types") or [_FULL_ATTENTION] if kind != _FULL_ATTENTION}
    if kinds:
        raise ValueError(f"{prefix}layer_types holds {sorted(kinds)}; only {_FULL_ATTENTION} is implemented.")
    rope = _section(text, "rope_parameters", prefix)
    _expect(rope, "rope_type", "default", f"{prefix}rope_parameters.")


def _check_audio(audio: dict[str, Any], hidden_size: int) -> None:
    """Refuse an audio side that is not discrete, delayed, tied codebooks of the backbone's width."""
    prefix = "audio_encoder_config."
    _expect(audio, "encoder_type", "discrete", prefix)
    if audio.get("use_delay_pattern") is not True:
        raise ValueError(f"{prefix}use_delay_pattern is not true, which this model class does not implement.")
    if audio.get("tie_word_embeddings") is not True:
        raise ValueError(f"{prefix}tie_word_embeddings is not true, which this model class does not implement.")
    if audio.get("out_dim") != hidden_size:
        raise ValueError(f"{prefix}out_dim is {audio.get('out_dim')!r}, but text_config.hidden_size is {hidden_size}.")


def _section(parent: dict[str, Any], key: str, prefix: str) -> dict[str, Any]:
    """Return the JSON object at *key* of *parent*, or raise naming ``prefix + key``."""
    value = _need(parent, key, prefix)
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}{key} is not a JSON object.")
    return value


def _need(parent: dict[str, Any], key: str, prefix: str) -> Any:
    """Return *parent*'s value for *key*, or raise naming ``prefix + key`` when it is missing or null."""
    value = parent.get(key)
    if value is None:
        raise ValueError(f"{prefix}{key} is missing.")
    return value


def _integer(parent: dict[str, Any], key: str, prefix: str) -> int:
    """Return *parent*'s whole-number value for *key*, or raise naming ``prefix + key``."""
    value = _need(parent, key, prefix)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{prefix}{key} is {value!r}, not a whole number.")
    return value


def _expect(parent: dict[str, Any], key: str, expected: str, prefix: str) -> None:
    """Raise naming ``prefix + key`` unless *parent*'s value for it is *expected*."""
    value = parent.get(key)
    if value != expected:
        raise ValueError(f"{prefix}{key} is {value!r}; this model class implements only {expected!r}.")
