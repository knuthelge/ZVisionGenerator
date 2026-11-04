"""Read the official Higgs TTS 3 ``config.json`` into the few numbers our model class needs.

The file is plain JSON, pinned to a commit in ``config.yaml``. It is read without an Auto class, ``auto_map`` or
``PretrainedConfig``, so no remote code can be named by it. The mapping to transformers' ``Qwen3Config`` lives in
``higgs3_model``.

This module imports neither torch nor transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Higgs3Config", "read_higgs3_config"]


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


def read_higgs3_config(path: Path) -> Higgs3Config:
    """Return the network sizes held by the pinned official ``config.json`` at *path*."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    text = raw["text_config"]
    audio = raw["audio_encoder_config"]
    return Higgs3Config(
        hidden_size=text["hidden_size"],
        intermediate_size=text["intermediate_size"],
        num_layers=text["num_hidden_layers"],
        num_heads=text["num_attention_heads"],
        num_kv_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        rms_norm_eps=float(text["rms_norm_eps"]),
        rope_theta=float(text["rope_parameters"]["rope_theta"]),
        text_vocab_size=text["vocab_size"],
        max_positions=text["max_position_embeddings"],
        num_codebooks=audio["num_codebooks"],
        codebook_size=audio["vocab_size"],
    )
