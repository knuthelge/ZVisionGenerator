"""Shared helpers for the backend tests: a tiny Higgs TTS 3 config and a checkpoint in the official naming. Never a real model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import save_file

from zvisiongenerator.backends.audio.higgs3_config import Higgs3Config

if TYPE_CHECKING:
    from zvisiongenerator.backends.audio.higgs3_model import Higgs3Model

TINY = Higgs3Config(
    hidden_size=32,
    intermediate_size=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=8,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    text_vocab_size=64,
    max_positions=128,
    num_codebooks=3,
    codebook_size=6,
)


def checkpoint_tensors(config: Higgs3Config = TINY, *, seed: int = 0) -> dict[str, torch.Tensor]:
    """Return random bf16 tensors named as the official checkpoint names them, with the unused ones too (both head copies, a codec tensor)."""
    generator = torch.Generator().manual_seed(seed)
    d, h, kv, hd = config.hidden_size, config.num_heads, config.num_kv_heads, config.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "body.norm.weight": (d,),
        "tied.embedding.text_embedding.weight": (config.text_vocab_size, d),
        "tied.embedding.modality_embeddings.0.embedding.weight": (config.num_codebooks * config.codebook_size, d),
        "tied.head.modality_heads.0.weight": (config.num_codebooks * config.codebook_size, d),
        "tied.head.text_head.weight": (config.text_vocab_size, d),
        "tied.embedding.modality_embeddings.0.model.encoder.conv.weight": (5,),
    }
    for i in range(config.num_layers):
        prefix = f"body.layers.{i}"
        shapes |= {
            f"{prefix}.self_attn.q_proj.weight": (h * hd, d),
            f"{prefix}.self_attn.k_proj.weight": (kv * hd, d),
            f"{prefix}.self_attn.v_proj.weight": (kv * hd, d),
            f"{prefix}.self_attn.o_proj.weight": (d, h * hd),
            f"{prefix}.self_attn.q_norm.weight": (hd,),
            f"{prefix}.self_attn.k_norm.weight": (hd,),
            f"{prefix}.mlp.gate_proj.weight": (config.intermediate_size, d),
            f"{prefix}.mlp.up_proj.weight": (config.intermediate_size, d),
            f"{prefix}.mlp.down_proj.weight": (d, config.intermediate_size),
            f"{prefix}.input_layernorm.weight": (d,),
            f"{prefix}.post_attention_layernorm.weight": (d,),
        }
    return {name: (torch.randn(shape, generator=generator) * 0.2).to(torch.bfloat16) for name, shape in shapes.items()}


def write_checkpoint(path: Path, tensors: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """Write *tensors* (default: :func:`checkpoint_tensors`) to the safetensors file *path* and return them."""
    tensors = checkpoint_tensors() if tensors is None else tensors
    save_file(tensors, str(path))
    return tensors


def tiny_random_model(seed: int = 0) -> Higgs3Model:
    """Return a random-init ``Higgs3Model`` of the :data:`TINY` config in fp32 on the CPU, with the audio head tied to the audio embedding."""
    from zvisiongenerator.backends.audio.higgs3_model import Higgs3Model

    torch.manual_seed(seed)
    model = Higgs3Model(TINY)
    torch.nn.init.normal_(model.audio_embedding.weight, std=0.5)
    model.audio_head.weight = model.audio_embedding.weight
    return model.eval().requires_grad_(False)
