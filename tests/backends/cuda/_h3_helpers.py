"""Shared builders for the H3 tests: a tiny transformer and layout, tiny VAE configs, and LoRA / VAE file writers.

Import this module only after ``pytest.importorskip("torch")``, ``"diffusers"`` and ``"safetensors"`` have run.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

TEXT_TAG, AUDIO_TAG, VIDEO_TAG = 1, 2, 0
PATCH = (1, 2, 2)
HIDDEN = 64
LAYERS = 2
TIME_EMBED_DIM = 32

VAE_CONFIG = {
    "latent_channels": 4,
    "block_out_channels": (4, 4, 4, 4, 4, 4),
    "layers_per_block": 1,
    "norm_num_groups": 2,
    "decoder_num_layers": 2,
    "decoder_num_attention_heads": 2,
    "decoder_attention_head_dim": 8,
    "decoder_num_register_tokens": 1,
    "decoder_ffn_mult": 2,
    "latents_mean": (0.1, 0.2, -0.1, 0.3),
    "latents_std": (1.0, 2.0, 0.5, 1.5),
}
AUDIO_CONFIG = {
    "encoder_dim": 4,
    "encoder_rates": (2, 2),
    "latent_dim": 8,
    "latent_channels": 4,
    "num_attention_heads": 2,
    "decoder_dim": 8,
    "decoder_rates": (2, 2),
    "decoder_kernel_sizes": (4, 4),
    "resblock_kernel_sizes": (3,),
    "resblock_dilation_sizes": ((1,),),
    "latents_mean": [0.1, -0.2, 0.3, 0.0],
    "latents_std": [2.0, 1.0, 0.5, 1.5],
}


def transformer_config() -> dict:
    """Return the constructor arguments of the tiny H3 transformer (64 hidden, 2 blocks)."""
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 32,
        "hidden_size": HIDDEN,
        "num_layers": LAYERS,
        "num_refiner_layers": 1,
        "ffn_dim": 2 * HIDDEN,
        "in_channels": 24,
        "audio_in_channels": 32,
        "patch_size": PATCH,
        "text_dim": 48,
        "freq_dim": 32,
        "time_embed_hidden_dim": HIDDEN,
        "time_embed_dim": TIME_EMBED_DIM,
        "rope_freq_dim": 4,
    }


def tiny_transformer(seed: int = 0):
    """Return a random-init transformer with the checkpoint's dtype mix: bf16 blocks, fp32 timestep MLP, patch projections and heads."""
    from diffusers import MiniMaxH3Transformer3DModel

    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(**transformer_config())
    keep = set(model._keep_in_fp32_modules)
    for name, module in model.named_modules():
        if not any(part in keep for part in name.split(".")):
            for parameter in module.parameters(recurse=False):
                parameter.data = parameter.data.to(torch.bfloat16)
    return model.eval()


def tiny_layout() -> SimpleNamespace:
    """Return a tiny packed layout: 4 text rows, 6 audio rows and 4 video rows, with the structural tensors the forward takes."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep

    text_tokens = 4
    tags = torch.full((text_tokens,), TEXT_TAG, dtype=torch.long)
    position_ids, token_tags, video_indices, audio_indices, text_indices, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(tags, 2, 2, 4, 3, PATCH, 2, AUDIO_TAG, VIDEO_TAG, ())
    return SimpleNamespace(
        text_tokens=text_tokens,
        video_indices=video_indices,
        audio_indices=audio_indices,
        kwargs={"position_ids": position_ids, "token_tags": token_tags, "video_indices": video_indices, "audio_indices": audio_indices, "text_indices": text_indices},
    )


def layout_inputs(layout, seed: int = 1) -> dict:
    """Return random hidden states for *layout*, shaped for the tiny transformer."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "hidden_states": torch.randn(1, len(layout.video_indices), 96, generator=generator),
        "audio_hidden_states": torch.randn(1, len(layout.audio_indices), 32, generator=generator),
        "encoder_hidden_states": torch.randn(1, layout.text_tokens, 48, generator=generator),
    }


def lora_factors(shapes: dict[str, tuple[int, int]], rank: int, seed: int = 0) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return random ``(down, up)`` factors per module, given each module's ``(output rows, input columns)``."""
    generator = torch.Generator().manual_seed(seed)
    return {module: (torch.randn(rank, columns, generator=generator), torch.randn(rows, rank, generator=generator)) for module, (rows, columns) in shapes.items()}


def write_lora(path: Path, factors: dict[str, tuple[torch.Tensor, torch.Tensor]], metadata: dict[str, str] | None = None) -> Path:
    """Write an ai-toolkit style LoRA file (checkpoint module names under ``diffusion_model.``)."""
    tensors = {}
    for module, (down, up) in factors.items():
        tensors[f"diffusion_model.{module}.lora_A.weight"] = down.contiguous()
        tensors[f"diffusion_model.{module}.lora_B.weight"] = up.contiguous()
    save_file(tensors, str(path), metadata=metadata)
    return path


def write_vae_dir(directory: Path, vae, *, keep, shards: bool = False) -> None:
    """Save *vae*'s config and the weights whose name *keep* accepts, as one file or two shards, to *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    config = {"_class_name": "AutoencoderKLMiniMaxH3", **{key: list(value) if isinstance(value, tuple) else value for key, value in VAE_CONFIG.items()}}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    state = {name: tensor.contiguous() for name, tensor in vae.state_dict().items() if keep(name)}
    if not shards:
        save_file(state, str(directory / "diffusion_pytorch_model.safetensors"))
        return
    names = sorted(state)
    halves = {"model-00001-of-00002.safetensors": names[: len(names) // 2], "model-00002-of-00002.safetensors": names[len(names) // 2 :]}
    for file, part in halves.items():
        save_file({name: state[name] for name in part}, str(directory / file))
    weight_map = {name: file for file, part in halves.items() for name in part}
    (directory / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
