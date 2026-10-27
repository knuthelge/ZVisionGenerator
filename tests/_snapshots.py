"""Builders for fake model snapshots: diffusers-style folders and Hub-cache layouts."""

from __future__ import annotations

import json
from pathlib import Path

LTX25_WEIGHTED = {"audio_vae": 1, "connectors": 2, "text_encoder": 5, "transformer": 4, "vae": 1, "vocoder": 1}
H3_SHARDS = {"transformer": 3, "vae": 1, "transformer_ref": 2, "text_encoder": 5}


def touch(path: Path, content: str = "x") -> Path:
    """Write *content* to *path*, creating its folders, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def cache_snapshot_dir(cache: Path, repo_id: str = "owner/repo", commit: str = "c0ffee") -> Path:
    """Return the snapshot folder of *repo_id* at *commit* inside the Hub cache *cache* (not created)."""
    return cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / commit


def write_model_index(model_dir: Path, class_name: str = "ZImagePipeline") -> None:
    """Write a diffusers ``model_index.json`` with a transformer, text encoder, vae, scheduler and tokenizer."""
    index = {
        "_class_name": class_name,
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
        "text_encoder": ["transformers", "Qwen3Model"],
        "vae": ["diffusers", "AutoencoderKL"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "safety_checker": [None, None],
    }
    touch(model_dir / "model_index.json", json.dumps(index))


def write_ltx25_index(model_dir: Path, class_name: str = "LTX2Pipeline", text_encoder: str = "Gemma4UnifiedForConditionalGeneration") -> None:
    """Write a ``model_index.json`` shaped like the LTX-2.5 diffusers repo's."""
    index = {
        "_class_name": class_name,
        "audio_vae": ["diffusers", "AutoencoderKLLTX2Audio"],
        "connectors": ["ltx2", "LTX2TextConnectors"],
        "diffusion_decoder": ["diffusers", "LTX2VideoDiffusionDecoderModel"],
        "duration_head": ["ltx2", "LTX2DurationHead"],
        "processor": ["transformers", "Gemma4Processor"],
        "prompt_enhancer": ["transformers", "Gemma4ForConditionalGeneration"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", text_encoder],
        "tokenizer": ["transformers", "GemmaTokenizer"],
        "transformer": ["diffusers", "LTX2VideoTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLLTX2Video"],
        "vocoder": ["ltx2", "LTX2VocoderWithBWE"],
    }
    touch(model_dir / "model_index.json", json.dumps(index))


def _write_weights(model_dir: Path, shards_by_component: dict[str, int]) -> None:
    """Write a config and single-file or sharded weights for every component."""
    for component, shards in shards_by_component.items():
        touch(model_dir / component / "config.json")
        if shards == 1:
            touch(model_dir / component / "diffusion_pytorch_model.safetensors")
            continue
        for shard in range(1, shards + 1):
            touch(model_dir / component / f"model-{shard:05d}-of-{shards:05d}.safetensors")


def write_ltx25_snapshot(model_dir: Path) -> None:
    """Lay out a complete LTX-2.5 snapshot: sharded weights, config-only extras and an unlisted ``latent_upsampler``."""
    write_ltx25_index(model_dir)
    _write_weights(model_dir, LTX25_WEIGHTED)
    touch(model_dir / "latent_upsampler" / "diffusion_pytorch_model.safetensors")
    for component in ("duration_head", "diffusion_decoder", "scheduler", "tokenizer"):
        touch(model_dir / component / "config.json")


def write_h3_index(model_dir: Path, class_name: str = "MiniMaxH3ModularPipeline") -> None:
    """Write a modular-pipeline ``model_index.json`` whose entries carry a third (component spec) element."""
    spec = {"pretrained_model_name_or_path": "owner/h3"}
    index = {
        "_class_name": class_name,
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration", spec],
        "tokenizer": ["transformers", "Qwen2Tokenizer", spec],
        "transformer": ["diffusers", "MiniMaxH3Transformer", spec],
        "transformer_ref": ["diffusers", "MiniMaxH3Transformer", spec],
        "vae": ["diffusers", "AutoencoderKLMiniMaxH3", spec],
        "scheduler": ["diffusers", "MiniMaxH3Scheduler", spec],
    }
    touch(model_dir / "model_index.json", json.dumps(index))


def write_h3_snapshot(model_dir: Path) -> None:
    """Lay out a complete H3 snapshot: sharded transformer, single-file vae, and the components the app skips."""
    write_h3_index(model_dir)
    _write_weights(model_dir, H3_SHARDS)
    touch(model_dir / "scheduler" / "scheduler_config.json")
    touch(model_dir / "tokenizer" / "tokenizer.json")
