"""Resolve the on-disk files of the MiniMax H3 model for the CUDA backend, downloading what is missing."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.h3_preset import H3Files, H3Preset
from zvisiongenerator.utils.hf_snapshot import snapshot_download as hub_snapshot_download
from zvisiongenerator.utils.hf_snapshot import snapshot_with
from zvisiongenerator.utils.paths import split_huggingface_revision

__all__ = ["base_repo_allow", "resolve_base_files", "resolve_files", "transformer_files"]

_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
_TRANSFORMER_DIR = "transformer"
_MODEL_INDEX = "model_index.json"


def _repo_dir(model_ref: str, preset: H3Preset, download: Callable[..., str]) -> Path:
    """Return the folder of the H3 repository *model_ref*: a local folder as it is, a Hub reference downloaded with the preset's filters.

    A reference without a revision takes the preset's when it names the preset's repository.

    Raises:
        FileNotFoundError: If a local folder holds no ``model_index.json``.
        RuntimeError: If the download fails.
    """
    local = Path(model_ref).expanduser()
    if local.is_dir():
        if not (local / _MODEL_INDEX).is_file():
            raise FileNotFoundError(f"{local} holds no {_MODEL_INDEX}; point at a MiniMax H3 folder or repository.")
        return local
    repo_id, revision = split_huggingface_revision(model_ref)
    if revision is None and repo_id == preset.repo:
        revision = preset.revision
    try:
        return Path(download(repo_id=repo_id, revision=revision, allow_patterns=list(preset.repo_allow), ignore_patterns=list(preset.repo_ignore)))
    except Exception as exc:
        raise RuntimeError(f"Failed to download the MiniMax H3 model '{model_ref}': {exc}") from exc


def resolve_files(model_ref: str, preset: H3Preset, *, upscale: bool = False, lora: bool = True, snapshot_download: Callable[..., str] | None = None) -> H3Files:
    """Return the on-disk files of the CUDA H3 model, downloading what is missing.

    The repository, the text encoder and the turbo LoRA are each fetched as one snapshot with just their files. With
    *upscale*, the latent upscaler's file is fetched the same way.

    Args:
        model_ref: A local H3 folder or a Hub reference (``owner/repo`` or ``owner/repo@revision``).
        preset: The CUDA preset: the repository's file filters, the text encoder's files, the turbo LoRA and the latent upscaler.
        upscale: Whether to resolve the latent upscaler's file too; without it the field is ``None`` and nothing more is fetched.
        lora: Whether to resolve the turbo LoRA; without it the field is ``None`` (a converted folder holds its merge).
        snapshot_download: Called as ``snapshot_download(repo_id=, revision=, allow_patterns=[, ignore_patterns=])``;
            defaults to the Hub's.

    Raises:
        FileNotFoundError: If a local folder holds no ``model_index.json``, or the text encoder, LoRA or latent upscaler file is missing.
        RuntimeError: If the repository download fails.
        ValueError: If the preset's LoRA is not exactly one file.
    """
    download = hub_snapshot_download if snapshot_download is None else snapshot_download
    if len(preset.lora.files) != 1:
        raise ValueError(f"The turbo LoRA must be one file, got {len(preset.lora.files)}.")
    repo_dir = _repo_dir(model_ref, preset, download)
    text_encoder_dir = snapshot_with(preset.text_encoder, download)
    shards = tuple(text_encoder_dir / name for name in preset.text_encoder.files if name.endswith(".safetensors"))
    for path in (text_encoder_dir / "config.json", *shards):
        if not path.is_file():
            raise FileNotFoundError(f"Text encoder file not found: {path}")
    lora_file = snapshot_with(preset.lora, download) / preset.lora.files[0] if lora else None
    if lora_file is not None and not lora_file.is_file():
        raise FileNotFoundError(f"Turbo LoRA not found: {lora_file}")
    upsampler_file = None
    if upscale:
        upsampler_file = snapshot_with(preset.latent_upsampler, download) / preset.latent_upsampler.files[0]
        if not upsampler_file.is_file():
            raise FileNotFoundError(f"Latent upsampler not found: {upsampler_file}")
    return H3Files(repo_dir=repo_dir, text_encoder_dir=text_encoder_dir, text_encoder_shards=shards, lora_file=lora_file, latent_upsampler_file=upsampler_file)


def base_repo_allow(preset: H3Preset) -> tuple[str, ...]:
    """Return what a converted H3 folder downloads of the stock repository: the preset's files without the transformer's weights (its config is kept)."""
    return (*(pattern for pattern in preset.repo_allow if not pattern.startswith(f"{_TRANSFORMER_DIR}/")), f"{_TRANSFORMER_DIR}/config.json")


def resolve_base_files(preset: H3Preset, *, upscale: bool = False, lora: bool = False, snapshot_download: Callable[..., str] | None = None) -> H3Files:
    """Return the files a converted H3 folder takes from the stock repository (see :func:`base_repo_allow`), and the turbo LoRA only with *lora* (to merge it).

    See :func:`resolve_files` for the arguments and errors.
    """
    return resolve_files(f"{preset.repo}@{preset.revision}", replace(preset, repo_allow=base_repo_allow(preset)), upscale=upscale, lora=lora, snapshot_download=snapshot_download)


def transformer_files(repo_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    """Return the transformer's safetensors files and its config for the stored copy.

    The files are those the folder's index lists, else every ``*.safetensors`` in it.

    Raises:
        FileNotFoundError: If the folder has no ``config.json`` or no weights.
    """
    folder = repo_dir / _TRANSFORMER_DIR
    config_path = folder / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} not found; the MiniMax H3 download is incomplete.")
    index = folder / _INDEX_NAME
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = [folder / name for name in sorted(set(weight_map.values()))]
    else:
        shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"{folder} holds no transformer weights; the MiniMax H3 download is incomplete.")
    return shards, json.loads(config_path.read_text(encoding="utf-8"))
