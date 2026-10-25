"""Converted LTX-2.3 and LTX-2.5 folders on Windows/Linux: write one from a single-file checkpoint.

A converted folder is a diffusers LTX-2 folder that holds its own quantized weights:

- ``transformer/``: the checkpoint's transformer in NF4 (q4) or FP8 (q8), in the prepared-copy layout.
- ``text_encoder/``: the shared FP8 text encoder (a prepared copy with its own manifest, hard-linked from another
  one when it exists) plus its ``config.json``.
- the diffusers repo's index and other components (connectors, VAEs, vocoder, tokenizer, scheduler), hard-linked.
- the folder manifest (:mod:`zvisiongenerator.utils.ltx_converted`), written last.

It loads without the source checkpoint.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda.video.weights import (
    count_transformer_elements,
    ensure_prepared,
    prepared_text_encoders,
    text_encoder_sources,
    transformer_name_map,
    write_text_encoder,
    write_transformer_tensors,
)
from zvisiongenerator.converters.ltx_checkpoint import connectors_status, transformer_tensors
from zvisiongenerator.utils.file_links import link_tree
from zvisiongenerator.utils.ltx_converted import write_converted_ltx

if TYPE_CHECKING:
    from zvisiongenerator.converters.ltx_checkpoint import LtxCheckpoint

__all__ = ["convert", "write_converted"]

# Components written here rather than linked from the diffusers repo.
_OWN_COMPONENTS = ("transformer", "text_encoder")
_STORAGE = {4: "nf4", 8: "fp8-e4m3"}

Report = Callable[[str, int, int], None]


def convert(checkpoint: LtxCheckpoint, target: Path, bits: int, *, config: dict[str, Any], version: str | None, models_dir: Path, lora: Path | None, report: Report) -> None:
    """Write the converted folder of *checkpoint* at *bits* into *target* on this machine's GPU, with the components of LTX version *version*'s diffusers repo.

    Raises:
        RuntimeError: When there is no usable NVIDIA GPU, or the repo can't be downloaded.
    """
    import torch

    from zvisiongenerator.backends.cuda.driver import require_cuda
    from zvisiongenerator.backends.cuda.video.backend import _repo_dir
    from zvisiongenerator.utils.config import ltx_diffusers_repo, ltx_repo_ignore

    require_cuda(torch, "Converting an LTX checkpoint on Windows and Linux")
    report("download", 0, 1)
    repo_dir = _repo_dir(ltx_diffusers_repo(config, version), with_transformer=False, extra_ignore=ltx_repo_ignore(config, version))
    write_converted(checkpoint, target, bits, repo_dir=repo_dir, models_dir=models_dir, lora=lora, device=torch.device("cuda"), report=report)


def write_converted(
    checkpoint: LtxCheckpoint,
    target: Path,
    bits: int,
    *,
    repo_dir: Path,
    models_dir: Path,
    lora: Path | None,
    device: Any,
    report: Report,
) -> None:
    """Write the converted folder of *checkpoint* at *bits* into the new folder *target*.

    Args:
        checkpoint: The checked checkpoint.
        target: The folder to write (a partial folder; the caller moves it into place).
        bits: 4 (NF4) or 8 (FP8).
        repo_dir: The diffusers LTX repo of the checkpoint's version the other components come from (its transformer weights are not needed).
        models_dir: The models folder, searched for a text-encoder copy to reuse.
        lora: Lightricks' distilled LoRA to merge in, or ``None``.
        device: The CUDA device NF4 weights are packed on (and the LoRA merged on).
        report: Called with ``(stage, done, total)`` as the conversion advances.

    Raises:
        ValueError: When the checkpoint's text connectors differ from stock's (only stock ones are supported here).
        RuntimeError: When the checkpoint does not cover the transformer.
    """
    report("connectors", 0, 1)
    connectors = connectors_status(checkpoint)
    if connectors == "changed":
        raise ValueError(f"{checkpoint.path.name} changes the text connectors; on Windows and Linux only checkpoints with stock connectors can be converted.")
    if connectors == "unknown":
        print("Note: the checkpoint's text connectors are quantized, so whether they were fine-tuned can't be told; the stock connectors are used.")
    config = json.loads((repo_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    names = transformer_name_map(checkpoint.transformer)
    tensors = transformer_tensors(checkpoint, lora=lora, merge_device=device, progress=lambda done, total: report("transformer", done, total))
    write_transformer_tensors(((names[name], tensor) for name, tensor in tensors if name in names), config, target / "transformer", bits, device, label=checkpoint.path.name)
    report("text_encoder", 0, 1)
    ensure_prepared(target / "text_encoder", text_encoder_sources(repo_dir), 8, lambda folder: write_text_encoder(repo_dir, folder), reuse=prepared_text_encoders(models_dir))
    _replace_file(repo_dir / "text_encoder" / "config.json", target / "text_encoder" / "config.json")
    report("components", 0, 1)
    _link_components(repo_dir, target)
    block_matrices, transformer_rest = count_transformer_elements((names[name], checkpoint.shapes[name]) for name in checkpoint.transformer if name in names)
    manifest = {
        "bits": bits,
        "storage": _STORAGE[bits],
        "block_matrices": block_matrices,
        "transformer_rest": transformer_rest,
        "source": checkpoint.path.name,
        "distilled_lora": lora.name if lora is not None else None,
    }
    write_converted_ltx(target, manifest, version=checkpoint.version)


def _link_components(repo_dir: Path, target: Path) -> None:
    """Copy the diffusers repo's ``model_index.json`` into *target* and hard-link its other components there."""
    index = json.loads((repo_dir / "model_index.json").read_text(encoding="utf-8"))
    for name, spec in index.items():
        if name.startswith("_") or name in _OWN_COMPONENTS or not isinstance(spec, list) or not (repo_dir / name).is_dir():
            continue
        link_tree(repo_dir / name, target / name)
    _replace_file(repo_dir / "model_index.json", target / "model_index.json")


def _replace_file(source: Path, target: Path) -> None:
    """Copy *source* to *target* as a file of its own (never writing through a hard link another folder shares)."""
    target.unlink(missing_ok=True)
    shutil.copyfile(source, target)
