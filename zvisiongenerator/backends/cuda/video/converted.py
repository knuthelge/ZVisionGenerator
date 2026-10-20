"""Converted LTX-2.3 folders on Windows/Linux: write one from a single-file checkpoint, and check one before it loads.

A converted folder is a diffusers LTX-2 folder that holds its own quantized weights:

- ``transformer/``: the checkpoint's transformer in NF4 (q4) or FP8 (q8), in the prepared-copy layout.
- ``text_encoder/``: the shared FP8 text encoder (a prepared copy with its own manifest, hard-linked from another
  one when it exists) plus its ``config.json``.
- the diffusers repo's index and other components (connectors, VAEs, vocoder, tokenizer, scheduler), hard-linked.
- the folder manifest (:mod:`zvisiongenerator.utils.ltx_converted`), written last.

It loads without the source checkpoint. NF4 copies record how bitsandbytes serialized them and a dequantized
sample, so a bitsandbytes update that reads them differently refuses the folder instead of loading noise.
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda.video.weights import (
    NF4_LAYOUT,
    count_transformer_elements,
    ensure_prepared,
    nf4_sample,
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

__all__ = ["check_converted", "convert", "write_converted"]

# Components written here rather than linked from the diffusers repo.
_OWN_COMPONENTS = ("transformer", "text_encoder")
_STORAGE = {4: "nf4", 8: "fp8-e4m3"}
# A bitsandbytes update may round differently; anything beyond this share of the sample's largest value is a changed format.
_NF4_SAMPLE_TOLERANCE = 0.01

Report = Callable[[str, int, int], None]


def convert(checkpoint: LtxCheckpoint, target: Path, bits: int, *, config: dict[str, Any], models_dir: Path, lora: Path | None, report: Report) -> None:
    """Write the converted folder of *checkpoint* at *bits* into *target* on this machine's GPU, with the configured diffusers repo's components.

    Raises:
        RuntimeError: When there is no usable NVIDIA GPU, or the repo can't be downloaded.
    """
    import torch

    from zvisiongenerator.backends.cuda.driver import require_cuda
    from zvisiongenerator.backends.cuda.video.backend import _repo_dir
    from zvisiongenerator.utils.config import ltx_diffusers_repo

    require_cuda(torch, "Converting an LTX checkpoint on Windows and Linux")
    report("download", 0, 1)
    repo_dir = _repo_dir(ltx_diffusers_repo(config), with_transformer=False)
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
        repo_dir: The diffusers LTX-2.3 repo the other components come from (its transformer weights are not needed).
        models_dir: The models folder, searched for a text-encoder copy to reuse.
        lora: Lightricks' distilled LoRA to merge in, or ``None``.
        device: The CUDA device NF4 weights are packed on (and the LoRA merged on).
        report: Called with ``(stage, done, total)`` as the conversion advances.

    Raises:
        ValueError: When the checkpoint's text connectors differ from stock's (only stock ones are supported here).
        RuntimeError: When the checkpoint does not cover the transformer.
    """
    report("connectors", 0, 1)
    if connectors_status(checkpoint) == "changed":
        raise ValueError(f"{checkpoint.path.name} changes the text connectors; on Windows and Linux only checkpoints with stock connectors can be converted.")
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
    manifest: dict[str, Any] = {
        "bits": bits,
        "storage": _STORAGE[bits],
        "block_matrices": block_matrices,
        "transformer_rest": transformer_rest,
        "source": checkpoint.path.name,
        "distilled_lora": lora.name if lora is not None else None,
    }
    if bits == 4:
        version = importlib.metadata.version("bitsandbytes")
        weight, sample = nf4_sample(target / "transformer", device)
        manifest["nf4"] = {**NF4_LAYOUT, "sample_weight": weight, "sample": sample, "bitsandbytes": version, "checked_bitsandbytes": version}
    write_converted_ltx(target, manifest)


def check_converted(model_dir: Path, manifest: dict[str, Any], device: Any) -> None:
    """Check that the converted folder *model_dir* can be read by the installed libraries before it loads.

    An NF4 copy saved in another layout is refused. After a bitsandbytes update, its sample weight is dequantized
    again and compared with the recorded values; when they match, the new version is recorded so the check runs
    once per version.

    Raises:
        RuntimeError: When the folder can't be read as it was written; it has to be converted again.
    """
    if manifest.get("storage") != _STORAGE[manifest["bits"]]:
        raise _reconvert(model_dir, "it was stored in another format")
    if manifest["bits"] != 4:
        return
    nf4 = manifest.get("nf4") or {}
    if {key: nf4.get(key) for key in NF4_LAYOUT} != NF4_LAYOUT:
        raise _reconvert(model_dir, "its NF4 weights were saved in another layout")
    version = importlib.metadata.version("bitsandbytes")
    if nf4.get("checked_bitsandbytes") == version:
        return
    _weight, sample = nf4_sample(model_dir / "transformer", device)
    recorded = nf4.get("sample") or []
    limit = _NF4_SAMPLE_TOLERANCE * max((abs(value) for value in recorded), default=0.0)
    if len(sample) != len(recorded) or any(abs(a - b) > limit for a, b in zip(sample, recorded, strict=True)):
        raise _reconvert(model_dir, f"bitsandbytes {version} reads its NF4 weights differently from {nf4.get('bitsandbytes')}, which wrote them")
    write_converted_ltx(model_dir, {key: value for key, value in manifest.items() if key not in ("kind", "layout")} | {"nf4": {**nf4, "checked_bitsandbytes": version}})


def _reconvert(model_dir: Path, reason: str) -> RuntimeError:
    """Return the error for a converted folder that can't load: *reason*, and what to do."""
    return RuntimeError(f"The converted LTX model {model_dir.name} can't be loaded: {reason}. Convert the original checkpoint again with 'ziv model model --model-type ltx-2.3'.")


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
