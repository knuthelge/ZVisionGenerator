"""Converted MiniMax H3 folders on Windows/Linux: write one from a fine-tune.

A converted folder is a folded 4-bit copy of the fine-tune's transformer, in the layout of stock's ``h3@q4``, plus the
folder manifest (:mod:`zvisiongenerator.utils.h3_converted`), written last. Every other component (text encoder,
VAEs, schedulers, the latent upscaler) comes from the stock repository at load time, without stock's transformer.
The turbo LoRA is merged into the copy unless the checkpoint is already step-distilled, so a converted model never
gets it at run time.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends.cuda.video.h3 import lora as h3_lora
from zvisiongenerator.backends.cuda.video.h3 import stored_copy
from zvisiongenerator.backends.cuda.video.h3.sources import resolve_base_files
from zvisiongenerator.backends.cuda.video.runtime import configure_torch_runtime
from zvisiongenerator.converters.h3_checkpoint import diffusers_tensors, stock_time_embedder
from zvisiongenerator.utils.h3_converted import write_converted_h3
from zvisiongenerator.utils.h3_preset import h3_preset

if TYPE_CHECKING:
    from zvisiongenerator.converters.h3_checkpoint import H3Checkpoint
    from zvisiongenerator.utils.h3_preset import H3Preset

__all__ = ["STORAGE", "convert", "write_converted"]

STORAGE = "nf4-folded"

Report = Callable[[str, int, int], None]


def convert(checkpoint: H3Checkpoint, target: Path, *, turbo: bool, config: dict[str, Any], report: Report) -> None:
    """Write the converted folder of *checkpoint* into *target* on this machine's GPU.

    Raises:
        RuntimeError: When there is no usable NVIDIA GPU, or a base file can't be downloaded.
    """
    import torch

    from zvisiongenerator.backends.cuda.driver import require_cuda

    require_cuda(torch, "Converting a MiniMax H3 checkpoint on Windows and Linux")
    configure_torch_runtime(torch)  # as the backend runs stock's copy build, so the tables match
    preset = h3_preset(config)
    report("download", 0, 1)
    files = resolve_base_files(preset, lora=not turbo)
    time_embedder = stock_time_embedder(preset.repo, preset.revision) if checkpoint.curve_rank is not None else None
    write_converted(checkpoint, target, preset=preset, repo_dir=files.repo_dir, lora_file=None if turbo else files.lora_file, time_embedder=time_embedder, device=torch.device("cuda"), report=report)
    print("Note: keep the checkpoint; the folder holds a 4-bit copy tied to this version's H3 settings, and a change to them asks for converting again.")


def write_converted(
    checkpoint: H3Checkpoint,
    target: Path,
    *,
    preset: H3Preset,
    repo_dir: Path,
    lora_file: Path | None,
    time_embedder: dict[str, Any] | None,
    device: Any,
    report: Report,
) -> None:
    """Write the converted folder of *checkpoint* into the new folder *target*.

    Args:
        checkpoint: The checked checkpoint.
        target: The folder to write (a partial folder; the caller moves it into place).
        preset: The H3 preset: its schedules decide the fold keys, its turbo scale the merge.
        repo_dir: The stock repository (its transformer config and schedulers are read).
        lora_file: The turbo LoRA to merge, or ``None`` for an already step-distilled checkpoint.
        time_embedder: Stock's time embedder, which un-prunes a pruned checkpoint.
        device: The CUDA device weights are decoded and packed on.
        report: Called with ``(stage, done, total)`` as the conversion advances.
    """
    from zvisiongenerator.backends.cuda.video.h3 import adaln

    config = json.loads((repo_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    keys = adaln.fold_keys(preset.schedules, repo_dir)
    turbo = (h3_lora.load_pairs(lora_file), preset.lora_scale) if lora_file is not None else None

    def tensors(wanted: Callable[[str], bool] | None) -> Any:
        progress = (lambda done, total: report("transformer", done, total)) if wanted is None else None
        return diffusers_tensors(checkpoint, wanted, device=device, progress=progress, time_embedder=time_embedder)

    stored_copy.write_folded(tensors, config, turbo, keys, target, device, label=checkpoint.path.name)
    write_converted_h3(
        target,
        {
            "storage": STORAGE,
            "turbo": "merged" if turbo is not None else "distilled",
            "turbo_lora": f"{preset.lora.repo_id}@{preset.lora.revision}/{preset.lora.files[0]}" if turbo is not None else None,
            "copy": stored_copy.copy_settings(preset.lora_scale if turbo is not None else None, keys),
            "pruned_rank": checkpoint.curve_rank,
            "source": checkpoint.path.name,
            "source_bytes": sum(path.stat().st_size for path in checkpoint.files),
        },
    )
