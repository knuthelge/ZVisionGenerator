"""Convert a MiniMax H3 fine-tune into a models-folder entry of its own (``--model-type h3``).

The converted model runs like stock H3: at 4 or 8 steps, with LoRAs, image to video, first and last frame, Extend
and upscale. H3 runs that fast only through lightx2v's turbo LoRA, which stock gets merged in; a checkpoint that is
already step-distilled must not get it twice. Nothing in a file tells reliably which kind it is, so the caller says
(``--turbo``: already distilled), and the inspection only shows hints.

The folder's layout depends on the platform (see :func:`zvisiongenerator.backends.convert_h3_checkpoint`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zvisiongenerator.converters.convert_target import Progress, checked_target, write_into
from zvisiongenerator.converters.h3_checkpoint import H3Checkpoint, inspect_h3
from zvisiongenerator.utils.quantized_weights import precision_label

__all__ = ["H3Inspection", "convert_h3", "inspect_h3_for_conversion", "turbo_advice"]

_LAYOUTS = {"comfyui": "ComfyUI single file", "native": "MiniMax native shards", "diffusers": "diffusers"}


@dataclass(frozen=True)
class H3Inspection:
    """What the converter found in an H3 checkpoint, before converting it.

    Attributes:
        checkpoint: The checked checkpoint.
        advice: What the turbo hints mean for converting it.
    """

    checkpoint: H3Checkpoint
    advice: str

    @property
    def summary(self) -> str:
        """One line on the layout and storage: ``ComfyUI single file, pruned (rank 8), INT8 (195)``."""
        parts = [_LAYOUTS[self.checkpoint.layout]]
        if self.checkpoint.curve_rank is not None:
            parts.append(f"pruned (rank {self.checkpoint.curve_rank})")
        parts.append(precision_label(self.checkpoint.precision))
        return ", ".join(parts)


def inspect_h3_for_conversion(path: Path, turbo: str = "--turbo") -> H3Inspection:
    """Check the H3 checkpoint at *path* and say what its turbo hints mean, naming the switch as *turbo*; nothing is downloaded.

    Raises:
        ValueError: If it is not an H3 checkpoint the converter reads.
    """
    checkpoint = inspect_h3(path)
    return H3Inspection(checkpoint=checkpoint, advice=turbo_advice(checkpoint.turbo_hints, turbo))


def turbo_advice(hints: tuple[str, ...], turbo: str) -> str:
    """Return what the turbo *hints* mean for converting, naming the switch as *turbo*."""
    if hints:
        return (
            f"This may already be step-distilled: {'; '.join(hints)}. If its model page says it runs at 4 or 8 steps on its own, "
            f"convert it with {turbo}; otherwise the turbo LoRA is merged in. A wrong choice renders blurry or burnt clips: convert it again."
        )
    return (
        "Nothing in the file says it is step-distilled, so the turbo LoRA is merged in, as for stock. Only a checkpoint that already runs at "
        f"4 or 8 steps (a turbo, lightning or distilled merge) should be converted with {turbo}."
    )


def convert_h3(input_path: Path, models_dir: Path, name: str, *, turbo: bool, force: bool, config: dict[str, Any]) -> Path:
    """Convert the H3 checkpoint *input_path* into ``<models_dir>/<name>``; return the folder.

    Args:
        input_path: A ``.safetensors`` file, or any shard of a sharded set.
        models_dir: The models folder.
        name: The new model's folder name; neither an existing folder (unless *force*) nor a configured alias.
        turbo: The checkpoint is already step-distilled: the turbo LoRA is not merged.
        force: Replace an existing folder of that name, once the new one is complete.
        config: The loaded config.

    Raises:
        ValueError: If the name is taken by an alias, or the checkpoint isn't one the converter reads.
        FileExistsError: If the folder exists and *force* is not set.
        RuntimeError: If a base file can't be downloaded, or the conversion fails.
    """
    target = checked_target(models_dir, name, force=force, config=config)
    inspection = inspect_h3_for_conversion(input_path)
    checkpoint = inspection.checkpoint
    print(f"Checkpoint: {input_path.name} ({inspection.summary})")
    print("--turbo: the checkpoint is already step-distilled, so the turbo LoRA is not merged." if turbo else "Merging the turbo LoRA, as for stock.")
    if inspection.checkpoint.turbo_hints and not turbo:
        print(f"Note: {inspection.advice}")
    from zvisiongenerator.backends import convert_h3_checkpoint

    return write_into(target, lambda partial: convert_h3_checkpoint(checkpoint, partial, turbo=turbo, config=config, report=Progress()))
