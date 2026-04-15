"""Generate final filename with timestamp and settings information."""

from __future__ import annotations

from datetime import datetime
import os
import re


def generate_filename(
    set_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    scheduler: str | None = None,
    model: str | None = None,
    lora_paths: list[str] | None = None,
    lora_weights: list[float] | None = None,
    num_frames: int | None = None,
) -> str:
    if set_name:
        safe_name = re.sub(r'[/\\:*?"<>|]', "_", set_name)
        safe_name = safe_name.replace("..", "_")
        safe_name = safe_name.strip(". ")
        set_name = safe_name or None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if model:
        name = os.path.basename(model)
        for ext in (".safetensors", ".ckpt", ".bin", ".pt"):
            if name.endswith(ext):
                name = name[: -len(ext)]
                break
        model_part = f"_{name.replace(' ', '_')}"
    else:
        model_part = ""
    if lora_paths:
        lora_parts = []
        for p, w in zip(lora_paths, lora_weights or []):
            name = os.path.basename(p).split(".")[0]
            lora_parts.append(f"{name}_{int(w * 100)}")
        lora_part = "_" + "_".join(lora_parts)
    else:
        lora_part = ""
    dims_part = f"{width}x{height}"
    frames_part = f"_{num_frames}f" if num_frames is not None else ""
    filename = f"{set_name + '_' if set_name else ''}{timestamp}_{dims_part}{frames_part}{model_part}{'_' + scheduler if scheduler else ''}{lora_part}_steps{steps}{f'_cfg{guidance}' if guidance is not None else ''}_seed{seed}"
    return filename
