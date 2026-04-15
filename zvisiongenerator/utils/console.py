"""Generate console output with timing and settings information."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts


def _fmt_time(secs: float | None) -> str:
    if secs is None:
        return "–"
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def format_generation_info(
    request: ImageGenerationRequest,
    artifacts: ImageWorkingArtifacts,
    *,
    run_number: int,
    total_runs: int,
    ran_iterations: int,
    total_iterations: int,
    set_name: str,
    prompt_idx: int,
    total_prompts: int,
    elapsed_secs: float | None = None,
    avg_secs: float | None = None,
    eta_secs: float | None = None,
) -> str:
    # Derive display dimensions (request stores pre-upscale sizes)
    upscale = request.upscale_factor
    width = request.width * upscale if upscale else request.width
    height = request.height * upscale if upscale else request.height

    model_name = request.model_name
    model_type = request.model_family if request.model_family != "unknown" else None
    model_status = f"Model: {model_name.split('/')[-1]} ({model_type})" if model_type else f"Model: {model_name.split('/')[-1]}" if model_name else "Model: default"
    lora_paths = request.lora_paths
    lora_weights = request.lora_weights or []
    if lora_paths:
        lora_status = ", ".join(f"{os.path.basename(p.replace('.safetensors', ''))} ({w})" for p, w in zip(lora_paths, lora_weights))
    else:
        lora_status = "disabled"
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 80
    if avg_secs is not None:
        timing_line = f"Elapsed: {_fmt_time(elapsed_secs)} | Avg: {_fmt_time(avg_secs)}/img | ETA: ~{_fmt_time(eta_secs)}"
    else:
        timing_line = f"Elapsed: {_fmt_time(elapsed_secs)} | ETA: calculating..."
    ratio = request.ratio
    size = request.size
    dims_display = f"Ratio: {ratio}, Size: {size}, {width}\u00d7{height}" if ratio is not None else f"{width}\u00d7{height}"
    return f"\n{'–' * terminal_width}\nGenerating image number {ran_iterations}/{total_iterations}, in run nr {run_number + 1}/{total_runs}.\nGenerating prompt set '{set_name}' (prompt {prompt_idx + 1}/{total_prompts}).\nSteps: {request.steps}, Guidance: {request.guidance}, {dims_display}. {'Upscaling enabled' if upscale else 'Upscaling disabled'}\n{model_status}. LoRA: {lora_status}\n{timing_line}\n{'–' * terminal_width}\n"
