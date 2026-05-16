"""Build and embed reusable config metadata in generated assets, and construct full provenance payloads."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.utils.paths import display_stem

PROVENANCE_SCHEMA = "zvisiongenerator.asset-provenance.v1"
IMAGE_CONFIG_SCHEMA = "zvisiongenerator.config.v1"
VIDEO_CONFIG_SCHEMA = IMAGE_CONFIG_SCHEMA
_PNG_CONFIG_KEY = "zvisiongenerator.config"
_MP4_CONFIG_KEY = "zvisiongenerator.config"


def build_image_config_payload(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> dict[str, Any]:
    """Build a minimal reusable config payload for an image generation.

    Contains only fields that are meaningful for re-generation: schema, workflow,
    prompt, model, seed, steps, guidance, dimensions, ratio, size, image_path, lora.
    Excludes output paths, runtime internals, resolved_prompt, and postprocess state.
    """
    width = artifacts.image.width if artifacts.image is not None else request.width
    height = artifacts.image.height if artifacts.image is not None else request.height
    return _drop_unserializable(
        {
            "schema": IMAGE_CONFIG_SCHEMA,
            "workflow": "img2img" if request.image_path else "txt2img",
            "prompt": request.prompt,
            "model": request.model_name,
            "seed": request.seed,
            "steps": request.steps,
            "guidance": request.guidance,
            "width": width,
            "height": height,
            "ratio": request.ratio,
            "size": request.size,
            "image_path": request.image_path,
            "lora": _format_loras(request.lora_paths, request.lora_weights),
        }
    )


def build_video_config_payload(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> dict[str, Any]:
    """Build a minimal reusable config payload for a video generation.

    Contains only fields meaningful for re-generation: schema, workflow, prompt,
    model, seed, steps, width, height, ratio, size, frame_count, image_path, lora.
    Excludes model_family, duplicate model_name, resolved_prompt, generation_time,
    output paths, media_type, and runtime/postprocess/audio internals.
    """
    return _drop_unserializable(
        {
            "schema": VIDEO_CONFIG_SCHEMA,
            "workflow": "img2vid" if request.image_path else "txt2vid",
            "prompt": request.prompt,
            "model": request.model_name,
            "seed": request.seed,
            "steps": request.steps,
            "width": request.width,
            "height": request.height,
            "ratio": None,
            "size": None,
            "frame_count": request.num_frames,
            "image_path": request.image_path,
            "lora": _format_loras(request.lora_paths, request.lora_weights),
        }
    )


def embed_mp4_config(video_path: str | Path, payload: dict[str, Any]) -> None:
    """Embed a config payload as JSON in MP4 container metadata under zvisiongenerator.config.

    Uses ffmpeg to copy streams and attach the metadata key, replacing the original file.

    Raises:
        subprocess.CalledProcessError: If ffmpeg exits with a non-zero status.
    """
    video_path = Path(video_path)
    tmp = video_path.with_suffix(".tmp.mp4")
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-c", "copy", "-metadata", f"{_MP4_CONFIG_KEY}={value}", str(tmp)],
            check=True,
            capture_output=True,
        )
        tmp.replace(video_path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def read_mp4_config(video_path: str | Path) -> dict[str, Any] | None:
    """Read the zvisiongenerator.config payload from MP4 container metadata, or None if absent."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    raw = data.get("format", {}).get("tags", {}).get(_MP4_CONFIG_KEY)
    if raw is None:
        return None
    return json.loads(raw)


def embed_png_config(png_info: PngInfo, payload: dict[str, Any]) -> None:
    """Embed a config payload as JSON in PNG metadata under zvisiongenerator.config."""
    png_info.add_text(_PNG_CONFIG_KEY, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def read_png_config(image_path: str | Path) -> dict[str, Any] | None:
    """Read the zvisiongenerator.config payload from PNG metadata, or None if absent."""
    from PIL import Image as _Image

    with _Image.open(image_path) as img:
        raw = img.info.get(_PNG_CONFIG_KEY)
    if raw is None:
        return None
    return json.loads(raw)


def build_image_provenance(asset_path: str | Path, request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> dict[str, Any]:
    """Build conservative JSON provenance for an image generation."""
    prompt = artifacts.resolved_prompt or request.prompt
    width = artifacts.image.width if artifacts.image is not None else request.width
    height = artifacts.image.height if artifacts.image is not None else request.height
    return _drop_unserializable(
        {
            "schema": PROVENANCE_SCHEMA,
            "media_type": "image",
            "workflow": "img2img" if request.image_path else "txt2img",
            "prompt": request.prompt,
            "resolved_prompt": prompt,
            "negative_prompt": request.negative_prompt,
            "model": request.model_name,
            "model_name": request.model_name,
            "model_family": request.model_family,
            "seed": request.seed,
            "steps": request.steps,
            "guidance": request.guidance,
            "scheduler": request.scheduler,
            "width": width,
            "height": height,
            "ratio": request.ratio,
            "size": request.size,
            "frame_count": None,
            "image_path": request.image_path,
            "image_strength": request.image_strength if request.image_path else None,
            "lora": _format_loras(request.lora_paths, request.lora_weights),
            "loras": _build_loras(request.lora_paths, request.lora_weights),
            "generation": {
                "generation_time": artifacts.generation_time,
                "was_upscaled": artifacts.was_upscaled,
                "upscale_factor": request.upscale_factor,
                "upscale_denoise": request.upscale_denoise,
                "upscale_steps": request.upscale_steps,
                "upscale_guidance": request.upscale_guidance,
                "sharpen": request.sharpen,
                "contrast": request.contrast,
                "saturation": request.saturation,
            },
            "output": {
                "asset_path": str(Path(asset_path)),
                "filename": Path(asset_path).name,
            },
        }
    )


def build_video_provenance(asset_path: str | Path, request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> dict[str, Any]:
    """Build conservative JSON provenance for a video generation."""
    return _drop_unserializable(
        {
            "schema": PROVENANCE_SCHEMA,
            "media_type": "video",
            "workflow": "img2vid" if request.image_path else "txt2vid",
            "prompt": request.prompt,
            "resolved_prompt": artifacts.resolved_prompt or request.prompt,
            "model": request.model_name,
            "model_name": request.model_name,
            "model_family": request.model_family,
            "seed": request.seed,
            "steps": request.steps,
            "guidance": None,
            "width": request.width,
            "height": request.height,
            "ratio": None,
            "size": None,
            "frame_count": request.num_frames,
            "image_path": request.image_path,
            "image_strength": None,
            "lora": _format_loras(request.lora_paths, request.lora_weights),
            "loras": _build_loras(request.lora_paths, request.lora_weights),
            "generation": {
                "generation_time": artifacts.generation_time,
                "output_format": request.output_format,
                "upscale_factor": request.upscale,
                "upscale_steps": request.upscale_steps,
                "audio": not request.no_audio,
            },
            "output": {
                "asset_path": str(Path(asset_path)),
                "filename": Path(asset_path).name,
            },
        }
    )


def _build_loras(paths: list[str] | None, weights: list[float] | None) -> list[dict[str, Any]]:
    if not paths:
        return []
    return [
        {
            "name": display_stem(path),
            "path": path,
            "weight": weights[index] if weights is not None and index < len(weights) else 1.0,
        }
        for index, path in enumerate(paths)
    ]


def _format_loras(paths: list[str] | None, weights: list[float] | None) -> str | None:
    loras = _build_loras(paths, weights)
    if not loras:
        return None
    return ",".join(f"{item['path']}:{item['weight']:g}" for item in loras)


def _drop_unserializable(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, default=str))
