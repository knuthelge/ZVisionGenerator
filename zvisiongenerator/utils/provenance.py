"""Build, embed and read the config metadata saved inside generated images and videos."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.utils.paths import collapse_home, display_stem, expand_home

IMAGE_CONFIG_SCHEMA = "zvisiongenerator.config.v1"
VIDEO_CONFIG_SCHEMA = IMAGE_CONFIG_SCHEMA
_PNG_CONFIG_KEY = "zvisiongenerator.config"
_MP4_CONFIG_KEY = "zvisiongenerator.config"
# EXIF ImageDescription tag, where saved images also carry the rendered prompt.
EXIF_IMAGE_DESCRIPTION = 0x010E
# The recorded ``generation.upscale.method`` values: refine with the image model, or upscale with SeedVR2.
UPSCALE_METHOD_CREATIVE = "creative"
UPSCALE_METHOD_SEEDVR2 = "seedvr2"


def build_image_config_payload(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> dict[str, Any]:
    """Build the config payload embedded in a saved image.

    The top-level keys are the reusable settings (schema, workflow, prompt, negative prompt, model, family, seed,
    steps, guidance, scheduler, dimensions, ratio, size, reference image, strength, LoRAs). ``generation`` records
    how the file was made (time, upscale, post-processing) and ``source`` the image an upscale started from.
    An auto-enhanced prompt replaces the template prompt so the asset records what rendered.
    """
    width = artifacts.image.width if artifacts.image is not None else request.width
    height = artifacts.image.height if artifacts.image is not None else request.height
    negative_prompt = None if artifacts.metadata.get("negative_suppressed") else request.negative_prompt
    # A SeedVR2 upscale of a source with no recorded sampling settings must not save class defaults as if they were its own.
    unknown_source = request.upscale_method == UPSCALE_METHOD_SEEDVR2 and not request.source_settings_known
    payload: dict[str, Any] = {
        "schema": IMAGE_CONFIG_SCHEMA,
        "workflow": _image_workflow(request),
        "prompt": artifacts.metadata.get("enhanced_prompt") or request.prompt,
        "negative_prompt": negative_prompt or None,
        "model": _portable_path(request.model_name),
        "model_family": request.model_family,
        "seed": None if unknown_source else request.seed,
        "steps": None if unknown_source else request.steps,
        "guidance": None if unknown_source else request.guidance,
        "scheduler": request.scheduler_name,
        "width": width,
        "height": height,
        "ratio": request.ratio,
        "size": request.size,
        "image_path": _portable_path(request.image_path),
        "lora": _format_loras(request.lora_paths, request.lora_weights),
        "generation": _image_generation_details(request, artifacts),
    }
    if request.image_path:
        payload["image_strength"] = request.image_strength
    source = artifacts.metadata.get("upscale_source")
    if source:
        # The workflow that made the original, so Reuse settings regenerates it (img2img keeps its reference).
        payload["source"] = {**source, "path": _portable_path(source.get("path")), "workflow": "img2img" if request.image_path else "txt2img"}
    return _drop_unserializable(payload)


def build_video_config_payload(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> dict[str, Any]:
    """Build the config payload embedded in a saved video.

    The top-level keys are the reusable settings (schema, workflow, prompt, model, family, seed, steps,
    dimensions, frame count, reference image, LoRAs); ``generation`` records time, upscale, audio and format.
    An auto-enhanced prompt replaces the template prompt so the asset records what rendered. The prompt is the
    action alone; the scene is recorded apart as ``scene`` (omitted when there is none), and ``scene_input`` and
    ``prompt_input`` keep the user's own text when the scene or prompt was enhanced.

    An extension (``workflow`` ``"extend"``) records the action as sent, the ``source_video`` it continued and the
    new frames as ``extend_frames``; ``frame_count`` is the whole output clip and the size is the source's displayed one.
    """
    upscale = _drop_empty({"factor": request.upscale, "steps": request.upscale_steps}) if request.upscale else None
    extend = request.extend
    audio = not request.no_audio if extend is None else _extend_audio_output(extend.source, no_audio=request.no_audio)
    payload = {
        "schema": VIDEO_CONFIG_SCHEMA,
        "workflow": _video_workflow(request),
        "prompt": _video_prompt(request, artifacts),
        **_video_scene_fields(request, artifacts),
        "model": _portable_path(request.model_name),
        "model_family": request.model_family,
        "seed": request.seed,
        "steps": request.steps,
        "width": extend.source.width if extend is not None else request.width,
        "height": extend.source.height if extend is not None else request.height,
        "ratio": None,
        "size": None,
        "frame_count": extend.source.frame_count + extend.frames if extend is not None else request.num_frames,
        "image_path": _portable_path(request.image_path),
        "lora": _format_loras(request.lora_paths, request.lora_weights),
        "generation": _drop_empty(
            {
                "time": _round_seconds(artifacts.generation_time),
                "upscale": upscale,
                "audio": audio,
                "output_format": request.output_format,
            }
        ),
    }
    if extend is not None:
        payload["source_video"] = _portable_path(str(extend.source.path))
        payload["extend_frames"] = extend.frames
    return _drop_unserializable(payload)


def _extend_audio_output(source: Any, *, no_audio: bool) -> bool:
    """Return whether an extension has sound (see ``extend_audio_output``); imported late because ``video_extend`` imports this module."""
    from zvisiongenerator.utils.video_extend import extend_audio_output

    return extend_audio_output(source, no_audio=no_audio)


def _video_workflow(request: VideoGenerationRequest) -> str:
    """Return the video workflow name: ``extend``, ``img2vid`` or ``txt2vid``."""
    if request.extend is not None:
        return "extend"
    return "img2vid" if request.image_path else "txt2vid"


def _video_prompt(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> str:
    """Return the prompt a video records: the enhanced action, else the template; an extension records the action it sent.

    The action sent differs from the template when quoted sentences of a legacy source's prompt were stripped.
    """
    enhanced = artifacts.metadata.get("enhanced_prompt")
    if request.extend is not None and not enhanced:
        return artifacts.metadata.get("prompt_sent", request.prompt)
    return enhanced or request.prompt


def _video_scene_fields(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> dict[str, str]:
    """Return the video payload's scene keys that apply: ``scene``, and the user's inputs when the text was enhanced.

    The inputs come from the request when its scene or prompt already holds a rewrite made before the job; otherwise,
    for a rewrite made by the job, they are the scene and prompt as written. A job that rewrote only the action has
    no ``enhanced_scene``: ``scene`` is the user's own and ``scene_input`` repeats it, which tells it from a rewrite
    of a blank scene (a ``scene`` with no ``scene_input``).
    """
    metadata = artifacts.metadata
    enhanced = bool(metadata.get("enhanced_prompt"))
    fields = {
        "scene": metadata.get("enhanced_scene") or metadata.get("scene"),
        "scene_input": request.scene_input or (request.scene if enhanced else None),
        "prompt_input": request.prompt_input or (request.prompt if enhanced else None),
    }
    return {key: text for key, value in fields.items() if (text := optional_text(value)) is not None}


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
            ["ffmpeg", "-y", "-i", str(video_path), "-c", "copy", "-movflags", "use_metadata_tags", "-metadata", f"{_MP4_CONFIG_KEY}={value}", str(tmp)],
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


@dataclass(frozen=True)
class RecordedSettings:
    """Generation settings embedded in a saved image or video, as typed values; unrecorded ones are None."""

    workflow: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    model: str | None = None
    model_family: str | None = None
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    scheduler: str | None = None
    width: int | None = None
    height: int | None = None
    ratio: str | None = None
    size: str | None = None
    frame_count: int | None = None
    image_path: str | None = None
    image_strength: float | None = None
    # Video scene as generated, and the user's own scene and prompt when the recorded ones were enhanced.
    scene: str | None = None
    scene_input: str | None = None
    prompt_input: str | None = None
    # An extension: the clip it continued and the frames it added (``frame_count`` is the whole clip).
    source_video: str | None = None
    extend_frames: int | None = None
    # ``path:weight`` entries joined by commas, as ``--lora`` takes them.
    lora: str | None = None
    # How the file was made: time, quantize, upscale and post-processing (empty for files that predate it).
    generation: dict[str, Any] = field(default_factory=dict)
    # The image an upscale started from: ``{path, width, height, workflow}``.
    source: dict[str, Any] | None = None


def recorded_settings(config: dict[str, Any]) -> RecordedSettings:
    """Read the top-level settings of an embedded config payload; malformed values become None."""
    return RecordedSettings(
        workflow=optional_text(config.get("workflow")),
        prompt=optional_text(config.get("prompt")),
        negative_prompt=optional_text(config.get("negative_prompt")),
        model=_local_path(optional_text(config.get("model"))),
        model_family=optional_text(config.get("model_family")),
        seed=optional_int(config.get("seed")),
        steps=optional_int(config.get("steps")),
        guidance=optional_float(config.get("guidance")),
        scheduler=optional_text(config.get("scheduler")),
        width=optional_int(config.get("width")),
        height=optional_int(config.get("height")),
        ratio=optional_text(config.get("ratio")),
        size=optional_text(config.get("size")),
        frame_count=optional_int(config.get("frame_count")),
        image_path=_local_path(optional_text(config.get("image_path"))),
        image_strength=optional_float(config.get("image_strength")),
        scene=optional_text(config.get("scene")),
        scene_input=optional_text(config.get("scene_input")),
        prompt_input=optional_text(config.get("prompt_input")),
        source_video=_local_path(optional_text(config.get("source_video"))),
        extend_frames=optional_int(config.get("extend_frames")),
        lora=_lora_string(config.get("lora")),
        generation=_generation_block(config.get("generation"), is_video=_is_video_config(config)),
        source=_source_block(config.get("source")),
    )


def _lora_string(value: Any) -> str | None:
    """Return LoRAs as ``name[:weight]`` entries joined by commas; older files may record a list or a mapping."""
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        return _expand_lora_entries(value.strip()) or None
    items = value if isinstance(value, list) else [value]
    entries: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            entries.append(item.strip())
        elif isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip():
            weight = item.get("weight")
            entries.append(item["name"].strip() if weight in (None, "") else f"{item['name'].strip()}:{weight}")
    return _expand_lora_entries(",".join(entries)) or None


def _expand_lora_entries(text: str) -> str:
    """Expand a leading ``~`` in each comma-separated LoRA entry."""
    return ",".join(expand_home(entry) for entry in text.split(",")) if text else text


def _is_video_config(config: dict[str, Any]) -> bool:
    """Return whether an embedded config payload describes a video."""
    return optional_int(config.get("frame_count")) is not None or optional_text(config.get("workflow")) in ("txt2vid", "img2vid", "extend")


def _generation_block(value: Any, *, is_video: bool = False) -> dict[str, Any]:
    """Keep the known, well-typed keys of a recorded ``generation`` block; videos record no upscale method."""
    if not isinstance(value, dict):
        return {}
    generation: dict[str, Any] = {}
    for key in ("time", "sharpen", "contrast", "saturation"):
        number = optional_float(value.get(key))
        if number is not None:
            generation[key] = number
    quantize = optional_int(value.get("quantize"))
    if quantize is not None:
        generation["quantize"] = quantize
    if isinstance(value.get("audio"), bool):
        generation["audio"] = value["audio"]
    output_format = optional_text(value.get("output_format"))
    if output_format is not None:
        generation["output_format"] = output_format
    upscale = value.get("upscale")
    if isinstance(upscale, dict):
        coerced = {
            "factor": optional_int(upscale.get("factor")),
            "denoise": optional_float(upscale.get("denoise")),
            "steps": optional_int(upscale.get("steps")),
            "guidance": optional_float(upscale.get("guidance")),
            "pre_sharpen": optional_float(upscale.get("pre_sharpen")),
            "model": optional_text(upscale.get("model")),
            "quantize": optional_int(upscale.get("quantize")),
            "softness": optional_float(upscale.get("softness")),
            "seed": optional_int(upscale.get("seed")),
        }
        coerced = {key: item for key, item in coerced.items() if item is not None}
        if coerced and is_video:
            generation["upscale"] = coerced
        elif coerced:
            # Image files that predate the method were all refined by the image model.
            method = UPSCALE_METHOD_SEEDVR2 if upscale.get("method") == UPSCALE_METHOD_SEEDVR2 else UPSCALE_METHOD_CREATIVE
            generation["upscale"] = {"method": method, **coerced}
    return generation


def _source_block(value: Any) -> dict[str, Any] | None:
    """Return a recorded upscale source as ``{path, width, height, workflow}``, or None."""
    if not isinstance(value, dict):
        return None
    path = optional_text(value.get("path"))
    if path is None:
        return None
    return {
        "path": expand_home(path),
        "width": optional_int(value.get("width")),
        "height": optional_int(value.get("height")),
        "workflow": optional_text(value.get("workflow")),
    }


def image_prompt_text(image: Any) -> str | None:
    """Return the rendered prompt saved in an open PIL image (PNG ``Description`` or EXIF ImageDescription)."""
    return optional_text(image.info.get("Description") or image.getexif().get(EXIF_IMAGE_DESCRIPTION))


def optional_text(value: Any) -> str | None:
    """Return a recorded value as stripped text, or None when it is missing or blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_int(value: Any) -> int | None:
    """Return a recorded value as an int, or None when it is missing, blank, a bool or not a number."""
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def optional_float(value: Any) -> float | None:
    """Return a recorded value as a float, or None when it is missing, blank, a bool or not a number."""
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _build_loras(paths: list[str] | None, weights: list[float] | None) -> list[dict[str, Any]]:
    if not paths:
        return []
    return [
        {
            "name": display_stem(path),
            "path": collapse_home(path),
            "weight": weights[index] if weights is not None and index < len(weights) else 1.0,
        }
        for index, path in enumerate(paths)
    ]


def _portable_path(path: Any) -> Any:
    """Write a recorded path with the home directory as ``~``, so shared files do not reveal the username."""
    return collapse_home(path) if isinstance(path, str) else path


def _local_path(path: str | None) -> str | None:
    """Read a recorded path back with ``~`` expanded to this user's home directory."""
    return expand_home(path) if path is not None else None


def _format_loras(paths: list[str] | None, weights: list[float] | None) -> str | None:
    loras = _build_loras(paths, weights)
    if not loras:
        return None
    return ",".join(f"{item['path']}:{item['weight']:g}" for item in loras)


def _image_workflow(request: ImageGenerationRequest) -> str:
    if request.upscale_source:
        return "upscale"
    return "img2img" if request.image_path else "txt2img"


def _image_generation_details(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> dict[str, Any]:
    upscale = _upscale_details(request) if artifacts.was_upscaled and request.upscale_factor else None
    return _drop_empty(
        {
            "time": _round_seconds(artifacts.generation_time),
            "quantize": request.quantize,
            "upscale": upscale,
            "sharpen": _sharpen_amount(request, artifacts) if request.sharpen else None,
            "contrast": request.contrast_amount if request.contrast else None,
            "saturation": request.saturation_amount if request.saturation else None,
        }
    )


def _upscale_details(request: ImageGenerationRequest) -> dict[str, Any]:
    """Return the recorded ``generation.upscale`` block: the method plus the settings that method used."""
    if request.upscale_method == UPSCALE_METHOD_SEEDVR2:
        return _drop_empty(
            {
                "method": UPSCALE_METHOD_SEEDVR2,
                "factor": request.upscale_factor,
                "model": request.upscale_model,
                "quantize": request.upscale_quantize,
                "softness": request.upscale_softness,
                "seed": request.seed,
            }
        )
    return _drop_empty(
        {
            "method": UPSCALE_METHOD_CREATIVE,
            "factor": request.upscale_factor,
            "denoise": request.upscale_denoise,
            "steps": request.upscale_steps,
            "guidance": request.upscale_guidance,
            "pre_sharpen": request.sharpen_amount_pre_upscale if request.upscale_sharpen else None,
        }
    )


def _sharpen_amount(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> float:
    if request.sharpen_amount_override is not None:
        return request.sharpen_amount_override
    return request.sharpen_amount_upscaled if artifacts.was_upscaled else request.sharpen_amount_normal


def _round_seconds(seconds: float) -> float | None:
    return round(seconds, 1) if seconds > 0 else None


def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _drop_unserializable(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, default=str))
