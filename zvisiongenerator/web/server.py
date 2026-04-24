"""Serve the maintained FastAPI backend, Svelte SPA shell, and JSON/SSE job endpoints for the Web UI."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Any
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import yaml

from zvisiongenerator.converters.list_assets import list_loras, list_models, list_video_models
from zvisiongenerator.converters.lora_import import import_lora_hf, import_lora_local
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import VideoGenerationRequest
from zvisiongenerator.utils.config import resolve_defaults, resolve_video_defaults, validate_scheduler
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.lora import parse_lora_arg
from zvisiongenerator.utils.paths import get_ziv_data_dir, resolve_lora_path, resolve_model_path
from zvisiongenerator.utils.video_model_detect import detect_video_model
from zvisiongenerator.video_cli import _align_ltx_frames, _align_resolution
from zvisiongenerator.web.config import WebUiConfig, load_web_config
from zvisiongenerator.web.web_runner import JobConflictError, UnsupportedJobControlError, WebRunner


_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".mkv"})

_CANONICAL_WORKFLOW_VALUES = ("txt2img", "img2img", "txt2vid", "img2vid")
_WORKFLOW_ALIASES = {
    "txt2img": "txt2img",
    "texttoimage": "txt2img",
    "image": "txt2img",
    "img2img": "img2img",
    "i2i": "img2img",
    "imagetoimage": "img2img",
    "txt2vid": "txt2vid",
    "texttovideo": "txt2vid",
    "video": "txt2vid",
    "img2vid": "img2vid",
    "i2v": "img2vid",
    "imagetovideo": "img2vid",
}
_WORKFLOW_DEFINITIONS: dict[str, dict[str, Any]] = {
    "txt2img": {
        "mode": "image",
        "model_kind": "image",
        "supports_reference_image": False,
        "requires_reference_image": False,
        "clear_fields": ["image_path", "image_strength", "frames", "audio", "low_memory"],
    },
    "img2img": {
        "mode": "image",
        "model_kind": "image",
        "supports_reference_image": True,
        "requires_reference_image": True,
        "clear_fields": ["frames", "audio", "low_memory"],
    },
    "txt2vid": {
        "mode": "video",
        "model_kind": "video",
        "supports_reference_image": False,
        "requires_reference_image": False,
        "clear_fields": [
            "negative_prompt",
            "guidance",
            "image_path",
            "image_strength",
            "quantize",
            "sharpen_enabled",
            "sharpen_amount",
            "contrast_enabled",
            "contrast_amount",
            "saturation_enabled",
            "saturation_amount",
            "upscale",
            "upscale_denoise",
            "upscale_guidance",
            "upscale_sharpen",
            "upscale_save_pre",
        ],
    },
    "img2vid": {
        "mode": "video",
        "model_kind": "video",
        "supports_reference_image": True,
        "requires_reference_image": True,
        "clear_fields": [
            "negative_prompt",
            "guidance",
            "quantize",
            "sharpen_enabled",
            "sharpen_amount",
            "contrast_enabled",
            "contrast_amount",
            "saturation_enabled",
            "saturation_amount",
            "upscale",
            "upscale_denoise",
            "upscale_guidance",
            "upscale_sharpen",
            "upscale_save_pre",
        ],
    },
}

web_runner = WebRunner()


_IMAGE_BOOTSTRAP_STRENGTH = 0.5
_IMAGE_BOOTSTRAP_POSTPROCESS = {
    "sharpen": 0.8,
    "contrast": False,
    "saturation": False,
}
_IMAGE_BOOTSTRAP_UPSCALE = {
    "enabled": False,
    "factor": None,
    "denoise": None,
    "steps": None,
    "guidance": None,
    "sharpen": True,
    "save_pre": False,
}
_VIDEO_BOOTSTRAP_UPSCALE = {
    "enabled": False,
    "factor": 2,
    "steps": None,
}


@dataclass(frozen=True)
class GalleryAsset:
    """Renderable gallery item metadata."""

    name: str
    kind: str
    extension: str
    filesystem_path: str
    modified_at: float
    modified_label: str
    path_label: str
    media_url: str
    detail_url: str
    reuse_workspace_url: str
    reuse_settings_url: str
    prompt: str
    model_label: str
    width: int | None
    height: int | None
    seed: int | None
    steps: int | None
    guidance: float | None
    dimensions_label: str
    seed_label: str
    steps_label: str
    guidance_label: str
    workflow: str | None
    ratio: str | None
    size: str | None
    frame_count: int | None
    reference_image_path: str | None
    lora: str | None


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Own the background worker pool for the life of the FastAPI app."""
    try:
        yield
    finally:
        web_runner.shutdown()


app = FastAPI(title="Z-Vision Generator Web UI", lifespan=_lifespan)
app.mount("/app-static", StaticFiles(directory=str(Path(__file__).with_name("static") / "app"), check_dir=False), name="app-static")


@app.get("/docs/assets/{asset_name}")
async def docs_asset(asset_name: str) -> FileResponse:
    """Serve preview branding assets referenced by the web docs and browser clients."""
    asset_path = Path(__file__).resolve().parents[2] / "docs" / "assets" / asset_name
    if not asset_path.is_file():
        raise HTTPException(status_code=404, detail=f"Unknown asset: {asset_name}")
    return FileResponse(asset_path)


@app.get("/media/{asset_path:path}")
async def output_media(asset_path: str) -> FileResponse:
    """Serve generated media files from the configured output directory."""
    root = Path(load_web_config().output_dir).resolve()
    candidate = (root / asset_path).resolve()
    if not candidate.is_file() or not candidate.is_relative_to(root):
        raise HTTPException(status_code=404, detail=f"Unknown media asset: {asset_path}")
    return FileResponse(candidate)


@app.get("/")
async def root_redirect() -> RedirectResponse:
    """Redirect the root URL to the Svelte SPA."""
    return RedirectResponse(url="/app", status_code=302)


@app.get("/app")
async def spa_root() -> FileResponse:
    """Serve the Svelte SPA entry point."""
    spa_index = Path(__file__).with_name("static") / "app" / "index.html"
    if not spa_index.is_file():
        raise HTTPException(status_code=503, detail="Svelte SPA not built. Run: make frontend-build")
    return FileResponse(spa_index)


@app.post("/api/pick-directory")
async def pick_directory(request: Request) -> dict[str, str | None]:
    """Open a native directory picker on the local machine running the Web UI."""
    payload = await request.json()
    selected_path = _pick_directory(_coerce_optional_string(payload.get("initial_dir")))
    return {"path": selected_path}


@app.post("/api/generate")
async def generate(request: Request) -> JSONResponse:
    """Accept multipart form submissions and queue image or video generation jobs."""
    form = await request.form()
    web_config = load_web_config()
    mode = str(form.get("mode", "image")).strip().lower() or "image"

    try:
        if mode == "video":
            job_context = _submit_video_job(form, web_config)
        else:
            job_context = _submit_image_job(form, web_config)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    requested_workflow = _canonicalize_workflow(_optional_text(form, "workflow"), fallback=_default_workflow_for_mode(mode))
    workflow = _canonicalize_workflow(job_context.get("job_type"), fallback=requested_workflow)

    return JSONResponse(
        {
            "job_id": job_context["job_id"],
            "workflow": workflow,
            "prompt": job_context.get("prompt", ""),
            "model": job_context.get("title", ""),
            "runs": 1,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "events_url": job_context.get("events_url", ""),
            "status_url": job_context.get("status_url", ""),
            "supported_controls": list(job_context.get("supported_controls", ())),
            "meta": job_context.get("meta", ""),
        }
    )


@app.post("/jobs/{job_id}/controls/{action}")
async def control_job(job_id: str, action: str) -> dict[str, Any]:
    """Queue a supported control action for an active Web job."""
    try:
        return web_runner.queue_job_control(job_id, action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    except UnsupportedJobControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/jobs/dummy", status_code=202)
async def create_dummy_job(steps: int = 5, delay_seconds: float = 0.25) -> dict[str, str | int]:
    """Start a dummy background job that emits skeleton progress events."""
    if steps < 1:
        raise HTTPException(status_code=400, detail="steps must be at least 1")
    if delay_seconds <= 0:
        raise HTTPException(status_code=400, detail="delay_seconds must be greater than 0")

    job_id = web_runner.submit_dummy_job(total_steps=steps, delay_seconds=delay_seconds)
    return {
        "job_id": job_id,
        "status_url": f"/jobs/{job_id}",
        "events_url": f"/jobs/{job_id}/events",
        "steps": steps,
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, object]:
    """Return a snapshot of a background job tracked by the web runner."""
    try:
        return web_runner.get_job_snapshot(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc


@app.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str) -> StreamingResponse:
    """Stream job progress as SSE frames for browser and programmatic consumers."""
    try:
        web_runner.get_job_snapshot(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc

    return StreamingResponse(
        web_runner.stream_job_events(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _build_workspace_bootstrap_view(web_config: WebUiConfig) -> dict[str, Any]:
    """Resolve per-model bootstrap defaults using the same config layering as the CLI."""
    image_default_model = _preferred_option(web_config.default_models.image, web_config.image_model_options)
    video_default_model = _preferred_option(web_config.default_models.video, web_config.video_model_options)
    image_defaults = {model_name: _build_image_bootstrap_defaults(model_name, web_config) for model_name in web_config.image_model_options}
    video_defaults = {model_name: _build_video_bootstrap_defaults(model_name, web_config) for model_name in web_config.video_model_options}
    return {
        "image_default_model": image_default_model,
        "video_default_model": video_default_model,
        "image_model_defaults": image_defaults,
        "video_model_defaults": video_defaults,
    }


def _preferred_option(preferred: str | None, options: tuple[str, ...]) -> str | None:
    """Return the preferred option when it exists, otherwise the first available item."""
    if preferred in options:
        return preferred
    return options[0] if options else None


def _resolve_ratio_and_size(
    preferred_ratio: str | None,
    ratios: tuple[str, ...],
    size_options_map: dict[str, tuple[str, ...]],
    preferred_size: str | None,
    *,
    fallback_ratio: str,
    fallback_size: str,
) -> tuple[str, str]:
    """Resolve the nearest valid ratio and size from config-backed options."""
    ratio = _preferred_option(preferred_ratio, ratios) or fallback_ratio
    size_options = size_options_map.get(ratio, ())
    size = _preferred_option(preferred_size, size_options) or fallback_size
    return ratio, size


def _resolve_image_bootstrap_ratio_size(web_config: WebUiConfig) -> tuple[str, str]:
    """Resolve image bootstrap ratio and size from layered config defaults."""
    app_config = web_config.app_config
    return _resolve_ratio_and_size(
        app_config.get("generation", {}).get("default_ratio"),
        web_config.image_ratios,
        web_config.image_size_options,
        app_config.get("generation", {}).get("default_size"),
        fallback_ratio="2:3",
        fallback_size="m",
    )


def _resolve_video_bootstrap_ratio_size(web_config: WebUiConfig) -> tuple[str, str]:
    """Resolve video bootstrap ratio and size from layered config defaults."""
    app_config = web_config.app_config
    return _resolve_ratio_and_size(
        app_config.get("video_generation", {}).get("default_ratio"),
        web_config.video_ratios,
        web_config.video_size_options,
        app_config.get("video_generation", {}).get("default_size"),
        fallback_ratio="16:9",
        fallback_size="m",
    )


def _resolve_image_bootstrap_dimensions(app_config: dict[str, Any], ratio: str, size: str) -> dict[str, int]:
    """Resolve image dimensions from config for the selected ratio and size."""
    dims = app_config.get("sizes", {}).get(ratio, {}).get(size, {})
    return {
        "width": dims.get("width", 1024),
        "height": dims.get("height", 1024),
    }


def _resolve_video_bootstrap_family(app_config: dict[str, Any], family: str | None) -> str:
    """Choose a config-backed video family for bootstrap fallback resolution."""
    if family and family != "unknown":
        return family
    video_sizes = app_config.get("video_sizes", {})
    if "ltx" in video_sizes:
        return "ltx"
    return next(iter(video_sizes), "ltx")


def _default_video_max_steps(app_config: dict[str, Any], family: str) -> int | None:
    """Expose an optional UI steps cap from the same preset family used by the CLI."""
    value = app_config.get("video_model_presets", {}).get(family, {}).get("default_steps")
    return value if isinstance(value, int) else None


def _image_bootstrap_postprocess() -> dict[str, Any]:
    """Return the default image postprocess bootstrap payload."""
    return dict(_IMAGE_BOOTSTRAP_POSTPROCESS)


def _image_bootstrap_upscale() -> dict[str, Any]:
    """Return the default image upscale bootstrap payload."""
    return dict(_IMAGE_BOOTSTRAP_UPSCALE)


def _video_bootstrap_upscale() -> dict[str, Any]:
    """Return the default video upscale bootstrap payload."""
    return dict(_VIDEO_BOOTSTRAP_UPSCALE)


def _build_image_bootstrap_defaults(model_name: str, web_config: WebUiConfig) -> dict[str, Any]:
    """Resolve image workflow bootstrap defaults for a single model option."""
    app_config = web_config.app_config
    ratio, size = _resolve_image_bootstrap_ratio_size(web_config)
    try:
        resolved_model = resolve_model_path(model_name, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
        model_info = detect_image_model(resolved_model)
        backend_name = "mflux" if sys.platform == "darwin" else "diffusers"
        defaults = resolve_defaults(model_info, app_config, {}, backend_name)
    except Exception:
        defaults = {
            "steps": app_config.get("generation", {}).get("default_steps", 10),
            "guidance": app_config.get("generation", {}).get("default_guidance", 3.5),
            "scheduler": None,
            "supports_negative_prompt": False,
        }
    dims = _resolve_image_bootstrap_dimensions(app_config, ratio, size)
    return {
        "ratio": ratio,
        "size": size,
        "width": dims["width"],
        "height": dims["height"],
        "steps": defaults["steps"],
        "guidance": defaults["guidance"],
        "scheduler": defaults.get("scheduler"),
        "supports_negative_prompt": bool(defaults.get("supports_negative_prompt", False)),
        "supports_quantize": bool(web_config.quantize_options),
        "quantize": None,
        "image_strength": _IMAGE_BOOTSTRAP_STRENGTH,
        "postprocess": _image_bootstrap_postprocess(),
        "upscale": _image_bootstrap_upscale(),
    }


def _build_video_bootstrap_defaults(model_name: str, web_config: WebUiConfig) -> dict[str, Any]:
    """Resolve video workflow bootstrap defaults for a single model option."""
    app_config = web_config.app_config
    ratio, size = _resolve_video_bootstrap_ratio_size(web_config)
    supports_i2v = False
    fps = 24
    try:
        resolved_model = resolve_model_path(model_name, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
        model_info = detect_video_model(resolved_model)
        family = _resolve_video_bootstrap_family(app_config, model_info.family)
        supports_i2v = model_info.supports_i2v
        fps_value = getattr(model_info, "default_fps", 24)
        fps = fps_value if isinstance(fps_value, int) else 24
    except Exception:
        family = _resolve_video_bootstrap_family(app_config, None)
    defaults = resolve_video_defaults(family, app_config, {"ratio": ratio, "size": size})
    return {
        "ratio": defaults.get("ratio", ratio),
        "size": defaults.get("size", size),
        "steps": defaults["steps"],
        "width": defaults["width"],
        "height": defaults["height"],
        "frame_count": defaults["num_frames"],
        "audio": True,
        "low_memory": True,
        "supports_i2v": supports_i2v,
        "supports_quantize": False,
        "quantize": None,
        "max_steps": _default_video_max_steps(app_config, family),
        "fps": fps,
        "upscale": _video_bootstrap_upscale(),
    }


def _canonicalize_workflow(value: Any, *, fallback: str | None = None) -> str | None:
    """Map backend and legacy workflow labels to the SPA-facing canonical vocabulary."""
    if value is None:
        return fallback
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    return _WORKFLOW_ALIASES.get(normalized, fallback)


def _default_workflow_for_mode(mode: str) -> str:
    return "txt2vid" if mode == "video" else "txt2img"


def _workflow_mode(workflow: str) -> str:
    return "video" if workflow in {"txt2vid", "img2vid"} else "image"


def _build_workflow_contract() -> dict[str, Any]:
    return {
        "values": list(_CANONICAL_WORKFLOW_VALUES),
        "legacy_aliases": dict(_WORKFLOW_ALIASES),
        "definitions": {name: dict(value) for name, value in _WORKFLOW_DEFINITIONS.items()},
        "field_precedence": {
            "defaults": ["cli", "model_variant", "model_family", "global"],
            "dimensions": "explicit_width_height_overrides_ratio_size",
        },
    }


def _submit_image_job(form: Any, web_config: WebUiConfig) -> dict[str, Any]:
    app_config = web_config.app_config
    workflow = _canonicalize_workflow(_optional_text(form, "workflow"), fallback="txt2img") or "txt2img"
    prompt = _required_text(form, "prompt")
    negative_prompt = _optional_text(form, "negative_prompt")
    model_name = _text_or_default(form, "model", web_config.default_models.image)
    if not model_name:
        raise ValueError("An image model is required.")

    output_dir = _resolve_output_dir(_text_or_default(form, "output", web_config.output_dir))
    args = argparse.Namespace(
        ratio=_choice_or_default(form, "ratio", web_config.image_ratios, app_config["generation"].get("default_ratio", "2:3")),
        size=None,
        width=_optional_int(form, "width"),
        height=_optional_int(form, "height"),
        runs=_optional_int(form, "runs") or 1,
        seed=_optional_int(form, "seed"),
        steps=_optional_int(form, "steps"),
        guidance=_optional_float(form, "guidance"),
        scheduler=_optional_text(form, "scheduler"),
        upscale=_optional_int(form, "upscale"),
        upscale_denoise=_optional_float(form, "upscale_denoise"),
        upscale_steps=_optional_int(form, "upscale_steps"),
        upscale_guidance=_optional_float(form, "upscale_guidance"),
        upscale_sharpen=_checkbox(form, "upscale_sharpen", default=True),
        upscale_save_pre=_checkbox(form, "upscale_save_pre"),
        image_path=_resolve_reference_image(form, output_dir),
        image_strength=_optional_float(form, "image_strength"),
        output=output_dir,
        model=model_name,
        quantize=_optional_int(form, "quantize"),
        sharpen=_resolve_numeric_toggle(form, "sharpen_enabled", "sharpen_amount", default_enabled=True),
        contrast=_resolve_numeric_toggle(form, "contrast_enabled", "contrast_amount", default_enabled=False),
        saturation=_resolve_numeric_toggle(form, "saturation_enabled", "saturation_amount", default_enabled=False),
        lora_paths=None,
        lora_weights=None,
    )
    args.size = _resolve_image_size(form, web_config, args.ratio)
    if args.image_strength is None:
        args.image_strength = 0.5
    if _WORKFLOW_DEFINITIONS[workflow]["requires_reference_image"] and args.image_path is None:
        raise ValueError("Image-to-image requires a reference image.")
    _validate_image_args(args, quantize_options=web_config.quantize_options)

    resolved_model = resolve_model_path(args.model, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
    model_info = detect_image_model(resolved_model)
    backend_name = "mflux" if sys.platform == "darwin" else "diffusers"
    defaults = resolve_defaults(
        model_info,
        app_config,
        {key: value for key, value in {"steps": args.steps, "guidance": args.guidance, "scheduler": args.scheduler}.items() if value is not None},
        backend_name,
    )
    args.steps = defaults["steps"]
    args.guidance = defaults["guidance"]
    args.scheduler = defaults["scheduler"]
    validate_scheduler(args.scheduler, app_config)
    if args.upscale and args.upscale_steps is None:
        args.upscale_steps = max(1, args.steps // 2)

    args.lora_paths, args.lora_weights = _resolve_loras(form)
    prompts_data = {"web": [(prompt, negative_prompt)]}
    dims = app_config["sizes"][args.ratio][args.size]

    request = ImageGenerationRequest(
        backend=None,
        model=None,
        prompt=prompt,
        model_name=model_name,
        model_family=model_info.family,
        supports_negative_prompt=defaults.get("supports_negative_prompt", False),
        lora_paths=args.lora_paths,
        lora_weights=args.lora_weights,
        negative_prompt=negative_prompt,
        ratio=args.ratio,
        size=args.size,
        width=args.width or dims["width"],
        height=args.height or dims["height"],
        seed=args.seed or 0,
        steps=args.steps,
        guidance=args.guidance,
        scheduler=args.scheduler,
        upscale_factor=args.upscale,
        upscale_denoise=args.upscale_denoise,
        upscale_steps=args.upscale_steps,
        upscale_guidance=args.upscale_guidance,
        upscale_sharpen=args.upscale_sharpen,
        upscale_save_pre=args.upscale_save_pre,
        image_path=args.image_path,
        image_strength=args.image_strength,
        sharpen=args.sharpen is not False,
        sharpen_amount_override=args.sharpen if isinstance(args.sharpen, float) else None,
        contrast=args.contrast is not False,
        contrast_amount=args.contrast if isinstance(args.contrast, float) else app_config.get("contrast", {}).get("default_amount", 1.0),
        saturation=args.saturation is not False,
        saturation_amount=args.saturation if isinstance(args.saturation, float) else app_config.get("saturation", {}).get("default_amount", 1.0),
        output_dir=args.output,
    )
    job_id = web_runner.submit_image_request_job(
        request=request,
        prompts_data=prompts_data,
        config=app_config,
        args=args,
        model_ref=resolved_model,
        quantize=args.quantize,
    )
    return {
        "job_id": job_id,
        "job_type": "Image to Image" if args.image_path else "Text to Image",
        "title": model_name,
        "prompt": prompt,
        "events_url": f"/jobs/{job_id}/events",
        "status_url": f"/jobs/{job_id}",
        "supported_controls": ("next", "pause", "repeat", "quit"),
        "meta": f"{args.ratio} · {args.size} · {args.steps} steps",
    }


def _submit_video_job(form: Any, web_config: WebUiConfig) -> dict[str, Any]:
    app_config = web_config.app_config
    workflow = _canonicalize_workflow(_optional_text(form, "workflow"), fallback="txt2vid") or "txt2vid"
    prompt = _required_text(form, "prompt")
    model_name = _text_or_default(form, "model", web_config.default_models.video)
    if not model_name:
        raise ValueError("A video model is required.")

    ratio = _choice_or_default(form, "ratio", web_config.video_ratios, app_config.get("video_generation", {}).get("default_ratio", "16:9"))
    size = _resolve_video_size(form, web_config, ratio)
    output_dir = _resolve_output_dir(_text_or_default(form, "output", web_config.output_dir))
    image_path = _resolve_reference_image(form, output_dir)
    audio_enabled = _checkbox(form, "audio", default=True)
    args = argparse.Namespace(
        model=model_name,
        prompt=prompt,
        prompts_file=None,
        image_path=image_path,
        ratio=ratio,
        size=size,
        width=_optional_int(form, "width"),
        height=_optional_int(form, "height"),
        frames=_optional_int(form, "frames"),
        num_frames=None,
        steps=_optional_int(form, "steps"),
        seed=_optional_int(form, "seed"),
        runs=_optional_int(form, "runs") or 1,
        low_memory=_checkbox(form, "low_memory", default=True),
        output=output_dir,
        format="mp4",
        lora=None,
        lora_paths=[],
        lora_weights=[],
        upscale=_optional_int(form, "upscale"),
        upscale_steps=None,
        no_audio=not audio_enabled,
        audio=audio_enabled,
    )
    if _WORKFLOW_DEFINITIONS[workflow]["requires_reference_image"] and image_path is None:
        raise ValueError("Image-to-video requires a reference image.")
    _validate_video_args(args)

    resolved_model = resolve_model_path(args.model, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
    model_info = detect_video_model(resolved_model)
    if model_info.family == "unknown":
        raise ValueError(f"Could not detect a supported video model for '{model_name}'.")
    if image_path and not model_info.supports_i2v:
        raise ValueError(f"Model '{model_name}' does not support image-to-video.")

    family_sizes = app_config.get("video_sizes", {}).get(model_info.family, {})
    if family_sizes:
        if args.ratio not in family_sizes:
            raise ValueError(f"Unknown ratio '{args.ratio}' for {model_info.family}. Valid: {list(family_sizes.keys())}")
        if args.size not in family_sizes.get(args.ratio, {}):
            raise ValueError(f"Unknown size '{args.size}' for ratio '{args.ratio}'. Valid: {list(family_sizes.get(args.ratio, {}).keys())}")

    args.lora_paths, args.lora_weights = _resolve_loras(form)
    cli_overrides = {
        key: value
        for key, value in {
            "ratio": args.ratio,
            "size": args.size,
            "steps": args.steps,
            "width": args.width,
            "height": args.height,
            "num_frames": args.frames,
        }.items()
        if value is not None
    }
    defaults = resolve_video_defaults(model_info.family, app_config, cli_overrides)
    args.steps = defaults["steps"]
    args.width = defaults["width"]
    args.height = defaults["height"]
    args.num_frames = defaults["num_frames"]
    _normalize_video_args(args, app_config, model_info)

    prompts_data = {"web": [(prompt, None)]}
    request = VideoGenerationRequest(
        backend=None,
        model=None,
        prompt=prompt,
        model_name=model_name,
        model_family=model_info.family,
        lora_paths=args.lora_paths,
        lora_weights=args.lora_weights,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        seed=args.seed or 0,
        steps=args.steps,
        image_path=image_path,
        upscale=args.upscale,
        upscale_steps=args.upscale_steps,
        no_audio=args.no_audio,
        output_dir=args.output,
        output_format=args.format,
    )
    job_id = web_runner.submit_video_request_job(
        request=request,
        prompts_data=prompts_data,
        config=app_config,
        args=args,
        model_ref=resolved_model,
    )
    mode_label = "Image to Video" if image_path else "Text to Video"
    return {
        "job_id": job_id,
        "job_type": mode_label,
        "title": model_name,
        "prompt": prompt,
        "events_url": f"/jobs/{job_id}/events",
        "status_url": f"/jobs/{job_id}",
        "supported_controls": (),
        "meta": f"{args.width}x{args.height} · {args.num_frames} frames · {args.steps} steps",
    }


def _required_text(form: Any, key: str) -> str:
    value = _optional_text(form, key)
    if not value:
        raise ValueError(f"Field '{key}' is required.")
    return value


def _optional_text(form: Any, key: str) -> str | None:
    value = form.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_or_default(form: Any, key: str, default: str | None) -> str | None:
    return _optional_text(form, key) or default


def _optional_int(form: Any, key: str) -> int | None:
    value = _optional_text(form, key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Field '{key}' must be an integer.") from exc


def _optional_float(form: Any, key: str) -> float | None:
    value = _optional_text(form, key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Field '{key}' must be a number.") from exc


def _checkbox(form: Any, key: str, *, default: bool = False) -> bool:
    if key not in form:
        return default
    values = getattr(form, "getlist", lambda _key: [form.get(_key, "")])(key)
    normalized = [str(value).strip().lower() for value in values]
    return any(value not in {"", "0", "false", "off", "no"} for value in normalized)


def _choice_or_default(form: Any, key: str, choices: tuple[str, ...], default: str) -> str:
    value = _optional_text(form, key) or default
    if value not in choices:
        raise ValueError(f"Field '{key}' must be one of {list(choices)}.")
    return value


def _resolve_image_size(form: Any, web_config: WebUiConfig, ratio: str) -> str:
    choices = web_config.image_size_options[ratio]
    default = web_config.app_config["generation"].get("default_size", "m")
    return _choice_or_default(form, "size", choices, default)


def _resolve_video_size(form: Any, web_config: WebUiConfig, ratio: str) -> str:
    choices = web_config.video_size_options[ratio]
    default = web_config.app_config.get("video_generation", {}).get("default_size", "m")
    return _choice_or_default(form, "size", choices, default)


def _optional_path(form: Any, key: str) -> str | None:
    value = _optional_text(form, key)
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"Path '{path}' for '{key}' does not exist.")
    return str(path)


def _required_path(form: Any, key: str) -> str:
    """Return a validated path field or raise a field-specific error."""
    value = _optional_path(form, key)
    if value is None:
        raise ValueError(f"Field '{key}' is required.")
    return value


def _resolve_reference_image(form: Any, output_dir: str) -> str | None:
    uploaded_file = form.get("image_file")
    if _is_uploaded_file(uploaded_file):
        return _save_uploaded_reference_image(uploaded_file, output_dir)
    return _optional_path(form, "image_path")


def _is_uploaded_file(value: Any) -> bool:
    return bool(getattr(value, "filename", None) and getattr(value, "file", None))


def _save_uploaded_reference_image(uploaded_file: Any, output_dir: str) -> str:
    original_name = Path(str(uploaded_file.filename)).name
    suffix = Path(original_name).suffix.lower() or ".png"
    if suffix not in _IMAGE_EXTENSIONS:
        raise ValueError("Reference image must be a PNG, JPEG, or WebP file.")

    upload_dir = Path(output_dir) / ".web_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    candidate = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    uploaded_file.file.seek(0)
    candidate.write_bytes(uploaded_file.file.read())
    try:
        with Image.open(candidate) as image:
            image.verify()
    except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as exc:
        candidate.unlink(missing_ok=True)
        raise ValueError("Uploaded reference image is invalid or unreadable.") from exc
    return str(candidate)


def _resolve_output_dir(value: str | None) -> str:
    if value is None:
        raise ValueError("An output directory is required.")
    output_dir = Path(value).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def _resolve_numeric_toggle(form: Any, enabled_key: str, amount_key: str, *, default_enabled: bool) -> bool | float:
    enabled = _checkbox(form, enabled_key, default=default_enabled)
    amount = _optional_float(form, amount_key)
    if not enabled:
        return False
    if amount is None:
        return True
    if amount < 0:
        raise ValueError(f"Field '{amount_key}' must be non-negative.")
    return amount


def _resolve_loras(form: Any) -> tuple[list[str] | None, list[float] | None]:
    lora_value = _optional_text(form, "lora")
    if lora_value is None:
        return None, None
    try:
        parsed = parse_lora_arg(lora_value)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return (
        [resolve_lora_path(name) for name, _ in parsed],
        [weight for _, weight in parsed],
    )


def _validate_image_args(args: argparse.Namespace, *, quantize_options: tuple[int, ...] = (4, 8)) -> None:
    if args.runs < 1:
        raise ValueError("Runs must be at least 1.")
    if args.quantize is not None and args.quantize not in quantize_options:
        raise ValueError(f"Quantize must be one of {list(quantize_options)}.")
    if args.width is not None and args.width <= 0:
        raise ValueError("Width must be positive.")
    if args.height is not None and args.height <= 0:
        raise ValueError("Height must be positive.")
    if args.width is not None and args.width % 16 != 0:
        raise ValueError("Width must be a multiple of 16.")
    if args.height is not None and args.height % 16 != 0:
        raise ValueError("Height must be a multiple of 16.")
    if args.upscale is not None and args.upscale not in (2, 4):
        raise ValueError("Upscale must be 2 or 4.")
    if args.upscale_denoise is not None and not (0.0 <= args.upscale_denoise <= 1.0):
        raise ValueError("Upscale denoise must be between 0.0 and 1.0.")
    if args.steps is not None and args.steps < 1:
        raise ValueError("Steps must be at least 1.")
    if args.upscale_steps is not None and args.upscale_steps < 1:
        raise ValueError("Upscale steps must be at least 1.")
    if args.guidance is not None and args.guidance < 0:
        raise ValueError("Guidance must be non-negative.")
    if args.upscale_guidance is not None and args.upscale_guidance < 0:
        raise ValueError("Upscale guidance must be non-negative.")
    if isinstance(args.sharpen, float) and args.sharpen < 0:
        raise ValueError("Sharpen amount must be non-negative.")
    if isinstance(args.contrast, float) and args.contrast < 0:
        raise ValueError("Contrast amount must be non-negative.")
    if isinstance(args.saturation, float) and args.saturation < 0:
        raise ValueError("Saturation amount must be non-negative.")
    if args.upscale is not None:

        def _round16(value: int) -> int:
            return ((value + 15) // 16) * 16

        for dim_name, dim_val in (("Width", args.width), ("Height", args.height)):
            if dim_val is None:
                continue
            base = dim_val // args.upscale
            final = _round16(base) * args.upscale
            if final != dim_val:
                raise ValueError(f"{dim_name} {dim_val} is not compatible with upscale {args.upscale}: base size {base} rounds to {_round16(base)}, giving final size {final} instead of {dim_val}.")
    if not (0.0 <= args.image_strength <= 1.0):
        raise ValueError("Image strength must be between 0.0 and 1.0.")


def _validate_video_args(args: argparse.Namespace) -> None:
    if args.runs < 1:
        raise ValueError("Runs must be at least 1.")
    if args.steps is not None and args.steps < 1:
        raise ValueError("Steps must be at least 1.")
    if args.upscale is not None and args.upscale != 2:
        raise ValueError("Upscale only supports factor 2 (LTX spatial upscaler).")


def _normalize_video_args(args: argparse.Namespace, config: dict[str, Any], model_info: Any) -> None:
    steps_explicitly_set = args.steps is not None
    if args.upscale:
        upscale_cfg = config.get("video_model_presets", {}).get("ltx", {}).get("upscale", {})
        if not steps_explicitly_set:
            args.steps = upscale_cfg.get("default_upscale_steps", 8)
        args.upscale_steps = args.steps
    else:
        args.upscale_steps = None

    if args.steps < 1:
        raise ValueError("Steps must be at least 1.")

    if model_info.family == "ltx" and args.steps > 8:
        args.steps = 8
        if args.upscale_steps is not None:
            args.upscale_steps = args.steps

    alignment = 64 if args.upscale else model_info.resolution_alignment
    args.width, args.height = _align_resolution(args.width, args.height, alignment, model_info.family.upper())
    args.num_frames = _align_ltx_frames(args.num_frames, model_info.frame_alignment)
    if args.width < 64 or args.height < 64:
        raise ValueError(f"Resolved dimensions {args.width}x{args.height} are too small (minimum 64x64)")


def _list_gallery_assets(output_dir: str) -> list[GalleryAsset]:
    root = Path(output_dir)
    if not root.exists():
        return []

    assets: list[GalleryAsset] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        asset = _build_gallery_asset(root, candidate)
        if asset is not None:
            assets.append(asset)
    assets.sort(key=lambda item: item.modified_at, reverse=True)
    return assets


def _build_gallery_asset(root: Path, candidate: Path) -> GalleryAsset | None:
    """Build gallery metadata for a single media asset under the output root."""
    kind = _asset_kind(candidate)
    if kind is None:
        return None
    relative_path = candidate.relative_to(root).as_posix()
    metadata = _read_asset_metadata(candidate, kind)
    return GalleryAsset(
        name=candidate.name,
        kind=kind,
        extension=candidate.suffix.lower().lstrip("."),
        filesystem_path=str(candidate),
        modified_at=candidate.stat().st_mtime,
        modified_label=_format_age(candidate.stat().st_mtime),
        path_label=relative_path,
        media_url=f"/media/{quote(relative_path)}",
        detail_url="",
        reuse_workspace_url="",
        reuse_settings_url="",
        prompt=metadata["prompt"],
        model_label=metadata["model_label"],
        width=metadata["width"],
        height=metadata["height"],
        seed=metadata["seed"],
        steps=metadata["steps"],
        guidance=metadata["guidance"],
        dimensions_label=metadata["dimensions_label"],
        seed_label=metadata["seed_label"],
        steps_label=metadata["steps_label"],
        guidance_label=metadata["guidance_label"],
        workflow=metadata["workflow"],
        ratio=metadata["ratio"],
        size=metadata["size"],
        frame_count=metadata["frame_count"],
        reference_image_path=metadata["reference_image_path"],
        lora=metadata["lora"],
    )


def _read_asset_metadata(asset_path: Path, kind: str) -> dict[str, Any]:
    """Read best-effort display metadata for a gallery asset."""
    sidecar_metadata = _read_json_sidecar(asset_path)
    filename_metadata = _parse_generated_filename(asset_path)
    image_metadata = _read_image_metadata(asset_path) if kind == "image" else {}

    prompt = _coerce_text(
        _metadata_value(sidecar_metadata, "prompt", "prompt_text", "resolved_prompt", "description", "positive_prompt") or image_metadata.get("prompt") or asset_path.stem.replace("_", " ")
    )
    model_label = _coerce_text(_metadata_value(sidecar_metadata, "model_name", "model", "model_ref", "model_id") or kind.capitalize())

    width = _coerce_int(_metadata_value(sidecar_metadata, "width") or image_metadata.get("width") or filename_metadata.get("width"))
    height = _coerce_int(_metadata_value(sidecar_metadata, "height") or image_metadata.get("height") or filename_metadata.get("height"))
    seed = _coerce_int(_metadata_value(sidecar_metadata, "seed") or filename_metadata.get("seed"))
    steps = _coerce_int(_metadata_value(sidecar_metadata, "steps", "num_inference_steps", "inference_steps") or filename_metadata.get("steps"))
    guidance = _coerce_float(_metadata_value(sidecar_metadata, "guidance", "guidance_scale", "cfg", "cfg_scale") or filename_metadata.get("guidance"))
    workflow = _coerce_optional_text(_metadata_value(sidecar_metadata, "workflow", "workflow_name", "job_type", "mode"))
    ratio = _coerce_optional_text(_metadata_value(sidecar_metadata, "ratio", "aspect_ratio"))
    size = _coerce_optional_text(_metadata_value(sidecar_metadata, "size", "resolution_size"))
    frame_count = _coerce_int(_metadata_value(sidecar_metadata, "frame_count", "frames", "num_frames"))
    reference_image_path = _coerce_optional_text(_metadata_value(sidecar_metadata, "image_path", "reference_image", "reference_image_path", "input_image", "source_image"))
    lora = _coerce_lora_string(_metadata_value(sidecar_metadata, "lora", "loras"))

    return {
        "prompt": prompt,
        "model_label": model_label,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "dimensions_label": f"{width}x{height}" if width is not None and height is not None else "Unavailable",
        "seed_label": str(seed) if seed is not None else "Unavailable",
        "steps_label": str(steps) if steps is not None else "Unavailable",
        "guidance_label": _format_guidance(guidance),
        "workflow": workflow,
        "ratio": ratio,
        "size": size,
        "frame_count": frame_count,
        "reference_image_path": reference_image_path,
        "lora": lora,
    }


def _read_json_sidecar(asset_path: Path) -> dict[str, Any]:
    """Read adjacent JSON metadata using the common sidecar naming patterns."""
    candidates = [asset_path.with_suffix(".json"), asset_path.with_name(f"{asset_path.name}.json")]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except OSError, UnicodeDecodeError, json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _read_image_metadata(asset_path: Path) -> dict[str, Any]:
    """Read embedded prompt and dimensions from an image when possible."""
    try:
        with Image.open(asset_path) as image:
            exif = image.getexif()
            return {
                "width": image.width,
                "height": image.height,
                "prompt": image.info.get("Description") or exif.get(0x010E),
            }
    except FileNotFoundError, OSError, UnidentifiedImageError, ValueError:
        return {}


def _parse_generated_filename(asset_path: Path) -> dict[str, Any]:
    """Parse shared generated filename tokens when sidecar metadata is absent."""
    match = re.search(
        r"_(?P<width>\d+)x(?P<height>\d+)(?:_\d+f)?(?:_.*?)?_steps(?P<steps>\d+)(?:_cfg(?P<guidance>[-+]?\d+(?:\.\d+)?))?_seed(?P<seed>\d+)",
        asset_path.stem,
    )
    if match is None:
        return {}
    parsed = match.groupdict()
    return {
        "width": _coerce_int(parsed.get("width")),
        "height": _coerce_int(parsed.get("height")),
        "steps": _coerce_int(parsed.get("steps")),
        "guidance": _coerce_float(parsed.get("guidance")),
        "seed": _coerce_int(parsed.get("seed")),
    }


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    """Find a key in nested metadata mappings."""
    if not metadata:
        return None
    for mapping in _walk_mappings(metadata):
        for key in keys:
            if key in mapping and mapping[key] not in (None, ""):
                return mapping[key]
    return None


def _walk_mappings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return nested mappings in depth-first order for metadata lookup."""
    mappings = [payload]
    for value in payload.values():
        if isinstance(value, dict):
            mappings.extend(_walk_mappings(value))
    return mappings


def _coerce_text(value: Any) -> str:
    """Convert a metadata value to a non-empty string."""
    if value is None:
        return "Unavailable"
    text = str(value).strip()
    return text or "Unavailable"


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: Any) -> int | None:
    """Convert a metadata value to an integer when possible."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _coerce_float(value: Any) -> float | None:
    """Convert a metadata value to a float when possible."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _coerce_lora_string(value: Any) -> str | None:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        entries: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    entries.append(text)
                continue
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                weight = item.get("weight")
                if weight in (None, ""):
                    entries.append(item["name"].strip())
                else:
                    entries.append(f"{item['name'].strip()}:{weight}")
        return ",".join(entry for entry in entries if entry) or None
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        weight = value.get("weight")
        if weight in (None, ""):
            return value["name"].strip() or None
        return f"{value['name'].strip()}:{weight}"
    return None


def _format_guidance(value: float | None) -> str:
    """Format a guidance value for gallery display."""
    if value is None:
        return "Unavailable"
    return f"{value:g}"


def _paginate_assets(assets: list[GalleryAsset], *, page: int, page_size: int) -> tuple[list[GalleryAsset], int | None]:
    start = (page - 1) * page_size
    end = start + page_size
    next_page = page + 1 if end < len(assets) else None
    return assets[start:end], next_page


def _filter_and_sort_assets(assets: list[GalleryAsset], *, media_filter: str, sort_order: str) -> list[GalleryAsset]:
    """Apply gallery filter and sort controls to a list of assets."""
    normalized_filter = media_filter.strip().lower()
    if normalized_filter not in {"all", "image", "video"}:
        normalized_filter = "all"
    normalized_sort = sort_order.strip().lower()
    if normalized_sort not in {"newest", "oldest"}:
        normalized_sort = "newest"

    filtered = [asset for asset in assets if normalized_filter == "all" or asset.kind == normalized_filter]
    return sorted(filtered, key=lambda asset: asset.modified_at, reverse=normalized_sort == "newest")


def _asset_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    return None


def _resolve_selected_asset(assets: list[GalleryAsset], selected: str | None) -> GalleryAsset | None:
    if not assets:
        return None
    if selected is None:
        return assets[0]
    for asset in assets:
        if asset.path_label == selected:
            return asset
    return assets[0]


def _delete_gallery_assets(output_dir: str, selected_paths: list[str]) -> None:
    """Delete selected gallery assets and adjacent JSON sidecars."""
    root = Path(output_dir).resolve()
    for relative_path in dict.fromkeys(path for path in selected_paths if path.strip()):
        candidate = _resolve_output_asset_path(root, relative_path)
        if candidate is None or not candidate.is_file():
            continue
        candidate.unlink(missing_ok=True)
        for sidecar in (candidate.with_suffix(".json"), candidate.with_name(f"{candidate.name}.json")):
            sidecar.unlink(missing_ok=True)


def _resolve_output_asset_path(root: Path, relative_path: str) -> Path | None:
    """Resolve a relative output asset path safely under the configured root."""
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate


def _build_config_view(web_config: WebUiConfig) -> dict[str, Any]:
    app_config = web_config.app_config
    default_ratio = app_config.get("generation", {}).get("default_ratio", web_config.image_ratios[0])
    image_sizes = app_config.get("sizes", {}).get(default_ratio, {})
    image_size_labels = [
        {
            "value": size_name,
            "label": f"{size_name} ({size_config['width']}x{size_config['height']})",
        }
        for size_name, size_config in image_sizes.items()
    ]
    token_var = _huggingface_token_env_var()
    data_dir = Path(web_config.output_dir).expanduser().parent
    model_cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface" / "hub")
    return {
        "image_size_labels": image_size_labels,
        "model_cache_dir": model_cache_dir,
        "lora_dir": str(data_dir / "loras"),
        "huggingface_token_configured": token_var is not None,
        "huggingface_token_env_var": token_var,
        "default_image_size": app_config.get("generation", {}).get("default_size", "m"),
    }


def _persist_web_config(form: Any) -> None:
    """Persist the subset of config fields the app actually consumes today."""
    current = load_web_config()
    override_config = _read_user_config_override()

    image_model = _optional_text(form, "ui.default_models.image")
    if image_model:
        if image_model not in current.image_model_options:
            raise ValueError("Default image model must be one of the discovered image models.")
        _set_nested_mapping_value(override_config, ("ui", "default_models", "image"), image_model)

    video_model = _optional_text(form, "ui.default_models.video")
    if video_model:
        if video_model not in current.video_model_options:
            raise ValueError("Default video model must be one of the discovered video models.")
        _set_nested_mapping_value(override_config, ("ui", "default_models", "video"), video_model)

    default_image_size = _optional_text(form, "generation.default_size")
    default_ratio = current.app_config.get("generation", {}).get("default_ratio", current.image_ratios[0])
    valid_sizes = current.image_size_options.get(default_ratio, ())
    if default_image_size:
        if default_image_size not in valid_sizes:
            raise ValueError(f"Default base resolution must be one of {list(valid_sizes)} for ratio '{default_ratio}'.")
        _set_nested_mapping_value(override_config, ("generation", "default_size"), default_image_size)

    output_dir = _optional_text(form, "ui.output_dir")
    if output_dir:
        _set_nested_mapping_value(override_config, ("ui", "output_dir"), _normalize_user_directory(output_dir))

    _write_user_config_override(override_config)


def _convert_model_from_form(form: Any) -> dict[str, str]:
    """Validate and run checkpoint conversion from the Web models page."""
    input_path = _required_path(form, "input_path")
    model_type = _optional_text(form, "model_type") or "zimage"
    if model_type not in {"zimage", "flux2-klein-4b", "flux2-klein-9b"}:
        raise ValueError("Model type must be one of zimage, flux2-klein-4b, or flux2-klein-9b.")

    args = ["model", "--input", input_path, "--model-type", model_type]
    model_name = _optional_text(form, "name")
    if model_name:
        args.extend(["--name", model_name])
    if model_type == "zimage":
        args.extend(["--base-model", _optional_text(form, "base_model") or "Tongyi-MAI/Z-Image-Turbo"])
    if _checkbox(form, "copy"):
        args.append("--copy")

    detail = _run_model_management_command(args)
    return {
        "tone": "success",
        "message": "Converted the checkpoint into an installed model directory.",
        "detail": detail,
    }


def _import_local_lora_from_form(form: Any) -> dict[str, str]:
    """Validate and import a LoRA from a local safetensors file."""
    source_path = Path(_required_path(form, "source_path"))
    imported = import_lora_local(source_path, get_ziv_data_dir() / "loras", name=_optional_text(form, "name"))
    return {
        "tone": "success",
        "message": f"Imported local LoRA '{imported.stem}'.",
        "detail": str(imported),
    }


def _import_hf_lora_from_form(form: Any) -> dict[str, str]:
    """Validate and import a LoRA from Hugging Face."""
    repo_id = _required_text(form, "repo_id")
    imported = import_lora_hf(
        repo_id,
        get_ziv_data_dir() / "loras",
        filename=_optional_text(form, "filename"),
        name=_optional_text(form, "name"),
    )
    return {
        "tone": "success",
        "message": f"Imported Hugging Face LoRA '{imported.stem}'.",
        "detail": str(imported),
    }


def _run_model_management_command(args: list[str]) -> str:
    """Run the existing model-management CLI and return a concise status summary."""
    result = subprocess.run(
        [sys.executable, "-m", "zvisiongenerator.converters.convert_checkpoint", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "").strip()
    error_output = (result.stderr or "").strip()
    if result.returncode != 0:
        message = error_output or output or "Model-management command failed."
        raise RuntimeError(message)

    for line in output.splitlines():
        if line.startswith("Conversion complete!") or line.startswith("LoRA imported:"):
            return line
    return output.splitlines()[-1] if output else "Operation completed successfully."


def _huggingface_token_env_var() -> str | None:
    """Return the active Hugging Face token variable name when configured."""
    for name in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(name):
            return name
    return None


def _normalize_user_directory(value: str) -> str:
    """Normalize a user-provided directory path for persisted config."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _read_user_config_override() -> dict[str, Any]:
    """Read the mutable user config override file, if present."""
    config_path = get_ziv_data_dir() / "config.yaml"
    if not config_path.is_file():
        return {}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Failed to read user config override: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("User config override must be a YAML mapping.")
    return payload


def _write_user_config_override(payload: dict[str, Any]) -> None:
    """Write the mutable user config override file."""
    config_path = get_ziv_data_dir() / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _set_nested_mapping_value(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Assign a nested mapping value, creating intermediate dicts as needed."""
    cursor = payload
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pick_directory(initial_dir: str | None) -> str | None:
    if sys.platform == "darwin":
        return _pick_directory_macos(initial_dir)
    return _pick_directory_tk(initial_dir)


def _pick_directory_macos(initial_dir: str | None) -> str | None:
    script = "set chosenFolder to choose folder\nPOSIX path of chosenFolder"
    if initial_dir:
        candidate = Path(initial_dir).expanduser()
        if candidate.exists():
            escaped_dir = str(candidate).replace('"', '\\"')
            script = f'set defaultLocation to POSIX file "{escaped_dir}"\nset chosenFolder to choose folder default location defaultLocation\nPOSIX path of chosenFolder'
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().lower()
        if "user canceled" in stderr or "cancelled" in stderr:
            return None
        raise RuntimeError(result.stderr.strip() or "Native directory picker failed.")
    selected = result.stdout.strip()
    return selected or None


def _pick_directory_tk(initial_dir: str | None) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("Native directory picker is unavailable in this environment.") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askdirectory(initialdir=initial_dir or str(Path.home()), mustexist=True)
    finally:
        root.destroy()
    return selected or None


def _format_age(timestamp: float) -> str:
    seconds = max(0, int(time() - timestamp))
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


# ─── JSON API endpoints (Svelte SPA) ─────────────────────────────────────────


def _gallery_asset_to_json(asset: GalleryAsset, web_config: WebUiConfig) -> dict[str, Any]:
    """Convert a GalleryAsset to the JSON shape expected by the Svelte SPA."""
    created_at = datetime.fromtimestamp(asset.modified_at, tz=timezone.utc).isoformat()
    default_workflow = _default_workflow_for_mode(asset.kind)
    requested_workflow = _canonicalize_workflow(asset.workflow, fallback=default_workflow)
    fallback_reasons: list[str] = []
    workflow_available = True
    if _workflow_mode(requested_workflow) != asset.kind:
        requested_workflow = default_workflow
        workflow_available = False
        fallback_reasons.append("workflow_media_mismatch")

    resolved_workflow = requested_workflow
    if _WORKFLOW_DEFINITIONS[resolved_workflow]["requires_reference_image"] and asset.reference_image_path is None:
        resolved_workflow = default_workflow
        workflow_available = False
        fallback_reasons.append("missing_reference_image")

    model_options = web_config.image_model_options if _workflow_mode(resolved_workflow) == "image" else web_config.video_model_options
    default_model = _preferred_option(
        web_config.default_models.image if _workflow_mode(resolved_workflow) == "image" else web_config.default_models.video,
        model_options,
    )
    requested_model = None if asset.model_label == "Unavailable" else asset.model_label
    model_available = requested_model is None or requested_model in model_options
    resolved_model = requested_model if requested_model in model_options else default_model
    if requested_model is not None and not model_available:
        fallback_reasons.append("model_not_configured")

    reuse_params: dict[str, str] = {
        "workflow": resolved_workflow,
        "prompt": asset.prompt,
    }
    if resolved_model:
        reuse_params["model"] = resolved_model
    if asset.lora:
        reuse_params["lora"] = asset.lora
    if asset.steps is not None:
        reuse_params["steps"] = str(asset.steps)
    if asset.guidance is not None and _workflow_mode(resolved_workflow) == "image":
        reuse_params["guidance"] = f"{asset.guidance:g}"
    if asset.seed is not None:
        reuse_params["seed"] = str(asset.seed)
    if asset.ratio is not None:
        reuse_params["ratio"] = asset.ratio
    if asset.size is not None:
        reuse_params["size"] = asset.size
    if asset.width is not None:
        reuse_params["width"] = str(asset.width)
    if asset.height is not None:
        reuse_params["height"] = str(asset.height)
    if asset.frame_count is not None and _workflow_mode(resolved_workflow) == "video":
        reuse_params["frames"] = str(asset.frame_count)
    if _WORKFLOW_DEFINITIONS[resolved_workflow]["requires_reference_image"] and asset.reference_image_path is not None:
        reuse_params["image_path"] = asset.reference_image_path
    return {
        "path": asset.filesystem_path,
        "url": asset.media_url,
        "thumbnail_url": asset.media_url,
        "filename": asset.name,
        "created_at": created_at,
        "workflow": requested_workflow,
        "prompt": asset.prompt,
        "model": asset.model_label,
        "width": asset.width,
        "height": asset.height,
        "ratio": asset.ratio,
        "size": asset.size,
        "frame_count": asset.frame_count,
        "image_path": asset.reference_image_path,
        "media_type": asset.kind,
        "reuse_state": {
            "requested_workflow": requested_workflow,
            "resolved_workflow": resolved_workflow,
            "workflow_available": workflow_available,
            "requested_model": requested_model,
            "resolved_model": resolved_model,
            "model_available": model_available,
            "fallback_reasons": fallback_reasons,
        },
        "reuse_workspace_url": f"#/workspace?{urlencode(reuse_params)}",
    }


def _build_gallery_page_json(assets: list[GalleryAsset], web_config: WebUiConfig, *, page: int, page_size: int) -> dict[str, Any]:
    """Build a paginated GalleryPage response dict."""
    total_count = len(assets)
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    page_assets, _ = _paginate_assets(assets, page=page, page_size=page_size)
    return {
        "assets": [_gallery_asset_to_json(a, web_config) for a in page_assets],
        "page": page,
        "total_pages": total_pages,
        "total_count": total_count,
    }


def _build_api_config_response(web_config: WebUiConfig) -> dict[str, Any]:
    """Build a JSON-serializable AppConfig dict from WebUiConfig."""
    config_view = _build_config_view(web_config)
    return {
        "output_dir": web_config.output_dir,
        "log_level": "info",
        "ui": {
            "visible_sections": list(web_config.visible_sections),
            "theme": web_config.theme,
            "startup_view": web_config.startup_view,
            "gallery_page_size": web_config.gallery_page_size,
            "output_dir": web_config.output_dir,
            "default_models": {
                "image": web_config.default_models.image,
                "video": web_config.default_models.video,
            },
            "image_model_options": list(web_config.image_model_options),
            "video_model_options": list(web_config.video_model_options),
            "model_cache_dir": config_view["model_cache_dir"],
            "loras_dir": config_view["lora_dir"],
            "huggingface_token_configured": config_view["huggingface_token_configured"],
            "huggingface_token_env_var": config_view["huggingface_token_env_var"],
            "image_size_labels": config_view["image_size_labels"],
            "default_image_size": config_view["default_image_size"],
        },
        "models": {},
    }


@app.get("/api/workspace")
async def api_workspace() -> dict[str, Any]:
    """Return WorkspaceContext JSON for the Svelte SPA."""
    web_config = load_web_config()
    data_dir = get_ziv_data_dir()

    all_assets = _list_gallery_assets(web_config.output_dir)
    history_assets = [_gallery_asset_to_json(a, web_config) for a in all_assets[:20]]

    image_models = [{"id": name, "label": name, "type": "image"} for name in web_config.image_model_options]
    video_models = [{"id": name, "label": name, "type": "video"} for name in web_config.video_model_options]
    loras_dir = data_dir / "loras"
    loras = [{"name": name, "path": str(loras_dir / f"{name}.safetensors")} for name in web_config.lora_options]

    form_view = _build_workspace_bootstrap_view(web_config)
    image_default_model = form_view["image_default_model"]
    video_default_model = form_view["video_default_model"]
    image_model_defaults_map = form_view["image_model_defaults"]
    video_model_defaults_map = form_view["video_model_defaults"]
    image_defaults = image_model_defaults_map.get(image_default_model) or _build_image_bootstrap_defaults(image_default_model or "", web_config)
    video_defaults = video_model_defaults_map.get(video_default_model) or _build_video_bootstrap_defaults(video_default_model or "", web_config)

    return {
        "image_models": image_models,
        "video_models": video_models,
        "loras": loras,
        "history_assets": history_assets,
        "defaults": image_defaults,
        "video_defaults": video_defaults,
        "image_model_defaults": image_model_defaults_map,
        "video_model_defaults": video_model_defaults_map,
        "current_image_model": image_default_model,
        "current_video_model": video_default_model,
        "output_dir": web_config.output_dir,
        "quantize_options": list(web_config.quantize_options),
        "image_ratios": list(web_config.image_ratios),
        "video_ratios": list(web_config.video_ratios),
        "image_size_options": {ratio: list(options) for ratio, options in web_config.image_size_options.items()},
        "video_size_options": {ratio: list(options) for ratio, options in web_config.video_size_options.items()},
        "scheduler_options": list(web_config.scheduler_options),
        "workflow_contract": _build_workflow_contract(),
        "config": {
            "visible_sections": list(web_config.visible_sections),
            "theme": web_config.theme,
            "gallery_page_size": web_config.gallery_page_size,
            "startup_view": web_config.startup_view,
            "output_dir": web_config.output_dir,
            "default_models": {
                "image": web_config.default_models.image,
                "video": web_config.default_models.video,
            },
        },
    }


@app.get("/api/history")
async def api_history(
    page: int = Query(1, ge=1),
    media_filter: str = Query("all"),
    sort_order: str = Query("newest"),
) -> dict[str, Any]:
    """Return paginated gallery history for the Svelte SPA."""
    web_config = load_web_config()
    all_assets = _filter_and_sort_assets(
        _list_gallery_assets(web_config.output_dir),
        media_filter=media_filter,
        sort_order=sort_order,
    )
    return _build_gallery_page_json(all_assets, web_config, page=page, page_size=web_config.gallery_page_size)


@app.get("/api/gallery")
async def api_gallery_json(
    page: int = Query(1, ge=1),
    filter: str = Query("all"),
    sort_order: str = Query("newest"),
) -> dict[str, Any]:
    """Return paginated gallery assets as JSON for the Svelte SPA gallery view."""
    web_config = load_web_config()
    all_assets = _filter_and_sort_assets(
        _list_gallery_assets(web_config.output_dir),
        media_filter=filter,
        sort_order=sort_order,
    )
    return _build_gallery_page_json(all_assets, web_config, page=page, page_size=web_config.gallery_page_size)


@app.get("/api/config")
async def api_get_config() -> dict[str, Any]:
    """Return current Web UI config as JSON for the Svelte SPA."""
    web_config = load_web_config()
    return _build_api_config_response(web_config)


@app.post("/api/config")
async def api_save_config(request: Request) -> dict[str, Any]:
    """Update Web UI config from a JSON body and return the updated config."""
    payload = await request.json()
    try:
        _persist_web_config(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    web_config = load_web_config()
    return _build_api_config_response(web_config)


@app.get("/api/models")
async def api_models() -> dict[str, Any]:
    """Return installed model inventory as JSON for the Svelte SPA."""
    data_dir = get_ziv_data_dir()
    token_var = _huggingface_token_env_var()
    image_models = [{"name": m.name, "family": m.family, "size": m.size} for m in list_models(data_dir)]
    video_models = [{"name": m.name, "family": m.family, "supports_i2v": m.supports_i2v} for m in list_video_models(data_dir)]
    loras = [{"name": lora.name, "file_size_mb": lora.file_size_mb, "size_label": f"{lora.file_size_mb} MB"} for lora in list_loras(data_dir)]
    return {
        "models_dir": str(data_dir / "models"),
        "loras_dir": str(data_dir / "loras"),
        "image_models": image_models,
        "video_models": video_models,
        "loras": loras,
        "available_surfaces": [],
        "huggingface_configured": token_var is not None,
        "huggingface_token_env_var": token_var,
    }


@app.post("/api/models/convert")
async def api_models_convert(request: Request) -> dict[str, Any]:
    """Convert a local checkpoint into an installed model directory (JSON API)."""
    payload = await request.json()
    try:
        notice = _convert_model_from_form(payload)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "tone": notice["tone"], "message": notice["message"]}


@app.post("/api/models/import-lora/local")
async def api_models_import_lora_local(request: Request) -> dict[str, Any]:
    """Import a local LoRA file into the configured data directory (JSON API)."""
    payload = await request.json()
    try:
        notice = _import_local_lora_from_form(payload)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "tone": notice["tone"], "message": notice["message"]}


@app.post("/api/models/import-lora/hf")
async def api_models_import_lora_hf(request: Request) -> dict[str, Any]:
    """Import a LoRA from Hugging Face into the configured data directory (JSON API)."""
    payload = await request.json()
    try:
        notice = _import_hf_lora_from_form(payload)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "tone": notice["tone"], "message": notice["message"]}


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel_job(job_id: str) -> dict[str, str]:
    """Cancel a running job. Returns cancelled status or 404 if the job is unknown."""
    try:
        web_runner.get_job_snapshot(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    try:
        web_runner.queue_job_control(job_id, "quit")
    except UnsupportedJobControlError, KeyError:
        pass
    return {"status": "cancelled"}


@app.delete("/api/gallery/{asset_path:path}")
async def api_delete_gallery_asset(asset_path: str) -> dict[str, str]:
    """Delete a single gallery asset by relative path under the output directory."""
    web_config = load_web_config()
    root = Path(web_config.output_dir).resolve()
    candidate = _resolve_output_asset_path(root, asset_path)
    if candidate is None or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")
    _delete_gallery_assets(web_config.output_dir, [asset_path])
    return {"status": "deleted"}
