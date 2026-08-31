"""Serve the maintained FastAPI backend, Svelte SPA shell, and JSON/SSE job endpoints for the Web UI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from zvisiongenerator.backends import get_backend_name
from zvisiongenerator.converters.lora_import import import_lora_hf, import_lora_local
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_types import VideoGenerationRequest
from zvisiongenerator.utils.alignment import align_ltx_frames, align_resolution
from zvisiongenerator.utils.config import resolve_defaults, resolve_video_defaults, validate_scheduler
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.lora import parse_lora_arg
from zvisiongenerator.utils.paths import get_ziv_data_dir, resolve_lora_path, resolve_model_path
from zvisiongenerator.utils.video_model_detect import detect_video_model
from zvisiongenerator.web.config import WebUiConfig, load_web_config
from zvisiongenerator.web.config_api import build_api_config_response, huggingface_token_env_var
from zvisiongenerator.web.config_contract import persist_writable_config_patch, resolve_output_dir as resolve_config_output_dir
from zvisiongenerator.web.defaults import default_image_size_for_ratio, default_video_size_for_ratio
from zvisiongenerator.web.gallery import build_gallery_page_json, delete_gallery_assets, filter_and_sort_assets, gallery_asset_to_json, list_gallery_assets, resolve_output_asset_path
from zvisiongenerator.web.path_picker import pick_path
from zvisiongenerator.web.prompt_files import inspect_prompt_file, read_prompt_file, resolve_prompt_file_option, write_prompt_file
from zvisiongenerator.web.job_contract import IMAGE_SUPPORTED_CONTROLS, VIDEO_SUPPORTED_CONTROLS
from zvisiongenerator.web.web_runner import JobConflictError, UnsupportedJobControlError, WebRunner
from zvisiongenerator.web.workspace_api import build_models_response, build_workspace_bootstrap_view, build_workspace_response
from zvisiongenerator.web.workspace_contract import (
    CANONICAL_WORKFLOW_VALUES,
    DEFAULT_PROMPT_SOURCE,
    PROMPT_FILE_CONTRACT,
    PROMPT_SOURCE_VALUES,
    WORKFLOW_DEFINITIONS,
    build_workflow_contract,
    canonicalize_workflow,
    default_workflow_for_mode,
)


_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_DOCS_ASSETS_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets"

_WORKFLOW_DEFINITIONS = WORKFLOW_DEFINITIONS

_PROMPT_FILE_EXTENSIONS = tuple(PROMPT_FILE_CONTRACT["accepted_extensions"])

web_runner = WebRunner()


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
    asset_path = _resolve_docs_asset_path(asset_name)
    if asset_path is None or not asset_path.is_file():
        raise HTTPException(status_code=404, detail=f"Unknown asset: {asset_name}")
    return FileResponse(asset_path)


@app.get("/media/{asset_path:path}")
async def output_media(asset_path: str) -> FileResponse:
    """Serve generated media files by output-root-relative asset ID."""
    root = Path(load_web_config().output_dir).resolve()
    candidate = resolve_output_asset_path(root, asset_path)
    if candidate is None or not candidate.is_file():
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


@app.post("/api/picker")
async def open_picker(request: Request) -> dict[str, str | None]:
    """Open a shared native picker on the local machine running the Web UI host."""
    payload = await request.json()
    kind = _required_json_string(payload, "kind")
    purpose = _required_json_string(payload, "purpose")
    initial_path = _coerce_optional_string(payload.get("initial_path"))
    try:
        return pick_path(kind, purpose=purpose, initial_path=initial_path).to_payload()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/prompt-files/inspect")
async def api_prompt_file_inspect(request: Request) -> dict[str, Any]:
    """Inspect a host-local prompt file and return active option metadata."""
    payload = await request.json()
    try:
        document = inspect_prompt_file(_required_json_string(payload, "path"), accepted_extensions=_PROMPT_FILE_EXTENSIONS)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"path": document.path, "options": document.options}


@app.post("/api/prompt-files/read")
async def api_prompt_file_read(request: Request) -> dict[str, Any]:
    """Read raw prompt-file YAML plus active option metadata."""
    payload = await request.json()
    try:
        document = read_prompt_file(_required_json_string(payload, "path"), accepted_extensions=_PROMPT_FILE_EXTENSIONS)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"path": document.path, "raw_text": document.raw_text, "options": document.options}


@app.put("/api/prompt-files/write")
async def api_prompt_file_write(request: Request) -> dict[str, Any]:
    """Validate and atomically replace a host-local prompt file."""
    payload = await request.json()
    try:
        document = write_prompt_file(
            _required_json_string(payload, "path"),
            _required_json_string(payload, "raw_text"),
            accepted_extensions=_PROMPT_FILE_EXTENSIONS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"path": document.path, "options": document.options}


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
    created_at = datetime.now(tz=timezone.utc).isoformat()
    try:
        web_runner.update_job_context(
            str(job_context["job_id"]),
            {
                "workflow": workflow,
                "job_type": job_context.get("job_type", workflow),
                "prompt": job_context.get("prompt", ""),
                "model": job_context.get("title", ""),
                "runs": job_context["runs"],
                "created_at": created_at,
                "output_dir": job_context.get("output_dir", web_config.output_dir),
            },
        )
    except KeyError:
        pass

    return JSONResponse(
        {
            "job_id": job_context["job_id"],
            "workflow": workflow,
            "prompt": job_context.get("prompt", ""),
            "model": job_context.get("title", ""),
            "runs": job_context["runs"],
            "created_at": created_at,
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
    return build_workspace_bootstrap_view(web_config)


def _build_workspace_response(web_config: WebUiConfig, history_assets: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the workspace payload from backend-owned contract helpers."""
    return build_workspace_response(
        web_config,
        history_assets,
        active_job=web_runner.get_active_exclusive_job_snapshot(),
        prompt_sources=list(PROMPT_SOURCE_VALUES),
        default_prompt_source=DEFAULT_PROMPT_SOURCE,
        prompt_file_contract=dict(PROMPT_FILE_CONTRACT),
        workflow_contract=_build_workflow_contract(),
        build_bootstrap_view=_build_workspace_bootstrap_view,
    )


def _build_models_response(web_config: WebUiConfig, *, token_var: str | None) -> dict[str, Any]:
    """Build the models payload from shared backend inventory helpers."""
    return build_models_response(web_config, token_var=token_var)


def _resolve_docs_asset_path(asset_name: str) -> Path | None:
    candidate = (_DOCS_ASSETS_DIR / asset_name).resolve()
    try:
        candidate.relative_to(_DOCS_ASSETS_DIR)
    except ValueError:
        return None
    return candidate


def _preferred_option(preferred: str | None, options: tuple[str, ...]) -> str | None:
    """Return the preferred option when it exists, otherwise the first available item."""
    if preferred in options:
        return preferred
    return options[0] if options else None


def _canonicalize_workflow(value: Any, *, fallback: str | None = None) -> str | None:
    """Return a canonical workflow value, or fallback when value is absent or unsupported."""
    return canonicalize_workflow(value, fallback=fallback)


def _workflow_from_form(value: str | None, *, fallback: str) -> str:
    """Resolve a submitted workflow, rejecting non-canonical values."""
    if value is None:
        return fallback
    workflow = _canonicalize_workflow(value)
    if workflow is None:
        valid = ", ".join(CANONICAL_WORKFLOW_VALUES)
        raise ValueError(f"Unknown workflow '{value}'. Valid workflows: {valid}.")
    return workflow


def _default_workflow_for_mode(mode: str) -> str:
    return default_workflow_for_mode(mode)


def _build_workflow_contract() -> dict[str, Any]:
    return build_workflow_contract()


def _submit_image_job(form: Any, web_config: WebUiConfig) -> dict[str, Any]:
    app_config = web_config.app_config
    workflow = _workflow_from_form(_optional_text(form, "workflow"), fallback="txt2img")
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
        first_sigma=None,
        json_prompt_enabled=False,
    )
    args.size = _resolve_image_size(form, web_config, args.ratio)
    if args.image_strength is None:
        args.image_strength = 0.5
    if _WORKFLOW_DEFINITIONS[workflow]["requires_reference_image"] and args.image_path is None:
        raise ValueError("Image-to-image requires a reference image.")
    _validate_image_args(args, quantize_options=web_config.quantize_options)

    resolved_model = resolve_model_path(args.model, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
    model_info = detect_image_model(resolved_model)
    defaults = resolve_defaults(
        model_info,
        app_config,
        {key: value for key, value in {"steps": args.steps, "guidance": args.guidance, "scheduler": args.scheduler}.items() if value is not None},
        get_backend_name(),
    )
    json_prompt_text = _optional_text(form, "json_prompt")
    if json_prompt_text is not None:
        # JSON caption mode: bypass _resolve_prompt_submission so a JSON-only submission
        # is not rejected by the required-prompt check.
        if _optional_text(form, "prompt") is not None:
            raise ValueError("Provide either a prompt or a structured JSON caption, not both.")
        prompt_source = _optional_text(form, "prompt_source") or DEFAULT_PROMPT_SOURCE
        if prompt_source == "file":
            raise ValueError("A structured JSON caption cannot be combined with prompt-file mode.")
        try:
            parsed = json.loads(json_prompt_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'json_prompt must be a JSON object, e.g. \'{{"high_level_description": "..."}}\': {exc}') from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"json_prompt must be a JSON object, got {type(parsed).__name__}.")
        prompt = json_prompt_text
        negative_prompt = None
        prompts_data = {"prompt": [(json_prompt_text, None)]}
        args.json_prompt_enabled = True
    else:
        _prompt_source, prompt, negative_prompt, prompts_data = _resolve_prompt_submission(form)
    first_sigma = _optional_float(form, "first_sigma")
    if first_sigma is not None and not (0.0 < first_sigma <= 2.0):
        raise ValueError(f"first_sigma must be in (0.0, 2.0], got {first_sigma}.")
    args.first_sigma = first_sigma
    if args.json_prompt_enabled and not defaults.get("supports_json_prompt", False):
        raise ValueError("This model does not support structured JSON captions.")
    if args.first_sigma is not None and not defaults.get("supports_first_sigma", False):
        raise ValueError("This model does not support the first-step sigma control.")
    # Capture explicit-intent flags on args BEFORE the resolved defaults overwrite
    # args.steps/args.guidance, so run_batch's authoritative request rebuild sees them
    # via getattr(args, ...) (mirrors image_cli.py's args.steps_explicit assignment).
    args.steps_explicit = args.steps is not None
    args.guidance_explicit = args.guidance is not None
    args.steps = defaults["steps"]
    args.guidance = defaults["guidance"]
    args.scheduler = defaults["scheduler"]
    validate_scheduler(args.scheduler, app_config)
    if args.upscale and args.upscale_steps is None:
        args.upscale_steps = max(1, args.steps // 2)

    if not defaults.get("supports_negative_prompt", False):
        negative_prompt = None
        prompts_data = _replace_prompt_negatives(prompts_data, negative_prompt=None)

    args.lora_paths, args.lora_weights = _resolve_loras(form)
    dims = app_config["sizes"][args.ratio][args.size]
    eff_width = args.width or dims["width"]
    eff_height = args.height or dims["height"]
    _validate_model_capabilities(args, defaults, eff_width, eff_height)

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
        steps_explicit=args.steps_explicit,
        guidance_explicit=args.guidance_explicit,
        first_sigma=args.first_sigma,
        json_prompt=args.json_prompt_enabled,
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
        "supported_controls": IMAGE_SUPPORTED_CONTROLS,
        "runs": args.runs,
        "output_dir": args.output,
        "meta": f"{args.ratio} · {args.size} · {args.steps} steps",
    }


def _submit_video_job(form: Any, web_config: WebUiConfig) -> dict[str, Any]:
    app_config = web_config.app_config
    workflow = _workflow_from_form(_optional_text(form, "workflow"), fallback="txt2vid")
    model_name = _text_or_default(form, "model", web_config.default_models.video)
    if not model_name:
        raise ValueError("A video model is required.")

    ratio = _choice_or_default(form, "ratio", web_config.video_ratios, app_config.get("video_generation", {}).get("default_ratio", "16:9"))
    size = _resolve_video_size(form, web_config, ratio)
    output_dir = _resolve_output_dir(_text_or_default(form, "output", web_config.output_dir))
    image_path = _resolve_reference_image(form, output_dir)
    audio_enabled = _checkbox(form, "audio", default=True)
    _prompt_source, prompt, _negative_prompt, prompts_data = _resolve_prompt_submission(form)
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

    video_sizes = app_config.get("video_sizes", {})
    if video_sizes:
        if args.ratio not in video_sizes:
            raise ValueError(f"Unknown ratio '{args.ratio}'. Valid: {list(video_sizes.keys())}")
        if args.size not in video_sizes.get(args.ratio, {}):
            raise ValueError(f"Unknown size '{args.size}' for ratio '{args.ratio}'. Valid: {list(video_sizes.get(args.ratio, {}).keys())}")

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
        "supported_controls": VIDEO_SUPPORTED_CONTROLS,
        "runs": args.runs,
        "output_dir": args.output,
        "meta": f"{args.width}x{args.height} · {args.num_frames} frames · {args.steps} steps",
    }


def _resolve_prompt_submission(form: Any) -> tuple[str, str, str | None, dict[str, list[tuple[str, str | None]]]]:
    """Resolve inline or prompt-file submission into a single prompt payload."""
    prompt_source = _text_or_default(form, "prompt_source", DEFAULT_PROMPT_SOURCE)
    if prompt_source not in PROMPT_SOURCE_VALUES:
        raise ValueError(f"Unknown prompt source '{prompt_source}'.")
    if prompt_source == "file":
        _normalized_path, option = resolve_prompt_file_option(
            _required_text(form, "prompts_file"),
            _required_text(form, "prompt_option_id"),
            accepted_extensions=_PROMPT_FILE_EXTENSIONS,
        )
        prompts_data = {option.set_name: [(option.prompt, option.negative_prompt)]}
        return prompt_source, option.prompt, option.negative_prompt, prompts_data

    prompt = _required_text(form, "prompt")
    negative_prompt = _optional_text(form, "negative_prompt")
    return prompt_source, prompt, negative_prompt, {"web": [(prompt, negative_prompt)]}


def _replace_prompt_negatives(
    prompts_data: dict[str, list[tuple[str, str | None]]],
    *,
    negative_prompt: str | None,
) -> dict[str, list[tuple[str, str | None]]]:
    return {set_name: [(prompt, negative_prompt) for prompt, _existing_negative in entries] for set_name, entries in prompts_data.items()}


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
    default = default_image_size_for_ratio(web_config, ratio)
    return _choice_or_default(form, "size", choices, default)


def _resolve_video_size(form: Any, web_config: WebUiConfig, ratio: str) -> str:
    choices = web_config.video_size_options[ratio]
    default = default_video_size_for_ratio(web_config, ratio)
    return _choice_or_default(form, "size", choices, default)


def _optional_path(form: Any, key: str) -> str | None:
    value = _optional_text(form, key)
    if value is None:
        return None
    if "://" in value:
        raise ValueError(f"Field '{key}' must be a host-local file path on the machine running the Web UI host.")
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"Path '{path}' for '{key}' must be an existing host-local file.")
    return str(path.resolve())


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
    return str(resolve_config_output_dir(value))


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


def _validate_model_capabilities(args: argparse.Namespace, defaults: dict[str, Any], eff_width: int, eff_height: int) -> None:
    """Reject requests that violate the resolved model's capability flags.

    Args:
        args: The resolved image-generation args Namespace.
        defaults: The per-model effective defaults from ``resolve_defaults``.
        eff_width: The effective width (custom or size-preset) to validate.
        eff_height: The effective height (custom or size-preset) to validate.

    Raises:
        ValueError: When the model does not support a requested reference
            image / upscale / quantization, or when effective dimensions fall
            outside the model's ``dimension_min``/``dimension_max``/``dimension_step``.
    """
    if args.image_path is not None and not defaults.get("supports_img2img", True):
        raise ValueError("This model does not support reference-image (img2img) steering.")
    if args.upscale is not None and not defaults.get("supports_upscale", True):
        raise ValueError("This model does not support upscaling.")
    if getattr(args, "quantize", None) is not None and not defaults.get("supports_quantize", True):
        raise ValueError("This model does not support quantization.")
    dmin = int(defaults.get("dimension_min", 16))
    dmax = defaults.get("dimension_max", None)
    dstep = int(defaults.get("dimension_step", 16))
    for label, value in (("Width", eff_width), ("Height", eff_height)):
        if value < dmin or (dmax is not None and value > dmax) or value % dstep != 0:
            upper = dmax if dmax is not None else "∞"
            raise ValueError(f"{label} {value} must be between {dmin} and {upper} and a multiple of {dstep} for this model.")


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
    args.width, args.height = align_resolution(args.width, args.height, alignment, model_info.family.upper())
    args.num_frames = align_ltx_frames(args.num_frames, model_info.frame_alignment)
    if args.width < 64 or args.height < 64:
        raise ValueError(f"Resolved dimensions {args.width}x{args.height} are too small (minimum 64x64)")


def _persist_web_config(form: Any) -> None:
    """Persist the backend-owned writable config patch."""
    persist_writable_config_patch(form, load_web_config())


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


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_json_string(payload: dict[str, Any], key: str) -> str:
    value = _coerce_optional_string(payload.get(key))
    if value is None:
        raise HTTPException(status_code=422, detail=f"Field '{key}' is required.")
    return value


# ─── JSON API endpoints (Svelte SPA) ─────────────────────────────────────────


@app.get("/api/workspace")
async def api_workspace(include_history: bool = Query(True)) -> dict[str, Any]:
    """Return WorkspaceContext JSON for the Svelte SPA."""
    web_config = load_web_config()
    history_assets: list[dict[str, Any]] = []
    if include_history:
        all_assets = list_gallery_assets(web_config.output_dir)
        history_assets = [gallery_asset_to_json(asset, web_config) for asset in all_assets[:20]]
    return _build_workspace_response(web_config, history_assets)


@app.get("/api/history")
async def api_history(
    page: int = Query(1, ge=1),
    media_filter: str = Query("all"),
    sort_order: str = Query("newest"),
) -> dict[str, Any]:
    """Return paginated gallery history for the Svelte SPA."""
    web_config = load_web_config()
    all_assets = filter_and_sort_assets(
        list_gallery_assets(web_config.output_dir),
        media_filter=media_filter,
        sort_order=sort_order,
    )
    return build_gallery_page_json(all_assets, web_config, page=page, page_size=web_config.gallery_page_size)


@app.get("/api/gallery")
async def api_gallery_json(
    page: int = Query(1, ge=1),
    filter: str = Query("all"),
    sort_order: str = Query("newest"),
) -> dict[str, Any]:
    """Return paginated gallery assets as JSON for the Svelte SPA gallery view."""
    web_config = load_web_config()
    all_assets = filter_and_sort_assets(
        list_gallery_assets(web_config.output_dir),
        media_filter=filter,
        sort_order=sort_order,
    )
    return build_gallery_page_json(all_assets, web_config, page=page, page_size=web_config.gallery_page_size)


@app.get("/api/config")
async def api_get_config() -> dict[str, Any]:
    """Return current Web UI config as JSON for the Svelte SPA."""
    web_config = load_web_config()
    return build_api_config_response(web_config)


@app.post("/api/config")
async def api_save_config(request: Request) -> dict[str, Any]:
    """Update Web UI config from a JSON body and return the updated config."""
    payload = await request.json()
    try:
        _persist_web_config(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    web_config = load_web_config()
    return build_api_config_response(web_config)


@app.get("/api/models")
async def api_models() -> dict[str, Any]:
    """Return installed model inventory as JSON for the Svelte SPA."""
    web_config = load_web_config()
    token_var = huggingface_token_env_var()
    return _build_models_response(web_config, token_var=token_var)


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
async def api_cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a running job. Returns cancelled status or 404 if the job is unknown."""
    try:
        snapshot = web_runner.get_job_snapshot(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    if snapshot["status"] in {"completed", "failed", "cancelled"}:
        return {"job_id": job_id, "status": snapshot["status"]}
    supported_controls = set(snapshot.get("supported_controls") or [])
    if "quit" not in supported_controls and "cancel" not in supported_controls:
        raise HTTPException(status_code=409, detail="This job does not support cancellation.")
    try:
        return web_runner.queue_job_control(job_id, "quit")
    except UnsupportedJobControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/api/gallery/{asset_path:path}")
async def api_delete_gallery_asset(asset_path: str) -> dict[str, str]:
    """Delete a single gallery asset by output-root-relative asset ID."""
    web_config = load_web_config()
    root = Path(web_config.output_dir).resolve()
    candidate = resolve_output_asset_path(root, asset_path)
    if candidate is None or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")
    delete_gallery_assets(web_config.output_dir, [asset_path])
    return {"status": "deleted"}
