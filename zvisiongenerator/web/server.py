"""Serve the maintained FastAPI backend, Svelte SPA shell, and JSON/SSE job endpoints for the Web UI."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import random
import stat
import subprocess
import threading
import sys
import uuid
import warnings
from collections.abc import Iterator
from contextlib import ExitStack, asynccontextmanager, contextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from zvisiongenerator.backends import (
    accelerator_memory_budget_or_none as memory_budget,
    get_backend_name,
    get_prompt_enhancer_session,
    supports_video_audio,
    supports_video_extend,
    supports_video_file_upscale,
    supports_video_upscale_review,
    video_family_on_platform,
)
from zvisiongenerator.backends.prompt_enhancer_session import runs_on_cpu, smart_budget_bytes
from zvisiongenerator.converters.convert_checkpoint import H3_MODEL_TYPE, LTX_MODEL_TYPES, MODEL_TYPES, VIDEO_MODEL_TYPES
from zvisiongenerator.converters.lora_import import import_lora_hf, import_lora_local
from zvisiongenerator.core.ideogram4 import compact_caption
from zvisiongenerator.core.image_types import ImageGenerationRequest
from zvisiongenerator.core.video_family import UNDESCRIBED_UPSCALE_STEPS, SettledSteps, settle_video_steps
from zvisiongenerator.core.video_types import VideoGenerationRequest
from zvisiongenerator.downloads.flights import DownloadFlights
from zvisiongenerator.downloads.hub_access import token_source as hub_token_source
from zvisiongenerator.utils import app_db
from zvisiongenerator.utils.config import (
    ENHANCER_TIERS,
    enhancer_tier_models,
    load_config,
    model_display_name,
    model_reference,
    resolve_defaults,
    resolve_enhancer_model,
    resolve_upscale_steps,
    validate_scheduler,
    validate_sharpen_amount,
    video_family,
)
from zvisiongenerator.utils.enhance_record import parse_enhance_record
from zvisiongenerator.utils.ffmpeg import require_ffmpeg
from zvisiongenerator.utils.subprocess_env import utf8_child_env
from zvisiongenerator.utils.image_model_detect import detect_image_model
from zvisiongenerator.utils.lora import resolve_lora_references
from zvisiongenerator.utils.paths import get_ziv_data_dir, resolve_model_path
from zvisiongenerator.utils.model_memory import clip_memory_fit
from zvisiongenerator.utils.prompt_compose import expand_random_choices
from zvisiongenerator.utils.prompt_enhance import (
    ENHANCE_FORMATS,
    EnhanceContext,
    EnhanceResult,
    EnhanceSettings,
    enhance_context,
    enhance_options,
    image_enhance_entries,
    enhance_prompt,
    resolve_parts,
    settings_from_mapping,
    validate_settings,
)
from zvisiongenerator.utils.prompt_enhance_caption import enhance_caption
from zvisiongenerator.utils.prompt_enhance_video import EnhancePairResult, enhance_video_prompt, split_scene_action
from zvisiongenerator.utils.prompts import enhance_by_set as group_enhance_by_set
from zvisiongenerator.utils.prompts import scene_by_set as group_scene_by_set
from zvisiongenerator.utils.provenance import UPSCALE_METHOD_SEEDVR2
from zvisiongenerator.utils.upscale import VIDEO_UPSCALE_FACTOR
from zvisiongenerator.utils.seedvr2 import SeedVR2Setup, seedvr2_row_name, seedvr2_tier_label
from zvisiongenerator.upscale_runner import read_upscale_source
from zvisiongenerator.video_file_upscale_runner import read_video_upscale_source
from zvisiongenerator.utils.video_extend import apply_extend_args, extend_step, require_extend_family
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.utils.video_support import unsupported_video_request
from zvisiongenerator.workflows import build_seedvr2_upscale_workflow, build_upscale_workflow
from zvisiongenerator.web.config import WebUiConfig, load_web_config
from zvisiongenerator.web.config_api import build_api_config_response
from zvisiongenerator.web.config_contract import persist_first_run_skipped, persist_writable_config_patch, resolve_output_dir as resolve_config_output_dir
from zvisiongenerator.web.download_lane import DownloadLane
from zvisiongenerator.web.defaults import default_image_size_for_ratio, default_video_size_for_ratio
from zvisiongenerator.web.extend_api import EXTEND_UNAVAILABLE_MESSAGE, extend_context_from_body, extend_source_response, load_source_video, requested_extend_frames
from zvisiongenerator.web.gallery import (
    GALLERY_MEDIA_EXTENSIONS,
    VIDEO_EXTENSIONS,
    build_gallery_page_json,
    delete_gallery_assets,
    filter_and_sort_assets,
    gallery_asset_to_json,
    list_gallery_assets,
    resolve_output_asset_path,
)
from zvisiongenerator.web.job_log import JOB_LOG_PAGE_SIZE, JobLog, JobLogError
from zvisiongenerator.web.json_fields import json_field
from zvisiongenerator.web.model_download_plan import PlanUnchecked, plan_summary, row_plan
from zvisiongenerator.web.path_browser import BrowsePlace, BrowsePurpose, DirectoryListing, browse_places, create_folder, default_place_dir, host_drives, list_directory, resolve_purpose
from zvisiongenerator.web.request_guard import LocalRequestGuardMiddleware
from zvisiongenerator.web.upscale_api import plan_seedvr2_upscale, plan_upscale, plan_video_upscale, upscale_json_request, upscale_size_label, video_upscale_size_label
from zvisiongenerator.web.video_clip import clip_defaults, require_upscale_factor, resolve_video_clip, resolve_video_model
from zvisiongenerator.web.lora_fit import lora_bytes_for_fit
from zvisiongenerator.web.prompt_builder import PromptFileChangedError, append_entry_payload, create_prompt_file, load_document_payload, preview_document_payload, save_document_payload
from zvisiongenerator.web.prompt_files import inspect_prompt_file, resolve_prompt_file_options, write_prompt_file
from zvisiongenerator.web.presets import create_preset, delete_preset, list_presets, rename_preset, update_preset
from zvisiongenerator.web.model_delete import DeleteTarget, delete_lora, delete_model, delete_scope, enhancer_delete_target, model_delete_target, seedvr2_delete_target
from zvisiongenerator.review import review_enabled
from zvisiongenerator.web.job_contract import UPSCALE_SUPPORTED_CONTROLS, image_supported_controls, video_supported_controls
from zvisiongenerator.web.web_runner import JobConflictError, UnsupportedJobControlError, WebRunner, worker_runtime_context
from zvisiongenerator.web.workspace_api import build_models_response, build_workspace_bootstrap_view, build_workspace_response, find_other_downloads
from zvisiongenerator.utils.hf_cache import free_stranded_shared_files
from zvisiongenerator.utils.model_files import disk_size_label, huggingface_cache_dir
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

download_flights = DownloadFlights()
job_log = JobLog(app_db.app_db_path)
web_runner = WebRunner(download_flights=download_flights, job_log=job_log)
download_lane = DownloadLane(flights=download_flights)
logger = logging.getLogger(__name__)
_ENHANCE_TERMINAL_FRAMES = frozenset({"done", "error"})
# How long shutting down waits for a running Models-page download's child process to stop and clean up.
_LANE_SHUTDOWN_SECS = 10.0


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Own the background worker pool for the life of the FastAPI app."""
    try:
        yield
    finally:
        download_lane.shutdown(join_timeout=_LANE_SHUTDOWN_SECS)
        web_runner.shutdown()


# Blocking handler work (folder listings, conversion, imports) is offloaded with run_in_threadpool (AnyIO's worker
# pool), not asyncio.to_thread: the SSE streams wait on asyncio's default executor, so long listings or conversions
# must not occupy its threads and stall progress updates.
app = FastAPI(title="Z-Vision Generator Web UI", lifespan=_lifespan)
app.add_middleware(LocalRequestGuardMiddleware)
app.mount("/app-static", StaticFiles(directory=str(Path(__file__).with_name("static") / "app"), check_dir=False), name="app-static")


@app.get("/docs/assets/{asset_name}")
async def docs_asset(asset_name: str) -> FileResponse:
    """Serve preview branding assets referenced by the web docs and browser clients."""
    asset_path = _resolve_docs_asset_path(asset_name)
    if asset_path is None or not asset_path.is_file():
        raise HTTPException(status_code=404, detail=f"Unknown asset: {asset_name}")
    return FileResponse(asset_path)


@app.get("/media/{asset_path:path}")
def output_media(asset_path: str) -> FileResponse:
    """Serve generated media files by output-root-relative asset ID."""
    candidate = resolve_output_asset_path(_media_output_root(), asset_path)
    if candidate is None or candidate.suffix.lower() not in GALLERY_MEDIA_EXTENSIONS or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Unknown media asset: {asset_path}")
    return FileResponse(candidate)


_media_output_root_cache: tuple[tuple[object, ...], Path] | None = None


def _media_output_root() -> Path:
    """Resolve the output root without building the full Web UI config (model inventory) per media request.

    Cached on the user config file's identity, so a gallery page of thumbnails parses the config once; a save
    (atomic replace) changes the stat and refreshes it.
    """
    global _media_output_root_cache
    data_dir = get_ziv_data_dir()
    try:
        config_stat = (data_dir / "config.yaml").stat()
        cache_key: tuple[object, ...] = (str(data_dir), config_stat.st_ino, config_stat.st_mtime_ns, config_stat.st_size)
    except OSError:
        cache_key = (str(data_dir), None)
    cached = _media_output_root_cache
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    ui_config = load_config().get("ui") or {}
    output_dir = ui_config.get("output_dir") if isinstance(ui_config, dict) else None
    root = resolve_config_output_dir(output_dir).resolve()
    _media_output_root_cache = (cache_key, root)
    return root


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


@app.get("/api/browse")
async def api_browse(purpose: str, path: str = "", hidden: bool = False, show_all: bool = Query(False, alias="all")) -> dict[str, Any]:
    """List one host-local folder for the in-app path browser, keeping what *purpose* can choose."""
    browse_purpose = _browse_purpose(purpose)
    try:
        listing = await run_in_threadpool(_list_browse_directory, browse_purpose, path, show_hidden=hidden, show_all=show_all)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {exc.filename or path}.") from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return listing.to_payload()


@app.get("/api/browse/places")
async def api_browse_places(purpose: str) -> dict[str, Any]:
    """Return what *purpose* chooses and the path browser's sidebar shortcuts for it."""
    browse_purpose = _browse_purpose(purpose)
    places = await run_in_threadpool(_browse_places, browse_purpose)
    return {"purpose": browse_purpose.to_payload(), "places": [place.to_payload() for place in places]}


@app.post("/api/browse/folder")
async def api_browse_folder(request: Request) -> dict[str, str]:
    """Create a folder from the path browser for a folder purpose (New folder)."""
    payload = await request.json()
    browse_purpose = _browse_purpose(_required_json_string(payload, "purpose"))
    parent = _required_json_string(payload, "path")
    name = _required_json_string(payload, "name")
    try:
        created = await run_in_threadpool(create_folder, browse_purpose, parent, name)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=f"'{name}' already exists.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {parent}.") from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"path": str(created)}


def _browse_purpose(name: str) -> BrowsePurpose:
    try:
        return resolve_purpose(name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _browse_places(purpose: BrowsePurpose) -> tuple[BrowsePlace, ...]:
    return browse_places(purpose, home=Path.home(), output_root=_media_output_root(), data_dir=get_ziv_data_dir(), drives=host_drives())


def _list_browse_directory(purpose: BrowsePurpose, path: str, *, show_hidden: bool, show_all: bool) -> DirectoryListing:
    """List *path*, or the purpose's own place when it is empty; blocking, so it runs in the thread pool."""
    # Only an empty path opens at the purpose's place, which reads the config.
    default_dir = None if path.strip() else default_place_dir(purpose, home=Path.home(), output_root=_media_output_root(), data_dir=get_ziv_data_dir())
    return list_directory(purpose, path, default_dir=default_dir, show_hidden=show_hidden, show_all=show_all)


@app.post("/api/extend/source")
async def api_extend_source(request: Request) -> dict[str, Any]:
    """Inspect a host-local video that may be extended: its facts, recorded scene and prompt, problems and memory fit."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Request body must be JSON.") from exc
    try:
        return await run_in_threadpool(extend_source_response, payload, load_web_config(), _media_output_root())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/video/memory-fit")
async def api_video_memory_fit(request: Request) -> dict[str, Any]:
    """Check whether a Text to Video or Image to Video clip of the given size and length fits this machine's memory; queues nothing."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Request body must be JSON.") from exc
    try:
        return await run_in_threadpool(_video_memory_fit_response, payload, load_web_config())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/prompt-files/inspect")
async def api_prompt_file_inspect(request: Request) -> dict[str, Any]:
    """Inspect a host-local prompt file and return active option metadata."""
    payload = await request.json()
    try:
        document = await run_in_threadpool(inspect_prompt_file, _required_json_string(payload, "path"), accepted_extensions=_PROMPT_FILE_EXTENSIONS)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"path": document.path, "options": document.options}


@app.put("/api/prompt-files/write")
async def api_prompt_file_write(request: Request) -> dict[str, Any]:
    """Validate and atomically replace a host-local prompt file."""
    payload = await request.json()
    try:
        document = await run_in_threadpool(
            write_prompt_file,
            _required_json_string(payload, "path"),
            _required_json_string(payload, "raw_text"),
            accepted_extensions=_PROMPT_FILE_EXTENSIONS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"path": document.path, "options": document.options}


@app.post("/api/prompt-files/document")
async def api_prompt_file_document(request: Request) -> dict[str, Any]:
    """Load a prompt file as a builder document (or its raw text and problem, for repair)."""
    payload = await request.json()
    try:
        return await run_in_threadpool(load_document_payload, _required_json_string(payload, "path"), accepted_extensions=_PROMPT_FILE_EXTENSIONS)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.put("/api/prompt-files/document")
async def api_prompt_file_document_save(request: Request) -> dict[str, Any]:
    """Save a builder document onto its prompt file; 409 when the file changed on disk since it was loaded."""
    payload = await request.json()
    base_text = payload.get("base_text")
    try:
        return await run_in_threadpool(
            save_document_payload,
            _required_json_string(payload, "path"),
            _required_json_string(payload, "revision"),
            payload.get("document"),
            base_text=base_text if isinstance(base_text, str) else None,
            force=payload.get("force") is True,
            accepted_extensions=_PROMPT_FILE_EXTENSIONS,
        )
    except PromptFileChangedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/prompt-files/preview")
async def api_prompt_file_preview(request: Request) -> dict[str, Any]:
    """Preview a builder document: resolved prompts, problems, snippet use counts, and an optional roll."""
    payload = await request.json()
    roll_entry_id = payload.get("roll_entry_id")
    try:
        return await run_in_threadpool(preview_document_payload, payload.get("document"), roll_entry_id=roll_entry_id if isinstance(roll_entry_id, str) else None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/prompt-files/create")
async def api_prompt_file_create(request: Request) -> dict[str, Any]:
    """Create a new, empty prompt file in an existing folder."""
    payload = await request.json()
    try:
        return await run_in_threadpool(
            create_prompt_file,
            _required_json_string(payload, "directory"),
            _required_json_string(payload, "name"),
            accepted_extensions=_PROMPT_FILE_EXTENSIONS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/prompt-files/append")
async def api_prompt_file_append(request: Request) -> dict[str, Any]:
    """Append a prompt to the ``prompts`` set of a prompt file (the set is added at the end when missing)."""
    payload = await request.json()
    path = _required_json_string(payload, "path")
    try:
        prompt = json_field(payload, "prompt", str, "a string") or ""
        scene = json_field(payload, "scene", str, "a string")
        negative = json_field(payload, "negative", str, "a string")
        return await run_in_threadpool(append_entry_payload, path, prompt, scene=scene, negative=negative, accepted_extensions=_PROMPT_FILE_EXTENSIONS)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/prompt/enhance")
async def enhance_prompt_endpoint(request: Request) -> StreamingResponse:
    """Stream an on-demand prompt enhancement as NDJSON frames: ``status`` → ``text``* → ``done`` | ``error``.

    Image frames carry ``text`` / ``prompt``. Video frames also carry the split ``scene`` and ``prompt`` parts: ``text`` frames
    ``{text, scene, prompt}`` and the ``done`` frame ``{scene, prompt, clamped, warning?}``.
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Request body must be JSON.") from exc
    try:
        job = await run_in_threadpool(_validate_enhance_body, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    session = get_prompt_enhancer_session()
    reservation: list[object] = []

    def _reserve() -> None:
        token = session.reserve()
        if token is None:
            raise JobConflictError("Another prompt enhancement is in progress.")
        reservation.append(token)

    try:
        # Atomic with job admission: no job can start between this check and the enhancer claiming the slot.
        web_runner.admit_exclusive(_reserve, busy_message="Prompt enhancement is available when all jobs have finished.")
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    frames: queue.Queue[dict[str, Any]] = queue.Queue()
    cancelled = threading.Event()
    # One dedicated thread owns the whole enhancement (load + generate): MLX streams are thread-local.
    threading.Thread(target=_run_enhancement, args=(job, frames, cancelled, reservation[0]), name="ziv-prompt-enhance", daemon=True).start()

    async def _stream():
        try:
            while True:
                frame = await asyncio.to_thread(frames.get)
                yield json.dumps(frame) + "\n"
                if frame["type"] in _ENHANCE_TERMINAL_FRAMES:
                    return
        finally:
            cancelled.set()

    return StreamingResponse(_stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


def _reject_while_busy() -> None:
    """Job admission check: refuse a job while an on-demand enhancement or a model quantization holds memory.

    Both can only start while no job is active, so with jobs active a busy enhancer belongs to a job's own
    preflight and the new job simply queues behind it.
    """
    if web_runner.has_active_jobs():
        return
    if get_prompt_enhancer_session().busy():
        raise JobConflictError("Prompt enhancement in progress. Wait for it to finish before starting a job.")
    if _quantize_slot["busy"]:
        raise JobConflictError("A model is being quantized on the Models page. Wait for it to finish before starting a job.")


# A converter run that saves a quantized copy loads a full model, so it excludes generation jobs (and vice versa).
# Claimed and checked under the web runner's job lock (``admit_exclusive`` / ``admission_check``).
_quantize_slot = {"busy": False}


@contextmanager
def _claim_quantize_slot() -> Iterator[None]:
    """Hold the quantize slot for one converter run; raise JobConflictError while a job runs or another holds it."""

    def _claim() -> None:
        if _quantize_slot["busy"]:
            raise JobConflictError("Another model is being quantized. Wait for it to finish.")
        _quantize_slot["busy"] = True

    web_runner.admit_exclusive(_claim, busy_message="Wait for all jobs to finish before saving a quantized copy; both would need a full model in memory.")
    try:
        yield
    finally:
        _quantize_slot["busy"] = False


def _validate_enhance_body(body: Any) -> dict[str, Any]:
    """Validate an enhance request body into the job parameters for :func:`_run_enhancement`."""
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    mode = body.get("mode", "image")
    if mode not in ("image", "video"):
        raise ValueError("mode must be 'image' or 'video'.")
    prompt = body.get("prompt")
    scene = ""
    if mode == "video":
        scene = body.get("scene", "")
        if not isinstance(scene, str):
            raise ValueError("scene must be a string.")
        if prompt is None:
            prompt = ""
        if not isinstance(prompt, str):
            raise ValueError("Enter a scene or a prompt to enhance.")
    elif not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a prompt to enhance.")
    context = _enhance_context_from_body(body.get("context"), mode=mode, prompt=prompt)
    # Extending a clip may start from nothing: the enhancer then continues the scene naturally.
    if mode == "video" and not context.extends and not (prompt.strip() or scene.strip()):
        raise ValueError("Enter a scene or a prompt to enhance.")
    settings = settings_from_mapping(body.get("settings") or {}, mode=mode, motion_in_image="warn")
    tier = _body_choice(body, "enhance_model", ENHANCER_TIERS)
    caption = mode == "image" and _body_choice(body, "format", ENHANCE_FORMATS) == "json"
    validate_settings(settings, mode=mode)
    app_config = load_config()
    options = enhance_options(app_config)
    max_words = body.get("max_words")
    if max_words is None:
        max_words = int(options["length"]["max_words"])
    if isinstance(max_words, bool) or not isinstance(max_words, int) or not 1 <= max_words <= 1000:
        raise ValueError("max_words must be an integer between 1 and 1000.")
    repo, revision = resolve_enhancer_model(app_config, platform_key=sys.platform, cli_model=tier, budget_bytes=smart_budget_bytes(tier))
    return {
        "prompt": prompt,
        "scene": scene,
        "mode": mode,
        "caption": caption,
        "settings": settings,
        "context": context,
        "ceiling": max_words,
        "repo": repo,
        "revision": revision,
        "options": options,
        "idle_seconds": options["idle_release_seconds"],
    }


def _body_choice(body: dict[str, Any], key: str, allowed: tuple[str, ...]) -> str | None:
    """Return the request body's *key* when it is one of *allowed*, ``None`` when absent."""
    value = body.get(key)
    if value is not None and value not in allowed:
        raise ValueError(f"{key} must be one of: {', '.join(allowed)}.")
    return value


def _enhance_context_from_body(raw: Any, *, mode: str, prompt: str = "") -> EnhanceContext:
    """Validate the enhance request's job facts (frame size or ratio, frames, fps, audio, reference image).

    A video request with ``context.source_video`` extends that clip: the server reads the clip's size, frame rate,
    recorded prompt and recorded scene itself, so only the source, the new length and the audio setting come from the client.

    Args:
        prompt: The user's prompt template, compared with a legacy source's recorded prompt.
    """
    if raw is None:
        return EnhanceContext()
    if not isinstance(raw, dict):
        raise ValueError("context must be a JSON object.")
    if mode == "video" and raw.get("source_video") is not None:
        return extend_context_from_body(raw, prompt)

    def _positive(key: str) -> int | float | None:
        value = raw.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"context.{key} must be a positive number.")
        return value

    ratio = raw.get("ratio")
    if ratio is not None and not isinstance(ratio, str):
        raise ValueError("context.ratio must be a string such as '16:9'.")
    return enhance_context(
        mode=mode,
        width=_positive("width"),
        height=_positive("height"),
        ratio=ratio,
        frames=_positive("frames"),
        fps=_positive("fps"),
        audio=raw.get("audio") is True and supports_video_audio(),
        reference_image=raw.get("reference_image") is True,
        end_image=raw.get("end_image") is True,
    )


def _run_enhancement(job: dict[str, Any], frames: queue.Queue[dict[str, Any]], cancelled: threading.Event, reservation: object | None = None) -> None:
    """Load (if needed) and run the enhancer on this thread, pushing NDJSON frames to *frames*."""
    try:
        with worker_runtime_context():
            session = get_prompt_enhancer_session()
            with session.acquire(job["repo"], job["revision"], idle_seconds=job["idle_seconds"], on_phase=lambda phase: frames.put({"type": "status", "phase": phase})) as enhancer:
                frames.put({"type": "status", "phase": "generating_cpu" if runs_on_cpu(enhancer) else "generating"})
                result = _enhance_for_job(job, enhancer, frames, cancelled)
        frames.put(_enhance_done_frame(job["mode"], result, model=model_reference(job["repo"], job["revision"])))
    except (RuntimeError, ValueError) as exc:
        frames.put({"type": "error", "detail": str(exc)})
    except Exception as exc:  # noqa: BLE001 - any loader/runtime failure must still end the stream
        logger.exception("Prompt enhancement failed")
        frames.put({"type": "error", "detail": f"Prompt enhancement failed: {exc}"})
    finally:
        if reservation is not None:
            get_prompt_enhancer_session().cancel_reservation(reservation)


def _enhance_for_job(job: dict[str, Any], enhancer: Any, frames: queue.Queue[dict[str, Any]], cancelled: threading.Event) -> EnhanceResult | EnhancePairResult:
    """Run the enhancement for *job*, pushing ``text`` frames (video frames carry the split ``scene`` and ``prompt``)."""
    options = job["options"]
    common = {
        "seed": random.randint(0, 2**31 - 1),
        "ceiling": job["ceiling"],
        "context": job.get("context"),
        "length_cfg": options["length"],
        "temperature": options["temperature"],
        "max_tokens": options["max_new_tokens"],
        "cancelled": cancelled.is_set,
    }
    if job["mode"] != "video":
        rewrite = enhance_caption if job.get("caption") else partial(enhance_prompt, mode="image")
        return rewrite(enhancer, expand_random_choices(job["prompt"]), job["settings"], on_text=lambda text: frames.put({"type": "text", "text": text}), **common)
    scene = expand_random_choices(job.get("scene", ""))
    context = job.get("context")
    # Only the action is written when the settings ask for it and there is a scene to keep: no scene text streams.
    action_only = resolve_parts(job["settings"], scene=scene, extends=bool(context is not None and context.extends), reference_image=bool(context is not None and context.reference_image)) == "action"

    def _on_text(text: str) -> None:
        part_scene, part_prompt = split_scene_action(text, action_only=action_only)
        frames.put({"type": "text", "text": text, "scene": part_scene, "prompt": part_prompt})

    return enhance_video_prompt(enhancer, scene, expand_random_choices(job["prompt"]), job["settings"], on_text=_on_text, **common)


def _enhance_done_frame(mode: str, result: EnhanceResult | EnhancePairResult, *, model: str) -> dict[str, Any]:
    """Return the terminal ``done`` frame: ``{prompt, clamped, model, warning?}`` for images, plus ``scene`` for video.

    ``model`` is the enhancer that wrote it (``REPO[@REVISION]``), which the client sends back with Generate so the asset
    records it. A video frame's ``scene`` is ``""`` when the scene was not rewritten (action only), so the client keeps the user's scene.
    """
    frame: dict[str, Any] = {"type": "done", "prompt": result.prompt, "clamped": result.clamped, "model": model}
    if isinstance(result, EnhancePairResult):
        frame = {"type": "done", "scene": result.scene, **{key: frame[key] for key in ("prompt", "clamped", "model")}}
    if result.warning:
        frame["warning"] = result.warning
    return frame


@app.post("/api/generate")
async def generate(request: Request) -> JSONResponse:
    """Accept multipart form submissions and queue image or video generation jobs."""
    form = await request.form()
    return await run_in_threadpool(_generate_from_form, form)


def _generate_from_form(form: Any) -> JSONResponse:
    """Validate a parsed generate form and queue the job; runs off the event loop."""
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

    return _job_response(str(job_context["job_id"]), job_context["context"], job_context.get("supported_controls", ()))


def _job_response(job_id: str, context: dict[str, Any], supported_controls: tuple[str, ...]) -> JSONResponse:
    """Return the response to a job submission: its public context, event URLs and place in the queue."""
    try:
        queue_position = web_runner.get_job_snapshot(job_id)["queue_position"]
    except KeyError:
        queue_position = None
    return JSONResponse(
        {
            "job_id": job_id,
            "workflow": context["workflow"],
            "job_type": context["job_type"],
            "prompt": context["prompt"],
            "model": context["model"],
            "model_label": context.get("model_label") or context["model"],
            "runs": context["runs"],
            "created_at": context["created_at"],
            "events_url": f"/jobs/{job_id}/events",
            "status_url": f"/jobs/{job_id}",
            "supported_controls": list(supported_controls),
            "meta": context["meta"],
            "width": context.get("width"),
            "height": context.get("height"),
            "notices": list(context.get("notices", ())),
            "queue_position": queue_position,
            **({"media_type": context["media_type"]} if "media_type" in context else {}),
        }
    )


def _public_job_context(
    form: Any,
    *,
    mode: str,
    job_type: str,
    model: str,
    model_label: str,
    prompt: str,
    runs: int,
    meta: str,
    width: int,
    height: int,
    output_dir: str,
    uploads: dict[str, str | None],
) -> dict[str, Any]:
    """Build the public context a job carries from submission on (shown in snapshots and the queue).

    ``width`` and ``height`` are the size the job renders at, which gives its preview the output's shape.
    ``uploads`` maps each path setting (``image_path``, ``end_image_path``) to the file the job uses for it.
    """
    requested_workflow = _canonicalize_workflow(_optional_text(form, "workflow"), fallback=_default_workflow_for_mode(mode))
    workflow = _canonicalize_workflow(job_type, fallback=requested_workflow)
    return {
        "workflow": workflow,
        "job_type": job_type,
        "prompt": prompt,
        "model": model,
        "model_label": model_label,
        "runs": runs,
        "meta": meta,
        "width": width,
        "height": height,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "output_dir": output_dir,
        "settings": _submitted_settings(form, uploads=uploads),
    }


def _submitted_settings(form: Any, *, uploads: dict[str, str | None]) -> dict[str, str | list[str]]:
    """Return the submitted text fields verbatim (repeated fields as lists); an upload becomes its saved path.

    Args:
        form: The submitted form.
        uploads: Each path setting (such as ``image_path``) mapped to the file the job uses for it. An entry is recorded
            only when the form carries the matching upload field (``image_file`` for ``image_path``).
    """
    settings: dict[str, str | list[str]] = {}
    # Starlette's FormData repeats a key per value; plain mappings (tests, internal callers) have one item per key.
    items = form.multi_items() if hasattr(form, "multi_items") else form.items()
    for key, value in items:
        if isinstance(value, list):
            settings[key] = [str(item) for item in value]
            continue
        if not isinstance(value, str):
            continue
        if key not in settings:
            settings[key] = value
            continue
        existing = settings[key]
        settings[key] = [*existing, value] if isinstance(existing, list) else [existing, value]
    for path_key, saved_path in uploads.items():
        if saved_path and _is_uploaded_file(form.get(_upload_field(path_key))):
            settings[path_key] = saved_path
    return settings


@app.post("/api/upscale")
async def upscale(request: Request) -> JSONResponse:
    """Queue a job that upscales an existing gallery image 2× or 4×, refined with its recorded settings."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Request body must be JSON.") from exc
    return await run_in_threadpool(_upscale_from_body, body)


def _upscale_from_body(body: Any) -> JSONResponse:
    """Validate an upscale request, plan the job and submit it; runs off the event loop."""
    try:
        asset_id, factor, method = upscale_json_request(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    source_path = resolve_output_asset_path(_media_output_root(), asset_id)
    if source_path is None:
        raise HTTPException(status_code=422, detail="The file must be inside the output folder.")
    if source_path.suffix.lower() in VIDEO_EXTENSIONS:
        return _video_upscale_from_body(source_path, factor, method)
    if source_path.suffix.lower() not in _IMAGE_EXTENSIONS or not source_path.is_file():
        raise HTTPException(status_code=404, detail=f"Unknown image: {asset_id}")

    web_config = load_web_config()
    setup = web_config.seedvr2
    use_seedvr2 = method == UPSCALE_METHOD_SEEDVR2
    if use_seedvr2 and setup is None:
        raise HTTPException(status_code=422, detail="SeedVR2 upscaling is not available on this machine.")
    try:
        source = read_upscale_source(source_path)
        if use_seedvr2:
            plan = plan_seedvr2_upscale(source, factor, setup=setup)
        else:
            plan = plan_upscale(source, factor, web_config, backend_name=get_backend_name())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # An upscale has no Workspace form, so it carries no settings to load back from the queue.
    upscale_model = seedvr2_tier_label(setup.tier.model, setup.weights) if use_seedvr2 else plan.request.model_name or ""
    context = {
        "workflow": "upscale",
        "job_type": "Upscale",
        "prompt": plan.request.resolved_prompt or plan.request.prompt,
        "model": upscale_model,
        "model_label": model_display_name(web_config.app_config, upscale_model),
        "runs": 1,
        "meta": upscale_size_label(plan),
        "width": plan.option.width,
        "height": plan.option.height,
        "notices": list(plan.notices),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "output_dir": web_config.output_dir,
        "settings": {},
    }
    try:
        job_id = web_runner.submit_upscale_job(
            request=plan.request,
            workflow=build_seedvr2_upscale_workflow() if use_seedvr2 else build_upscale_workflow(sharpen=plan.request.sharpen),
            model_ref=plan.model_ref,
            admission_check=_reject_while_busy,
            context=context,
        )
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_response(job_id, context, UPSCALE_SUPPORTED_CONTROLS)


def _video_upscale_from_body(source_path: Path, factor: int, method: str) -> JSONResponse:
    """Validate an upscale request for a gallery video, plan the job and submit it; runs off the event loop."""
    if method != UPSCALE_METHOD_SEEDVR2:
        raise HTTPException(status_code=422, detail="Videos are upscaled with SeedVR2 only.")
    if factor != VIDEO_UPSCALE_FACTOR:
        raise HTTPException(status_code=422, detail=f"Videos are upscaled {VIDEO_UPSCALE_FACTOR}× only.")
    web_config = load_web_config()
    setup = web_config.seedvr2
    if setup is None or not supports_video_file_upscale():
        raise HTTPException(status_code=422, detail="Video upscaling is not available on this machine.")
    try:
        require_ffmpeg()
        source, recorded_config = read_video_upscale_source(source_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    plan = plan_video_upscale(source, recorded_config, setup=setup)
    upscale_model = seedvr2_tier_label(setup.tier.model, setup.weights)
    context = {
        "workflow": "upscale",
        "job_type": "Upscale",
        "media_type": "video",
        "prompt": source.prompt or "",
        "model": upscale_model,
        "model_label": model_display_name(web_config.app_config, upscale_model),
        "runs": 1,
        "meta": video_upscale_size_label(plan),
        "width": plan.request.width,
        "height": plan.request.height,
        "notices": list(plan.notices),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "output_dir": web_config.output_dir,
        "settings": {},
    }
    try:
        job_id = web_runner.submit_video_file_upscale_job(request=plan.request, admission_check=_reject_while_busy, context=context)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _job_response(job_id, context, UPSCALE_SUPPORTED_CONTROLS)


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


@app.get("/jobs/{job_id}/preview")
async def get_job_preview(job_id: str) -> Response:
    """Return the latest in-memory live preview of a running generation as a JPEG."""
    try:
        preview = web_runner.get_job_preview(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    if preview is None:
        raise HTTPException(status_code=404, detail="No live preview is available for this job.")
    return Response(content=preview, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/jobs/{job_id}/review")
async def get_job_review(job_id: str) -> Response:
    """Serve the small render of a video that waits for review as an mp4; ``v`` only busts the browser cache."""
    try:
        path = web_runner.get_job_review_path(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    try:
        file_stat = os.stat(path) if path is not None else None
    except FileNotFoundError, NotADirectoryError:  # the job resumed and removed its temporary files
        file_stat = None
    if file_stat is None or not stat.S_ISREG(file_stat.st_mode):
        raise HTTPException(status_code=404, detail="No render is waiting for review.")
    return FileResponse(path, stat_result=file_stat, media_type="video/mp4", headers={"Cache-Control": "no-store"})


@app.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str, request: Request, after: int | None = Query(None, ge=0)) -> StreamingResponse:
    """Stream job progress as SSE frames for browser and programmatic consumers."""
    try:
        web_runner.get_job_snapshot(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc

    last_event_id = request.headers.get("last-event-id")
    header_event_id = int(last_event_id) if last_event_id and last_event_id.isascii() and last_event_id.isdecimal() else None
    after_event_id = header_event_id if header_event_id is not None else after

    return StreamingResponse(
        web_runner.stream_job_events(job_id, after_event_id=after_event_id),
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
    jobs = web_runner.list_jobs()
    return build_workspace_response(
        web_config,
        history_assets,
        active_job=jobs["active_job"],
        queued_jobs=jobs["queued_jobs"],
        prompt_sources=list(PROMPT_SOURCE_VALUES),
        default_prompt_source=DEFAULT_PROMPT_SOURCE,
        prompt_file_contract=dict(PROMPT_FILE_CONTRACT),
        workflow_contract=_build_workflow_contract(),
        build_bootstrap_view=_build_workspace_bootstrap_view,
    )


def _build_models_response(web_config: WebUiConfig, *, token_source: str | None) -> dict[str, Any]:
    """Build the models payload from shared backend inventory helpers."""
    return build_models_response(web_config, token_source=token_source)


def _resolve_docs_asset_path(asset_name: str) -> Path | None:
    candidate = (_DOCS_ASSETS_DIR / asset_name).resolve()
    try:
        candidate.relative_to(_DOCS_ASSETS_DIR)
    except ValueError:
        return None
    return candidate


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
    return build_workflow_contract(video_audio=supports_video_audio(), video_upscale_review=supports_video_upscale_review(), video_extend=supports_video_extend())


def _submit_image_job(form: Any, web_config: WebUiConfig) -> dict[str, Any]:
    app_config = web_config.app_config
    workflow = _workflow_from_form(_optional_text(form, "workflow"), fallback="txt2img")
    model_name = _text_or_default(form, "model", web_config.default_models.image)
    if not model_name:
        raise ValueError("An image model is required.")

    # The configured output directory is backend-owned. Accepting a copy from the
    # Workspace form lets a stale browser tab write to one root while Gallery and
    # /media serve another, which breaks previews and makes new assets disappear.
    output_dir = _resolve_output_dir(web_config.output_dir)
    upscale = _optional_int(form, "upscale")
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
        upscale=upscale,
        upscale_denoise=_optional_float(form, "upscale_denoise"),
        upscale_steps=_optional_int(form, "upscale_steps"),
        upscale_guidance=_optional_float(form, "upscale_guidance"),
        upscale_sharpen=_checkbox(form, "upscale_sharpen", default=True),
        upscale_save_pre=_checkbox(form, "upscale_save_pre"),
        review=bool(upscale) and _checkbox(form, "review"),
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
        # The Web UI shows captions pretty-printed; Ideogram 4 reads the single-line form.
        prompt = compact_caption(parsed)
        negative_prompt = None
        prompts_data = {"prompt": [(prompt, None)]}
        args.json_prompt_enabled = True
        enhance_by_set = None
    else:
        _prompt_source, prompt, negative_prompt, prompts_data, enhance_by_set, _scene_by_set = _resolve_prompt_submission_with_enhance(form)
        enhance_by_set = image_enhance_entries(enhance_by_set, caption=_optional_text(form, "enhance_format") == "json")
    _apply_enhance_submission(args, form, mode="image", json_caption=args.json_prompt_enabled)
    args.prompt_input = _optional_text(form, "prompt_input")
    args.enhance_record = _enhance_record_from_form(form, mode="image", app_config=app_config)
    first_sigma = _optional_float(form, "first_sigma")
    if first_sigma is not None and not (0.0 < first_sigma <= 2.0):
        raise ValueError(f"first_sigma must be in (0.0, 2.0], got {first_sigma}.")
    args.first_sigma = first_sigma
    if (args.json_prompt_enabled or args.enhance_format == "json") and not defaults.get("supports_json_prompt", False):
        raise ValueError("This model does not support structured JSON captions.")
    if args.first_sigma is not None and not defaults.get("supports_first_sigma", False):
        raise ValueError("This model does not support the first-step sigma control.")
    # Capture explicit-intent flags on args BEFORE the resolved defaults overwrite
    # args.steps/args.guidance, so run_batch's authoritative request rebuild sees them
    # via getattr(args, ...) (mirrors image_cli.py's args.steps_explicit assignment).
    # The form always sends both, so only a value that differs from the model's default counts as chosen.
    model_defaults = resolve_defaults(model_info, app_config, {}, get_backend_name())
    args.steps_explicit = args.steps is not None and args.steps != model_defaults["steps"]
    args.guidance_explicit = args.guidance is not None and args.guidance != model_defaults["guidance"]
    args.steps = defaults["steps"]
    args.guidance = defaults["guidance"]
    # A model with its own sampler ignores a scheduler carried over from another model, as with negative prompts.
    args.scheduler = defaults["scheduler"] if defaults.get("supports_scheduler", True) else None
    validate_scheduler(args.scheduler, app_config)
    if args.upscale and args.upscale_steps is None:
        args.upscale_steps = resolve_upscale_steps(defaults, args.steps)

    if not defaults.get("supports_negative_prompt", False):
        negative_prompt = None
        prompts_data = _replace_prompt_negatives(prompts_data, negative_prompt=None)

    args.lora_paths, args.lora_weights = _resolve_loras(form, require_file=True)
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
        width=eff_width,
        height=eff_height,
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
    context = _public_job_context(
        form,
        mode="image",
        job_type="Image to Image" if args.image_path else "Text to Image",
        model=model_name,
        model_label=model_display_name(app_config, model_name),
        prompt=prompt,
        runs=args.runs,
        # A custom size has no ratio or size preset of its own; the ratio fields hold the form's defaults.
        meta=f"{eff_width}x{eff_height} · {args.steps} steps" if args.width else f"{args.ratio} · {args.size} · {args.steps} steps",
        width=eff_width,
        height=eff_height,
        output_dir=args.output,
        uploads={"image_path": args.image_path},
    )
    supported_controls = image_supported_controls(review=review_enabled(args))
    job_id = web_runner.submit_image_request_job(
        request=request,
        prompts_data=prompts_data,
        config=app_config,
        args=args,
        model_ref=resolved_model,
        quantize=args.quantize,
        enhance_by_set=enhance_by_set,
        admission_check=_reject_while_busy,
        context=context,
        supported_controls=supported_controls,
    )
    return {
        "job_id": job_id,
        "context": context,
        "supported_controls": supported_controls,
    }


def _submit_video_job(form: Any, web_config: WebUiConfig) -> dict[str, Any]:
    app_config = web_config.app_config
    workflow = _workflow_from_form(_optional_text(form, "workflow"), fallback="txt2vid")
    extend = workflow == "extend"
    if extend and not supports_video_extend():
        raise ValueError(EXTEND_UNAVAILABLE_MESSAGE)
    model_name = _video_model_name(form, web_config)

    # An extension takes its size, frame rate and length from the source clip, so the size and reference-image fields are ignored.
    ratio, size = (None, None) if extend else _video_ratio_and_size(form, web_config)
    output_dir = _resolve_output_dir(web_config.output_dir)
    image_path = None if extend else _resolve_reference_image(form, output_dir)
    end_image_path = None if extend else _resolve_reference_image(form, output_dir, path_key="end_image_path")
    # The source is checked again here: the file may have changed since the Workspace inspected it.
    source = load_source_video(_optional_text(form, "source_video"), key="source_video") if extend else None
    audio_enabled = _checkbox(form, "audio", default=True) and supports_video_audio()
    scene = _optional_text(form, "scene")
    if source is not None and "scene" not in form:
        scene = source.scene
    _prompt_source, prompt, _negative_prompt, prompts_data, enhance_by_set, scene_by_set = _resolve_prompt_submission_with_enhance(form, allow_empty_prompt=scene is not None or extend)
    upscale = None if extend else _optional_int(form, "upscale")
    args = argparse.Namespace(
        model=model_name,
        prompt=prompt,
        scene=scene,
        scene_input=_optional_text(form, "scene_input"),
        prompt_input=_optional_text(form, "prompt_input"),
        prompts_file=None,
        image_path=image_path,
        end_image_path=end_image_path,
        ratio=ratio,
        size=size,
        width=None if extend else _optional_int(form, "width"),
        height=None if extend else _optional_int(form, "height"),
        frames=None if extend else _optional_int(form, "frames"),
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
        upscale=upscale,
        upscale_steps=None,
        no_audio=not audio_enabled,
        audio=audio_enabled,
        review=bool(upscale) and _checkbox(form, "review"),
    )
    if review_enabled(args) and not supports_video_upscale_review():
        raise HTTPException(status_code=400, detail="Review before upscaling is not available for video on this platform.")
    _apply_enhance_submission(args, form, mode="video", json_caption=False)
    args.enhance_record = _enhance_record_from_form(form, mode="video", app_config=app_config)
    if _WORKFLOW_DEFINITIONS[workflow]["requires_reference_image"] and image_path is None:
        raise ValueError("Image-to-video requires a reference image.")
    if end_image_path is not None and image_path is None:
        raise ValueError("A last-frame image needs a first-frame reference image.")
    _validate_video_args(args)

    resolved_model, model_info = resolve_video_model(app_config, model_name)
    if image_path and not model_info.supports_i2v:
        raise ValueError(f"Model '{model_name}' does not support image-to-video.")
    refusal = unsupported_video_request(
        app_config,
        model_info.family,
        image=image_path is not None,
        extend=extend,
        review=review_enabled(args),
        end_image=end_image_path is not None,
        upscale=bool(args.upscale),
    )
    if refusal is not None:
        raise ValueError(refusal)

    args.lora_paths, args.lora_weights = _resolve_loras(form, require_file=False)
    if source is not None:
        # The window keeps the source's size and frame rate; no resolution or frame alignment applies.
        args.steps = clip_defaults(app_config, model_info, steps=args.steps)["steps"]
        family = video_family_on_platform(app_config, model_info.family)
        require_extend_family(source, family)
        apply_extend_args(
            args,
            source,
            requested_extend_frames(_optional_int(form, "extend_frames"), _optional_float(form, "extend_seconds"), source.fps, step=extend_step(family)),
            family,
        )
    else:
        clip = resolve_video_clip(app_config, model_info, ratio=args.ratio, size=args.size, steps=args.steps, width=args.width, height=args.height, frames=args.frames, upscale=bool(args.upscale))
        args.steps, args.width, args.height, args.num_frames = clip.steps, clip.width, clip.height, clip.num_frames
        args.fps = model_info.default_fps
    _normalize_video_steps(args, app_config, model_info)

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
        end_image_path=end_image_path,
        upscale=args.upscale,
        upscale_steps=args.upscale_steps,
        no_audio=args.no_audio,
        output_dir=args.output,
        output_format=args.format,
    )
    try:
        require_ffmpeg()
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc

    if source is not None:
        job_type, meta = "Extend video", f"{args.width}x{args.height} · +{args.extend_frames} frames · {args.steps} steps"
    else:
        job_type, meta = "Image to Video" if image_path else "Text to Video", f"{args.width}x{args.height} · {args.num_frames} frames · {args.steps} steps"
    context = _public_job_context(
        form,
        mode="video",
        job_type=job_type,
        model=model_name,
        model_label=model_display_name(app_config, model_name),
        prompt=prompt or scene or "",
        runs=args.runs,
        meta=meta,
        width=args.width,
        height=args.height,
        output_dir=args.output,
        uploads={"image_path": image_path, "end_image_path": end_image_path},
    )
    supported_controls = video_supported_controls(review=review_enabled(args))
    job_id = web_runner.submit_video_request_job(
        request=request,
        prompts_data=prompts_data,
        config=app_config,
        args=args,
        model_ref=resolved_model,
        enhance_by_set=enhance_by_set,
        scene_by_set=scene_by_set,
        admission_check=_reject_while_busy,
        context=context,
        supported_controls=supported_controls,
    )
    return {
        "job_id": job_id,
        "context": context,
        "supported_controls": supported_controls,
    }


def _resolve_prompt_submission_with_enhance(
    form: Any,
    *,
    allow_empty_prompt: bool = False,
) -> tuple[
    str,
    str,
    str | None,
    dict[str, list[tuple[str, str | None]]],
    dict[str, list[EnhanceSettings | None]] | None,
    dict[str, list[str | None]] | None,
]:
    """Resolve a prompt submission plus prompt-file ``enhance:`` settings and ``scene:`` texts (``None`` for inline prompts).

    Args:
        form: The submitted form.
        allow_empty_prompt: Accept a blank inline prompt (a video job whose scene carries the text).
    """
    prompt_source = _text_or_default(form, "prompt_source", DEFAULT_PROMPT_SOURCE)
    if prompt_source not in PROMPT_SOURCE_VALUES:
        raise ValueError(f"Unknown prompt source '{prompt_source}'.")
    if prompt_source == "file":
        option_ids = form.getlist("prompt_option_id") if hasattr(form, "getlist") else [form.get("prompt_option_id")]
        _normalized_path, options = resolve_prompt_file_options(
            _required_text(form, "prompts_file"),
            [str(option_id).strip() for option_id in option_ids if option_id and str(option_id).strip()],
            accepted_extensions=_PROMPT_FILE_EXTENSIONS,
        )
        prompts_data: dict[str, list[tuple[str, str | None]]] = {}
        for option in options:
            prompts_data.setdefault(option.set_name, []).append((option.prompt, option.negative_prompt))
        first = options[0]
        return prompt_source, first.prompt, first.negative_prompt, prompts_data, group_enhance_by_set(options), group_scene_by_set(options)

    prompt = (_optional_text(form, "prompt") or "") if allow_empty_prompt else _required_text(form, "prompt")
    negative_prompt = _optional_text(form, "negative_prompt")
    return prompt_source, prompt, negative_prompt, {"web": [(prompt, negative_prompt)]}, None, None


def _apply_enhance_submission(args: argparse.Namespace, form: Any, *, mode: str, json_caption: bool) -> None:
    """Set ``args.enhance`` from the auto-enhance toggle (job-wide settings) or ``None``, plus the job's enhancer
    (``enhance_model``: fast or smart) and, for images, what it writes (``enhance_format``: text or json); never for JSON captions.
    """
    args.no_enhance = False
    args.enhance = None
    args.enhance_model = None if json_caption else _form_choice(form, "enhance_model", ENHANCER_TIERS)
    args.enhance_format = None if json_caption or mode != "image" else _form_choice(form, "enhance_format", ENHANCE_FORMATS)
    if json_caption or not _checkbox(form, "enhance_auto"):
        return
    raw = _optional_text(form, "enhance_settings")
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"enhance_settings must be a JSON object: {exc}") from exc
    settings = settings_from_mapping(data, mode=mode, motion_in_image="warn")
    validate_settings(settings, mode=mode)
    args.enhance = settings


def _enhance_record_from_form(form: Any, *, mode: str, app_config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the ``enhance_record`` sent with Enhanced text (how the Web UI rewrite was made), validated; None when absent.

    The tier is worked out from the model, not taken from the browser. The record only describes how the text was made,
    so an invalid one (e.g. naming an option that has since been removed) is dropped with a warning instead of failing the job.
    """
    raw = _optional_text(form, "enhance_record")
    if raw is None:
        return None
    try:
        return parse_enhance_record(json.loads(raw), mode=mode, config=app_config, platform_key=sys.platform)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        warnings.warn(f"Ignoring the enhance record sent with the prompt: {exc}", stacklevel=2)
        return None


def _form_choice(form: Any, key: str, allowed: tuple[str, ...]) -> str | None:
    """Return the form's *key* when it is one of *allowed*, ``None`` when absent."""
    value = _optional_text(form, key)
    if value is not None and value not in allowed:
        raise ValueError(f"{key} must be one of: {', '.join(allowed)}.")
    return value


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


def _video_memory_fit_response(payload: Any, web_config: WebUiConfig) -> dict[str, Any]:
    """Build the ``POST /api/video/memory-fit`` response: the size and length a video job would use, and how that clip fits memory.

    The size and length resolve exactly as in a Text to Video or Image to Video job. ``image`` (an Image to Video clip,
    which holds a keyframe), ``steps`` (settled by the model's step rules), ``lora`` (the job's ``name:weight,...`` list,
    whose files H3 holds in memory) and ``review`` (the upscale pauses to review, so H3 loads its transformer twice)
    refine MiniMax H3's estimate; LTX ignores all four. ``memory_fit`` is ``None`` when this machine has no memory budget or the model has no estimate: an LTX model
    that is not a downloaded MLX model with its text encoder, or a MiniMax H3 clip size it cannot take. MiniMax H3
    adds ``expected_seconds``, an estimate of the generation time.

    Raises:
        ValueError: If the body, model, ratio, size, steps, upscale or review is invalid, or an H3 clip names a LoRA that is not there.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    for key in ("model", "ratio", "size"):
        json_field(payload, key, str, "a string")
    width, height, frames, upscale, steps = (json_field(payload, key, int, "a positive integer", minimum=1) for key in ("width", "height", "frames", "upscale", "steps"))
    low_memory = json_field(payload, "low_memory", bool, "a boolean")
    image = json_field(payload, "image", bool, "a boolean")
    lora = json_field(payload, "lora", str, "a string")
    review = json_field(payload, "review", bool, "a boolean")

    app_config = web_config.app_config
    model_name = _video_model_name(payload, web_config)
    ratio, size = _video_ratio_and_size(payload, web_config)
    require_upscale_factor(upscale)
    resolved_model, model_info = resolve_video_model(app_config, model_name)
    clip = resolve_video_clip(app_config, model_info, ratio=ratio, size=size, steps=steps, width=width, height=height, frames=frames, upscale=bool(upscale), quiet=True)
    forwards = _settled_video_steps(clip.steps, bool(upscale), app_config, model_info).steps
    memory_fit = clip_memory_fit(
        resolved_model,
        width=clip.width,
        height=clip.height,
        frames=clip.num_frames,
        low_memory=low_memory is not False,
        upscale=bool(upscale),
        budget=memory_budget(),
        model_name=model_name,
        app_config=app_config,
        family=model_info.family,
        mode="i2v" if image else "t2v",
        forwards=forwards or 8,
        loras_bytes=lora_bytes_for_fit(lora, family=model_info.family),
        review=bool(upscale) and review is True,
    )
    return {"memory_fit": memory_fit, "width": clip.width, "height": clip.height, "frames": clip.num_frames}


def _required_path(form: Any, key: str) -> str:
    """Return a validated path field or raise a field-specific error."""
    value = _optional_path(form, key)
    if value is None:
        raise ValueError(f"Field '{key}' is required.")
    return value


def _checkpoint_path(form: Any, key: str) -> str:
    """Return the checkpoint file the field *key* names, absolute but with symlinks kept (a Hugging Face cache file links to a blob named by its hash, and a sharded checkpoint is found through the links beside it).

    Raises:
        ValueError: As :func:`_required_path`.
    """
    _required_path(form, key)
    return str(Path(str(_optional_text(form, key))).expanduser().absolute())


def _upload_field(path_key: str) -> str:
    """Return the form field that uploads the file a path field names (``image_path`` -> ``image_file``)."""
    return f"{path_key.removesuffix('_path')}_file"


def _resolve_reference_image(form: Any, output_dir: str, *, path_key: str = "image_path") -> str | None:
    """Return the reference image a form carries: an upload (saved under the output folder) wins over a typed path.

    Args:
        form: The submitted form.
        output_dir: The output folder uploads are saved under.
        path_key: The path field to read; its upload field is derived from it (``end_image_path`` -> ``end_image_file``).
    """
    uploaded_file = form.get(_upload_field(path_key))
    if _is_uploaded_file(uploaded_file):
        return _save_uploaded_reference_image(uploaded_file, output_dir)
    return _optional_path(form, path_key)


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


def _resolve_loras(form: Any, *, require_file: bool) -> tuple[list[str] | None, list[float] | None]:
    lora_value = _optional_text(form, "lora")
    if lora_value is None:
        return None, None
    return resolve_lora_references(lora_value, require_file=require_file)


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
    if isinstance(args.sharpen, float):
        validate_sharpen_amount(args.sharpen)
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
    require_upscale_factor(args.upscale)


def _video_model_name(form: Any, web_config: WebUiConfig) -> str:
    """Return the video model a request names, else the configured default.

    Raises:
        ValueError: If neither names one.
    """
    model_name = _text_or_default(form, "model", web_config.default_models.video)
    if not model_name:
        raise ValueError("A video model is required.")
    return model_name


def _video_ratio_and_size(form: Any, web_config: WebUiConfig) -> tuple[str, str]:
    """Return the ratio and size a request names, else the configured defaults.

    Raises:
        ValueError: If a value is not one the Web UI offers.
    """
    ratio = _choice_or_default(form, "ratio", web_config.video_ratios, web_config.app_config.get("video_generation", {}).get("default_ratio", "16:9"))
    return ratio, _resolve_video_size(form, web_config, ratio)


def _settled_video_steps(steps: int | None, upscale: bool, config: dict[str, Any], model_info: VideoModelInfo) -> SettledSteps:
    """Settle the requested *steps* of a video job by the upscale and the model family's step rules."""
    family = video_family(config, model_info.family)
    return settle_video_steps(steps, upscale=upscale, family=family, default_upscale_steps=family.default_upscale_steps if family is not None else UNDESCRIBED_UPSCALE_STEPS)


def _normalize_video_steps(args: argparse.Namespace, config: dict[str, Any], model_info: VideoModelInfo) -> None:
    """Settle the step counts of a video job: upscale steps and the model family's step rules (silently)."""
    if args.steps is not None and args.steps < 1:
        raise ValueError("Steps must be at least 1.")
    settled = _settled_video_steps(args.steps, bool(args.upscale), config, model_info)
    args.steps = settled.steps
    args.upscale_steps = settled.upscale_steps


def _persist_web_config(form: Any) -> None:
    """Persist the backend-owned writable config patch."""
    persist_writable_config_patch(form, load_web_config())


def _convert_model_from_form(form: Any) -> dict[str, str]:
    """Validate and run checkpoint conversion from the Web models page.

    Raises:
        HTTPException: 409 when a quantized copy is requested while a generation job runs or another model is
            being quantized (each holds a full model in memory). Jobs are refused while it runs.
    """
    input_path = _checkpoint_path(form, "input_path")
    model_type = _optional_text(form, "model_type") or "zimage"
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Model type must be one of {', '.join(MODEL_TYPES)}.")

    args = ["model", "--input", input_path, "--model-type", model_type]
    model_name = _optional_text(form, "name")
    if model_name:
        args.extend(["--name", model_name])
    if model_type in VIDEO_MODEL_TYPES:
        return _start_video_conversion(form, args, model_type, model_name or Path(input_path).stem)
    if model_type == "zimage":
        args.extend(["--base-model", _optional_text(form, "base_model") or "Tongyi-MAI/Z-Image-Turbo"])
    if _checkbox(form, "copy"):
        args.append("--copy")
    quantize = _optional_int(form, "quantize")
    if quantize is not None:
        if quantize not in (4, 8):
            raise ValueError("Quantize must be 4 or 8.")
        args.extend(["--quantize", str(quantize)])
        try:
            with _claim_quantize_slot():
                detail = _run_model_management_command(args)
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    else:
        detail = _run_model_management_command(args)
    return {
        "tone": "success",
        "message": "Converted the checkpoint into an installed model directory." if quantize is None else f"Converted the checkpoint and saved a q{quantize} copy.",
        "detail": detail,
    }


def _start_video_conversion(form: Any, args: list[str], model_type: str, name: str) -> dict[str, str]:
    """Start a video conversion as a background job (see :mod:`zvisiongenerator.web.video_conversion`); it holds the quantize slot until it ends.

    An LTX conversion takes the quantize level and Distill; an H3 conversion takes "already step-distilled" (``turbo``).

    Raises:
        ValueError: Without a quantize level of 4 or 8 for LTX (the only level it stores).
        HTTPException: 409 while a generation job runs or another conversion holds the slot.
    """
    from zvisiongenerator.web.video_conversion import start_conversion

    quantize = _optional_int(form, "quantize")
    if model_type in LTX_MODEL_TYPES:
        if quantize not in (4, 8):
            raise ValueError(f"Converting an {model_type.upper()} checkpoint needs a quantize level of 4 or 8: that level is the only one stored.")
        args.extend(["--quantize", str(quantize)])
        if _checkbox(form, "distill"):
            args.append("--distill")
    elif model_type == H3_MODEL_TYPE and _checkbox(form, "turbo"):
        args.append("--turbo")
    slot = ExitStack()
    try:
        slot.enter_context(_claim_quantize_slot())
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    start_conversion(args, name, slot)
    level = f" at q{quantize}" if model_type in LTX_MODEL_TYPES else ""
    return {"tone": "success", "message": f"Converting the checkpoint into '{name}'{level}. This takes several minutes.", "detail": "", "background": "1"}


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
        encoding="utf-8",
        errors="replace",
        env=utf8_child_env(),
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
def api_workspace(include_history: bool = Query(True)) -> dict[str, Any]:
    """Return WorkspaceContext JSON for the Svelte SPA."""
    web_config = load_web_config()
    history_assets: list[dict[str, Any]] = []
    if include_history:
        all_assets = list_gallery_assets(web_config.output_dir)
        history_assets = [gallery_asset_to_json(asset, web_config) for asset in all_assets[:20]]
    return _build_workspace_response(web_config, history_assets)


@app.get("/api/gallery")
def api_gallery_json(
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
def api_get_config() -> dict[str, Any]:
    """Return current Web UI config as JSON for the Svelte SPA."""
    web_config = load_web_config()
    return build_api_config_response(web_config)


@app.post("/api/config")
async def api_save_config(request: Request) -> dict[str, Any]:
    """Update Web UI config from a JSON body and return the updated config."""
    payload = await request.json()
    try:
        await run_in_threadpool(_persist_web_config, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    web_config = await run_in_threadpool(load_web_config)
    return build_api_config_response(web_config)


@app.post("/api/first-run/skip")
async def api_first_run_skip() -> dict[str, bool]:
    """Remember that the first-run dialog was skipped, so the Workspace stops offering it."""
    try:
        await run_in_threadpool(persist_first_run_skipped)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"skipped": True}


@contextmanager
def _preset_http_errors() -> Iterator[None]:
    """Map the preset store's errors to HTTP statuses: 409 taken, 404 missing, 422 invalid, 500 unwritable."""
    try:
        yield
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not access the presets: {exc}") from exc


@app.get("/api/presets")
def api_presets_list() -> dict[str, Any]:
    """Return the saved settings presets of both modes, sorted by name."""
    return {"presets": [preset.to_dict() for preset in list_presets()]}


@app.post("/api/presets")
def api_presets_create(body: dict[str, Any]) -> dict[str, Any]:
    """Create a preset from a JSON body ``{name, mode, model, settings}``; the slug follows the name."""
    with _preset_http_errors():
        return create_preset(body).to_dict()


@app.put("/api/presets/{mode}/{slug}")
def api_presets_update(mode: str, slug: str, body: dict[str, Any]) -> dict[str, Any]:
    """Overwrite the preset *slug* of *mode* from a JSON body ``{name, model, settings}`` whose name keeps the slug."""
    with _preset_http_errors():
        return update_preset(mode, slug, body).to_dict()


@app.post("/api/presets/{mode}/{slug}/rename")
def api_presets_rename(mode: str, slug: str, body: dict[str, Any]) -> dict[str, Any]:
    """Rename the preset *slug* of *mode* from a JSON body ``{name}``; its slug follows the new name."""
    with _preset_http_errors():
        return rename_preset(mode, slug, body.get("name")).to_dict()


@app.delete("/api/presets/{mode}/{slug}")
def api_presets_delete(mode: str, slug: str) -> dict[str, str]:
    """Delete the preset *slug* of *mode*."""
    with _preset_http_errors():
        delete_preset(mode, slug)
    return {"status": "ok"}


@app.get("/api/models")
def api_models() -> dict[str, Any]:
    """Return installed model inventory as JSON for the Svelte SPA."""
    web_config = load_web_config()
    return _build_models_response(web_config, token_source=hub_token_source())


@app.post("/api/models/convert")
async def api_models_convert(request: Request) -> dict[str, Any]:
    """Convert a local checkpoint into an installed model directory (JSON API)."""
    payload = await request.json()
    try:
        notice = await run_in_threadpool(_convert_model_from_form, payload)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "tone": notice["tone"], "message": notice["message"], "background": bool(notice.get("background"))}


@app.post("/api/models/convert/inspect")
async def api_models_convert_inspect(request: Request) -> dict[str, Any]:
    """Inspect a video checkpoint before converting it as the model type given: a summary, advice, and what to preselect."""
    from zvisiongenerator.web.video_conversion import inspect_checkpoint_for_form

    payload = await request.json()
    try:
        path = Path(_checkpoint_path(payload, "input_path"))
        model_type = _optional_text(payload, "model_type") or LTX_MODEL_TYPES[0]
        return await run_in_threadpool(inspect_checkpoint_for_form, path, load_web_config().app_config, model_type)
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models/convert/status")
def api_models_convert_status() -> dict[str, Any]:
    """Return the state and progress of the current or last video conversion."""
    from zvisiongenerator.web.video_conversion import conversion_status

    return conversion_status()


@app.post("/api/models/import-lora/local")
async def api_models_import_lora_local(request: Request) -> dict[str, Any]:
    """Import a local LoRA file into the configured data directory (JSON API)."""
    payload = await request.json()
    try:
        notice = await run_in_threadpool(_import_local_lora_from_form, payload)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "tone": notice["tone"], "message": notice["message"]}


@app.post("/api/models/import-lora/hf")
async def api_models_import_lora_hf(request: Request) -> dict[str, Any]:
    """Import a LoRA from Hugging Face into the configured data directory (JSON API)."""
    payload = await request.json()
    try:
        notice = await run_in_threadpool(_import_hf_lora_from_form, payload)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "tone": notice["tone"], "message": notice["message"]}


# The download routes are declared before DELETE /api/models/{name} so that path pattern cannot shadow them.
@app.get("/api/models/downloads")
def api_models_downloads() -> dict[str, Any]:
    """Return the download lane: running, queued and recently finished Models-page downloads."""
    return download_lane.snapshot()


@app.get("/api/models/download/plan")
async def api_models_download_plan(kind: str, name: str) -> dict[str, Any]:
    """Return what downloading a Models-page row would fetch: parts, sizes and the free space on the download drive."""
    web_config = await run_in_threadpool(load_web_config)
    try:
        plan = await run_in_threadpool(row_plan, kind, name, web_config)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    try:
        return await run_in_threadpool(plan_summary, plan)
    except PlanUnchecked as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc


@app.post("/api/models/download")
async def api_models_download(request: Request) -> dict[str, Any]:
    """Queue the download of a Models-page row; an already queued or running one for the same row is returned."""
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Expected a JSON object.")
    kind = _required_json_string(payload, "kind")
    name = _required_json_string(payload, "name")
    try:
        plan = await run_in_threadpool(lambda: row_plan(kind, name, load_web_config()))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    try:
        return download_lane.submit(kind=kind, name=name, label=plan.label, plan=lambda: row_plan(kind, name, load_web_config()).targets)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/models/downloads/{item_id}/cancel")
def api_models_download_cancel(item_id: str) -> dict[str, Any]:
    """Cancel a queued download, or stop the running one (finished files are kept)."""
    try:
        return download_lane.cancel(item_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown download: {item_id}") from exc


def _seedvr2_copy_target(name: str, setup: SeedVR2Setup, models_dir: Path) -> DeleteTarget | None:
    """Return the delete target when *name* is the Models-page row of the SeedVR2 tier in use and its weights are on disk."""
    if name != seedvr2_row_name(setup.tier.model, setup.weights):
        return None
    return seedvr2_delete_target(setup.tier.model, models_dir, setup.weights)


@app.delete("/api/models/{name}")
def api_delete_model(name: str) -> dict[str, Any]:
    """Delete an installed model folder, or an alias model's HuggingFace download (the alias itself stays)."""
    _ensure_no_active_job()
    web_config = load_web_config()
    entry = next((entry for entry in (*web_config.image_inventory, *web_config.video_inventory) if entry.name == name), None)
    inventory = (*web_config.image_inventory, *web_config.video_inventory)
    target = model_delete_target(entry, Path(web_config.data_dir) / "models", **delete_scope(entry, inventory, web_config.app_config)) if entry is not None else None
    if entry is None and web_config.seedvr2 is not None:
        target = _seedvr2_copy_target(name, web_config.seedvr2, Path(web_config.models_dir))
    if target is None:
        raise HTTPException(status_code=404, detail=f"No deletable model named '{name}'.")
    _ensure_not_downloading(repo_ids=(target.repo_id,) if target.repo_id else (), paths=(target.path,))
    try:
        delete_model(target)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model '{name}' has nothing on disk to delete.") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete model '{name}': {exc}") from exc
    messages = {"installed": f"Deleted model '{name}'.", "copy": f"Deleted the prepared copy of '{name}'."}
    message = messages.get(target.kind, f"Deleted the HuggingFace download of '{target.repo_id}'.")
    return {"status": "ok", "tone": "success", "message": message}


@app.delete("/api/enhancer-models/{tier}")
def api_delete_enhancer_model(tier: str) -> dict[str, Any]:
    """Delete the HuggingFace download of the prompt-enhancer model configured for *tier* (``fast``, ``smart``, ``custom``)."""
    model = enhancer_tier_models(load_web_config().app_config, platform_key=sys.platform).get(tier)
    target = enhancer_delete_target(model[0]) if model is not None else None
    if target is None:
        raise HTTPException(status_code=404, detail=f"No downloaded enhancer model for '{tier}'.")
    session = get_prompt_enhancer_session()
    reservation: list[object] = []

    def _reserve() -> None:
        token = session.reserve()
        if token is None:
            raise JobConflictError("Wait for the prompt enhancement to finish before deleting its model.")
        reservation.append(token)

    try:
        # Holding the reservation keeps enhancements and jobs from starting while the files go.
        web_runner.admit_exclusive(_reserve, busy_message="Wait for all jobs to finish before deleting models or LoRAs.")
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        _ensure_not_downloading(repo_ids=(target.repo_id,))
        session.release()
        delete_model(target)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No downloaded enhancer model for '{tier}'.") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete enhancer model '{target.repo_id}': {exc}") from exc
    finally:
        session.cancel_reservation(reservation[0])
    return {"status": "ok", "tone": "success", "message": f"Deleted the HuggingFace download of '{target.repo_id}'."}


# Declared before /api/hf-cache/{repo_id:path}, which would otherwise match it.
@app.delete("/api/hf-cache/stranded")
def api_free_stranded_shared_files() -> dict[str, Any]:
    """Delete the files in the HuggingFace cache's shared store that no downloaded repo links to."""
    _ensure_no_active_job()
    _ensure_not_downloading(any_cache=True)
    try:
        freed = free_stranded_shared_files(huggingface_cache_dir())
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete unused cache files: {exc}") from exc
    return {"status": "ok", "tone": "success", "message": f"Freed {disk_size_label(freed)} of unused cache files."}


@app.delete("/api/hf-cache/{repo_id:path}")
def api_delete_other_download(repo_id: str) -> dict[str, Any]:
    """Delete a HuggingFace download nothing on the Models page uses; downloads in use have their own delete."""
    _ensure_no_active_job()
    repo = next((repo for repo in find_other_downloads(load_web_config()) if repo.repo_id == repo_id), None)
    if repo is None:
        raise HTTPException(status_code=404, detail=f"No unused HuggingFace download named '{repo_id}'.")
    _ensure_not_downloading(repo_ids=(repo.repo_id,))
    try:
        delete_model(DeleteTarget("huggingface", repo.path, repo_id=repo.repo_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No unused HuggingFace download named '{repo_id}'.") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete '{repo_id}': {exc}") from exc
    return {"status": "ok", "tone": "success", "message": f"Deleted the HuggingFace download of '{repo_id}'."}


@app.delete("/api/loras/{name}")
def api_delete_lora(name: str) -> dict[str, Any]:
    """Delete a LoRA file from the data directory."""
    _ensure_no_active_job()
    try:
        delete_lora(Path(load_web_config().data_dir) / "loras", name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete LoRA '{name}': {exc}") from exc
    return {"status": "ok", "tone": "success", "message": f"Deleted LoRA '{name}'."}


def _ensure_no_active_job() -> None:
    """Refuse model and LoRA deletes while a generation job may be reading their files."""
    if web_runner.get_active_exclusive_job_snapshot() is not None:
        raise HTTPException(status_code=409, detail="Wait for all jobs to finish before deleting models or LoRAs.")


def _ensure_not_downloading(*, repo_ids: tuple[str, ...] = (), paths: tuple[Path, ...] = (), any_cache: bool = False) -> None:
    """Refuse a delete while a Models-page download is writing to the repos, folders or cache it would remove."""
    repo = download_flights.conflict(repo_ids=repo_ids, paths=paths, any_cache=any_cache)
    if repo is not None:
        raise HTTPException(status_code=409, detail=f"Wait for the download of {repo} to finish, or cancel it, before deleting it.")


def _job_log_unavailable(exc: JobLogError) -> HTTPException:
    """Return the 500 for a job log that could not be read or written."""
    return HTTPException(status_code=500, detail=f"Could not read or write the job log: {exc}")


# Declared before the parameterised /api/jobs/{job_id} routes.
@app.get("/api/jobs")
def api_list_jobs() -> dict[str, Any]:
    """Return the active generation job and the queued ones, oldest first."""
    return web_runner.list_jobs()


@app.get("/api/jobs/log")
def api_job_log(page: int = Query(1, ge=1), status: Literal["all", "failed", "completed", "cancelled"] = "all") -> dict[str, Any]:
    """Return one newest-first page of the finished-job log, filtered by outcome."""
    try:
        return job_log.page(status, page, JOB_LOG_PAGE_SIZE)
    except JobLogError as exc:
        raise _job_log_unavailable(exc) from exc


@app.delete("/api/jobs/log/{job_id}")
def api_delete_job_log_entry(job_id: str) -> dict[str, str]:
    """Remove one entry from the job log; 404 when it is not there."""
    try:
        deleted = job_log.delete(job_id)
    except JobLogError as exc:
        raise _job_log_unavailable(exc) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown log entry: {job_id}")
    return {"deleted": job_id}


@app.delete("/api/jobs/log")
def api_clear_job_log() -> dict[str, int]:
    """Remove every entry from the job log and return how many there were."""
    try:
        return {"deleted_count": job_log.clear()}
    except JobLogError as exc:
        raise _job_log_unavailable(exc) from exc


@app.delete("/api/jobs/queue")
def api_clear_queue() -> dict[str, list[str]]:
    """Remove every queued job; the active job keeps running."""
    return {"cancelled": web_runner.clear_queue()}


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a running job. Returns cancelled status or 404 if the job is unknown."""
    try:
        snapshot = web_runner.get_job_snapshot(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
    if snapshot["status"] in {"completed", "failed", "cancelled"}:
        return {"job_id": job_id, "status": snapshot["status"]}
    # A job that has not started is removed from the queue, whatever controls it supports once running.
    if web_runner.cancel_queued(job_id):
        return {"job_id": job_id, "status": "cancelled"}
    supported_controls = set(snapshot.get("supported_controls") or [])
    if "quit" not in supported_controls and "cancel" not in supported_controls:
        raise HTTPException(status_code=409, detail="This job does not support cancellation.")
    try:
        return web_runner.queue_job_control(job_id, "quit")
    except UnsupportedJobControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/api/gallery/{asset_path:path}")
def api_delete_gallery_asset(asset_path: str) -> dict[str, str]:
    """Delete a single gallery asset by output-root-relative asset ID."""
    web_config = load_web_config()
    root = Path(web_config.output_dir).resolve()
    candidate = resolve_output_asset_path(root, asset_path)
    if candidate is None or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")
    delete_gallery_assets(web_config.output_dir, [asset_path])
    return {"status": "deleted"}
