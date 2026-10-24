"""Own gallery inventory scanning and gallery response serialization."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import time
from typing import Any
from urllib.parse import quote, unquote, urlencode

from PIL import Image, UnidentifiedImageError

from zvisiongenerator.backends import supports_video_extend, supports_video_file_upscale
from zvisiongenerator.core.ideogram4 import is_json_caption
from zvisiongenerator.utils import app_db
from zvisiongenerator.utils.enhance_record import enhance_record_summary
from zvisiongenerator.utils.ffmpeg import STAGING_FILE_PREFIX
from zvisiongenerator.utils.provenance import (
    UPSCALE_WORKFLOW,
    RecordedSettings,
    image_prompt_text,
    mp4_config_from_probe,
    optional_float,
    optional_int,
    optional_text,
    read_png_config,
    recorded_settings,
)
from zvisiongenerator.utils.config import model_capabilities
from zvisiongenerator.utils.seedvr2 import SeedVR2Setup, seedvr2_download_bytes, seedvr2_size_label, seedvr2_tier_label
from zvisiongenerator.utils.upscale import UPSCALE_FACTORS, UpscaleOption, seedvr2_options, seedvr2_video_option, upscale_options, upscale_output_size
from zvisiongenerator.utils.video_probe import ffprobe_json, parse_probe
from zvisiongenerator.utils.video_splice import SPLICE_DIR_PREFIX
from zvisiongenerator.web.config import WebUiConfig, preferred_option
from zvisiongenerator.web.gallery_index import FactsRead, GalleryIndex, ScannedFile
from zvisiongenerator.web.workspace_contract import WORKFLOW_DEFINITIONS, canonicalize_workflow, default_workflow_for_mode, workflow_mode


_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".mkv"})
GALLERY_MEDIA_EXTENSIONS = _IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
_STAGING_DIR_NAMES = frozenset({".web_uploads"})
# Scratch names the video tools create beside the finished outputs: the splice's directories (see ``video_splice``),
# and the hidden temp files of an in-place MP4 rewrite and the video upscale's work folders (see ``ffmpeg.staging_path``).
_STAGING_PREFIXES = (SPLICE_DIR_PREFIX, STAGING_FILE_PREFIX)
EXTEND_UPSCALED_REASON = "Upscaled videos can't be extended yet."
EXTEND_MISSING_FILE_REASON = "The file is missing."
_GALLERY_INDEX = GalleryIndex(app_db.app_db_path)


@dataclass(frozen=True)
class GalleryAsset:
    """Represent gallery item metadata derived from one output file."""

    id: str
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
    has_reusable_config: bool
    negative_prompt: str | None = None
    scene: str | None = None
    scene_input: str | None = None
    prompt_input: str | None = None
    # How the prompt was enhanced: ``{settings, model, tier, format}``; None when it was not, or the file predates it.
    enhance: dict[str, Any] | None = None
    scheduler: str | None = None
    model_family: str | None = None
    image_strength: float | None = None
    generation: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] | None = None
    # An extension: the clip it continued and the frames it added; ``upscaled`` marks a video that was upscaled.
    source_video: str | None = None
    extend_frames: int | None = None
    upscaled: bool = False
    # The image a first-and-last-frame clip ends on.
    end_image_path: str | None = None


def list_gallery_assets(output_dir: str) -> list[GalleryAsset]:
    """Scan an output directory for renderable image and video assets.

    The walk only stats files; the facts read from each file come from the gallery index, which reads a file again
    only when it is new or changed.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return []

    files, walk_complete = _scan_output_root(root)
    facts = _GALLERY_INDEX.facts_for(_index_root_key(root), files, _read_scanned_facts, prune=walk_complete and bool(files))
    assets = [asset for file in files if (asset := _build_gallery_asset(root, file, facts[file.rel_path])) is not None]
    assets.sort(key=lambda item: (item.modified_at, item.id), reverse=True)
    return assets


def gallery_asset_for_output_path(output_dir: str, output_path: str) -> GalleryAsset | None:
    """Build gallery metadata for one generated output under the configured output root.

    The file is read directly: the gallery index is neither consulted nor written.
    """
    root = Path(output_dir).expanduser().resolve()
    candidate = Path(output_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(root):
        return None
    if _is_hidden_from_inventory(root, resolved):
        return None
    file = _scanned_file(root, resolved)
    if file is None:
        return None
    return _build_gallery_asset(root, file, _read_scanned_facts(file).facts)


def _index_root_key(root: Path) -> str:
    """Return the key the gallery index stores *root* under: its resolved path."""
    try:
        return str(root.expanduser().resolve())
    except OSError:
        return str(root.expanduser().absolute())


def _scanned_file(root: Path, path: Path) -> ScannedFile | None:
    """Stat *path* and describe it as a media file under *root*; ``None`` when it vanished or is not a regular file."""
    try:
        st = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    return ScannedFile(path.relative_to(root).as_posix(), path, st.st_mtime_ns, st.st_ctime_ns, st.st_mtime, st.st_size)


def _scan_output_root(root: Path) -> tuple[list[ScannedFile], bool]:
    """Walk *root* with stat only and return its media files and whether the walk was error-free.

    Staging directories are never descended into and staging or non-media names are skipped before any stat.
    A file that vanishes during the walk is skipped.
    """
    errors: list[OSError] = []
    files: list[ScannedFile] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=errors.append):
        dirnames[:] = [name for name in dirnames if not _is_staging_part(name)]
        for name in filenames:
            path = Path(dirpath, name)
            if _is_staging_part(name) or _asset_kind(path) is None:
                continue
            file = _scanned_file(root, path)
            if file is not None:
                files.append(file)
    return files, not errors


def filter_and_sort_assets(assets: list[GalleryAsset], *, media_filter: str, sort_order: str) -> list[GalleryAsset]:
    """Apply gallery filter and sort controls to a list of assets."""
    normalized_filter = media_filter.strip().lower()
    if normalized_filter not in {"all", "image", "video"}:
        normalized_filter = "all"
    normalized_sort = sort_order.strip().lower()
    if normalized_sort not in {"newest", "oldest"}:
        normalized_sort = "newest"

    filtered = [asset for asset in assets if normalized_filter == "all" or asset.kind == normalized_filter]
    return sorted(filtered, key=lambda asset: asset.modified_at, reverse=normalized_sort == "newest")


def build_gallery_page_json(assets: list[GalleryAsset], web_config: WebUiConfig, *, page: int, page_size: int) -> dict[str, Any]:
    """Build a paginated gallery response."""
    total_count = len(assets)
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    page_assets, _ = _paginate_assets(assets, page=page, page_size=page_size)
    return {
        "assets": [gallery_asset_to_json(asset, web_config) for asset in page_assets],
        "page": page,
        "total_pages": total_pages,
        "total_count": total_count,
    }


def gallery_asset_to_json(asset: GalleryAsset, web_config: WebUiConfig) -> dict[str, Any]:
    """Convert one gallery asset to the current SPA JSON shape."""
    created_at = datetime.fromtimestamp(asset.modified_at, tz=timezone.utc).isoformat()
    default_workflow = default_workflow_for_mode(asset.kind)
    # An upscale is regenerated with the workflow that made its source.
    recorded_workflow = asset.source.get("workflow") if asset.source and asset.source.get("workflow") else asset.workflow
    requested_workflow = canonicalize_workflow(recorded_workflow, fallback=default_workflow)
    fallback_reasons: list[str] = []
    workflow_available = True
    if workflow_mode(requested_workflow) != asset.kind:
        requested_workflow = default_workflow
        workflow_available = False
        fallback_reasons.append("workflow_media_mismatch")

    resolved_workflow = requested_workflow
    video_extend = supports_video_extend()
    if resolved_workflow == "extend" and not video_extend:
        resolved_workflow = default_workflow
        workflow_available = False
        fallback_reasons.append("extend_unavailable")
    if WORKFLOW_DEFINITIONS[resolved_workflow]["requires_reference_image"] and asset.reference_image_path is None:
        resolved_workflow = default_workflow
        workflow_available = False
        fallback_reasons.append("missing_reference_image")

    model_options = web_config.image_model_options if workflow_mode(resolved_workflow) == "image" else web_config.video_model_options
    default_model = preferred_option(
        web_config.default_models.image if workflow_mode(resolved_workflow) == "image" else web_config.default_models.video,
        model_options,
    )
    requested_model = None if asset.model_label == "Unavailable" else asset.model_label
    if requested_model is None:
        model_available = True
        resolved_model = None
    else:
        model_available = requested_model in model_options
        resolved_model = requested_model if model_available else default_model
        if not model_available:
            fallback_reasons.append("model_not_configured")

    extending = resolved_workflow == "extend"
    if extending and not _source_video_exists(asset.source_video):
        # The workflow stays extend: the source is chosen again in the Workspace.
        workflow_available = False
        fallback_reasons.append("missing_source_video")

    reuse_params: dict[str, str] = {"workflow": resolved_workflow}
    if asset.has_reusable_config:
        reuse_params.update(_reuse_text_params(asset, video=workflow_mode(resolved_workflow) == "video"))
        if resolved_model:
            reuse_params["model"] = resolved_model
        # Always named, even blank: a reused job runs with exactly the source's LoRAs, never the ones already chosen.
        reuse_params["lora"] = asset.lora or ""
        if workflow_mode(resolved_workflow) == "image":
            # Always named, even blank, like the LoRAs. A model that fell back to the default is not the one the level was
            # recorded with (a stored-quant ``<name>@q<bits>`` id is such a case), so the level is not carried over.
            level = asset.generation.get("quantize")
            reuse_params["quantize"] = str(level) if level is not None and model_available else ""
        if asset.steps is not None:
            reuse_params["steps"] = str(asset.steps)
        if asset.guidance is not None and workflow_mode(resolved_workflow) == "image":
            reuse_params["guidance"] = f"{asset.guidance:g}"
        if asset.seed is not None:
            reuse_params["seed"] = str(asset.seed)
        if extending:
            # An extension takes its size, frame rate and length from the source; only the source and the new frames carry over.
            reuse_params.update(_extend_reuse_params(asset, source_missing="missing_source_video" in fallback_reasons))
            # Both boxes are named even when blank: a Reuse that recorded no text keeps them blank instead of taking the source's.
            reuse_params.setdefault("scene", "")
            reuse_params.setdefault("prompt", "")
        else:
            if asset.ratio is not None:
                reuse_params["ratio"] = asset.ratio
            if asset.size is not None:
                reuse_params["size"] = asset.size
            if asset.width is not None:
                reuse_params["width"] = str(asset.width)
            if asset.height is not None:
                reuse_params["height"] = str(asset.height)
            if asset.frame_count is not None and workflow_mode(resolved_workflow) == "video":
                reuse_params["frames"] = str(asset.frame_count)
        if WORKFLOW_DEFINITIONS[resolved_workflow]["requires_reference_image"] and asset.reference_image_path is not None:
            reuse_params["image_path"] = asset.reference_image_path
            # The last frame travels with its first frame, so a Reuse that falls back to text-to-video drops both.
            if resolved_workflow == "img2vid" and asset.end_image_path is not None:
                reuse_params["end_image_path"] = asset.end_image_path
        if asset.negative_prompt and workflow_mode(resolved_workflow) == "image":
            reuse_params["negative_prompt"] = asset.negative_prompt
        if asset.scheduler and workflow_mode(resolved_workflow) == "image":
            reuse_params["scheduler"] = asset.scheduler
        # An upscale is regenerated from its source settings, at the source's size.
        if asset.source is not None:
            if asset.source.get("width") is not None:
                reuse_params["width"] = str(asset.source["width"])
            if asset.source.get("height") is not None:
                reuse_params["height"] = str(asset.source["height"])
    return {
        "id": asset.id,
        "url": asset.media_url,
        "thumbnail_url": asset.media_url,
        "filename": asset.name,
        "created_at": created_at,
        "workflow": requested_workflow,
        "prompt": asset.prompt or asset.scene or "",
        "scene": asset.scene,
        "model": asset.model_label,
        "width": asset.width,
        "height": asset.height,
        "ratio": asset.ratio,
        "size": asset.size,
        "frame_count": asset.frame_count,
        "source_video": asset.source_video,
        "extend_frames": asset.extend_frames,
        "image_path": asset.reference_image_path,
        "end_image_path": asset.end_image_path,
        "file_path": asset.filesystem_path,
        "seed": asset.seed,
        "steps": asset.steps,
        "guidance": asset.guidance,
        "lora": asset.lora,
        "media_type": asset.kind,
        "has_reusable_config": asset.has_reusable_config,
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
        "details": _asset_details_json(asset),
        "enhance": _asset_enhance_json(asset),
        "upscale": _asset_upscale_json(asset, web_config, resolved_model) if asset.kind == "image" else _video_upscale_json(asset, web_config.seedvr2),
        "extend": _asset_extend_json(asset) if asset.kind == "video" and video_extend else None,
    }


def _asset_enhance_json(asset: GalleryAsset) -> dict[str, Any] | None:
    """Return how the asset's prompt was enhanced, for the viewer: the one-line summary and the user's own text; None when it was not."""
    if asset.enhance is None:
        return None
    return {"summary": enhance_record_summary(asset.enhance), "prompt_input": asset.prompt_input, "scene_input": asset.scene_input, **asset.enhance}


def _asset_extend_json(asset: GalleryAsset) -> dict[str, Any]:
    """Return whether a video can be extended as ``{"available", "reason"}``; the reason is ``None`` when it can."""
    if asset.upscaled:
        return {"available": False, "reason": EXTEND_UPSCALED_REASON}
    if not Path(asset.filesystem_path).is_file():
        return {"available": False, "reason": EXTEND_MISSING_FILE_REASON}
    return {"available": True, "reason": None}


def _extend_reuse_params(asset: GalleryAsset, *, source_missing: bool) -> dict[str, str]:
    """Return the Reuse query params that belong to an extension: the source, its new frames and, when the source moved, the flag."""
    params: dict[str, str] = {}
    if asset.source_video:
        params["source_video"] = asset.source_video
    if asset.extend_frames is not None:
        params["extend_frames"] = str(asset.extend_frames)
    if source_missing:
        params["missing_source_video"] = "true"
    return params


def _source_video_exists(path: str | None) -> bool:
    """Return whether the recorded source video of an extension is still a file."""
    if not path:
        return False
    try:
        return Path(path).expanduser().is_file()
    except OSError:
        return False


def output_media_url(root: Path, path: Path) -> str | None:
    """Return the ``/media`` URL of a media file inside the output root, or ``None`` when the server would not serve it."""
    try:
        asset_id = path.resolve().relative_to(root.resolve()).as_posix()
    except OSError, ValueError:
        return None
    if normalize_asset_id(asset_id) is None or path.suffix.lower() not in GALLERY_MEDIA_EXTENSIONS:
        return None
    return f"/media/{quote(asset_id, safe='/')}"


def _reuse_text_params(asset: GalleryAsset, *, video: bool) -> dict[str, str]:
    """Return the Reuse query params for an asset's text: the user's own text, plus the generated text where it differs.

    An enhanced video asset records its generated scene apart from its generated prompt, so the prompt params never
    carry scene text. ``enhanced_scene`` and ``enhanced_prompt`` appear only for a part whose recorded text differs
    from the recorded input of that part: a run that rewrote only the action reuses with the user's scene and the
    enhanced action. Without recorded inputs (plain or older assets) the recorded text is what the user wrote.
    """
    if not video:
        if asset.prompt_input and asset.prompt != asset.prompt_input:
            # An enhanced image: the user's text in Your text and the rewrite (or caption) in Enhanced, with how it was made.
            return {"prompt": asset.prompt_input, "enhanced_prompt": asset.prompt, **_reuse_enhance_params(asset)}
        # A structured caption reuses into the JSON box: as a text prompt its braces would read as {a|b} choices.
        return {"json_prompt": asset.prompt} if is_json_caption(asset.prompt) else {"prompt": asset.prompt}
    if asset.scene_input or asset.prompt_input:
        pairs = {
            "scene": asset.scene_input,
            "prompt": asset.prompt_input,
            "enhanced_scene": asset.scene if asset.scene != asset.scene_input else None,
            "enhanced_prompt": asset.prompt if asset.prompt != asset.prompt_input else None,
        }
    else:
        pairs = {"scene": asset.scene, "prompt": asset.prompt}
    params = {key: value for key, value in pairs.items() if value}
    if params.get("enhanced_scene") or params.get("enhanced_prompt"):
        params.update(_reuse_enhance_params(asset))
    return params


def _reuse_enhance_params(asset: GalleryAsset) -> dict[str, str]:
    """Return the ``enhance`` Reuse param: the recorded options, model, tier and format as JSON, when the asset has them."""
    return {"enhance": json.dumps(asset.enhance, separators=(",", ":"))} if asset.enhance else {}


def delete_gallery_assets(output_dir: str, selected_paths: list[str]) -> None:
    """Delete selected gallery media assets."""
    root = Path(output_dir).resolve()
    removed: list[str] = []
    for asset_id in dict.fromkeys(path for path in selected_paths if path.strip()):
        rel_path = normalize_asset_id(asset_id)
        candidate = resolve_output_asset_path(root, asset_id)
        if rel_path is None or candidate is None or not candidate.is_file():
            continue
        candidate.unlink(missing_ok=True)
        removed.append(rel_path)
    _GALLERY_INDEX.forget(_index_root_key(root), removed)


def resolve_output_asset_path(root: Path, asset_id: str) -> Path | None:
    """Resolve an output-root-relative POSIX asset ID safely under the configured root."""
    normalized = normalize_asset_id(asset_id)
    if normalized is None:
        return None
    candidate = (root / normalized).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate


def normalize_asset_id(asset_id: str) -> str | None:
    """Return a canonical output-root-relative POSIX asset ID, or None when invalid."""
    text = unquote(str(asset_id)).strip()
    if not text or "\\" in text or re.match(r"^[A-Za-z]:", text):
        return None
    path = PurePosixPath(text)
    if path.is_absolute():
        return None
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} or _is_staging_part(part) for part in parts):
        return None
    return path.as_posix()


def _build_gallery_asset(root: Path, file: ScannedFile, facts: dict[str, Any]) -> GalleryAsset | None:
    """Build the asset of one scanned file from its facts; per-request fields are computed here, never stored."""
    candidate = file.path
    kind = _asset_kind(candidate)
    if kind is None:
        return None
    asset_id = file.rel_path
    metadata = _asset_metadata(candidate, kind, facts)
    return GalleryAsset(
        id=asset_id,
        name=candidate.name,
        kind=kind,
        extension=candidate.suffix.lower().lstrip("."),
        filesystem_path=str(candidate),
        modified_at=file.mtime,
        modified_label=_format_age(file.mtime),
        path_label=asset_id,
        media_url=f"/media/{quote(asset_id, safe='/')}",
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
        end_image_path=metadata["end_image_path"],
        lora=metadata["lora"],
        has_reusable_config=metadata["has_reusable_config"],
        negative_prompt=metadata["negative_prompt"],
        scene=metadata["scene"],
        scene_input=metadata["scene_input"],
        prompt_input=metadata["prompt_input"],
        enhance=metadata["enhance"],
        scheduler=metadata["scheduler"],
        model_family=metadata["model_family"],
        image_strength=metadata["image_strength"],
        generation=metadata["generation"],
        source=_source_with_asset_id(root, metadata["source"]),
        source_video=metadata["source_video"],
        extend_frames=metadata["extend_frames"],
        upscaled=metadata["upscaled"],
    )


def _asset_metadata(asset_path: Path, kind: str, facts: dict[str, Any]) -> dict[str, Any]:
    """Derive the display and reuse metadata of an asset from the facts read from its file."""
    # Embedded config (PNG tEXt chunk / MP4 container tag) is the only reusable generation settings source.
    config = facts.get("config") or {}
    recorded = recorded_settings(config)
    filename_metadata = _parse_generated_filename(asset_path)
    image_metadata = facts.get("image") or {}

    prompt = recorded.prompt or image_metadata.get("prompt")
    if not prompt and (recorded.scene or (kind == "video" and config)):
        # A video that recorded its settings but no prompt (scene-only, or a blank-text extension) stays empty; the file
        # name stands in only for assets with no recorded settings, where it is display text and never reusable text.
        prompt = ""
    else:
        prompt = _coerce_text(prompt or asset_path.stem.replace("_", " "))
    video_metadata = facts.get("video") or {}
    width = recorded.width or optional_int(image_metadata.get("width") or filename_metadata.get("width")) or optional_int(video_metadata.get("width"))
    height = recorded.height or optional_int(image_metadata.get("height") or filename_metadata.get("height")) or optional_int(video_metadata.get("height"))

    return {
        "prompt": prompt,
        "model_label": _coerce_text(recorded.model),
        "width": width,
        "height": height,
        "seed": recorded.seed,
        "steps": recorded.steps,
        "guidance": recorded.guidance,
        "dimensions_label": f"{width}x{height}" if width is not None and height is not None else "Unavailable",
        "seed_label": str(recorded.seed) if recorded.seed is not None else "Unavailable",
        "steps_label": str(recorded.steps) if recorded.steps is not None else "Unavailable",
        "guidance_label": _format_guidance(recorded.guidance),
        "workflow": recorded.workflow,
        "ratio": recorded.ratio,
        "size": recorded.size,
        "frame_count": recorded.frame_count,
        "reference_image_path": recorded.image_path,
        "end_image_path": recorded.end_image_path,
        "lora": recorded.lora,
        "has_reusable_config": bool(config) and not (kind == "video" and _is_orphan_upscale(recorded)),
        "negative_prompt": recorded.negative_prompt,
        "scene": recorded.scene,
        "scene_input": recorded.scene_input,
        "prompt_input": recorded.prompt_input,
        "enhance": recorded.enhance,
        "scheduler": recorded.scheduler,
        "model_family": recorded.model_family,
        "image_strength": recorded.image_strength,
        "generation": recorded.generation,
        "source": recorded.source,
        "source_video": recorded.source_video,
        "extend_frames": recorded.extend_frames,
        "upscaled": bool(recorded.generation.get("upscale")),
    }


def _is_orphan_upscale(recorded: RecordedSettings) -> bool:
    """Return whether *recorded* is an upscale of a video that recorded no settings, so there is nothing to reuse."""
    return recorded.workflow == UPSCALE_WORKFLOW and recorded.source is not None and recorded.source.get("workflow") is None


def _asset_details_json(asset: GalleryAsset) -> dict[str, Any]:
    """Return the recorded settings the asset viewer shows beyond the top-level reuse fields."""
    return {
        "recorded_workflow": asset.workflow,
        "negative_prompt": asset.negative_prompt,
        "scheduler": asset.scheduler,
        "model_family": asset.model_family,
        "image_strength": asset.image_strength,
        "generation": dict(asset.generation),
        "source": dict(asset.source) if asset.source is not None else None,
    }


def _asset_upscale_json(asset: GalleryAsset, web_config: WebUiConfig, resolved_model: str | None) -> dict[str, Any] | None:
    """Return the upscale menu for an image: the creative factors and the SeedVR2 upscale.

    ``factors`` gives each creative factor's size and whether the refinement model allows it. That model is the
    one recorded in the image, or the default image model when that one is missing or not configured, matching
    what ``POST /api/upscale`` will use. ``seedvr2`` is independent of the image's model: ``None`` where SeedVR2
    is unsupported, else its options (never disabled, possibly warned), tier label and download state.
    """
    if asset.width is None or asset.height is None:
        return None
    model = resolved_model or preferred_option(web_config.default_models.image, web_config.image_model_options)
    if model is None:
        options = [UpscaleOption(factor, *upscale_output_size(asset.width, asset.height, factor), allowed=False, reason="No image model is configured.") for factor in UPSCALE_FACTORS]
    else:
        # The inventory's family gives the same capabilities resolve_defaults uses for POST /api/upscale, without detection.
        family = next((entry.family for entry in web_config.image_inventory if entry.name == model), "unknown")
        options = upscale_options(asset.width, asset.height, model_capabilities(web_config.app_config, family), web_config.app_config)
    setup = web_config.seedvr2
    seedvr2_menu = seedvr2_options(asset.width, asset.height, setup.tier, budget_bytes=setup.budget_bytes) if setup is not None else []
    return {"factors": [asdict(option) for option in options], "seedvr2": _seedvr2_upscale_json(seedvr2_menu, setup)}


def _seedvr2_upscale_json(options: list[UpscaleOption], setup: SeedVR2Setup | None) -> dict[str, Any] | None:
    """Return the SeedVR2 part of the upscale menu for *options*, or ``None`` when SeedVR2 is unsupported."""
    if setup is None:
        return None
    model = setup.tier.model
    return {
        "options": [asdict(option) for option in options],
        "model": seedvr2_tier_label(model, setup.weights),
        "downloaded": setup.downloaded,
        "download_size_label": None if setup.downloaded else seedvr2_size_label(seedvr2_download_bytes(model, setup.weights)),
    }


def _video_upscale_json(asset: GalleryAsset, setup: SeedVR2Setup | None) -> dict[str, Any] | None:
    """Return the upscale menu for a video: one SeedVR2 2× option, or ``None`` where it is unsupported or the size is unknown.

    The size is the recorded one, else the displayed size ffprobe reported. The option is never disabled, only warned.
    """
    if setup is None or asset.width is None or asset.height is None or not supports_video_file_upscale():
        return None
    option = seedvr2_video_option(asset.width, asset.height, setup.tier, budget_bytes=setup.budget_bytes)
    return {"factors": [], "seedvr2": _seedvr2_upscale_json([option], setup)}


def _source_with_asset_id(root: Path, source: dict[str, Any] | None) -> dict[str, Any] | None:
    """Add the gallery ID of an upscale source that still exists under *root*, else ``id: None``."""
    if source is None:
        return None
    asset_id = None
    try:
        resolved_root = root.expanduser().resolve()
        candidate = Path(source["path"]).expanduser().resolve()
        if candidate.is_relative_to(resolved_root) and candidate.is_file():
            asset_id = candidate.relative_to(resolved_root).as_posix()
    except OSError, ValueError:
        asset_id = None
    return {**source, "id": asset_id}


# Changing what this reads requires bumping ``gallery_index.GALLERY_INDEX_VERSION``.
def _read_asset_facts(asset_path: Path, kind: str | None) -> FactsRead:
    """Read the raw facts of a media file: the embedded config, and the size (plus prompt text for an image).

    The facts are incomplete when a reader failed, so the index does not store them.
    """
    if kind == "video":
        return _read_video_facts(asset_path)
    complete = True
    config: dict[str, Any] | None = None
    try:
        if kind == "image" and asset_path.suffix.lower() == ".png":
            config = read_png_config(asset_path)
    except Exception:
        complete = False
    image: dict[str, Any] = {}
    if kind == "image":
        try:
            with Image.open(asset_path) as opened:
                image = {"width": opened.width, "height": opened.height, "prompt": image_prompt_text(opened)}
        except FileNotFoundError, OSError, UnidentifiedImageError, ValueError:
            complete = False
    return FactsRead({"config": config, "image": image}, complete)


def _read_video_facts(asset_path: Path) -> FactsRead:
    """Read a video's embedded config and displayed size with one ffprobe call.

    The config is only read from ``.mp4`` files. The size is what a player shows (rotation applied), and is left out
    when the file has no video stream.
    """
    try:
        data = ffprobe_json(asset_path, count_packets=False)
    except Exception:
        return FactsRead({"config": None, "image": {}}, False)
    complete = True
    config: dict[str, Any] | None = None
    try:
        config = mp4_config_from_probe(data) if asset_path.suffix.lower() == ".mp4" else None
    except ValueError:
        complete = False
    try:
        probe = parse_probe(data)
        video = {"width": probe.width, "height": probe.height}
    except ValueError:
        video = {}
    return FactsRead({"config": config, "image": {}, "video": video}, complete)


def _read_scanned_facts(file: ScannedFile) -> FactsRead:
    """Read the facts of a scanned file; the reader the gallery index calls for new or changed files."""
    return _read_asset_facts(file.path, _asset_kind(file.path))


def _parse_generated_filename(asset_path: Path) -> dict[str, Any]:
    match = re.search(
        r"_(?P<width>\d+)x(?P<height>\d+)(?:_\d+f)?(?:_.*?)?_steps(?P<steps>\d+)(?:_cfg(?P<guidance>[-+]?\d+(?:\.\d+)?))?_seed(?P<seed>\d+)",
        asset_path.stem,
    )
    if match is None:
        return {}
    parsed = match.groupdict()
    return {
        "width": optional_int(parsed.get("width")),
        "height": optional_int(parsed.get("height")),
        "steps": optional_int(parsed.get("steps")),
        "guidance": optional_float(parsed.get("guidance")),
        "seed": optional_int(parsed.get("seed")),
    }


def _coerce_text(value: Any) -> str:
    return optional_text(value) or "Unavailable"


def _format_guidance(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:g}"


def _paginate_assets(assets: list[GalleryAsset], *, page: int, page_size: int) -> tuple[list[GalleryAsset], int | None]:
    start = (page - 1) * page_size
    end = start + page_size
    next_page = page + 1 if end < len(assets) else None
    return assets[start:end], next_page


def _asset_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def _is_hidden_from_inventory(root: Path, candidate: Path) -> bool:
    return any(_is_staging_part(part) for part in candidate.relative_to(root).parts)


def _is_staging_part(part: str) -> bool:
    """Return whether a path component names an upload or scratch directory that never holds gallery assets."""
    return part in _STAGING_DIR_NAMES or part.startswith(_STAGING_PREFIXES)


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
