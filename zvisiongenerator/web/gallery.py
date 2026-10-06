"""Own gallery inventory scanning and gallery response serialization."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import time
from typing import Any
from urllib.parse import quote, unquote, urlencode

from PIL import Image, UnidentifiedImageError

from zvisiongenerator.backends import get_backend_name
from zvisiongenerator.utils.provenance import image_prompt_text, optional_float, optional_int, optional_text, read_mp4_config, read_png_config, recorded_settings
from zvisiongenerator.utils.config import model_capabilities
from zvisiongenerator.utils.upscale import UPSCALE_FACTORS, UpscaleOption, upscale_options, upscale_output_size
from zvisiongenerator.web.config import WebUiConfig, preferred_option
from zvisiongenerator.web.workspace_contract import WORKFLOW_DEFINITIONS, canonicalize_workflow, default_workflow_for_mode, workflow_mode


_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".mkv"})
GALLERY_MEDIA_EXTENSIONS = _IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS
_STAGING_DIR_NAMES = frozenset({".web_uploads"})


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
    scheduler: str | None = None
    model_family: str | None = None
    image_strength: float | None = None
    generation: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] | None = None


def list_gallery_assets(output_dir: str) -> list[GalleryAsset]:
    """Scan an output directory for renderable image and video assets."""
    root = Path(output_dir)
    if not root.exists():
        return []

    assets: list[GalleryAsset] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        if _is_hidden_from_inventory(root, candidate):
            continue
        asset = _build_gallery_asset(root, candidate)
        if asset is not None:
            assets.append(asset)
    assets.sort(key=lambda item: item.modified_at, reverse=True)
    return assets


def gallery_asset_for_output_path(output_dir: str, output_path: str) -> GalleryAsset | None:
    """Build gallery metadata for one generated output under the configured output root."""
    root = Path(output_dir).expanduser().resolve()
    candidate = Path(output_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(root):
        return None
    if _is_hidden_from_inventory(root, resolved):
        return None
    return _build_gallery_asset(root, resolved)


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

    reuse_params: dict[str, str] = {"workflow": resolved_workflow}
    if asset.has_reusable_config:
        reuse_params["prompt"] = asset.prompt
        if resolved_model:
            reuse_params["model"] = resolved_model
        if asset.lora:
            reuse_params["lora"] = asset.lora
        if asset.steps is not None:
            reuse_params["steps"] = str(asset.steps)
        if asset.guidance is not None and workflow_mode(resolved_workflow) == "image":
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
        if asset.frame_count is not None and workflow_mode(resolved_workflow) == "video":
            reuse_params["frames"] = str(asset.frame_count)
        if WORKFLOW_DEFINITIONS[resolved_workflow]["requires_reference_image"] and asset.reference_image_path is not None:
            reuse_params["image_path"] = asset.reference_image_path
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
        "prompt": asset.prompt,
        "model": asset.model_label,
        "width": asset.width,
        "height": asset.height,
        "ratio": asset.ratio,
        "size": asset.size,
        "frame_count": asset.frame_count,
        "image_path": asset.reference_image_path,
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
        "upscale": _asset_upscale_json(asset, web_config, resolved_model) if asset.kind == "image" else None,
    }


def delete_gallery_assets(output_dir: str, selected_paths: list[str]) -> None:
    """Delete selected gallery media assets."""
    root = Path(output_dir).resolve()
    for asset_id in dict.fromkeys(path for path in selected_paths if path.strip()):
        candidate = resolve_output_asset_path(root, asset_id)
        if candidate is None or not candidate.is_file():
            continue
        candidate.unlink(missing_ok=True)


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
    if not parts or any(part in {"", ".", ".."} or part in _STAGING_DIR_NAMES for part in parts):
        return None
    return path.as_posix()


def _build_gallery_asset(root: Path, candidate: Path) -> GalleryAsset | None:
    kind = _asset_kind(candidate)
    if kind is None:
        return None
    asset_id = candidate.relative_to(root).as_posix()
    metadata = _read_asset_metadata(candidate, kind)
    return GalleryAsset(
        id=asset_id,
        name=candidate.name,
        kind=kind,
        extension=candidate.suffix.lower().lstrip("."),
        filesystem_path=str(candidate),
        modified_at=candidate.stat().st_mtime,
        modified_label=_format_age(candidate.stat().st_mtime),
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
        lora=metadata["lora"],
        has_reusable_config=metadata["has_reusable_config"],
        negative_prompt=metadata["negative_prompt"],
        scheduler=metadata["scheduler"],
        model_family=metadata["model_family"],
        image_strength=metadata["image_strength"],
        generation=metadata["generation"],
        source=_source_with_asset_id(root, metadata["source"]),
    )


def _read_asset_metadata(asset_path: Path, kind: str) -> dict[str, Any]:
    # Embedded config (PNG tEXt chunk / MP4 container tag) is the only reusable generation settings source.
    config = _read_embedded_config(asset_path, kind) or {}
    recorded = recorded_settings(config)
    filename_metadata = _parse_generated_filename(asset_path)
    image_metadata = _read_image_metadata(asset_path) if kind == "image" else {}

    prompt = _coerce_text(recorded.prompt or image_metadata.get("prompt") or asset_path.stem.replace("_", " "))
    width = recorded.width or optional_int(image_metadata.get("width") or filename_metadata.get("width"))
    height = recorded.height or optional_int(image_metadata.get("height") or filename_metadata.get("height"))

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
        "lora": recorded.lora,
        "has_reusable_config": bool(config),
        "negative_prompt": recorded.negative_prompt,
        "scheduler": recorded.scheduler,
        "model_family": recorded.model_family,
        "image_strength": recorded.image_strength,
        "generation": recorded.generation,
        "source": recorded.source,
    }


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
    """Return the upscale menu for an image: each factor's size and whether the refinement model allows it.

    The refinement model is the one recorded in the image, or the default image model when that one is
    missing or not configured, matching what ``POST /api/upscale`` will use.
    """
    if asset.width is None or asset.height is None:
        return None
    model = resolved_model or preferred_option(web_config.default_models.image, web_config.image_model_options)
    if model is None:
        options = [UpscaleOption(factor, *upscale_output_size(asset.width, asset.height, factor), allowed=False, reason="No image model is configured.") for factor in UPSCALE_FACTORS]
    else:
        # The inventory's family gives the same capabilities resolve_defaults uses for POST /api/upscale, without detection.
        family = next((entry.family for entry in web_config.image_inventory if entry.name == model), "unknown")
        options = upscale_options(asset.width, asset.height, model_capabilities(web_config.app_config, family, get_backend_name()), web_config.app_config)
    return {"factors": [asdict(option) for option in options]}


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


def _read_embedded_config(asset_path: Path, kind: str) -> dict[str, Any] | None:
    """Read the embedded zvisiongenerator.config payload from a PNG or MP4 asset.

    Returns the parsed dict when present, or None for all other formats and all errors.
    """
    try:
        if kind == "image" and asset_path.suffix.lower() == ".png":
            return read_png_config(asset_path)
        if kind == "video" and asset_path.suffix.lower() == ".mp4":
            return read_mp4_config(asset_path)
    except Exception:
        return None
    return None


def _read_image_metadata(asset_path: Path) -> dict[str, Any]:
    try:
        with Image.open(asset_path) as image:
            return {"width": image.width, "height": image.height, "prompt": image_prompt_text(image)}
    except FileNotFoundError, OSError, UnidentifiedImageError, ValueError:
        return {}


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
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    return None


def _is_hidden_from_inventory(root: Path, candidate: Path) -> bool:
    return any(part in _STAGING_DIR_NAMES for part in candidate.relative_to(root).parts)


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
