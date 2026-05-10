"""Own gallery inventory scanning and gallery response serialization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import time
from typing import Any
from urllib.parse import quote, unquote, urlencode

from PIL import Image, UnidentifiedImageError

from zvisiongenerator.utils.provenance import read_mp4_config, read_png_config
from zvisiongenerator.web.config import WebUiConfig
from zvisiongenerator.web.workspace_contract import WORKFLOW_DEFINITIONS, canonicalize_workflow, default_workflow_for_mode, workflow_mode


_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".mkv"})
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
    requested_workflow = canonicalize_workflow(asset.workflow, fallback=default_workflow)
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
    default_model = _preferred_option(
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
    )


def _read_asset_metadata(asset_path: Path, kind: str) -> dict[str, Any]:
    # Embedded config (PNG tEXt chunk / MP4 container tag) is the only reusable generation settings source.
    primary = _read_embedded_config(asset_path, kind) or {}
    has_reusable_config = bool(primary)
    filename_metadata = _parse_generated_filename(asset_path)
    image_metadata = _read_image_metadata(asset_path) if kind == "image" else {}

    prompt = _coerce_text(_metadata_value(primary, "prompt") or image_metadata.get("prompt") or asset_path.stem.replace("_", " "))

    model_label = _coerce_text(_metadata_value(primary, "model"))
    width = _coerce_int(_metadata_value(primary, "width") or image_metadata.get("width") or filename_metadata.get("width"))
    height = _coerce_int(_metadata_value(primary, "height") or image_metadata.get("height") or filename_metadata.get("height"))
    seed = _coerce_int(_metadata_value(primary, "seed"))
    steps = _coerce_int(_metadata_value(primary, "steps"))
    guidance = _coerce_float(_metadata_value(primary, "guidance"))
    workflow = _coerce_optional_text(_metadata_value(primary, "workflow"))
    ratio = _coerce_optional_text(_metadata_value(primary, "ratio"))
    size = _coerce_optional_text(_metadata_value(primary, "size"))
    frame_count = _coerce_int(_metadata_value(primary, "frame_count"))
    reference_image_path = _coerce_optional_text(_metadata_value(primary, "image_path"))
    lora = _coerce_lora_string(_metadata_value(primary, "lora"))

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
        "has_reusable_config": has_reusable_config,
    }


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
            exif = image.getexif()
            return {
                "width": image.width,
                "height": image.height,
                "prompt": image.info.get("Description") or exif.get(0x010E),
            }
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
        "width": _coerce_int(parsed.get("width")),
        "height": _coerce_int(parsed.get("height")),
        "steps": _coerce_int(parsed.get("steps")),
        "guidance": _coerce_float(parsed.get("guidance")),
        "seed": _coerce_int(parsed.get("seed")),
    }


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    if not metadata:
        return None
    for mapping in _walk_mappings(metadata):
        for key in keys:
            if key in mapping and mapping[key] not in (None, ""):
                return mapping[key]
    return None


def _walk_mappings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    mappings = [payload]
    for value in payload.values():
        if isinstance(value, dict):
            mappings.extend(_walk_mappings(value))
    return mappings


def _coerce_text(value: Any) -> str:
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
    if value in (None, ""):
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _coerce_float(value: Any) -> float | None:
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


def _preferred_option(preferred: str | None, options: tuple[str, ...]) -> str | None:
    if preferred in options:
        return preferred
    return options[0] if options else None
