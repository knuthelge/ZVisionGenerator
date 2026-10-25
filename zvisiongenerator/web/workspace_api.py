"""Build shared SPA payloads for workspace and models routes."""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.backends import accelerator_memory_budget_or_none as memory_budget, get_backend_name, supports_stored_quants, supports_video_audio, video_capabilities
from zvisiongenerator.converters.list_assets import list_loras
from zvisiongenerator.downloads.presence import target_present_offline
from zvisiongenerator.backends.prompt_enhancer_session import is_model_downloaded
from zvisiongenerator.utils.config import (
    enhancer_tier_models,
    model_display_name,
    resolve_defaults,
    resolve_enhancer_model,
    resolve_video_defaults,
    smart_enhancer_model,
    smart_unavailable,
    video_companion_repos,
    video_family,
)
from zvisiongenerator.utils.alignment import align_frames
from zvisiongenerator.utils.ideogram4_instant import is_instant_dir
from zvisiongenerator.utils.image_model_detect import ImageModelInfo, detect_image_model
from zvisiongenerator.utils.h3_converted import CONVERTED_H3_STORAGES, read_converted_h3
from zvisiongenerator.utils.ltx_converted import read_converted_ltx
from zvisiongenerator.utils.hf_cache import CachedRepo, cached_model_repos, repo_disk_bytes, stranded_shared_files
from zvisiongenerator.utils.model_files import (
    directory_bytes,
    disk_size_label,
    find_local_model_dir,
    huggingface_cache_dir,
    huggingface_cache_repo_dir,
    ltx_mlx_text_encoder_repo,
)
from zvisiongenerator.utils.paths import parse_huggingface_repo_reference, resolve_model_path
from zvisiongenerator.utils.prompt_enhance import matrix_contract, resolve_enhance_ceiling
from zvisiongenerator.utils.stored_quant import stored_quant_dirs_for
from zvisiongenerator.utils.seedvr2 import SEEDVR2_REPO, SeedVR2Setup, seedvr2_download_bytes, seedvr2_row_name, seedvr2_size_label
from zvisiongenerator.utils.video_extend import extend_step
from zvisiongenerator.utils.video_model_detect import detect_video_model
from zvisiongenerator.web.config import WebUiConfig, preferred_option
from zvisiongenerator.web.model_download_plan import KIND_MODEL, enhancer_download_needed, row_download_needed
from zvisiongenerator.web.first_run import first_run_payload
from zvisiongenerator.web.lora_index import lookup as lookup_lora_families
from zvisiongenerator.web.defaults import resolve_image_ratio_size_defaults, resolve_video_ratio_size_defaults
from zvisiongenerator.web.model_delete import cache_repos_linked_from, delete_scope, enhancer_delete_target, installed_models_linking_to, model_delete_target, seedvr2_delete_target, used_repo_ids
from zvisiongenerator.web.model_inventory import ImageInventoryEntry, VideoInventoryEntry, declared_image_family, stored_quant_of
from zvisiongenerator.web.model_status import describe_model_status
from zvisiongenerator.web.workspace_contract import END_FRAME_CONTROLS, hidden_video_controls

if TYPE_CHECKING:
    from zvisiongenerator.downloads.target import DownloadTarget

_UNKNOWN_STATUS: dict[str, Any] = {"downloaded": None, "memory_fit": None}
_IMAGE_BOOTSTRAP_STRENGTH = 0.5
_IMAGE_BOOTSTRAP_POSTPROCESS = {
    # On with no amount: the backend uses the config's sharpening amounts unless the user sets one.
    "sharpen": True,
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


def build_workspace_bootstrap_view(web_config: WebUiConfig) -> dict[str, Any]:
    """Resolve per-model bootstrap defaults using shared backend config authority."""
    image_default_model = preferred_option(web_config.default_models.image, web_config.image_model_options)
    video_default_model = preferred_option(web_config.default_models.video, web_config.video_model_options)
    image_defaults = {model_name: _build_image_bootstrap_defaults(model_name, web_config) for model_name in web_config.image_model_options}
    video_defaults = {model_name: _build_video_bootstrap_defaults(model_name, web_config) for model_name in web_config.video_model_options}
    return {
        "image_default_model": image_default_model,
        "video_default_model": video_default_model,
        "image_model_defaults": image_defaults,
        "video_model_defaults": video_defaults,
    }


def build_workspace_response(
    web_config: WebUiConfig,
    history_assets: list[dict[str, Any]],
    *,
    active_job: dict[str, Any] | None,
    queued_jobs: list[dict[str, Any]] | None = None,
    prompt_sources: list[str],
    default_prompt_source: str,
    prompt_file_contract: dict[str, Any],
    workflow_contract: dict[str, Any],
    build_bootstrap_view: Any = build_workspace_bootstrap_view,
) -> dict[str, Any]:
    """Build the workspace bootstrap payload consumed by the SPA."""
    form_view = build_bootstrap_view(web_config)
    image_default_model = form_view["image_default_model"]
    video_default_model = form_view["video_default_model"]
    image_model_defaults_map = form_view["image_model_defaults"]
    video_model_defaults_map = form_view["video_model_defaults"]

    status = _status_resolver(web_config, image_model_defaults_map, instant_sources_count=True)
    image_entries = {entry.name: entry for entry in web_config.image_inventory}
    video_entries = {entry.name: entry for entry in web_config.video_inventory}
    model_labels = web_config.app_config.get("model_labels") or {}
    image_models = [
        {
            "id": name,
            "label": model_display_name(web_config.app_config, name),
            "type": "image",
            "download_size_label": _size_label(model_labels.get(name) or {}),
            **status(image_entries.get(name), "image"),
        }
        for name in web_config.image_model_options
    ]
    video_models = [{"id": name, "label": model_display_name(web_config.app_config, name), "type": "video", **status(video_entries.get(name), "video")} for name in web_config.video_model_options]
    loras = _lora_rows(web_config)
    image_defaults = image_model_defaults_map.get(image_default_model) or _build_image_bootstrap_defaults(image_default_model or "", web_config)
    video_defaults = video_model_defaults_map.get(video_default_model) or _build_video_bootstrap_defaults(video_default_model or "", web_config)

    return {
        "image_models": image_models,
        "video_models": video_models,
        "loras": loras,
        "history_assets": history_assets,
        "active_job": active_job,
        "queued_jobs": queued_jobs or [],
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
        "image_size_dimensions": {ratio: {size: list(wh) for size, wh in sizes.items()} for ratio, sizes in web_config.image_size_dimensions.items()},
        "video_size_dimensions": {ratio: {size: list(wh) for size, wh in sizes.items()} for ratio, sizes in web_config.video_size_dimensions.items()},
        "scheduler_options": list(web_config.scheduler_options),
        "prompt_sources": prompt_sources,
        "default_prompt_source": default_prompt_source,
        "prompt_file": prompt_file_contract,
        "workflow_contract": workflow_contract,
        "prompt_enhancer": build_prompt_enhancer_contract(web_config),
        "first_run": first_run_payload(web_config.app_config, memory_budget(), web_config.image_model_options),
        "config": {
            "gallery_page_size": web_config.gallery_page_size,
            "startup_view": web_config.startup_view,
            "output_dir": web_config.output_dir,
            "default_models": {
                "image": web_config.default_models.image,
                "video": web_config.default_models.video,
            },
        },
    }


def build_prompt_enhancer_contract(
    web_config: WebUiConfig,
    *,
    downloaded: Callable[[str, str | None], bool] = is_model_downloaded,
    budget_bytes: Callable[[], int | None] | None = None,
) -> dict[str, Any]:
    """Describe the prompt enhancer for the SPA: option matrix, effective (Fast or custom) model, download state, and Smart.

    ``smart`` is ``None`` when a custom model is configured (no Fast/Smart choice); otherwise it holds the Smart model's
    first-download size label (``None`` once downloaded) and ``unavailable``, the reason it cannot run here, or ``None``.

    Args:
        budget_bytes: Returns the machine's accelerator memory, or ``None`` when unknown.
    """
    app_config = web_config.app_config
    section = app_config.get("prompt_enhancer") or {}
    contract: dict[str, Any] = {
        "matrix": matrix_contract(),
        "model": None,
        "revision": None,
        "downloaded": False,
        "download_size_label": _size_label(section),
        "default_max_words": resolve_enhance_ceiling(app_config, family=None, mode="image"),
        "video_audio": supports_video_audio(),
        "smart": None,
        "error": None,
    }
    try:
        repo, revision = resolve_enhancer_model(app_config, platform_key=sys.platform)
    except ValueError as exc:
        contract["error"] = str(exc)
        return contract
    contract["model"], contract["revision"] = repo, revision
    contract["downloaded"] = downloaded(repo, revision)
    # The size label describes the built-in default; a custom model's size is unknown.
    if contract["downloaded"] or section.get("user_model"):
        contract["download_size_label"] = None
    if not section.get("user_model"):
        smart = smart_enhancer_model(app_config, platform_key=sys.platform)
        budget = (budget_bytes or _gpu_budget_bytes)()
        contract["smart"] = {
            "download_size_label": _size_label(section.get("smart") or {}) if smart is not None and not downloaded(*smart) else None,
            "unavailable": smart_unavailable(app_config, platform_key=sys.platform, budget_bytes=budget),
        }
    return contract


def _size_label(section: dict[str, Any]) -> str | None:
    """Return a tier section's download size label for this platform, if any."""
    sizes = section.get("download_size_label") or {}
    label = sizes.get(sys.platform) if isinstance(sizes, dict) else None
    return label if isinstance(label, str) else None


def _gpu_budget_bytes() -> int | None:
    """Return this machine's accelerator memory in bytes, or ``None`` when unknown."""
    budget = memory_budget()
    return budget.gpu_bytes if budget is not None else None


def build_models_response(
    web_config: WebUiConfig,
    *,
    token_source: str | None,
    image_defaults_for: Callable[[str, WebUiConfig], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the models inventory payload from the inventory discovered with the Web UI config.

    Args:
        web_config: The Web UI config, including its discovered model inventory.
        token_source: Where the Hugging Face token comes from (an environment variable name or ``hf auth login``), if any.
        image_defaults_for: Resolves one image model's bootstrap defaults; only its ``supports_quantize`` is used.
    """
    data_dir = Path(web_config.data_dir)
    image_defaults_for = image_defaults_for or _build_image_bootstrap_defaults
    # Quantize levels only shape memory estimates, so skip resolving them where there is no budget (no GPU found).
    image_defaults = {entry.name: image_defaults_for(entry.name, web_config) for entry in web_config.image_inventory} if memory_budget() else {}
    find_local_dir = functools.cache(find_local_model_dir)
    status = _status_resolver(web_config, image_defaults, find_local_dir)
    inventory = (*web_config.image_inventory, *web_config.video_inventory)
    delete_info = _delete_info_resolver(data_dir / "models", find_local_dir, inventory=inventory, app_config=web_config.app_config)
    disk_size = _disk_size_resolver(data_dir / "models", listed={entry.name for entry in inventory}, app_config=web_config.app_config)
    download_needed = _download_needed_resolver(web_config)
    loras = [{"name": lora.name, "file_size_mb": lora.file_size_mb, "size_label": f"{lora.file_size_mb} MB"} for lora in list_loras(data_dir)]
    image_rows = [
        {
            "name": entry.name,
            "label": model_display_name(web_config.app_config, entry.name),
            "family": entry.family,
            "size_label": entry.size or "Unknown",
            "source": entry.source,
            "stored_quant": _stored_quant_payload(entry),
            **status(entry, "image"),
            "delete": delete_info(entry),
            "disk_size_label": disk_size(entry),
        }
        for entry in web_config.image_inventory
    ] + _seedvr2_model_rows(web_config.seedvr2, Path(web_config.models_dir))
    video_rows = [
        {
            "name": entry.name,
            "label": model_display_name(web_config.app_config, entry.name),
            "family": entry.family,
            "supports_i2v": entry.supports_i2v,
            "source": entry.source,
            "converted_bits": _converted_bits(entry),
            **status(entry, "video"),
            "delete": delete_info(entry),
            "disk_size_label": disk_size(entry),
        }
        for entry in web_config.video_inventory
    ]
    for row in (*image_rows, *video_rows):
        row["download_needed"] = download_needed(KIND_MODEL, row["name"], row["downloaded"])
    return {
        "models_dir": web_config.models_dir,
        "loras_dir": web_config.loras_dir,
        "image_models": image_rows,
        "video_models": video_rows,
        "enhancer_models": _enhancer_model_rows(web_config.app_config),
        "other_downloads": _other_downloads_payload(web_config),
        "loras": loras,
        "stored_quants_supported": supports_stored_quants(),
        "huggingface_configured": token_source is not None,
        "huggingface_token_source": token_source,
    }


def _download_needed_resolver(web_config: WebUiConfig) -> Callable[[str, str, bool | None], bool]:
    """Return a resolver for each model row's ``download_needed``: whether anything is still to be downloaded.

    The answer comes from what is on disk only (no network, no backend import). One row that cannot be planned
    warns and reads as not needing a download, like an unreadable status.
    """

    def _needed(kind: str, name: str, downloaded: bool | None) -> bool:
        try:
            return row_download_needed(kind, name, web_config, downloaded=downloaded)
        except Exception as exc:  # noqa: BLE001 - one unreadable row must not break the whole listing
            warnings.warn(f"Could not determine whether '{name}' needs a download: {exc}", stacklevel=2)
            return False

    return _needed


def _seedvr2_model_rows(setup: SeedVR2Setup | None, models_dir: Path) -> list[dict[str, Any]]:
    """Return the Models-page row for the SeedVR2 tier this machine uses, or no rows where SeedVR2 is off.

    The row has the shape of an image model's, plus its weights folder's disk use and the size of the first-use
    download, which the page shows in the not-downloaded tooltip and the delete confirmation.
    """
    if setup is None:
        return []
    model = setup.tier.model
    target = seedvr2_delete_target(model, models_dir, setup.weights)
    return [
        {
            "name": seedvr2_row_name(model, setup.weights),
            "label": seedvr2_row_name(model, setup.weights),
            "family": "seedvr2",
            "size_label": model.removeprefix("seedvr2-"),
            "source": "seedvr2",
            "stored_quant": None,
            "downloaded": setup.downloaded,
            "memory_fit": None,
            "delete": {"kind": target.kind, "repo_id": target.repo_id, "linked_by": [], "stored_quants": []} if target is not None else None,
            "disk_size_label": seedvr2_size_label(directory_bytes(target.path)) if target is not None else None,
            "download_size_label": seedvr2_size_label(seedvr2_download_bytes(model, setup.weights)),
        }
    ]


def _enhancer_model_rows(
    app_config: dict[str, Any],
    *,
    downloaded: Callable[[str, str | None], bool] = is_model_downloaded,
    budget_bytes: Callable[[], int | None] | None = None,
    present: Callable[[DownloadTarget], bool] = target_present_offline,
) -> list[dict[str, Any]]:
    """Return one Models-page row per configured enhancer tier, with its disk use, whether it can be deleted and whether it needs a download.

    Args:
        budget_bytes: Returns the machine's accelerator memory, for Smart's ``unavailable`` reason.
        present: Tells whether a download target is fully on disk without using the network.
    """
    section = app_config.get("prompt_enhancer") or {}
    size_sections = {"fast": section, "smart": section.get("smart") or {}}
    rows = []
    for tier, (repo, revision) in enhancer_tier_models(app_config, platform_key=sys.platform).items():
        target = enhancer_delete_target(repo)
        disk_bytes = repo_disk_bytes(target.path) if target is not None else 0
        unavailable = smart_unavailable(app_config, platform_key=sys.platform, budget_bytes=(budget_bytes or _gpu_budget_bytes)()) if tier == "smart" else None
        rows.append(
            {
                "tier": tier,
                "repo": repo,
                "revision": revision,
                "downloaded": downloaded(repo, revision),
                "disk_size_label": disk_size_label(disk_bytes) if disk_bytes else None,
                "download_size_label": _size_label(size_sections[tier]) if tier in size_sections else None,
                "unavailable": unavailable,
                "deletable": target is not None,
                "download_needed": _enhancer_download_needed(tier, repo, revision, unavailable, present),
            }
        )
    return rows


def _enhancer_download_needed(tier: str, repo: str, revision: str | None, unavailable: str | None, present: Callable[[DownloadTarget], bool]) -> bool:
    """Return an enhancer row's ``download_needed``; a tier that cannot be planned warns and needs none."""
    try:
        return enhancer_download_needed(repo, revision, unavailable=unavailable, present=present)
    except Exception as exc:  # noqa: BLE001 - one unreadable row must not break the whole listing
        warnings.warn(f"Could not determine whether the '{tier}' enhancer model needs a download: {exc}", stacklevel=2)
        return False


def find_other_downloads(web_config: WebUiConfig, *, cache_dir: Path | None = None) -> tuple[CachedRepo, ...]:
    """Return the model repos in the HuggingFace cache that nothing on the Models page uses.

    Used means an alias or enhancer tier loads it (with LTX companions, the MLX LTX text encoder and, where the
    SeedVR2 engine downloads through the cache to make its stored copy, the SeedVR2 repo), or an installed model
    links into it.
    """
    cache_dir = cache_dir or huggingface_cache_dir()
    # The MLX LTX text encoder counts as used everywhere; only the macOS backend downloads it.
    extra = [ltx_mlx_text_encoder_repo(), *([SEEDVR2_REPO] if web_config.seedvr2 is not None and web_config.seedvr2.weights.stores_copy else [])]
    used = used_repo_ids((*web_config.image_inventory, *web_config.video_inventory), web_config.app_config, platform_key=sys.platform, extra=extra)
    linked = cache_repos_linked_from(Path(web_config.models_dir), cache_dir)
    return tuple(repo for repo in cached_model_repos(cache_dir) if repo.repo_id.casefold() not in used and repo.path.name not in linked)


def _other_downloads_payload(web_config: WebUiConfig) -> dict[str, Any]:
    """Describe the unused HuggingFace downloads and the shared files nothing links to, for the Models page."""
    cache_dir = huggingface_cache_dir()
    stranded = stranded_shared_files(cache_dir)
    return {
        "cache_dir": str(cache_dir),
        "repos": [{"repo_id": repo.repo_id, "size_label": disk_size_label(repo.freed_bytes)} for repo in find_other_downloads(web_config, cache_dir=cache_dir)],
        "stranded": {"count": len(stranded), "size_label": disk_size_label(sum(stranded.values()))} if stranded else None,
    }


def _stored_quant_payload(entry: ImageInventoryEntry) -> dict[str, Any] | None:
    """Describe an installed stored quant (its base model and bits), or ``None`` for other models."""
    stored = stored_quant_of(entry)
    return None if stored is None else {"base_model": stored[0], "bits": stored[1]}


def _converted_bits(entry: VideoInventoryEntry) -> int | None:
    """Return the quantize level a converted LTX or H3 checkpoint was stored at, or ``None`` for other video models."""
    if entry.source != "installed":
        return None
    folder = Path(entry.resolved_path)
    manifest = read_converted_ltx(folder)
    if manifest is not None:
        return manifest["bits"]
    h3 = read_converted_h3(folder)
    return None if h3 is None else CONVERTED_H3_STORAGES[h3["storage"]]


def _status_resolver(
    web_config: WebUiConfig,
    image_model_defaults: dict[str, dict[str, Any]],
    find_local_dir: Callable[[str], Path | None] | None = None,
    *,
    instant_sources_count: bool = False,
) -> Callable[[ImageInventoryEntry | VideoInventoryEntry | None, str], dict[str, Any]]:
    """Return a per-request resolver for each model's ``downloaded``/``memory_fit`` fields.

    Image quantize levels come from the same bootstrap defaults that drive the workspace quantize picker, and
    download lookups are memoised for the request (LTX MLX models share one Gemma text-encoder lookup).

    With *instant_sources_count*, an Ideogram 4 Instant folder that is not built yet reads as downloaded once all
    its sources are on disk (the first job builds the folder from them), with ``memory_fit`` still ``None``.
    The Models page leaves it off so a row shows it still has to be built.
    """
    budget = memory_budget()
    find_local_dir = find_local_dir or functools.cache(find_local_model_dir)

    def _status(entry: ImageInventoryEntry | VideoInventoryEntry | None, kind: str) -> dict[str, Any]:
        if entry is None:
            return dict(_UNKNOWN_STATUS)
        quantize_options = _image_quantize_levels(web_config, image_model_defaults.get(entry.name, {})) if kind == "image" else ()
        try:
            described = describe_model_status(
                entry.resolved_path,
                kind=kind,
                quantize_options=quantize_options,
                budget=budget,
                find_local_dir=find_local_dir,
                model_name=entry.name,
                app_config=web_config.app_config,
                family=entry.family if kind == "video" else None,
            )
            if instant_sources_count and not described["downloaded"] and is_instant_dir(entry.resolved_path):
                described["downloaded"] = not row_download_needed(KIND_MODEL, entry.name, web_config, downloaded=False)
            return described
        except Exception as exc:  # noqa: BLE001 - one unreadable model must not break the whole listing
            warnings.warn(f"Could not determine status for model '{entry.name}': {exc}", stacklevel=2)
            return dict(_UNKNOWN_STATUS)

    return _status


def _delete_info_resolver(
    models_dir: Path,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
    *,
    inventory: tuple[ImageInventoryEntry | VideoInventoryEntry, ...] = (),
    app_config: dict[str, Any] | None = None,
) -> Callable[[ImageInventoryEntry | VideoInventoryEntry], dict[str, Any] | None]:
    """Return a resolver for each model's ``delete`` field: what deleting it removes, or ``None`` when it cannot be deleted.

    HuggingFace downloads list the installed models linking into them (``linked_by``), since deleting the download
    breaks those models.
    """

    def _delete_info(entry: ImageInventoryEntry | VideoInventoryEntry) -> dict[str, Any] | None:
        target = model_delete_target(entry, models_dir, find_local_dir=find_local_dir, **delete_scope(entry, inventory, app_config or {}))
        if target is None or not (target.path.exists() or target.path.is_symlink()):
            return None
        linked_by = list(installed_models_linking_to(target.path, models_dir)) if target.kind == "huggingface" else []
        stored_quants = [path.name for path in ((target.path, *target.stored_quants) if target.kind == "copy" else target.stored_quants)]
        info = {"kind": target.kind, "repo_id": target.repo_id, "linked_by": linked_by, "stored_quants": stored_quants}
        return {**info, "shared_with": list(target.shared_with)} if target.kind == "copy" else info

    return _delete_info


def _disk_size_resolver(models_dir: Path, *, listed: set[str], app_config: dict[str, Any]) -> Callable[[ImageInventoryEntry | VideoInventoryEntry], str | None]:
    """Return a resolver for each model's ``disk_size_label``: the disk its files use, or ``None`` when nothing is on disk.

    An installed model counts its folder (not the HuggingFace files it links to), an alias its HuggingFace download
    with any companion repos. Stored quants count with their model unless they are rows of their own (in *listed*).
    """

    def _disk_size(entry: ImageInventoryEntry | VideoInventoryEntry) -> str | None:
        stored_quants = [path for path in stored_quant_dirs_for(models_dir, entry.name) if path.name not in listed]
        local = [*stored_quants]
        hub: list[Path] = []
        if entry.source == "installed":
            local.append(models_dir / entry.name)
        elif (repo := parse_huggingface_repo_reference(entry.resolved_path)) is not None:
            hub = [huggingface_cache_repo_dir(repo_id) for repo_id in (repo.repo_id, *video_companion_repos(entry.family, app_config, entry.resolved_path))]
        total = sum(directory_bytes(path) for path in local) + sum(repo_disk_bytes(path) for path in hub)
        return disk_size_label(total) if total else None

    return _disk_size


def _image_quantize_levels(web_config: WebUiConfig, capabilities: dict[str, Any]) -> tuple[int, ...]:
    """Return the quantize levels offered for an image model: one rule for the picker and the memory estimates."""
    return tuple(web_config.quantize_options) if capabilities.get("supports_quantize", True) else ()


def _resolve_image_bootstrap_dimensions(app_config: dict[str, Any], ratio: str, size: str) -> dict[str, int]:
    dims = app_config.get("sizes", {}).get(ratio, {}).get(size, {})
    return {
        "width": dims.get("width", 1024),
        "height": dims.get("height", 1024),
    }


def _dimension_is_supported(value: int, *, minimum: int, maximum: int | None, step: int) -> bool:
    return value >= minimum and (maximum is None or value <= maximum) and value % step == 0


def _is_supported_image_preset(app_config: dict[str, Any], ratio: str, size: str, defaults: dict[str, Any]) -> bool:
    dims = _resolve_image_bootstrap_dimensions(app_config, ratio, size)
    minimum = int(defaults.get("dimension_min", 16))
    maximum = defaults.get("dimension_max", None)
    step = int(defaults.get("dimension_step", 16))
    return _dimension_is_supported(dims["width"], minimum=minimum, maximum=maximum, step=step) and _dimension_is_supported(dims["height"], minimum=minimum, maximum=maximum, step=step)


def _pick_supported_image_bootstrap_preset(web_config: WebUiConfig, defaults: dict[str, Any], preferred_ratio: str, preferred_size: str) -> tuple[str, str]:
    app_config = web_config.app_config
    ratios = list(getattr(web_config, "image_ratios", ()))
    if preferred_ratio in ratios:
        ratio_candidates = [preferred_ratio, *[ratio for ratio in ratios if ratio != preferred_ratio]]
    else:
        ratio_candidates = ratios

    for ratio in ratio_candidates:
        size_options = list(getattr(web_config, "image_size_options", {}).get(ratio, ()))
        if not size_options:
            continue
        if ratio == preferred_ratio and preferred_size in size_options:
            preferred_index = size_options.index(preferred_size)
            candidate_indices = [preferred_index]
            for offset in range(1, len(size_options)):
                lower = preferred_index - offset
                upper = preferred_index + offset
                if lower >= 0:
                    candidate_indices.append(lower)
                if upper < len(size_options):
                    candidate_indices.append(upper)
            ordered_sizes = [size_options[index] for index in candidate_indices]
        else:
            ordered_sizes = size_options

        for size in ordered_sizes:
            if _is_supported_image_preset(app_config, ratio, size, defaults):
                return ratio, size

    return preferred_ratio, preferred_size


def _known_family(family: object) -> str | None:
    """Return a detected model family, or None when detection could not name one."""
    return family if isinstance(family, str) and family and family != "unknown" else None


def _resolve_video_bootstrap_family(app_config: dict[str, Any], family: str | None) -> str:
    if family and family != "unknown":
        return family
    video_presets = app_config.get("video_model_presets", {})
    if "ltx" in video_presets:
        return "ltx"
    return next(iter(video_presets), "ltx")


def _default_video_max_steps(app_config: dict[str, Any], family: str) -> int | None:
    value = app_config.get("video_model_presets", {}).get(family, {}).get("default_steps")
    return value if isinstance(value, int) else None


def _image_bootstrap_postprocess() -> dict[str, Any]:
    return dict(_IMAGE_BOOTSTRAP_POSTPROCESS)


def _image_bootstrap_upscale() -> dict[str, Any]:
    return dict(_IMAGE_BOOTSTRAP_UPSCALE)


def _video_bootstrap_upscale() -> dict[str, Any]:
    return dict(_VIDEO_BOOTSTRAP_UPSCALE)


def _lora_rows(web_config: WebUiConfig) -> list[dict[str, Any]]:
    """Return one ``{name, path, family, size_variant}`` row per LoRA."""
    families = lookup_lora_families(web_config.loras_dir, web_config.lora_options)
    return [
        {
            "name": name,
            "path": str(Path(web_config.loras_dir) / f"{name}.safetensors"),
            "family": families[name].family,
            "size_variant": families[name].size_variant,
        }
        for name in web_config.lora_options
    ]


def _build_image_bootstrap_defaults(model_name: str, web_config: WebUiConfig) -> dict[str, Any]:
    app_config = web_config.app_config
    preferred_ratio, preferred_size = resolve_image_ratio_size_defaults(web_config)
    family: str | None = None
    size_variant: str | None = None
    try:
        resolved_model = resolve_model_path(model_name, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
        declared = declared_image_family(app_config, model_name)
        if declared is not None:
            model_info = ImageModelInfo(family=declared, is_distilled=False, size=None)
        else:
            model_info = detect_image_model(resolved_model)
        defaults = resolve_defaults(model_info, app_config, {}, get_backend_name())
        family = _known_family(model_info.family)
        size_variant = model_info.size
        enhance_max_words = resolve_enhance_ceiling(app_config, family=model_info.family, mode="image")
    except Exception:
        enhance_max_words = resolve_enhance_ceiling(app_config, family=None, mode="image")
        defaults = {
            "steps": app_config.get("generation", {}).get("default_steps", 10),
            "guidance": app_config.get("generation", {}).get("default_guidance", 3.5),
            "scheduler": None,
            "supports_negative_prompt": False,
            "supports_img2img": True,
            "supports_upscale": True,
            "supports_json_prompt": False,
            "supports_first_sigma": False,
            "supports_scheduler": True,
            "dimension_min": 16,
            "dimension_max": None,
            "dimension_step": 16,
        }
    ratio, size = _pick_supported_image_bootstrap_preset(web_config, defaults, preferred_ratio, preferred_size)
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
        "supports_quantize": bool(_image_quantize_levels(web_config, defaults)),
        "quantize": None,
        "image_strength": _IMAGE_BOOTSTRAP_STRENGTH,
        "postprocess": _image_bootstrap_postprocess(),
        "upscale": _image_bootstrap_upscale(),
        "supports_img2img": bool(defaults.get("supports_img2img", True)),
        "supports_upscale": bool(defaults.get("supports_upscale", True)),
        "supports_json_prompt": bool(defaults.get("supports_json_prompt", False)),
        "supports_first_sigma": bool(defaults.get("supports_first_sigma", False)),
        "supports_scheduler": bool(defaults.get("supports_scheduler", True)),
        "dimension_min": int(defaults.get("dimension_min", 16)),
        "dimension_max": defaults.get("dimension_max", None),
        "dimension_step": int(defaults.get("dimension_step", 16)),
        "enhance_max_words": enhance_max_words,
        "family": family,
        "size_variant": size_variant,
    }


def _build_video_bootstrap_defaults(model_name: str, web_config: WebUiConfig) -> dict[str, Any]:
    app_config = web_config.app_config
    ratio, size = resolve_video_ratio_size_defaults(web_config)
    supports_i2v = False
    fps = 24
    frame_alignment = 1
    detected_family: str | None = None
    try:
        resolved_model = resolve_model_path(model_name, aliases=app_config.get("model_aliases", {}), platform_key=sys.platform)
        model_info = detect_video_model(resolved_model)
        detected_family = _known_family(getattr(model_info, "family", None))
        family = _resolve_video_bootstrap_family(app_config, detected_family)
        supports_i2v = bool(getattr(model_info, "supports_i2v", False))
        fps_value = getattr(model_info, "default_fps", 24)
        fps = fps_value if isinstance(fps_value, int) else 24
        alignment_value = getattr(model_info, "frame_alignment", 1)
        frame_alignment = alignment_value if isinstance(alignment_value, int) and alignment_value > 0 else 1
    except Exception:
        family = _resolve_video_bootstrap_family(app_config, None)
    defaults = resolve_video_defaults(family, app_config, {"ratio": ratio, "size": size})
    described = video_family(app_config, detected_family) if detected_family else None
    capabilities = video_capabilities(app_config, detected_family) if detected_family else None
    frame_count = defaults["num_frames"]
    if described is not None:
        frame_count = align_frames(frame_count, described.frame_rule)
    frame_rule = described.frame_rule.to_json() if described is not None else {"alignment": frame_alignment, "offset": 1, "round": "nearest"}
    return {
        "ratio": defaults.get("ratio", ratio),
        "size": defaults.get("size", size),
        "steps": defaults["steps"],
        "width": defaults["width"],
        "height": defaults["height"],
        "frame_count": frame_count,
        "audio": True,
        "low_memory": True,
        "supports_i2v": supports_i2v,
        "supports_quantize": False,
        "quantize": None,
        "max_steps": described.max_steps if described is not None else _default_video_max_steps(app_config, family),
        "fps": fps,
        "frame_rule": frame_rule,
        # A family the config does not describe keeps every capability but the opt-in last frame.
        "hidden_controls": hidden_video_controls(capabilities) if capabilities is not None else sorted(END_FRAME_CONTROLS),
        "supports_extend": capabilities.extend if capabilities is not None else True,
        "step_options": list(described.step_options) if described is not None and described.step_options else None,
        "extend_frame_step": extend_step(described),
        "upscale": _video_bootstrap_upscale(),
        "enhance_max_words": resolve_enhance_ceiling(app_config, family=family, mode="video"),
        "family": detected_family,
    }
