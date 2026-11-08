"""Read the facts about the LTX video models from the packaged model metadata: which version a model is, the files it and its companions download, and the options its backends load it with.

Nothing here reads settings; the functions take a model name and a platform and return what the metadata declares for them.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

from zvisiongenerator.downloads.declaration import companion_targets, model_targets, role_targets
from zvisiongenerator.downloads.target import DownloadTarget, HubFiles
from zvisiongenerator.utils.ltx_converted import LTX_FAMILY_MODEL_TYPE, read_converted_ltx
from zvisiongenerator.utils.model_metadata import family_block, load_model_metadata, model_entry, model_ids
from zvisiongenerator.utils.paths import get_ziv_data_dir, is_folder_name, join_huggingface_revision

__all__ = [
    "EULER_ANCESTRAL",
    "LTX_SAMPLERS",
    "SEPARATE_TRANSFORMER_IGNORE",
    "ltx_companion_file",
    "ltx_converter_base",
    "ltx_converter_repo_ignore",
    "ltx_diffusers_quantize",
    "ltx_distilled_lora_licence",
    "ltx_distilled_lora_source",
    "ltx_gemma_repo_id",
    "ltx_load_options",
    "ltx_mlx_base_pack",
    "ltx_version",
    "ltx_version_for_type",
]

EULER_ANCESTRAL: Final = "euler_ancestral"
LTX_SAMPLERS: Final = frozenset({EULER_ANCESTRAL})
"""The LTX denoising samplers a model version may select; the single definition of the names."""

SEPARATE_TRANSFORMER_IGNORE: Final = ("transformer/*.safetensors", "transformer/*.safetensors.index.json")
"""The patterns an LTX diffusers repo is downloaded without when a single-file transformer replaces the repo's own weights."""


def ltx_version_for_type(model_type: str) -> str | None:
    """Return the id of the LTX version a converter model type belongs to (its ``model_type``); ``None`` for ``ltx-2.3``, the family block.

    Raises:
        ValueError: If no version has *model_type*.
    """
    if model_type == LTX_FAMILY_MODEL_TYPE:
        return None
    for version, block in load_model_metadata()["video_families"]["ltx"].get("versions", {}).items():
        if block.get("model_type") == model_type:
            return version
    raise ValueError(f"No LTX version has model_type '{model_type}'.")


def ltx_version(model_name: str | None) -> str | None:
    """Return the id of the LTX version the model *model_name* is, or ``None``.

    A built-in model names its version in its entry; a converted LTX folder (by name or path) is the version whose
    ``model_type`` is its manifest's kind. ``None`` means the family block, which makes the model LTX-2.3: a ``None``
    model name, an LTX-2.3 alias, a repo id or a local path.
    """
    if model_name is None:
        return None
    entry = model_entry(model_name)
    if entry is not None:
        return entry.get("version") if entry["kind"] == "video" and entry.get("family") == "ltx" else None
    kind = _converted_model_type(model_name)
    return None if kind is None else ltx_version_for_type(kind)


def _converted_model_type(model_name: str) -> str | None:
    """Return the kind of the converted LTX folder *model_name* names (a folder in the models folder, or a path), or ``None`` when it is not one."""
    folder = get_ziv_data_dir() / "models" / model_name if is_folder_name(model_name) else Path(model_name).expanduser()
    manifest = read_converted_ltx(folder) if folder.is_dir() else None
    return manifest["kind"] if manifest is not None else None


def _hub_file(pairs: Iterable[tuple[str, DownloadTarget]], role: str) -> tuple[str, str] | None:
    """Return the single-file target of *pairs* with *role* as ``(repo[@revision], filename)``, or ``None`` when there is none."""
    targets = role_targets(pairs, role)
    if not targets:
        return None
    target = targets[0]
    return join_huggingface_revision(target.repo_id, target.revision), target.allow_patterns[0]


def ltx_companion_file(model_name: str | None, platform_key: str, role: str) -> tuple[str, str] | None:
    """Return the single-file companion of the LTX model *model_name* with *role* (``transformer`` or ``latent_upsampler``) on *platform_key* as ``(repo[@revision], filename)``.

    ``None`` means the model has none there: it uses its repo's own file (the LTX-2.5 transformer), or *platform_key* declares none.
    """
    return _hub_file(companion_targets("ltx", ltx_version(model_name), platform_key), role)


def ltx_distilled_lora_source(version: str | None = None) -> tuple[str, str]:
    """Return Lightricks' distilled LoRA of LTX version *version* (``distilled_lora``) as ``(repo[@revision], filename)``.

    LTX checkpoint conversion merges it into plain-dev checkpoints with ``--distill``. *version* is a version id
    (:func:`ltx_version_for_type`); ``None`` is the family block, LTX-2.3.
    """
    lora = family_block("ltx", version)["distilled_lora"]
    return lora["ref"], lora["file"]


def ltx_distilled_lora_licence(version: str | None = None) -> str:
    """Return the id of the licence record that covers the distilled LoRA of LTX version *version*."""
    return family_block("ltx", version)["distilled_lora"]["licence"]


def _model_target(model_id: str, platform_key: str) -> DownloadTarget | None:
    """Return the model itself (its first ``role: model`` target) that *model_id* downloads on *platform_key*, or ``None``."""
    return next(iter(role_targets(model_targets(model_id, platform_key) or (), "model")), None)


def _pack(target: DownloadTarget) -> HubFiles:
    """Return the pinned MLX pack *target* declares, with the files the app loads from it."""
    return HubFiles(target.repo_id, target.revision, target.allow_patterns)


def _mlx_packs() -> dict[str, HubFiles]:
    """Return the pinned MLX LTX packs, keyed by their ``owner/repo@revision`` reference, each with the files the app loads from it."""
    packs: dict[str, HubFiles] = {}
    for model_id in model_ids("video"):
        if model_entry(model_id)["family"] == "ltx":
            target = _model_target(model_id, "darwin")
            packs.setdefault(join_huggingface_revision(target.repo_id, target.revision), _pack(target))
    return packs


def _converter_base_id(bits: int, version: str | None) -> str:
    """Return the id of the model whose repo a converted LTX checkpoint of *version* at *bits* takes its other components from (``converter_base.q<bits>``)."""
    return family_block("ltx", version)["converter_base"][f"q{bits}"]


def ltx_converter_base(bits: int, version: str | None, platform_key: str) -> str:
    """Return the reference of the repo a converted LTX checkpoint of *version* at *bits* takes its other components from on *platform_key*.

    It is the model target of the model the version's ``converter_base.q<bits>`` names: the MLX pack on macOS, the diffusers repo on Windows and Linux.
    """
    target = _model_target(_converter_base_id(bits, version), platform_key)
    return join_huggingface_revision(target.repo_id, target.revision)


def ltx_mlx_base_pack(bits: int, version: str | None = None) -> HubFiles:
    """Return the pinned files of the MLX pack a converted checkpoint of *version* at *bits* takes its other components from on macOS (see :func:`ltx_converter_base`)."""
    return _pack(_model_target(_converter_base_id(bits, version), "darwin"))


def ltx_converter_repo_ignore(bits: int, version: str | None, platform_key: str) -> tuple[str, ...]:
    """Return the patterns the base repo of a converted LTX checkpoint of *version* at *bits* is downloaded without on *platform_key*.

    The converted checkpoint replaces the repo's own transformer, so :data:`SEPARATE_TRANSFORMER_IGNORE` always applies,
    followed by what the base model's target declares beyond them.
    """
    declared = _model_target(_converter_base_id(bits, version), platform_key).ignore_patterns or ()
    return (*SEPARATE_TRANSFORMER_IGNORE, *(pattern for pattern in declared if pattern not in SEPARATE_TRANSFORMER_IGNORE))


def _repo_ignore(model_name: str | None, platform_key: str, *, transformer_source: tuple[str, str] | None) -> tuple[str, ...]:
    """Return the patterns the repo of the LTX model *model_name* is downloaded and loaded without.

    A built-in model's own declaration is the whole list. Any other repo skips its transformer weights when a single-file
    *transformer_source* replaces them, and is downloaded whole otherwise.
    """
    target = None if model_name is None else _model_target(model_name, platform_key)
    if target is not None:
        return tuple(target.ignore_patterns or ())
    return SEPARATE_TRANSFORMER_IGNORE if transformer_source else ()


def ltx_gemma_repo_id() -> str:
    """Return the repo id of the Gemma 3 text encoder the macOS LTX-2.3 packs encode prompts with, without its revision."""
    return role_targets(companion_targets("ltx", None, "darwin"), "text_encoder")[0].repo_id


def ltx_diffusers_quantize(model_name: str | None) -> int | None:
    """Return the quantize level (4 or 8) the diffusers LTX backend loads the built-in model *model_name* at (``ltx-4`` is 4).

    ``None`` means *model_name* is no built-in LTX model (a repo id, a path or a converted folder). The Windows/Linux backend
    runs only 4-bit or 8-bit and refuses a model that has no level, unless it is a converted folder, which carries its own.
    """
    entry = None if model_name is None else model_entry(model_name)
    return entry["quantize"] if entry is not None and entry["kind"] == "video" and entry.get("family") == "ltx" else None


def ltx_load_options(model_name: str | None, platform_key: str, *, upscale: bool) -> dict[str, Any]:
    """Return the keyword arguments a video backend's ``load_model`` takes for the LTX model picked as *model_name* on *platform_key*.

    The diffusers options (``quantize``, ``transformer_source``, ``model_name``, ``latent_upsampler`` and the
    ``repo_ignore``, and the version-only ``sampler`` and ``image_crf``) are ignored by the MLX backend, and the MLX options
    (``mlx_packs``, and ``mlx_text_encoder`` when the version has one) by the diffusers backend. The single-file
    companions are those *platform_key* declares. ``repo_ignore`` is the finished list of patterns the diffusers repo is downloaded
    without, always present: a built-in model's own declaration, else :data:`SEPARATE_TRANSFORMER_IGNORE` when a single-file
    transformer replaces the repo's own, else none. The two version-only keys are present only when the version of *model_name* sets them.
    """
    version = ltx_version(model_name)
    block = family_block("ltx", version)
    pairs = companion_targets("ltx", version, platform_key)
    options: dict[str, Any] = {
        "quantize": ltx_diffusers_quantize(model_name),
        "transformer_source": _hub_file(pairs, "transformer"),
        "model_name": model_name,
        "mlx_packs": _mlx_packs(),
    }
    encoder = role_targets(companion_targets("ltx", version, "darwin"), "text_encoder")
    if encoder:
        options["mlx_text_encoder"] = join_huggingface_revision(encoder[0].repo_id, encoder[0].revision)
    options["repo_ignore"] = _repo_ignore(model_name, platform_key, transformer_source=options["transformer_source"])
    for key in ("sampler", "image_crf"):
        if key in block:
            options[key] = block[key]
    if upscale:
        options["upscale"] = True
        options["latent_upsampler"] = _hub_file(pairs, "latent_upsampler")
    return options
