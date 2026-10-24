"""Locate fully downloaded model weights on disk without touching the network."""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Final

from zvisiongenerator.utils.paths import parse_huggingface_repo_reference

# Used only when ltx-pipelines-mlx cannot be imported to read its own ``gemma_model_id`` default.
_LTX_MLX_TEXT_ENCODER_FALLBACK = "mlx-community/gemma-3-12b-it-4bit"
_LTX_MLX_MARKER = "connector.safetensors"
_LTX_MLX_TRANSFORMER = "transformer.safetensors"
_LTX_MLX_DISTILLED_GLOB = "transformer-distilled-*.safetensors"
_LTX_MLX_DISTILLED_UNVERSIONED = "transformer-distilled.safetensors"
# The spatial upsampler names ltx-pipelines-mlx looks for (``_resolve_upsampler_path``), 2.5 packs first.
_LTX_MLX_UPSAMPLER_STEMS = ("spatial_upscaler_x2_v1_0", "ltx-2.3-spatial-upscaler-x2", "spatial_upscaler_x2_v1_1")
# Bundled Gemma 4 text encoder of an LTX-2.5 pack: ltx_core_mlx ``encoder_configurator._TEXT_ENCODER_WEIGHTS`` / ``_TEXT_ENCODER_CONFIG``.
_LTX_MLX_BUNDLED_ENCODER_WEIGHTS = "text_encoder.safetensors"
_LTX_MLX_BUNDLED_ENCODER_CONFIG = "text_encoder_config.json"
# Video VAE decoder names: ltx_pipelines_mlx ``blocks._video_vae_names`` (the conv decoder of 2.5 packs wins over the 2.3 one).
_LTX_MLX_DECODER_CONV = "vae_decoder_conv.safetensors"
_LTX_MLX_DECODER = "vae_decoder.safetensors"
_WEIGHTLESS_CLASS_HINTS = ("Tokenizer", "Scheduler", "Processor", "FeatureExtractor")
_TRANSFORMER_COMPONENT = "transformer"
# The diffusers pipeline classes of LTX-2 repos. Those with a Gemma 3 text encoder (LTX-2.3) are downloaded without their
# transformer weights on Windows/Linux, which load the transformer from a single-file checkpoint
# (video_model_presets.ltx.diffusers.transformer).
_LTX2_PIPELINES = ("LTX2Pipeline", "LTX2ImageToVideoPipeline")
# Text-encoder class prefix of the LTX-2.5 layout, whose repos carry their own transformer weights.
_GEMMA4_PREFIX = "Gemma4"
# Components of an LTX-2 ``model_index.json`` that the app never loads, so they need no weights on disk.
LTX2_UNLOADED_COMPONENTS: Final[frozenset[str]] = frozenset({"processor", "prompt_enhancer", "duration_head", "diffusion_decoder"})
_SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d+)-of-(?P<total>\d+)\.safetensors$")
# Diffusers precision variants (e.g. ``diffusion_pytorch_model.fp16.safetensors``, optionally sharded).
_VARIANT_RE = re.compile(r"\.(?:fp16|bf16|fp32|fp8)(?:-\d+-of-\d+)?\.safetensors$")

_ltx_mlx_text_encoder_repo: str | None = None


def ltx_mlx_text_encoder_repo() -> str:
    """Return the Gemma repo ltx-pipelines-mlx encodes prompts with (its ``gemma_model_id`` default).

    The value read from the pipeline is cached; the fallback is not, so a transient import failure is retried.
    """
    global _ltx_mlx_text_encoder_repo
    if _ltx_mlx_text_encoder_repo is None:
        try:
            from ltx_pipelines_mlx import BasePipeline

            default = inspect.signature(BasePipeline.__init__).parameters["gemma_model_id"].default
        except Exception:  # noqa: BLE001 - the MLX pipeline is unavailable off macOS; fall back to the known default
            return _LTX_MLX_TEXT_ENCODER_FALLBACK
        if not isinstance(default, str) or not default:
            return _LTX_MLX_TEXT_ENCODER_FALLBACK
        _ltx_mlx_text_encoder_repo = default
    return _ltx_mlx_text_encoder_repo


def is_ltx_mlx_layout(model_dir: Path) -> bool:
    """Return whether *model_dir* uses the single-file LTX MLX layout (separate Gemma text encoder)."""
    return (model_dir / _LTX_MLX_MARKER).is_file()


def is_ltx_model_dir(model_dir: Path) -> bool:
    """Return whether the folder *model_dir* holds an LTX model by its contents: the MLX layout, or an LTX-2 diffusers index."""
    if is_ltx_mlx_layout(model_dir):
        return True
    index = _read_model_index(model_dir)
    return index is not None and index.get("_class_name") in _LTX2_PIPELINES


def ltx_mlx_transformer_file(model_dir: Path) -> Path | None:
    """Return the transformer file ltx-pipelines-mlx loads from *model_dir*, if present.

    Mirrors the pipeline's own choice: ``transformer.safetensors``, else the lexicographically last
    versioned ``transformer-distilled-*.safetensors`` (the newest release), else the unversioned
    ``transformer-distilled.safetensors``.
    """
    plain = model_dir / _LTX_MLX_TRANSFORMER
    if plain.is_file():
        return plain
    versioned = sorted(path for path in model_dir.glob(_LTX_MLX_DISTILLED_GLOB) if path.is_file())
    if versioned:
        return versioned[-1]
    unversioned = model_dir / _LTX_MLX_DISTILLED_UNVERSIONED
    return unversioned if unversioned.is_file() else None


def ltx_mlx_bundled_text_encoder(model_dir: Path) -> Path | None:
    """Return the text encoder bundled in an LTX-2.5 MLX pack, or ``None`` for a pack that needs a separate Gemma 3.

    Mirrors the pipeline's ``select_text_encoder``: the root ``text_encoder.safetensors`` counts only when
    ``text_encoder_config.json`` sits next to it.
    """
    weights = model_dir / _LTX_MLX_BUNDLED_ENCODER_WEIGHTS
    config = model_dir / _LTX_MLX_BUNDLED_ENCODER_CONFIG
    return weights if weights.is_file() and config.is_file() else None


def ltx_mlx_video_decoder_file(model_dir: Path) -> Path | None:
    """Return the video VAE decoder file ltx-pipelines-mlx loads from *model_dir*, if present.

    Mirrors the pipeline's ``_video_vae_names``: ``vae_decoder_conv.safetensors`` (LTX-2.5) when present,
    else ``vae_decoder.safetensors`` (LTX-2.3).
    """
    for name in (_LTX_MLX_DECODER_CONV, _LTX_MLX_DECODER):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    return None


def ltx_mlx_upsampler_file(model_dir: Path) -> Path | None:
    """Return the spatial upsampler file ltx-pipelines-mlx loads from *model_dir* for an upscaled clip, if present.

    Mirrors the pipeline's lookup: for each known name, the lexicographically last versioned file
    (``{name}-*.safetensors``), else ``{name}.safetensors``.
    """
    for stem in _LTX_MLX_UPSAMPLER_STEMS:
        versioned = sorted(path for path in model_dir.glob(f"{stem}-*.safetensors") if path.is_file())
        if versioned:
            return versioned[-1]
        plain = model_dir / f"{stem}.safetensors"
        if plain.is_file():
            return plain
    return None


def find_local_model_dir(reference: str) -> Path | None:
    """Return the on-disk directory for *reference* when its weights are fully downloaded.

    Args:
        reference: A local model directory or a HuggingFace repo id (optionally ``@revision``).

    Returns:
        The model directory (a HuggingFace cache snapshot for repo ids), or ``None`` when the
        weights are missing or only partially downloaded.
    """
    local = Path(reference).expanduser()
    if local.is_dir():
        return local if has_complete_weights(local) else None
    repo = parse_huggingface_repo_reference(reference)
    if repo is None:
        return None
    return next((snapshot for snapshot in cached_snapshot_dirs(repo.repo_id, repo.revision) if has_complete_weights(snapshot)), None)


def model_weight_files(model_dir: Path, components: tuple[str, ...] | None = None) -> tuple[Path, ...]:
    """Return the safetensors files the loader reads from *model_dir*, in one directory walk.

    For diffusers layouts only the weight-bearing components listed in ``model_index.json`` count, so
    extra files in a full repo download (e.g. a root-level single-file checkpoint) are ignored. For Krea 2 models
    whose ``transformer`` folder holds no weights, root-level files stand in for it: mflux downloads only Krea 2
    Turbo's single-file ``turbo.safetensors`` and saves Krea 2 stored quants as root-level shards.
    Precision variants (``*.fp16.safetensors``) are ignored in any folder that also holds the plain weights.
    Symlinked component folders are followed: converted checkpoints link their text encoder and VAE from the base
    repo.

    Args:
        model_dir: The model directory.
        components: Its :func:`weighted_components`, when the caller already read them.
    """
    files = _without_redundant_variants(sorted(path for path in model_dir.rglob("*.safetensors", recurse_symlinks=True) if path.is_file()))
    components = weighted_components(model_dir) if components is None else components
    if components is None:
        return tuple(files)
    allowed = set(components)
    selected = [path for path in files if component_of(model_dir, path) in allowed]
    if _TRANSFORMER_COMPONENT in allowed and not any(component_of(model_dir, path) == _TRANSFORMER_COMPONENT for path in selected) and is_krea2_model(model_dir):
        selected += [path for path in files if component_of(model_dir, path) == ""]
    return tuple(selected)


def weighted_components(model_dir: Path) -> tuple[str, ...] | None:
    """Return the weight-bearing component folders from ``model_index.json``, or ``None`` without a readable index."""
    index = _read_model_index(model_dir)
    if index is None:
        return None
    skipped = LTX2_UNLOADED_COMPONENTS if index.get("_class_name") in _LTX2_PIPELINES else frozenset()
    return tuple(
        name
        for name, spec in index.items()
        if not name.startswith("_") and name not in skipped and isinstance(spec, list) and len(spec) == 2 and all(spec) and not any(hint in str(spec[1]) for hint in _WEIGHTLESS_CLASS_HINTS)
    )


def _read_model_index(model_dir: Path) -> dict | None:
    """Return the parsed ``model_index.json`` of *model_dir*, or ``None`` when it is missing or unreadable."""
    index_path = model_dir / "model_index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return index if isinstance(index, dict) else None


def is_krea2_model(model_dir: Path) -> bool:
    """Return whether *model_dir*'s ``model_index.json`` declares a Krea 2 pipeline."""
    index = _read_model_index(model_dir)
    return index is not None and index.get("_class_name") == "Krea2Pipeline"


def is_ideogram4_model(model_dir: Path) -> bool:
    """Return whether *model_dir*'s ``model_index.json`` declares an Ideogram 4 pipeline."""
    index = _read_model_index(model_dir)
    return index is not None and index.get("_class_name") == "Ideogram4Pipeline"


def has_complete_weights(model_dir: Path, only: set[str] | None = None) -> bool:
    """Return whether *model_dir* holds every weight file its layout declares.

    Args:
        model_dir: The model directory.
        only: Restricts the components that must hold weights to these folder names (``set()`` requires none);
            ``None`` requires every declared one. File selection and shard completeness are checked either way.
    """
    components = weighted_components(model_dir)
    if components is None and (model_dir / "model_index.json").is_file():
        return False  # An unreadable index cannot vouch for the download.
    files = model_weight_files(model_dir, components)
    if not files and (only is None or components is None):
        return False  # Without an index nothing declares what a complete snapshot holds.
    if is_ltx_mlx_layout(model_dir):
        return ltx_mlx_transformer_file(model_dir) is not None and ltx_mlx_video_decoder_file(model_dir) is not None and _bundled_encoder_downloaded(model_dir)
    # Root-level files are only selected when they stand in for the transformer folder.
    present = {component_of(model_dir, path) or _TRANSFORMER_COMPONENT for path in files}
    required = [name for name in components or () if (only is None or name in only) and not (name == _TRANSFORMER_COMPONENT and _has_separate_transformer(model_dir))]
    return all(name in present for name in required) and _shards_complete(files)


def _bundled_encoder_downloaded(model_dir: Path) -> bool:
    """Return whether an MLX pack that declares a bundled text encoder (its config file) also holds the weights.

    HuggingFace snapshots link a file only once it is complete, so the small config appears first in an interrupted download.
    """
    return not (model_dir / _LTX_MLX_BUNDLED_ENCODER_CONFIG).is_file() or (model_dir / _LTX_MLX_BUNDLED_ENCODER_WEIGHTS).is_file()


def ltx2_text_encoder_class(model_dir: Path) -> str | None:
    """Return the ``text_encoder`` class name of an LTX-2 diffusers repo, read from its ``model_index.json``.

    Returns:
        The class name (e.g. ``Gemma3ForConditionalGeneration``), or ``None`` when the index is missing, unreadable or
        not an LTX-2 pipeline, or when its ``text_encoder`` entry is malformed.
    """
    index = _read_model_index(model_dir)
    if index is None or index.get("_class_name") not in _LTX2_PIPELINES:
        return None
    spec = index.get("text_encoder")
    if not isinstance(spec, list) or len(spec) != 2 or not isinstance(spec[1], str) or not spec[1]:
        return None
    return spec[1]


def _has_separate_transformer(model_dir: Path) -> bool:
    """Return whether *model_dir* is an LTX-2 pipeline whose transformer may load from a separate checkpoint instead of its folder.

    LTX-2.5 repos (Gemma 4 text encoder) hold their own transformer weights, so they never qualify.
    """
    text_encoder = ltx2_text_encoder_class(model_dir)
    return text_encoder is not None and not text_encoder.startswith(_GEMMA4_PREFIX)


def directory_bytes(path: Path) -> int:
    """Return the total size of the regular files under *path*, not following symlinks; 0 when it is missing."""
    if not path.is_dir():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            file = Path(root, name)
            if file.is_symlink():
                continue
            try:
                total += file.stat().st_size
            except OSError:
                continue
    return total


def disk_size_label(num_bytes: int) -> str:
    """Return *num_bytes* in decimal units: "34.7 GB", "92 MB" or "4 KB"."""
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.1f} GB"
    if num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.0f} MB"
    return f"{max(1, round(num_bytes / 1e3))} KB"


def huggingface_cache_dir() -> Path:
    """Return the HuggingFace hub cache folder (``HF_HUB_CACHE``, which follows ``HF_HOME``)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def huggingface_cache_repo_dir(repo_id: str) -> Path:
    """Return the HuggingFace cache folder that holds every downloaded revision of *repo_id*."""
    return huggingface_cache_dir() / f"models--{repo_id.replace('/', '--')}"


def cached_snapshot_dirs(repo_id: str, revision: str | None) -> list[Path]:
    """Return candidate cache snapshots, newest first, as mflux's offline resolution considers them.

    With a *revision* only that snapshot counts: a commit hash is its folder's name, and a tag or branch name maps to
    its commit through ``refs/``. The cache folder is read directly, because the Hub's own offline lookup refuses a
    snapshot that lacks any file of the repo, and diffusers only downloads the files it loads.
    """
    repo_dir = huggingface_cache_repo_dir(repo_id)
    if revision is not None:
        if revision in (".", ".."):
            return []
        ref = repo_dir / "refs" / revision
        try:
            commit = ref.read_text(encoding="utf-8").strip() if ref.is_file() else revision
        except OSError:
            return []
        snapshot = repo_dir / "snapshots" / commit
        return [snapshot] if snapshot.is_dir() else []
    snapshots_dir = repo_dir / "snapshots"
    try:
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    except OSError:
        return []
    return sorted(snapshots, key=lambda path: path.stat().st_mtime, reverse=True)


def _without_redundant_variants(files: list[Path]) -> list[Path]:
    """Drop precision-variant files from folders that also hold the plain (non-variant) weights."""
    folders_with_plain = {path.parent for path in files if not _VARIANT_RE.search(path.name)}
    return [path for path in files if not (_VARIANT_RE.search(path.name) and path.parent in folders_with_plain)]


def component_of(model_dir: Path, path: Path) -> str:
    """Return the top-level component folder of *path*, or ``""`` for files at the model root."""
    parts = path.relative_to(model_dir).parts
    return parts[0] if len(parts) > 1 else ""


def _shards_complete(files: tuple[Path, ...]) -> bool:
    """Check that every ``name-0000k-of-0000N.safetensors`` shard set is complete.

    Shard filenames are used instead of ``*.safetensors.index.json`` because some repos ship a stale index.
    """
    shard_sets: dict[tuple[Path, str, int], set[int]] = {}
    for path in files:
        match = _SHARD_RE.match(path.name)
        if match is None:
            continue
        key = (path.parent, match["prefix"], int(match["total"]))
        shard_sets.setdefault(key, set()).add(int(match["index"]))
    return all(indices >= set(range(1, total + 1)) for (_, _, total), indices in shard_sets.items())
