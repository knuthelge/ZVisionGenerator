"""Locate fully downloaded model weights on disk without touching the network."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from zvisiongenerator.utils.paths import parse_huggingface_repo_reference

# Used only when ltx-pipelines-mlx cannot be imported to read its own ``gemma_model_id`` default.
_LTX_MLX_TEXT_ENCODER_FALLBACK = "mlx-community/gemma-3-12b-it-4bit"
_LTX_MLX_MARKER = "connector.safetensors"
_LTX_MLX_TRANSFORMERS = ("transformer.safetensors", "transformer-distilled.safetensors")
_WEIGHTLESS_CLASS_HINTS = ("Tokenizer", "Scheduler", "Processor", "FeatureExtractor")
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
            from ltx_pipelines_mlx import TextToVideoPipeline

            default = inspect.signature(TextToVideoPipeline.__init__).parameters["gemma_model_id"].default
        except Exception:  # noqa: BLE001 - the MLX pipeline is unavailable off macOS; fall back to the known default
            return _LTX_MLX_TEXT_ENCODER_FALLBACK
        if not isinstance(default, str) or not default:
            return _LTX_MLX_TEXT_ENCODER_FALLBACK
        _ltx_mlx_text_encoder_repo = default
    return _ltx_mlx_text_encoder_repo


def is_ltx_mlx_layout(model_dir: Path) -> bool:
    """Return whether *model_dir* uses the single-file LTX MLX layout (separate Gemma text encoder)."""
    return (model_dir / _LTX_MLX_MARKER).is_file()


def ltx_mlx_transformer_file(model_dir: Path) -> Path | None:
    """Return the transformer file ltx-pipelines-mlx loads from *model_dir*, if present."""
    for name in _LTX_MLX_TRANSFORMERS:
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
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
    return next((snapshot for snapshot in _cached_snapshots(repo.repo_id, repo.revision) if has_complete_weights(snapshot)), None)


def model_weight_files(model_dir: Path, components: tuple[str, ...] | None = None) -> tuple[Path, ...]:
    """Return the safetensors files the loader reads from *model_dir*, in one directory walk.

    For diffusers layouts only the weight-bearing components listed in ``model_index.json`` count, so
    extra files in a full repo download (e.g. a root-level single-file checkpoint) are ignored. Precision
    variants (``*.fp16.safetensors``) are ignored in any folder that also holds the plain weights. Symlinked
    component folders are followed: converted checkpoints link their text encoder and VAE from the base repo.

    Args:
        model_dir: The model directory.
        components: Its :func:`weighted_components`, when the caller already read them.
    """
    files = _without_redundant_variants(sorted(path for path in model_dir.rglob("*.safetensors", recurse_symlinks=True) if path.is_file()))
    components = weighted_components(model_dir) if components is None else components
    if components is None:
        return tuple(files)
    allowed = set(components)
    return tuple(path for path in files if component_of(model_dir, path) in allowed)


def weighted_components(model_dir: Path) -> tuple[str, ...] | None:
    """Return the weight-bearing component folders from ``model_index.json``, or ``None`` without a readable index."""
    index_path = model_dir / "model_index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(index, dict):
        return None
    return tuple(
        name
        for name, spec in index.items()
        if not name.startswith("_") and isinstance(spec, list) and len(spec) == 2 and all(spec) and not any(hint in str(spec[1]) for hint in _WEIGHTLESS_CLASS_HINTS)
    )


def has_complete_weights(model_dir: Path) -> bool:
    """Return whether *model_dir* holds every weight file its layout declares."""
    components = weighted_components(model_dir)
    if components is None and (model_dir / "model_index.json").is_file():
        return False  # An unreadable index cannot vouch for the download.
    files = model_weight_files(model_dir, components)
    if not files:
        return False
    if is_ltx_mlx_layout(model_dir):
        return ltx_mlx_transformer_file(model_dir) is not None and (model_dir / "vae_decoder.safetensors").is_file()
    present = {component_of(model_dir, path) for path in files}
    return all(name in present for name in components or ()) and _shards_complete(files)


def huggingface_cache_repo_dir(repo_id: str) -> Path:
    """Return the HuggingFace cache folder that holds every downloaded revision of *repo_id*."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"


def _cached_snapshots(repo_id: str, revision: str | None) -> list[Path]:
    """Return candidate cache snapshots, newest first, as mflux's offline resolution considers them."""
    if revision is not None:
        from huggingface_hub import snapshot_download

        try:
            return [Path(snapshot_download(repo_id, revision=revision, local_files_only=True))]
        except Exception:  # noqa: BLE001 - any cache miss or malformed cache simply means "not downloaded"
            return []
    snapshots_dir = huggingface_cache_repo_dir(repo_id) / "snapshots"
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
