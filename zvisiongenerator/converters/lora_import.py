"""LoRA import logic — local copy and HuggingFace download."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_INVALID_NAME_PATTERN = re.compile(r"[/\\]|\.\.|[\x00]")


def _validate_name(name: str) -> str:
    """Validate and normalise a LoRA name.

    Strips trailing .safetensors if present, then rejects names containing
    ``/``, ``\\``, ``..``, or null bytes.

    Returns:
        The sanitised name (without extension).

    Raises:
        ValueError: If the name is empty or contains forbidden characters.
    """
    if name.endswith(".safetensors"):
        name = name.removesuffix(".safetensors")
    if not name:
        raise ValueError("LoRA name must not be empty.")
    if _INVALID_NAME_PATTERN.search(name):
        raise ValueError(f"Invalid LoRA name '{name}': must not contain '/', '\\', '..', or null bytes.")
    return name


def import_lora_local(source: Path, dest_dir: Path, name: str | None = None) -> Path:
    """Copy a local .safetensors file to dest_dir/<name>.safetensors.

    Args:
        source: Path to local .safetensors file.
        dest_dir: Target directory (e.g., ~/.ziv/loras/).
        name: Optional custom name (without extension). Defaults to source stem.

    Returns:
        Path to the copied file.

    Raises:
        FileNotFoundError: source doesn't exist.
        ValueError: source is not a .safetensors file, or name contains invalid chars.
        FileExistsError: destination already exists.
    """
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    if source.suffix != ".safetensors":
        raise ValueError(f"Source must be a .safetensors file, got: {source.name}")

    final_name = _validate_name(name if name is not None else source.stem)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{final_name}.safetensors"

    if dest.exists():
        raise FileExistsError(f"Destination already exists: {dest}")

    shutil.copy2(source, dest)
    return dest


def import_lora_hf(
    repo_id: str,
    dest_dir: Path,
    filename: str | None = None,
    name: str | None = None,
) -> Path:
    """Download a LoRA .safetensors from a HuggingFace repo.

    Args:
        repo_id: HuggingFace repo ID (e.g., "user/lora-name").
        dest_dir: Target directory (e.g., ~/.ziv/loras/).
        filename: Specific .safetensors file in the repo. If None, auto-detects
                  (error if repo has multiple .safetensors files).
        name: Optional custom name (without extension). Defaults to filename stem.

    Returns:
        Path to the downloaded file.

    Raises:
        ValueError: No .safetensors found, ambiguous (multiple files without
            --file), or invalid name.
        FileExistsError: destination already exists.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    if filename is None:
        all_files = list_repo_files(repo_id)
        safetensors = [f for f in all_files if f.endswith(".safetensors")]
        if not safetensors:
            raise ValueError(f"No .safetensors files found in repo '{repo_id}'.")
        if len(safetensors) > 1:
            names = ", ".join(safetensors)
            raise ValueError(f"Multiple .safetensors files in repo '{repo_id}': {names}. Specify one with the filename argument.")
        filename = safetensors[0]

    if not filename.endswith(".safetensors"):
        raise ValueError(f"File must be a .safetensors file, got: {filename}")

    stem = Path(filename).stem
    final_name = _validate_name(name if name is not None else stem)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{final_name}.safetensors"

    if dest.exists():
        raise FileExistsError(f"Destination already exists: {dest}")

    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copy2(cached, dest)
    return dest
