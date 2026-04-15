"""Image model type detection for Z-Vision Generator.

Detects image model family (ZImage, FLUX.2 Klein, etc.) from model_index.json
for both local model directories and HuggingFace repo IDs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

_HF_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")

_CLASS_NAME_MAP: dict[str, str] = {
    "ZImagePipeline": "zimage",
    "Flux2KleinPipeline": "flux2_klein",
    "Flux2Pipeline": "flux2",
    "FluxPipeline": "flux1",
}


@dataclass(frozen=True)
class ImageModelInfo:
    family: str  # "zimage" | "flux2_klein" | "flux2" | "flux1" | "unknown"
    is_distilled: bool  # True for distilled Klein, False otherwise
    size: str | None  # "4b" | "9b" | None


def detect_image_model(model_path: str) -> ImageModelInfo:
    """Detect model family from model_index.json.

    Args:
        model_path: Local directory path or HuggingFace repo ID.

    Returns:
        ImageModelInfo with detected properties.

    Raises:
        FileNotFoundError: model_index.json missing for local paths.
        ValueError: Unsupported model type or missing _class_name.
    """
    # Detect local filesystem paths that don't exist — avoid confusing HF lookups
    is_local = os.path.isdir(model_path)

    if not is_local:
        # Explicitly local paths: absolute, starts with ./ ../ ~/, or has backslash
        _explicitly_local = os.path.isabs(model_path) or model_path.startswith(("./", "../", "~/")) or "\\" in model_path
        if _explicitly_local:
            raise FileNotFoundError(f"Local model directory not found: {model_path}")

        # Paths with / that don't match HF repo pattern (org/repo)
        if "/" in model_path and _HF_REPO_PATTERN.match(model_path) is None:
            raise FileNotFoundError(f"Local model directory not found: {model_path}")
        # Remaining: bare names or org/repo → treat as HF repo ID

    index = _read_model_index(model_path, is_local)

    class_name = index.get("_class_name")
    if not class_name:
        raise ValueError(f"model_index.json for '{model_path}' is missing the '_class_name' field.")

    family = _CLASS_NAME_MAP.get(class_name, "unknown")

    if family == "flux2":
        raise ValueError("FLUX.2-dev (32B) is not supported. Use FLUX.2 Klein variants (4B/9B).")

    if family == "flux2_klein":
        is_distilled = index.get("is_distilled", True)
        size = _detect_klein_size(model_path, is_local)
        info = ImageModelInfo(family=family, is_distilled=is_distilled, size=size)
    elif family == "flux1":
        info = ImageModelInfo(family="flux1", is_distilled=False, size=None)
    elif family == "zimage":
        info = ImageModelInfo(family="zimage", is_distilled=False, size=None)
    else:
        info = ImageModelInfo(family="unknown", is_distilled=False, size=None)

    return info


def _read_model_index(model_path: str, is_local: bool) -> dict:
    """Read model_index.json from a local directory or HuggingFace Hub."""
    if is_local:
        index_path = os.path.join(model_path, "model_index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"model_index.json not found in {model_path}. Is this a valid HuggingFace model directory?")
        with open(index_path, encoding="utf-8") as f:
            return json.load(f)

    # HuggingFace repo ID — download model_index.json
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub is required to detect model type from HuggingFace repo IDs. Install it with: uv add huggingface_hub")

    try:
        downloaded_path = hf_hub_download(repo_id=model_path, filename="model_index.json")
    except Exception as e:
        first_segment = model_path.split("/")[0]
        if os.path.isdir(first_segment):
            raise FileNotFoundError(f"Could not find model '{model_path}'. The '{first_segment}/' directory exists locally — did you mean './{model_path}'?") from e
        raise
    with open(downloaded_path, encoding="utf-8") as f:
        return json.load(f)


def _detect_klein_size(model_path: str, is_local: bool) -> str | None:
    """Detect Klein model size (4B vs 9B).

    Strategy:
    1. Check if model_path contains "4b" or "9b" (case-insensitive).
    2. Fallback: read transformer/config.json and check num_single_layers.
    3. Final fallback: return None.
    """
    # Check path name
    match = re.search(r"\b(4b|9b)\b", model_path, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Fallback: read transformer config
    try:
        if is_local:
            config_path = os.path.join(model_path, "transformer", "config.json")
            if not os.path.isfile(config_path):
                return None
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                return None
            downloaded = hf_hub_download(repo_id=model_path, filename="transformer/config.json")
            with open(downloaded, encoding="utf-8") as f:
                config = json.load(f)

        num_single = config.get("num_single_layers")
        if num_single is not None:
            # 4B has 20 single layers, 9B has 24 single layers
            return "4b" if num_single <= 20 else "9b"
    except OSError, KeyError, ValueError:
        pass
    except Exception as e:
        if "EntryNotFoundError" in type(e).__name__ or "HfHubHTTPError" in type(e).__name__:
            return None
        raise

    return None
