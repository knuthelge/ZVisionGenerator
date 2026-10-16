"""System-memory estimate of LTX on the CUDA video backend, read from the weight files' headers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from zvisiongenerator.backends.cuda.video import weights
from zvisiongenerator.utils.config import ltx_transformer_source
from zvisiongenerator.utils.model_files import find_local_model_dir
from zvisiongenerator.utils.model_memory import LtxCudaMemoryEstimate, estimate_ltx_cuda_system_memory

__all__ = ["cached_ltx_system_estimate", "ltx_system_estimate"]


def ltx_system_estimate(repo_dir: Path, transformer_input: Path, quantize: int) -> LtxCudaMemoryEstimate:
    """Estimate the system memory of LTX at *quantize* from the headers of the repo at *repo_dir* and the transformer at *transformer_input*."""
    block_matrices, transformer_rest = weights.transformer_elements(transformer_input)
    return estimate_ltx_cuda_system_memory(
        block_matrices=block_matrices,
        transformer_rest=transformer_rest,
        connectors=weights.component_elements(repo_dir / "connectors"),
        text_encoder=weights.text_encoder_bytes(repo_dir / "text_encoder"),
        quantize=quantize,
    )


def cached_ltx_system_estimate(
    config: dict[str, Any],
    resolved_model: str,
    quantize: int,
    *,
    find_local_dir: Callable[[str], Path | None] = find_local_model_dir,
) -> LtxCudaMemoryEstimate | None:
    """Estimate LTX's system memory from files already on disk, or return ``None`` when some are missing.

    Nothing is downloaded: the repo and the configured single-file transformer must be in the Hugging Face cache.

    Args:
        config: The loaded config, which names the single-file transformer.
        resolved_model: The model's resolved path (a local directory or a Hugging Face repo id).
        quantize: 4 or 8.
        find_local_dir: Locates a downloaded model directory.
    """
    repo_dir = find_local_dir(resolved_model)
    if repo_dir is None:
        return None
    source = ltx_transformer_source(config)
    transformer_input = _cached_hub_file(*source) if source else repo_dir / "transformer"
    if transformer_input is None:
        return None
    try:
        return ltx_system_estimate(repo_dir, transformer_input, quantize)
    except OSError, ValueError, KeyError, TypeError:
        return None


def _cached_hub_file(repo_id: str, filename: str) -> Path | None:
    """Return the local path of a Hugging Face Hub file if it is in the cache, without downloading."""
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id, filename)
    return Path(cached) if isinstance(cached, str) else None
