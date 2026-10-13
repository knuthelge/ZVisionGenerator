"""MLX unified-memory helpers for the macOS backends."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

# Freed buffers MLX may keep for reuse. Its default (the whole memory limit) let the cache grow to 16-29 GB
# alongside the model and pushed macOS into swap; 4 GB showed no step slowdown.
BUFFER_CACHE_LIMIT_BYTES = 4 * 1024**3


def release_memory() -> None:
    """Drop unreachable model references and return MLX's buffer cache to the system."""
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    aggressive_cleanup()


def memory_budget_bytes() -> int | None:
    """Return Apple's recommended GPU working-set size for this machine, in bytes."""
    import mlx.core as mx

    value = mx.device_info().get("max_recommended_working_set_size")
    return int(value) if isinstance(value, int) and value > 0 else None


def memory_size_bytes() -> int | None:
    """Return the machine's total unified memory, in bytes."""
    import mlx.core as mx

    value = mx.device_info().get("memory_size")
    return int(value) if isinstance(value, int) and value > 0 else None


@contextlib.contextmanager
def mlx_cache_limit(limit: int) -> Iterator[None]:
    """Set the process-wide MLX buffer-cache limit for the block, then restore the previous one.

    MLX has no cache-limit getter; ``set_cache_limit`` returns the value it replaces.
    The previous limit is restored even when the block raises.

    Args:
        limit: Cache limit in bytes to apply inside the block; 0 disables caching.

    Yields:
        None.
    """
    import mlx.core as mx

    previous = mx.set_cache_limit(limit)
    try:
        yield
    finally:
        mx.set_cache_limit(previous)


def materialize_weights(model: Any) -> None:
    """Evaluate every parameter once so the weights stay resident.

    mflux loads weights lazily; unevaluated, every generation rebuilds them from the safetensors
    files (and quantizes them again), costing tens of seconds and a near-bf16 memory peak per image.
    """
    import mlx.core as mx

    mx.eval(model.parameters())


def apply_buffer_cache_policy() -> None:
    """Cap MLX's free-buffer cache and drop what loading left behind."""
    import mlx.core as mx

    mx.set_cache_limit(BUFFER_CACHE_LIMIT_BYTES)
    mx.clear_cache()
