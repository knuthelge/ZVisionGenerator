"""MLX unified-memory helpers for the macOS backends."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator


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
