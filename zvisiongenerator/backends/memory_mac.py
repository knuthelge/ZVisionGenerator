"""MLX unified-memory helpers for the macOS backends."""

from __future__ import annotations


def release_memory() -> None:
    """Drop unreachable model references and return MLX's buffer cache to the system."""
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    aggressive_cleanup()


def memory_budget_bytes() -> int | None:
    """Return Apple's recommended GPU working-set size for this machine, in bytes."""
    import mlx.core as mx

    value = mx.device_info().get("max_recommended_working_set_size")
    return int(value) if isinstance(value, int) and value > 0 else None
