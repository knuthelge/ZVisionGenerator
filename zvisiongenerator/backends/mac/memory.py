"""MLX unified-memory helpers for the macOS backends."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
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
    """Evaluate every parameter, one array at a time, so the weights stay resident.

    mflux loads weights lazily; unevaluated, every generation rebuilds them from the safetensors
    files (and quantizes them again), costing tens of seconds and a near-bf16 memory peak per image.
    Evaluating array by array keeps the peak near the final weights; one batched evaluation of the
    whole tree holds noticeably more memory while it runs.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    for _, parameter in tree_flatten(model.parameters()):
        mx.eval(parameter)


def evaluate[T](value: T) -> T:
    """Evaluate the lazy MLX *value* (an array, or a list or tuple of them) now and return it."""
    import mlx.core as mx

    mx.eval(value)
    return value


def apply_buffer_cache_policy() -> None:
    """Cap MLX's free-buffer cache and drop what loading left behind."""
    import mlx.core as mx

    mx.set_cache_limit(BUFFER_CACHE_LIMIT_BYTES)
    mx.clear_cache()


def load_resident[T](build: Callable[[], T]) -> T:
    """Build a model with MLX's buffer cache disabled, keep its weights resident, then cap the cache.

    ``build`` and the weight evaluation run with the buffer cache at 0, so freed buffers (the bf16
    originals while quantizing) return to the system instead of piling up beside the model. The cache
    policy is applied after the previous limit is restored. When ``build`` or the evaluation raises,
    the previous limit is restored and the policy is not applied.

    Args:
        build: Callable that constructs and returns the model.

    Returns:
        The model ``build`` returned, with every parameter evaluated.
    """
    with mlx_cache_limit(0):
        model = build()
        materialize_weights(model)
    apply_buffer_cache_policy()
    return model
