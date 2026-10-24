"""Training-free block-tail residual cache for MiniMax-H3 denoising."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx


@dataclass(frozen=True)
class BlockCacheConfig:
    """Controls when a denoising step may reuse the previous block tail."""

    sigma_threshold: float = 0.12
    start_percent: float = 0.10
    end_percent: float = 0.90
    max_consecutive: int = 2
    cache_depth: float = 0.75

    def __post_init__(self) -> None:
        if self.sigma_threshold < 0:
            raise ValueError("sigma_threshold must be non-negative")
        if not 0.0 <= self.start_percent <= self.end_percent <= 1.0:
            raise ValueError("cache window must satisfy 0 <= start <= end <= 1")
        if self.max_consecutive < 0:
            raise ValueError("max_consecutive must be non-negative")
        if not 0.0 <= self.cache_depth < 1.0:
            raise ValueError("cache_depth must satisfy 0 <= depth < 1")

    @property
    def enabled(self) -> bool:
        return (
            self.sigma_threshold > 0
            and self.max_consecutive > 0
            and self.cache_depth > 0
        )


class BlockResidualCache:
    """Reuse the trailing block-stack residual when adjacent sigmas are close.

    A full step records ``tail_residual = full_output - warm_prefix_output``.
    An eligible cache step recomputes only the warm prefix and adds that saved
    residual. The first and last denoising steps always execute every block.
    """

    def __init__(self, config: BlockCacheConfig | None = None) -> None:
        self.config = config or BlockCacheConfig()
        self.reset()

    def reset(self) -> None:
        self.residual: mx.array | None = None
        self.signature: tuple | None = None
        self.previous_sigma: float | None = None
        self.previous_step = -1
        self.consecutive_hits = 0
        self.full_steps = 0
        self.cache_steps = 0
        self.executed_blocks = 0
        self.skipped_blocks = 0

    def stats(self) -> dict[str, int | float]:
        total = self.executed_blocks + self.skipped_blocks
        return {
            "full_steps": self.full_steps,
            "cache_steps": self.cache_steps,
            "executed_blocks": self.executed_blocks,
            "skipped_blocks": self.skipped_blocks,
            "saved_fraction": self.skipped_blocks / total if total else 0.0,
        }

    def _warm_blocks(self, block_count: int) -> int:
        return max(
            0,
            min(
                block_count - 1,
                round(block_count * (1.0 - self.config.cache_depth)),
            ),
        )

    def run(
        self,
        x: mx.array,
        *,
        block_count: int,
        run_range: Callable[[mx.array, int, int], mx.array],
        sigma: float,
        step_index: int,
        total_steps: int,
    ) -> mx.array:
        """Run a full or cached block stack for one denoising step."""
        if block_count < 1:
            return x
        if not self.config.enabled:
            return run_range(x, 0, block_count)

        signature = (tuple(x.shape), str(x.dtype), block_count, total_steps)
        if signature != self.signature or step_index <= self.previous_step:
            self.reset()
            self.signature = signature

        progress = step_index / max(total_steps - 1, 1)
        sigma_delta = (
            float("inf")
            if self.previous_sigma is None
            else abs(float(sigma) - self.previous_sigma)
        )
        can_cache = (
            self.residual is not None
            and 0 < step_index < total_steps - 1
            and self.config.start_percent <= progress <= self.config.end_percent
            and sigma_delta < self.config.sigma_threshold
            and self.consecutive_hits < self.config.max_consecutive
        )

        warm_blocks = self._warm_blocks(block_count)
        if can_cache:
            warm = run_range(x, 0, warm_blocks) if warm_blocks else x
            out = warm + self.residual
            mx.eval(out)
            self.cache_steps += 1
            self.consecutive_hits += 1
            self.executed_blocks += warm_blocks
            self.skipped_blocks += block_count - warm_blocks
        else:
            warm = run_range(x, 0, warm_blocks) if warm_blocks else x
            out = run_range(warm, warm_blocks, block_count)
            residual = out - warm
            mx.eval(out, residual)
            self.residual = residual
            self.full_steps += 1
            self.consecutive_hits = 0
            self.executed_blocks += block_count

        self.previous_sigma = float(sigma)
        self.previous_step = step_index
        return out


__all__ = ["BlockCacheConfig", "BlockResidualCache"]