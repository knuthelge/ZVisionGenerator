"""Bench-only overrides, applied inside the process that runs one case; the app has no flag or setting for any of them.

Each one patches a function of the app in this process, so nothing a user can set reaches it and the app's code carries no
bench hook: ``show_card_of`` makes the app read a smaller card, ``show_mac_of`` a Mac with less memory, ``show_machine_of`` whichever this
machine is, and ``watch_placement`` records the placement the CUDA image
loader chose and, when asked, forces one in its place (a probe of the loader's fragmentation term needs a placement the
loader would step down from). On a Mac, ``trace_mlx_job`` samples the process's footprint and MLX's active and cached memory at every denoise step and around the decode.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Callable, Iterator
from typing import Any

GIB = 1024**3
_IMAGE_BACKEND = "zvisiongenerator.backends.cuda.image.backend"


def show_card_of(gib: float) -> None:
    """Make the app read a card of *gib* GiB of GPU memory, as it would on a smaller card; its system memory and the driver's free memory stay the machine's."""
    from zvisiongenerator.backends.cuda import memory

    real = memory.memory_sizes()
    if real is None:
        return

    def sizes() -> tuple[int, int]:
        return int(gib * GIB), real[1]

    memory.memory_sizes = sizes
    memory.card_total_bytes = lambda: int(gib * GIB)  # the features tuned on the card's whole memory read the same simulated card
    backend = sys.modules.get(_IMAGE_BACKEND)
    if backend is not None:  # the image backend bound the function by name when it was imported
        backend.memory_sizes = sizes


_MAC_VIDEO = "zvisiongenerator.backends.mac.video"
_LTX_DECODE_BUDGET_ENV = "LTX2_VAE_DECODE_BUDGET_GB"


def show_mac_of(total_gib: float) -> None:
    """Make the app read a Mac whose unified memory is *total_gib* GiB, for the verdict and for the paths that depend on the machine's size.

    The app builds its memory budget in :func:`zvisiongenerator.backends.get_accelerator_memory_budget`, which every verdict reads
    through the cached ``accelerator_memory_budget_or_none``; this replaces the first and clears the second. The LTX loader reads the
    machine's size from ``memory_size_bytes`` to stream its transformer on a 24 GB Mac or less, so that is replaced too, and the decode budget
    LTX reads from ``LTX2_VAE_DECODE_BUDGET_GB`` (half the unified memory when unset) is set to half of *total_gib* unless the variable
    is already set. Nothing else about the machine changes, so a job still peaks where this machine's MLX puts it.
    """
    from dataclasses import replace

    from zvisiongenerator import backends
    from zvisiongenerator.backends.mac import memory

    total = int(total_gib * GIB)
    real = backends.get_accelerator_memory_budget

    def budget() -> Any:
        machine = real()
        return None if machine is None else replace(machine, unified_bytes=total)

    def size() -> int:
        return total

    backends.get_accelerator_memory_budget = budget
    backends.accelerator_memory_budget_or_none.cache_clear()
    memory.memory_size_bytes = size
    video = sys.modules.get(_MAC_VIDEO)
    if video is not None:  # the Mac video backend bound the function by name when it was imported
        video.memory_size_bytes = size
    os.environ.setdefault(_LTX_DECODE_BUDGET_ENV, f"{total_gib / 2:g}")


def show_machine_of(budget_gib: float | None = None, total_gib: float | None = None) -> None:
    """Make the app read a smaller machine of this kind: a Mac of *total_gib* GiB of memory, or a CUDA card of *budget_gib* GiB.

    Raises:
        ValueError: *budget_gib* is given on a Mac, which is graded against its total memory only, or *total_gib* on a machine with a CUDA card, whose system memory stays the machine's, or neither is given.
    """
    if sys.platform == "darwin":
        if budget_gib is not None or total_gib is None:
            raise ValueError("A Mac is simulated by its total memory only: give --total-gib, not --budget-gib.")
        show_mac_of(total_gib)
        return
    if total_gib is not None or budget_gib is None:
        raise ValueError("A CUDA card is simulated by its memory (--budget-gib); its system memory stays the machine's.")
    from zvisiongenerator import backends

    show_card_of(budget_gib)
    backends.accelerator_memory_budget_or_none.cache_clear()


def watch_placement(forced: str | None) -> Callable[[], dict[str, Any]]:
    """Record the placement the CUDA image loader chooses, and take *forced* instead of it when that is set.

    Args:
        forced: ``"resident"``, ``"turns"`` or ``"streamed"``, or ``None`` to leave the loader's choice alone.

    Returns:
        A function that reports, after the run, ``loaded_placement`` (what the loader took, ``None`` if it never chose), ``placement_warning``
        (its step-down or shortage line), ``forced_placement`` and ``loader_free_bytes`` (the memory the job could have when the loader chose: the driver's free memory plus what the process already held on the card).

    Raises:
        ValueError: *forced* is not a placement the loader knows.
    """
    from zvisiongenerator.backends.cuda.image import backend
    from zvisiongenerator.backends.cuda.image.placement import PlacementChoice
    from zvisiongenerator.utils.memory_verdict import RESIDENT, STREAMED, TURNS

    if forced is not None and forced not in (RESIDENT, TURNS, STREAMED):
        raise ValueError(f"Unknown placement {forced!r}; expected resident, turns or streamed.")
    seen: dict[str, Any] = {"loaded_placement": None, "placement_warning": None, "forced_placement": forced, "loader_free_bytes": None}
    real = backend.choose_placement

    def choose(*args: Any, **kwargs: Any) -> PlacementChoice:
        choice = PlacementChoice(forced) if forced is not None else real(*args, **kwargs)
        seen["loaded_placement"], seen["placement_warning"], seen["loader_free_bytes"] = choice.placement, choice.warning, kwargs.get("free_bytes")
        return choice

    backend.choose_placement = choose
    return lambda: dict(seen)


class _StepTrace:
    """An mflux callback that reads the memory counters before the loop and after every denoise step, and the seconds each step took."""

    def __init__(self, readings: list[list[Any]], seconds: list[float]) -> None:
        self._readings, self._seconds, self._step, self._mark = readings, seconds, 0, time.perf_counter()

    def call_before_loop(self, *_: Any, **__: Any) -> None:
        self._readings.append(_reading(0))
        self._mark = time.perf_counter()

    def call_in_loop(self, t: int, seed: Any, prompt: Any, latents: Any, *_: Any, **__: Any) -> None:
        import mlx.core as mx

        mx.eval(latents)  # mflux calls this before it evaluates the step, as the app's own progress callback does
        elapsed = time.perf_counter() - self._mark
        self._step += 1
        self._seconds.append(round(elapsed, 3))
        self._readings.append(_reading(self._step))
        self._mark = time.perf_counter()


def _reading(label: int | str) -> list[Any]:
    """Return ``[label, footprint, active, cache]`` in bytes, the footprint ``None`` where the kernel call fails."""
    import mlx.core as mx

    from bench.memory.measure import current_footprint

    return [label, current_footprint(), int(mx.get_active_memory()), int(mx.get_cache_memory())]


def trace_mlx_job() -> Callable[[], dict[str, Any]]:
    """Sample the process's footprint and MLX's active and cached memory through one Mac image job, in this process only.

    A reading is taken before the denoise loop (step 0) and after each step, from a callback this registers on the model for the duration of
    every generation (wrapping the Mac backend's ``_reporting``, which the CLI's run does not otherwise register a progress callback in), and
    right before and after the first VAE decode, whose result is evaluated before the second reading so the decode's memory is in it. The
    callback evaluates the step's latents first, as the app's own progress callback does, so each reading is of a finished step. A job with
    more than one denoise loop (a refine after a render, an upscale) numbers the steps of each from 0 again; a job with more than one decode records the first.

    Returns:
        A function that reports, after the run, ``trace``: ``[[step, footprint_bytes, active_bytes, cache_bytes], ...]`` with ``step`` an
        integer, or ``"before_decode"`` and ``"after_decode"``, and ``step_seconds``: the time each denoise step took (the callback's
        evaluation of it included).
    """
    import functools

    import mlx.core as mx
    from mflux.models.common.vae.vae_util import VAEUtil

    from zvisiongenerator.backends.mac import image as mac_image

    readings: list[list[Any]] = []
    seconds: list[float] = []
    decoded = {"seen": False}
    real_reporting = mac_image._reporting
    real_decode = VAEUtil.decode

    @contextlib.contextmanager
    def reporting(model: Any, *args: Any, **kwargs: Any) -> Iterator[None]:
        tracer = _StepTrace(readings, seconds)
        model.callbacks.register(tracer)
        try:
            with real_reporting(model, *args, **kwargs):
                yield
        finally:
            mac_image._unregister_callback(model, tracer)

    @functools.wraps(real_decode)
    def decode(*args: Any, **kwargs: Any) -> Any:
        first = not decoded["seen"]
        decoded["seen"] = True
        if first:
            readings.append(_reading("before_decode"))
        result = real_decode(*args, **kwargs)
        if first:
            if isinstance(result, getattr(mx, "array", ())):
                mx.eval(result)
            readings.append(_reading("after_decode"))
        return result

    mac_image._reporting = reporting
    VAEUtil.decode = staticmethod(decode)
    return lambda: {"trace": [list(reading) for reading in readings], "step_seconds": list(seconds)}
