"""Make memory room for the background removal chain by swapping the image model out and back in."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.core.matte import MatteChain

__all__ = ["FOREGROUND_BYTES_PER_PIXEL", "MIN_FREE_BYTES", "MatteRoom", "needs_swap", "required_free_bytes"]

# The cut-out models take about 1.5 to 3 GB beside the image model; below this the image model makes room.
MIN_FREE_BYTES = 2 * 1024**3
# The foreground colour estimate runs at the full output size on the accelerator: about 200 bytes per pixel on top of
# its inputs, so a 4x upscale to 4096x4096 needs several gigabytes more than a 1024x1024 image.
FOREGROUND_BYTES_PER_PIXEL = 256


def required_free_bytes(pixels: int) -> int:
    """Return the free accelerator memory a cut-out of an image with *pixels* pixels needs beside the image model."""
    return MIN_FREE_BYTES + pixels * FOREGROUND_BYTES_PER_PIXEL


def needs_swap(free: int | None, min_free: int) -> bool:
    """Return whether *free* bytes are too few for the chain; unknown free memory never asks for a swap."""
    return free is not None and free < min_free


@dataclass
class MatteRoom:
    """Working state of one job's memory guard for the background removal chain.

    Before the chain's first load, :meth:`make_room` checks the free accelerator memory. When it is short, the job
    swaps: the image model is released before every cut-out, and :meth:`model_for_render` loads it again before the
    next render. Without a *reload* the guard never releases anything.

    Attributes:
        backend: Owns the loaded model and releases it.
        model: The model handle the next render uses; replaced by each reload.
        reload: Loads the image model again, or ``None`` when the caller cannot.
        free_bytes: Returns the free accelerator memory in bytes, or ``None`` when unknown.
        release_memory: Returns freed memory to the system.
        swapping: Whether the guard has tripped; stays on for the rest of the job.
        released: Whether the model is released and has to be loaded again before the next render.
    """

    backend: ImageBackend
    model: Any
    reload: Callable[[], Any] | None
    free_bytes: Callable[[], int | None]
    release_memory: Callable[[], None]
    swapping: bool = False
    released: bool = False

    def make_room(self, pixels: int) -> None:
        """Release the image model when the chain has too little memory, or always once the job swaps.

        Args:
            pixels: The pixel count of the image about to be cut out.
        """
        if self.reload is None:
            return
        if not self.swapping and not needs_swap(self.free_bytes(), required_free_bytes(pixels)):
            return
        self.swapping = True
        self.backend.release_model(self.model)
        self.release_memory()
        self.released = True

    def model_for_render(self, chain: MatteChain, prepare: Callable[[Any], None]) -> Any:
        """Return the model for the next render, loading it again first when it was released.

        The chain is closed before the reload so the two never share the accelerator.

        Args:
            chain: The job's background removal chain.
            prepare: Encodes the job's prompts for a freshly loaded model.
        """
        if self.released and self.reload is not None:
            chain.close()
            self.release_memory()
            self.model = self.reload()
            prepare(self.model)
            self.released = False
        return self.model
