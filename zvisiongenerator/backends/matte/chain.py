"""The background removal chain: BiRefNet finds the subject, ViTMatte refines its edge, the background colour is removed."""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from zvisiongenerator.backends.matte.birefnet import load_birefnet
from zvisiongenerator.backends.matte.vitmatte import load_vitmatte
from zvisiongenerator.processing import MASK_SIZE, TRIMAP_RADIUS, VITMATTE_MAX, estimate_foreground, merge_alpha, to_rgba, trimap, working_size
from zvisiongenerator.utils.hf_snapshot import snapshot_with
from zvisiongenerator.utils.matte import matte_pins

if TYPE_CHECKING:
    import torch

__all__ = ["TorchMatteChain"]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class TorchMatteChain:
    """Cut a subject out of an image with BiRefNet and ViTMatte on one torch device.

    The models load on the first :meth:`cut_out` and stay resident until :meth:`close`. All the torch code is
    platform-neutral; the caller picks the device and the memory hooks.
    """

    def __init__(self, device: str, *, download: Callable[..., str], release_memory: Callable[[], None], after_cut_out: Callable[[], None]) -> None:
        """Create an unloaded chain.

        Args:
            device: Torch device string the models run on (``"cuda"`` or ``"mps"``).
            download: Hub snapshot downloader, used only for files that are not cached yet.
            release_memory: Called by :meth:`close` to return the freed accelerator memory to the system.
            after_cut_out: Called at the end of every :meth:`cut_out`.
        """
        self._device = device
        self._download = download
        self._release_memory = release_memory
        self._after_cut_out = after_cut_out
        self._birefnet: Any = None
        self._vitmatte: tuple[Any, Any] | None = None

    @property
    def loaded(self) -> bool:
        """Whether both models are resident."""
        return self._birefnet is not None and self._vitmatte is not None

    def cut_out(self, image: Image.Image) -> Image.Image:
        """Return *image* (RGB) as an RGBA image of the same size with the background transparent."""
        import torch

        self._ensure_loaded()
        try:
            width, height = image.size
            rgb_image = image.convert("RGB")
            work = working_size(width, height, longest=VITMATTE_MAX)
            mask = _resize(self._mask(rgb_image), work, Image.Resampling.BILINEAR)
            tri = trimap(mask, radius=TRIMAP_RADIUS)
            refined = self._refine(rgb_image.resize(work, Image.Resampling.BOX) if work != (width, height) else rgb_image, tri)
            alpha = merge_alpha(tri, refined)
            if work != (width, height):
                alpha = merge_alpha(_resize(tri, (width, height), Image.Resampling.NEAREST), _resize(alpha, (width, height), Image.Resampling.BILINEAR))
            rgb = torch.from_numpy(np.asarray(rgb_image, np.float32) / 255.0).to(self._device)
            foreground = estimate_foreground(rgb, torch.from_numpy(alpha).to(self._device)).cpu().numpy()
            return to_rgba(foreground, alpha)
        finally:
            self._after_cut_out()

    def close(self) -> None:
        """Drop both models and return their memory."""
        self._birefnet = None
        self._vitmatte = None
        gc.collect()
        self._release_memory()

    def _ensure_loaded(self) -> None:
        """Load both models from their pinned snapshots, downloading only what is missing."""
        if self.loaded:
            return
        import torch

        birefnet_pin, vitmatte_pin = matte_pins()
        birefnet = load_birefnet(snapshot_with(birefnet_pin, self._download), self._device, torch.float16)
        self._vitmatte = load_vitmatte(snapshot_with(vitmatte_pin, self._download), self._device)
        self._birefnet = birefnet

    def _mask(self, rgb_image: Image.Image) -> np.ndarray:
        """Return BiRefNet's foreground probability as a float32 ``(MASK_SIZE, MASK_SIZE)`` array."""
        import torch

        square = rgb_image.resize((MASK_SIZE, MASK_SIZE), Image.Resampling.BILINEAR)
        normalised = (np.asarray(square, np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        batch = torch.from_numpy(normalised.transpose(2, 0, 1)[None].copy()).to(self._device, torch.float16)
        with torch.no_grad():
            return self._birefnet(batch)[-1].sigmoid().float().cpu().numpy()[0, 0]

    def _refine(self, small_rgb: Image.Image, tri: np.ndarray) -> np.ndarray:
        """Return ViTMatte's alpha for *small_rgb* given its trimap, as a float32 array of the trimap's shape."""
        import torch

        processor, model = self._vitmatte
        height, width = tri.shape
        inputs = processor(images=np.asarray(small_rgb, np.float32) / 255.0, trimaps=tri, do_rescale=False, return_tensors="pt").to(self._device)
        with torch.no_grad():
            alpha: torch.Tensor = model(**inputs).alphas[0, 0]
        return alpha.float().cpu().numpy()[:height, :width]


def _resize(values: np.ndarray, size: tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    """Return the float32 ``(H, W)`` array *values* resized to *size* (width, height)."""
    if (values.shape[1], values.shape[0]) == size:
        return values
    return np.asarray(Image.fromarray(values.astype(np.float32), mode="F").resize(size, resample), dtype=np.float32)
