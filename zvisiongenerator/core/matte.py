"""MatteChain Protocol — contract for the models that cut a subject out of an image."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL import Image


@runtime_checkable
class MatteChain(Protocol):
    """Background removal for one job: a segmentation model followed by a matting model.

    Required interface:
    - loaded: whether the models are resident on the accelerator
    - cut_out(): return an image with the subject kept and the background transparent
    - close(): drop the models and return their memory
    """

    @property
    def loaded(self) -> bool:
        """Whether the models are loaded (they load on the first ``cut_out``)."""
        ...

    def cut_out(self, image: Image.Image) -> Image.Image:
        """Return *image* (RGB) as an RGBA image of the same size with the background transparent."""
        ...

    def close(self) -> None:
        """Drop the models and free their memory; a later ``cut_out`` loads them again."""
        ...
