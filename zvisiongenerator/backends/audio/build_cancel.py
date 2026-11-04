"""Let a Stop reach the middle of building a model: every layer the library creates is a place to check."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from zvisiongenerator.core.audio_backend import raise_if_cancelled

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck

__all__ = ["stop_while_building"]


@contextmanager
def stop_while_building(cancelled: CancelCheck | None) -> Iterator[None]:
    """Check *cancelled* each time a module or parameter is registered in the calling thread while the block runs.

    A library builds a model in one call that cannot be interrupted from outside. PyTorch calls a global hook for every
    layer and tensor the call creates, so a Stop is seen within the time one layer takes to build. Work that creates
    no layers (reading a file, copying to the GPU) is not covered; the caller checks before and after it.

    Raises:
        AudioCancelled: From inside the block, once *cancelled* returns ``True``.
    """
    if cancelled is None:
        yield
        return
    from torch.nn.modules.module import register_module_module_registration_hook, register_module_parameter_registration_hook

    owner = threading.get_ident()

    def check(*_: Any) -> None:
        # The hooks are process-wide: only the thread that loads is stopped, never another thread building a module.
        if threading.get_ident() == owner:
            raise_if_cancelled(cancelled)

    handles = (register_module_module_registration_hook(check), register_module_parameter_registration_hook(check))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
