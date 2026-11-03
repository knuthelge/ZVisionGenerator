"""Snapshot and restore the process-wide PyTorch numerics flags that audio model code changes, and set the default dtype for a block.

The vendored audio kits switch these flags when a model is built (deterministic cuDNN, TF32 off, the strictest float32
matmul precision). They are global, so a backend that is closed would leave every later image and video job slower.
The audio runner takes a snapshot before it loads a backend and restores it when the backend is closed.

PyTorch keeps the TF32 and matmul-precision settings in per-backend ``fp32_precision`` values that the older
``allow_tf32`` flags and ``set_float32_matmul_precision`` also write. Code that mixes the two leaves them out of step,
so ``torch.get_float32_matmul_precision()`` raises; the snapshot holds every ``fp32_precision`` value as well, and they
are restored last, so a mixed state comes back exactly.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from operator import attrgetter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = ["TorchFlags", "capture_torch_flags", "default_dtype", "preserved_torch_flags", "restore_torch_flags"]

TorchFlags = dict[str, Any]
"""A snapshot: each flag's path under ``torch.backends`` to its value, plus ``float32_matmul_precision``."""

_FLAGS = (
    "cudnn.deterministic",
    "cudnn.benchmark",
    "cuda.matmul.allow_tf32",
    "cudnn.allow_tf32",
    "cuda.matmul.allow_fp16_reduced_precision_reduction",
)
_PRECISIONS = (
    "cudnn.fp32_precision",
    "cudnn.conv.fp32_precision",
    "cudnn.rnn.fp32_precision",
    "cuda.matmul.fp32_precision",
    "mkldnn.matmul.fp32_precision",
)
"""Restored after the coarse flags and the matmul precision, so these per-backend values are what stays."""


def capture_torch_flags() -> TorchFlags:
    """Return the current values of the flags."""
    import torch

    flags = {path: attrgetter(path)(torch.backends) for path in (*_FLAGS, *_PRECISIONS)}
    try:
        flags["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except RuntimeError:  # the legacy and the new settings disagree
        flags["float32_matmul_precision"] = None
    return flags


def restore_torch_flags(flags: TorchFlags) -> None:
    """Set the global flags back to the values in *flags*: the coarse settings first, the per-backend precisions last."""
    import torch

    for path in _FLAGS:
        _set(torch.backends, path, flags[path])
    if flags["float32_matmul_precision"] is not None:
        torch.set_float32_matmul_precision(flags["float32_matmul_precision"])
    for path in _PRECISIONS:
        _set(torch.backends, path, flags[path])


@contextmanager
def preserved_torch_flags() -> Iterator[None]:
    """Give the global flags back as they were on entry, whatever the block (a model build, a take) changed and however it ends."""
    flags = capture_torch_flags()
    try:
        yield
    finally:
        restore_torch_flags(flags)


@contextmanager
def default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Make new floating-point parameters *dtype* for the length of the block, then restore the previous default dtype."""
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _set(root: Any, path: str, value: Any) -> None:
    """Set the attribute at dotted *path* under *root*."""
    parent, _, name = path.rpartition(".")
    setattr(attrgetter(parent)(root), name, value)
