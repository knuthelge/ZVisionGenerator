"""Hint at an NVIDIA driver update when a CUDA build of PyTorch cannot start."""

from __future__ import annotations

# Oldest NVIDIA driver that runs CUDA 13 builds of PyTorch on Windows and Linux.
_CUDA13_MIN_DRIVER = 580


def cuda_driver_hint(available: bool, cuda_version: str | None) -> str | None:
    """Return a sentence asking for a newer NVIDIA driver, or None when the driver is not the likely cause.

    A CUDA 13 build of PyTorch reports CUDA as unavailable on a machine whose driver is older than
    ``_CUDA13_MIN_DRIVER``. The hint stays silent when CUDA works, or when the build is not CUDA 13 or newer.

    Args:
        available: Whether ``torch.cuda.is_available()`` is true.
        cuda_version: ``torch.version.cuda`` (such as ``"13.0"``), or None for a build without CUDA.

    Returns:
        The hint, or None.
    """
    if available or not isinstance(cuda_version, str):
        return None
    try:
        major = int(cuda_version.split(".")[0])
    except ValueError:
        return None
    if major < 13:
        return None
    return f"If this machine has an NVIDIA GPU, update its driver to {_CUDA13_MIN_DRIVER} or newer (this PyTorch build uses CUDA {cuda_version})."
