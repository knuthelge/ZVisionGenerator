"""Check that PyTorch can use an NVIDIA GPU, and hint at a driver update when a CUDA build cannot start."""

from __future__ import annotations

from typing import Any

# Oldest NVIDIA driver that runs CUDA 13 builds of PyTorch on Windows and Linux.
_CUDA13_MIN_DRIVER = 580

# How to get a CUDA build of PyTorch where PyPI's wheel has none (Windows).
CUDA_INSTALL_COMMAND = "uv tool install z-vision-generator --index https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match"


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


def require_cuda(torch: Any, backend: str) -> None:
    """Raise when *torch* cannot use an NVIDIA GPU, naming *backend* and asking for a driver update where that is the likely cause.

    Args:
        torch: The imported ``torch`` module (or a stand-in with ``cuda.is_available()`` and ``version.cuda``).
        backend: What needs the GPU, as the message's subject, such as "The SeedVR2 upscale on Windows and Linux".

    Raises:
        RuntimeError: If CUDA is unavailable or this PyTorch build has no CUDA. A build without CUDA (PyPI's
            Windows wheel) gets the install command that fetches one, since no driver or GPU change would help.
    """
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if torch.cuda.is_available() and cuda_version is not None:
        return
    if cuda_version is None:
        raise RuntimeError(f"{backend} cannot run: the installed PyTorch has no CUDA. Install again with PyTorch's CUDA index: {CUDA_INSTALL_COMMAND}")
    message = f"CUDA is not available. {backend} requires an NVIDIA GPU with CUDA support."
    hint = cuda_driver_hint(False, cuda_version)
    raise RuntimeError(f"{message} {hint}" if hint else message)
