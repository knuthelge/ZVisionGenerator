"""Tests for zvisiongenerator.backends.cuda.driver."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from zvisiongenerator.backends.cuda.driver import CUDA_INSTALL_COMMAND, cuda_driver_hint, require_cuda


def _torch(available: bool, cuda_version: str | None) -> MagicMock:
    torch = MagicMock()
    torch.cuda.is_available.return_value = available
    torch.version.cuda = cuda_version
    return torch


def test_require_cuda_passes_when_a_cuda_build_sees_a_gpu():
    require_cuda(_torch(True, "13.0"), "The image backend")


def test_require_cuda_names_the_cpu_only_build_and_the_install_command():
    with pytest.raises(RuntimeError) as exc_info:
        require_cuda(_torch(False, None), "The image backend")

    message = str(exc_info.value)
    assert message.startswith("The image backend cannot run: the installed PyTorch has no CUDA.")
    assert CUDA_INSTALL_COMMAND in message
    assert "NVIDIA GPU" not in message


def test_require_cuda_asks_for_a_driver_when_a_cuda_13_build_sees_no_gpu():
    with pytest.raises(RuntimeError) as exc_info:
        require_cuda(_torch(False, "13.0"), "The image backend")

    message = str(exc_info.value)
    assert "requires an NVIDIA GPU" in message
    assert "580" in message
    assert CUDA_INSTALL_COMMAND not in message


@pytest.mark.parametrize("cuda_version", ["13.0", "13.2", "14.0"])
def test_hint_when_cuda_13_or_newer_is_unavailable(cuda_version):
    hint = cuda_driver_hint(False, cuda_version)

    assert hint is not None
    assert "580" in hint
    assert cuda_version in hint


@pytest.mark.parametrize(
    ("available", "cuda_version"),
    [
        (True, "13.0"),
        (False, "12.6"),
        (False, None),
        (False, "x"),
        (False, MagicMock()),
    ],
)
def test_no_hint_when_the_driver_is_not_the_likely_cause(available, cuda_version):
    assert cuda_driver_hint(available, cuda_version) is None
