"""The suite runs with the GPU hidden, so no test can open a CUDA context or take GPU memory."""

from __future__ import annotations


import pytest


def test_torch_sees_no_cuda_device() -> None:
    torch = pytest.importorskip("torch")

    assert not torch.cuda.is_available()
    assert torch.cuda.device_count() == 0
