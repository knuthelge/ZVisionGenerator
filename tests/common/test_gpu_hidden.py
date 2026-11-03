"""The suite runs with the GPU hidden, so no test can open a CUDA context or take GPU memory."""

from __future__ import annotations

import os

import pytest


def test_the_gpu_is_hidden_from_the_environment() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_torch_sees_no_cuda_device() -> None:
    torch = pytest.importorskip("torch")

    assert not torch.cuda.is_available()
    assert torch.cuda.device_count() == 0
