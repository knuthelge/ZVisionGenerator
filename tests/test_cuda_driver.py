"""Tests for zvisiongenerator.backends.cuda_driver."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from zvisiongenerator.backends.cuda_driver import cuda_driver_hint


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
