"""Shared helpers for the download, model and registry tests."""

from __future__ import annotations

import copy
import functools
import tempfile
from pathlib import Path
from typing import Any

import pytest


@functools.cache
def _load_packaged_config_once() -> dict[str, Any]:
    """Parse the packaged ``config.yaml`` once per process, with no user override."""
    from zvisiongenerator.utils.config import load_config

    with tempfile.TemporaryDirectory() as empty, pytest.MonkeyPatch.context() as patch:
        data_dir = Path(empty) / ".ziv"
        patch.setattr("zvisiongenerator.utils.config.get_ziv_data_dir", lambda: data_dir)
        patch.setattr("zvisiongenerator.utils.paths.get_ziv_data_dir", lambda: data_dir)
        return load_config()


def packaged_config() -> dict[str, Any]:
    """Return a private copy of the packaged config, so a test may change it freely."""
    return copy.deepcopy(_load_packaged_config_once())
