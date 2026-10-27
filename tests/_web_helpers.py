"""Shared helpers for the Web UI test modules."""

from __future__ import annotations

import copy
import os
import tempfile
from functools import cache
from typing import Any
from unittest import mock

from zvisiongenerator.utils.config import load_config


@cache
def _packaged_config_once() -> dict[str, Any]:
    """Parse the packaged config once, with an empty data dir so no user override applies."""
    with tempfile.TemporaryDirectory() as data_dir, mock.patch.dict(os.environ, {"ZIV_DATA_DIR": data_dir}):
        return load_config()


def packaged_config() -> dict[str, Any]:
    """Return a private deep copy of the packaged config (parsed once per process)."""
    return copy.deepcopy(_packaged_config_once())
