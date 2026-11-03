"""Helper that serves test licence records in place of the packaged ``licences.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from zvisiongenerator.utils import licences as licences_module

__all__ = ["use_licences"]


def use_licences(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, records: dict[str, dict[str, Any]]) -> None:
    """Make ``load_licences`` read *records* (record id to record) instead of the packaged file for this test."""
    path = tmp_path / "licences.yaml"
    path.write_text(yaml.safe_dump(records), encoding="utf-8")
    monkeypatch.setattr(licences_module, "DATA_FILE", path)
