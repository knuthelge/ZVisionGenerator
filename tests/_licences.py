"""Helper that serves test licence records in place of the packaged ones in ``model_metadata.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests._factories import use_model_metadata

__all__ = ["use_licences"]


def use_licences(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, records: dict[str, dict[str, Any]]) -> None:
    """Make the licence readers see only *records* (record id to record) for this test.

    A record lists the ids of the models it covers under ``models``, which this writes as the ``licences`` of one metadata
    entry per model, in record order; the metadata then holds no other model.
    """
    covered: dict[str, list[str]] = {}
    for record_id, record in records.items():
        for model_id in record["models"]:
            covered.setdefault(model_id, []).append(record_id)
    data = {
        "licences": {record_id: {key: value for key, value in record.items() if key != "models"} for record_id, record in records.items()},
        "models": {model_id: {"kind": "image", "licences": record_ids} for model_id, record_ids in covered.items()},
    }
    use_model_metadata(monkeypatch, tmp_path, data)
