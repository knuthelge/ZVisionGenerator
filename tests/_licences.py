"""Helper that serves test licence records in place of the packaged ones in ``model_metadata.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests._factories import packaged_metadata_copy, use_model_metadata

__all__ = ["use_licences"]


def use_licences(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, records: dict[str, dict[str, Any]], *, models: dict[str, dict[str, Any]] | None = None) -> None:
    """Make the licence readers see only *records* (record id to record) for this test.

    A record lists the ids of the models it covers under ``models``, which this writes as the ``licences`` of one metadata
    entry per model (an image model downloading ``owner/<id>``), in record order; the metadata then holds no other model. A model in *models* (id to metadata entry)
    is kept as that entry, with its ``licences`` set to the records that cover it. The video families stay as packaged: they carry
    no licences, and readers of the Models page need them.
    """
    covered: dict[str, list[str]] = {}
    for record_id, record in records.items():
        for model_id in record["models"]:
            covered.setdefault(model_id, []).append(record_id)
    data = {
        "licences": {record_id: {key: value for key, value in record.items() if key != "models"} for record_id, record in records.items()},
        "models": {model_id: {"kind": "image", "licences": record_ids, "downloads": [{"ref": f"owner/{model_id}"}]} for model_id, record_ids in covered.items()},
        "video_families": packaged_metadata_copy()["video_families"],
    }
    for model_id, entry in (models or {}).items():
        data["models"][model_id] = {**entry, "licences": covered.get(model_id, [])}
    use_model_metadata(monkeypatch, tmp_path, data)
