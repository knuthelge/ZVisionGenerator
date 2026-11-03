"""Licence records: the terms a model comes with, read from the packaged ``licences.yaml``.

The records are fixed facts about the models, so they live in their own file, read directly and never merged with the
user's config. A contract test checks that every URL is a pinned file or a model page and every record names a known model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import yaml

if TYPE_CHECKING:
    from zvisiongenerator.downloads.target import DownloadTarget

__all__ = ["DATA_FILE", "MATTE_LICENCE_MODEL", "NOTICE_HEADER", "LicenceRecord", "licence_notice", "licence_payload", "licence_text", "licences_for", "load_licences"]

DATA_FILE: Final = Path(__file__).with_name("licences.yaml")
"""The packaged records: a mapping of record id to record."""

MATTE_LICENCE_MODEL: Final = "matte"
"""The model id of the background removal models, which are no alias but have a licence record."""

NOTICE_HEADER: Final = "By downloading you agree to these terms:"
"""The first line of a licence notice."""


@dataclass(frozen=True)
class LicenceRecord:
    """One licence and the models it covers (aliases, audio presets or ``matte``); URLs are pinned to a revision or are a model page."""

    id: str
    name: str
    supplier: str
    summary: str
    url: str
    models: tuple[str, ...]
    url_label: str = "Licence"
    attribution: str | None = None


def load_licences() -> tuple[LicenceRecord, ...]:
    """Return the packaged records in the order they are written; the file is read once."""
    return _load(DATA_FILE)


@cache
def _load(path: Path) -> tuple[LicenceRecord, ...]:
    """Read the records in *path*; cached per path."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return tuple(
        LicenceRecord(
            id=str(record_id),
            name=record["name"],
            supplier=record["supplier"],
            summary=record["summary"],
            url=record["url"],
            models=tuple(record["models"]),
            url_label=record.get("url_label", "Licence"),
            attribution=record.get("attribution"),
        )
        for record_id, record in raw.items()
    )


def licences_for(records: Sequence[LicenceRecord], model_id: str) -> tuple[LicenceRecord, ...]:
    """Return the *records* that cover *model_id*, in the order given; empty when it has none."""
    return tuple(record for record in records if model_id in record.models)


def licence_payload(records: Sequence[LicenceRecord], model_id: str) -> list[dict[str, Any]]:
    """Return the *records* that cover *model_id* as JSON-ready dicts, in the order given."""
    return [{"name": r.name, "supplier": r.supplier, "summary": r.summary, "url": r.url, "url_label": r.url_label, "attribution": r.attribution} for r in licences_for(records, model_id)]


def licence_text(record: LicenceRecord) -> str:
    """Format *record* as ``<name> (<supplier>): <summary>``."""
    return f"{record.name} ({record.supplier}): {record.summary}"


def licence_notice(
    records: Sequence[LicenceRecord],
    model_id: str,
    plan: Callable[[], Sequence[DownloadTarget] | None],
    present: Callable[[DownloadTarget], bool] | None = None,
) -> tuple[str, ...]:
    """Return the lines that name the terms of *model_id* before its download starts.

    Args:
        records: The licence records.
        model_id: The model the user picked (the alias, not the resolved path), or ``matte``.
        plan: Plans the model's downloads; ``None`` means they cannot be planned offline and counts as missing. Called
            only when the model has a record, so a model without one costs no Hub call.
        present: Whether one target is already on disk; defaults to the offline presence check.

    Returns:
        The header, then per record its text, ``"  <url>"`` and, if it has one, ``"  <attribution>"``. Empty when the
        model has no record or every target is present.
    """
    covering = licences_for(records, model_id)
    if not covering:
        return ()
    targets = plan()
    if present is None:
        from zvisiongenerator.downloads.presence import target_present_offline

        present = target_present_offline
    if targets is not None and all(present(target) for target in targets):
        return ()
    lines = [NOTICE_HEADER]
    for record in covering:
        lines += [licence_text(record), f"  {record.url}"]
        if record.attribution is not None:
            lines.append(f"  {record.attribution}")
    return tuple(lines)
