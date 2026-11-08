"""Licence records: the terms a model comes with, read from the packaged ``model_metadata.yaml``.

The records are fixed facts about the models, read directly and never merged with the user's config. Each model entry of the
metadata names the records that cover it. A contract test checks that every URL is a pinned file or a model page and that
every record is used and every id an entry names exists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from zvisiongenerator.utils import model_metadata

if TYPE_CHECKING:
    from zvisiongenerator.downloads.target import DownloadTarget

__all__ = [
    "MATTE_LICENCE_MODEL",
    "NOTICE_FOOTER",
    "NOTICE_HEADER",
    "LicenceRecord",
    "licence_notice",
    "licence_payload",
    "record_notice",
    "licence_text",
    "licences_for",
    "load_licences",
]

MATTE_LICENCE_MODEL: Final = "matte"
"""The model id of the background removal models, which are no alias but have a licence record."""

NOTICE_HEADER: Final = "By downloading you agree to these terms:"
"""The first line of a licence notice."""

NOTICE_FOOTER: Final = "These are short summaries; the linked licence is what applies."
"""The last line of a licence notice."""


@dataclass(frozen=True)
class LicenceRecord:
    """One licence; URLs are pinned to a revision or are a model page. The models it covers name it in their metadata entry."""

    id: str
    name: str
    supplier: str
    summary: str
    url: str
    url_label: str = "Licence"
    attribution: str | None = None


def load_licences() -> tuple[LicenceRecord, ...]:
    """Return the packaged records in the order they are written; each metadata file is read once."""
    return _records(model_metadata.METADATA_FILE)


@cache
def _records(path: Path) -> tuple[LicenceRecord, ...]:
    """Build the records of the metadata file *path*; cached per path."""
    raw = model_metadata.load_model_metadata().get("licences") or {}
    return tuple(
        LicenceRecord(
            id=str(record_id),
            name=record["name"],
            supplier=record["supplier"],
            summary=record["summary"],
            url=record["url"],
            url_label=record.get("url_label", "Licence"),
            attribution=record.get("attribution"),
        )
        for record_id, record in raw.items()
    )


def licences_for(records: Sequence[LicenceRecord], model_id: str) -> tuple[LicenceRecord, ...]:
    """Return the *records* that the metadata entry of *model_id* names, in the entry's order; empty for an unknown model."""
    entry = model_metadata.model_entry(model_id)
    if entry is None:
        return ()
    by_id = {record.id: record for record in records}
    return tuple(by_id[record_id] for record_id in entry["licences"])


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
    return _notice_lines(licences_for(records, model_id), plan, present)


def record_notice(
    records: Sequence[LicenceRecord],
    record_id: str,
    plan: Callable[[], Sequence[DownloadTarget] | None],
    present: Callable[[DownloadTarget], bool] | None = None,
) -> tuple[str, ...]:
    """Return the lines that name the terms of the record *record_id* before the files *plan* lists download.

    For a download that is no model of its own, such as the LoRA a checkpoint conversion merges, so only the record that covers
    it is named and not every record of the model it belongs to. Same lines and rules as :func:`licence_notice`: empty when
    no record has that id or every target is present.
    """
    return _notice_lines(tuple(record for record in records if record.id == record_id), plan, present)


def _notice_lines(covering: Sequence[LicenceRecord], plan: Callable[[], Sequence[DownloadTarget] | None], present: Callable[[DownloadTarget], bool] | None) -> tuple[str, ...]:
    """Return the notice for the *covering* records; nothing when there are none (the plan is not called) or every target is present."""
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
    lines.append(NOTICE_FOOTER)
    return tuple(lines)
