"""Shared licence notice of the CLIs: name a model's terms before its download starts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from zvisiongenerator.utils.licences import licence_notice, load_licences, record_notice

if TYPE_CHECKING:
    from zvisiongenerator.downloads.target import DownloadTarget

__all__ = ["print_licence_notice", "print_record_notice"]


def print_licence_notice(model_id: str, plan: Callable[[], Sequence[DownloadTarget] | None]) -> None:
    """Print the licence notice of *model_id*, one line at a time; nothing when it has no record or is downloaded."""
    for line in licence_notice(load_licences(), model_id, plan):
        print(line, flush=True)


def print_record_notice(record_id: str, plan: Callable[[], Sequence[DownloadTarget] | None]) -> None:
    """Print the terms of the record *record_id*, one line at a time; nothing when no record has that id or its files are downloaded."""
    for line in record_notice(load_licences(), record_id, plan):
        print(line, flush=True)
