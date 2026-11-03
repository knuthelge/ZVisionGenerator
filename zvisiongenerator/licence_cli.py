"""Shared licence notice of the image and video CLIs: name a model's terms before its download starts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from zvisiongenerator.utils.licences import LicenceRecord, licence_notice, licences_for, load_licences

if TYPE_CHECKING:
    from zvisiongenerator.downloads.target import DownloadTarget

__all__ = ["licence_lines", "print_licence_notice"]


def licence_lines(
    records: Sequence[LicenceRecord],
    model_id: str,
    plan: Callable[[], Sequence[DownloadTarget] | None],
    *,
    present: Callable[[DownloadTarget], bool],
) -> tuple[str, ...]:
    """Return the notice lines for *model_id*, planning its downloads only when it has a licence record.

    Args:
        records: The licence records.
        model_id: The model the user picked (the alias, not the resolved path), or ``matte``.
        plan: Plans the model's downloads the way the CLI's download step would; ``None`` means they cannot be planned offline
            and counts as missing. Not called for a model without a record, so such a model costs no Hub call.
        present: Whether one target is already on disk.

    Returns:
        The lines of :func:`~zvisiongenerator.utils.licences.licence_notice`; empty when the model has no record or is already downloaded.
    """
    if not licences_for(records, model_id):
        return ()
    return licence_notice(records, model_id, plan(), present)


def print_licence_notice(
    model_id: str,
    plan: Callable[[], Sequence[DownloadTarget] | None],
    *,
    present: Callable[[DownloadTarget], bool],
) -> None:
    """Print the notice lines of :func:`licence_lines` for the packaged records, one per line, and nothing when there are none."""
    for line in licence_lines(load_licences(), model_id, plan, present=present):
        print(line, flush=True)
