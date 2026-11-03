"""Licence records: the terms a model comes with, read from the packaged ``licences.yaml``.

Each record names a licence and the models it covers. Every URL is pinned to a revision, so the text a user reads is the
text of the weights the app downloads. The records are fixed facts about the models, so they live in their own file,
read directly and never merged with the user's config. :func:`load_licences` is the one reader (once, cached); every
other function is pure: it takes the records, the known model ids or a presence check from its caller and does no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlparse

import yaml

if TYPE_CHECKING:
    from zvisiongenerator.downloads.target import DownloadTarget

__all__ = [
    "DATA_FILE",
    "MATTE_LICENCE_MODEL",
    "NOTICE_HEADER",
    "LicenceRecord",
    "check_licence_models",
    "known_model_ids",
    "licence_notice",
    "licence_payload",
    "licence_text",
    "licences_for",
    "load_licences",
    "parse_licences",
]

DATA_FILE: Final = Path(__file__).with_name("licences.yaml")
"""The packaged records: a mapping of record id to record."""

MATTE_LICENCE_MODEL: Final = "matte"
"""The model id of the background removal models, which are no alias but have a licence record."""

NOTICE_HEADER: Final = "By downloading you agree to these terms:"
"""The first line of a licence notice."""

DEFAULT_URL_LABEL: Final = "Licence"

_REQUIRED_TEXT_KEYS: Final = ("name", "supplier", "summary", "url")
_PINNED_PATH: Final = re.compile(r"(?:^|[/@])[0-9a-f]{40}(?:[/?#]|$)")
"""A 40-hex commit as a whole path segment (Hugging Face and GitHub ``/<sha>/``) or after ``@`` (jsDelivr ``repo@<sha>``)."""


@dataclass(frozen=True)
class LicenceRecord:
    """One licence and the models it covers.

    Attributes:
        id: The record's key in ``licences.yaml``.
        name: The licence's name, such as ``FLUX Non-Commercial License v2.1``.
        supplier: Who licenses the model.
        summary: The terms in a short plain-language phrase.
        url: The licence text (or model card), pinned to a revision.
        models: The model ids the licence covers: aliases, audio presets or ``matte``.
        notice_url: The NOTICE file, pinned to a revision, or ``None``.
        url_label: What the link to *url* is called.
    """

    id: str
    name: str
    supplier: str
    summary: str
    url: str
    models: tuple[str, ...]
    notice_url: str | None = None
    url_label: str = DEFAULT_URL_LABEL


def load_licences() -> tuple[LicenceRecord, ...]:
    """Read and parse the packaged ``licences.yaml``, in the order it is written; the file is read once.

    Returns:
        One frozen record per entry. Whether the models they name exist is checked by :func:`check_licence_models`.

    Raises:
        FileNotFoundError: If the packaged file is missing.
        ValueError: If the file is not valid YAML or a record is malformed (see :func:`parse_licences`).
    """
    return _load(DATA_FILE)


@cache
def _load(path: Path) -> tuple[LicenceRecord, ...]:
    """Read and parse the records in *path*; cached per path."""
    with open(path, encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse {path.name}: {e}") from e
    return parse_licences(raw)


def parse_licences(raw: object) -> tuple[LicenceRecord, ...]:
    """Validate the parsed *raw* records and return them in the order they are written.

    Args:
        raw: The parsed YAML: a mapping of record id to record. ``None`` (an empty file) means no records.

    Returns:
        One frozen record per entry.

    Raises:
        ValueError: If *raw* or a record is malformed, a URL has no revision or ``models`` is empty or not a list of
            strings.
    """
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise ValueError(f"Licence records must be a mapping of record id to record, got {type(raw).__name__}.")
    return tuple(_parse_record(str(record_id), record) for record_id, record in raw.items())


def known_model_ids(config: Mapping[str, Any]) -> frozenset[str]:
    """Return every model id a record may name: the ``model_aliases`` and ``audio_model_presets`` keys of *config* and ``matte``."""
    ids = {MATTE_LICENCE_MODEL}
    for key in ("model_aliases", "audio_model_presets"):
        section = config.get(key)
        if isinstance(section, Mapping):
            ids.update(str(model) for model in section)
    return frozenset(ids)


def check_licence_models(records: Sequence[LicenceRecord], known_models: Collection[str]) -> None:
    """Raise ``ValueError`` if a record names a model that is not in *known_models*."""
    for record in records:
        for model in record.models:
            if model not in known_models:
                raise ValueError(f"Licence record '{record.id}' names an unknown model {model!r}. Known: a model alias, an audio model preset or '{MATTE_LICENCE_MODEL}'.")


def licences_for(records: Sequence[LicenceRecord], model_id: str) -> tuple[LicenceRecord, ...]:
    """Return the *records* that cover *model_id*, in the order given; empty when it has none."""
    return tuple(record for record in records if model_id in record.models)


def licence_payload(records: Sequence[LicenceRecord], model_id: str) -> list[dict[str, Any]]:
    """Return the *records* that cover *model_id* as JSON-ready dicts (``name``, ``supplier``, ``summary``, ``url``, ``url_label``, ``notice_url``), in the order given."""
    return [{"name": r.name, "supplier": r.supplier, "summary": r.summary, "url": r.url, "url_label": r.url_label, "notice_url": r.notice_url} for r in licences_for(records, model_id)]


def licence_text(record: LicenceRecord) -> str:
    """Format *record* as ``<name> (<supplier>): <summary>``."""
    return f"{record.name} ({record.supplier}): {record.summary}"


def licence_notice(
    records: Sequence[LicenceRecord],
    model_id: str,
    targets: Sequence[DownloadTarget] | None,
    present: Callable[[DownloadTarget], bool],
) -> tuple[str, ...]:
    """Return the lines that name the terms of *model_id* when downloading it is about to start.

    Args:
        records: The licence records.
        model_id: The model the user picked.
        targets: The model's downloads, or ``None`` when they cannot be planned offline (counted as missing).
        present: Whether one target is already on disk. Not called when the model has no record.

    Returns:
        The header, then per record its text, ``"  <url>"`` and, if it has one, ``"  NOTICE: <url>"``. Empty when the
        model has no record or every target is present.
    """
    covering = licences_for(records, model_id)
    if not covering:
        return ()
    if targets is not None and all(present(target) for target in targets):
        return ()
    lines = [NOTICE_HEADER]
    for record in covering:
        lines += [licence_text(record), f"  {record.url}"]
        if record.notice_url is not None:
            lines.append(f"  NOTICE: {record.notice_url}")
    return tuple(lines)


def _parse_record(record_id: str, raw: object) -> LicenceRecord:
    """Validate one record and return it."""
    where = f"licence record '{record_id}'"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(raw).__name__}.")
    for key in (*_REQUIRED_TEXT_KEYS, "models"):
        if key not in raw:
            raise ValueError(f"{where} is missing the required key '{key}'.")
    text = {key: _text(raw[key], f"{where}.{key}") for key in _REQUIRED_TEXT_KEYS}
    _require_pinned(text["url"], f"{where}.url")
    notice_url = raw.get("notice_url")
    if notice_url is not None:
        notice_url = _text(notice_url, f"{where}.notice_url")
        _require_pinned(notice_url, f"{where}.notice_url")
    url_label = raw.get("url_label")
    url_label = DEFAULT_URL_LABEL if url_label is None else _text(url_label, f"{where}.url_label")
    return LicenceRecord(id=record_id, models=_models(raw["models"], f"{where}.models"), notice_url=notice_url, url_label=url_label, **text)


def _text(value: object, where: str) -> str:
    """Return *value* when it is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string.")
    return value.strip()


def _require_pinned(url: str, where: str) -> None:
    """Raise ``ValueError`` unless *url* is an https link whose path carries a 40-hex commit."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{where} must be an https URL, got {url!r}.")
    if not _PINNED_PATH.search(parsed.path):
        raise ValueError(f"{where} must be pinned to a revision (a 40-character commit in the path), got {url!r}.")


def _models(value: object, where: str) -> tuple[str, ...]:
    """Return the model ids in *value*, which must be a non-empty list of strings."""
    if not isinstance(value, list) or not value or not all(isinstance(model, str) for model in value):
        raise ValueError(f"{where} must be a non-empty list of model ids.")
    return tuple(value)
