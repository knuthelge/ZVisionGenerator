"""Licence records: the terms a model comes with, read from the config's ``licences`` section.

Each record names a licence and the models it covers. Every URL is pinned to a revision, so the text a user reads is the
text of the weights the app downloads. The functions here are pure: they read a config ``dict`` and a presence check
passed in by the caller, and do no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlparse

if TYPE_CHECKING:
    from zvisiongenerator.downloads.target import DownloadTarget

__all__ = ["MATTE_LICENCE_MODEL", "NOTICE_HEADER", "LicenceRecord", "licence_notice", "licence_payload", "licence_text", "licences_for", "parse_licences"]

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
        id: The record's key in the ``licences`` section.
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


def parse_licences(config: Mapping[str, Any]) -> tuple[LicenceRecord, ...]:
    """Parse and validate the ``licences`` section of *config*, in the order it is written.

    Args:
        config: The loaded config. A missing ``licences`` section means no records.

    Returns:
        One frozen record per entry.

    Raises:
        ValueError: If the section or a record is malformed, a URL has no revision, ``models`` is empty or names an
            unknown model. The model ids that exist are the ``model_aliases`` and ``audio_model_presets`` keys and
            ``matte``.
    """
    section = config.get("licences")
    if section is None:
        return ()
    if not isinstance(section, Mapping):
        raise ValueError(f"config 'licences' must be a mapping of record id to record, got {type(section).__name__}.")
    known = _known_models(config)
    return tuple(_parse_record(str(record_id), raw, known) for record_id, raw in section.items())


def licences_for(config: Mapping[str, Any], model_id: str) -> tuple[LicenceRecord, ...]:
    """Return the records that cover *model_id*, in config order; empty when it has none."""
    return tuple(record for record in parse_licences(config) if model_id in record.models)


def licence_payload(config: Mapping[str, Any], model_id: str) -> list[dict[str, Any]]:
    """Return the records that cover *model_id* as JSON-ready dicts (``name``, ``supplier``, ``summary``, ``url``, ``url_label``, ``notice_url``), in config order."""
    return [{"name": r.name, "supplier": r.supplier, "summary": r.summary, "url": r.url, "url_label": r.url_label, "notice_url": r.notice_url} for r in licences_for(config, model_id)]


def licence_text(record: LicenceRecord) -> str:
    """Format *record* as ``<name> (<supplier>): <summary>``."""
    return f"{record.name} ({record.supplier}): {record.summary}"


def licence_notice(
    config: Mapping[str, Any],
    model_id: str,
    targets: Sequence[DownloadTarget] | None,
    present: Callable[[DownloadTarget], bool],
) -> tuple[str, ...]:
    """Return the lines that name the terms of *model_id* when downloading it is about to start.

    Args:
        config: The loaded config.
        model_id: The model the user picked.
        targets: The model's downloads, or ``None`` when they cannot be planned offline (counted as missing).
        present: Whether one target is already on disk. Not called when the model has no record.

    Returns:
        The header, then per record its text, ``"  <url>"`` and, if it has one, ``"  NOTICE: <url>"``. Empty when the
        model has no record or every target is present.
    """
    records = licences_for(config, model_id)
    if not records:
        return ()
    if targets is not None and all(present(target) for target in targets):
        return ()
    lines = [NOTICE_HEADER]
    for record in records:
        lines += [licence_text(record), f"  {record.url}"]
        if record.notice_url is not None:
            lines.append(f"  NOTICE: {record.notice_url}")
    return tuple(lines)


def _known_models(config: Mapping[str, Any]) -> frozenset[str]:
    """Return every model id a record may name: the aliases, the audio presets (when the config has them) and ``matte``."""
    aliases = config.get("model_aliases")
    audio = config.get("audio_model_presets")
    ids = {MATTE_LICENCE_MODEL}
    for section in (aliases, audio):
        if isinstance(section, Mapping):
            ids.update(str(key) for key in section)
    return frozenset(ids)


def _parse_record(record_id: str, raw: object, known: frozenset[str]) -> LicenceRecord:
    """Validate one record and return it."""
    where = f"config 'licences.{record_id}'"
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
    return LicenceRecord(id=record_id, models=_models(raw["models"], known, f"{where}.models"), notice_url=notice_url, url_label=url_label, **text)


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


def _models(value: object, known: frozenset[str], where: str) -> tuple[str, ...]:
    """Return the model ids in *value*, which must be a non-empty list of known ids."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of model ids.")
    for model in value:
        if not isinstance(model, str) or model not in known:
            raise ValueError(f"{where} names an unknown model {model!r}. Known: a model alias, an audio model preset or '{MATTE_LICENCE_MODEL}'.")
    return tuple(value)
