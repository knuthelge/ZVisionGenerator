"""The memory bench's case lists: one ``cases/<media>.yaml`` per medium, loaded and expanded into the cases a run executes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CASES_DIR = Path(__file__).with_name("cases")
MEDIA = ("image", "video", "audio")


def load_cases(media: str, directory: Path = CASES_DIR) -> dict[str, Any]:
    """Return the case file of *media*: shared ``defaults``, a ``prompt`` and ``families`` by name, plus what the medium reads besides.

    Raises:
        ValueError: *media* is not image, video or audio.
    """
    if media not in MEDIA:
        raise ValueError(f"Unknown media {media!r}; expected one of {list(MEDIA)}.")
    return yaml.safe_load((directory / f"{media}.yaml").read_text(encoding="utf-8"))


def expand_cases(cases: Mapping[str, Any], *, media: str, platform: str, families: list[str] | None = None, ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Return the cases to run on *platform*, each with the shared settings filled in and its ``media`` and ``family`` set.

    Args:
        cases: The loaded case file.
        media: The medium the file is for; it becomes each case's ``media``.
        platform: ``"cuda"`` or ``"mac"``; a case with ``only`` that leaves it out is dropped.
        families: Family names to keep; ``None`` keeps all.
        ids: Case ids to keep; ``None`` keeps all.

    Returns:
        The cases in file order.

    Raises:
        ValueError: A requested family or id is not in the file.
    """
    known = cases["families"]
    unknown = [name for name in families or [] if name not in known]
    if unknown:
        raise ValueError(f"Unknown family {unknown}; the case file has {sorted(known)}.")
    expanded = []
    for family, entries in known.items():
        if families and family not in families:
            continue
        for entry in entries:
            if platform not in entry.get("only", [platform]):
                continue
            case = dict(cases.get("defaults", {})) | entry | {"family": family, "media": media}
            case["prompt"] = entry.get("prompt") or (cases["edit_prompt"] if case.get("edit") else cases["prompt"])
            expanded.append(case)
    missing = [name for name in ids or [] if name not in {case["id"] for case in expanded}]
    if missing:
        raise ValueError(f"No case {missing} for {platform}; the ids are {[case['id'] for case in expanded]}.")
    return [case for case in expanded if not ids or case["id"] in ids]
