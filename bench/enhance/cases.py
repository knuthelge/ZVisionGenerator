"""Load and validate the bench cases from YAML."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from zvisiongenerator.utils.prompt_enhance import EnhanceSettings, settings_from_mapping, validate_settings

CASES_PATH = Path(__file__).with_name("cases.yaml")
CASE_MODES = ("image", "video", "caption")
CONTEXT_KEYS = frozenset({"ratio", "width", "height", "frames", "fps", "audio", "reference_image", "extends", "previous", "extract_from", "new_frames", "prompt_is_original"})
# Seeds are at least eight digits and stable per case id, so reordering or adding cases never changes another case's seeds.
_SEED_FLOOR = 10_000_000
_SEED_SPAN = 900_000_000
_CASE_KEYS = frozenset({"id", "tags", "mode", "prompt", "scene", "settings", "context", "family", "keep", "count", "forbid"})


@dataclass(frozen=True)
class Case:
    """One bench case: the input, its enhance settings and job context, and what the output must keep or avoid."""

    id: str
    tags: tuple[str, ...]
    mode: str
    prompt: str
    scene: str
    settings: EnhanceSettings
    raw_settings: dict[str, Any]
    context: dict[str, Any]
    family: str | None
    keep: tuple[str, ...]
    count: str | None
    forbid: tuple[str, ...]

    @property
    def media(self) -> str:
        """Return the enhancer mode the case runs in: ``video`` or ``image`` (captions are images)."""
        return "video" if self.mode == "video" else "image"


def case_seeds(case_id: str, count: int) -> tuple[int, ...]:
    """Return *count* consecutive seeds for *case_id*, starting at a stable eight- or nine-digit base."""
    digest = int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:8], "big")
    base = _SEED_FLOOR + digest % _SEED_SPAN
    return tuple(base + offset for offset in range(count))


def load_cases(path: Path = CASES_PATH) -> tuple[Case, ...]:
    """Load every case from *path*.

    Raises:
        ValueError: Naming the case when one is malformed or two share an id.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("cases")
    if not isinstance(entries, list):
        raise ValueError(f"{path.name}: expected a top-level 'cases' list.")
    cases = tuple(parse_case(entry) for entry in entries)
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"Duplicate case id '{case.id}'.")
        seen.add(case.id)
    return cases


def parse_case(entry: Any) -> Case:
    """Build one case from its YAML mapping.

    Raises:
        ValueError: For unknown keys, a bad mode, invalid settings or context, or no text to enhance.
    """
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
        raise ValueError(f"Every case needs a mapping with a string 'id', got {entry!r}.")
    case_id = entry["id"]
    unknown = sorted(set(entry) - _CASE_KEYS)
    if unknown:
        raise ValueError(f"Case '{case_id}': unknown key(s) {', '.join(unknown)}.")
    mode = entry.get("mode", "image")
    if mode not in CASE_MODES:
        raise ValueError(f"Case '{case_id}': mode must be one of {', '.join(CASE_MODES)}.")
    context = entry.get("context") or {}
    if not isinstance(context, dict) or set(context) - CONTEXT_KEYS:
        raise ValueError(f"Case '{case_id}': context keys must be among {', '.join(sorted(CONTEXT_KEYS))}.")
    prompt = str(entry.get("prompt") or "")
    scene = str(entry.get("scene") or "")
    if mode != "video" and scene:
        raise ValueError(f"Case '{case_id}': only video cases take a scene.")
    if not (prompt.strip() or scene.strip() or (mode == "video" and context.get("extends"))):
        raise ValueError(f"Case '{case_id}': needs a prompt or a scene.")
    raw_settings = dict(entry.get("settings") or {})
    media = "video" if mode == "video" else "image"
    try:
        settings = settings_from_mapping(raw_settings, mode=media)
        validate_settings(settings, mode=media)
    except ValueError as exc:
        raise ValueError(f"Case '{case_id}': {exc}") from exc
    return Case(
        id=case_id,
        tags=tuple(str(tag) for tag in entry.get("tags") or ()),
        mode=mode,
        prompt=prompt,
        scene=scene,
        settings=settings,
        raw_settings=raw_settings,
        context=dict(context),
        family=entry.get("family"),
        keep=tuple(str(item) for item in entry.get("keep") or ()),
        count=entry.get("count"),
        forbid=tuple(str(item) for item in entry.get("forbid") or ()),
    )


def select_cases(cases: tuple[Case, ...], *, tags: set[str] | None = None, ids: set[str] | None = None) -> tuple[Case, ...]:
    """Return the cases that carry any of *tags* and whose id is in *ids*; ``None`` means no filter."""
    return tuple(case for case in cases if (not tags or tags & set(case.tags)) and (not ids or case.id in ids))
