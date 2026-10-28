"""Read the trigger text of a LoRA from the places that name one: a model card's ``instance_prompt`` and a file's caption metadata.

Both readers are pure. The file name and ``ss_output_name`` are never a trigger.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any

__all__ = ["MAX_TRIGGER_LENGTH", "caption_trigger", "card_triggers", "clean_trigger"]

MAX_TRIGGER_LENGTH = 200
_PHRASE_BREAK = re.compile(r"[,;\n\r]")


def clean_trigger(text: object) -> str | None:
    """Return *text* trimmed to one line of at most ``MAX_TRIGGER_LENGTH`` characters, or None when it is not text or is blank."""
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.split())[:MAX_TRIGGER_LENGTH].strip()
    return cleaned or None


def card_triggers(instance_prompt: object) -> str | None:
    """Return the trigger words of a model card's ``instance_prompt`` as one comma-separated line, or None when it names none.

    The prompt (or each entry of a list) is split at commas, semicolons and line breaks; blanks and repeats (in any case)
    are dropped, so ``"t3chnic4lly, 1960s film"`` gives ``"t3chnic4lly, 1960s film"`` and a sentence stays one word. The
    line is cut at ``MAX_TRIGGER_LENGTH`` characters on a word boundary.
    """
    parts = instance_prompt if isinstance(instance_prompt, list) else [instance_prompt]
    seen: set[str] = set()
    words: list[str] = []
    for part in parts:
        if not isinstance(part, str):
            continue
        for piece in _PHRASE_BREAK.split(part):
            word = " ".join(piece.split()).rstrip(".").strip()
            if word and word.casefold() not in seen:
                seen.add(word.casefold())
                words.append(word)
    if not words:
        return None
    line = ""
    for word in words:
        candidate = f"{line}, {word}" if line else word
        if len(candidate) > MAX_TRIGGER_LENGTH:
            break
        line = candidate
    # A first word that is longer than the limit on its own is cut, as any typed trigger is.
    return line or clean_trigger(words[0])


def caption_trigger(metadata: Mapping[str, Any]) -> str | None:
    """Return the one caption a LoRA was trained on, from its ``ss_tag_frequency`` metadata, or None.

    The metadata holds, per training folder, how often each caption tag occurred. Exactly one distinct tag across all
    folders is a single caption, which is the trigger; several tags are a dataset's vocabulary and name no trigger.
    """
    raw = metadata.get("ss_tag_frequency")
    try:
        frequencies = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    if not isinstance(frequencies, Mapping):
        return None
    tags = {tag.strip() for folder in frequencies.values() if isinstance(folder, Mapping) for tag in folder if isinstance(tag, str) and tag.strip()}
    return clean_trigger(next(iter(tags))) if len(tags) == 1 else None
