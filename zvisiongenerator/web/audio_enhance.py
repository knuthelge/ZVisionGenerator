"""The audio side of ``POST /api/prompt/enhance``: validate a ``mode: "audio"`` body, run its task and shape the stream frames.

The body names a ``task`` (``sound`` or ``style``, which rewrite a caption, or a writer task, ``lyrics`` or ``script``, which edits a
longer text), the audio ``model`` and ``kind``, the ``prompt`` the task works on, the task's ``settings`` (the audio options of a
caption, see ``audio_writers`` for a writer's) and the take's ``context`` (see :func:`audio_context_from_body`). The stream is the
one image and video use (``status`` frames, then ``text`` frames, then ``done`` or ``error``), but its ``text`` and ``done`` frames
carry the written texts in ``fields``, a mapping of field name to text, so a task that writes several fields uses the same frame.
The ``fill`` task is the one that does: Describe it's idea is the ``prompt``, ``settings.fields`` names the fields still empty, and the ``done``
frame's ``fields`` holds the validated value of each one that could be written (``style``, ``tempo``, ``key``, ``lyrics``, ``vibe``, ``text``) as text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from zvisiongenerator.utils.audio_preset import AUDIO_KINDS
from zvisiongenerator.utils.audio_settings import parse_key, parse_sound_type, parse_tempo
from zvisiongenerator.utils.audio_writers import (
    FILL_FIELDS,
    FILL_TASK,
    STYLE_CONTEXT_CHARS,
    WRITER_FIELDS,
    WRITER_KINDS,
    WRITER_TASKS,
    FillResult,
    check_fill_request,
    check_lyrics_request,
    check_script_request,
    fill_settings_from_mapping,
    lyrics_settings_from_mapping,
    run_writer,
    script_settings_from_mapping,
)
from zvisiongenerator.utils.prompt_enhance import EnhanceResult
from zvisiongenerator.utils.prompt_enhance_audio import (
    AUDIO_MODE,
    AUDIO_TASKS,
    LYRICS_CONTEXT_CHARS,
    SHORT_TEXT_MAX_CHARS,
    TASK_KINDS,
    AudioEnhanceContext,
    audio_settings_from_mapping,
    enhance_audio_caption,
)
from zvisiongenerator.web.audio_models import find_audio_model
from zvisiongenerator.web.json_fields import json_field

__all__ = ["AUDIO_MODE", "ENHANCE_TASKS", "TASK_FIELDS", "audio_context_from_body", "audio_done_frame", "audio_text_frame", "run_audio_enhance", "validate_audio_enhance_body"]

ENHANCE_TASKS: tuple[str, ...] = (*AUDIO_TASKS, *WRITER_TASKS)
"""Every task an audio enhance request can name: the caption tasks, then the writer tasks."""

TASK_FIELDS: dict[str, str] = {"sound": "prompt", "style": "style", **WRITER_FIELDS}
"""The field each task writes in a ``text`` or ``done`` frame's ``fields``."""

_KINDS: dict[str, str] = {**TASK_KINDS, **WRITER_KINDS}
_FILL_KIND_DEFAULT = "music"

_EMPTY_TEXT = {"sound": "Enter a sound to enhance.", "style": "Enter a style to enhance."}


def audio_context_from_body(raw: Any) -> AudioEnhanceContext:
    """Validate the audio request's ``context``: the take's facts that shape a rewrite; an absent one is the defaults.

    Keys: ``lyrics`` and ``text`` (the song's lyrics or the speech text, only the first
    ``LYRICS_CONTEXT_CHARS`` characters are kept), ``style`` (text, only its first ``STYLE_CONTEXT_CHARS`` are kept: what the lyrics task writes to suit), ``instrumental`` (``true`` for a song without singing), ``tempo`` and ``key`` (as the Settings take them; blank is Auto),
    ``seconds`` (a positive number) and ``sound_type``. Other keys are ignored.

    Raises:
        ValueError: When *raw* is not an object or a value is invalid, in the words the user sees.
    """
    if raw is None:
        return AudioEnhanceContext()
    if not isinstance(raw, dict):
        raise ValueError("context must be a JSON object.")
    lyrics = json_field(raw, "lyrics", str, "text", name="context.lyrics") or ""
    text = json_field(raw, "text", str, "text", name="context.text") or ""
    style = json_field(raw, "style", str, "text", name="context.style") or ""
    seconds = json_field(raw, "seconds", (int, float), "a positive number", name="context.seconds")
    if seconds is not None and seconds <= 0:
        raise ValueError("context.seconds must be a positive number.")
    return AudioEnhanceContext(
        lyrics=lyrics[:LYRICS_CONTEXT_CHARS],
        text=text[:LYRICS_CONTEXT_CHARS],
        style=style[:STYLE_CONTEXT_CHARS],
        instrumental=raw.get("instrumental") is True,
        tempo=parse_tempo(raw.get("tempo")),
        key=parse_key(raw.get("key")),
        seconds=seconds,
        sound_type=parse_sound_type(raw.get("sound_type")),
    )


def validate_audio_enhance_body(body: Mapping[str, Any], *, app_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an audio enhance body into the job parameters for :func:`run_audio_enhance`.

    The ``kind`` defaults to the task's (``sound`` for the sound task, ``music`` for the style and lyrics tasks, ``speech`` for the script task; the fill
    task works on music and on speech, and defaults to music) and must be one the ``model`` makes. The fill task's ``prompt`` is the Describe it text
    (at most ``DESCRIBE_MAX_CHARS``), and its ``settings.fields`` the fields still empty, which the model must be able to take. A caption's text must not be empty and is kept to
    ``SHORT_TEXT_MAX_CHARS``; a writer task works on a longer text (``LONG_TEXT_MAX_CHARS``) that must hold what it edits. The lyrics task
    needs a model that sings lyrics; the script task works on a speech model's text.

    Returns:
        The job: ``mode``, ``task``, ``kind``, ``model`` (the preset id), ``family`` (its backend family, which picks the caption rules),
        ``prompt``, ``settings`` (the audio options of a caption, ``LyricsSettings`` for lyrics, ``ScriptSettings`` for a script, ``FillSettings`` for a fill) and ``context``.

    Raises:
        ValueError: When the task, model, kind, text, settings or context is invalid, in the words the user sees.
    """
    task = body.get("task")
    if task not in ENHANCE_TASKS:
        raise ValueError(f"task must be one of: {', '.join(ENHANCE_TASKS)}.")
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("model must be an audio model id.")
    preset = find_audio_model(app_config, model)
    kind = body.get("kind", _FILL_KIND_DEFAULT if task == FILL_TASK else _KINDS[task])
    if kind not in AUDIO_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(AUDIO_KINDS)}.")
    if task == FILL_TASK:
        if kind not in FILL_FIELDS:
            raise ValueError(f"Describe it fills {' and '.join(FILL_FIELDS)}, not {kind}.")
    elif kind != _KINDS[task]:
        raise ValueError(f"The {task} task works on {_KINDS[task]}, not {kind}.")
    if not preset.has_kind(kind):
        raise ValueError(f"{preset.label} does not make {kind}.")
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")
    context = audio_context_from_body(body.get("context"))
    if task in WRITER_TASKS:
        settings = _writer_settings(task, body.get("settings") or {}, prompt=prompt, preset=preset, kind=kind)
    else:
        if len(prompt) > SHORT_TEXT_MAX_CHARS:
            raise ValueError(f"The text is too long ({len(prompt)} characters); keep it to {SHORT_TEXT_MAX_CHARS:,}.")
        if not prompt.strip():
            raise ValueError(_EMPTY_TEXT[task])
        settings = audio_settings_from_mapping(body.get("settings") or {}, task=task)
    return {"mode": AUDIO_MODE, "task": task, "kind": kind, "model": preset.id, "family": preset.family, "prompt": prompt, "settings": settings, "context": context}


def _writer_settings(task: str, raw: Any, *, prompt: str, preset: Any, kind: str) -> Any:
    """Validate a writer task's settings against its text and the model; each task checks what it needs."""
    if task == FILL_TASK:
        settings = fill_settings_from_mapping(raw, kind=kind)
        if "lyrics" in settings.fields and not preset.takes_lyrics(kind):
            raise ValueError(f"{preset.label} does not take lyrics.")
        check_fill_request(prompt)
        return settings
    if task == "lyrics":
        if not preset.takes_lyrics(_KINDS[task]):
            raise ValueError(f"{preset.label} does not take lyrics.")
        settings = lyrics_settings_from_mapping(raw)
        check_lyrics_request(prompt, settings)
        return settings
    if task == "script":
        settings = script_settings_from_mapping(raw)
        check_script_request(prompt)
        return settings
    raise ValueError(f"task must be one of: {', '.join(ENHANCE_TASKS)}.")


def run_audio_enhance(
    job: Mapping[str, Any],
    enhancer: Any,
    *,
    seed: int,
    temperature: float,
    on_text: Callable[[str], None],
    cancelled: Callable[[], bool],
) -> EnhanceResult | FillResult:
    """Run *job*'s task with *enhancer* and return its result; *on_text* receives a caption task's in-progress text (a writer task reports only its result).

    Raises:
        ValueError: When there is nothing to rewrite.
        RuntimeError: When cancelled or when the enhancer returns nothing usable.
    """
    if job["task"] in WRITER_TASKS:
        return run_writer(job["task"], enhancer, job["prompt"], job["settings"], context=job["context"], family=job["family"], seed=seed, temperature=temperature, cancelled=cancelled)
    return enhance_audio_caption(
        enhancer,
        job["prompt"],
        job["settings"],
        task=job["task"],
        family=job["family"],
        context=job["context"],
        seed=seed,
        temperature=temperature,
        on_text=on_text,
        cancelled=cancelled,
    )


def audio_text_frame(task: str, text: str) -> dict[str, Any]:
    """Return the ``text`` frame of an in-progress caption rewrite: ``{type, task, text, fields}`` with the text under the task's field."""
    return {"type": "text", "task": task, "text": text, "fields": {TASK_FIELDS[task]: text}}


def audio_done_frame(task: str, result: EnhanceResult | FillResult, *, model: str) -> dict[str, Any]:
    """Return the terminal ``done`` frame: ``{type, task, fields, clamped, model, warning?}``.

    ``fields`` maps each field the task wrote to its text (a fill writes several; its tempo is the number as text). ``model`` is the enhancer that wrote it (``REPO[@REVISION]``), which the client
    sends back with Generate so the take records it.
    """
    fields = dict(result.fields) if isinstance(result, FillResult) else {TASK_FIELDS[task]: result.prompt}
    frame: dict[str, Any] = {"type": "done", "task": task, "fields": fields, "clamped": False if isinstance(result, FillResult) else result.clamped, "model": model}
    if result.warning:
        frame["warning"] = result.warning
    return frame
