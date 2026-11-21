"""Answer "does this job fit?" for the Web UI: the body of ``POST /api/memory`` and the ``memory`` of each model row.

Every answer comes from one call per row to :func:`~zvisiongenerator.utils.memory_verdict.judge`. This module builds the job
of each medium from what the caller knows, resolves the weights, and shapes the verdicts; it holds no threshold or figure of
its own.

The weights come from the model's files when it is downloaded, else from the shipped table (see
:mod:`zvisiongenerator.utils.weight_totals`); the Hugging Face header is never read, so a request stays fast and offline.

The answer is ``None`` when the machine has no memory budget or the weights cannot be resolved; otherwise:

- ``kind``: ``"discrete"`` (a card and system RAM) or ``"unified"`` (a Mac).
- ``budget``: ``{"gpu_gb", "ram_gb"}``; on a Mac ``gpu_gb`` is the total unified memory the job is graded against and ``ram_gb`` is ``None``.
- ``control``: what the rows vary, ``"quantize"`` or ``"low_memory"``, or ``None`` for a single row.
- ``current``: the key of the row the job as set up falls on (``"none"``, ``"8"``, ``"4"``, ``"on"``, ``"off"`` or ``"only"``).
- ``rows``: ``[{"key", "status", "resource", "gpu_gb", "ram_gb", "word", "label", "sentence", "details"}]``, one per choice of the control. ``word`` is the status as
  a word, ``label`` the level's name (empty for a single row), ``sentence`` what the status means for this computer, and ``details`` what was judged at that level.
- ``title``, ``control_title`` and ``table_heading``: the heading of the answer, the title of its level column and the heading over its table of levels (the last two ``None`` for a single row).
  Every word comes from :mod:`zvisiongenerator.utils.memory_wording`; the client lays them out and words nothing.
- ``judged``: ``{"frame", "width", "height", "seconds", "quantize", "low_memory", "references", "loras"}``, what was judged.
- ``download_size_label``: the download a model that is not on disk needs, as the model metadata words it (such as ``"35 GB"``), else ``None``.
- ``expected_seconds``: MiniMax H3's estimated generation time, else ``None``.
- ``notes``: present only when the job was judged on something other than what was asked: short sentences naming each LoRA that is not
  available (counted as 0 bytes). Only LoRAs produce notes; an Extend length that is not a multiple of the model's step is judged at the
  nearest multiple without one.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from zvisiongenerator.backends import video_family_on_platform
from zvisiongenerator.core.video_family import UNDESCRIBED_UPSCALE_STEPS, settle_video_steps
from zvisiongenerator.utils.audio_memory import audio_verdict
from zvisiongenerator.utils.audio_preset import AudioPreset
from zvisiongenerator.utils.edit_request import parse_extend
from zvisiongenerator.utils.image_memory_job import file_megapixels, sized_edit_job, sized_image_job, starts_from_picture
from zvisiongenerator.utils.lora import parse_lora_arg
from zvisiongenerator.utils.memory_rows import NONE_KEY, ONLY_KEY, judge_rows, level_key, low_memory_key, low_memory_rows, quantize_choices, quantize_rows
from zvisiongenerator.utils.memory_verdict import MemoryJob, MemoryVerdict, MemoryWeights
from zvisiongenerator.utils.memory_wording import control_title, judged_details, judged_title, level_label, memory_sentence, status_word, table_heading
from zvisiongenerator.utils.model_files import find_local_model_dir
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget
from zvisiongenerator.utils.model_metadata import alias_map, model_download_size_label
from zvisiongenerator.utils.paths import resolve_model_path
from zvisiongenerator.utils.video_extend import extend_step, resolve_extend_frames, window_frames
from zvisiongenerator.utils.video_memory_job import clip_memory_job
from zvisiongenerator.utils.weight_totals import resolve_image_weights, resolve_ltx_weights
from zvisiongenerator.web.config import WebUiConfig
from zvisiongenerator.web.browse_media import media_file
from zvisiongenerator.web.json_fields import json_field
from zvisiongenerator.web.lora_fit import LoraSize, lora_size_for_fit
from zvisiongenerator.web.model_status import model_downloaded, nothing_to_download
from zvisiongenerator.web.path_browser import resolve_purpose
from zvisiongenerator.web.request_fields import image_ratio, image_size_name, optional_text, settled_video_steps, video_model_name, video_ratio_and_size
from zvisiongenerator.web.video_clip import require_upscale_factor, resolve_video_clip, resolve_video_model

__all__ = [
    "audio_memory",
    "memory_response",
    "reference_image_memory",
    "reference_video_memory",
]

IMAGE_FRAME = "image"
EDIT_FRAME = "edit"
CLIP_FRAME = "clip"
EXTENSION_FRAME = "extension"
REFERENCE_FRAME = "reference"
"""What a verdict judged: a job of the medium, or its reference job (a model listed without a job to judge, which no request asks for)."""

_FRAMES = {"image": (IMAGE_FRAME, EDIT_FRAME), "video": (CLIP_FRAME, EXTENSION_FRAME)}

# The reference clip a listed video model is judged on: 5 s at 768x448, text to video. A length is 8n + 1 frames for LTX and
# 17n + 5 for MiniMax H3, so 5 s at 24 fps is 121 and 124 frames.
_REFERENCE_CLIP_SIZE = (768, 448)
_REFERENCE_CLIP_FRAMES = {"ltx": 121, "h3": 124}
_REFERENCE_CLIP_SECONDS = 5.0

# The reference image of a listed model: a new 1024x1024 image with no references and no LoRAs.
_REFERENCE_IMAGE_SIZE = (1024, 1024)

type FindLocalDir = Callable[[str], Path | None]


def reference_video_memory(
    name: str,
    resolved_path: str,
    family: str,
    *,
    budget: MemoryBudget | None,
    downloaded: bool | None,
    find_local_dir: FindLocalDir = find_local_model_dir,
) -> dict[str, Any] | None:
    """Return the memory answer of a listed video model: its reference clip of 5 s at 768x448, text to video, no upscale and no LoRAs, with Low memory on; ``None`` for a family without one."""
    frames = _REFERENCE_CLIP_FRAMES.get(family)
    if frames is None:
        return None
    job = clip_memory_job(family, width=_REFERENCE_CLIP_SIZE[0], height=_REFERENCE_CLIP_SIZE[1], frames=frames)
    return video_memory(name, resolved_path, job, frame=REFERENCE_FRAME, seconds=_REFERENCE_CLIP_SECONDS, low_memory=True, budget=budget, downloaded=downloaded, find_local_dir=find_local_dir)


def reference_image_memory(
    name: str,
    resolved_path: str,
    *,
    quantize: int | None = None,
    levels: tuple[int, ...],
    budget: MemoryBudget | None,
    downloaded: bool | None,
    find_local_dir: FindLocalDir = find_local_model_dir,
) -> dict[str, Any] | None:
    """Return the memory answer of a listed image model: its reference job of a new 1024x1024 image with no references and no LoRAs."""
    return image_memory(
        name, resolved_path, job=sized_image_job(*_REFERENCE_IMAGE_SIZE), frame=REFERENCE_FRAME, quantize=quantize, levels=levels, budget=budget, downloaded=downloaded, find_local_dir=find_local_dir
    )


def image_memory(
    name: str,
    resolved_path: str,
    *,
    job: MemoryJob,
    frame: str,
    quantize: int | None,
    levels: tuple[int, ...],
    budget: MemoryBudget | None,
    downloaded: bool | None,
    loras: int = 0,
    notes: Sequence[str] = (),
    find_local_dir: FindLocalDir = find_local_model_dir,
) -> dict[str, Any] | None:
    """Return the memory answer of an image job, with one row per quantize level, or ``None`` when there is no budget or the weights are unknown.

    A model stored at one level has one row at that level whatever level is asked for.

    Args:
        name: The model as picked.
        resolved_path: What the name resolves to: a model folder or ``owner/repo[@revision]``.
        job: The size the model draws, its references and its LoRAs.
        frame: What the job is (``"image"``, ``"edit"`` or ``"reference"``).
        quantize: The level the job is set to; ``None`` is unquantized.
        levels: The quantize levels the Web UI offers.
        budget: This machine's memory, or ``None`` when it has none.
        downloaded: Whether the model is on disk; ``None`` when unknown.
        loras: How many LoRAs the job applies.
        notes: What the job was judged on instead of what was asked, as the answer's ``notes``.
        find_local_dir: Locates a downloaded model directory.
    """
    if not _has_budget(budget):
        return None
    resolved = resolve_image_weights(name, resolved_path, platform_key=sys.platform, find_local_dir=find_local_dir)
    if resolved is None:
        return None
    choices = quantize_choices(resolved.stored_bits, resolved.family, levels)
    current = quantize if quantize in choices else choices[0]
    template = replace(job, family=resolved.family, model_size=resolved.model_size)
    verdicts = judge_rows(quantize_rows(template, choices), budget, resolved.weights)
    if verdicts is None:
        return None
    if level_key(current) not in dict(verdicts):
        # The level the job is set to is one the engine cannot judge (a Quant option such as 6), so the answer is about the first level it could judge.
        current = next(level for level in choices if level_key(level) == verdicts[0][0])
    # An edit's picture is in the job as its one reference, but it is not one the user added, so the line does not count it.
    references = len(job.reference_megapixels) - (frame == EDIT_FRAME)
    judged = _judged(frame, width=job.width, height=job.height, quantize=current, references=references, loras=loras)
    return _answer(budget, verdicts, control="quantize" if len(choices) > 1 else None, current=level_key(current), judged=judged, download=_download_label(name, downloaded), notes=notes)


def video_memory(
    name: str,
    resolved_path: str,
    job: MemoryJob,
    *,
    frame: str,
    seconds: float | None,
    low_memory: bool,
    budget: MemoryBudget | None,
    downloaded: bool | None,
    loras: int = 0,
    notes: Sequence[str] = (),
    find_local_dir: FindLocalDir = find_local_model_dir,
) -> dict[str, Any] | None:
    """Return the memory answer of a video clip, or ``None`` when there is no budget, the family has no estimate or the weights are unknown.

    LTX has two rows, Low memory on and off. MiniMax H3 has one, and its answer carries the expected time.
    Args:
        name: The model as picked.
        resolved_path: What the name resolves to: a model folder or ``owner/repo[@revision]``.
        job: The clip: its family, size, frames, mode and the fields that family reads. Its ``low_memory`` is set per row.
        frame: What the job is (``"clip"``, ``"extension"`` or ``"reference"``).
        seconds: The clip's length in seconds, for the judged line.
        low_memory: Whether the job has Low memory on (LTX).
        budget: This machine's memory, or ``None`` when it has none.
        downloaded: Whether the model is on disk; ``None`` when unknown.
        loras: How many LoRAs the job applies.
        notes: What the job was judged on instead of what was asked, as the answer's ``notes``.
        find_local_dir: Locates a downloaded model directory.
    """
    if not _has_budget(budget):
        return None
    if job.family == "ltx":
        resolved = resolve_ltx_weights(name, resolved_path, platform_key=sys.platform, find_local_dir=find_local_dir)
        if resolved is None:
            return None
        weights, switch = resolved.weights, True
    elif job.family == "h3":
        weights, switch = MemoryWeights(), False
    else:
        return None
    verdicts = judge_rows(low_memory_rows(job) if switch else [(ONLY_KEY, job)], budget, weights)
    if verdicts is None:
        return None
    judged = _judged(frame, width=job.width, height=job.height, seconds=seconds, low_memory=low_memory if switch else None, references=job.references, loras=loras)
    current = low_memory_key(low_memory) if switch else ONLY_KEY
    return _answer(budget, verdicts, control="low_memory" if switch else None, current=current, judged=judged, download=_download_label(name, downloaded), notes=notes)


def audio_memory(preset: AudioPreset, *, platform_key: str, budget: MemoryBudget | None, downloaded: bool | None) -> dict[str, Any] | None:
    """Return the memory answer of an audio model: its measured peak on *platform_key* against the machine, in one row; ``None`` without a budget or a measured peak.

    Args:
        preset: The audio model as this machine runs it.
        platform_key: The accelerator key (``"cuda"`` or ``"mps"``) its peak was measured under.
        budget: This machine's memory, or ``None`` when it has none.
        downloaded: Whether the model is on disk; ``None`` when unknown.
    """
    verdict = audio_verdict(preset, platform_key, budget)
    if verdict is None or budget is None:
        return None
    return _answer(budget, [(ONLY_KEY, verdict)], control=None, current=ONLY_KEY, judged=_judged(REFERENCE_FRAME), download=_download_label(preset.id, downloaded))


def memory_response(
    payload: Any,
    web_config: WebUiConfig,
    *,
    budget: MemoryBudget | None,
    downloading_loras: Mapping[str, int] | None = None,
    find_local_dir: FindLocalDir = find_local_model_dir,
) -> dict[str, Any] | None:
    """Build the ``POST /api/memory`` answer for a request body; ``None`` when this machine has no budget or the model's weights are unknown.

    ``media`` (``"image"`` or ``"video"``) and ``model`` pick the job; ``kind`` says which of the medium's frames it is (``"image"``, ``"edit"``; ``"clip"``,
    ``"extension"``). The other fields:

    - **Image:** ``quantize``; ``ratio`` and ``size``, or ``width`` and ``height`` (each overriding the preset's axis); the reference pictures, each judged at its real size:
      ``reference_sizes`` (the ``[width, height]`` of each picture the browser measured, in pixels) and ``reference_paths`` (the files on this machine, whose headers
      are read, under the rules of the file browser's ``reference_image`` purpose; one that cannot be read counts nothing, as for the command line); ``lora``. An edit sends ``source_width`` and ``source_height`` (the picture's size), ``masked`` and
      ``extend`` (``side=amount``); the size judged is what the model draws, and the picture is its one reference. A plain image may send
      ``upscale`` (2 or 4) and ``image`` (a start image is set): it is judged at its final size, starting from a picture.
    - **Video:** ``low_memory`` (default on); a clip sends ``ratio``, ``size``, ``width``, ``height``, ``frames``, ``upscale``, ``steps``, ``image``
      (an image to video clip), ``review``, ``references`` (how many) and ``lora``. An extension sends ``source`` (the clip's path),
      ``extend_frames`` (the default length when absent), ``steps``, ``references`` and ``lora``; the window the model generates is judged.

    Args:
        payload: The request body.
        web_config: The Web UI config.
        budget: This machine's memory, or ``None`` when it has none.
        downloading_loras: ``{name: expected bytes}`` of the LoRAs still downloading, which count at the size they will have.
        find_local_dir: Locates a downloaded model directory.

    Raises:
        ValueError: If the body, a field, the model or a LoRA is invalid, or the source clip of an extension cannot be used.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    media = json_field(payload, "media", str, "a string")
    if media not in _FRAMES:
        raise ValueError(f"media must be one of {list(_FRAMES)}.")
    kind = _text(payload, "kind") or _FRAMES[media][0]
    if kind not in _FRAMES[media]:
        raise ValueError(f"kind must be one of {list(_FRAMES[media])} for {media}.")
    if media == "image":
        return _image_response(payload, kind, web_config, budget, downloading_loras, find_local_dir)
    return _video_response(payload, kind, web_config, budget, downloading_loras, find_local_dir)


# --- requests -----------------------------------------------------------------------------------------------------------


def _image_response(
    payload: dict[str, Any], kind: str, web_config: WebUiConfig, budget: MemoryBudget | None, downloading_loras: Mapping[str, int] | None, find_local_dir: FindLocalDir
) -> dict[str, Any] | None:
    app_config = web_config.app_config
    name = _text(payload, "model")
    if not name:
        raise ValueError("A model is required.")
    quantize = json_field(payload, "quantize", int, "an integer")
    if quantize is not None and quantize not in web_config.quantize_options:
        raise ValueError(f"quantize must be one of {list(web_config.quantize_options)}.")
    resolved_path = resolve_model_path(name, aliases=alias_map(app_config), platform_key=sys.platform)
    levels = tuple(web_config.quantize_options)
    downloaded = nothing_to_download(resolved_path, downloaded=model_downloaded(resolved_path, kind="image", app_config=app_config, find_local_dir=find_local_dir))
    lora_size, loras = _loras(payload, downloading_loras)
    lora_bytes = lora_size.total
    if kind == EDIT_FRAME:
        source_width, source_height = (json_field(payload, key, int, "a positive integer", minimum=1) for key in ("source_width", "source_height"))
        if source_width is None or source_height is None:
            raise ValueError("An edit needs source_width and source_height.")
        masked = json_field(payload, "masked", bool, "a boolean")
        extend = _text(payload, "extend")
        job = sized_edit_job((source_width, source_height), masked=masked is True, extend=parse_extend(extend) if extend else None, lora_bytes=lora_bytes)
    else:
        width, height = _image_size(payload, web_config)
        upscale = json_field(payload, "upscale", int, "a positive integer", minimum=1)
        if upscale not in (None, 2, 4):
            raise ValueError("An image upscale is 2 or 4.")
        start_image = json_field(payload, "image", bool, "a boolean") is True
        # The size is the one the run draws and refines to: an upscale's final size, as the run's own job has it.
        job = sized_image_job(width, height, reference_megapixels=_reference_megapixels(payload), lora_bytes=lora_bytes, from_picture=starts_from_picture(upscale=upscale, start_image=start_image))
    return image_memory(
        name, resolved_path, job=job, frame=kind, quantize=quantize, levels=levels, budget=budget, downloaded=downloaded, loras=loras, notes=_lora_notes(lora_size), find_local_dir=find_local_dir
    )


def _video_response(
    payload: dict[str, Any], kind: str, web_config: WebUiConfig, budget: MemoryBudget | None, downloading_loras: Mapping[str, int] | None, find_local_dir: FindLocalDir
) -> dict[str, Any] | None:
    app_config = web_config.app_config
    name = video_model_name(payload, web_config)
    low_memory = json_field(payload, "low_memory", bool, "a boolean") is not False
    resolved_path, model_info = resolve_video_model(app_config, name)
    family = model_info.family
    downloaded = model_downloaded(resolved_path, kind="video", family=family, model_name=name, app_config=app_config, find_local_dir=find_local_dir)
    lora_size, loras = _loras(payload, downloading_loras)
    notes = _lora_notes(lora_size)
    references = json_field(payload, "references", int, "a non-negative whole number", minimum=0) or 0
    steps = json_field(payload, "steps", int, "a positive integer", minimum=1)
    if kind == EXTENSION_FRAME:
        job, seconds = _extension_job(payload, name, family, web_config, steps=steps, references=references, lora_bytes=lora_size.total)
    else:
        job, seconds = _clip_job(payload, web_config, model_info, steps=steps, references=references, lora_bytes=lora_size.total)
    return video_memory(name, resolved_path, job, frame=kind, seconds=seconds, low_memory=low_memory, budget=budget, downloaded=downloaded, loras=loras, notes=notes, find_local_dir=find_local_dir)


def _clip_job(payload: dict[str, Any], web_config: WebUiConfig, model_info: Any, *, steps: int | None, references: int, lora_bytes: int) -> tuple[MemoryJob, float]:
    """Return the job of a clip and its length in seconds: its size, length and steps resolve exactly as in a Text to Video or Image to Video job."""
    app_config = web_config.app_config
    width, height, frames, upscale = (json_field(payload, key, int, "a positive integer", minimum=1) for key in ("width", "height", "frames", "upscale"))
    ratio, size = video_ratio_and_size(payload, web_config)
    require_upscale_factor(upscale)
    clip = resolve_video_clip(app_config, model_info, ratio=ratio, size=size, steps=steps, width=width, height=height, frames=frames, upscale=bool(upscale), quiet=True)
    settled = settled_video_steps(clip.steps, bool(upscale), app_config, model_info)
    job = clip_memory_job(
        model_info.family,
        width=clip.width,
        height=clip.height,
        frames=clip.num_frames,
        mode="i2v" if json_field(payload, "image", bool, "a boolean") else "t2v",
        references=references,
        upscale=bool(upscale),
        review=json_field(payload, "review", bool, "a boolean") is True,
        steps=settled.steps,
        lora_bytes=lora_bytes,
    )
    return job, clip.num_frames / (model_info.default_fps or 24)


def _extension_job(payload: dict[str, Any], name: str, family: str, web_config: WebUiConfig, *, steps: int | None, references: int, lora_bytes: int) -> tuple[MemoryJob, float]:
    """Return the job of an extension and the seconds of its window: the window is the frames the model generates, the carried frames plus the new ones.

    A length in frames that is not a multiple of the model's step is judged at the nearest multiple (the job itself refuses it).
    """
    from zvisiongenerator.web.extend_api import load_source_video  # extend_api reaches the gallery, which reads the audio rows this module feeds

    source = load_source_video(payload.get("source"), key="source")
    described = video_family_on_platform(web_config.app_config, family)
    extend_frames = json_field(payload, "extend_frames", int, "a positive integer", minimum=1)
    new_frames, _rounded = resolve_extend_frames(frames=extend_frames, seconds=None, fps=source.fps, strict=False, step=extend_step(described))
    window = window_frames(new_frames, described)
    settled = settle_video_steps(steps, upscale=False, family=described, default_upscale_steps=UNDESCRIBED_UPSCALE_STEPS)
    job = clip_memory_job(family, width=source.width, height=source.height, frames=window, mode="extend", references=references, steps=settled.steps, lora_bytes=lora_bytes)
    return job, window / source.fps


def _image_size(payload: dict[str, Any], web_config: WebUiConfig) -> tuple[int, int]:
    """Return the size an image job draws: the preset of its ratio and size, each axis overridden by ``width`` or ``height``."""
    width, height = (json_field(payload, key, int, "a positive integer", minimum=1) for key in ("width", "height"))
    ratio = image_ratio(payload, web_config)
    dims = web_config.app_config["sizes"][ratio][image_size_name(payload, web_config, ratio)]
    return width or dims["width"], height or dims["height"]


def _reference_megapixels(payload: dict[str, Any]) -> tuple[float, ...]:
    """Return the decimal megapixels of each reference picture of the request: the ``reference_sizes`` it was told, then the files of ``reference_paths``, whose headers are read.

    Raises:
        ValueError: If ``reference_sizes`` is not a list of ``[width, height]`` pairs of positive whole numbers, or ``reference_paths`` not a list of text.
    """
    sizes = json_field(payload, "reference_sizes", list, "a list of [width, height] pairs")
    megapixels: list[float] = []
    for size in sizes or ():
        if not (isinstance(size, list) and len(size) == 2 and all(isinstance(side, int) and not isinstance(side, bool) and side > 0 for side in size)):
            raise ValueError("reference_sizes must be a list of [width, height] pairs.")
        megapixels.append(size[0] * size[1] / 1e6)
    paths = json_field(payload, "reference_paths", list, "a list of file paths")
    if not all(isinstance(path, str) for path in paths or ()):
        raise ValueError("reference_paths must be a list of file paths.")
    return (*megapixels, *file_megapixels(_reference_files(paths or ())))


def _reference_files(paths: Sequence[str]) -> list[Path]:
    """Return the files of *paths* the file browser would serve as reference images; a path it would refuse is left out."""
    purpose = resolve_purpose("reference_image")
    files: list[Path] = []
    for path in paths:
        try:
            files.append(media_file(purpose, path, ("image",)).path)
        except FileNotFoundError, ValueError:
            continue
    return files


def _loras(payload: dict[str, Any], downloading: Mapping[str, int] | None) -> tuple[LoraSize, int]:
    """Return the size of the request's ``lora`` list and how many LoRAs it names; a LoRA still downloading counts the size it will have.

    A LoRA that is not available counts 0 bytes and is named in the size's ``unavailable``: the estimate still judges the job.

    Raises:
        ValueError: If the list is malformed.
    """
    text = _text(payload, "lora") or ""
    return lora_size_for_fit(text, downloading), len(parse_lora_arg(text)) if text else 0


def _lora_notes(size: LoraSize) -> list[str]:
    """Return the note for the LoRAs a request names that are not available, or none."""
    if not size.unavailable:
        return []
    if len(size.unavailable) == 1:
        return [f"LoRA {size.unavailable[0]} isn't available, so it isn't counted."]
    return [f"LoRAs {', '.join(size.unavailable)} aren't available, so they aren't counted."]


# --- answers ------------------------------------------------------------------------------------------------------------


def _answer(
    budget: MemoryBudget, verdicts: list[tuple[str, MemoryVerdict]], *, control: str | None, current: str, judged: dict[str, Any], download: str | None, notes: Sequence[str] = ()
) -> dict[str, Any]:
    """Shape the verdicts of a job's rows as the answer, words included; the expected time is the current row's, and ``notes`` is left out when there are none."""
    chosen = dict(verdicts).get(current) or verdicts[0][1]
    frame = judged["frame"]
    answer = {
        "kind": "unified" if budget.system_bytes is None else "discrete",
        "budget": {
            "gpu_gb": _gb(budget.unified_bytes or budget.gpu_bytes if budget.system_bytes is None else budget.gpu_bytes),
            "ram_gb": None if budget.system_bytes is None else _gb(budget.system_bytes),
        },
        "control": control,
        "current": current,
        "rows": [
            {
                "key": key,
                "status": verdict.status,
                "resource": verdict.resource,
                "gpu_gb": _gb(verdict.gpu_bytes),
                "ram_gb": None if verdict.ram_bytes is None else _gb(verdict.ram_bytes),
                "word": status_word(verdict.status),
                "label": level_label(control, key),
                "sentence": memory_sentence(verdict.status, verdict.resource),
                "details": _row_details(judged, control, key),
            }
            for key, verdict in verdicts
        ],
        "title": judged_title(frame, width=judged["width"], height=judged["height"], seconds=judged["seconds"]),
        "control_title": control_title(control),
        "table_heading": table_heading(control, frame),
        "judged": judged,
        "download_size_label": download,
        "expected_seconds": chosen.expected_seconds,
    }
    return {**answer, "notes": list(notes)} if notes else answer


def _row_details(judged: dict[str, Any], control: str | None, key: str) -> str:
    """Return what was judged at one row's level: the job's facts with the row's Quant level or Low memory setting."""
    facts = {name: judged[name] for name in ("width", "height", "seconds", "quantize", "low_memory", "references", "loras")}
    if control == "quantize":
        facts["quantize"] = None if key == NONE_KEY else int(key)
    elif control == "low_memory":
        facts["low_memory"] = key == "on"
    return judged_details(judged["frame"], **facts)


def _judged(
    frame: str,
    *,
    width: int | None = None,
    height: int | None = None,
    seconds: float | None = None,
    quantize: int | None = None,
    low_memory: bool | None = None,
    references: int = 0,
    loras: int = 0,
) -> dict[str, Any]:
    """Return what a verdict judged, so the client can word it: the frame, the size and length, the choices and the counts."""
    return {
        "frame": frame,
        "width": width,
        "height": height,
        "seconds": None if seconds is None else round(seconds, 1),
        "quantize": quantize,
        "low_memory": low_memory,
        "references": references,
        "loras": loras,
    }


def _download_label(name: str, downloaded: bool | None) -> str | None:
    """Return the one-time download size label of a model that is not on disk and has a recorded size; otherwise ``None``."""
    return model_download_size_label(name, sys.platform) if downloaded is False else None


def _has_budget(budget: MemoryBudget | None) -> bool:
    return budget is not None and bool(budget.gpu_bytes)


def _gb(value: int) -> float:
    return round(value / GIB, 1)


def _text(payload: dict[str, Any], key: str) -> str | None:
    """Return the stripped text of *key*, ``None`` when it is absent or blank.

    Raises:
        ValueError: If the field is not a string.
    """
    json_field(payload, key, str, "a string")
    return optional_text(payload, key)
