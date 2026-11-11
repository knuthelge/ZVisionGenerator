"""Build, embed and read the config metadata saved inside generated images and videos."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL.PngImagePlugin import PngInfo

from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageWorkingArtifacts
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.utils.edit_request import parse_extend
from zvisiongenerator.utils.enhance_record import parse_enhance_record
from zvisiongenerator.utils.ffmpeg import staging_path
from zvisiongenerator.utils.paths import collapse_home, display_stem, expand_home

IMAGE_CONFIG_SCHEMA = "zvisiongenerator.config.v1"
VIDEO_CONFIG_SCHEMA = IMAGE_CONFIG_SCHEMA
AUDIO_CONFIG_SCHEMA = IMAGE_CONFIG_SCHEMA
_PNG_CONFIG_KEY = "zvisiongenerator.config"
# The container tag every video and audio file carries its config under.
_CONFIG_KEY = "zvisiongenerator.config"
# ffmpeg's output option that lets MP4 keep a custom tag.
_MP4_TAG_OPTIONS = ("-movflags", "use_metadata_tags")
# EXIF ImageDescription tag, where saved images also carry the rendered prompt.
EXIF_IMAGE_DESCRIPTION = 0x010E
# The recorded ``generation.upscale.method`` values: refine with the image model, or upscale with SeedVR2.
UPSCALE_METHOD_CREATIVE = "creative"
UPSCALE_METHOD_SEEDVR2 = "seedvr2"
# A video file upscaled on macOS: LTX's spatial upsampler and a short refine of the clip's own latent.
UPSCALE_METHOD_LTX = "ltx"
UPSCALE_WORKFLOW = "upscale"
# What an edit can do, as recorded in the ``edit`` block.
_EDIT_MODES = ("paint", "extend", "whole")


def build_image_config_payload(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> dict[str, Any]:
    """Build the config payload embedded in a saved image.

    The top-level keys are the reusable settings (schema, workflow, prompt, negative prompt, model, family, seed,
    steps, guidance, scheduler, dimensions, ratio, size, start image, strength, references (``reference_images``, only
    when set), LoRAs). ``generation`` records
    how the file was made (time, upscale, post-processing) and ``source`` the image an upscale started from.
    An auto-enhanced prompt replaces the template prompt so the asset records what rendered. An enhanced prompt also
    records the user's own text as ``prompt_input`` and how it was enhanced as ``enhance`` (options, model, tier, format).
    """
    width = artifacts.image.width if artifacts.image is not None else request.width
    height = artifacts.image.height if artifacts.image is not None else request.height
    negative_prompt = None if artifacts.metadata.get("negative_suppressed") else request.negative_prompt
    # A SeedVR2 upscale of a source with no recorded sampling settings must not save class defaults as if they were its own.
    unknown_source = request.upscale_method == UPSCALE_METHOD_SEEDVR2 and not request.source_settings_known
    # SeedVR2 runs no image model, so a source that recorded no family leaves it unrecorded rather than "unknown".
    family_unknown = request.upscale_method == UPSCALE_METHOD_SEEDVR2 and request.model_family == "unknown"
    payload: dict[str, Any] = {
        "schema": IMAGE_CONFIG_SCHEMA,
        "workflow": _image_workflow(request),
        "prompt": artifacts.metadata.get("enhanced_prompt") or request.prompt,
        "negative_prompt": negative_prompt or None,
        "model": _portable_path(request.model_name),
        "model_family": None if family_unknown else request.model_family,
        "seed": None if unknown_source else request.seed,
        "steps": None if unknown_source else request.steps,
        "guidance": None if unknown_source else request.guidance,
        "scheduler": request.scheduler_name,
        "width": width,
        "height": height,
        "ratio": request.ratio,
        "size": request.size,
        "image_path": _portable_path(request.image_path),
        "lora": _format_loras(request.lora_paths, request.lora_weights),
        "generation": _image_generation_details(request, artifacts),
    }
    if request.image_path:
        payload["image_strength"] = request.image_strength
    if request.reference_paths:
        payload["reference_images"] = [_portable_path(path) for path in request.reference_paths]
    payload.update(_enhance_fields(request.enhance_record, prompt_input=_image_prompt_input(request, artifacts)))
    source = artifacts.metadata.get("upscale_source")
    if source:
        # The workflow that made the original, so Reuse settings regenerates it (img2img keeps its reference).
        payload["source"] = {**source, "path": _portable_path(source.get("path")), "workflow": "img2img" if request.image_path else "txt2img"}
    if request.edit_source:
        payload.update(_edit_fields(request, artifacts))
    return _drop_unserializable(payload)


def build_video_config_payload(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> dict[str, Any]:
    """Build the config payload embedded in a saved video.

    The top-level keys are the reusable settings (schema, workflow, prompt, model, family, seed, steps,
    dimensions, frame count, reference image, last frame (``end_image_path``, only when set), LoRAs); ``generation`` records time, upscale, audio and format.
    An auto-enhanced prompt replaces the template prompt so the asset records what rendered. The prompt is the
    action alone; the scene is recorded apart as ``scene`` (omitted when there is none), and ``scene_input`` and
    ``prompt_input`` keep the user's own text when the scene or prompt was enhanced.

    An extension (``workflow`` ``"extend"``) records the action as sent, the ``source_video`` it continued and the
    new frames as ``extend_frames``; ``frame_count`` is the whole output clip and the size is the source's displayed one.
    An extension that continued from a point records the original clip as ``source_video`` and the snapped time of the
    last frame it kept as ``extend_from``; the temporary cut it was made from is not recorded.

    A clip that followed an audio track records its ``audio_track_path``, ``audio_track_start`` and ``audio_track_end``
    (where in the track the clip's sound ends), and ``generation.audio`` is true.
    """
    upscale = _drop_empty({"factor": request.upscale, "steps": request.upscale_steps}) if request.upscale else None
    extend = request.extend
    audio = _video_has_sound(request)
    payload = {
        "schema": VIDEO_CONFIG_SCHEMA,
        "workflow": _video_workflow(request),
        "prompt": _video_prompt(request, artifacts),
        **_video_scene_fields(request, artifacts),
        "model": _portable_path(request.model_name),
        "model_family": request.model_family,
        "seed": request.seed,
        "steps": request.steps,
        "width": extend.source.width if extend is not None else request.width,
        "height": extend.source.height if extend is not None else request.height,
        "ratio": None,
        "size": None,
        "frame_count": extend.source.frame_count + extend.frames if extend is not None else request.num_frames,
        "image_path": _portable_path(request.image_path),
        "lora": _format_loras(request.lora_paths, request.lora_weights),
        "generation": _drop_empty(
            {
                "time": _round_seconds(artifacts.generation_time),
                "upscale": upscale,
                "audio": audio,
                "output_format": request.output_format,
            }
        ),
    }
    if request.end_image_path:
        payload["end_image_path"] = _portable_path(request.end_image_path)
    payload.update(_audio_track_fields(request))
    if extend is not None:
        payload["source_video"] = _portable_path(str(extend.original_path))
        payload["extend_frames"] = extend.frames
        if extend.cut is not None:
            payload["extend_from"] = extend.cut.seconds
    payload.update(_enhance_fields(request.enhance_record))
    return _drop_unserializable(payload)


def build_audio_config_payload(request: AudioGenerationRequest, clip: AudioClip, *, parts: int | None, generation_time: float) -> dict[str, Any]:
    """Build the config payload embedded in a saved audio take.

    The top-level keys are the reusable settings: ``workflow`` and ``kind`` (``music``, ``speech`` or ``sound``), the
    ``model``, the ``prompt`` (the song's style, the user's full text or the sound's description, never the parts a long
    text was read in), the ``lyrics`` (music), the ``seconds`` the take was made at (the resolved length, for a sound or
    music of a model that sets its length) and the ``requested_seconds`` (only when the length was set, so Reuse leaves a length on Auto), the ``tempo`` and ``key`` of a song (each only when set), the ``sound_type``
    (sound), the ``speaker`` (the built-in voice of a speech take) with its delivery (``emotion``, ``manner``, ``pace``,
    ``pitch`` and ``expressiveness``), the ``voice_path`` and ``voice_text`` of a cloned voice, and the ``seed``. A take
    read in a built-in voice has no ``voice_*`` keys, and one that follows a recording has no ``speaker`` or delivery. A
    take whose text was written by the enhancer records how as ``enhance`` and the user's own text as ``prompt_input``,
    which is saved even when it is empty. The speaker's consent is never recorded. ``generation`` records how the file was made: the ``time`` it took, the actual ``duration`` in
    seconds, the number of ``parts`` of a speech take, ``cut`` when the model reached its length limit, and the ``notes`` the model left about the take.
    """
    voice = request.voice
    payload = {
        "schema": AUDIO_CONFIG_SCHEMA,
        "workflow": request.kind,
        "kind": request.kind,
        "model": request.preset,
        "prompt": request.prompt,
        "lyrics": request.lyrics or None,
        "seconds": request.seconds,
        "requested_seconds": request.requested_seconds,
        "tempo": request.tempo,
        "key": request.key,
        "sound_type": request.sound_type,
        "speaker": request.speaker,
        **(asdict(request.delivery) if request.delivery is not None else {}),
        "voice_path": _portable_path(voice.path) if voice is not None else None,
        "voice_text": voice.text if voice is not None else None,
        "seed": request.seed,
        "generation": _drop_empty(
            {
                "time": _round_seconds(generation_time),
                "duration": round(clip.seconds, 2),
                "parts": parts,
                "cut": True if clip.cut else None,
                "notes": list(clip.notes) or None,
            }
        ),
    }
    payload = _drop_empty(payload)
    if request.enhance_record:
        # Whether a take is enhanced is whether it records how; the user's own text is kept even when empty.
        payload.update(_enhance_fields(request.enhance_record))
        payload["prompt_input"] = request.prompt_input or ""
    return _drop_unserializable(payload)


def _audio_track_fields(request: VideoGenerationRequest) -> dict[str, Any]:
    """Return the payload keys of the audio track a clip followed, or none.

    ``audio_track_end`` is where in the track the clip's sound ends: its start plus the clip's frames over the frame
    rate, or for an extension its new frames over the source's frame rate. It is left out when the frame rate is unknown.
    """
    track = request.audio_track
    if track is None:
        return {}
    fields: dict[str, Any] = {"audio_track_path": _portable_path(track.path), "audio_track_start": track.start}
    end = request.audio_track_end
    if end is not None:
        fields["audio_track_end"] = end
    return fields


def build_video_upscale_payload(
    source_config: Mapping[str, Any] | None,
    *,
    source_path: str,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
    frame_count: int,
    audio: bool,
    upscale: Mapping[str, Any],
    generation_time: float,
) -> dict[str, Any]:
    """Build the config payload embedded in a video made by upscaling an existing video.

    It starts from the source's own recorded settings (*source_config*, the raw payload; ``None`` for a clip with
    none), so Reuse and the viewer show how the original was made, and replaces what the upscale changes: the
    workflow becomes ``upscale``, and the size and ``frame_count`` are the upscaled clip's. ``generation`` records
    the time, the *upscale* block and whether the clip has sound; ``source`` the clip it came from (``path``, its
    displayed ``width`` and ``height``) and ``workflow``. When the source was itself an upscale, the size and the
    workflow are the original's, recorded in its own ``source``, so Reuse fills in the original's size and workflow
    however many times the clip was upscaled. ``workflow`` is ``None`` for a clip with no recorded settings, which
    therefore cannot be reused.
    """
    origin = _upscale_origin(source_config)
    kept = {key: value for key, value in (source_config or {}).items() if key not in ("schema", "generation", "source")}
    payload = {
        **kept,
        "schema": VIDEO_CONFIG_SCHEMA,
        "workflow": UPSCALE_WORKFLOW,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "generation": _drop_empty({"time": _round_seconds(generation_time), "upscale": dict(upscale), "audio": audio}),
        "source": {
            "path": _portable_path(source_path),
            "width": optional_int(origin.get("width")) or source_width,
            "height": optional_int(origin.get("height")) or source_height,
            "workflow": original_video_workflow(source_config),
        },
    }
    return _drop_unserializable(payload)


def _upscale_origin(source_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the ``source`` block of *source_config* when it is an upscale's (it describes the original behind it), else an empty mapping."""
    if not source_config or optional_text(source_config.get("workflow")) != UPSCALE_WORKFLOW:
        return {}
    source = source_config.get("source")
    return source if isinstance(source, Mapping) else {}


def original_video_workflow(source_config: Mapping[str, Any] | None) -> str | None:
    """Return the workflow that made the original behind *source_config*: its own, or the one its upscale source recorded."""
    if not source_config:
        return None
    if optional_text(source_config.get("workflow")) == UPSCALE_WORKFLOW:
        return optional_text(_upscale_origin(source_config).get("workflow"))
    return optional_text(source_config.get("workflow"))


def _image_prompt_input(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> str | None:
    """Return the user's own prompt of an enhanced image: the text sent beside a Web UI rewrite, or the template of a job rewrite."""
    if request.prompt_input:
        return request.prompt_input
    return request.prompt if artifacts.metadata.get("enhanced_prompt") else None


def _enhance_fields(record: dict[str, Any] | None, *, prompt_input: str | None = None) -> dict[str, Any]:
    """Return the payload keys of an enhanced prompt: ``prompt_input`` (images) and ``enhance``, each only when known."""
    fields: dict[str, Any] = {}
    if optional_text(prompt_input) is not None:
        fields["prompt_input"] = prompt_input
    if record:
        fields["enhance"] = record
    return fields


def _video_has_sound(request: VideoGenerationRequest) -> bool:
    """Return whether the finished clip has sound; an audio track is the clip's sound, whatever the Audio switch said.

    ``video_extend`` is imported late because it imports this module.
    """
    track = request.audio_track is not None
    if request.extend is None:
        return track or not request.no_audio
    from zvisiongenerator.utils.video_extend import extend_join_has_sound

    return extend_join_has_sound(request.extend.source, no_audio=request.no_audio, track=track)


def _video_workflow(request: VideoGenerationRequest) -> str:
    """Return the video workflow name: ``extend``, ``img2vid`` or ``txt2vid``."""
    if request.extend is not None:
        return "extend"
    return "img2vid" if request.image_path else "txt2vid"


def _video_prompt(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> str:
    """Return the prompt a video records: the enhanced action, else the template; an extension records the action it sent.

    The action sent differs from the template when quoted sentences of a legacy source's prompt were stripped.
    """
    enhanced = artifacts.metadata.get("enhanced_prompt")
    if request.extend is not None and not enhanced:
        return artifacts.metadata.get("prompt_sent", request.prompt)
    return enhanced or request.prompt


def _video_scene_fields(request: VideoGenerationRequest, artifacts: VideoWorkingArtifacts) -> dict[str, str]:
    """Return the video payload's scene keys that apply: ``scene``, and the user's inputs when the text was enhanced.

    The inputs come from the request when its scene or prompt already holds a rewrite made before the job; otherwise,
    for a rewrite made by the job, they are the scene and prompt as written. A job that rewrote only the action has
    no ``enhanced_scene``: ``scene`` is the user's own and ``scene_input`` repeats it, which tells it from a rewrite
    of a blank scene (a ``scene`` with no ``scene_input``).
    """
    metadata = artifacts.metadata
    enhanced = bool(metadata.get("enhanced_prompt"))
    fields = {
        "scene": metadata.get("enhanced_scene") or metadata.get("scene"),
        "scene_input": request.scene_input or (request.scene if enhanced else None),
        "prompt_input": request.prompt_input or (request.prompt if enhanced else None),
    }
    return {key: text for key, value in fields.items() if (text := optional_text(value)) is not None}


def config_args(payload: dict[str, Any]) -> list[str]:
    """Return the ffmpeg output options that store a config payload as JSON in a container tag under zvisiongenerator.config.

    The tag is the same in every container ffmpeg writes tags to; MP4 also needs the extra option :func:`mp4_config_args` adds.
    """
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return ["-metadata", f"{_CONFIG_KEY}={value}"]


def mp4_config_args(payload: dict[str, Any]) -> list[str]:
    """Return the ffmpeg output options that store a config payload as JSON in MP4 metadata under zvisiongenerator.config."""
    return [*_MP4_TAG_OPTIONS, *config_args(payload)]


def embed_mp4_config(video_path: str | Path, payload: dict[str, Any]) -> None:
    """Embed a config payload as JSON in MP4 container metadata under zvisiongenerator.config.

    Uses ffmpeg to copy streams and attach the metadata key, replacing the original file.

    Raises:
        subprocess.CalledProcessError: If ffmpeg exits with a non-zero status.
    """
    video_path = Path(video_path)
    tmp = staging_path(video_path)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-c", "copy", *mp4_config_args(payload), str(tmp)],
            stdin=subprocess.DEVNULL,
            check=True,
            capture_output=True,
        )
        tmp.replace(video_path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def read_config(path: str | Path) -> dict[str, Any] | None:
    """Read the zvisiongenerator.config payload from a media file's container metadata, or None if absent.

    Raises:
        subprocess.CalledProcessError: If ffprobe cannot read the file.
    """
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )
    return config_from_probe(json.loads(result.stdout))


def config_from_probe(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the zvisiongenerator.config payload in ffprobe's JSON for a file (its ``format`` block), or None if absent.

    The tag name is matched without regard to case, because some tag editors change the case of tag names.
    """
    tags = data.get("format", {}).get("tags", {})
    raw = next((value for key, value in tags.items() if key.lower() == _CONFIG_KEY), None)
    if raw is None:
        return None
    return json.loads(raw)


def embed_png_config(png_info: PngInfo, payload: dict[str, Any]) -> None:
    """Embed a config payload as JSON in PNG metadata under zvisiongenerator.config."""
    png_info.add_text(_PNG_CONFIG_KEY, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def read_png_config(image_path: str | Path) -> dict[str, Any] | None:
    """Read the zvisiongenerator.config payload from PNG metadata, or None if absent."""
    from PIL import Image as _Image

    with _Image.open(image_path) as img:
        raw = img.info.get(_PNG_CONFIG_KEY)
    if raw is None:
        return None
    return json.loads(raw)


@dataclass(frozen=True)
class RecordedSettings:
    """Generation settings embedded in a saved image or video, as typed values; unrecorded ones are None."""

    workflow: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    model: str | None = None
    model_family: str | None = None
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    scheduler: str | None = None
    width: int | None = None
    height: int | None = None
    ratio: str | None = None
    size: str | None = None
    frame_count: int | None = None
    image_path: str | None = None
    # The image a video clip ends on (a first-and-last-frame clip).
    end_image_path: str | None = None
    image_strength: float | None = None
    # The references an image was made with, in prompt order.
    reference_images: tuple[str, ...] = ()
    # Video scene as generated, and the user's own scene and prompt when the recorded ones were enhanced.
    scene: str | None = None
    scene_input: str | None = None
    prompt_input: str | None = None
    # An extension: the clip it continued and the frames it added (``frame_count`` is the whole clip).
    source_video: str | None = None
    extend_frames: int | None = None
    # The time of the last frame of ``source_video`` an extension kept, when it continued from a point rather than the end.
    extend_from: float | None = None
    # The audio file a video followed, where in it the clip's sound started and where it ended.
    audio_track_path: str | None = None
    audio_track_start: float | None = None
    audio_track_end: float | None = None
    # ``path:weight`` entries joined by commas, as ``--lora`` takes them.
    lora: str | None = None
    # How the prompt was enhanced: ``{settings, model, tier, format}``; None when it was not (or the file predates it).
    enhance: dict[str, Any] | None = None
    # How the file was made: time, quantize, upscale and post-processing (empty for files that predate it).
    generation: dict[str, Any] = field(default_factory=dict)
    # The image an upscale or an edit started from: ``{path, width, height, workflow}`` (an edit's ``workflow`` is None).
    source: dict[str, Any] | None = None
    # What an edit did: ``{mode, extend}``, where ``extend`` is ``{side, amount}`` when it extended and else absent; None for any other file.
    edit: dict[str, Any] | None = None


def recorded_settings(config: dict[str, Any]) -> RecordedSettings:
    """Read the top-level settings of an embedded config payload; malformed values become None."""
    return RecordedSettings(
        workflow=optional_text(config.get("workflow")),
        prompt=optional_text(config.get("prompt")),
        negative_prompt=optional_text(config.get("negative_prompt")),
        model=_local_path(optional_text(config.get("model"))),
        model_family=optional_text(config.get("model_family")),
        seed=optional_int(config.get("seed")),
        steps=optional_int(config.get("steps")),
        guidance=optional_float(config.get("guidance")),
        scheduler=optional_text(config.get("scheduler")),
        width=optional_int(config.get("width")),
        height=optional_int(config.get("height")),
        ratio=optional_text(config.get("ratio")),
        size=optional_text(config.get("size")),
        frame_count=optional_int(config.get("frame_count")),
        image_path=_local_path(optional_text(config.get("image_path"))),
        end_image_path=_local_path(optional_text(config.get("end_image_path"))),
        image_strength=optional_float(config.get("image_strength")),
        reference_images=_local_paths(config.get("reference_images")),
        scene=optional_text(config.get("scene")),
        scene_input=optional_text(config.get("scene_input")),
        prompt_input=optional_text(config.get("prompt_input")),
        source_video=_local_path(optional_text(config.get("source_video"))),
        extend_frames=optional_int(config.get("extend_frames")),
        extend_from=optional_float(config.get("extend_from")),
        audio_track_path=_local_path(optional_text(config.get("audio_track_path"))),
        audio_track_start=optional_float(config.get("audio_track_start")),
        audio_track_end=optional_float(config.get("audio_track_end")),
        lora=_lora_string(config.get("lora")),
        generation=_generation_block(config.get("generation"), is_video=_is_video_config(config)),
        source=_source_block(config.get("source")),
        edit=_edit_block(config.get("edit")),
        enhance=_enhance_block(config.get("enhance"), is_video=_is_video_config(config)),
    )


def _enhance_block(value: Any, *, is_video: bool) -> dict[str, Any] | None:
    """Return a recorded ``enhance`` block normalized, or None when absent or malformed (an old or hand-edited file)."""
    try:
        return parse_enhance_record(value, mode="video" if is_video else "image")
    except ValueError:
        return None


def _lora_string(value: Any) -> str | None:
    """Return LoRAs as ``name[:weight]`` entries joined by commas; older files may record a list or a mapping."""
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        return _expand_lora_entries(value.strip()) or None
    items = value if isinstance(value, list) else [value]
    entries: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            entries.append(item.strip())
        elif isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip():
            weight = item.get("weight")
            entries.append(item["name"].strip() if weight in (None, "") else f"{item['name'].strip()}:{weight}")
    return _expand_lora_entries(",".join(entries)) or None


def _expand_lora_entries(text: str) -> str:
    """Expand a leading ``~`` in each comma-separated LoRA entry."""
    return ",".join(expand_home(entry) for entry in text.split(",")) if text else text


def _is_video_config(config: dict[str, Any]) -> bool:
    """Return whether an embedded config payload describes a video."""
    return optional_int(config.get("frame_count")) is not None or optional_text(config.get("workflow")) in ("txt2vid", "img2vid", "extend")


def _generation_block(value: Any, *, is_video: bool = False) -> dict[str, Any]:
    """Keep the known, well-typed keys of a recorded ``generation`` block; a video keeps the upscale method only when it is ``seedvr2`` or ``ltx``."""
    if not isinstance(value, dict):
        return {}
    generation: dict[str, Any] = {}
    for key in ("time", "sharpen", "contrast", "saturation"):
        number = optional_float(value.get(key))
        if number is not None:
            generation[key] = number
    quantize = optional_int(value.get("quantize"))
    if quantize is not None:
        generation["quantize"] = quantize
    if isinstance(value.get("audio"), bool):
        generation["audio"] = value["audio"]
    if value.get("transparent") is True:
        generation["transparent"] = True
    output_format = optional_text(value.get("output_format"))
    if output_format is not None:
        generation["output_format"] = output_format
    upscale = value.get("upscale")
    if isinstance(upscale, dict):
        coerced = {
            "factor": optional_int(upscale.get("factor")),
            "denoise": optional_float(upscale.get("denoise")),
            "steps": optional_int(upscale.get("steps")),
            "guidance": optional_float(upscale.get("guidance")),
            "pre_sharpen": optional_float(upscale.get("pre_sharpen")),
            "model": optional_text(upscale.get("model")),
            "quantize": optional_int(upscale.get("quantize")),
            "softness": optional_float(upscale.get("softness")),
            "seed": optional_int(upscale.get("seed")),
        }
        coerced = {key: item for key, item in coerced.items() if item is not None}
        if coerced and is_video:
            # A video upscaled from a file records its method; the in-run upscale of a generated clip records none.
            method = {"method": upscale["method"]} if upscale.get("method") in (UPSCALE_METHOD_SEEDVR2, UPSCALE_METHOD_LTX) else {}
            generation["upscale"] = {**method, **coerced}
        elif coerced:
            # Image files that predate the method were all refined by the image model.
            method = UPSCALE_METHOD_SEEDVR2 if upscale.get("method") == UPSCALE_METHOD_SEEDVR2 else UPSCALE_METHOD_CREATIVE
            generation["upscale"] = {"method": method, **coerced}
    return generation


def _source_block(value: Any) -> dict[str, Any] | None:
    """Return a recorded upscale source as ``{path, width, height, workflow}``, or None."""
    if not isinstance(value, dict):
        return None
    path = optional_text(value.get("path"))
    if path is None:
        return None
    return {
        "path": expand_home(path),
        "width": optional_int(value.get("width")),
        "height": optional_int(value.get("height")),
        "workflow": optional_text(value.get("workflow")),
    }


def _edit_block(value: Any) -> dict[str, Any] | None:
    """Return a recorded ``edit`` block as ``{mode}`` plus ``extend`` when it extended, or None when absent or malformed."""
    if not isinstance(value, dict):
        return None
    mode = optional_text(value.get("mode"))
    if mode not in _EDIT_MODES:
        return None
    block: dict[str, Any] = {"mode": mode}
    extend = value.get("extend")
    if isinstance(extend, dict):
        try:
            side, amount = parse_extend((extend.get("side") or "", extend.get("amount") or ""))
        except ValueError:
            return block
        block["extend"] = {"side": side, "amount": amount}
    return block


def image_prompt_text(image: Any) -> str | None:
    """Return the rendered prompt saved in an open PIL image (PNG ``Description`` or EXIF ImageDescription)."""
    return optional_text(image.info.get("Description") or image.getexif().get(EXIF_IMAGE_DESCRIPTION))


def optional_text(value: Any) -> str | None:
    """Return a recorded value as stripped text, or None when it is missing or blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_int(value: Any) -> int | None:
    """Return a recorded value as an int, or None when it is missing, blank, a bool or not a number."""
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def optional_float(value: Any) -> float | None:
    """Return a recorded value as a float, or None when it is missing, blank, a bool or not a number."""
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _build_loras(paths: list[str] | None, weights: list[float] | None) -> list[dict[str, Any]]:
    if not paths:
        return []
    return [
        {
            "name": display_stem(path),
            "path": collapse_home(path),
            "weight": weights[index] if weights is not None and index < len(weights) else 1.0,
        }
        for index, path in enumerate(paths)
    ]


def _portable_path(path: Any) -> Any:
    """Write a recorded path with the home directory as ``~``, so shared files do not reveal the username."""
    return collapse_home(path) if isinstance(path, str) else path


def _local_path(path: str | None) -> str | None:
    """Read a recorded path back with ``~`` expanded to this user's home directory."""
    return expand_home(path) if path is not None else None


def _local_paths(value: Any) -> tuple[str, ...]:
    """Read recorded paths back with ``~`` expanded; anything but a list of non-blank strings gives none."""
    if not isinstance(value, list):
        return ()
    return tuple(expand_home(item) for item in value if isinstance(item, str) and item.strip())


def _format_loras(paths: list[str] | None, weights: list[float] | None) -> str | None:
    loras = _build_loras(paths, weights)
    if not loras:
        return None
    return ",".join(f"{item['path']}:{item['weight']:g}" for item in loras)


def _edit_fields(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> dict[str, Any]:
    """Return the ``source`` and ``edit`` blocks of an edit: the picture it started from, what it did, and how far it extended."""
    source = artifacts.metadata["edit_source"]
    edit: dict[str, Any] = {"mode": "paint" if request.edit_paint is not None else "extend" if request.edit_extend else "whole"}
    if request.edit_extend:
        side, amount = request.edit_extend
        edit["extend"] = {"side": side, "amount": amount}
    return {"source": {**source, "path": _portable_path(source.get("path"))}, "edit": edit}


def _image_workflow(request: ImageGenerationRequest) -> str:
    if request.edit_source:
        return "edit"
    if request.upscale_source:
        return "upscale"
    return "img2img" if request.image_path else "txt2img"


def _image_generation_details(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> dict[str, Any]:
    if request.edit_source:
        # An edit runs no post-processing; pixels outside the paint stay the source's own. A cut-out stays one: its alpha is kept.
        cutout = artifacts.edit is not None and artifacts.edit.cutout is not None
        return _drop_empty({"time": _round_seconds(artifacts.generation_time), "quantize": request.quantize, "transparent": True if cutout else None})
    upscale = _upscale_details(request) if artifacts.was_upscaled and request.upscale_factor else None
    return _drop_empty(
        {
            "time": _round_seconds(artifacts.generation_time),
            "quantize": request.quantize,
            "upscale": upscale,
            "sharpen": _sharpen_amount(request, artifacts) if request.sharpen else None,
            "contrast": request.contrast_amount if request.contrast else None,
            "saturation": request.saturation_amount if request.saturation else None,
            "transparent": True if request.transparent else None,
        }
    )


def _upscale_details(request: ImageGenerationRequest) -> dict[str, Any]:
    """Return the recorded ``generation.upscale`` block: the method plus the settings that method used."""
    if request.upscale_method == UPSCALE_METHOD_SEEDVR2:
        return _drop_empty(
            {
                "method": UPSCALE_METHOD_SEEDVR2,
                "factor": request.upscale_factor,
                "model": request.upscale_model,
                "quantize": request.upscale_quantize,
                "softness": request.upscale_softness,
                "seed": request.seed,
            }
        )
    return _drop_empty(
        {
            "method": UPSCALE_METHOD_CREATIVE,
            "factor": request.upscale_factor,
            "denoise": request.upscale_denoise,
            "steps": request.upscale_steps,
            "guidance": request.upscale_guidance,
            "pre_sharpen": request.sharpen_amount_pre_upscale if request.upscale_sharpen else None,
        }
    )


def _sharpen_amount(request: ImageGenerationRequest, artifacts: ImageWorkingArtifacts) -> float:
    if request.sharpen_amount_override is not None:
        return request.sharpen_amount_override
    return request.sharpen_amount_upscaled if artifacts.was_upscaled else request.sharpen_amount_normal


def _round_seconds(seconds: float) -> float | None:
    return round(seconds, 1) if seconds > 0 else None


def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _drop_unserializable(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, default=str))
