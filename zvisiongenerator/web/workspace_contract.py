"""Own backend workflow values and static workspace capability metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zvisiongenerator.utils.audio_preset import AUDIO_KINDS

if TYPE_CHECKING:
    from collections.abc import Collection

    from zvisiongenerator.core.video_family import VideoCapabilities


# One repeated form field holds every reference (a file or a host path per row), so the rows keep their order.
REFERENCE_FIELD = "reference"

AUDIO_WORKFLOW_VALUES = AUDIO_KINDS
"""The workflows of the audio mode, one per kind of audio (a workflow is named after its kind); the contract lists those a runnable model makes."""
CANONICAL_WORKFLOW_VALUES = ("txt2img", "img2img", "edit", "txt2vid", "img2vid", "extend", *AUDIO_WORKFLOW_VALUES)
PROMPT_SOURCE_VALUES = ("inline", "file")
DEFAULT_PROMPT_SOURCE = "inline"
PROMPT_FILE_CONTRACT = {
    "accepted_extensions": [".yaml", ".yml"],
    "selection_required": True,
    "trust_boundary": {
        "scope": "server_host_only",
        "manual_entry": "submitted_value_kept_until_backend_validation",
        "picker": "in_app_browser",
        "read_write": "yaml_files_only",
    },
    "help": {
        "path": "Enter or browse for a prompt YAML file on the machine running the Web UI host. The submitted value stays visible until the backend accepts or rejects it.",
        "option_required": "Select one or more prompts to run together. Runs applies to each selected prompt.",
        "option_optional": "Select prompts from the file.",
        "empty_options": "This prompt file has no active prompt options.",
        "stale_selection": "Some selected prompts are no longer active and have been unchecked.",
        "loaded": "Prompt file loaded.",
        "ignored_negative_video": "Negative prompt entries are ignored for video workflows.",
        "ignored_negative_unsupported": "The current image model ignores negative prompt entries.",
        "ignored_scene_image": "Scene entries are ignored for image workflows.",
    },
}

# The controls of the last-frame slot. Only a family that declares ``end_frame`` offers them.
END_FRAME_CONTROLS = ("end_image", "end_image_path", "end_image_clear")

# The controls of a speech model's Voice section. Only a model that can follow a voice offers them.
VOICE_CONTROLS = ("voice", "voice_path", "voice_text", "voice_consent")

# The controls of the Audio track section. Only a family that declares ``audio_track`` offers them.
AUDIO_TRACK_CONTROLS = ("audio_track", "audio_track_path", "audio_track_start")

WORKFLOW_DEFINITIONS: dict[str, dict[str, Any]] = {
    "txt2img": {
        "mode": "image",
        "model_kind": "image",
        "visible_controls": [
            "workflow",
            "model",
            "preset",
            "quantize",
            "loras",
            "prompt_source",
            "prompt_inline",
            "json_prompt",
            "prompt_enhance",
            "prompt_enhance_auto",
            "negative_prompt",
            "prompt_file_path",
            "prompt_file_option",
            "prompt_file_preview",
            "prompt_file_edit",
            "reference_image",
            "reference_image_path",
            "reference_image_clear",
            "references",
            "references_clear",
            "ratio",
            "size",
            "custom_dimensions",
            "runs",
            "steps",
            "guidance",
            "seed",
            "first_sigma",
            "scheduler",
            "postprocess_sharpen",
            "postprocess_contrast",
            "postprocess_saturation",
            "postprocess_transparent",
            "image_upscale_enabled",
            "image_upscale_factor",
            "image_upscale_denoise",
            "image_upscale_steps",
            "image_upscale_guidance",
            "image_upscale_sharpen",
            "image_upscale_review",
        ],
        "requires_reference_image": False,
        "requires_source_video": False,
        "clear_fields": ["frames", "audio", "low_memory"],
    },
    "img2img": {
        "mode": "image",
        "model_kind": "image",
        "visible_controls": [
            "workflow",
            "model",
            "preset",
            "quantize",
            "loras",
            "prompt_source",
            "prompt_inline",
            "json_prompt",
            "prompt_enhance",
            "prompt_enhance_auto",
            "negative_prompt",
            "prompt_file_path",
            "prompt_file_option",
            "prompt_file_preview",
            "prompt_file_edit",
            "reference_image",
            "reference_image_path",
            "reference_image_clear",
            "references",
            "references_clear",
            "ratio",
            "size",
            "custom_dimensions",
            "runs",
            "steps",
            "guidance",
            "image_strength",
            "seed",
            "first_sigma",
            "scheduler",
            "postprocess_sharpen",
            "postprocess_contrast",
            "postprocess_saturation",
            "postprocess_transparent",
            "image_upscale_enabled",
            "image_upscale_factor",
            "image_upscale_denoise",
            "image_upscale_steps",
            "image_upscale_guidance",
            "image_upscale_sharpen",
            "image_upscale_review",
        ],
        "requires_reference_image": True,
        "requires_source_video": False,
        "clear_fields": ["frames", "audio", "low_memory"],
    },
    "edit": {
        "mode": "image",
        "model_kind": "image",
        "visible_controls": ["workflow", "model", "quantize", "loras", "prompt_inline", "runs", "steps", "seed"],
        "requires_reference_image": False,
        "requires_source_video": False,
        "clear_fields": [],
    },
    "txt2vid": {
        "mode": "video",
        "model_kind": "video",
        "visible_controls": [
            "workflow",
            "model",
            "preset",
            "loras",
            "prompt_source",
            "prompt_scene",
            "prompt_inline",
            "prompt_enhance",
            "prompt_enhance_auto",
            "prompt_file_path",
            "prompt_file_option",
            "prompt_file_preview",
            "prompt_file_edit",
            "reference_image",
            "reference_image_path",
            "reference_image_clear",
            *END_FRAME_CONTROLS,
            *AUDIO_TRACK_CONTROLS,
            "ratio",
            "size",
            "custom_dimensions",
            "runs",
            "frame_count",
            "steps",
            "seed",
            "audio",
            "low_memory",
            "video_upscale_enabled",
            "video_upscale_factor",
            "video_upscale_review",
        ],
        "requires_reference_image": False,
        "requires_source_video": False,
        "clear_fields": [
            "negative_prompt",
            "guidance",
            "quantize",
            "sharpen_enabled",
            "sharpen_amount",
            "contrast_enabled",
            "contrast_amount",
            "saturation_enabled",
            "saturation_amount",
            "transparent",
            "upscale",
            "upscale_denoise",
            "upscale_steps",
            "upscale_guidance",
            "upscale_sharpen",
            "upscale_save_pre",
        ],
    },
    "img2vid": {
        "mode": "video",
        "model_kind": "video",
        "visible_controls": [
            "workflow",
            "model",
            "preset",
            "loras",
            "prompt_source",
            "prompt_scene",
            "prompt_inline",
            "prompt_enhance",
            "prompt_enhance_auto",
            "prompt_file_path",
            "prompt_file_option",
            "prompt_file_preview",
            "prompt_file_edit",
            "reference_image",
            "reference_image_path",
            "reference_image_clear",
            *END_FRAME_CONTROLS,
            *AUDIO_TRACK_CONTROLS,
            "ratio",
            "size",
            "custom_dimensions",
            "runs",
            "frame_count",
            "steps",
            "seed",
            "audio",
            "low_memory",
            "video_upscale_enabled",
            "video_upscale_factor",
            "video_upscale_review",
        ],
        "requires_reference_image": True,
        "requires_source_video": False,
        "clear_fields": [
            "negative_prompt",
            "guidance",
            "quantize",
            "sharpen_enabled",
            "sharpen_amount",
            "contrast_enabled",
            "contrast_amount",
            "saturation_enabled",
            "saturation_amount",
            "transparent",
            "upscale",
            "upscale_denoise",
            "upscale_steps",
            "upscale_guidance",
            "upscale_sharpen",
            "upscale_save_pre",
        ],
    },
    "extend": {
        "mode": "video",
        "model_kind": "video",
        "visible_controls": [
            "workflow",
            "model",
            "preset",
            "loras",
            "prompt_source",
            "prompt_scene",
            "prompt_inline",
            "prompt_enhance",
            "prompt_enhance_auto",
            "prompt_file_path",
            "prompt_file_option",
            "prompt_file_preview",
            "prompt_file_edit",
            "source_video",
            "source_video_path",
            "source_video_clear",
            "extend_length",
            *AUDIO_TRACK_CONTROLS,
            "runs",
            "steps",
            "seed",
            "audio",
            "low_memory",
        ],
        "requires_reference_image": False,
        "requires_source_video": True,
        "clear_fields": [
            "negative_prompt",
            "guidance",
            "quantize",
            "sharpen_enabled",
            "sharpen_amount",
            "contrast_enabled",
            "contrast_amount",
            "saturation_enabled",
            "saturation_amount",
            "transparent",
            "upscale",
            "upscale_denoise",
            "upscale_steps",
            "upscale_guidance",
            "upscale_sharpen",
            "upscale_save_pre",
            "ratio",
            "size",
            "width",
            "height",
            "frames",
            "video_upscale_enabled",
            "video_upscale_factor",
        ],
    },
}


def _audio_definition(*controls: str) -> dict[str, Any]:
    """Return the definition of an audio workflow that shows *controls* between the model and the run controls."""
    return {
        "mode": "audio",
        "model_kind": "audio",
        "visible_controls": ["workflow", "model", *controls, "runs", "seed"],
        "requires_reference_image": False,
        "requires_source_video": False,
        "clear_fields": [],
    }


AUDIO_WORKFLOW_DEFINITIONS: dict[str, dict[str, Any]] = {
    "music": _audio_definition("music_style", "lyrics", "tempo", "key", "length"),
    "speech": _audio_definition("speech_text", *VOICE_CONTROLS, "delivery"),
    "sound": _audio_definition("sound_prompt", "sound_type", "length"),
}
WORKFLOW_DEFINITIONS.update(AUDIO_WORKFLOW_DEFINITIONS)

# The workflow controls a video family loses when it lacks a capability, by capability name.
_CONTROLS_WITHOUT_CAPABILITY = {
    "lora": ("loras",),
    "upscale": ("video_upscale_enabled", "video_upscale_factor", "video_upscale_review"),
    "upscale_review": ("video_upscale_review",),
    "low_memory": ("low_memory",),
    "audio": ("audio",),
    "end_frame": END_FRAME_CONTROLS,
    "audio_track": AUDIO_TRACK_CONTROLS,
}


def _hidden_controls(capabilities: object, table: dict[str, tuple[str, ...]]) -> list[str]:
    """Return the control ids in *table* whose capability *capabilities* lacks, sorted."""
    return sorted({control for name, controls in table.items() if not getattr(capabilities, name) for control in controls})


def hidden_video_controls(capabilities: VideoCapabilities) -> list[str]:
    """Return the workflow controls a video model does not have, sorted, from what it cannot do on this platform."""
    return _hidden_controls(capabilities, _CONTROLS_WITHOUT_CAPABILITY)


def canonicalize_workflow(value: Any, *, fallback: str | None = None) -> str | None:
    """Return a canonical workflow value, or fallback when value is absent or unsupported."""
    if value is None:
        return fallback
    workflow = str(value).strip()
    return workflow if workflow in CANONICAL_WORKFLOW_VALUES else fallback


def default_workflow_for_mode(mode: str) -> str:
    """Return the default workflow for an image/video submission mode."""
    return "txt2vid" if mode == "video" else "txt2img"


def workflow_mode(workflow: str) -> str:
    """Return the media mode (``image``, ``video`` or ``audio``) for a workflow."""
    if workflow in AUDIO_WORKFLOW_VALUES:
        return "audio"
    return "video" if workflow in {"txt2vid", "img2vid", "extend"} else "image"


def build_workflow_contract(*, video_audio: bool = True, video_upscale_review: bool = True, video_extend: bool = True, audio_kinds: Collection[str] = ()) -> dict[str, Any]:
    """Build the backend-owned workflow and control visibility contract.

    Args:
        video_audio: Whether the video backend writes an audio track; without one the Audio control is hidden.
        video_upscale_review: Whether the video backend can pause for review after the half-size render; without it the video Review control is hidden.
        video_extend: Whether the video backend can extend a clip; without it the ``extend`` workflow is left out.
        audio_kinds: The kinds (``music``, ``speech``, ``sound``) that have a model which runs on this machine. Only their
            workflows are listed; with none, no audio workflow is listed and the Audio tab stays hidden.
    """
    hidden = frozenset(name for name, hide in (("audio", not video_audio), ("video_upscale_review", not video_upscale_review)) if hide)
    names = [name for name in CANONICAL_WORKFLOW_VALUES if _is_listed(name, video_extend=video_extend, audio_kinds=audio_kinds)]
    return {
        "values": names,
        "definitions": {name: _definition_for_platform(WORKFLOW_DEFINITIONS[name], hidden=hidden) for name in names},
        "field_precedence": {
            "defaults": ["cli", "model_variant", "model_family", "global"],
            "dimensions": "explicit_width_height_overrides_ratio_size",
        },
    }


def _is_listed(workflow: str, *, video_extend: bool, audio_kinds: Collection[str]) -> bool:
    """Return whether the contract lists *workflow*: an extension needs a backend that extends, an audio workflow a runnable model of its kind."""
    if workflow == "extend":
        return video_extend
    if workflow in AUDIO_WORKFLOW_VALUES:
        return workflow in audio_kinds
    return True


def _definition_for_platform(definition: dict[str, Any], *, hidden: frozenset[str]) -> dict[str, Any]:
    """Copy a workflow definition, dropping the *hidden* controls the platform's video backend cannot honour."""
    if not hidden:
        return dict(definition)
    return {**definition, "visible_controls": [control for control in definition["visible_controls"] if control not in hidden]}
