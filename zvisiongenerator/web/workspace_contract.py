"""Own backend workflow values and static workspace capability metadata."""

from __future__ import annotations

from typing import Any


CANONICAL_WORKFLOW_VALUES = ("txt2img", "img2img", "txt2vid", "img2vid", "extend")
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
WORKFLOW_DEFINITIONS: dict[str, dict[str, Any]] = {
    "txt2img": {
        "mode": "image",
        "model_kind": "image",
        "visible_controls": [
            "workflow",
            "model",
            "quantize",
            "loras",
            "prompt_source",
            "prompt_inline",
            "prompt_enhance",
            "prompt_enhance_auto",
            "negative_prompt",
            "prompt_file_path",
            "prompt_file_option",
            "prompt_file_preview",
            "prompt_file_edit",
            "ratio",
            "size",
            "custom_dimensions",
            "runs",
            "steps",
            "guidance",
            "seed",
            "scheduler",
            "postprocess_sharpen",
            "postprocess_contrast",
            "postprocess_saturation",
            "image_upscale_enabled",
            "image_upscale_factor",
            "image_upscale_denoise",
            "image_upscale_steps",
            "image_upscale_guidance",
            "image_upscale_sharpen",
            "image_upscale_review",
        ],
        "supports_reference_image": False,
        "requires_reference_image": False,
        "requires_source_video": False,
        "clear_fields": ["image_path", "image_strength", "frames", "audio", "low_memory"],
    },
    "img2img": {
        "mode": "image",
        "model_kind": "image",
        "visible_controls": [
            "workflow",
            "model",
            "quantize",
            "loras",
            "prompt_source",
            "prompt_inline",
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
            "ratio",
            "size",
            "custom_dimensions",
            "runs",
            "steps",
            "guidance",
            "image_strength",
            "seed",
            "scheduler",
            "postprocess_sharpen",
            "postprocess_contrast",
            "postprocess_saturation",
            "image_upscale_enabled",
            "image_upscale_factor",
            "image_upscale_denoise",
            "image_upscale_steps",
            "image_upscale_guidance",
            "image_upscale_sharpen",
            "image_upscale_review",
        ],
        "supports_reference_image": True,
        "requires_reference_image": True,
        "requires_source_video": False,
        "clear_fields": ["frames", "audio", "low_memory"],
    },
    "txt2vid": {
        "mode": "video",
        "model_kind": "video",
        "visible_controls": [
            "workflow",
            "model",
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
        "supports_reference_image": False,
        "requires_reference_image": False,
        "requires_source_video": False,
        "clear_fields": [
            "negative_prompt",
            "guidance",
            "image_path",
            "image_strength",
            "quantize",
            "sharpen_enabled",
            "sharpen_amount",
            "contrast_enabled",
            "contrast_amount",
            "saturation_enabled",
            "saturation_amount",
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
        "supports_reference_image": True,
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
            "runs",
            "steps",
            "seed",
            "audio",
            "low_memory",
        ],
        "supports_reference_image": False,
        "requires_reference_image": False,
        "requires_source_video": True,
        "clear_fields": [
            "negative_prompt",
            "guidance",
            "image_path",
            "image_strength",
            "quantize",
            "sharpen_enabled",
            "sharpen_amount",
            "contrast_enabled",
            "contrast_amount",
            "saturation_enabled",
            "saturation_amount",
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
    """Return the media mode for a canonical workflow."""
    return "video" if workflow in {"txt2vid", "img2vid", "extend"} else "image"


def build_workflow_contract(*, video_audio: bool = True, video_upscale_review: bool = True, video_extend: bool = True) -> dict[str, Any]:
    """Build the backend-owned workflow and control visibility contract.

    Args:
        video_audio: Whether the video backend writes an audio track; without one the Audio control is hidden.
        video_upscale_review: Whether the video backend can pause for review after the half-size render; without it the video Review control is hidden.
        video_extend: Whether the video backend can extend a clip; without it the ``extend`` workflow is left out.
    """
    hidden = frozenset(name for name, hide in (("audio", not video_audio), ("video_upscale_review", not video_upscale_review)) if hide)
    names = [name for name in CANONICAL_WORKFLOW_VALUES if video_extend or name != "extend"]
    return {
        "values": names,
        "definitions": {name: _definition_for_platform(WORKFLOW_DEFINITIONS[name], hidden=hidden) for name in names},
        "field_precedence": {
            "defaults": ["cli", "model_variant", "model_family", "global"],
            "dimensions": "explicit_width_height_overrides_ratio_size",
        },
    }


def _definition_for_platform(definition: dict[str, Any], *, hidden: frozenset[str]) -> dict[str, Any]:
    """Copy a workflow definition, dropping the *hidden* controls the platform's video backend cannot honour."""
    if not hidden:
        return dict(definition)
    return {**definition, "visible_controls": [control for control in definition["visible_controls"] if control not in hidden]}
