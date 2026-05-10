"""Own backend workflow values and static workspace capability metadata."""

from __future__ import annotations

from typing import Any


CANONICAL_WORKFLOW_VALUES = ("txt2img", "img2img", "txt2vid", "img2vid")
PROMPT_SOURCE_VALUES = ("inline", "file")
DEFAULT_PROMPT_SOURCE = "inline"
PROMPT_FILE_CONTRACT = {
    "accepted_extensions": [".yaml", ".yml"],
    "browse_kind": "existing_file",
    "selection_required": True,
    "trust_boundary": {
        "scope": "server_host_only",
        "manual_entry": "submitted_value_kept_until_backend_validation",
        "picker": "server_host_native_picker",
        "read_write": "existing_yaml_files_only",
    },
    "help": {
        "path": "Enter or browse for a prompt YAML file on the machine running the Web UI host. The submitted value stays visible until the backend accepts or rejects it.",
        "editor": "Reads and writes happen on the machine running the Web UI host. Saves replace the file atomically only after YAML validation succeeds.",
        "option_required": "Select an active prompt option before generating.",
        "option_optional": "Select an active prompt option from the file.",
        "empty_options": "This prompt file has no active prompt options.",
        "stale_selection": "The previously selected prompt option is no longer active.",
        "loaded": "Prompt file loaded.",
        "saved": "Prompt file saved.",
        "ignored_negative_video": "Negative prompt entries are ignored for video workflows.",
        "ignored_negative_unsupported": "The current image model ignores negative prompt entries.",
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
        ],
        "supports_reference_image": False,
        "requires_reference_image": False,
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
        ],
        "supports_reference_image": True,
        "requires_reference_image": True,
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
            "prompt_inline",
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
        ],
        "supports_reference_image": False,
        "requires_reference_image": False,
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
            "prompt_inline",
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
        ],
        "supports_reference_image": True,
        "requires_reference_image": True,
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
    return "video" if workflow in {"txt2vid", "img2vid"} else "image"


def build_workflow_contract() -> dict[str, Any]:
    """Build the backend-owned workflow and control visibility contract."""
    return {
        "values": list(CANONICAL_WORKFLOW_VALUES),
        "definitions": {name: dict(value) for name, value in WORKFLOW_DEFINITIONS.items()},
        "field_precedence": {
            "defaults": ["cli", "model_variant", "model_family", "global"],
            "dimensions": "explicit_width_height_overrides_ratio_size",
        },
    }
