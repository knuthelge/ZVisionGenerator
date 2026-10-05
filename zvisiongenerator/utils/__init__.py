"""Utility modules — config, filenames, model detection, prompt handling."""

from __future__ import annotations

from .alignment import round_to_alignment
from .config import load_config, resolve_defaults, resolve_video_defaults, validate_scheduler
from .console import format_generation_info
from .ffmpeg import ensure_ffmpeg, strip_audio
from .filename import generate_filename, unique_output_path
from .image_model_detect import ImageModelInfo, detect_image_model
from .interactive import SkipSignal
from .lora import parse_lora_arg
from .paths import (
    HuggingFaceRepoReference,
    display_basename,
    display_stem,
    get_ziv_data_dir,
    is_explicit_local_path,
    is_huggingface_repo_id,
    is_remote_lora_reference,
    parse_huggingface_repo_reference,
    resolve_lora_path,
    resolve_model_path,
)
from .platform import AliasMap, AliasValue, PlatformInfo, get_platform_info, resolve_alias
from .provenance import (
    EXIF_IMAGE_DESCRIPTION,
    IMAGE_CONFIG_SCHEMA,
    RecordedSettings,
    VIDEO_CONFIG_SCHEMA,
    build_image_config_payload,
    build_video_config_payload,
    embed_mp4_config,
    embed_png_config,
    image_prompt_text,
    optional_float,
    optional_int,
    optional_text,
    read_mp4_config,
    read_png_config,
    recorded_settings,
)
from .prompt_compose import expand_random_choices
from .prompt_enhance import EnhanceResult, EnhanceSettings, enhance_prompt, matrix_contract, parse_enhance_entry, parse_enhance_spec
from .prompts import load_prompts_file
from .video_model_detect import VideoModelInfo, detect_video_model

__all__ = [
    "ImageModelInfo",
    "AliasMap",
    "AliasValue",
    "HuggingFaceRepoReference",
    "EnhanceResult",
    "EnhanceSettings",
    "EXIF_IMAGE_DESCRIPTION",
    "IMAGE_CONFIG_SCHEMA",
    "PlatformInfo",
    "RecordedSettings",
    "VIDEO_CONFIG_SCHEMA",
    "SkipSignal",
    "VideoModelInfo",
    "build_image_config_payload",
    "build_video_config_payload",
    "detect_image_model",
    "detect_video_model",
    "display_basename",
    "display_stem",
    "embed_mp4_config",
    "embed_png_config",
    "enhance_prompt",
    "ensure_ffmpeg",
    "expand_random_choices",
    "format_generation_info",
    "generate_filename",
    "get_platform_info",
    "get_ziv_data_dir",
    "image_prompt_text",
    "is_explicit_local_path",
    "is_huggingface_repo_id",
    "is_remote_lora_reference",
    "load_config",
    "load_prompts_file",
    "matrix_contract",
    "optional_float",
    "optional_int",
    "optional_text",
    "parse_huggingface_repo_reference",
    "parse_enhance_entry",
    "parse_enhance_spec",
    "parse_lora_arg",
    "read_mp4_config",
    "read_png_config",
    "recorded_settings",
    "resolve_alias",
    "resolve_defaults",
    "resolve_lora_path",
    "resolve_model_path",
    "resolve_video_defaults",
    "round_to_alignment",
    "strip_audio",
    "unique_output_path",
    "validate_scheduler",
]
