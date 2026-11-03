"""Hugging Face model downloads: targets, offline presence, dry-run checks, free-space estimates, Hub refusal messages, child-process runs and the ensure step."""

from __future__ import annotations

from zvisiongenerator.downloads.check import check_target
from zvisiongenerator.downloads.cli_report import CliDownloadReporter, eta_text
from zvisiongenerator.downloads.child import ChildResult, child_env, run_in_child
from zvisiongenerator.downloads.ensure import DETACHED, READY, STOPPED, UNCHECKED, DownloadReport, ensure_downloaded, require_checked
from zvisiongenerator.downloads.flights import DownloadFlights, Flight
from zvisiongenerator.downloads.hub_access import LOGIN_SOURCE, TOKEN_ENV_VARS, describe_refusal, refusal_message, token_source
from zvisiongenerator.downloads.plans import audio_plan, built_model_plan, enhancer_plan, image_plan, matte_plan, seedvr2_plan, video_model_downloaded, video_plan
from zvisiongenerator.downloads.presence import remove_partial_files, target_present_offline
from zvisiongenerator.downloads.progress import ByteCounts, SpeedMeter, make_byte_sink, progress_from_counts
from zvisiongenerator.downloads.space import SpaceNeed, space_need
from zvisiongenerator.downloads.target import DownloadProgress, DownloadTarget, TargetCheck, is_literal

__all__ = [
    "DETACHED",
    "LOGIN_SOURCE",
    "READY",
    "STOPPED",
    "TOKEN_ENV_VARS",
    "UNCHECKED",
    "ByteCounts",
    "ChildResult",
    "CliDownloadReporter",
    "DownloadFlights",
    "DownloadProgress",
    "DownloadReport",
    "DownloadTarget",
    "Flight",
    "SpaceNeed",
    "SpeedMeter",
    "TargetCheck",
    "audio_plan",
    "built_model_plan",
    "check_target",
    "child_env",
    "describe_refusal",
    "enhancer_plan",
    "eta_text",
    "ensure_downloaded",
    "image_plan",
    "is_literal",
    "make_byte_sink",
    "matte_plan",
    "progress_from_counts",
    "refusal_message",
    "remove_partial_files",
    "require_checked",
    "run_in_child",
    "seedvr2_plan",
    "space_need",
    "target_present_offline",
    "token_source",
    "video_model_downloaded",
    "video_plan",
]
