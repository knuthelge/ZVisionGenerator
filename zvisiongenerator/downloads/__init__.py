"""Hugging Face model downloads: targets, offline presence, dry-run checks, free-space estimates, child-process runs and the ensure step."""

from __future__ import annotations

from zvisiongenerator.downloads.check import check_target
from zvisiongenerator.downloads.child import ChildResult, child_env, run_in_child
from zvisiongenerator.downloads.ensure import DETACHED, READY, STOPPED, UNCHECKED, DownloadReport, ensure_downloaded
from zvisiongenerator.downloads.flights import DownloadFlights, Flight
from zvisiongenerator.downloads.plans import enhancer_plan, image_plan, instant_plan, seedvr2_plan, video_plan
from zvisiongenerator.downloads.presence import remove_partial_files, target_present_offline
from zvisiongenerator.downloads.progress import ByteCounts, SpeedMeter, make_byte_sink, progress_from_counts
from zvisiongenerator.downloads.space import SpaceNeed, space_need
from zvisiongenerator.downloads.target import DownloadProgress, DownloadTarget, TargetCheck, is_literal

__all__ = [
    "DETACHED",
    "READY",
    "STOPPED",
    "UNCHECKED",
    "ByteCounts",
    "ChildResult",
    "DownloadFlights",
    "DownloadProgress",
    "DownloadReport",
    "DownloadTarget",
    "Flight",
    "SpaceNeed",
    "SpeedMeter",
    "TargetCheck",
    "check_target",
    "child_env",
    "enhancer_plan",
    "ensure_downloaded",
    "image_plan",
    "instant_plan",
    "is_literal",
    "make_byte_sink",
    "progress_from_counts",
    "remove_partial_files",
    "run_in_child",
    "seedvr2_plan",
    "space_need",
    "target_present_offline",
    "video_plan",
]
