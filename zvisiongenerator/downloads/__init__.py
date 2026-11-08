"""Hugging Face model downloads: targets, the declared downloads, offline presence, dry-run checks, free-space estimates, Hub refusal messages, child-process runs and the ensure step.

The planners that decide which targets a model needs live in :mod:`zvisiongenerator.downloads.plans`, which is not imported here: they read the
model facts of ``utils``, which read the declarations of this package.
"""

from __future__ import annotations

from zvisiongenerator.downloads.check import check_target
from zvisiongenerator.downloads.cli_report import CliDownloadReporter, eta_text
from zvisiongenerator.downloads.child import ChildResult, child_env, run_in_child
from zvisiongenerator.downloads.declaration import companion_targets, declared_repo_ids, model_targets, role_targets
from zvisiongenerator.downloads.ensure import DETACHED, READY, STOPPED, UNCHECKED, DownloadReport, ensure_downloaded, require_checked
from zvisiongenerator.downloads.flights import DownloadFlights, Flight
from zvisiongenerator.downloads.hub_access import LOGIN_SOURCE, TOKEN_ENV_VARS, describe_refusal, refusal_message, token_source
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
    "check_target",
    "child_env",
    "companion_targets",
    "declared_repo_ids",
    "describe_refusal",
    "eta_text",
    "ensure_downloaded",
    "is_literal",
    "make_byte_sink",
    "model_targets",
    "progress_from_counts",
    "refusal_message",
    "remove_partial_files",
    "require_checked",
    "role_targets",
    "run_in_child",
    "space_need",
    "target_present_offline",
    "token_source",
]
