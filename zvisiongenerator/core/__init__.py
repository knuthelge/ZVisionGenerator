"""Core abstractions — Backend Protocols, workflow types, and stage definitions."""

from __future__ import annotations

from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageStage, ImageWorkingArtifacts, JobPrompt
from zvisiongenerator.core.job_plan import EnhanceStatus, IterationPlan, JobPlan
from zvisiongenerator.core.seedvr2 import SeedVR2Upscaler
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_backend import VideoBackend
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoStage, VideoWorkingArtifacts
from zvisiongenerator.core.workflow import GenerationWorkflow

__all__ = [
    "EnhanceStatus",
    "GenerationWorkflow",
    "ImageBackend",
    "ImageGenerationRequest",
    "ImageStage",
    "ImageWorkingArtifacts",
    "IterationPlan",
    "JobPlan",
    "JobPrompt",
    "SeedVR2Upscaler",
    "StageOutcome",
    "VideoBackend",
    "VideoGenerationRequest",
    "VideoStage",
    "VideoWorkingArtifacts",
]
