"""Core abstractions — Backend Protocols, workflow types, and stage definitions."""

from __future__ import annotations

from zvisiongenerator.core.image_backend import ImageBackend
from zvisiongenerator.core.image_types import ImageGenerationRequest, ImageStage, ImageWorkingArtifacts
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_backend import VideoBackend
from zvisiongenerator.core.video_types import VideoGenerationRequest, VideoStage, VideoWorkingArtifacts
from zvisiongenerator.core.workflow import GenerationWorkflow

__all__ = [
    "GenerationWorkflow",
    "ImageBackend",
    "ImageGenerationRequest",
    "ImageStage",
    "ImageWorkingArtifacts",
    "StageOutcome",
    "VideoBackend",
    "VideoGenerationRequest",
    "VideoStage",
    "VideoWorkingArtifacts",
]
