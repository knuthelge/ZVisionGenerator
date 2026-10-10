"""VideoBackend Protocol — contract for platform-specific video inference engines.

Protocol requires name, load_model, text_to_video, image_to_video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from zvisiongenerator.utils.video_model_detect import VideoModelInfo


@runtime_checkable
class VideoBackend(Protocol):
    """Platform-specific video inference engine.

    Required interface:
    - name: str identifier ("ltx")
    - load_model(): load video model with backend-specific kwargs
    - text_to_video(): generate video from text prompt
    - image_to_video(): generate video from image + text prompt
    """

    name: str

    def load_model(
        self,
        model_path: str,
        **kwargs: Any,
    ) -> tuple[Any, VideoModelInfo]:
        """Load a video model. Returns (model_handle, video_model_info).

        The model_handle type varies by backend:
        - LTX: Backend-private handle (pipeline plus frame rate and upscale mode),
          reusable across generate calls. Callers pass it back unchanged.

        Backend-specific keyword arguments:
        - LTX: low_memory (bool), loras (list[tuple[str, float]]), upscale (bool).
          ``mode`` is accepted and ignored.
        """
        ...

    def text_to_video(
        self,
        model: Any,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        steps: int,
        output_path: str,
        step_callback: Any | None = None,
        **kwargs: Any,
    ) -> Path | None:
        """Generate video from text. Returns output path or None if failed."""
        ...

    def image_to_video(
        self,
        model: Any,
        image_path: str,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        seed: int,
        steps: int,
        output_path: str,
        step_callback: Any | None = None,
        **kwargs: Any,
    ) -> Path | None:
        """Generate video from image + text. Returns output path or None if failed."""
        ...
