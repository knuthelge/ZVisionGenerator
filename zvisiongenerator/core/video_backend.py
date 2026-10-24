"""VideoBackend Protocol — contract for platform-specific video inference engines.

Protocol requires name, load_model, text_to_video, image_to_video. ``VideoExtender`` is a separate, optional
capability that both LTX backends and the MiniMax H3 backends provide.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from zvisiongenerator.utils.video_model_detect import VideoModelInfo

if TYPE_CHECKING:
    from zvisiongenerator.core.video_types import ExtendSegment, ExtendSource


@runtime_checkable
class VideoBackend(Protocol):
    """Platform-specific video inference engine.

    Required interface:
    - name: str identifier ("ltx", "h3")
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
        - H3: Backend-private handle (the resolved files, the stored 4-bit transformer copy and the load mode).
          Each stage of a clip loads and frees its own models, so the handle holds no weights.

        Backend-specific keyword arguments:
        - LTX: low_memory (bool), loras (list[tuple[str, float]]), upscale (bool),
          latent_upsampler (tuple[str, str] | None, diffusers only).
          ``mode`` is accepted and ignored.
        - H3 (Windows/Linux): mode ("t2v", "i2v" or "extend"), loras (list[tuple[str, float]]), upscale (bool),
          preset (the ``H3Preset``, required), model_name and low_memory (both accepted and ignored). Any other
          keyword raises ``ValueError`` naming it.
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
        """Generate video from text. Returns output path or None if failed.

        Optional keyword arguments:
        - ``stage1_steps``: run the two-stage upscale with this many half-resolution steps.
        - ``review``: ``Callable[[Path], None]``, valid only with ``stage1_steps``. The backend decodes the
          half-resolution result to a temporary video file, calls ``review(path)`` and continues with the upscale
          when it returns; the file is deleted afterwards. ``review`` may raise to end the generation. A backend
          that cannot pause after the half-size render raises ``ValueError`` (see ``supports_video_upscale_review``).
        - H3: ``steps`` must be a step count of the preset's schedules (4 or 8); ``stage1_steps`` is the same count
          for the half-size render, and the full-size result comes from the SeedVR2 video upscale.
        """
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
        """Generate video from image + text. Returns output path or None if failed.

        Takes the same optional ``stage1_steps`` and ``review`` keyword arguments as ``text_to_video``. H3 also
        takes ``end_image``, the path of a last-frame image: the clip then runs from ``image_path`` to it.
        """
        ...


class VideoExtender(Protocol):
    """Optional capability: continue an existing clip from its last frames.

    Deliberately not ``runtime_checkable`` and not part of ``VideoBackend``: a backend offers it only where
    ``supports_video_extend()`` says so. Callers gate on that and look the method up on the backend.
    """

    def extend_video(
        self,
        model: Any,
        source: ExtendSource,
        prompt: str,
        *,
        extend_frames: int,
        seed: int,
        steps: int,
        segment_dir: str | Path,
        audio_output: bool,
        step_callback: Any | None = None,
    ) -> ExtendSegment | None:
        """Generate the segment that follows ``source``; returns its video and optional audio files, or None if failed."""
        ...
