"""The keyword arguments the video stages send must bind to both MiniMax H3 backends' method signatures.

The stages call ``text_to_video``, ``image_to_video`` and ``extend_video`` by keyword. A spec'd mock of each backend
class rejects a keyword its method does not take, so a stage that starts sending a new one fails here instead of on a
Mac or a GPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

import pytest

from zvisiongenerator.backends.cuda.video.h3.backend import H3VideoBackend
from zvisiongenerator.backends.mac.h3_video import H3MlxVideoBackend
from zvisiongenerator.core.types import StageOutcome
from zvisiongenerator.core.video_types import ExtendSource, ExtendSpec, VideoGenerationRequest, VideoWorkingArtifacts
from zvisiongenerator.workflows import video_stages

_BACKENDS = [pytest.param(H3MlxVideoBackend, id="mac"), pytest.param(H3VideoBackend, id="cuda")]


def _request(backend, tmp_path: Path, **overrides) -> VideoGenerationRequest:
    return VideoGenerationRequest(backend=backend, model=object(), prompt="a fox", output_dir=str(tmp_path), step_callback=lambda _event: None, **overrides)


def _artifacts() -> VideoWorkingArtifacts:
    artifacts = VideoWorkingArtifacts()
    artifacts.filename = "clip.mp4"
    return artifacts


@pytest.mark.parametrize("backend_class", _BACKENDS)
@pytest.mark.parametrize("upscale", [False, True])
@pytest.mark.parametrize("end_image", [None, "last.png"])
def test_the_generation_stages_send_keywords_the_backend_takes(backend_class, upscale, end_image, tmp_path):
    backend = create_autospec(backend_class, instance=True)
    backend.text_to_video.return_value = backend.image_to_video.return_value = tmp_path / "clip.mp4"
    upscaling = {"upscale": 2, "upscale_steps": 4, "review": lambda _path: None} if upscale else {}

    assert video_stages.text_to_video_stage(_request(backend, tmp_path, **upscaling), _artifacts()) == StageOutcome.success
    assert video_stages.image_to_video_stage(_request(backend, tmp_path, image_path="first.png", end_image_path=end_image, **upscaling), _artifacts()) == StageOutcome.success

    sent = backend.image_to_video.call_args.kwargs
    assert ("end_image" in sent) is (end_image is not None)
    assert {"stage1_steps", "review"} <= set(sent) if upscale else not {"stage1_steps", "review"} & set(sent)


@pytest.mark.parametrize("backend_class", _BACKENDS)
def test_the_extend_stage_sends_keywords_the_backend_takes(backend_class, tmp_path):
    backend = create_autospec(backend_class, instance=True)
    backend.extend_video.return_value = None
    source = ExtendSource(path=tmp_path / "source.mp4", width=64, height=64, fps=24.0, fps_text="24/1", frame_count=22, has_audio=True, video_start=0.0)

    video_stages.extend_video_stage(_request(backend, tmp_path, extend=ExtendSpec(source, 17)), _artifacts())

    assert set(backend.extend_video.call_args.kwargs) == {"model", "source", "prompt", "extend_frames", "seed", "steps", "segment_dir", "audio_output", "step_callback"}
