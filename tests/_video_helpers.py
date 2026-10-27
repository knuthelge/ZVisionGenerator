"""Shared builders for the video tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.core.video_types import VideoGenerationRequest
from zvisiongenerator.utils.video_model_detect import VideoModelInfo


def make_video_request(**overrides: Any) -> VideoGenerationRequest:
    """Build a minimal ``VideoGenerationRequest`` with mock backend and model."""
    defaults: dict[str, Any] = {"backend": MagicMock(), "model": MagicMock(), "prompt": "a beautiful sunset"}
    defaults.update(overrides)
    return VideoGenerationRequest(**defaults)


def make_ltx_info(**overrides: Any) -> VideoModelInfo:
    """Build the ``VideoModelInfo`` of an LTX model (8-frame and 32-pixel alignment, 24 fps)."""
    fields: dict[str, Any] = {"family": "ltx", "backend": "ltx", "supports_i2v": True, "default_fps": 24, "frame_alignment": 8, "resolution_alignment": 32}
    fields.update(overrides)
    return VideoModelInfo(**fields)


class _VideoCliEnhancer:
    def __init__(self, order: list[str]):
        self.order = order

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        yield "A dog sprints across a sunlit beach, camera tracking low."

    def close(self) -> None:
        self.order.append("close")


def patch_video_cli_for_preflight(monkeypatch: pytest.MonkeyPatch, order: list[str], *, load_error: Exception | None = None):
    """Patch ``video_cli`` so ``main()`` runs its preflight with a fake enhancer and records the steps in ``order``.

    Returns the patched module, a dict that receives the batch's plan, and the list of ``enhance`` flags the workflow was built with.
    """
    import zvisiongenerator.video_cli as video_cli
    from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession

    model_info = make_ltx_info()
    backend = MagicMock()
    backend.load_model.side_effect = lambda *a, **k: (order.append("load_model"), (MagicMock(), model_info))[1]

    def _factory(repo, revision):
        if load_error is not None:
            raise load_error
        return _VideoCliEnhancer(order)

    session = PromptEnhancerSession(_factory, scheduler=lambda delay, callback: None, downloaded=lambda repo, revision: True)
    monkeypatch.setattr(
        video_cli,
        "load_config",
        lambda: {
            "video_generation": {"default_ratio": "16:9", "default_size": "m"},
            "video_sizes": {"16:9": {"m": {"width": 704, "height": 448, "frames": 49}}},
            "prompt_enhancer": {"model": {"darwin": "a/b", "win32": "a/b", "linux": "a/b"}},
        },
    )
    monkeypatch.setattr(video_cli, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(video_cli, "detect_video_model", lambda _model: model_info)
    monkeypatch.setattr(video_cli, "resolve_video_defaults", lambda _family, _config, cli_overrides: {"steps": 8, "width": 704, "height": 448, "num_frames": 49})
    monkeypatch.setattr(video_cli, "get_video_backend", lambda _family: backend)
    workflows: list[bool] = []
    monkeypatch.setattr(video_cli, "build_video_workflow", lambda _args, *, enhance: workflows.append(enhance) or object())
    monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, rev: True)
    monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.get_prompt_enhancer_session", lambda: session)
    monkeypatch.setattr("zvisiongenerator.preflight.release_accelerator_memory", lambda: order.append("release_memory"))
    captured: dict = {}

    def _fake_run_video_batch(backend, model, model_info, workflow, prompts_data, config, args, *, plan, skip_signal=None):
        order.append("run_video_batch")
        captured["plan"] = plan

    monkeypatch.setattr(video_cli, "run_video_batch", _fake_run_video_batch)
    return video_cli, captured, workflows
