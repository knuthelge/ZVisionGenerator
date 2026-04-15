"""Shared test helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

from zvisiongenerator.utils.video_model_detect import VideoModelInfo


def _make_args(**overrides):
    """Return a minimal argparse Namespace for run_batch()."""
    defaults = dict(
        ratio="2:3",
        size="m",
        width=None,
        height=None,
        runs=1,
        seed=42,
        steps=4,
        guidance=1.0,
        scheduler=None,
        upscale=None,
        upscale_denoise=None,
        upscale_steps=None,
        upscale_guidance=None,
        upscale_sharpen=False,
        upscale_save_pre=False,
        image_path=None,
        image_strength=0.5,
        output="outputs",
        model="test-model",
        sharpen=True,
        contrast=False,
        saturation=False,
        lora_paths=None,
        lora_weights=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_video_args(**overrides):
    """Return a minimal argparse Namespace for video CLI / run_video_batch()."""
    defaults = dict(
        model="dgrauet/ltx-2.3-mlx-q4",
        prompt=None,
        prompts_file="prompts.yaml",
        image_path=None,
        ratio="16:9",
        size="m",
        width=704,
        height=448,
        num_frames=49,
        steps=8,
        seed=42,
        runs=1,
        low_memory=True,
        output=".",
        format="mp4",
        lora=None,
        lora_paths=[],
        lora_weights=[],
        upscale=None,
        upscale_steps=None,
        no_audio=False,
        audio=True,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_mock_video_backend(name="ltx"):
    """Return a MagicMock satisfying the VideoBackend Protocol."""
    mock = MagicMock()
    mock.name = name
    mock.text_to_video.return_value = Path("/tmp/test.mp4")
    mock.image_to_video.return_value = Path("/tmp/test.mp4")
    mock.load_model.return_value = (
        MagicMock(),
        VideoModelInfo(family=name, backend=name, supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32),
    )
    return mock
