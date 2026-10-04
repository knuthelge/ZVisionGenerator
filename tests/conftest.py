"""Shared test helpers."""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

from zvisiongenerator.core.job_plan import JobPlan
from zvisiongenerator.preflight import plan_iterations
from zvisiongenerator.utils.video_model_detect import VideoModelInfo

# Starlette's TestClient sends ``Host: testserver``; allow it through the Web UI request guard.
os.environ.setdefault("ZIV_UI_ALLOWED_HOSTS", "testserver")
# Entry points start the error log; keep tests from writing ~/.ziv/logs or replacing exception hooks.
os.environ.setdefault("ZIV_LOG", "off")


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


def _make_plan(prompts_data, args, *, config=None, enhance_by_set=None) -> JobPlan:
    """Return the preflight plan of a job without enhancement rewrites (statuses stay ``off``)."""
    generation = (config or {}).get("generation", {})
    return JobPlan(
        plan_iterations(
            prompts_data,
            runs=args.runs,
            seed=args.seed,
            seed_min=generation.get("seed_min", 4),
            seed_max=generation.get("seed_max", 2**32 - 1),
            json_prompt=bool(getattr(args, "json_prompt_enabled", False)),
            disabled=bool(getattr(args, "no_enhance", False)),
            override=getattr(args, "enhance", None),
            enhance_by_set=enhance_by_set,
        )
    )


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
