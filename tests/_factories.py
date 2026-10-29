"""Plain builders shared by test modules in more than one area.

Import them with ``from tests._factories import ...``; they are functions, not fixtures, so a test calls them with the overrides it needs.
"""

from __future__ import annotations

import copy
import functools
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.core.job_plan import JobPlan
from zvisiongenerator.core.video_types import ExtendSource
from zvisiongenerator.preflight import plan_iterations
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.utils.video_probe import VideoProbe


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
        review=False,
        image_path=None,
        image_strength=0.5,
        reference_paths=None,
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


def _make_plan(prompts_data, args, *, config=None, enhance_by_set=None, scene=None, scene_by_set=None) -> JobPlan:
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
            scene=scene,
            scene_by_set=scene_by_set,
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


def make_extend_source(**overrides: Any) -> ExtendSource:
    """Return a checked 24 fps, 512x256, 49-frame source with sound that was recorded with a scene and an action."""
    values: dict[str, Any] = dict(
        path=Path("/videos/clip.mp4"),
        width=512,
        height=256,
        fps=24.0,
        fps_text="24/1",
        frame_count=49,
        has_audio=True,
        video_start=0.0,
        scene="A woman in a red raincoat on a rainy street.",
        prompt="She opens a black umbrella.",
    )
    values.update(overrides)
    return ExtendSource(**values)


def make_probe(**overrides: Any) -> VideoProbe:
    """Return the probe of a clean 512x256, 24 fps, 73-frame h264 clip with audio and no colour tags."""
    values: dict[str, Any] = dict(
        coded_width=512,
        coded_height=256,
        width=512,
        height=256,
        codec="h264",
        profile="High",
        pix_fmt="yuv420p",
        bits_per_sample=8,
        sample_aspect_ratio="1:1",
        r_frame_rate="24/1",
        avg_frame_rate="24/1",
        time_base="1/12288",
        field_order="progressive",
        color_space=None,
        color_primaries=None,
        color_trc=None,
        rotation=0,
        frame_count=73,
        duration=3.04,
        video_start=0.0,
        format_start=0.0,
        has_audio=True,
    )
    values.update(overrides)
    return VideoProbe(**values)


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


def _patch_image_main_dependencies(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub everything ``image_cli.main`` loads and return the fake backend; the batch run itself is a mock."""
    import zvisiongenerator.image_cli as image_cli
    import zvisiongenerator.utils.lora as lora_utils

    backend = MagicMock()
    backend.name = "stub-image-backend"
    backend.load_model = MagicMock(return_value=(object(), MagicMock(family="zimage", size=None)))

    monkeypatch.setattr(
        image_cli,
        "load_config",
        lambda: {
            "generation": {"default_ratio": "2:3", "default_size": "m"},
            "sizes": {"2:3": {"m": {"width": 832, "height": 1216}}},
            "model_aliases": {},
            "schedulers": {},
        },
    )
    monkeypatch.setattr(image_cli, "select_ratio_size_defaults", lambda *args, **kwargs: ("2:3", "m"))
    monkeypatch.setattr(image_cli, "resolve_model_path", lambda model, **kwargs: model)
    monkeypatch.setattr(image_cli, "resolve_defaults", lambda *args, **kwargs: {"steps": 10, "guidance": 0.5, "scheduler": None})
    monkeypatch.setattr(image_cli, "validate_scheduler", lambda *args, **kwargs: None)
    monkeypatch.setattr(image_cli, "detect_image_model", lambda _: MagicMock(family="zimage", size=None))
    monkeypatch.setattr(image_cli, "get_backend", lambda: backend)
    monkeypatch.setattr(lora_utils, "resolve_lora_path", lambda name: f"/resolved/{Path(name).name}")
    monkeypatch.setattr(image_cli, "run_batch", MagicMock())
    monkeypatch.setattr(lora_utils.os.path, "isfile", lambda _path: True)
    return backend


@functools.cache
def shared_packaged_config() -> dict[str, Any]:
    """Return the packaged ``config.yaml`` parsed once per process, with an empty data dir so no user override applies; never mutate it."""
    from zvisiongenerator.utils import config as config_module

    with tempfile.TemporaryDirectory() as empty, pytest.MonkeyPatch.context() as patch:
        data_dir = Path(empty) / ".ziv"
        patch.setenv("ZIV_DATA_DIR", str(data_dir))
        patch.setattr(config_module, "get_ziv_data_dir", lambda: data_dir)
        return config_module.load_config()


def load_packaged_config() -> dict[str, Any]:
    """Return a private deep copy of the packaged config (parsed once per process), so a test may change it freely.

    Inside a test, the ``packaged_config`` fixture returns the same thing.
    """
    return copy.deepcopy(shared_packaged_config())
