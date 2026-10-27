"""Shared test helpers."""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from zvisiongenerator.core.job_plan import JobPlan
from zvisiongenerator.core.video_types import ExtendSource
from zvisiongenerator.preflight import plan_iterations
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.utils.video_probe import VideoProbe

# One BLAS/OpenMP thread per process: under ``-n auto`` every worker otherwise starts one thread per core and they starve each
# other (the torch-heavy files run several times slower). Set before anything imports torch or numpy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Starlette's TestClient sends ``Host: testserver``; allow it through the Web UI request guard.
os.environ.setdefault("ZIV_UI_ALLOWED_HOSTS", "testserver")
# Entry points start the error log; keep tests from writing ~/.ziv/logs or replacing exception hooks.
os.environ.setdefault("ZIV_LOG", "off")


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``real_download_plans`` marker."""
    config.addinivalue_line("markers", "real_download_plans: keep the web runner's real download planners (they may read model indexes)")


@pytest.fixture(autouse=True)
def _no_web_download_planning(request, monkeypatch):
    """Plan no downloads in the web runner (``None`` = cannot plan), so no web job test reaches Hugging Face through a planner.

    Tests install the planners they need after this runs; ``@pytest.mark.real_download_plans`` keeps the real ones.
    """
    if request.node.get_closest_marker("real_download_plans"):
        return
    from zvisiongenerator.web import web_runner

    for name in ("image_plan", "video_plan", "seedvr2_plan", "enhancer_plan"):
        monkeypatch.setattr(web_runner, name, lambda *_a, **_kw: None)


@pytest.fixture(autouse=True)
def _no_hugging_face_token(monkeypatch):
    """Keep every test from seeing the real Hugging Face token (env or ``hf auth login`` file); tests that need one inject it."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(name, raising=False)


def _lazy_tmp_dir(factory: pytest.TempPathFactory, name: str):
    """Return a getter that makes a fresh temp dir on first call, so autouse fixtures cost nothing for tests that never touch it.

    pytest rescans its temp root on every ``mktemp``, so an eager dir per test slows every later test down.
    """
    made: list[Path] = []

    def _get() -> Path:
        if not made:
            made.append(factory.mktemp(name))
        return made[0]

    return _get


@pytest.fixture(autouse=True)
def _presets_in_tmp(tmp_path_factory, monkeypatch):
    """Keep every test's presets in a temporary data directory, never the real ``~/.ziv/presets``."""
    monkeypatch.setattr("zvisiongenerator.web.presets.get_ziv_data_dir", _lazy_tmp_dir(tmp_path_factory, "ziv-data"))


_HUB_DOWNLOADS = ("hf_hub_download", "snapshot_download")


@pytest.fixture(autouse=True)
def _no_hub_downloads(monkeypatch):
    """Fail any test that reaches ``huggingface_hub``'s download functions unmocked.

    The stubs raise ``RuntimeError`` and record the call; the test fails afterwards if any was recorded, because app
    code often catches the error and carries on. A test that patches either name itself replaces the stub and wins.
    Use ``hub_offline`` for tests whose subject is "nothing cached, no network".
    """
    import huggingface_hub

    calls: list[str] = []

    def _stub(name: str):
        def _blocked(*args, **kwargs):
            repo_id = kwargs.get("repo_id", args[0] if args else None)
            calls.append(f"{name}(repo_id={repo_id!r})")
            raise RuntimeError(f"Tests must not reach huggingface_hub.{name}; mock it.")

        return _blocked

    for name in _HUB_DOWNLOADS:
        monkeypatch.setattr(huggingface_hub, name, _stub(name))
    yield
    if calls:
        pytest.fail("Test reached the Hugging Face Hub unmocked: " + ", ".join(calls), pytrace=False)


@pytest.fixture
def hub_offline(monkeypatch):
    """Make ``huggingface_hub``'s download functions report nothing cached and no network (``LocalEntryNotFoundError``), without failing the test."""
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def _offline(*_args, **_kwargs):
        raise LocalEntryNotFoundError("Hub unavailable in tests.")

    for name in _HUB_DOWNLOADS:
        monkeypatch.setattr(huggingface_hub, name, _offline)


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


@pytest.fixture(autouse=True)
def _no_real_memory_budget(monkeypatch):
    """Keep ziv-video from querying the real accelerator when it checks a clip against memory; tests that care patch it again."""
    import zvisiongenerator.video_cli as video_cli

    monkeypatch.setattr(video_cli, "accelerator_memory_budget_or_none", lambda: None)


@pytest.fixture()
def fake_peft_no_match(monkeypatch):
    """Install a fake ``peft`` module and return its ``NoMatchingPeftModuleError`` (a ``ValueError``, as in peft 0.21)."""

    class NoMatchingPeftModuleError(ValueError):
        pass

    peft = ModuleType("peft")
    peft.NoMatchingPeftModuleError = NoMatchingPeftModuleError
    monkeypatch.setitem(sys.modules, "peft", peft)
    return NoMatchingPeftModuleError


@pytest.fixture(autouse=True)
def _seedvr2_off_in_web_config(monkeypatch):
    """Keep SeedVR2 off in the Web UI config so tests behave the same on every platform; opt-in tests patch it back on."""
    monkeypatch.setattr("zvisiongenerator.web.config.supports_seedvr2", lambda: False)


@pytest.fixture(autouse=True)
def _isolated_app_db(monkeypatch, tmp_path_factory):
    """Point the app database at a throwaway data dir so no test reads or writes the real ``ziv.sqlite3``.

    Returns the dir getter; the dir is made on first use.
    """
    data_dir = _lazy_tmp_dir(tmp_path_factory, "ziv-app-db")
    monkeypatch.setattr("zvisiongenerator.utils.app_db.get_ziv_data_dir", data_dir)
    return data_dir
