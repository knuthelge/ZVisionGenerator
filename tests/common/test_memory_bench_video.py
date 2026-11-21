"""Tests for the memory bench's LTX video cases: the start picture, Extend and the upscale."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.memory import video
from bench.memory.measure import GIB
from bench.memory.cases import expand_cases, load_cases
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.model_memory import MemoryBudget


def _prepared(case_id: str, platform: str = "cuda") -> tuple[dict, dict]:
    cases = load_cases("video")
    case = {case["id"]: case for case in expand_cases(cases, media="video", platform=platform)}[case_id]
    video.prepare(case, cases, load_config())
    return case, cases


# --- the LTX cases -----------------------------------------------------------------------------------------------


def test_the_case_file_has_the_ltx_cases_at_the_default_size_and_the_medium_one() -> None:
    ids = {case["id"]: case for case in expand_cases(load_cases("video"), media="video", platform="cuda")}

    assert {"ltx4-s", "ltx4-m", "ltx4-s-i2v", "ltx4-s-extend", "ltx4-s-upscale", "ltx25-4-s"} <= set(ids)
    assert (ids["ltx4-s"]["model"], ids["ltx4-s"]["size"]) == ("ltx-4", "s") and ids["ltx4-m"]["size"] == "m"
    assert ids["ltx25-4-s"]["model"] == "ltx25-4" and ids["ltx4-s-upscale"]["upscale"] == 2
    assert all(case["steps"] == 2 for case in ids.values())


def test_a_start_image_case_starts_from_a_synthetic_picture_of_the_clips_size(tmp_path: Path) -> None:
    case, cases = _prepared("ltx4-s-i2v")

    video.ensure_sources(case, tmp_path, cases)
    argv = video.case_argv(case, tmp_path)

    picture = Path(argv[argv.index("--image") + 1])
    from PIL import Image

    assert picture.is_file() and Image.open(picture).size == (case["width"], case["height"])
    assert argv[argv.index("--size") + 1] == "s"


def test_an_extend_case_continues_its_source_at_the_sources_size_and_carries_no_size_or_frames(tmp_path: Path) -> None:
    case, _cases = _prepared("ltx4-s-extend")
    argv = video.case_argv(case, tmp_path)

    assert (case["width"], case["height"]) == (576, 320)
    assert case["frames"] == 25 + 72  # the carried frames and 3 s at 24 fps
    assert argv[argv.index("--extend") + 1] == str(tmp_path / "sources" / "extend-xs.mp4") and argv[argv.index("--extend-seconds") + 1] == "3"
    assert not {"--size", "--ratio", "--width", "--height", "--frames", "--upscale"} & set(argv)


def test_the_extend_source_clip_is_rendered_once_by_the_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    case, cases = _prepared("ltx4-s-extend")
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append(command)
        out = Path(command[command.index("--output") + 1])
        (out / "clip.mp4").write_bytes(b"x")

    monkeypatch.setattr(subprocess, "run", fake_run)
    video.ensure_sources(case, tmp_path, cases)
    video.ensure_sources(case, tmp_path, cases)

    assert len(calls) == 1 and (tmp_path / "sources" / "extend-xs.mp4").read_bytes() == b"x"
    command = calls[0]
    assert command[command.index("-m") + 1] == "ltx-4" and command[command.index("--size") + 1] == "xs" and command[command.index("--frames") + 1] == "49" and "--no-enhance" in command
    assert list((tmp_path / "sources").iterdir()) == [tmp_path / "sources" / "extend-xs.mp4"]


def test_a_source_clip_that_was_not_written_is_an_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    case, cases = _prepared("ltx4-s-extend")
    monkeypatch.setattr(subprocess, "run", lambda command, check: None)

    with pytest.raises(FileNotFoundError):
        video.ensure_sources(case, tmp_path, cases)


def test_the_video_estimate_names_the_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from zvisiongenerator.utils.memory_verdict import MemoryJob

    seen: list[MemoryJob] = []
    monkeypatch.setattr("zvisiongenerator.backends.accelerator_memory_budget_or_none", lambda: MemoryBudget(10 * GIB, 32 * GIB))
    monkeypatch.setattr(
        "zvisiongenerator.utils.memory_verdict.judge", lambda job, machine, weights: seen.append(job) or SimpleNamespace(gpu_bytes=1, ram_bytes=1, status="fits", resource=None, placement=None)
    )
    base = {"family": "h3", "width": 512, "height": 320, "frames": 22}

    video.estimate(base)
    video.estimate(base | {"start_image": True})
    video.estimate(base | {"extend_source": "extend-xs"})

    assert [job.mode for job in seen] == ["t2v", "i2v", "extend"]
