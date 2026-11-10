"""Tests for cutting a clip at a frame so an extension continues from there; ffmpeg and ffprobe are mocked."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from tests._factories import make_extend_source, make_probe
from zvisiongenerator.core.video_types import ExtendCut
from zvisiongenerator.utils import video_cut
from zvisiongenerator.utils.video_cut import (
    CUT_DIR_PREFIX,
    MIN_CUT_FRAMES,
    applied_extend_cut,
    build_cut_command,
    cut_extend_source,
    extend_cut_point,
    moved_track_end,
    sweep_stale_cuts,
)


def _source(**overrides):
    return make_extend_source(**{"path": Path("/videos/walk.mp4"), "frame_count": 120, **overrides})


def _cut(keep: int = 60, **overrides) -> ExtendCut:
    return ExtendCut(**{"keep_frames": keep, "seconds": (keep - 1) / 24, "original": Path("/videos/walk.mp4"), **overrides})


@pytest.fixture
def cutting(monkeypatch) -> MagicMock:
    """Mock ffmpeg and ffprobe so a cut "succeeds": the probe reads back the frames the command kept. Returns the ffmpeg mock."""
    ffmpeg = MagicMock(return_value=None)
    ffmpeg.keep = 60  # the frames the "cut file" turns out to hold; a test sets another to simulate a bad cut

    def _probe(_path):
        return make_probe(frame_count=ffmpeg.keep, video_start=0.0, format_start=0.0)

    monkeypatch.setattr(video_cut, "run_ffmpeg", ffmpeg)
    monkeypatch.setattr(video_cut, "probe_video", _probe)
    return ffmpeg


# ── Which frame ──────────────────────────────────────────────────────────────


def test_a_time_keeps_the_frame_it_lies_in_and_everything_before_it():
    cut = extend_cut_point(_source(), 2.5)

    assert cut == ExtendCut(keep_frames=61, seconds=2.5, original=Path("/videos/walk.mp4"))


def test_the_recorded_time_is_the_frame_s_own_not_the_one_asked_for():
    cut = extend_cut_point(_source(), 2.51)

    assert cut is not None
    assert (cut.keep_frames, cut.seconds) == (61, 2.5)


def test_the_recorded_time_includes_where_the_video_stream_starts():
    cut = extend_cut_point(_source(video_start=0.08), 2.58)

    assert cut is not None
    assert (cut.keep_frames, cut.seconds) == (61, pytest.approx(2.58))


@pytest.mark.parametrize("seconds", [119 / 24, 5.0, 500.0])
def test_a_point_at_or_past_the_last_frame_keeps_the_whole_clip(seconds):
    assert extend_cut_point(_source(), seconds) is None


def test_the_last_frame_before_the_end_still_cuts_one_frame():
    cut = extend_cut_point(_source(), 118 / 24)

    assert cut is not None
    assert cut.keep_frames == 119


def test_a_point_that_keeps_exactly_the_minimum_is_allowed():
    cut = extend_cut_point(_source(), (MIN_CUT_FRAMES - 1) / 24)

    assert cut is not None
    assert cut.keep_frames == MIN_CUT_FRAMES


def test_a_point_under_the_minimum_is_refused_and_says_the_limit():
    with pytest.raises(ValueError, match=rf"keeps {MIN_CUT_FRAMES - 1} frames.*at least {MIN_CUT_FRAMES} frames"):
        extend_cut_point(_source(), (MIN_CUT_FRAMES - 2) / 24)


@pytest.mark.parametrize("seconds", [-0.5, math.nan, math.inf])
def test_a_negative_or_non_finite_point_is_refused(seconds):
    with pytest.raises(ValueError, match="finite number of seconds"):
        extend_cut_point(_source(), seconds)


# ── The command ──────────────────────────────────────────────────────────────


def _option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_the_command_keeps_the_frames_and_encodes_them_at_the_joins_quality():
    command = build_cut_command(_source(), _cut(60), "out.mp4")

    assert _option(command, "-vf").startswith("trim=end_frame=60,setpts=PTS-STARTPTS")
    assert (_option(command, "-crf"), _option(command, "-c:v"), _option(command, "-r")) == ("12", "libx264", "24/1")
    assert "-preset" not in command
    assert command[-1] == "out.mp4"


def test_the_sound_is_moved_to_the_first_frame_and_cut_to_the_kept_length():
    command = build_cut_command(_source(video_start=0.021), _cut(60), "out.mp4")

    chain = _option(command, "-af")
    assert "atrim=start=0.021000" in chain
    assert "atrim=end_sample=120000" in chain  # 60 frames at 24 fps is 2.5 s, 120 000 samples at 48 kHz
    assert (_option(command, "-c:a"), "-an" in command) == ("aac", False)


def test_the_kept_length_of_the_sound_follows_a_fractional_frame_rate():
    command = build_cut_command(_source(fps=30000 / 1001, fps_text="30000/1001"), _cut(60), "out.mp4")

    assert "atrim=end_sample=96096" in _option(command, "-af")  # 60 * 1001 / 30000 s * 48 000


def test_a_silent_source_is_cut_without_sound():
    command = build_cut_command(_source(has_audio=False), _cut(60), "out.mp4")

    assert "-an" in command
    assert not {"-af", "-c:a", "0:a:0"} & set(command)


def test_the_colour_tags_are_stamped_on_the_cut():
    command = build_cut_command(_source(color_space="bt709", color_primaries="bt709", color_trc="bt709"), _cut(60), "out.mp4")

    video_filter = _option(command, "-vf")
    assert "scale=out_color_matrix=bt709" in video_filter
    assert "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709" in video_filter


def test_an_untagged_source_adds_no_colour_conversion():
    assert _option(build_cut_command(_source(), _cut(60), "out.mp4"), "-vf").endswith(",null")


def test_the_command_drops_the_metadata():
    command = build_cut_command(_source(), _cut(60), "out.mp4")

    assert _option(command, "-map_metadata") == "-1"


# ── The cut source ───────────────────────────────────────────────────────────


def test_the_cut_source_is_built_from_the_cut_file_and_keeps_the_originals_settings(cutting, tmp_path):
    source = _source(scene="A street.", prompt="She waves.", model="ltx-8", lora="/loras/a.safetensors:0.8", audio_track_path="/music/song.wav", audio_track_end=10.0, color_space="bt709")
    cutting.keep = 60

    cut_source = cut_extend_source(source, _cut(60), tmp_path)

    assert cut_source.path == (tmp_path / "cut.mp4").resolve()
    assert cut_source.frame_count == 60
    assert (cut_source.scene, cut_source.prompt, cut_source.model, cut_source.lora) == ("A street.", "She waves.", "ltx-8", "/loras/a.safetensors:0.8")
    assert cut_source.audio_track_path == "/music/song.wav"
    assert cut_source.audio_track_end == pytest.approx(10.0 - 60 / 24)
    assert cut_source.probe is not None and cut_source.probe.frame_count == 60


def test_the_cut_source_takes_its_timing_from_the_cut_file_not_the_original(monkeypatch, cutting, tmp_path):
    monkeypatch.setattr(video_cut, "probe_video", lambda _path: make_probe(frame_count=60, video_start=0.02, format_start=0.0, r_frame_rate="24/1"))

    cut_source = cut_extend_source(_source(video_start=0.5), _cut(60), tmp_path)

    assert cut_source.video_start == pytest.approx(0.02)


def test_the_cut_is_made_by_one_ffmpeg_call(cutting, tmp_path):
    cut_extend_source(_source(), _cut(60), tmp_path)

    cutting.assert_called_once()
    assert cutting.call_args.args[0][-1] == str(tmp_path / "cut.mp4")
    assert cutting.call_args.args[1] == "ffmpeg could not cut the video"


def test_a_cut_with_the_wrong_number_of_frames_fails(cutting, tmp_path):
    cutting.keep = 59

    with pytest.raises(RuntimeError, match="59 frames, expected 60"):
        cut_extend_source(_source(), _cut(60), tmp_path)


def test_a_cut_with_another_size_fails(monkeypatch, cutting, tmp_path):
    monkeypatch.setattr(video_cut, "probe_video", lambda _path: make_probe(frame_count=60, width=256, height=128))

    with pytest.raises(RuntimeError, match="256x128, expected 512x256"):
        cut_extend_source(_source(), _cut(60), tmp_path)


def test_a_failing_ffmpeg_fails_the_cut(monkeypatch, tmp_path):
    monkeypatch.setattr(video_cut, "run_ffmpeg", MagicMock(side_effect=RuntimeError("ffmpeg could not cut the video: boom")))

    with pytest.raises(RuntimeError, match="boom"):
        cut_extend_source(_source(), _cut(60), tmp_path)


# ── The track ────────────────────────────────────────────────────────────────


def test_the_track_end_moves_back_by_the_removed_frames():
    assert moved_track_end(_source(audio_track_end=10.0), _cut(60)) == pytest.approx(10.0 - 60 / 24)


def test_the_track_end_never_goes_below_zero():
    assert moved_track_end(_source(audio_track_end=1.0), _cut(30)) == 0.0


def test_a_source_without_a_track_end_has_none_to_move():
    assert moved_track_end(_source(), _cut(60)) is None


# ── The job's cut ────────────────────────────────────────────────────────────


def _args(source, cut, output) -> argparse.Namespace:
    return argparse.Namespace(extend_source=source, extend_cut=cut, output=str(output))


def test_a_job_without_a_cut_makes_nothing(cutting, tmp_path):
    save = tmp_path / "not-made"
    source = _source()
    args = _args(source, None, save)

    with applied_extend_cut(args):
        assert args.extend_source is source

    cutting.assert_not_called()
    assert not save.exists()


def test_a_job_with_a_cut_runs_on_the_cut_and_gets_its_original_back(cutting, tmp_path):
    source = _source()
    args = _args(source, _cut(60), tmp_path)

    with applied_extend_cut(args):
        assert args.extend_source is not source
        assert args.extend_source.frame_count == 60
        assert args.extend_source.path.parent.parent == tmp_path.resolve()
        assert args.extend_source.path.parent.name.startswith(".ziv-tmp-cut-")

    assert args.extend_source is source


def test_the_cut_is_made_once_whatever_the_job_does_inside(cutting, tmp_path):
    args = _args(_source(), _cut(60), tmp_path)

    with applied_extend_cut(args):
        pass

    cutting.assert_called_once()


def test_the_work_folder_is_removed_when_the_job_ends(cutting, tmp_path):
    with applied_extend_cut(_args(_source(), _cut(60), tmp_path)):
        assert [entry.name for entry in tmp_path.iterdir()]

    assert list(tmp_path.iterdir()) == []


def test_the_work_folder_is_removed_when_the_job_fails_or_is_stopped(cutting, tmp_path):
    args = _args(_source(), _cut(60), tmp_path)
    original = args.extend_source

    with pytest.raises(KeyboardInterrupt):
        with applied_extend_cut(args):
            raise KeyboardInterrupt

    assert list(tmp_path.iterdir()) == []
    assert args.extend_source is original


def test_the_work_folder_is_removed_when_the_cut_itself_fails(cutting, tmp_path):
    cutting.keep = 5
    args = _args(_source(), _cut(60), tmp_path)

    with pytest.raises(RuntimeError, match="expected 60"):
        with applied_extend_cut(args):
            pytest.fail("the job must not start on a bad cut")

    assert list(tmp_path.iterdir()) == []
    assert args.extend_source is not None


def test_a_save_folder_that_does_not_exist_yet_is_created_first(cutting, tmp_path):
    save = tmp_path / "new" / "project"

    with applied_extend_cut(_args(_source(), _cut(60), save)):
        assert save.is_dir()

    assert save.is_dir()
    assert list(save.iterdir()) == []


# ── Cuts a killed process left behind ────────────────────────────────────────


@pytest.fixture
def dead_pid() -> int:
    """The id of a process that has already ended."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _leftover(save: Path, pid: int, suffix: str = "abc123") -> Path:
    folder = save / f"{CUT_DIR_PREFIX}{pid}-{suffix}"
    folder.mkdir()
    (folder / "cut.mp4").write_bytes(b"cut")
    return folder


def test_the_work_folder_is_named_for_the_process_that_made_it(cutting, tmp_path):
    with applied_extend_cut(_args(_source(), _cut(60), tmp_path)) as _:
        names = [entry.name for entry in tmp_path.iterdir()]

    assert len(names) == 1
    assert names[0].startswith(f"{CUT_DIR_PREFIX}{os.getpid()}-")


def test_a_cut_a_killed_process_left_behind_is_removed(dead_pid, tmp_path):
    stale = _leftover(tmp_path, dead_pid)

    sweep_stale_cuts(tmp_path)

    assert not stale.exists()


def test_the_cut_of_a_job_still_running_in_another_process_is_left_alone(tmp_path):
    live = _leftover(tmp_path, os.getppid())

    sweep_stale_cuts(tmp_path)

    assert (live / "cut.mp4").is_file()


def test_a_cut_whose_process_id_another_process_has_taken_is_removed(tmp_path):
    reused = _leftover(tmp_path, os.getppid())
    os.utime(reused, (0, 0))  # last written before the process that holds the id now started

    sweep_stale_cuts(tmp_path)

    assert not reused.exists()


def test_the_sweep_leaves_this_process_s_folder_other_folders_and_files_alone(dead_pid, tmp_path):
    own = _leftover(tmp_path, os.getpid())
    nameless = tmp_path / f"{CUT_DIR_PREFIX}abc123"
    nameless.mkdir()
    other = tmp_path / f".ziv-tmp-track-{dead_pid}-x"
    other.mkdir()
    (tmp_path / "clip.mp4").write_bytes(b"clip")
    (tmp_path / f"{CUT_DIR_PREFIX}{dead_pid}-file").write_bytes(b"not a folder")

    sweep_stale_cuts(tmp_path)

    assert own.is_dir()
    assert nameless.is_dir()
    assert other.is_dir()
    assert (tmp_path / "clip.mp4").is_file()
    assert (tmp_path / f"{CUT_DIR_PREFIX}{dead_pid}-file").is_file()


def test_a_save_folder_that_is_not_there_has_nothing_to_sweep(tmp_path):
    sweep_stale_cuts(tmp_path / "missing")


def test_a_job_with_a_cut_sweeps_the_leftovers_before_it_makes_its_own(cutting, dead_pid, tmp_path):
    stale = _leftover(tmp_path, dead_pid)

    with applied_extend_cut(_args(_source(), _cut(60), tmp_path)):
        assert not stale.exists()

    assert list(tmp_path.iterdir()) == []


def test_a_job_without_a_cut_sweeps_nothing(cutting, dead_pid, tmp_path):
    stale = _leftover(tmp_path, dead_pid)

    with applied_extend_cut(_args(_source(), None, tmp_path)):
        pass

    assert stale.is_dir()
