"""Tests for checking a clip that a video is extended from, and for the extension window arithmetic."""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from conftest import make_extend_source, make_probe

from zvisiongenerator.core.video_types import ExtendSource
from zvisiongenerator.utils import video_extend
from zvisiongenerator.utils.config import load_config, video_family
from zvisiongenerator.utils.paths import collapse_home
from zvisiongenerator.utils.provenance import RecordedSettings
from zvisiongenerator.utils.video_extend import (
    AUDIO_FADE_SAMPLES,
    AUDIO_SAMPLE_RATE,
    CARRY_FRAMES,
    ExtendInspection,
    ExtendProblem,
    apply_extend_args,
    build_extend_source,
    carry_latent_frames,
    audio_lead_samples,
    cut_window_audio,
    extend_audio_output,
    extend_family_problems,
    extend_frames_from_seconds,
    extend_source_problems,
    inspect_extend_source,
    load_extend_source,
    extend_enhance_context,
    recorded_prompts,
    require_extend_family,
    resolve_extend_frames,
    snap_extend_frames,
    window_audio_cut,
    window_frames,
    with_family_problems,
)


# A clean 512x256, 24 fps, 72-frame clip with audio and BT.709 colour tags.
_probe = partial(make_probe, frame_count=72, duration=3.0, color_space="bt709", color_primaries="bt709", color_trc="bt709")


def _codes(problems: list[ExtendProblem]) -> list[str]:
    return [problem.code for problem in problems]


class TestExtendSourceProblems:
    @pytest.mark.parametrize(
        ("code", "overrides"),
        [
            ("unreadable", {"frame_count": None}),
            ("unreadable", {"r_frame_rate": "0/0", "avg_frame_rate": "0/0"}),
            ("unreadable", {"rotation": 45}),
            ("off_grid", {"coded_width": 512, "coded_height": 250, "width": 250, "height": 512, "rotation": 90}),  # checked by the displayed size
            ("too_short", {"frame_count": CARRY_FRAMES - 1}),
            ("off_grid", {"width": 500, "coded_width": 500}),
            ("off_grid", {"height": 250, "coded_height": 250}),
            ("too_small", {"width": 32, "coded_width": 32}),
            ("too_small", {"height": 32, "coded_height": 32}),
            ("variable_frame_rate", {"avg_frame_rate": "2400/101"}),
            ("non_square_pixels", {"sample_aspect_ratio": "4:3"}),
            ("high_bit_depth", {"pix_fmt": "yuv420p10le"}),
            ("high_bit_depth", {"pix_fmt": "yuv444p12be"}),
            ("high_bit_depth", {"pix_fmt": "p010le"}),
            ("high_bit_depth", {"bits_per_sample": 10}),
            ("hdr", {"color_trc": "smpte2084"}),
            ("hdr", {"color_trc": "arib-std-b67"}),
            ("hdr", {"color_space": "bt2020nc"}),
            ("hdr", {"color_space": "bt2020c"}),
            ("interlaced", {"field_order": "tt"}),
            ("interlaced", {"field_order": "bb"}),
            ("interlaced", {"field_order": "tb"}),
        ],
    )
    def test_each_problem_is_reported_with_its_code(self, code, overrides):
        problems = extend_source_problems(_probe(**overrides), None)

        assert code in _codes(problems)
        assert all(problem.message for problem in problems)

    def test_an_upscaled_clip_is_flagged(self):
        recorded = RecordedSettings(generation={"upscale": {"factor": 2}})

        assert _codes(extend_source_problems(_probe(), recorded)) == ["upscaled"]

    def test_unreadable_audio_is_flagged_only_when_the_clip_has_audio(self):
        assert _codes(extend_source_problems(_probe(), None, audio_ok=False)) == ["unreadable_audio"]
        assert extend_source_problems(_probe(has_audio=False), None, audio_ok=False) == []

    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"frame_count": CARRY_FRAMES},
            {"coded_width": 512, "coded_height": 256, "width": 256, "height": 512, "rotation": 90},  # checked by the displayed size
            {"sample_aspect_ratio": "1:1"},
            {"sample_aspect_ratio": "0:1"},
            {"sample_aspect_ratio": None},
            {"field_order": None},
            {"field_order": "progressive"},
            {"field_order": "unknown"},
            {"pix_fmt": "yuvj420p"},
            {"pix_fmt": "yuv420p", "bits_per_sample": None},
            {"color_space": None, "color_primaries": None, "color_trc": None},
            {"avg_frame_rate": "0/0"},
            {"r_frame_rate": "30000/1001", "avg_frame_rate": "1440000/48047"},
        ],
    )
    def test_ordinary_variations_are_not_problems(self, overrides):
        assert extend_source_problems(_probe(**overrides), None) == []

    def test_every_problem_is_listed_not_just_the_first(self):
        problems = extend_source_problems(_probe(width=500, coded_width=500, field_order="tt", color_trc="smpte2084"), None)

        assert sorted(_codes(problems)) == ["hdr", "interlaced", "off_grid"]


class TestExtendFamilyProblems:
    @pytest.fixture(scope="class")
    def h3(self):
        return video_family(load_config(), "h3")

    @pytest.fixture(scope="class")
    def ltx(self):
        return video_family(load_config(), "ltx")

    def test_an_h3_source_at_30_fps_gives_a_frame_rate_problem(self, h3):
        (problem,) = extend_family_problems(30.0, h3)

        assert problem.code == "frame_rate"
        assert "30 fps" in problem.message and "24 fps" in problem.message and "MiniMax H3" in problem.message

    @pytest.mark.parametrize("fps", [24.0, 24.0005, 23.9995])
    def test_an_h3_source_at_24_fps_passes(self, h3, fps):
        assert extend_family_problems(fps, h3) == []

    @pytest.mark.parametrize("fps", [24.002, 23.976, 25.0])
    def test_an_h3_source_off_24_fps_beyond_the_tolerance_is_refused(self, h3, fps):
        assert _codes(extend_family_problems(fps, h3)) == ["frame_rate"]

    @pytest.mark.parametrize("fps", [24.0, 30.0, 29.97, 60.0])
    def test_ltx_extends_a_clip_at_its_own_rate(self, ltx, fps):
        assert extend_family_problems(fps, ltx) == []

    def test_a_model_without_a_description_and_an_unreadable_rate_add_nothing(self, h3):
        assert extend_family_problems(30.0, None) == []
        assert extend_family_problems(0.0, h3) == []

    def test_the_inspection_of_a_30_fps_source_loses_its_source_and_keeps_its_probe(self, h3):
        probe = _probe(r_frame_rate="30/1", avg_frame_rate="30/1")
        inspection = ExtendInspection(probe, None, (), build_extend_source("clip.mp4", probe, None))

        checked = with_family_problems(inspection, h3)

        assert _codes(list(checked.problems)) == ["frame_rate"]
        assert checked.source is None and checked.probe is probe
        assert with_family_problems(inspection, None) is inspection
        assert with_family_problems(ExtendInspection(None, None, (ExtendProblem("unreadable", "x"),), None), h3).problems == (ExtendProblem("unreadable", "x"),)

    def test_a_30_fps_source_is_required_to_be_24_fps_by_h3_only(self, h3, ltx):
        source = make_extend_source(fps=30.0, fps_text="30/1")

        with pytest.raises(ValueError, match="clip.mp4 cannot be extended: .*24 fps clips only"):
            require_extend_family(source, h3)
        require_extend_family(source, ltx)
        require_extend_family(make_extend_source(), h3)

    def test_the_h3_rate_is_the_one_the_backends_check(self, h3):
        from zvisiongenerator.utils.h3_memory import H3_FPS

        assert h3.extend_fps == H3_FPS == 24


class TestRecordedPrompts:
    def test_a_clip_with_a_scene_gives_both_parts(self):
        assert recorded_prompts(RecordedSettings(scene="A castle at dusk.", prompt="A knight rides in.")) == ("A castle at dusk.", "A knight rides in.")

    def test_a_clip_with_one_prompt_gives_no_scene(self):
        assert recorded_prompts(RecordedSettings(prompt="A knight rides to a castle at dusk.")) == (None, "A knight rides to a castle at dusk.")

    @pytest.mark.parametrize("recorded", [None, RecordedSettings(), RecordedSettings(scene="  ", prompt="")])
    def test_a_clip_without_settings_gives_none(self, recorded):
        assert recorded_prompts(recorded) == (None, None)


class TestBuildExtendSource:
    def test_carries_the_displayed_geometry_timing_and_recorded_settings(self):
        probe = _probe(coded_width=512, coded_height=256, width=256, height=512, rotation=90, r_frame_rate="30000/1001", frame_count=90, video_start=0.05, format_start=0.03)
        recorded = RecordedSettings(scene="A castle.", prompt="A knight rides in.", model="/models/ltx", lora="/loras/style.safetensors:0.8")

        source = build_extend_source("/clips/clip.mp4", probe, recorded)

        assert source == ExtendSource(
            path=source.path,
            width=256,
            height=512,
            fps=pytest.approx(29.97, abs=0.001),
            fps_text="30000/1001",
            frame_count=90,
            has_audio=True,
            video_start=pytest.approx(0.02),
            color_space="bt709",
            color_primaries="bt709",
            color_trc="bt709",
            scene="A castle.",
            prompt="A knight rides in.",
            model="/models/ltx",
            lora="/loras/style.safetensors:0.8",
        )
        assert str(source.path) == "/clips/clip.mp4"
        assert source.probe is probe

    def test_a_relative_path_is_recorded_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        source = build_extend_source("clips/../clip.mp4", _probe(), None)

        assert source.path == tmp_path.resolve() / "clip.mp4"

    def test_a_relative_path_under_home_is_recorded_with_a_tilde(self, tmp_path, monkeypatch):
        home = tmp_path.resolve()
        (home / "videos").mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(home / "videos")

        source = build_extend_source("clip.mp4", _probe(), None)

        assert collapse_home(str(source.path)) == "~/videos/clip.mp4"

    def test_the_home_directory_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))

        assert build_extend_source("~/clip.mp4", _probe(), None).path == tmp_path.resolve() / "clip.mp4"

    def test_a_missing_frame_count_is_rejected(self):
        with pytest.raises(ValueError, match="length"):
            build_extend_source("/clips/clip.mp4", _probe(frame_count=None), None)

    def test_a_clip_with_one_recorded_prompt_is_legacy(self):
        legacy = build_extend_source("/clips/clip.mp4", _probe(), RecordedSettings(prompt="A knight rides to a castle."))
        scened = build_extend_source("/clips/clip.mp4", _probe(), RecordedSettings(scene="A castle.", prompt="A knight rides in."))
        scene_only = build_extend_source("/clips/clip.mp4", _probe(), RecordedSettings(scene="A castle."))
        bare = build_extend_source("/clips/clip.mp4", _probe(), None)

        assert (legacy.legacy, scened.legacy, scene_only.legacy, bare.legacy) == (True, False, False, False)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"")
    return path


@pytest.fixture
def probed(monkeypatch):
    """Patch the external readers; the returned dict sets what they give."""
    state = {"probe": _probe(), "config": None, "audio_ok": True}

    def read_config(_path):
        if isinstance(state["config"], Exception):
            raise state["config"]
        return state["config"]

    monkeypatch.setattr(video_extend, "probe_video", MagicMock(side_effect=lambda _path: state["probe"]))
    monkeypatch.setattr(video_extend, "read_mp4_config", read_config)
    monkeypatch.setattr(video_extend, "audio_decodes", MagicMock(side_effect=lambda *_a, **_k: state["audio_ok"]))
    return state


class TestInspectExtendSource:
    def test_a_clean_clip_gives_a_source_and_no_problems(self, clip, probed):
        probed["config"] = {"scene": "A castle.", "prompt": "A knight rides in.", "model": "/models/ltx"}

        inspection = inspect_extend_source(clip)

        assert inspection.problems == ()
        assert inspection.probe is probed["probe"]
        assert inspection.source is not None
        assert (inspection.source.scene, inspection.source.prompt, inspection.source.model) == ("A castle.", "A knight rides in.", "/models/ltx")
        assert inspection.source.probe is inspection.probe
        assert inspection.config == probed["config"]

    def test_the_settings_of_a_mov_are_read_like_those_of_an_mp4(self, tmp_path, probed):
        mov = tmp_path / "clip.mov"
        mov.write_bytes(b"")
        probed["config"] = {"prompt": "A knight rides in."}

        inspection = inspect_extend_source(mov)

        assert inspection.config == probed["config"]
        assert inspection.recorded is not None and inspection.recorded.prompt == "A knight rides in."

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing.mp4"):
            inspect_extend_source(tmp_path / "missing.mp4")

    def test_a_file_without_a_video_stream(self, clip, monkeypatch):
        monkeypatch.setattr(video_extend, "probe_video", MagicMock(side_effect=ValueError("no video stream")))

        inspection = inspect_extend_source(clip)

        assert _codes(list(inspection.problems)) == ["no_video_stream"]
        assert inspection.source is None and inspection.probe is None

    @pytest.mark.parametrize(
        ("failure", "detail"),
        [(ValueError("video stream has no size"), "no size"), (RuntimeError("ffprobe could not read clip.mp4: Invalid data"), "Invalid data")],
        ids=["no size", "unreadable file"],
    )
    def test_a_file_that_cannot_be_probed_is_unreadable_not_missing(self, clip, monkeypatch, failure, detail):
        monkeypatch.setattr(video_extend, "probe_video", MagicMock(side_effect=failure))

        inspection = inspect_extend_source(clip)

        assert _codes(list(inspection.problems)) == ["unreadable"]
        assert detail in inspection.problems[0].message

    def test_upscaled_files_are_flagged_from_their_recorded_settings(self, clip, probed):
        probed["config"] = {"prompt": "A knight.", "generation": {"upscale": {"factor": 2}}}

        inspection = inspect_extend_source(clip)

        assert _codes(list(inspection.problems)) == ["upscaled"]
        assert inspection.source is None

    def test_unreadable_audio_is_flagged(self, clip, probed):
        probed["audio_ok"] = False

        assert _codes(list(inspect_extend_source(clip).problems)) == ["unreadable_audio"]

    def test_the_audio_check_covers_the_carried_frames_at_the_end(self, clip, probed):
        probed["probe"] = _probe(frame_count=72, video_start=0.1, format_start=0.0)

        inspect_extend_source(clip)

        call = video_extend.audio_decodes.call_args
        assert call.kwargs["start"] == pytest.approx(0.1 + (72 - CARRY_FRAMES) / 24)
        assert call.kwargs["duration"] == pytest.approx(CARRY_FRAMES / 24)

    def test_a_silent_clip_skips_the_audio_check(self, clip, probed):
        probed["probe"] = _probe(has_audio=False)

        inspect_extend_source(clip)

        video_extend.audio_decodes.assert_not_called()

    @pytest.mark.parametrize("failure", [None, subprocess.CalledProcessError(1, ["ffprobe"]), ValueError("bad json"), OSError("gone")])
    def test_missing_or_unreadable_settings_are_not_a_problem(self, clip, probed, failure):
        probed["config"] = failure

        inspection = inspect_extend_source(clip)

        assert inspection.problems == ()
        assert inspection.recorded is None and inspection.config is None
        assert inspection.source is not None
        assert (inspection.source.scene, inspection.source.prompt, inspection.source.model) == (None, None, None)


class TestLoadExtendSource:
    def test_returns_the_checked_source(self, clip, probed):
        probed["config"] = {"prompt": "A knight rides to a castle."}

        source = load_extend_source(clip)

        assert source.path == clip
        assert (source.width, source.height, source.fps_text, source.frame_count) == (512, 256, "24/1", 72)
        assert source.legacy is True

    def test_a_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_extend_source(tmp_path / "missing.mp4")

    def test_names_every_problem(self, clip, probed):
        probed["probe"] = _probe(width=500, coded_width=500, field_order="tt")

        with pytest.raises(ValueError) as caught:
            load_extend_source(clip)

        message = str(caught.value)
        assert "clip.mp4 cannot be extended" in message
        assert "multiples of 32" in message and "interlaced" in message


_source = partial(make_extend_source, path=Path("clip.mp4"), frame_count=97, scene=None, prompt=None)


class TestResolveExtendFrames:
    @pytest.mark.parametrize("strict", [False, True])
    def test_neither_length_gives_the_two_second_default(self, strict):
        assert resolve_extend_frames(frames=None, seconds=None, fps=24.0, strict=strict) == (48, None)
        assert resolve_extend_frames(frames=None, seconds=None, fps=30.0, strict=strict)[0] == 64

    @pytest.mark.parametrize("strict", [False, True])
    def test_frames_win_over_seconds(self, strict):
        assert resolve_extend_frames(frames=24, seconds=10.0, fps=24.0, strict=strict) == (24, None)

    @pytest.mark.parametrize(("requested", "snapped"), [(50, 48), (53, 56), (9, 8), (1, 8)])
    def test_lenient_frames_snap_and_say_so_in_the_callers_words(self, requested, snapped):
        frames, note = resolve_extend_frames(frames=requested, seconds=None, fps=24.0, strict=False, frames_name="--extend-frames")
        assert frames == snapped
        assert note is not None and "--extend-frames" in note and str(requested) in note and str(snapped) in note

    @pytest.mark.parametrize("requested", [12, 0, -8, 4, 7])
    def test_strict_frames_must_be_a_multiple_of_eight_of_at_least_eight(self, requested):
        with pytest.raises(ValueError, match="extend_frames must be a multiple of 8, at least 8"):
            resolve_extend_frames(frames=requested, seconds=None, fps=24.0, strict=True)

    def test_strict_error_names_the_callers_option(self):
        with pytest.raises(ValueError, match="frames_field must be a multiple of 8"):
            resolve_extend_frames(frames=12, seconds=None, fps=24.0, strict=True, frames_name="frames_field")

    def test_lenient_seconds_on_a_step_give_no_note(self):
        assert resolve_extend_frames(frames=None, seconds=2.0, fps=24.0, strict=False) == (48, None)

    def test_lenient_seconds_off_a_step_explain_the_conversion(self):
        frames, note = resolve_extend_frames(frames=None, seconds=2.0, fps=30.0, strict=False, seconds_name="--extend-seconds")
        assert frames == 64
        assert note is not None and "--extend-seconds" in note and "64" in note

    def test_strict_seconds_convert_silently(self):
        assert resolve_extend_frames(frames=None, seconds=2.0, fps=30.0, strict=True) == (64, None)

    @pytest.mark.parametrize("seconds", [0.0, -1.0])
    def test_strict_seconds_must_be_positive(self, seconds):
        with pytest.raises(ValueError, match="extend_seconds must be positive"):
            resolve_extend_frames(frames=None, seconds=seconds, fps=24.0, strict=True)

    def test_strict_seconds_report_an_unusable_frame_rate_under_their_name(self):
        with pytest.raises(ValueError, match="extend_seconds: The frame rate must be a finite number"):
            resolve_extend_frames(frames=None, seconds=2.0, fps=0.0, strict=True)


class TestExtendEnhanceContext:
    def test_describes_the_window_the_source_and_the_new_frames(self):
        context = extend_enhance_context(_source(scene="A street.", prompt="She waves."), 48, no_audio=False)
        assert context.extends is True
        assert context.seconds == pytest.approx(73 / 24)
        assert context.new_seconds == pytest.approx(2.0)
        assert context.previous == "She waves."
        assert context.extract_from is None
        assert context.audio is True
        assert context.reference_image is False

    def test_the_window_is_in_seconds_at_the_source_frame_rate_and_the_source_size_picks_the_orientation(self):
        context = extend_enhance_context(_source(width=256, height=512, fps=30000 / 1001, fps_text="30000/1001"), 96, no_audio=False)
        assert context.seconds == pytest.approx(121 / (30000 / 1001))
        assert context.new_seconds == pytest.approx(96 / (30000 / 1001))
        assert context.orientation == "portrait"

    def test_a_legacy_source_is_extracted_from_and_a_silent_job_has_no_sound(self):
        context = extend_enhance_context(_source(prompt="She waves."), 48, no_audio=True)
        assert context.extract_from == "She waves."
        assert context.audio is False

    def test_a_silent_source_stays_silent(self):
        assert extend_enhance_context(_source(has_audio=False), 48, no_audio=False).audio is False

    @pytest.mark.parametrize(
        ("source_scene", "prompt", "expected"),
        [(None, " She  waves. ", True), (None, "She leaves.", False), (None, None, False), ("A street.", "She waves.", False)],
    )
    def test_the_prompt_is_original_only_for_a_legacy_source_that_it_still_equals(self, source_scene, prompt, expected):
        context = extend_enhance_context(_source(scene=source_scene, prompt="She waves."), 48, no_audio=False, prompt=prompt)
        assert context.prompt_is_original is expected


class TestStreamOffset:
    @pytest.mark.parametrize(("video_start", "format_start", "expected"), [(0.5, 0.0, 0.5), (0.5, 0.25, 0.25), (0.0, 0.5, 0.0), (None, None, 0.0), (0.2, None, 0.2)])
    def test_the_source_records_how_late_its_video_starts_after_the_container(self, video_start, format_start, expected):
        probe = _probe(video_start=video_start, format_start=format_start)
        source = build_extend_source("clip.mp4", probe, None)
        assert source.video_start == pytest.approx(expected)


class TestExtendAudioOutput:
    @pytest.mark.parametrize(("has_audio", "no_audio", "expected"), [(True, False, True), (True, True, False), (False, False, False), (False, True, False)])
    def test_sound_needs_a_source_with_audio_and_no_no_audio(self, has_audio, no_audio, expected):
        assert extend_audio_output(_source(has_audio=has_audio), no_audio=no_audio) is expected


class TestWindowArithmetic:
    @pytest.mark.parametrize(("extend", "expected"), [(48, 73), (8, 33)])
    def test_window_frames_add_the_carry(self, extend, expected):
        assert window_frames(extend) == expected

    @pytest.mark.parametrize(("family", "extend", "expected"), [("ltx", 16, 41), ("h3", 17, 39), ("h3", 34, 56)])
    def test_window_frames_add_the_carry_of_the_family(self, family, extend, expected):
        assert window_frames(extend, video_family(load_config(), family)) == expected

    def test_the_carry_fills_whole_latent_frames(self):
        assert carry_latent_frames() == 4

    def test_the_new_latent_frames_of_a_window_are_n(self):
        assert [(window_frames(8 * n) - 1) // 8 + 1 - carry_latent_frames() for n in (1, 6)] == [1, 6]

    @pytest.mark.parametrize(("frames", "expected"), [(0, 8), (1, 8), (11, 8), (12, 16), (13, 16), (47, 48), (50, 48), (52, 48), (60, 64), (100, 96)])
    def test_snap_extend_frames(self, frames, expected):
        assert snap_extend_frames(frames) == expected

    @pytest.mark.parametrize(
        ("seconds", "fps", "expected"),
        [
            (2.0, 24, 48),
            (2.0, 30, 64),  # 60 frames is 7.5 steps: ties go to the even step
            (4.0, 25, 96),  # 100 frames is 12.5 steps: ties go to the even step
            (0.1, 24, 8),
            (3.0, 24, 72),
            (2.0, 29.97, 56),  # 59.94 frames is just under 7.5 steps
        ],
    )
    def test_extend_frames_from_seconds(self, seconds, fps, expected):
        assert extend_frames_from_seconds(seconds, fps) == expected

    @pytest.mark.parametrize("seconds", [float("inf"), float("-inf"), float("nan"), 1e999, 0.0, -1.0])
    def test_extend_frames_from_seconds_rejects_a_length_that_is_not_finite_and_positive(self, seconds):
        with pytest.raises(ValueError, match="finite number greater than 0"):
            extend_frames_from_seconds(seconds, 24)

    @pytest.mark.parametrize("fps", [0.0, -24.0, float("inf"), float("nan")])
    def test_extend_frames_from_seconds_rejects_an_unusable_frame_rate(self, fps):
        with pytest.raises(ValueError, match="finite number greater than 0"):
            extend_frames_from_seconds(2.0, fps)

    @pytest.mark.parametrize("frames", [float("inf"), float("nan")])
    def test_snap_extend_frames_rejects_a_count_that_is_not_finite(self, frames):
        with pytest.raises(ValueError, match="finite number"):
            snap_extend_frames(frames)

    def test_apply_extend_args_sets_what_the_workflow_reads(self):
        source = _source(width=256, height=512, fps=30.0, fps_text="30/1", frame_count=90)
        args = argparse.Namespace(width=704, height=448, fps=24, num_frames=121)

        apply_extend_args(args, source, 48)

        assert (args.extend_source, args.extend_frames) == (source, 48)
        assert (args.width, args.height, args.fps, args.num_frames) == (256, 512, 30.0, 73)

    @pytest.mark.parametrize(("family", "extend", "window"), [("ltx", 16, 41), ("h3", 17, 39)])
    def test_apply_extend_args_sizes_the_window_by_the_family(self, family, extend, window):
        args = argparse.Namespace(width=704, height=448, fps=24, num_frames=121)

        apply_extend_args(args, _source(), extend, video_family(load_config(), family))

        assert args.num_frames == window

    @pytest.mark.parametrize(("extend", "window"), [(51, 56), (17, 22), (34, 39), (102, 107)])
    def test_a_family_that_carries_one_frame_moves_the_window_to_its_frame_rule(self, extend, window):
        mac = dataclasses.replace(video_family(load_config(), "h3"), carry_frames=1)
        args = argparse.Namespace(width=704, height=448, fps=24, num_frames=121)

        apply_extend_args(args, _source(), extend, mac)

        assert window_frames(extend, mac) == args.num_frames == window

    def test_the_rewrite_context_of_a_one_frame_carry_spans_the_window_the_mac_renders(self):
        mac = dataclasses.replace(video_family(load_config(), "h3"), carry_frames=1)

        context = extend_enhance_context(_source(), 51, no_audio=False, family=mac)

        assert context.seconds == pytest.approx(56 / 24)
        assert context.new_seconds == pytest.approx(51 / 24)

    def test_the_rewrite_context_spans_the_window_of_the_family(self):
        context = extend_enhance_context(_source(), 17, no_audio=False, family=video_family(load_config(), "h3"))

        assert context.seconds == pytest.approx(39 / 24)
        assert context.new_seconds == pytest.approx(17 / 24)

    @pytest.mark.parametrize(
        ("extend", "fps", "expected"),
        [(48, 24, (50000, 96000)), (8, 24, (50000, 16000)), (64, 30, (40000, 102400))],
    )
    def test_window_audio_cut_in_samples(self, extend, fps, expected):
        assert window_audio_cut(extend, fps) == expected

    @pytest.mark.parametrize(("fps", "expected"), [(24, 12000), (30, 12000), (8, 12000), (1000, 1200)])
    def test_audio_lead_is_a_quarter_second_at_most(self, fps, expected):
        assert audio_lead_samples(fps) == min(expected, round(CARRY_FRAMES * AUDIO_SAMPLE_RATE / fps))

    def test_window_audio_cut_rounds_to_whole_samples(self):
        skip, keep = window_audio_cut(48, 29.97)

        assert (skip, keep) == (round(25 * 48000 / 29.97), round(48 * 48000 / 29.97))


class TestCutWindowAudio:
    def _samples(self, length: int) -> np.ndarray:
        return np.tile(np.arange(length, dtype=np.float32), (2, 1))

    def test_drops_the_skipped_samples_and_keeps_the_rest(self):
        cut = cut_window_audio(self._samples(10), skip=3, keep=4)

        assert cut.shape == (2, 4)
        assert cut[0].tolist() == [3.0, 4.0, 5.0, 6.0]

    def test_a_short_end_fades_out_and_is_then_zero(self):
        samples = np.ones((2, 1000 + 2000 - 500), dtype=np.float32)

        cut = cut_window_audio(samples, skip=1000, keep=2000)

        assert cut.shape == (2, 2000)
        fade_start = 1500 - AUDIO_FADE_SAMPLES
        assert cut[0, :fade_start].min() == 1.0
        assert np.all(np.diff(cut[0, fade_start:1500]) < 0)
        assert cut[0, 1499] == 0.0
        assert not cut[:, 1500:].any()

    def test_a_short_end_shorter_than_the_fade_still_ends_at_zero(self):
        cut = cut_window_audio(self._samples(8), skip=3, keep=8)

        assert cut.shape == (2, 8)
        assert cut[1].tolist()[-3:] == [0.0, 0.0, 0.0]
        assert cut[1, 0] == 3.0

    def test_a_full_length_end_is_not_faded(self):
        cut = cut_window_audio(np.ones((2, 5000), dtype=np.float32), skip=1000, keep=4000)

        assert cut[:, -1].min() == 1.0

    def test_the_lead_comes_from_before_the_skip(self):
        cut = cut_window_audio(self._samples(10), skip=5, keep=4, lead=3)

        assert cut.shape == (2, 7)
        assert cut[0].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    def test_a_short_end_still_gives_the_lead_and_the_new_samples(self):
        cut = cut_window_audio(np.ones((2, 1500), dtype=np.float32), skip=1000, keep=800, lead=300)

        assert cut.shape == (2, 1100)
        assert cut[0, :300].min() == 1.0

    def test_the_input_is_not_modified(self):
        samples = np.ones((2, 100), dtype=np.float32)

        cut_window_audio(samples, skip=10, keep=200)

        assert samples.min() == 1.0

    def test_audio_shorter_than_the_skip_is_all_zeros(self):
        cut = cut_window_audio(self._samples(2), skip=5, keep=4)

        assert cut.shape == (2, 4)
        assert not cut.any()

    def test_a_long_end_is_trimmed(self):
        assert cut_window_audio(self._samples(100), skip=10, keep=20).shape == (2, 20)

    def test_keeps_the_channel_count_and_dtype(self):
        cut = cut_window_audio(np.zeros((1, 10), dtype=np.float32), skip=1, keep=12)

        assert cut.shape == (1, 12) and cut.dtype == np.float32


class TestResolveExtendFramesStep:
    """A family whose extension adds frames in another step (17) rounds and checks against that step."""

    def test_a_lenient_length_snaps_to_the_step_with_a_note(self):
        assert resolve_extend_frames(frames=58, seconds=None, fps=24.0, strict=False, frames_name="--extend-frames", step=17) == (51, "--extend-frames 58 rounded to 51 (a multiple of 17).")

    @pytest.mark.parametrize(("frames", "expected"), [(26, 34), (4, 17), (8, 17), (68, 68)])
    def test_lenient_lengths_snap_to_a_multiple_of_the_step(self, frames, expected):
        assert resolve_extend_frames(frames=frames, seconds=None, fps=24.0, strict=False, step=17)[0] == expected

    def test_a_tie_rounds_to_the_even_multiple_like_the_ltx_step(self):
        # 25.5 is halfway between 17 and 34 multiples of 17 (1.5): halves go to the even count, 2 * 17.
        assert snap_extend_frames(25.5, 17) == 34
        assert snap_extend_frames(42.5, 17) == 34

    def test_strict_frames_must_be_a_multiple_of_the_step(self):
        assert resolve_extend_frames(frames=51, seconds=None, fps=24.0, strict=True, step=17) == (51, None)
        with pytest.raises(ValueError, match=r"^extend_frames must be a multiple of 17, at least 17\.$"):
            resolve_extend_frames(frames=58, seconds=None, fps=24.0, strict=True, step=17)

    def test_the_default_and_seconds_lengths_use_the_step(self):
        assert resolve_extend_frames(frames=None, seconds=None, fps=24.0, strict=False, step=17) == (51, None)
        assert resolve_extend_frames(frames=None, seconds=2.0, fps=24.0, strict=True, step=17) == (51, None)
        assert resolve_extend_frames(frames=None, seconds=2.0, fps=30.0, strict=True, step=17) == (68, None)
        frames, note = resolve_extend_frames(frames=None, seconds=2.0, fps=24.0, strict=False, seconds_name="--extend-seconds", step=17)
        assert frames == 51 and note == "--extend-seconds 2 is 48 frames at 24 fps; using 51 frames (2.12 s)."

    def test_the_step_defaults_to_the_ltx_step(self):
        assert resolve_extend_frames(frames=58, seconds=None, fps=24.0, strict=False)[0] == 56
