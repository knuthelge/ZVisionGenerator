"""Tests for joining a generated segment onto a source clip."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests._factories import make_extend_source, make_probe
from tests._fake_media import FakeProcess, completed, f32le

from zvisiongenerator.core.video_types import ExtendSegment, ExtendSource
from zvisiongenerator.utils import video_splice
from zvisiongenerator.utils.video_splice import (
    build_audio_graph,
    build_audio_prep_command,
    build_concat_copy_command,
    build_final_mux_command,
    build_join_audio,
    build_mux_command,
    build_reencode_command,
    build_segment_command,
    build_video_only_command,
    can_stream_copy,
    color_filter,
    mux_audio,
    run_ffmpeg,
    silent_end_length,
    splice_clips,
    write_pcm16_wav,
    write_segment_frames,
)


def _source(path: Path = Path("source.mp4"), **overrides: Any) -> ExtendSource:
    return make_extend_source(**{"path": path, "frame_count": 73, "scene": None, "prompt": None, **overrides})


def _arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class TestAudioPrepCommand:
    def test_an_aac_track_is_copied(self):
        command = build_audio_prep_command("in.mp4", "audio.m4a", copy=True)

        assert _arg(command, "-c:a") == "copy"
        assert "-b:a" not in command

    def test_any_other_track_is_encoded_as_aac_at_192k(self):
        command = build_audio_prep_command("in.webm", "audio.m4a", copy=False)

        assert (_arg(command, "-c:a"), _arg(command, "-b:a")) == ("aac", "192k")

    @pytest.mark.parametrize("copy", [True, False])
    def test_only_the_first_audio_stream_is_written(self, copy):
        command = build_audio_prep_command("in.mp4", "audio.m4a", copy=copy)

        assert _arg(command, "-map") == "0:a:0"
        assert "-vn" in command
        assert command[-1] == "audio.m4a"


class TestFinalMuxCommand:
    def test_copies_both_streams_with_the_video_moved_to_the_sources_first_frame(self):
        command = build_final_mux_command("video.mp4", "audio.m4a", "out.mp4", video_offset=0.021333)

        assert command.index("-itsoffset") < command.index("-i")
        assert _arg(command, "-itsoffset") == "0.021333"
        assert command[command.index("-i") + 1] == "video.mp4"
        assert [command[i + 1] for i, part in enumerate(command) if part == "-map"] == ["0:v:0", "1:a:0"]
        assert _arg(command, "-c") == "copy"
        assert command[-1] == "out.mp4"

    def test_the_audio_keeps_its_own_start_and_neither_stream_is_cut_to_the_other(self):
        command = build_final_mux_command("video.mp4", "audio.m4a", "out.mp4", video_offset=0.0)

        assert "-copyts" in command
        assert "-shortest" not in command

    def test_a_silent_result_copies_the_video_alone(self):
        command = build_final_mux_command("video.mp4", None, "out.mp4", video_offset=0.25)

        assert command.count("-i") == 1
        assert "-itsoffset" not in command
        assert [command[i + 1] for i, part in enumerate(command) if part == "-map"] == ["0:v:0"]

    @pytest.mark.parametrize("audio", ["audio.m4a", None])
    def test_the_only_metadata_is_what_is_passed(self, audio):
        command = build_final_mux_command("video.mp4", audio, "out.mp4", video_offset=0.0, metadata=["-metadata", "key=value"])

        assert _arg(command, "-map_metadata") == "-1"
        assert command[-3:-1] == ["-metadata", "key=value"]
        assert command.index("-map_metadata") < command.index("-metadata")


class TestColorFilter:
    def test_bt709_converts_with_the_bt709_matrix_and_sets_the_tags(self):
        assert color_filter("bt709", "bt709", "bt709") == "scale=out_color_matrix=bt709,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"

    @pytest.mark.parametrize("space", ["smpte170m", "bt470bg"])
    def test_sd_spaces_use_the_bt601_matrix(self, space):
        assert color_filter(space, "smpte170m", "smpte170m") == f"scale=out_color_matrix=bt601,setparams=colorspace={space}:color_primaries=smpte170m:color_trc=smpte170m"

    def test_untagged_is_a_null_filter(self):
        assert color_filter(None, None, None) == "null"

    def test_tags_are_copied_exactly_as_probed(self):
        assert color_filter("bt709", None, "iec61966-2-1") == "scale=out_color_matrix=bt709,setparams=colorspace=bt709:color_trc=iec61966-2-1"

    def test_unknown_space_keeps_its_tags_without_converting(self):
        assert color_filter("smpte240m", "smpte240m", None) == "setparams=colorspace=smpte240m:color_primaries=smpte240m"


class TestCanStreamCopy:
    def test_equal_streams_copy(self):
        assert can_stream_copy(make_probe(), make_probe(frame_count=16, has_audio=False))

    @pytest.mark.parametrize(
        "change",
        [
            {"codec": "hevc"},
            {"profile": "Main"},
            {"pix_fmt": "yuv444p"},
            {"coded_width": 640, "width": 640},
            {"height": 288, "coded_height": 288},
            {"time_base": "1/90000"},
            {"r_frame_rate": "30/1"},
            {"color_space": "bt709"},
            {"color_primaries": "bt709"},
            {"color_trc": "bt709"},
        ],
    )
    def test_any_difference_re_encodes(self, change):
        assert not can_stream_copy(make_probe(), make_probe(**change))

    def test_equal_colour_tags_copy(self):
        tags = {"color_space": "bt709", "color_primaries": "bt709", "color_trc": "bt709"}
        assert can_stream_copy(make_probe(**tags), make_probe(**tags))

    def test_rotated_source_is_always_re_encoded(self):
        rotated = make_probe(rotation=90, coded_width=256, coded_height=512, width=512, height=256)
        assert not can_stream_copy(rotated, make_probe())
        assert not can_stream_copy(make_probe(), rotated)
        assert not can_stream_copy(rotated, rotated)

    def test_non_square_pixels_re_encode(self):
        assert not can_stream_copy(make_probe(sample_aspect_ratio="4:3"), make_probe())
        assert can_stream_copy(make_probe(sample_aspect_ratio=None), make_probe(sample_aspect_ratio="0:1"))


class TestSegmentCommand:
    def test_reads_raw_frames_at_the_sources_geometry_and_encodes_at_crf_18(self):
        command = build_segment_command(512, 256, "24/1", "seg.mp4", color_space="bt709", color_primaries="bt709", color_trc="bt709")
        assert _arg(command, "-f") == "rawvideo"
        assert _arg(command, "-s") == "512x256"
        assert _arg(command, "-r") == "24/1"
        assert _arg(command, "-i") == "-"
        assert _arg(command, "-c:v") == "libx264"
        assert _arg(command, "-crf") == "18"
        assert command.count("-pix_fmt") == 2
        assert command[command.index("-c:v") : command.index("-c:v") + 4] == ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        assert _arg(command, "-vf").startswith("scale=out_color_matrix=bt709,setparams=colorspace=bt709")
        assert "-an" in command
        assert command[-1] == "seg.mp4"

    def test_untagged_source_gets_a_null_filter(self):
        assert _arg(build_segment_command(64, 64, "30000/1001", "seg.mp4"), "-vf") == "null"


class TestAudioGraph:
    def test_keeps_the_head_of_the_source_audio_then_appends_the_join(self):
        graph = build_audio_graph(source_input=1, wav_input=2, head_samples=66000, offset=0.25)
        assert graph.startswith("[1:a:0]aresample=48000:async=1:first_pts=0,")
        assert "atrim=start=0.250000,asetpts=PTS-STARTPTS,atrim=end_sample=66000[a0]" in graph
        assert "[2:a:0]aresample=48000," in graph
        assert graph.endswith("[a0][a1]concat=n=2:v=0:a=1[a]")

    def test_the_source_audio_is_never_padded(self):
        assert "apad" not in build_audio_graph(source_input=0, wav_input=2, head_samples=1000, offset=0.0)


class TestVideoOnlyCommand:
    def test_copies_the_first_video_stream_alone_without_metadata(self):
        command = build_video_only_command("source.mp4", "video.mp4")
        assert [command[i + 1] for i, part in enumerate(command) if part == "-map"] == ["0:v:0"]
        assert _arg(command, "-c") == "copy"
        assert "-an" in command
        assert _arg(command, "-map_metadata") == "-1"
        assert command[command.index("-i") + 1] == "source.mp4"
        assert command[-1] == "video.mp4"


class TestCopyCommand:
    def test_stream_copies_the_listed_video_and_encodes_one_audio_track(self):
        graph = build_audio_graph(source_input=1, wav_input=2, head_samples=66000, offset=0.0)
        command = build_concat_copy_command("list.txt", "source.mp4", "seg.wav", "out.mp4", audio_graph=graph)
        assert command[command.index("-f") : command.index("-f") + 6] == ["-f", "concat", "-safe", "0", "-i", "list.txt"]
        assert command.count("-i") == 3
        assert _arg(command, "-c:v") == "copy"
        assert _arg(command, "-c:a") == "aac"
        assert _arg(command, "-filter_complex") == graph
        assert [command[i + 1] for i, part in enumerate(command) if part == "-map"] == ["0:v:0", "[a]"]
        assert _arg(command, "-map_metadata") == "-1"
        assert "libx264" not in command
        assert command[-1] == "out.mp4"

    def test_no_audio_copies_the_video_alone(self):
        command = build_concat_copy_command("list.txt", "source.mp4", None, "out.mp4", audio_graph=None)
        assert command[command.index("-map") : command.index("-map") + 2] == ["-map", "0:v:0"]
        assert _arg(command, "-c") == "copy"
        assert "-an" in command
        assert "-filter_complex" not in command and "-c:a" not in command
        assert _arg(command, "-map_metadata") == "-1"


class TestReencodeCommand:
    def test_one_pass_with_the_colour_filter_inside_the_graph(self):
        graph = build_audio_graph(source_input=0, wav_input=2, head_samples=66000, offset=0.0)
        command = build_reencode_command("source.mp4", "seg.mp4", "seg.wav", "30000/1001", "out.mp4", color_space="bt709", color_primaries="bt709", color_trc="bt709", audio_graph=graph)
        filter_complex = _arg(command, "-filter_complex")
        assert filter_complex.startswith(
            "[0:v:0]setpts=PTS-STARTPTS[v0];[1:v:0]setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[c];[c]scale=out_color_matrix=bt709,setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709[v];"
        )
        assert filter_complex.endswith(graph)
        assert "-vf" not in command
        assert [command[i + 1] for i, part in enumerate(command) if part == "-map"] == ["[v]", "[a]"]
        assert _arg(command, "-c:v") == "libx264"
        assert _arg(command, "-crf") == "12"
        assert _arg(command, "-pix_fmt") == "yuv420p"
        assert _arg(command, "-r") == "30000/1001"
        assert _arg(command, "-c:a") == "aac"
        assert _arg(command, "-map_metadata") == "-1"
        assert command.count("-i") == 3

    def test_decodes_with_auto_rotation_on(self):
        command = build_reencode_command("source.mp4", "seg.mp4", None, "24/1", "out.mp4")
        assert "-noautorotate" not in command

    def test_untagged_source_uses_a_null_filter_and_silent_output_has_no_audio(self):
        command = build_reencode_command("source.mp4", "seg.mp4", None, "24/1", "out.mp4")
        assert _arg(command, "-filter_complex") == "[0:v:0]setpts=PTS-STARTPTS[v0];[1:v:0]setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[c];[c]null[v]"
        assert "-an" in command
        assert "-c:a" not in command
        assert command.count("-i") == 2
        assert _arg(command, "-map_metadata") == "-1"


class TestMuxCommand:
    def test_copies_the_video_and_encodes_the_wav_as_the_only_audio_track(self):
        command = build_mux_command("video.mp4", "audio.wav", "out.mp4")
        assert [command[i + 1] for i, part in enumerate(command) if part == "-i"] == ["video.mp4", "audio.wav"]
        assert [command[i + 1] for i, part in enumerate(command) if part == "-map"] == ["0:v:0", "1:a:0"]
        assert _arg(command, "-c:v") == "copy"
        assert _arg(command, "-c:a") == "aac"
        assert _arg(command, "-map_metadata") == "-1"
        assert "-shortest" not in command and "libx264" not in command
        assert command[-1] == "out.mp4"

    def test_a_failing_ffmpeg_raises_with_its_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(video_splice.subprocess, "run", lambda *_a, **_k: completed(returncode=1, stderr=b"Invalid data found"))
        with pytest.raises(RuntimeError, match="add the audio.*Invalid data found"):
            mux_audio(tmp_path / "video.mp4", tmp_path / "audio.wav", tmp_path / "out.mp4")


class TestWritePcmWav:
    def test_writes_every_sample_of_every_channel(self, tmp_path):
        samples = np.stack([np.linspace(-1, 1, 1234, dtype=np.float32), np.zeros(1234, dtype=np.float32)])
        path = tmp_path / "audio.wav"
        write_pcm16_wav(samples, path)
        with wave.open(str(path)) as reader:
            assert (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes()) == (2, 2, 48000, 1234)
            data = np.frombuffer(reader.readframes(1234), dtype=np.int16).reshape(1234, 2)
        assert data[0, 0] == -32767 and data[-1, 0] == 32767
        assert not data[:, 1].any()

    @pytest.mark.parametrize(("extra", "expected"), [((), 48000), ((32000,), 32000)])
    def test_the_sample_rate_defaults_to_the_joins_and_can_be_given(self, tmp_path, extra, expected):
        path = tmp_path / "audio.wav"
        write_pcm16_wav(np.zeros((2, 10), dtype=np.float32), path, *extra)
        with wave.open(str(path)) as reader:
            assert reader.getframerate() == expected

    def test_clips_out_of_range_values(self, tmp_path):
        path = tmp_path / "audio.wav"
        write_pcm16_wav(np.array([[2.0, -2.0]], dtype=np.float32), path)
        with wave.open(str(path)) as reader:
            assert np.frombuffer(reader.readframes(2), dtype=np.int16).tolist() == [32767, -32767]


class TestSpliceClipsMocked:
    """The orchestration with ffmpeg and ffprobe replaced."""

    def _run_splice(self, tmp_path, monkeypatch, *, source_probe=None, segment_probe=None, joined_probe=None, include_audio=True, wav=True, fail=False, existing=None, **source_overrides):
        source = _source(tmp_path / "source.mp4", **source_overrides)
        segment = ExtendSegment(tmp_path / "segment.mp4", tmp_path / "segment.wav" if wav else None)
        probes = {"source.mp4": source_probe or make_probe(), "segment.mp4": segment_probe or make_probe(frame_count=16, has_audio=False), "joined": joined_probe or make_probe(frame_count=89)}
        commands: list[list[str]] = []

        self.probed = []

        def fake_probe(path):
            name = str(path)
            self.probed.append(Path(name).name)
            return next(probe for key, probe in probes.items() if key in name)

        def fake_run(command):
            commands.append(command)
            if fail:
                raise RuntimeError("ffmpeg could not join the clips: boom")
            Path(command[-1]).write_bytes(b"joined")

        def fake_prepare(source, segment, work):
            self.prepared.append((source, segment))
            return 123456, work / "join.wav"

        self.prepared = []
        monkeypatch.setattr(video_splice, "probe_video", fake_probe)
        monkeypatch.setattr(video_splice, "run_ffmpeg", fake_run)
        monkeypatch.setattr(video_splice, "_prepare_join_audio", fake_prepare)
        monkeypatch.setattr(video_splice, "_prepare_silent_join_audio", fake_prepare)
        output = tmp_path / "out" / "result.mp4"
        output.parent.mkdir()
        if existing is not None:
            output.write_bytes(existing)
        try:
            result = splice_clips(source, segment, output, extend_frames=16, include_audio=include_audio)
        finally:
            self.output, self.commands = output, commands
        return result

    def test_matching_segment_takes_the_stream_copy_branch(self, tmp_path, monkeypatch):
        result = self._run_splice(tmp_path, monkeypatch)
        assert result == self.output
        assert self.output.read_bytes() == b"joined"
        remux, command = self.commands
        assert remux[remux.index("-map") + 1] == "0:v:0" and "-an" in remux  # the source's video alone, so its audio cannot offset it
        assert _arg(command, "-c:v") == "copy"
        assert "libx264" not in command

    def test_source_probe_on_the_source_is_reused(self, tmp_path, monkeypatch):
        self._run_splice(tmp_path, monkeypatch, probe=make_probe())
        assert "source.mp4" not in self.probed
        assert self.probed.count("segment.mp4") == 1  # the segment and the joined file are still probed
        assert len(self.commands) == 2  # the reused probe still matched the segment: stream copy

    def test_reused_probe_decides_the_branch(self, tmp_path, monkeypatch):
        self._run_splice(tmp_path, monkeypatch, probe=make_probe(profile="Main"))
        (command,) = self.commands
        assert _arg(command, "-c:v") == "libx264"
        assert "source.mp4" not in self.probed

    def test_a_rotated_source_is_always_re_encoded_upright(self, tmp_path, monkeypatch):
        self._run_splice(
            tmp_path,
            monkeypatch,
            source_probe=make_probe(rotation=90, width=256, height=512, coded_width=512, coded_height=256),
            width=256,
            height=512,
            joined_probe=make_probe(frame_count=89, width=256, height=512, coded_width=256, coded_height=512),
        )
        (command,) = self.commands
        assert _arg(command, "-c:v") == "libx264"
        assert "-noautorotate" not in command

    def test_source_without_a_probe_is_probed_once(self, tmp_path, monkeypatch):
        self._run_splice(tmp_path, monkeypatch)
        assert self.probed.count("source.mp4") == 1

    def test_concat_list_names_both_files_and_pins_the_sources_duration(self, tmp_path, monkeypatch):
        listings: list[str] = []
        monkeypatch.setattr(video_splice.Path, "write_text", lambda self, text, **kwargs: listings.append(text) or len(text))
        self._run_splice(tmp_path, monkeypatch)
        (listing,) = listings
        video_only = self.commands[0][-1]
        assert Path(video_only).name == "source-video.mp4"
        assert listing.splitlines() == [f"file '{video_only}'", f"duration {73 / 24:.6f}", f"file '{tmp_path / 'segment.mp4'}'"]

    def test_mismatching_segment_is_re_encoded_in_one_pass(self, tmp_path, monkeypatch):
        self._run_splice(tmp_path, monkeypatch, segment_probe=make_probe(profile="Main", frame_count=16, has_audio=False), color_space="bt709", color_primaries="bt709", color_trc="bt709")
        (command,) = self.commands
        assert _arg(command, "-c:v") == "libx264"
        assert _arg(command, "-crf") == "12"
        assert "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709" in _arg(command, "-filter_complex")
        assert _arg(command, "-map_metadata") == "-1"

    def test_audio_takes_the_head_the_join_asks_for_and_then_the_join(self, tmp_path, monkeypatch):
        self._run_splice(tmp_path, monkeypatch, video_start=0.04)
        command = self.commands[-1]
        graph = _arg(command, "-filter_complex")
        assert "atrim=start=0.040000" in graph
        assert "atrim=end_sample=123456[a0]" in graph  # the head that the join preparation asked for
        assert "[2:a:0]" in graph
        assert Path(command[[i for i, part in enumerate(command) if part == "-i"][2] + 1]).name == "join.wav"

    def test_fractional_frame_rate_is_exact(self, tmp_path, monkeypatch):
        self._run_splice(
            tmp_path,
            monkeypatch,
            fps=30000 / 1001,
            fps_text="30000/1001",
            frame_count=30,
            source_probe=make_probe(r_frame_rate="30000/1001"),
            segment_probe=make_probe(r_frame_rate="30000/1001", frame_count=16),
            joined_probe=make_probe(frame_count=46),
        )
        assert self.prepared
        assert video_splice._source_samples(self.prepared[0][0]) == 48048  # 30 * 1001 / 30000 * 48000

    @pytest.mark.parametrize("segment_probe", [None, make_probe(profile="Main", frame_count=16, has_audio=False)])
    @pytest.mark.parametrize(
        "case",
        [{"has_audio": False, "include_audio": False, "source_probe": make_probe(has_audio=False)}, {"include_audio": False}, {"wav": False}],
        ids=["silent source, audio not wanted", "audio not wanted", "no wav"],
    )
    def test_output_without_audio_has_no_audio_stream_or_map(self, tmp_path, monkeypatch, case, segment_probe):
        self._run_splice(tmp_path, monkeypatch, segment_probe=segment_probe, joined_probe=make_probe(frame_count=89, has_audio=False), **case)
        command = self.commands[-1]
        assert "-an" in command
        assert "-c:a" not in command
        assert "[a]" not in command
        assert "anullsrc" not in " ".join(command)
        if "source_probe" in case:
            assert not self.prepared  # no audio is decoded or built

    def test_failing_command_leaves_no_output_and_no_temporary_directory(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match="boom"):
            self._run_splice(tmp_path, monkeypatch, fail=True)
        assert list(self.output.parent.iterdir()) == []

    def test_wrong_frame_count_is_refused_and_cleaned_up(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match="88 frames, expected 89"):
            self._run_splice(tmp_path, monkeypatch, joined_probe=make_probe(frame_count=88))
        assert list(self.output.parent.iterdir()) == []

    def test_wrong_size_or_rotation_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match="upright"):
            self._run_splice(tmp_path, monkeypatch, joined_probe=make_probe(frame_count=89, rotation=90))

    def test_missing_audio_in_the_result_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match="audio"):
            self._run_splice(tmp_path, monkeypatch, joined_probe=make_probe(frame_count=89, has_audio=False))

    def test_an_existing_output_survives_a_refused_result(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError):
            self._run_splice(tmp_path, monkeypatch, existing=b"old", joined_probe=make_probe(frame_count=1))
        assert self.output.read_bytes() == b"old"

    def test_an_existing_output_is_replaced_by_a_checked_result(self, tmp_path, monkeypatch):
        self._run_splice(tmp_path, monkeypatch, existing=b"old")
        assert self.output.read_bytes() == b"joined"
        assert [path.name for path in self.output.parent.iterdir()] == ["result.mp4"]


RATE = 48000


class TestWriteSegmentFrames:
    def _popen(self, monkeypatch, **process) -> list[FakeProcess]:
        made: list[FakeProcess] = []

        def popen(command, **kwargs):
            made.append(FakeProcess(stderr_file=kwargs.get("stderr"), **process))
            return made[0]

        monkeypatch.setattr(video_splice.subprocess, "Popen", popen)
        return made

    def test_pipes_every_frame_as_raw_rgb_bytes_in_order(self, monkeypatch):
        made = self._popen(monkeypatch)
        frames = [np.full((4, 6, 3), value, dtype=np.uint8) for value in (1, 2, 3)]

        write_segment_frames(iter(frames), ["ffmpeg"])

        assert made[0].stdin.written == b"".join(frame.tobytes() for frame in frames)
        assert made[0].stdin.closed

    def test_other_frame_types_are_sent_as_contiguous_uint8(self, monkeypatch):
        made = self._popen(monkeypatch)
        frame = np.arange(2 * 3 * 3, dtype=np.int64).reshape(2, 3, 3)[:, ::-1]  # not contiguous, not uint8

        write_segment_frames([frame], ["ffmpeg"])

        assert made[0].stdin.written == np.ascontiguousarray(frame, dtype=np.uint8).tobytes()

    def test_a_failing_ffmpeg_raises_with_its_message_and_the_description(self, monkeypatch):
        self._popen(monkeypatch, returncode=1, stderr=b"Invalid argument: colour tag\n")

        with pytest.raises(RuntimeError, match="could not encode the clip.*colour tag"):
            write_segment_frames([np.zeros((2, 2, 3), dtype=np.uint8)], ["ffmpeg"], description="the clip")

    def test_reports_ffmpegs_error_when_its_pipe_breaks(self, monkeypatch):
        # ffmpeg stopped before reading: both the writes and the flush of the close fail, and its exit code says why.
        self._popen(monkeypatch, returncode=1, stderr=b"Invalid argument: colour tag\n", broken_stdin=True)

        with pytest.raises(RuntimeError, match="extension segment.*colour tag"):
            write_segment_frames(np.zeros((8, 4, 4, 3), dtype=np.uint8), ["ffmpeg"])


class TestRunFfmpeg:
    def test_runs_the_command_without_a_stdin_and_hides_its_output(self, monkeypatch):
        run = MagicMock(return_value=completed())
        monkeypatch.setattr(video_splice.subprocess, "run", run)

        run_ffmpeg(["ffmpeg", "-i", "in.mp4", "out.mp4"])

        assert run.call_args.args[0] == ["ffmpeg", "-i", "in.mp4", "out.mp4"]
        assert run.call_args.kwargs["stdin"] == video_splice.subprocess.DEVNULL
        assert run.call_args.kwargs["capture_output"] is True

    def test_a_failure_raises_with_the_message_and_ffmpegs_error_without_its_banner(self, monkeypatch):
        stderr = b"ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers\nbuilt with gcc 14\nout.mp4: Permission denied\n"
        monkeypatch.setattr(video_splice.subprocess, "run", MagicMock(return_value=completed(returncode=1, stderr=stderr)))

        with pytest.raises(RuntimeError, match="^ffmpeg could not join the clips: out.mp4: Permission denied$"):
            run_ffmpeg(["ffmpeg"])

    def test_the_message_can_be_given(self, monkeypatch):
        monkeypatch.setattr(video_splice.subprocess, "run", MagicMock(return_value=completed(returncode=1, stderr=b"boom")))

        with pytest.raises(RuntimeError, match="^could not mux: boom$"):
            run_ffmpeg(["ffmpeg"], "could not mux")


class TestConcatList:
    def test_names_the_video_only_copy_then_the_segment_and_pins_the_sources_duration(self, tmp_path):
        source = _source(tmp_path / "source.mp4", frame_count=73)

        listing = video_splice._concat_list(source, tmp_path / "work" / "source-video.mp4", tmp_path / "segment.mp4")

        assert listing.splitlines() == [f"file '{tmp_path / 'work' / 'source-video.mp4'}'", f"duration {73 / 24:.6f}", f"file '{tmp_path / 'segment.mp4'}'"]

    def test_quotes_in_a_path_are_escaped_for_the_concat_demuxer(self, tmp_path):
        folder = tmp_path / "it's a folder"
        source = _source(folder / "my clip.mp4")

        listing = video_splice._concat_list(source, folder / "source-video.mp4", folder / "segment.mp4")

        escaped = str(folder).replace("'", "'\\''")
        assert listing.splitlines()[0] == f"file '{escaped}/source-video.mp4'"
        assert listing.splitlines()[2] == f"file '{escaped}/segment.mp4'"


def _wav(path: Path, audio: np.ndarray, rate: int = RATE) -> Path:
    write_pcm16_wav(audio, path, rate)
    return path


class TestSilentJoinAudio:
    """A silent source gets silence for its video length, then the new part of the segment's sound."""

    @pytest.mark.parametrize("lead", [0, 6000])
    def test_silence_for_the_sources_length_then_the_new_part_after_the_lead(self, tmp_path, lead):
        source = _source(tmp_path / "source.mp4", has_audio=False, frame_count=33)
        new_part = np.full((2, 32000), 0.5, dtype=np.float32)
        wav = _wav(tmp_path / "segment.wav", np.concatenate([np.full((2, lead), -0.5, dtype=np.float32), new_part], axis=1))
        segment = ExtendSegment(tmp_path / "segment.mp4", wav, lead_samples=lead)
        work = tmp_path / "work"
        work.mkdir()

        head, join_wav = video_splice._prepare_silent_join_audio(source, segment, work)

        audio = video_splice._read_wav(join_wav)
        silence = 33 * RATE // 24
        assert head == 0
        assert audio.shape == (2, silence + 32000)
        assert not audio[:, :silence].any()
        assert np.allclose(audio[:, silence:], 0.5, atol=1e-3)  # the lead-in stand-in is not used


class TestJoinAudio:
    """A source with sound keeps the head the join asks for, then gets the join WAV."""

    def test_the_head_and_the_join_add_up_to_the_video_plus_the_new_samples(self, tmp_path, monkeypatch):
        source = _source(tmp_path / "source.mp4", frame_count=48)
        samples = 48 * RATE // 24
        short = 1500  # the sound ends this many samples before the video
        tail = np.full((2, 480), 0.2, dtype=np.float32)
        monkeypatch.setattr(video_splice, "_decode_source_tail", lambda src, wanted, fade: (samples - short, tail))
        window = np.full((2, 12000 + 16 * 2000), 0.2, dtype=np.float32)
        segment = ExtendSegment(tmp_path / "segment.mp4", _wav(tmp_path / "segment.wav", window), lead_samples=12000)
        work = tmp_path / "work"
        work.mkdir()

        head, join_wav = video_splice._prepare_join_audio(source, segment, work)

        join = video_splice._read_wav(join_wav)
        assert head + join.shape[1] == samples + 16 * 2000
        assert head <= samples - short  # the blend takes the end of the head from the window

    def test_a_sound_that_reaches_the_end_of_the_video_is_kept_whole(self, tmp_path, monkeypatch):
        source = _source(tmp_path / "source.mp4", frame_count=48)
        samples = 48 * RATE // 24
        monkeypatch.setattr(video_splice, "_decode_source_tail", lambda src, wanted, fade: (samples, np.zeros((2, 480), dtype=np.float32)))
        window = np.full((2, 12000 + 32000), 0.3, dtype=np.float32)
        segment = ExtendSegment(tmp_path / "segment.mp4", _wav(tmp_path / "segment.wav", window), lead_samples=12000)
        work = tmp_path / "work"
        work.mkdir()

        head, join_wav = video_splice._prepare_join_audio(source, segment, work)

        assert head == samples
        assert video_splice._read_wav(join_wav).shape == (2, 32000)


class TestDecodeSourceTail:
    SAMPLES = 48000

    def _decode(self, monkeypatch, audio: np.ndarray, *, returncode: int = 0, stderr: bytes = b"", tail: int = 480):
        popen = MagicMock(side_effect=lambda command, **kw: FakeProcess(f32le(audio), returncode=returncode, stderr=stderr, stderr_file=kw.get("stderr")))
        monkeypatch.setattr(video_splice.subprocess, "Popen", popen)
        return video_splice._decode_source_tail(_source(Path("/clips/clip.mp4"), video_start=0.25), self.SAMPLES, tail), popen

    def test_decodes_the_first_audio_stream_onto_the_video_time_axis_as_stereo_float(self, monkeypatch):
        _, popen = self._decode(monkeypatch, np.ones((2, 1000), dtype=np.float32))

        command = popen.call_args.args[0]
        assert command[command.index("-map") + 1] == "0:a:0"
        assert command[command.index("-af") + 1] == video_splice.source_audio_chain(samples=self.SAMPLES, offset=0.25)
        assert (command[command.index("-f") + 1], command[command.index("-ac") + 1]) == ("f32le", "2")

    def test_returns_the_length_and_the_last_samples(self, monkeypatch):
        audio = np.stack([np.arange(5000), np.arange(5000)]).astype(np.float32)

        (length, tail), _ = self._decode(monkeypatch, audio)

        assert length == 5000
        assert tail.shape == (2, 480)
        assert np.array_equal(tail, audio[:, -480:])

    def test_a_silent_end_is_not_counted_and_not_in_the_tail(self, monkeypatch):
        audio = np.concatenate([np.full((2, 5000), 0.5, dtype=np.float32), np.zeros((2, 1000), dtype=np.float32)], axis=1)

        (length, tail), _ = self._decode(monkeypatch, audio)

        assert length == 5000
        assert (tail == 0.5).all()

    def test_a_failing_ffmpeg_raises_with_its_message(self, monkeypatch):
        with pytest.raises(RuntimeError, match="could not read the source's audio.*Invalid data found"):
            self._decode(monkeypatch, np.zeros((2, 0), dtype=np.float32), returncode=1, stderr=b"Invalid data found")


class TestBuildJoinAudio:
    """The pure assembly of the audio that follows the source's own."""

    LEAD = 100
    SAMPLES = 1000

    def _window(self, new: int = 400) -> np.ndarray:
        """Window audio whose sample ``j`` is the number ``SAMPLES - LEAD + j`` (the source time axis), on two channels."""
        return np.tile(np.arange(self.SAMPLES - self.LEAD, self.SAMPLES + new, dtype=np.float32), (2, 1))

    def _tail(self, length: int, size: int) -> np.ndarray:
        """The source's last ``size`` samples when its audio is ``length`` long, as the negative of their position."""
        return np.tile(-np.arange(length - size, length, dtype=np.float32), (2, 1))

    def _join(self, length: int, *, fade: int = 8, lead: int | None = None, new: int = 400):
        lead = self.LEAD if lead is None else lead
        window = self._window(new)[:, self.LEAD - lead :]
        return build_join_audio(self._tail(length, fade), length, window, lead=lead, samples=self.SAMPLES, fade=fade)

    @pytest.mark.parametrize("length", [1000, 969, 900, 950, 905, 899, 850, 0])
    @pytest.mark.parametrize("lead", [0, 100])
    def test_head_and_join_always_add_up_to_the_video_plus_the_new_samples(self, length, lead):
        head, join = self._join(length, lead=lead)
        assert head + join.shape[1] == self.SAMPLES + 400

    def test_a_track_that_reaches_the_end_of_the_video_is_left_alone(self):
        head, join = self._join(1000)
        assert head == 1000
        np.testing.assert_array_equal(join, self._window()[:, self.LEAD :])

    def test_the_missing_end_is_filled_from_the_window_after_a_blend(self):
        head, join = self._join(969, fade=8)
        assert head == 969 - 8
        assert join.shape == (2, 8 + (self.SAMPLES - 969) + 400)
        # After the blend come the window's own samples, which are the missing end of the track and then the new part.
        np.testing.assert_array_equal(join[:, 8:], self._window()[:, 969 - (self.SAMPLES - self.LEAD) :])
        assert not (join[:, 8:] == 0).any()
        assert join[0, 0] == pytest.approx(-961 * (1 - 0.5 / 8) + 961 * (0.5 / 8))  # almost entirely the source's sample

    def test_the_blend_steps_from_the_source_to_the_window(self):
        window = np.ones((2, 100 + 40), dtype=np.float32)
        tail = np.zeros((2, 10), dtype=np.float32)
        _, join = build_join_audio(tail, 950, window, lead=100, samples=1000, fade=10)
        blend = join[0, :10]
        assert np.all(np.diff(blend) > 0)
        assert 0 < blend[0] < 0.1 and 0.9 < blend[-1] < 1

    def test_a_gap_longer_than_the_lead_stays_silent_before_the_window(self):
        head, join = self._join(850, fade=8)
        assert head == 850  # nothing to blend with: the window does not reach back to the end of the source's audio
        assert join.shape[1] == (self.SAMPLES - self.LEAD - 850) + self.LEAD + 400
        assert not join[:, :50].any()
        np.testing.assert_array_equal(join[:, 50:], self._window())

    def test_a_blend_shorter_than_the_fade_when_the_window_barely_reaches_the_audio_end(self):
        head, join = self._join(905, fade=8)  # the window starts at sample 900: 5 samples to blend over
        assert head == 900
        assert join.shape[1] + head == self.SAMPLES + 400

    def test_without_a_lead_the_gap_is_silent(self):
        head, join = self._join(969, lead=0)
        assert head == 969
        assert not join[:, :31].any()
        assert join[0, 31] == 1000

    def test_the_input_is_not_modified(self):
        window, tail = self._window(), self._tail(969, 8)
        before = window.copy(), tail.copy()
        build_join_audio(tail, 969, window, lead=self.LEAD, samples=self.SAMPLES, fade=8)
        np.testing.assert_array_equal(window, before[0])
        np.testing.assert_array_equal(tail, before[1])


class TestSilentEndLength:
    def _audio(self, loud: int, silent: int, level: float = 0.0) -> np.ndarray:
        return np.concatenate([np.full((2, loud), 0.5, dtype=np.float32), np.full((2, silent), level, dtype=np.float32)], axis=1)

    def test_a_run_of_zeros_at_the_end_is_found(self):
        assert silent_end_length(self._audio(1000, 1100)) == 1100

    def test_samples_within_two_lsb_count_as_silent(self):
        assert silent_end_length(self._audio(1000, 500, level=2 / 32767)) == 500
        assert silent_end_length(self._audio(1000, 500, level=3 / 32767)) == 0

    def test_a_run_of_two_milliseconds_or_less_is_not_missing_sound(self):
        assert silent_end_length(self._audio(1000, 96)) == 0
        assert silent_end_length(self._audio(1000, 97)) == 97

    def test_audio_that_is_loud_to_its_end_has_none(self):
        assert silent_end_length(self._audio(1000, 0)) == 0

    def test_a_silence_longer_than_the_lead_in_is_the_clips_own(self):
        assert silent_end_length(self._audio(1000, 12001)) == 0
        assert silent_end_length(self._audio(1000, 12000)) == 12000

    def test_both_channels_must_be_silent(self):
        audio = self._audio(1000, 500)
        audio[1, -200] = 0.5
        assert silent_end_length(audio) == 199

    def test_silence_in_the_middle_is_ignored(self):
        audio = np.concatenate([self._audio(100, 500), self._audio(300, 0)], axis=1)
        assert silent_end_length(audio) == 0

    def test_no_samples_is_not_a_silent_end(self):
        assert silent_end_length(np.zeros((2, 0), dtype=np.float32)) == 0
