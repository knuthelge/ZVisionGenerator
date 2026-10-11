"""Tests for joining a generated segment onto a source clip."""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _media import REAL, ffmpeg, first_frame, numbered_frames, rotated_copy
from conftest import make_extend_source, make_probe

from zvisiongenerator.core.video_types import ExtendSegment, ExtendSource
from zvisiongenerator.utils import video_splice
from zvisiongenerator.utils.video_extend import build_extend_source, cut_window_audio
from zvisiongenerator.utils.video_probe import probe_video
from zvisiongenerator.utils.video_splice import (
    build_audio_graph,
    build_concat_copy_command,
    build_join_audio,
    build_reencode_command,
    build_segment_command,
    build_video_only_command,
    can_stream_copy,
    color_filter,
    silent_end_length,
    splice_clips,
    write_pcm16_wav,
)


def _source(path: Path = Path("source.mp4"), **overrides: Any) -> ExtendSource:
    return make_extend_source(**{"path": path, "frame_count": 73, "scene": None, "prompt": None, **overrides})


def _arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


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
        monkeypatch.setattr(video_splice, "_run", fake_run)
        monkeypatch.setattr(video_splice, "_prepare_join_audio", fake_prepare)
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
        [{"has_audio": False, "source_probe": make_probe(has_audio=False)}, {"include_audio": False}, {"wav": False}],
        ids=["silent source", "audio not wanted", "no wav"],
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


FPS = 24
RATE = 48000


def _make_clip(path: Path, frames: np.ndarray, *, audio_seconds: float | None = None, video_args: tuple[str, ...] = ("-crf", "18"), metadata: bool = False) -> ExtendSource:
    """Encode raw frames (and a sine track) into ``path`` and return the source for it."""
    _, height, width, _ = frames.shape
    audio = ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={audio_seconds}"] if audio_seconds else []
    maps = ["-map", "1:v", "-map", "0:a", "-c:a", "aac"] if audio_seconds else []
    tags = ["-metadata", "title=source-title"] if metadata else []
    ffmpeg(
        *audio,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(FPS),
        "-i",
        "-",
        *maps,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        *video_args,
        *tags,
        str(path),
        data=frames.tobytes(),
    )
    return build_extend_source(path, probe_video(path), None)


def _make_segment(directory: Path, source: ExtendSource, frames: np.ndarray, *, with_audio: bool = True) -> ExtendSegment:
    """Encode generated frames with the segment helper, and a tone as their audio."""
    video = directory / "segment.mp4"
    command = build_segment_command(source.width, source.height, source.fps_text, video, color_space=source.color_space, color_primaries=source.color_primaries, color_trc=source.color_trc)
    subprocess.run(command, input=frames.tobytes(), capture_output=True, check=True)
    wav = None
    if with_audio:
        wav = directory / "segment.wav"
        count = round(len(frames) / FPS * RATE)
        tone = 0.3 * np.sin(2 * np.pi * 300 * np.arange(count) / RATE)
        write_pcm16_wav(np.stack([tone, tone]), wav)
    return ExtendSegment(video, wav)


def _packet_hashes(path: Path, *, decoded: bool = False) -> list[str]:
    """Return a hash per video packet, or per decoded frame."""
    codec = [] if decoded else ["-c", "copy"]
    lines = ffmpeg("-i", str(path), "-map", "0:v:0", *codec, "-f", "framemd5", "-").decode().splitlines()
    return [line.rsplit(",", 1)[1].strip() for line in lines if line and not line.startswith("#")]


def _audio_pcm(path: Path) -> np.ndarray:
    data = ffmpeg("-i", str(path), "-map", "0:a:0", "-f", "s16le", "-ac", "2", "-ar", str(RATE), "-")
    return np.frombuffer(data, dtype=np.int16).reshape(-1, 2)


def _red_overlap(frame: np.ndarray, reference: np.ndarray) -> float:
    """Return the share of the reference's red patch that is also red in the frame."""

    def red(image: np.ndarray) -> np.ndarray:
        return (image[:, :, 0] > 200) & (image[:, :, 1] < 60)

    return float((red(frame) & red(reference)).sum() / red(reference).sum())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@REAL
class TestRealStreamCopy:
    def test_source_and_segment_are_kept_bit_for_bit(self, tmp_path):
        source = _make_clip(tmp_path / "source.mp4", numbered_frames(33, 128, 64), audio_seconds=1.4, metadata=True)
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64, offset=100))
        assert can_stream_copy(probe_video(source.path), probe_video(segment.video_path))
        before = _sha256(source.path)
        output = tmp_path / "joined.mp4"

        assert splice_clips(source, segment, output, extend_frames=16, include_audio=True) == output

        probe = probe_video(output)
        assert probe.frame_count == 33 + 16
        assert (probe.width, probe.height, probe.rotation) == (128, 64, 0)
        # Every decoded frame is the source's or the segment's own, and so is every packet except the first of each
        # part, which the muxer rewrites to carry the stream's parameter sets.
        decoded = _packet_hashes(output, decoded=True)
        assert decoded[:33] == _packet_hashes(source.path, decoded=True)
        assert decoded[33:] == _packet_hashes(segment.video_path, decoded=True)
        joined = _packet_hashes(output)
        assert joined[1:33] == _packet_hashes(source.path)[1:]
        assert joined[34:] == _packet_hashes(segment.video_path)[1:]
        assert _sha256(source.path) == before

    def test_parts_encoded_with_different_x264_settings_both_decode_exactly(self, tmp_path):
        # The mp4 keeps one avcC, but the concat demuxer writes each part's parameter sets in-band, so a segment
        # encoded with other settings than the source still decodes as it did on its own.
        source = _make_clip(tmp_path / "source.mp4", numbered_frames(33, 128, 64), audio_seconds=1.4, video_args=("-crf", "12"))
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64, offset=100))
        _make_clip(segment.video_path, numbered_frames(16, 128, 64, offset=100), video_args=("-crf", "18", "-refs", "1", "-bf", "0"))
        assert can_stream_copy(probe_video(source.path), probe_video(segment.video_path))
        output = tmp_path / "joined.mp4"

        splice_clips(source, segment, output, extend_frames=16, include_audio=True)

        assert _packet_hashes(output)[1:33] == _packet_hashes(source.path)[1:]  # the stream-copy branch kept the packets
        decoded = _packet_hashes(output, decoded=True)
        assert decoded[:33] == _packet_hashes(source.path, decoded=True)
        assert decoded[33:] == _packet_hashes(segment.video_path, decoded=True)

    def test_audio_is_the_sources_fitted_length_plus_the_new_audio(self, tmp_path):
        source = _make_clip(tmp_path / "source.mp4", numbered_frames(33, 128, 64), audio_seconds=1.4)  # audio longer than the video
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64))
        output = tmp_path / "joined.mp4"
        splice_clips(source, segment, output, extend_frames=16, include_audio=True)
        assert len(_audio_pcm(output)) == round(33 / FPS * RATE) + round(16 / FPS * RATE) == 98000

    def test_short_audio_is_padded_to_the_videos_length(self, tmp_path):
        source = _make_clip(tmp_path / "source.mp4", numbered_frames(48, 128, 64), audio_seconds=1.0)
        segment = _make_segment(tmp_path, source, numbered_frames(8, 128, 64))
        output = tmp_path / "joined.mp4"
        splice_clips(source, segment, output, extend_frames=8, include_audio=True)
        assert len(_audio_pcm(output)) == 48 * 2000 + 8 * 2000

    def test_video_timestamps_stay_even_when_the_audio_outlasts_the_video(self, tmp_path):
        source = _make_clip(tmp_path / "source.mp4", numbered_frames(33, 128, 64), audio_seconds=3.0)
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64))
        output = tmp_path / "joined.mp4"
        splice_clips(source, segment, output, extend_frames=16, include_audio=True)
        times = ffmpeg("-i", str(output), "-map", "0:v:0", "-c", "copy", "-f", "framemd5", "-").decode().splitlines()
        stamps = sorted(int(line.split(",")[2]) for line in times if line and not line.startswith("#"))
        steps = {later - earlier for earlier, later in zip(stamps, stamps[1:])}
        assert len(steps) == 1

    def test_no_source_metadata_is_carried_over(self, tmp_path):
        source = _make_clip(tmp_path / "source.mp4", numbered_frames(33, 128, 64), audio_seconds=1.4, metadata=True)
        assert "source-title" in ffmpeg("-i", str(source.path), "-f", "ffmetadata", "-").decode()
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64))
        output = tmp_path / "joined.mp4"
        splice_clips(source, segment, output, extend_frames=16, include_audio=True)
        assert "source-title" not in ffmpeg("-i", str(output), "-f", "ffmetadata", "-").decode()

    def test_leaves_only_the_output_in_its_directory(self, tmp_path):
        source = _make_clip(tmp_path / "source.mp4", numbered_frames(33, 128, 64))
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64), with_audio=False)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        splice_clips(source, segment, out_dir / "joined.mp4", extend_frames=16, include_audio=False)
        assert [path.name for path in out_dir.iterdir()] == ["joined.mp4"]

    def test_a_missing_wav_leaves_nothing_behind(self, tmp_path):
        source = _make_clip(tmp_path / "source.mp4", numbered_frames(33, 128, 64), audio_seconds=1.4)
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64), with_audio=False)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        broken = ExtendSegment(segment.video_path, tmp_path / "missing.wav")
        with pytest.raises(FileNotFoundError, match="missing.wav"):
            splice_clips(source, broken, out_dir / "joined.mp4", extend_frames=16, include_audio=True)
        assert list(out_dir.iterdir()) == []


def _tagged_source(tmp_path, space: str, **kwargs: Any) -> ExtendSource:
    """Encode a Main-profile clip with colour tags, so a High-profile segment cannot be stream copied onto it."""
    graph = {"bt709": "scale=out_color_matrix=bt709", "smpte170m": "scale=out_color_matrix=bt601"}[space]
    tags = f"{graph},setparams=colorspace={space}:color_primaries={space}:color_trc={space}"
    return _make_clip(tmp_path / "source.mp4", numbered_frames(33, 128, 64), video_args=("-crf", "18", "-profile:v", "main", "-vf", tags), **kwargs)


@REAL
class TestRealWithoutAudio:
    """Either join branch leaves no audio stream when none is wanted, and a silent source stays silent."""

    @staticmethod
    def _source(tmp_path, branch: str, **kwargs: Any) -> ExtendSource:
        if branch == "reencode":
            return _tagged_source(tmp_path, "bt709", **kwargs)
        return _make_clip(tmp_path / "source.mp4", numbered_frames(33, 128, 64), **kwargs)

    @pytest.mark.parametrize("branch", ["copy", "reencode"])
    def test_without_audio_the_result_has_no_audio_stream(self, tmp_path, branch):
        source = self._source(tmp_path, branch, audio_seconds=1.4)
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64))
        output = tmp_path / "joined.mp4"
        splice_clips(source, segment, output, extend_frames=16, include_audio=False)
        probe = probe_video(output)
        assert not probe.has_audio
        assert probe.frame_count == 49

    @pytest.mark.parametrize("branch", ["copy", "reencode"])
    def test_silent_source_stays_silent_even_with_new_audio(self, tmp_path, branch):
        source = self._source(tmp_path, branch)
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64))
        assert segment.audio_path is not None
        output = tmp_path / "joined.mp4"
        splice_clips(source, segment, output, extend_frames=16, include_audio=True)
        probe = probe_video(output)
        assert not probe.has_audio
        assert probe.frame_count == 49


@REAL
class TestRealReencode:
    @pytest.mark.parametrize("space", ["bt709", "smpte170m"])
    def test_colour_tags_survive(self, tmp_path, space):
        source = _tagged_source(tmp_path, space)
        assert (source.color_space, source.color_primaries, source.color_trc) == (space, space, space)
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64, offset=100), with_audio=False)
        assert not can_stream_copy(probe_video(source.path), probe_video(segment.video_path))  # Main against High
        output = tmp_path / "joined.mp4"
        splice_clips(source, segment, output, extend_frames=16, include_audio=False)
        probe = probe_video(output)
        assert probe.frame_count == 49
        assert (probe.color_space, probe.color_primaries, probe.color_trc) == (space, space, space)
        assert probe.pix_fmt == "yuv420p"

    def test_audio_sample_count_is_exact_and_the_result_has_one_audio_stream(self, tmp_path):
        source = _tagged_source(tmp_path, "bt709", audio_seconds=1.2)
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64))
        output = tmp_path / "joined.mp4"
        splice_clips(source, segment, output, extend_frames=16, include_audio=True)
        assert probe_video(output).has_audio
        assert len(_audio_pcm(output)) == 98000

    def test_a_re_encoded_result_stream_copies_on_the_next_extension(self, tmp_path):
        source = _tagged_source(tmp_path, "bt709", audio_seconds=1.2)
        first_segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64, offset=100))
        assert not can_stream_copy(probe_video(source.path), probe_video(first_segment.video_path))  # Main against High
        joined = tmp_path / "joined.mp4"
        splice_clips(source, first_segment, joined, extend_frames=16, include_audio=True)

        # The result carries the segment encoder's own stream parameters, so the next extension joins without re-encoding.
        next_source = build_extend_source(joined, probe_video(joined), None)
        second_dir = tmp_path / "second"
        second_dir.mkdir()
        second_segment = _make_segment(second_dir, next_source, numbered_frames(16, 128, 64, offset=200))
        assert can_stream_copy(probe_video(joined), probe_video(second_segment.video_path))
        extended = tmp_path / "extended.mp4"
        splice_clips(next_source, second_segment, extended, extend_frames=16, include_audio=True)

        assert probe_video(extended).frame_count == 33 + 16 + 16
        assert _packet_hashes(extended, decoded=True)[: 33 + 16] == _packet_hashes(joined, decoded=True)

    def test_audio_follows_a_source_whose_video_starts_late(self, tmp_path):
        click = np.zeros(3 * RATE, dtype=np.float32)
        click[RATE : RATE + 240] = np.random.default_rng(0).uniform(-1, 1, 240)  # a burst 1.0 s into the file
        wav = tmp_path / "click.wav"
        write_pcm16_wav(click[None, :], wav)
        plain = tmp_path / "plain.mp4"
        _make_clip(plain, numbered_frames(72, 128, 64))
        late = tmp_path / "late.mkv"
        ffmpeg("-i", str(wav), "-itsoffset", "0.5", "-i", str(plain), "-map", "1:v", "-map", "0:a", "-c:v", "copy", "-c:a", "aac", str(late))
        source = build_extend_source(late, probe_video(late), None)
        assert source.video_start == pytest.approx(0.5)
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64), with_audio=False)
        silent_wav = tmp_path / "quiet.wav"
        write_pcm16_wav(np.zeros((2, 32000), dtype=np.float32), silent_wav)
        output = tmp_path / "joined.mp4"
        splice_clips(source, ExtendSegment(segment.video_path, silent_wav), output, extend_frames=16, include_audio=True)
        pcm = _audio_pcm(output)
        assert len(pcm) == 72 * 2000 + 32000
        peak = int(np.abs(pcm[:, 0].astype(int)).argmax())
        assert abs(peak - 0.5 * RATE) < 1500  # 1.0 s in the file is 0.5 s after the first frame, not 1.0 s


@REAL
class TestRealRotated:
    @pytest.mark.parametrize(("angle", "size", "rotation"), [("-90", (64, 128), 90), ("90", (64, 128), 270), ("180", (128, 64), 180)])
    def test_rotated_source_comes_out_upright(self, tmp_path, angle, size, rotation):
        base = tmp_path / "base.mp4"
        _make_clip(base, numbered_frames(33, 128, 64), audio_seconds=1.4)
        rotated = rotated_copy(base, tmp_path / "rotated.mp4", angle)
        source = build_extend_source(rotated, probe_video(rotated), None)
        assert ((source.width, source.height), source.probe.rotation) == (size, rotation)
        segment = _make_segment(tmp_path, source, numbered_frames(16, *size, offset=100))
        assert not can_stream_copy(probe_video(source.path), probe_video(segment.video_path))
        output = tmp_path / "joined.mp4"

        splice_clips(source, segment, output, extend_frames=16, include_audio=True)

        probe = probe_video(output)
        assert ((probe.coded_width, probe.coded_height), probe.rotation) == (size, 0)
        assert probe.frame_count == 49
        assert len(_audio_pcm(output)) == 98000
        expected = first_frame(rotated, *size)  # ffmpeg's own auto-rotated decode of the source
        first = first_frame(output, *size)
        assert np.abs(first.astype(int) - expected.astype(int)).mean() < 3
        assert _red_overlap(first, expected) > 0.9  # the corner patch is where the player shows it
        assert _red_overlap(np.rot90(expected, 2), expected) < 0.1  # so a wrong orientation would be told apart


@REAL
class TestRealAdversarial:
    """Cases an independent reviewer added: wrong frame counts, NTSC rates, awkward paths."""

    def _clip_pair(self, tmp_path, *, count=33, extend=16, audio_seconds=1.4, name="source.mp4", **clip_args):
        source = _make_clip(tmp_path / name, numbered_frames(count, 128, 64), audio_seconds=audio_seconds, **clip_args)
        return source, _make_segment(tmp_path, source, numbered_frames(extend, 128, 64, offset=100))

    @pytest.mark.parametrize("error", [-1, 1])
    def test_a_frame_count_off_by_one_is_refused_and_the_existing_output_survives(self, tmp_path, error):
        source, segment = self._clip_pair(tmp_path)
        wrong = dataclasses.replace(source, frame_count=source.frame_count + error)  # as a bad nb_frames would give
        output = tmp_path / "joined.mp4"
        output.write_bytes(b"old")
        with pytest.raises(RuntimeError, match="frames, expected"):
            splice_clips(wrong, segment, output, extend_frames=16, include_audio=True)
        assert output.read_bytes() == b"old"
        assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".splice-")] == []

    @pytest.mark.parametrize("branch", ["copy", "reencode"])
    def test_ntsc_rate_gets_exact_frames_and_samples(self, tmp_path, branch):
        frames = numbered_frames(30, 128, 64)
        profile = ("-profile:v", "main") if branch == "reencode" else ()
        path = tmp_path / "source.mp4"
        sine = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1.4"]
        raw = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "128x64", "-r", "30000/1001", "-i", "-"]
        ffmpeg(*sine, *raw, "-map", "1:v", "-map", "0:a", "-c:a", "aac", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", *profile, str(path), data=frames.tobytes())
        source = build_extend_source(path, probe_video(path), None)
        assert (source.fps_text, source.frame_count) == ("30000/1001", 30)
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64, offset=100), with_audio=False)
        wav = tmp_path / "segment.wav"
        added = round(16 * 1001 / 30000 * RATE)  # 25626 samples
        write_pcm16_wav(np.zeros((2, added), dtype=np.float32), wav)
        assert can_stream_copy(probe_video(source.path), probe_video(segment.video_path)) == (branch == "copy")
        output = tmp_path / "joined.mp4"
        splice_clips(source, ExtendSegment(segment.video_path, wav), output, extend_frames=16, include_audio=True)
        assert probe_video(output).frame_count == 46
        assert len(_audio_pcm(output)) == round(30 * 1001 / 30000 * RATE) + added == 48048 + 25626

    def test_paths_with_quotes_and_spaces_survive_the_concat_list(self, tmp_path):
        folder = tmp_path / "it's a folder"
        folder.mkdir()
        source = _make_clip(folder / "my clip.mp4", numbered_frames(33, 128, 64))
        segment = _make_segment(folder, source, numbered_frames(16, 128, 64, offset=100), with_audio=False)
        output = folder / "out put.mp4"
        splice_clips(source, segment, output, extend_frames=16, include_audio=False)
        assert probe_video(output).frame_count == 49

    def test_copy_branch_of_a_source_whose_video_starts_late_keeps_every_frame(self, tmp_path):
        plain = tmp_path / "plain.mp4"
        _make_clip(plain, numbered_frames(33, 128, 64))
        late = tmp_path / "late.mp4"
        ffmpeg("-itsoffset", "0.5", "-i", str(plain), "-c", "copy", str(late))
        source = build_extend_source(late, probe_video(late), None)
        segment = _make_segment(tmp_path, source, numbered_frames(16, 128, 64, offset=100), with_audio=False)
        output = tmp_path / "joined.mp4"
        splice_clips(source, segment, output, extend_frames=16, include_audio=False)
        decoded = _packet_hashes(output, decoded=True)
        assert decoded[:33] == _packet_hashes(source.path, decoded=True)
        assert decoded[33:] == _packet_hashes(segment.video_path, decoded=True)


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


_NOISE = np.random.default_rng(7).uniform(-0.2, 0.2, 40 * RATE).astype(np.float32)  # the sound of the chain tests, at every moment


def _noise_clip(path: Path, frames: int, *, audio_short: int, click_frame: int) -> ExtendSource:
    """Encode an app-like clip (raw frames to libx264 with B-frames, 48 kHz stereo AAC) whose sound is noise and one click.

    The audio ends ``audio_short`` samples before the video does, as a clip made by the app does.
    """
    samples = round(frames / FPS * RATE) - audio_short
    sound = _NOISE[:samples].copy()
    click = round(click_frame / FPS * RATE)
    sound[click : click + 24] = 0.9
    wav = path.with_suffix(".wav")
    write_pcm16_wav(np.stack([sound, sound]), wav)
    raw = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "128x64", "-r", str(FPS), "-i", "-"]
    encode = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "192k"]
    ffmpeg("-i", str(wav), *raw, "-map", "1:v", "-map", "0:a", *encode, str(path), data=numbered_frames(frames, 128, 64).tobytes())
    wav.unlink()
    return build_extend_source(path, probe_video(path), None)


def _window_segment(directory: Path, source: ExtendSource, extend_frames: int, *, lead: int = 12000) -> ExtendSegment:
    """Build a segment whose window sound is the chain's noise at the right moments: ``lead`` samples before the cut, then the new part."""
    video = _make_segment(directory, source, numbered_frames(extend_frames, 128, 64, offset=100), with_audio=False).video_path
    cut = round(source.frame_count / FPS * RATE)
    keep = round(extend_frames / FPS * RATE)
    wav = directory / "segment.wav"
    write_pcm16_wav(np.stack([_NOISE[cut - lead : cut + keep]] * 2), wav)
    return ExtendSegment(video, wav, lead_samples=lead)


def _best_lag(reference: np.ndarray, heard: np.ndarray, span: int = 2400) -> int:
    """Return how many samples ``heard`` lags behind ``reference`` (cross-correlation of their left channels)."""
    count = min(len(reference), len(heard), 60000)
    a, b = reference[:count, 0].astype(float), heard[:count, 0].astype(float)
    size = 1 << (2 * count - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    lags = np.concatenate([np.arange(0, span + 1), np.arange(-span, 0)])
    return int(lags[int(np.argmax(np.concatenate([corr[: span + 1], corr[-span:]])))])


def _longest_silence(pcm: np.ndarray, start: int, stop: int, threshold: int = 2) -> int:
    """Return the longest run of digital silence (both channels within ``threshold``) in ``pcm[start:stop]``."""
    quiet = (np.abs(pcm[start:stop].astype(int)).max(axis=1) <= threshold).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet, [0]])))
    return int((edges[1::2] - edges[::2]).max(initial=0))


@REAL
class TestRealChainedExtensions:
    """Extending an app-like clip (B-frames, AAC) again and again keeps the picture at zero and the sound in step with it."""

    FRAMES = 48
    EXTEND = 16
    CLICK_FRAME = 10

    def _clip(self, path: Path) -> ExtendSource:
        return _noise_clip(path, self.FRAMES, audio_short=1500, click_frame=self.CLICK_FRAME)

    def _chain(self, tmp_path, steps: int):
        """Extend a clip ``steps`` times, each time the previous result; yield each step's source, segment and output."""
        source = self._clip(tmp_path / "step0.mp4")
        for step in range(1, steps + 1):
            directory = tmp_path / f"work{step}"
            directory.mkdir()
            segment = _window_segment(directory, source, self.EXTEND)
            output = tmp_path / f"step{step}.mp4"
            assert can_stream_copy(probe_video(source.path), probe_video(segment.video_path))
            splice_clips(source, segment, output, extend_frames=self.EXTEND, include_audio=True)
            yield step, source, segment, output
            source = build_extend_source(output, probe_video(output), None)

    def test_three_chained_extensions_keep_video_at_zero_audio_in_step_and_no_silent_hole(self, tmp_path):
        first = self._clip(tmp_path / "step0.mp4")
        assert first.video_start == 0
        reference = _audio_pcm(first.path)
        click = round(self.CLICK_FRAME / FPS * RATE)
        for step, source, segment, output in self._chain(tmp_path, 3):
            probe = probe_video(output)
            frames = self.FRAMES + step * self.EXTEND
            assert probe.frame_count == frames
            assert abs(probe.video_start or 0.0) < 0.001
            assert abs(probe.format_start or 0.0) < 0.001
            assert probe.r_frame_rate == "24/1"
            assert probe.avg_frame_rate == "24/1"
            pcm = _audio_pcm(output)
            assert len(pcm) == round(frames / FPS * RATE)
            # The click is where it was in the source, to within a millisecond, after every number of extensions.
            assert abs(int(np.abs(pcm[: 2 * click, 0].astype(int)).argmax()) - click) < RATE // 1000
            assert abs(_best_lag(reference, pcm)) < RATE // 1000
            # The source's own frames are kept exactly, in every step, and so are the segment's.
            decoded = _packet_hashes(output, decoded=True)
            assert decoded[: source.frame_count] == _packet_hashes(source.path, decoded=True)
            assert decoded[source.frame_count :] == _packet_hashes(segment.video_path, decoded=True)
            assert _packet_hashes(output)[1 : source.frame_count] == _packet_hashes(source.path)[1:]
            # No hole of digital silence at the join or anywhere else (under 2 ms).
            join = round(source.frame_count / FPS * RATE)
            assert _longest_silence(pcm, join - 4800, join + 4800) < RATE // 500
            assert _longest_silence(pcm, 0, len(pcm)) < RATE // 500

    def test_the_gap_is_silent_without_a_lead_in(self, tmp_path):
        """A segment with no lead-in cannot fill the gap, so it stays silent: the lead-in is what removes the hole."""
        source = self._clip(tmp_path / "step0.mp4")
        directory = tmp_path / "work"
        directory.mkdir()
        window = _window_segment(directory, source, self.EXTEND)
        wav = directory / "plain.wav"
        write_pcm16_wav(_read_segment_wav(window.audio_path)[:, window.lead_samples :], wav)
        output = tmp_path / "out.mp4"
        splice_clips(source, ExtendSegment(window.video_path, wav), output, extend_frames=self.EXTEND, include_audio=True)
        pcm = _audio_pcm(output)
        assert len(pcm) == round((self.FRAMES + self.EXTEND) / FPS * RATE)
        assert 1000 < _longest_silence(pcm, 94000, 98000) <= 1600

    def test_the_new_audio_ends_with_a_fade_not_a_cut(self, tmp_path):
        source = self._clip(tmp_path / "step0.mp4")
        directory = tmp_path / "work"
        directory.mkdir()
        keep, lead, early = round(self.EXTEND / FPS * RATE), 12000, 1500
        cut_at = round(self.FRAMES / FPS * RATE)
        # A window whose sound ends early, as the model's does, cut as the backend cuts it.
        window = np.stack([_NOISE[cut_at - lead : cut_at + keep - early]] * 2)
        cut = cut_window_audio(window, skip=lead, keep=keep, lead=lead)
        wav = directory / "segment.wav"
        write_pcm16_wav(cut, wav)
        video = _make_segment(directory, source, numbered_frames(self.EXTEND, 128, 64, offset=100), with_audio=False).video_path
        output = tmp_path / "out.mp4"
        splice_clips(source, ExtendSegment(video, wav, lead_samples=lead), output, extend_frames=self.EXTEND, include_audio=True)
        pcm = np.abs(_audio_pcm(output).astype(int))
        end = len(pcm)
        assert end == round((self.FRAMES + self.EXTEND) / FPS * RATE)
        real_end = end - early
        assert pcm[real_end - 3000 : real_end - 2000, 0].max() > 1000  # loud noise before the fade
        assert pcm[real_end - 50 : real_end, 0].max() < 0.2 * pcm[real_end - 3000 : real_end - 2000, 0].max()  # faded out where the sound ended
        assert pcm[real_end + 100 :].max() <= 100  # silent after it (codec ringing only)

    def test_reencode_branch_keeps_the_audio_in_step_and_fills_the_gap(self, tmp_path):
        source = self._clip(tmp_path / "step0.mp4")
        directory = tmp_path / "work"
        directory.mkdir()
        window = _window_segment(directory, source, self.EXTEND)
        tags = dataclasses.replace(source, color_space="bt709", color_primaries="bt709", color_trc="bt709")
        video = _make_segment(directory, tags, numbered_frames(self.EXTEND, 128, 64), with_audio=False).video_path
        assert not can_stream_copy(probe_video(source.path), probe_video(video))
        output = tmp_path / "out.mp4"
        splice_clips(source, ExtendSegment(video, window.audio_path, window.lead_samples), output, extend_frames=self.EXTEND, include_audio=True)
        pcm = _audio_pcm(output)
        assert probe_video(output).frame_count == self.FRAMES + self.EXTEND
        assert len(pcm) == round((self.FRAMES + self.EXTEND) / FPS * RATE)
        assert abs(_best_lag(_audio_pcm(source.path), pcm)) < RATE // 1000
        assert _longest_silence(pcm, 0, len(pcm)) < RATE // 500


def _read_segment_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as reader:
        data = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16).reshape(-1, reader.getnchannels())
    return data.T.astype(np.float32) / 32767


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


@REAL
class TestRealShortWindowChain:
    """The model's audio is about 1520 samples short of its video, so each extension ends in a silent tail that the next one must fill."""

    FRAMES = 48
    EXTEND = 16
    EARLY = 1520
    LEAD = 12000

    def _short_window(self, directory: Path, source: ExtendSource) -> ExtendSegment:
        video = _make_segment(directory, source, numbered_frames(self.EXTEND, 128, 64, offset=100), with_audio=False).video_path
        cut = round(source.frame_count / FPS * RATE)
        keep = round(self.EXTEND / FPS * RATE)
        window = np.stack([_NOISE[cut - self.LEAD : cut + keep - self.EARLY]] * 2)  # the model's sound ends before its video does
        wav = directory / "segment.wav"
        write_pcm16_wav(cut_window_audio(window, skip=self.LEAD, keep=keep, lead=self.LEAD), wav)
        return ExtendSegment(video, wav, lead_samples=self.LEAD)

    def test_no_silent_hole_at_either_join_of_two_extensions(self, tmp_path):
        source = _noise_clip(tmp_path / "step0.mp4", self.FRAMES, audio_short=1500, click_frame=10)
        joins = []
        for step in (1, 2):
            directory = tmp_path / f"work{step}"
            directory.mkdir()
            joins.append(round(source.frame_count / FPS * RATE))
            output = tmp_path / f"step{step}.mp4"
            splice_clips(source, self._short_window(directory, source), output, extend_frames=self.EXTEND, include_audio=True)
            pcm = _audio_pcm(output)
            assert len(pcm) == round((self.FRAMES + step * self.EXTEND) / FPS * RATE)
            for join in joins:
                assert _longest_silence(pcm, join - 4800, join + 4800) < RATE // 500  # under 2 ms
            source = build_extend_source(output, probe_video(output), None)
        # The second extension's own end is silent again (the model's sound is short), as the third would find it.
        assert _longest_silence(pcm, len(pcm) - 4800, len(pcm)) > 500
        assert abs(_best_lag(_NOISE[None, : len(pcm)].T.repeat(2, axis=1) * 32767, pcm)) < RATE // 1000
