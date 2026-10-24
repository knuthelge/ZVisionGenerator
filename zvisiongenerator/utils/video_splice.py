"""Join a generated segment onto the end of a source clip."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import wave
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from zvisiongenerator.utils.ffmpeg import tool_error
from zvisiongenerator.utils.video_extend import AUDIO_FADE_SAMPLES, AUDIO_LEAD_SECONDS, AUDIO_SAMPLE_RATE
from zvisiongenerator.utils.video_probe import VideoProbe, parse_fraction, probe_video

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np

    from zvisiongenerator.core.video_types import ExtendSegment, ExtendSource

# Prefix of the temporary directory a join works in, beside its output.
SPLICE_DIR_PREFIX = ".splice-"
# Quality of the original part when the join re-encodes it: crf 12 keeps it within about 1 ΔE of the source, where
# 18, the generated segment's quality, moves it by about 1.4.
REENCODE_CRF = 12
SEGMENT_CRF = 18
AUDIO_BITRATE = "192k"
# An end of the source's audio that is digital silence (within 2 LSB of 16 bits) for longer than 2 ms and no longer
# than the window's lead-in is taken as missing sound, not as music or speech: the model's audio comes out short of its
# video, so a clip made by an extension ends in zeros.
SILENT_END_LEVEL = 2 / 32767
SILENT_END_MIN_SAMPLES = 96
SILENT_END_MAX_SAMPLES = round(AUDIO_LEAD_SECONDS * AUDIO_SAMPLE_RATE)

_FFMPEG_QUIET = ["ffmpeg", "-y", "-v", "error"]
_AUDIO_ENCODER = ["-c:a", "aac", "-b:a", AUDIO_BITRATE]
# The matrix that converts RGB to YUV for a colour space tag.
_MATRICES = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601"}
# Fields of two probes that must agree for their streams to be joined without re-encoding.
_COPY_FIELDS = ("codec", "profile", "pix_fmt", "coded_width", "coded_height", "width", "height", "time_base", "r_frame_rate", "color_space", "color_primaries", "color_trc")


def color_filter(space: str | None, primaries: str | None, trc: str | None) -> str:
    """Return the video filter that converts with the right matrix and stamps the colour tags; ``null`` when untagged.

    The tags are set with ``setparams`` inside the filter graph: the ``-colorspace`` family of output options does
    not reach the x264 stream header once a ``scale`` filter has run.
    """
    parts = []
    if matrix := _MATRICES.get(space or ""):
        parts.append(f"scale=out_color_matrix={matrix}")
    tags = [f"{name}={value}" for name, value in (("colorspace", space), ("color_primaries", primaries), ("color_trc", trc)) if value]
    if tags:
        parts.append("setparams=" + ":".join(tags))
    return ",".join(parts) or "null"


def can_stream_copy(source: VideoProbe, segment: VideoProbe) -> bool:
    """Return whether the segment can be appended to the source's video without re-encoding it.

    The streams must agree on codec, profile, pixel format, size, time base, frame rate and colour tags and have
    square pixels. A rotated clip is always re-encoded, since a stream copy cannot turn it upright.
    """
    if source.rotation != 0 or segment.rotation != 0:
        return False
    if not (source.square_pixels and segment.square_pixels):
        return False
    return all(getattr(source, field) == getattr(segment, field) for field in _COPY_FIELDS)


def build_segment_command(width: int, height: int, fps_text: str, output: str | Path, *, color_space: str | None = None, color_primaries: str | None = None, color_trc: str | None = None) -> list[str]:
    """Build the ffmpeg command that encodes raw RGB frames from stdin into the generated segment's video.

    The segment is encoded with the source's size, frame rate and colour tags so that it can be stream-copied onto it.

    Args:
        width: Displayed width of the source.
        height: Displayed height of the source.
        fps_text: The source's frame rate as ffprobe writes it (``24/1``).
        output: The segment file.
        color_space: The source's colour space tag.
        color_primaries: The source's colour primaries tag.
        color_trc: The source's transfer characteristics tag.
    """
    colors = color_filter(color_space, color_primaries, color_trc)
    return [
        *_FFMPEG_QUIET,
        *("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", fps_text, "-i", "-"),
        *("-vf", colors),
        *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(SEGMENT_CRF), "-an"),
        str(output),
    ]


def write_segment_frames(frames: Iterable[np.ndarray], command: list[str], *, description: str = "the extension segment") -> None:
    """Pipe ``height x width x 3`` uint8 RGB frames into a segment command from ``build_segment_command``.

    Args:
        frames: The frames to encode.
        command: The command from ``build_segment_command``.
        description: What is being written, for the error text ("ffmpeg could not encode <description>").

    Raises:
        RuntimeError: If ffmpeg exits with an error.
    """
    import numpy as np

    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors) as process:
            assert process.stdin is not None
            # A BrokenPipeError means ffmpeg stopped early; its exit code and message say why. Closing stdin can raise it
            # again while flushing what is buffered, so the close is guarded too.
            with contextlib.suppress(BrokenPipeError):
                for frame in frames:
                    process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()
        if process.returncode != 0:
            errors.seek(0)
            raise tool_error(f"ffmpeg could not encode {description}", errors.read())


def source_audio_chain(*, samples: int, offset: float) -> str:
    """Return the filter chain that decodes the source's audio onto the video's time axis, cut to at most ``samples``.

    The audio is moved to the first video frame (``offset`` seconds are cut or, when the audio starts late, padded)
    and is not padded at its end: a track shorter than its video stays short.
    """
    layout = f"aformat=sample_fmts=fltp:sample_rates={AUDIO_SAMPLE_RATE}:channel_layouts=stereo"
    return f"aresample={AUDIO_SAMPLE_RATE}:async=1:first_pts=0,{layout},atrim=start={offset:.6f},asetpts=PTS-STARTPTS,atrim=end_sample={samples}"


def build_audio_graph(*, source_input: int, wav_input: int, head_samples: int, offset: float) -> str:
    """Build the filter graph that makes the joined audio: the first ``head_samples`` of the source's, then the join WAV.

    Args:
        source_input: Index of the ffmpeg input holding the source's audio.
        wav_input: Index of the ffmpeg input holding the rest of the audio (the join WAV from ``build_join_audio``).
        head_samples: Samples taken from the source's audio, which is moved to the first video frame first.
        offset: Seconds from the start of the file to the first video frame.

    Returns:
        A graph whose output is labelled ``[a]``.
    """
    layout = f"aformat=sample_fmts=fltp:sample_rates={AUDIO_SAMPLE_RATE}:channel_layouts=stereo"
    head = f"[{source_input}:a:0]{source_audio_chain(samples=head_samples, offset=offset)}[a0]"
    return f"{head};[{wav_input}:a:0]aresample={AUDIO_SAMPLE_RATE},{layout}[a1];[a0][a1]concat=n=2:v=0:a=1[a]"


def build_join_audio(tail: np.ndarray, source_length: int, window: np.ndarray, *, lead: int, samples: int, fade: int = AUDIO_FADE_SAMPLES) -> tuple[int, np.ndarray]:
    """Return how much of the source's audio to keep and the audio that follows it, filling a track that ends short.

    The source's audio normally ends a few tens of milliseconds before its video does. That gap is filled with the
    window's own reconstruction of the same moment (its carried frames' sound, ``lead`` samples before the new part),
    blended in over ``fade`` samples ending where the source's audio does. Where the window holds nothing for the
    gap (a gap longer than ``lead``) it stays silent. A track that reaches the end of its video is left as it is.

    Args:
        tail: The last samples of the source's audio, ``(channels, n)`` with ``n`` at most ``source_length``.
        source_length: Samples of the source's audio on the video's time axis, at most ``samples``.
        window: ``(channels, lead + new samples)``: the window's sound from ``lead`` before the cut.
        lead: Samples of the window that precede the new part.
        samples: Length of the source's part in samples (its video length).
        fade: Samples of the blend at the end of the source's audio.

    Returns:
        ``(head_samples, join)``: the samples of the source's audio to keep, then the audio to append, so that
        ``head_samples + join.shape[1] == samples + new samples``.
    """
    import numpy as np

    if source_length >= samples:
        return samples, window[:, lead:]
    window_start = samples - lead  # the source sample that the window's first sample lines up with
    overlap = min(fade, tail.shape[1], max(0, source_length - window_start))
    first = max(0, source_length - window_start)  # the window sample that lines up with the source's first missing sample
    silence = np.zeros((window.shape[0], max(0, window_start - source_length)), dtype=np.float32)
    ramp = ((np.arange(overlap) + 0.5) / max(overlap, 1)).astype(np.float32)
    blend = tail[:, tail.shape[1] - overlap :] * (1.0 - ramp) + window[:, first - overlap : first] * ramp
    return source_length - overlap, np.concatenate([blend.astype(np.float32), silence, window[:, first:]], axis=1)


def build_video_only_command(source: str | Path, output: str | Path) -> list[str]:
    """Build the ffmpeg command that stream-copies a clip's first video stream alone into its own file.

    The concat demuxer shifts a file's video by the start offset of the file's other streams. An AAC track begins
    one frame (1024 samples) before zero, which moves the copied video later by that much: 21 ms at 48 kHz, which
    adds up over chained extensions and leaves the audio behind the picture. A file that holds only the video has
    no such offset, and its packets are copied unchanged. The copy also starts at zero, where the audio graph
    puts the first audio sample, even when the source's video starts late in its file.
    """
    return [*_FFMPEG_QUIET, "-i", str(source), "-map", "0:v:0", "-c", "copy", "-an", "-map_metadata", "-1", str(output)]


def build_concat_copy_command(list_file: str | Path, source: str | Path, wav: str | Path | None, output: str | Path, *, audio_graph: str | None) -> list[str]:
    """Build the ffmpeg command that stream-copies the source's and the segment's video (listed in a concat file).

    Args:
        list_file: Concat-demuxer list holding the source's video-only copy and the segment.
        source: The source file, read again for its audio.
        wav: The new audio, or ``None`` for a silent result.
        output: The joined file.
        audio_graph: The graph from ``build_audio_graph`` (input 1 is the source and input 2 the WAV), or ``None`` for no audio.
    """
    inputs = ["-f", "concat", "-safe", "0", "-i", str(list_file)]
    if audio_graph is not None and wav is not None:
        return [
            *_FFMPEG_QUIET,
            *inputs,
            "-i",
            str(source),
            "-i",
            str(wav),
            "-filter_complex",
            audio_graph,
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            *_AUDIO_ENCODER,
            "-map_metadata",
            "-1",
            str(output),
        ]
    return [*_FFMPEG_QUIET, *inputs, "-map", "0:v:0", "-c", "copy", "-an", "-map_metadata", "-1", str(output)]


def build_reencode_command(
    source: str | Path,
    segment: str | Path,
    wav: str | Path | None,
    fps_text: str,
    output: str | Path,
    *,
    color_space: str | None = None,
    color_primaries: str | None = None,
    color_trc: str | None = None,
    audio_graph: str | None = None,
) -> list[str]:
    """Build the ffmpeg command that decodes both clips and encodes their concatenation in one pass.

    ffmpeg's auto-rotation stays on, so a rotated source comes out upright with no display matrix. The colour
    filter runs inside the graph after the concat.

    Args:
        source: The source file (input 0).
        segment: The generated segment's video (input 1).
        wav: The new audio (input 2), or ``None`` for a silent result.
        fps_text: The source's frame rate as ffprobe writes it.
        output: The joined file.
        color_space: The source's colour space tag, copied as probed.
        color_primaries: The source's colour primaries tag.
        color_trc: The source's transfer characteristics tag.
        audio_graph: The graph from ``build_audio_graph`` (input 0 is the source and input 2 the WAV), or ``None`` for no audio.
    """
    colors = color_filter(color_space, color_primaries, color_trc)
    # Each clip's timestamps start at zero, so a source whose first frame is late in its file is not padded with repeated frames.
    graph = f"[0:v:0]setpts=PTS-STARTPTS[v0];[1:v:0]setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[c];[c]{colors}[v]"
    video = ["-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(REENCODE_CRF), "-r", fps_text]
    inputs = ["-i", str(source), "-i", str(segment)]
    if audio_graph is not None and wav is not None:
        return [*_FFMPEG_QUIET, *inputs, "-i", str(wav), "-filter_complex", f"{graph};{audio_graph}", *video, "-map", "[a]", *_AUDIO_ENCODER, "-map_metadata", "-1", str(output)]
    return [*_FFMPEG_QUIET, *inputs, "-filter_complex", graph, *video, "-an", "-map_metadata", "-1", str(output)]


def build_mux_command(video: str | Path, wav: str | Path, output: str | Path) -> list[str]:
    """Build the ffmpeg command that stream-copies a video-only file and encodes a WAV as its AAC audio track.

    The video is copied unchanged, so a clip written once by ``build_segment_command`` is never encoded twice. The result
    carries no metadata from either input.

    Args:
        video: The video-only file.
        wav: The audio, at any sample rate.
        output: The muxed file.
    """
    return [*_FFMPEG_QUIET, "-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *_AUDIO_ENCODER, "-map_metadata", "-1", str(output)]


def build_audio_prep_command(source: str | Path, output: str | Path, *, copy: bool) -> list[str]:
    """Build the ffmpeg command that writes a clip's first audio stream, and nothing else, to an audio file.

    Args:
        source: The video file.
        output: The audio file (``.m4a``).
        copy: Whether to copy the stream unchanged (it is AAC already); otherwise it is encoded as AAC.
    """
    codec = ["-c:a", "copy"] if copy else _AUDIO_ENCODER
    return [*_FFMPEG_QUIET, "-i", str(source), "-map", "0:a:0", "-vn", *codec, "-map_metadata", "-1", str(output)]


def build_final_mux_command(video: str | Path, audio: str | Path | None, output: str | Path, *, video_offset: float, metadata: Sequence[str] = ()) -> list[str]:
    """Build the ffmpeg command that stream-copies a video-only file, and an AAC audio file when there is one, into one MP4.

    ``-copyts`` keeps the audio file's own start (a late-starting track is delayed by an edit list, which ffmpeg would
    otherwise shift back to zero); the video is moved to *video_offset* seconds, where the source's first frame was, so the
    two stay in step. Neither stream is cut to the other's length. Without audio the video is copied as it is. The result
    carries no metadata but *metadata*.

    Args:
        video: The video-only file, which starts at zero.
        audio: The audio file from ``build_audio_prep_command``, or ``None`` for a silent result.
        output: The muxed file.
        video_offset: Seconds from the start of the source file to its first video frame; unused without audio.
        metadata: Output options that add metadata, such as those from ``mp4_config_args``.
    """
    if audio is None:
        return [*_FFMPEG_QUIET, "-i", str(video), "-map", "0:v:0", "-c", "copy", "-map_metadata", "-1", *metadata, str(output)]
    return [
        *_FFMPEG_QUIET,
        *("-itsoffset", f"{video_offset:.6f}", "-i", str(video)),
        *("-i", str(audio)),
        *("-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-copyts", "-map_metadata", "-1"),
        *metadata,
        str(output),
    ]


def mux_audio(video: str | Path, wav: str | Path, output: str | Path) -> None:
    """Write *output* as *video* with *wav* as its audio track (see ``build_mux_command``).

    Raises:
        RuntimeError: If ffmpeg exits with an error.
    """
    run_ffmpeg(build_mux_command(video, wav, output), "ffmpeg could not add the audio to the video")


def write_pcm16_wav(samples: np.ndarray, path: str | Path, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
    """Write audio of shape ``(channels, samples)`` with values in -1..1 as a 16-bit PCM WAV file, by default at ``AUDIO_SAMPLE_RATE``."""
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(pcm.shape[0])
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(np.ascontiguousarray(pcm.T).tobytes())


def splice_clips(source: ExtendSource, segment: ExtendSegment, output: str | Path, *, extend_frames: int, include_audio: bool) -> Path:
    """Append a generated segment to a source clip and write the result atomically.

    The source's video is stream-copied when the segment matches it, otherwise both are decoded and encoded
    again in one pass (crf 12). A rotated source always takes the second path and comes out upright. With
    audio, the source's audio is fitted to its video length (a track that ends short is filled from the segment's
    lead-in sound) and the new audio follows it, in one AAC encode.

    Args:
        source: The clip to continue.
        segment: The generated segment: its video, and its WAV when it has audio.
        output: The joined file; written to a temporary file beside it first.
        extend_frames: Frames in the segment, which are added to the source's.
        include_audio: Whether the result has audio. It needs the segment's WAV and a source with audio: a silent source stays silent.

    Returns:
        The output path.

    Raises:
        RuntimeError: If ffmpeg fails, or the result does not have the expected frames, size, rotation and audio.
    """
    output = Path(output)
    wav = segment.audio_path if include_audio and source.has_audio else None
    work = Path(tempfile.mkdtemp(dir=output.parent, prefix=SPLICE_DIR_PREFIX))
    try:
        joined = work / f"joined{output.suffix or '.mp4'}"
        head_samples = 0
        if wav is not None:
            head_samples, wav = _prepare_join_audio(source, segment, work)
        if can_stream_copy(source.probe or probe_video(source.path), probe_video(segment.video_path)):
            video_only = work / "source-video.mp4"
            run_ffmpeg(build_video_only_command(source.path, video_only))
            list_file = work / "clips.txt"
            list_file.write_text(_concat_list(source, video_only, segment.video_path), encoding="utf-8")
            graph = _audio_graph(source, wav, head_samples, source_input=1, wav_input=2)
            command = build_concat_copy_command(list_file, source.path, wav, joined, audio_graph=graph)
        else:
            graph = _audio_graph(source, wav, head_samples, source_input=0, wav_input=2)
            command = build_reencode_command(
                source.path,
                segment.video_path,
                wav,
                source.fps_text,
                joined,
                color_space=source.color_space,
                color_primaries=source.color_primaries,
                color_trc=source.color_trc,
                audio_graph=graph,
            )
        run_ffmpeg(command)
        _check_joined(probe_video(joined), source, extend_frames=extend_frames, audio=wav is not None)
        os.replace(joined, output)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return output


def _source_samples(source: ExtendSource) -> int:
    """Return the length of the source's video in samples at the joined audio's rate."""
    rate = parse_fraction(source.fps_text) or Fraction(source.fps).limit_denominator(1_000_000)
    return round(Fraction(source.frame_count * AUDIO_SAMPLE_RATE) / rate)


def _audio_graph(source: ExtendSource, wav: Path | None, head_samples: int, *, source_input: int, wav_input: int) -> str | None:
    """Return the audio graph for a join, or ``None`` when the result has no audio."""
    if wav is None:
        return None
    return build_audio_graph(source_input=source_input, wav_input=wav_input, head_samples=head_samples, offset=source.video_start)


def _prepare_join_audio(source: ExtendSource, segment: ExtendSegment, work: Path) -> tuple[int, Path]:
    """Write the audio that follows the source's own and return ``(samples of the source's audio to keep, that WAV)``.

    The source's audio is decoded once to find where it ends and to read its last samples; the join (see
    ``build_join_audio``) is then written to the working directory.
    """
    samples = _source_samples(source)
    length, tail = _decode_source_tail(source, samples, AUDIO_FADE_SAMPLES)
    head_samples, join = build_join_audio(tail, length, _read_wav(segment.audio_path), lead=segment.lead_samples, samples=samples)
    join_wav = work / "join.wav"
    write_pcm16_wav(join, join_wav)
    return head_samples, join_wav


def silent_end_length(audio: np.ndarray) -> int:
    """Return the length of the digital silence that ends ``(channels, samples)`` audio, or 0 when that is not missing sound.

    A run of near-zero samples (see ``SILENT_END_LEVEL``) counts when it is longer than ``SILENT_END_MIN_SAMPLES``
    and no longer than ``SILENT_END_MAX_SAMPLES``: a longer silence is the clip's own.
    """
    import numpy as np

    quiet = np.abs(audio).max(axis=0) <= SILENT_END_LEVEL if audio.shape[1] else np.zeros(0, dtype=bool)
    loud = np.flatnonzero(~quiet)
    run = quiet.size if loud.size == 0 else quiet.size - 1 - int(loud[-1])
    return run if SILENT_END_MIN_SAMPLES < run <= SILENT_END_MAX_SAMPLES else 0


def _decode_source_tail(source: ExtendSource, samples: int, tail_samples: int) -> tuple[int, np.ndarray]:
    """Decode the source's audio as the join graph does; return its length in samples and its last ``tail_samples``.

    The length stops where the sound does: a silent end that ``silent_end_length`` finds is not counted, so the
    join fills it like any other gap. The audio streams through a pipe, so a long clip is not held in memory.

    Raises:
        RuntimeError: If ffmpeg cannot decode the audio.
    """
    import numpy as np

    command = [*_FFMPEG_QUIET, "-i", str(source.path), "-map", "0:a:0", "-af", source_audio_chain(samples=samples, offset=source.video_start), "-f", "f32le", "-ac", "2", "-"]
    frame = 2 * 4  # bytes of one stereo float32 sample
    total, tail = 0, b""
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as process:
            assert process.stdout is not None
            while chunk := process.stdout.read(1 << 20):
                total += len(chunk)
                tail = (tail + chunk)[-(tail_samples + SILENT_END_MAX_SAMPLES) * frame :]
        if process.returncode != 0:
            errors.seek(0)
            raise tool_error("ffmpeg could not read the source's audio", errors.read())
    audio = np.frombuffer(tail[len(tail) % frame :], dtype=np.float32).reshape(-1, 2).T
    silent = silent_end_length(audio)
    return total // frame - silent, audio[:, : audio.shape[1] - silent][:, -tail_samples:]


def _read_wav(path: Path | None) -> np.ndarray:
    """Read a 16-bit PCM WAV as ``(2, samples)`` float32 in -1..1 (a mono file is doubled)."""
    import numpy as np

    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16).reshape(-1, channels).T
    audio = pcm.astype(np.float32) / 32767
    return audio if channels == 2 else np.repeat(audio[:1], 2, axis=0)


def _concat_list(source: ExtendSource, video_only: Path, segment: Path) -> str:
    """Return the concat-demuxer list that puts the segment right after the source's video.

    The source's duration is set to its video length: left to the demuxer it would be the file's, which a longer
    audio track stretches, leaving a gap before the segment.
    """

    def quoted(path: Path) -> str:
        return str(path.resolve()).replace("'", "'\\''")

    return f"file '{quoted(video_only)}'\nduration {source.frame_count / source.fps:.6f}\nfile '{quoted(segment)}'\n"


def run_ffmpeg(command: list[str], message: str = "ffmpeg could not join the clips") -> None:
    """Run an ffmpeg command, hiding its output.

    Raises:
        RuntimeError: If ffmpeg exits with an error; *message* leads the error text.
    """
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise tool_error(message, result.stderr)


def _check_joined(probe: VideoProbe, source: ExtendSource, *, extend_frames: int, audio: bool) -> None:
    """Check a joined file against the source and the frames added.

    Raises:
        RuntimeError: If the frame count, displayed size, rotation or audio is not as expected.
    """
    expected = source.frame_count + extend_frames
    if probe.frame_count != expected:
        raise RuntimeError(f"The joined video has {probe.frame_count} frames, expected {expected}.")
    if (probe.width, probe.height) != (source.width, source.height) or probe.rotation != 0:
        raise RuntimeError(f"The joined video is {probe.width}x{probe.height} (rotation {probe.rotation}), expected {source.width}x{source.height} upright.")
    if probe.has_audio != audio:
        raise RuntimeError(f"The joined video {'has no' if audio else 'has'} audio, expected {'audio' if audio else 'none'}.")
