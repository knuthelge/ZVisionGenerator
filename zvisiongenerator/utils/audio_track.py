"""Read a user's audio track and build the window sound of an extension.

Everything here is a pure function of its inputs apart from the ffmpeg and ffprobe calls that read an audio file. Every
backend and workflow stage reads a track through ``read_track`` instead of keeping its own reader.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zvisiongenerator.core.video_types import AudioTrack
from zvisiongenerator.utils.ffmpeg import tool_error
from zvisiongenerator.utils.video_extend import AUDIO_LEAD_SECONDS, AUDIO_SAMPLE_RATE, cut_window_audio
from zvisiongenerator.utils.video_probe import audio_decodes, ffprobe_json

if TYPE_CHECKING:
    import numpy as np

    from zvisiongenerator.core.video_types import ExtendSource

__all__ = [
    "TRACK_UNREADABLE",
    "continue_track",
    "extension_audio",
    "probe_track_seconds",
    "read_track",
    "require_readable_track",
    "window_waveform",
]

# The message of the ``ValueError`` an unreadable track raises; callers show it as it is.
TRACK_UNREADABLE = "Can't read the audio track"


def read_track(track: AudioTrack, rate: int, seconds: float) -> np.ndarray:
    """Read *seconds* of *track*, from its start, as stereo float32 audio of shape ``(2, samples)``.

    A mono file is doubled; a stretch that runs past the end of the file is padded with silence (all of it when *track*
    starts at or past the end), so the result always holds ``round(seconds * rate)`` samples.

    Args:
        track: The audio (or video) file, whose first audio stream is read, and the seconds into it where the stretch begins
            (negative counts as 0).
        rate: The sample rate to resample to, in Hz.
        seconds: Length of the stretch in seconds.

    Raises:
        ValueError: With ``TRACK_UNREADABLE`` if ffmpeg cannot decode the file; the cause holds ffmpeg's message.
    """
    import numpy as np

    samples = max(0, round(seconds * rate))
    if samples == 0:
        return np.zeros((2, 0), dtype=np.float32)
    # ffmpeg's own mono to stereo upmix lowers each channel by 3 dB; a mono track is copied to both channels as it is.
    channels = ["-af", "pan=stereo|c0=c0|c1=c0"] if _is_mono(track.path) else ["-ac", "2"]
    command = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, track.start):.6f}", "-t", f"{seconds:.6f}", "-i", str(track.path), "-map", "0:a:0", "-vn", *channels, "-ar", str(rate), "-f", "f32le", "-"]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, check=False)
    if result.returncode != 0:
        raise ValueError(TRACK_UNREADABLE) from tool_error("ffmpeg could not read the audio track", result.stderr)
    data = result.stdout
    audio = np.frombuffer(data[: len(data) - len(data) % 8], dtype=np.float32).reshape(-1, 2).T[:, :samples]
    return np.pad(audio, ((0, 0), (0, samples - audio.shape[1])))


def _is_mono(path: str) -> bool:
    """Return whether the first audio stream of *path* has one channel; a file ffprobe cannot read is not mono (ffmpeg then reports it)."""
    try:
        data = ffprobe_json(path, count_packets=False)
    except RuntimeError:
        return False
    stream = next((s for s in data.get("streams") or [] if isinstance(s, dict) and s.get("codec_type") == "audio"), None)
    return stream is not None and stream.get("channels") == 1


def probe_track_seconds(path: str) -> float:
    """Return the length of an audio file's first audio stream in seconds, checking that it decodes.

    Raises:
        ValueError: With ``TRACK_UNREADABLE`` if the file is missing, has no audio stream, has no length or does not decode.
    """
    try:
        data = ffprobe_json(path, count_packets=False)
    except RuntimeError as exc:
        raise ValueError(TRACK_UNREADABLE) from exc
    stream = next((s for s in data.get("streams") or [] if isinstance(s, dict) and s.get("codec_type") == "audio"), None)
    if stream is None:
        raise ValueError(TRACK_UNREADABLE)
    seconds = _positive_seconds(stream.get("duration")) or _positive_seconds((data.get("format") or {}).get("duration"))
    if seconds is None or not audio_decodes(path, start=0.0, duration=min(seconds, 1.0)):
        raise ValueError(TRACK_UNREADABLE)
    return seconds


def require_readable_track(track: AudioTrack) -> None:
    """Check that *track* can be read, so a request fails at submit, before any model loads.

    Raises:
        ValueError: With ``TRACK_UNREADABLE`` if the file is missing, has no audio stream, has no length or does not decode.
    """
    probe_track_seconds(track.path)


def continue_track(track: AudioTrack, source: ExtendSource) -> AudioTrack:
    """Return *track* starting where *source*'s sound ended, when *source* followed the same file; else *track* unchanged.

    The files are compared by their resolved paths. A source that records no track, a different file or no end gives *track* back.
    """
    if source.audio_track_path is None or source.audio_track_end is None:
        return track
    if Path(source.audio_track_path).expanduser().resolve() != Path(track.path).expanduser().resolve():
        return track
    return AudioTrack(track.path, source.audio_track_end)


def window_waveform(source: ExtendSource, track: AudioTrack, *, carry_frames: int, extend_frames: int, rate: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """Return the sound of an extension's window: the source's sound under its last *carry_frames*, then the track from its start.

    The source part is silence for a silent source, and for any stretch past where the source's sound ends. The track
    part is as long as the new frames. Backends encode this waveform to hold the window's sound, and the Extend sound
    stage writes the extension's WAV from it, so the held and the written sound are the same samples.

    Args:
        source: The clip being extended.
        track: The track the extension follows.
        carry_frames: The frames of the source's end the window carries.
        extend_frames: The new frames.
        rate: The sample rate of the result, in Hz.

    Returns:
        Float32 audio of shape ``(2, round(carry_frames * rate / fps) + round(extend_frames * rate / fps))``.

    Raises:
        ValueError: With ``TRACK_UNREADABLE`` if the track or the source's sound cannot be decoded.
    """
    import numpy as np

    carry_seconds = carry_frames / source.fps
    if source.has_audio:
        # An input seek is relative to the start of the file, and the video's first frame may not be at its start.
        begin = source.video_start + max(0, source.frame_count - carry_frames) / source.fps
        carried = read_track(AudioTrack(str(source.path), begin), rate, carry_seconds)
    else:
        carried = np.zeros((2, round(carry_seconds * rate)), dtype=np.float32)
    return np.concatenate([carried, read_track(track, rate, extend_frames / source.fps)], axis=1)


def extension_audio(source: ExtendSource, track: AudioTrack, *, carry_frames: int, extend_frames: int, rate: int = AUDIO_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Return the audio of an extension's segment WAV and its lead-in, cut from the window waveform.

    The WAV starts *lead* samples before the new part (the end of the source's own sound, which the join blends over a
    source track that ends short of its video) and holds the new part after it.

    Args:
        source: The clip being extended.
        track: The track the extension follows.
        carry_frames: The frames of the source's end the window carries.
        extend_frames: The new frames.
        rate: The sample rate of the result, in Hz.

    Returns:
        ``(audio, lead)``: stereo float32 audio of ``lead + round(extend_frames * rate / fps)`` samples, and the lead-in's samples.

    Raises:
        ValueError: With ``TRACK_UNREADABLE`` if the track or the source's sound cannot be decoded.
    """
    window = window_waveform(source, track, carry_frames=carry_frames, extend_frames=extend_frames, rate=rate)
    skip = round(carry_frames * rate / source.fps)
    keep = round(extend_frames * rate / source.fps)
    lead = min(skip, round(AUDIO_LEAD_SECONDS * rate))
    return cut_window_audio(window, skip, keep, lead), lead


def _positive_seconds(value: Any) -> float | None:
    """Return *value* as seconds when it is a positive finite number, else ``None``."""
    try:
        seconds = float(value)
    except TypeError, ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None
