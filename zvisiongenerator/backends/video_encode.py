"""Write generated frames, with an optional audio track, to an MP4 file through PyAV.

The settings match the macOS LTX pipeline's ffmpeg command: H.264 at CRF 18 in yuv420p, and AAC audio at the
encoder's default rate. diffusers' own ``encode_video`` leaves x264 at its default CRF 23, which smears fine detail.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = ["VIDEO_CRF", "encode_video"]

VIDEO_CRF = 18


def encode_video(frames: Sequence[Any], fps: int, output_path: str, audio: torch.Tensor | None = None, audio_sample_rate: int | None = None) -> None:
    """Encode *frames* (RGB PIL images or ``HxWx3`` uint8 arrays) at *fps* to *output_path*, muxing *audio* when given.

    Args:
        frames: The video frames, all the same size.
        fps: Frames per second.
        output_path: The MP4 file to write.
        audio: A float waveform in ``[-1, 1]``, ``(channels, samples)``, mono or stereo.
        audio_sample_rate: The waveform's sample rate; required with *audio*.

    Raises:
        ValueError: If there are no frames, or audio comes without a sample rate.
    """
    import av
    import numpy as np

    if not frames:
        raise ValueError("No frames to encode.")
    if audio is not None and not audio_sample_rate:
        raise ValueError("audio_sample_rate is required when audio is provided.")
    first = np.asarray(frames[0])
    with av.open(output_path, mode="w") as container:
        stream = container.add_stream("libx264", rate=int(fps), options={"crf": str(VIDEO_CRF)})
        stream.width = first.shape[1]
        stream.height = first.shape[0]
        stream.pix_fmt = "yuv420p"
        audio_stream = _audio_stream(container, audio_sample_rate) if audio is not None else None
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(np.asarray(frame, dtype=np.uint8), format="rgb24")))
        container.mux(stream.encode())
        if audio_stream is not None:
            _write_audio(container, audio_stream, audio, audio_sample_rate)


def _audio_stream(container: Any, sample_rate: int) -> Any:
    """Add a stereo AAC stream at *sample_rate* to *container*."""
    stream = container.add_stream("aac", rate=sample_rate)
    stream.codec_context.layout = "stereo"
    stream.codec_context.time_base = Fraction(1, sample_rate)
    return stream


def _write_audio(container: Any, stream: Any, waveform: torch.Tensor, sample_rate: int) -> None:
    """Encode the float *waveform* into *stream* as stereo, resampled to the encoder's sample format."""
    import av

    samples = _stereo_int16(waveform)
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="stereo")
    frame.sample_rate = sample_rate
    resampler = av.audio.resampler.AudioResampler(format=stream.codec_context.format or "fltp", layout="stereo", rate=sample_rate)
    next_pts = 0
    for resampled in [*resampler.resample(frame), *resampler.resample(None)]:  # None flushes what it buffered
        if resampled.pts is None:
            resampled.pts = next_pts
        next_pts += resampled.samples
        container.mux(stream.encode(resampled))
    container.mux(stream.encode())


def _stereo_int16(waveform: torch.Tensor) -> Any:
    """Return the ``(channels, samples)`` *waveform* as an interleaved ``(samples, 2)`` int16 array; mono is duplicated.

    Raises:
        ValueError: If the waveform is not mono or stereo, channels first.
    """
    import numpy as np

    samples = waveform.detach().float().cpu().numpy()
    samples = samples[None, :] if samples.ndim == 1 else samples
    if samples.ndim != 2 or samples.shape[0] > 2:
        raise ValueError(f"Expected mono or stereo audio as (channels, samples); got shape {tuple(waveform.shape)}.")
    samples = samples.T  # (samples, channels)
    if samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).copy(order="C")
