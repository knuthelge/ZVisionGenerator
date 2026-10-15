"""Tests for writing generated frames and audio to MP4 with PyAV."""

from __future__ import annotations

import pytest

av = pytest.importorskip("av")
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from PIL import Image  # noqa: E402

from zvisiongenerator.backends.video_encode import VIDEO_CRF, encode_video  # noqa: E402


def _frames(count=5, size=(64, 48)):
    return [Image.new("RGB", size, (40 * index % 255, 120, 200)) for index in range(count)]


def _x264_settings(path):
    data = path.read_bytes()
    start = data.find(b"x264 - core")
    return data[start : data.find(b"\x00", start)].decode(errors="replace")


def test_video_is_h264_at_the_macos_quality(tmp_path):
    path = tmp_path / "clip.mp4"

    encode_video(_frames(), 24, str(path))

    with av.open(str(path)) as container:
        (stream,) = container.streams.video
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == (64, 48)
        assert stream.average_rate == 24
        assert sum(1 for _ in container.decode(stream)) == 5
    assert f"crf={VIDEO_CRF}.0" in _x264_settings(path)
    assert VIDEO_CRF == 18


@pytest.mark.parametrize("shape", [(2, 48000), (1, 48000), (48000,)])
def test_audio_is_muxed_as_stereo_aac(tmp_path, shape):
    path = tmp_path / "clip.mp4"
    waveform = torch.sin(torch.linspace(0, 440 * 2 * 3.14159, 48000)).expand(*shape) * 0.5 if len(shape) > 1 else torch.zeros(48000)

    encode_video(_frames(24), 24, str(path), audio=waveform, audio_sample_rate=48000)

    with av.open(str(path)) as container:
        (audio,) = container.streams.audio
        assert audio.codec_context.name == "aac"
        assert audio.codec_context.sample_rate == 48000
        assert audio.codec_context.channels == 2
        assert audio.duration * audio.time_base == pytest.approx(1.0, abs=0.05)


def test_numpy_frames_encode_like_images(tmp_path):
    frames = [np.full((48, 64, 3), 128, dtype=np.uint8) for _ in range(3)]

    encode_video(frames, 24, str(tmp_path / "clip.mp4"))

    with av.open(str(tmp_path / "clip.mp4")) as container:
        assert sum(1 for _ in container.decode(video=0)) == 3
