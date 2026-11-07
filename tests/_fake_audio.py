"""In-memory stand-ins for the ffmpeg and ffprobe calls that write and read audio files, so no test starts either tool.

A "fake take" is a file with a ``.flac`` (or any audio) name whose bytes are a NumPy archive of the samples, the sample rate and
the embedded settings. :func:`install_fake_audio` replaces the product functions that save, probe, decode and read tags with
versions that understand these files and the 16-bit WAV files ``write_pcm16_wav`` makes; any other file reads as unreadable,
as a corrupt file does for the real tools. What our code does with the probe's JSON, the tags and the peaks still runs for real.
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_CONFIG_TAG = "zvisiongenerator.config"
_PEAK_RATE = 8000


@dataclass(frozen=True)
class FakeAudio:
    """What a fake audio file holds: float32 ``(channels, frames)`` samples, the rate, and the embedded settings (``None`` for a file without any)."""

    samples: np.ndarray
    rate: int
    payload: dict[str, Any] | None
    codec: str

    @property
    def seconds(self) -> float:
        return self.samples.shape[1] / self.rate


def write_fake_audio(samples: np.ndarray, sample_rate: int, path: str | Path, payload: dict[str, Any] | None) -> Path:
    """Write a fake take to *path* (the signature of ``save_flac``, with ``payload=None`` for a file with no settings) and return *path*."""
    path = Path(path)
    stored = "" if payload is None else json.dumps(payload, ensure_ascii=False)
    with path.open("wb") as handle:
        np.savez(handle, samples=np.asarray(samples, dtype=np.float32), rate=sample_rate, has_payload=payload is not None, payload=stored)
    return path


def read_fake_audio(path: str | Path) -> FakeAudio:
    """Return what the audio file at *path* holds; a fake take or a 16-bit PCM WAV.

    Raises:
        ValueError: If the file is missing or is neither, like a file the real tools cannot read.
    """
    path = Path(path)
    try:
        head = path.read_bytes()[:4]
    except OSError as exc:
        raise ValueError(f"{path} cannot be read") from exc
    if head == b"RIFF":
        with wave.open(str(path), "rb") as reader:
            channels, rate = reader.getnchannels(), reader.getframerate()
            pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16)
        return FakeAudio((pcm.reshape(-1, channels).T / 32768.0).astype(np.float32), rate, None, "pcm_s16le")
    if head == b"PK\x03\x04":
        with np.load(path, allow_pickle=False) as archive:
            payload = json.loads(str(archive["payload"])) if bool(archive["has_payload"]) else None
            return FakeAudio(archive["samples"], int(archive["rate"]), payload, "flac")
    raise ValueError(f"{path} is not audio")


def fake_save_flac(samples: np.ndarray, sample_rate: int, path: str | Path, payload: dict[str, Any]) -> Path:
    """Stand in for ``save_flac``: write a fake take holding *samples*, *sample_rate* and *payload*."""
    return write_fake_audio(samples, sample_rate, path, payload)


def fake_read_config(path: str | Path) -> dict[str, Any] | None:
    """Stand in for ``read_config``: return the settings a fake take holds, ``None`` for a file without any.

    Raises:
        subprocess.CalledProcessError: If the file cannot be read, as ffprobe's exit does.
    """
    try:
        return read_fake_audio(path).payload
    except ValueError as exc:
        raise subprocess.CalledProcessError(1, ["ffprobe"], stderr=str(exc)) from exc


def fake_ffprobe_json(path: str | Path, *, count_packets: bool = True, timeout: float | None = None) -> dict[str, Any]:
    """Stand in for ``ffprobe_json``: return the JSON ffprobe would print for a fake take or WAV.

    Raises:
        RuntimeError: If the file cannot be read, as the real function does when ffprobe fails.
    """
    try:
        audio = read_fake_audio(path)
    except ValueError as exc:
        raise RuntimeError(f"ffprobe could not read {path}: {exc}") from exc
    duration = f"{audio.seconds:.6f}"
    stream = {"codec_type": "audio", "codec_name": audio.codec, "channels": audio.samples.shape[0], "sample_rate": str(audio.rate), "duration": duration}
    tags = {} if audio.payload is None else {_CONFIG_TAG: json.dumps(audio.payload, ensure_ascii=False, sort_keys=True)}
    return {"streams": [stream], "format": {"duration": duration, "tags": tags}}


def fake_audio_decodes(path: str | Path, *, start: float, duration: float) -> bool:
    """Stand in for ``audio_decodes``: whether the file is a fake take or WAV."""
    try:
        read_fake_audio(path)
    except ValueError:
        return False
    return True


def fake_read_waveform(path: str | Path, count: int = 100, *, timeout: float | None = None) -> tuple[list[int], float]:
    """Stand in for ``read_waveform``: return the real peaks computation of the fake take's samples and its length in seconds.

    Raises:
        RuntimeError: If the file cannot be read, as the real function does when ffmpeg fails.
    """
    from zvisiongenerator.utils.audio_peaks import waveform_peaks

    try:
        audio = read_fake_audio(path)
    except ValueError as exc:
        raise RuntimeError(f"ffmpeg could not decode {path}: {exc}") from exc
    return waveform_peaks(audio.samples.mean(axis=0), count), audio.seconds


def fake_read_waveform_peaks(path: str | Path, count: int = 100) -> list[int]:
    """Stand in for ``read_waveform_peaks``."""
    return fake_read_waveform(path, count)[0]


def decode_fake(path: str | Path) -> tuple[np.ndarray, int]:
    """Return the samples ``(channels, frames)`` and the rate of the fake take or WAV at *path*."""
    audio = read_fake_audio(path)
    return audio.samples, audio.rate


# Product name -> (the module that defines it, its stand-in). The stand-in replaces the name in the defining module and in every
# loaded module that imported it, and in any module imported later, which reads it from the defining module.
_SEAMS = {
    "ffprobe_json": ("zvisiongenerator.utils.video_probe", fake_ffprobe_json),
    "audio_decodes": ("zvisiongenerator.utils.video_probe", fake_audio_decodes),
    "read_waveform": ("zvisiongenerator.utils.audio_peaks", fake_read_waveform),
    "read_waveform_peaks": ("zvisiongenerator.utils.audio_peaks", fake_read_waveform_peaks),
    "read_config": ("zvisiongenerator.utils.provenance", fake_read_config),
    "save_flac": ("zvisiongenerator.utils.audio_save", fake_save_flac),
}


def install_fake_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make saving, probing, decoding and tag reading of audio files work on fake takes and WAVs instead of starting ffmpeg or ffprobe."""
    import importlib

    for name, (module_name, fake) in _SEAMS.items():
        original = getattr(importlib.import_module(module_name), name)
        for module in [item for key, item in list(sys.modules.items()) if key.startswith("zvisiongenerator") and item is not None]:
            if getattr(module, name, None) is original:
                monkeypatch.setattr(module, name, fake)
