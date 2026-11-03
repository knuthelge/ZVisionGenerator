"""Shared builders for the audio tests: synthetic clips, decoding a saved file, and the runner's and CLI's patched surroundings."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from tests._audio_presets import audio_config

from zvisiongenerator.core.audio_types import AudioClip, AudioGenerationRequest, VoiceRef
from zvisiongenerator.utils.audio_preset import AudioPreset, parse_audio_presets
from zvisiongenerator.utils.audio_save import save_flac
from zvisiongenerator.utils.video_splice import write_pcm16_wav


def preset(model_id: str) -> AudioPreset:
    """Return the parsed fixture preset *model_id* (``yue2``, ``higgs3`` or ``sa3``)."""
    return next(item for item in parse_audio_presets(audio_config()) if item.id == model_id)


def sine(seconds: float, rate: int = 24000, amplitude: float = 0.5, frequency: float = 220.0, channels: int = 1) -> np.ndarray:
    """Return a float32 sine shaped ``(channels, frames)``."""
    wave = amplitude * np.sin(2 * np.pi * frequency * np.arange(round(seconds * rate)) / rate)
    return np.tile(wave.astype(np.float32), (channels, 1))


def clip(seconds: float = 0.5, rate: int = 24000, amplitude: float = 0.5, channels: int = 1, **fields: Any) -> AudioClip:
    """Return an :class:`AudioClip` of a sine, with *fields* (``cut``, ``preclamp``) set."""
    return AudioClip(samples=sine(seconds, rate, amplitude, channels=channels), sample_rate=rate, **fields)


def decode(path: Path, channels: int = 1) -> tuple[np.ndarray, int]:
    """Decode an audio file with ffmpeg to float32 ``(channels, frames)`` and return it with the file's sample rate."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True
    )
    rate, found = (int(part) for part in probe.stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, found).T.copy(), rate


def fake_backend(*clips: AudioClip, events: list | None = None) -> MagicMock:
    """Return a backend whose ``generate`` returns *clips* in turn (the last one repeats) and notes load, generate and close in *events*."""
    backend = MagicMock()
    queue = list(clips)
    log = events if events is not None else []
    backend.load.side_effect = lambda *args, **kwargs: log.append("load")
    backend.close.side_effect = lambda: log.append("close")

    def generate(request, on_progress, cancelled):
        log.append("generate")
        return queue.pop(0) if len(queue) > 1 else queue[0]

    backend.generate.side_effect = generate
    backend.events = log
    return backend


def patch_runner(monkeypatch, backend: Any, *, ensure_outcome: str = "ready") -> SimpleNamespace:
    """Patch the runner's surroundings: this machine is CUDA, the backend is *backend*, nothing downloads.

    Returns the patched ``ensure_downloaded`` and enhancer session mocks.
    """
    from zvisiongenerator import audio_cli, audio_runner

    ensure = MagicMock(return_value=ensure_outcome)
    session = MagicMock()
    monkeypatch.setattr(audio_cli, "audio_platform_key", lambda: "cuda")
    monkeypatch.setattr(audio_runner, "audio_platform_key", lambda: "cuda")
    monkeypatch.setattr(audio_runner, "get_audio_backend", MagicMock(return_value=backend))
    monkeypatch.setattr(audio_runner, "ensure_downloaded", ensure)
    monkeypatch.setattr(audio_runner, "prompt_enhancer_session", session)
    return SimpleNamespace(ensure=ensure, session=session)


def voice_folder(root: Path) -> Path:
    """Return a folder under *root* with a recording (``me.wav``), an *Any* take (``any.flac``) and a cloned take (``cloned.flac``), each 4 s."""
    folder = root / "voices"
    folder.mkdir()
    tone = sine(4.0, rate=24000)
    write_pcm16_wav(tone, folder / "me.wav", 24000)
    save_flac(tone, 24000, folder / "any.flac", {"kind": "speech", "prompt": "Welcome back."})
    save_flac(tone, 24000, folder / "cloned.flac", {"kind": "speech", "prompt": "x", "voice_path": "~/me.wav", "voice_text": "Hello, this is me."})
    return folder


def case_request(item: dict, voices: Path) -> AudioGenerationRequest:
    """Return the request a ``tests/fixtures/audio_cli_cases.json`` case expects, with its ``/voices/`` paths moved into *voices*."""
    voice = item["voice"]
    return AudioGenerationRequest(
        preset=item["preset"],
        kind=item["kind"],
        prompt=item["prompt"],
        lyrics=item["lyrics"],
        seconds=item["seconds"],
        voice=None if voice is None else VoiceRef(voice["path"].replace("/voices/", f"{voices}/"), voice["text"]),
        seed=item["seed"],
    )
