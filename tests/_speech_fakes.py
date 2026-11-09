"""Fakes for the speech backends: a scripted speech engine, and a fake ``faster_qwen3_tts`` with the patches that make the Qwen backend use it.

Import them with ``from tests._speech_fakes import ...``. Nothing here loads a real model.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tests._audio_presets import audio_config, qwen3_tts_preset

from zvisiongenerator.backends.audio import qwen3_tts as qwen3_tts_module
from zvisiongenerator.utils.audio_preset import AudioPreset, parse_audio_presets

RATE = 24000


def qwen_preset() -> AudioPreset:
    """Return the parsed ``qwen3_tts`` fixture preset."""
    return next(item for item in parse_audio_presets(audio_config(qwen3_tts=qwen3_tts_preset())) if item.id == "qwen3_tts")


def tone(seconds: float, level: float = 0.3) -> np.ndarray:
    """Return a mono float32 sine of *seconds* at 24 kHz."""
    return (level * np.sin(2 * np.pi * 220 * np.arange(round(seconds * RATE)) / RATE)).astype(np.float32)


class FakeSpeechEngine:
    """A :class:`SpeechEngine` that returns scripted parts and records what it was asked.

    ``waves`` are the parts' waveforms in order (the last one repeats); ``cuts`` are the 0-based parts that report a cut.
    """

    sample_rate = RATE

    def __init__(self, waves: list[np.ndarray] | None = None, cuts: tuple[int, ...] = (), events: list | None = None) -> None:
        self.waves = list(waves) if waves is not None else [tone(0.5)]
        self.cuts = cuts
        self.events = events if events is not None else []
        self.prepared: list[Any] = []
        self.parts: list[dict[str, Any]] = []

    def prepare_voice(self, voice: Any) -> Any:
        self.events.append("prepare")
        self.prepared.append(voice)
        return ("prompt", voice.text)

    def read_part(self, text: str, seed: int, *, speaker: str | None, reference: Any, delivery: Any, max_new_tokens: int) -> tuple[np.ndarray, bool]:
        index = len(self.parts)
        self.events.append(("part", index))
        self.parts.append({"text": text, "seed": seed, "speaker": speaker, "reference": reference, "delivery": delivery, "max_new_tokens": max_new_tokens})
        return self.waves[min(index, len(self.waves) - 1)], index in self.cuts


class FakeFasterModel:
    """A stand-in for ``FasterQwen3TTS``: it records loads and calls, and answers every call with the next scripted wave.

    The class keeps ``loaded`` (every ``from_pretrained``, in order) and ``waves`` (the scripted answers, the last one repeating);
    :func:`install_fake_faster_qwen` resets them for each test.
    """

    loaded: list[dict[str, Any]] = []
    waves: list[np.ndarray] = []
    fail_warmup = False

    def __init__(self, folder: str, kwargs: dict[str, Any]) -> None:
        self.folder = folder
        self.kwargs = kwargs
        self.warmups = 0
        self.calls: list[dict[str, Any]] = []
        self.prompts: list[dict[str, Any]] = []
        self.model = SimpleNamespace(create_voice_clone_prompt=self._create_voice_clone_prompt)

    @classmethod
    def from_pretrained(cls, folder: str, **kwargs: Any) -> FakeFasterModel:
        instance = cls(folder, kwargs)
        cls.loaded.append({"folder": folder, **kwargs, "instance": instance})
        return instance

    def warmup(self) -> None:
        if type(self).fail_warmup:
            raise RuntimeError("graph capture failed")
        self.warmups += 1

    def _create_voice_clone_prompt(self, ref_audio: Any, ref_text: str) -> list[str]:
        self.prompts.append({"ref_audio": ref_audio, "ref_text": ref_text})
        return [f"prompt:{ref_text}"]

    def _answer(self, call: dict[str, Any]) -> tuple[list[np.ndarray], int]:
        import torch

        call["seed"] = torch.initial_seed()
        self.calls.append(call)
        waves = type(self).waves
        return [waves[min(len(self.calls) - 1, len(waves) - 1)]], RATE

    def generate_custom_voice(self, text: str, speaker: str, language: str, instruct: str | None = None, max_new_tokens: int = 1200, **_: Any) -> tuple[list[np.ndarray], int]:
        return self._answer({"kind": "custom_voice", "text": text, "speaker": speaker, "language": language, "instruct": instruct, "max_new_tokens": max_new_tokens})

    def generate_voice_clone(self, text: str, language: str, voice_clone_prompt: Any = None, max_new_tokens: int = 1200, **_: Any) -> tuple[list[np.ndarray], int]:
        return self._answer({"kind": "voice_clone", "text": text, "language": language, "voice_clone_prompt": voice_clone_prompt, "max_new_tokens": max_new_tokens})

    def generate_voice_design(self, text: str, instruct: str, language: str, max_new_tokens: int = 2048, **_: Any) -> tuple[list[np.ndarray], int]:
        return self._answer({"kind": "voice_design", "text": text, "instruct": instruct, "language": language, "max_new_tokens": max_new_tokens})


def install_fake_faster_qwen(monkeypatch: Any, tmp_path: Path, *, waves: list[np.ndarray] | None = None) -> SimpleNamespace:
    """Make the Qwen backend load :class:`FakeFasterModel` from fake snapshot folders and note the shim and the memory release.

    Returns the ``events`` list (``("shim",)``, ``("load", repo)``, ``("release",)`` in order) beside the model class.
    """
    events: list[tuple] = []
    FakeFasterModel.loaded = []
    FakeFasterModel.waves = list(waves) if waves is not None else [tone(0.5)]
    FakeFasterModel.fail_warmup = False
    module = types.ModuleType("faster_qwen3_tts")
    original_load = FakeFasterModel.from_pretrained.__func__

    def from_pretrained(cls, folder: str, **kwargs: Any) -> FakeFasterModel:
        events.append(("load", Path(folder).name))
        return original_load(cls, folder, **kwargs)

    monkeypatch.setattr(FakeFasterModel, "from_pretrained", classmethod(from_pretrained))
    module.FasterQwen3TTS = FakeFasterModel
    monkeypatch.setitem(sys.modules, "faster_qwen3_tts", module)
    monkeypatch.setattr(qwen3_tts_module, "apply_qwen_shim", lambda: events.append(("shim",)))
    monkeypatch.setattr(qwen3_tts_module, "snapshot_dir", lambda files: tmp_path / files.repo_id.rsplit("/", 1)[1])
    monkeypatch.setattr(qwen3_tts_module, "read_voice", lambda voice, max_seconds, rate: tone(min(2.0, max_seconds)))
    monkeypatch.setattr("zvisiongenerator.backends.cuda.memory.release_memory", lambda: events.append(("release",)))
    return SimpleNamespace(events=events, model=FakeFasterModel)
