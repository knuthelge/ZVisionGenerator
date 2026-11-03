"""Higgs TTS 3 speech backend: our own model class on the official weights, transformers' codec, and long text read in parts.

The network is :class:`~zvisiongenerator.backends.audio.higgs3_model.Higgs3Model`, filled from the official
``bosonai/higgs-tts-3-4b`` snapshot, and the codes of a take come from ``higgs3_generate.generate_speech_codes``. The audio
codec is transformers' ``HiggsAudioV2TokenizerModel`` at the pinned revision, loaded without remote code on the preset's
``codec_device``. Both repositories are read from the Hugging Face cache only; loading never downloads. Text is read in
parts (``utils/speech_text``), each part seeded ``seed + index``, and the parts are joined into one take
(``utils/speech_join``). The generation loop calls back once per generated frame, which is how progress and Stop reach it.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from zvisiongenerator.backends.audio.higgs3_config import read_higgs3_config
from zvisiongenerator.backends.audio.higgs3_generate import generate_speech_codes
from zvisiongenerator.backends.audio.higgs3_prompt import load_tokenizer, special_ids
from zvisiongenerator.core.audio_backend import AudioCancelled
from zvisiongenerator.core.audio_types import AudioClip
from zvisiongenerator.core.video_types import AudioTrack
from zvisiongenerator.utils.audio_request import check_speech_request
from zvisiongenerator.utils.audio_resample import resample_audio
from zvisiongenerator.utils.audio_track import probe_track_seconds, read_track
from zvisiongenerator.utils.h3_preset import HubFiles
from zvisiongenerator.utils.hf_snapshot import cached_snapshot
from zvisiongenerator.utils.speech_join import join_speech_parts
from zvisiongenerator.utils.speech_text import split_speech_text
from zvisiongenerator.utils.torch_flags import TorchFlags, capture_torch_flags, restore_torch_flags

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.backends.audio.higgs3_prompt import SpecialIds
    from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
    from zvisiongenerator.utils.audio_preset import AudioFiles, AudioLimits, AudioPlatform, AudioPreset

__all__ = ["SAMPLE_RATE", "Higgs3Backend"]

SAMPLE_RATE = 24000
"""The model's sample rate, and the rate of every take."""

SPEECH_PHASE = "Speech tokens"
DECODE_PHASE = "Decode"

_FRAME_STRIDE = 10
"""Report every this many frames (about 2.5 a second at the codec's 25 frames a second), so a long take does not flood the progress events."""

_OFFICIAL_REPO = "bosonai/higgs-tts-3-4b"
_CODEC_REPO = "bosonai/higgs-audio-v2-tokenizer"

# A recording is read at twice the model's rate and then polyphase-resampled to it, which matched the model's own
# resampling in level and word error rate.
_READ_RATE = 2 * SAMPLE_RATE


@dataclass
class _Progress:
    """The position of the part being made, which the frame callback and the decoder report from."""

    on_progress: ProgressCallback
    cancelled: CancelCheck
    max_new_tokens: int
    n_parts: int
    part: int = 1
    step: int = 0
    reported: int = 0


class Higgs3Backend:
    """Speech from Higgs TTS 3 on a torch device, with long text read in parts and an optional voice to follow."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._ids: SpecialIds | None = None
        self._codec: Any = None
        self._device = ""
        self._codec_device = "cpu"
        self._boc_id = 0
        self._limits: AudioLimits | None = None
        self._settings: Mapping[str, Any] = {}
        self._progress: _Progress | None = None
        self._flags: TorchFlags | None = None

    def load(self, preset: AudioPreset, platform_entry: AudioPlatform, device: str) -> None:
        """Load the model onto *device* and the codec onto the preset's ``codec_device``, from the Hugging Face cache.

        Nothing is downloaded: both repositories must already be in the cache.

        Args:
            preset: The ``higgs3`` preset, whose limits and settings apply.
            platform_entry: The preset's entry for this platform, which pins the model's and the codec's files.
            device: ``cuda`` or ``mps``.

        Raises:
            ValueError: If the platform entry does not list the files of the model or the codec.
            FileNotFoundError: If a repository is not in the cache with every listed file.
        """
        official = _cached_folder(_entry_files(preset, platform_entry, _OFFICIAL_REPO))
        codec_dir = _cached_folder(_entry_files(preset, platform_entry, _CODEC_REPO))
        self.close()
        self._limits = preset.limits
        self._settings = {**preset.settings, **platform_entry.settings}
        self._device = device
        self._codec_device = str(self._settings.get("codec_device", "cpu"))
        self._flags = capture_torch_flags()  # model code may set global numerics flags; close() puts them back
        try:
            self._load_models(official, codec_dir)
        except BaseException:
            self.close()
            raise

    def generate(self, request: AudioGenerationRequest, on_progress: ProgressCallback, cancelled: CancelCheck) -> AudioClip:
        """Read the request's text, in parts when it is long, and return it as one take.

        Part *i* is seeded ``request.seed + i``. Part 1 follows the voice recording (with its transcript) when there is
        one, and is read in the model's own voice when there is not. When there are more parts, part 1's own waveform
        is encoded once and is the reference (with part 1's text as the transcript) for parts 2 to *n*, so one voice
        and room reads the whole text.

        Args:
            request: The take to make.
            on_progress: Called as ``on_progress("Speech tokens", step, max_new_tokens, part, n_parts)`` for the first
                frame of a part, every ``_FRAME_STRIDE`` frames after it and the last frame, and with phase ``Decode`` around each part's decode.
            cancelled: Checked before each part and on every frame; ``True`` stops the take.

        Returns:
            One mono take at 24 kHz, with ``cut`` set when some part reached ``max_new_tokens`` without finishing.

        Raises:
            AudioCancelled: If ``cancelled`` returned ``True``.
            ValueError: If the request is refused by :func:`~zvisiongenerator.utils.audio_request.check_speech_request`.
            RuntimeError: If the backend is not loaded, or the model made no sound.
        """
        if self._model is None or self._limits is None:
            raise RuntimeError("Higgs TTS 3 is not loaded; call load() first.")
        limits = self._limits
        check_speech_request(request, limits)
        parts = split_speech_text(request.prompt, int(limits.part_chars), max_text_chars=int(limits.max_text_chars))
        max_new_tokens = int(limits.max_new_tokens)
        reference = self._voice_reference(request.voice) if request.voice is not None else None  # for part 1 only
        progress = _Progress(on_progress, cancelled, max_new_tokens, len(parts))
        self._progress = progress
        waves: list[np.ndarray] = []
        cut = False
        try:
            for index, text in enumerate(parts):
                if cancelled():
                    raise AudioCancelled
                progress.part, progress.step, progress.reported = index + 1, 0, 0
                wave, part_cut = self._read_part(text, request.seed + index, reference, max_new_tokens)
                cut = cut or part_cut
                if index == 0 and len(parts) > 1:
                    if not wave.size:
                        raise RuntimeError("Higgs TTS 3 made no sound for the first part of the text, so the rest has no voice to follow. Try another seed.")
                    reference = (self._encode(wave), text)
                waves.append(wave)
        finally:
            self._progress = None
        if not any(wave.size for wave in waves):
            raise RuntimeError("Higgs TTS 3 made no sound for this text. Try another seed.")
        joined = join_speech_parts(waves, SAMPLE_RATE, self._settings)
        return AudioClip(samples=joined.reshape(1, -1), sample_rate=SAMPLE_RATE, cut=cut)

    def close(self) -> None:
        """Free the model, the codec and the accelerator memory they held, and restore the torch flags.

        Safe to call again, after a failed load and after a failed take.
        """
        model, device, flags = self._model, self._device, self._flags
        self._model = self._tokenizer = self._ids = self._codec = None
        self._progress = None
        self._flags = None
        try:
            del model
            gc.collect()
            if device:
                _empty_cache(device)
        finally:
            if flags is not None:
                restore_torch_flags(flags)

    def _load_models(self, official: Path, codec_dir: Path) -> None:
        """Load the tokenizer, the model and the codec from their cached folders."""
        import torch
        from transformers import HiggsAudioV2TokenizerModel

        from zvisiongenerator.backends.audio.higgs3_model import load_higgs3_model

        config = read_higgs3_config(official / "config.json")
        tokenizer = load_tokenizer(official)
        ids = special_ids(tokenizer)
        model = load_higgs3_model(config, official / "model.safetensors", self._device)
        # No remote code: the codec class is native to transformers.
        codec = HiggsAudioV2TokenizerModel.from_pretrained(str(codec_dir), dtype=torch.float32).to(self._codec_device).eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        self._boc_id = config.boc_id
        self._tokenizer, self._ids, self._model, self._codec = tokenizer, ids, model, codec

    def _decode(self, codes: Any) -> np.ndarray:
        """Decode raw codes ``[frames, codebooks]`` into a mono 24 kHz float32 waveform on the codec's device, reporting the Decode phase."""
        import torch

        progress = self._progress
        if progress is not None:
            progress.on_progress(DECODE_PHASE, 0, 1, progress.part, progress.n_parts)
        with torch.no_grad():
            codes = torch.where(codes >= self._boc_id, torch.zeros_like(codes), codes)  # the two ids above the codec's values are stream markers
            batch = codes.transpose(0, 1).unsqueeze(0).to(self._codec_device, torch.long)
            audio = self._codec.decode(batch).audio_values.squeeze(0).squeeze(0).detach().cpu().float()
        if progress is not None:
            progress.on_progress(DECODE_PHASE, 1, 1, progress.part, progress.n_parts)
        return np.asarray(audio.numpy(), dtype=np.float32)

    def _count_frame(self) -> None:
        """Run once per generated frame: stop if asked, else count and report the first frame and every ``_FRAME_STRIDE``-th."""
        progress = self._progress
        if progress is None:
            return
        if progress.cancelled():
            raise AudioCancelled
        progress.step += 1
        if progress.step == 1 or progress.step % _FRAME_STRIDE == 0:
            self._report_frames(progress)

    @staticmethod
    def _report_frames(progress: _Progress) -> None:
        """Report the frame count of the part being made and remember that it was reported."""
        progress.reported = progress.step
        progress.on_progress(SPEECH_PHASE, progress.step, progress.max_new_tokens, progress.part, progress.n_parts)

    def _read_part(self, text: str, seed: int, reference: tuple[Any, str | None] | None, max_new_tokens: int) -> tuple[np.ndarray, bool]:
        """Return the 24 kHz mono waveform of one part and whether it was cut, generated with *seed* and following *reference* (codes and transcript) when given."""
        import torch

        torch.manual_seed(seed)
        codes, transcript = reference if reference is not None else (None, None)
        result = generate_speech_codes(
            self._model,
            self._tokenizer,
            self._ids,
            text,
            reference_codes=codes,
            reference_text=transcript,
            max_new_tokens=max_new_tokens,
            temperature=self._setting("temperature"),
            top_p=self._setting("top_p"),
            on_frame=self._count_frame,
        )
        progress = self._progress
        if progress is not None and progress.reported != progress.step:  # the last frame of a part is always reported
            self._report_frames(progress)
        wave = self._decode(result.codes) if result.codes.shape[0] else np.zeros(0, dtype=np.float32)  # no decode is reported for a part without sound
        return wave, result.cut

    def _voice_reference(self, voice: VoiceRef) -> tuple[Any, str | None]:
        """Return the codes and transcript of the voice recording: read, polyphase-resampled to 24 kHz and encoded once."""
        seconds = min(probe_track_seconds(voice.path), float(self._limits.max_voice_seconds))
        stereo = read_track(AudioTrack(voice.path, 0.0), _READ_RATE, seconds)
        mono = resample_audio(stereo.mean(axis=0, keepdims=True), _READ_RATE, SAMPLE_RATE)[0]
        return self._encode(mono), (voice.text or None)

    def _encode(self, wave: np.ndarray) -> Any:
        """Encode a 24 kHz mono waveform with the codec into ``[frames, codebooks]`` int64 codes on the CPU."""
        import torch

        if wave.shape[0] < SAMPLE_RATE:  # a reference shorter than a second is padded to one
            wave = np.pad(wave, (0, SAMPLE_RATE - wave.shape[0]))
        batch = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)).reshape(1, 1, -1).to(self._codec_device, torch.float32)
        with torch.no_grad():
            codes = self._codec.encode(batch).audio_codes
        return codes.squeeze(0).transpose(0, 1).to(torch.long).cpu()

    def _setting(self, name: str) -> Any:
        """Return the preset setting *name*, or raise ``ValueError`` naming it."""
        if name not in self._settings:
            raise ValueError(f"The Higgs TTS 3 preset needs the setting '{name}'.")
        return self._settings[name]


def _entry_files(preset: AudioPreset, platform_entry: AudioPlatform, repo_id: str) -> AudioFiles:
    """Return the files the platform entry lists for *repo_id*.

    Raises:
        ValueError: If the entry lists none.
    """
    files = next((files for files in platform_entry.files if files.repo_id == repo_id), None)
    if files is None:
        raise ValueError(f"The platform entry of '{preset.id}' must list the files of {repo_id}.")
    return files


def _cached_folder(files: AudioFiles) -> Path:
    """Return the cached snapshot folder that holds every listed file of *files*; never download.

    Raises:
        FileNotFoundError: If the revision is not in the cache or lacks a listed file.
    """
    folder = cached_snapshot(HubFiles(files.repo_id, files.revision, files.patterns))
    if folder is None:
        raise FileNotFoundError(f"Higgs TTS 3 needs {files.repo_id}@{files.revision}, which is not downloaded.")
    return folder


def _empty_cache(device: str) -> None:
    """Return the cached accelerator memory of *device* to the system."""
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
