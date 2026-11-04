"""Higgs TTS 3 speech backend: our own model class on the official weights, transformers' codec, and long text read in parts.

The network is :class:`~zvisiongenerator.backends.audio.higgs3_model.Higgs3Model`, filled from the official
``bosonai/higgs-tts-3-4b`` snapshot, and the codes of a take come from ``higgs3_generate.generate_speech_codes``. The audio
codec is transformers' ``HiggsAudioV2TokenizerModel`` at the pinned revision, loaded without remote code on the preset's
``codec_device``. Both repositories are read from the Hugging Face cache only; loading never downloads. Text is read in
parts (``utils/speech_text``), each part seeded ``seed + index``, and the parts are joined into one take
(``utils/speech_join``). The generation loop calls back once per generated frame, which is how progress and Stop reach it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from zvisiongenerator.backends.audio.build_cancel import stop_while_building
from zvisiongenerator.backends.audio.higgs3_config import read_higgs3_config
from zvisiongenerator.backends.audio.higgs3_generate import generate_speech_codes
from zvisiongenerator.backends.audio.higgs3_prompt import load_tokenizer, special_ids
from zvisiongenerator.core.audio_backend import DECODE_PHASE, AudioCancelled, raise_if_cancelled
from zvisiongenerator.core.audio_types import AudioClip
from zvisiongenerator.core.video_types import AudioTrack
from zvisiongenerator.utils.audio_resample import resample_audio
from zvisiongenerator.utils.audio_track import probe_track_seconds, read_track
from zvisiongenerator.backends.audio import snapshot_dir
from zvisiongenerator.utils.speech_join import join_speech_parts
from zvisiongenerator.utils.speech_text import split_speech_text

if TYPE_CHECKING:
    from zvisiongenerator.core.audio_backend import CancelCheck, ProgressCallback
    from zvisiongenerator.backends.audio.higgs3_prompt import SpecialIds
    from zvisiongenerator.core.audio_types import AudioGenerationRequest, VoiceRef
    from zvisiongenerator.utils.audio_preset import AudioLimits, AudioPreset

__all__ = ["SAMPLE_RATE", "Higgs3Backend"]

SAMPLE_RATE = 24000
"""The model's sample rate, and the rate of every take."""

SPEECH_PHASE = "Speech tokens"

_FRAME_STRIDE = 10
"""Report every this many frames (about 2.5 a second at the codec's 25 frames a second), so a long take does not flood the progress events."""

_FRAMES_PER_SECOND = 25
"""The codec's frame rate: frames of speech codes per second of sound."""

_CHARS_PER_SECOND = 16.5
"""How many characters of text the model speaks a second, in the middle of the 16 to 18 measured over real takes."""

_FRAMES_PER_CHAR = _FRAMES_PER_SECOND / _CHARS_PER_SECOND
"""How many frames a character of text takes to read, which turns a part's length into the frames it should need."""

# A recording is read at twice the model's rate and then polyphase-resampled to it, which matched the model's own
# resampling in level and word error rate.
_READ_RATE = 2 * SAMPLE_RATE


def _estimate_part_frames(text: str, max_new_tokens: int) -> int:
    """Return how many frames reading *text* should take, at least 1 and at most *max_new_tokens*.

    The estimate is the text's length at the model's usual speaking rate; a part with more pauses or a slower voice
    needs more, which :func:`_part_total` allows for.
    """
    return max(1, min(max_new_tokens, round(len(text) * _FRAMES_PER_CHAR)))


def _part_total(estimate: int, step: int, max_new_tokens: int) -> int:
    """Return the total to report at frame *step* of a part expected to take *estimate* frames.

    It is the estimate until the part runs past it; after that it stays one stride ahead, so the bar waits near the end
    instead of overflowing. It never exceeds *max_new_tokens*, which ends a part.
    """
    return min(max_new_tokens, max(estimate, step + _FRAME_STRIDE))


def _trim_reference(wave: np.ndarray, text: str, max_seconds: float) -> tuple[np.ndarray, str]:
    """Return the start of *wave* up to *max_seconds* with the same share of the words of *text*, or both as they are when *wave* is short enough.

    A voice reference is held to the length of a recording; the transcript is cut in proportion so it still matches what is heard.
    """
    keep = int(max_seconds * SAMPLE_RATE)
    if wave.shape[0] <= keep:
        return wave, text
    words = text.split()
    kept_words = max(1, int(len(words) * keep / wave.shape[0]))
    return wave[:keep], " ".join(words[:kept_words])


@dataclass
class _Progress:
    """The position of the part being made, which the frame callback and the decoder report from."""

    on_progress: ProgressCallback
    cancelled: CancelCheck
    max_new_tokens: int
    n_parts: int
    part: int = 1
    estimate: int = 1
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

    def load(self, preset: AudioPreset, device: str, cancelled: CancelCheck | None = None) -> None:
        """Load the model onto *device* and the codec onto the preset's ``codec_device`` (the CPU on CUDA and MPS on a Mac by default), from the Hugging Face cache.

        Nothing is downloaded: both repositories must already be in the cache.

        Args:
            preset: The ``higgs3`` preset: its pinned model and codec files, its limits and its settings for *device*.
            device: ``cuda`` or ``mps``.
            cancelled: Called between the load steps (tokenizer, network, codec), while the network and the codec are built and
                before each tensor of the model is copied; ``True`` stops the load.

        Raises:
            FileNotFoundError: If a repository is not in the cache with every listed file.
            AudioCancelled: If *cancelled* returned ``True``; everything loaded so far is freed.
        """
        model_files, codec_files = preset.files
        official, codec_dir = snapshot_dir(model_files), snapshot_dir(codec_files)
        self.close()
        self._limits = preset.limits
        self._settings = preset.settings_for(device)
        self._device = device
        self._codec_device = str(self._settings.get("codec_device", "mps" if device == "mps" else "cpu"))
        try:
            self._load_models(official, codec_dir, cancelled)
        except BaseException:
            self.close()
            raise

    def generate(self, request: AudioGenerationRequest, on_progress: ProgressCallback, cancelled: CancelCheck) -> AudioClip:
        """Read the request's text, in parts when it is long, and return it as one take.

        Part *i* is seeded ``request.seed + i``. With a voice recording, every part follows the recording (with its
        transcript). Without one, part 1 is read in the model's own voice; when there are more parts, part 1's own
        waveform is encoded once and is the reference for parts 2 to *n* (with part 1's text as the transcript), so one
        voice and room reads the whole text. That reference is cut to the preset's ``max_voice_seconds``, the same
        length a recording is held to, with the transcript cut to the same share of its words.

        Args:
            request: The take to make.
            on_progress: Called as ``on_progress("Speech tokens", step, total, part, n_parts)`` for the first frame of a
                part, every ``_FRAME_STRIDE`` frames after it and the last frame, and with phase ``Decode`` around each
                part's decode. ``total`` is the frames the part's text should need (``_estimate_part_frames``); the last
                frame reports the frames the part took as its total, so a part's bar ends full.
            cancelled: Checked before each part and on every frame; ``True`` stops the take.

        Returns:
            One mono take at 24 kHz, with ``cut`` set when some part reached ``max_new_tokens`` without finishing. The codec's
            output is not limited to -1..1, so a loud take may peak above 1.0; the runner levels it.

        Raises:
            AudioCancelled: If ``cancelled`` returned ``True``.
            RuntimeError: If the model made no sound.
        """
        limits = self._limits
        parts = split_speech_text(request.prompt, int(limits.part_chars), max_text_chars=int(limits.max_text_chars))
        max_new_tokens = int(limits.max_new_tokens)
        voice_reference = self._voice_reference(request.voice) if request.voice is not None else None
        reference = voice_reference  # a recording is every part's reference; otherwise part 1 has none and the later parts follow part 1
        progress = _Progress(on_progress, cancelled, max_new_tokens, len(parts))
        self._progress = progress
        waves: list[np.ndarray] = []
        cut = False
        try:
            for index, text in enumerate(parts):
                if cancelled():
                    raise AudioCancelled
                progress.part, progress.step, progress.reported = index + 1, 0, 0
                progress.estimate = _estimate_part_frames(text, max_new_tokens)
                wave, part_cut = self._read_part(text, request.seed + index, reference, max_new_tokens)
                cut = cut or part_cut
                if index == 0 and len(parts) > 1 and voice_reference is None:
                    if not wave.size:
                        raise RuntimeError("Higgs TTS 3 made no sound for the first part of the text, so the rest has no voice to follow. Try another seed.")
                    reference_wave, reference_text = _trim_reference(wave, text, float(limits.max_voice_seconds))
                    reference = (self._encode(reference_wave), reference_text)
                waves.append(wave)
        finally:
            self._progress = None
        if not any(wave.size for wave in waves):
            raise RuntimeError("Higgs TTS 3 made no sound for this text. Try another seed.")
        joined = join_speech_parts(waves, SAMPLE_RATE, self._settings)
        return AudioClip(samples=joined.reshape(1, -1), sample_rate=SAMPLE_RATE, cut=cut)

    def close(self) -> None:
        """Drop the model and the codec; the runner returns the accelerator memory and the torch flags.

        Safe to call again, after a failed load and after a failed take.
        """
        self._model = self._tokenizer = self._ids = self._codec = None
        self._progress = None

    def _load_models(self, official: Path, codec_dir: Path, cancelled: CancelCheck | None = None) -> None:
        """Load the tokenizer, the model and the codec from their cached folders."""
        import torch
        from transformers import HiggsAudioV2TokenizerModel

        from zvisiongenerator.backends.audio.higgs3_model import load_higgs3_model

        raise_if_cancelled(cancelled)  # the imports above take seconds
        config = read_higgs3_config(official / "config.json")
        tokenizer = load_tokenizer(official)
        ids = special_ids(tokenizer)
        raise_if_cancelled(cancelled)
        model = load_higgs3_model(config, official / "model.safetensors", self._device, cancelled)
        raise_if_cancelled(cancelled)
        # No remote code: the codec class is native to transformers.
        with stop_while_building(cancelled):
            codec = HiggsAudioV2TokenizerModel.from_pretrained(str(codec_dir), dtype=torch.float32)
        raise_if_cancelled(cancelled)
        codec = codec.to(self._codec_device).eval()
        raise_if_cancelled(cancelled)
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
    def _report_frames(progress: _Progress, *, finished: bool = False) -> None:
        """Report the frame count of the part being made against its estimated total, and remember that it was reported.

        A *finished* part reports the frames it took as its total, so its bar is full.
        """
        progress.reported = progress.step
        total = progress.step if finished else _part_total(progress.estimate, progress.step, progress.max_new_tokens)
        progress.on_progress(SPEECH_PHASE, progress.step, total, progress.part, progress.n_parts)

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
            temperature=self._settings["temperature"],
            top_p=self._settings["top_p"],
            on_frame=self._count_frame,
        )
        progress = self._progress
        if progress is not None and progress.step:  # the last frame of a part is always reported, with the frames it took as the total
            self._report_frames(progress, finished=True)
        wave = self._decode(result.codes) if result.codes.shape[0] else np.zeros(0, dtype=np.float32)  # no decode is reported for a part without sound
        return wave, result.cut

    def _voice_reference(self, voice: VoiceRef) -> tuple[Any, str]:
        """Return the codes and transcript of the voice recording: read, polyphase-resampled to 24 kHz and encoded once."""
        seconds = min(probe_track_seconds(voice.path), float(self._limits.max_voice_seconds))
        stereo = read_track(AudioTrack(voice.path, 0.0), _READ_RATE, seconds)
        mono = resample_audio(stereo.mean(axis=0, keepdims=True), _READ_RATE, SAMPLE_RATE)[0]
        return self._encode(mono), voice.text

    def _encode(self, wave: np.ndarray) -> Any:
        """Encode a 24 kHz mono waveform with the codec into ``[frames, codebooks]`` int64 codes on the CPU."""
        import torch

        if wave.shape[0] < SAMPLE_RATE:  # a reference shorter than a second is padded to one
            wave = np.pad(wave, (0, SAMPLE_RATE - wave.shape[0]))
        batch = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)).reshape(1, 1, -1).to(self._codec_device, torch.float32)
        with torch.no_grad():
            codes = self._codec.encode(batch).audio_codes
        return codes.squeeze(0).transpose(0, 1).to(torch.long).cpu()
