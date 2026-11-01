"""Pin a clip's last sound at the start of an LTX-2 extension window on diffusers.

``LTX2ConditionPipeline`` pins video frames but gives every audio token the same timestep, so the carried sound
cannot be held. :func:`pin_audio_prefix` does it around the pipeline call, as LTX's denoise mask does on macOS:
the transformer sees the carried audio tokens clean, at timestep 0, on every step, and :func:`restore_audio_prefix`
puts the clean tokens back into the result before it is decoded.

:func:`pin_audio_track` holds a whole user track instead: every audio token is clean at timestep 0 and audio sigma 0 on
every step, so the video is generated against that sound and the model's own audio is left to follow it.

The carried sound is encoded the way the MLX pipeline encodes it: a log-mel spectrogram of the stereo waveform
at the audio VAE's sample rate (torchaudio ``MelSpectrogram`` settings: Hann window, centred reflect padding,
magnitude, Slaney mel scale and norm), through the audio VAE encoder's mean.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

__all__ = ["AudioCarry", "encode_audio_carry", "log_mel", "pin_audio_prefix", "pin_audio_track", "restore_audio_prefix", "slaney_mel_basis"]

_N_FFT = 1024
_LOG_FLOOR = 1e-5


@dataclass(frozen=True)
class AudioCarry:
    """The carried sound: its latent ``(1, channels, tokens, mel_bins)`` and the same tokens packed and normalized as the denoiser sees them."""

    latent: torch.Tensor
    tokens: torch.Tensor

    @property
    def token_count(self) -> int:
        """Return how many audio tokens at the start of the window are pinned."""
        return int(self.latent.shape[2])


def slaney_mel_basis(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Return a Slaney-scale mel filterbank with Slaney area normalization, shape ``(n_mels, n_fft // 2 + 1)``.

    Matches ``torchaudio.transforms.MelSpectrogram(mel_scale="slaney", norm="slaney")`` from 0 Hz to Nyquist.
    """
    import numpy as np

    def hz_to_mel(f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=np.float64)
        return np.where(f < 1000.0, 3.0 * f / 200.0, 15.0 + 27.0 * np.log(np.maximum(f, 1e-10) / 1000.0) / np.log(6.4))

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float64)
        return np.where(m < 15.0, 200.0 * m / 3.0, 1000.0 * np.exp((m - 15.0) * np.log(6.4) / 27.0))

    hz = mel_to_hz(np.linspace(float(hz_to_mel(np.array(0.0))), float(hz_to_mel(np.array(sample_rate / 2.0))), n_mels + 2))
    freqs = np.linspace(0, sample_rate / 2.0, n_fft // 2 + 1)
    lower, center, upper = hz[:-2, None], hz[1:-1, None], hz[2:, None]
    rising = np.maximum(0.0, (freqs - lower) / (center - lower)) * (freqs <= center)
    falling = np.maximum(0.0, (upper - freqs) / (upper - center)) * (freqs > center)
    return ((rising + falling) * (2.0 / (upper - lower))).astype(np.float32)


def log_mel(waveform: torch.Tensor, *, sample_rate: int, hop_length: int, n_mels: int) -> torch.Tensor:
    """Return the log-mel spectrogram of a ``(channels, samples)`` waveform, shape ``(1, channels, frames, n_mels)``.

    A waveform too short for the centred reflect padding (half an FFT window or less) is padded with silence first.
    """
    import torch

    short = _N_FFT // 2 + 1 - waveform.shape[-1]
    if short > 0:
        waveform = torch.nn.functional.pad(waveform, (0, short))
    window = torch.hann_window(_N_FFT, periodic=True, device=waveform.device, dtype=torch.float32)
    spectrum = torch.stft(waveform.float(), _N_FFT, hop_length=hop_length, window=window, center=True, pad_mode="reflect", return_complex=True).abs()
    basis = torch.from_numpy(slaney_mel_basis(sample_rate, _N_FFT, n_mels)).to(waveform.device)
    mel = torch.einsum("mf,cft->ctm", basis, spectrum)
    return torch.log(mel.clamp_min(_LOG_FLOOR)).unsqueeze(0)


def encode_audio_carry(pipeline: Any, waveform: np.ndarray, token_count: int) -> AudioCarry:
    """Encode the carried sound, a ``(2, samples)`` waveform at the audio VAE's sample rate, into ``token_count`` tokens.

    The latent is cut, or padded with zeros when the sound ends early, to the tokens that cover the carried frames.
    """
    import torch

    audio_vae = pipeline.audio_vae
    config = audio_vae.config
    device = torch.device("cuda")
    mel = log_mel(torch.from_numpy(waveform).to(device), sample_rate=config.sample_rate, hop_length=config.mel_hop_length, n_mels=config.mel_bins)
    latent = audio_vae.encode(mel.to(audio_vae.dtype), return_dict=False)[0].mode().float()
    latent = latent[:, :, :token_count]
    if latent.shape[2] < token_count:
        latent = torch.nn.functional.pad(latent, (0, 0, 0, token_count - latent.shape[2]))
    tokens = pipeline._normalize_audio_latents(pipeline._pack_audio_latents(latent), audio_vae.latents_mean, audio_vae.latents_std)  # noqa: SLF001
    return AudioCarry(latent=latent, tokens=tokens)


@contextmanager
def pin_audio_prefix(transformer: Any, carry: AudioCarry) -> Iterator[None]:
    """Make every forward pass of *transformer* see the carried audio tokens clean and at timestep 0.

    The audio timestep becomes per token, as the video's is for its pinned frames; the audio sigma that the
    prompt modulation uses stays per sample.
    """
    count = carry.token_count

    def _hook(_module: Any, args: Any, kwargs: dict[str, Any]) -> Any:
        audio = kwargs.get("audio_hidden_states")
        timestep = kwargs.get("audio_timestep")
        if audio is None or timestep is None:
            return None
        audio = audio.clone()
        audio[:, :count] = carry.tokens.to(device=audio.device, dtype=audio.dtype)
        per_token = timestep.reshape(-1, 1).expand(audio.shape[0], audio.shape[1]).clone() if timestep.ndim < 2 else timestep.clone()
        per_token[:, :count] = 0
        return args, {**kwargs, "audio_hidden_states": audio, "audio_timestep": per_token}

    handle = transformer.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def pin_audio_track(transformer: Any, tokens: torch.Tensor) -> Iterator[None]:
    """Make every forward pass of *transformer* see a whole audio track clean: its tokens, a per-token timestep of 0 and an audio sigma of 0.

    Unlike :func:`pin_audio_prefix` the hold does not depend on what the pipeline passes. ``LTX2Pipeline`` passes no
    ``audio_timestep`` and the others pass one per sample, so the per-token timestep is always set; every pipeline passes
    ``sigma``, which the transformer uses for the audio sigma when none is given, so the audio sigma is set from it.

    Args:
        transformer: The transformer to hook.
        tokens: The track packed and normalized as the denoiser sees it, shape ``(1, tokens, channels)``.

    Raises:
        RuntimeError: From a forward pass whose audio has another token count than *tokens*.
    """
    import torch

    def _hook(_module: Any, args: Any, kwargs: dict[str, Any]) -> Any:
        audio = kwargs.get("audio_hidden_states")
        if audio is None:
            return None
        batch, count = audio.shape[0], audio.shape[1]
        if tokens.shape[1] != count:
            raise RuntimeError(f"The audio track holds {tokens.shape[1]} tokens but the clip's sound has {count}.")
        reference = kwargs.get("audio_timestep")
        if reference is None:
            reference = kwargs["timestep"]
        updated = {
            **kwargs,
            "audio_hidden_states": tokens.to(device=audio.device, dtype=audio.dtype).expand(batch, -1, -1).clone(),
            "audio_timestep": torch.zeros((batch, count), device=audio.device, dtype=reference.dtype),
        }
        sigma = kwargs.get("audio_sigma")
        if sigma is None:
            sigma = kwargs.get("sigma")
        if sigma is not None:
            updated["audio_sigma"] = torch.zeros_like(sigma)
        return args, updated

    handle = transformer.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def restore_audio_prefix(audio_latents: torch.Tensor, carry: AudioCarry) -> None:
    """Put the clean carried tokens back at the start of a result's ``(batch, channels, tokens, mel_bins)`` audio latents, in place."""
    audio_latents[:, :, : carry.token_count] = carry.latent.to(device=audio_latents.device, dtype=audio_latents.dtype)
