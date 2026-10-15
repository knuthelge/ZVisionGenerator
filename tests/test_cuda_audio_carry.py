"""Tests for pinning a clip's last sound at the start of a diffusers LTX-2 extension window."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from zvisiongenerator.backends.cuda.video.audio_carry import AudioCarry, encode_audio_carry, log_mel, pin_audio_prefix, restore_audio_prefix, slaney_mel_basis


def _reference_log_mel(waveform: np.ndarray) -> np.ndarray:
    """The LTX audio processor's recipe in numpy: centred reflect padding, periodic Hann, |rfft|, Slaney mel, log."""
    window = np.hanning(1025)[:-1]
    basis = slaney_mel_basis(16000, 1024, 64)
    channels = []
    for channel in waveform:
        padded = np.concatenate([channel[1:513][::-1], channel, channel[-513:-1][::-1]])
        count = (len(padded) - 1024) // 160 + 1
        frames = np.stack([padded[i * 160 : i * 160 + 1024] for i in range(count)]) * window
        channels.append(np.log(np.maximum(np.abs(np.fft.rfft(frames)) @ basis.T, 1e-5)))
    return np.stack(channels)[None]


class TestLogMel:
    def test_matches_the_ltx_audio_processor_recipe(self):
        waveform = np.random.default_rng(0).standard_normal((2, 8000)).astype(np.float32) * 0.1
        got = log_mel(torch.from_numpy(waveform), sample_rate=16000, hop_length=160, n_mels=64).numpy()
        assert got.shape == (1, 2, 51, 64)
        np.testing.assert_allclose(got, _reference_log_mel(waveform), atol=1e-4)

    def test_slaney_filters_have_unit_area_per_mel_and_cover_the_spectrum(self):
        basis = slaney_mel_basis(16000, 1024, 64)
        assert basis.shape == (64, 513)
        assert (basis >= 0).all() and (basis.sum(axis=0)[1:-1] > 0).all()


class _FakeAudioVae:
    """Encoder whose posterior mean is ``tokens`` latent frames of ones, like AutoencoderKLLTX2Audio."""

    dtype = torch.float32
    config = SimpleNamespace(sample_rate=16000, mel_hop_length=160, mel_bins=64)
    latents_mean = torch.full((128,), 0.5)
    latents_std = torch.full((128,), 2.0)

    def __init__(self, tokens: int):
        self.tokens = tokens
        self.mels: list[torch.Tensor] = []

    def encode(self, mel, return_dict=True):
        self.mels.append(mel)
        return (SimpleNamespace(mode=lambda: torch.ones((1, 8, self.tokens, 16))),)


class _FakePipeline:
    def __init__(self, tokens: int):
        self.audio_vae = _FakeAudioVae(tokens)

    @staticmethod
    def _pack_audio_latents(latents):
        return latents.transpose(1, 2).flatten(2, 3)

    @staticmethod
    def _normalize_audio_latents(latents, mean, std):
        return (latents - mean) / std


@pytest.fixture
def cpu_device(monkeypatch):
    real_device = torch.device
    monkeypatch.setattr(torch, "device", lambda name: real_device("cpu") if name == "cuda" else real_device(name))


class TestEncodeAudioCarry:
    @pytest.mark.parametrize(("encoded", "wanted", "ones"), [(30, 26, 26), (20, 26, 20), (26, 26, 26)])
    def test_the_latent_is_cut_or_zero_padded_to_the_carried_tokens(self, cpu_device, encoded, wanted, ones):
        pipeline = _FakePipeline(encoded)
        carry = encode_audio_carry(pipeline, np.zeros((2, 16000), dtype=np.float32), wanted)
        assert carry.token_count == wanted
        assert carry.latent.shape == (1, 8, wanted, 16)
        assert carry.latent[:, :, :ones].eq(1).all() and carry.latent[:, :, ones:].eq(0).all()

    def test_the_pinned_tokens_are_packed_and_normalized_as_the_denoiser_sees_them(self, cpu_device):
        carry = encode_audio_carry(_FakePipeline(26), np.zeros((2, 16000), dtype=np.float32), 26)
        assert carry.tokens.shape == (1, 26, 128)
        assert torch.allclose(carry.tokens, torch.full((1, 26, 128), 0.25))

    def test_the_encoder_gets_a_stereo_log_mel_of_the_waveform(self, cpu_device):
        pipeline = _FakePipeline(26)
        encode_audio_carry(pipeline, np.zeros((2, 16000), dtype=np.float32), 26)
        (mel,) = pipeline.audio_vae.mels
        assert mel.shape == (1, 2, 101, 64)


class _Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, *, audio_hidden_states, audio_timestep, audio_sigma, **_kwargs):
        self.calls.append({"audio": audio_hidden_states, "timestep": audio_timestep, "sigma": audio_sigma})


def _carry(count: int = 2) -> AudioCarry:
    return AudioCarry(latent=torch.full((1, 8, count, 16), 7.0), tokens=torch.full((1, count, 128), 3.0))


class TestPinAudioPrefix:
    def test_the_transformer_sees_the_carried_tokens_clean_at_timestep_zero(self):
        transformer = _Recorder()
        audio = torch.zeros((1, 5, 128))
        timestep = torch.tensor([900.0])
        with pin_audio_prefix(transformer, _carry()):
            transformer(audio_hidden_states=audio, audio_timestep=timestep, audio_sigma=timestep)
        (call,) = transformer.calls
        assert call["audio"][:, :2].eq(3).all() and call["audio"][:, 2:].eq(0).all()
        assert call["timestep"].tolist() == [[0.0, 0.0, 900.0, 900.0, 900.0]]
        assert call["sigma"].tolist() == [900.0]  # the prompt modulation keeps one sigma per sample
        assert audio.eq(0).all()  # the pipeline's own state is not changed

    def test_the_hook_is_removed_on_exit_even_after_an_error(self):
        transformer = _Recorder()
        with pytest.raises(RuntimeError), pin_audio_prefix(transformer, _carry()):
            raise RuntimeError("boom")
        timestep = torch.tensor([900.0])
        transformer(audio_hidden_states=torch.zeros((1, 5, 128)), audio_timestep=timestep, audio_sigma=timestep)
        assert transformer.calls[0]["timestep"] is timestep

    def test_restore_puts_the_clean_latent_back_at_the_start(self):
        audio_latents = torch.zeros((1, 8, 5, 16))
        restore_audio_prefix(audio_latents, _carry())
        assert audio_latents[:, :, :2].eq(7).all() and audio_latents[:, :, 2:].eq(0).all()
