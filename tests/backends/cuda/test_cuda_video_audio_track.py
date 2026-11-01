"""Tests for holding a user's audio track clean through the diffusers LTX backend: the transformer hook, the encode order, both passes and Extend."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests._factories import make_extend_source
from tests.backends.cuda._helpers import (
    _FakeImagePipeline,
    _FakeTextPipeline,
    _load_upscale_model,
    _make_runtime,
    _patch_runtime,
    _text_to_video_args,
    install_fake_weights,
)
from zvisiongenerator.backends.cuda.video import backend as backend_module
from zvisiongenerator.backends.cuda.video.audio_carry import pin_audio_track
from zvisiongenerator.backends.cuda.video.backend import DiffusersVideoBackend
from zvisiongenerator.core.video_types import AudioTrack

_TRACK = AudioTrack("/audio/voice.wav", 1.6)
_TOKENS = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)


@pytest.fixture(autouse=True)
def fake_weights(monkeypatch):
    """Fake the weights, placement and memory checks of ``load_model`` (see ``_cuda_helpers.install_fake_weights``)."""
    return install_fake_weights(monkeypatch)


class _Transformer(torch.nn.Module):
    """Records the keyword arguments of each forward pass, with the defaults of diffusers' ``LTX2VideoTransformer3DModel``."""

    def __init__(self):
        super().__init__()
        self.passes: list[dict[str, object]] = []

    def forward(self, hidden_states, audio_hidden_states=None, timestep=None, sigma=None, audio_timestep=None, audio_sigma=None):
        self.passes.append(dict(hidden_states=hidden_states, audio_hidden_states=audio_hidden_states, timestep=timestep, sigma=sigma, audio_timestep=audio_timestep, audio_sigma=audio_sigma))
        return hidden_states


def _noisy_audio(batch: int = 1, count: int = 3) -> torch.Tensor:
    return torch.full((batch, count, 2), 7.0)


class TestPinAudioTrack:
    """The hook always sets the clean tokens, a per-token audio timestep of 0 and an audio sigma of 0, whatever the pipeline passed."""

    @staticmethod
    def _call(call_shape: str, *, batch: int = 1, count: int = 3) -> tuple[_Transformer, dict[str, object]]:
        transformer = _Transformer()
        timestep = torch.full((batch,), 0.7)
        video = torch.ones(batch, 4, 2)
        shapes = {
            # LTX2Pipeline (text to video): no audio_timestep, no audio_sigma
            "text": dict(timestep=timestep, sigma=timestep),
            # LTX2ImageToVideoPipeline: the audio timestep is the video's
            "image": dict(timestep=timestep, sigma=timestep, audio_timestep=timestep),
            # LTX2ConditionPipeline: a separate audio timestep and audio sigma
            "condition": dict(timestep=timestep, sigma=timestep, audio_timestep=torch.full((batch,), 0.4), audio_sigma=torch.full((batch,), 0.4)),
        }
        with pin_audio_track(transformer, _TOKENS):
            transformer(hidden_states=video, audio_hidden_states=_noisy_audio(batch, count), **shapes[call_shape])
        (seen,) = transformer.passes
        return transformer, seen

    @pytest.mark.parametrize("call_shape", ["text", "image", "condition"])
    def test_the_audio_is_the_clean_tokens_at_timestep_and_sigma_zero(self, call_shape):
        _, seen = self._call(call_shape)
        assert torch.equal(seen["audio_hidden_states"], _TOKENS)
        assert seen["audio_timestep"].shape == (1, 3) and not seen["audio_timestep"].any()
        assert seen["audio_sigma"].shape == (1,) and not seen["audio_sigma"].any()

    def test_every_sample_of_a_batch_gets_the_tokens(self):
        _, seen = self._call("text", batch=2)
        assert seen["audio_hidden_states"].shape == (2, 3, 2)
        assert torch.equal(seen["audio_hidden_states"][1], _TOKENS[0])
        assert seen["audio_timestep"].shape == (2, 3)

    def test_the_tokens_follow_the_audio_dtype_and_device_of_the_pass(self):
        transformer = _Transformer()
        timestep = torch.full((1,), 0.5, dtype=torch.bfloat16)
        with pin_audio_track(transformer, _TOKENS):
            transformer(hidden_states=torch.ones(1, 4, 2), audio_hidden_states=_noisy_audio().to(torch.bfloat16), timestep=timestep, sigma=timestep)
        seen = transformer.passes[0]
        assert seen["audio_hidden_states"].dtype == torch.bfloat16
        assert seen["audio_timestep"].dtype == torch.bfloat16

    def test_a_pass_without_audio_is_left_alone(self):
        transformer = _Transformer()
        with pin_audio_track(transformer, _TOKENS):
            transformer(hidden_states=torch.ones(1, 4, 2), timestep=torch.zeros(1), sigma=torch.zeros(1))
        assert transformer.passes[0]["audio_hidden_states"] is None and transformer.passes[0]["audio_timestep"] is None

    def test_a_pass_with_another_token_count_fails(self):
        with pytest.raises(RuntimeError, match="3 tokens.*has 5"):
            self._call("text", count=5)

    def test_the_hook_is_removed_when_the_pass_fails(self):
        transformer = _Transformer()
        with pytest.raises(RuntimeError), pin_audio_track(transformer, _TOKENS):
            raise RuntimeError("out of memory")
        assert not transformer._forward_pre_hooks


class _TrackPipelineMixin:
    """The audio parts of a diffusers LTX-2 pipeline that the track path reads: the audio VAE's rate and the token rate."""

    audio_sampling_rate = 16000
    audio_hop_length = 160
    audio_vae_temporal_compression_ratio = 4

    def _init_audio(self):
        self.audio_vae = SimpleNamespace(config=SimpleNamespace(sample_rate=16000))


class _TrackTextPipeline(_TrackPipelineMixin, _FakeTextPipeline):
    def __init__(self):
        super().__init__()
        self._init_audio()


class _TrackImagePipeline(_TrackPipelineMixin, _FakeImagePipeline):
    def __init__(self):
        super().__init__()
        self._init_audio()


@pytest.fixture
def held(monkeypatch, fake_weights):
    """Replace the track reader, the encoder and the hold; record what each saw and when, in the weights' event list."""
    calls = SimpleNamespace(reads=[], windows=[], encoded=[], holds=[], events=fake_weights.events, waveform=np.ones((2, 100), dtype=np.float32), tokens=_TOKENS)

    def _read(track, rate, seconds):
        calls.reads.append((track.path, rate, track.start, seconds))
        return calls.waveform

    def _window(source, track, *, carry_frames, extend_frames, rate):
        calls.windows.append((source, track, carry_frames, extend_frames, rate))
        return calls.waveform

    def _encode(pipeline, waveform, token_count):
        calls.encoded.append((pipeline, waveform, token_count))
        calls.events.append("encode track")
        return SimpleNamespace(tokens=calls.tokens)

    @contextmanager
    def _pin(transformer, tokens):
        entered = len(transformer_owner[0].calls) if transformer_owner else None
        calls.holds.append(SimpleNamespace(transformer=transformer, tokens=tokens, calls_on_enter=entered))
        calls.events.append("hold")
        yield
        calls.holds[-1].calls_on_exit = len(transformer_owner[0].calls) if transformer_owner else None
        calls.events.append("release")

    transformer_owner: list = []
    calls.owner = transformer_owner
    monkeypatch.setattr(backend_module, "read_track", _read)
    monkeypatch.setattr(backend_module, "window_waveform", _window)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.audio_carry.encode_audio_carry", _encode)
    monkeypatch.setattr("zvisiongenerator.backends.cuda.video.audio_carry.pin_audio_track", _pin)
    return calls


def _load(monkeypatch, *, mode="t2v", upscale=False, low_memory=True, runtime=None):
    runtime = runtime or _make_runtime(text_pipeline=_TrackTextPipeline, image_pipeline=_TrackImagePipeline)
    if upscale:
        return _load_upscale_model(monkeypatch, low_memory=low_memory, mode=mode, runtime=runtime)
    _patch_runtime(monkeypatch, runtime)
    return DiffusersVideoBackend().load_model("diffusers/LTX-2.3-Distilled-Diffusers", quantize=4, mode=mode, low_memory=low_memory)[0]


class TestTextToVideoWithATrack:
    def test_reads_the_window_at_the_audio_vae_rate_from_start_at(self, monkeypatch, tmp_path, held):
        model = _load(monkeypatch)
        DiffusersVideoBackend().text_to_video(model=model, audio_track=_TRACK, **_text_to_video_args(tmp_path))
        assert held.reads == [("/audio/voice.wav", 16000, 1.6, 49 / 24)]
        ((pipeline, waveform, token_count),) = held.encoded
        assert pipeline is model.text_to_video and waveform is held.waveform
        assert token_count == round(49 / 24 * 25)

    def test_holds_the_tokens_during_the_one_stage_pass(self, monkeypatch, tmp_path, held):
        model = _load(monkeypatch)
        held.owner.append(model.text_to_video)
        DiffusersVideoBackend().text_to_video(model=model, audio_track=_TRACK, **_text_to_video_args(tmp_path))
        (hold,) = held.holds
        assert hold.tokens is held.tokens
        assert (hold.calls_on_enter, hold.calls_on_exit) == (0, 1)

    def test_a_clip_without_a_track_reads_nothing_and_holds_nothing(self, monkeypatch, tmp_path, held):
        model = _load(monkeypatch)
        DiffusersVideoBackend().text_to_video(model=model, **_text_to_video_args(tmp_path))
        assert held.reads == held.encoded == held.holds == []

    def test_the_track_is_encoded_before_the_transformer_is_attached(self, monkeypatch, tmp_path, held, fake_weights):
        fake_weights.fits_together = False  # low memory: the transformer loads after the prompt is encoded
        model = _load(monkeypatch)
        assert "load transformer" not in held.events
        DiffusersVideoBackend().text_to_video(model=model, audio_track=_TRACK, **_text_to_video_args(tmp_path))
        assert held.events.index("encode track") < held.events.index("load transformer") < held.events.index("hold")

    def test_both_passes_of_the_two_stage_upscale_are_held(self, monkeypatch, tmp_path, held):
        model = _load(monkeypatch, upscale=True, low_memory=False)
        held.owner.append(model.text_to_video)
        DiffusersVideoBackend().text_to_video(model=model, audio_track=_TRACK, **_text_to_video_args(tmp_path, stage1_steps=4))
        assert len(model.text_to_video.calls) == 2
        assert [(hold.calls_on_enter, hold.calls_on_exit) for hold in held.holds] == [(0, 1), (1, 2)]
        assert all(hold.tokens is held.tokens for hold in held.holds)
        assert len(held.encoded) == 1  # encoded once for both passes


class TestImageToVideoWithATrack:
    @staticmethod
    def _run(model, tmp_path, **overrides):
        from PIL import Image

        image = tmp_path / "first.png"
        Image.new("RGB", (64, 64)).save(image)
        arguments = {"image_path": str(image), "prompt": "a city", "width": 512, "height": 384, "num_frames": 49, "seed": 7, "steps": 8, "output_path": str(tmp_path / "out.mp4"), **overrides}
        return DiffusersVideoBackend().image_to_video(model=model, **arguments)

    def test_holds_both_passes_of_the_upscale_with_one_encode(self, monkeypatch, tmp_path, held):
        model = _load(monkeypatch, mode="i2v", upscale=True, low_memory=False)
        held.owner.append(model.image_to_video)
        self._run(model, tmp_path, audio_track=_TRACK, stage1_steps=4)
        assert held.reads == [("/audio/voice.wav", 16000, 1.6, 49 / 24)]
        assert [(hold.calls_on_enter, hold.calls_on_exit) for hold in held.holds] == [(0, 1), (1, 2)]
        assert len(held.encoded) == 1


class TestExtendWithATrack:
    @pytest.fixture
    def segments(self, monkeypatch):
        encoded = []
        monkeypatch.setattr("zvisiongenerator.utils.video_tail.read_source_tail", lambda source: np.zeros((25, 256, 512, 3), dtype=np.uint8))
        monkeypatch.setattr("zvisiongenerator.utils.video_tail.read_source_tail_audio", lambda source, rate: pytest.fail("the tail's sound is part of the window waveform"))
        monkeypatch.setattr("zvisiongenerator.utils.video_splice.write_segment_frames", lambda frames, command: encoded.append(list(frames)))
        return encoded

    @staticmethod
    def _extend(model, tmp_path, source=None, **overrides):
        kwargs = {"extend_frames": 16, "seed": 7, "steps": 8, "segment_dir": tmp_path, "audio_output": False, "audio_track": _TRACK, **overrides}
        return DiffusersVideoBackend().extend_video(model, source or make_extend_source(), "She closes the umbrella.", **kwargs)

    def test_holds_the_whole_window_not_the_prefix(self, monkeypatch, tmp_path, held, segments):
        model = _load(monkeypatch, mode="extend", runtime=_make_runtime())
        held.owner.append(model.extend)
        source = make_extend_source()
        self._extend(model, tmp_path, source)
        assert held.windows == [(source, _TRACK, 25, 16, 16000)]
        ((pipeline, waveform, token_count),) = held.encoded
        assert pipeline is model.extend and waveform is held.waveform
        assert token_count == round(41 / 24 * 25)  # the 25 carried frames and the 16 new ones
        (hold,) = held.holds
        assert hold.tokens is held.tokens and (hold.calls_on_enter, hold.calls_on_exit) == (0, 1)

    @pytest.mark.parametrize("has_audio", [True, False])
    def test_writes_no_segment_wav(self, monkeypatch, tmp_path, held, segments, has_audio):
        model = _load(monkeypatch, mode="extend", runtime=_make_runtime())
        segment = self._extend(model, tmp_path, make_extend_source(has_audio=has_audio))
        assert segment.audio_path is None
        assert not (tmp_path / "segment.wav").exists()
        assert len(segments) == 1  # the new frames are written

    def test_the_window_is_encoded_before_the_transformer_is_attached(self, monkeypatch, tmp_path, held, segments, fake_weights):
        fake_weights.fits_together = False
        model = _load(monkeypatch, mode="extend", runtime=_make_runtime())
        self._extend(model, tmp_path)
        assert held.events.index("encode track") < held.events.index("load transformer") < held.events.index("hold")
