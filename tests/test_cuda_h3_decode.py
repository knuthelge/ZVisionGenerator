"""Tests for the H3 chunk-by-chunk video decode and the stereo audio decode, against diffusers' own decode steps."""

from __future__ import annotations

import json
import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")

from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from zvisiongenerator.backends.cuda.video.h3 import decode as decode_module  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3.audio import AUDIO_SAMPLE_RATE, decode_audio, load_audio_vae  # noqa: E402
from zvisiongenerator.backends.cuda.video.h3.decode import (  # noqa: E402
    PIXEL_MEAN,
    PIXEL_STD,
    decode_chunked,
    decode_video,
    denormalise,
    iter_decoded_chunks,
    load_video_vae,
    to_uint8,
)

_VAE_CONFIG = {
    "latent_channels": 4,
    "block_out_channels": (4, 4, 4, 4, 4, 4),
    "layers_per_block": 1,
    "norm_num_groups": 2,
    "decoder_num_layers": 2,
    "decoder_num_attention_heads": 2,
    "decoder_attention_head_dim": 8,
    "decoder_num_register_tokens": 1,
    "decoder_ffn_mult": 2,
    "latents_mean": (0.1, 0.2, -0.1, 0.3),
    "latents_std": (1.0, 2.0, 0.5, 1.5),
}
_AUDIO_CONFIG = {
    "encoder_dim": 4,
    "encoder_rates": (2, 2),
    "latent_dim": 8,
    "latent_channels": 4,
    "num_attention_heads": 2,
    "decoder_dim": 8,
    "decoder_rates": (2, 2),
    "decoder_kernel_sizes": (4, 4),
    "resblock_kernel_sizes": (3,),
    "resblock_dilation_sizes": ((1,),),
    "latents_mean": [0.1, -0.2, 0.3, 0.0],
    "latents_std": [2.0, 1.0, 0.5, 1.5],
}
_PAD_FREE_LENGTHS = (7, 12)  # 5 * n + 2 latent frames fill whole chunks
_PADDED_LENGTHS = (3, 4, 9)  # the rest repeat the last latent frame, and the decode cuts the extra pixel frames again


def _tiny_vae() -> AutoencoderKLMiniMaxH3:
    """Return a seeded tiny video VAE in fp32 whose output spans the whole [0, 1] pixel range (so the clamp matters)."""
    torch.manual_seed(0)
    vae = AutoencoderKLMiniMaxH3(**_VAE_CONFIG)
    with torch.no_grad():
        vae.decoder.proj_out.weight.mul_(40)
        vae.decoder.proj_out.bias.normal_()
    return vae.eval().requires_grad_(False)


def _latents(frames: int, size: int = 2, seed: int = 1) -> torch.Tensor:
    return torch.randn(1, 4, frames, size, size, generator=torch.Generator().manual_seed(seed))


def _decode_step_reference(vae, latents: torch.Tensor) -> np.ndarray:
    """Return the frames diffusers' ``MiniMaxH3VideoDecodeStep`` (decoders.py:188-193) produces, as ``(T, H, W, 3)`` uint8."""
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    video = vae.decode(latents * std + mean, return_dict=False)[0]
    pixel_mean = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    return (video * 255).round().to(torch.uint8)[0].permute(1, 2, 3, 0).numpy()


@pytest.fixture(scope="module")
def vae():
    return _tiny_vae()


@pytest.fixture(scope="module")
def tiled_vae():
    """A tiny VAE whose 64 px frames split into overlapping 32 px tiles."""
    tiny = _tiny_vae()
    tiny.enable_tiling(tile_sample_min_height=32, tile_sample_min_width=32, tile_sample_min_overlap_height=16, tile_sample_min_overlap_width=16)
    return tiny


# --- contract against diffusers' decode step (0.41.0) ------------------------------------------------------------


class TestAgainstTheDiffusersDecodeStep:
    @pytest.mark.parametrize("frames", [*_PAD_FREE_LENGTHS, *_PADDED_LENGTHS])
    def test_the_frames_equal_the_decode_step_bitwise(self, vae, frames):
        latents = _latents(frames)
        with torch.no_grad():
            expected = _decode_step_reference(vae, latents)
        assert np.array_equal(np.stack(decode_video(vae, latents)), expected)

    @pytest.mark.parametrize("frames", [7, 4])
    def test_spatial_tiling_is_followed(self, tiled_vae, frames):
        latents = _latents(frames, size=4)
        with torch.no_grad():
            expected = _decode_step_reference(tiled_vae, latents)
        assert expected.shape[1:3] == (64, 64)
        assert np.array_equal(np.stack(decode_video(tiled_vae, latents)), expected)

    def test_the_test_vae_exercises_the_whole_range_and_the_clamp(self, vae):
        frames = np.stack(decode_video(vae, _latents(7)))
        assert frames.min() == 0 and frames.max() == 255
        assert len(np.unique(frames)) > 100

    def test_denormalise_is_the_decode_steps_latent_maths(self, vae):
        latents = _latents(3)
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        assert torch.equal(denormalise(vae, latents), latents * std + mean)

    def test_to_uint8_reverts_the_imagenet_normalisation_then_clamps_and_rounds(self):
        decoded = torch.zeros(1, 3, 1, 1, 2)
        decoded[0, :, 0, 0, 1] = 100.0  # far above range: clamps to 255
        frame = to_uint8(decoded)
        assert frame.dtype == torch.uint8
        assert frame.shape == (1, 1, 2, 3)  # (T, H, W, 3)
        assert frame[0, 0, 0].tolist() == [round(value * 255) for value in PIXEL_MEAN]  # zero decodes to the pixel mean
        assert frame[0, 0, 1].tolist() == [255, 255, 255]


class TestAgainstDecodeClipAndBlend:
    @pytest.mark.parametrize("frames", [*_PAD_FREE_LENGTHS, *_PADDED_LENGTHS])
    def test_the_chunks_joined_equal_the_vaes_own_decode_before_the_trim(self, vae, frames):
        z = _latents(frames)
        with torch.no_grad():
            joined = torch.cat(list(iter_decoded_chunks(vae, z)), dim=2)
            expected = vae._decode(z)
        assert torch.equal(joined[:, :, : expected.shape[2]], expected)
        assert joined.shape[2] - expected.shape[2] == decode_module._trailing_pad_frames(vae, frames)

    def test_every_chunk_comes_from_decode_clip_and_the_overlap_goes_through_blend(self, vae, monkeypatch):
        clips, blends = [], []
        decode_clip, blend = vae._decode_clip, vae._blend
        monkeypatch.setattr(vae, "_decode_clip", lambda z: clips.append(tuple(z.shape)) or decode_clip(z))
        monkeypatch.setattr(vae, "_blend", lambda a, b, extent, dim: blends.append((tuple(a.shape), tuple(b.shape), extent, dim)) or blend(a, b, extent, dim))
        with torch.no_grad():
            chunks = list(iter_decoded_chunks(vae, _latents(12)))
        # 12 + 3 dropped = 15 tokens = 3 chunks of 5, minus one: two clips of 5 + 2 overlapping latent frames
        assert clips == [(1, 4, 7, 2, 2)] * 2
        assert len(chunks) == 3
        assert blends == [((1, 3, vae.frame_overlap, 32, 32), (1, 3, 20 - vae.frame_pre_padding, 32, 32), vae.frame_overlap, -3)]

    def test_chunks_are_produced_one_at_a_time(self, vae, monkeypatch):
        calls = []
        decode_clip = vae._decode_clip
        monkeypatch.setattr(vae, "_decode_clip", lambda z: calls.append(1) or decode_clip(z))
        with torch.no_grad():
            chunks = iter_decoded_chunks(vae, _latents(12))
            next(chunks)
            assert len(calls) == 1
            list(chunks)
        assert len(calls) == 2


# --- the public decode -------------------------------------------------------------------------------------------


class TestDecodeVideo:
    def test_it_returns_a_list_of_contiguous_height_width_3_uint8_arrays(self, vae):
        frames = decode_video(vae, _latents(9))
        assert isinstance(frames, list)
        assert all(isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.shape == (32, 32, 3) and frame.flags["C_CONTIGUOUS"] for frame in frames)

    @pytest.mark.parametrize(("latent_frames", "pixel_frames"), [(7, 22), (12, 39), (3, 9), (4, 13), (9, 30)])
    def test_it_trims_the_padding(self, vae, latent_frames, pixel_frames):
        assert len(decode_video(vae, _latents(latent_frames))) == pixel_frames

    def test_it_decodes_without_gradients(self, vae, monkeypatch):
        seen = []
        decode_clip = vae._decode_clip
        monkeypatch.setattr(vae, "_decode_clip", lambda z: seen.append(torch.is_grad_enabled()) or decode_clip(z))
        latents = _latents(7).requires_grad_(True)
        assert torch.is_grad_enabled()
        decode_video(vae, latents)
        assert seen == [False] and torch.is_grad_enabled()

    def test_decode_chunked_matches_decode_video(self, vae):
        latents = _latents(4)
        with torch.no_grad():
            chunked = decode_chunked(vae, latents)
        assert np.array_equal(np.stack(chunked), np.stack(decode_video(vae, latents)))

    @pytest.mark.parametrize("shape", [(4, 2, 2), (2, 4, 7, 2, 2)])
    def test_it_refuses_latents_that_are_not_one_clip(self, vae, shape):
        with pytest.raises(ValueError, match="Expected latents"):
            decode_video(vae, torch.zeros(shape))


# --- loading the decoder side ------------------------------------------------------------------------------------


_KEPT_IN_FP32 = ("norm1", "norm2", "norm_out", "scale1", "scale2")


def _write_vae(directory, vae, *, drop_prefixes=("encoder.", "quant_conv."), shards=False):
    """Save *vae*'s config and its decoder-side weights (a file with no encoder tensors) to *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLMiniMaxH3", **{k: list(v) if isinstance(v, tuple) else v for k, v in _VAE_CONFIG.items()}}), encoding="utf-8")
    state = {name: tensor.contiguous() for name, tensor in vae.state_dict().items() if not name.startswith(drop_prefixes)}
    if not shards:
        save_file(state, str(directory / "diffusion_pytorch_model.safetensors"))
        return
    names = sorted(state)
    halves = {"model-00001-of-00002.safetensors": names[: len(names) // 2], "model-00002-of-00002.safetensors": names[len(names) // 2 :]}
    for file, part in halves.items():
        save_file({name: state[name] for name in part}, str(directory / file))
    weight_map = {name: file for file, part in halves.items() for name in part}
    (directory / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")


class TestLoadVideoVae:
    @pytest.mark.parametrize("shards", [False, True])
    def test_it_loads_the_decoder_side_without_the_encoder_weights(self, vae, tmp_path, shards):
        _write_vae(tmp_path, vae, shards=shards)
        loaded = load_video_vae(tmp_path, torch.device("cpu"))
        assert isinstance(loaded.encoder, torch.nn.Identity) and isinstance(loaded.quant_conv, torch.nn.Identity)
        assert not loaded.training
        assert not any(parameter.requires_grad for parameter in loaded.parameters())
        assert torch.equal(loaded.decoder.proj_out.weight, vae.decoder.proj_out.weight.to(torch.bfloat16))

    def test_the_fp32_keep_list_is_matched_by_name_part_and_everything_else_is_bf16(self, vae, tmp_path):
        _write_vae(tmp_path, vae)
        loaded = load_video_vae(tmp_path, torch.device("cpu"))
        kept = {name for name, _ in loaded.named_parameters() if any(part in _KEPT_IN_FP32 for part in name.split("."))}
        assert any(name.endswith(".scale1") for name in kept)  # bare parameters count, not only the modules
        assert any(name.endswith(".norm1.weight") for name in kept) and any(name.startswith("decoder.norm_out.") for name in kept)
        for name, parameter in loaded.named_parameters():
            assert parameter.dtype == (torch.float32 if name in kept else torch.bfloat16), name
        assert torch.equal(loaded.decoder.transformer_blocks[0].scale1, vae.decoder.transformer_blocks[0].scale1)  # fp32 values are not rounded

    def test_the_rotary_frequencies_stay_exactly_the_constructors(self, vae, tmp_path):
        _write_vae(tmp_path, vae)
        loaded = load_video_vae(tmp_path, torch.device("cpu"))
        assert loaded.decoder.rope.inv_freq.dtype == torch.float32
        assert torch.equal(loaded.decoder.rope.inv_freq, vae.decoder.rope.inv_freq)

    def test_the_vae_moves_to_the_device_when_post_quant_conv_first_runs(self, vae, tmp_path):
        _write_vae(tmp_path, vae)
        loaded = load_video_vae(tmp_path, torch.device("cpu"))
        assert len(loaded.post_quant_conv._forward_pre_hooks) == 1
        assert len(loaded.decoder._forward_pre_hooks) == 0

    def test_it_asks_to_load_the_vae_when_used_on_the_given_device(self, vae, tmp_path, monkeypatch):
        _write_vae(tmp_path, vae)
        calls = []
        monkeypatch.setattr("zvisiongenerator.backends.cuda.block_streaming.load_when_used", lambda component, entry_points, device: calls.append((component, entry_points, device)) or [])
        loaded = load_video_vae(tmp_path, "the-device")
        assert calls == [(loaded, [loaded.post_quant_conv], "the-device")]

    def test_a_decoder_missing_tensors_is_refused(self, vae, tmp_path):
        _write_vae(tmp_path, vae, drop_prefixes=("encoder.", "quant_conv.", "decoder.proj_out."))
        with pytest.raises(RuntimeError, match="do not cover the model"):
            load_video_vae(tmp_path, torch.device("cpu"))

    def test_the_loaded_vae_decodes_close_to_the_fp32_one(self, vae, tmp_path):
        _write_vae(tmp_path, vae)
        loaded = load_video_vae(tmp_path, torch.device("cpu"))
        latents = _latents(7)
        reference = np.stack(decode_video(vae, latents)).astype(np.int16)
        frames = decode_video(loaded, latents)
        assert len(frames) == 22
        assert float(np.abs(np.stack(frames).astype(np.int16) - reference).mean()) < 40  # bf16 weights: same picture, not the same bits


# --- audio -------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def audio_vae():
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch's weight_norm deprecation, raised inside diffusers
        return AutoencoderKLMiniMaxH3Audio(**_AUDIO_CONFIG).eval().requires_grad_(False)


def _audio_latents(frames: int = 5) -> torch.Tensor:
    return torch.randn(2, 4, frames, generator=torch.Generator().manual_seed(2))


class TestDecodeAudio:
    def test_the_sample_rate_is_the_audio_vaes(self, audio_vae):
        assert AUDIO_SAMPLE_RATE == 32000 == audio_vae.config.sampling_rate

    def test_it_returns_a_stereo_float32_waveform_channels_first(self, audio_vae):
        waveform = decode_audio(audio_vae, _audio_latents(5))
        assert waveform.dtype == torch.float32
        assert waveform.shape == (2, 5 * audio_vae.hop_length)
        assert waveform.device.type == "cpu"
        assert float(waveform.abs().max()) > 0

    def test_it_matches_the_audio_decode_step_of_diffusers(self, audio_vae):
        latents = _audio_latents()
        mean = torch.tensor(audio_vae.config.latents_mean).view(1, -1, 1)
        std = torch.tensor(audio_vae.config.latents_std).view(1, -1, 1)
        with torch.no_grad():
            expected = audio_vae.decode(latents * std + mean, return_dict=False)[0].float().permute(1, 0, 2)  # decoders.py: (1, 2, samples)
        assert expected.shape[0] == 1
        assert torch.equal(decode_audio(audio_vae, latents), expected[0])

    def test_the_two_channels_are_decoded_independently(self, audio_vae):
        latents = _audio_latents()
        both = decode_audio(audio_vae, latents)
        first = decode_audio(audio_vae, torch.cat([latents[:1], latents[:1]]))
        assert torch.allclose(both[0], first[0], atol=1e-6)
        assert torch.equal(first[0], first[1])

    def test_it_decodes_without_gradients(self, audio_vae):
        latents = _audio_latents().requires_grad_(True)
        assert not decode_audio(audio_vae, latents).requires_grad

    @pytest.mark.parametrize("shape", [(4, 5), (1, 4, 5), (3, 4, 5)])
    def test_it_refuses_latents_that_are_not_two_channels(self, audio_vae, shape):
        with pytest.raises(ValueError, match="Expected audio latents"):
            decode_audio(audio_vae, torch.zeros(shape))


class TestLoadAudioVae:
    def test_it_loads_the_vae_in_eval_mode_with_the_decoder_in_fp32(self, audio_vae, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio_vae.save_pretrained(str(tmp_path))
            loaded = load_audio_vae(tmp_path, torch.device("cpu"))
        assert not loaded.training
        assert not any(parameter.requires_grad for parameter in loaded.parameters())
        assert {parameter.dtype for parameter in loaded.parameters()} == {torch.float32}
        latents = _audio_latents()
        assert torch.allclose(decode_audio(loaded, latents), decode_audio(audio_vae, latents), atol=1e-5)


# --- real encode -------------------------------------------------------------------------------------------------


def test_the_decoded_frames_and_audio_encode_to_an_mp4(vae, audio_vae, tmp_path):
    av = pytest.importorskip("av")
    from zvisiongenerator.backends.cuda.video.encode import encode_video

    frames = decode_video(vae, _latents(7))
    waveform = decode_audio(audio_vae, _audio_latents(40))
    output = tmp_path / "clip.mp4"
    encode_video(frames, 24, str(output), audio=waveform, audio_sample_rate=AUDIO_SAMPLE_RATE)
    with av.open(str(output)) as container:
        video = container.streams.video[0]
        audio = container.streams.audio[0]
        assert (video.width, video.height) == (32, 32)
        assert sum(1 for _ in container.decode(video=0)) == len(frames) == 22
        assert audio.rate == AUDIO_SAMPLE_RATE and audio.codec_context.channels == 2
