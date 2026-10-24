"""Tests for the H3 conditioning encoders, against diffusers' own keyframe, reference and prepare-latents steps."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")
PIL_Image = pytest.importorskip("PIL.Image")

from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Scheduler  # noqa: E402
from diffusers.modular_pipelines import PipelineState  # noqa: E402
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3VideoReference  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.encoders import (  # noqa: E402
    MiniMaxH3KeyframeVaeEncoderStep,
    MiniMaxH3Ref2VAReferenceEncoderStep,
)
from safetensors.torch import save_file  # noqa: E402

from zvisiongenerator.backends.cuda.video.h3.conditioning import (  # noqa: E402
    KEYFRAME_ENCODE_SEED,
    TAIL_AUDIO_LATENTS,
    TAIL_FRAMES,
    TAIL_LATENT_FRAMES,
    encode_keyframe,
    encode_tail,
    encode_tail_audio,
    fit_keyframe,
    load_vae_encoder,
    prenoise,
    tail_audio_rows,
    tail_first_audio_order,
)
from zvisiongenerator.backends.cuda.video.h3.schedule import AUDIO_COND_T, KEYFRAME_T, row_timesteps  # noqa: E402

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
_HOP = 4  # samples per audio latent of the tiny audio VAE (the real one has 800)
_CANVAS = (64, 32)  # width, height: 4 x 2 latent cells


def _tiny_vae() -> AutoencoderKLMiniMaxH3:
    torch.manual_seed(0)
    return AutoencoderKLMiniMaxH3(**_VAE_CONFIG).eval().requires_grad_(False)


@pytest.fixture(scope="module")
def vae():
    return _tiny_vae()


@pytest.fixture(scope="module")
def audio_vae():
    torch.manual_seed(0)
    return AutoencoderKLMiniMaxH3Audio(**_AUDIO_CONFIG).eval().requires_grad_(False)


def _picture(width: int, height: int, seed: int = 0) -> PIL_Image.Image:
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return PIL_Image.fromarray(pixels)


def _frames(count: int = TAIL_FRAMES, seed: int = 1) -> np.ndarray:
    width, height = _CANVAS
    return np.random.default_rng(seed).integers(0, 256, (count, height, width, 3), dtype=np.uint8)


def _components(vae=None, audio_vae=None) -> SimpleNamespace:
    """Return the slice of ``MiniMaxH3ModularPipeline`` the diffusers encoder steps read."""
    return SimpleNamespace(
        vae=vae,
        audio_vae=audio_vae,
        _execution_device=torch.device("cpu"),
        pixel_mean=(0.485, 0.456, 0.406),
        pixel_std=(0.229, 0.224, 0.225),
        keyframe_encode_seed=42,
        vae_frames_per_chunk=17,
        vae_latents_per_chunk=5,
        audio_latent_channels=4,
    )


def _run_step(step, components, **inputs):
    state = PipelineState()
    for name, value in inputs.items():
        state.set(name, value)
    with torch.no_grad():
        _, state = step(components, state)
    return state


# --- keyframes ---------------------------------------------------------------------------------------------------


class TestEncodeKeyframe:
    def test_it_equals_the_keyframe_step_of_diffusers_bitwise(self, vae):
        image = _picture(*_CANVAS)
        state = _run_step(MiniMaxH3KeyframeVaeEncoderStep(), _components(vae=vae), keyframes=[image])
        expected = state.get("condition_latents")[0]
        latents = encode_keyframe(vae, image, *_CANVAS)
        assert latents.shape == (1, 4, 1, 2, 4)
        assert torch.equal(latents, expected)

    def test_the_seed_is_42_and_not_the_global_or_request_state(self, vae):
        assert KEYFRAME_ENCODE_SEED == 42
        image = _picture(*_CANVAS)
        torch.manual_seed(1)
        first = encode_keyframe(vae, image, *_CANVAS)
        torch.manual_seed(2)
        assert torch.equal(encode_keyframe(vae, image, *_CANVAS), first)

    def test_the_sample_is_rounded_to_fp16_before_it_is_normalised(self, vae):
        latents = encode_keyframe(vae, _picture(*_CANVAS), *_CANVAS)
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        raw = latents * std + mean
        assert torch.allclose(raw, raw.half().float(), rtol=0, atol=1e-3)  # on the fp16 grid, up to the float32 arithmetic of the normalisation

    def test_it_returns_float32_on_the_cpu(self, vae):
        latents = encode_keyframe(vae, _picture(*_CANVAS), *_CANVAS)
        assert latents.dtype == torch.float32 and latents.device.type == "cpu" and not latents.requires_grad

    def test_it_fits_an_image_of_another_aspect_before_encoding(self, vae):
        wide = _picture(200, 50)
        state = _run_step(MiniMaxH3KeyframeVaeEncoderStep(), _components(vae=vae), keyframes=[fit_keyframe(wide, *_CANVAS)])
        assert torch.equal(encode_keyframe(vae, wide, *_CANVAS), state.get("condition_latents")[0])

    @pytest.mark.parametrize("canvas", [(65, 32), (64, 0), (0, 32), (64, 40)])
    def test_it_refuses_a_canvas_that_is_not_a_multiple_of_the_compression(self, vae, canvas):
        with pytest.raises(ValueError, match="multiple of 16"):
            encode_keyframe(vae, _picture(64, 32), *canvas)


class TestFitKeyframe:
    def test_it_matches_the_ltx_centre_crop_fit(self):
        from zvisiongenerator.backends.cuda.video.backend import _fit_image

        for size in ((200, 50), (50, 200), (64, 32), (31, 99)):
            image = _picture(*size)
            assert fit_keyframe(image, *_CANVAS).tobytes() == _fit_image(image, *_CANVAS).tobytes()

    def test_it_crops_the_overflow_equally_from_both_sides(self):
        stripes = np.zeros((10, 30, 3), dtype=np.uint8)
        stripes[:, 10:20] = 255  # a white centre between two black bands
        fitted = np.asarray(fit_keyframe(PIL_Image.fromarray(stripes), 10, 10))
        assert fitted.shape == (10, 10, 3) and fitted.min() == fitted.max() == 255

    def test_an_image_at_the_target_size_is_unchanged(self):
        image = _picture(*_CANVAS)
        assert fit_keyframe(image, *_CANVAS).tobytes() == image.tobytes()

    def test_it_converts_to_rgb(self):
        assert fit_keyframe(_picture(10, 10).convert("L"), 8, 8).mode == "RGB"
        assert fit_keyframe(_picture(10, 10).convert("RGBA"), 8, 8).mode == "RGB"


# --- the extension tail ------------------------------------------------------------------------------------------


class TestEncodeTail:
    def test_it_equals_the_video_reference_encode_of_diffusers_bitwise(self, vae, audio_vae):
        frames = _frames()
        reference = MiniMaxH3VideoReference(frames=frames, fps=24.0)
        state = _run_step(MiniMaxH3Ref2VAReferenceEncoderStep(), _components(vae=vae, audio_vae=audio_vae), normalized_references=[reference])
        expected = state.get("condition_latents")[0]
        latents = encode_tail(vae, frames)
        assert latents.shape == (1, 4, TAIL_LATENT_FRAMES, 2, 4)
        assert torch.equal(latents, expected)

    def test_a_tail_is_22_frames_and_7_latents(self):
        assert (TAIL_FRAMES, TAIL_LATENT_FRAMES, TAIL_AUDIO_LATENTS) == (22, 7, 37)

    @pytest.mark.parametrize("count", [1, 21, 23, 39])
    def test_it_refuses_a_tail_that_is_not_22_frames(self, vae, count):
        with pytest.raises(ValueError, match="22 uint8 RGB frames"):
            encode_tail(vae, _frames(count))

    def test_it_refuses_float_frames(self, vae):
        with pytest.raises(ValueError, match="uint8"):
            encode_tail(vae, _frames().astype(np.float32))

    def test_it_refuses_a_canvas_that_is_not_a_multiple_of_the_compression(self, vae):
        with pytest.raises(ValueError, match="multiple of 16"):
            encode_tail(vae, _frames()[:, :, :60])


# --- the tail audio ----------------------------------------------------------------------------------------------


def _oracle_audio(audio_vae, waveform: torch.Tensor) -> torch.Tensor:
    """Return the soundtrack rows ``(2 x frames, channels)`` diffusers' reference encoder emits (encoders.py:759)."""
    reference = MiniMaxH3AudioReference(audio=waveform, sample_rate=32000)
    state = _run_step(MiniMaxH3Ref2VAReferenceEncoderStep(), _components(audio_vae=audio_vae), normalized_references=[reference])
    return state.get("audio_condition_latents")[0]


def _trim_or_pad(rows: torch.Tensor, frames: int) -> torch.Tensor:
    """Cut each channel of channel-major *rows* to *frames* latents, repeating the last one when short."""
    per_channel = rows.reshape(2, -1, rows.shape[-1])
    if per_channel.shape[1] < frames:
        per_channel = torch.cat([per_channel, per_channel[:, -1:].expand(-1, frames - per_channel.shape[1], -1)], dim=1)
    return per_channel[:, :frames].reshape(2 * frames, -1)


class TestEncodeTailAudio:
    @pytest.mark.parametrize("latents", [TAIL_AUDIO_LATENTS - 5, TAIL_AUDIO_LATENTS, TAIL_AUDIO_LATENTS + 9])
    def test_it_equals_the_reference_audio_encode_trimmed_or_padded_to_37(self, audio_vae, latents):
        waveform = torch.randn(2, latents * _HOP, generator=torch.Generator().manual_seed(3))
        rows = encode_tail_audio(audio_vae, waveform)
        assert rows.shape == (2 * TAIL_AUDIO_LATENTS, 4)
        assert torch.equal(rows, _trim_or_pad(_oracle_audio(audio_vae, waveform), TAIL_AUDIO_LATENTS))

    def test_a_short_tail_repeats_its_last_latent(self, audio_vae):
        rows = encode_tail_audio(audio_vae, torch.randn(2, 30 * _HOP)).reshape(2, TAIL_AUDIO_LATENTS, 4)
        assert torch.equal(rows[:, 30:], rows[:, 29:30].expand(-1, 7, -1))

    def test_rows_are_channel_major(self, audio_vae):
        waveform = torch.zeros(2, TAIL_AUDIO_LATENTS * _HOP)
        waveform[1] = torch.randn(TAIL_AUDIO_LATENTS * _HOP, generator=torch.Generator().manual_seed(4))
        rows = encode_tail_audio(audio_vae, waveform)
        alone = encode_tail_audio(audio_vae, waveform.flip(0))  # channels swapped
        assert torch.allclose(rows[:TAIL_AUDIO_LATENTS], alone[TAIL_AUDIO_LATENTS:], atol=1e-5)
        assert torch.allclose(rows[TAIL_AUDIO_LATENTS:], alone[:TAIL_AUDIO_LATENTS], atol=1e-5)
        assert not torch.equal(rows[:TAIL_AUDIO_LATENTS], rows[TAIL_AUDIO_LATENTS:])

    def test_it_returns_float32_on_the_cpu_and_takes_the_posterior_mean(self, audio_vae):
        waveform = torch.randn(2, TAIL_AUDIO_LATENTS * _HOP)
        rows = encode_tail_audio(audio_vae, waveform)
        assert rows.dtype == torch.float32 and rows.device.type == "cpu" and not rows.requires_grad
        assert torch.equal(rows, encode_tail_audio(audio_vae, waveform))  # no sampling: repeatable

    @pytest.mark.parametrize("shape", [(5,), (1, 100), (3, 100), (2, 1, 100)])
    def test_it_refuses_a_waveform_that_is_not_stereo(self, audio_vae, shape):
        with pytest.raises(ValueError, match="stereo waveform"):
            encode_tail_audio(audio_vae, torch.zeros(shape))


# --- held audio rows ---------------------------------------------------------------------------------------------


class TestTailAudioRows:
    def test_the_held_rows_are_the_first_37_of_each_channel(self):
        rows = tail_audio_rows(50)
        assert rows.tolist() == [*range(37), *range(50, 87)]

    def test_the_tail_first_order_puts_both_tails_first_then_the_rest_ascending(self):
        order = tail_first_audio_order(50)
        assert order.tolist() == [*range(37), *range(50, 87), *range(37, 50), *range(87, 100)]
        assert sorted(order.tolist()) == list(range(100))

    def test_a_clip_whose_audio_is_all_tail(self):
        assert tail_first_audio_order(37).tolist() == list(range(74))
        assert tail_audio_rows(37).tolist() == list(range(74))

    @pytest.mark.parametrize(("audio_latents", "tail"), [(36, 37), (50, 0), (50, 51)])
    def test_it_refuses_a_tail_outside_the_clip(self, audio_latents, tail):
        with pytest.raises(ValueError, match="audio tail"):
            tail_first_audio_order(audio_latents, tail)
        with pytest.raises(ValueError, match="audio tail"):
            tail_audio_rows(audio_latents, tail)

    def test_row_timesteps_pins_exactly_the_held_rows(self):
        """The leading 74 entries of the order get the clean level; the other audio rows take the step's level."""
        text, audio_latents = 3, 50
        audio_idx = text + tail_first_audio_order(audio_latents)
        video_idx = torch.arange(text + 2 * audio_latents, text + 2 * audio_latents + 6)
        distinct, index = row_timesteps(video_idx, audio_idx, text, 0.5, 0.25, condition_video_rows=2, condition_audio_rows=2 * TAIL_AUDIO_LATENTS)
        by_row = distinct[index]
        held = text + tail_audio_rows(audio_latents)
        assert (by_row[held] == AUDIO_COND_T).all()
        free = text + torch.tensor([row for row in range(2 * audio_latents) if row not in set(tail_audio_rows(audio_latents).tolist())])
        assert (by_row[free] == 0.25).all()
        assert (by_row[video_idx[:2]] == KEYFRAME_T).all()


# --- pre-noising -------------------------------------------------------------------------------------------------


class TestPrenoise:
    def test_it_equals_scale_noise_of_the_generators_noise(self):
        scheduler = MiniMaxH3Scheduler()
        latents = torch.randn(1, 4, 1, 2, 4, generator=torch.Generator().manual_seed(7))
        noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(8))
        mixed = prenoise(scheduler, latents, KEYFRAME_T, torch.Generator().manual_seed(8))
        assert torch.equal(mixed, scheduler.scale_noise(latents, KEYFRAME_T, noise))
        assert torch.allclose(mixed, 0.999 * latents + 0.001 * noise)

    def test_it_matches_the_prepare_latents_step_of_diffusers(self):
        """The released step noises each condition with ``randn_tensor`` under the request generator, then ``scale_noise(..., 0.999)``."""
        from diffusers.utils.torch_utils import randn_tensor

        scheduler = MiniMaxH3Scheduler()
        conditions = [torch.randn(1, 4, 1, 2, 4, generator=torch.Generator().manual_seed(i)) for i in range(2)]
        reference_generator = torch.Generator().manual_seed(11)
        expected = [scheduler.scale_noise(c, 0.999, randn_tensor(c.shape, generator=reference_generator, device=c.device, dtype=torch.float32)) for c in conditions]
        generator = torch.Generator().manual_seed(11)
        assert all(torch.equal(prenoise(scheduler, c, KEYFRAME_T, generator), e) for c, e in zip(conditions, expected, strict=True))

    def test_a_clean_level_returns_the_latents_but_still_draws_the_noise(self):
        scheduler = MiniMaxH3Scheduler()
        latents = torch.randn(74, 4)
        generator = torch.Generator().manual_seed(1)
        assert torch.equal(prenoise(scheduler, latents, AUDIO_COND_T, generator), latents)
        after = torch.Generator().manual_seed(1)
        torch.randn(latents.shape, generator=after)
        assert torch.equal(torch.randn(3, generator=generator), torch.randn(3, generator=after))


# --- loading the encoder side ------------------------------------------------------------------------------------


def _write_encoder_side(directory, vae, *, shards=False):
    """Save *vae*'s config and its encoder-side weights (a file with no decoder tensors) to *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    config = {"_class_name": "AutoencoderKLMiniMaxH3", **{key: list(value) if isinstance(value, tuple) else value for key, value in _VAE_CONFIG.items()}}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    state = {name: tensor.contiguous() for name, tensor in vae.state_dict().items() if name.startswith(("encoder.", "quant_conv."))}
    if not shards:
        save_file(state, str(directory / "diffusion_pytorch_model.safetensors"))
        return
    names = sorted(state)
    halves = {"model-00001-of-00002.safetensors": names[: len(names) // 2], "model-00002-of-00002.safetensors": names[len(names) // 2 :]}
    for file, part in halves.items():
        save_file({name: state[name] for name in part}, str(directory / file))
    weight_map = {name: file for file, part in halves.items() for name in part}
    (directory / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")


class TestLoadVaeEncoder:
    @pytest.mark.parametrize("shards", [False, True])
    def test_it_loads_the_encoder_without_decoder_weights_and_encodes_like_the_original(self, vae, tmp_path, shards):
        _write_encoder_side(tmp_path, vae, shards=shards)
        loaded = load_vae_encoder(tmp_path, torch.device("cpu"))
        assert isinstance(loaded.decoder, torch.nn.Identity) and isinstance(loaded.post_quant_conv, torch.nn.Identity)
        image = _picture(*_CANVAS)
        assert torch.equal(encode_keyframe(loaded, image, *_CANVAS), encode_keyframe(vae, image, *_CANVAS))

    def test_it_is_eval_mode_without_gradients_and_in_float32(self, vae, tmp_path):
        _write_encoder_side(tmp_path, vae)
        loaded = load_vae_encoder(tmp_path, torch.device("cpu"))
        assert not loaded.training
        parameters = list(loaded.parameters())
        assert parameters and not any(parameter.requires_grad for parameter in parameters)
        assert all(parameter.dtype == torch.float32 for parameter in parameters)

    def test_it_refuses_weights_that_miss_the_encoder(self, vae, tmp_path):
        _write_encoder_side(tmp_path, vae)
        file = tmp_path / "diffusion_pytorch_model.safetensors"
        file.unlink()
        save_file({"quant_conv.weight": vae.quant_conv.weight.contiguous(), "quant_conv.bias": vae.quant_conv.bias.contiguous()}, str(file))
        with pytest.raises(RuntimeError):
            load_vae_encoder(tmp_path, torch.device("cpu"))
