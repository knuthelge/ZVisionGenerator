"""Pin the diffusers LTX-2 internals that extending a clip on Windows and Linux relies on.

Introspection, plus tiny-tensor checks of static helpers: no pipeline is constructed and no weights are loaded. A failure
here after a diffusers bump means a name, signature or layout the CUDA extend path uses has changed.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import torch
from diffusers import (
    AutoencoderKLLTX2Audio,
    LTX2ConditionPipeline,
    LTX2ImageToVideoPipeline,
    LTX2Pipeline,
    LTX2VideoTransformer3DModel,
    LTXEulerAncestralRFScheduler,
)
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, LTX2_5_IMAGE_CRF, apply_image_conditioning_crf, resolve_default_image_crf


def _parameters(callable_obj) -> set[str]:
    return set(inspect.signature(callable_obj).parameters)


def test_the_video_condition_takes_frames_a_latent_index_and_a_strength():
    fields = {field.name for field in dataclasses.fields(LTX2VideoCondition)}
    assert {"frames", "index", "strength"} <= fields


def test_the_condition_pipeline_call_takes_the_arguments_the_backend_passes():
    assert {"conditions", "frame_rate", "sigmas", "num_inference_steps", "output_type", "callback_on_step_end", "prompt_embeds", "prompt_attention_mask", "generator"} <= _parameters(
        LTX2ConditionPipeline.__call__
    )


def test_audio_latents_pack_to_tokens_of_channels_times_mel_bins_and_normalize_per_feature():
    latent = torch.arange(1 * 8 * 3 * 16, dtype=torch.float32).reshape(1, 8, 3, 16)
    packed = LTX2ConditionPipeline._pack_audio_latents(latent)  # noqa: SLF001
    assert packed.shape == (1, 3, 128)
    assert torch.equal(packed[0, 1, 16:32], latent[0, 1, 1])  # channel-major: c * 16 + mel bin
    normalized = LTX2ConditionPipeline._normalize_audio_latents(packed, torch.ones(128), torch.full((128,), 2.0))  # noqa: SLF001
    assert torch.equal(normalized, (packed - 1) / 2)


def test_the_transformer_takes_audio_hidden_states_and_separate_audio_timestep_and_sigma():
    assert {"audio_hidden_states", "audio_timestep", "audio_sigma", "sigma", "use_cross_timestep"} <= _parameters(LTX2VideoTransformer3DModel.forward)


def test_the_audio_vae_encodes_a_stereo_mel_to_a_posterior_of_8_channels_16_bins_and_a_quarter_of_the_frames():
    vae = AutoencoderKLLTX2Audio(base_channels=8, causality_axis="height", norm_type="pixel", output_channels=2).eval()
    assert (vae.config.sample_rate, vae.config.mel_hop_length, vae.config.mel_bins) == (16000, 160, 64)
    with torch.no_grad():
        (posterior,) = vae.encode(torch.zeros(1, 2, 101, 64), return_dict=False)
    assert posterior.mode().shape == (1, 8, 26, 16)


def _tiny_transformer() -> LTX2VideoTransformer3DModel:
    """A one-layer transformer with LTX-2.3's switches (gated attention, cross-attention modulation, prompt AdaLN)."""
    torch.manual_seed(0)
    sizes = {
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "audio_num_attention_heads": 2,
        "audio_attention_head_dim": 8,
        "cross_attention_dim": 16,
        "audio_cross_attention_dim": 16,
        "caption_channels": 16,
    }
    channels = {"in_channels": 8, "out_channels": 8, "audio_in_channels": 8, "audio_out_channels": 8}
    return LTX2VideoTransformer3DModel(num_layers=1, gated_attn=True, audio_gated_attn=True, cross_attn_mod=True, audio_cross_attn_mod=True, use_prompt_embeddings=False, **sizes, **channels).eval()


def test_the_transformer_takes_the_audio_timestep_per_token_and_equal_values_match_one_per_sample():
    model = _tiny_transformer()
    video, audio, text = torch.randn(1, 8, 8), torch.randn(1, 5, 8), torch.randn(1, 4, 16)
    sigma = torch.tensor([600.0])

    def run(audio_timestep):
        with torch.no_grad():
            return model(
                hidden_states=video,
                audio_hidden_states=audio,
                encoder_hidden_states=text,
                audio_encoder_hidden_states=text,
                timestep=sigma.expand(1, 8),
                audio_timestep=audio_timestep,
                sigma=sigma,
                audio_sigma=sigma,
                encoder_attention_mask=torch.ones(1, 4),
                audio_encoder_attention_mask=torch.ones(1, 4),
                num_frames=2,
                height=2,
                width=2,
                fps=24.0,
                audio_num_frames=5,
                use_cross_timestep=True,
                return_dict=False,
            )[1]

    per_sample = run(sigma)
    assert torch.allclose(run(sigma.reshape(1, 1).expand(1, 5).clone()), per_sample, atol=1e-5)
    pinned = sigma.reshape(1, 1).expand(1, 5).clone()
    pinned[:, :2] = 0
    assert not torch.allclose(run(pinned)[:, :2], per_sample[:, :2])


def test_the_ancestral_scheduler_keeps_the_terminal_timestep_so_the_sampler_trims_it():
    scheduler = LTXEulerAncestralRFScheduler()
    scheduler.set_timesteps(sigmas=[*DISTILLED_SIGMA_VALUES, 0.0], device="cpu")
    assert len(scheduler.sigmas) == len(DISTILLED_SIGMA_VALUES) + 1
    assert len(scheduler.timesteps) == len(DISTILLED_SIGMA_VALUES) + 1  # a forward at t = 0 unless trimmed
    assert float(scheduler.timesteps[-1]) == 0.0


def test_the_ancestral_scheduler_set_timesteps_takes_sigmas_and_mu_and_step_takes_a_generator():
    assert {"sigmas", "device", "mu"} <= _parameters(LTXEulerAncestralRFScheduler.set_timesteps)
    assert {"model_output", "timestep", "sample", "generator", "return_dict"} <= _parameters(LTXEulerAncestralRFScheduler.step)
    config = LTXEulerAncestralRFScheduler().config
    assert {"num_train_timesteps", "eta", "s_noise"} <= set(config)


def test_the_ancestral_scheduler_final_step_is_the_denoised_sample_and_set_begin_index_exists():
    scheduler = LTXEulerAncestralRFScheduler(eta=1.0, s_noise=1.0)
    scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.0], device="cpu")
    scheduler.set_begin_index(0)
    sample, velocity = torch.ones(1, 2), torch.full((1, 2), 0.25)
    generator = torch.Generator(device="cpu").manual_seed(0)
    mid = scheduler.step(velocity, scheduler.timesteps[0], sample, generator=generator, return_dict=False)[0]
    last = scheduler.step(velocity, scheduler.timesteps[1], mid, generator=generator, return_dict=False)[0]
    assert torch.allclose(last, mid - 0.5 * velocity)


def test_the_three_ltx2_pipelines_list_the_optional_components_and_take_no_diffusion_decoder():
    for pipeline in (LTX2Pipeline, LTX2ImageToVideoPipeline, LTX2ConditionPipeline):
        assert {"processor", "prompt_enhancer", "duration_head"} <= set(pipeline._optional_components)  # noqa: SLF001
        parameters = _parameters(pipeline.__init__)
        assert {"processor", "prompt_enhancer", "duration_head"} <= parameters
        assert "diffusion_decoder" not in parameters


def test_the_image_conditioning_crf_defaults_and_zero_skips_recompression():
    assert resolve_default_image_crf(None) == 33
    assert LTX2_5_IMAGE_CRF == 18
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    assert apply_image_conditioning_crf(image, 0) is image
