"""Tests for the MOSS-Image sampling loop and latent helpers, including golden checks against the vendored pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers import Flux2Pipeline
from moss_image.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
from torch import nn

from zvisiongenerator.backends.cuda.image.moss_sampler import (
    Conditioning,
    decode,
    denoise,
    encode_image,
    encode_prompt,
    initial_noise,
    noised,
    set_schedule,
    to_vae_latents,
)
from zvisiongenerator.core.moss import img2img_run_steps

_TEXT_DIM = 32
_LATENT_SHAPE = (1, 128, 4, 4)


class _StubVae(nn.Module):
    """FLUX.2-VAE stand-in: random batch-norm statistics and a fixed 1x1 conv plus 8x upsample as the decoder."""

    def __init__(self) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(3)
        self.bn = nn.BatchNorm2d(128)
        self.bn.running_mean.copy_(torch.randn(128, generator=gen))
        self.bn.running_var.copy_(torch.rand(128, generator=gen) + 0.5)
        self.config = SimpleNamespace(batch_norm_eps=1e-4)
        self.conv = nn.Conv2d(32, 3, 1)
        nn.init.normal_(self.conv.weight, generator=gen)
        self.encoded: torch.Tensor | None = None

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        out = nn.functional.interpolate(self.conv(z), scale_factor=8, mode="nearest")
        return (out,)

    def encode(self, pixels: torch.Tensor):
        assert self.encoded is not None
        mode = SimpleNamespace(mode=lambda: self.encoded)
        return SimpleNamespace(latent_dist=mode)


@pytest.fixture(scope="module")
def transformer() -> MOSSImageTransformer:
    torch.manual_seed(0)
    return MOSSImageTransformer(dim=64, ffn_dim=128, num_heads=2, num_layers=3, text_dim=_TEXT_DIM, freq_dim=16).eval()


def _conditioning(seed: int, length: int = 5) -> Conditioning:
    gen = torch.Generator().manual_seed(seed)
    mask = torch.ones(1, length, dtype=torch.bool)
    mask[0, -1] = False
    return Conditioning(torch.randn(1, length, _TEXT_DIM, generator=gen), mask)


def _vendored_pipeline(transformer, vae=None) -> MOSSImagePipeline:
    return MOSSImagePipeline(transformer, MOSSImageScheduler(), vae=vae)


@pytest.mark.parametrize("guidance", [3.0, 1.0])
@pytest.mark.parametrize("steps", [4, 6])
def test_loop_matches_the_vendored_pipeline_latents(transformer, guidance, steps):
    cond, uncond = _conditioning(1), _conditioning(2, length=7)
    skip = (1,)
    pipe = _vendored_pipeline(transformer)
    with torch.no_grad():
        expected = pipe(
            prompt_embeds=cond.embeds,
            prompt_attention_mask=cond.mask,
            negative_prompt_embeds=uncond.embeds,
            negative_prompt_attention_mask=uncond.mask,
            height=64,
            width=64,
            num_inference_steps=steps,
            guidance_scale=guidance,
            uncond_skip_layers=skip,
            generator=torch.Generator("cpu").manual_seed(11),
            output_type="latent",
            return_dict=False,
        )[0]

    scheduler = MOSSImageScheduler()
    start = set_schedule(scheduler, steps, 1.0, "cpu")
    noise = initial_noise(_LATENT_SHAPE, torch.Generator("cpu").manual_seed(11), "cpu")
    with torch.no_grad():
        ours = denoise(
            transformer,
            scheduler,
            noise,
            cond=cond,
            uncond=uncond if guidance != 1.0 else None,
            guidance=guidance,
            skip_layers=skip,
            start=start,
            on_step=lambda i, t, x: False,
        )
    assert start == 0
    torch.testing.assert_close(ours, expected, rtol=0, atol=1e-6)


def test_decode_matches_the_vendored_pt_output(transformer):
    vae = _StubVae()
    pipe = _vendored_pipeline(transformer, vae=vae)
    latents = torch.randn(_LATENT_SHAPE, generator=torch.Generator().manual_seed(5))
    # One step from the given latents, then the vendored decode (upstream lines 431-437) vs ours on the stepped latents.
    cond = _conditioning(1)
    with torch.no_grad():
        expected = pipe(
            prompt_embeds=cond.embeds,
            prompt_attention_mask=cond.mask,
            height=64,
            width=64,
            num_inference_steps=1,
            guidance_scale=1.0,
            latents=latents,
            uncond_skip_layers=(),
            output_type="pt",
            return_dict=False,
        )[0]
        stepped = _stepped_latents(transformer, cond, latents, steps=1)
        ours = decode(vae, pipe.image_processor, stepped, output_type="pt")
    assert ours.shape == (1, 3, 64, 64)
    torch.testing.assert_close(ours, expected, rtol=0, atol=1e-6)


def _stepped_latents(transformer, cond, latents, *, steps):
    scheduler = MOSSImageScheduler()
    start = set_schedule(scheduler, steps, 1.0, "cpu")
    return denoise(
        transformer,
        scheduler,
        latents,
        cond=cond,
        uncond=None,
        guidance=1.0,
        skip_layers=(),
        start=start,
        on_step=lambda i, t, x: False,
    )


@pytest.mark.parametrize("steps", [1, 4, 8, 20, 30, 50])
def test_scheduler_realises_every_requested_step(steps):
    scheduler = MOSSImageScheduler()
    scheduler.set_timesteps(steps, device="cpu")
    assert len(scheduler.timesteps) == steps


@pytest.mark.parametrize("strength", [1.0, 0.8, 0.5, 0.25, 0.0])
def test_set_schedule_start_index_and_begin_index(strength):
    scheduler = MOSSImageScheduler()
    start = set_schedule(scheduler, 10, strength, "cpu")
    total = len(scheduler.timesteps)
    assert start == total - img2img_run_steps(total, strength)
    assert scheduler.begin_index == start
    if strength >= 1.0:
        assert start == 0
    assert 0 <= start < total


def test_img2img_loop_uses_the_sigma_tail_and_counts_callbacks(transformer):
    scheduler = MOSSImageScheduler()
    start = set_schedule(scheduler, 8, 0.5, "cpu")
    total = len(scheduler.timesteps)
    run = total - start
    assert 0 < start < total
    seen: list[tuple[int, float]] = []
    latents = noised(torch.zeros(_LATENT_SHAPE), initial_noise(_LATENT_SHAPE, torch.Generator().manual_seed(1), "cpu"), float(scheduler.sigmas[start]))
    with torch.no_grad():
        out = denoise(
            transformer,
            scheduler,
            latents,
            cond=_conditioning(1),
            uncond=None,
            guidance=1.0,
            skip_layers=(),
            start=start,
            on_step=lambda i, t, x: seen.append((i, float(t))) or False,
        )
    assert out is not None
    assert [i for i, _ in seen] == list(range(run))
    assert [t for _, t in seen] == [float(t) for t in scheduler.timesteps[start:]]


def test_img2img_resume_equals_the_tail_of_a_full_run(transformer):
    """Starting at ``start`` from the full run's latents after step ``start - 1`` reproduces the full run's result."""
    steps, strength = 8, 0.5
    cond = _conditioning(1)
    noise = initial_noise(_LATENT_SHAPE, torch.Generator().manual_seed(2), "cpu")

    full_scheduler = MOSSImageScheduler()
    full_start = set_schedule(full_scheduler, steps, 1.0, "cpu")
    resume_scheduler = MOSSImageScheduler()
    start = set_schedule(resume_scheduler, steps, strength, "cpu")
    assert full_start == 0 and 0 < start < len(full_scheduler.timesteps)

    captured: dict[str, torch.Tensor] = {}

    def capture(i, t, x):
        if i == start - 1:
            captured["x"] = x.clone()
        return False

    with torch.no_grad():
        full = denoise(transformer, full_scheduler, noise, cond=cond, uncond=None, guidance=1.0, skip_layers=(), start=0, on_step=capture)
        resumed = denoise(transformer, resume_scheduler, captured["x"], cond=cond, uncond=None, guidance=1.0, skip_layers=(), start=start, on_step=lambda i, t, x: False)
    torch.testing.assert_close(resumed, full, rtol=0, atol=1e-6)


@pytest.mark.parametrize("strength", [0.75, 0.5, 0.25])
def test_img2img_noising_matches_the_scheduler_sigmas(strength):
    """With the exact flow-matching velocity (noise - clean), denoising from ``noised(clean, noise, sigmas[start])`` recovers *clean*."""

    class _ExactVelocity:
        def __init__(self, velocity: torch.Tensor) -> None:
            self.velocity = velocity

        def __call__(self, latents, t, embeds, mask, **kwargs):
            return SimpleNamespace(sample=self.velocity)

    clean = torch.randn(_LATENT_SHAPE, generator=torch.Generator().manual_seed(6))
    noise = initial_noise(_LATENT_SHAPE, torch.Generator().manual_seed(7), "cpu")
    scheduler = MOSSImageScheduler()
    start = set_schedule(scheduler, 10, strength, "cpu")
    assert start > 0
    latents = noised(clean, noise, float(scheduler.sigmas[start]))
    out = denoise(
        _ExactVelocity(noise - clean),
        scheduler,
        latents,
        cond=_conditioning(1),
        uncond=None,
        guidance=1.0,
        skip_layers=(),
        start=start,
        on_step=lambda i, t, x: False,
    )
    assert float(scheduler.sigmas[-1]) == 0.0
    torch.testing.assert_close(out, clean, rtol=0, atol=1e-5)


def test_on_step_returning_true_stops_and_returns_none(transformer):
    scheduler = MOSSImageScheduler()
    start = set_schedule(scheduler, 6, 1.0, "cpu")
    calls: list[int] = []

    def stop_after_two(i, t, x):
        calls.append(i)
        return i == 1

    with torch.no_grad():
        out = denoise(
            transformer,
            scheduler,
            torch.zeros(_LATENT_SHAPE),
            cond=_conditioning(1),
            uncond=None,
            guidance=1.0,
            skip_layers=(),
            start=start,
            on_step=stop_after_two,
        )
    assert out is None
    assert calls == [0, 1]


def test_noised_blends_by_sigma():
    clean, noise = torch.full((2, 2), 2.0), torch.full((2, 2), 6.0)
    torch.testing.assert_close(noised(clean, noise, 0.25), torch.full((2, 2), 3.0))
    torch.testing.assert_close(noised(clean, noise, 0.0), clean)
    torch.testing.assert_close(noised(clean, noise, 1.0), noise)


def test_initial_noise_is_fp32_and_seeded():
    a = initial_noise(_LATENT_SHAPE, torch.Generator().manual_seed(9), "cpu")
    b = initial_noise(_LATENT_SHAPE, torch.Generator().manual_seed(9), "cpu")
    assert a.dtype == torch.float32 and a.shape == _LATENT_SHAPE
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_encode_image_normalises_and_to_vae_latents_inverts_it():
    vae = _StubVae()
    z = torch.randn(1, 32, 8, 8, generator=torch.Generator().manual_seed(4))
    vae.encoded = z
    processor = SimpleNamespace(preprocess=lambda image, height, width: torch.zeros(1, 3, height, width))
    latents = encode_image(vae, processor, object(), 64, 64)
    assert latents.shape == (1, 128, 4, 4)
    patchified = Flux2Pipeline._patchify_latents(z)
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    torch.testing.assert_close(latents, (patchified - mean) / std)
    torch.testing.assert_close(to_vae_latents(vae, latents), z, rtol=0, atol=1e-5)


def test_encode_prompt_passes_the_device_and_returns_conditioning():
    seen: dict[str, object] = {}

    class _Batch(dict):
        def to(self, device):
            seen["device"] = device
            return self

    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, text, return_tensors, truncation, max_length):
            seen["max_length"] = max_length
            return _Batch(input_ids=torch.zeros(1, 3, dtype=torch.long))

    class _Encoder:
        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.ones(1, 3, 4))

    cond = encode_prompt(_Tokenizer(), _Encoder(), "a cat", device="cpu", max_length=77)
    assert seen == {"max_length": 77, "device": "cpu"}
    assert cond.embeds.shape == (1, 3, 4)
    assert cond.mask.tolist() == [[True, True, True]]
