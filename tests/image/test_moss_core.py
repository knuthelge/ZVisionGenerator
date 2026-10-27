"""Tests for the pure MOSS-Image settings."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from zvisiongenerator.core.moss import UNCOND_SKIP_LAYERS, img2img_run_steps, sampler_settings


def _settings(**overrides):
    kwargs = {"turbo": False, "steps": 30, "guidance": 4.0, "negative_prompt": None}
    return sampler_settings(**{**kwargs, **overrides})


def test_turbo_ignores_guidance_with_a_warning():
    with pytest.warns(UserWarning, match="guidance 4 is ignored"):
        sampler = _settings(turbo=True, steps=8, guidance=4.0, negative_prompt="blurry")
    assert (sampler.steps, sampler.guidance, sampler.negative_prompt, sampler.skip_layers) == (8, 1.0, None, ())
    assert not sampler.do_cfg


def test_turbo_at_guidance_one_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler = _settings(turbo=True, steps=8, guidance=1.0)
    assert sampler.guidance == 1.0


def test_sft_guided_run_uses_negative_prompt_and_skip_layers():
    sampler = _settings(negative_prompt="blurry")
    assert sampler.do_cfg
    assert sampler.guidance == 4.0
    assert sampler.negative_prompt == "blurry"
    assert sampler.skip_layers == UNCOND_SKIP_LAYERS


def test_sft_guided_run_without_negative_prompt_uses_empty_string():
    assert _settings(negative_prompt=None).negative_prompt == ""


def test_sft_without_guidance_has_no_negative_prompt():
    sampler = _settings(guidance=1.0, negative_prompt="blurry")
    assert not sampler.do_cfg
    assert sampler.negative_prompt is None


@pytest.mark.parametrize(
    ("total", "strength", "expected"),
    [(4, 0.4, 2), (5, 0.4, 2), (50, 1.0, 50), (4, 0.0, 1)],
)
def test_img2img_run_steps(total, strength, expected):
    assert img2img_run_steps(total, strength) == expected


def test_turbo_warning_points_at_the_caller():
    with pytest.warns(UserWarning, match="ignored") as caught:
        _settings(turbo=True, guidance=4.0)

    assert caught[0].filename == __file__


@pytest.fixture(scope="module")
def cuda_image_backend():
    """Import the diffusers backend for its img2img grid, and leave ``sys.modules`` as found."""
    key = "zvisiongenerator.backends.cuda.image.backend"
    had = key in sys.modules
    try:
        module = importlib.import_module(key)
    except ImportError:
        pytest.skip("the CUDA image backend cannot be imported here")
    yield module
    if not had:
        sys.modules.pop(key, None)


@pytest.mark.parametrize("total", [1, 2, 4, 5, 8, 30, 50])
@pytest.mark.parametrize("strength", [0.0, 0.1, 0.4, 0.5, 0.75, 1.0])
def test_img2img_run_steps_matches_the_cuda_backends_sigma_tail(cuda_image_backend, total, strength):
    """The count equals what the diffusers backend's img2img grid runs, as the docstring promises."""
    assert img2img_run_steps(total, strength) == len(cuda_image_backend._img2img_sigmas(total, strength))
