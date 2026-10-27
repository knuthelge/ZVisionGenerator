"""Tests for the Euler-ancestral sampler of the distilled LTX-2.5 rows (seed and sigma maths, shared noise, scheduler swap).

The scheduler tests run diffusers' real scheduler on CPU tensors of a few elements; no pipeline or weights are loaded.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from zvisiongenerator.backends.cuda.video.sampler import (
    STAGE_SEED_OFFSETS,
    SharedNoise,
    ancestral_scheduler,
    stage_seed,
    swap_scheduler,
    with_terminal_sigma,
)

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from diffusers.pipelines.ltx2.pipeline_ltx2 import retrieve_timesteps  # noqa: E402
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES  # noqa: E402


class TestStageSeed:
    def test_stage_offsets_are_10000_and_20000(self):
        assert dict(STAGE_SEED_OFFSETS) == {1: 10_000, 2: 20_000}

    @pytest.mark.parametrize(("seed", "stage", "expected"), [(5, 1, 10_005), (5, 2, 20_005), (0, 1, 10_000), (2**32 - 1, 2, 2**32 + 19_999)])
    def test_adds_the_stage_offset_without_wrapping(self, seed, stage, expected):
        assert stage_seed(seed, stage) == expected

    @pytest.mark.parametrize("stage", [1, 2])
    def test_no_seed_gives_no_stage_seed(self, stage):
        assert stage_seed(None, stage) is None

    @pytest.mark.parametrize(("stage", "seed"), [(0, 7), ("1", None)])
    def test_bad_stage_raises_with_or_without_a_seed(self, stage, seed):
        with pytest.raises(ValueError, match="stage"):
            stage_seed(seed, stage)


class TestWithTerminalSigma:
    def test_appends_zero(self):
        assert with_terminal_sigma([1.0, 0.5]) == [1.0, 0.5, 0.0]

    def test_keeps_an_existing_terminal_zero(self):
        assert with_terminal_sigma([1.0, 0.5, 0.0]) == [1.0, 0.5, 0.0]

    def test_returns_a_new_list_of_floats(self):
        source = (1, 0)
        result = with_terminal_sigma(source)
        assert result == [1.0, 0.0]
        assert all(isinstance(value, float) for value in result)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            with_terminal_sigma([])


def test_importing_the_module_loads_neither_torch_nor_diffusers():
    code = "import sys; import zvisiongenerator.backends.cuda.video.sampler; bad = [m for m in ('torch', 'diffusers') if m in sys.modules]; assert not bad, bad"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True)


class TestSharedNoise:
    def test_deepcopy_returns_the_same_holder(self):
        noise = SharedNoise(object())
        assert copy.deepcopy(noise) is noise

    def test_deepcopy_of_a_container_keeps_the_holder_shared(self):
        noise = SharedNoise(object())
        clone = copy.deepcopy({"noise": noise, "other": [1, 2]})
        assert clone["noise"] is noise


VIDEO_SHAPE = (1, 4, 2, 2, 2)
AUDIO_SHAPE = (1, 3, 5)


def _generator(seed: int):
    return torch.Generator(device="cpu").manual_seed(seed)


def _scheduler(seed: int = 3, table=DISTILLED_SIGMA_VALUES):
    """Return an ancestral scheduler set up the way the pipeline does it: ``retrieve_timesteps`` then ``set_begin_index(0)``."""
    scheduler = ancestral_scheduler(_generator(seed))
    retrieve_timesteps(scheduler, None, "cpu", None, sigmas=table, mu=1.0)
    scheduler.set_begin_index(0)
    return scheduler


def _expected_step(sample, velocity, sigma, sigma_next, noise):
    """Euler-ancestral RF update with eta 1 and s_noise 1, written out independently of diffusers."""
    denoised = sample - sigma * velocity
    if sigma_next == 0.0:
        return denoised
    sigma_down = sigma_next * (sigma_next / sigma)
    alpha_next, alpha_down = 1.0 - sigma_next, 1.0 - sigma_down
    ratio = sigma_down / sigma
    mean = ratio * sample + (1.0 - ratio) * denoised
    renoise = (sigma_next**2 - sigma_down**2 * alpha_next**2 / (alpha_down**2 + 1e-12)) ** 0.5
    return (alpha_next / (alpha_down + 1e-12)) * mean + noise * renoise


def _run_loop(video, audio, steps: int | None = None):
    """Step a video and an audio scheduler in the pipeline's order (video first); return both final samples."""
    x, a = torch.ones(VIDEO_SHAPE), torch.ones(AUDIO_SHAPE)
    for t in video.timesteps[:steps]:
        x = video.step(torch.full_like(x, 0.1), t, x, return_dict=False)[0]
        a = audio.step(torch.full_like(a, 0.2), t, a, return_dict=False)[0]
    return x, a


class TestSchedule:
    def test_eight_sigma_table_gives_eight_timesteps_and_a_terminal_zero(self):
        scheduler = _scheduler()
        assert len(scheduler.timesteps) == 8
        assert len(scheduler.sigmas) == 9
        assert float(scheduler.sigmas[-1]) == 0.0
        assert [round(float(s), 6) for s in scheduler.sigmas[:-1]] == pytest.approx(DISTILLED_SIGMA_VALUES)

    def test_timesteps_are_sigmas_times_the_train_timesteps_without_the_zero(self):
        scheduler = _scheduler()
        assert float(scheduler.timesteps[0]) == pytest.approx(1000.0)
        assert float(scheduler.timesteps[-1]) == pytest.approx(421.875)
        assert float(scheduler.timesteps.min()) > 0.0

    def test_num_train_timesteps_scales_the_timesteps(self):
        scheduler = ancestral_scheduler(_generator(1), num_train_timesteps=500)
        scheduler.set_timesteps(sigmas=DISTILLED_SIGMA_VALUES, device="cpu")
        assert float(scheduler.timesteps[0]) == pytest.approx(500.0)

    def test_stage_two_table_gives_three_timesteps(self):
        scheduler = ancestral_scheduler(_generator(3))
        timesteps, steps = retrieve_timesteps(scheduler, None, "cpu", None, sigmas=STAGE_2_DISTILLED_SIGMA_VALUES, mu=1.0)
        assert (len(timesteps), steps) == (3, 3)
        assert len(scheduler.sigmas) == 4
        assert float(scheduler.sigmas[-1]) == 0.0

    def test_set_timesteps_requires_explicit_sigmas(self):
        with pytest.raises(ValueError, match="sigmas"):
            ancestral_scheduler(_generator(3)).set_timesteps(8)

    def test_the_scheduler_is_eta_one_and_s_noise_one(self):
        config = ancestral_scheduler(_generator(3)).config
        assert (config.eta, config.s_noise) == (1.0, 1.0)

    def test_each_call_builds_a_fresh_scheduler_of_one_cached_class(self):
        first, second = ancestral_scheduler(_generator(1)), ancestral_scheduler(_generator(2))
        assert first is not second
        assert type(first) is type(second)


class TestStepping:
    def test_the_first_step_matches_the_ancestral_formula(self):
        scheduler = _scheduler(seed=11)
        sample, velocity = torch.ones(VIDEO_SHAPE), torch.full(VIDEO_SHAPE, 0.1)
        noise = torch.randn(VIDEO_SHAPE, generator=_generator(11))
        result = scheduler.step(velocity, scheduler.timesteps[0], sample, return_dict=False)[0]
        sigma, sigma_next = DISTILLED_SIGMA_VALUES[0], DISTILLED_SIGMA_VALUES[1]
        assert torch.allclose(result, _expected_step(sample, velocity, sigma, sigma_next, noise), atol=1e-5)

    def test_the_final_step_is_the_denoised_sample_and_draws_no_noise(self):
        scheduler = _scheduler(seed=11)
        x = torch.ones(VIDEO_SHAPE)
        for t in scheduler.timesteps[:-1]:
            x = scheduler.step(torch.full_like(x, 0.1), t, x, return_dict=False)[0]
        state_before = scheduler.noise.generator.get_state()
        last = scheduler.step(torch.full_like(x, 0.1), scheduler.timesteps[-1], x, return_dict=False)[0]
        assert torch.allclose(last, x - DISTILLED_SIGMA_VALUES[-1] * 0.1)
        assert torch.equal(scheduler.noise.generator.get_state(), state_before)

    def test_an_explicit_generator_overrides_the_shared_one(self):
        scheduler = _scheduler(seed=11)
        sample, velocity = torch.ones(VIDEO_SHAPE), torch.full(VIDEO_SHAPE, 0.1)
        state_before = scheduler.noise.generator.get_state()
        result = scheduler.step(velocity, scheduler.timesteps[0], sample, generator=_generator(99), return_dict=False)[0]
        noise = torch.randn(VIDEO_SHAPE, generator=_generator(99))
        assert torch.allclose(result, _expected_step(sample, velocity, DISTILLED_SIGMA_VALUES[0], DISTILLED_SIGMA_VALUES[1], noise), atol=1e-5)
        assert torch.equal(scheduler.noise.generator.get_state(), state_before)


class TestSharedGenerator:
    def test_a_deep_copied_scheduler_shares_the_noise_holder_and_generator(self):
        video = _scheduler()
        audio = copy.deepcopy(video)
        assert audio is not video
        assert audio.noise is video.noise
        assert audio.noise.generator is video.noise.generator

    def test_the_deep_copy_keeps_its_own_step_state(self):
        video = _scheduler()
        audio = copy.deepcopy(video)
        video.step(torch.zeros(VIDEO_SHAPE), video.timesteps[0], torch.ones(VIDEO_SHAPE), return_dict=False)
        assert video.step_index == 1
        assert audio.step_index != 1

    def test_video_and_audio_steps_consume_the_generator_in_order(self):
        video = _scheduler(seed=21)
        audio = copy.deepcopy(video)
        x, a = torch.ones(VIDEO_SHAPE), torch.ones(AUDIO_SHAPE)
        vx = video.step(torch.full_like(x, 0.1), video.timesteps[0], x, return_dict=False)[0]
        ax = audio.step(torch.full_like(a, 0.2), audio.timesteps[0], a, return_dict=False)[0]

        reference = _generator(21)
        video_noise = torch.randn(VIDEO_SHAPE, generator=reference)
        audio_noise = torch.randn(AUDIO_SHAPE, generator=reference)
        sigma, sigma_next = DISTILLED_SIGMA_VALUES[0], DISTILLED_SIGMA_VALUES[1]
        assert torch.allclose(vx, _expected_step(x, torch.full_like(x, 0.1), sigma, sigma_next, video_noise), atol=1e-5)
        assert torch.allclose(ax, _expected_step(a, torch.full_like(a, 0.2), sigma, sigma_next, audio_noise), atol=1e-5)

    def test_drawing_audio_noise_before_video_noise_changes_the_video_result(self):
        x, a = torch.ones(VIDEO_SHAPE), torch.ones(AUDIO_SHAPE)

        video = _scheduler(seed=21)
        audio = copy.deepcopy(video)
        video_first = video.step(torch.full_like(x, 0.1), video.timesteps[0], x, return_dict=False)[0]
        audio.step(torch.full_like(a, 0.2), audio.timesteps[0], a, return_dict=False)

        video = _scheduler(seed=21)
        audio = copy.deepcopy(video)
        audio.step(torch.full_like(a, 0.2), audio.timesteps[0], a, return_dict=False)
        video_second = video.step(torch.full_like(x, 0.1), video.timesteps[0], x, return_dict=False)[0]

        assert not torch.equal(video_first, video_second)


class TestDeterminism:
    def _run(self, seed: int):
        video = _scheduler(seed=seed)
        return _run_loop(video, copy.deepcopy(video))

    def test_the_same_seed_gives_identical_tensors(self):
        (x1, a1), (x2, a2) = self._run(5), self._run(5)
        assert torch.equal(x1, x2)
        assert torch.equal(a1, a2)

    def test_a_different_seed_gives_different_tensors(self):
        (x1, a1), (x2, a2) = self._run(5), self._run(6)
        assert not torch.equal(x1, x2)
        assert not torch.equal(a1, a2)

    def test_the_stage_seeds_give_independent_noise(self):
        first, second = self._run(stage_seed(5, 1)), self._run(stage_seed(5, 2))
        assert not torch.equal(first[0], second[0])


class TestSwapScheduler:
    def test_installs_the_scheduler_during_the_body_and_restores_the_original(self):
        original, replacement = object(), object()
        pipeline = SimpleNamespace(scheduler=original)
        with swap_scheduler(pipeline, replacement):
            assert pipeline.scheduler is replacement
        assert pipeline.scheduler is original

    def test_restores_after_an_exception(self):
        original = FlowMatchEulerDiscreteScheduler()
        pipeline = SimpleNamespace(scheduler=original)
        with pytest.raises(RuntimeError, match="boom"), swap_scheduler(pipeline, ancestral_scheduler(_generator(1))):
            assert pipeline.scheduler is not original
            raise RuntimeError("boom")
        assert pipeline.scheduler is original
