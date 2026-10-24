"""Tests for the MiniMax H3 row and frame arithmetic and clip geometry."""

from __future__ import annotations

from pathlib import Path

import pytest

from zvisiongenerator.backends.cuda.video.h3.frames import Geometry, geometry
from zvisiongenerator.utils import h3_memory
from zvisiongenerator.utils.h3_memory import (
    H3_TEXT_TOKENS,
    estimate_h3_mlx_clip_memory,
    estimate_h3_mlx_seconds,
    h3_audio_latents,
    h3_latent_frames,
    h3_mlx_clip_fit,
    h3_sequence_rows,
    h3_video_rows,
)
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget


@pytest.mark.parametrize(("frames", "latents"), [(22, 7), (39, 12), (124, 37), (243, 72)])
def test_latent_frames(frames, latents):
    assert h3_latent_frames(frames) == latents


@pytest.mark.parametrize("frames", [0, 1, 21, 23, 125, -12])
def test_latent_frames_refuses_other_counts(frames):
    with pytest.raises(ValueError, match="17 \\* n \\+ 5"):
        h3_latent_frames(frames)


@pytest.mark.parametrize(("frames", "latents"), [(22, 37), (124, 207), (243, 405)])
def test_audio_latents(frames, latents):
    assert h3_audio_latents(frames) == latents


def test_video_rows_pin():
    assert h3_video_rows(864, 480, 124) == 14_985


def test_sequence_rows_pin_adds_text_and_audio():
    assert h3_sequence_rows(864, 480, 124) == 15_499
    assert h3_sequence_rows(864, 480, 124) == 14_985 + 207 * 2 + H3_TEXT_TOKENS


@pytest.mark.parametrize(
    ("width", "height", "video", "sequence"),
    [
        (576, 320, 6_660, 7_174),
        (768, 448, 12_432, 12_946),
        (1024, 576, 21_312, 21_826),
        (1280, 704, 32_560, 33_074),
        (1600, 896, 51_800, 52_314),
        (384, 384, 5_328, 5_842),
        (576, 576, 11_988, 12_502),
        (768, 768, 21_312, 21_826),
        (960, 960, 33_300, 33_814),
        (1216, 1216, 53_428, 53_942),
    ],
)
def test_rows_at_124_frames(width, height, video, sequence):
    assert (h3_video_rows(width, height, 124), h3_sequence_rows(width, height, 124)) == (video, sequence)


def test_keyframes_add_one_latent_frame_of_rows_each():
    cell = (864 // 32) * (480 // 32)
    assert cell == 405
    assert h3_sequence_rows(864, 480, 124, keyframes=1) == 15_499 + cell
    assert h3_sequence_rows(864, 480, 124, keyframes=2) == 15_499 + 2 * cell


def test_tail_adds_no_rows():
    assert h3_sequence_rows(864, 480, 124, tail_frames=22) == h3_sequence_rows(864, 480, 124)


@pytest.mark.parametrize(
    "kwargs",
    [{"keyframes": -1}, {"tail_frames": 21}, {"tail_frames": 141}],
)
def test_sequence_rows_refuses_bad_conditioning(kwargs):
    with pytest.raises(ValueError):
        h3_sequence_rows(864, 480, 124, **kwargs)


@pytest.mark.parametrize(("width", "height"), [(864, 481), (850, 480), (0, 480), (864, 0), (-32, 480)])
def test_unaligned_canvas_is_refused(width, height):
    with pytest.raises(ValueError, match="multiple of 32"):
        h3_video_rows(width, height, 124)
    with pytest.raises(ValueError, match="multiple of 32"):
        geometry(width, height, 124)


def test_geometry_pin():
    geo = geometry(864, 480, 124)
    assert geo == Geometry(864, 480, 124, latent_frames=37, latent_height=30, latent_width=54, audio_latents=207)
    assert geo.video_rows == 14_985
    assert geo.audio_rows == 414


@pytest.mark.parametrize(("width", "height", "frames"), [(864, 480, 124), (768, 448, 22), (1280, 736, 243), (64, 32, 5)])
def test_geometry_agrees_with_the_arithmetic(width, height, frames):
    geo = geometry(width, height, frames)
    assert geo.video_rows == h3_video_rows(width, height, frames)
    assert geo.audio_rows + geo.video_rows + H3_TEXT_TOKENS == h3_sequence_rows(width, height, frames)


def test_geometry_refuses_a_bad_frame_count():
    with pytest.raises(ValueError, match="17 \\* n \\+ 5"):
        geometry(864, 480, 120)


def test_geometry_is_frozen():
    with pytest.raises(AttributeError):
        geometry(864, 480, 124).width = 1  # type: ignore[misc]


def test_public_names_are_exported():
    assert all(hasattr(h3_memory, name) for name in h3_memory.__all__)


class TestMacMemory:
    """The macOS (MLX) estimate, from the fork's M4 Pro measurements."""

    def test_the_reference_clip_is_15_gib(self):
        assert h3_sequence_rows(768, 448, 124) == 12_946
        assert estimate_h3_mlx_clip_memory(768, 448, 124) == 15 * GIB
        assert estimate_h3_mlx_clip_memory(768, 448, 124, mode="t2v") == 15 * GIB

    def test_text_to_video_grows_160_kib_per_row(self):
        extra = h3_sequence_rows(1280, 704, 124) - 12_946

        assert estimate_h3_mlx_clip_memory(1280, 704, 124) == 15 * GIB + 160 * 1024 * extra

    def test_a_small_clip_shrinks_below_the_reference(self):
        assert estimate_h3_mlx_clip_memory(512, 288, 22) < 15 * GIB

    @pytest.mark.parametrize("mode", ["i2v", "extend"])
    def test_an_image_mode_is_the_load_phase_when_that_is_larger(self, mode):
        # text encoder + vision 14.5, static transformer 1.7, VAEs 1.5
        assert estimate_h3_mlx_clip_memory(768, 448, 124, mode=mode) == pytest.approx(17.7 * GIB, abs=1)

    @pytest.mark.parametrize("mode", ["i2v", "extend"])
    def test_an_image_mode_is_the_denoise_phase_with_its_keyframe_when_that_is_larger(self, mode):
        rows = h3_sequence_rows(1920, 1088, 243, keyframes=1)
        denoise = 15 * GIB + 160 * 1024 * (rows - 12_946)

        assert denoise > 17.7 * GIB
        assert estimate_h3_mlx_clip_memory(1920, 1088, 243, mode=mode) == denoise

    def test_an_image_mode_never_needs_less_than_text_to_video(self):
        for width, height, frames in [(512, 288, 22), (768, 448, 124), (1280, 704, 243), (1920, 1088, 243)]:
            assert estimate_h3_mlx_clip_memory(width, height, frames, mode="i2v") >= estimate_h3_mlx_clip_memory(width, height, frames)

    def test_an_unknown_mode_or_a_bad_clip_is_refused(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            estimate_h3_mlx_clip_memory(768, 448, 124, mode="fl2va")
        with pytest.raises(ValueError, match="17 \\* n \\+ 5"):
            estimate_h3_mlx_clip_memory(768, 448, 121)
        with pytest.raises(ValueError, match="multiple of 32"):
            estimate_h3_mlx_clip_memory(770, 448, 124)

    def test_the_time_is_23_minutes_at_the_reference_clip_and_8_forwards(self):
        assert estimate_h3_mlx_seconds(768, 448, 124, forwards=8) == 23 * 60

    def test_the_time_is_the_fixed_part_plus_the_forwards(self):
        fixed = 240
        per_forward = (23 * 60 - fixed) / 8

        assert estimate_h3_mlx_seconds(768, 448, 124, forwards=4) == round(fixed + 4 * per_forward)
        assert estimate_h3_mlx_seconds(768, 448, 124, forwards=1) == round(fixed + per_forward)

    def test_a_larger_clip_takes_longer_and_an_image_mode_adds_its_keyframe_rows(self):
        assert estimate_h3_mlx_seconds(1280, 704, 124, forwards=8) > estimate_h3_mlx_seconds(768, 448, 124, forwards=8)
        assert estimate_h3_mlx_seconds(768, 448, 124, forwards=8, mode="i2v") > estimate_h3_mlx_seconds(768, 448, 124, forwards=8)

    def test_the_time_needs_a_forward(self):
        with pytest.raises(ValueError, match="forwards"):
            estimate_h3_mlx_seconds(768, 448, 124, forwards=0)

    @pytest.mark.parametrize(
        ("gpu_gib", "mode", "status", "required_gb"),
        [
            (24, "t2v", "fits", 15.0),
            (24, "i2v", "fits", 17.7),
            (16, "t2v", "fits", 15.0),
            (13, "t2v", "tight", 15.0),
            (9, "t2v", "too_large", 15.0),
            (16, "i2v", "tight", 17.7),
            (11, "extend", "too_large", 17.7),
        ],
    )
    def test_the_fit_classifies_against_the_gpu_working_set(self, gpu_gib, mode, status, required_gb):
        fit = h3_mlx_clip_fit(768, 448, 124, MemoryBudget(gpu_gib * GIB, unified_bytes=32 * GIB), forwards=8, mode=mode)

        assert fit == {"status": status, "required_gb": required_gb, "budget_gb": float(gpu_gib), "expected_seconds": estimate_h3_mlx_seconds(768, 448, 124, forwards=8, mode=mode)}

    def test_an_image_mode_can_be_tight_where_text_to_video_fits(self):
        budget = MemoryBudget(16 * GIB, unified_bytes=24 * GIB)

        assert h3_mlx_clip_fit(768, 448, 124, budget, forwards=8)["status"] == "fits"
        assert h3_mlx_clip_fit(768, 448, 124, budget, forwards=8, mode="i2v")["status"] == "tight"

    @pytest.mark.parametrize("budget", [None, MemoryBudget(0), MemoryBudget(10 * GIB, system_bytes=30 * GIB)])
    def test_there_is_no_fit_without_a_unified_budget(self, budget):
        assert h3_mlx_clip_fit(768, 448, 124, budget, forwards=8) is None


class TestCudaEstimate:
    """The RTX 3080 calibration: GPU peak, system peak and time of an H3 clip on a discrete GPU."""

    BUDGET = MemoryBudget(10 * GIB, system_bytes=30 * GIB)

    @pytest.fixture
    def tier(self):
        from zvisiongenerator.utils.config import load_config

        return h3_memory.h3_seedvr2_tier(load_config(), self.BUDGET)

    # Measured on the RTX 3080 (generation time of clean runs, excluding the one-time weight preparation).
    @pytest.mark.parametrize(
        ("width", "height", "frames", "forwards", "seconds"),
        [
            (864, 480, 124, 8, 279),
            (864, 480, 124, 4, 210),
            (864, 480, 73, 8, 203),
            (864, 480, 107, 8, 253),
            (864, 480, 243, 8, 513),
            (864, 480, 345, 8, 790),
            (1280, 736, 124, 8, 597),
        ],
    )
    def test_the_time_matches_the_measured_pins_within_5_percent(self, width, height, frames, forwards, seconds):
        assert h3_memory.h3_expected_seconds(width, height, frames, forwards=forwards) == pytest.approx(seconds, rel=0.05)

    def test_the_upscale_time_matches_the_measured_pin_within_5_percent(self):
        assert h3_memory.h3_expected_seconds(768, 448, 124, forwards=8, upscale=True) == pytest.approx(346, rel=0.05)

    def test_an_image_mode_adds_the_encoder_time_and_a_lora_slows_each_forward(self):
        plain = h3_memory.h3_expected_seconds(864, 480, 124, forwards=8)

        assert h3_memory.h3_expected_seconds(864, 480, 124, forwards=8, mode="i2v") > plain + 30
        assert h3_memory.h3_expected_seconds(864, 480, 124, forwards=8, loras=True) > plain

    def test_an_extension_encodes_its_tail_like_a_keyframe_without_extra_rows(self):
        t2v = h3_memory.h3_expected_seconds(864, 480, 124, forwards=8)

        assert h3_memory.h3_expected_seconds(864, 480, 124, forwards=8, mode="extend") == t2v + 30

    def test_an_upscaled_clip_renders_at_half_size_then_adds_the_seedvr2_phase(self):
        single = h3_memory.h3_expected_seconds(1536, 896, 124, forwards=8)
        upscaled = h3_memory.h3_expected_seconds(1536, 896, 124, forwards=8, upscale=True)
        stage_one = h3_memory.h3_expected_seconds(768, 448, 124, forwards=8)

        assert upscaled > stage_one
        assert upscaled - stage_one > 0
        assert single > stage_one

    def test_the_mlx_platform_delegates_to_the_mac_estimate(self):
        assert h3_memory.h3_expected_seconds(768, 448, 124, forwards=8, platform="mlx") == estimate_h3_mlx_seconds(768, 448, 124, forwards=8)

    @pytest.mark.parametrize(("kwargs", "message"), [({"forwards": 0}, "forwards"), ({"forwards": 8, "platform": "tpu"}, "platform"), ({"forwards": 8, "mode": "v2v"}, "mode")])
    def test_the_time_refuses_bad_arguments(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            h3_memory.h3_expected_seconds(864, 480, 124, **kwargs)

    def test_the_gpu_peak_of_the_reference_clip_is_the_decode_plus_the_desktop_share(self):
        estimate = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124)

        assert estimate.gpu_bytes == pytest.approx((5.857 + 9.48e-7 * 864 * 480 + 1.0) * GIB, abs=0.01 * GIB)

    # The process peaks of the M1 runs (the chunked feed-forward at 1280x736 and 345 frames); an estimate must not fall below them.
    @pytest.mark.parametrize(
        ("width", "height", "frames", "measured_gib"),
        [(864, 480, 73, 5.34), (864, 480, 107, 5.70), (864, 480, 124, 6.30), (1280, 736, 124, 7.84), (864, 480, 345, 8.90)],
    )
    def test_the_gpu_estimate_covers_the_measured_peak_without_overshooting_it_far(self, width, height, frames, measured_gib):
        estimate = h3_memory.estimate_h3_cuda_clip_memory(width, height, frames)

        assert measured_gib * GIB <= estimate.gpu_bytes <= (measured_gib + 2.0) * GIB

    def test_the_transformer_grows_151_kib_per_row_and_a_keyframe_adds_its_rows(self):
        plain = h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124)
        image = h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124, mode="i2v")

        assert image.gpu_bytes - plain.gpu_bytes == 151 * 1024 * 40 * 23
        assert h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124, mode="extend").gpu_bytes == plain.gpu_bytes

    def test_an_image_to_video_clip_adds_the_vision_tower_to_system_memory(self):
        plain = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124)
        image = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124, mode="i2v")

        assert image.system_bytes - plain.system_bytes == int(1.5 * GIB)
        assert h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124, mode="extend").system_bytes == plain.system_bytes

    def test_a_user_lora_adds_2_7_times_its_size_to_the_transformer_and_all_of_it_to_system_memory(self):
        lora = 310_000_000
        plain = h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124)
        loaded = h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124, loras_bytes=lora)

        assert loaded.gpu_bytes - plain.gpu_bytes == pytest.approx(2.7 * lora, abs=1024)
        assert loaded.system_bytes - plain.system_bytes == lora

    def test_a_lora_clip_at_the_reference_size_covers_the_measured_peak(self):
        assert h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124, loras_bytes=310_000_000).gpu_bytes >= 7.16 * GIB

    def test_system_memory_is_the_base_plus_the_decoded_frames(self):
        estimate = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124)

        assert estimate.system_bytes == int(17.7 * GIB) + 124 * 864 * 480 * 3

    def test_a_large_or_long_clip_fits_the_card_that_the_chunked_feed_forward_serves(self):
        large = h3_memory.h3_clip_fit(1280, 736, 124, self.BUDGET, forwards=8)
        long = h3_memory.h3_clip_fit(864, 480, 345, self.BUDGET, forwards=8)

        assert large["status"] == "fits"
        assert long["status"] == "tight"

    def test_an_upscale_needs_a_tier(self):
        with pytest.raises(ValueError, match="tier"):
            h3_memory.estimate_h3_cuda_clip_memory(768, 448, 124, upscale=True)

    def test_an_upscale_runs_the_transformer_at_half_size(self, tier):
        upscaled = h3_memory.estimate_h3_cuda_clip_memory(1536, 896, 124, upscale=True, seedvr2_tier=tier)
        half = h3_memory.estimate_h3_cuda_clip_memory(768, 448, 124)

        assert upscaled.gpu_bytes == half.gpu_bytes

    def test_the_system_peak_is_the_larger_phase_not_their_sum(self, tier):
        estimate = h3_memory.estimate_h3_cuda_clip_memory(768, 448, 124, upscale=True, seedvr2_tier=tier)
        seedvr2_phase = 12 * GIB + 124 * 384 * 224 * 3 + 124 * 768 * 448 * 3 * 5
        h3_phase = int(17.7 * GIB) + 124 * 384 * 224 * 3

        assert estimate.system_bytes == max(h3_phase, seedvr2_phase)
        assert estimate.system_bytes < h3_phase + 12 * GIB

    def test_system_bytes_without_a_tier_or_with_a_tier_without_a_system_need(self):
        from zvisiongenerator.utils.seedvr2 import SeedVR2Tier

        bare = SeedVR2Tier("seedvr2-3b", 1.0, 1.0, 1.0, 1.0)

        assert h3_memory.h3_cuda_system_bytes() == int(17.7 * GIB)
        assert h3_memory.h3_cuda_system_bytes(seedvr2_tier=bare) == int(17.7 * GIB)
        streaming = SeedVR2Tier("seedvr2-7b", 1, 1, 1, 1, system_gb=22)
        assert h3_memory.h3_cuda_system_bytes(loras_bytes=GIB, seedvr2_tier=streaming) == int(17.7 * GIB) + GIB
        assert h3_memory.h3_cuda_system_bytes(seedvr2_tier=streaming, seedvr2_frame_bytes=7 * GIB) == 19 * GIB

    def test_the_tier_is_the_one_the_upscale_picks(self, tier):
        from zvisiongenerator.seedvr2_loader import build_seedvr2_setup
        from zvisiongenerator.utils.config import load_config
        from zvisiongenerator.utils.seedvr2 import FP8_WEIGHTS

        setup = build_seedvr2_setup(load_config(), weights=FP8_WEIGHTS, budget_bytes=self.BUDGET.gpu_bytes, system_bytes=self.BUDGET.system_bytes, models_dir=Path("/nowhere"))

        assert setup.tier == tier

    def test_a_default_upscale_on_30_gib_is_not_too_large(self, tier):
        fit = h3_memory.h3_clip_fit(768, 448, 124, self.BUDGET, forwards=8, upscale=True, seedvr2_tier=tier)

        assert fit["status"] != "too_large"
        assert fit["resource"] in ("gpu", "system")

    def test_the_fit_carries_status_budget_resource_family_and_the_estimated_time(self):
        fit = h3_memory.h3_clip_fit(864, 480, 124, self.BUDGET, forwards=8)

        assert fit == {
            "status": "fits",
            "required_gb": 7.3,
            "budget_gb": 10.0,
            "kind": "discrete",
            "resource": "gpu",
            "family": "h3",
            "expected_seconds": h3_memory.h3_expected_seconds(864, 480, 124, forwards=8),
        }

    @pytest.mark.parametrize(
        ("gpu_gib", "system_gib", "status", "resource"),
        [(10, 30, "fits", "gpu"), (8, 30, "tight", "gpu"), (7, 30, "too_large", "gpu"), (10, 20, "tight", "system"), (10, 16, "too_large", "system")],
    )
    def test_the_fit_classifies_both_memories(self, gpu_gib, system_gib, status, resource):
        fit = h3_memory.h3_clip_fit(864, 480, 124, MemoryBudget(gpu_gib * GIB, system_bytes=system_gib * GIB), forwards=8)

        assert (fit["status"], fit["resource"]) == (status, resource)

    @pytest.mark.parametrize("budget", [None, MemoryBudget(0, system_bytes=GIB), MemoryBudget(24 * GIB, unified_bytes=32 * GIB)])
    def test_there_is_no_fit_without_a_discrete_budget(self, budget):
        assert h3_memory.h3_clip_fit(864, 480, 124, budget, forwards=8) is None

    def test_the_lora_files_are_summed_and_a_missing_path_counts_nothing(self, tmp_path):
        (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
        folder = tmp_path / "folder"
        folder.mkdir()
        (folder / "b.safetensors").write_bytes(b"y" * 5)

        assert h3_memory.h3_lora_bytes([tmp_path / "a.safetensors", folder, tmp_path / "missing"]) == 15

    def test_an_extension_window_carries_the_22_frame_tail(self):
        assert h3_memory.h3_extend_window_frames(34) == 56
