"""Tests for the MiniMax H3 row and frame arithmetic and clip geometry."""

from __future__ import annotations


import pytest

from zvisiongenerator.backends.cuda.video.h3.frames import Geometry, geometry
from zvisiongenerator.utils import h3_memory
from zvisiongenerator.utils.lora_size import lora_files_bytes
from zvisiongenerator.utils.h3_memory import (
    H3_TEXT_TOKENS,
    estimate_h3_mlx_clip_memory,
    estimate_h3_mlx_seconds,
    h3_audio_latents,
    h3_frames_of_latent_frames,
    h3_latent_frames,
    h3_nominal_references,
    h3_reference_block_rows,
    h3_refine_windows,
    h3_sequence_rows,
    h3_video_rows,
)
from zvisiongenerator.utils.memory_offsets import CUDA_OFFSET_LINUX_BYTES
from zvisiongenerator.utils.memory_verdict import MemoryJob, MemoryWeights, judge
from zvisiongenerator.utils.model_memory import GIB, MemoryBudget


def _h3_fit(width, height, frames, budget, *, forwards, mode="t2v", loras_bytes=0, upscale=False, review=False, references=0):
    """Judge an H3 clip the way every surface does and return the figures the estimates are asserted on; ``None`` without a discrete budget."""
    if budget is None or not budget.gpu_bytes or budget.system_bytes is None:
        return None
    job = MemoryJob(media="video", family="h3", width=width, height=height, frames=frames, mode=mode, references=references, lora_bytes=loras_bytes, upscale=upscale, review=review, forwards=forwards)
    verdict = judge(job, budget, MemoryWeights())
    return {
        "status": verdict.status,
        "required_gb": round(verdict.gpu_bytes / GIB, 1),
        "budget_gb": round(budget.gpu_bytes / GIB, 1),
        "kind": "discrete",
        "resource": "system" if verdict.resource == "ram" else verdict.resource,
        "family": "h3",
        "expected_seconds": verdict.expected_seconds,
    }


def _h3_mac_fit(width, height, frames, budget, *, forwards, mode="t2v", references=0):
    """Judge an H3 clip on a Mac the way every surface does; ``None`` without a unified budget."""
    if budget is None or not budget.unified_bytes or budget.system_bytes is not None:
        return None
    verdict = judge(MemoryJob(media="video", family="h3", width=width, height=height, frames=frames, mode=mode, references=references, forwards=forwards), budget, MemoryWeights())
    return {"status": verdict.status, "required_gb": round(verdict.gpu_bytes / GIB, 1), "budget_gb": round(budget.unified_bytes / GIB, 1), "expected_seconds": verdict.expected_seconds}


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
        (1024, 576, 21_312, 21_826),
        (1600, 896, 51_800, 52_314),
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


SQUARE = (1024, 1024)
WIDE = (1792, 1024)


@pytest.mark.parametrize(("size", "rows"), [(SQUARE, 1024 + 1024 + 8), (WIDE, 1792 + 1792 + 8)])
def test_a_reference_adds_its_latent_rows_and_its_vision_tokens(size, rows):
    assert h3_reference_block_rows(*size) == rows


def test_a_references_latent_rows_are_those_the_ref2va_layout_packs():
    import torch

    from zvisiongenerator.utils.h3_ref2va import ref2va_sequence

    for width, height in (SQUARE, WIDE):
        shape = (1, height // 16, width // 16)  # one latent frame at a sixteenth of the size
        layout = ref2va_sequence([1] * 3, [shape], latent_frames=2, latent_height=4, latent_width=4, audio_latents=1, anchors=[], patch=(1, 2, 2), audio_channels=2, audio_tag=2, video_tag=0)
        assert isinstance(layout.position_ids, torch.Tensor)
        assert layout.keyframe_rows == h3_reference_block_rows(width, height) - (width // 32) * (height // 32) - 8


def test_references_add_their_blocks_to_the_sequence():
    base = h3_sequence_rows(864, 480, 124)

    assert h3_sequence_rows(864, 480, 124, references=[SQUARE]) == base + 2056
    assert h3_sequence_rows(864, 480, 124, references=[SQUARE, WIDE]) == base + 2056 + 3592
    assert h3_sequence_rows(864, 480, 124, keyframes=1, references=[WIDE]) == h3_sequence_rows(864, 480, 124, keyframes=1) + 3592


def test_a_tail_beside_references_is_shown_to_the_text_encoder_as_one_canvas_sized_block():
    cell = (864 // 32) * (480 // 32)
    without = h3_sequence_rows(864, 480, 124, references=[SQUARE])

    assert h3_sequence_rows(864, 480, 124, tail_frames=22, references=[SQUARE]) == without + cell + 14
    assert h3_sequence_rows(864, 480, 124, tail_frames=22) == h3_sequence_rows(864, 480, 124)


def test_references_of_a_known_count_share_the_canvas_aspect():
    assert h3_nominal_references(0, 1344, 768) == ()
    assert h3_nominal_references(2, 1344, 768) == (WIDE, WIDE)
    assert h3_nominal_references(1, 768, 1344) == ((1024, 1792),)


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


@pytest.mark.parametrize(("latent_frames", "frames"), [(2, 5), (7, 22), (22, 73), (37, 124), (72, 243)])
def test_frames_of_latent_frames_inverts_latent_frames(latent_frames, frames):
    assert h3_frames_of_latent_frames(latent_frames) == frames
    assert h3_latent_frames(frames) == latent_frames


@pytest.mark.parametrize("latent_frames", [0, 1, 3, 21, 23])
def test_frames_of_latent_frames_refuses_other_counts(latent_frames):
    with pytest.raises(ValueError, match="5 \\* n \\+ 2"):
        h3_frames_of_latent_frames(latent_frames)


@pytest.mark.parametrize("latent_frames", [2, 7, 12, 22])
def test_a_clip_of_22_latent_frames_or_fewer_is_one_window(latent_frames):
    assert h3_refine_windows(latent_frames) == ((0, latent_frames),)


@pytest.mark.parametrize(
    ("latent_frames", "windows"),
    [
        (27, ((0, 22), (5, 27))),
        (37, ((0, 22), (15, 37))),
        (72, ((0, 22), (15, 37), (30, 52), (45, 67), (50, 72))),
    ],
)
def test_a_longer_clip_gets_22_frame_windows_every_15_with_the_last_pulled_back(latent_frames, windows):
    assert h3_refine_windows(latent_frames) == windows


def test_the_windows_cover_the_clip_in_order_and_are_all_22_long():
    # Up to 150 latent frames: past the 481-frame clip (142 latent frames).
    for latent_frames in range(23, 151):
        windows = h3_refine_windows(latent_frames)

        assert windows[0][0] == 0, latent_frames
        assert windows[-1][1] == latent_frames, latent_frames
        assert all(end - start == 22 for start, end in windows), latent_frames
        assert all(earlier[0] < later[0] < earlier[1] for earlier, later in zip(windows, windows[1:], strict=False)), latent_frames


class TestMacMemory:
    """The macOS (MLX) estimate, from the process's peak footprint."""

    def test_the_two_measured_text_to_video_clips_are_reproduced(self):
        assert h3_sequence_rows(512, 320, 124) == 6_434
        assert estimate_h3_mlx_clip_memory(512, 320, 124) == pytest.approx(12.03 * GIB, abs=0.01 * GIB)
        assert estimate_h3_mlx_clip_memory(768, 448, 22) == pytest.approx(11.33 * GIB, abs=0.01 * GIB)
        assert estimate_h3_mlx_clip_memory(768, 448, 124, mode="t2v") == estimate_h3_mlx_clip_memory(768, 448, 124)

    def test_text_to_video_grows_by_a_fixed_amount_per_row(self):
        small, large = (512, 320, 124), (1280, 704, 124)
        rows = h3_sequence_rows(*large) - h3_sequence_rows(*small)

        assert (estimate_h3_mlx_clip_memory(*large) - estimate_h3_mlx_clip_memory(*small)) / rows == pytest.approx(186.5 * 1024, rel=1e-3)

    def test_a_small_clip_shrinks_below_the_measured_one(self):
        assert estimate_h3_mlx_clip_memory(512, 288, 22) < estimate_h3_mlx_clip_memory(512, 320, 124)

    @pytest.mark.parametrize("mode", ["i2v", "extend"])
    def test_an_image_mode_is_the_conditioning_phase_when_that_is_larger(self, mode):
        assert estimate_h3_mlx_clip_memory(576, 320, 73, mode=mode) == pytest.approx(15.34 * GIB, abs=1)

    @pytest.mark.parametrize("mode", ["i2v", "extend"])
    def test_an_image_mode_is_the_denoise_phase_with_its_keyframe_when_that_is_larger(self, mode):
        large = estimate_h3_mlx_clip_memory(1920, 1088, 243, mode=mode)

        assert large > 15.34 * GIB
        assert large > estimate_h3_mlx_clip_memory(1920, 1088, 243)  # it also holds the keyframe's rows

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
        ("unified_gib", "mode", "status", "required_gb"),
        [
            (24, "t2v", "fits", 13.2),
            (24, "i2v", "fits", 15.3),
            (32, "t2v", "fits", 13.2),
            (16, "t2v", "tight", 13.2),
            (12, "t2v", "too_large", 13.2),
            (18, "i2v", "tight", 15.3),
            (14, "extend", "too_large", 15.3),
        ],
    )
    def test_the_fit_classifies_against_the_total_memory(self, unified_gib, mode, status, required_gb):
        fit = _h3_mac_fit(768, 448, 124, MemoryBudget(8 * GIB, unified_bytes=unified_gib * GIB), forwards=8, mode=mode)

        assert fit == {"status": status, "required_gb": required_gb, "budget_gb": float(unified_gib), "expected_seconds": estimate_h3_mlx_seconds(768, 448, 124, forwards=8, mode=mode)}

    def test_an_image_mode_can_be_tight_where_text_to_video_fits(self):
        budget = MemoryBudget(int(12.8 * GIB), unified_bytes=int(19.5 * GIB))

        assert _h3_mac_fit(576, 320, 73, budget, forwards=8)["status"] == "fits"
        assert _h3_mac_fit(576, 320, 73, budget, forwards=8, mode="i2v")["status"] == "tight"

    def test_text_to_video_with_references_runs_the_staged_path_so_the_conditioning_phase_sets_a_floor(self):
        plain = estimate_h3_mlx_clip_memory(576, 320, 73)
        with_reference = estimate_h3_mlx_clip_memory(576, 320, 73, references=[SQUARE])

        assert plain < 15.34 * GIB
        assert with_reference == pytest.approx(15.34 * GIB, abs=1)
        assert estimate_h3_mlx_clip_memory(576, 320, 73, mode="i2v") == with_reference

    def test_the_reference_rows_grow_the_denoise_phase_by_the_per_row_amount(self):
        plain = estimate_h3_mlx_clip_memory(1280, 704, 243)
        with_references = estimate_h3_mlx_clip_memory(1280, 704, 243, references=[SQUARE, WIDE])

        assert with_references - plain == int(186.5 * 1024) * (2056 + 3592)

    def test_an_extension_has_no_tail_block_on_the_mac(self):
        assert estimate_h3_mlx_clip_memory(1280, 704, 243, mode="extend", references=[SQUARE]) == estimate_h3_mlx_clip_memory(1280, 704, 243, mode="i2v", references=[SQUARE])

    def test_references_take_longer(self):
        assert estimate_h3_mlx_seconds(768, 448, 124, forwards=8, references=[SQUARE]) > estimate_h3_mlx_seconds(768, 448, 124, forwards=8)

    def test_the_fit_counts_the_references_at_the_canvas_aspect(self):
        budget = MemoryBudget(16 * GIB, unified_bytes=32 * GIB)
        sizes = h3_nominal_references(2, 768, 448)
        fit = _h3_mac_fit(768, 448, 124, budget, forwards=8, references=2)

        assert fit["required_gb"] == round(estimate_h3_mlx_clip_memory(768, 448, 124, references=sizes) / GIB, 1)
        assert fit["required_gb"] > _h3_mac_fit(768, 448, 124, budget, forwards=8)["required_gb"]
        assert fit["expected_seconds"] == estimate_h3_mlx_seconds(768, 448, 124, forwards=8, references=sizes)

    @pytest.mark.parametrize("budget", [None, MemoryBudget(0), MemoryBudget(10 * GIB, system_bytes=32 * GIB)])
    def test_there_is_no_fit_without_a_unified_budget(self, budget):
        assert _h3_mac_fit(768, 448, 124, budget, forwards=8) is None


class TestCudaEstimate:
    """The CUDA calibration: GPU peak, system peak and time of an H3 clip on a discrete GPU."""

    BUDGET = MemoryBudget(10 * GIB, system_bytes=32 * GIB)  # a clip's process memory is about 25 GiB, which reads Fits up to 80 % of 32 GiB

    # Measured generation time of clean runs, excluding the one-time weight preparation.
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

    def test_an_image_mode_adds_the_encoder_time_and_a_lora_slows_each_forward(self):
        plain = h3_memory.h3_expected_seconds(864, 480, 124, forwards=8)

        assert h3_memory.h3_expected_seconds(864, 480, 124, forwards=8, mode="i2v") > plain + 30
        assert h3_memory.h3_expected_seconds(864, 480, 124, forwards=8, loras=True) > plain

    def test_an_extension_encodes_its_tail_like_a_keyframe_without_extra_rows(self):
        t2v = h3_memory.h3_expected_seconds(864, 480, 124, forwards=8)

        assert h3_memory.h3_expected_seconds(864, 480, 124, forwards=8, mode="extend") == t2v + 30

    # The measured chunked totals of the upscale's phases (stage 1, net, transformer load, refine, decode): 424, 598 and 1,069 s.
    @pytest.mark.parametrize(("width", "height", "frames", "seconds"), [(1280, 704, 124, 424), (1600, 896, 124, 598), (1600, 896, 243, 1069)])
    def test_the_upscale_time_matches_the_measured_chunked_totals_within_15_percent(self, width, height, frames, seconds):
        assert h3_memory.h3_expected_seconds(width, height, frames, forwards=8, upscale=True) == pytest.approx(seconds, rel=0.15)

    def test_the_refine_pays_two_forwards_for_each_window(self):
        one = h3_memory.h3_expected_seconds(1280, 704, 73, forwards=8, upscale=True)
        two = h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8, upscale=True)
        five = h3_memory.h3_expected_seconds(1280, 704, 243, forwards=8, upscale=True)

        assert one < two < five

    def test_an_image_mode_upscale_encodes_the_images_a_second_time_at_the_full_size(self):
        t2v = h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8, upscale=True)
        i2v = h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8, mode="i2v", upscale=True)
        keyframe_rows = h3_memory._cuda_rows(640, 352, 124, "i2v") - h3_memory._cuda_rows(640, 352, 124, "t2v")
        assert keyframe_rows > 0
        assert i2v - t2v >= 2 * h3_memory._CUDA_IMAGE_ENCODE_SECONDS

    def test_a_lora_slows_the_refine_forwards_as_well_as_stage_one(self):
        plain = h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8, upscale=True)
        loaded = h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8, upscale=True, loras=True)
        stage_one_only = h3_memory.h3_expected_seconds(640, 352, 124, forwards=8, loras=True) - h3_memory.h3_expected_seconds(640, 352, 124, forwards=8)

        assert loaded - plain > stage_one_only

    def test_a_larger_upscale_takes_longer(self):
        assert h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8, upscale=True) < h3_memory.h3_expected_seconds(1600, 896, 124, forwards=8, upscale=True)

    def test_the_mlx_platform_delegates_to_the_mac_estimate(self):
        assert h3_memory.h3_expected_seconds(768, 448, 124, forwards=8, platform="mlx") == estimate_h3_mlx_seconds(768, 448, 124, forwards=8)

    @pytest.mark.parametrize(("kwargs", "message"), [({"forwards": 0}, "forwards"), ({"forwards": 8, "platform": "tpu"}, "platform"), ({"forwards": 8, "mode": "v2v"}, "mode")])
    def test_the_time_refuses_bad_arguments(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            h3_memory.h3_expected_seconds(864, 480, 124, **kwargs)

    def test_the_gpu_peak_of_the_reference_clip_is_the_video_decode(self):
        estimate = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124)

        assert estimate.gpu_bytes == pytest.approx(h3_memory._CUDA_DECODE_BASE_BYTES + h3_memory._CUDA_DECODE_BYTES_PER_PIXEL * 864 * 480, abs=0.01 * GIB)

    # Measured figures, in the quantity the bench compares (a probed case: the middle of its hold-probe bracket; else the allocated peak plus the CUDA offset): h3-5s (512x320) and h3-s (768x448 at 22 frames), 4.77 GiB allocated at both;
    # h3-5s-864 (4.98 allocated); then the three the transformer sets: h3-10s-864 (6.20, the stand-in of 5.80 allocated plus the offset, not probed), h3-5s-1280 (6.65, bracket 6.50..6.80) and h3-14s-864 (7.95, bracket 7.81..8.09).
    # The estimate plus the CUDA offset is within the bench's tolerance of each, and the transformer's three within 0.15 GiB (a line through three points that do not lie on one).
    @pytest.mark.parametrize(
        ("width", "height", "frames", "figure_gib"),
        [(512, 320, 124, 4.77 + 0.40), (768, 448, 22, 4.77 + 0.40), (864, 480, 124, 4.98 + 0.40), (864, 480, 243, 6.20), (1280, 736, 124, 6.65), (864, 480, 345, 7.95)],
    )
    def test_the_gpu_estimate_is_within_the_bench_tolerance_of_the_measured_figure(self, width, height, frames, figure_gib):
        figure = h3_memory.estimate_h3_cuda_clip_memory(width, height, frames).gpu_bytes + CUDA_OFFSET_LINUX_BYTES
        measured = figure_gib * GIB

        assert abs(figure - measured) <= max(0.10 * measured, GIB)
        if frames > 124 or width > 864:
            assert abs(figure - measured) <= 0.15 * GIB

    def test_two_references_take_the_text_encoders_turn_to_the_measured_peak(self):
        """A 5 s clip at 512x320 with two references (h3-5s-refs) peaked at 6.45 GiB allocated, 1.68 GiB above the plain clip."""
        plain = h3_memory.estimate_h3_cuda_clip_memory(512, 320, 124)
        referenced = h3_memory.estimate_h3_cuda_clip_memory(512, 320, 124, references=h3_nominal_references(2, 512, 320))

        assert abs(referenced.gpu_bytes + CUDA_OFFSET_LINUX_BYTES - (6.45 * GIB + CUDA_OFFSET_LINUX_BYTES)) <= 0.1 * GIB
        assert referenced.gpu_bytes > plain.gpu_bytes

    def test_the_transformer_grows_150_kib_per_row_and_a_keyframe_adds_its_rows(self):
        plain = h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124)
        image = h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124, mode="i2v")

        assert image.gpu_bytes - plain.gpu_bytes == 150 * 1024 * 40 * 23
        assert h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124, mode="extend").gpu_bytes == plain.gpu_bytes

    def test_an_image_to_video_clip_adds_the_vision_tower_to_system_memory(self):
        plain = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124)
        image = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124, mode="i2v")

        assert image.system_bytes - plain.system_bytes == h3_memory._CUDA_VISION_SYSTEM_BYTES
        assert h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124, mode="extend").system_bytes == plain.system_bytes

    def test_references_add_their_rows_to_the_transformer_in_every_mode(self):
        plain = h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124)
        per_row = 150 * 1024

        assert h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124, references=[WIDE]).gpu_bytes - plain.gpu_bytes == per_row * 3592
        assert h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124, references=[SQUARE, WIDE]).gpu_bytes - plain.gpu_bytes == per_row * (2056 + 3592)
        image = h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124, mode="i2v")
        assert h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124, mode="i2v", references=[WIDE]).gpu_bytes - image.gpu_bytes == per_row * 3592
        # An extension with references also shows its tail to the text encoder: one block of the canvas's rows.
        extension = h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124, mode="extend", references=[WIDE])
        assert extension.gpu_bytes - plain.gpu_bytes == per_row * (3592 + 40 * 23 + 14)

    @pytest.mark.parametrize("mode", ["t2v", "extend"])
    def test_references_load_the_vision_tower_whatever_the_mode(self, mode):
        plain = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124, mode=mode)
        referenced = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124, mode=mode, references=[SQUARE])

        assert referenced.system_bytes - plain.system_bytes == h3_memory._CUDA_VISION_SYSTEM_BYTES

    def test_the_vision_tower_is_counted_once_for_a_keyframe_and_references(self):
        image = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124, mode="i2v")

        assert h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124, mode="i2v", references=[SQUARE]).system_bytes == image.system_bytes

    def test_references_add_time_in_their_rows_and_one_encode_to_text_to_video(self):
        plain = h3_memory.h3_expected_seconds(864, 480, 124, forwards=8)
        rows = h3_sequence_rows(864, 480, 124)
        added = 8 * (h3_memory._cuda_forward_seconds(rows + 2056) - h3_memory._cuda_forward_seconds(rows)) + h3_memory._CUDA_IMAGE_ENCODE_SECONDS

        assert h3_memory.h3_expected_seconds(864, 480, 124, forwards=8, references=[SQUARE]) == pytest.approx(plain + added, abs=1)

    def test_an_image_mode_with_references_pays_the_encode_once_not_twice(self):
        plain = h3_memory.h3_expected_seconds(864, 480, 124, forwards=8, mode="i2v")
        rows = h3_sequence_rows(864, 480, 124, keyframes=1)
        added = 8 * (h3_memory._cuda_forward_seconds(rows + 2056) - h3_memory._cuda_forward_seconds(rows))

        assert h3_memory.h3_expected_seconds(864, 480, 124, forwards=8, mode="i2v", references=[SQUARE]) == pytest.approx(plain + added, abs=1)

    def test_references_can_turn_an_upscale_that_keeps_the_transformer_into_one_that_loads_twice(self):
        card = None
        for gib in (x / 4 for x in range(24, 80)):
            if h3_memory.h3_upscale_keeps_transformer(1344, 768, 124, int(gib * GIB)) and not h3_memory.h3_upscale_keeps_transformer(1344, 768, 124, int(gib * GIB), references=[WIDE, WIDE]):
                card = int(gib * GIB)
                break

        assert card is not None
        fit = _h3_fit(1344, 768, 124, MemoryBudget(card, system_bytes=32 * GIB), forwards=8, upscale=True, references=2)
        plain = _h3_fit(1344, 768, 124, MemoryBudget(card, system_bytes=32 * GIB), forwards=8, upscale=True)
        assert plain["expected_seconds"] == h3_memory.h3_expected_seconds(1344, 768, 124, forwards=8, upscale=True, keep_transformer=True)
        assert fit["expected_seconds"] == h3_memory.h3_expected_seconds(1344, 768, 124, forwards=8, upscale=True, references=[WIDE, WIDE])

    def test_the_fit_counts_the_references_at_the_canvas_aspect(self):
        sizes = h3_nominal_references(2, 1344, 768)
        fit = _h3_fit(1344, 768, 124, self.BUDGET, forwards=8, references=2)
        estimate = h3_memory.estimate_h3_cuda_clip_memory(1344, 768, 124, references=sizes)

        assert fit["required_gb"] == round((estimate.gpu_bytes + CUDA_OFFSET_LINUX_BYTES) / GIB, 1)
        assert fit["expected_seconds"] == h3_memory.h3_expected_seconds(1344, 768, 124, forwards=8, references=sizes)

    def test_an_extension_with_two_references_is_flagged_on_a_card_where_it_fits_without(self):
        def figure(references: int) -> int:
            sizes = h3_nominal_references(references, 1344, 768)
            return h3_memory.estimate_h3_cuda_clip_memory(1344, 768, 90, mode="extend", references=sizes).gpu_bytes + CUDA_OFFSET_LINUX_BYTES

        card = MemoryBudget(int((figure(1) + figure(2)) / 2 / 0.9), system_bytes=40 * GIB)  # the fits line lies between one reference and two
        plain = _h3_fit(1344, 768, 90, card, forwards=8, mode="extend")
        one = _h3_fit(1344, 768, 90, card, forwards=8, mode="extend", references=1)
        two = _h3_fit(1344, 768, 90, card, forwards=8, mode="extend", references=2)

        assert (plain["status"], one["status"]) == ("fits", "fits")
        assert two["status"] != "fits"

    def test_a_user_lora_adds_a_share_of_its_size_to_the_transformer_and_a_multiple_of_it_to_system_memory(self):
        lora = 310_000_000
        plain = h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124)
        loaded = h3_memory.estimate_h3_cuda_clip_memory(1280, 736, 124, loras_bytes=lora)

        assert loaded.gpu_bytes - plain.gpu_bytes == pytest.approx(h3_memory._CUDA_DIT_LORA_BYTES_PER_BYTE * lora, abs=1024)
        assert loaded.system_bytes - plain.system_bytes == pytest.approx(h3_memory._CUDA_SYSTEM_LORA_BYTES_PER_BYTE * lora, abs=1)

    def test_a_user_lora_does_not_lift_a_small_clips_gpu_peak_above_the_decode(self):
        lora = 1_400_000_000
        plain = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124)
        loaded = h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124, loras_bytes=lora)

        assert loaded.gpu_bytes == plain.gpu_bytes

    def test_a_large_or_long_clip_fits_the_card_that_the_chunked_feed_forward_serves(self):
        large = _h3_fit(1280, 736, 124, self.BUDGET, forwards=8)
        long = _h3_fit(864, 480, 345, self.BUDGET, forwards=8)

        assert large["status"] == "fits"
        assert long["status"] == "fits"

    def test_an_upscale_needs_no_tier(self):
        assert h3_memory.estimate_h3_cuda_clip_memory(1280, 704, 124, upscale=True).gpu_bytes > 0

    def test_the_upscale_gpu_estimate_covers_the_full_size_decode_peak(self):
        # 5.7 GiB: the full-size decode at 1280x704, a reserved peak.
        assert h3_memory.estimate_h3_cuda_clip_memory(1280, 704, 124, upscale=True).gpu_bytes + CUDA_OFFSET_LINUX_BYTES >= 5.7 * GIB

    def test_the_upscale_gpu_estimate_is_the_refines_allocated_peak(self):
        # h3-5s-1600-upscale: 5 s at 1600x896, rendered at half size and refined at the full size, peaked at 5.97 GiB allocated (the refine's transformer).
        assert abs(h3_memory.estimate_h3_cuda_clip_memory(1600, 896, 124, upscale=True).gpu_bytes - 5.97 * GIB) <= 0.1 * GIB

    @pytest.mark.parametrize(("width", "height"), [(1280, 704), (1600, 896)])
    def test_the_large_upscales_fit_a_10_gib_card(self, width, height):
        estimate = h3_memory.estimate_h3_cuda_clip_memory(width, height, 124, upscale=True)
        fit = _h3_fit(width, height, 124, self.BUDGET, forwards=8, upscale=True)

        assert estimate.gpu_bytes <= 9.0 * GIB
        assert fit["status"] == "fits"
        assert fit["resource"] == "gpu"

    @pytest.mark.parametrize(
        ("kwargs", "extra"),
        [
            ({}, 124 * 864 * 480 * 3),
            ({"upscale": True}, 124 * 1280 * 704 * 3),
            ({"mode": "i2v", "loras_bytes": GIB, "upscale": True}, h3_memory._CUDA_VISION_SYSTEM_BYTES + h3_memory._CUDA_SYSTEM_LORA_BYTES_PER_BYTE * GIB + 124 * 1280 * 704 * 3),
        ],
    )
    def test_the_system_bytes_are_the_base_the_vision_tower_the_loras_and_the_decoded_frames(self, kwargs, extra):
        width, height = (1280, 704) if kwargs.get("upscale") else (864, 480)

        assert h3_memory.estimate_h3_cuda_clip_memory(width, height, 124, **kwargs).system_bytes == pytest.approx(h3_memory._CUDA_SYSTEM_BASE_BYTES + extra, abs=1)
        assert h3_memory.h3_cuda_system_bytes() == h3_memory._CUDA_SYSTEM_BASE_BYTES
        assert (
            h3_memory.h3_cuda_system_bytes(loras_bytes=GIB, frame_bytes=2 * GIB, vision=True)
            == h3_memory._CUDA_SYSTEM_BASE_BYTES + h3_memory._CUDA_VISION_SYSTEM_BYTES + int(h3_memory._CUDA_SYSTEM_LORA_BYTES_PER_BYTE * GIB) + 2 * GIB
        )

    def test_a_clip_of_73_frames_or_fewer_refines_at_its_own_rows(self):
        short = h3_memory.estimate_h3_cuda_clip_memory(1600, 896, 39, upscale=True)
        full_window = h3_memory.estimate_h3_cuda_clip_memory(1600, 896, 73, upscale=True)

        assert short.gpu_bytes <= full_window.gpu_bytes

    # The peaks of the latent upscaler measured alone: (width, height, frames, GiB).
    @pytest.mark.parametrize(("width", "height", "frames", "measured_gib"), [(1280, 704, 124, 1.71), (1600, 896, 124, 2.28), (1600, 896, 243, 4.14), (1600, 896, 481, 7.44)])
    def test_the_upsampler_stage_covers_the_measured_peaks(self, width, height, frames, measured_gib):
        assert h3_memory._cuda_upsampler_bytes(width, height, frames) >= measured_gib * GIB

    def test_the_upsampler_stage_grows_linearly_with_the_latent_frames(self):
        steps = [h3_memory._cuda_upsampler_bytes(1600, 896, frames + 17) - h3_memory._cuda_upsampler_bytes(1600, 896, frames) for frames in (124, 141, 158)]

        assert steps[0] > 0
        assert steps[0] == pytest.approx(steps[1]) == pytest.approx(steps[2])

    def test_the_upsampler_stage_is_part_of_the_gpu_peak(self, monkeypatch):
        monkeypatch.setattr(h3_memory, "_cuda_upsampler_bytes", lambda width, height, frames: 20 * GIB)

        assert h3_memory.estimate_h3_cuda_clip_memory(1280, 704, 124, upscale=True).gpu_bytes == 20 * GIB
        assert h3_memory.estimate_h3_cuda_clip_memory(1280, 704, 124).gpu_bytes < 20 * GIB

    def test_a_default_upscale_on_30_gib_fits_with_the_gpu_as_the_limit(self):
        fit = _h3_fit(1280, 704, 124, self.BUDGET, forwards=8, upscale=True)

        assert fit["status"] == "fits"
        assert fit["resource"] == "gpu"
        assert fit["expected_seconds"] == h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8, upscale=True, keep_transformer=True)

    # --- an upscale that keeps the transformer loaded between its passes

    def test_keeping_the_transformer_saves_no_time_without_an_upscale(self):
        assert h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8, keep_transformer=True) == h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8)

    @pytest.mark.parametrize(("width", "height"), [(1280, 704), (1600, 896)])
    @pytest.mark.parametrize("mode", ["t2v", "i2v"])
    def test_a_5_s_large_upscale_fits_a_10_gib_card_with_the_transformer_resident(self, width, height, mode):
        kept = h3_memory.estimate_h3_cuda_clip_memory(width, height, 124, mode=mode, upscale=True, keep_transformer=True)
        loaded_twice = h3_memory.estimate_h3_cuda_clip_memory(width, height, 124, mode=mode, upscale=True)

        assert kept.gpu_bytes >= loaded_twice.gpu_bytes
        assert kept.gpu_bytes <= 9.0 * GIB
        assert kept.system_bytes == loaded_twice.system_bytes
        assert h3_memory.h3_upscale_keeps_transformer(width, height, 124, 10 * GIB, mode=mode)

    def test_keeping_the_transformer_changes_nothing_without_an_upscale(self):
        assert h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124, keep_transformer=True) == h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124)

    def test_a_clip_whose_resident_stages_would_not_fit_loads_twice(self, monkeypatch):
        monkeypatch.setattr(h3_memory, "_cuda_upsampler_bytes", lambda width, height, frames: 7 * GIB)

        assert not h3_memory.h3_upscale_keeps_transformer(1280, 704, 124, 10 * GIB)
        assert h3_memory.h3_upscale_keeps_transformer(1280, 704, 124, 24 * GIB)

    def test_a_card_too_small_for_the_clip_loads_twice(self):
        assert not h3_memory.h3_upscale_keeps_transformer(1600, 896, 124, 6 * GIB)

    def test_the_fit_of_an_upscale_without_review_skips_the_second_load(self):
        no_review = _h3_fit(1280, 704, 124, self.BUDGET, forwards=8, upscale=True)
        review = _h3_fit(1280, 704, 124, self.BUDGET, forwards=8, upscale=True, review=True)

        assert no_review["expected_seconds"] == h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8, upscale=True, keep_transformer=True)
        assert review["expected_seconds"] == h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8, upscale=True)
        assert review["expected_seconds"] - no_review["expected_seconds"] == h3_memory._CUDA_REFINE_LOAD_SECONDS

    def test_the_fit_loads_twice_where_the_single_load_does_not_fit(self, monkeypatch):
        monkeypatch.setattr(h3_memory, "_cuda_upsampler_bytes", lambda width, height, frames: 7 * GIB)

        fit = _h3_fit(1280, 704, 124, self.BUDGET, forwards=8, upscale=True)

        assert fit["expected_seconds"] == h3_memory.h3_expected_seconds(1280, 704, 124, forwards=8, upscale=True)

    def test_the_review_flag_does_not_change_a_clip_without_an_upscale(self):
        assert _h3_fit(864, 480, 124, self.BUDGET, forwards=8, review=True) == _h3_fit(864, 480, 124, self.BUDGET, forwards=8)

    def test_the_fit_carries_status_budget_resource_family_and_the_estimated_time(self):
        fit = _h3_fit(864, 480, 124, self.BUDGET, forwards=8)

        assert fit == {
            "status": "fits",
            "required_gb": round((h3_memory.estimate_h3_cuda_clip_memory(864, 480, 124).gpu_bytes + CUDA_OFFSET_LINUX_BYTES) / GIB, 1),
            "budget_gb": 10.0,
            "kind": "discrete",
            "resource": "gpu",
            "family": "h3",
            "expected_seconds": h3_memory.h3_expected_seconds(864, 480, 124, forwards=8),
        }

    @pytest.mark.parametrize(
        ("gpu_gib", "system_gib", "status", "resource"),
        [(10, 32, "fits", "gpu"), (5.5, 32, "tight", "gpu"), (5, 32, "too_large", "gpu"), (10, 17, "tight", "system"), (10, 15, "too_large", "system")],
    )
    def test_the_fit_classifies_both_memories(self, gpu_gib, system_gib, status, resource):
        fit = _h3_fit(864, 480, 124, MemoryBudget(int(gpu_gib * GIB), system_bytes=system_gib * GIB), forwards=8)

        assert (fit["status"], fit["resource"]) == (status, resource)

    @pytest.mark.parametrize("budget", [None, MemoryBudget(0, system_bytes=GIB), MemoryBudget(24 * GIB, unified_bytes=32 * GIB)])
    def test_there_is_no_fit_without_a_discrete_budget(self, budget):
        assert _h3_fit(864, 480, 124, budget, forwards=8) is None

    def test_the_lora_files_are_summed_and_a_missing_path_counts_nothing(self, tmp_path):
        (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
        folder = tmp_path / "folder"
        folder.mkdir()
        (folder / "b.safetensors").write_bytes(b"y" * 5)

        assert lora_files_bytes([tmp_path / "a.safetensors", folder, tmp_path / "missing"]) == 15
