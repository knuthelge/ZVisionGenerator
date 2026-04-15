"""Tests for video CLI argument parsing and alignment helpers."""

from __future__ import annotations

import warnings

import pytest

from zvisiongenerator.video_cli import _align_ltx_frames, _align_resolution, _build_video_parser


# ---------------------------------------------------------------------------
# _build_video_parser
# ---------------------------------------------------------------------------


class TestBuildVideoParser:
    """Verify argument parser configuration and default handling."""

    def test_minimal_valid_args(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "dgrauet/ltx-2.3-mlx-q4"])
        assert args.model == "dgrauet/ltx-2.3-mlx-q4"
        assert args.runs == 1
        assert args.output == "."

    def test_all_args_parsed(self):
        parser = _build_video_parser()
        args = parser.parse_args(
            [
                "-m",
                "dgrauet/ltx-2.3-mlx-q4",
                "--prompt",
                "a sunset",
                "--image",
                "/tmp/img.png",
                "-W",
                "512",
                "-H",
                "384",
                "--frames",
                "25",
                "--steps",
                "50",
                "--seed",
                "123",
                "-r",
                "3",
                "-o",
                "/tmp/out",
                "--format",
                "mp4",
                "--no-low-memory",
            ]
        )
        assert args.model == "dgrauet/ltx-2.3-mlx-q4"
        assert args.prompt == "a sunset"
        assert args.image_path == "/tmp/img.png"
        assert args.width == 512
        assert args.height == 384
        assert args.frames == 25
        assert args.steps == 50
        assert args.seed == 123
        assert args.runs == 3
        assert args.output == "/tmp/out"
        assert args.format == "mp4"
        assert args.low_memory is False

    def test_model_defaults_to_none(self):
        parser = _build_video_parser()
        args = parser.parse_args([])
        assert args.model is None

    def test_lora_single(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--lora", "path/to/lora.safetensors"])
        assert args.lora == "path/to/lora.safetensors"

    def test_lora_with_strength(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--lora", "lora1:0.5,lora2:0.8"])
        assert args.lora == "lora1:0.5,lora2:0.8"

    def test_defaults_none_for_optional_args(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model"])
        assert args.width is None
        assert args.height is None
        assert args.frames is None
        assert args.steps is None
        assert args.seed is None
        assert args.prompt is None
        assert args.image_path is None
        assert args.ratio is None
        assert args.size is None

    def test_ratio_flag(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--ratio", "9:16"])
        assert args.ratio == "9:16"

    def test_size_flag(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "-s", "l"])
        assert args.size == "l"

    def test_size_long_flag(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--size", "s"])
        assert args.size == "s"

    def test_ratio_and_size_together(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--ratio", "1:1", "-s", "m"])
        assert args.ratio == "1:1"
        assert args.size == "m"

    def test_ratio_with_width_override(self):
        parser = _build_video_parser()
        args = parser.parse_args(["-m", "model", "--ratio", "16:9", "-W", "640"])
        assert args.ratio == "16:9"
        assert args.width == 640


# ---------------------------------------------------------------------------
# _align_resolution
# ---------------------------------------------------------------------------


class TestAlignResolution:
    """Verify resolution alignment to multiples of a given divisor."""

    @pytest.mark.parametrize(
        ("w_in", "h_in", "w_out", "h_out"),
        [
            (704, 480, 704, 480),  # already aligned
            (512, 512, 512, 512),  # already aligned
            (700, 475, 704, 480),  # rounded up
            (710, 490, 704, 480),  # rounded to nearest multiple of 32
            (100, 100, 96, 96),  # small values
        ],
        ids=["already-aligned-704x480", "already-aligned-512x512", "rounded-700x475", "rounded-710x490", "small-100x100"],
    )
    def test_alignment(self, w_in, h_in, w_out, h_out):
        result_w, result_h = _align_resolution(w_in, h_in, 32, "Test")
        assert result_w % 32 == 0
        assert result_h % 32 == 0
        assert result_w == w_out
        assert result_h == h_out

    def test_already_aligned_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _align_resolution(704, 480, 32, "Test")
            assert len(w) == 0

    def test_unaligned_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _align_resolution(700, 475, 32, "Test")
            assert len(w) == 1
            assert "divisible by 32" in str(w[0].message)


# ---------------------------------------------------------------------------
# _align_ltx_frames
# ---------------------------------------------------------------------------


class TestAlignLtxFrames:
    """Verify LTX frame alignment to 8k+1 pattern."""

    @pytest.mark.parametrize(
        ("frames_in", "frames_out"),
        [
            (9, 9),  # 8*1+1 = 9
            (17, 17),  # 8*2+1 = 17
            (49, 49),  # 8*6+1 = 49
            (97, 97),  # 8*12+1 = 97
            (121, 121),  # 8*15+1 = 121
            (50, 49),  # rounds to 49
            (48, 49),  # rounds to 49
            (10, 9),  # rounds to 9
            (14, 17),  # rounds to 17 (14-1=13, 13/8=1.625, round=2, 8*2+1=17)
            (1, 9),  # min k=1 → 9
        ],
        ids=["9", "17", "49", "97", "121", "50->49", "48->49", "10->9", "14->17", "1->9"],
    )
    def test_alignment(self, frames_in, frames_out):
        result = _align_ltx_frames(frames_in)
        assert result == frames_out
        # Verify 8k+1 pattern
        assert (result - 1) % 8 == 0

    def test_valid_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _align_ltx_frames(49)
            assert len(w) == 0

    def test_invalid_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _align_ltx_frames(50)
            assert len(w) == 1
            assert "8k+1" in str(w[0].message)
