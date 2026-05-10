"""Behavior tests for unified LoRA parsing across image and video CLIs."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import zvisiongenerator.image_cli as image_cli
import zvisiongenerator.utils as utils
import zvisiongenerator.video_cli as video_cli
from zvisiongenerator.utils.lora import parse_lora_arg


def _patch_image_main_dependencies(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    backend = MagicMock()
    backend.name = "stub-image-backend"
    backend.load_model = MagicMock(return_value=(object(), MagicMock(family="zimage", size=None)))

    monkeypatch.setattr(
        image_cli,
        "load_config",
        lambda: {
            "generation": {"default_ratio": "2:3", "default_size": "m"},
            "sizes": {"2:3": {"m": {"width": 832, "height": 1216}}},
            "model_aliases": {},
            "schedulers": {},
        },
    )
    monkeypatch.setattr(image_cli, "select_ratio_size_defaults", lambda *args, **kwargs: ("2:3", "m"))
    monkeypatch.setattr(image_cli, "resolve_model_path", lambda model, **kwargs: model)
    monkeypatch.setattr(image_cli, "resolve_defaults", lambda *args, **kwargs: {"steps": 10, "guidance": 0.5, "scheduler": None})
    monkeypatch.setattr(image_cli, "validate_scheduler", lambda *args, **kwargs: None)
    monkeypatch.setattr(image_cli, "detect_image_model", lambda _: MagicMock(family="zimage", size=None))
    monkeypatch.setattr(image_cli, "get_backend", lambda: backend)
    monkeypatch.setattr(image_cli, "resolve_lora_path", lambda name: f"/resolved/{Path(name).name}")
    monkeypatch.setattr(image_cli, "run_batch", MagicMock())
    monkeypatch.setattr(image_cli.os.path, "isfile", lambda _path: True)
    return backend


def _patch_video_main_dependencies(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    backend = MagicMock()
    backend.load_model = MagicMock(return_value=(object(), MagicMock(family="ltx")))

    monkeypatch.setattr(
        video_cli,
        "load_config",
        lambda: {
            "video_generation": {"default_ratio": "16:9", "default_size": "m"},
            "video_sizes": {"16:9": {"m": {"width": 768, "height": 432}}},
            "video_model_presets": {"ltx": {"upscale": {}}},
            "model_aliases": {},
        },
    )
    monkeypatch.setattr(video_cli, "select_ratio_size_defaults", lambda *args, **kwargs: ("16:9", "m"))
    monkeypatch.setattr(video_cli, "resolve_model_path", lambda model, **kwargs: model)
    monkeypatch.setattr(video_cli, "resolve_video_defaults", lambda *args, **kwargs: {"steps": 8, "width": 768, "height": 432, "num_frames": 49})
    monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(
        video_cli,
        "detect_video_model",
        lambda _: MagicMock(family="ltx", supports_i2v=True, backend="stub-video-backend", resolution_alignment=32, frame_alignment=8),
    )
    monkeypatch.setattr(video_cli, "resolve_lora_path", lambda name: f"/resolved/{Path(name).name}")
    monkeypatch.setattr(video_cli, "get_video_backend", lambda _backend_name: backend)
    monkeypatch.setattr(video_cli, "build_video_workflow", lambda _args: object())
    monkeypatch.setattr(video_cli, "run_video_batch", MagicMock())
    monkeypatch.setattr(video_cli.os, "makedirs", lambda *_args, **_kwargs: None)
    return backend


class TestParseLoraArgBehavior:
    def test_single_name_defaults_to_weight_one(self):
        assert parse_lora_arg("style") == [("style", 1.0)]

    def test_comma_separated_values_preserve_weights(self):
        assert parse_lora_arg("style:0.8,detail:0.5") == [("style", 0.8), ("detail", 0.5)]

    def test_invalid_numeric_suffix_is_treated_as_part_of_name(self):
        assert parse_lora_arg("style:notanumber") == [("style:notanumber", 1.0)]

    def test_whitespace_is_stripped_before_parsing(self):
        assert parse_lora_arg(" style : 0.8 , detail ") == [("style", 0.8), ("detail", 1.0)]

    def test_empty_entry_raises_value_error(self):
        with pytest.raises(ValueError, match="Empty LoRA entry"):
            parse_lora_arg("")

    def test_empty_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Empty LoRA name"):
            parse_lora_arg(":0.5")


class TestPublicImportContract:
    def test_utils_re_exports_parse_lora_arg(self):
        assert utils.parse_lora_arg is parse_lora_arg


class TestParserParity:
    @pytest.mark.parametrize("builder", [image_cli._build_parser, video_cli._build_video_parser], ids=["image", "video"])
    def test_lora_defaults_to_none(self, builder):
        args = builder().parse_args(["-m", "model"])
        assert args.lora is None

    @pytest.mark.parametrize("builder", [image_cli._build_parser, video_cli._build_video_parser], ids=["image", "video"])
    def test_lora_accepts_single_comma_separated_argument(self, builder):
        args = builder().parse_args(["-m", "model", "--lora", "style:0.8,detail"])
        assert args.lora == "style:0.8,detail"
        assert parse_lora_arg(args.lora) == [("style", 0.8), ("detail", 1.0)]

    def test_image_and_video_parsers_produce_matching_lora_values(self):
        lora_value = "lora1:0.9,lora2,lora3:0.3"
        image_args = image_cli._build_parser().parse_args(["-m", "model", "--lora", lora_value])
        video_args = video_cli._build_video_parser().parse_args(["-m", "model", "--lora", lora_value])

        assert image_args.lora == video_args.lora
        assert parse_lora_arg(image_args.lora) == [("lora1", 0.9), ("lora2", 1.0), ("lora3", 0.3)]

    @pytest.mark.parametrize("builder", [image_cli._build_parser, video_cli._build_video_parser], ids=["image", "video"])
    def test_missing_lora_value_exits(self, builder):
        with pytest.raises(SystemExit) as exc_info:
            builder().parse_args(["-m", "model", "--lora"])

        assert exc_info.value.code == 2


class TestRuntimeHandoff:
    def test_image_main_resolves_and_passes_loras_to_backend(self, monkeypatch: pytest.MonkeyPatch):
        backend = _patch_image_main_dependencies(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["ziv-image", "-m", "model", "--prompt", "a cat", "--lora", "style:0.8,detail"])

        image_cli.main()

        assert backend.load_model.call_args.kwargs["lora_paths"] == ["/resolved/style", "/resolved/detail"]
        assert backend.load_model.call_args.kwargs["lora_weights"] == [0.8, 1.0]

    def test_video_main_resolves_and_passes_loras_to_backend(self, monkeypatch: pytest.MonkeyPatch):
        backend = _patch_video_main_dependencies(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["ziv-video", "-m", "model", "--prompt", "a cat", "--lora", "style:0.8,detail"])

        video_cli.main()

        assert backend.load_model.call_args.kwargs["loras"] == [("/resolved/style", 0.8), ("/resolved/detail", 1.0)]


class TestInvalidLoraInputInCli:
    def test_image_main_rejects_invalid_lora_value(self, monkeypatch: pytest.MonkeyPatch):
        _patch_image_main_dependencies(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["ziv-image", "-m", "model", "--prompt", "a cat", "--lora", ""])

        with pytest.raises(SystemExit) as exc_info:
            image_cli.main()

        assert exc_info.value.code == 2

    def test_video_main_rejects_invalid_lora_value(self, monkeypatch: pytest.MonkeyPatch):
        _patch_video_main_dependencies(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["ziv-video", "-m", "model", "--prompt", "a cat", "--lora", ""])

        with pytest.raises(SystemExit) as exc_info:
            video_cli.main()

        assert exc_info.value.code == 2
