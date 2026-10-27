"""Behavior tests for unified LoRA parsing across image and video CLIs."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests._factories import _patch_image_main_dependencies, load_packaged_config

import zvisiongenerator.image_cli as image_cli
import zvisiongenerator.utils.lora as lora_utils
import zvisiongenerator.video_cli as video_cli
from zvisiongenerator.utils.lora import parse_lora_arg


def _patch_video_main_dependencies(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    backend = MagicMock()
    backend.load_model = MagicMock(return_value=(object(), MagicMock(family="ltx")))

    monkeypatch.setattr(
        video_cli,
        "load_config",
        lambda: {
            "video_generation": {"default_ratio": "16:9", "default_size": "m"},
            "video_sizes": {"16:9": {"m": {"width": 768, "height": 432}}},
            "video_model_presets": {"ltx": {**load_packaged_config()["video_model_presets"]["ltx"], "upscale": {}}},
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
    monkeypatch.setattr(lora_utils, "resolve_lora_path", lambda name: f"/resolved/{Path(name).name}")
    monkeypatch.setattr(lora_utils.os.path, "exists", lambda path: str(path).startswith("/resolved/") or os.path.lexists(path))
    monkeypatch.setattr(video_cli, "get_video_backend", lambda _backend_name: backend)
    monkeypatch.setattr(video_cli, "build_video_workflow", lambda _args, **_kwargs: object())
    monkeypatch.setattr(video_cli, "run_video_batch", MagicMock())
    monkeypatch.setattr(video_cli.os, "makedirs", lambda *_args, **_kwargs: None)
    return backend


class TestParserParity:
    @pytest.mark.parametrize("builder", [image_cli._build_parser, video_cli._build_video_parser], ids=["image", "video"])
    def test_lora_accepts_single_comma_separated_argument(self, builder):
        args = builder().parse_args(["-m", "model", "--lora", "style:0.8,detail"])
        assert args.lora == "style:0.8,detail"
        assert parse_lora_arg(args.lora) == [("style", 0.8), ("detail", 1.0)]


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

    def test_image_main_rejects_remote_lora_before_backend_load(self, monkeypatch: pytest.MonkeyPatch):
        backend = _patch_image_main_dependencies(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["ziv-image", "-m", "model", "--prompt", "a cat", "--lora", "org/lora:0.8"])

        with pytest.raises(SystemExit) as exc_info:
            image_cli.main()

        assert exc_info.value.code == 2
        backend.load_model.assert_not_called()


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
