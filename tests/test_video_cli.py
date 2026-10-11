"""Tests for video CLI argument parsing and alignment helpers."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from zvisiongenerator.utils.alignment import align_ltx_frames, align_resolution
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.video_cli import _build_video_parser


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

    @pytest.mark.parametrize(
        ("argv", "dest", "expected"),
        [
            (["--lora", "lora1:0.5,lora2:0.8"], "lora", "lora1:0.5,lora2:0.8"),
            (["--ratio", "9:16"], "ratio", "9:16"),
            (["-s", "l"], "size", "l"),
            (["--size", "s"], "size", "s"),
            (["-W", "640"], "width", 640),
        ],
    )
    def test_flag_reaches_its_option(self, argv, dest, expected):
        assert getattr(_build_video_parser().parse_args(["-m", "model", *argv]), dest) == expected

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


class TestVideoCliExecution:
    """Verify executable CLI behavior beyond argument parsing."""

    def test_main_reports_unknown_model_guidance_with_platform_alias(self, monkeypatch, capsys):
        import zvisiongenerator.video_cli as video_cli

        monkeypatch.setattr(video_cli, "load_config", lambda: {"model_aliases": {}})
        monkeypatch.setattr(video_cli, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(
            video_cli,
            "detect_video_model",
            lambda _model: VideoModelInfo(
                family="unknown",
                backend="unknown",
                supports_i2v=False,
                default_fps=24,
                frame_alignment=1,
                resolution_alignment=1,
            ),
        )
        monkeypatch.setattr(video_cli.sys, "platform", "win32")

        with patch("sys.argv", ["ziv-video", "-m", "unknown/repo", "--prompt", "a dog"]):
            with pytest.raises(SystemExit, match="2"):
                video_cli.main()

        err = capsys.readouterr().err
        assert "ltx-2.3" in err
        assert "dgrauet/ltx" not in err
        assert "supported/configured HuggingFace LTX repo IDs" in err
        assert "local path containing 'ltx'" in err
        assert "compatible local path/HuggingFace LTX repo" not in err

    def test_main_rejects_remote_lora_before_backend_load(self, monkeypatch, capsys):
        import zvisiongenerator.video_cli as video_cli

        model_info = VideoModelInfo(
            family="ltx",
            backend="ltx",
            supports_i2v=True,
            default_fps=24,
            frame_alignment=8,
            resolution_alignment=32,
        )
        backend = MagicMock()

        monkeypatch.setattr(
            video_cli,
            "load_config",
            lambda: {
                "video_generation": {"default_ratio": "16:9", "default_size": "m"},
                "video_sizes": {"16:9": {"m": {"width": 704, "height": 448, "frames": 49}}},
                "video_model_presets": {"ltx": {"upscale": {}}},
                "model_aliases": {},
            },
        )
        monkeypatch.setattr(video_cli, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(video_cli, "detect_video_model", lambda _model: model_info)
        monkeypatch.setattr(
            video_cli,
            "resolve_video_defaults",
            lambda _family, _config, cli_overrides: {"steps": 8, "width": 704, "height": 448, "num_frames": 49, **cli_overrides},
        )
        monkeypatch.setattr(video_cli, "get_video_backend", lambda _family: backend)

        with patch("sys.argv", ["ziv-video", "-m", "ltx-2.3", "--prompt", "a dog", "--lora", "org/lora:0.8"]):
            with pytest.raises(SystemExit, match="2"):
                video_cli.main()

        err = capsys.readouterr().err
        assert "Remote HuggingFace LoRA references are not supported" in err
        assert "org/lora" in err
        backend.load_model.assert_not_called()

    def test_main_reports_platform_alias_mismatch_via_parser_error(self, monkeypatch):
        import zvisiongenerator.video_cli as video_cli

        monkeypatch.setattr(
            video_cli,
            "load_config",
            lambda: {
                "model_aliases": {
                    "ltx-4": {
                        "darwin": "dgrauet/ltx-2.3-mlx-q4",
                        "win32": {"message": "Alias 'ltx-4' is macOS-only. On Windows, use 'ltx-2.3' for the CUDA diffusers backend."},
                    }
                }
            },
        )
        monkeypatch.setattr(video_cli.sys, "platform", "win32")

        with patch("sys.argv", ["ziv-video", "-m", "ltx-4", "--prompt", "a dog"]):
            with pytest.raises(SystemExit, match="2"):
                video_cli.main()

    def test_main_reports_backend_load_errors_via_parser_error(self, monkeypatch, tmp_path):
        import zvisiongenerator.video_cli as video_cli

        model_info = VideoModelInfo(
            family="ltx",
            backend="ltx",
            supports_i2v=True,
            default_fps=24,
            frame_alignment=8,
            resolution_alignment=32,
        )
        backend = MagicMock()
        backend.load_model.side_effect = RuntimeError("CUDA is not available. The Linux diffusers video backend requires an NVIDIA GPU with CUDA support.")

        monkeypatch.setattr(
            video_cli,
            "load_config",
            lambda: {
                "video_generation": {"default_ratio": "16:9", "default_size": "m"},
                "video_sizes": {"16:9": {"m": {"width": 704, "height": 448, "frames": 49}}},
                "video_model_presets": {"ltx": {"upscale": {"default_upscale_steps": 8}}},
            },
        )
        monkeypatch.setattr(video_cli, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(video_cli, "detect_video_model", lambda _model: model_info)
        monkeypatch.setattr(
            video_cli,
            "resolve_video_defaults",
            lambda _family, _config, cli_overrides: {"steps": 8, "width": 704, "height": 448, "num_frames": 49, **cli_overrides},
        )
        monkeypatch.setattr(video_cli, "get_video_backend", lambda _family: backend)

        with patch("sys.argv", ["ziv-video", "-m", "ltx-2.3", "--prompt", "a dog", "-o", str(tmp_path)]):
            with pytest.raises(SystemExit, match="2"):
                video_cli.main()

    def test_main_caps_upscale_steps_before_running_batch(self, monkeypatch, tmp_path):
        import zvisiongenerator.video_cli as video_cli

        model_info = VideoModelInfo(
            family="ltx",
            backend="ltx",
            supports_i2v=True,
            default_fps=24,
            frame_alignment=8,
            resolution_alignment=32,
        )
        backend = MagicMock()
        backend.load_model.return_value = (MagicMock(), model_info)
        captured: dict[str, int] = {}

        monkeypatch.setattr(
            video_cli,
            "load_config",
            lambda: {
                "video_generation": {"default_ratio": "16:9", "default_size": "m"},
                "video_sizes": {"16:9": {"m": {"width": 704, "height": 448, "frames": 49}}},
                "video_model_presets": {"ltx": {"upscale": {"default_upscale_steps": 8}}},
            },
        )
        monkeypatch.setattr(video_cli, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(video_cli, "detect_video_model", lambda _model: model_info)
        monkeypatch.setattr(
            video_cli,
            "resolve_video_defaults",
            lambda _family, _config, cli_overrides: {
                "steps": cli_overrides.get("steps", 8),
                "width": 704,
                "height": 448,
                "num_frames": 49,
            },
        )
        monkeypatch.setattr(video_cli, "get_video_backend", lambda _family: backend)
        monkeypatch.setattr(video_cli, "build_video_workflow", lambda _args, **_kwargs: object())

        def _fake_run_video_batch(backend, model, model_info, workflow, prompts_data, config, args, *, plan, skip_signal=None):
            captured["steps"] = args.steps
            captured["upscale_steps"] = args.upscale_steps

        monkeypatch.setattr(video_cli, "run_video_batch", _fake_run_video_batch)

        with patch("sys.argv", ["ziv-video", "-m", "ltx-2.3-mlx-q4", "--prompt", "a dog", "--upscale", "2", "--steps", "12", "-o", str(tmp_path)]):
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                video_cli.main()

        assert captured["steps"] == 8
        assert captured["upscale_steps"] == 8
        assert any("max 8 denoising steps" in str(item.message) for item in seen)


class _VideoCliEnhancer:
    def __init__(self, order: list[str]):
        self.order = order

    def generate(self, messages, *, seed, max_tokens, temperature, cancelled=None):
        yield "A dog sprints across a sunlit beach, camera tracking low."

    def close(self) -> None:
        self.order.append("close")


class TestVideoCliPreflight:
    """ziv-video runs preflight (rewrites, enhancer unload, memory release) before loading the video model."""

    def _patch(self, monkeypatch, order: list[str], *, load_error=None):
        import zvisiongenerator.video_cli as video_cli
        from zvisiongenerator.backends.prompt_enhancer_session import PromptEnhancerSession

        model_info = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)
        backend = MagicMock()
        backend.load_model.side_effect = lambda *a, **k: (order.append("load_model"), (MagicMock(), model_info))[1]

        def _factory(repo, revision):
            if load_error is not None:
                raise load_error
            return _VideoCliEnhancer(order)

        session = PromptEnhancerSession(_factory, scheduler=lambda delay, callback: None, downloaded=lambda repo, revision: True)
        monkeypatch.setattr(
            video_cli,
            "load_config",
            lambda: {
                "video_generation": {"default_ratio": "16:9", "default_size": "m"},
                "video_sizes": {"16:9": {"m": {"width": 704, "height": 448, "frames": 49}}},
                "prompt_enhancer": {"model": {"darwin": "a/b", "win32": "a/b", "linux": "a/b"}},
            },
        )
        monkeypatch.setattr(video_cli, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(video_cli, "detect_video_model", lambda _model: model_info)
        monkeypatch.setattr(video_cli, "resolve_video_defaults", lambda _family, _config, cli_overrides: {"steps": 8, "width": 704, "height": 448, "num_frames": 49})
        monkeypatch.setattr(video_cli, "get_video_backend", lambda _family: backend)
        workflows: list[bool] = []
        monkeypatch.setattr(video_cli, "build_video_workflow", lambda _args, *, enhance: workflows.append(enhance) or object())
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.ensure_available", lambda repo, rev: True)
        monkeypatch.setattr("zvisiongenerator.backends.prompt_enhancer_session.get_prompt_enhancer_session", lambda: session)
        monkeypatch.setattr("zvisiongenerator.preflight.release_accelerator_memory", lambda: order.append("release_memory"))
        captured: dict = {}

        def _fake_run_video_batch(backend, model, model_info, workflow, prompts_data, config, args, *, plan, skip_signal=None):
            order.append("run_video_batch")
            captured["plan"] = plan

        monkeypatch.setattr(video_cli, "run_video_batch", _fake_run_video_batch)
        return video_cli, captured, workflows

    def test_preflight_before_model_load(self, monkeypatch, tmp_path):
        order: list[str] = []
        video_cli, captured, workflows = self._patch(monkeypatch, order)
        with patch("sys.argv", ["ziv-video", "-m", "ltx-2.3", "--prompt", "a dog runs", "--enhance", "-o", str(tmp_path)]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                video_cli.main()
        assert order == ["close", "release_memory", "load_model", "run_video_batch"]
        assert captured["plan"].has_rewrites and workflows == [True]

    def test_scene_reaches_the_rewrite(self, monkeypatch, tmp_path):
        order: list[str] = []
        video_cli, captured, _workflows = self._patch(monkeypatch, order)
        with patch("sys.argv", ["ziv-video", "-m", "ltx-2.3", "--scene", "A sunlit beach.", "--prompt", "a dog runs", "--enhance", "-o", str(tmp_path)]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                video_cli.main()
        (iteration,) = captured["plan"].iterations
        assert iteration.enhanced_prompt == "A dog sprints across a sunlit beach, camera tracking low."
        assert iteration.enhanced_scene == "A sunlit beach."

    def test_enhancer_load_error_exits_before_model_load(self, monkeypatch, tmp_path):
        order: list[str] = []
        video_cli, _captured, _workflows = self._patch(monkeypatch, order, load_error=RuntimeError("Could not load prompt enhancer model a/b"))
        with patch("sys.argv", ["ziv-video", "-m", "ltx-2.3", "--prompt", "a dog runs", "--enhance", "-o", str(tmp_path)]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(SystemExit, match="2"):
                    video_cli.main()
        assert order == []


class TestVideoCliScene:
    """``--scene`` reaches the plan; scene-only runs are valid; a run with no scene and no prompt is refused."""

    def _main(self, monkeypatch, tmp_path, argv):
        """Run ``main()`` with heavy steps patched out; return what the batch received, or ``None`` if the CLI exited first."""
        import zvisiongenerator.video_cli as video_cli

        model_info = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)
        backend = MagicMock()
        backend.load_model.return_value = (MagicMock(), model_info)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            video_cli, "load_config", lambda: {"video_generation": {"default_ratio": "16:9", "default_size": "m"}, "video_sizes": {"16:9": {"m": {"width": 704, "height": 448, "frames": 49}}}}
        )
        monkeypatch.setattr(video_cli, "resolve_model_path", lambda model, **_: model)
        monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(video_cli, "detect_video_model", lambda _model: model_info)
        monkeypatch.setattr(video_cli, "resolve_video_defaults", lambda _family, _config, cli_overrides: {"steps": 8, "width": 704, "height": 448, "num_frames": 49})
        monkeypatch.setattr(video_cli, "get_video_backend", lambda _family: backend)
        monkeypatch.setattr(video_cli, "build_video_workflow", lambda _args, **_kwargs: object())
        captured: dict = {}

        def _fake_run_video_batch(backend, model, model_info, workflow, prompts_data, config, args, *, plan, skip_signal=None):
            captured.update(prompts_data=prompts_data, args=args, plan=plan)

        monkeypatch.setattr(video_cli, "run_video_batch", _fake_run_video_batch)
        with patch("sys.argv", ["ziv-video", "-m", "ltx-2.3", "-o", str(tmp_path / "out"), *argv]):
            video_cli.main()
        return captured

    def test_parser_has_scene_flag_defaulting_to_none(self):
        parser = _build_video_parser()
        assert parser.parse_args(["-m", "model"]).scene is None
        assert parser.parse_args(["-m", "model", "--scene", "A castle."]).scene == "A castle."

    def test_prompts_file_flag_defaults_to_none(self):
        assert _build_video_parser().parse_args(["-m", "model"]).prompts_file is None

    def test_scene_alone_is_a_scene_only_run(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--scene", "A castle at dusk."])
        assert captured["prompts_data"] == {"prompt": [("", None)]}
        (iteration,) = captured["plan"].iterations
        assert iteration.scene == "A castle at dusk." and iteration.resolved_scene == "A castle at dusk." and iteration.resolved_prompt == ""
        assert captured["args"].scene == "A castle at dusk."
        assert captured["args"].scene_input is None and captured["args"].prompt_input is None

    def test_scene_with_prompt_reaches_the_plan(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--scene", "  A castle.  ", "--prompt", "a knight rides in"])
        assert captured["prompts_data"] == {"prompt": [("a knight rides in", None)]}
        (iteration,) = captured["plan"].iterations
        assert (iteration.scene, iteration.resolved_prompt) == ("A castle.", "a knight rides in")

    def test_scene_with_an_empty_prompt_is_allowed(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--scene", "A castle.", "--prompt", ""])
        assert captured["prompts_data"] == {"prompt": [("", None)]}
        assert captured["plan"].iterations[0].scene == "A castle."

    def test_empty_prompt_without_a_scene_is_refused(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--prompt", "  "])
        assert "--prompt must not be empty" in capsys.readouterr().err

    def test_blank_scene_and_empty_prompt_is_refused(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--scene", "  ", "--prompt", ""])
        assert "--prompt must not be empty" in capsys.readouterr().err

    def test_blank_scene_alone_falls_back_to_the_default_prompts_file(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--scene", ""])
        assert "Prompts file not found: prompts.yaml" in capsys.readouterr().err

    def test_prompt_alone_is_unchanged(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--prompt", "a dog"])
        assert captured["prompts_data"] == {"prompt": [("a dog", None)]}
        assert captured["args"].scene is None and captured["plan"].iterations[0].scene is None

    def test_no_prompt_source_still_reads_the_default_prompts_file(self, monkeypatch, tmp_path):
        (tmp_path / "prompts.yaml").write_text("s:\n  - prompt: from the default file\n")
        captured = self._main(monkeypatch, tmp_path, [])
        assert captured["prompts_data"] == {"s": [("from the default file", None)]}

    def test_prompts_file_entry_scene_beats_the_job_scene(self, monkeypatch, tmp_path):
        prompts = tmp_path / "mine.yaml"
        prompts.write_text("s:\n  - prompt: one\n    scene: Entry scene.\n  - prompt: two\n")
        captured = self._main(monkeypatch, tmp_path, ["-p", str(prompts), "--scene", "Job scene."])
        assert [it.scene for it in captured["plan"].iterations] == ["Entry scene.", "Job scene."]

    def test_prompts_file_without_scenes_is_unchanged(self, monkeypatch, tmp_path):
        prompts = tmp_path / "mine.yaml"
        prompts.write_text("s:\n  - prompt: one\n")
        captured = self._main(monkeypatch, tmp_path, ["-p", str(prompts)])
        assert captured["plan"].iterations[0].scene is None

    def test_prompt_overrides_a_prompts_file_and_scene_still_applies(self, monkeypatch, tmp_path):
        prompts = tmp_path / "mine.yaml"
        prompts.write_text("s:\n  - prompt: one\n    scene: Entry scene.\n")
        captured = self._main(monkeypatch, tmp_path, ["-p", str(prompts), "--prompt", "inline", "--scene", "Job scene."])
        assert captured["prompts_data"] == {"prompt": [("inline", None)]}
        assert captured["plan"].iterations[0].scene == "Job scene."


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
        result_w, result_h = align_resolution(w_in, h_in, 32, "Test")
        assert result_w % 32 == 0
        assert result_h % 32 == 0
        assert result_w == w_out
        assert result_h == h_out

    def test_already_aligned_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            align_resolution(704, 480, 32, "Test")
            assert len(w) == 0

    def test_unaligned_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            align_resolution(700, 475, 32, "Test")
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
        result = align_ltx_frames(frames_in)
        assert result == frames_out
        # Verify 8k+1 pattern
        assert (result - 1) % 8 == 0

    def test_valid_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            align_ltx_frames(49)
            assert len(w) == 0

    def test_invalid_emits_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            align_ltx_frames(50)
            assert len(w) == 1
            assert "8k+1" in str(w[0].message)
