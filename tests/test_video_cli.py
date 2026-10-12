"""Tests for video CLI argument parsing and alignment helpers."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_extend_source

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
        """The reply has no markers, so the scene is not rewritten: the plan carries no enhanced scene (it once echoed the scene)."""
        order: list[str] = []
        video_cli, captured, _workflows = self._patch(monkeypatch, order)
        with patch("sys.argv", ["ziv-video", "-m", "ltx-2.3", "--scene", "A sunlit beach.", "--prompt", "a dog runs", "--enhance", "-o", str(tmp_path)]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                video_cli.main()
        (iteration,) = captured["plan"].iterations
        assert iteration.enhanced_prompt == "A dog sprints across a sunlit beach, camera tracking low."
        assert iteration.enhanced_scene is None and iteration.scene == "A sunlit beach."

    def test_enhancer_load_error_exits_before_model_load(self, monkeypatch, tmp_path):
        order: list[str] = []
        video_cli, _captured, _workflows = self._patch(monkeypatch, order, load_error=RuntimeError("Could not load prompt enhancer model a/b"))
        with patch("sys.argv", ["ziv-video", "-m", "ltx-2.3", "--prompt", "a dog runs", "--enhance", "-o", str(tmp_path)]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(SystemExit, match="2"):
                    video_cli.main()
        assert order == []


def _cli_source(**overrides):
    return make_extend_source(**{"path": Path("clip.mp4"), "height": 288, "frame_count": 120, "scene": "A rainy street.", "prompt": "She waves.", **overrides})


def _run_cli(
    monkeypatch,
    tmp_path,
    argv,
    *,
    backend,
    source=None,
    load_error=None,
    config=None,
    local_models=("org/recorded-model", "ltx-8"),
    extend_supported=True,
    fit=None,
):
    """Run ``main()`` with heavy steps patched out; return what the batch received plus the backend and memory-check mocks."""
    import zvisiongenerator.video_cli as video_cli

    model_info = VideoModelInfo(family="ltx", backend="ltx", supports_i2v=True, default_fps=24, frame_alignment=8, resolution_alignment=32)
    backend.load_model.return_value = (MagicMock(), model_info)
    memory = MagicMock(return_value=fit)
    monkeypatch.chdir(tmp_path)
    app_config = {"ui": {"default_models": {"video": "ltx-8"}}} if config is None else config
    monkeypatch.setattr(video_cli, "load_config", lambda: app_config)
    monkeypatch.setattr(video_cli, "resolve_model_path", lambda model, **_: model)
    monkeypatch.setattr(video_cli, "find_local_model_dir", lambda ref: tmp_path if ref in local_models else None)
    monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(video_cli, "supports_video_extend", lambda: extend_supported)
    monkeypatch.setattr(video_cli, "supports_video_audio", lambda: True)
    monkeypatch.setattr(video_cli, "detect_video_model", lambda _model: model_info)
    monkeypatch.setattr(video_cli, "resolve_video_defaults", lambda _family, _config, cli_overrides: {"steps": 8, "width": 704, "height": 448, "num_frames": 49})
    monkeypatch.setattr(video_cli, "get_video_backend", lambda _family: backend)
    monkeypatch.setattr(video_cli, "get_accelerator_memory_budget", lambda: MagicMock(gpu_bytes=16 * 1024**3))
    monkeypatch.setattr(video_cli, "window_memory_fit", memory)
    monkeypatch.setattr(video_cli, "build_video_workflow", lambda _args, **_kwargs: object())
    loaded: list[str] = []

    def _load_source(path):
        loaded.append(path)
        if load_error is not None:
            raise load_error
        return source or _cli_source()

    monkeypatch.setattr(video_cli, "load_extend_source", _load_source)
    captured: dict = {"backend": backend, "memory": memory, "loaded": loaded}

    def _fake_run_video_batch(backend, model, model_info, workflow, prompts_data, config, args, *, plan, skip_signal=None):
        captured.update(prompts_data=prompts_data, args=args, plan=plan)

    monkeypatch.setattr(video_cli, "run_video_batch", _fake_run_video_batch)
    with patch("sys.argv", ["ziv-video", "-o", str(tmp_path / "out"), *argv]):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            try:
                video_cli.main()
            finally:
                captured["warnings"] = [str(item.message) for item in seen]
                captured["memory_warnings"] = [message for message in captured["warnings"] if "GB" in message]
    return captured


class TestVideoCliScene:
    """``--scene`` reaches the plan; scene-only runs are valid; a run with no scene and no prompt is refused."""

    def _main(self, monkeypatch, tmp_path, argv):
        return _run_cli(monkeypatch, tmp_path, ["-m", "ltx-2.3", *argv], backend=MagicMock())

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


class TestVideoCliExtend:
    """``--extend`` loads and checks the source before the model, takes defaults from it, and sets the args the extend workflow reads."""

    _source = staticmethod(_cli_source)

    def _main(self, monkeypatch, tmp_path, argv, **options):
        """Run ``main()`` through :func:`_run_cli`; ``self.backend`` holds the backend mock even when the CLI exits first."""
        backend = self.backend = MagicMock()
        return _run_cli(monkeypatch, tmp_path, argv, backend=backend, **options)

    def test_parser_extend_options(self):
        args = _build_video_parser().parse_args(["--extend", "clip.mp4", "--extend-seconds", "1.5"])
        assert (args.extend_path, args.extend_seconds, args.extend_frames) == ("clip.mp4", 1.5, None)
        args = _build_video_parser().parse_args(["--extend", "clip.mp4", "--extend-frames", "40"])
        assert (args.extend_seconds, args.extend_frames) == (None, 40)

    def test_extend_length_options_are_exclusive(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-seconds", "2", "--extend-frames", "48"])
        assert "not allowed with argument" in capsys.readouterr().err

    @pytest.mark.parametrize("flag", ["--extend-seconds", "--extend-frames"])
    def test_length_without_extend_is_refused(self, monkeypatch, tmp_path, capsys, flag):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["-m", "m", "--prompt", "a dog", flag, "16"])
        assert f"{flag} requires --extend" in capsys.readouterr().err

    def test_extend_frames_below_eight_is_refused(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-frames", "4"])
        assert "--extend-frames must be at least 8" in capsys.readouterr().err

    def test_extend_seconds_must_be_positive(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-seconds", "0"])
        assert "--extend-seconds must be a finite number greater than 0" in capsys.readouterr().err

    @pytest.mark.parametrize("seconds", ["nan", "inf"])
    def test_extend_seconds_must_be_finite(self, monkeypatch, tmp_path, capsys, seconds):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-seconds", seconds])
        assert "--extend-seconds must be a finite number greater than 0" in capsys.readouterr().err
        self.backend.load_model.assert_not_called()

    @pytest.mark.parametrize("path", ["", "   "])
    def test_a_blank_extend_path_is_refused(self, monkeypatch, tmp_path, capsys, path):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", path, "--prompt", "a dog"])
        assert "--extend needs a video path" in capsys.readouterr().err
        self.backend.load_model.assert_not_called()

    def test_capability_gate_refuses_extend_before_the_source_is_read(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], extend_supported=False)
        assert "macOS only" in capsys.readouterr().err
        self.backend.load_model.assert_not_called()

    def test_capability_gate_does_not_touch_other_runs(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a dog"], extend_supported=False)
        assert captured["args"].extend_source is None and captured["loaded"] == []

    @pytest.mark.parametrize(
        ("extra", "flag"),
        [
            (["--upscale", "2"], "--upscale"),
            (["--image", "pic.png"], "--image"),
            (["--ratio", "16:9"], "--ratio"),
            (["--size", "m"], "--size"),
            (["-W", "512"], "--width"),
            (["-H", "288"], "--height"),
            (["--frames", "49"], "--frames"),
        ],
    )
    def test_options_that_choose_a_size_are_refused(self, monkeypatch, tmp_path, capsys, extra, flag):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", *extra])
        assert f"{flag} cannot be used with --extend" in capsys.readouterr().err
        self.backend.load_model.assert_not_called()

    def test_source_problems_are_printed_and_exit_before_the_model_loads(self, monkeypatch, tmp_path, capsys):
        error = ValueError("clip.mp4 cannot be extended: The video is HDR. The video is interlaced.")
        with pytest.raises(SystemExit) as exit_info:
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], load_error=error)
        assert exit_info.value.code == 2
        err = capsys.readouterr().err
        assert "The video is HDR." in err and "The video is interlaced." in err
        self.backend.load_model.assert_not_called()

    def test_missing_source_file_exits_before_the_model_loads(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "gone.mp4"], load_error=FileNotFoundError("Video not found: gone.mp4"))
        assert "Video not found: gone.mp4" in capsys.readouterr().err
        self.backend.load_model.assert_not_called()

    @pytest.mark.parametrize(
        ("argv", "overrides", "expected"),
        [
            (["--prompt", ""], {}, ""),
            ([], {}, ""),  # a source recorded with a scene continues from an empty action
            ([], {"scene": None, "prompt": None}, ""),
            ([], {"scene": None, "prompt": 'She says "Hi." She waves.'}, 'She says "Hi." She waves.'),  # a legacy source continues from its own prompt
            (["--prompt", ""], {"scene": None, "prompt": "She waves."}, ""),
        ],
        ids=["explicit empty", "omitted", "no recorded prompt", "omitted for a legacy source", "explicit empty for a legacy source"],
    )
    def test_an_omitted_or_empty_prompt_plans_one_iteration(self, monkeypatch, tmp_path, argv, overrides, expected):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", *argv], source=self._source(**overrides))
        assert captured["prompts_data"] == {"prompt": [(expected, None)]}
        (iteration,) = captured["plan"].iterations
        assert iteration.resolved_prompt == expected
        assert not any("Still the original prompt" in message for message in captured["warnings"])

    def test_a_prompts_file_is_not_replaced_by_a_legacy_prompt(self, monkeypatch, tmp_path):
        prompts = tmp_path / "mine.yaml"
        prompts.write_text("s:\n  - prompt: one\n")
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "-p", str(prompts)], source=self._source(scene=None, prompt="She waves."))
        assert captured["prompts_data"] == {"s": [("one", None)]}

    def test_a_prompt_is_used_as_given(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--prompt", "She turns away."])
        assert captured["prompts_data"] == {"prompt": [("She turns away.", None)]}

    def test_a_prompts_file_is_read_when_given(self, monkeypatch, tmp_path):
        prompts = tmp_path / "mine.yaml"
        prompts.write_text("s:\n  - prompt: one\n  - prompt: two\n")
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "-p", str(prompts)])
        assert captured["prompts_data"] == {"s": [("one", None), ("two", None)]}

    def test_a_prompts_file_entry_without_a_prompt_is_refused_by_the_loader(self, monkeypatch, tmp_path, capsys):
        prompts = tmp_path / "mine.yaml"
        prompts.write_text("s:\n  - prompt: one\n  - prompt: ''\n")
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "-p", str(prompts)])
        assert "empty prompt" in capsys.readouterr().err
        self.backend.load_model.assert_not_called()

    _KEEP_ALL = "style=keep,mood=keep,details=,length=same,motion=,sound="

    def _preflight_spy(self, monkeypatch):
        """Replace the preflight with a spy so no enhancer loads; the spy's ``call_args`` holds what the CLI planned."""
        import zvisiongenerator.video_cli as video_cli

        spy = MagicMock(return_value=MagicMock(cancelled=False, has_rewrites=False))
        monkeypatch.setattr(video_cli, "run_preflight", spy)
        return spy

    def test_keep_everything_runs_when_the_prompt_splits_into_scene_and_action(self, monkeypatch, tmp_path):
        from zvisiongenerator.utils.prompt_enhance import EnhanceSettings

        self._preflight_spy(monkeypatch)
        args = self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a fox", "--enhance", self._KEEP_ALL])["args"]
        assert args.enhance == EnhanceSettings(style="keep", mood="keep", details=(), length="same", motion=(), sound=())

    def test_parts_action_is_parsed_and_is_not_a_noop_with_a_scene(self, monkeypatch, tmp_path):
        self._preflight_spy(monkeypatch)
        argv = ["-m", "ltx-8", "--scene", "A castle.", "--prompt", "a fox", "--enhance", f"{self._KEEP_ALL},parts=action"]
        assert self._main(monkeypatch, tmp_path, argv)["args"].enhance.parts == "action"

    def test_parts_stays_auto_so_an_extension_with_a_scene_writes_the_action(self, monkeypatch, tmp_path):
        """The default is resolved per job (see ``resolve_parts``): the settings stay unset (auto) and an extension picks the action."""
        self._preflight_spy(monkeypatch)
        args = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--prompt", "she waves", "--enhance"])["args"]
        assert args.enhance.parts is None

    def test_an_invalid_parts_value_is_refused(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a fox", "--enhance", "parts=scene"])
        assert "both, action" in capsys.readouterr().err

    def test_keep_everything_is_refused_as_a_noop_with_a_scene(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--scene", "A castle.", "--prompt", "a fox", "--enhance", self._KEEP_ALL])
        assert "Nothing to enhance" in capsys.readouterr().err

    def test_keep_everything_runs_for_an_extension_even_with_a_scene(self, monkeypatch, tmp_path):
        self._preflight_spy(monkeypatch)
        args = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--scene", "A castle.", "--prompt", "she waves", "--enhance", self._KEEP_ALL])["args"]
        assert args.enhance is not None

    def test_an_entry_with_keep_everything_settings_is_kept_for_an_extension(self, monkeypatch, tmp_path):
        import zvisiongenerator.video_cli as video_cli
        from zvisiongenerator.utils.prompt_enhance import EnhanceSettings

        keep_all = EnhanceSettings(style="keep", mood="keep", details=(), length="same", motion=(), sound=())
        prompts = tmp_path / "mine.yaml"
        prompts.write_text("s:\n  - prompt: one\n")
        inspection = MagicMock(prompts_data={"s": [("one", None)]}, scene_by_set={"s": ["Entry scene."]}, enhance_by_set={"s": [keep_all]})
        monkeypatch.setattr(video_cli, "inspect_prompts_file", lambda _path: inspection)
        spy = self._preflight_spy(monkeypatch)
        self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "-p", str(prompts)])
        assert spy.call_args.kwargs["enhance_by_set"] == {"s": [keep_all]}

    def test_a_missing_prompts_file_is_an_error_only_when_given(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "-p", str(tmp_path / "nope.yaml")])
        assert "Prompts file not found" in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("fps", "seconds", "frames"),
        [(24.0, "2", 48), (30.0, "2", 64), (25.0, "4", 96), (24.0, "0.1", 8)],
    )
    def test_seconds_convert_at_the_source_frame_rate(self, monkeypatch, tmp_path, fps, seconds, frames):
        args = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-seconds", seconds], source=self._source(fps=fps))["args"]
        assert (args.extend_frames, args.num_frames, args.fps) == (frames, frames + 25, fps)

    def test_seconds_that_do_not_land_on_a_step_warn(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-seconds", "2"], source=self._source(fps=30.0))
        assert any("--extend-seconds" in message for message in captured["warnings"])

    def test_seconds_on_a_step_do_not_warn(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-seconds", "2"])
        assert not any("--extend-seconds" in message for message in captured["warnings"])

    @pytest.mark.parametrize(("requested", "snapped"), [(48, 48), (50, 48), (53, 56), (9, 8)])
    def test_frames_snap_to_a_multiple_of_eight(self, monkeypatch, tmp_path, requested, snapped):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-frames", str(requested)])
        assert captured["args"].extend_frames == snapped and captured["args"].num_frames == snapped + 25
        assert any("--extend-frames" in message for message in captured["warnings"]) is (requested != snapped)

    def test_args_for_the_extend_workflow(self, monkeypatch, tmp_path):
        source = self._source()
        args = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], source=source)["args"]
        assert args.extend_source is source
        assert (args.width, args.height) == (512, 288)
        assert (args.extend_frames, args.num_frames, args.fps) == (48, 73, 24.0)
        assert args.ratio is None and args.size is None and args.upscale_steps is None

    def test_size_is_not_aligned_or_resolved(self, monkeypatch, tmp_path):
        args = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], source=self._source(width=96, height=64))["args"]
        assert (args.width, args.height) == (96, 64)

    def test_scene_defaults_to_the_sources_scene(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--prompt", "She leaves."])
        assert captured["args"].scene == "A rainy street."
        assert captured["plan"].iterations[0].scene == "A rainy street."

    def test_an_explicit_scene_wins(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--scene", " A beach. "])
        assert captured["args"].scene == "A beach."

    def test_an_explicit_empty_scene_means_no_scene(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--scene", "", "--prompt", "She leaves."])
        assert captured["args"].scene is None and captured["plan"].iterations[0].scene is None

    def test_a_legacy_source_has_no_default_scene(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--prompt", "She leaves."], source=self._source(scene=None, prompt="She waves."))
        assert captured["args"].scene is None

    def test_a_legacy_prompt_left_unchanged_warns(self, monkeypatch, tmp_path):
        source = self._source(scene=None, prompt="She waves,  then smiles.")
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--prompt", "She waves, then smiles."], source=source)
        assert any("Still the original prompt" in message for message in captured["warnings"])

    def test_a_legacy_prompt_that_was_edited_does_not_warn(self, monkeypatch, tmp_path):
        source = self._source(scene=None, prompt="She waves.")
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--prompt", "She sits down."], source=source)
        assert not any("Still the original prompt" in message for message in captured["warnings"])

    def test_the_original_prompt_of_a_scene_bearing_source_does_not_warn(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--prompt", "She waves."])
        assert not any("Still the original prompt" in message for message in captured["warnings"])

    def test_model_defaults_to_the_sources_model(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], source=self._source(model="org/recorded-model"))
        assert captured["args"].model == "org/recorded-model"
        assert captured["backend"].load_model.call_args.args[0] == "org/recorded-model"
        assert captured["backend"].load_model.call_args.kwargs["mode"] == "extend"
        assert not any("not available locally" in message for message in captured["warnings"])

    def test_an_explicit_model_wins(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "-m", "org/other"], source=self._source(model="org/recorded-model"))
        assert captured["args"].model == "org/other"

    def test_an_unavailable_source_model_falls_back_to_the_config_default_with_a_warning(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], source=self._source(model="org/not-downloaded"))
        assert captured["args"].model == "ltx-8"
        assert any("org/not-downloaded" in message and "not available locally" in message for message in captured["warnings"])

    def test_a_source_without_a_model_uses_the_config_default_silently(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], source=self._source(model=None))
        assert captured["args"].model == "ltx-8"
        assert not any("not available locally" in message for message in captured["warnings"])

    def test_no_model_anywhere_is_an_error(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], source=self._source(model=None), config={})
        assert "--model is required" in capsys.readouterr().err
        self.backend.load_model.assert_not_called()

    def test_loras_default_to_the_sources_existing_loras(self, monkeypatch, tmp_path):
        kept = tmp_path / "kept.safetensors"
        kept.write_text("x")
        source = self._source(lora=f"{kept}:0.8,{tmp_path / 'gone.safetensors'}:0.5")
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], source=source)
        assert captured["args"].lora_paths == [str(kept)] and captured["args"].lora_weights == [0.8]
        assert captured["backend"].load_model.call_args.kwargs["loras"] == [(str(kept), 0.8)]
        assert any("gone.safetensors" in message and "not found" in message for message in captured["warnings"])

    def test_all_loras_missing_means_none(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], source=self._source(lora=f"{tmp_path / 'gone.safetensors'}:0.5"))
        assert captured["args"].lora_paths is None
        assert captured["backend"].load_model.call_args.kwargs["loras"] is None

    def test_explicit_loras_win(self, monkeypatch, tmp_path):
        mine = tmp_path / "mine.safetensors"
        mine.write_text("x")
        source = self._source(lora=f"{tmp_path / 'gone.safetensors'}:0.5")
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--lora", f"{mine}:0.3"], source=source)
        assert captured["args"].lora_paths == [str(mine)] and captured["args"].lora_weights == [0.3]
        assert not any("gone.safetensors" in message for message in captured["warnings"])

    def test_a_window_that_does_not_fit_warns_and_the_run_continues(self, monkeypatch, tmp_path):
        fit = {"status": "too_large", "required_gb": 40.2, "budget_gb": 16.0}
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "-m", "org/recorded-model"], fit=fit)
        (warning,) = captured["memory_warnings"]
        assert "40.2 GB" in warning and "too large" in warning and "--extend-seconds" in warning and "--extend-frames" in warning
        assert "args" in captured
        call = captured["memory"].call_args
        assert call.args == ("org/recorded-model",)
        assert (call.kwargs["width"], call.kwargs["height"], call.kwargs["frames"], call.kwargs["low_memory"], call.kwargs["budget_bytes"]) == (512, 288, 73, True, 16 * 1024**3)

    def test_a_window_that_fits_does_not_warn(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], fit={"status": "fits", "required_gb": 9.0, "budget_gb": 16.0})
        assert captured["memory_warnings"] == []

    def test_an_unknown_memory_fit_does_not_warn(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], fit=None)
        assert captured["memory_warnings"] == []

    def test_audio_follows_the_source_and_the_flag(self, monkeypatch, tmp_path):
        from zvisiongenerator.utils.video_extend import extend_audio_output

        with_audio = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"])["args"]
        assert with_audio.no_audio is False and extend_audio_output(with_audio.extend_source, no_audio=with_audio.no_audio)
        muted = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--no-audio"])["args"]
        assert muted.no_audio is True and not extend_audio_output(muted.extend_source, no_audio=muted.no_audio)

    def test_runs_and_seed_reach_the_batch(self, monkeypatch, tmp_path):
        args = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "-r", "3", "--seed", "7"])["args"]
        assert (args.runs, args.seed) == (3, 7)

    def test_the_source_path_is_expanded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        captured = self._main(monkeypatch, tmp_path, ["--extend", "~/clip.mp4"])
        assert captured["loaded"] == [str(tmp_path / "clip.mp4")]

    def test_a_non_extend_run_is_unchanged(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a dog"])
        args = captured["args"]
        assert args.extend_source is None and (args.width, args.height, args.num_frames) == (704, 448, 49)
        assert captured["memory_warnings"] == []
        assert captured["backend"].load_model.call_args.kwargs["mode"] == "t2v"

    def test_model_is_still_required_without_extend(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--prompt", "a dog"])
        assert "--model is required" in capsys.readouterr().err


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
