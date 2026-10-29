"""Tests for the ziv-video command line: parsing, refusals before the model loads, defaults and memory checks."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub.utils import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

from tests._factories import make_extend_source
from tests.video._helpers import make_ltx_info, patch_video_cli_for_preflight

from zvisiongenerator import video_cli
from zvisiongenerator.utils.config import load_config
from zvisiongenerator.utils.model_memory import MemoryBudget
from zvisiongenerator.utils.video_model_detect import VideoModelInfo
from zvisiongenerator.video_cli import _build_video_parser

_LTX_FAMILY = {
    key: value
    for key, value in load_config()["video_model_presets"]["ltx"].items()
    if key in ("label", "default_steps", "max_steps", "frame_rule", "extend_frame_step", "carry_frames", "capabilities")
}


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


class TestVideoCliHubRefusals:
    """A Hub refusal ends ziv-video with a usage error that says what to do, not a traceback."""

    @staticmethod
    def _gated() -> GatedRepoError:
        return GatedRepoError("boom", response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/api/models/owner/model")))

    @pytest.mark.parametrize("stage", ["detect", "load"])
    def test_gated_model_exits_2_with_the_next_step(self, monkeypatch, tmp_path, capsys, stage):
        backend = MagicMock()
        options = {}
        if stage == "load":
            backend.load_model.side_effect = self._gated()
        else:

            def refuse(_model):
                raise self._gated()

            options["detect"] = refuse
        with pytest.raises(SystemExit, match="2"):
            _run_cli(monkeypatch, tmp_path, ["-m", "owner/model", "--prompt", "a dog"], backend=backend, **options)
        error = capsys.readouterr().err.strip().splitlines()[-1]
        assert "hf auth login" in error
        assert "https://huggingface.co" in error
        assert "Traceback" not in error

    def test_other_load_errors_keep_their_text(self, monkeypatch, tmp_path, capsys):
        backend = MagicMock()
        backend.load_model.side_effect = RuntimeError("out of memory")
        with pytest.raises(SystemExit, match="2"):
            _run_cli(monkeypatch, tmp_path, ["-m", "owner/model", "--prompt", "a dog"], backend=backend)
        assert "out of memory" in capsys.readouterr().err


class TestVideoCliExecution:
    """Verify executable CLI behavior beyond argument parsing."""

    def test_main_reports_unknown_model_guidance_with_the_ltx_aliases(self, monkeypatch, capsys):
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

        assert "'ltx-4' or 'ltx-8'" in capsys.readouterr().err

    def test_main_rejects_remote_lora_before_backend_load(self, monkeypatch, capsys):
        import zvisiongenerator.video_cli as video_cli

        model_info = make_ltx_info()
        backend = MagicMock()

        monkeypatch.setattr(
            video_cli,
            "load_config",
            lambda: {
                "video_generation": {"default_ratio": "16:9", "default_size": "m"},
                "video_sizes": {"16:9": {"m": {"width": 704, "height": 448, "frames": 49}}},
                "video_model_presets": {"ltx": {**_LTX_FAMILY, "upscale": {}}},
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

        with patch("sys.argv", ["ziv-video", "-m", "ltx-4", "--prompt", "a dog", "--lora", "org/lora:0.8"]):
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

    def test_main_loads_the_ltx25_alias_on_macos_from_the_pinned_mirror(self, monkeypatch, tmp_path):
        import zvisiongenerator.video_cli as video_cli
        from zvisiongenerator.utils.config import load_config

        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "ziv"))
        monkeypatch.setattr(video_cli, "load_config", load_config)
        monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(video_cli.sys, "platform", "darwin")
        backend = MagicMock()
        backend.load_model.side_effect = RuntimeError("stop after the load request")
        monkeypatch.setattr(video_cli, "get_video_backend", lambda _family: backend)

        with patch("sys.argv", ["ziv-video", "-m", "ltx25-4", "--prompt", "a dog", "-o", str(tmp_path)]):
            with pytest.raises(SystemExit, match="2"):
                video_cli.main()

        backend.load_model.assert_called_once()
        call = backend.load_model.call_args
        assert "MarceloFSO/ltx-2.5-mlx-q4-fast@281bf05bd23207d6feef708fc2e270f84ba992b5" in (*call.args, *call.kwargs.values())

    @pytest.mark.parametrize("with_image", [False, True])
    def test_main_loads_h3_on_macos_with_options_the_mlx_backend_accepts(self, monkeypatch, tmp_path, with_image):
        import zvisiongenerator.video_cli as video_cli
        from zvisiongenerator.backends.mac import h3_video
        from zvisiongenerator.utils.config import load_config
        from zvisiongenerator.utils.model_memory import MemoryBudget

        monkeypatch.setenv("ZIV_DATA_DIR", str(tmp_path / "ziv"))
        monkeypatch.setattr(video_cli, "load_config", load_config)
        monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(video_cli.sys, "platform", "darwin")
        monkeypatch.setattr(video_cli, "accelerator_memory_budget_or_none", lambda: MemoryBudget(24 * 1024**3, unified_bytes=32 * 1024**3))
        monkeypatch.setattr(h3_video, "resolve_mlx_files", lambda *_a, **_kw: MagicMock(lora_file="turbo.safetensors"))
        monkeypatch.setattr(h3_video, "_turbo_rank", lambda _path: 4)
        monkeypatch.setattr(h3_video, "ensure_text_encoder_copy", lambda models_dir, name, *_a: models_dir / name)
        loaded = []
        monkeypatch.setattr(video_cli, "get_video_backend", lambda _family: h3_video.H3MlxVideoBackend())
        monkeypatch.setattr(video_cli, "build_video_workflow", lambda _args, **_kwargs: object())
        monkeypatch.setattr(video_cli, "run_video_batch", lambda *, model, **_kw: loaded.append(model))
        argv = ["ziv-video", "-m", "h3", "--prompt", "a dog", "-o", str(tmp_path)]
        if with_image:
            image = tmp_path / "first.png"
            image.write_bytes(b"x")
            argv += ["--image", str(image)]

        with patch("sys.argv", argv):
            video_cli.main()

        (model,) = loaded
        assert (model.mode, model.verbose) == ("i2v" if with_image else "t2v", True)
        assert model.turbo_lora_alpha == 0.0625 * 4

    @staticmethod
    def _run_main_for_load_kwargs(monkeypatch, tmp_path, *, preset_extra, extra_argv):
        """Run ``main`` with a stubbed backend and return that backend's mock."""
        import zvisiongenerator.video_cli as video_cli

        model_info = make_ltx_info()
        backend = MagicMock()
        backend.load_model.return_value = (MagicMock(), model_info)
        ltx_preset = {**_LTX_FAMILY, "upscale": {"default_upscale_steps": 8}, **preset_extra}

        monkeypatch.setattr(
            video_cli,
            "load_config",
            lambda: {
                "video_generation": {"default_ratio": "16:9", "default_size": "m"},
                "video_sizes": {"16:9": {"m": {"width": 704, "height": 448, "frames": 49}}},
                "video_model_presets": {"ltx": ltx_preset},
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
        monkeypatch.setattr(video_cli, "build_video_workflow", lambda _args, **_kwargs: object())
        monkeypatch.setattr(video_cli, "run_video_batch", lambda *_a, **_k: None)

        argv = ["ziv-video", "-m", "ltx-4", "--prompt", "a dog", "-o", str(tmp_path), *extra_argv]
        with patch("sys.argv", argv):
            video_cli.main()
        return backend

    def test_main_passes_latent_upsampler_source_when_upscaling(self, monkeypatch, tmp_path):
        upsampler = {"repo": "your-org/upsampler", "filename": "up.safetensors"}

        backend = self._run_main_for_load_kwargs(monkeypatch, tmp_path, preset_extra={"diffusers": {"latent_upsampler": upsampler}}, extra_argv=["--upscale", "2"])

        kwargs = backend.load_model.call_args.kwargs
        assert kwargs["upscale"] is True
        assert kwargs["latent_upsampler"] == ("your-org/upsampler", "up.safetensors")

    def test_main_omits_latent_upsampler_without_upscale(self, monkeypatch, tmp_path):
        upsampler = {"repo": "your-org/upsampler", "filename": "up.safetensors"}

        backend = self._run_main_for_load_kwargs(monkeypatch, tmp_path, preset_extra={"diffusers": {"latent_upsampler": upsampler}}, extra_argv=[])

        assert "latent_upsampler" not in backend.load_model.call_args.kwargs

    def test_main_passes_none_when_upsampler_not_configured(self, monkeypatch, tmp_path):
        backend = self._run_main_for_load_kwargs(monkeypatch, tmp_path, preset_extra={}, extra_argv=["--upscale", "2"])

        assert backend.load_model.call_args.kwargs["latent_upsampler"] is None

    def test_main_reports_malformed_upsampler_config_via_parser_error(self, monkeypatch, tmp_path):
        bad = {"diffusers": {"latent_upsampler": {"repo": " ", "filename": "up.safetensors"}}}

        with pytest.raises(SystemExit, match="2"):
            self._run_main_for_load_kwargs(monkeypatch, tmp_path, preset_extra=bad, extra_argv=["--upscale", "2"])


class TestVideoCliPreflight:
    """ziv-video runs preflight (rewrites, enhancer unload, memory release) before loading the video model."""

    def _patch(self, monkeypatch, order: list[str], *, load_error=None):
        return patch_video_cli_for_preflight(monkeypatch, order, load_error=load_error)

    def test_preflight_before_model_load(self, monkeypatch, tmp_path):
        order: list[str] = []
        video_cli, captured, workflows = self._patch(monkeypatch, order)
        with patch("sys.argv", ["ziv-video", "-m", "ltx-4", "--prompt", "a dog runs", "--enhance", "-o", str(tmp_path)]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                video_cli.main()
        assert order == ["close", "release_memory", "load_model", "run_video_batch"]
        assert captured["plan"].has_rewrites and workflows == [True]

    def test_scene_reaches_the_rewrite(self, monkeypatch, tmp_path):
        """The reply has no markers, so the scene is not rewritten: the plan carries no enhanced scene (it once echoed the scene)."""
        order: list[str] = []
        video_cli, captured, _workflows = self._patch(monkeypatch, order)
        with patch("sys.argv", ["ziv-video", "-m", "ltx-4", "--scene", "A sunlit beach.", "--prompt", "a dog runs", "--enhance", "-o", str(tmp_path)]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                video_cli.main()
        (iteration,) = captured["plan"].iterations
        assert iteration.enhanced_prompt == "A dog sprints across a sunlit beach, camera tracking low."
        assert iteration.enhanced_scene is None and iteration.scene == "A sunlit beach."

    def test_enhancer_load_error_exits_before_model_load(self, monkeypatch, tmp_path):
        order: list[str] = []
        video_cli, _captured, _workflows = self._patch(monkeypatch, order, load_error=RuntimeError("Could not load prompt enhancer model a/b"))
        with patch("sys.argv", ["ziv-video", "-m", "ltx-4", "--prompt", "a dog runs", "--enhance", "-o", str(tmp_path)]):
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
    resolve=None,
    detect=None,
):
    """Run ``main()`` with heavy steps patched out; return what the batch received plus the backend and memory-check mocks."""
    import zvisiongenerator.video_cli as video_cli

    model_info = make_ltx_info()
    backend.load_model.return_value = (MagicMock(), model_info)
    memory = MagicMock(return_value=fit)
    monkeypatch.chdir(tmp_path)
    app_config = {"ui": {"default_models": {"video": "ltx-8"}}} if config is None else config
    monkeypatch.setattr(video_cli, "load_config", lambda: app_config)
    monkeypatch.setattr(video_cli, "resolve_model_path", resolve or (lambda model, **_: model))
    if local_models is not None:
        monkeypatch.setattr(video_cli, "video_model_downloaded", lambda ref, **_: ref in local_models)
    monkeypatch.setattr(video_cli, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(video_cli, "supports_video_extend", lambda: extend_supported)
    monkeypatch.setattr(video_cli, "supports_video_audio", lambda: True)
    monkeypatch.setattr(video_cli, "detect_video_model", detect or (lambda _model: model_info))
    monkeypatch.setattr(video_cli, "resolve_video_defaults", lambda _family, _config, cli_overrides: {"steps": 8, "width": 704, "height": 448, "num_frames": 49})
    monkeypatch.setattr(video_cli, "get_video_backend", lambda _family: backend)
    monkeypatch.setattr(video_cli, "accelerator_memory_budget_or_none", lambda: MemoryBudget(16 * 1024**3))
    monkeypatch.setattr(video_cli, "clip_memory_fit", memory)
    fits: list = []
    real_clip_fit = video_cli._clip_fit

    def _spy_clip_fit(args, config, *, upscale):
        fits.append(real_clip_fit(args, config, upscale=upscale))
        return fits[-1]

    monkeypatch.setattr(video_cli, "_clip_fit", _spy_clip_fit)
    monkeypatch.setattr(video_cli, "build_video_workflow", lambda _args, **_kwargs: object())
    loaded: list[str] = []

    def _load_source(path):
        loaded.append(path)
        if load_error is not None:
            raise load_error
        return source or _cli_source()

    monkeypatch.setattr(video_cli, "load_extend_source", _load_source)
    captured: dict = {"backend": backend, "memory": memory, "loaded": loaded, "fits": fits}

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


_H3_CONFIG = {**load_config(), "ui": {"default_models": {"video": "ltx-8"}}}


def _h3_repo(model: str, **_) -> str:
    """Resolve the alias ``h3`` to a repo reference, as the real resolver does; other names stay as they are."""
    return "org/h3" if model == "h3" else model


class TestVideoCliScene:
    """``--scene`` reaches the plan; scene-only runs are valid; a run with no scene and no prompt is refused."""

    def _main(self, monkeypatch, tmp_path, argv):
        return _run_cli(monkeypatch, tmp_path, ["-m", "ltx-4", *argv], backend=MagicMock())

    def test_the_picked_model_name_is_kept_beside_the_resolved_model(self, monkeypatch, tmp_path):
        def _resolve(model, **_):
            return f"org/repo-for-{model}"

        captured = _run_cli(monkeypatch, tmp_path, ["-m", "ltx-4", "--prompt", "a dog"], backend=MagicMock(), resolve=_resolve)
        assert (captured["args"].model_name, captured["args"].model) == ("ltx-4", "org/repo-for-ltx-4")

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
        assert "--prompt" in capsys.readouterr().err

    def test_blank_scene_and_empty_prompt_is_refused(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--scene", "  ", "--prompt", ""])
        assert "--prompt" in capsys.readouterr().err

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
        assert flag in capsys.readouterr().err

    def test_extend_frames_below_eight_is_refused(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-frames", "4"])
        assert "--extend-frames" in capsys.readouterr().err

    def test_extend_seconds_must_be_positive(self, monkeypatch, tmp_path, capsys):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-seconds", "0"])
        assert "--extend-seconds" in capsys.readouterr().err

    @pytest.mark.parametrize("seconds", ["nan", "inf"])
    def test_extend_seconds_must_be_finite(self, monkeypatch, tmp_path, capsys, seconds):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-seconds", seconds])
        assert "--extend-seconds" in capsys.readouterr().err
        self.backend.load_model.assert_not_called()

    @pytest.mark.parametrize("path", ["", "   "])
    def test_a_blank_extend_path_is_refused(self, monkeypatch, tmp_path, capsys, path):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", path, "--prompt", "a dog"])
        assert "--extend" in capsys.readouterr().err
        self.backend.load_model.assert_not_called()

    def test_capability_gate_refuses_extend_before_the_source_is_read(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], extend_supported=False)
        self.backend.load_model.assert_not_called()

    def test_a_family_that_cannot_extend_refuses_the_extension_before_the_model_loads(self, monkeypatch, tmp_path):
        ltx = load_config()["video_model_presets"]["ltx"]
        config = {"ui": {"default_models": {"video": "ltx-8"}}, "video_model_presets": {"ltx": {**ltx, "capabilities": {**ltx["capabilities"], "extend": False}}}}

        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], config=config)

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
        assert flag in capsys.readouterr().err
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

    def test_keep_everything_is_a_general_rewrite_with_a_scene(self, monkeypatch, tmp_path):
        self._preflight_spy(monkeypatch)
        args = self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--scene", "A castle.", "--prompt", "a fox", "--enhance", self._KEEP_ALL])["args"]
        assert args.enhance is not None

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

    def test_seconds_convert_at_the_source_frame_rate(self, monkeypatch, tmp_path):
        args = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-seconds", "2"], source=self._source(fps=30.0))["args"]
        assert (args.extend_frames, args.num_frames, args.fps) == (64, 64 + 25, 30.0)

    def test_seconds_that_do_not_land_on_a_step_warn(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-seconds", "2"], source=self._source(fps=30.0))
        assert any("--extend-seconds" in message for message in captured["warnings"])

    def test_seconds_on_a_step_do_not_warn(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-seconds", "2"])
        assert not any("--extend-seconds" in message for message in captured["warnings"])

    def test_frames_snap_to_a_multiple_of_eight(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--extend-frames", "53"])
        assert captured["args"].extend_frames == 56 and captured["args"].num_frames == 56 + 25
        assert any("--extend-frames" in message for message in captured["warnings"])

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

    def test_a_source_model_whose_every_download_is_present_is_kept(self, monkeypatch, tmp_path):
        from zvisiongenerator.downloads import DownloadTarget

        targets = (DownloadTarget("org/h3", "rev1", ()), DownloadTarget("org/h3-text", "rev2", ()))
        monkeypatch.setattr("zvisiongenerator.downloads.plans.video_plan", lambda *_args, **_kwargs: targets)
        present = {"org/h3", "org/h3-text"}
        monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: target.repo_id in present)
        h3_info = make_ltx_info(family="h3", backend="h3", frame_alignment=17)
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], source=self._source(model="h3"), local_models=None, detect=lambda _model: h3_info, config=_H3_CONFIG, resolve=_h3_repo)
        assert captured["args"].model_name == "h3"
        assert not any("not available locally" in message for message in captured["warnings"])

    def test_a_source_model_with_one_download_missing_falls_back_with_a_warning(self, monkeypatch, tmp_path):
        from zvisiongenerator.downloads import DownloadTarget

        targets = (DownloadTarget("org/h3", "rev1", ()), DownloadTarget("org/h3-text", "rev2", ()))
        monkeypatch.setattr("zvisiongenerator.downloads.plans.video_plan", lambda *_args, **_kwargs: targets)
        monkeypatch.setattr("zvisiongenerator.downloads.presence.target_present_offline", lambda target: target.repo_id == "org/h3")
        h3_info = make_ltx_info(family="h3", backend="h3", frame_alignment=17)
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4"], source=self._source(model="h3"), local_models=None, detect=lambda _model: h3_info, config=_H3_CONFIG, resolve=_h3_repo)
        assert captured["args"].model == "ltx-8"
        assert any("(h3)" in message and "not available locally" in message for message in captured["warnings"])

    def test_a_typed_repo_id_of_an_alias_runs_as_that_alias(self, monkeypatch, tmp_path):
        config = {"model_aliases": {"ltx-4": "org/ltx-mlx-q4"}, "ui": {"default_models": {"video": "ltx-8"}}}
        captured = _run_cli(monkeypatch, tmp_path, ["-m", "org/ltx-mlx-q4", "--prompt", "a dog"], backend=MagicMock(), config=config)
        assert captured["args"].model_name == "ltx-4"

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
        assert captured["fits"][0]["status"] == "too_large" and "--extend-seconds" in warning and "--extend-frames" in warning
        assert "args" in captured
        call = captured["memory"].call_args
        assert call.args == ("org/recorded-model",)
        assert (call.kwargs["width"], call.kwargs["height"], call.kwargs["frames"], call.kwargs["low_memory"], call.kwargs["upscale"], call.kwargs["budget"]) == (
            512,
            288,
            73,
            True,
            False,
            MemoryBudget(16 * 1024**3),
        )

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


class TestVideoCliClipMemory:
    """A text-to-video or image-to-video run checks its aligned size and length against memory once and warns; it never stops."""

    _main = TestVideoCliExtend._main
    _source = staticmethod(TestVideoCliExtend._source)
    _TOO_LARGE = {"status": "too_large", "required_gb": 40.2, "budget_gb": 16.0}

    def test_a_clip_that_does_not_fit_warns_once_and_the_run_continues(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a dog", "--runs", "3"], fit=self._TOO_LARGE)

        assert [fit["status"] for fit in captured["fits"]] == ["too_large"]
        assert len(captured["memory_warnings"]) == 1
        assert "args" in captured

    def test_a_tight_clip_is_reported_as_tight(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a dog"], fit={"status": "tight", "required_gb": 18.0, "budget_gb": 16.0})

        assert [fit["status"] for fit in captured["fits"]] == ["tight"]
        assert len(captured["memory_warnings"]) == 1

    @pytest.mark.parametrize(("flags", "advised"), [([], False), (["--no-low-memory"], True)])
    def test_the_warning_points_at_low_memory_only_when_it_is_off(self, monkeypatch, tmp_path, flags, advised):
        captured = self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a dog", *flags], fit=self._TOO_LARGE)

        (warning,) = captured["memory_warnings"]
        assert ("--low-memory" in warning) is advised
        assert "--frames" in warning and "--size" in warning

    @pytest.mark.parametrize("fit", [{"status": "fits", "required_gb": 9.0, "budget_gb": 16.0}, None])
    def test_a_clip_that_fits_or_is_not_estimated_does_not_warn(self, monkeypatch, tmp_path, fit):
        captured = self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a dog"], fit=fit)

        assert captured["memory_warnings"] == []

    def test_the_check_sees_the_aligned_size_and_length(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a dog", "-W", "700", "-H", "450", "--frames", "50"])

        call = captured["memory"].call_args
        assert call.args == ("ltx-8",)
        assert (call.kwargs["width"], call.kwargs["height"], call.kwargs["frames"]) == (704, 448, 49)
        assert (call.kwargs["low_memory"], call.kwargs["upscale"], call.kwargs["budget"]) == (True, False, MemoryBudget(16 * 1024**3))

    def test_an_upscaled_clip_is_checked_at_its_final_64_aligned_size(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a dog", "--upscale", "2", "-W", "700", "-H", "450"])

        call = captured["memory"].call_args
        assert (call.kwargs["width"], call.kwargs["height"], call.kwargs["upscale"]) == (704, 448, True)


# ---------------------------------------------------------------------------
# aligned_video_dims
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _align_resolution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# align_frames (the LTX rule)
# ---------------------------------------------------------------------------


def test_a_clip_limited_by_system_memory_points_at_the_model_not_the_clip(monkeypatch):
    import argparse

    import zvisiongenerator.video_cli as video_cli

    monkeypatch.setattr(video_cli, "_clip_fit", lambda *_args, **_kwargs: {"status": "too_large", "required_gb": 40.0, "budget_gb": 32.0, "kind": "discrete", "resource": "system"})
    args = argparse.Namespace(upscale=None, model_name="ltx-4", width=704, height=448, num_frames=121, low_memory=True)

    with pytest.warns(UserWarning) as caught:
        video_cli._warn_if_too_large(args, {}, extending=False)

    assert len(caught) == 1
    assert "--frames" not in str(caught[0].message)


class TestVideoCliH3:
    """MiniMax H3 on the CLI: ``--end-image``, the 17k+5 frame rule, and refusals only for what a family lacks."""

    _H3 = make_ltx_info(family="h3", backend="h3", frame_alignment=17)

    @pytest.fixture(autouse=True)
    def _linux(self, monkeypatch):
        import zvisiongenerator.backends as backends_module

        monkeypatch.setattr(backends_module.sys, "platform", "linux")

    def _main(self, monkeypatch, tmp_path, argv, *, family="h3", config=None, backend=None, **options):
        info = self._H3 if family == "h3" else make_ltx_info()
        return _run_cli(monkeypatch, tmp_path, argv, backend=backend or MagicMock(), config=config or load_config(), detect=lambda _model: info, **options)

    def _images(self, tmp_path):
        from PIL import Image

        paths = []
        for name in ("first.png", "last.png"):
            path = tmp_path / name
            Image.new("RGB", (8, 8)).save(path)
            paths.append(str(path))
        return paths

    def test_parser_end_image_option(self):
        args = _build_video_parser().parse_args(["-m", "h3", "--image", "a.png", "--end-image", "b.png"])
        assert (args.image_path, args.end_image_path) == ("a.png", "b.png")
        assert _build_video_parser().parse_args(["-m", "h3"]).end_image_path is None

    def test_an_end_image_without_an_image_is_refused(self, monkeypatch, tmp_path):
        _first, last = self._images(tmp_path)

        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["-m", "h3", "--prompt", "a lake", "--end-image", last])

    def test_a_missing_end_image_file_is_refused(self, monkeypatch, tmp_path, capsys):
        first, _last = self._images(tmp_path)

        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["-m", "h3", "--prompt", "a lake", "--image", first, "--end-image", str(tmp_path / "gone.png")])

        assert str(tmp_path / "gone.png") in capsys.readouterr().err

    def test_a_family_without_the_capability_refuses_the_end_image(self, monkeypatch, tmp_path, capsys):
        first, last = self._images(tmp_path)

        config = load_config()
        config["video_model_presets"]["ltx"]["capabilities"]["end_frame"] = False

        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a lake", "--image", first, "--end-image", last], family="ltx", config=config)

        assert "does not support a last-frame image" in capsys.readouterr().err

    @pytest.mark.parametrize(("extra", "end_image"), [(["--end-image"], True), ([], False)])
    def test_ltx_loads_for_the_last_frame_image_only_when_one_is_given(self, monkeypatch, tmp_path, extra, end_image):
        first, last = self._images(tmp_path)
        backend = MagicMock()
        backend.load_model.return_value = (MagicMock(), make_ltx_info())
        monkeypatch.setattr(video_cli, "run_video_batch", MagicMock())

        self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--prompt", "a lake", "--image", first, *[a for flag in extra for a in (flag, last)]], family="ltx", backend=backend)

        assert backend.load_model.call_args.kwargs["mode"] == "i2v"
        assert backend.load_model.call_args.kwargs["end_image"] is end_image

    def test_an_extension_refuses_the_end_image(self, monkeypatch, tmp_path, capsys):
        first, last = self._images(tmp_path)

        with pytest.raises(SystemExit, match="2"):
            self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "--image", first, "--end-image", last])

        assert "cannot be used with --extend" in capsys.readouterr().err

    def test_an_h3_extension_of_a_30_fps_source_is_refused_before_preflight_and_the_model(self, monkeypatch, tmp_path, capsys):
        backend = MagicMock()
        preflight = MagicMock()
        monkeypatch.setattr(video_cli, "run_preflight", preflight)
        source = _cli_source(fps=30.0, fps_text="30/1")

        with pytest.raises(SystemExit, match="2"):
            _run_cli(monkeypatch, tmp_path, ["-m", "h3", "--extend", "clip.mp4", "--prompt", "a lake"], backend=backend, config=load_config(), detect=lambda _model: self._H3, source=source)

        assert "30 fps" in capsys.readouterr().err
        preflight.assert_not_called()
        backend.load_model.assert_not_called()

    def test_an_h3_extension_of_a_24_fps_source_reaches_the_batch(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["-m", "h3", "--extend", "clip.mp4", "--prompt", "a lake"], source=_cli_source())

        assert captured["args"].extend_frames is not None

    def test_an_ltx_extension_of_a_30_fps_source_reaches_the_batch(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["-m", "ltx-8", "--extend", "clip.mp4", "--prompt", "a lake"], family="ltx", source=_cli_source(fps=30.0, fps_text="30/1"))

        assert captured["args"].extend_frames is not None

    def test_h3_on_macos_refuses_an_upscale_before_the_enhancer_and_the_model(self, monkeypatch, tmp_path, capsys):
        import zvisiongenerator.backends as backends_module

        monkeypatch.setattr(backends_module.sys, "platform", "darwin")
        backend = MagicMock()
        preflight = MagicMock()
        monkeypatch.setattr(video_cli, "run_preflight", preflight)

        with pytest.raises(SystemExit, match="2"):
            _run_cli(monkeypatch, tmp_path, ["-m", "h3", "--prompt", "a lake", "--upscale", "2"], backend=backend, config=load_config(), detect=lambda _model: self._H3)

        assert "MiniMax H3 can't upscale a clip on this platform." in capsys.readouterr().err
        preflight.assert_not_called()
        backend.load_model.assert_not_called()

    def test_h3_with_an_upscale_on_cuda_reaches_the_batch(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["-m", "h3", "--prompt", "a lake", "--upscale", "2"])

        assert captured["args"].upscale == 2

    @pytest.mark.parametrize(("family", "extend_frames", "window"), [("h3", "17", 39), ("ltx", "16", 41)])
    def test_an_extension_records_the_window_of_its_family(self, monkeypatch, tmp_path, family, extend_frames, window):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "-m", "h3" if family == "h3" else "ltx-8", "--extend-frames", extend_frames], family=family)

        assert (captured["args"].extend_frames, captured["args"].num_frames) == (int(extend_frames), window)
        assert captured["memory"].call_args.kwargs["frames"] == window

    def test_h3_with_both_images_reaches_the_batch(self, monkeypatch, tmp_path):
        first, last = self._images(tmp_path)

        captured = self._main(monkeypatch, tmp_path, ["-m", "h3", "--prompt", "a lake", "--image", first, "--end-image", last])

        args = captured["args"]
        assert (args.image_path, args.end_image_path, args.model_family) == (first, last, "h3")
        load = captured["backend"].load_model.call_args
        assert load.kwargs["mode"] == "i2v"
        assert load.kwargs["preset"].repo == "MiniMaxAI/MiniMax-H3"
        assert "upscale" not in load.kwargs

    def test_h3_without_an_end_image_leaves_it_unset(self, monkeypatch, tmp_path):
        first, _last = self._images(tmp_path)

        captured = self._main(monkeypatch, tmp_path, ["-m", "h3", "--prompt", "a lake", "--image", first])

        assert captured["args"].end_image_path is None

    def test_the_upscale_is_passed_to_the_h3_load(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["-m", "h3", "--prompt", "a lake", "--upscale", "2"])

        assert captured["backend"].load_model.call_args.kwargs["upscale"] is True

    def test_frames_follow_the_17k_plus_5_rule(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["-m", "h3", "--prompt", "a lake"])

        assert captured["args"].num_frames == 56  # the patched defaults give 49: up to the next 17k+5

    def test_extend_frames_are_multiples_of_17(self, monkeypatch, tmp_path):
        captured = self._main(monkeypatch, tmp_path, ["--extend", "clip.mp4", "-m", "h3", "--extend-frames", "40"])

        assert captured["args"].extend_frames == 34


class TestEstimatedTimeLine:
    """The H3 run prints its estimated time, labelled as estimated, on a discrete and on a unified budget."""

    _DISCRETE = MemoryBudget(10 * 1024**3, system_bytes=30 * 1024**3)
    _UNIFIED = MemoryBudget(24 * 1024**3, unified_bytes=32 * 1024**3)

    def _args(self, **overrides):
        import argparse

        values = dict(
            model="MiniMaxAI/MiniMax-H3",
            upscale=None,
            model_name="h3",
            model_family="h3",
            width=864,
            height=480,
            num_frames=124,
            low_memory=True,
            steps=8,
            image_path=None,
            extend_source=None,
            lora_paths=None,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def _run(self, monkeypatch, budget, args):
        monkeypatch.setattr(video_cli, "accelerator_memory_budget_or_none", lambda: budget)
        video_cli._warn_if_too_large(args, load_config(), extending=args.extend_source is not None)

    def test_the_line_names_the_size_the_frames_and_the_estimated_minutes(self, monkeypatch, capsys):
        self._run(monkeypatch, self._DISCRETE, self._args())

        out = capsys.readouterr().out
        assert "864x480" in out and "124 frames" in out and "estimated" in out

    def test_a_unified_budget_prints_the_line_too(self, monkeypatch, capsys):
        self._run(monkeypatch, self._UNIFIED, self._args(width=768, height=448))

        out = capsys.readouterr().out
        assert "768x448" in out and "124 frames" in out and "estimated" in out

    def test_an_extension_estimates_the_window_in_num_frames(self, monkeypatch, capsys):
        self._run(monkeypatch, self._DISCRETE, self._args(extend_source=make_extend_source(), extend_frames=34, num_frames=56))

        assert "864x480, 56 frames" in capsys.readouterr().out

    def test_the_clip_fit_is_asked_with_the_mode_and_steps_of_the_run(self, monkeypatch):
        calls = []
        monkeypatch.setattr(video_cli, "accelerator_memory_budget_or_none", lambda: self._DISCRETE)
        monkeypatch.setattr(video_cli, "clip_memory_fit", lambda model, **kwargs: calls.append(kwargs))

        video_cli._clip_fit(self._args(steps=4, image_path="a.png"), {}, upscale=False)
        video_cli._clip_fit(self._args(extend_source=make_extend_source(), extend_frames=34, num_frames=56), {}, upscale=False)
        video_cli._clip_fit(self._args(model_family="ltx", steps=8), {}, upscale=False)

        assert [(call["mode"], call["forwards"], call["frames"]) for call in calls] == [("i2v", 4, 124), ("extend", 8, 56), ("t2v", 8, 124)]

    def test_the_clip_fit_knows_whether_the_upscale_pauses_for_review(self, monkeypatch):
        calls = []
        monkeypatch.setattr(video_cli, "accelerator_memory_budget_or_none", lambda: self._DISCRETE)
        monkeypatch.setattr(video_cli, "clip_memory_fit", lambda model, **kwargs: calls.append(kwargs))

        video_cli._clip_fit(self._args(review=True), {}, upscale=True)
        video_cli._clip_fit(self._args(review=False), {}, upscale=True)
        video_cli._clip_fit(self._args(), {}, upscale=True)

        assert [call["review"] for call in calls] == [True, False, False]

    def test_ltx_prints_no_time_line(self, monkeypatch, capsys):
        monkeypatch.setattr(video_cli, "clip_memory_fit", lambda *_a, **_k: {"status": "fits", "required_gb": 8.0, "budget_gb": 10.0})

        self._run(monkeypatch, self._DISCRETE, self._args(model_family="ltx"))

        assert capsys.readouterr().out == ""

    def test_the_system_warning_of_h3_has_no_ltx_advice(self, monkeypatch):
        with pytest.warns(UserWarning) as caught:
            self._run(monkeypatch, MemoryBudget(10 * 1024**3, system_bytes=16 * 1024**3), self._args())

        assert "system memory" in str(caught[0].message)
        assert "ltx" not in str(caught[0].message)
