# Development

## Development Setup

```bash
git clone https://github.com/knuthelge/ZVisionGenerator && cd ZVisionGenerator
uv sync
```

## Make Targets

| Target | Description |
|--------|-------------|
| `make install` | Install all dependencies (`uv sync`) |
| `make lock` | Regenerate uv.lock |
| `make lint` | Run ruff linter |
| `make lint-fix` | Run ruff linter with auto-fix |
| `make format` | Format code with ruff |
| `make format-check` | Check formatting without changes |
| `make test` | Run tests with pytest |
| `make check` | Full CI gate: lint + format-check + test |
| `make build` | Build wheel and sdist |
| `make clean` | Remove build artifacts, caches, venv |
| `make run` | Run `ziv-image` CLI (use `ARGS="..."`) |
| `make model` | Run `ziv-model` CLI (use `ARGS="..."`) |

## Testing Conventions

- Tests live in `tests/`. Name files `test_<module>.py`, functions `test_<behavior>`.
- Group related tests in `Test`-prefixed classes.
- Never import or instantiate real backends/models. Use `unittest.mock.MagicMock` with controlled return values.
- Helper pattern: `_make_mock_backend()` returning a MagicMock with `.text_to_image.return_value = Image.new("RGB", (64, 64))`.
- Use `conftest.py::_make_args(**overrides)` to build minimal `argparse.Namespace` objects.
- Use `tmp_path` fixture for filesystem operations.
- Use `@pytest.mark.parametrize` for data-driven tests with many input/output pairs.
- Skip heavy dependencies: `pytest.importorskip("torch")`, `@pytest.mark.skipif(sys.platform == ...)` for platform-specific tests.

## Code Style

- Python 3.14+, `from __future__ import annotations` in every `.py` file.
- Modern type syntax: `str | None`, `list[str]`, `dict[str, Any]`. Never use `Optional`, `Union`, `List`, `Dict`, or `Tuple` from `typing`.
- Import order: stdlib → third-party → local (`zvisiongenerator.*`). Separate each group with a blank line.
- Use `TYPE_CHECKING` guards for imports that pull in heavy runtime dependencies (torch, mflux, diffusers).
- Line length limit: 200 (enforced by ruff).
- Docstrings: Google-style. Module-level docstrings on all non-trivial modules. First line is an imperative fragment. Multi-line use `Args:`, `Returns:`, `Raises:` sections.

## Project Structure

```
zvisiongenerator/
├── __init__.py
├── image_cli.py                   Image CLI entry point (ziv-image)
├── video_cli.py                   Video CLI entry point (ziv-video)
├── cli.py                         Unified CLI entry point (ziv)
├── image_runner.py                Image generation run orchestration
├── video_runner.py                Video generation run orchestration
├── config.yaml                    Default configuration (sizes, model presets)
├── backends/
│   ├── image_mac.py               macOS image backend (mflux/MLX)
│   ├── image_win.py               Windows image backend (diffusers/CUDA)
│   ├── video_mac.py               macOS video backend (LTX via MLX)
│   └── video_win.py               Windows video backend (LTX via CUDA)
├── converters/
│   ├── convert_checkpoint.py      Safetensors checkpoint → diffusers converter (ziv-model model)
│   ├── list_assets.py             List installed models, video models, and LoRAs (ziv-model list)
│   └── lora_import.py             LoRA import — local copy and HF download (ziv-model lora)
├── core/
│   ├── types.py                   Shared types (StageOutcome)
│   ├── image_types.py             Image request and artifacts
│   ├── video_types.py             Video request and artifacts
│   ├── image_backend.py           Image backend protocol
│   ├── video_backend.py           Video backend protocol
│   └── workflow.py                Unified workflow engine (image + video)
├── processing/
│   ├── contrast.py                Contrast adjustment
│   ├── saturation.py              Saturation adjustment
│   └── sharpen.py                 AMD CAS post-processing
├── schedulers/
│   └── beta_scheduler.py          Beta-distribution sigma scheduler
├── utils/
│   ├── alignment.py               Pixel-alignment helpers for resolution
│   ├── config.py                  Config loading (image + video)
│   ├── console.py                 Console formatting
│   ├── ffmpeg.py                  ffmpeg availability check and install
│   ├── filename.py                Output filename generation
│   ├── image_model_detect.py      Image model type detection
│   ├── interactive.py             Keyboard interrupt handling
│   ├── lora.py                    LoRA CLI argument parsing
│   ├── paths.py                   ~/.ziv/ model store resolution
│   ├── prompt_compose.py          Structured prompt flattening & snippets
│   ├── prompts.py                 Prompt file loading
│   └── video_model_detect.py      Video model type detection
└── workflows/
    ├── image_stages.py            Image pipeline stage definitions
    └── video_stages.py            Video pipeline stage definitions

prompts.yaml                       Default prompt definitions
```

## Architecture Overview

### Backend Protocol

Platform backends live in `backends/image_mac.py` (mflux/MLX) and `backends/image_win.py` (diffusers/CUDA). Video backends live in `backends/video_mac.py` (LTX via MLX) and `backends/video_win.py` (LTX via CUDA). Image backends implement the `ImageBackend` Protocol from `core/image_backend.py`; video backends implement the `VideoBackend` Protocol from `core/video_backend.py`. Platform selection uses `sys.platform` in `backends/__init__.py` — never check platform elsewhere.

### Workflow Stages

Image stage functions in `workflows/image_stages.py` have the uniform signature `(ImageGenerationRequest, ImageWorkingArtifacts) -> StageOutcome`. Video stage functions in `workflows/video_stages.py` have the signature `(VideoGenerationRequest, VideoWorkingArtifacts) -> StageOutcome`. Stages are composed dynamically by `build_workflow()` and `build_video_workflow()`.

### Data Types

Use `@dataclass(frozen=True)` for immutable value objects (inputs, detection results). Use mutable `@dataclass` only for working state. No pydantic or attrs.

### Config Layering

CLI flags > model preset variant > model preset family > global defaults. Config is a plain `dict` loaded from YAML, not a dataclass.

### Platform Support

- Platform definitions live in `config.yaml` under the `platforms` section, mapping `sys.platform` keys to human-readable labels.
- `PlatformInfo` from `zvisiongenerator.utils.platform` is the typed interface for resolved platform metadata.
- To add a new platform:
    - Add an entry to the `platforms` section in `config.yaml`.
    - Create the backend module(s) in `backends/`.
    - Register the platform in `backends/__init__.py` by updating `_PLATFORM_BACKEND_KEY` and adding the import in the relevant registration function.
- `model_aliases` use value types to signal availability: a string means supported (model path), a dict with `message` means unsupported with guidance, and a missing platform key means unsupported without guidance.

### Error Conventions

Raise `ValueError`, `FileNotFoundError`, `RuntimeError` directly with descriptive f-string messages. Use `warnings.warn()` with `stacklevel=2` for non-fatal conditions. No custom exception classes except private sentinels.
