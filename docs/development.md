# Development

## Development Setup

```bash
git clone https://github.com/knuthelge/ZVisionGenerator && cd ZVisionGenerator
make install
```

## Make Targets

| Target | Description |
|--------|-------------|
| `make install` | Install Python dependencies with `uv sync` and frontend dependencies with `pnpm --dir frontend install --frozen-lockfile` |
| `make lock` | Regenerate uv.lock |
| `make lint` | Run ruff linter |
| `make lint-fix` | Run ruff linter with auto-fix |
| `make format` | Format code with ruff |
| `make format-check` | Check formatting without changes |
| `make test` | Run tests with pytest |
| `make docs-check` | Build documentation with `mkdocs build --strict` |
| `make check` | Full CI gate: lint + format-check + pytest + frontend checks + docs build + packaged SPA static gate |
| `make frontend-build` | Build the Svelte app into `zvisiongenerator/web/static/app/` |
| `make frontend-static-check` | Rebuild the Svelte app and fail if packaged static artifacts changed or are untracked |
| `make frontend-test` | Run frontend type checks and Vitest tests |
| `make build` | Build wheel and sdist |
| `make clean` | Remove build artifacts, caches, venv |
| `make run` | Run `ziv-image` CLI (use `ARGS="..."`) |
| `make model` | Run `ziv-model` CLI (use `ARGS="..."`) |

`make check` covers the full verification gate: the Python test suite, frontend type checks and Vitest tests, packaged SPA artifact drift detection, and a strict docs build. The narrower targets (`make frontend-test`, `make frontend-static-check`, `make docs-check`) are available for iterating on a single surface.

## Testing Conventions

- Tests live in `tests/`. Name files `test_<module>.py`, functions `test_<behavior>`.
- Group related tests in `Test`-prefixed classes.
- Never import or instantiate real backends/models. Use `unittest.mock.MagicMock` with controlled return values.
- Mock backend pattern: a MagicMock with `.text_to_image.return_value = Image.new("RGB", (64, 64))` (see `_make_mock_backend()` in `tests/test_workflow.py`).
- Use `conftest.py::_make_args(**overrides)` to build minimal `argparse.Namespace` objects.
- Use `tmp_path` fixture for filesystem operations.
- Use `@pytest.mark.parametrize` for data-driven tests with many input/output pairs.
- Skip heavy dependencies: `pytest.importorskip("torch")`, `@pytest.mark.skipif(sys.platform == ...)` for platform-specific tests.
- Assert behavior and machine-readable contracts, not documentation prose, help text wording, CSS utility classes, source-code text, or incidental selector details.
- String assertions are appropriate only when the string is the contract under test, such as routes, config keys, storage keys, parser options, event names, filenames, enum values, structured statuses, and accessibility or control names required for operability.

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
├── enhance_cli.py                 Shared --enhance / --no-enhance / --enhance-model CLI handling
├── image_runner.py                Image generation run orchestration
├── video_runner.py                Video generation run orchestration
├── config.yaml                    Default configuration (sizes, model presets)
├── backends/
│   ├── image_mac.py               macOS image backend (mflux/MLX)
│   ├── image_mac_preview.py       Cheap latent → RGB live previews for the mflux backend
│   ├── image_win.py               Windows/Linux image backend (diffusers/CUDA)
│   ├── image_win_preview.py       Cheap latent → RGB live previews for the diffusers backend
│   ├── memory_mac.py              MLX memory release and recommended GPU memory budget
│   ├── prompt_enhancer_mac.py     macOS prompt-enhancer LLM (mlx-lm)
│   ├── prompt_enhancer_win.py     Windows/Linux prompt-enhancer LLM (transformers)
│   ├── prompt_enhancer_session.py Single resident prompt-enhancer model: exclusive use and idle release
│   ├── video_mac.py               macOS video backend (LTX via MLX)
│   └── video_diffusers.py         Windows/Linux video backend (LTX via diffusers/CUDA)
├── converters/
│   ├── convert_checkpoint.py      Safetensors checkpoint → diffusers converter (ziv-model model)
│   ├── list_assets.py             List installed models, video models, and LoRAs (ziv-model list)
│   └── lora_import.py             LoRA import — local copy and HF download (ziv-model lora)
├── core/
│   ├── types.py                   Shared types (StageOutcome)
│   ├── progress_events.py         Shared image/video workflow progress event helpers
│   ├── latent_preview.py          Latent → RGB projection factors shared by live previews
│   ├── image_types.py             Image request and artifacts
│   ├── video_types.py             Video request and artifacts
│   ├── image_backend.py           Image backend protocol
│   ├── video_backend.py           Video backend protocol
│   ├── prompt_enhancer.py         Prompt-enhancer protocol
│   └── workflow.py                Unified workflow engine (image + video)
├── processing/
│   ├── contrast.py                Contrast adjustment
│   ├── saturation.py              Saturation adjustment
│   └── sharpen.py                 AMD CAS post-processing
├── schedulers/
│   └── beta_scheduler.py          Beta-distribution sigma scheduler
├── utils/
│   ├── alignment.py               Pixel-alignment helpers for resolution
│   ├── app_log.py                 Error log file, cleared on every start
│   ├── atomic_write.py            Atomic file replacement for config and prompt-file writers
│   ├── config.py                  Config loading (image + video)
│   ├── console.py                 Console formatting
│   ├── ffmpeg.py                  ffmpeg availability check and install
│   ├── filename.py                Output filename generation
│   ├── image_model_detect.py      Image model type detection
│   ├── interactive.py             Keyboard interrupt handling
│   ├── lora.py                    LoRA CLI argument parsing
│   ├── model_files.py             Offline check for fully downloaded model weights
│   ├── model_memory.py            Header-only memory estimates for MLX models
│   ├── paths.py                   ~/.ziv/ model store resolution
│   ├── platform.py                Platform metadata and platform-aware alias values
│   ├── prompt_compose.py          Structured prompt flattening & snippets
│   ├── prompt_enhance.py          Prompt-enhancement build, run, and post-processing (no model I/O)
│   ├── prompts.py                 Prompt file loading
│   ├── provenance.py              Embedded asset config (PNG/MP4) and full provenance payload builders
│   └── video_model_detect.py      Video model type detection
├── web/
│   ├── config.py                  Web UI config loading and model inventory discovery
│   ├── config_api.py              JSON config response assembly
│   ├── config_contract.py         Writable config semantics and path readback helpers
│   ├── defaults.py                Ratio and size defaults for Web UI contracts
│   ├── gallery.py                 Gallery inventory and response serialization
│   ├── job_contract.py            Web job lifecycle, terminal event, and control contract
│   ├── model_delete.py            Delete installed models, LoRAs, and HF downloads
│   ├── model_inventory.py         Backend model inventory for Web UI contracts
│   ├── model_status.py            Per-model downloaded / memory-fit status for the SPA
│   ├── path_picker.py             Host-local directory and file picker
│   ├── prompt_files.py            Read and atomically update host-local prompt files
│   ├── request_guard.py           DNS-rebinding and cross-site request protection
│   ├── web_runner.py              Background workers for generation batches
│   ├── workspace_api.py           Shared SPA payloads for workspace and models routes
│   ├── workspace_contract.py      Workflow aliases and static workspace capabilities
│   └── server.py                  FastAPI route wiring and request parsing
└── workflows/
    ├── enhance_stage.py           Shared auto-enhance step for image and video workflows
    ├── image_stages.py            Image pipeline stage definitions
    └── video_stages.py            Video pipeline stage definitions

prompts.yaml                       Default prompt definitions
```

## Architecture Overview

### Backend Protocol

Platform backends live in `backends/image_mac.py` (mflux/MLX), `backends/image_win.py` (Windows/Linux diffusers image), `backends/video_mac.py` (MLX video), and `backends/video_diffusers.py` (Windows/Linux diffusers video). Image backends implement the `ImageBackend` Protocol from `core/image_backend.py`; video backends implement the `VideoBackend` Protocol from `core/video_backend.py`. The prompt-enhancer LLM follows the same split: `backends/prompt_enhancer_mac.py` (mlx-lm) and `backends/prompt_enhancer_win.py` (transformers) implement the `PromptEnhancer` Protocol from `core/prompt_enhancer.py`. Backend and accelerator selection stays centralized in `backends/__init__.py` — never branch on `sys.platform` elsewhere to pick a backend. OS-level checks unrelated to backends (file pickers, ffmpeg install hints, terminal handling) are fine where they are needed.

The `image_win.py` module is the shared diffusers/CUDA image backend for both Windows and Linux. The filename is historical; platform selection still happens only in `backends/__init__.py`.

Windows and Linux video resolution is config-driven through `video_model_presets.ltx.diffusers.default_repo` and the `ltx-2.3` alias. The backend assumes a diffusers-compatible repository layout and keeps the configured default overrideable from user config. macOS keeps the MLX-only `ltx-4` and `ltx-8` aliases.

The Windows/Linux video backend is CUDA-only. Validation happens lazily during backend load so importing the package or running non-video code paths does not require torch, diffusers, or a GPU.

The Windows/Linux diffusers image backend is also CUDA-only. Validation happens lazily during image backend load so importing the package or running non-image code paths does not require torch, diffusers, or a GPU.

### Workflow Stages

Image stage functions in `workflows/image_stages.py` have the uniform signature `(ImageGenerationRequest, ImageWorkingArtifacts) -> StageOutcome`. Video stage functions in `workflows/video_stages.py` have the signature `(VideoGenerationRequest, VideoWorkingArtifacts) -> StageOutcome`. Stages are composed dynamically by `build_workflow()` and `build_video_workflow()`.

### Data Types

Use `@dataclass(frozen=True)` for immutable value objects (inputs, detection results). Use mutable `@dataclass` only for working state. No pydantic or attrs.

### Config Layering

CLI flags > model preset variant > model preset family > global defaults. Config is a plain `dict` loaded from YAML, not a dataclass.

### Web Model Inventory

A repo-id alias can declare its image family in config via `model_alias_families` (for example `ideo: ideogram4`). The Web UI inventory and workspace bootstrap consult this declared family as an offline fast-path, so a known alias is surfaced without the network `detect_image_model` round-trip that a bare Hugging Face repo id would otherwise require. Content-based detection for arbitrary local model directories is unaffected; only aliases with a declared family skip detection.

Per-model Web UI capability flags — `supports_img2img`, `supports_upscale`, `supports_json_prompt`, `supports_first_sigma`, `dimension_min`, `dimension_max`, and `dimension_step` — are data-driven: they originate in the `model_presets` family entries, flow through `resolve_defaults`, and are emitted into the SPA's `ImageModelDefaults`. Families that do not set them inherit permissive defaults (img2img and upscale enabled, JSON caption and first-sigma disabled, `16`/`null`/`16` dimension bounds), so gating is centralized in config rather than branching on family in the web layer.

### Error Conventions

Raise `ValueError`, `FileNotFoundError`, `RuntimeError` directly with descriptive f-string messages. Use `warnings.warn()` with `stacklevel=2` for non-fatal conditions. Add a custom exception class only when callers must catch it distinctly (for example `JobConflictError` in `web/web_runner.py`) or as a private sentinel.

### Test Strategy

Mock heavy image and video dependencies in tests. Diffusers backend tests patch the lazy runtime loader, torch CUDA checks, export helpers, and PIL image loading so the suite never instantiates a real model, downloads weights, or requires a real CUDA device. Platform dispatch, alias resolution, and Web inventory tests should assert behavior through config and protocol boundaries rather than backend internals.

## Proposals

Planned and possible future work is written up in [`proposals/`](https://github.com/knuthelge/ZVisionGenerator/tree/main/proposals) at the repository root, one Markdown file per proposal with a status line, indexed in `proposals/README.md`. Check it before starting larger work, and update a proposal's status when it ships.

## Releasing

Pushing a `v*` tag runs the Release workflow, which builds the package, publishes it to PyPI and creates a GitHub Release. The package version comes from the tag (`hatch-vcs`), so the tag must be a valid PEP 440 version:

- **Final releases:** `vX.Y.Z`, e.g. `v0.13.0`.
- **Betas:** `vX.Y.ZbN`, e.g. `v0.13.0b1`, then `b2`, `b3`, … before the final `vX.Y.Z`. Use `aN` and `rcN` the same way for alphas and release candidates.

Pre-release tags are published as GitHub pre-releases, and `pip` installs them only with `--pre` or an exact pin (`pip install z-vision-generator==0.13.0b1`). Before tagging, move the `[Unreleased]` CHANGELOG entries under a `## [<version>] - <date>` heading in a `chore(release): v<version>` commit whose body lists the release highlights. PyPI never accepts the same version twice, so a broken release needs a new tag.
