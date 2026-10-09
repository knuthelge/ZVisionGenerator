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
| `make frontend-palette-check` | Fail on raw colour classes or literals in Svelte files (`PALETTE_STRICT=0` only warns) |
| `make build` | Build wheel and sdist |
| `make clean` | Remove build artifacts, caches, venv |
| `make run` | Run `ziv-image` CLI (use `ARGS="..."`) |
| `make model` | Run `ziv-model` CLI (use `ARGS="..."`) |

`make check` covers the full verification gate: the Python test suite, frontend type checks and Vitest tests, the raw colour check, packaged SPA artifact drift detection, and a strict docs build. The narrower targets (`make frontend-test`, `make frontend-static-check`, `make docs-check`) are available for iterating on a single surface.

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
├── preflight.py                   Preflight phase: plan seeds and prompts, rewrite them before the model loads
├── image_model_loader.py          Image model loading through stored quants (reuse, or save on first use)
├── image_runner.py                Image generation run orchestration
├── upscale_runner.py              Upscale one existing image: read its recorded settings, run the upscale workflow
├── video_runner.py                Video generation run orchestration
├── config.yaml                    Default configuration (sizes, model presets)
├── backends/
│   ├── image_mac.py               macOS image backend (mflux/MLX)
│   ├── image_mac_preview.py       Cheap latent → RGB live previews for the mflux backend
│   ├── image_win.py               Windows/Linux image backend (diffusers/CUDA)
│   ├── image_win_preview.py       Cheap latent → RGB live previews for the diffusers backend
│   ├── memory_cuda.py             CUDA allocator options and memory release after a job
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
│   ├── job_plan.py                Job plan value types (JobPlan, IterationPlan, EnhanceStatus)
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
│   ├── filename.py                Output filenames ({set}_{timestamp}) and collision counters
│   ├── image_model_detect.py      Image model type detection
│   ├── interactive.py             Keyboard interrupt handling
│   ├── lora.py                    LoRA CLI argument parsing
│   ├── model_files.py             Offline check for fully downloaded model weights
│   ├── model_memory.py            Header-only memory estimates for MLX models
│   ├── paths.py                   ~/.ziv/ model store resolution
│   ├── platform.py                Platform metadata and platform-aware alias values
│   ├── prompt_compose.py          Structured prompt flattening & snippets
│   ├── prompt_document.py         Prompt file ↔ editable document for the Web UI builder (comment-preserving, ruamel.yaml)
│   ├── prompt_document_check.py   Builder document previews and problems, with the prompt loader's rules
│   ├── prompt_enhance.py          Prompt-enhancement build, run, and post-processing (no model I/O)
│   ├── prompts.py                 Prompt file loading
│   ├── provenance.py              Settings embedded in generated PNG/MP4 files: build, embed, read
│   ├── stored_quant.py            Stored-quant naming, manifests and folder handling
│   ├── upscale.py                 Upscale output sizes, size limits and denoise defaults
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
│   ├── prompt_builder.py          Prompts page API: load, preview, save (revision-checked) and create prompt files
│   ├── prompt_files.py            Read and atomically update host-local prompt files
│   ├── request_guard.py           DNS-rebinding and cross-site request protection
│   ├── upscale_api.py             Plan a viewer upscale job: refinement model, carried settings, limits
│   ├── web_runner.py              Background workers for generation batches
│   ├── workspace_api.py           Shared SPA payloads for workspace and models routes
│   ├── workspace_contract.py      Workflow aliases and static workspace capabilities
│   └── server.py                  FastAPI route wiring and request parsing
└── workflows/
    ├── enhance_stage.py           Applies the planned rewrite in image and video workflows
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

Image stage functions in `workflows/image_stages.py` have the uniform signature `(ImageGenerationRequest, ImageWorkingArtifacts) -> StageOutcome`. Video stage functions in `workflows/video_stages.py` have the signature `(VideoGenerationRequest, VideoWorkingArtifacts) -> StageOutcome`. Stages are composed dynamically by `build_workflow()` and `build_video_workflow()`. `build_upscale_workflow()` upscales an existing image (`load_source_stage` → `upscale_stage` → `sharpen_stage` → `save_image_stage`); the Web UI runs it through `WebRunner.submit_upscale_job` and `upscale_runner.run_upscale`, which emits the same events as a one-image batch, without preflight or the batch loop.

### Job Phases: Preflight and Generation

Every image and video job (CLI and Web UI) runs in two phases, and only one model is resident at a time:

1. **Preflight** (`preflight.run_preflight`) runs before the generation model loads. It plans every iteration with `plan_iterations` (seed, `{a|b}` expansion, enhancement settings) and, when any iteration asks for auto enhancement, loads the prompt enhancer, rewrites those prompts, closes the enhancer and frees accelerator memory. It returns a frozen `JobPlan` (`core/job_plan.py`).
2. **Generation**: the entry point loads the model and passes the plan to `run_batch` / `run_video_batch` as the required `plan=` argument. Runners raise `ValueError` for a cancelled plan or one whose length differs from the batch, and `RuntimeError` when an iteration is out of order. Retry and repeat keep the planned text and draw a new random seed.

`JobPlan.iterations` holds one `IterationPlan` per iteration in run → set → prompt order, the order the runners loop in. Each carries `seed`, `resolved_prompt`, `enhanced_prompt` and `enhance_status` (`off`, `enhanced`, `failed`, `skipped`). The workflow's `enhance_prompt_stage` only applies the planned rewrite; no workflow stage calls the LLM.

Entry points own the control signal. `image_cli.main` creates the `SkipSignal` and its key listener before preflight and stops it after `run_batch`; Web UI image jobs pass their control signal to both phases; video jobs pass none. During preflight there is a control boundary before each rewrite and a final one after the enhancer is released: `skip` skips the next rewrite, `pause` waits, `quit` cancels, and `repeat` is dropped. A rewrite stopped by the user (the `cancelled` callback saw Next or Quit) is `skipped`, never `failed`. On Quit, preflight releases the enhancer and memory before emitting `batch_cancelled`.

Preflight events (none carry `eta_secs`, `avg_secs` or `elapsed_secs`):

| Event | When | Payload |
|---|---|---|
| `preflight_started` | first event of every preflight | `mode`, `total_iterations`, `total_rewrites` |
| `enhancer_loading` | 0..n, only when the enhancer loads; `cpu` when it runs on the CPU | `mode`, `phase` |
| `prompts_enhancing` | once per rewrite, after its boundary | `mode`, `index`, `total` |
| `prompt_enhance_failed` | a rewrite failed without a user stop | `mode`, `index`, `total`, `message` |
| `job_paused` / `job_resumed` | pause at a boundary | `mode`, `completed_iterations`, `total_iterations` |
| `batch_cancelled` | Quit (web: `job_cancelled`) | `mode`, `completed_iterations`, `total_iterations` |
| `preflight_finished` | last event unless cancelled | `mode`, `total_iterations`, `enhanced`, `failed`, `skipped` |

`prompt_started` also carries the iteration's `enhance_status`.

Tests must never run the real memory cleanup: patch `zvisiongenerator.preflight.release_accelerator_memory` (and `zvisiongenerator.web.web_runner.release_accelerator_memory` for web jobs), or pass `release_memory=` to `run_preflight`.

To add pre-load work (for example LoRA validation or a memory-fit check), add a field to `JobPlan` or `IterationPlan` and a private step in `run_preflight`. There is no stage pipeline or registry; promote `preflight.py` to a package only once it holds several concerns.

### Web Job Queue

`WebRunner` (`web/web_runner.py`) runs every Web UI job on a single worker thread. Generation jobs are never refused for being second: `_submit_job` registers the job, publishes `job_submitted`, appends it to the FIFO queue and dispatches it, all under `_jobs_lock`, so the worker receives jobs in queue order. When the worker reaches a job, `_claim` takes it off the queue and publishes `job_started` (status `running`, `started_at`; `elapsed_secs` counts from here). A job removed with `cancel_queued`, `clear_queue` or `shutdown` is no longer queued, so the claim skips it even if its work item was already picked up; it ends with `job_cancelled` and `reason: "removed"`.

The *active* job is the claimed one or, between jobs, the head of the queue; every other unfinished generation job is *queued*, with a 1-based `queue_position`. `GET /api/jobs` and `/api/workspace` return them as `active_job` and `queued_jobs`, and a job is never in both. `DELETE /api/jobs/queue` removes the queued jobs and keeps the active one. Actions that need the model memory to themselves (manual prompt enhancement, saving a quantized copy, deleting models or LoRAs) still wait until no generation job is running or queued.

### Data Types

Use `@dataclass(frozen=True)` for immutable value objects (inputs, detection results). Use mutable `@dataclass` only for working state. No pydantic or attrs.

### Config Layering

CLI flags > model preset variant > model preset family > global defaults. Config is a plain `dict` loaded from YAML, not a dataclass.

### Web Model Inventory

A repo-id alias can declare its image family in config via `model_alias_families` (for example `ideo: ideogram4`). The Web UI inventory and workspace bootstrap consult this declared family as an offline fast-path, so a known alias is surfaced without the network `detect_image_model` round-trip that a bare Hugging Face repo id would otherwise require. Content-based detection for arbitrary local model directories is unaffected; only aliases with a declared family skip detection.

Per-model Web UI capability flags — `supports_img2img`, `supports_upscale`, `supports_scheduler`, `supports_json_prompt`, `supports_first_sigma`, `dimension_min`, `dimension_max`, and `dimension_step` — are data-driven: they originate in the `model_presets` family entries, flow through `resolve_defaults`, and are emitted into the SPA's `ImageModelDefaults`. Families that do not set them inherit permissive defaults (img2img, upscale and scheduler enabled, JSON caption and first-sigma disabled, `16`/`null`/`16` dimension bounds), so gating is centralized in config rather than branching on family in the web layer.

### Error Conventions

Raise `ValueError`, `FileNotFoundError`, `RuntimeError` directly with descriptive f-string messages. Use `warnings.warn()` with `stacklevel=2` for non-fatal conditions. Add a custom exception class only when callers must catch it distinctly (for example `JobConflictError` in `web/web_runner.py`) or as a private sentinel.

### Web UI Confirmations

Ask the user to approve an action with `ConfirmDialog` (`frontend/src/lib/components/molecules/`), never `window.confirm`. It has no title: it shows the question, an optional line of information, and a cancel button plus a confirm button named after the action (danger-styled for destructive actions). Enter confirms (except on the focused cancel button), Esc cancels, and a held key never confirms; while it is open, no other key reaches the page or a viewer underneath. Call `requestConfirm({ question, info, confirmLabel })`, which resolves `true` on confirm, or render `<ConfirmDialog>` directly when the dialog needs a body or stays open while the action runs (`pending`).

### Web UI Toasts

Report the outcome of an action with `addToast(message, type, options)` from `$lib/state/toasts.svelte`. One toast shows at a time at the bottom centre of the page content (pages with a fixed left sidebar set `--toast-inset-left`); the rest wait their turn, shown as "+N". A repeat of a waiting or showing message merges into it ("×2"), and an error goes ahead of every non-error. Each tone has a default time on screen (success 2.5 s, info 4 s, warning 7 s); errors stay until dismissed, or give way after 8 s once other toasts are waiting, and hovering or focusing a toast pauses it. Pass `{ action: { label, run } }` for one follow-up such as Retry, and `{ timeout }` only when the default doesn't fit. An action that acts on a page must not outlive it: keep the id `addToast` returns and call `dismissToast(id)` in the page's `onDestroy`. Raise one toast per action, and skip it when the screen already shows the result.

### Web UI Design System

The Web UI has one design system ([proposal](https://github.com/knuthelge/ZVisionGenerator/blob/main/proposals/web-design-system.md); borders and colours in [No borders and Graphite](https://github.com/knuthelge/ZVisionGenerator/blob/main/proposals/web-no-borders-graphite.md)). Its tokens live in `@theme` in `frontend/src/app/global.css`, and its component classes are the `ui-*` classes in the same file. Build pages from the shared components in `frontend/src/lib/components/` (`Button`, `Input`, `Select`, `Toggle`, `Badge`, `Panel`, `Segmented`, `ActionBar`, `PageHeader`, `KeyValueList`, `Alert`, `EmptyState`, `InspectorSection`, `InspectorRow`) or, where no component fits, the `ui-*` classes (for example `ui-chip`), rather than styling elements by hand.

- **Size:** controls are 28px (`h-control`), small ones 22px, rows 30px. Buttons that sit together share one height; inside an `ActionBar` every button is 36px and the one marked `main` takes the remaining width.
- **Type:** four sizes only: `text-meta` (11px), `text-ui` (12px), `text-content` (13px), `text-title` (16px). Nunito (`font-heading`) is for titles, area names and the main action; monospace is for numbers, paths and shortcuts.
- **Labels:** field and row labels are sentence case (`ui-label`). The one small-caps style (`ui-area-label`) names areas: panels, sections, table headers and menu groups.
- **Shape and surfaces:** corners are 4px for chips and badges, 6px for controls and 10px for containers. Overlays (menus, popovers, tooltips, toasts, dialogs) use `bg-overlay`, darker than any page surface, with a faint ring and never a drop shadow; `bg-raised` is for section header bands and `bg-fill` for the prompt box.
- **No borders:** containers are separated by shade, and buttons, fields, chips and badges are filled shapes. Dividers between rows and sections stay. Draw an outline only where it means something: focus, a field with an error, the selected tile, and the dashed edge of drop zones and add buttons. Segmented controls are a recessed well with the chosen option as a lighter pill.
- **Main action:** each section (a bar, panel, form, dialog or popover) has at most one main action, a `primary` button: teal fill and a bold Nunito label. Every other button in the section is secondary.
- **Reference pages:** Config and Models centre their content at `max-w-page` (1200px), with the page bar lined up to it. Settings use label-left rows (`FormField layout="row"`), and tables use `ui-table-comfortable`.
- **Colour:** neutrals are Graphite greys with no tint. Teal means on, selected or the main action; coral (`accent-coral`, Blob's beret) marks `$snippets` and amber `{a|b}` choices; red is for errors and destructive actions, which always show red text. Info messages stay teal, never coral. Text is at least 4.5:1 on every surface. Use tokens, never raw palette classes or colour literals; `make check` fails on them.
- **Focus:** every control shows the same 2px teal outline with a 2px offset.

### Test Strategy

Mock heavy image and video dependencies in tests. Diffusers backend tests patch the lazy runtime loader, torch CUDA checks, export helpers, and PIL image loading so the suite never instantiates a real model, downloads weights, or requires a real CUDA device. Platform dispatch, alias resolution, and Web inventory tests should assert behavior through config and protocol boundaries rather than backend internals.

## Proposals

Planned and possible future work is written up in [`proposals/`](https://github.com/knuthelge/ZVisionGenerator/tree/main/proposals) at the repository root, one Markdown file per proposal with a status line, indexed in `proposals/README.md`. Check it before starting larger work, and update a proposal's status when it ships.

## Releasing

Pushing a `v*` tag runs the Release workflow, which builds the package, publishes it to PyPI and creates a GitHub Release. The package version comes from the tag (`hatch-vcs`), so the tag must be a valid PEP 440 version:

- **Final releases:** `vX.Y.Z`, e.g. `v0.13.0`.
- **Betas:** `vX.Y.ZbN`, e.g. `v0.13.0b1`, then `b2`, `b3`, … before the final `vX.Y.Z`. Use `aN` and `rcN` the same way for alphas and release candidates.

Pre-release tags are published as GitHub pre-releases, and `pip` installs them only with `--pre` or an exact pin (`pip install z-vision-generator==0.13.0b1`). Before tagging, move the `[Unreleased]` CHANGELOG entries under a `## [<version>] - <date>` heading in a `chore(release): v<version>` commit whose body lists the release highlights. PyPI never accepts the same version twice, so a broken release needs a new tag.
