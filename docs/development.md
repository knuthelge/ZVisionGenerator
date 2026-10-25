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
| `make frontend-install` | Install only the frontend pnpm dependencies |
| `make lock` | Regenerate uv.lock |
| `make lint` | Run ruff linter |
| `make lint-fix` | Run ruff linter with auto-fix |
| `make format` | Format code with ruff |
| `make format-check` | Check formatting without changes |
| `make test` | Run tests with pytest |
| `make docs-check` | Build documentation with `mkdocs build --strict` |
| `make check` | Full CI gate: lint + format-check + pytest + frontend checks + docs build + packaged SPA static gate |
| `make frontend-dev` | Start the Vite dev server |
| `make frontend-build` | Build the Svelte app into `zvisiongenerator/web/static/app/` |
| `make frontend-static-check` | Rebuild the Svelte app and fail if packaged static artifacts changed or are untracked |
| `make frontend-test` | Run frontend type checks and Vitest tests |
| `make frontend-palette-check` | Fail on raw colour classes or literals in Svelte files (`PALETTE_STRICT=0` only warns) |
| `make update-ltx` | Re-vendor `ltx-core-mlx` and `ltx-pipelines-mlx` into `packages/` |
| `make update-seedvr2` | Re-vendor the SeedVR2 PyTorch inference core into `packages/seedvr2_torch` |
| `make update-h3-mlx` | Re-vendor the MiniMax H3 MLX pipeline into `packages/minimax_h3_mlx` |
| `make build` | Build wheel and sdist |
| `make clean` | Remove build artifacts, caches, venv |
| `make run` | Run `ziv-image` CLI (use `ARGS="..."`) |
| `make model` | Run `ziv-model` CLI (use `ARGS="..."`) |
| `make enhance-bench` | Run the [prompt-enhancer bench](#prompt-enhancer-bench) (use `ARGS="..."`; loads a real model, not part of `make check`) |

`make check` covers the full verification gate: the Python test suite, frontend type checks and Vitest tests, the raw colour check, packaged SPA artifact drift detection, and a strict docs build. The narrower targets (`make frontend-test`, `make frontend-static-check`, `make docs-check`) are available for iterating on a single surface.

## Testing Conventions

- Tests live in `tests/`. Name files `test_<module>.py`, functions `test_<behavior>`.
- Group related tests in `Test`-prefixed classes.
- Never import or instantiate real backends/models. Use `unittest.mock.MagicMock` with controlled return values.
- Mock backend pattern: a MagicMock with `.text_to_image.return_value = Image.new("RGB", (64, 64))` (see `_make_mock_backend()` in `tests/test_workflow.py`).
- Use `conftest.py::_make_args(**overrides)` to build minimal `argparse.Namespace` objects.
- Use `tmp_path` fixture for filesystem operations.
- An autouse fixture, `_isolated_app_db`, points `zvisiongenerator.utils.app_db.get_ziv_data_dir` at a per-test temporary directory, so no test touches the real `ziv.sqlite3`. Files a test creates are under 2 seconds old and the gallery index never persists them; to test caching, backdate them with `os.utime` (or pass `clock=` to `GalleryIndex`) and count reader calls.
- Use `@pytest.mark.parametrize` for data-driven tests with many input/output pairs.
- Tests that run the real `ffmpeg` and `ffprobe` (in `tests/test_video_probe.py`, `tests/test_video_splice.py` and `tests/test_video_tail.py`) are marked to skip when the tools are missing and are meant to be run locally with ffmpeg installed; run them before changing the extend join. Everything else mocks ffmpeg.
- An autouse guard fails any test that reaches `huggingface_hub.hf_hub_download` or `snapshot_download` unmocked, even when the code under test swallows the error. Use the `hub_offline` fixture for a test whose subject is "nothing cached, no network", or mock the function in the test.
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
├── enhance_cli.py                 Shared --enhance / --no-enhance / --enhance-model / --enhance-format CLI handling
├── preflight.py                   Preflight phase: plan seeds and prompts, rewrite them before the model loads
├── ideogram4_instant_builder.py   Build the Ideogram 4 Instant model folder on first use (download, convert, link the base components)
├── image_model_loader.py          Image model loading through stored quants (reuse, or save on first use)
├── image_review.py                Review of an image's small render before the upscale
├── image_runner.py                Image generation run orchestration
├── upscale_runner.py              Upscale one existing image: read its recorded settings, run the upscale workflow
├── video_file_upscale_runner.py   Upscale one existing video: check it, stream its frames through SeedVR2, keep its sound and settings
├── review.py                      Shared wait for a review decision
├── seedvr2_loader.py              SeedVR2 loading from its weights folder: a stored 8-bit copy (macOS) or the downloaded FP8 files
├── video_runner.py                Video generation run orchestration
├── config.yaml                    Default configuration (sizes, model presets)
├── backends/                      Backend selection (__init__.py, the only place that picks a platform)
│   ├── prompt_enhancer_session.py Single resident prompt-enhancer model: exclusive use and idle release
│   ├── cuda/                      Windows and Linux (diffusers/CUDA); what image and video share sits here
│   │   ├── block_streaming.py     Stream a model's blocks to the GPU as they run (group offloading, or a parameter swap that carries NF4)
│   │   ├── download_plan.py       Hugging Face downloads the diffusers image, LTX video and transformers enhancer backends need (no torch import)
│   │   ├── driver.py              NVIDIA driver hint when a CUDA build of PyTorch cannot start
│   │   ├── lora_peft.py           Detect and report LoRA files that match no module (peft)
│   │   ├── memory.py              CUDA memory release after a job, and GPU and system memory sizes
│   │   ├── prompt_enhancer.py     Prompt-enhancer LLM (transformers)
│   │   ├── quant.py               NF4 (q4) and FP8 (q8) weights, at load and as stored quants, and the NF4 pre-pack of a model's block linears
│   │   ├── seedvr2.py             SeedVR2 upscaler (vendored PyTorch core, FP8 streamed from system memory)
│   │   ├── image/
│   │   │   ├── backend.py         Image backend
│   │   │   ├── lokr.py            LyCORIS LoKr adapters (applied like mflux)
│   │   │   └── preview.py         Cheap latent → RGB live previews
│   │   └── video/
│   │       ├── audio_carry.py     Pin a clip's last sound at the start of an extension window
│   │       ├── backend.py         LTX video backend
│   │       ├── converted.py       Converted LTX-2.3 folders: write one from a single-file checkpoint
│   │       ├── encode.py          Write frames and audio to MP4 with PyAV (H.264 CRF 18, AAC)
│   │       ├── h3/                MiniMax H3 video backend (CUDA), in atomic modules
│   │       │   ├── adaln.py       The exact AdaLN fold: per-block modulation tables keyed by a forward's distinct timesteps
│   │       │   ├── audio.py       Audio VAE: decode audio latents to a stereo waveform
│   │       │   ├── backend.py     The backend: stages loaded and freed one at a time, Extend, upscale with review
│   │       │   ├── conditioning.py Encode keyframes, an extension tail and its audio into latents
│   │       │   ├── decode.py      Decode video latents chunk by chunk to 8-bit frames, so GPU memory stays flat
│   │       │   ├── frames.py      Clip geometry: canvas and frame count to latent cells and rows
│   │       │   ├── latent_upsampler.py The 3D latent upscaler network (ported, MIT) with its strict fp16 loader
│   │       │   ├── lora.py        Merge the turbo LoRA in fp32; split a user LoRA by target (AdaLN, refiner, PEFT)
│   │       │   ├── refine.py      The windowed refine's helpers: noise drawn once, per-window audio and keyframes, the cross-fade
│   │       │   ├── sampler.py     The denoise loop over a packed sequence with pinned conditioning rows
│   │       │   ├── schedule.py    Sigma schedules, row sets and per-row timesteps
│   │       │   ├── sources.py     Resolve the model's files on disk, downloading what is missing
│   │       │   ├── stored_copy.py The folded 4-bit transformer copy `<name>@q4` (`h3@q4` for the stock model): written once, loaded per clip
│   │       │   └── text_encoder.py The 50-layer NF4 Qwen3-VL encoder and its vision tower
│   │       ├── image_crf.py       Re-compress an image-to-video conditioning image through H.264 at a model version's CRF
│   │       ├── memory_estimate.py System-memory estimate of LTX, read from the weight files' headers
│   │       ├── runtime.py         Helpers both CUDA video backends share: torch runtime tuning, fitting a conditioning image to a canvas
│   │       ├── sampler.py         Euler-ancestral sampling for LTX-2.5's distilled rows: seed and sigma maths, the shared noise holder and the scheduler swap
│   │       ├── streaming_decode.py Decode LTX-2 video latents a few frames at a time, straight to 8-bit frames
│   │       ├── timestep_runs.py   Run the LTX-2 transformer on per-token (image-to-video) timesteps by runs
│   │       ├── token_chunking.py  Run token-wise layers (feed-forward, RMSNorm) over a long sequence in chunks
│   │       └── weights.py         LTX weights: the transformer (single-file for LTX-2.3, the repo's own for LTX-2.5) and the Gemma 3 or Gemma 4 text encoder, prepared once per quantize level
│   └── mac/                       macOS (MLX)
│       ├── download_plan.py       Hugging Face downloads the mflux image, MLX video and mlx-lm enhancer backends need (no mflux import until planning)
│       ├── h3_video.py            MiniMax H3 video backend (MLX): file resolution, the 4-bit text-encoder copy, LoRA combining and Extend around the vendored port
│       ├── image.py               Image backend (mflux)
│       ├── image_preview.py       Cheap latent → RGB live previews for the mflux backend
│       ├── memory.py              MLX memory release, recommended GPU memory budget, machine memory size and scoped cache limit
│       ├── prompt_enhancer.py     Prompt-enhancer LLM (mlx-lm)
│       ├── seedvr2.py             SeedVR2 upscaler (mflux)
│       └── video.py               LTX video backend (MLX)
├── converters/
│   ├── convert_checkpoint.py      Safetensors checkpoint → diffusers converter (ziv-model model)
│   ├── ideogram4_instant.py       Ideogram 4 Instant's single-file transformer → diffusers BF16 or mflux FP8 layout, one tensor at a time
│   ├── list_assets.py             List installed models, video models, and LoRAs (ziv-model list)
│   └── lora_import.py             LoRA import — local copy and HF download (ziv-model lora)
├── core/
│   ├── types.py                   Shared types (StageOutcome)
│   ├── job_plan.py                Job plan value types (JobPlan, IterationPlan, EnhanceStatus)
│   ├── progress_events.py         Shared image/video workflow progress event helpers
│   ├── latent_preview.py          Latent → RGB projection factors shared by live previews
│   ├── image_types.py             Image request and artifacts
│   ├── video_types.py             Video request and artifacts, and the extend source, spec and segment
│   ├── image_backend.py           Image backend protocol
│   ├── video_backend.py           Video backend protocol and the optional VideoExtender protocol
│   ├── video_family.py            Video family value types (frame rule, capabilities, steps) and the step settling
│   ├── prompt_enhancer.py         Prompt-enhancer protocol
│   ├── seedvr2.py                 SeedVR2 upscaler protocol
│   └── workflow.py                Unified workflow engine (image + video)
├── downloads/                     Hugging Face model downloads shown as progress (see Model Downloads)
│   ├── check.py                   Dry-run check: files and bytes a target still needs, or None when offline
│   ├── child.py                   Run one download in a child process; relay its byte counts, stop it on request
│   ├── ensure.py                  Skip, wait for or run each part of a plan before a model loads
│   ├── flights.py                 Single-flight registry: one owner per destination, waiters mirror its progress
│   ├── plans.py                   Compose a model's targets from the platform planners, Instant sources and SeedVR2 tier files
│   ├── presence.py                Offline presence test and partial-file cleanup
│   ├── progress.py                Byte counting sink for tqdm, speed meter and progress from counts
│   ├── protocol.py                Wire prefix of the child's stdout messages
│   ├── space.py                   Free-space estimate for the main download drive
│   ├── target.py                  DownloadTarget, TargetCheck and DownloadProgress value objects
│   └── worker.py                  Child entry point: snapshot_download one target, print byte counts
├── processing/
│   ├── contrast.py                Contrast adjustment
│   ├── saturation.py              Saturation adjustment
│   └── sharpen.py                 AMD CAS post-processing
├── schedulers/
│   └── beta_scheduler.py          Beta-distribution sigma scheduler
├── utils/
│   ├── alignment.py               Pixel-alignment helpers for resolution, and the size and frame alignment of a video job
│   ├── app_db.py                  The app database `ziv.sqlite3`: path, connections with the standard pragmas, per-component schema versions, corruption quarantine
│   ├── app_log.py                 Error log file, cleared on every start
│   ├── atomic_write.py            Atomic file replacement for config and prompt-file writers
│   ├── audio_resample.py          Polyphase resampling of multi-channel audio
│   ├── config.py                  Config loading (image + video)
│   ├── console.py                 Console formatting
│   ├── ffmpeg.py                  ffmpeg availability check and install, the error for a failed ffmpeg or ffprobe run, and the hidden temp path of in-place rewrites
│   ├── file_links.py              Hard-link model files into a new folder, copying where linking fails
│   ├── filename.py                Output filenames ({set}_{timestamp}) and collision counters
│   ├── h3_lora.py                 Read MiniMax H3 LoRA files of any format into diffusers' converted form
│   ├── h3_memory.py               MiniMax H3 row arithmetic, memory estimates (CUDA GPU and system memory, macOS unified memory) and estimated generation time
│   ├── h3_preset.py               Value types of the MiniMax H3 preset and its resolved files, parsed from the config entry
│   ├── hf_snapshot.py             Find or download the files of one pinned Hugging Face revision (shared by the CUDA and macOS H3 backends)
│   ├── ideogram4_instant.py       Ideogram 4 Instant's pinned build sources, model folder and build record
│   ├── image_model_detect.py      Image model type detection
│   ├── interactive.py             Keyboard interrupt handling and review state
│   ├── lora.py                    LoRA CLI argument parsing
│   ├── lora_family.py             Pure detector of the model family a LoRA was trained for, from its safetensors header and metadata (`LORA_FAMILY_VERSION`)
│   ├── model_files.py             Offline check for fully downloaded model weights
│   ├── model_memory.py            Header-only memory estimates (MLX unified memory, with a per-stage LTX clip estimate by size and length; CUDA GPU and system memory)
│   ├── page_cache.py              Drop model files from the page cache once written or read
│   ├── paths.py                   ~/.ziv/ model store resolution
│   ├── platform.py                Platform metadata and platform-aware alias values
│   ├── prompt_compose.py          Structured prompt flattening & snippets; video scene helpers (compose scene + prompt, quoted-line stripping, original-prompt check)
│   ├── prompt_document.py         Prompt file ↔ editable document for the Web UI builder (comment-preserving, ruamel.yaml)
│   ├── prompt_document_check.py   Builder document previews and problems, with the prompt loader's rules
│   ├── prompt_enhance.py          Prompt-enhancement settings, length plans, chat messages, run, and post-processing (no model I/O)
│   ├── prompt_enhance_video.py    Video prompts as a scene and an action: word plans, chat messages, output parsing and the run (no model I/O)
│   ├── prompt_enhance_caption.py  Ideogram 4 JSON captions: the model writes labelled lines, code builds the caption (no model I/O)
│   ├── prompts.py                 Prompt file loading
│   ├── provenance.py              Settings embedded in generated PNG/MP4 files: build, embed, read
│   ├── safetensors_stream.py      Read and write safetensors one tensor at a time, keeping them out of the page cache; reads a header and its metadata in one bounded read
│   ├── seedvr2.py                 SeedVR2 weight layouts, settings, memory estimate and tier choice
│   ├── stored_quant.py            Stored-quant naming, manifests and folder handling, and the write-once stored-copy routine
│   ├── upscale.py                 Upscale output sizes, size limits, SeedVR2 options (images and video), the video batch size and denoise defaults
│   ├── video_extend.py            Extend: source checks and problem codes, loading the source, frame and audio-sample arithmetic of the window
│   ├── video_model_detect.py      Video model type detection
│   ├── video_probe.py             ffprobe a clip: stream facts, rotation and the checks' inputs
│   ├── video_splice.py            Join a generated segment onto the source clip (stream copy or one re-encode) with the audio
│   ├── video_support.py           Refuse a video request the model's family cannot do on this platform
│   └── video_tail.py              Read a clip's last frames, or all its frames, at their displayed size
├── web/
│   ├── config.py                  Web UI config loading and model inventory discovery
│   ├── config_api.py              JSON config response assembly
│   ├── config_contract.py         Writable config semantics and path readback helpers
│   ├── defaults.py                Ratio and size defaults for Web UI contracts
│   ├── download_lane.py           Models-page downloads: a queue run one at a time on its own thread, beside the job queue
│   ├── extend_api.py              Extend routes: check and describe a source video, fit its window to memory, build the rewrite context of an extension
│   ├── first_run.py               The model the first-run dialog preselects, and the dialog's payload block
│   ├── gallery.py                 Gallery inventory and response serialization
│   ├── gallery_index.py           Gallery component of the app database: facts read from gallery files, reconciled with the folder on each listing
│   ├── job_contract.py            Web job lifecycle, terminal event, and control contract
│   ├── job_log.py                 Persisted log of finished jobs: entry builder, filters, paging and the capped JSON-lines store
│   ├── json_fields.py             Typed reads of optional fields in a JSON request body
│   ├── lora_index.py              LoRA component of the app database: the cached family of each LoRA file, detected from its header when first seen
│   ├── lora_fit.py                Size the LoRAs a Web UI memory estimate covers (MiniMax H3 only), resolved as a job resolves them
│   ├── model_delete.py            Delete installed models, LoRAs, and HF downloads
│   ├── model_download_plan.py     Plan a Models-page row's downloads: targets, whether it needs any, sizes and free space
│   ├── model_inventory.py         Backend model inventory for Web UI contracts
│   ├── model_status.py            Per-model downloaded / memory-fit status for the SPA
│   ├── path_browser.py            Host folder listings for the in-app path browser
│   ├── presets.py                 Settings presets for the Web UI: file format, slugs, validation, and the store (list, write, rename, delete under one lock)
│   ├── prompt_builder.py          Prompts page API: load, preview, save (revision-checked), create prompt files, and append a Workspace prompt
│   ├── prompt_files.py            Read and atomically update host-local prompt files
│   ├── request_guard.py           DNS-rebinding and cross-site request protection
│   ├── upscale_api.py             Plan a viewer upscale job: refinement model, carried settings, limits
│   ├── video_clip.py              Resolve the model, steps, size and length a video job runs at, shared by the job and its memory estimate
│   ├── web_runner.py              Background workers for generation batches
│   ├── workspace_api.py           Shared SPA payloads for workspace and models routes
│   ├── workspace_contract.py      Workflow aliases and static workspace capabilities
│   └── server.py                  FastAPI route wiring and request parsing
└── workflows/
    ├── enhance_stage.py           Applies the planned rewrite in image and video workflows
    ├── image_stages.py            Image pipeline stage definitions
    ├── seedvr2_stages.py          SeedVR2 upscale stage
    └── video_stages.py            Video pipeline stage definitions, including extend_video and join_clips

prompts.yaml                       Default prompt definitions

bench/enhance/                     Prompt-enhancer bench (not packaged): cases, checks, runner and compare report

packages/                          Vendored third-party code, never edited by hand: ltx_core_mlx, ltx_pipelines_mlx, seedvr2_torch and minimax_h3_mlx (see the Vendored sections)
```

## Architecture Overview

### Backend Protocol

Platform backends live in `backends/mac/image.py` (mflux/MLX), `backends/cuda/image/backend.py` (Windows/Linux diffusers image), `backends/mac/video.py` (MLX video), and `backends/cuda/video/backend.py` (Windows/Linux diffusers video). MiniMax H3 has its own video backends, `backends/cuda/video/h3/backend.py` and `backends/mac/h3_video.py`, chosen by model family (see [Video Families](#video-families)). Image backends implement the `ImageBackend` Protocol from `core/image_backend.py`; video backends implement the `VideoBackend` Protocol from `core/video_backend.py`. The prompt-enhancer LLM follows the same split: `backends/mac/prompt_enhancer.py` (mlx-lm) and `backends/cuda/prompt_enhancer.py` (transformers) implement the `PromptEnhancer` Protocol from `core/prompt_enhancer.py`. Backend and accelerator selection stays centralized in `backends/__init__.py` — never branch on `sys.platform` elsewhere to pick a backend. OS-level checks unrelated to backends (drive listings, ffmpeg install hints, terminal handling) are fine where they are needed.

SeedVR2, the plain upscale of existing images, has its own small Protocol, `SeedVR2Upscaler` in `core/seedvr2.py`, implemented by `backends/mac/seedvr2.py` (mflux) and `backends/cuda/seedvr2.py` (the vendored PyTorch core, see [Vendored SeedVR2](#vendored-seedvr2)). `supports_seedvr2()` and `get_seedvr2_upscaler()` in `backends/__init__.py` expose it, so other code asks for the capability instead of checking the platform. Each engine's `weights` (`utils/seedvr2.SeedVR2Weights`) say which Hub files it reads (repo and file each), its precision label, which `seedvr2` config tier list prices it (`tiers` or `discrete_tiers`) and whether it keeps a stored copy. `seedvr2_loader.py` follows that, with one hidden weights folder per tier (`<models>/.seedvr2/<model>@<precision>`): on macOS it loads through a stored 8-bit copy in the same way `image_model_loader.py` does for image models; on CUDA the engine downloads the FP8 transformer and the VAE into the folder, because the vendored core reads both from one folder. The CUDA 7B tier uses the FP8 file whose last block is FP16 (`AInVFX/SeedVR2_comfyUI`), since the plain FP8 file draws streaks in flat areas. `tests/test_mflux_seedvr2_contract.py` records the mflux surface the macOS engine relies on; re-run it on every mflux bump. The CUDA engine also implements the optional `SeedVR2VideoUpscaler`: `upscale_video` takes an iterator of frames and yields the enlarged frames, two batches of the core per chunk, and reports `(batches_done, batches_total)` after each decoded batch. `supports_video_file_upscale()` in `backends/__init__.py` tells whether the platform's engine has it, so callers ask for the capability instead of checking the platform. The macOS engine upscales images only.

Which Hugging Face files a model needs is not part of any backend Protocol, because a plan must be available without loading a backend. Each platform has a light planner module, `backends/mac/download_plan.py` and `backends/cuda/download_plan.py`, with module functions `image_download_plan`, `video_download_plan` and `enhancer_download_plan` that import neither torch, diffusers nor (until planning) mflux. `backends/__init__.py` holds the dispatch (`image_download_plan`, `video_download_plan`, `enhancer_download_plan`), so the platform is still chosen only there. See [Model Downloads](#model-downloads).

Backends are grouped by platform: `backends/cuda/` (Windows and Linux) and `backends/mac/`. A task with helpers of its own gets a folder (`cuda/image/`, `cuda/video/`); what two tasks share stays one level up. The platform folders never import from each other. Platform selection still happens only in `backends/__init__.py`.

Windows and Linux video resolution is config-driven through `video_model_presets.ltx.diffusers`: `ltx-4` and `ltx-8` both resolve to `default_repo`, `quantize` maps each name to its level (there is no unquantized path), and `utils.config.ltx_load_options` turns the config into the backend's load arguments for both the CLI and the Web UI. Only distilled LTX checkpoints are supported (LTX-2.3 and, as `ltx25-4`, LTX-2.5).

**Model versions.** The family block describes LTX-2.3. A version is `video_model_presets.<family>.versions.<id>`: an `aliases` list plus any keys of the family block, merged over it (`utils.config.video_preset`) for the aliases it lists, with dicts merged and every other value, `null` included, replacing. The merge order is CLI flags, then the version, then the family, then the global defaults. An alias belongs to at most one version, and a malformed `versions` block or `aliases` list is a `ValueError`. `ltx25` lists `ltx25-4` and sets its own `default_repo` (a pinned revision of the ungated `unsloth/LTX-2.5-FP8`), a `null` `transformer` (the repo's own transformer, so the download plan skips the single-file checkpoint), `repo_ignore` for the repo's root single-file checkpoints, `image_crf` and `sampler`. Only the `diffusers.*` keys are read from a version block; `ltx_load_options` adds `repo_ignore`, `sampler` and `image_crf` only when the model's preset sets them, so the options of the 2.3 aliases are unchanged. `ltx_version` returns the id of the version that lists a model, or `None`; `ltx_load_options` refuses `euler_ancestral` for any model that is in no version block (a `None` model, an LTX-2.3 alias, a repo id or a path), on every platform. Step defaults and the other keys outside `diffusers` stay family-level. On macOS `ltx25-4` resolves to a pinned, ungated MLX pack through `model_aliases.ltx25-4.darwin` in `config.yaml` (`MarceloFSO/ltx-2.5-mlx-q4-fast` at a fixed revision: the distilled files only, with the sha256 values of `dgrauet/ltx-2.5-mlx-q4`); version membership does not choose the pack. That pack bundles its Gemma 4 text encoder at the folder root, and version membership only drives that: `downloads/plans.video_plan` passes `text_encoder_bundled=True` for any alias that is in a version, and `backends/mac/download_plan` then plans the pack alone, without the Gemma 3 repo; the CUDA planner accepts the keyword and ignores it. A repo or folder that is not an alias is not in a version: it plans Gemma 3 as before, and once its files are on disk the status and memory estimate read them as they are. `backends/mac/video.load_model` ignores `sampler`, `image_crf` and `repo_ignore` on purpose: the pack decides the sampler (`is_ltx25_pack`), the MLX conditioner sets the image quality, and the pack has no root checkpoints to skip. It resolves a pinned `owner/repo@revision` to a snapshot folder itself, because the vendored pipeline downloads by repo id only, and it raises `ValueError` for a 2.5 pack with no root text encoder.

LTX-2.5 uses a Gemma 4 text encoder, which `cuda/video/weights` selects by the `text_encoder` class in the repo's `model_index.json` (`utils.model_files.ltx2_text_encoder_class`). Gemma 4 has its own checkpoint renames, ahead of the Gemma 3 set, and the FP8 skip patterns are shared. Completeness checks (`utils.model_files`) ignore the LTX-2 components the app never loads (`LTX2_UNLOADED_COMPONENTS`: `processor`, `prompt_enhancer`, `duration_head` and `diffusion_decoder`), and the backend passes those as `None` when it builds the pipeline.

`cuda/video/backend.load_model` keeps LTX within the system memory of a 32 GB machine and the GPU memory of a 10 GB one. The rules a change must keep:

- The text encoder and the transformer never share system memory unless both fit (`_check_system_memory`). When they do not, they take turns: the transformer is released while the text encoder encodes each new prompt (`_encode_prompt`, `_attach_transformer`, `_release_transformer`); when they do, the text encoder stays loaded between prompts. LoRAs are applied when the transformer is attached.
- The transformer is the single-file checkpoint in `video_model_presets.ltx.diffusers.transformer`, or the repo's own `transformer` folder when a version sets it to `null` (LTX-2.5). `cuda/video/weights` prepares it and the text encoder once per level under `<models>/<alias>@q<bits>/`, streaming tensor by tensor through `safetensors_stream` so a 46 GB file never sits in memory. The NF4 pre-pack (`nf4_linear_names`, `pack_nf4`, `swap_in_linear4bit`, `place_nf4`) is `cuda/quant`'s, and copies are written through `utils/stored_quant.ensure_stored_copy`.
- In low-memory mode the transformer streams to the GPU block by block (`block_streaming`): FP8 with diffusers' group offloading, NF4 with `stream_blocks`, because group offloading leaves bitsandbytes' quantization state on the CPU. The VAEs and vocoder move to the GPU only to encode or decode. Video never uses `enable_model_cpu_offload()`.
- Nothing may hold a full-length activation per token beyond the hidden states: the token-wise layers run in chunks (`token_chunking`), image-to-video's per-token timesteps are applied per run of equal timesteps (`timestep_runs`), and the final pass returns latents that `streaming_decode` decodes a few frames at a time. Each keeps diffusers' output unchanged, and its tests compare against diffusers.
- Clips are written by `video_encode` (H.264 CRF 18 and AAC, as on macOS).

A version's `sampler` selects how the distilled sigmas are walked. `euler_ancestral` (LTX-2.5) swaps in diffusers' `LTXEulerAncestralRFScheduler` for each denoising pass through `cuda/video/sampler`: a terminal sigma of 0 is appended and the last timestep dropped, the video and audio schedulers share one noise generator, and each stage gets a fresh scheduler seeded at the job seed plus 10000 (stage 1) or 20000 (stage 2). A pass that carries conditions, which is every Extend pass, always uses the stock deterministic scheduler, because the ancestral step would re-noise the pinned tail. macOS follows the same rule with the vendored loop: `_generate_one_stage` walks `LTX_2_5_DISTILLED_SIGMAS` through `euler_ancestral_denoising_loop` for text to video and image to video on a 2.5 pack, and Extend stays on `denoise_loop`, so Extend is deterministic on both backends. A version's `image_crf` re-compresses an image-to-video conditioning image through `cuda/video/image_crf` before the app's fit to the clip's size, as upstream does; the LTX-2.3 aliases set neither key and keep diffusers' CRF 33 inside the pipeline.

`backends/cuda/video/backend.py` passes the distilled sigma tables from `diffusers.pipelines.ltx2.utils` on every generation call, with `num_inference_steps` equal to their length and every guidance term at its neutral value (`_DISTILLED_GUIDANCE`); diffusers' own defaults for the audio, spatio-temporal and modality terms over-cook a distilled model. Fewer steps than the table keep its first sigma and its tail. The `--upscale 2` flow is two-stage:

1. Stage 1 runs the generation pipeline at half the requested size with `output_type="latent"`.
2. An `LTX2LatentUpsamplePipeline` upsamples the latents 2×. It is built around the generation pipeline's own VAE and is never moved with `.to()` or given CPU offload, since either would re-hook that shared VAE. In low-memory mode the upsampler model waits on the CPU and visits the GPU only for this pass.
3. Stage 2 calls the same LoRA-bearing generation pipeline at full size with the upsampled latents, the stage-1 audio latents and the stage-2 sigmas. The scheduler is swapped for a static-shift copy for that call only and restored afterwards, even when the call fails.

The upsampler weights are one safetensors file resolved with `hf_hub_download` from `video_model_presets.ltx.diffusers.latent_upsampler` (`repo`, `filename`; read with `utils.config.ltx_latent_upsampler_source`) before the generation pipeline loads, and only when upscaling. Both stages report steps as one run under the `video_upscale_stage_1` and `video_upscale_stage_2` phases, as the macOS backend does.

Extending a clip is an optional capability, not part of `VideoBackend`: the `VideoExtender` Protocol in `core/video_backend.py` declares `extend_video(model, source, prompt, *, extend_frames, seed, steps, segment_dir, audio_output, step_callback=None)`, which both LTX backends and both MiniMax H3 backends implement (`extend_frames` is a multiple of the family's `extend_frame_step`). Callers gate on `supports_video_extend()` in `backends/__init__.py` (beside `supports_video_audio()`) and look the method up on the backend; the Web UI contract drops the `extend` workflow, and the CLI refuses `--extend`, when it is false.

On Windows and Linux, `load_model(mode="extend")` loads `LTX2ConditionPipeline`, and `extend_video` pins the tail frames with `LTX2VideoCondition(frames=tail, index=0, strength=1.0)` in a one-stage window at the source's size and frame rate. diffusers gives every audio token one timestep, so `cuda/video/audio_carry` pins the carried sound itself, as the macOS denoise mask does: the tail's sound (read by `utils/video_tail.read_source_tail_audio` at the audio VAE's rate) becomes a log-mel spectrogram with the LTX audio processor's settings and is encoded with `audio_vae.encode(...)[0].mode()`; `pin_audio_prefix` is a transformer pre-hook that, on every forward pass, puts those tokens (packed and normalized with the pipeline's `_pack_audio_latents` and `_normalize_audio_latents`) clean at the start of `audio_hidden_states` and gives them `audio_timestep` 0, leaving `audio_sigma` per sample; `restore_audio_prefix` puts the clean latent back into the result's audio latents (`output_type="latent"` returns them denormalized as `(batch, 8, tokens, 16)`) before they are decoded. The window's frames go to the segment through `utils/video_splice.write_segment_frames`. `tests/test_diffusers_ltx2_contract.py` pins the diffusers names and layouts this uses, including the Euler-ancestral scheduler, the optional pipeline components and the image-conditioning CRF helper. Re-run it on every diffusers bump.

The Windows/Linux video backend is CUDA-only. Validation happens lazily during backend load so importing the package or running non-video code paths does not require torch, diffusers, or a GPU.

The Windows/Linux diffusers image backend is also CUDA-only. Validation happens lazily during image backend load so importing the package or running non-image code paths does not require torch, diffusers, or a GPU.

### Video Families

LTX and MiniMax H3 share one seam, so code asks a family what it can do instead of branching on its name.

- **Registry.** `backends/__init__.py` holds the `(platform, family)` map of backend factories: `get_video_backend(family)` creates a backend on first use and `video_family_available(family)` says whether the platform has one. `utils/video_model_detect` maps an alias, a repository prefix or a local folder's contents to a family (`ltx` or `h3`); an H3 folder is recognised before the LTX path rule, since its path may contain "ltx".
- **Description.** `video_model_presets.<family>` in the config is read by `utils.config.video_family` into a frozen `core.video_family.VideoFamily`: the label, the `frame_rule` (alignment, offset and rounding of valid frame counts), the `capabilities` (`i2v`, `end_frame`, `extend`, `upscale`, `upscale_review`, `lora`, `audio`, `low_memory`), the default and maximum steps and optional `step_options`, the `default_frames`, the `extend_frame_step`, the `carry_frames`, the optional `extend_fps` (the one frame rate an extension source must run at; H3 24, LTX none, because it extends at the source's rate) and the optional `upscale.default_upscale_steps` (the family's `default_steps` without it).
- **Capabilities with a platform overlay.** `platform_capabilities` replaces some capabilities per `sys.platform` (H3 loses `upscale` and `upscale_review` on `darwin`). `backends.video_capabilities(config, family)` returns the result limited further by what this platform's backend supports, and nothing at all for a family without a backend here. `utils/video_support.unsupported_video_request` turns a missing capability into the one refusal message the CLI and the Web UI share.
- **Frame rule.** `utils.alignment.align_frames(frames, rule)` applies it for LTX (8k+1, nearest) and H3 (17k+5, rounding up). The rule is part of the Workspace payload, so the frontend's duration maths uses it instead of a copy.
- **Steps.** `core.video_family.settle_video_steps` moves a request to the family's step options (the nearest, a tie to the higher one) or lowers it to the maximum, and returns a message when it changed the request. The Web UI offers `step_options` as the steps choice.
- **Default frames and extend step.** A family's `default_frames` (H3: 124) replaces the size table's frame count; `extend_frame_step` (LTX 8, H3 17) is the unit of an extension's length, read by `utils.video_extend.extend_step`, the CLI and the Extend routes. `carry_frames` (LTX 25, H3 22) is how many frames of the source's end an extension carries, and `platform_carry_frames` (H3: `darwin: 1`, because the macOS backend continues from the last frame alone) overrides it per `sys.platform`. `backends.video_family_on_platform(config, family)` returns the family with this platform's carry applied, and the callers that size a window (the CLI, preflight and the Extend routes) read the family through it. `utils.video_extend.window_frames(extend_frames, family)` adds the carry to the new frames and moves the sum to the family's `frame_rule` (a no-op for LTX and for H3 on CUDA, whose carries fill whole blocks; 51 new frames with one carried frame give a 56-frame window on a Mac), and `apply_extend_args` and `extend_enhance_context` take the family so that every consumer of the window (the memory estimate, the rewrite context and the request) sees the same count; saved settings record the finished clip's length. It matches `utils.video_tail.CARRY_FRAMES` and the H3 backend's `TAIL_FRAMES`. `utils.video_extend.extend_family_problems` turns a source whose rate differs from `extend_fps` (beyond `EXTEND_FPS_TOLERANCE`, which the H3 backends' own check shares) into a `frame_rate` problem: `/api/extend/source` adds it to the source's problems, and the job submit, the Enhance context and the CLI refuse it through `require_extend_family`.
- **Hidden controls.** `web.workspace_contract.hidden_video_controls(capabilities)` lists the workflow controls a model lacks. The Workspace payload carries them as `hidden_controls` in `video_model_defaults`, beside `frame_rule`, `step_options`, `extend_frame_step`, `supports_extend` and `supports_i2v`.
- **Per-family estimates and downloads.** `utils.model_memory` dispatches on `family` to LTX's estimators or to `utils/h3_memory` (CUDA GPU and system memory, macOS unified memory, and the expected time), and `downloads/plans.py` dispatches the download plan, so a new family adds a branch to each rather than to every caller.

**MiniMax H3 on CUDA.** `backends/cuda/video/h3/backend.py` loads one model for each stage just before it runs and frees it right after (text encoder, keyframe and tail encoders, denoise, video decode, audio decode), so the stages never overlap in memory. It never imports a private name from the LTX backend; helpers both need live in `cuda/video/runtime`. The diffusers pieces it uses (the scheduler, the packed-sequence layout, the VAE's chunked decode) are pinned by contract tests against diffusers 0.41.0 (the tests assert the version, so an upgrade fails them first); after one, re-check `tests/test_cuda_h3_*.py` and update `h3/decode`, which copies the VAE's chunked decode.

- **Conditioning rows.** The packed sequence is `[text | keyframe rows | audio | video]`. A keyframe (image to video, first and last frame) adds one latent frame of rows in front of the video block at timestep 0.999. An Extend tail is the first 7 latent frames of the video block (the source's last 22 frames) plus the first 37 audio latents of each of the two channels; the audio rows are channel-major, so the held audio is two runs. Held rows are mixed with noise once and then never written: each step applies the schedulers to the generated rows only, selected by a boolean mask.
- **Fold keys.** One forward serves every row, each at its own timestep, and a block's AdaLN output depends only on the distinct timesteps of that forward. The sorted tuple of those timesteps is the fold key. `schedule.ROW_SETS` (`plain`, `keyframe`, `extend`) and the 4- and 8-step schedules decide which keys occur, and `adaln` builds a table per block and key; the stored copy holds the union (24 keys, about 0.7 GB), so every step count and mode loads a subset of one file, and the AdaLN projections themselves are not stored. The key list is part of the copy's manifest: a changed schedule, row set or turbo LoRA scale rebuilds the copy. The upscale's refine loads only the tail keys (from step 6 of the 8-step schedule) of the row sets its windows use: `keyframe` for a window that holds an image and `plain` for the others, so the keys are the union of both, because a middle window and an outer one differ in the keys they ask for.
- **User LoRAs.** `lora.split_user_lora` routes a LoRA by target. AdaLN modules become a delta added to the fold tables, token-refiner modules are merged into the bf16 refiner, and everything else goes to PEFT on the NF4 layers. A target on the timestep embedding is refused, because the tables are a function of it.
- **Long clips.** The feed-forward layers run in chunks (`ffn_chunk_tokens` in the preset, via `cuda/video/token_chunking`), which bounds their temporaries and is what lets 1280×736 and 345-frame clips fit a 10 GB GPU. The decode converts each chunk to 8-bit frames and moves it to system memory at once, so its GPU use does not grow with the clip.
- **Upscale.** `_render_upscaled` runs stage 1 at half size (`_generate` returns the latents, the text embeddings and the token tags), decodes the stage-1 waveform, which the clip keeps, and decodes the half-size video only when there is a `review`, which sees it as a temporary video file. `h3/latent_upsampler.py` then enlarges the stage-1 latents 2x. Its network is a port of LBH-123-AI's `LatentResizer3D` with the attribute names unchanged, so the pinned fp16 weights load strictly. The sampler's latents are already in the VAE's normalised space, and `upscale_latents` normalises them a second time with the VAE's `latents_mean` and `latents_std`: the reference node feeds the network that way and the upscale was validated that way, so the double normalisation is deliberate and a test pins it. The refine (`sampler.refine`) re-noises the enlarged latents and the stage-1 audio rows to step 6 of the 8-step schedule (whatever the stage-1 step count) and runs the last 2 forwards with the user LoRAs, with the start and end images held as full-size keyframe rows; the refined audio is discarded. It runs in windows of 22 latent frames (73 frames) with an overlap of 7, planned by `h3_refine_windows` from the clip length alone (one window for a clip of 73 frames or fewer), so a seed gives the same clip on every card. One CPU draw seeded `seed + 1` provides the noise, `h3/refine.py` slices it and the stage-1 audio per window, the first window holds the first image and the last window the last, and the windows are cross-faded into one latent that is decoded once at full size. Progress counts the stage-1 forwards plus 2 for each window. The estimators in `utils/h3_memory.py` model these stages: the net's GPU peak is linear in the full-size latent volume. Without a `review` the transformer is loaded once (`_load_shared_transformer`, with the union of the stage-1 and refine fold keys) and stays loaded through the waveform decode, the latent upscale and the keyframe encode until the refine uses it; with a `review` the half-size decode needs the memory, so the stage-1 transformer is released and the refine loads its own. `h3_memory.h3_upscale_keeps_transformer` decides from the card's memory: the resident transformer (the layers outside the blocks, the NF4 statistics and the fold tables, `_CUDA_RESIDENT_TRANSFORMER_BYTES`, computed from the model's sizes and not yet measured on a card) is added to the peaks of those stages, and a clip whose peak would pass the share of the card a clip may use loads twice, as it did before. `h3_clip_fit` and `h3_expected_seconds` take the same decision (`review` and `keep_transformer`), so the estimate has no second load (`_CUDA_REFINE_LOAD_SECONDS`) when the backend will not make one.

**MiniMax H3 on macOS.** `backends/mac/h3_video.py` wraps the vendored port (see [Vendored MiniMax H3 MLX](#vendored-minimax-h3-mlx)). It resolves the files, converts the text encoder once into the 4-bit `h3@mlx-te4` copy (which keeps the vision tower), combines user LoRAs with the turbo LoRA into one rank-concatenated adapter, drops the text encoder after its single encode through a one-shot proxy, and extends a clip as image to video from its last frame. Upscale and review are refused, and the Workspace hides them through the platform overlay. It is tested on Linux with a fake `minimax_h3_mlx` module plus syntax-tree contract tests; running it needs a Mac.

### Workflow Stages

Image stage functions in `workflows/image_stages.py` have the uniform signature `(ImageGenerationRequest, ImageWorkingArtifacts) -> StageOutcome`. Video stage functions in `workflows/video_stages.py` have the signature `(VideoGenerationRequest, VideoWorkingArtifacts) -> StageOutcome`. Stages are composed dynamically by `build_workflow()` and `build_video_workflow()`. A video workflow starts with `resolve_prompt_stage`, which composes the scene and the prompt into the text the model receives (`compose_video_prompt`), and `enhance_prompt_stage` applies a planned scene and action rewrite. A job whose arguments carry an `extend_source` swaps the generation stage for the extend pair: `extend_video` (calls `extend_video` on the backend for the new segment only) and `join_clips` (`utils/video_splice.splice_clips` joins the segment onto the untouched source and removes the temporary segment directory). The extend variant has no `strip_audio` stage, since the joined clip already has the right audio; its provenance `workflow` is `extend`, with `source_video` and `extend_frames` recorded. `build_upscale_workflow()` upscales an existing image (`load_source_stage` → `upscale_stage` → `sharpen_stage` → `save_image_stage`); the Web UI runs it through `WebRunner.submit_upscale_job` and `upscale_runner.run_upscale`, which emits the same events as a one-image batch, without preflight or the batch loop. `build_seedvr2_upscale_workflow()` is the plain upscale (`resolve_prompt_stage` → `load_source_stage` → `seedvr2_stages.upscale_stage` → `save_image_stage`), chosen when the request's `upscale_method` is `seedvr2`; it does not sharpen. The tier is chosen by `utils/seedvr2.pick_seedvr2_tier` from the `seedvr2` config block and the Mac's memory budget, and `generation.upscale.method` records `seedvr2` or `creative` in the saved file.

### Review Before Upscaling

`--review` pauses a generation after the small render of an upscale until the user approves or discards it. The review state lives in `SkipSignal` (`utils/interactive.py`), the shared wait in `review.py`, and images pass through `review_stage` before `upscale_stage`. Video passes a `review` keyword to the backend, which honours it after its half-size render (the CUDA H3 backend decodes the half-size video, which an unreviewed upscale does not do, and writes it to a temporary video file for the review). `supports_video_upscale_review()` and the `upscale_review` capability of the model's family (see [Video Families](#video-families)) tell the form, server and CLI whether to offer it; MiniMax H3 on macOS has no upscale at all. The Web UI sends approve and discard through the job controls and shows a video review from `GET /jobs/{id}/review`. Everything stays in memory while the review waits, so an approved output equals one made without review.

### Job Phases: Preflight and Generation

Every image and video job (CLI and Web UI) runs in two phases, and only one model is resident at a time:

1. **Preflight** (`preflight.run_preflight`) runs before the generation model loads. It plans every iteration with `plan_iterations` (seed, `{a|b}` expansion, enhancement settings) and, when any iteration asks for auto enhancement, loads the prompt enhancer, rewrites those prompts, closes the enhancer and frees accelerator memory. It returns a frozen `JobPlan` (`core/job_plan.py`).
2. **Generation**: the entry point loads the model and passes the plan to `run_batch` / `run_video_batch` as the required `plan=` argument. Runners raise `ValueError` for a cancelled plan or one whose length differs from the batch, and `RuntimeError` when an iteration is out of order. Retry and repeat keep the planned text and draw a new random seed.

`JobPlan.iterations` holds one `IterationPlan` per iteration in run → set → prompt order, the order the runners loop in. Each carries `seed`, `resolved_prompt`, `enhanced_prompt` and `enhance_status` (`off`, `enhanced`, `failed`, `skipped`); video iterations also carry `scene`, `resolved_scene` and `enhanced_scene`, and a video rewrite returns the scene and the action together. `EnhanceSettings.parts` (`both`, `action` or `None` for auto) picks what a video rewrite writes; `resolve_parts()` turns it into `action` or `both` per job (auto means `action` when extending a clip that has a scene, and a blank scene always means `both`). A rewrite that keeps the scene returns `EnhancePairResult.scene == ""`, so `enhanced_scene` is `None`, no `enhanced_scene` is stored in the metadata, and the provenance `scene` is the user's own, with `scene_input` repeating it. The workflow's `enhance_prompt_stage` only applies the planned rewrite; no workflow stage calls the LLM.

Entry points own the control signal. `image_cli.main` creates the `SkipSignal` and its key listener before preflight and stops it after `run_batch`; Web UI image and video jobs pass their control signal to both phases. Video jobs offer only Quit: `run_video_batch` checks it before each video and raises from the step callback so a running video stops at its next denoising step, then emits `batch_cancelled`. During preflight there is a control boundary before each rewrite and a final one after the enhancer is released: `skip` skips the next rewrite, `pause` waits, `quit` cancels, and `repeat` is dropped. A rewrite stopped by the user (the `cancelled` callback saw Next or Quit) is `skipped`, never `failed`. On Quit, preflight releases the enhancer and memory before emitting `batch_cancelled`.

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

Downloads started from the Models page take a second path: `DownloadLane` (`web/download_lane.py`) runs them one at a time on its own thread. The lane never touches the job executor or the exclusive slot, so generation jobs are admitted while it downloads. Deleting a model, an enhancer model or a Hugging Face download is refused with 409 while a download touches its repo or folder (`_ensure_not_downloading` in `web/server.py`, using `DownloadFlights.conflict`). See [Model Downloads](#model-downloads).

`web/job_log.py` keeps a log of finished jobs in `jobs.jsonl` in the ZIV data directory, one JSON object per line, oldest first. `WebRunner` takes it as the `job_log` argument and writes nothing without one.

- **Which jobs.** Only image, video and upscale jobs that started (`started_at` is set) are logged. A job removed from the queue never started, and `dummy` jobs are not generation jobs, so neither is logged.
- **Once only.** A job is logged when it first reaches a terminal status (`completed_at` still unset). A later terminal event changes the in-memory status but writes no second line, so the log keeps the first outcome.
- **Where the write happens.** `_publish_event` copies the plain facts of the job under `record.lock` and writes last, after the subscriber fan-out and the release of the claim, holding no lock. Every logged terminal is published on the worker thread, so the write never blocks the event loop; the routes of the log are plain `def` and run in the threadpool. If a future change logs queue removals, the write must move off the loop.
- **Failures never reach the job.** Building the entry and writing it share one `try/except Exception` that only emits a `UserWarning`, so the terminal event, the claim release and the next queued job do not depend on the log.
- **Partial failures.** The record counts `generation_finished` events with `status: "failed"` (`failed_runs`) and keeps the last reason. A job that still ends `completed` or `cancelled` keeps that status, and the `failed` filter of `GET /api/jobs/log` matches it through `failed_runs`.
- **Storage.** Every change (append, delete, clear) reads the file, edits the entries and replaces it with `write_text_atomic` under one lock, so a reader never sees a partial line. Lines that are blank, not JSON, not an object or without a string `job_id` are skipped on read and dropped on the next rewrite. Past 2000 entries only the newest 1500 are kept.
- **Best effort.** Two `ziv ui` servers on one data directory race on the rewrite and can lose an entry. A server stopped while a job runs may exit before the worker writes that job's line.
- **Routes.** `GET /api/jobs/log` (50 per page, newest first, `status` filter), `DELETE /api/jobs/log/{job_id}` and `DELETE /api/jobs/log` are declared before the `/api/jobs/{job_id}/...` routes. A file system error from the store is a 500 with a `detail`.
- **Tests.** An autouse fixture in `tests/conftest.py` points `job_log.get_ziv_data_dir` at a temporary directory, so no test writes the real log.

**Video upscale jobs.** `POST /api/upscale` on a gallery video queues an `upscale` job whose context, submit response and snapshot carry `media_type: "video"` (image jobs leave it out); `WebRunner.submit_video_file_upscale_job` runs it through `video_file_upscale_runner.run_video_file_upscale` after the same SeedVR2 download plan and load as the image upscale. The runner builds the result in a hidden `.ziv-tmp-upscale-*` folder beside the source, which the gallery and `/media` never list or serve, and moves it into place at the end, so a quit or a failure leaves nothing. The result embeds the source's config through `provenance.build_video_upscale_payload`, with `workflow: "upscale"` and a `source` block that follows back to the original.

### App Database

`ziv.sqlite3` in the ZIV data directory is one shared SQLite file for features that need to keep data between runs. `utils/app_db.py` is stdlib only, so the CLI and the web layer can both use it; it holds no lock and no module-level connection, and never creates, alters or drops a feature's tables.

- **Path and connections.** `app_db_path()` returns `get_ziv_data_dir() / APP_DB_FILENAME`, resolved at call time so tests can patch `app_db.get_ziv_data_dir`. `connect(path=None, *, components=())` returns a fresh `AppConnection` (a `sqlite3.Connection` that remembers `db_path` and `db_identity`, the `(st_dev, st_ino)` of the file recorded before opening). It sets WAL, a `busy_timeout` of 5 s, `foreign_keys=ON` and `synchronous=NORMAL`, uses autocommit, creates `schema_versions` and brings each listed component to its version. WAL keeps `ziv.sqlite3-wal` and `ziv.sqlite3-shm` beside the file while it is open.
- **Components.** A feature declares `Component(name, version, migrate)`. Its tables are named `<name>_*`, and it never reads or writes another component's tables.
- **`schema_versions`.** The table `schema_versions(component, version)` stores, per component, the highest version that has run on this file, and that number is never lowered. When the stored version (0 if there is no row) differs from the component's, `connect` runs `BEGIN IMMEDIATE`, re-reads the stored version, calls `migrate(conn, stored)`, stores `max(stored, version)` and commits. If `migrate` raises, the transaction rolls back, the row is unchanged and the error propagates. `migrate` also receives a stored version higher than its own, which means a newer ziv wrote the file; each component decides what that means. A data component upgrades step by step and raises, so its data is left alone. The gallery is a cache and keeps a table per version (see [Gallery Index](#gallery-index)).
- **Corruption.** `recover_if_corrupt(conn, exc)` is the one place the policy lives. When `is_corruption(exc)` (`SQLITE_CORRUPT` or `SQLITE_NOTADB`), it closes the connection, calls `quarantine(path, identity)` and warns. `quarantine` moves the file only if the file at the path still has the identity recorded before the connection opened, so a file another process has already replaced is left alone. It renames `-wal`, then `-shm`, then the main file to `ziv.sqlite3.corrupt-<YYYYmmdd-HHMMSS>[-n]` plus the same suffixes, so a new main file is never paired with an old WAL. The app never deletes the database. Errors that do not mean a damaged file (cannot open, read-only, busy or locked, I/O error, a directory at the path) never rename anything. `connect` uses the helper while opening and migrating, then retries once.
- **Connection rules for a feature.** Open one short-lived connection per operation, `with closing(app_db.connect(components=(MY_COMPONENT,))) as conn:`; never keep a module-global connection and never share one across threads. Keep `BEGIN IMMEDIATE ... COMMIT` writes short and free of slow I/O. A lock belongs to the feature that needs it. When an operation fails, pass the exception to `recover_if_corrupt` before deciding what to do. `rollback(conn)` ends an open transaction and ignores a failure to do so; use it in the `except` path of a write.

A future feature, such as the job log, would define `JOBS = app_db.Component("jobs", 1, _migrate_jobs)`, where `_migrate_jobs` raises `RuntimeError` for a `from_version` above 1 and creates `jobs_entries` when it is 0. A gallery rebuild never touches `jobs_entries`, and a quarantine keeps its data in the `.corrupt-*` file.

### Gallery Index

`web/gallery_index.py` is the gallery component of the app database. `list_gallery_assets` calls it on every listing, so `/api/gallery` and `/api/workspace?include_history=true` read a file's embedded metadata once per file version instead of once per request, including the first request after a restart.

- **Raw facts.** A row stores what was read from the file, `{"config": <embedded payload or null>, "image": {width, height, prompt} or {}}`, not the derived `GalleryAsset` fields. The derivation (recorded settings, filename parsing, prompt fallback) runs on every request, so fields that depend on other files or on the clock (`source.id`, `reuse_state`, `missing_source_video`, `modified_label`) are never stale, and a change to the derivation needs no migration. `gallery._read_asset_facts` is the only metadata I/O.
- **Reconciliation.** `_scan_output_root` walks the root with `stat` only, never descending into staging directories, and `GalleryIndex.facts_for` matches each `ScannedFile` against the rows of that root. A row is used only when its `(mtime_ns, ctime_ns, size)` equals the walked stat, so a stale or orphan row costs at most one re-read and can never show a file or wrong metadata. Only walked files are returned. A change of ctime alone (a backup tool, chmod) causes one re-read.
- **When a read is persisted.** Only when no reader failed, the file's mtime is at least 2 s before now, and a re-stat after the read returns the same key. A read that is not persisted is still used in the response and is tried again on the next listing. Persisted reads are committed every 64 files, so an interrupted cold scan keeps its progress.
- **Pruning.** A missing root returns `[]` and keeps its rows. Rows for files the walk did not see are deleted only by a walk without errors that found at least one media file, because `resolve_output_dir` creates the folder and an unmounted drive looks like an empty folder. The delete is conditional on the key the call loaded. A deleted file is therefore never shown, and its row is removed by the next error-free listing of that root that finds at least one file, including when it was deleted while the app was not running. `delete_gallery_assets` calls `GalleryIndex.forget()`, which removes the rows at once.
- **Versioned table.** The table is `gallery_assets_v{N}`, with `N = GALLERY_INDEX_VERSION`. Bump the constant when the columns change or when `_read_asset_facts` reads something different. `_migrate_gallery` always runs `CREATE TABLE IF NOT EXISTS` for its own table, and drops the lower-versioned gallery tables only when the stored highest version is below `N`; it never drops a higher one. Two ziv versions sharing a data folder (a development checkout and an installed one) each keep their own table, so they do not rebuild each other's: the older one re-reads once after the first bump and is then stable.
- **Fallback.** The gallery never fails because of the database. An error while opening or loading falls back to reading every file uncached, with a warning and without touching the file (a corrupt file is first quarantined by `recover_if_corrupt`). A write error keeps the results already read and stops writing for that call.
- **Locking.** `GalleryIndex` holds one `threading.Lock` around a whole reconciliation, so concurrent cold requests read each file once. `forget()` does not take it, so a delete never waits behind a scan.
- **Job completion.** A finished job does not write a row. The next listing picks the output up, and `gallery_asset_for_output_path` reads uncached and never touches the database.
- **Hidden temp.** `embed_mp4_config` and `strip_audio` write their temporary copy as `.ziv-tmp-<name>` (`staging_path()` in `utils/ffmpeg.py`), which the gallery staging rules hide from listings, the walk and `/media`.

### LoRA Index

`web/lora_index.py` is the LoRA component of the app database. It caches which model family each LoRA file was trained for, so the Workspace can list the LoRAs made for the selected model first. `utils/lora_family.py` holds the pure detector (`detect_lora_family(header, metadata)`: metadata first, then a key and shape fingerprint); `utils/safetensors_stream.read_header_and_metadata` reads a header in one read and refuses a declared length above `min(100 MiB, file_size - 8)` before reading it, so no tensor data is read.

- **Table and stat key.** The index is declared as `Component("loras", LORA_FAMILY_VERSION, ...)` on `app_db.app_db_path`. A row of `loras_entries_v{N}` is keyed by `(root, name)` and holds `family` and `size_variant` beside the stat key `(mtime_ns, ctime_ns, size)`; a row is used only when that key equals the file's current stat.
- **Lookup.** `lookup(loras_dir, names)` does one `os.stat` per name and one database read. A file with no row or a stale key is detected inline from its header and stored, so each file is read once until it changes; a header that cannot be read or classified is stored as an unknown family. The same write deletes the folder's rows for LoRAs no longer listed, unless the list is empty: an empty list returns at once without opening the database, so an unreadable folder does not wipe the cache.
- **Cache only.** A database error in a lookup warns once and every LoRA is reported with an unknown family, so the picker falls back to a flat list. Requests never fail because of the index.
- **Versioning.** The table is `loras_entries_v{N}` with `N = LORA_FAMILY_VERSION`. Bump the constant when the detection rules or the columns change; the next lookup creates the new table, drops lower-versioned `loras_entries_v*` tables and re-detects.
- **Payload.** `build_workspace_response` adds `family` and `size_variant` to each `loras` entry; the SPA groups them in `lib/state/loras.ts` (`groupLorasForModel`).
- **Tests.** The autouse `_isolated_app_db` fixture in `tests/conftest.py` gives every test a throwaway database, so no test touches the real `ziv.sqlite3`.

### Settings Presets

Presets are a Web UI feature; the CLI does not know them. A preset is one YAML file, `<data dir>/presets/<mode>/<slug>.yaml` (`mode` is `image` or `video`), with the top-level keys `name`, `model` (or null) and `settings`; the folder decides the mode, so the file does not repeat it. `web/presets.py` owns the format, the slug rules and the store (writes under one lock with `write_text_atomic`).

- **Settings.** The `settings` keys are the Workspace form field names. Unknown keys are ignored when reading and writing, and missing keys count as not set. Prompt, seed, reference image and the workflow are never stored.
- **Routes.** `GET /api/presets` lists both modes sorted by name and skips invalid files. `POST /api/presets` creates one: the server derives the slug from the name and answers 409 when it is taken. `PUT /api/presets/{mode}/{slug}` overwrites (a `mode` in the body is ignored), `POST /api/presets/{mode}/{slug}/rename` renames, `DELETE /api/presets/{mode}/{slug}` removes; an unknown mode or a malformed slug in the path is 422. `POST` takes the mode in its body. Names and slugs are unique within a mode, so image and video presets can share a name.
- **Web apply.** `draft.applyPreset(ctx, preset)` (helpers in `lib/state/presets.ts`) applies a preset the same way in every workflow: keys outside the preset (prompt, seed, reference image, workflow) are untouched, and the capability gates re-run so the new model clears what it cannot use, as a manual model switch does. A model or LoRA that is not installed is skipped without a message.
- **Deselect.** A mode switch, Reset, and an extend or reuse that loads a source's settings clear the applied preset; a preset can be picked again afterwards. The applied slug is a single value in the draft (not per mode); it moves on rename and clears on delete.
- **UI parts.** `PresetMenu.svelte` is the toolbar control and, on a phone, the sheet behind the first chip; `PresetNameDialog.svelte` is the Save as and Rename dialog; `lib/state/presets.svelte.ts` is the list store.
- **Tests.** An autouse fixture in `tests/conftest.py` points the presets folder at a temporary directory, so no test reads or writes the real one.

### Model Downloads

A model that is not on disk is fetched before it loads, as a step of its own, so the Web UI can show bytes, speed and time left and offer Stop. The pieces are plain functions and small value objects in `downloads/`, composed by `ensure_downloaded`:

1. **Plan.** A planner returns `DownloadTarget` values (repo, revision, allow and ignore patterns, optional `local_dir`) for the files the load will read: `downloads/plans.py` composes the platform planners with the Ideogram 4 Instant sources and the SeedVR2 tier files (`image_plan`, `video_plan`, `instant_plan`, `seedvr2_plan`, `enhancer_plan`). A plan is `()` when nothing is to be fetched (a local folder, a built Instant folder) and `None` when it cannot be made now (offline with no cached file index); callers skip the ensure step for `None`. Planners mirror the patterns of the code that will load the model (mflux's weight definitions, mlx-lm's `DEFAULT_ALLOW_PATTERNS`, diffusers' pipeline files) so the downloaded files are exactly the loader's.
2. **Presence.** `target_present_offline` decides without the network whether a target is fully on disk (cached files, complete weights, or a complete `local_dir`). A present part is skipped.
3. **Dry-run.** `check_target` calls `snapshot_download(..., dry_run=True)` and returns the files and bytes still missing. It returns `None` when the Hub cannot be reached (offline mode, no connection, timeout), and raises `RuntimeError` when the Hub refuses (gated or missing repo or revision, server error). A gated or unauthorized refusal is one line from `downloads/hub_access.py` (`refusal_message`): the repo's Hub link plus `hf auth login` when `token_source()` finds no token; the worker, the CUDA image planner (`RuntimeError` instead of `LookupError`), job failures and both CLIs use the same `describe_refusal`, and any other error keeps its own text. `dry_run` needs the network: offline, the part is **unchecked** and the load proceeds as before, with the loader's own text phases.
4. **Child.** `run_in_child` runs `python -m zvisiongenerator.downloads.worker '<target json>'` unbuffered, in its own session (Windows: a new process group) so a terminal Ctrl+C does not reach it. The worker calls `snapshot_download` with a `tqdm_class` sink that counts the two aggregate byte bars ("Downloading bytes" as wire, "Reconstructing" as written; they overlap and are never summed) and prints `ZIV-DOWNLOAD {json}` lines on stdout every 0.5 s while the counts change, then a `done` or `error` line. The parent drains the lines on a reader thread, polls `cancelled()` and the exit code, and treats a missing `done` line as failure. The wire bytes of a xet transfer are a little below the file bytes (deduplication and compression), so a part's last report snaps to 100%. The child exits when its stdin closes, so a hard server kill leaves no orphan.
5. **Flights.** `DownloadFlights` keeps one flight per destination (`DownloadTarget.key`). The first caller owns the download; others wait and mirror its progress with `shared: true`. When a flight ends stopped or unchecked, a waiter re-checks its own target and claims it, so a stale result is never read. Finished files stay, so the re-claimer downloads only the rest.
6. **Ensure.** `ensure_downloaded(targets, flights=, report=, cancelled=)` walks the parts in order and returns `READY`, `UNCHECKED` (a part could not be checked; the rest is skipped and the load proceeds as before), `STOPPED` (this caller owned a download and stopped it) or `DETACHED` (this caller stopped waiting for someone else's download). It raises `RuntimeError` for a refused dry-run or a failed download, naming the repo and the bytes fetched. Reports are throttled to one per second, except a part's first and an owned part's final one. Every collaborator (`present`, `check`, `run`, `clock`) is a parameter.

**MiniMax H3 plans.** `backends/__init__.h3_download_plan` dispatches by platform, and `downloads/plans.py` routes the `h3` family to it (`utils.config.video_companion_repos` lists the repositories downloaded beside the main one, for the Models page and for deleting). Windows and Linux (`cuda/download_plan.h3_download_plan`) list the main repository at its pinned revision with `repo_allow` and `repo_ignore` patterns (the parts the app never loads are left out), the first four shards of the NF4 text encoder, the turbo LoRA file, and with `upscale` the latent upscaler's file (0.69 GB, from its own repository at a pinned revision, which `video_companion_repos` lists with the other companions). macOS (`mac/download_plan.h3_mlx_download_plan`) lists the INT8 transformer's files (unless `-m` is a local folder that carries its own transformer), the `FL2VA` components (tokenizer, processor, both VAEs and text-encoder shards 1-11 and 14) and the turbo LoRA, every file literally, so the presence check cannot pass on an empty or partial folder. Each platform then builds a stored copy from the download on first use: `<name>@q4` (CUDA; `stored_copy.copy_name` names it after the model it was picked by, so the stock `h3` keeps `h3@q4` and a fine-tune has its own) and `h3@mlx-te4` (the macOS text encoder); deleting the model removes its own copy with it.

**Local H3 folders on macOS.** `h3_video.local_int8_transformer` decides what a local `-m` folder contributes, and `resolve_mlx_files` and `h3_mlx_download_plan` both follow it through `local_transformer_dirs`, so the plan and the loader agree. It looks in `FL2VA/transformer` and then `transformer/`. A folder in the port's INT8 layout (`config.json`, `quant_config.json`, `model.safetensors.index.json` and every shard it names) is loaded as is and its plan has no transformer target; an INT8 folder missing its index or a shard raises `FileNotFoundError`. A folder with no transformer weights falls back to the preset's pinned INT8 transformer, as a Hub reference does. A folder with any other transformer (the official bf16 release or a fine-tune of it) raises `ValueError`, because the port cannot convert one yet and the stock transformer must not stand in for it.

**Stored-copy sources.** `utils.stored_quant.file_identities` records each source file's identity in a copy's manifest (`ziv-quant.json`), and a copy whose manifest differs from the current sources is rebuilt. A resolved Hugging Face cache blob is named by its content hash, so its name and size are enough. Any other file, such as one in a local model folder, also records `mtime_ns` and `inode`, since a replaced file can keep its name and size; hashing tens of gigabytes was rejected. A copy built from local files therefore rebuilds once after this identity was introduced, and again when the file is replaced or moved. Copies built from Hub downloads keep their manifests unchanged. `utils.quantized_weights.decoded_tensors` multiplies scaled FP8 inputs by their scales instead of casting them, which only matters for local FP8 fine-tunes; it also decodes ComfyUI's INT8, ConvRot, W4A4, W4A8, W6A8, NVFP4 and MXFP8 weights for the video converters.

**Stop and detach.** Stopping an owning job or lane item terminates the child (killing it after `grace_secs`) and deletes its `*.incomplete` files; finished files are kept and resume is per file, so the file in progress starts over. A job that only waits on another download is detached instead: the job ends and the download continues. Stop has to end the process: raising from the tqdm sink does not stop hf_xet's transfer, and an xet download does not grow its `*.incomplete` file as it goes, so there is nothing to poll in the cache either; progress comes only from the sink.

**Memory.** The child's memory is bounded by hf_xet's download buffer, which it sizes from host RAM (about RAM/16 plus RAM/64 per file in flight), not by file size, so a multi-GB file does not need that much memory. `child_env()` sets `HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE=512mb` unless the parent environment already sets it, which about halves the peak at the same speed, so a download can run beside a generation.

**Event contract.** Web jobs publish `model_downloading` with `mode` (`image` or `video`), `role` (`model` or `enhancer`), `model` (label), `repo`, `part` and `parts` (1-based, this job's plan), `bytes_done`, `bytes_total` (null if unknown), `bytes_per_sec` and `download_eta_secs` (null until known) and `shared`. It is published at most once a second per job plus each owned part's final report, and replaces the previous `model_downloading` in `record.history` so a client that connects later replays only the latest one. A job stopped during a download ends with `job_cancelled` carrying `download: "stopped"` or `"detached"`. The on-demand Enhance button's own NDJSON download frames are separate and unchanged. The Svelte job card turns the event into a row inside the load step (the enhancer's into the Enhance step), and ends it at `model_loading` (the enhancer's at its loading or rewriting events).

**Models page.** `web/model_download_plan.py` plans a row (`row_plan`), judges offline whether it needs a download (`row_download_needed`, used by the workspace payload for the Download button) and summarizes a plan with sizes and free space (`plan_summary`, using `downloads/space.py`). `DownloadLane` (`web/download_lane.py`) queues items, runs them with the same `ensure_downloaded` and the server's shared `DownloadFlights`, and reports them as `DownloadItem`s. The routes:

| Route | Purpose |
|---|---|
| `GET /api/models/downloads` | The running item, the queued ones in order and recently finished ones |
| `GET /api/models/download/plan?kind=&name=` | Parts, sizes and free space of a row; 404 for an unknown row, 424 when Hugging Face refuses (gated or missing repo; also from the planner on Windows and Linux), 503 when it cannot be reached |
| `POST /api/models/download` | Queue a row (`kind`, `name`); an already queued or running item for the row is returned; 424 when planning the row is refused by Hugging Face |
| `POST /api/models/downloads/{id}/cancel` | Remove a queued item or stop the running one |

The free-space figure is an estimate: it checks only the main download drive, counts shared xet blobs twice and ignores `HF_XET_CACHE`. The confirmation lets the user proceed anyway.

**Version bumps.** The planners depend on how the loaders choose files. After changing mflux, mlx-lm, diffusers, huggingface_hub or hf-xet, run the pattern pins (`tests/test_mac_download_plan.py` on macOS, which compare the planners with mflux's `get_download_patterns()` and mlx-lm's `DEFAULT_ALLOW_PATTERNS`; `tests/test_cuda_download_plan.py` on Linux, which pin the diffusers and CUDA video patterns against recorded file lists), and repeat the hand test with a small public repo: start a download from the Models page, check that progress moves, cancel mid-file, check that no `*.incomplete` remains and finished files stay, then start a job on the same model while the lane downloads it and check that Stop on the job leaves the download running. A renamed tqdm bar shows as progress stuck at 0 until each part completes, never as a failed download. A pattern that drifts only costs the progress display: the backend downloads the missing files during the load.

### Data Types

Use `@dataclass(frozen=True)` for immutable value objects (inputs, detection results). Use mutable `@dataclass` only for working state. No pydantic or attrs.

### Config Layering

CLI flags > model preset variant > model preset family > global defaults. Config is a plain `dict` loaded from YAML, not a dataclass.

A video family's preset sits at `video_model_presets.<family>`, with per-version overlays under `versions` (see [Backend Protocol](#backend-protocol)). MiniMax H3's preset holds the pinned `repo_ref`, which the `h3` entry of `model_aliases` references on every platform so the pin exists once, then the family keys (`frame_rule`, `step_options`, `default_frames`, `extend_frame_step`, `carry_frames`, `capabilities` and `platform_capabilities`) and one block per platform: `diffusers` for Windows and Linux (download patterns, the text encoder, the turbo LoRA and its scale, the schedules, the attention backend and `ffn_chunk_tokens`) and `mlx` for macOS (the transformer and component file lists, the text-encoder quantization); the macOS preset also carries the family `frame_rule` and the `diffusers` schedules, so both platforms run the same step counts, shifts and clip lengths. `utils/h3_preset` parses a block into frozen values and raises a `ValueError` naming the key when it is malformed; `backends.h3_load_options` passes the parsed preset to the backend's `load_model`.

### Web Model Inventory

A repo-id alias can declare its image family in config via `model_alias_families` (for example `ideo: ideogram4`). The Web UI inventory and workspace bootstrap consult this declared family as an offline fast-path, so a known alias is surfaced without the network `detect_image_model` round-trip that a bare Hugging Face repo id would otherwise require. Content-based detection for arbitrary local model directories is unaffected; only aliases with a declared family skip detection.

`model_labels.<alias>.name` in `config.yaml` is how the Web UI names an alias; `utils.config.model_display_name(config, name)` returns it, or the alias itself when none is set. The server fills `label` on the workspace model options, the Models rows, download plans (and so the lane items) and `model_label` on a job's context and snapshot. The alias stays the id in every API field, CLI flag, prompt file and recorded setting; only the Workspace picker, job card, queue row and Models page show the name.

`model_labels.<alias>.download_size_label` (a `darwin` / `win32` / `linux` map, whole GB) is the one-time download size of an image alias on each platform; the workspace payload carries this platform's value as `download_size_label` on the image options. It is data entered from a dry-run download plan against an empty cache, so update it when an alias's repo or files change. The first-run dialog (`FirstRunDialog.svelte`) opens when every image model is not downloaded, no image download is queued or running and `ui.first_run_skipped` is not set. `first_run` in `config.yaml` names the models it recommends: `web/first_run.recommended_first_run_model` returns `large_model` when the machine's memory (Mac unified memory, CUDA total VRAM) is at least `min_memory_gb` within 5%, else `small_model`, and the workspace payload's `first_run` block carries that and the skip flag. `POST /api/first-run/skip` writes `ui.first_run_skipped` to the user config; it is not a writable config field.

A repo-id alias points at the upstream repo when it is ungated, otherwise at a mirror whose weight files match upstream's sha256.

Per-model Web UI capability flags — `supports_img2img`, `supports_upscale`, `supports_scheduler`, `supports_json_prompt`, `supports_first_sigma`, `dimension_min`, `dimension_max`, and `dimension_step` — are data-driven: they originate in the `model_presets` family entries, flow through `resolve_defaults`, and are emitted into the SPA's `ImageModelDefaults`. Families that do not set them inherit permissive defaults (img2img, upscale and scheduler enabled, JSON caption and first-sigma disabled, `16`/`null`/`16` dimension bounds), so gating is centralized in config rather than branching on family in the web layer.

### Error Conventions

Raise `ValueError`, `FileNotFoundError`, `RuntimeError` directly with descriptive f-string messages. Use `warnings.warn()` with `stacklevel=2` for non-fatal conditions. Add a custom exception class only when callers must catch it distinctly (for example `JobConflictError` in `web/web_runner.py`) or as a private sentinel.

### Web UI Confirmations

Ask the user to approve an action with `ConfirmDialog` (`frontend/src/lib/components/molecules/`), never `window.confirm`. It has no title: it shows the question, an optional line of information, and a cancel button plus a confirm button named after the action (danger-styled for destructive actions). Enter confirms (except on the focused cancel button), Esc cancels, and a held key never confirms; while it is open, no other key reaches the page or a viewer underneath. Call `requestConfirm({ question, info, confirmLabel })`, which resolves `true` on confirm, or render `<ConfirmDialog>` directly when the dialog needs a body or stays open while the action runs (`pending`).

Let the user choose a path on the server's machine with `PathField`, or `PathInput` (a path box with completion) and `PathBrowser` (the browse dialog) where a field needs its own layout. Each takes a browse purpose; the purposes, with their folder or file kind and accepted extensions, live in `web/path_browser.py`, so add one there for a new kind of path.

### Web UI Toasts

Report the outcome of an action with `addToast(message, type, options)` from `$lib/state/toasts.svelte`. One toast shows at a time at the bottom centre of the page content (pages with a fixed left sidebar set `--toast-inset-left`); the rest wait their turn, shown as "+N". A repeat of a waiting or showing message merges into it ("×2"), and an error goes ahead of every non-error. Each tone has a default time on screen (success 2.5 s, info 4 s, warning 7 s); errors stay until dismissed, or give way after 8 s once other toasts are waiting, and hovering or focusing a toast pauses it. Pass `{ action: { label, run } }` for one follow-up such as Retry, and `{ timeout }` only when the default doesn't fit. An action that acts on a page must not outlive it: keep the id `addToast` returns and call `dismissToast(id)` in the page's `onDestroy`. The exception is the download lane (`lib/state/downloads.svelte.ts`): its store raises the done and failed toasts, and their Retry runs the store's request flow rather than page code, so they stay after the page that began the download is gone. Raise one toast per action, and skip it when the screen already shows the result.

Send requests through `api` in `$lib/api/client`, which throws `ApiError` with the HTTP `status` and the server's `detail`. Show a failed request with `errorText(err, fallback)`, never `err.message`, which holds the method, path and raw body. Offer Retry only when `isRetryable(err)`: a 4xx means the server refused the input, and the same request fails the same way again.

### Web UI Design System

The Web UI has one design system. Its tokens live in `@theme` in `frontend/src/app/global.css`, and its component classes are the `ui-*` classes in the same file. Build pages from the shared components in `frontend/src/lib/components/` (`Button`, `Input`, `Select`, `Toggle`, `Badge`, `Panel`, `Segmented`, `ActionBar`, `PageHeader`, `KeyValueList`, `Alert`, `EmptyState`, `InspectorSection`, `InspectorRow`) or, where no component fits, the `ui-*` classes (for example `ui-chip`), rather than styling elements by hand.

- **Size:** controls are 28px (`h-control`), small ones 22px, rows 30px. Buttons that sit together share one height; inside an `ActionBar` every button is 36px and the one marked `main` takes the remaining width.
- **Type:** four sizes: `text-meta` (11px), `text-ui` (12px), `text-content` (13px), `text-title` (16px), plus `text-display` (30px, 24px on compact screens) for the first-run welcome headline and the recommended model's name only. Nunito (`font-heading`) is for titles, area names and the main action; monospace is for numbers, paths and shortcuts.
- **Labels:** field and row labels are sentence case (`ui-label`). The one small-caps style (`ui-area-label`) names areas: panels, sections, table headers and menu groups.
- **Shape and surfaces:** corners are 4px for chips and badges, 6px for controls and 10px for containers. Overlays (menus, popovers, tooltips, toasts, dialogs) use `bg-overlay`, darker than any page surface, with a faint ring and never a drop shadow; `bg-raised` is for section header bands and `bg-fill` for the prompt box.
- **No borders:** containers are separated by shade, and buttons, fields, chips and badges are filled shapes. Dividers between rows and sections stay. Draw an outline only where it means something: focus, a field with an error, the selected tile, and the dashed edge of drop zones and add buttons. Segmented controls are a recessed well with the chosen option as a lighter pill.
- **Main action:** each section (a bar, panel, form, dialog or popover) has at most one main action, a `primary` button: teal fill and a bold Nunito label. Every other button in the section is secondary.
- **Reference pages:** Config and Models centre their content at `max-w-page` (1200px), with the page bar lined up to it. Settings use label-left rows (`FormField layout="row"`), and tables use `ui-table-comfortable`.
- **Colour:** neutrals are Graphite greys with no tint. Teal means on, selected or the main action; coral (`accent-coral`, Blob's beret) marks `$snippets` and amber `{a|b}` choices; red is for errors and destructive actions, which always show red text. Info messages stay teal, never coral. Text is at least 4.5:1 on every surface. Use tokens, never raw palette classes or colour literals; `make check` fails on them.
- **Focus:** every control shows the same 2px teal outline with a 2px offset.
- **Touch is a layer, not a breakpoint:** `@media (pointer: coarse)` in `global.css` raises the size tokens (controls 40px, small controls 32px, rows and action bars 44px, the switch 40 × 24) at every width, so an iPad in the desktop layout gets finger-sized targets and a mouse keeps the desktop density. Build sizes on the tokens, never on literal pixels, so they follow. Numbers get − and + steppers (`InspectorNumber`, press and hold to repeat) because a label cannot be scrubbed when a drag scrolls; `InspectorRow` skips scrubbing on a coarse pointer. The `layout` store (`lib/state/layoutMode.svelte.ts`) tells components the composition (`compact | tall | wide`) and whether the pointer is coarse.
- **Hover only where there is hover:** a touch screen keeps `:hover` on whatever was tapped last, so every `:hover` rule sits inside `@media (hover: hover)` (`make frontend-hover-check`, part of `make check`, fails on a bare one; Tailwind's `hover:` variant already does this). A finger gets an `:active` pressed state under `@media (hover: none)` instead, and anything a mouse reveals on hover is always shown under `(hover: none)`. On a phone, fields are at least 16px so focusing one does not zoom the page, and the page never scrolls or bounces (`overscroll-behavior: none` on `html` and `body`).
- **The visible viewport:** `lib/state/viewport.ts` (started in `main.ts`) sets `--app-height` and `--app-top` from `window.visualViewport`, and puts a page the browser scrolled back at the top when no field is being edited. The app root and every full-screen layer use `.ui-viewport`, which is fixed to those values, so they sit above an on-screen keyboard (iOS ignores `interactive-widget`) and follow Safari's toolbar at once. Build full-screen layers on `.ui-viewport`, never on `inset: 0` or `100vh`.
- **Three compositions of one Workspace:** the same panes (`ComposePane`, `SettingsPane`, the toolbar, the stage, `GalleryStrip`) are arranged by `layout.mode`. *Wide* (1024px and up) is the sidebar beside the stage. *Tall* (640 to 1023px, a portrait tablet) puts the stage and strip over Compose and Settings side by side. *Compact* (under 640px, or under 500px tall) makes the stage the screen: a `TopStrip` and `NavDrawer` replace `TopNav`; a `SettingsStrip` of chips (`settingsChips.ts`) opens one `SettingPopover` at a time, All opens `SettingsPane` in a `Sheet`, and a composer at the foot hosts `ComposePane` folded or open. Everything stays inside the Workspace form, so every field submits in every composition. A running job shows `JobPreview` large with `JobCard` beside it on wide and tall and `JobBar` over it on compact; `ActionMenu` and `Modal` render as a `Sheet` on compact. Nothing is hidden behind a phone pattern on a wide screen.
- **Stable frames:** nothing moves under a finger while a page loads. *The frame first:* a page draws its complete frame synchronously from its composition (bars, strips, panels, the grid, action rows) and requests only fill it; a loading state lives inside the slot it is loading (placeholder chips in `SettingsStrip`, the disabled actions of `AdminPageShell`, the Gallery's placeholder tiles) and never changes the height of anything around it. *Pages remember:* a page component is rebuilt on every visit, so what it loaded lives in `lib/state/pageMemory.ts` (the Workspace context, the Gallery view with its scroll position, the inventory, the config); the page adopts the remembered value on mount, draws it at once, then asks the server again and replaces it. *Anchors do not move:* the top strip, the composer, the settings strip and page bars keep their place and size; content that changes size changes inside a box reserved from the start. A new page follows all three, and page switches stay a hard cut, which is what a native tab switch does.
- **One Gallery:** the strip on the Workspace, the stage's stepping through earlier outputs, the Gallery page and the viewer all read `lib/state/gallery.svelte.ts`, which keeps one list per query (`filter` and `sort`). The default query is the strip's and the stage's and the store loads it; the Gallery page loads its own query's pages and commits them with `setView`, so `removeAssets` and `mergeOutputs` reach every list, a visit comes back where it was, and the page's first visit shows the strip's list. There is no history route or store; `/api/gallery` serves both.
- **One viewer, native on each device:** `molecules/AssetViewer.svelte` is a panel over the page on wide and tall layouts (a backdrop that closes on click, the details column beside the image) and the whole screen on compact. It takes the tile that opened it (`origin`) and flies a copy of the thumbnail between the tile's box and the image's box on open and close (`viewerFlight.ts`, a FLIP on the transform), with the thumbnail standing in until the image has loaded. On touch, a track of three panes follows a sideways drag and settles to the neighbour before the index changes (`viewerPager.ts` decides from offset and velocity); a pull down shrinks the image and clears the backdrop (`--pull`), and lets go into the tile through the same flight; a pull up opens the details as a half sheet in flow under the image; one tap hides the chrome. A hold on the image (`actions/hold.ts`, which swallows the click the lifting finger makes wherever it lands) or a right-click on a tile or the image opens the asset menu, anchored at the pointer through a 1px anchor. The viewer is an escape layer, so Escape closes it, the shortcut list first.
- **One sheet, and drags that follow the finger:** every sheet on a phone is `molecules/Sheet.svelte` (the Job and All settings sheets, the enhancer, menus, dialogs): it slides in and out, is as tall as its content up to the screen under the top strip, and pulls down to close through `actions/sheetDrag.ts`. The grab bar and head carry `data-sheet-handle` and always pull; the body pulls once its content is at the top. The `SettingPopover` and the viewer's details use the same action. Drags are built on `actions/pan.ts`, which decides on the first movement whether a touch is the surface's (and then cancels every move, so the browser cannot start scrolling under it) or the page's; deciding later loses to native scrolling on iOS and Chrome. Its pure helpers (`lockAxis`, `releaseVelocity` from the last 80 ms, `rubberBand`, `canScrollToward`) carry the decisions and are unit-tested. A press and hold is `actions/hold.ts` (fingers and pens only; it marks the node `data-holding` with `--hold-ms` while held, so `.ui-holdable` fills a bar along its foot, and swallows the click and the browser's long-press menu once it fires): Generate's new-seed menu and the asset menu on every `AssetTile` use it. The `NavDrawer` swipes closed on the same tracker, and the Workspace stage uses it to follow the finger sideways through history and up to the Gallery (`features/workspace/stageSwipe.ts` decides the outcome, and a trackpad's sideways wheel steps once per gesture). The Prompts page arranges sets, entries and snippets with pointer events, not HTML drag and drop (which never fires for a finger): `reorderDrag` in `features/prompts/dragDrop.svelte.ts` starts at once from a grip for a finger and after a few pixels for a mouse, and `dropTargetAt` reads the drop zones from `data-dnd-entry`, `data-dnd-set` and `data-dnd-snippet` under the pointer.

### Test Strategy

Mock heavy image and video dependencies in tests. Diffusers backend tests patch the lazy runtime loader, torch CUDA checks, upsampler weight download and loading, export helpers, and PIL image loading so the suite never instantiates a real model, downloads weights, or requires a real CUDA device. Platform dispatch, alias resolution, and Web inventory tests should assert behavior through config and protocol boundaries rather than backend internals.

Download tests never use the network or a real `snapshot_download`: `check_target` and the worker take a fake `snapshot_download`, `run_in_child` takes a fake `popen` (a wrapper command that patches `snapshot_download` with a fake updating the sink), and `ensure_downloaded` and `DownloadLane` take fake `present`, `check`, `run` and `ensure` callables and a fake clock. The pattern pins that need mflux, mlx-lm or diffusers skip themselves with `pytest.importorskip` where those aren't installed.

## Prompt-Enhancer Bench

The unit tests mock the enhancer, so they cover how a rewrite is asked for and parsed, not what a real model writes. The bench in `bench/enhance/` runs the real enhancer over fixed cases and counts how well the rewrites follow the request. Run it before and after changing the enhancer's wording (rules, option instructions, length plans) or a pinned enhancer model, and put the compare summary in the commit body or PR.

```bash
make enhance-bench ARGS="run --label main-fast"                  # all cases, Fast, 3 seeds each
make enhance-bench ARGS="run --label main-smart --model smart"
make enhance-bench ARGS="run --label try --tags quotes,short --seeds 1"
make enhance-bench ARGS="compare main-fast try"                  # writes bench/enhance/runs/compare-main-fast-try.md
make enhance-bench ARGS="rescore main-fast try"                  # recompute checks from stored outputs, no model
```

- **Cases** live in `bench/enhance/cases.yaml`: a prompt (and a scene for video), the `enhance:` settings, the job context (ratio, frames, fps, audio, reference image, the extend source's prompt) and optional expectations (`keep`, `count`, `forbid`). Seeds come from the case id, so a run repeats with the same model, platform and settings.
- **A run** loads the enhancer once and calls the same functions preflight uses (`enhance_prompt`, `enhance_video_prompt`, `enhance_caption`). It writes `results.jsonl` (one line per case and seed, with the system prompt, the output, every model call and the checks) and `meta.json` (commit, model, revision, temperature) to `bench/enhance/runs/<label>/`, which is ignored by version control.
- **Checks:** usable, first try (no retry), format (scene and action pair, caption shape), length within the planned range, quoted text kept exactly, `keep`, `count`, `forbid`, and for extensions no repeat of the earlier prompt. A silent video checks only quoted lettering, since it drops quoted speech by design.
- **`keep`** items match case-insensitively, each word as a stem ("river" finds "riverbank"), with up to two words in between ("green lamp" finds "green desk lamp"); `|` lists alternatives ("old man|elderly man"). **`count`** accepts a number word or digit with up to two words before the noun ("three fluffy cats").
- **Rescore** recomputes the checks of stored runs from their outputs, after a check or a case's expectations change. A video's format result is kept, since the stored text can't show whether the pair needed the fallback.
- **Compare** reports each check overall and per tag, speed, and the ten cases that changed most, side by side with a diff of their system prompts.

The bench loads one enhancer model and nothing else. `tests/test_enhance_bench.py` covers its checks and plumbing with a fake model.

## Vendored LTX

`packages/ltx_core_mlx` and `packages/ltx_pipelines_mlx` are vendored from [`dgrauet/ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx). They are not published to PyPI, and PyPI rejects direct-URL dependencies, so they live in the repository. Never edit them by hand.

- **Pin.** `LTX_COMMIT` in the `Makefile` must be the commit of a release tag; the comment above it names the tag.
- **Update.** `make update-ltx` replaces both packages with the pinned commit. It does not carry their dependencies: mirror the `dependencies` of upstream's `packages/*/pyproject.toml` into ours by hand, macOS-only and capped at the minor version that upstream's `uv.lock` tests (for example `mlx-arsenal>=0.2.4,<0.3`).
- **Internals we rely on.** `backends/mac/video.py` uses upstream attributes and call shapes that are not public API. Re-check each on every bump. `tests/test_ltx_mlx_contract.py` pins their names and signatures by introspection, and the mocked tests in `tests/test_video_upscale.py` cover our calls:
    - `_pending_loras` on the pipeline (LoRAs applied at load)
    - the duck-typed `stepwise` slot and its `bind(..., stage=)` contract, which returns the stage's `on_step` callable
    - `_pre_denoise_flush(video_state, audio_state)` (guards against the Metal watchdog) and `_decode_and_save_video(frame_rate=, seed=)`
    - `dit`, `load()`, `prompt_encoder`, `video_patchifier`, `audio_patchifier`, `low_memory`, and `image_conditioner` (`__call__(fn, free_after=)` and `resolve_crf`)
    - resetting `vae_decoder`, `audio_decoder` and `vocoder` to `None` to free the decoders
    - `utils._orchestration.combined_image_conditionings`, `utils.args.ImageConditioningInput`, and `utils.helpers.create_noised_state(legacy_scalar_blend=True)`
    - `ltx_core_mlx`'s `X0Model`, `snap_output_dimensions(two_stage=)`, `compute_video_latent_shape`, `compute_audio_token_count`, `compute_audio_positions`, `compute_video_positions` and `aggressive_cleanup`
    - `scheduler.shorten_schedule`, `DISTILLED_SIGMAS` and `STAGE_2_SIGMAS`, and `utils.samplers.denoise_loop` with its `on_step(step_idx, total, video_x0, sigma)` signature
    - for LTX-2.5 packs: `utils.generation.is_ltx25_pack` and `BasePipeline.model_dir`, `scheduler.LTX_2_5_DISTILLED_SIGMAS` and `LTX_2_5_STAGE_2_DISTILLED_SIGMAS`, `components.diffusion_steps.EulerAncestralDiffusionStep(eta, s_noise)`, `utils.samplers.euler_ancestral_denoising_loop` (keyword arguments `transformer=`, `stepper=`, `noise_seed=` and `on_step=`), `distilled.ANCESTRAL_ETA`, `ANCESTRAL_S_NOISE` and `ANCESTRAL_NOISE_SEED_OFFSET`
    - the two file rules `utils.model_files` mirrors: `ltx_core_mlx`'s `select_text_encoder` (a root `text_encoder.safetensors` with `text_encoder_config.json` marks a bundled Gemma 4 encoder) and `ltx_pipelines_mlx`'s `blocks._video_vae_names` (the conv decoder of 2.5 packs wins over the 2.3 one)
    - for review before upscaling: `_stage1` (shadowed per call), `Stage1Result.video_tokens`/`audio_tokens`/`latent_dims`, `_load_decoders`, `dit`/`_loaded` (restored after the preview decode), and the model state released after an aborted generation
    - `DistilledPipeline.generate_and_save` for upscale, and `BasePipeline`'s `gemma_model_id` default, the transformer-file preference order and the `low_memory` and `low_ram_streaming` constructor arguments
    - Extending a clip (`LtxVideoBackend.extend_video`), video side: `conditioning.types.VideoConditionByLatentIndex(frame_indices, clean_latent, strength)` pins the carried tail at the start of the window; `retake.encode_video_tensor(encoder, video_tensor)` and `utils.media_io.video_preprocess(frames, height, width)` encode the tail frames inside `image_conditioner(fn, free_after=)`
    - Extending a clip, audio side: `ltx_core_mlx.utils.audio.load_audio(path, target_sample_rate, start_time, max_duration, mono)` reads the tail's sound at its position in the file; `audio_vae.encode_audio(waveform, sample_rate, encoder, processor)` encodes it inside `audio_conditioner(fn, free_after=)`; `extend_distilled.fit_audio_latent(audio_latent, audio_tokens)` fits the `(1, 8, T, 16)` latent before `audio_patchifier.patchify`, and `extend_distilled.carry_audio_tokens(frame_rate, carry_pixel_frames)` gives the token count
    - Extending a clip, decode: the pipeline's `_loaded` flag (cleared with `dit = None` so the next `load()` reloads the transformer), `video_decoder_block.load()` and `video_decoder_block.video_decoder` (the diffusion decoder is rejected), `video_vae.VideoDecoder.tiled_decode(latent, tiling_config)`, `video_vae._compute_decode_tiling(latent_shape, frame_rate, budget_bytes)` (which may return `None` or temporal tiles), `video_vae.decode_cache_limit()`, `video_vae.stream_chunks_to_ffmpeg(chunks, proc)` and `video_vae._ffmpeg_sink(cmd)`, and `audio_decoder_block.__call__(audio_latent)`
    - Extending a clip: `ltx_core_mlx.utils.audio.load_audio` returns an `AudioData` whose `.waveform` and `.sample_rate` are passed to `encode_audio`, and `video_patchifier.patchify` and `audio_patchifier.patchify` return `(tokens, dims)` tuples, so the tokens are `patchify(...)[0]`
- **Converted folders.** `backends/mac/ltx_converted.py` writes LTX checkpoints in the layout of dgrauet's MLX folders, which upstream loads: `transformer.`-prefixed names with the renames of `LTXV_LORA_COMFY_RENAMING_MAP`, the block Linears quantized at group size 64 (the bit width is read back from the shapes), `connector.`-prefixed connectors, and `embedded_config.json`. Re-check that layout against a published folder on every bump.
- **Cache limit.** `low_ram_streaming` makes the pipeline set the MLX buffer-cache limit to 0 for the whole process. `backends/mac/memory.mlx_cache_limit` scopes that to model construction and each generation, then restores the previous limit.

## Vendored SeedVR2

`packages/seedvr2_torch` is the PyTorch SeedVR2 inference core from [`numz/ComfyUI-SeedVR2_VideoUpscaler`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (Apache-2.0), which the Windows and Linux plain upscale runs. Upstream is not a pip package, so it lives in the repository. Never edit it by hand.

- **Pin.** `SEEDVR2_COMMIT` in the `Makefile` must be the commit of a release; the comment above it names the version.
- **Layout.** `make update-seedvr2` copies upstream's `src/` without `src/interfaces/` (the ComfyUI nodes), the two model configs, the fixed text embeddings `pos_emb.pt` and `neg_emb.pt`, and the licence, and writes its own `__init__.py` in place of upstream's, which loads the nodes. Upstream's code imports itself relatively and finds its configs and embeddings three folders above `src/utils/constants.py`, so the copy keeps upstream's layout and is imported as `seedvr2_torch.src`.
- **Update.** It does not carry dependencies: mirror upstream's `requirements.txt` into ours by hand, Windows and Linux only, leaving out `gguf` (only imported for GGUF weights) and `matplotlib` (debug output). `torchvision` comes from the same CUDA index as `torch`, and `opencv-python` is replaced by `opencv-python-headless`.
- **Internals we rely on.** `tests/test_seedvr2_torch_contract.py` pins the entry points the upscaler calls, their keyword arguments and the bundled files.

## Vendored MiniMax H3 MLX

`packages/minimax_h3_mlx` is the MLX pipeline for MiniMax H3 from [`Argus-AiTeam/minimax-h3-mac`](https://github.com/Argus-AiTeam/minimax-h3-mac) (Apache-2.0), which the macOS H3 backend runs. Upstream is not published to PyPI, so it lives in the repository. Never edit it by hand.

- **Pin.** `H3_MLX_COMMIT` in the `Makefile` names the vendored commit and the comment above it names the branch and date. `tests/test_h3_mlx_contract.py` checks it against `H3_MLX_COMMIT` in `backends/mac/h3_video.py`.
- **Layout.** `make update-h3-mlx` copies upstream's `minimax_h3_mlx/` without its command-line modules (`generate_cli.py`, `doctor.py` and `cleanroom_smoke.py`), adds `scripts/quantize_text_encoder.py` as `quantize_text_encoder.py`, and copies the license. Upstream imports itself absolutely as `minimax_h3_mlx`, so the directory keeps that name and is imported as a top-level package.
- **Update.** It does not carry dependencies: mirror upstream's `dependencies` into ours by hand, macOS-only. `mlx-vlm` is pinned exactly, because upstream imports its Qwen3-VL vision model, and a newer `mlx-vlm` pulls a newer `mlx` than the one LTX is tested with. Check that `uv lock` adds packages and leaves `mlx` where it was. On macOS the pinned `mlx-vlm` also needs a newer `starlette` (the Web UI's server library) than Windows and Linux use, so run the Web UI on a Mac after a bump.
- **Internals we rely on.** `tests/test_h3_mlx_contract.py` reads the source as a syntax tree, since Linux has no MLX, and pins the `MiniMaxH3Pipeline.from_pretrained` and `__call__` parameters, the single read of `self.text_encoder` in `__call__`, the `GenerationResult` fields, the low-memory image refusal, the Turbo LoRA key pattern and its six targets, and `quantize_text_encoder.convert`. Import checks for the same names run where MLX is installed.

## Releasing

Pushing a `v*` tag runs the Release workflow, which builds the package, publishes it to PyPI and creates a GitHub Release. The package version comes from the tag (`hatch-vcs`), so the tag must be a valid PEP 440 version:

- **Final releases:** `vX.Y.Z`, e.g. `v0.13.0`.
- **Betas:** `vX.Y.ZbN`, e.g. `v0.13.0b1`, then `b2`, `b3`, … before the final `vX.Y.Z`. Use `aN` and `rcN` the same way for alphas and release candidates.

Pre-release tags are published as GitHub pre-releases, and `pip` installs them only with `--pre` or an exact pin (`pip install z-vision-generator==0.13.0b1`). Before tagging, move the `[Unreleased]` CHANGELOG entries under a `## [<version>] - <date>` heading in a `chore(release): v<version>` commit whose body lists the release highlights. PyPI never accepts the same version twice, so a broken release needs a new tag.
