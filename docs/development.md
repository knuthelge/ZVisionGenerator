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
- Tests that run the real `ffmpeg` and `ffprobe` (in `tests/test_video_probe.py`, `tests/test_video_splice.py` and `tests/test_video_tail.py`) are marked to skip when the tools are missing and are meant to be run locally with ffmpeg installed; run them before changing the extend join. Everything else mocks ffmpeg.
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
├── review.py                      Shared wait for a review decision
├── seedvr2_loader.py              SeedVR2 loading through its stored 8-bit copy (download, quantize, save, delete the download)
├── video_runner.py                Video generation run orchestration
├── config.yaml                    Default configuration (sizes, model presets)
├── backends/                      Backend selection (__init__.py, the only place that picks a platform)
│   ├── prompt_enhancer_session.py Single resident prompt-enhancer model: exclusive use and idle release
│   ├── cuda/                      Windows and Linux (diffusers/CUDA); what image and video share sits here
│   │   ├── block_streaming.py     Stream a model's blocks to the GPU as they run (group offloading, or a parameter swap that carries NF4)
│   │   ├── driver.py              NVIDIA driver hint when a CUDA build of PyTorch cannot start
│   │   ├── lora_peft.py           Detect and report LoRA files that match no module (peft)
│   │   ├── memory.py              CUDA memory release after a job, and GPU and system memory sizes
│   │   ├── prompt_enhancer.py     Prompt-enhancer LLM (transformers)
│   │   ├── quant.py               NF4 (q4) and FP8 (q8) weights, at load and as stored quants
│   │   ├── image/
│   │   │   ├── backend.py         Image backend
│   │   │   ├── lokr.py            LyCORIS LoKr adapters (applied like mflux)
│   │   │   └── preview.py         Cheap latent → RGB live previews
│   │   └── video/
│   │       ├── audio_carry.py     Pin a clip's last sound at the start of an extension window
│   │       ├── backend.py         LTX video backend
│   │       ├── encode.py          Write frames and audio to MP4 with PyAV (H.264 CRF 18, AAC)
│   │       ├── streaming_decode.py Decode LTX-2 video latents a few frames at a time, straight to 8-bit frames
│   │       ├── timestep_runs.py   Run the LTX-2 transformer on per-token (image-to-video) timesteps by runs
│   │       ├── token_chunking.py  Run token-wise layers (feed-forward, RMSNorm) over a long sequence in chunks
│   │       └── weights.py         LTX-2.3 weights: the single-file transformer and the Gemma text encoder, prepared once per quantize level
│   └── mac/                       macOS (MLX)
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
│   ├── prompt_enhancer.py         Prompt-enhancer protocol
│   ├── seedvr2.py                 SeedVR2 upscaler protocol
│   └── workflow.py                Unified workflow engine (image + video)
├── processing/
│   ├── contrast.py                Contrast adjustment
│   ├── saturation.py              Saturation adjustment
│   └── sharpen.py                 AMD CAS post-processing
├── schedulers/
│   └── beta_scheduler.py          Beta-distribution sigma scheduler
├── utils/
│   ├── alignment.py               Pixel-alignment helpers for resolution, and the size and frame alignment of a video job
│   ├── app_log.py                 Error log file, cleared on every start
│   ├── atomic_write.py            Atomic file replacement for config and prompt-file writers
│   ├── config.py                  Config loading (image + video)
│   ├── console.py                 Console formatting
│   ├── ffmpeg.py                  ffmpeg availability check and install, and the error for a failed ffmpeg or ffprobe run
│   ├── file_links.py              Hard-link model files into a new folder, copying where linking fails
│   ├── filename.py                Output filenames ({set}_{timestamp}) and collision counters
│   ├── ideogram4_instant.py       Ideogram 4 Instant's pinned build sources, model folder and build record
│   ├── image_model_detect.py      Image model type detection
│   ├── interactive.py             Keyboard interrupt handling and review state
│   ├── lora.py                    LoRA CLI argument parsing
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
│   ├── safetensors_stream.py      Read and write safetensors one tensor at a time, keeping them out of the page cache
│   ├── seedvr2.py                 SeedVR2 settings, memory estimate and per-Mac tier choice
│   ├── stored_quant.py            Stored-quant naming, manifests and folder handling
│   ├── upscale.py                 Upscale output sizes, size limits, SeedVR2 options and denoise defaults
│   ├── video_extend.py            Extend: source checks and problem codes, loading the source, frame and audio-sample arithmetic of the window
│   ├── video_model_detect.py      Video model type detection
│   ├── video_probe.py             ffprobe a clip: stream facts, rotation and the checks' inputs
│   ├── video_splice.py            Join a generated segment onto the source clip (stream copy or one re-encode) with the audio
│   └── video_tail.py              Read a clip's last frames at their displayed size
├── web/
│   ├── config.py                  Web UI config loading and model inventory discovery
│   ├── config_api.py              JSON config response assembly
│   ├── config_contract.py         Writable config semantics and path readback helpers
│   ├── defaults.py                Ratio and size defaults for Web UI contracts
│   ├── extend_api.py              Extend routes: check and describe a source video, fit its window to memory, build the rewrite context of an extension
│   ├── gallery.py                 Gallery inventory and response serialization
│   ├── job_contract.py            Web job lifecycle, terminal event, and control contract
│   ├── json_fields.py             Typed reads of optional fields in a JSON request body
│   ├── model_delete.py            Delete installed models, LoRAs, and HF downloads
│   ├── model_inventory.py         Backend model inventory for Web UI contracts
│   ├── model_status.py            Per-model downloaded / memory-fit status for the SPA
│   ├── path_browser.py            Host folder listings for the in-app path browser
│   ├── prompt_builder.py          Prompts page API: load, preview, save (revision-checked) and create prompt files
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
```

## Architecture Overview

### Backend Protocol

Platform backends live in `backends/mac/image.py` (mflux/MLX), `backends/cuda/image/backend.py` (Windows/Linux diffusers image), `backends/mac/video.py` (MLX video), and `backends/cuda/video/backend.py` (Windows/Linux diffusers video). Image backends implement the `ImageBackend` Protocol from `core/image_backend.py`; video backends implement the `VideoBackend` Protocol from `core/video_backend.py`. The prompt-enhancer LLM follows the same split: `backends/mac/prompt_enhancer.py` (mlx-lm) and `backends/cuda/prompt_enhancer.py` (transformers) implement the `PromptEnhancer` Protocol from `core/prompt_enhancer.py`. Backend and accelerator selection stays centralized in `backends/__init__.py` — never branch on `sys.platform` elsewhere to pick a backend. OS-level checks unrelated to backends (drive listings, ffmpeg install hints, terminal handling) are fine where they are needed.

SeedVR2, the plain upscale of existing images, is macOS-only and has its own small Protocol, `SeedVR2Upscaler` in `core/seedvr2.py`, implemented by `backends/mac/seedvr2.py` (mflux). `supports_seedvr2()` and `get_seedvr2_upscaler()` in `backends/__init__.py` expose it, so other code asks for the capability instead of checking the platform. `seedvr2_loader.py` loads it through a stored 8-bit copy in the same way `image_model_loader.py` does for image models, and `tests/test_mflux_seedvr2_contract.py` records the mflux surface it relies on; re-run it on every mflux bump.

Backends are grouped by platform: `backends/cuda/` (Windows and Linux) and `backends/mac/`. A task with helpers of its own gets a folder (`cuda/image/`, `cuda/video/`); what two tasks share stays one level up. The platform folders never import from each other. Platform selection still happens only in `backends/__init__.py`.

Windows and Linux video resolution is config-driven through `video_model_presets.ltx.diffusers`: `ltx-4` and `ltx-8` both resolve to `default_repo`, `quantize` maps each name to its level (there is no unquantized path), and `utils.config.ltx_load_options` turns the config into the backend's load arguments for both the CLI and the Web UI. Only distilled LTX-2.3 checkpoints are supported.

`cuda/video/backend.load_model` keeps LTX within the system memory of a 32 GB machine and the GPU memory of a 10 GB one. The rules a change must keep:

- The text encoder and the transformer never share system memory unless both fit (`_check_system_memory`). When they do not, they take turns: the transformer is released while the text encoder encodes each new prompt (`_encode_prompt`, `_attach_transformer`, `_release_transformer`); when they do, the text encoder stays loaded between prompts. LoRAs are applied when the transformer is attached.
- The transformer is the single-file checkpoint in `video_model_presets.ltx.diffusers.transformer`. `cuda/video/weights` prepares it and the text encoder once per level under `<models>/<alias>@q<bits>/`, streaming tensor by tensor through `safetensors_stream` so a 46 GB file never sits in memory.
- In low-memory mode the transformer streams to the GPU block by block (`block_streaming`): FP8 with diffusers' group offloading, NF4 with `stream_blocks`, because group offloading leaves bitsandbytes' quantization state on the CPU. The VAEs and vocoder move to the GPU only to encode or decode. Video never uses `enable_model_cpu_offload()`.
- Nothing may hold a full-length activation per token beyond the hidden states: the token-wise layers run in chunks (`token_chunking`), image-to-video's per-token timesteps are applied per run of equal timesteps (`timestep_runs`), and the final pass returns latents that `streaming_decode` decodes a few frames at a time. Each keeps diffusers' output unchanged, and its tests compare against diffusers.
- Clips are written by `video_encode` (H.264 CRF 18 and AAC, as on macOS).

`backends/cuda/video/backend.py` passes the distilled sigma tables from `diffusers.pipelines.ltx2.utils` on every generation call, with `num_inference_steps` equal to their length and every guidance term at its neutral value (`_DISTILLED_GUIDANCE`); diffusers' own defaults for the audio, spatio-temporal and modality terms over-cook a distilled model. Fewer steps than the table keep its first sigma and its tail. The `--upscale 2` flow is two-stage:

1. Stage 1 runs the generation pipeline at half the requested size with `output_type="latent"`.
2. An `LTX2LatentUpsamplePipeline` upsamples the latents 2×. It is built around the generation pipeline's own VAE and is never moved with `.to()` or given CPU offload, since either would re-hook that shared VAE. In low-memory mode the upsampler model waits on the CPU and visits the GPU only for this pass.
3. Stage 2 calls the same LoRA-bearing generation pipeline at full size with the upsampled latents, the stage-1 audio latents and the stage-2 sigmas. The scheduler is swapped for a static-shift copy for that call only and restored afterwards, even when the call fails.

The upsampler weights are one safetensors file resolved with `hf_hub_download` from `video_model_presets.ltx.diffusers.latent_upsampler` (`repo`, `filename`; read with `utils.config.ltx_latent_upsampler_source`) before the generation pipeline loads, and only when upscaling. Both stages report steps as one run under the `video_upscale_stage_1` and `video_upscale_stage_2` phases, as the macOS backend does.

Extending a clip is an optional capability, not part of `VideoBackend`: the `VideoExtender` Protocol in `core/video_backend.py` declares `extend_video(model, source, prompt, *, extend_frames, seed, steps, segment_dir, audio_output, step_callback=None)`, which both LTX backends implement. Callers gate on `supports_video_extend()` in `backends/__init__.py` (beside `supports_video_audio()`) and look the method up on the backend; the Web UI contract drops the `extend` workflow, and the CLI refuses `--extend`, when it is false.

On Windows and Linux, `load_model(mode="extend")` loads `LTX2ConditionPipeline`, and `extend_video` pins the tail frames with `LTX2VideoCondition(frames=tail, index=0, strength=1.0)` in a one-stage window at the source's size and frame rate. diffusers gives every audio token one timestep, so `cuda/video/audio_carry` pins the carried sound itself, as the macOS denoise mask does: the tail's sound (read by `utils/video_tail.read_source_tail_audio` at the audio VAE's rate) becomes a log-mel spectrogram with the LTX audio processor's settings and is encoded with `audio_vae.encode(...)[0].mode()`; `pin_audio_prefix` is a transformer pre-hook that, on every forward pass, puts those tokens (packed and normalized with the pipeline's `_pack_audio_latents` and `_normalize_audio_latents`) clean at the start of `audio_hidden_states` and gives them `audio_timestep` 0, leaving `audio_sigma` per sample; `restore_audio_prefix` puts the clean latent back into the result's audio latents (`output_type="latent"` returns them denormalized as `(batch, 8, tokens, 16)`) before they are decoded. The window's frames go to the segment through `utils/video_splice.write_segment_frames`. `tests/test_diffusers_ltx2_contract.py` pins the diffusers names and layouts this uses.

The Windows/Linux video backend is CUDA-only. Validation happens lazily during backend load so importing the package or running non-video code paths does not require torch, diffusers, or a GPU.

The Windows/Linux diffusers image backend is also CUDA-only. Validation happens lazily during image backend load so importing the package or running non-image code paths does not require torch, diffusers, or a GPU.

### Workflow Stages

Image stage functions in `workflows/image_stages.py` have the uniform signature `(ImageGenerationRequest, ImageWorkingArtifacts) -> StageOutcome`. Video stage functions in `workflows/video_stages.py` have the signature `(VideoGenerationRequest, VideoWorkingArtifacts) -> StageOutcome`. Stages are composed dynamically by `build_workflow()` and `build_video_workflow()`. A video workflow starts with `resolve_prompt_stage`, which composes the scene and the prompt into the text the model receives (`compose_video_prompt`), and `enhance_prompt_stage` applies a planned scene and action rewrite. A job whose arguments carry an `extend_source` swaps the generation stage for the extend pair: `extend_video` (calls `extend_video` on the backend for the new segment only) and `join_clips` (`utils/video_splice.splice_clips` joins the segment onto the untouched source and removes the temporary segment directory). The extend variant has no `strip_audio` stage, since the joined clip already has the right audio; its provenance `workflow` is `extend`, with `source_video` and `extend_frames` recorded. `build_upscale_workflow()` upscales an existing image (`load_source_stage` → `upscale_stage` → `sharpen_stage` → `save_image_stage`); the Web UI runs it through `WebRunner.submit_upscale_job` and `upscale_runner.run_upscale`, which emits the same events as a one-image batch, without preflight or the batch loop. `build_seedvr2_upscale_workflow()` is the plain upscale (`resolve_prompt_stage` → `load_source_stage` → `seedvr2_stages.upscale_stage` → `save_image_stage`), chosen when the request's `upscale_method` is `seedvr2`; it does not sharpen. The tier is chosen by `utils/seedvr2.pick_seedvr2_tier` from the `seedvr2` config block and the Mac's memory budget, and `generation.upscale.method` records `seedvr2` or `creative` in the saved file.

### Review Before Upscaling

`--review` pauses a generation after the small render of an upscale until the user approves or discards it. The review state lives in `SkipSignal` (`utils/interactive.py`), the shared wait in `review.py`, and images pass through `review_stage` before `upscale_stage`. Video passes a `review` keyword to the backend: only the macOS backend (`mac/video`) honours it, the CUDA one (`cuda/video/backend`) refuses, and `supports_video_upscale_review()` in `backends/__init__.py` tells the form, server and CLI whether to offer it. The Web UI sends approve and discard through the job controls and shows a video review from `GET /jobs/{id}/review`. Everything stays in memory while the review waits, so an approved output equals one made without review.

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

Let the user choose a path on the server's machine with `PathField`, or `PathInput` (a path box with completion) and `PathBrowser` (the browse dialog) where a field needs its own layout. Each takes a browse purpose; the purposes, with their folder or file kind and accepted extensions, live in `web/path_browser.py`, so add one there for a new kind of path.

### Web UI Toasts

Report the outcome of an action with `addToast(message, type, options)` from `$lib/state/toasts.svelte`. One toast shows at a time at the bottom centre of the page content (pages with a fixed left sidebar set `--toast-inset-left`); the rest wait their turn, shown as "+N". A repeat of a waiting or showing message merges into it ("×2"), and an error goes ahead of every non-error. Each tone has a default time on screen (success 2.5 s, info 4 s, warning 7 s); errors stay until dismissed, or give way after 8 s once other toasts are waiting, and hovering or focusing a toast pauses it. Pass `{ action: { label, run } }` for one follow-up such as Retry, and `{ timeout }` only when the default doesn't fit. An action that acts on a page must not outlive it: keep the id `addToast` returns and call `dismissToast(id)` in the page's `onDestroy`. Raise one toast per action, and skip it when the screen already shows the result.

Send requests through `api` in `$lib/api/client`, which throws `ApiError` with the HTTP `status` and the server's `detail`. Show a failed request with `errorText(err, fallback)`, never `err.message`, which holds the method, path and raw body. Offer Retry only when `isRetryable(err)`: a 4xx means the server refused the input, and the same request fails the same way again.

### Web UI Design System

The Web UI has one design system. Its tokens live in `@theme` in `frontend/src/app/global.css`, and its component classes are the `ui-*` classes in the same file. Build pages from the shared components in `frontend/src/lib/components/` (`Button`, `Input`, `Select`, `Toggle`, `Badge`, `Panel`, `Segmented`, `ActionBar`, `PageHeader`, `KeyValueList`, `Alert`, `EmptyState`, `InspectorSection`, `InspectorRow`) or, where no component fits, the `ui-*` classes (for example `ui-chip`), rather than styling elements by hand.

- **Size:** controls are 28px (`h-control`), small ones 22px, rows 30px. Buttons that sit together share one height; inside an `ActionBar` every button is 36px and the one marked `main` takes the remaining width.
- **Type:** four sizes only: `text-meta` (11px), `text-ui` (12px), `text-content` (13px), `text-title` (16px). Nunito (`font-heading`) is for titles, area names and the main action; monospace is for numbers, paths and shortcuts.
- **Labels:** field and row labels are sentence case (`ui-label`). The one small-caps style (`ui-area-label`) names areas: panels, sections, table headers and menu groups.
- **Shape and surfaces:** corners are 4px for chips and badges, 6px for controls and 10px for containers. Overlays (menus, popovers, tooltips, toasts, dialogs) use `bg-overlay`, darker than any page surface, with a faint ring and never a drop shadow; `bg-raised` is for section header bands and `bg-fill` for the prompt box.
- **No borders:** containers are separated by shade, and buttons, fields, chips and badges are filled shapes. Dividers between rows and sections stay. Draw an outline only where it means something: focus, a field with an error, the selected tile, and the dashed edge of drop zones and add buttons. Segmented controls are a recessed well with the chosen option as a lighter pill.
- **Main action:** each section (a bar, panel, form, dialog or popover) has at most one main action, a `primary` button: teal fill and a bold Nunito label. Every other button in the section is secondary.
- **Reference pages:** Config and Models centre their content at `max-w-page` (1200px), with the page bar lined up to it. Settings use label-left rows (`FormField layout="row"`), and tables use `ui-table-comfortable`.
- **Colour:** neutrals are Graphite greys with no tint. Teal means on, selected or the main action; coral (`accent-coral`, Blob's beret) marks `$snippets` and amber `{a|b}` choices; red is for errors and destructive actions, which always show red text. Info messages stay teal, never coral. Text is at least 4.5:1 on every surface. Use tokens, never raw palette classes or colour literals; `make check` fails on them.
- **Focus:** every control shows the same 2px teal outline with a 2px offset.
- **Touch is a layer, not a breakpoint:** `@media (pointer: coarse)` in `global.css` raises the size tokens (controls 36px, small controls 28px, rows 40px, action bars 44px, the switch 40 × 24) at every width, so an iPad in the desktop layout gets finger-sized targets and a mouse keeps the desktop density. Build sizes on the tokens, never on literal pixels, so they follow. Numbers get − and + steppers (`InspectorNumber`, press and hold to repeat) because a label cannot be scrubbed when a drag scrolls; `InspectorRow` skips scrubbing on a coarse pointer. The `layout` store (`lib/state/layoutMode.svelte.ts`) tells components the composition (`compact | tall | wide`) and whether the pointer is coarse.
- **Three compositions of one Workspace:** the same panes (`ComposePane`, `SettingsPane`, the toolbar, the stage, `HistoryStrip`) are arranged by `layout.mode`. *Wide* (1024px and up) is the sidebar beside the stage. *Tall* (640 to 1023px, a portrait tablet) puts the stage and strip over Compose and Settings side by side. *Compact* (under 640px, or under 500px tall) makes the stage the screen: a `TopStrip` and `NavDrawer` replace `TopNav`; a `SettingsStrip` of chips (`settingsChips.ts`) opens one `SettingPopover` at a time, All opens `SettingsPane` in a `BottomSheet`, and a composer at the foot hosts `ComposePane` folded or open. Everything stays inside the Workspace form, so every field submits in every composition. A running job shows `JobPreview` large with `JobCard` beside it on wide and tall and `JobBar` over it on compact; `ActionMenu` and `Modal` render as sheets on compact. Nothing is hidden behind a phone pattern on a wide screen.

### Test Strategy

Mock heavy image and video dependencies in tests. Diffusers backend tests patch the lazy runtime loader, torch CUDA checks, upsampler weight download and loading, export helpers, and PIL image loading so the suite never instantiates a real model, downloads weights, or requires a real CUDA device. Platform dispatch, alias resolution, and Web inventory tests should assert behavior through config and protocol boundaries rather than backend internals.

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
    - for review before upscaling: `_stage1` (shadowed per call), `Stage1Result.video_tokens`/`audio_tokens`/`latent_dims`, `_load_decoders`, `dit`/`_loaded` (restored after the preview decode), and the model state released after an aborted generation
    - `DistilledPipeline.generate_and_save` for upscale, and `BasePipeline`'s `gemma_model_id` default, the transformer-file preference order and the `low_memory` and `low_ram_streaming` constructor arguments
    - Extending a clip (`LtxVideoBackend.extend_video`), video side: `conditioning.types.VideoConditionByLatentIndex(frame_indices, clean_latent, strength)` pins the carried tail at the start of the window; `retake.encode_video_tensor(encoder, video_tensor)` and `utils.media_io.video_preprocess(frames, height, width)` encode the tail frames inside `image_conditioner(fn, free_after=)`
    - Extending a clip, audio side: `ltx_core_mlx.utils.audio.load_audio(path, target_sample_rate, start_time, max_duration, mono)` reads the tail's sound at its position in the file; `audio_vae.encode_audio(waveform, sample_rate, encoder, processor)` encodes it inside `audio_conditioner(fn, free_after=)`; `extend_distilled.fit_audio_latent(audio_latent, audio_tokens)` fits the `(1, 8, T, 16)` latent before `audio_patchifier.patchify`, and `extend_distilled.carry_audio_tokens(frame_rate, carry_pixel_frames)` gives the token count
    - Extending a clip, decode: the pipeline's `_loaded` flag (cleared with `dit = None` so the next `load()` reloads the transformer), `video_decoder_block.load()` and `video_decoder_block.video_decoder` (the diffusion decoder is rejected), `video_vae.VideoDecoder.tiled_decode(latent, tiling_config)`, `video_vae._compute_decode_tiling(latent_shape, frame_rate, budget_bytes)` (which may return `None` or temporal tiles), `video_vae.decode_cache_limit()`, `video_vae.stream_chunks_to_ffmpeg(chunks, proc)` and `video_vae._ffmpeg_sink(cmd)`, and `audio_decoder_block.__call__(audio_latent)`
    - Extending a clip: `ltx_core_mlx.utils.audio.load_audio` returns an `AudioData` whose `.waveform` and `.sample_rate` are passed to `encode_audio`, and `video_patchifier.patchify` and `audio_patchifier.patchify` return `(tokens, dims)` tuples, so the tokens are `patchify(...)[0]`
- **Cache limit.** `low_ram_streaming` makes the pipeline set the MLX buffer-cache limit to 0 for the whole process. `backends/mac/memory.mlx_cache_limit` scopes that to model construction and each generation, then restores the previous limit.

## Vendored SeedVR2

`packages/seedvr2_torch` is the PyTorch SeedVR2 inference core from [`numz/ComfyUI-SeedVR2_VideoUpscaler`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (Apache-2.0), which the Windows and Linux plain upscale runs. Upstream is not a pip package, so it lives in the repository. Never edit it by hand.

- **Pin.** `SEEDVR2_COMMIT` in the `Makefile` must be the commit of a release; the comment above it names the version.
- **Layout.** `make update-seedvr2` copies upstream's `src/` without `src/interfaces/` (the ComfyUI nodes), the two model configs, the fixed text embeddings `pos_emb.pt` and `neg_emb.pt`, and the licence, and writes its own `__init__.py` in place of upstream's, which loads the nodes. Upstream's code imports itself relatively and finds its configs and embeddings three folders above `src/utils/constants.py`, so the copy keeps upstream's layout and is imported as `seedvr2_torch.src`.
- **Update.** It does not carry dependencies: mirror upstream's `requirements.txt` into ours by hand, Windows and Linux only, leaving out `gguf` (only imported for GGUF weights) and `matplotlib` (debug output). `torchvision` comes from the same CUDA index as `torch`, and `opencv-python` is replaced by `opencv-python-headless`.
- **Internals we rely on.** `tests/test_seedvr2_torch_contract.py` pins the entry points the upscaler calls, their keyword arguments and the bundled files.

## Releasing

Pushing a `v*` tag runs the Release workflow, which builds the package, publishes it to PyPI and creates a GitHub Release. The package version comes from the tag (`hatch-vcs`), so the tag must be a valid PEP 440 version:

- **Final releases:** `vX.Y.Z`, e.g. `v0.13.0`.
- **Betas:** `vX.Y.ZbN`, e.g. `v0.13.0b1`, then `b2`, `b3`, … before the final `vX.Y.Z`. Use `aN` and `rcN` the same way for alphas and release candidates.

Pre-release tags are published as GitHub pre-releases, and `pip` installs them only with `--pre` or an exact pin (`pip install z-vision-generator==0.13.0b1`). Before tagging, move the `[Unreleased]` CHANGELOG entries under a `## [<version>] - <date>` heading in a `chore(release): v<version>` commit whose body lists the release highlights. PyPI never accepts the same version twice, so a broken release needs a new tag.
