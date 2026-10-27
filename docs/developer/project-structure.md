# Project structure

```text
zvisiongenerator/
├── __init__.py
├── image_cli.py                   Image CLI entry point (ziv-image)
├── video_cli.py                   Video CLI entry point (ziv-video)
├── cli.py                         Unified CLI entry point (ziv)
├── enhance_cli.py                 Shared --enhance / --no-enhance / --enhance-model / --enhance-format CLI handling
├── preflight.py                   Preflight phase: plan seeds and prompts, rewrite them before the model loads
├── built_models.py                One entry point for the models built on first use: recognise their folders, prepare them, name the step
├── ideogram4_instant_builder.py   Build the Ideogram 4 Instant model folder on first use (download, convert, link the base components)
├── image_model_loader.py          Image model loading through stored quants (reuse, or save on first use)
├── image_review.py                Review of an image's small render before the upscale
├── image_runner.py                Image generation run orchestration
├── moss_builder.py                Build the MOSS-Image model folders on first use (link the downloaded SFT files, and the Turbo LoRA for moss)
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
│   │   │   ├── ideogram4.py       Ideogram 4 Instant: load, placement, the first-step sigma and the pipeline call
│   │   │   ├── lokr.py            LyCORIS LoKr adapters (applied like mflux)
│   │   │   ├── moss.py            MOSS-Image: load, placement, the sampling run and the stored q4/q8 copies
│   │   │   ├── moss_lora.py       Unmerged LoRA and LoKr adapters for the MOSS-Image transformer (kohya, ai-toolkit, diffusers/PEFT)
│   │   │   ├── moss_sampler.py    MOSS-Image denoising loop, prompt encoding and latent helpers (mirrors the vendored pipeline)
│   │   │   └── preview.py         Cheap latent → RGB live previews
│   │   └── video/
│   │       ├── audio_carry.py     Pin a clip's last sound at the start of an extension window
│   │       ├── backend.py         LTX video backend
│   │       ├── converted.py       Write a converted LTX-2.3 or LTX-2.5 folder from a single-file checkpoint
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
│       ├── h3_converted.py        Write a converted MiniMax H3 folder (8-bit MLX transformer) from a fine-tune
│       ├── h3_staged.py           MiniMax H3 image modes (MLX): keyframe, vision, prompt, denoise and decode phases that each load one component and drop it before the next
│       ├── h3_video.py            MiniMax H3 video backend (MLX): file resolution, the 4-bit text-encoder copy, LoRA combining and Extend around the vendored port
│       ├── image.py               Image backend (mflux)
│       ├── image_preview.py       Cheap latent → RGB live previews for the mflux backend
│       ├── ltx_converted.py       Write a converted LTX-2.3 MLX folder from a single-file checkpoint
│       ├── memory.py              MLX memory release, recommended GPU memory budget, machine memory size and scoped cache limit
│       ├── prompt_enhancer.py     Prompt-enhancer LLM (mlx-lm)
│       ├── seedvr2.py             SeedVR2 upscaler (mflux)
│       └── video.py               LTX video backend (MLX)
├── converters/
│   ├── convert_checkpoint.py      Safetensors checkpoint → diffusers converter (ziv-model model)
│   ├── convert_target.py          Where a video checkpoint conversion writes, and how it reports progress
│   ├── h3_checkpoint.py           Read a MiniMax H3 fine-tune for conversion, one tensor at a time, under diffusers' names
│   ├── h3_convert.py              Convert a MiniMax H3 fine-tune into a models-folder entry of its own (`--model-type h3`)
│   ├── ideogram4_instant.py       Ideogram 4 Instant's single-file transformer → diffusers BF16 or mflux FP8 layout, one tensor at a time
│   ├── list_assets.py             List installed models, video models, and LoRAs (ziv-model list)
│   ├── ltx_checkpoint.py          Read a single-file LTX-2.3 or LTX-2.5 checkpoint, one tensor at a time
│   ├── ltx_convert.py             Convert an LTX-2.3 or LTX-2.5 checkpoint into a models-folder entry at one quantize level
│   └── lora_import.py             LoRA import — local copy and HF download (ziv-model lora)
├── core/
│   ├── types.py                   Shared types (StageOutcome)
│   ├── job_plan.py                Job plan value types (JobPlan, IterationPlan, EnhanceStatus)
│   ├── ideogram4.py               Ideogram 4 Instant settings both image backends share: first-step sigma, JSON captions, sampler
│   ├── moss.py                    MOSS-Image settings every image backend shares: sampler, img2img step count, text length
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
│   ├── hub_access.py              Where the Hugging Face token comes from, and one actionable line for a Hub refusal
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
│   ├── beta_scheduler.py          Beta-distribution sigma scheduler
│   └── beta_sigmas.py             Beta-distribution sigma grid shared by the mflux and diffusers backends
├── utils/
│   ├── alignment.py               Pixel-alignment helpers for resolution, and the size and frame alignment of a video job
│   ├── app_db.py                  The app database `ziv.sqlite3`: path, connections with the standard pragmas, per-component schema versions, corruption quarantine
│   ├── app_log.py                 Error log file, cleared on every start
│   ├── atomic_write.py            Atomic file replacement for config and prompt-file writers
│   ├── audio_resample.py          Polyphase resampling of multi-channel audio
│   ├── config.py                  Config loading (image + video)
│   ├── console.py                 Console formatting
│   ├── enhance_record.py          Record how a prompt was enhanced for a generated asset, and read it back
│   ├── ffmpeg.py                  ffmpeg availability check and install, the error for a failed ffmpeg or ffprobe run, and the hidden temp path of in-place rewrites
│   ├── file_links.py              Hard-link model files into a new folder, copying where linking fails
│   ├── filename.py                Output filenames ({set}_{timestamp}) and collision counters
│   ├── h3_lora.py                 Read MiniMax H3 LoRA files of any format into diffusers' converted form
│   ├── h3_memory.py               MiniMax H3 row arithmetic, memory estimates (CUDA GPU and system memory, macOS unified memory) and estimated generation time
│   ├── h3_preset.py               Value types of the MiniMax H3 preset and its resolved files, parsed from the config entry
│   ├── h3_converted.py            The manifest of a converted MiniMax H3 model folder
│   ├── hf_cache.py                Read the Hugging Face cache: repos, what deleting each frees, unused shared files
│   ├── hf_snapshot.py             Find or download the files of one pinned Hugging Face revision (shared by the CUDA and macOS H3 backends)
│   ├── ideogram4_instant.py       Ideogram 4 Instant's pinned build sources, model folder and build record
│   ├── image_model_detect.py      Image model type detection
│   ├── interactive.py             Keyboard interrupt handling and review state
│   ├── lora.py                    LoRA CLI argument parsing
│   ├── lora_family.py             Pure detector of the model family a LoRA was trained for, from its safetensors header and metadata (`LORA_FAMILY_VERSION`)
│   ├── ltx_converted.py           The manifest of a converted LTX model folder (LTX-2.3 or LTX-2.5)
│   ├── model_files.py             Offline check for fully downloaded model weights
│   ├── model_memory.py            Header-only memory estimates (MLX unified memory, with a per-stage LTX clip estimate by size and length; CUDA GPU and system memory)
│   ├── moss_build.py              MOSS-Image's pinned sources, model folders, build records and shared stored copies
│   ├── page_cache.py              Drop model files from the page cache once written or read
│   ├── paths.py                   ~/.ziv/ model store resolution
│   ├── platform.py                Platform metadata and platform-aware alias values
│   ├── prompt_compose.py          Structured prompt flattening & snippets; video scene helpers (compose scene + prompt, quoted-line stripping, original-prompt check)
│   ├── prompt_document.py         Prompt file ↔ editable document for the Web UI builder (comment-preserving, ruamel.yaml)
│   ├── prompt_document_check.py   Builder document previews and problems, with the prompt loader's rules
│   ├── prompt_enhance.py          Prompt-enhancement settings, length plans, chat messages, run, and post-processing (no model I/O)
│   ├── prompt_enhance_video.py    Video prompts as a scene and an action: word plans, chat messages, output parsing and the run (no model I/O)
│   ├── prompt_enhance_caption.py  Ideogram 4 JSON captions: the model writes labelled lines, code builds the caption (no model I/O)
│   ├── prompt_quotes.py           Find quoted text in a prompt and check that a rewrite keeps it exactly
│   ├── quantized_weights.py       Decode community checkpoints' quantized weights (scaled FP8, INT8, ConvRot, W4A4, W4A8, W6A8, NVFP4, MXFP8) one weight at a time
│   ├── prompts.py                 Prompt file loading
│   ├── provenance.py              Settings embedded in generated PNG/MP4 files: build, embed, read
│   ├── safetensors_stream.py      Read and write safetensors one tensor at a time, keeping them out of the page cache; reads a header and its metadata in one bounded read
│   ├── seedvr2.py                 SeedVR2 weight layouts, settings, memory estimate and tier choice
│   ├── stored_quant.py            Stored-quant naming, manifests and folder handling, and the write-once stored-copy routine
│   ├── subprocess_env.py          Environment for Python child processes whose output is piped or discarded
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
│   ├── instances.py               Record running Web UI servers for --background and --background-stop
│   ├── job_contract.py            Web job lifecycle, terminal event, and control contract
│   ├── job_log.py                 Job log component of the app database: entry builder, filters, paging and the `jobs_entries` store
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
│   ├── video_conversion.py        Video checkpoint conversion from the Models page (LTX-2.3, LTX-2.5, MiniMax H3): inspect, then convert as a background job with progress
│   ├── web_runner.py              Background workers for generation batches
│   ├── workspace_api.py           Shared SPA payloads for workspace and models routes
│   ├── workspace_contract.py      Workflow aliases and static workspace capabilities
│   └── server.py                  FastAPI route wiring and request parsing
└── workflows/
    ├── enhance_stage.py           Applies the planned rewrite in image and video workflows
    ├── image_stages.py            Image pipeline stage definitions
    ├── seedvr2_stages.py          SeedVR2 upscale stage
    └── video_stages.py            Video pipeline stage definitions, including extend_video and join_clips
```

## Outside the package

| Path | Holds |
|---|---|
| `frontend/` | The Web UI's source: Svelte 5, TypeScript, pnpm. Built into `zvisiongenerator/web/static/app/`, which is committed |
| `tests/` | The pytest suite, grouped by area: `web/`, `video/`, `image/`, `prompts/`, `downloads/`, `converters/`, `contracts/`, `backends/` (with `cuda/` and `mac/`) and `common/`. `conftest.py` holds the autouse guards and fixtures, `_factories.py` and the other `_*.py` files the helpers shared between areas, and `fixtures/` the JSON the frontend tests also read. See [Conventions](conventions.md#tests) |
| `bench/enhance/` | The [prompt-enhancer bench](enhancer-bench.md); not packaged |
| `packages/` | [Vendored](vendored.md) LTX MLX, SeedVR2, MiniMax H3 MLX and MOSS-Image code |
| `docs/` | These docs, built with Zensical from `zensical.toml` |
| `proposals/` | Write-ups of future work. See [Proposals](index.md#proposals) |
| `prompts.yaml` | Default prompt definitions |
