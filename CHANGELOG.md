# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.13.0b15] - 2026-10-09

### Added
- Web UI: Config marks each setting you have set yourself with a dot and a reset button that returns it to the default, like the Workspace settings; the help under each setting says which default applies

### Changed
- Web UI: a calmer look with fewer lines: panels, cards and controls no longer have outlines and are told apart by shade, with dividers kept between rows; buttons, fields, chips and badges are filled, and segmented controls show the chosen option as a lighter pill in a darker track
- Web UI: neutral grey surfaces replace the teal-tinted ones, so images carry the colour; teal stays the main colour, `$snippets` are coral like Blob's beret, and placeholders and grey badges are easier to read
- Web UI: dialogs, menus, popovers, tooltips and notifications sit on a dark surface with a faint ring instead of a drop shadow, so they stand out from the page; destructive buttons are red, and every control shows the same focus ring
- Web UI: each section's main action (Generate, Save, a form's submit button, a dialog's confirm button) has a teal fill and a bold label, so it stands apart from the other buttons
- Web UI: on wide screens, Config and Models keep their content in a centred column instead of stretching across the window; Config lists its settings as rows with the reset button beside each value, and Models shows models and LoRAs side by side, with the three add forms side by side below them
- Web UI: Workspace settings sections have a header band with a summary of the hidden settings, and switches, badges and path fields share one compact size
- Web UI: Config and Models share the page bar and panels; Config's **Save** and **Discard changes** sit in the page bar, both pages list folders and Hugging Face access the same way, and Models no longer cuts model names short
- Web UI: the Gallery uses the shared page bar with its filter, sort and **Delete selected** in it, and its filter and sort controls look like every other dropdown in the app; the asset viewer's buttons, details panel and shortcuts match the rest of the app
- Web UI: page titles match their tab names: Config, Models and Gallery
- Web UI: the top bar uses underline tabs; in the Workspace, ratio and resolution presets are joined button groups, Reset and Generate share one height, LoRA chips show their weight in a neutral chip, and the queue's **Clear queue** is red
- Web UI: the Prompts page uses the shared buttons, labels and page bar; all prompt text has one size, and an empty page says what to do next

### Fixed
- Web UI: the Workspace's **Add LoRA** menu closes on Escape, an outside click or a choice, and works with the arrow keys; before, it stayed open until you picked a LoRA, even under other menus
- Web UI: reusing an image's settings shows its LoRAs by name in the Workspace instead of by file path, so they match LoRAs added from the menu; negative LoRA weights are no longer cut off
- Web UI: the Models page shows the dividers under card titles and table headers, and highlights the table row under the pointer; before, both were the same colour as the card

## [0.13.0b14] - 2026-10-08

### Added
- Web UI: a **Prompts** page that builds prompt files visually. Define snippets, then build sets of prompt entries with `$snippet` chips, editable `{a|b}` choices (also while typing, and from selected words with `⌥↵`), text or named fields, an active switch, a negative prompt and full `enhance:` options. Drag sets, entries and snippets to arrange them. A preview shows each prompt as the model receives it, with a **Roll** of its choices. **Generate this one** queues a single prompt with the Workspace settings
- Web UI: saving a prompt file keeps its comments and layout, keeps the Workspace's selected prompts pointing at the same entries after reordering or renaming, and asks before overwriting a file that changed on disk. `⌘Z` undoes, and unsaved changes survive leaving the page
- Web UI: **New file…** creates an empty prompt file; a file that doesn't load opens in a repair view

### Changed
- Web UI: the job card shows only the job and its live preview; finished outputs appear in the History strip instead of a thumbnail grid on the card
- Web UI: the job card's progress bar fills in the step being processed, at the stage's average step pace, so it creeps forward instead of jumping
- Web UI: the job card's elapsed time counts up every second instead of only when a step finishes
- Web UI: **Edit YAML** in the prompt-file box is now **Edit** and opens the file on the Prompts page; the YAML text dialog is removed

### Fixed
- macOS: generation progress reports each step once it has finished, and live previews no longer hold a step back; before, the bar sat still at a preview and then jumped two steps

## [0.13.0b13] - 2026-10-07

### Added
- Windows/Linux: reference images (img2img) and upscale, including the viewer's **Upscale**, now work with Krea 2

### Changed
- Krea 2: the `krea2` alias uses `unsloth/Krea-2-Turbo`, an ungated mirror, so no Hugging Face token or license acceptance is needed
- Krea 2: only Krea 2 Turbo is supported. The Raw defaults and `ziv-model model --model-type krea2-raw` are removed
- Krea 2: negative prompts are turned off for it
- macOS: Krea 2 samples with Euler at the fixed timestep shift Krea trained it with, instead of mflux's er_sde sampler and size-dependent shift; upscales and reference images are no longer grainy
- Krea 2 upscales refine in 3 steps instead of 4, which ranked better in side-by-side tests and is about 25% faster
- Config: the per-backend `backends.<mflux|diffusers>` capability overrides added in 0.13.0b12 are removed; capability flags are per model family again

### Fixed
- Web UI: the asset viewer's Details panel lists every LoRA on its own row (`name · weight`, full path on hover); before, a second LoRA was cut off

## [0.13.0b12] - 2026-10-06

### Added
- **Krea 2 Turbo** image model (`krea2` alias, `krea/Krea-2-Turbo`) on macOS (mflux) and Windows/Linux (diffusers): 8 steps, guidance 1.0 (off), negative prompts, LoRAs, q4/q8 and live previews, with stored quants on macOS. On Windows and Linux, `-q 4` lets the text encoder and transformer take turns on the GPU so it runs on 10–12 GB cards. The model is gated on Hugging Face and needs its license accepted and a token
- Krea 2 Raw models (`krea/Krea-2-Raw` and Raw fine-tunes) default to 28 steps and guidance 5.5
- `ziv-model model --model-type krea2-turbo|krea2-raw` (and the Models page converter) converts native and ComfyUI Krea 2 checkpoints
- Config: a model preset's `backends.<mflux|diffusers>` entry overrides its capability flags for one backend. Krea 2 uses it to turn off reference images and upscale on Windows and Linux, where diffusers has no Krea 2 image-to-image pipeline; `ziv-image` rejects them there and the Web UI hides them
- Config: `supports_scheduler: false` hides the Scheduler control and rejects `--scheduler` for models that sample with their own schedule (Krea 2); a scheduler carried over from another model is dropped

### Changed
- macOS: memory estimates leave the text encoder unquantized for models whose loader keeps it in bfloat16 (Krea 2), so their q4/q8 estimates are no longer about 5–7 GB too low
- diffusers is updated to 0.40.0, the first release with Krea 2 that does not build the full attention matrix for its text padding mask

## [0.13.0b11] - 2026-10-05

### Added
- Web UI: **job queue**. While a job runs, Generate becomes **Add to queue**: change the settings and add more runs, which start one after another. The queue shows under the job card (**Up next**) with **Load settings**, remove and **Clear queue**, is shared by every tab, and keeps going when a job fails. New `GET /api/jobs` and `DELETE /api/jobs/queue` endpoints
- Web UI: more workspace shortcuts. `Alt+3` focuses the newest history tile, expanding the strip if needed. `P` pauses or resumes the running job, `N` skips to the next image and `R` repeats the current one
- Web UI: the Compose and Settings sidebar can be collapsed into a narrow strip with expand and Generate buttons; the choice is remembered in the browser
- Web UI: **Upscale → 2× / 4×** in the asset viewer (or **X** then **2** / **4**) and in the **⋯** menu on thumbnails upscales an existing image, refined with the settings recorded in it. Each option shows the output size and is disabled, with the reason, when the model can't upscale or the output is too large. The result is saved next to the original as `<name>_2x.png` / `<name>_4x.png`; jobs started from the Gallery open the Workspace
- Config: viewer upscales refine with `upscale.existing_denoise_small` (0.4) up to `upscale.existing_large_megapixels` (2) of output and `upscale.existing_denoise_large` (0.2) above it, so an upscale of an upscale refines as lightly as a direct 4×; `upscale.max_megapixels` (20) caps their output size. `sharpening.existing_upscaled` (1.2) and `sharpening.existing_pre_upscale` (0, off) set their final and pre-refinement sharpening
- Saved files also record the negative prompt, scheduler, model family, quantization, reference strength, generation time, upscale settings and the sharpen, contrast and saturation amounts
- Web UI: the asset viewer's **Details** panel shows everything recorded in the file, grouped into Generation, Post-processing and File, plus the original image of an upscale. **Reuse settings** also restores the negative prompt and scheduler

### Changed
- Web UI: the form stays editable while a job runs; the manual Enhance prompt button waits until all jobs have finished
- Web UI: the running job's card puts the live preview and outputs in a column to the right of the progress and controls
- Web UI: deleting assets, models and LoRAs asks in the app's own confirmation dialog instead of the browser's. `Enter` confirms and `Esc` cancels
- Output files are named `<set name>_<YYYY-MM-DD_HH-MM-SS>.png` (or the video's extension). Settings are no longer packed into the name, since they are embedded in the file; a taken name gets `_2`, `_3`… instead of being overwritten. Scripts that parse the old names need updating; the gallery still reads older names
- Step counts for img2img refinement and upscale passes show the steps that actually run (e.g. 3 / 3 instead of 3 / 7)
- Web UI: the **Sharpen** amount defaults to **auto**, which uses the config like `ziv-image` does (`sharpening.normal`, 1.0, for plain images and `sharpening.upscaled`, 1.2, for upscaled ones). Before, the Web UI always sent 0.8. A saved Workspace still on the old 0.8 switches to auto; type an amount to override

### Fixed
- Sharpen amounts above 1.5 are rejected (CLI, Web UI and config); higher values broke the sharpening filter. The Workspace control now stops at 1.5

## [0.13.0b10] - 2026-10-04

### Added
- macOS: **stored quants**. The first job that runs a model at q8 or q4 saves the quantized weights in the models folder (`<name>@q8`, `<name>@q4`, named after the installed model or alias, e.g. `zit@q8`); later jobs load that copy instead of quantizing again. LoRAs apply on top. Copies are listed on the Models page and are deleted with their model
- `ziv-model model --quantize 4|8` and **Quantized Copy** on the Models page save a quantized copy while converting a checkpoint

### Changed
- macOS: image models stay in memory for the whole job instead of re-reading (and re-quantizing) their weights for every image. On a 32 GB Mac, FLUX.2 Klein 9B q8 batches went from 60–80 s to about 28 s per image, with steady speed and no swap growth through the batch
- macOS: MLX's free-buffer cache is capped at 4 GB and cleared after each image, which kept up to 29 GB of unused buffers resident before
- `ziv-image` rejects `-q` for models that cannot be quantized (Ideogram 4), as the Web UI does

## [0.13.0b9] - 2026-10-04

### Added
- Prompt enhancer: **Mood** option (Serene, Joyful, Romantic, Melancholic, Mysterious, Eerie, Dramatic, Epic, Whimsical, Nostalgic) in the Web UI, `--enhance mood=…` and prompt-file `enhance:` entries. Defaults to Keep

## [0.13.0b8] - 2026-10-04

### Added
- Web UI: prompts that could not be enhanced, or were skipped, are marked in the progress panel
- **Next**, **Pause** and **Quit** work while prompts are enhanced (Web UI image jobs, `ziv-image` keys); `ziv-image` keys also work while the model loads

### Changed
- Jobs with auto enhancement rewrite all prompts first, so the first image starts after every rewrite. The progress panel shows *Enhancing prompt N of M*
- **Repeat** and automatic retries keep the image's `{a|b}` choices and enhanced prompt and only change the seed, in every job. The new seed is always random, even when a seed is set
- A prompt-enhancer error (e.g. the model cannot load) stops the job before the generation model loads

### Fixed
- Auto prompt enhancement: every prompt in a batch gets its rewrite, even when the image or video model is a tight memory fit. Prompts are rewritten before the model loads, and the enhancer is unloaded before generation starts

## [0.13.0b7] - 2026-10-04

### Added
- Web UI: keyboard shortcuts across the app; press `?` for the full list
  - Viewer: `Delete`/`Backspace` deletes (`Shift+Delete` without asking), `R` reuses settings, `E` uses as reference, `D` downloads, `C` copies the prompt, `Home`/`End` jump to the first/last asset
  - Gallery: arrow keys move between assets, `Space`/`X` selects, `⌘A`/`Ctrl+A` selects all, `Delete` deletes, `Esc` clears the selection
  - Workspace: `⌘⇧↵`/`Ctrl+Shift+↵` generates with a new seed, `⌘E`/`Ctrl+E` enhances the prompt, `Alt+1`/`Alt+2` focus the prompt/settings, `Esc` stops the running job
  - Anywhere: `G` then `W`/`G`/`M`/`C` opens Workspace, Gallery, Models or Config

## [0.13.0b6] - 2026-10-04

### Added
- Web UI: live previews at 25%, 50% and 75% of the steps now also work on Windows and Linux (diffusers/CUDA) for Z-Image, FLUX.2 Klein and FLUX.2

### Fixed
- Prompt enhancer (macOS): prompts after the first in a batch are enhanced again when the image or video model is a tight memory fit; the enhancer no longer runs out of memory after a generation

## [0.13.0b5] - 2026-10-03

### Added
- Prompt enhancer: new styles Candid, Street photography, Analog film, Black & white, Studio portrait, Product shot and Comic (`candid`, `street`, `film`, `bw`, `portrait`, `product`, `comic`)

## [0.13.0b4] - 2026-10-03

### Added
- Web UI: one asset tile and viewer everywhere (History, preview, run results, Gallery) with **Preview**, **Reuse settings**, **Use as reference**, **Download** and **Delete**
- Web UI: **Use as reference** sends an image to Image to Image or Image to Video
- Web UI: the viewer's **Details** panel (`I`) shows prompt, model, size, seed, steps, guidance and LoRAs
- Web UI: settings changed from the model default are marked and can be reset one by one
- Web UI: drag a setting's label to change its value; ↑/↓ nudge it, Shift ×10
- Web UI: seed lock
- Web UI: prompt-file chooser with filter and per-set All/None
- Web UI: All/None toggle for the enhancer's Details and Motion options
- Gallery API: assets include `file_path`, `seed`, `steps`, `guidance` and `lora`; the workspace context includes `video_size_dimensions`

### Changed
- Web UI: redesigned workspace with Compose and Settings on the left and History as a filmstrip under the preview
- Web UI: the prompt box has **Prompt** and **Enhanced** tabs; Enhance options open in a popover
- Web UI: Gallery uses a full-width grid; asset details moved into the viewer
- Web UI: settings are kept between visits and restarts
- Web UI: batch size up to 100
- Web UI: small images are no longer enlarged in the preview
- Web UI ✨ Enhance picks the prompt's `{a|b}` choices before rewriting, so the enhancer works on one plain prompt instead of placeholders. The Enhanced prompt no longer contains choices; use **Enhance each image when generating** for a fresh pick per image

### Fixed
- The prompt enhancer keeps every detail the prompt describes instead of dropping some of them; Shorter cuts only filler

## [0.13.0b3] - 2026-10-04

### Added
- The Web UI Models page can delete models and LoRAs. Deleting a converted model removes its folder in `~/.ziv/models/` and keeps the HuggingFace base-model files it links to. Deleting a HuggingFace model removes its download from the HuggingFace cache; the alias stays and downloads again on next use, and the confirmation lists converted models that rely on that download. Deletes are refused while a generation is running
- Prompt enhancement with a small local LLM (decensored Qwen3.5-4B by default; ≈2.4 GB download on macOS, ≈9 GB on Windows/Linux). Choose a style, details, a length (Shorter, Same, Longer ≈200 %, Extra long ≈300 %) and, for video, motion
- Web UI: ✨ Enhance writes a rewrite to a new **Enhanced prompt** box, which is generated instead of the prompt while it has text; it is marked out of date when the prompt changes. **Enhance each image when generating** rewrites every image's prompt at generation time, for inline prompts and prompt files
- `--enhance [SPEC]`, `--no-enhance` and `--enhance-model` for `ziv-image` and `ziv-video`, and a per-entry `enhance:` key in prompt files
- Config page: **Prompt Enhancer Model** to use a different Hugging Face model (`owner/name[@revision]`) or a local folder

### Changed
- The macOS memory badge uses wider limits: **Fits** up to 1.1 times Apple's recommended GPU memory and **Tight** up to 1.5 times, MLX's default memory limit. Models that run with some swapping, such as FLUX.2 Klein 9B unquantized on a 32 GB Mac, now show **Tight** instead of **Too large**. FLUX.2 Klein estimates no longer count the text encoder's LM head, which is never loaded
- Auto-enhanced images and videos store the enhanced prompt in their metadata, so Gallery shows and reuses the prompt that rendered
- `ziv-video` reads the prompt file before loading the model, so prompt-file errors are reported immediately
- Minimum versions raised to `mlx-lm>=0.31.2` and `transformers>=5.8.0` (first versions with Qwen3.5 support)

### Fixed
- Converted models showed as not downloaded and had no memory badge in the Web UI, because the base-model folders they link to were not followed when looking for weight files
- Converting a scaled FP8 checkpoint, such as an FP8 FLUX.2 Klein checkpoint saved by ComfyUI, failed with `chunk expects at least a 1-dimensional tensor`. The converter now multiplies FP8 weights by their stored scale and saves them as bfloat16. Before, a plain cast would have left them off by that scale. Block-wise and non-FP8 quantized checkpoints are rejected with a clear error

## [0.13.0b2] - 2026-10-02

### Added
- A reset button next to Generate in the Web UI workspace returns all settings to the selected model's defaults, keeping the model, LoRAs, quantize level and prompt
- Every command writes warnings and errors, with full tracebacks, to `~/.ziv/logs/<command>.log` (e.g. `ui.log`). The file is cleared when the command starts, so it always describes the latest run. If that folder is not writable, the log goes to the system temp folder instead. Set `ZIV_LOG=off` to disable it

### Changed
- mflux is now required at 0.20.x (`>=0.20.0,<0.21`), so installs use the mflux version the release was tested with

### Fixed
- Image generation failed immediately with mflux 0.20 (`unexpected keyword argument 'control_images'`), which a fresh install picks up: the progress and live-preview callbacks now accept arguments added by newer mflux versions

## [0.13.0b1] - 2026-10-02

### Added
- The workspace model picker and the Models page show whether each model is downloaded and, on macOS, whether it fits this Mac's memory (**Fits**, **Tight**, **Too large**) at the selected quantize level
- On macOS, the Web UI job card shows a live preview of the image at 25%, 50% and 75% of the denoising steps, so you can skip a generation early with Next. Previews come from a fast latent-to-RGB approximation (no VAE decode), live only in memory and are never saved

### Fixed
- The Web UI now returns a finished job's model memory to macOS, instead of holding it in MLX's cache until the next job, which could freeze machines with 16 GB of memory

## [0.12.0] - 2026-10-01

### Security
- The Web UI server now rejects requests addressed to unknown hostnames (DNS rebinding) and state-changing requests from other sites, so a page open in the same browser can no longer change settings, queue jobs or read files through the UI. IP addresses, `localhost`, the `--host` name and, for LAN binds, the machine's name are allowed; add others with `ZIV_UI_ALLOWED_HOSTS`
- `/media` now serves only gallery images and videos
- Fixed a command-injection issue in the macOS folder picker when the starting folder had a crafted name

### Added
- `ZIV_UI_ALLOWED_HOSTS` environment variable (comma-separated) for reaching the Web UI through extra hostnames

### Changed
- Gallery and page loads are much faster: model detection uses the local Hugging Face cache first, video models are no longer probed as image models, and thumbnails no longer rebuild the model list
- Native pickers, model conversion and LoRA imports no longer freeze the Web UI or its progress updates while they run, and only one picker can be open at a time
- On Linux and Windows the file picker now runs in a separate process
- Duplicate prompt set names now log a warning (the last one is used, as before)

### Fixed
- Stop and Pause clicked while the model is loading now take effect instead of being silently dropped
- The Workspace no longer stays stuck on "Generating…" when a job's progress stream ends; it recovers the job's real outcome, retries brief connection problems, and shows an "unknown" outcome only if the server restarted
- Finished jobs keep their final status after cleanup, so a page that reconnects later (e.g. after sleep) still sees the real result
- Restarting the Web UI quickly now keeps the same port instead of silently moving to the next one; `--host ::` works
- The Web UI now rejects remote or missing LoRAs like the CLI does, and uses the preset's default upscale steps like the CLI
- Prompt set names such as `2025:`, `true:` or `1.10:` are kept exactly as written instead of crashing the batch
- Settings saves are atomic and safe when made at the same time as other requests, and keep a symlinked `config.yaml` and its permissions
- Image models stored under a path containing "ltx" no longer disappear from the model list

## [0.11.0] - 2026-09-30

### Changed
- Refreshed the Web UI to match Blob's style: teal-tinted surfaces, rounder corners, pill-shaped tabs, chips, and toggles, and round icon badges on settings panels
- Headings, labels, tabs, and buttons now use the rounded Nunito typeface from Google Fonts, loaded without blocking render and falling back to rounded system fonts when unavailable
- Primary actions share one flat teal style with dark-teal text and a subtle lift on hover, including buttons on the Config and Models pages

## [0.10.0] - 2026-09-30

### Added
- Added Blob, an animated Workspace mascot that reacts to generation: thinking while queued or loading, painting while creating, cheering on completion, sad on failure, surprised when stopped, and sleeping while paused or idle
- Blob waves on arrival, watches the prompt while you type, and hops smoothly between spots in the preview as the Workspace view changes
- Latest-output previews now show a loading state until the image has painted instead of an empty stage

### Changed
- Replaced the app icon, favicon, and documentation logo with Blob-based artwork

## [0.9.0] - 2026-09-09

### Added
- Added multi-select prompt-file batches with select-all controls and preservation of valid selections after YAML edits
- Added live run and prompt counters with a compact visual sequence indicator during generation

### Changed
- Expanded prompt-file selection with concise two-line previews and optional full prompt and negative-prompt details
- Generation output previews now open in the shared Workspace lightbox with navigation and focus restoration
- LoRA weights in the Workspace now accept values from -10 through 10

## [0.8.1] - 2026-09-09

### Changed
- Refactored host-local path selection into one compact, accessible selector shared by model conversion, LoRA import, prompt files, and output configuration
- Reorganized the Models page for clearer responsive operation cards and reliable bottom spacing
- Added progressive batch output previews in both the Workspace preview area and its History pane

### Fixed
- Model conversion and local LoRA import now submit the selected host path reliably and preserve retry state after failures
- Configured output directories are now used consistently for generation, previews, Workspace history, and Gallery inventory
- Gallery filtering, pagination, selection, lightbox navigation, concurrent loading, and partial deletion remain synchronized under overlapping operations
- Required model and path controls now use native validation with accessible feedback
- Generation output grids remain scrollable and reachable for large batches and constrained window sizes

## [0.8.0] - 2026-09-08

### Changed
- Refined the Web UI with compact navigation, neutral surfaces, improved text contrast, and the original teal brand accent
- Tightened workspace controls with paired fields and collapsible post-processing, upscale, and video settings
- Expanded preview space with slimmer side panels and responsive workspace controls
- Redesigned generation cards with compact timing, readable stage labels, visual batch tracking, and a subtle progress sweep that respects reduced-motion preferences

### Fixed
- Generation controls now show pending, accepted, and failed requests and prevent duplicate clicks while a request is in flight
- Removed duplicate running-stage messages and conflicting pause/resume controls
- Kept the Generate bar clear of scrolling form controls

## [0.7.0] - 2026-09-08

### Added
- Packaged local Web UI with workspace, gallery, config, and model views
- FastAPI-backed local job APIs, status streaming, and background web runner support for browser-submitted image and video jobs
- Frontend build, test, and packaged static asset verification integrated into the standard Makefile and CI workflows
- Ideogram 4 image generation with structured JSON captions, configurable first-step sigma, and Web UI capability controls
- Windows and Linux video generation through the diffusers/CUDA LTX backend
- Generation provenance metadata for saved image and video outputs

### Changed
- Base installs now include the local Web UI runtime dependencies and packaged SPA assets
- Web configuration and model resolution now expose backend-aware defaults to the local UI without duplicating platform selection logic
- Model aliases may now resolve to different repositories depending on your operating system
- If a model alias is unavailable on your platform, a guidance message now explains why and suggests alternatives
- `ziv model list` now shows platform-specific availability for model aliases
- Bare `ziv` now shows command discovery/help, and the local Web UI is launched explicitly with `ziv ui` or `ziv-ui`
- Frontend dependency management now uses pnpm
- Dependency automation now groups major updates while ignoring minor and patch updates

### Fixed
- Local Web UI job recovery, status streaming, gallery updates, and runtime delivery are more resilient
- Image backend weight casting now handles both `bfloat16` and `float16` tensors correctly
- Model management frontend assets are included in packaged builds

## [0.6.5] - 2026-04

### Added
- New logo asset set for project and documentation branding

### Changed
- README branding updated to use the new logo treatment
- Documentation home page and MkDocs site branding updated to use the new logo assets

### Removed
- Older root-level logo assets replaced by the new documentation asset set

## [0.6.4] - 2026-04

### Added
- Bundled `ltx_core_mlx` and `ltx_pipelines_mlx` directly into the repository so the package installs correctly from PyPI without relying on an external distribution of `ltx-2-mlx`
- Dynamic versioning via `hatch-vcs`: package version is now derived automatically from git tags

### Changed
- `ziv` is now a unified parent command with subcommands: `ziv image`, `ziv video`, `ziv model`
- `ziv-image` is now the primary standalone image generation command (previously `ziv`)
- `ziv-convert` renamed to `ziv-model`

### Removed
- `ziv` no longer acts as the image generation command directly — use `ziv-image` or `ziv image`
- `ziv-convert` entry point removed — use `ziv-model` or `ziv model`

## [0.6.0] - 2026-04

### Added

- Video upscale: distilled-only two-stage 2× spatial pipeline
- Documentation site (MkDocs + GitHub Pages)
- CI/CD workflows: lint/test, automated PyPI release, docs deployment
- Community files: license, code of conduct, contributing guide, security policy
- Issue and PR templates

## [0.5.0] - 2026-04

### Added

- Image-to-video generation (`--image` flag in `ziv-video`)
- Video LoRA support (single, stacked, configurable weights)
- Audio included by default in video generation (`--no-audio` to strip)

## [0.4.0] - 2026-04

### Added

- `ziv-video` entry point for video generation
- Text-to-video with LTX-2.3 on macOS (MLX backend)
- Video size presets (s/m/l/xl × 3 aspect ratios)
- Video model auto-detection
- Low-memory mode for video generation
- Batch video generation from prompt files

## [0.3.0] - 2026-04

### Added

- Reference image steering (`--image` and `--image-strength` flags)
- Image-to-image generation on both macOS and Windows platforms

## [0.2.0] - 2026-04

### Added

- `ziv-convert` entry point for model and LoRA management
- Checkpoint conversion (Z-Image, FLUX.2 Klein 4B/9B formats)
- LoRA import from local files and HuggingFace Hub
- `ziv-convert list` to display installed models and LoRAs

## [0.1.0] - 2026-04

### Added

- Text-to-image generation (macOS via mflux/MLX, Windows via diffusers/CUDA)
- Model store at `~/.ziv/` with bare-name resolution and HuggingFace fallback
- Built-in model aliases with custom alias support
- YAML prompt files with multiple prompt sets and `active` filtering
- Prompt variable syntax (`{option1|option2}`) with nesting
- Structured prompts (dict/list flattening) and reusable snippets
- LoRA support (single and stacked, configurable weights)
- Upscale pipeline (generate small → Lanczos → img2img refine → CAS sharpen)
- 4-bit and 8-bit quantization on both platforms
- Post-processing: contrast, saturation, CAS sharpening
- Interactive keyboard controls during batch runs (skip, quit, pause, repeat)
- `ziv` and `ziv-image` entry points
