# Proposals

Planned and possible future work, written up before it is built. Each proposal is one Markdown file describing the problem, the evidence, the proposed change and the alternatives considered, so work can be picked up later without re-deriving the context.

These files are not part of the published docs site, and the docs don't link to them.

## Conventions

- One file per proposal, named with a short kebab-case slug (e.g. `image-editing.md`).
- Start each file with a **Status** line: `Proposed`, `Accepted`, `In progress`, `Done` or `Rejected`, plus the date it last changed.
- Open proposals (`Proposed`, `Accepted`, `In progress`) live in this folder. When a proposal ships or is rejected, set its status, link the release or commit, and `git mv` it to [`archive/`](archive/); the file stays as a design record of what was measured and decided. Fix relative links in both directions when moving it.
- Keep a proposal that is partly done open, with the shipped part summarised at the top and only the remaining work below it. Move leftover work that depends on another proposal into that proposal.
- Add every proposal to the index below, and move its row when it is archived.

## Open

| Proposal | Status | Summary |
|---|---|---|
| [Replace Ideogram 4 with Ideogram 4 Instant](ideogram-instant.md) | In progress | `ideo` becomes fal's 8-step distilled Instant (one transformer, no CFG), converted locally on first use from an ungated, tensor-checked copy; FP8 for mflux and NF4/FP8 stored quants on CUDA |
| [MOSS-Image on Windows/Linux, with a 4-step Turbo](moss-image.md) | Proposed | A `moss` family on CUDA from a vendored, pinned pipeline file; Turbo (SFT + unmerged FP32 4-step LoRA) by default and full SFT as a variant, q4/q8 stored quants; macOS waits for official mflux support |
| [Controls for Web UI video jobs](video-job-controls.md) | In progress | Quit has shipped; Next and Pause for video jobs remain |
| [First and last frame for image-to-video](video-first-last-frame.md) | Proposed | An optional end image for `img2vid`: macOS first (MLX keyframe on the true last frame), then CUDA via `LTX2ConditionPipeline.from_pipe` with generate-and-trim once a spike confirms it; Last frame picker, `--end-image`, recorded settings and a two-frame enhancer rule |
| [Upscale existing videos 2× on macOS](video-upscale-existing.md) | Proposed | Generate at `s`, extend, then upscale the finished video 2× from the viewer with LTX's own upsampler and a 1–3 step distilled refine, tiled over time for long clips; original audio kept. Replaces a SeedVR2 plan that won't fit a 32 GB Mac; a half-day spike sets steps and tiling |
| [Edit images: inpaint, extend and Klein edits](image-editing.md) | Proposed | Paint a mask and redraw it, extend the canvas, and (Klein) edit from a prompt, through one editor opened from any image |
| [First run: pick a model, download it, and no gated defaults](first-run.md) | Proposed | Ungated mirrors for the shipped aliases, a Welcome panel that recommends a model for this machine, downloads with progress, and a Hugging Face token field |
| [Show model download progress in Web UI jobs](model-download-progress.md) | Proposed | Download a missing model as its own step before loading, and show bytes, speed and ETA on the job card, with Stop and resume |
| [Faster generation on macOS](macos-speed.md) | Proposed | Ranked speed opportunities on Apple Silicon (Ideogram quants, model reuse across Web UI jobs, LTX dtype and streaming, Klein text encoder), with a measurement plan to size them first |
| [Show memory use while a job runs](live-memory-usage.md) | Proposed | Feasibility study: live memory use during generation is readable on macOS, Windows and Linux; Windows VRAM spill into shared memory is the one part still to verify |
| [Search and favourites in the Gallery](gallery-search-favourites.md) | Proposed | A search field and model, workflow, date and favourites filters in the Gallery, backed by an in-memory asset index; a star on every tile stored in a sidecar file in the output folder |
| [Variations and sweeps from the Workspace](seed-variations-and-sweeps.md) | Proposed | "Variations" on any output queues N new-seed runs of its settings; a Sweep section expands one or two settings over a list of values into queued jobs, with the axis values recorded on each output |
| [Recent prompts, rewrite history and "save to a prompt file"](workspace-prompt-recents.md) | Proposed | A recents list of submitted prompts, the last five enhancer rewrites with a stepper, and saving a Workspace prompt into a prompt file |
| [Know when a job is done](job-notifications.md) | Proposed | Progress and the result in the tab title and favicon, an optional sound, and browser notifications where the page is a secure context |
| [A log of past jobs](job-log.md) | Proposed | One JSON line per finished job on disk and a Jobs drawer on every page with duration, outputs, failure text, Load settings and a copyable command |
| [A phone layout and an installable app](phone-layout.md) | In progress | One Workspace in three compositions (wide, tall, compact) with a touch layer keyed to the pointer; on a phone the preview is the screen, with a settings strip of chips and a composer under it; the job's live preview fills the stage everywhere. Done: all ten stages; open: the Playwright check at a phone viewport |
| [Named presets, and what a LoRA is for](settings-presets.md) | Proposed | Server-side presets of settings, LoRAs and style text applied from the Settings header or `--preset`; LoRA trigger words and previews read from the file and sidecars |
| [Grab a frame from a video](video-frame-grab.md) | Proposed | Frame stepping in the viewer and a Grab frame action that writes a PNG with provenance and can make it the reference for the next image or clip |
| [One Gallery](one-gallery.md) | Proposed | Call the Workspace strip Gallery, share one store and one endpoint between the strip and the page |
| [Merge the Models and Config pages](admin-page-merge.md) | Proposed | One Setup page with General, Models, LoRAs and Folders tabs, redirects for the old routes, one nav slot fewer |
| [Copy a CLI command from any asset](asset-cli-command.md) | Proposed | Build the reproducing `ziv` command from an asset's recorded settings in the tile menu and viewer, with a parity test against Reuse settings |
| [Prompt enhancer v3: a test bench, sharper rules, better options](prompt-enhancer-v3.md) | In progress | Done: the bench; a word floor for Same, a labelled prompt and quoted text kept exactly; the enhance options recorded on every output, shown in the viewer and restored by Reuse; the new option set (Theme, Camera and Pace, a realism core for photos, 15 art styles) in an inspector-style popover. Open: per-model hints, worked examples and the bench's local judge |

## Archive

| Proposal | Status | Summary |
|---|---|---|
| [Touch that feels native, on every screen](archive/touch-interaction.md) | Done (unreleased) | One sheet primitive that always pulls down, a shared pan tracker so the stage and drawer follow the finger, press feedback and hold progress, and touch hygiene (no sticky hover or hover-only controls, no zoom on focus, keyboard inset, pointer reorder); desktop keeps its density and gains the same motion and a trackpad swipe |
| [Feature parity between macOS and Windows/Linux](archive/platform-parity.md) | Done (unreleased) | Klein img2img, video audio, two-stage video upscale and its review, quantized and streamed video, video memory estimates, extending a video, Ideogram 4, the Z-Image scheduler, ffmpeg install and the SeedVR2 plain upscale on Windows and Linux; hand tests still owed are listed in it |
| [Delete enhancer models, and a Hugging Face downloads list](archive/huggingface-cache-manager.md) | Done (unreleased) | Enhancer rows with delete, disk use on every model, an Other downloads table for cached repos nothing uses with a cleanup of stranded shared files, deletes that free Hugging Face's shared store, and LoRA imports that skip the cache |
| [An in-app path browser](archive/web-path-browser.md) | Done (unreleased) | A file browser drawn in the Web UI from a server listing, plus path completion in the field, replaces the host-native dialog (Tk on Linux, `osascript` on macOS), so Browse works on every OS and from any device |
| [Group the backends into platform folders](archive/backend-folders.md) | Done (unreleased) | `backends/` split into `cuda/` (with `image/` and `video/`) and a flat `mac/`; the `_win`/`_mac` names dropped. A pure move |
| [Extend a video](archive/video-extend.md) | Done (unreleased) | Continue a clip from its last 25 frames in the one-stage loop and splice the new part onto the untouched original, with audio, on macOS; Windows/Linux moved to platform parity |
| [A scene for video prompts](archive/video-scene.md) | Done (unreleased) | An optional Scene beside the video prompt (CLI, prompt files, Web UI), sent first, and an enhancer that writes both a scene and an action in one call |
| [Upscale existing images on macOS with SeedVR2](archive/image-upscale-seedvr2-mac.md) | Done (unreleased) | Plain 2×/4× in the viewer's Upscale menu are a one-step SeedVR2 upscale (no prompt or source model, about linear cost), today's path is 2×/4× creative; the model tier is picked per Mac from its memory, stored as an 8-bit copy. Spike: creative never won a test set |
| [Review before upscaling](archive/upscale-review.md) | Done (unreleased) | Pause each in-run upscale after the small render: Approve runs the full-size pass, Discard skips it. Images everywhere, video on macOS; CUDA video moved to platform parity |
| [Linux: keep systemd-oomd from killing the editor during model loads](archive/linux-oomd-model-loads.md) | Done (unreleased) | Relax oomd's memory-pressure limit with an optional drop-in, measured over an 18-job run, and documented for Linux users |
| [Prompt enhancer: audio, image awareness and model-aware rewrites](archive/prompt-enhancer-v2.md) | Done (v0.13.0b19) | A Sound option for video with audio, clip length and frame shape in the rewrite, image-to-video rewrites that describe change, and a free-text Direction; vision, per-model hints and length trimming rejected |
| [macOS: mflux 0.22 and a fresh LTX](archive/deps-macos.md) | Done (v0.13.0b19) | mflux 0.22 and mlx-lm 0.32, and LTX re-vendored at v0.16.3 with upstream's two-stage upscale and low-memory streaming |
| [Windows and Linux: torch 2.14, CUDA 13 and diffusers 0.41](archive/deps-cuda.md) | Done (v0.13.0b19) | Move torch to the cu130 index so RTX 50-series GPUs work, and update diffusers, transformers, accelerate and bitsandbytes |
| [Dependency refresh: shared packages and the frontend](archive/deps-shared.md) | Done (v0.13.0b19) | Drop the unused `hf-transfer`, refresh the web server and general libraries, and update the frontend minors, then Vite 8 and Vitest 5 |
| [Toasts as a bottom snackbar](archive/web-toast-snackbar.md) | Done (v0.13.0b17) | Show one notification at a time at the bottom of the page content instead of a stack over the navigation; merge repeats, put errors first and keep them, add a Retry action |
| [Quantization parity on CUDA](archive/cuda-quant-parity.md) | Done (v0.13.0b17) | FP8 q8 that streams like unquantized weights, stored q4/q8 quants and memory-fit badges on Windows and Linux |
| [Decode in float32 on CUDA](archive/cuda-vae-float32.md) | Rejected (2026-10-09) | Tested float32 VAE decode on CUDA: no visible difference on Z-Image, Klein or Krea 2, so the grain comes from sampling |
| [One design system for the Web UI](archive/web-design-system.md) | Done (v0.13.0b16) | Replace the four styling systems in the Web UI with one compact system: shared sizes, type scale, labels, panels and overlays, enforced by `make check` |
| [No borders and Graphite with teal and coral](archive/web-no-borders-graphite.md) | Done (v0.13.0b16) | Drop outlines except dividers and meaningful states, move to neutral Graphite greys, and keep teal as primary with Blob's coral as secondary |
| [Prompt builder in the Web UI](archive/web-prompt-builder.md) | Done (v0.13.0b16) | A Prompts page that builds prompt files visually (snippets, sets, entries, enhance), keeps comments on save, and keeps the Workspace selection right |
| [Upscale existing images](archive/image-upscale-from-gallery.md) | Done (v0.13.0b16) | Upscale any image 2× or 4× from the asset viewer, refined with its original settings; shorter filenames and every recorded setting in the viewer |
| [Queue jobs in the Web UI and collapse the sidebar](archive/web-job-queue.md) | Done (v0.13.0b16) | Queue generation jobs on the server instead of refusing them, collapse the left sidebar, and add a shared confirmation dialog |
| [Keep image models resident and reuse quantized weights](archive/image-model-memory.md) | Done (v0.13.0b16) | Load mflux weights eagerly, cap MLX's buffer cache, and store a selected quant next to the model for reuse |
| [Enhance prompts before the model loads](archive/enhance-before-model-load.md) | Done (v0.13.0b8) | Fix out-of-memory failures in auto-enhanced batches by rewriting every prompt before the generation model loads |
