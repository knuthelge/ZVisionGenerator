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
| [Controls for Web UI video jobs](video-job-controls.md) | In progress | Quit has shipped; Next and Pause for video jobs remain |
| [Feature parity between macOS and Windows/Linux](platform-parity.md) | Proposed | Klein img2img, video audio, two-stage video upscale with review, quantized and streamed video with memory badges on CUDA, extending a video; matching Z-Image scheduler, ffmpeg install and enhancer download |
| [First and last frame for image-to-video](video-first-last-frame.md) | Proposed | An optional end image for `img2vid`: macOS first (MLX keyframe on the true last frame), then CUDA via `LTX2ConditionPipeline.from_pipe` with generate-and-trim once a spike confirms it; Last frame picker, `--end-image`, recorded settings and a two-frame enhancer rule |
| [Upscale existing videos 2× on macOS](video-upscale-existing.md) | Proposed | Generate at `m`, extend, then upscale the finished video 2× from the viewer with LTX's own upsampler and a 1–3 step distilled refine, tiled over time for long clips; original audio kept. Replaces a SeedVR2 plan that won't fit a 32 GB Mac; a half-day spike sets steps and tiling |
| [Edit images: inpaint, extend and Klein edits](image-editing.md) | Proposed | Paint a mask and redraw it, extend the canvas, and (Klein) edit from a prompt, through one editor opened from any image |
| [First run: pick a model, download it, and no gated defaults](first-run.md) | Proposed | Ungated mirrors for the shipped aliases, a Welcome panel that recommends a model for this machine, downloads with progress, and a Hugging Face token field |
| [Show model download progress in Web UI jobs](model-download-progress.md) | Proposed | Download a missing model as its own step before loading, and show bytes, speed and ETA on the job card, with Stop and resume |
| [Faster generation on macOS](macos-speed.md) | Proposed | Ranked speed opportunities on Apple Silicon (Ideogram quants, model reuse across Web UI jobs, LTX dtype and streaming, Klein text encoder), with a measurement plan to size them first |
| [Show memory use while a job runs](live-memory-usage.md) | Proposed | Feasibility study: live memory use during generation is readable on macOS, Windows and Linux; Windows VRAM spill into shared memory is the one part still to verify |

## Archive

| Proposal | Status | Summary |
|---|---|---|
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
