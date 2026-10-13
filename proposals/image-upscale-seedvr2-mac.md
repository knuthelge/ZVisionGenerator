# Upscale existing images on macOS with SeedVR2

**Status:** Proposed (2026-10-12)

## Problem

Upscaling an existing image ([Upscale existing images](archive/image-upscale-from-gallery.md)) is a Lanczos resize followed by one img2img pass of the whole frame with the image's own diffusion model. That works well up to about the size the model was trained on (1–2 megapixels). Above that it has three problems:

- **Quality.** At a high denoise the model invents new detail and repeats patterns, because it never saw frames that large. `existing_image_denoise` drops to 0.2 above `upscale.existing_large_megapixels` (2 MP) to stop this. At 0.2 the pass mostly sharpens the Lanczos result and adds little real detail. So the large upscales people want for wallpapers and prints, 5–20 MP, are the ones that gain least.
- **Cost.** The DiT attends over the whole frame, so time and memory grow faster than the pixel count. A 4× of a 1.5 MP image refines 24 MP of latents in one pass.
- **Coverage.** The pass needs a model that can do img2img and a prompt. Ideogram 4 images can't be upscaled at all (`supports_upscale: false`). Images with no recorded settings get the default model and a prompt that may not match them.

SeedVR2 fits this job. It is a one-step diffusion super-resolution model, already in the mflux 0.22 we pin, with an image entry point. It needs no prompt and no source model, stays close to the input, and its attention works in fixed-size windows, so cost grows about linearly with pixel count.

### Naming

To the user both are an upscale, so the menu says **Upscale** and only a qualifier tells them apart:

- **2× / 4×** is SeedVR2: the same picture, larger and sharper. It's what people expect from "upscale".
- **2× creative / 4× creative** is today's path: the image's own model redraws the picture larger and adds new detail. "Creative upscale" is the established term for this (Krea, Magnific; Topaz calls its version Redefine).

In code, config and recorded settings, the methods are `seedvr2` and `creative`.

### Relation to the video proposal

[Upscale existing videos on macOS with SeedVR2](video-upscale-seedvr2-mac.md) uses the same model. This proposal is the smaller first step and builds the shared foundation the video one then reuses:

- the SeedVR2 loader and the capability flag in `backends/`
- the `seedvr2` config block and the automatic model choice
- the mflux contract test
- the first-use download, the stored 8-bit copy and its row on the Models page
- the memory estimate and warning for SeedVR2 jobs
- the plain 2×/4× items in the viewer's Upscale menu

With that in place, the video proposal only adds the clip loop, frame I/O and chunking. Shipping images first also gives the video spike a measured single-frame baseline (quality, seconds per megapixel, peak memory) from the app itself, not from a script.

### Not in scope

- **Windows and Linux.** diffusers 0.41 has no SeedVR2 pipeline (see the video proposal). There the menu keeps only the creative items, and a capability flag gates SeedVR2 off.
- **Removing the creative upscale.** It stays. It adds detail in the image's own style at moderate sizes, which SeedVR2 by design doesn't.
- **SeedVR2 during generation** (`--upscale` in a run). A run would then load two models: the generation model for the small render and SeedVR2 for the upscale. That doesn't fit the one-model-per-job flow, so in-run upscaling stays creative only. To get a SeedVR2 upscale, generate small and upscale from the viewer.
- **Factors other than 2× and 4×.** SeedVR2 takes any target size, but the menu, the keyboard chord and `UPSCALE_FACTORS` stay at 2 and 4. To reach a specific size, generate at the matching width and height.
- **Tiled diffusion** (img2img over overlapping tiles). It was considered as a way to raise the creative path's limit; see [Alternatives considered](#alternatives-considered).
- **EXIF orientation of rotated camera JPEGs.** The menu and the creative path read a rotated JPEG's stored size, so creative's size label (and the plain items' label) can show the sides swapped. mflux orients the input itself, so the SeedVR2 output is upright and the right size. Fixing the label is a separate, later change.
- **SeedVR2 in the model listing command.** The Models page shows it, and the CLI has no upscale command to use it with.

## Evidence

Checked against `main` at 3789d09 and the installed mflux 0.22.0 (2026-10-12). The spike ran on 2026-10-13.

### mflux SeedVR2 for images

| Check | Result |
|---|---|
| Entry point | `SeedVR2.generate_image(seed, image_path, resolution, softness=0.0) -> GeneratedImage` in `mflux/models/seedvr2/variants/upscale/seedvr2.py` |
| Input | Reads a **file path** itself (`SeedVR2Util.preprocess_image`, which applies EXIF orientation via `open_oriented`). That suits upscaling an existing gallery file |
| Output size | `resolution` is the target **short side** (an `int`) or a `ScaleFactor`. Each side is scaled by `resolution / min(w, h)` and rounded down to even. The input is padded to 16 px and the output cropped back, so the output needs no 16-px alignment |
| Softness | `softness` 0–1 maps to a 1–8× bicubic pre-downsample before upscaling. 0 (the default) turns it off. mflux's README suggests 0.5 as a starting point |
| Steps and prompt | One step, `scheduler="seedvr2_euler"`, a fixed text embedding (`SeedVR2TextEmbeddings.load_positive()`). No prompt, guidance or LoRAs |
| Colour | `SeedVR2Util.apply_color_correction` runs wavelet reconstruction and then a Lab histogram transfer against the bicubic-resized input, so colours follow the source |
| VAE tiling | `SeedVR2Initializer` sets `model.tiling_config = TilingConfig()`, which by default tiles the encode (512 px tiles, 64 px overlap) and splits the decode 8 ways per side. Large images don't decode in one piece |
| Attention | `WindowPartitioner._make_windows` fixes the window size in latent tokens (scaled from a 45×80 reference) and adds more windows as the image grows. Attention cost is therefore about linear in pixel count, not quadratic like a full-frame DiT pass |
| Models | `ModelConfig.seedvr2_3b()` (alias `seedvr2`) and `seedvr2_7b()`, both from Hugging Face `numz/SeedVR2_comfyUI`. `SeedVR2(quantize=, model_path=, model_config=)`. `mflux-save` writes quantized copies, and the variant is read back from the saved weights |
| Weight sizes | Download: 3B fp16 6.8 GB, 7B fp16 16.5 GB, plus a 0.5 GB VAE. mflux always downloads the fp16 file and quantizes in memory, so int8 saves memory but not disk until the download is deleted (step 2). Loaded: 7B int8 9.26 GB, 3B int8 4.72 GB. Community model cards report int4 degrading this model badly (see the video proposal) |
| Licence | Apache-2.0 (ByteDance-Seed model card) |

### Our code

| Area | Today |
|---|---|
| Workflow | `build_upscale_workflow()`: resolve prompt → suppress negative → `load_source_stage` → `upscale_stage` (Lanczos, optional CAS, `backend.image_to_image`) → `sharpen_stage` → `save_image_stage` |
| Planning | `web/upscale_api.plan_upscale` picks the model from the recorded settings, falls back to the default model, and enforces `upscale_options` |
| Limits | `utils/upscale.py`: `UPSCALE_FACTORS = (2, 4)`, `UpscaleOption`, `upscale.max_megapixels` (20). `dimension_max` is only set for Ideogram 4 |
| Menu | `_asset_upscale_json` in `web/gallery.py` returns `{factors: [...]}`. `AssetViewer.svelte` builds the Upscale menu from it and binds a keyboard chord to 2× and 4× |
| Job | `POST /api/upscale` `{asset_id, factor}` → `WebRunner.submit_upscale_job` → `upscale_runner.run_upscale` |
| Memory | `memory_mac.memory_budget_bytes()` (Apple's recommended working set), `classify_memory_fit`, `read_safetensors_totals`, and stored quants (`utils/stored_quant.py`) that keep a quantized copy next to the model |
| Platform flags | `supports_video_audio()` and `supports_video_upscale_review()` in `backends/__init__.py` |

### Spike results

Measured with `SeedVR2.generate_image` outside the app, on a 32 GB M5 Mac with a memory budget (Apple's recommended working set) of 26.80 GB. No 64 GB Mac was measured. Sources: eight purpose-built renders of 1–1.3 MP from the Krea 2, Z-Image Turbo and FLUX.2 Klein 9B presets (a close-up portrait, a street with small faces, a shop front with legible text, ink line art, foliage, architecture, an oil painting, a noisy night scene with neon), plus five portrait renders. The review was blind, one set at a time against a bicubic upscale of the original.

| Question | Result |
|---|---|
| SeedVR2 against creative | Creative never won a set. At 2× it came second in two of the eight test sets. Creative 4× doesn't run on this Mac (below), so at 4× SeedVR2 has no competitor |
| Tier ranking | Test set: 7B int8 won 7 of 8 sets at 2× and 6 of 8 at 4× (one tie; 3B bf16 won the other two). Several 4× sets were "almost a tie". Portraits: 7B int8 and 7B bf16 took the top two places in every 2× set, so int8 costs nothing visible. 3B bf16 and 3B int8 couldn't be told apart |
| Softness | Round 1 (0, 0.25, 0.5): 0.5 won 6 of 8 and 0 never won. Round 2 at 2× on 7B int8 (0.5, 0.75, 1.0): 0.75 placed first four times and second three times, and never last; 0.5 placed first three times; 1.0 never placed first. Differences between sources were small. Softness doesn't change memory or time |
| Stability | No Metal error up to 20 MP. 31.9 MP outputs (about 1.4× the budget) completed on 7B int8 and 3B bf16 while macOS swapped, with no error |

Peak memory and time per output size (one call):

| Tier | Weights | 4 MP | 8 MP | 16 MP | 20 MP | 31.9 MP |
|---|---|---|---|---|---|---|
| 7B int8 | 9.26 GB | 20.05 GB, 80 s | 20.12 GB, 163 s | 24.47 GB, 375 s | 27.61 GB, 485 s | 37.45 GB, 782 s |
| 3B int8 | 4.72 GB | 15.51 GB, 80 s | 15.58 GB, 156 s | 20.02 GB, 314 s | 23.60 GB, 401 s | not run |
| 3B bf16 (dropped) | 7.28 GB | 18.08 GB, 81 s | 18.15 GB, 142 s | 22.58 GB, 320 s | 26.16 GB, 401 s | 36.69 GB, 668 s |
| 7B bf16 (dropped) | 16.98 GB | 27.77 GB, 98 s | | | | |

- **Memory model.** Peak = weights + activations, with activations = max(10.79 GB, slope × MP + intercept): 0.83 GB/MP + 1.88 GB for 7B int8, 0.89 GB/MP + 1.10 GB for 3B int8. The flat floor holds up to about 8 MP. Fitted on 12–20 MP, it predicts the 31.9 MP runs within 0.2 GB.
- **Largest output inside the budget** on this Mac: 19.0 MP on 7B int8, 23.6 MP on 3B int8.
- **Time** is about linear in pixels: 20 s/MP on 3B, 20–24 s/MP on 7B int8. A 4× of a 1 MP render (16 MP) takes about 5 minutes on 3B and 6 on 7B int8.
- **Creative, for comparison.** 2× to 4 MP peaked at 43–45 GB, past physical memory, and took 4–9 minutes under swap. 2× to 8 MP and every 4× failed with a Metal out-of-memory error.
- **Load.** Quantizing in memory takes 1–3 s on a warm load, so a stored 8-bit copy saves disk, not time.

Decisions taken from these results:

- Tiers, largest first: **7B int8**, then **3B int8**. 3B bf16 is dropped: it looks the same as 3B int8 and needs more memory and disk. 7B bf16 is dropped: it looks the same as 7B int8 and doesn't fit a 32 GB Mac's budget even at 4 MP. A 32 GB Mac gets 7B int8; a Mac whose budget is under the 7B estimate at 16 MP (24.4 GB) gets 3B int8.
- Default softness 0.75, one value for every source.

## Proposed change

On macOS the viewer's Upscale menu shows **2×** and **4×** (SeedVR2) and, below them, **2× creative** and **4× creative** (today's upscale). Every item shows its output size and, when it's disabled, the reason. A SeedVR2 item that may need more memory than the Mac has stays enabled and shows a warning. Nothing moves behind a switch. A SeedVR2 job loads SeedVR2 instead of the image's model, runs one step, and saves the result next to the source, with the source's prompt and settings carried over for the record.

The app picks the SeedVR2 model for the machine, always 8-bit. Nobody has to choose.

Build it in one worktree, with a commit per step. Each step leaves the app working.

### 0. Spike, with recorded results

Done. SeedVR2 passed the gate: creative didn't win a single set. The numbers, the tier ranking, the activation costs and the softness default are in [Spike results](#spike-results).

### 1. Shared SeedVR2 backend, model choice and sizes (foundation for video too)

Backend:

- `core/seedvr2.py`: a `SeedVR2Upscaler` Protocol with `load_model(variant, quantize, model_path)`, `save_model(model, path)` (for the stored 8-bit copy in step 2) and `upscale_image(model, path, width, height, seed, softness) -> PIL.Image`.
- `backends/seedvr2_mac.py`: the mflux implementation, with mflux imported lazily. It passes the tier's `ModelConfig.seedvr2_3b()` or `seedvr2_7b()` explicitly, since `SeedVR2` doesn't read the variant back from saved weights. `upscale_image` calls `generate_image` with `resolution` set to the target short side and returns `.image`. The video proposal later adds `upscale_clip` to this module.
- `backends/memory_mac.py` takes over the two helpers that materialize loaded weights and cap MLX's buffer cache, so the image and SeedVR2 backends share them.
- `backends/__init__.py`: `get_seedvr2_upscaler()` and `supports_seedvr2()`. They return the mflux upscaler and `True` on macOS, and `None` and `False` elsewhere. No other module branches on platform. The video proposal's `supports_video_file_upscale()` builds on this flag.
- `tests/test_mflux_seedvr2_contract.py` pins by introspection only what the image path uses: `SeedVR2.__init__(quantize, model_path, model_config)`, `generate_image(seed, image_path, resolution, softness)` and its `.image`, `save_model(base_path)`, `ModelConfig.seedvr2_3b` / `seedvr2_7b`, each tier's transformer and VAE file names, and the saved copy's `transformer/` and `vae/` layout. The video proposal extends the same file with the latent creator, text embeddings and transformer call it relies on.

Automatic model choice:

- Config: a new top-level `seedvr2:` block with `model: auto`, `softness: 0.75`, `reference_megapixels: 16` and `tiers`. `tiers` lists, largest first, 7B int8 and 3B int8, each with its weights and the spike's activation constants (floor, slope, intercept), so new measurements only change `config.yaml`. Every tier is int8; bf16 and int4 aren't tiers, and there's no quantize setting. Setting `model` to `seedvr2-3b` or `seedvr2-7b` overrides the automatic choice; any other value is a config error. It follows the usual precedence rules.
- `utils/seedvr2.py`, pure:
  - `estimate_seedvr2_bytes(tier, megapixels)`: weights plus max(floor, slope × MP + intercept).
  - `pick_seedvr2_tier(settings, budget_bytes)` returns the first tier whose estimate at the reference size is within the budget (at most 1.0× it), and the last (smallest) tier when none is or the budget is unknown. It never disables SeedVR2. The same within-budget rule decides the menu warnings below, so the automatic tier never warns at the reference size.
- The choice is made once per machine, from `memory_budget_bytes()`, so a given Mac always uses (and downloads) the same model. It doesn't depend on the image, which would load different models for 2× and 4×. The reference size is a 4× of a 1 MP render (16 MP). A 32 GB Mac gets 7B int8 (24.4 GB at 16 MP, inside its 26.8 GB budget). A Mac whose budget is under that gets 3B int8 (20.1 GB at 16 MP): no swapping at 4× and less than half the download, for a small quality gap at 4×. `model: seedvr2-7b` still gives it 7B.
- The job card and the recorded settings show the model that ran (for example `seedvr2-7b q8`).
- Tests: the tier at a budget exactly equal to the 7B estimate at 16 MP and one byte under it; the smallest tier when the budget is unknown or nothing is within it; the `model` override wins; an unknown `model` is rejected.

Sizes:

- `utils/upscale.py`:
  - `seedvr2_output_size(width, height, factor)` matches `preprocess_image`: each side scaled and rounded down to even.
  - `seedvr2_options(width, height, tier, budget_bytes)` returns `UpscaleOption`s. SeedVR2 options are always enabled; there's no size cap. An option gets a `warning` when the tier's estimate for that size exceeds the memory budget. It ignores the image model's capabilities, so Ideogram 4 images and images without settings can be upscaled. Creative's options and its 20 MP limit don't change.
- Tests: sizes for odd and even sources; the warning just over the budget, and none at or under it.

### 2. Stored 8-bit copy

SeedVR2 isn't in the app's model inventory. Disk use comes first:

- Only the tier this Mac uses is downloaded: its fp16 transformer (16.5 GB for 7B, 6.8 GB for 3B) and the 0.5 GB VAE. mflux fetches them itself on the first load.
- mflux quantizes the tier to 8 bits while loading, and the app holds the weights in memory. It then stores the copy in a hidden `.seedvr2` folder in the models folder, the same way stored image-model copies are written. The copy carries its own transformer and VAE.
- Once the copy is saved, the app deletes both downloaded files, the transformer and the VAE, from the Hugging Face cache. Later jobs load the copy. If the save fails, nothing is deleted and the next job tries again.
- Whether SeedVR2 counts as downloaded is simply whether a current copy exists.

This is the step the video proposal's step 6 asked for. If [model download progress](model-download-progress.md) has shipped by then, it covers this; otherwise the job card shows "Downloading SeedVR2" with no progress bar. The first-use message says the copy saves disk, not that it loads faster: a warm in-memory quantize takes only seconds.

### 3. Workflow, runner and metadata

- `build_seedvr2_upscale_workflow()`: `resolve_prompt_stage` → `load_source_stage` → `upscale_stage` → `save_image_stage`. The prompt stage stays so the saved file carries the source's rendered prompt. There's no sharpen stage: SeedVR2 output is already sharp.
- The SeedVR2 `upscale_stage` lives in its own module (so the job card's "Upscaling" label keeps working), calls `request.backend.upscale_image` with the source path and the planned size, and replaces `artifacts.image`.
- `upscale_runner.run_upscale` runs it unchanged: it already takes a workflow and emits one-image batch events. SeedVR2 has one step, so the job card shows load → upscale → save without a step bar. A quit during the load is caught by `run_upscale`'s existing check before the workflow; a quit during the step saves nothing.
- Filename: `portrait_42_2x.png`, the same rule as creative. Since the method is recorded, the name doesn't need to differ.
- Recorded settings: `workflow: "upscale"`, the existing `source` block, and an `upscale` block with `method: "seedvr2"`, factor, model, quantize, softness and seed. The source's prompt, model and LoRAs are kept, so Reuse and the viewer still show how the original was made. A seed, steps or guidance the source didn't record is saved as empty, never as a default. Creative upscales gain `method: "creative"`. Old files without a method read as creative.
- Planning needs no image model: the recorded model is only kept for the record.
- Tests: stage order; the saved prompt; the payload's `method`; settings the source didn't record stay empty; events for success, quit and failure with a mocked backend.

### 4. Web server and contract

- `_asset_upscale_json` keeps `factors` (creative) and adds `seedvr2`: the options, the tier's name, whether it's downloaded and, until it is, the first-use download size; or `null` when `supports_seedvr2()` is false. It isn't tied to the image's model, so Ideogram 4 images get working 2× and 4× items while the creative items stay disabled with their existing reason.
- `POST /api/upscale` takes `method: "seedvr2" | "creative"`, defaulting to `"creative"` so existing clients keep working. `plan_seedvr2_upscale` (next to `plan_upscale` in `web/upscale_api.py`) builds the request.
- No memory admission check and no size cap: a SeedVR2 job over the estimated budget is queued like any other, since its item already carries the warning. Only `seedvr2` on a platform without it is refused.
- Tests: the gallery JSON on supported and unsupported backends, the route's method default, a warned size that queues, and queuing with the SeedVR2 workflow.

### 5. Web UI

- The Upscale menu lists 2×, 4×, 2× creative and 4× creative, each with its size and reason line. A warned SeedVR2 item stays enabled with a visible marker and the warning as its tooltip. Before the first SeedVR2 job, the plain items also say "First use downloads 17.0 GB" (or the 3B size). On Windows and Linux only the creative items appear, since `seedvr2` is `null`.
- Keyboard: the existing chord (X, then 2 or 4) runs plain 2× or 4×. That's SeedVR2 on macOS and creative where SeedVR2 isn't available. On macOS the creative items are in the menu only; there's no second chord. The shortcut help keeps its "Upscale 2×" and "Upscale 4×" labels.
- The job card shows "Upscale 2×" or "Upscale 2× creative", the output size, and the SeedVR2 model that ran.
- Run `make frontend-build` and commit the rebuilt SPA.
- Vitest: the four items render with their sizes; the plain items are absent when `seedvr2` is `null`; a warned item stays enabled; a plain item sends `method: "seedvr2"` and a creative one `method: "creative"`; the chord sends `seedvr2` on macOS and `creative` elsewhere.

### 6. Models page

- On macOS the Models page's Image models table gets one SeedVR2 row for the tier this Mac uses (for example `seedvr2-7b@q8`), with whether it's downloaded and the disk use of the stored copy.
- Its delete button removes the stored copy through the existing delete path and confirmation. The next plain upscale downloads the tier again. Downloaded files are never deleted from here (the app has already removed them).
- No memory badge on this row: memory is judged per output size in the Upscale menu, and a single badge would contradict it. SeedVR2 stays out of the model picker.
- Run `make frontend-build` and commit the rebuilt SPA.

### 7. Docs and changelog

- The Web UI guide's upscale section explains plain and creative upscales: when to use each, that plain uses SeedVR2 on macOS only, which model a Mac gets, the first-use download and what happens to it, and the time and memory from the spike.
- Add the `seedvr2` keys to the upscale section's config snippet and the `method` field to the embedded-settings list. The models guide mentions the SeedVR2 row on the Models page.
- `docs/development.md`: the new modules in Project Structure and the contract test as the record of the mflux internals we rely on.
- `CHANGELOG.md` entries under `[Unreleased]` land with the Web UI and Models page steps.
- Move this proposal to `archive/`, update the index, and update the video proposal's Evidence to point at the shipped foundation.

## Alternatives considered

- **Tiled img2img** (the "Ultimate SD Upscale" approach): run the creative pass over overlapping ~1024–1344 px tiles with the image's own model and feather the seams. It keeps the creative style and lifts its memory ceiling. But every tile sees the full prompt, so above about 0.35 denoise subjects repeat across tiles. Seams need tuning per model, and time grows with tile count times steps. Keep it as a later option if SeedVR2 is too conservative for painterly sources.
- **Make SeedVR2 the only upscale on macOS.** It's simpler, but it drops the creative upscale's "add detail in the model's style" at moderate sizes.
- **A model picker in the menu or a settings choice between 3B and 7B.** It puts a decision on the user that the app can make from memory alone. The config override covers anyone who wants a specific tier.
- **Picking the tier per image from its output size.** It uses the biggest model each time, but 2× and 4× of the same image could load and download different models. A per-machine choice is predictable.
- **The mflux ControlNet upscaler.** mflux's own README calls SeedVR2 the preferred, faster and more faithful option.
- **GAN upscalers (Real-ESRGAN and similar).** They're fast, but they'd add a dependency, and they smear or invent texture on generated images more than SeedVR2 does.
- **numz's PyTorch SeedVR2 on MPS**, which would also cover CUDA. It brings PyTorch into the macOS stack, which is MLX-only today; see the video proposal.

## Open questions

- **Softness per source.** Settled by the spike: the differences between sources were small, so there's one default (0.75) and no per-family value.
