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
- the model inventory entry and download
- the memory admission check for SeedVR2 jobs
- the plain 2×/4× items in the viewer's Upscale menu

With that in place, the video proposal only adds the clip loop, frame I/O and chunking. Shipping images first also gives the video spike a measured single-frame baseline (quality, seconds per megapixel, peak memory) from the app itself, not from a script.

### Not in scope

- **Windows and Linux.** diffusers 0.41 has no SeedVR2 pipeline (see the video proposal). There the menu keeps only the creative items, and a capability flag gates SeedVR2 off.
- **Removing the creative upscale.** It stays. It adds detail in the image's own style at moderate sizes, which SeedVR2 by design doesn't.
- **SeedVR2 during generation** (`--upscale` in a run). A run would then load two models: the generation model for the small render and SeedVR2 for the upscale. That doesn't fit the one-model-per-job flow, so in-run upscaling stays creative only. To get a SeedVR2 upscale, generate small and upscale from the viewer.
- **Factors other than 2× and 4×.** SeedVR2 takes any target size, but the menu, the keyboard chord and `UPSCALE_FACTORS` stay at 2 and 4. To reach a specific size, generate at the matching width and height.
- **Tiled diffusion** (img2img over overlapping tiles). It was considered as a way to raise the creative path's limit; see [Alternatives considered](#alternatives-considered).

## Evidence

Checked against `main` at 3789d09 and the installed mflux 0.22.0 (2026-10-12).

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
| Weight sizes | 3B fp16 is about 7 GB with its VAE. 7B is 16.5 GB at fp16 and about 8.8 GB at int8. Community model cards report 7B int8 output nearly identical to fp16, and int4 degrading this model badly (see the video proposal) |
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

## Proposed change

On macOS the viewer's Upscale menu shows **2×** and **4×** (SeedVR2) and, below them, **2× creative** and **4× creative** (today's upscale). Every item shows its output size and, when it's disabled, the reason. Nothing moves behind a switch. A SeedVR2 job loads SeedVR2 instead of the image's model, runs one step, and saves the result next to the source, with the source's prompt and settings carried over for the record.

The app picks the SeedVR2 model and quantization for the machine. Nobody has to choose.

Build it in one worktree, with a commit per step. Each step leaves the app working.

### 0. Spike, with recorded results

Run `SeedVR2.generate_image` directly, outside the app. Use six sources: a portrait, a landscape with foliage, text or line art, a Z-Image render, a Krea 2 render and an Ideogram 4 render. Each is 1–2 MP.

| Gate | Pass when |
|---|---|
| SeedVR2 against creative | In a blind side-by-side at 2× and 4×, SeedVR2 is preferred or tied on most sources against today's creative output at the same size. Note where creative wins (likely painterly styles at 2×) |
| Model tiers | Compare 7B int8, 3B bf16 and 3B int8 blind on the same sources. Confirm the ranking used by the automatic choice, and that 7B bf16 looks no better than 7B int8 |
| Softness | Pick the default from 0, 0.25 and 0.5 |
| Memory and time | Record `mx.get_peak_memory()` and seconds for each tier at 4, 8, 12 and 20 MP outputs, on a 32 GB and a 64 GB or larger Mac. Fit a per-megapixel activation cost per tier |
| Stability | No Metal command-buffer timeout at 20 MP. If there is one, record the largest size that works |

Write the numbers, the tier ranking, the activation costs and the softness default into this proposal's Evidence section before step 1. If SeedVR2 doesn't beat creative on most sources, stop and reject the proposal.

### 1. Shared SeedVR2 backend (foundation for video too)

- `core/seedvr2.py`: a `SeedVR2Upscaler` Protocol with `load_model(variant, quantize)` and `upscale_image(model, path, width, height, seed, softness) -> PIL.Image`.
- `backends/seedvr2_mac.py`: the mflux implementation, with mflux imported lazily. `upscale_image` calls `generate_image` with `resolution` set to the target short side and returns `.image`. A quantized tier is stored with `stored_quant` the first time, so later jobs load it without quantizing again. The video proposal later adds `upscale_clip` to this module.
- `backends/__init__.py`: `get_seedvr2_upscaler()` and `supports_seedvr2()`. They return the mflux upscaler and `True` on macOS, and `None` and `False` elsewhere. No other module branches on platform. The video proposal's `supports_video_file_upscale()` builds on this flag.
- `tests/test_mflux_seedvr2_contract.py` pins by introspection: `SeedVR2.__init__(quantize, model_path, model_config)`, `generate_image(seed, image_path, resolution, softness)`, `save_model(base_path)`, the `bits` and `tiling_config` attributes, and `ModelConfig.seedvr2_3b` / `seedvr2_7b`. The video proposal extends the same file with the latent creator, text embeddings and transformer call it relies on.
- Mocked tests: `upscale_image` passes the right short side for 2× and 4× and returns the PIL image.

### 2. Automatic model choice

- `utils/seedvr2.py`, pure:
  - `SEEDVR2_TIERS`: the tiers in quality order from the spike. The expected order is 7B int8, 3B bf16, 3B int8. 7B bf16 isn't a tier, because it looks the same as int8 at twice the memory. int4 isn't a tier either, because it degrades this model.
  - `pick_seedvr2_tier(budget_bytes, weight_bytes, activation_bytes_per_mp, reference_mp)` returns the first tier whose weights plus activations at the reference output size are `fits` under `classify_memory_fit`. It returns `None` when even the last tier doesn't fit, and then the SeedVR2 items are disabled with that reason.
  - `seedvr2_max_megapixels(tier, budget_bytes, ...)` returns the largest output the chosen tier fits, capped by `seedvr2.max_megapixels`.
- The choice is made once per machine, from `memory_budget_bytes()`, so a given Mac always uses (and downloads) the same model. It doesn't depend on the image, which would load different models for 2× and 4×. The reference size is the spike's typical 4× output (about 8 MP, to confirm).
- Config: a new top-level `seedvr2:` block with `model: auto`, `quantize: auto`, `softness` and `max_megapixels`. Setting `model` (`seedvr2-3b`, `seedvr2-7b`) or `quantize` (`null`, `8`) overrides the automatic choice. They follow the usual precedence rules.
- The job card and the recorded settings show the model that ran (for example `seedvr2-7b q8`).
- Tests: the picked tier for budgets of 16, 24, 32, 64 and 128 GB with fixed weight and activation numbers; `None` when nothing fits; config overrides win; the size cap.

### 3. Sizes and limits

- `utils/upscale.py`:
  - `seedvr2_output_size(width, height, factor)` matches `preprocess_image`: each side scaled and rounded down to even.
  - `seedvr2_options(width, height, max_mp)` returns `UpscaleOption`s against the tier's limit from step 2. It ignores the image model's capabilities, so Ideogram 4 images and images without settings can be upscaled.
- Tests: sizes for odd and even sources, and the limit reason.

### 4. Workflow, runner and metadata

- `build_seedvr2_upscale_workflow()`: `load_source_stage` → `seedvr2_upscale_stage` → `save_image_stage`. There's no prompt stage. Sharpening is off by default, since SeedVR2 output is already sharp; the spike can turn it on with `sharpening.existing_seedvr2`.
- `seedvr2_upscale_stage` calls `request.backend.upscale_image` with the source path and the planned size. It replaces `artifacts.image`.
- `upscale_runner.run_upscale` runs it unchanged: it already takes a workflow and emits one-image batch events. SeedVR2 has one step, so the job card shows load → upscale → save without a step bar.
- Filename: `portrait_42_2x.png`, the same rule as creative. Since the method is recorded, the name doesn't need to differ.
- Recorded settings: `workflow: "upscale"`, the existing `source` block, and an `upscale` block with `method: "seedvr2"`, factor, model, quantize, softness and seed. The source's prompt, model and LoRAs are kept, so Reuse and the viewer still show how the original was made. Creative upscales gain `method: "creative"`. Old files without a method read as creative.
- Memory: load under `mlx_cache_limit`, call `release_memory()` after the job, and keep one model resident at a time.
- Tests: stage order; the payload's `method`; events for success, quit and failure with a mocked backend.

### 5. Web server and contract

- `_asset_upscale_json` returns `{seedvr2: [...] | null, creative: [...]}`. `seedvr2` is `null` when `supports_seedvr2()` is false. It isn't tied to the image's model, so Ideogram 4 images get working 2× and 4× items while the creative items stay disabled with their existing reason.
- `POST /api/upscale` takes `method: "seedvr2" | "creative"`, defaulting to `"creative"` so existing clients keep working. `plan_seedvr2_upscale` (next to `plan_upscale` in `web/upscale_api.py`) builds the request and enforces `seedvr2_options`.
- Admission: estimate the chosen tier's memory from the safetensors headers plus the activation cost for this output size, and refuse with a clear message when it won't fit, as image jobs do.
- Tests: the gallery JSON on supported and unsupported backends, the route's method default, a refused size, and queuing with the SeedVR2 workflow.

### 6. Web UI

- The Upscale menu lists 2×, 4×, 2× creative and 4× creative, each with its size and reason line. On Windows and Linux only the creative items appear, since `seedvr2` is `null`.
- Keyboard: the existing chord runs plain 2× or 4×. That's SeedVR2 on macOS and creative where SeedVR2 isn't available. Shift with the same chord runs creative everywhere.
- The job card shows "Upscale 2×" or "Upscale 2× creative", the output size, and the SeedVR2 model that ran.
- Run `make frontend-build` and commit the rebuilt SPA.
- Vitest: the four items render with their sizes; the plain items are absent when `seedvr2` is `null`; a plain item sends `method: "seedvr2"` and a creative one `method: "creative"`; the chord and Shift chord send the right method.

### 7. Model download

SeedVR2 isn't in the app's model inventory. Add it to `ziv-model list` and the inventory, so the memory badge and the "not downloaded" state work. Only the tier picked for this Mac is downloaded, through the same path as other models. This is the step the video proposal's step 6 asked for. If [model download progress](model-download-progress.md) has shipped by then, it covers this; otherwise the job card shows "Downloading SeedVR2" with no progress bar.

### 8. Docs and changelog

- The Web UI guide's upscale section explains plain and creative upscales: when to use each, that plain uses SeedVR2 on macOS only, which model a Mac gets, and the time and memory from the spike.
- Add the `seedvr2` keys and the `method` field to the configuration reference.
- `docs/development.md`: the new modules in Project Structure, the mflux internals we rely on, and the contract test.
- A `CHANGELOG.md` entry under `[Unreleased]`.
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

- **Softness per source.** If the spike shows renders and photos want different softness, decide whether that's a config value per family or one default.
