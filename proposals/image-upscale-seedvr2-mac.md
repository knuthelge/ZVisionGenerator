# Upscale existing images on macOS with SeedVR2

**Status:** Proposed (2026-10-12)

## Problem

Upscaling an existing image ([Upscale existing images](archive/image-upscale-from-gallery.md)) is a Lanczos resize followed by one img2img pass of the whole frame with the image's own diffusion model. That works well up to about the size the model was trained on (1–2 megapixels). Above that it has three problems:

- **Quality.** At a high denoise the model invents new detail and repeats patterns, because it never saw frames that large. `existing_image_denoise` drops to 0.2 above `upscale.existing_large_megapixels` (2 MP) to stop this. At 0.2 the pass mostly sharpens the Lanczos result and adds little real detail. So the large upscales people want for wallpapers and prints, 5–20 MP, are the ones that gain least.
- **Cost.** The DiT attends over the whole frame, so time and memory grow faster than the pixel count. A 4× of a 1.5 MP image refines 24 MP of latents in one pass.
- **Coverage.** The refinement needs a model that can do img2img and a prompt. Ideogram 4 images can't be upscaled at all (`supports_upscale: false`). Images with no recorded settings get the default model and a prompt that may not match them.

SeedVR2 fits this job. It is a one-step diffusion restoration model, already in the mflux 0.22 we pin, with an image entry point. It needs no prompt and no source model, stays close to the input, and its attention works in fixed-size windows, so cost grows about linearly with pixel count.

### Relation to the video proposal

[Upscale existing videos on macOS with SeedVR2](video-upscale-seedvr2-mac.md) uses the same model. This proposal is the smaller first step and builds the shared foundation the video one then reuses:

- the SeedVR2 loader and the capability flag in `backends/`
- the `seedvr2` config block (model and quantization)
- the mflux contract test
- the model inventory entry and download
- the memory admission check for SeedVR2 jobs
- the Restore group in the viewer's Upscale menu

With that in place, the video proposal only adds the clip loop, frame I/O and chunking. Shipping images first also gives the video spike a measured single-frame baseline (quality, seconds per megapixel, peak memory) from the app itself, not from a script.

### Not in scope

- **Windows and Linux.** diffusers 0.41 has no SeedVR2 pipeline (see the video proposal). The SeedVR2 entries are gated off by a capability flag, and the refine path stays the only upscale there.
- **Replacing refinement.** Refinement stays. It can add detail in the image's own style at moderate sizes, which SeedVR2 by design doesn't. The two answer different questions: "re-imagine this bigger" and "make exactly this bigger".
- **In-run upscaling** (`--upscale` during generation). That stays as it is for now. See [Open questions](#open-questions).
- **Factors other than 2× and 4×.** SeedVR2 takes any target size, but the menu, the keyboard chord and `UPSCALE_FACTORS` stay at 2 and 4. To reach a specific size, generate at the matching width and height.
- **Tiled diffusion refinement** (img2img over overlapping tiles). It was considered as a way to raise the refine path's limit; see [Alternatives considered](#alternatives-considered).

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
| Models | `ModelConfig.seedvr2_3b()` (default, alias `seedvr2`) and `seedvr2_7b()`, both from Hugging Face `numz/SeedVR2_comfyUI`. `SeedVR2(quantize=, model_path=, model_config=)`. `mflux-save` writes quantized copies, and the variant is read back from the saved weights |
| Licence | Apache-2.0 (ByteDance-Seed model card) |

### Our code

| Area | Today |
|---|---|
| Workflow | `build_upscale_workflow()`: resolve prompt → suppress negative → `load_source_stage` → `upscale_stage` (Lanczos, optional CAS, `backend.image_to_image`) → `sharpen_stage` → `save_image_stage` |
| Planning | `web/upscale_api.plan_upscale` picks the refinement model from the recorded settings, falls back to the default model, and enforces `upscale_options` |
| Limits | `utils/upscale.py`: `UPSCALE_FACTORS = (2, 4)`, `UpscaleOption`, `upscale.max_megapixels` (20). `dimension_max` is only set for Ideogram 4 |
| Menu | `_asset_upscale_json` in `web/gallery.py` returns `{factors: [...]}`. `AssetViewer.svelte` builds the Upscale menu from it and binds a keyboard chord to 2× and 4× |
| Job | `POST /api/upscale` `{asset_id, factor}` → `WebRunner.submit_upscale_job` → `upscale_runner.run_upscale` |
| Platform flags | `supports_video_audio()` and `supports_video_upscale_review()` in `backends/__init__.py` |

## Proposed change

On macOS the viewer's Upscale menu gets a second group, **Restore** (SeedVR2), next to the existing refinement, which is renamed **Refine**. Each group has 2× and 4×, with output sizes and disabled reasons. Both groups are always visible, and nothing moves behind a switch. A Restore job loads SeedVR2 instead of the image's model, runs one step, and saves the result next to the source with the source's prompt and settings carried over for the record.

Build it in one worktree, with a commit per step. Each step leaves the app working.

### 0. Spike, with recorded results

Run `SeedVR2.generate_image` directly, outside the app. Use six sources: a portrait, a landscape with foliage, text or line art, a Z-Image render, a Krea 2 render and an Ideogram 4 render. Each is 1–2 MP.

| Gate | Pass when |
|---|---|
| Quality against refine | In a blind side-by-side at 2× and 4×, Restore is preferred or tied on most sources against today's refine output at the same size. Note where refine wins (likely painterly styles at 2×) |
| Softness | Pick the default from 0, 0.25 and 0.5 |
| Memory and time | Record `mx.get_peak_memory()` and seconds for 3B bf16, 3B int8 and 7B int8 at 4, 8, 12 and 20 MP outputs, on a 32 GB and a 64 GB or larger Mac |
| Stability | No Metal command-buffer timeout at 20 MP. If there is one, record the largest size that works |

Write the numbers and the chosen defaults (model, quantization, softness, size limit) into this proposal's Evidence section before step 1. If Restore doesn't beat refine on most sources, stop and reject the proposal.

### 1. Shared SeedVR2 backend (foundation for video too)

- `core/restorer.py`: a `Restorer` Protocol with `load_model(variant, quantize)` and `restore_image(model, path, width, height, seed, softness) -> PIL.Image`.
- `backends/restore_mac.py`: the mflux implementation, with mflux imported lazily. `restore_image` calls `generate_image` with `resolution` set to the target short side and returns `.image`. The video proposal later adds `upscale_clip` to this module.
- `backends/__init__.py`: `get_restorer()` and `supports_seedvr2_upscale()`. They return the mflux restorer and `True` on macOS, and `None` and `False` elsewhere. No other module branches on platform. The video proposal's `supports_video_file_upscale()` becomes this flag plus its own checks.
- Config: a new top-level `seedvr2:` block with `model: seedvr2-3b`, `quantize`, `softness` and `max_megapixels`, using the defaults from the spike. The video proposal's `video_upscale:` block keeps only its clip settings (`chunk_frames`, `overlap_frames`) and reads model and quantization from `seedvr2`.
- `tests/test_mflux_seedvr2_contract.py` pins by introspection: `SeedVR2.__init__(quantize, model_path, model_config)`, `generate_image(seed, image_path, resolution, softness)`, the `bits` and `tiling_config` attributes, and `ModelConfig.seedvr2_3b` / `seedvr2_7b`. The video proposal extends the same file with the latent creator, text embeddings and transformer call it relies on.
- Mocked tests: `restore_image` passes the right short side for 2× and 4× and returns the PIL image.

### 2. Sizes and limits

- `utils/upscale.py`:
  - `restore_output_size(width, height, factor)` matches `preprocess_image`: each side scaled and rounded down to even.
  - `restore_options(width, height, config)` returns `UpscaleOption`s against `seedvr2.max_megapixels`. It ignores model capabilities, so Ideogram 4 images and images without settings can be restored.
- Tests: sizes for odd and even sources, and the limit reason.

### 3. Workflow, runner and metadata

- `build_restore_workflow()`: `load_source_stage` → `restore_stage` → `save_image_stage`. There's no prompt stage. Sharpening is off by default, since SeedVR2 output is already sharp; the spike can turn it on with `sharpening.existing_restored`.
- `restore_stage` calls `request.backend.restore_image` with the source path and the planned size. It replaces `artifacts.image`.
- `upscale_runner.run_upscale` runs it unchanged: it already takes a workflow and emits one-image batch events. SeedVR2 has one step, so the job card shows load → restore → save without a step bar.
- Filename: `portrait_42_2x.png`, the same rule as refine. Since the method is recorded, the name doesn't need to differ.
- Recorded settings: `workflow: "upscale"`, the existing `source` block, and an `upscale` block with `method: "restore"`, factor, `model` (`seedvr2-3b`), quantize, softness and seed. The source's prompt, model and LoRAs are kept, so Reuse and the viewer still show how the original was made. Refine uploads gain `method: "refine"`; old files without it read as refine.
- Memory: load under `mlx_cache_limit`, call `release_memory()` after the job, and keep one model resident at a time.
- Tests: stage order; the payload's `method`; events for success, quit and failure with a mocked backend.

### 4. Web server and contract

- `_asset_upscale_json` returns `{refine: [...], restore: [...] | null}`. `restore` is `null` when `supports_seedvr2_upscale()` is false. It isn't tied to the image's model, so Ideogram 4 images get a working Restore group while Refine stays disabled with its existing reason.
- `POST /api/upscale` takes `method: "refine" | "restore"`, defaulting to `"refine"` so existing clients keep working. `plan_restore` (next to `plan_upscale` in `web/upscale_api.py`) builds the request and enforces `restore_options`.
- Admission: estimate memory from the safetensors headers (`read_safetensors_totals`) plus the per-megapixel activation cost measured in the spike, and refuse with a clear message when it won't fit, as image jobs do.
- Tests: the gallery JSON on supported and unsupported backends, the route's method default, a refused size, and queuing with the restore workflow.

### 5. Web UI

- The Upscale menu shows two labelled groups, Refine and Restore, each with 2× and 4× and the same size and reason line as today. On Windows and Linux only Refine appears, since `restore` is `null`.
- Keyboard: the existing chord keeps its meaning (Refine 2× or 4×). Restore is reached through the menu. Adding a chord for it is an open question.
- The job card shows "Restore 2×" or "Refine 2×" and the output size.
- Run `make frontend-build` and commit the rebuilt SPA.
- Vitest: both groups render with their sizes, Restore is absent when `restore` is `null`, and choosing a Restore item sends `method: "restore"`.

### 6. Model download

SeedVR2 isn't in the app's model inventory. Add it to `ziv-model list` and the inventory, so the memory badge and the "not downloaded" state work, and download it through the same path as other models. This is the step the video proposal's step 6 asked for. If [model download progress](model-download-progress.md) has shipped by then, it covers this; otherwise the job card shows "Downloading SeedVR2" with no progress bar.

### 7. Docs and changelog

- The Web UI guide's upscale section explains Refine against Restore: when to use each, that Restore is macOS only, and the time and memory from the spike.
- Add the `seedvr2` keys to the configuration reference.
- `docs/development.md`: the new modules in Project Structure, the mflux internals we rely on, and the contract test.
- A `CHANGELOG.md` entry under `[Unreleased]`.
- Move this proposal to `archive/`, update the index, and update the video proposal's Evidence to point at the shipped foundation.

## Alternatives considered

- **Tiled img2img refinement** (the "Ultimate SD Upscale" approach): refine overlapping ~1024–1344 px tiles with the image's own model and feather the seams. It keeps the refine path's style and lifts its memory ceiling. But every tile sees the full prompt, so above about 0.35 denoise subjects repeat across tiles. Seams need tuning per model, and time grows with tile count times steps. Keep it as a later option if Restore is too conservative for painterly sources.
- **Make Restore the only upscale on macOS.** It's simpler, but it removes refinement's "add detail in the model's style" at moderate sizes, and the platforms would behave differently for the same menu item.
- **A config switch choosing which method the menu runs.** It hides one method behind settings. Two visible groups show both.
- **The mflux ControlNet upscaler.** mflux's own README calls SeedVR2 the preferred, faster and more faithful option.
- **GAN upscalers (Real-ESRGAN and similar).** They're fast, but they'd add a dependency, and they smear or invent texture on generated images more than SeedVR2 does.
- **numz's PyTorch SeedVR2 on MPS**, which would also cover CUDA. It brings PyTorch into the macOS stack, which is MLX-only today; see the video proposal.

## Open questions

- **Default model.** Choose 3B bf16, 3B int8 or 7B int8 from the spike. Prefer config over a menu choice, as in the video proposal.
- **In-run Restore.** `--upscale` during generation could offer Restore in place of refine, with a `--upscale-method` flag and a form control. That's worth doing if Restore wins the spike clearly, but as its own follow-up, because it touches the form, the CLI and review before upscaling.
- **A keyboard chord for Restore.** Today's chord stays on Refine. Decide whether Restore gets one, for example Shift with the same keys.
- **Softness per source.** If the spike shows renders and photos want different softness, decide whether that's a config value per family or one default.
