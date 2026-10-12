# Upscale existing videos on macOS with SeedVR2

**Status:** Proposed (2026-10-12)

## Problem

Images in the gallery can be upscaled 2× or 4× after the fact ([Upscale existing images](archive/image-upscale-from-gallery.md)). Videos can't. The only way to get a larger video is to choose the two-stage LTX upscale before generating, and that only works on LTX's own half-size render inside the same run. There's no way to enlarge a video that is already finished, for example after picking the best take from a batch of small, fast renders.

On macOS the model for this is already installed. mflux 0.22, which we pin, ships ByteDance's SeedVR2 (3B and 7B), a one-step diffusion restoration model that is the community default for video upscaling. mflux only exposes it for single images, but its transformer and VAE work on video. The work is a video loop around mflux's modules plus the same app plumbing the image upscale has. It doesn't need a model port or a new dependency.

### Not in scope

- **Windows and Linux.** diffusers 0.41 has no SeedVR2 pipeline, so CUDA would need a port or a new dependency. Gate the feature off there with a capability flag, the same way review before upscaling is gated, and leave CUDA to a later proposal or [platform parity](platform-parity.md).
- **Changing the in-run LTX upscale.** `--upscale` and two-stage generation stay as they are. This proposal covers upscaling a finished file.
- **Frame-rate upscaling** (temporal ×2). LTX's temporal upsampler (`dfr.py`) could do it later, but it's a separate feature.
- **Videos from outside the output folder.** The first version works on gallery videos only, like the image upscale. Uploading arbitrary videos can follow if people ask.
- **A command-line entry point.** Image upscaling is Web UI only today. Add `ziv-video` support only if people ask.

## Evidence

Checked against `main` at 2741388, the installed mflux 0.22.0 (released 2026-10-08, still the latest on PyPI), and the vendored LTX at v0.16.3 (checked 2026-10-12).

### mflux SeedVR2 (installed)

| Check | Result |
|---|---|
| Location | `mflux/models/seedvr2/`: `SeedVR2` class (`variants/upscale/seedvr2.py`), transformer, 3D VAE, latent creator, colour correction, `seedvr2_euler` scheduler |
| Public entry point | `SeedVR2.generate_image(seed, image_path, resolution, softness)`, which takes **one image** |
| Why it is image-only | `SeedVR2LatentCreator.create_noise_latents` hard-codes `shape=(B, 16, 1, H, W)`, and `SeedVR2Util.preprocess_image` reads one file with PIL |
| Transformer | `patch_in.py` unpacks `B, C, T, H, W` and patches along `T`. `window.py` builds attention windows over `(t, h, w)` (`wt = ceil(min(t, 30) / …)`), so it accepts multi-frame latents |
| VAE | `SeedVR2VAE` with `Encoder3D`/`Decoder3D`, `temporal_down_blocks=2` and `temporal_up_blocks=2`. Every conv is `CausalConv3d`, which pads time by repeating the first frame, so a clip of 4n+1 frames encodes to n+1 latent frames. Spatial scale is 8, with 16 latent channels |
| Text conditioning | No prompt. `SeedVR2TextEmbeddings.load_positive()` loads a fixed embedding shipped with mflux |
| Steps | One. `Config(num_inference_steps=1, guidance=1.0, scheduler="seedvr2_euler")` |
| Colour correction | `SeedVR2Util.apply_color_correction(content, style, luminance_weight=0.8)`: Lab colour transfer against the resized input |
| Tiling | `VAEUtil` and `VAETiler` are **image-only**. `encode_image_tiled` takes `(B, C, H, W)` and adds `T=1`. `decode_image_tiled` slices latents spatially but assembles its output as `(H, W, C)`. Neither can be used for clips |
| Model loading | `SeedVR2(quantize=, model_path=, model_config=ModelConfig.seedvr2_3b() \| seedvr2_7b())`. Model state lives in `vae`, `transformer`, `tiling_config`, `bits` and `callbacks` |
| Weights | Hugging Face `numz/SeedVR2_comfyUI`: `seedvr2_ema_3b_fp16.safetensors` or `seedvr2_ema_7b_fp16.safetensors`, plus `ema_vae_fp16.safetensors` |
| Licence | SeedVR and SeedVR2 are Apache-2.0 (ByteDance-Seed model card) |

### Community practice

The numz ComfyUI node (`ComfyUI-SeedVR2_VideoUpscaler`) is the reference implementation most people use:

- Frames go through in batches whose size follows the 4n+1 rule (`batch_size`, default 5; 21 and 33 are common).
- `temporal_overlap` (0–16 frames) is blended with a Hann window, and optional reversed `prepend_frames` steady the opening.
- VAE encode and decode tile spatially (1024 px tiles with 128 px overlap) only when memory runs out.
- Colour correction defaults to `wavelet` in the node and `lab` in its CLI.
- On macOS it runs on PyTorch MPS, not MLX. BlockSwap is turned off there, and the release notes record MPS allocator fixes.

Community MLX weights exist (for example a 7B int8 DiT at about 8.8 GB, against 16.5 GB fp16). Their cards report int8 output nearly identical to fp16 and say int4 degrades this model badly.

### Our code

| Area | Today |
|---|---|
| Image upscale | `build_upscale_workflow()` (load source → upscale → sharpen → save) and `upscale_runner.run_upscale`. Behind it: `POST /api/upscale`, `WebRunner.submit_upscale_job`, and `_asset_upscale_json` in `web/gallery.py`, which returns `null` for videos |
| Size rules | `utils/upscale.py`: `UPSCALE_FACTORS = (2, 4)`, `UpscaleOption`, `upscale.max_megapixels` |
| Video files | `utils/ffmpeg.py` has `require_ffmpeg` and `strip_audio`. `utils/provenance.py` has `embed_mp4_config` and `read_mp4_config` (ffprobe). There's no helper yet that reads frames or writes them |
| Platform flags | `supports_video_audio()` and `supports_video_upscale_review()` in `backends/__init__.py` |
| macOS memory | `memory_mac.release_memory`, `memory_budget_bytes` and `mlx_cache_limit`. `utils/model_memory.read_safetensors_totals` reads weight sizes from the file headers |

## Proposed change

Add **Upscale 2× / 4×** to the asset viewer for videos on macOS, using the same menu, job type and queue as images. The job reads the video in chunks, upscales each chunk with SeedVR2 through mflux's modules, blends the overlaps, writes the frames out through ffmpeg, copies the source audio and frame rate, and records the source and the upscale settings in the new file.

Build it in one worktree, with a commit per step. Each step leaves the app working.

### 0. Spike, with recorded results

Run SeedVR2 on clips directly against mflux, without the app. Write a ~50-line loop that copies `generate_image` but uses a `(1, 3, T, H, W)` clip and `T`-frame noise. Use four test clips: a slow camera move, fast motion, a face close-up, and a generated LTX clip with fine texture. Each is 121 frames at 480p and 540p.

| Gate | Pass when |
|---|---|
| Single-frame parity | With `T=1` and the same seed, the loop's output matches `generate_image` (PSNR ≥ 60 dB), which shows the copied loop is faithful |
| Temporal stability | Mean absolute difference between consecutive output frames, divided by the same measure on the bicubic-upscaled source, is lower for the clip path than for frame-by-frame `generate_image`. A visual check finds no flicker |
| Chunk seams | With chunks of 21 frames and 4 frames of overlap (Hann blend in pixels), a visual check at the seams finds no jump. Record whether sharing noise across the overlap (one noise tensor for the whole timeline, sliced per chunk) is needed |
| Memory and time | Record `mx.get_peak_memory()` and seconds per frame for 3B bf16 and 7B int8. Use outputs of 960p, 1080p and 1440p on the short side, chunks of 5, 13 and 21 frames, on a 32 GB and a 64 GB or larger Mac |
| Stability | No Metal command-buffer timeout at the largest setting that fits in memory |

Then:

- **Parity fails:** stop. The loop isn't reproducing mflux, so fix that before anything else.
- **Clip path is no more stable than per-frame:** the feature isn't worth a video path. Reconsider [the alternatives](#alternatives-considered).
- **1080p output doesn't fit on 32 GB at any chunk size:** step 2 adds spatial tiling before the feature ships, and the size limit in step 4 is set from the measurements.
- **Metal timeout:** split the DiT call or the VAE decode the way `_pre_denoise_flush` does for LTX, and measure again.

Write the numbers and the chosen defaults (chunk size, overlap, model, quantization, size limit) into this proposal's Evidence section before step 1.

### 1. Frame I/O and chunk planning (platform-free)

- `utils/video_frames.py`:
  - `probe_video(path) -> VideoProbe`: a frozen dataclass with width, height, frame count, frame rate, and whether the file has audio. It uses ffprobe.
  - `read_frames(path, start, count)`: decodes `rgb24` from an ffmpeg pipe.
  - `VideoFrameWriter`: a context manager that pipes `rgb24` into ffmpeg (H.264, the same CRF and pixel format as our LTX output) and, on close, muxes the audio from the source (`-map 1:a? -c:a copy`) at the source frame rate.
  - Frames stream chunk by chunk, so the whole video is never held in memory at full size.
- `utils/video_upscale.py`, pure:
  - `plan_chunks(frame_count, chunk_frames, overlap) -> list[ChunkSpan]`: every span is 4n+1 long, and a short final chunk is padded by repeating its last frame. The padding is dropped on write.
  - `blend_weights(overlap)`: the Hann ramp.
  - `video_upscale_output_size(width, height, factor)`: each side times the factor, rounded to 16 px, matching `preprocess_image`.
  - `video_upscale_options(...)`: returns `UpscaleOption`s with the reasons from the size limit.
- Tests:
  - Chunk spans: lengths, overlap, padding, and a single short clip.
  - Blend weights add up to 1 across an overlap.
  - Output sizes, and options with and without the limit.
  - `VideoFrameWriter` with a mocked `subprocess.Popen`: the arguments include the audio map only when the source has audio, and a failed encode removes the partial file.

### 2. macOS SeedVR2 video backend

- `core/video_upscaler.py`: a `VideoUpscaler` Protocol with `load_model(variant, quantize)` and `upscale_clip(model, frames, width, height, seed) -> frames`.
- `backends/video_upscale_mac.py`: the implementation, with mflux imported lazily.
  - `load_model` builds `SeedVR2(quantize=, model_config=ModelConfig.seedvr2_3b() | seedvr2_7b())`.
  - `upscale_clip` resizes the chunk with bicubic, pads to 16 px, encodes it with `model.vae.encode` as `(1, 3, T, H, W)`, and builds the condition with `SeedVR2LatentCreator.create_condition`. It creates `T`-frame noise (sliced from a timeline-wide tensor if the spike says so), takes one transformer step with `SeedVR2TextEmbeddings.load_positive()`, decodes, crops, and applies `SeedVR2Util.apply_color_correction` against the resized input.
  - Add spatial tiling here only if the spike requires it. It would be our own `(B, C, T, H, W)` tiler, because mflux's tilers are image-only.
- `backends/__init__.py`: add `get_video_upscaler()` and `supports_video_file_upscale()`. It returns the mflux upscaler and `True` on macOS, and `None` and `False` elsewhere. No other module branches on platform.
- Config: a new `video_upscale:` block with `model: seedvr2-3b`, `quantize: null`, `chunk_frames`, `overlap_frames` and `max_megapixels`, using the defaults from the spike. Follow the usual precedence rules.
- Contract tests, in a new `tests/test_mflux_seedvr2_contract.py` and following the pattern of `tests/test_ltx_mlx_contract.py`, pin by introspection:
  - `SeedVR2.__init__(quantize, model_path, model_config)` and the instance attributes `vae`, `transformer` and `bits`
  - `SeedVR2LatentCreator.create_condition(encoded_latent)`
  - `SeedVR2TextEmbeddings.load_positive(batch_size)`
  - `SeedVR2Util.apply_color_correction(content, style, luminance_weight)`
  - `ModelConfig.seedvr2_3b` and `seedvr2_7b`
  - that `"seedvr2_euler"` is registered in `mflux.models.common.schedulers`
  - `SeedVR2Transformer.__call__(txt, vid, timestep)`
- Mocked tests: `upscale_clip` sends 5D tensors with the right `T` to the mocked VAE and transformer, crops the padding, and applies colour correction once per chunk.

### 3. Workflow and runner

- `build_video_upscale_workflow()`: `load_video_source_stage` (probe the file and read its recorded settings) → `video_upscale_stage` (go through the chunk plan, blend, write) → `save_video_stage` (mux the audio and embed the metadata).
- `video_upscale_runner.run_video_upscale`: emits the same events as a one-video batch, the way `run_upscale` does for images. Each chunk is one step, so the job card's progress bar moves. Quit is checked between chunks and removes the partial file.
- Metadata: `build_video_config_payload` records `workflow: "upscale"`, a `source` block (path, size, frame count), and an `upscale` block (factor, model, quantize, chunk and overlap). It keeps the source's recorded prompt, model and LoRAs, so Reuse and the viewer still show how the original was made. The `source` block reuses the image upscale's shape, so `_source_with_asset_id` links back to the original.
- Memory: load the upscaler under `mlx_cache_limit`, call `release_memory()` after the job, and keep one model resident at a time, as the two-phase job model requires.
- Tests: stage order; events and statuses for success, quit and failure (with mocked I/O and backend); the payload's `workflow`, `source` and `upscale` keys; and removal of the partial file on quit or error.

### 4. Web server and contract

- `_asset_upscale_json` returns the factor menu for videos when `supports_video_file_upscale()` is true, built from `video_upscale_options`. Elsewhere it stays `null`.
- `POST /api/upscale` accepts video assets. It plans with the video planner and submits a job of type `upscale` with `media_type: "video"` in its context, so the queue and job card treat it like an image upscale.
- Admission check: estimate memory from the safetensors headers (`read_safetensors_totals`) plus the activation cost per chunk measured in the spike, and refuse with a clear message when it won't fit, as image jobs do.
- Tests:
  - The gallery JSON for a video on supported and unsupported backends.
  - The route rejects factors over the limit.
  - The route queues a job with the video workflow.
  - The admission check refuses when memory is short.

### 5. Web UI

- The asset viewer's existing Upscale menu shows up for videos too, with the same 2×/4× items, sizes and disabled reasons. This adds no new controls or tabs.
- The job card shows the chunk progress and the output size (`meta`), like an image upscale.
- Run `make frontend-build` and commit the rebuilt SPA.
- Vitest: the menu renders for a video asset that carries `upscale.factors` and is absent when `upscale` is `null`.

### 6. Model download

SeedVR2 weights aren't in the app's model inventory. The first video upscale would trigger mflux's own Hugging Face download (3B is about 7 GB with its VAE).

Show it as a normal model in `ziv-model list` and in the inventory, so the memory badge and the "not downloaded" state work. Download it through the same path as other models. If [model download progress](model-download-progress.md) has shipped by then, it covers this; otherwise the job card shows "Downloading SeedVR2" with no progress bar.

### 7. Docs and changelog

- A "Upscale a video" section in the Web UI guide, next to the image upscale. It covers what the model does, the time per frame and memory from the spike, the fact that it's macOS only, and that audio and frame rate are kept.
- Add the `video_upscale` keys to the configuration reference.
- `docs/development.md`: the new modules in Project Structure, a paragraph on the mflux internals we rely on (next to the vendored LTX list), and the contract test.
- A `CHANGELOG.md` entry under `[Unreleased]`.
- Move this proposal to `archive/` and update the index.

## Alternatives considered

- **LTX latent upscale or Pixel Spatial Upscaler IC-LoRA on the finished video.** All the pieces are vendored: `LatentUpsampler` spatial ×2, `encode_video_tensor` in `retake.py`, and IC-LoRA references with `reference_downscale_factor` in `ic_lora.py`. But it loads the 22B LTX transformer and Gemma to upscale, needs a prompt and several steps, follows LTX's 8n+1 frame and size rules, and changes detail more on footage LTX didn't make. Its IC-LoRA weights are gated. Keep it as an option if SeedVR2 is weak on LTX renders specifically.
- **Frame-by-frame `generate_image`.** No video loop to write, but it flickers, because the model never sees neighbouring frames. The spike uses it as the baseline to beat.
- **Contribute `generate_video` to mflux upstream.** This would remove our dependence on mflux internals and is worth offering once the loop works. Don't wait for it, though: the feature shouldn't depend on an upstream release.
- **numz's PyTorch implementation on MPS.** It's the most battle-tested, but it brings PyTorch MPS into a stack that is MLX-only on macOS, and it has known MPS allocator problems.
- **FlashVSR or DOVE.** Both are faster or newer in papers, but neither has an MLX port, and FlashVSR's speed depends on CUDA sparse-attention kernels.
- **GAN upscalers (Real-ESRGAN and similar).** Cheap and simple, but they work per frame and flicker.

## Open questions

- **Default model.** Choose 3B bf16 or 7B int8 from the spike's quality and memory results. Should the other be a config option only, or a choice in the menu? The menu would be a new control, so prefer config unless the quality gap is large.
- **4×.** At 4× even a 480p source becomes about 1920 px on the short side. Ship 2× only if 4× doesn't fit typical machines, and show the size-limit reason for 4× the way images do.
- **Long videos.** Time grows linearly with frame count. Decide whether the job card needs an estimated time remaining based on the first chunk.
- **Colour correction.** We'd use mflux's Lab transfer, which is what mflux ships. Wavelet is the node's default, so check in the spike whether it looks better on video.
