# Feature parity between macOS and Windows/Linux

**Status:** Proposed (2026-10-09)

## Problem

Most development has happened on macOS. The shared app is the same everywhere: the CLI controls, the Web UI (queue, gallery, prompt builder, config, file picker), text-to-image for Z-Image, FLUX.2 Klein and Krea 2, q4/q8 with stored quants, image memory badges, LoRAs and LoKr, live image previews, video text-to-video and image-to-video, video LoRAs and the prompt enhancer. [Quantization parity on CUDA](cuda-quant-parity.md) closed the image quantization gap.

What's left are the features that work on one platform and are missing or behave differently on the other. This proposal assumes the three dependency proposals have shipped ([shared](deps-shared.md), [CUDA](deps-cuda.md), [macOS](deps-macos.md)): diffusers 0.41.0 on Windows and Linux, mflux 0.22 and LTX MLX v0.16.3 on macOS. Several gaps get wider with them. macOS moves to upstream's two-stage video upscale and gains a float16 compute option and low-RAM video streaming, while the CUDA video backend doesn't change.

Missing on Windows and Linux:

1. **FLUX.2 Klein can't refine an existing image.** Reference images, `--upscale` and gallery Upscale fail with the default model `klein9b`. `DiffusersBackend.image_to_image` (`backends/image_win.py`) builds `AutoPipelineForImage2Image.from_pipe(model)` and passes `strength=`. diffusers maps the `flux2-klein` family to `Flux2KleinPipeline` for img2img, and its `__call__` has no `strength` parameter and no `**kwargs`, so the call raises `TypeError`. This is unchanged in 0.41.0. `flux2_klein` doesn't set `supports_img2img: false`, so the Web UI and CLI offer the feature. Z-Image (`ZImageImg2ImgPipeline`) and Krea 2 (our own latent path) work.
2. **Videos have no audio.** `LTX2PipelineOutput` returns `frames` and `audio`, but `video_diffusers._export_video` writes only the frames with `export_to_video`. `--audio`, which is on by default, does nothing.
3. **Video upscale isn't the two-stage upscale.** On macOS, `--upscale 2` generates at half resolution, upsamples the latents 2× and refines them at full resolution. On CUDA, stage 1 runs at the full requested size. `_upscale_video` then passes that size to `LTX2LatentUpsamplePipeline`, which reads `height`/`width` as the size of its *input* video. So the output is about 2× the requested size, nothing refines it, and stage 2 shows up as one jump in progress.
4. **No float16 compute option.** `deps-macos` adds `compute_precision` (config key and `--compute-precision`) for Z-Image and FLUX.2 Klein on mflux only. `DiffusersBackend.load_model` already maps `"float16"` to `torch.float16`, but `image_model_loader._PRECISION` fixes every load at `"bfloat16"`.
5. **No quantized video.** macOS has `ltx-4` and `ltx-8` (MLX q4/q8). CUDA has only `ltx-2.3` in bfloat16.
6. **No low-RAM video streaming.** `--low-memory` on CUDA means `enable_model_cpu_offload()`, which moves each whole component onto the GPU in turn. macOS gets DiT block streaming (`low_ram_streaming`). CUDA image models already stream block by block (`image_win._stream_to_gpu`), but video doesn't.
7. **No video memory-fit badges.** `model_status.describe_model_status` returns `memory_fit: None` for CUDA video ("CUDA video is not estimated").

Different defaults or behaviour for the same feature:

8. **Z-Image's default scheduler differs.** `model_presets.zimage.default_scheduler` is `beta` for mflux and `null` (FlowMatch Euler) for diffusers, so the same settings look different on each platform. The diffusers backend already supports `beta` (`FlowMatchEulerDiscreteScheduler(use_beta_sigmas=True)`).
9. **Missing ffmpeg.** On macOS, `utils/ffmpeg.ensure_ffmpeg` offers to install it with Homebrew or MacPorts. Elsewhere it prints a URL and exits.
10. **The prompt enhancer download is 9 GB on Windows and Linux**, against 2.4 GB on macOS. `coder3101/Qwen3.5-4B-heretic` is the full bfloat16 repo, quantized to NF4 at load.

### Not in scope

- **Ideogram 4** is mflux-only, and diffusers has no Ideogram 4 pipeline. It stays macOS-only, as `model_aliases.ideo` already says.
- **FLUX.1 and FLUX.2-dev** aren't supported models. The diffusers backend happens to load them through `AutoPipeline`, but that isn't a feature to match on macOS.
- **The GPU floor from `deps-cuda`** (RTX 20-series and newer, driver R580 or newer) is a deliberate trade-off, not a gap.
- **LTX-2.5** was a follow-up in both platform proposals. Decided 2026-10-10: LTX-2.5 not planned.

## Evidence

Checked against `main` at v0.13.0b17 and the diffusers 0.41.0 wheel from PyPI (2026-10-09).

| Check | Result |
|---|---|
| `AUTO_IMAGE2IMAGE_PIPELINES_MAPPING` in 0.41.0 | `flux2-klein` → `Flux2KleinPipeline`, `z-image` → `ZImageImg2ImgPipeline`, no `krea2` entry |
| `Flux2KleinPipeline.__call__` | No `strength`, no `**kwargs` (0.40.0 and 0.41.0) |
| `Flux2KleinInpaintPipeline` (0.41.0) | Takes `image`, `mask_image`, `image_reference` and `strength` |
| `LTX2PipelineOutput` | Has `frames` and `audio` |
| `diffusers.utils.export_utils.encode_video` (0.41.0) | `encode_video(video, fps, output_path, audio=None, audio_sample_rate=None, ...)`, through PyAV |
| `LTX2LatentUpsamplePipeline.__call__` | `height`/`width` are "the height in pixels of the input video (not the generated video, which will have a larger resolution)" |
| `video_mac._generate_upscaled` (to be replaced by `DistilledPipeline.generate_two_stage`) | Half-res distilled stage 1, 2× `LatentUpsampler`, distilled refine with `STAGE_2_SIGMAS` |

## Proposed change

Each step is its own PR, in this order. Steps 1, 2 and 8 are small and fix what users hit first. Steps 3, 5, 6 and 7 are video work and fit best after `deps-macos` PR 2, so both platforms move to the same upscale flow at once.

### 1. FLUX.2 Klein img2img on CUDA

- Give Klein the same treatment as Krea 2: a `_flux2_klein_image_to_image` path in `image_win.py` that VAE-encodes the image, noises it to the first sigma of the tail of the schedule, and runs `Flux2KleinPipeline` with `latents=` and `sigmas=`. Reuse the Krea 2 helpers where they're shape-agnostic (`_img2img_sigmas`, the sigma lookup), and move them to a small shared module if both families need them.
- The alternative is `Flux2KleinInpaintPipeline` with an all-white mask. It's an official pipeline with `strength`. But it runs the inpaint blending every step for nothing, and it doesn't exist in 0.40. Try it in the spike only as a reference for output quality.
- Until the fix lands, set `supports_img2img: false` and `supports_upscale: false` for `flux2_klein` on diffusers. That needs a backend-scoped capability, because the flags in `model_presets` apply to every platform today. If a per-backend override isn't worth adding for a short-lived gap, ship the fix directly.
- Tests: a mocked pipeline that rejects unknown kwargs (`inspect.signature` with no `**kwargs`), so a call with `strength=` fails the test the way it fails at runtime. Check the latents and sigmas handed to the pipeline for a given strength.

### 2. Audio in CUDA video

- In `video_diffusers`, read `result.audio` alongside the frames. When it's present, write the file with `diffusers.utils.export_utils.encode_video(frames, fps, output_path, audio=..., audio_sample_rate=...)`. The sample rate comes from the pipeline (the vocoder or audio VAE config; confirm the attribute in the spike). Without audio, keep `export_to_video`.
- `strip_audio_stage` already handles `--no-audio` for both platforms, so nothing changes in the workflow.
- `encode_video` uses PyAV. Confirm that `av` is installed with diffusers on Windows and Linux, or add `av; sys_platform != 'darwin'` to `dependencies`.
- Tests: the mocked pipeline returns `audio`; assert it reaches the encoder with the sample rate, and that a result without audio falls back to frames only.

### 3. Two-stage video upscale on CUDA

Match the macOS flow (half-res stage 1, 2× latent upsample, refine at full res):

- Stage 1: run the text-to-video or image-to-video pipeline at `width // 2` × `height // 2` with `output_type="latent"`. `video_cli` already aligns upscale sizes to 64, so the halves stay 32-aligned.
- Upsample: pass the latents to `LTX2LatentUpsamplePipeline` (`latents=`, with the half size as `height`/`width`), returning latents.
- Stage 2: denoise the upsampled latents at full size with the same distilled transformer over the stage-2 sigmas, then decode. In diffusers 0.41 that's the `LTX2Pipeline` call with `latents=` and `sigmas=` (the same trick the image backend uses for Krea 2), or `LTX2DFRPipeline` / `LTX2DFRTemporalRefinePipeline` if a spike shows they do this directly. The spike decides between them.
- Progress: stage 1 and stage 2 report real steps under the existing `video_upscale_stage_1` / `video_upscale_stage_2` phases, with one total, as on macOS.
- Confirm that the configured `ltx-2.3` repo (`dg845/LTX-2.3-Diffusers`) ships a `latent_upsampler` component. If not, load it from a separate repo, configured under `video_model_presets.ltx.diffusers`.
- Tests: mocked pipelines; assert the stage-1 size is half the target, the upsampler gets the half size, stage 2 runs at the target size, and the step totals.

### 4. `compute_precision` on CUDA

- Pass the resolved `compute_precision` to `load_model` instead of the fixed `_PRECISION`, on both backends. `float16` maps to `torch.float16` on diffusers, which `dtype_map` already supports. Keep `bfloat16` as the default.
- Apply it to the same families as on macOS (Z-Image and FLUX.2 Klein) and warn and ignore elsewhere, so the key means the same thing on both platforms.
- Watch for float16 overflow: on CUDA the VAE and text encoder share the pipeline dtype. If tests show NaNs or black images, keep the VAE (and text encoder, if needed) in bfloat16 and apply float16 to the transformer only.
- Ship it together with `deps-macos` PR 1, or right after, so the config key and the `--compute-precision` flag never exist on one platform only.
- Tests: the resolved precision reaches `load_model`, and unsupported families warn.

### 5. Quantized video on CUDA

- Add `--quantize 8` (FP8) and `--quantize 4` (NF4) for the LTX transformer, reusing `backends/image_win_quant.py`: `load_fp8_components` for q8, which streams like bfloat16, and `load_nf4_components` for q4. The Gemma text encoder gets the same treatment it gets for images.
- Store the quantized copy with the existing stored-quant flow (`<name>@q4` / `<name>@q8`), so the second run loads without quantizing.
- Expose the levels in the Web UI's video settings the way image models do.
- Sizes and quality need measuring: run t2v and i2v at q8 and q4 against bfloat16 with the same seed before turning it on.
- This was the prerequisite for LTX-2.5 on CUDA. Decided 2026-10-10: LTX-2.5 not planned.

### 6. Low-RAM video streaming on CUDA

- With `--low-memory`, apply block-level group offloading to the LTX transformer (as `image_win._stream_to_gpu` does for image transformers) instead of `enable_model_cpu_offload()`. The text encoder uses the same leaf-level rule (`_text_encoder_offload_type`).
- Measure peak VRAM and seconds per step for both modes on a 12 GB and a 24 GB card. Keep CPU offload if streaming isn't a clear win.

### 7. Video memory-fit badges on CUDA

- Add `estimate_cuda_video_memory` to `utils/model_memory.py`, mirroring how step 6 loads the model (streamed transformer, resident VAE and upsampler), for unquantized and each level from step 5.
- Use it in `model_status.describe_model_status` for CUDA video, in the same `discrete` shape as image models, with the existing `VIDEO_ESTIMATE_NOTE`.

### 8. Same Z-Image default scheduler

- Set `model_presets.zimage.default_scheduler.diffusers` to `beta`. The backend already implements it.
- Check with the same prompts and seeds on CUDA: if `beta` doesn't look better with diffusers' Z-Image, keep the default and note the difference in the model guide instead.

### 9. Offer to install ffmpeg on Windows and Linux

- Extend `ensure_ffmpeg` the way macOS works: on Windows offer `winget install ffmpeg` when `winget` is on `PATH`. On Linux, offer the distribution's package manager (`apt`, `dnf`, `pacman`) when present, with `sudo`.
- After installing on Windows, a new terminal may be needed before `PATH` picks up ffmpeg. Say so instead of failing with "still not found".

### 10. Smaller prompt-enhancer download on CUDA

- Look for a public, pinned NF4 or other pre-quantized copy of the same model that transformers loads directly. If one exists, point `prompt_enhancer.model.win32` / `.linux` at it and update `download_size_label`.
- Otherwise keep the 9 GB repo. It's a one-time download, and the label already tells users. Lowest priority.

## Verification

- Unit tests with mocked pipelines for each step, as above. CI runs on macOS only, so the CUDA steps also need a manual run on a Linux or Windows machine with an NVIDIA GPU. Record the GPU, driver, torch and diffusers versions in each PR.
- Step 1: `klein4b` and `klein9b` with a reference image, `--upscale 2`, and gallery Upscale, unquantized and at q4/q8.
- Steps 2, 3, 5, 6: `ltx-2.3` t2v and i2v, with and without `--audio`, with `--upscale 2`, at each quantize level, with `--low-memory` on and off. Compare against the same seeds on a Mac for overall look and audio sync, not pixels.
- Step 4: `zit` and `klein9b` at `float16` and `bfloat16`, with seconds per step and a check for NaN or black output.
- Step 7: compare the badge's estimate with the peak VRAM in the job log.

## Docs and changelog

- `CHANGELOG.md` `[Unreleased]`, one line per step as it ships, e.g. "FLUX.2 Klein reference images and upscale now work on Windows and Linux" and "Videos made on Windows and Linux now have sound".
- `README.md` platform table and `docs/getting-started.md`: drop the Windows/Linux video caveats as they close, and add q4/q8 for `ltx-2.3`.
- Model guide: `compute_precision` on both platforms, and the Z-Image scheduler default.

## Alternatives considered

- **One big parity PR.** Rejected: the video steps need hardware to verify and would hold back the small fixes (1, 2, 8).
- **Turn off Klein img2img on CUDA permanently.** Rejected: Klein is the default model, and refining images is a core feature.
- **Use only `LTX2DFRPipeline` for the CUDA upscale.** Possible, but it's new in 0.41 and its output isn't compared with the macOS two-stage flow yet. The spike in step 3 decides.
- **Keep `compute_precision` macOS-only and document it.** Rejected: a config key and CLI flag that mean nothing on one platform is the kind of drift this proposal is meant to remove.

## Open questions

- Which attribute holds the LTX-2 audio sample rate in diffusers 0.41 (step 2).
- Whether `dg845/LTX-2.3-Diffusers` includes the latent upsampler, and whether `LTX2Pipeline` accepts `latents=` plus `sigmas=` for the refine stage (step 3).
- Whether the per-backend capability override in step 1 is worth adding, or the fix should ship directly.
- How much quality q4 costs for LTX on CUDA, and whether only q8 should be offered (step 5).
