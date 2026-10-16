# Feature parity between macOS and Windows/Linux

**Status:** In progress (re-audited 2026-10-15 at v0.13.0b20). Steps 2 (audio in CUDA video), 3 (two-stage video upscale on CUDA, except review), 5 (quantized, streamed video on CUDA, which absorbed step 6) and 11 (extend a video on CUDA) are done. Steps 1, 3's review, 7, 8, 9, 10, 12 and 13 are still open. The re-audit added steps 12 (SeedVR2 upscale) and 13 (Ideogram 4), both accepted on 2026-10-15, and widened step 7 to cover the per-clip memory check.

## Where things stand

Re-audited against `main` at v0.13.0b20 (diffusers 0.41.0, mflux 0.22.0, torch 2.14.1, transformers 5.19.0, bitsandbytes 0.50.2). The check covered the capability functions in `backends/__init__.py`, every `sys.platform` branch outside it, the `config.yaml` per-platform keys, both backends' entry points, and the installed diffusers wheel.

| Feature | macOS | Windows / Linux | Step |
|---|---|---|---|
| Text to image: Z-Image, FLUX.2 Klein, Krea 2 | yes | yes | |
| Ideogram 4 | yes | no; refused in `load_model` | 13 |
| q4 / q8 with stored quants, image memory badges | yes | yes | |
| LoRAs and LoKr (image) | yes | yes | |
| Live image previews | yes | yes | |
| Image to image, `--upscale`, gallery creative upscale: Z-Image, Krea 2 | yes | yes | |
| Same, FLUX.2 Klein (the default model) | yes | **fails with `TypeError`** | 1 |
| Plain upscale of an existing image (SeedVR2) | yes | no; creative items only | 12 |
| Review before upscaling (image) | yes | yes | |
| Z-Image default scheduler | `beta` | FlowMatch Euler | 8 |
| Video t2v and i2v, `ltx-4` / `ltx-8`, LoRAs | yes | yes (LoRAs on NF4 not run on hardware) | 5 |
| Video audio | yes | yes | 2 |
| Two-stage video upscale | yes | yes | 3 |
| Review before upscaling (video) | yes | no; `--review` refused, option hidden | 3 |
| Extend a video, with its sound | yes | yes | 11 |
| Video model memory badge | yes | no (`memory_fit: None`) | 7 |
| Per-clip memory check (Duration note, Extend note, CLI warning) | yes | no (`clip_memory_fit` returns `None` for a discrete GPU) | 7 |
| Load-time memory check for video | n/a | yes (`estimate_ltx_cuda_system_memory`) | 5 |
| Prompt enhancer | yes, 2.4 GB download | yes, 9 GB download | 10 |
| Offer to install a missing ffmpeg | Homebrew, MacPorts | prints a URL and exits | 9 |
| Upscale an existing video | proposed, macOS only ([video-upscale-existing](video-upscale-existing.md)) | not planned yet | when it ships |

Same on both, nothing to do: the CLI, the Web UI (queue, gallery, prompt builder, config, file picker, Models page), `ziv-model` and the LoRA import, size presets, accelerator memory release, and the LTX-2 LoRA format (diffusers 0.41's `_convert_non_diffusers_ltx2_lora_to_diffusers` converts the official `diffusion_model.*` layout that macOS loads).

## Problem

Most development has happened on macOS. [Quantization parity on CUDA](archive/cuda-quant-parity.md) closed the image quantization gap, and steps 2, 3, 5 and 11 brought CUDA video level with macOS for generation, audio, upscale and extend. The table above lists what's left: features that work on one platform and are missing or behave differently on the other. This proposal assumes the three dependency proposals have shipped ([shared](archive/deps-shared.md), [CUDA](archive/deps-cuda.md), [macOS](archive/deps-macos.md)): diffusers 0.41.0 on Windows and Linux, mflux 0.22 and LTX MLX v0.16.3 on macOS.

Missing on Windows and Linux:

1. **FLUX.2 Klein can't refine an existing image.** Reference images, `--upscale` and gallery Upscale fail with the default model `klein9b`. `DiffusersBackend.image_to_image` (`backends/cuda/image/backend.py`) builds `AutoPipelineForImage2Image.from_pipe(model)` and passes `strength=`. diffusers maps the `flux2-klein` family to `Flux2KleinPipeline` for img2img, and its `__call__` has no `strength` parameter and no `**kwargs`, so the call raises `TypeError`. This is unchanged in 0.41.0. `flux2_klein` doesn't set `supports_img2img: false`, so the Web UI and CLI offer the feature. Z-Image (`ZImageImg2ImgPipeline`) and Krea 2 (our own latent path) work.
2. ~~**Videos have no audio.**~~ *Done (2026-10-14), see step 2.* `LTX2PipelineOutput` returns `frames` and `audio`, but `video_diffusers._export_video` writes only the frames with `export_to_video`. `--audio`, which is on by default, does nothing.
3. ~~**Video upscale isn't the two-stage upscale.**~~ *Done (2026-10-14), see step 3.* On macOS, `--upscale 2` generates at half resolution, upsamples the latents 2× and refines them at full resolution. On CUDA, stage 1 runs at the full requested size. `_upscale_video` then passes that size to `LTX2LatentUpsamplePipeline`, which reads `height`/`width` as the size of its *input* video. So the output is about 2× the requested size, nothing refines it, and stage 2 shows up as one jump in progress.
4. ~~**No float16 compute option.**~~ *Dropped (2026-10-11).* `deps-macos` proposed a `compute_precision` option for mflux, but it gave 0–4% on an M5 and was removed, so macOS has nothing to match. See [macos-speed.md](macos-speed.md) and [deps-macos.md](archive/deps-macos.md#what-changed-from-the-plan).
5. ~~**No quantized video, and video doesn't fit a 32 GB machine.**~~ *Done (2026-10-14), see step 5.* macOS has `ltx-4` and `ltx-8` (MLX q4/q8). CUDA had only `ltx-2.3` in bfloat16, which holds about 66 GB in system memory, and runs an older transformer (distilled 1.0) than macOS (1.1).
6. ~~**No low-RAM video streaming.**~~ *Merged into step 5.* `--low-memory` on CUDA meant `enable_model_cpu_offload()`, which moves each whole component onto the GPU in turn.
7. **No video memory estimates.** `model_status.describe_model_status` returns `memory_fit: None` for CUDA video ("CUDA video is not estimated"). The per-clip check that shipped on macOS later (the note under **Duration (s)**, the Extend Video note, `POST /api/video/memory-fit` and the `ziv-video` warning) is unified-memory only too: `clip_memory_fit` returns `None` when the budget has a separate system memory. On CUDA the clip length is what runs out first (XL 15 s fails on a 10 GB GPU, step 5), so this check matters more there than the badge.

Also missing on Windows and Linux, found in the 2026-10-15 re-audit:

- **Review before upscaling for video.** Planned as part of step 3 but not built: `supports_video_upscale_review()` is true only for LTX MLX, `video_cli` and the Web UI refuse `review`, and `_refuse_review` in `backends/cuda/video/backend.py` raises.
- **SeedVR2, the plain upscale of existing images** ([shipped on macOS](archive/image-upscale-seedvr2-mac.md)). `supports_seedvr2()` is macOS only, so the viewer offers only the creative items, and the X chord runs a creative upscale instead of a plain one. See step 12.
- **Ideogram 4.** This was out of scope because "diffusers has no Ideogram 4 pipeline", but diffusers 0.41.0 does ship one. See step 13.

~~Extending a video is also macOS only.~~ *Done (2026-10-15), see step 11.* `supports_video_extend()` was false on CUDA, which hid the Extend Video workflow and refused `--extend`.

Different defaults or behaviour for the same feature:

8. ~~**Z-Image's default scheduler differs.**~~ *Done (2026-10-16), see step 8.* `model_presets.zimage.default_scheduler` is `beta` for mflux and `null` (FlowMatch Euler) for diffusers, so the same settings look different on each platform. The diffusers backend already supports `beta` (`FlowMatchEulerDiscreteScheduler(use_beta_sigmas=True)`).
9. **Missing ffmpeg.** On macOS, `utils/ffmpeg.ensure_ffmpeg` offers to install it with Homebrew or MacPorts. Elsewhere it prints a URL and exits.
10. **The prompt enhancer download is 9 GB on Windows and Linux**, against 2.4 GB on macOS. `coder3101/Qwen3.5-4B-heretic` is the full bfloat16 repo, quantized to NF4 at load.

### Not in scope

- ~~**Ideogram 4** is mflux-only, and diffusers has no Ideogram 4 pipeline.~~ *In scope since 2026-10-15, as step 13:* diffusers 0.41.0 has `Ideogram4Pipeline`.
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
| `video_mac._generate_upscaled` (replaced by `DistilledPipeline.generate_and_save`) | Half-res distilled stage 1, 2× `LatentUpsampler`, distilled refine with `STAGE_2_SIGMAS` |
| `LTX2Pipeline.__call__` guidance defaults (checked 2026-10-14) | 0.38.0: `guidance_scale=4.0`, `stg_scale=0.0`, `modality_scale=1.0`. 0.40.0 and 0.41.0: `guidance_scale=3.0`, `stg_scale=1.0`, `modality_scale=3.0`, `audio_guidance_scale=7.0`, spatio-temporal guidance on block 28. Passing only `guidance_scale=1.0`, as the distilled model card does, therefore still runs extra guidance passes on 0.41 |
| `LTX2Pipeline.__call__` (0.41.0) | Accepts `latents=`, `audio_latents=`, `sigmas=` and `noise_scale=`, so the refine stage needs no DFR pipeline |
| LTX-2.3 on CUDA, by component (checked 2026-10-14) | Transformer 18.99B parameters (38 GB bfloat16), Gemma 3 text encoder 12.19B (46 GB on disk in float32), text connectors 3.17B (6.3 GB), VAE, audio VAE and vocoder 0.9B. With `enable_model_cpu_offload()` all of it sits in system memory (about 66 GB) and each whole component moves to the GPU in turn, so a 32 GB / 10 GB machine is killed for lack of memory while loading (systemd-oomd, 2026-10-14) and could never fit the 36 GB transformer on its GPU anyway |
| `diffusers/LTX-2.3-Distilled-Diffusers` (checked 2026-10-14) | A copy of `dg845/LTX-2.3-Distilled-Diffusers`, converted 2026-03-14 from distilled 1.0. Lightricks published `ltx-2.3-22b-distilled-1.1.safetensors` on 2026-04-13; the vendored MLX pipeline prefers 1.1 |
| Distilled 1.0 vs 1.1 (tensors compared by HTTP range reads, 2026-10-14) | Transformer blocks differ; text connectors (`video_embeddings_connector`, `audio_embeddings_connector`, `text_embedding_projection`), VAE and vocoder are byte-identical. Only the transformer has to come from the 1.1 file |
| diffusers single-file conversion (0.41.0) | `convert_ltx2_transformer_to_diffusers` renames the checkpoint's `model.diffusion_model.*` keys and drops the connectors |
| bitsandbytes NF4 under diffusers group offloading (0.41.0, bitsandbytes 0.50.2) | Fails with an illegal memory access: the hooks move `param.data` and leave the quantization state on the CPU, even when it is moved to the GPU first. Swapping whole `Params4bit` objects around a block's forward pass works and matches the resident output exactly |
| `dg845/LTX-2.3-Diffusers` | Not distilled and ships no `latent_upsampler` component. The distilled diffusers repo has none either; the x2 upsampler is a single file in `Lightricks/LTX-2.3` |

Re-checked 2026-10-15 at v0.13.0b20, against the installed diffusers 0.41.0 and the Hugging Face API:

| Check | Result |
|---|---|
| `Flux2KleinPipeline.__call__` | Still no `strength`. `AUTO_IMAGE2IMAGE_PIPELINES_MAPPING` still maps `flux2-klein` to it, and `DiffusersBackend.image_to_image` still passes `strength=` through `AutoPipelineForImage2Image.from_pipe`. `flux2_klein` still has no `supports_img2img: false` |
| `diffusers.Ideogram4Pipeline` (0.41.0) | Exists, with `Ideogram4Transformer2DModel`, `Ideogram4ModularPipeline` and `Ideogram4PromptEnhancerHead`. Components: `scheduler`, `vae`, `text_encoder` (`Qwen3VLModel`), `tokenizer`, `transformer`, `unconditional_transformer`, `prompt_enhancer_head`. `__call__` takes `guidance_scale`, a per-step `guidance_schedule` (default 45 × 7.0 then 3 × 3.0), `mu`/`std` for the timestep shift, `latents=` and `callback_on_step_end`. It takes no `image` and no `sigmas` |
| `ideogram-ai/ideogram-4-fp8` (the macOS `ideo` repo) | Already in diffusers layout (`model_index.json`: `Ideogram4Pipeline`), gated, 27.5 GB |
| `ideogram-ai/ideogram-4-nf4-diffusers` | Official, gated, 16.1 GB, created 2026-06-04. Transformer and text encoder are bitsandbytes NF4 (`quantization_config` in their `config.json`) |
| SeedVR2 in diffusers 0.41.0 | No pipeline. No usable diffusers-format repo on the Hub either (`TrgTuan10/SeedVR2_diffusers` is empty); the PyTorch weights are `ByteDance-Seed/SeedVR2-3B` / `-7B` and `numz/SeedVR2_comfyUI` |
| `clip_memory_fit` (`utils/model_memory.py`) | Returns `None` whenever `budget.system_bytes` is set, so on CUDA there is no Duration note, no Extend note and no CLI warning |
| `describe_model_status` (`web/model_status.py`) | `memory_fit` is `None` for CUDA video |
| `ensure_ffmpeg` (`utils/ffmpeg.py`) | Still offers Homebrew or MacPorts on macOS only |
| `model_presets.zimage.default_scheduler` | Still `mflux: beta`, `diffusers: null` |
| Pre-quantized `Qwen3.5-4B-heretic` for transformers | None among the 60 Hub results that aren't MLX or GGUF. The CUDA enhancer still downloads the 9 GB bfloat16 repo and quantizes it to NF4 when it loads |

## Proposed change

Each step is its own PR. Order for what's left:

1. Step 1 (Klein img2img). It's a hard failure on the default model, so it goes first.
2. Step 9 (ffmpeg install). Small. Step 8 (scheduler default) is done.
3. Step 7 (video memory estimates). It needs the clip-length measurements from step 5 and a few more at q8.
4. Step 3's review before upscaling.
5. Steps 13 and 12 (Ideogram 4, SeedVR2). Both are accepted. Each starts with a spike that picks the approach: which implementation for SeedVR2, which repo for Ideogram 4.
6. Step 10 (enhancer download). Lowest priority.

Step 4 was dropped. The step numbers stay as they are so references keep working.

### 1. FLUX.2 Klein img2img on CUDA

- Give Klein the same treatment as Krea 2: a `_flux2_klein_image_to_image` path in `image_win.py` that VAE-encodes the image, noises it to the first sigma of the tail of the schedule, and runs `Flux2KleinPipeline` with `latents=` and `sigmas=`. Reuse the Krea 2 helpers where they're shape-agnostic (`_img2img_sigmas`, the sigma lookup), and move them to a small shared module if both families need them.
- The alternative is `Flux2KleinInpaintPipeline` with an all-white mask. It's an official pipeline with `strength`. But it runs the inpaint blending every step for nothing, and it doesn't exist in 0.40. Try it in the spike only as a reference for output quality.
- Until the fix lands, set `supports_img2img: false` and `supports_upscale: false` for `flux2_klein` on diffusers. That needs a backend-scoped capability, because the flags in `model_presets` apply to every platform today. If a per-backend override isn't worth adding for a short-lived gap, ship the fix directly.
- Tests: a mocked pipeline that rejects unknown kwargs (`inspect.signature` with no `**kwargs`), so a call with `strength=` fails the test the way it fails at runtime. Check the latents and sigmas handed to the pipeline for a given strength.

### 2. Audio in CUDA video

**Done (2026-10-14).** `_export_video` muxes the first waveform of `result.audio` with `diffusers.utils.encode_video` at `pipeline.vocoder.config.output_sampling_rate` (48 kHz for the distilled repo); every clip, with or without audio, is written by that one encoder. The two-stage upscale exports the refine stage's audio. `av` is a dependency on Windows and Linux, and `supports_video_audio()` is true there. Still open: the manual run on an NVIDIA GPU, listening for sync against the same seed on a Mac.

Original plan:

- In `video_diffusers`, read `result.audio` alongside the frames. When it's present, write the file with `diffusers.utils.export_utils.encode_video(frames, fps, output_path, audio=..., audio_sample_rate=...)`. The sample rate comes from the pipeline (the vocoder or audio VAE config; confirm the attribute in the spike). Without audio, keep `export_to_video`.
- `strip_audio_stage` already handles `--no-audio` for both platforms, so nothing changes in the workflow.
- `encode_video` uses PyAV. Confirm that `av` is installed with diffusers on Windows and Linux, or add `av; sys_platform != 'darwin'` to `dependencies`.
- Tests: the mocked pipeline returns `audio`; assert it reaches the encoder with the sample rate, and that a result without audio falls back to frames only.

### 3. Two-stage video upscale on CUDA

**Done (2026-10-14), except review before upscaling** (the last bullet of the original plan below), which is still open. The default repo is now the distilled `diffusers/LTX-2.3-Distilled-Diffusers`, the upsampler is loaded from `Lightricks/LTX-2.3` (`video_model_presets.ltx.diffusers.latent_upsampler`), stage 1 runs at half size, and the refine runs through `LTX2Pipeline(latents=, sigmas=)` on the same pipeline, so DFR was not needed. Every diffusers call now passes the distilled sigmas and `guidance_scale=1.0`. Still open: the manual run on an NVIDIA GPU (t2v and i2v, `--low-memory` on and off, LoRA, Cancel during stage 2). The guidance question is answered in step 5: the 0.41 defaults cost speed and change the look, and are now turned off.

Original plan:

Match the macOS flow (half-res stage 1, 2× latent upsample, refine at full res):

- Stage 1: run the text-to-video or image-to-video pipeline at `width // 2` × `height // 2` with `output_type="latent"`. `video_cli` already aligns upscale sizes to 64, so the halves stay 32-aligned.
- Upsample: pass the latents to `LTX2LatentUpsamplePipeline` (`latents=`, with the half size as `height`/`width`), returning latents.
- Stage 2: denoise the upsampled latents at full size with the same distilled transformer over the stage-2 sigmas, then decode. In diffusers 0.41 that's the `LTX2Pipeline` call with `latents=` and `sigmas=` (the same trick the image backend uses for Krea 2), or `LTX2DFRPipeline` / `LTX2DFRTemporalRefinePipeline` if a spike shows they do this directly. The spike decides between them. (Resolved in step 3: `LTX2Pipeline` was enough.)
- Progress: stage 1 and stage 2 report real steps under the existing `video_upscale_stage_1` / `video_upscale_stage_2` phases, with one total, as on macOS.
- Confirm that the configured `ltx-2.3` repo ships a `latent_upsampler` component. If not, load it from a separate repo, configured under `video_model_presets.ltx.diffusers`. (It does not; see the evidence table.)
- Tests: mocked pipelines; assert the stage-1 size is half the target, the upsampler gets the half size, stage 2 runs at the target size, and the step totals.
- **Review before upscaling for video** (moved here from [Review before upscaling](archive/upscale-review.md), which shipped for images everywhere and video on macOS). Once stage 1 returns latents, decode them to a temporary mp4 and call the backend's `review(path)` before stage 2, as `video_mac` does. Then make `supports_video_upscale_review()` in `backends/__init__.py` true on CUDA and drop the `ValueError` that `video_diffusers` raises for `review`. Measure the peak VRAM during the preview decode against the no-review peak; if it is higher, free the transformer for the preview and reload it on Approve. Tests: the mocked stage-1 latents are decoded and passed to `review` before stage 2 runs, and a discard skips stage 2.

### 4. `compute_precision` on CUDA (dropped)

Dropped on 2026-10-11. This step existed to keep a config key and CLI flag identical on both platforms. The macOS option it mirrored was implemented, measured at 0–4% on an M5 (noise), and removed, so no key or flag exists to match. A float16 mode on CUDA would have to stand on its own measurements; the numbers are in [macos-speed.md](macos-speed.md). The step number stays so the other steps keep theirs.

### 5. Quantized, streamed video on CUDA, with the 1.1 transformer (steps 5 and 6 merged)

**Done (2026-10-14).** The two steps were merged because neither fixes a 32 GB machine on its own: quantization alone still holds the text encoder and transformer together (about 36 GB at FP8), and streaming alone still holds 66 GB of bfloat16 weights.

- **The 1.1 transformer.** `video_model_presets.ltx.diffusers.transformer` names the single-file checkpoint (`Lightricks/LTX-2.3`, `ltx-2.3-22b-distilled-1.1.safetensors`). Its transformer tensors are renamed with diffusers' own `convert_ltx2_transformer_to_diffusers`; everything else (text encoder, connectors, VAEs, vocoder, scheduler, tokenizer) comes from `default_repo`, whose own transformer weights are no longer downloaded. Without the entry, the repo's transformer is used as before.
- **Levels by alias.** `ltx-4` (NF4 transformer) and `ltx-8` (FP8 transformer) now resolve on Windows and Linux too, both with an FP8 text encoder and connectors. Decided 2026-10-14: `ltx-2.3` (bfloat16) is removed and the backend refuses to load without a level, which also drops the unquantized loading code. The level comes from `video_model_presets.ltx.diffusers.quantize` (`ltx-4: 4`, `ltx-8: 8`), looked up by the picked model name, so the Web UI needs no new control and the macOS default `ltx-8` works everywhere.
- **Prepared copies.** At a quantize level the transformer and the text encoder are written once as the alias's stored copy in the models folder, `~/.ziv/models/ltx-4@q4/{transformer,text_encoder}` (first kept in a separate `~/.ziv/cache/ltx/`, moved 2026-10-15 so the Models page counts and deletes them with the alias; the text encoder is hard-linked between the two aliases), as sharded safetensors with a manifest of the source files (resolved name and size), the level and the storage format. A changed source or format writes a new copy. The write reads one tensor at a time with plain reads and drops each read range from the page cache (`backends/cuda/safetensors_stream.py`), so a 46 GB checkpoint never fills memory. NF4 weights are packed by bitsandbytes on the GPU and saved in `Linear4bit`'s own layout; only the linear layers of the transformer blocks are packed, the rest stays bfloat16. Unquantized loads read the sources directly.
- **The text encoder in its own phase.** The pipelines are built without a text encoder. For each new prompt it loads (FP8, group-offloaded), encodes, and is released before the transformer runs; the last prompt's embeddings are kept, and both upscale stages use them. When the estimate says the two do not fit in system memory together (below 80% of it), they take turns: the transformer loads only after the first prompt is encoded, and a new prompt releases it (and unloads its LoRAs, which also sit on the connectors) before the text encoder loads, then reloads it from the prepared copy and applies the LoRAs again. A first run that kept the transformer loaded while the text encoder loaded peaked at about 25 GB on the 30 GB machine and was killed; that is what this fixes. A repeated prompt reloads nothing.
- **Streaming.** In low-memory mode the FP8 and bfloat16 transformers use diffusers' block-level group offloading, as images do, and the connectors leaf-level. The NF4 transformer streams with `block_streaming.stream_blocks`, which swaps each block's (pinned) parameters for GPU copies around its forward pass and keeps the rest of the model on the GPU. The VAEs and vocoder live on the GPU. `enable_model_cpu_offload()` is gone from video.
- **A memory check instead of a kill.** Before loading, `estimate_ltx_cuda_system_memory` (`utils/model_memory.py`) estimates the larger of the two phases from the file headers. A level that cannot fit raises a clear error naming `ltx-4`; one above 80% of system memory warns.
- **Guidance off.** The first manual runs gave morphed video and metallic audio in t2v and i2v. The cause was step 3's choice to pass only `guidance_scale=1.0`: diffusers 0.41 then still runs audio classifier-free guidance at 7, spatio-temporal guidance at 1 and modality guidance at 3. The same prompt and seed at 704×448, 49 frames, q4: defaults looked over-sharpened and stylized, at about 9 s/step with a GPU peak of 8.3 GB and allocator OOM retries; all terms neutral looked natural at 2–3 s/step and 7.1 GB. Every term is now neutral, as on macOS, and prompts are encoded without a negative prompt. The converted 1.1 weights were checked as well: all 230 compared tensors match the diffusers 1.0 transformer's shapes and differ from it by at most 19% (relative norm), as a fine-tune does.
- **Output quality.** After the guidance fix the clips were still slightly smudgy at both levels. Two causes were checked: the output encoder, which was diffusers' `encode_video` with x264's default CRF 23 (macOS writes CRF 18), now replaced by `backends/cuda/video/encode.py` at CRF 18; and the FP8 connectors, which differ from bfloat16 by 1.5% (video) and 2.3% (audio) relative error with cosine similarity of about 1.0 on a real prompt, too little to matter. Image to Video now fits the image like macOS (cover and center-crop) instead of diffusers' stretch.
- **Longer clips (2026-10-15).** The first runs topped out at M 10 s and L 5 s, and XL failed: the VAE tiled only in space (diffusers leaves `use_framewise_decoding` off), so each tile decoded every frame, and the feed-forward layers widened every token of the clip at once. Frame-wise decoding and chunked feed-forward layers (`backends/cuda/video/token_chunking.py`, 4096 tokens per chunk, outputs unchanged) hold the GPU peak at 6.75 GB at every size below:

  | ltx-4, t2v, 8 steps | Result | Time | GPU peak | RSS peak |
  |---|---|---|---|---|
  | M 704×448, 10 / 15 / 20 s | ok | 152 / 119 / 163 s | 6.1–6.75 GB | 20–22 GB |
  | L 960×512, 5 / 10 / 15 s | ok | 71 / 119 / 197 s | 6.75 GB | 18–23 GB |
  | XL 1408×896, 5 s (direct / `--upscale 2`) | ok | 165 / 270 s | 6.75 GB | 23 / 26 GB |
  | XL 1408×896, 10 s | GPU out of memory in the first denoising step | | 7.5 GB | |

  Pushing XL further (same day): the VAEs and vocoder now stay off the GPU while the transformer runs (`block_streaming.load_when_used` / `unload_before`); the final pass returns latents and `backends/cuda/video/streaming_decode.py` decodes them with the VAE's own tiles and blend but sends each finished chunk to the CPU as 8-bit frames (diffusers' frame-wise decode kept every tile on the GPU, about three times the clip, and the pipeline then made a float32 copy of the whole clip in system memory); and every RMSNorm in the blocks is chunked like the feed-forward layers (their float32 upcast of a 57,000-token XL 15 s state is 0.93 GB per temporary).

  | ltx-4, XL 1408×896 | Result | Time | GPU peak | RSS peak |
  |---|---|---|---|---|
  | 10 s direct | ok | 462 s | 5.8 GB | 20.7 GB |
  | 10 s `--upscale 2` | ok | 370 s | 5.8 GB | 25.6 GB |
  | 15 s direct / upscaled | GPU out of memory at denoising step 3 / refine step 2 | 75 s per step | 7.6 GB | |

  XL 15 s still fails on a 929 MB float32 temporary (57,000 tokens × 4096 × 4 bytes) outside the chunked layers, most likely the rotary embedding tables or their application in attention, and it would take over 10 minutes per clip on this card even if it fit. XL stops at 10 s on a 10 GB GPU for now.

  Image-to-video (same day): diffusers gives every video token its own timestep (the conditioned first frame is clean), so the timestep modulation was expanded per token, about 2.8 GB held for the whole pass plus 2.8 GB in each block at XL 10 s, and L 10 s and XL 10 s ran out of GPU memory. `backends/cuda/video/timestep_runs.py` applies it per run of equal timesteps instead, with the same outputs. i2v with `ltx-4`: L 10 s ok in 210 s (GPU peak 4.4 GB), XL 10 s ok in 383 s (GPU 6.7 GB, RSS 21.4 GB).
- Still open: the q8-vs-q4 quality comparison against bfloat16 and against the same seeds on a Mac, LoRAs on the NF4 transformer on real hardware, a progress event for the one-time preparation (it only logs today), and encoding all of a batch's prompts in one text-encoder load.

Measured 2026-10-14 on a 10 GB GPU with about 32 GB of RAM, Linux, torch 2.14 / CUDA 13, diffusers 0.41.0, bitsandbytes 0.50.2, at q4, t2v 512×320, 25 frames, 8 steps, two different prompts in one load:

| | Result |
|---|---|
| Preparing the copies (once) | transformer about 4 min (reads 38 GB, writes 9.8 GB), text encoder about 3 min (reads 46 GB, writes 12 GB); peak RSS 6–10 GB, no memory pressure |
| Load (copies prepared) | 24 s; RSS 4.1 GB (connectors in FP8 and the CUDA runtime; diffusers' `from_pretrained` had held the connectors' bfloat16 copy too, 10 GB) |
| A new prompt | text encoder load and encode about 40 s, transformer reload about 30 s |
| Denoising | 4.6 s/step; GPU peak 8.3–8.6 GB |
| Peak RSS | 20–21 GB, PSI near 0; before the take-turns fix and the pinned-memory release, 25–27 GB with PSI up to 50% (and one kernel OOM kill) |

`estimate_ltx_cuda_system_memory` is calibrated to these runs (6 GB beyond the weights): q4 20 GB, q8 27 GB, bfloat16 47 GB. q8, i2v, `--upscale 2` and LoRAs were not run on hardware in this step.

Original plans:

- (5) Add `--quantize 8` (FP8) and `--quantize 4` (NF4) for the LTX transformer, reusing `backends/cuda/quant.py`, store the copy with the stored-quant flow (`<name>@q4` / `<name>@q8`), and expose the levels in the Web UI's video settings. (Became aliases and a cache folder: the transformer is not an installed model with a folder of its own.)
- (6) With `--low-memory`, apply block-level group offloading to the LTX transformer instead of `enable_model_cpu_offload()`. Measure peak VRAM and seconds per step on a 12 GB and a 24 GB card.

### 7. Video memory estimates on CUDA: badges and the per-clip check

- Add `estimate_cuda_video_memory` to `utils/model_memory.py`, mirroring how step 5 loads the model, for `ltx-4` and `ltx-8` (there's no unquantized path any more). The transformer streams; the VAEs and vocoder stay off the GPU while it runs, and the decode streams its chunks. The system-memory half exists as `estimate_ltx_cuda_system_memory`. It needs the transformer checkpoint's header, which `huggingface_hub.get_safetensors_metadata` can read before the download.
- Use it in `model_status.describe_model_status` for CUDA video, in the same `discrete` shape as image models, with the existing `VIDEO_ESTIMATE_NOTE`.
- **Per clip.** Let `clip_memory_fit` handle a discrete budget too: GPU memory as a function of tokens (width × height × frames) and mode (t2v, i2v, upscale, extend), checked against VRAM, and system memory from `estimate_ltx_cuda_system_memory`, with the `classify_cuda_memory_fit` thresholds. The Duration note, the Extend note and the `ziv-video` warning then work unchanged. Fit the GPU curve to step 5's tables (the 6.75 GB plateau from M up to XL 10 s, then XL 15 s out of memory at 7.6 GB, on a 10 GB card), plus a few q8 runs. A 12 GB and a 24 GB card are needed to check that it scales.
- Tests: the estimate for a known header and size; `clip_memory_fit` with a discrete budget returns `tight` / `too_large` around the fitted limit; the model status gives a `discrete` `memory_fit` for CUDA video.

### 8. Same Z-Image default scheduler

**Done (2026-10-16, needs a look on other GPUs).** `model_presets.zimage.default_scheduler.diffusers` is now `beta`. Setting it alone would not have matched macOS, so the backend changed too:

- **diffusers' `use_beta_sigmas` is a different schedule from mflux's.** mflux takes the beta(0.6, 0.6) grid and then applies the resolution-dependent shift. diffusers shifts first and spreads the beta spacing linearly between the shifted end points, so at 10 steps and 1024×1024 it ends at sigma 0.25 where mflux ends at 0.115 (diffusers' linear default also ends at 0.25).
- **Fix.** `_apply_beta_scheduler` (`backends/cuda/image/backend.py`) gives Z-Image the beta grid from `schedulers/beta_sigmas.py` as `sigmas=` and turns on dynamic shifting, so the pipeline shifts with the same `mu` as mflux. The mflux `BetaScheduler` uses the same grid function. Other families keep `use_beta_sigmas`.
- **Check on a 10 GB NVIDIA GPU (diffusers 0.41.0), `zit` at `-q 8`, 10 steps, 1024×1024, seed 1234, four prompts:** the default, the new beta and diffusers' own beta all give clean images with no artifacts, and the new beta changes composition and detail a little, as a different schedule should. The images are a matter of taste, so beta is the default for parity, not because it is clearly better.

### 9. Offer to install ffmpeg on Windows and Linux

- Extend `ensure_ffmpeg` the way macOS works: on Windows offer `winget install ffmpeg` when `winget` is on `PATH`. On Linux, offer the distribution's package manager (`apt`, `dnf`, `pacman`) when present, with `sudo`.
- After installing on Windows, a new terminal may be needed before `PATH` picks up ffmpeg. Say so instead of failing with "still not found".

### 10. Smaller prompt-enhancer download on CUDA

- Look for a public, pinned NF4 or other pre-quantized copy of the same model that transformers loads directly. If one exists, point `prompt_enhancer.model.win32` / `.linux` at it and update `download_size_label`.
- Otherwise keep the 9 GB repo. It's a one-time download, and the label already tells users. Lowest priority.
- Re-checked 2026-10-15: no such copy exists yet (see the evidence table). A second option is to save our own NF4 copy with the stored-quant flow and drop the bfloat16 download afterwards. That cuts disk use from 9 GB to about 3 GB and load time, but not the download. It's only worth it if the load-time quantization shows up in timings.

### 11. Extend a video on CUDA

*Done (2026-10-15).* macOS extends a clip by pinning its last 25 frames at the start of a one-stage window and splicing the new part onto the original ([Extend a video](archive/video-extend.md)). The splice, source checks and prompts were already shared; CUDA got the generation step, with sound:

- `load_model(mode="extend")` loads `LTX2ConditionPipeline` through the same quantized, streamed load as the other modes (LoRAs apply as usual). The CLI and the Web UI load it for extend jobs.
- The tail is pinned with `LTX2VideoCondition(frames=tail, index=0, strength=1.0)`: 25 frames encode to exactly 4 latent frames, which overwrite the first latent frames at full strength, as the MLX pin does. The window runs at the source's size and frame rate.
- The condition pipeline has no audio pin (one audio timestep per sample), so `backends/cuda/video/audio_carry.py` adds one: the tail's sound is encoded with the LTX audio processor's log-mel settings and the audio VAE, a transformer pre-hook gives the carried tokens to the model clean and at timestep 0 on every step, and the clean tokens are put back into the result before decoding.
- The window's frames are written with the shared segment command, so a clip made by the app is stream-copied at the join.
- No upscale for extend, as on macOS.
- Found on the way: the CLI recorded the repository an alias resolves to, and on CUDA `ltx-4` and `ltx-8` share one, so Reuse and Extend could not tell the quantize level. It now records the name picked, as the Web UI does.

Spike results (a 10 GB GPU with 32 GB of system memory, torch 2.14, diffusers 0.41.0, `ltx-4`, 768×512 at 24 fps, +48 frames):

- 97 frames out; the original 49 decode bit-identically (stream-copy branch).
- The join is a 3.2 ΔE step against a 0.9 median between neighbouring frames, about 2.3 above normal motion. That matches the 2–2.5 ΔE macOS measured and accepted (the VAE round trip of the carry).
- Memory, `ltx-4` with low memory on and each window as long as the text-to-video limits from step 5 (at the presets of that time): 704×448 +19 s peaked at 6.0 GB of GPU and 21.4 GB of system memory, 960×512 +14 s at 6.1 GB and 22.9 GB, 1408×896 +9 s at 6.8 GB and 22.2 GB. Each came out with the expected frame count. Text to video at the same size and frame count (481, 361 and 241 frames) peaked at 6.2, 6.2 and 8.5 GB of GPU and 21.3, 20.8 and 20.2 GB of system memory (plus 0.7 GB of swap at 1408×896), and took within 5% of the extension's time. Extend never needed more GPU memory, and at most about 2 GB more system memory, which is within run-to-run noise (the peak also counts cached model files). So extend fits wherever the same-length text to video does. `ltx-8` at 704×448 +8 s peaked at 5.2 GB of GPU and 25.3 GB of system memory, in a run that also prepared its q8 copy.
- The audio carry encodes and decodes back to the source's sound (a 1 kHz test tone comes back as 100% 1 kHz, within 3 dB). Pinning changes the new part's sound at the same seed. A steady synthetic tone is not continued with the pin or without it, so the speech and ambience continuity is judged by ear, not by that metric.

### 12. SeedVR2 plain upscale on CUDA

macOS runs SeedVR2 through mflux ([shipped](archive/image-upscale-seedvr2-mac.md)). diffusers 0.41 has no SeedVR2 pipeline, so CUDA would need a PyTorch implementation of its own:

- **Spike first.** Candidates: ByteDance's reference code (`ByteDance-Seed/SeedVR`, Apache-2.0) with the `ByteDance-Seed/SeedVR2-3B` / `-7B` weights, or numz's ComfyUI node, which already does tiling and block swap on consumer GPUs but is built around ComfyUI. Vendor only the inference path (DiT, VAE, window attention). Measure 2× and 4× of a 1 MP image on a 10 GB GPU at 3B and 7B, with the transformer streamed by `block_streaming` and the VAE tiled.
- **If it fits:** a `backends/cuda/seedvr2.py` that implements the `SeedVR2Upscaler` Protocol (`core/seedvr2.py`), registered in `_SEEDVR2_UPSCALERS_MAP` for `win32` and `linux`. The tier choice in `config.yaml` (`seedvr2.tiers`) needs CUDA estimates in the `discrete` shape. The stored 8-bit copy goes through `seedvr2_loader.py` like the image models' copies, using `backends/cuda/quant.py`. The Web UI, the X chord and the Models page row need no changes, because they already follow `supports_seedvr2()`.
- **If 7B doesn't fit a 10 GB GPU,** the tier choice picks 3B there, as it does on smaller Macs.
- Tests: the Protocol contract with a mocked model, the tier pick against a `discrete` budget, and the registry returning the upscaler on Linux.

### 13. Ideogram 4 on CUDA

diffusers 0.41.0 ships `Ideogram4Pipeline`, and both official repos are already in its layout (see the evidence table). So the "macOS only" limit now comes from our code, not from diffusers.

- **Spike first.** Load `ideogram-ai/ideogram-4-nf4-diffusers` (16 GB, NF4 transformer and Qwen3-VL text encoder) on a 10 GB GPU with model or group offloading and measure 1 MP and 2 MP. Then try the FP8 repo the `ideo` alias already uses, through `load_fp8_components` and block streaming. If it loads as is, both platforms share one repo and the stored quant flow can apply. Check the `unconditional_transformer`: it doubles the transformer weights and may be the first thing that doesn't fit.
- **Load.** Drop the `RuntimeError` for `ideogram4` in `DiffusersBackend.load_model` and give `model_aliases.ideo` a `win32` / `linux` repo. Ideogram 4's `supports_img2img: false`, `supports_upscale: false` and the dimension limits already apply on every platform. `supports_quantize: false` holds only if CUDA also uses one fixed-precision repo.
- **First sigma and JSON prompts.** macOS overrides the first denoising step's timestep by wrapping mflux's `Ideogram4Scheduler.make_timesteps` (`backends/mac/image.py`). The diffusers pipeline takes no `sigmas`, so the CUDA side needs the same hook on its scheduler's `set_timesteps`, or `supports_first_sigma` has to become backend-scoped. That's the same per-backend capability question as step 1. JSON prompts are only text, so they should carry over. Check that against the pipeline's prompt handling.
- **Guidance.** Map our single guidance value to the pipeline's `guidance_schedule` the way mflux does. Compare against the same seed on a Mac.
- Tests: a mocked `Ideogram4Pipeline` receives the mapped guidance schedule and the first-sigma override, and an `ideo` alias resolves on Linux.

## Verification

- Unit tests with mocked pipelines for each step, as above. CI runs on macOS only, so the CUDA steps also need a manual run on a Linux or Windows machine with an NVIDIA GPU. Record the GPU, driver, torch and diffusers versions in each PR.
- Step 1: `klein4b` and `klein9b` with a reference image, `--upscale 2`, and gallery Upscale, unquantized and at q4/q8.
- Steps 2, 3, 5: `ltx-8` and `ltx-4` t2v and i2v, with and without `--audio`, with `--upscale 2`, at each quantize level, with `--low-memory` on and off. Compare against the same seeds on a Mac for overall look and audio sync, not pixels.
- Step 7: compare the badge's estimate with the peak VRAM in the job log. For the per-clip check, run one clip just under the limit it predicts and one just over.
- Step 12: 2× and 4× of the same images as the macOS spike, comparing time, peak VRAM and system memory, and the look against macOS.
- Step 13: the same prompts and seeds as on a Mac, with and without first sigma and with a JSON prompt, at 1 MP and 2 MP.
- Still owed from the done steps, on an NVIDIA GPU: audio sync against a Mac (step 2); `--low-memory` off, LoRA and Cancel during the refine stage (step 3); q8 against q4 for quality, and LoRAs on the NF4 transformer (step 5).

## Docs and changelog

- `CHANGELOG.md` `[Unreleased]`, one line per step as it ships, e.g. "FLUX.2 Klein reference images and upscale now work on Windows and Linux" and "Videos made on Windows and Linux now have sound".
- `README.md` platform table and `docs/getting-started.md`: drop the Windows/Linux video caveats as they close, and add `ltx-4` and `ltx-8` there.
- Model guide: the Z-Image scheduler default.
- Step 7: the "Memory Check" section of `docs/guides/video.md` stops saying macOS only.
- Steps 12 and 13: `docs/guides/image.md` drops "macOS only" from the plain upscale and Ideogram 4 sections, and the README and `docs/index.md` feature lists lose the Ideogram 4 caveat.

## Alternatives considered

- **One big parity PR.** Rejected: the video steps need hardware to verify and would hold back the small fixes (1, 2, 8).
- **Turn off Klein img2img on CUDA permanently.** Rejected: Klein is the default model, and refining images is a core feature.
- **Use only `LTX2DFRPipeline` for the CUDA upscale.** Possible, but it's new in 0.41 and its output isn't compared with the macOS two-stage flow yet. The spike in step 3 decided: `LTX2Pipeline` was enough.

## Open questions

- ~~Which attribute holds the LTX-2 audio sample rate in diffusers 0.41 (step 2).~~ Answered (2026-10-14): `pipeline.vocoder.config.output_sampling_rate`.
- Whether Ideogram 4 on CUDA should use the NF4 diffusers repo or the same FP8 repo as macOS (step 13). This depends on the spike and on whether `supports_quantize: false` stays platform-wide.
- Which PyTorch SeedVR2 to vendor for step 12: ByteDance's reference code or numz's node.
- Whether "Upscale an existing video" gets a CUDA step here when [it ships](video-upscale-existing.md) on macOS. The building blocks exist: `LTX2LatentUpsamplePipeline` and step 3's `LTX2Pipeline(latents=, sigmas=)` refine.
- ~~Whether the configured repo includes the latent upsampler, and whether `LTX2Pipeline` accepts `latents=` plus `sigmas=` for the refine stage (step 3).~~ Answered (2026-10-14): the repo has no upsampler, which is loaded from `Lightricks/LTX-2.3`, and `LTX2Pipeline` accepts both.
- Whether the per-backend capability override in step 1 is worth adding, or the fix should ship directly.
- How much quality q4 costs for LTX on CUDA (step 5). Both levels are offered; q4 is the one that fits a 32 GB machine.
