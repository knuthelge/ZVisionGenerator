# Faster generation on macOS

**Status:** Proposed (2026-10-11)

## Problem

Generation on Apple Silicon is slower than it needs to be in a few places, and most of them are not in the denoising math. This proposal ranks the opportunities found by reading our macOS backends against mflux 0.22 and the vendored LTX MLX v0.16.3, and lists the measurements that would size them before anyone builds anything.

Nothing here has been implemented. Apart from the streaming measurement in item 4, no change has been timed yet. All other gains below are estimates, each with its basis.

### Measured context

[deps-macos.md](archive/deps-macos.md) proposed an opt-in float16 compute mode (`compute_precision`) for Z-Image and FLUX.2 Klein, based on mflux's release notes (8–35% faster, measured on an M1 Max). It was tried on an M5 with 32 GB and gave 0–4%, which is noise. mflux documents the mode as aimed at the M1 family, and the M5 Max shows no change either. It was dropped from deps-macos, so float16 image compute is not an opportunity on current Macs.

The lifetime peak of an LTX run is set before denoising (the text-encoder load and the pre-denoise phase), so streaming only lowers the steady-state footprint, not the peak. Item 4 has the numbers.

Baseline on that M5 (32 GB), `-s xs` (400x608). Timings are noisy (about ±25%):

| Model | Seconds per step |
|---|---|
| `zit` q8 | ~1.9 |
| `klein4b` q4 | ~1.1 |
| `klein9b` q4 | ~2.4 |
| `krea2` q4 (with LoRA) | ~7.0 |
| `ideo` (FP8) | ~10 |
| `ltx-4` t2v, 512x320x49 | ~6, for 8 steps, ~70 s wall |

## Opportunities

Ranked by estimated gain for effort and risk. "Output" says whether the same seed gives the same result.

### 1. Ideogram 4 stored q4/q8 quants

- **Change:** Pass `quantize` to `Ideogram4(...)` and set `supports_quantize` to true for `ideo`, so the stored-quant flow creates `ideo@q4` and `ideo@q8` like the other families.
- **Evidence:**
  - `zvisiongenerator/backends/image_mac.py:366-372` builds `Ideogram4` without `quantize`, with a comment that the transformer is FP8-locked.
  - `zvisiongenerator/config.yaml:158` sets `supports_quantize: false`.
  - The stored-quant flow is `zvisiongenerator/image_model_loader.py:196-254`.
  - mflux supports it (`mflux.models.ideogram4`, README section on quantization). Its `Fp8Linear` (`model/ideogram4_transformer/fp8_linear.py`) dequantizes FP8 to dense bf16 on every forward. The model is 27.5 GB (two 9.3 GB transformers plus an 8.8 GB text encoder) on a 32 GB machine. The README says a quantized copy "removes FP8 dequantization step from every forward".
- **Estimated gain:** up to about 2x per step. Basis: `ideo` takes about 5 s per forward, against `klein9b` q4 at 2.4 s per step for a similar-sized transformer.
- **Output:** changes. The upstream README reports 5.9–29% of subpixels differing at q8 and 38–66% at q4, "visually indistinguishable".
- **Effort / risk:** small / medium. Upstream measured the conversion peak at 22 GB (q4) and 28.7 GB (q8) on a 64 GB machine, so conversion may not fit a 32 GB Mac at q8.

### 2. Keep the last image model loaded between Web UI jobs

- **Change:** Cache the loaded backend, keyed by model, quant, LoRAs and weights. Free it on a model change, a video or upscale job, before prompt enhancement, or after an idle timeout.
- **Evidence:** every job loads the model (`zvisiongenerator/web/web_runner.py:664` and `:690`, through the loader at `:880-907`) and frees it at the end (`:606-607`).
- **Estimated gain:** 3–10 s per job, plus the compile retrace (item 6). Basis: the model-load timings in [image-model-memory.md](archive/image-model-memory.md).
- **Output:** identical.
- **Effort / risk:** medium / medium. On 32 GB an image model and the enhancer cannot both stay resident, so the eviction rules decide whether this helps or causes swap.

### 3. LTX compute dtype (shipped)

- **Change:** `_generation_scope` in `zvisiongenerator/backends/video_mac.py` defaults `LTX2_COMPUTE_DTYPE` to `float16` while a generation runs, unless the user already set the variable.
- **Evidence:** the vendored hook is in `packages/ltx_pipelines_mlx/_base.py`. The LTX DiT computes in float32 by default because float32 AdaLN tables promote the activations. Upstream reports, on an M1 Max, float16 cutting a run from 701 s to 563 s (-20%) and bfloat16 about 0%.
- **Measured on the M5 (32 GB), idle machine, `ltx-4`, same seed:**

  | Clip | Denoising per step, float32 → float16 | Whole render |
  |---|---|---|
  | 512x256, 49 frames | 4.71 → 3.59 s, 5.57 → 4.76 s (two pairs) | 53 → 44 s, 61 → 55 s |
  | 704x448, 97 frames | 22.3 → 16.6 s | 205 → 161 s |

  Peak Metal memory was the same (13.2 GB at 704x448). PSNR of float16 against float32 was 31.8 dB (small) and 35.1 dB (larger); float32 runs repeat exactly. Single-kernel timings agree: quantized matmul about 1.4x and attention about 2.3x faster than float32, with bfloat16 equal to float16 on this chip.
- **Output:** changes slightly for the same seed; judged "basically the same" by eye.
- **Decision (2026-10-14):** on by default with no option. `LTX2_COMPUTE_DTYPE=float32` restores the old path.

### 4. LTX block streaming on 32 GB (settled)

- **Status:** settled by the deps-macos measurement (2026-10-11). The low-memory streaming threshold moved from 32 GiB to 24 GiB, so 32 GB Macs no longer stream and `ltx-4` runs with the transformer resident. Nothing further to build here.
- **Measured** on an M5 with 32 GB, `ltx-4` at 512x320, 49 frames, streaming on and off interleaved, 3 runs each:
  - The lifetime peak footprint dropped only about 1.8 GB in every scenario (t2v, upscale 2, LoRA). Max RSS was 1–2 GB higher with streaming (mmap). There was no swap in either mode.
  - The steady-state denoise footprint fell from about 14 GB to 1.7–3.5 GB, but the peak is set earlier, so that did not help.
  - Seconds per step: t2v +5% and −1%, upscale stage 1 +6.7% and stage 2 −4%, LoRA +98% (17.5 against 8.8).
- **Decision:** the rule was to keep 32 only if the peak dropped by at least 4 GB (or swap was avoided) and no scenario regressed by more than 10%. Neither held, so the threshold is 24 GiB (`zvisiongenerator/backends/video_mac.py`, `_LOW_RAM_THRESHOLD_BYTES`).
- **Not measured:** 24 GB and 16 GB Macs. The 24 GiB threshold keeps streaming where the transformer plus the other components may not fit resident.

### 4a. LoRA under streaming is about 2x slower

- **Change:** Remove the roughly 2x step-time penalty that LTX LoRAs pay while the transformer streams. This still affects Macs with 24 GB or less.
- **Evidence:** the LoRA scenario ran at 17.5 s per step with streaming against 8.8 s resident (+98%), while the other scenarios moved by −4% to +6.7% (see item 4).
- **Likely cause (unconfirmed):** LoRA weights are fused into each block on every forward while streaming, instead of once. Confirm in the vendored pipeline before choosing an option.
- **Options to evaluate:**
  - Fuse the LoRAs once before streaming, if upstream allows it.
  - Skip streaming when LoRAs are set and the machine has enough memory.
  - Warn when streaming and LoRAs are combined.
- **Effort / risk:** not assessed.

### 5. Klein text encoding: fewer layers and a prompt cache

- **Change:**
  - Truncate the Qwen3 text encoder to the 27 layers Klein reads, after the stored-quant save (`zvisiongenerator/image_model_loader.py:243`).
  - Cache `_encode_prompt_pair` per prompt for Klein and Z-Image.
- **Evidence:** Klein takes hidden layers 9, 18 and 27 (`mflux.models.flux2.variants.txt2img.flux2_klein`, around lines 151-176), but the encoder runs all 36 (`mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder`, around lines 95-121). Klein and Z-Image have no prompt cache (the memory-saver callback in `mflux.callbacks.instances.memory_saver` is the only caching there).
- **Estimated gain:** 25% less text-encoder compute (27 of 36 layers), 1–2 GB less memory, and no re-encode when a batch repeats a prompt. The encoder was recorded at 7–8 s per Klein image in [image-model-memory.md](archive/image-model-memory.md), which may include step 1, so the absolute saving is uncertain.
- **Output:** identical.
- **Effort / risk:** small to medium / medium. Layer truncation reaches into mflux's module structure.

### 6. Reuse mflux's compiled predict across images

- **Change:** Cache the `mx.compile`d predict function per model instance instead of rebuilding it for each image.
- **Evidence:** mflux wraps a closure rebuilt on every `generate_image` call in `mx.compile` (`mflux.models.z_image` around line 110 and 250, `flux2_klein` around 93 and 299, `ideogram4` around 129-130), so each image retraces. Not applicable to Krea 2, whose closure captures the embeddings (`krea2` around 93 and 192-206).
- **Estimated gain:** unknown; a guess of 0.1–0.5 s per image. No basis beyond the retrace cost, so measure first.
- **Output:** identical.
- **Effort / risk:** small / medium. It depends on a private mflux API.

### 7. Load Gemma once per LTX batch

- **Change:** Encode every prompt of a job in one Gemma load.
- **Evidence:** each clip reloads Gemma (`zvisiongenerator/backends/video_mac.py:148`, `free_after=low_memory`) and drops the DiT and decoders (`:218-223`). Upstream's `PromptEncoder` accepts a list (`ltx_pipelines_mlx.utils.blocks`, around lines 234-266).
- **Estimated gain:** (N-1) Gemma loads and encodes saved for an N-clip batch. The size of one load is not measured.
- **Output:** identical.
- **Effort / risk:** medium / not assessed.

### 8. Two stage-2 steps instead of three in LTX upscale

- **Change:** Pass `stage2_steps=2` in the two-stage upscale path.
- **Evidence:** `zvisiongenerator/backends/video_mac.py:243-253`; `packages/ltx_pipelines_mlx/ti2vid_two_stages.py` around lines 735-746; upstream `docs/PIPELINES.md` around line 268. Stage 2 is the full-resolution stage.
- **Estimated gain:** about one third of the full-resolution stage. Basis: 2 of 3 steps.
- **Output:** changes.
- **Effort / risk:** small / not assessed. Needs a visual comparison before it is a default.

### 9. Image VAE and temp PNG

- **Change:**
  - Offer a bf16 VAE. We upcast the VAE to float32 even in bf16 mode (`zvisiongenerator/backends/image_mac.py:383-387`).
  - Write the img2img temporary PNG with `compress_level=0` (`:436-438` use the default compression).
- **Estimated gain:** a faster decode and encode at size `m` and on upscales for the VAE (not sized). For the PNG, an estimated 0.3–1 s per 4 MP upscale.
- **Output:** the VAE option changes the picture slightly; the PNG change is identical.
- **Effort / risk:** small. Experiment 5 sizes the VAE part.

### 10. Wire memory during generation

- **Change:** Call `mx.set_wired_limit` around generation.
- **Evidence:** mlx-lm does this (`mlx_lm.generate`, around lines 229-263); mflux, LTX and our code do not.
- **Estimated gain:** unknown. It can only help under memory pressure: `klein9b` q8, `ideo` and LTX.
- **Output:** identical.
- **Effort / risk:** small / not assessed.

### 11. Smaller items

- **`--no-audio` on LTX:** set `pipe.generate_audio=False` instead of decoding audio and stripping it with ffmpeg (`zvisiongenerator/workflows/__init__.py:83-84`; vendored `_base.py` around lines 494-506). Identical for the video.
- **Fewer steps (output changes):** Ideogram's `V4_TURBO_12` preset has 40% fewer steps than the hard-coded `V4_DEFAULT_20` (`zvisiongenerator/backends/image_mac.py:524`). `zit` could run 9 steps instead of 10 (`zvisiongenerator/config.yaml:116`).
- **Krea 2 with a LoRA:** mflux re-quantizes LoRA-touched q4 layers to q8 (`mflux.models.common.lora.mapping.lora_saver`, around lines 307-317). Compare speed without the LoRA and with `bake_lora=False` before deciding whether the 7.0 s per step is the LoRA's doing.
- **Enhancer:** mlx-lm's `batch_generate` (`mlx_lm.generate`, around line 1910) could shorten the 20–35 s preflight, but the rewrites would change.

## Measurement plan

Run in this order. Each is a measurement only, with no product change, on the M5 32 GB used for the baseline. Repeat each run, given the ±25% noise in the baseline.

1. **Ideogram FP8 against a q4 copy.** Seconds per step and swap (`vm_stat`). Sizes item 1.
2. **`ltx-4` streaming on and off, with `LTX2_COMPUTE_DTYPE` unset, `float16` and `bfloat16`.** Done: streaming is item 4, and float16 was measured and shipped as item 3 (bfloat16 was not timed end to end).
3. **`klein9b -n 3` with the same prompt.** Time `_encode_prompt_pair`, and compare step 1 against step 2 on images 2 and 3. Sizes items 5 and 6.
4. **Two identical `klein4b` Web UI jobs back to back.** Time from `model_loading` to the first step event. Sizes item 2.
5. **VAE decode in float32 against bf16** at size `m` and on a 2x upscale. Sizes item 9.

## Windows/Linux (secondary)

Not researched in depth. One candidate: the video backend enables VAE slicing and tiling unconditionally (`zvisiongenerator/backends/video_diffusers.py:164-169`). Enabling them only on low memory or at large sizes would avoid the slicing cost on GPUs with room. It needs a CUDA measurement before it is proposed properly.

## Rejected

- **mflux step cache.** Only supported for Qwen-Image 2.1, and runs under 10 steps skip nothing (`mflux.models.common` README, around lines 552-556; `step_cache.py:48`).
- **LTX sparse attention (`LTX2_SOL_TAU`).** Applies to the distilled stage 2 only and above 4096 tokens. Size `m` is about 2.2k tokens.
- **LTX TeaCache.** Upstream refuses it on distilled models (`docs/PIPELINES.md`, around line 266).
- **PiD decode.** It is a 4x image decode, slower, and changes the picture.
- **Live previews as a cost.** They are a linear projection at three milestones (`progress_events.py:16`), which is negligible.
- **Removing `mx.eval` in `_ProgressChecker`** (`zvisiongenerator/backends/image_mac.py:223`). mflux evaluates the same latents right afterwards (`z_image.py:133`), so there is nothing to save.
- **A smaller MLX cache cap or clearing the cache per image.** Already measured with no slowdown in [image-model-memory.md](archive/image-model-memory.md); not worth more work.
- **VAE tiling.** Already off (`flux2_initializer.py:48`).
- **Shorter text padding.** Changes the conditioning.
- **Fully loaded LTX (`--low-memory` off).** About 27 GB, too tight for 32 GB.
- **`LTX2_DIT_EVAL_EVERY=0`.** Risks the watchdog for little gain.
- **Unquantized Krea 2 text encoder (7.5 GB).** It runs once per prompt, which is cached, so it is not a per-step cost.
- **Float16 compute for mflux image models.** Measured at 0–4% on an M5 32 GB; see Problem.

## Alternatives considered

- **Only build the top-ranked items.** Rejected as a starting point: items 1 and 5 each depend on a measurement that is cheap compared with the build, so measure first.
- **Turn on LTX float16 by default after the M1 result.** Rejected at first, because the M1 gain did not carry over to the M5 for image models. It was then timed on the M5 (item 3), the gain held, and it shipped.
