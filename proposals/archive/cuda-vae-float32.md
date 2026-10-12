# Decode in float32 on CUDA

**Status:** Rejected (2026-10-09): step 1 found no visible difference, so the cause is in sampling, not the VAE

## Result of step 1

Run on an RTX 3080 (10 GB), 1024×1024, seed 7, no LoRAs, with a face at medium distance and a smooth dusk gradient. Each prompt was decoded three ways: bfloat16 VAE (current), float32 VAE with the latents cast to its dtype, and float32 with tiling off.

| Model | Mean / max pixel difference, bf16 vs fp32 | High-frequency energy, bf16 → fp32 (face; gradient) | Peak VRAM, bf16 → fp32 |
|---|---|---|---|
| Z-Image Turbo (10 steps) | 0.28 / 10 | eyes 3.575 → 3.563; sky 0.489 → 0.492 | 2.9 → 4.4–5.5 GB |
| FLUX.2 Klein 4B (4 steps) | 0.29 / 4 | 4.084 → 4.080; 0.966 → 0.973 | 2.9 → 4.4 GB, one 9.1 GB spike |
| Krea 2 Turbo, q4 (8 steps) | 0.30 / 35 (thin fence wires only) | 1.608 → 1.605; 0.767 → 0.764 | not recorded |

- Eyes, lashes and freckles at 4× zoom are indistinguishable, and grain in smooth skies measures the same (high-frequency energy is the mean absolute difference from a 1.5 px Gaussian blur).
- Tiling off changed nothing: at 1024×1024 diffusers does not tile.
- float32 costs VRAM, including one 9.1 GB peak on a 10 GB card (Klein), for no visible gain.

Per the outcomes below, this proposal is closed. The grain reported on CUDA is decided during sampling; the next step is a proposal comparing the mflux and diffusers scheduler configs for the same model.

## Problem

Images from the CUDA backend (Linux and Windows) look grainier than images from the same model, unquantized, on a Mac. Fine high-contrast detail such as eyes, lashes and catchlights can also look slightly pixelated or crunchy.

Nothing is quantized automatically on CUDA. Without `--quantize`, `DiffusersBackend.load_model` (`backends/image_win.py`) loads the pipeline in bfloat16. The difference between the platforms is in the VAE:

- **macOS (mflux):** `MfluxBackend.load_model` upcasts the VAE to float32 even in bfloat16 mode ("upcast VAE for better color gradients", `backends/image_mac.py`).
- **CUDA (diffusers):** the VAE is moved to the GPU in the pipeline dtype, `pipeline.vae.to(device="cuda", dtype=torch_dtype)`, so it stays in bfloat16. This happens in all three load paths: unquantized, NF4 and INT8.

bfloat16 keeps about 8 bits of mantissa. Decoding latents at that precision typically adds grain, banding in smooth gradients and aliasing on fine edges. That matches the reported symptoms.

Two other differences between the platforms may contribute:

- **VAE tiling and slicing are always on for CUDA.** Tiling can leave faint seams or texture differences at tile boundaries. mflux does not tile.
- **Sampling may differ.** mflux and diffusers may not use identical default schedules, shift or step handling. This is not verified. Different noise generators (torch CPU vs MLX) also mean the same seed gives a different image on each platform, so comparisons should look at overall texture, not the exact pixels.

The VAE explains grain and edge aliasing. It does not explain detail that is malformed or smeared, which gets decided in the latents. The first step is therefore to confirm the cause before changing anything.

## Step 1: Test

Run this on a Linux or Windows CUDA machine before implementing anything.

1. Pick 3–4 prompts with faces at medium distance (eyes at roughly 30–60 px), plus one with a smooth sky or studio gradient. Use a fixed seed, steps and guidance, no quantization and no LoRAs.
2. Render each prompt as a **baseline** (current code).
3. Render it again with the VAE in float32. A temporary local patch is enough: in `load_model`, replace the VAE line with `pipeline.vae.to(device="cuda", dtype=torch.float32)`, and cast the latents before decode (see the Flux 2 note below).
4. Render a third time with float32 VAE **and tiling off** (comment out `enable_vae_tiling()`), at a resolution that fits in VRAM.
5. Compare the three at 200–400 % zoom on the eyes, hair and gradients. Record decode time and peak VRAM (`torch.cuda.max_memory_allocated()`) for each.

Test Z-Image Turbo and at least one Flux 2 Klein model, because their pipelines handle the decode dtype differently.

**Outcomes:**

- **float32 removes the grain or edge aliasing:** go ahead with step 2.
- **Only tiling off helps:** change the tiling policy instead (see Alternatives).
- **No visible difference:** the cause is in sampling. Close this proposal and open one that compares the mflux and diffusers scheduler configs for the same model.

## Step 2: Implementation (if the test succeeds)

### Backend

- Add a private helper in `backends/image_win.py`, `_place_vae(pipeline)`, that moves the VAE to CUDA in float32. Use it in the unquantized path and in `_load_quantized` (both the NF4 and INT8 branches), replacing the three `pipeline.vae.to(device="cuda", dtype=torch_dtype)` lines.
- **Latent dtype at decode.** The pipelines do not all cast latents to the VAE dtype:
  - `ZImagePipeline` casts (`latents.to(self.vae.dtype)`) before decode, so it works as is.
  - `Flux2KleinPipeline` decodes the transformer's bfloat16 latents directly. With a float32 VAE this raises a dtype mismatch. Fix this inside the backend, not in the vendored diffusers code: wrap the VAE's `decode` (and `encode` for img2img, if needed) so it casts its input to the VAE dtype. A small `_cast_inputs_to_vae_dtype(vae)` helper that wraps the bound method is enough.
  - Check `Flux2Pipeline`, `FluxPipeline` and the img2img pipelines created with `AutoPipelineForImage2Image.from_pipe` the same way. `from_pipe` reuses the same VAE object, so the wrapper carries over.
- **Live previews.** `image_win_preview.py` reads `pipe.vae.bn` statistics and casts them itself, so it is unaffected. Confirm this in the test run.
- **Output dtype.** The decoded image is float32 before `image_processor.postprocess`, which is what we want. No change is needed there.

### Config

No new config key by default. float32 decode matches macOS behaviour and costs little. If the test shows a meaningful VRAM or speed cost on small GPUs, add `vae_precision: float32 | bfloat16` under the global image defaults (it can be overridden per model preset), defaulting to `float32`.

### Video

`backends/video_diffusers.py` also keeps the LTX VAE in the pipeline dtype. Video VAEs are much larger, and their decode memory grows with frame count. Leave video out of this change, and test it separately if the image result is positive.

### Tests and docs

- Unit tests in `tests/test_image_win.py` with mocked torch and diffusers: the VAE is moved to `torch.float32` in each of the three load paths, and the decode wrapper casts a bfloat16 input to the VAE dtype.
- `CHANGELOG.md` under `[Unreleased]`: "fix(image): decode in float32 on CUDA for cleaner fine detail, matching macOS."
- Note the float32 VAE on CUDA where `docs/guides/model.md` describes precision.

### Cost

The Z-Image and Flux VAEs are about 160 MB in bfloat16, so about 320 MB in float32, plus larger activations during decode. Decode time is expected to increase slightly per image. Tiling stays on to bound peak memory, so the extra VRAM should be modest.

## Alternatives considered

- **Use diffusers' `force_upcast` / `upcast_vae`.** Only some pipelines (SDXL-era) honour it. Z-Image and Flux 2 do not, so it would not cover the families we support.
- **Run the whole pipeline in float32 (`precision="float32"`).** This fixes precision everywhere, but doubles transformer memory and is much slower. The VAE is where precision matters most for the final pixels.
- **Turn tiling on only above a size threshold.** Worth doing if the test shows tiling contributes. It is independent of this change and can ship separately.

## Out of scope

- Matching mflux and diffusers sampling exactly (schedulers, shift, noise).
- Video VAE precision.
- Changes to quantized transformer or text-encoder compute dtypes.
