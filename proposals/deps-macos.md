# macOS: mflux 0.22 and a fresh LTX

**Status:** Proposed (2026-10-09)

This is one of three dependency proposals. It covers the Apple Silicon engines: mflux for images, the vendored `ltx_core_mlx` and `ltx_pipelines_mlx` for video, and mlx-lm for the prompt enhancer. See also [Shared packages and the frontend](deps-shared.md) and [Windows and Linux: torch 2.14, CUDA 13 and diffusers 0.41](deps-cuda.md).

## Problem

1. **mflux is capped at 0.20.x** (`mflux>=0.20.0,<0.21`, added in v0.13.0b2 after mflux 0.20 broke our callbacks). 0.22.0 is out, and it makes the image models we ship faster:
   - **Z-Image** now runs its transformer in bfloat16, about 30% faster denoising (measured on an M5 in the release notes).
   - A new **opt-in float16 compute precision** makes steps 8–35% faster for FLUX.2 Klein, Z-Image and Qwen-Image 2.1 (measured on an M1 Max).
   - **LoRAs** run in the model's loaded precision instead of float32, which saves memory on bfloat16 models.
2. **The vendored LTX is six months old.** `LTX_COMMIT` in the `Makefile` is `de4a12d` (2026-03-29). Upstream `dgrauet/ltx-2-mlx` has had 303 commits since, up to v0.16.3 (2026-10-09). We're missing many fixes that make output match Lightricks' reference: image-conditioning preprocessing, sampling of the distilled stage 2, stage-2 audio, and frame-rate handling. We're also missing **LTX-2.5 support** and memory features such as low-RAM block streaming.
3. **The longer we wait, the harder the LTX update gets.** Upstream renamed or removed every pipeline class we import, so this is already a port rather than a drop-in.

## Evidence

### mflux 0.22.0 (PyPI metadata and release notes, 2026-10-09)

- Requirements: `mlx>=0.32.0,<0.33`, `torch>=2.13.0,<3`, `transformers>=5.5.0,<6`, `huggingface-hub>=1.1.6,<2.0`, `pillow>=12.3.0`, `numpy>=2.0.1,<3`. torch and Pillow are raised by the other two proposals. mlx 0.32.3 is already locked, and so are torch 2.14 and transformers 5.19 (from [deps-cuda.md](deps-cuda.md)).
- **Every mflux internal we import still exists** in the 0.22.0 wheel:
  - model classes: `ZImageTurbo`, `Flux2Klein`, `Krea2`, `Ideogram4`
  - latent creators: Z-Image, FLUX.2, Ideogram 4
  - schedulers: `BaseScheduler`, `FlowMatchEulerDiscreteScheduler`, `LinearScheduler`, `Ideogram4Scheduler`
  - `ModelConfig`, `Config` and `StopImageGenerationException`
  - Checked by path in `backends/image_mac.py`, `image_mac_preview.py` and `schedulers/beta_scheduler.py`.
- Output changes we inherit:
  - **Z-Image changes slightly for the same seed**, because the transformer now runs in bfloat16. mflux keeps `float32=True` to reproduce old images.
  - The reloaded final-norm bias affects FLUX.1, FIBO and Qwen-Image only. None of our aliases use those, so stored quants of `zit`, `klein*`, `krea2` and `ideo` (from [image-model-memory](image-model-memory.md)) should stay valid. Verify anyway.
  - Weight-name mismatches and PEFT-layout DoRA adapters now raise errors instead of producing noise. That's better, but our LoRA import should surface the message clearly.
- `compute_precision` is a keyword argument on the Z-Image and FLUX.2 initializers (`ComputePrecision`). Krea 2 and Ideogram 4 don't take it.

### Vendored LTX: our pin vs v0.16.3

What `backends/video_mac.py` and `utils/model_files.py` use, compared with upstream v0.16.3 (`ae3b7ea`):

| We use | v0.16.3 |
|---|---|
| `ImageToVideoPipeline` (i2v without upscale) | **Removed** (`493aec2`). i2v is `image=` on any pipeline. |
| `TextToVideoPipeline` (t2v, i2v with upscale, Gemma repo lookup) | **Renamed** to match Lightricks' class names (`d6cc3d1`). The distilled one-stage path lives on `BasePipeline`, and the two-stage path on `DistilledPipeline`. |
| `generate_and_save(...)` | Only on some pipelines. `frame_rate` is now a required keyword everywhere (`b35254a`). |
| Our own distilled two-stage upscale (`_generate_upscaled`, about 200 lines), using `_load_text_encoder`, `_encode_text`, `_load_vae_encoder`, `_load_decoders`, `_decode_and_save_video`, `_fuse_pending_loras`, `DISTILLED_SIGMAS`, `STAGE_2_SIGMAS`, `denoise_loop` | **`DistilledPipeline.generate_two_stage`** does the same thing (half-res distilled, 2× spatial upsampler, distilled refine) and matches Lightricks. The private helpers still exist, but upstream has fixed this path several times since our copy: ancestral Euler for 2.5 stage 2, re-applying the conditioning mask in `res2s`, keeping stage-1 audio. |
| `_pending_loras` | Still supported, plus `LTX2_LORA_MODE=unfused`. |
| The ltx-core helpers we import (`compute_video_latent_shape`, `LatentState`, `LTXModel`, `X0Model`, `aggressive_cleanup`, the position helpers, `apply_quantization`, `load_split_safetensors`, `prepare_image_for_encoding`, `LatentUpsampler`) | All still exist with the same paths. |

Other breaking changes since our pin: hdr-ic-lora was replaced by an ACEScct HDR pipeline (`0e9f642`), the DiT config is now versioned and read from the checkpoint (`83a7011`), and the standalone upscale pipeline was removed (`e17c2af`).

New runtime dependency: `ltx-core-mlx` and `ltx-pipelines-mlx` now require `mlx-arsenal>=0.2.4`. OpenEXR is optional, needed only for HDR output.

New capabilities we would get:

- **LTX-2.5** checkpoints: `dgrauet/ltx-2.5-mlx-q4` (about 51 GB) and `dgrauet/ltx-2.5-mlx-q8` (about 75 GB). Both are gated with automatic approval. Decided 2026-10-10: LTX-2.5 not planned.
- Low-RAM DiT block streaming (`low_ram_streaming=`) and unfused LoRAs that work with it.
- Opt-in float16 DiT compute (`LTX2_COMPUTE_DTYPE`) and block-sparse stage-2 attention (`LTX2_SOL_TAU`).
- Distilled variants of retake, extend and audio-to-video, NAG guidance, keyframe interpolation, and auto-predicted duration on 2.5.

### mlx-lm

0.31.3 locked, 0.32.0 latest. It requires `mlx>=0.32.2` and `transformers>=5.7.0`. It's only used by `backends/prompt_enhancer_mac.py`.

## Proposed change

Two PRs, mflux first because it's smaller and independent of LTX.

### PR 1: mflux 0.22 and mlx-lm 0.32

- `mflux>=0.22.0,<0.23`. Keep the minor cap. It's the reason this upgrade is a deliberate step rather than a surprise.
- `mlx-lm>=0.32.0`.
- **Float16 compute as an option, off by default.** Add a `compute_precision` key (`bfloat16` | `float16`, default unset) to the image model presets in `config.yaml`. The usual precedence applies (CLI > variant > family > global), and there's a `--compute-precision` flag on `ziv-image`. `backends/image_mac.py` passes it to the Z-Image and FLUX.2 initializers only. For other families it warns once, with `warnings.warn(..., stacklevel=2)`, and ignores the setting. Leaving it off by default keeps output stable. Measure it on two Macs, then decide in a follow-up whether to turn it on for some families.
- **Accept the Z-Image output change** and add a CHANGELOG note: "Z-Image images change slightly for the same seed, and denoising is faster." Users can't pick image precision today (`image_model_loader._PRECISION` is fixed at bfloat16), so there's no old-output mode to keep.
- **Make sure our precision handling doesn't cancel the speedup.** `MfluxBackend.load_model` sets `ModelConfig.precision` and upcasts the VAE (`_upcast_model_weights`). Confirm that Z-Image's transformer actually runs in bfloat16 under 0.22 by timing a step, not just by reading the config.
- Check the callback signatures again (`image_mac.py` progress and preview callbacks, `image_mac_preview.py`). This is where 0.20 broke us. The existing `**kwargs`-tolerant callbacks should hold, but test with live previews in the Web UI.
- Tests: mocked unit tests for the `compute_precision` resolution and for passing it only to supported families.

### PR 2: re-vendor LTX at v0.16.3

- Set `LTX_COMMIT` to the v0.16.3 commit (`ae3b7ea9583db775bd4a1dfb506fb789585021e7`), not upstream `HEAD`, and run `make update-ltx`. Note the tag in the `Makefile` comment.
- Add `mlx-arsenal>=0.2.4; sys_platform == 'darwin'` to `dependencies`. `make update-ltx` doesn't carry the vendored packages' own dependencies.
- **Port `backends/video_mac.py` to the new pipeline classes:**
  - t2v and i2v without upscale go through the distilled one-stage path, with `image=` for i2v. That replaces both `TextToVideoPipeline` and `ImageToVideoPipeline`.
  - **Upscale goes through `DistilledPipeline.generate_two_stage`.** Delete our own `_generate_upscaled` reimplementation and its use of six private helpers. This is the main simplification. It removes about 200 lines that copy pipeline internals, and it picks up upstream's fixes for this exact path. Our `stage1_steps` maps to `stage1_steps=`. Stage 2 always runs the full `STAGE_2_SIGMAS` today, so leave `stage2_steps=` unset to keep that.
  - Pass `frame_rate=` explicitly everywhere, from the job's fps.
  - Keep LoRAs on `_pending_loras`, the supported hand-off. It still carries `# noqa: SLF001`, but upstream documents it now.
- `utils/model_files.ltx_mlx_text_encoder_repo` reads the `gemma_model_id` default from `BasePipeline.__init__` instead of `TextToVideoPipeline.__init__`. The fallback constant stays.
- **Pass `low_ram_streaming`** when `--low-memory` is set and the machine has 32 GB or less, so `ltx-4` fits on smaller Macs with less swapping. Measure peak memory on a 32 GB Mac before choosing the threshold.
- Tests: update the 25 LTX references in `tests/` (mocked pipelines) to the new class names and calls. The tests should check that upscale calls `generate_two_stage` with the mapped step counts, and that i2v passes `image=`.

### Not in this proposal (follow-ups)

- **LTX-2.5 aliases** (`ltx25-4`, `ltx25-8`). Re-vendoring makes them possible, and they're the most visible gain, but they need their own defaults (steps, sizes, auto duration), memory checks and docs. They're also gated, so they depend on the token handling in [first-run.md](first-run.md). Decided 2026-10-10: LTX-2.5 not planned.
- Retake, extend, audio-to-video, keyframe interpolation and HDR in the Web UI.
- Float16 LTX compute (`LTX2_COMPUTE_DTYPE`) and block-sparse attention as user options.

## Verification (Apple Silicon, by hand)

CI runs on `macos-latest`, so it covers unit tests and imports. It can't run models.

- **PR 1:**
  - `zit`, `klein4b`, `klein9b`, `krea2` and `ideo`: q4 and q8, with a LoRA, with live previews, with a 2× upscale, and loading a stored quant.
  - Record seconds per step before and after, with `compute_precision` unset and with `float16`. Note the Mac model and memory in the PR.
  - Run one enhance batch with mlx-lm 0.32.
- **PR 2:**
  - `ltx-4` and `ltx-8`: t2v, i2v, with upscale, with a LoRA, with `--low-memory` on.
  - Compare against clips rendered before the upgrade with the same seed. Expect differences, given the upstream fixes, but no regressions in motion, faces or audio sync.
  - Record peak memory.

## Docs and changelog

- `CHANGELOG.md` `[Unreleased]`:
  - "mflux 0.22: Z-Image is faster, and its images change slightly for the same seed. New `compute_precision` option (float16) for Z-Image and FLUX.2 Klein."
  - "macOS video uses LTX MLX v0.16.3: closer to the reference output, and lower memory with `--low-memory`."
- Model guide: the `compute_precision` key and flag.
- `docs/development.md`: the `mlx-arsenal` dependency in the vendoring section, and that `LTX_COMMIT` should point at a release tag.

## Alternatives considered

- **Raise the mflux cap to `<1.0`.** Rejected: 0.20 broke us within a day of release, and mflux changes its internals between minors.
- **Re-vendor LTX at upstream `HEAD`.** Rejected: tags are the commits upstream audits before release (for example `00ca7c0`, "align docs and CLI help with the code before 0.16.3"). HEAD may be mid-change.
- **Keep our own two-stage upscale and only fix the imports.** It would be less work now. But it keeps about 200 lines that depend on six private helpers, and it misses the upstream fixes to the very same path. Rejected.
- **Depend on published packages instead of vendoring.** Not possible today. `ltx-core-mlx` and `ltx-pipelines-mlx` aren't on PyPI (404 on 2026-10-09). A git URL dependency isn't an option either, because PyPI rejects uploads whose metadata contains direct URL dependencies, and we publish to PyPI. Revisit if upstream starts publishing.
- **Turn on float16 compute by default.** Rejected until it's measured on more than one Mac. The gains in the release notes come from one machine each.

## Open questions

- Whether the distilled one-stage path is reachable as a public method in v0.16.3, or only through `DistilledPipeline`'s `generate_two_stage` and `BasePipeline` internals. A one-hour spike against the vendored code settles which class t2v and i2v should use.
- Whether `low_ram_streaming` helps on 16 GB Macs at all with `ltx-4` (the repo is about 60 GB), or only on 32 GB machines.
- mflux 0.22's `transformers>=5.5` and `torch>=2.13` floors tie this PR to the [CUDA proposal](deps-cuda.md). If CUDA verification is slow, PR 1 could land alone by raising the torch floor to `>=2.13`, but then CUDA users get a torch they weren't tested on. The recommended order is CUDA first.
