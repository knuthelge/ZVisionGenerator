# macOS: mflux 0.22 and a fresh LTX

**Status:** Done (unreleased, 2026-10-11). Shipped on branch `deps-macos` in `10ca75a` (mflux 0.22 and mlx-lm 0.32), `4ace5cb` (LTX v0.16.3 re-vendor and port), `95d9115` (low-RAM streaming), `dfa148a` (docs) and `2139801` (MP4 settings fix). The plan was corrected in several places along the way; see [What changed from the plan](#what-changed-from-the-plan). Verified on one M5 with 32 GB; see [Results (2026-10-10/11)](#results-2026-1010-11).

This is one of three dependency proposals. It covers the Apple Silicon engines: mflux for images, the vendored `ltx_core_mlx` and `ltx_pipelines_mlx` for video, and mlx-lm for the prompt enhancer. See also [Shared packages and the frontend](deps-shared.md) and [Windows and Linux: torch 2.14, CUDA 13 and diffusers 0.41](deps-cuda.md).

The text below is the original plan. Where the build found it wrong, the passage carries a *Corrected* or *Dropped* marker that points to [What changed from the plan](#what-changed-from-the-plan).

## Problem

1. **mflux is capped at 0.20.x** (`mflux>=0.20.0,<0.21`, added in v0.13.0b2 after mflux 0.20 broke our callbacks). 0.22.0 is out, and it makes the image models we ship faster:
   - **Z-Image** now runs its transformer in bfloat16, about 30% faster denoising (measured on an M5 in the release notes).
   - A new **opt-in float16 compute precision** makes steps 8–35% faster for FLUX.2 Klein, Z-Image and Qwen-Image 2.1 (measured on an M1 Max). *Dropped: it gave 0–4% on an M5.*
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
  - The reloaded final-norm bias affects FLUX.1, FIBO and Qwen-Image only. None of our aliases use those. *Corrected: stored quants of `zit`, `klein*`, `krea2` and `ideo` (from [image-model-memory](image-model-memory.md)) do not stay valid. The manifest's `backend_format` is `mflux-<version>`, so every copy is rebuilt once.*
  - Weight-name mismatches and PEFT-layout DoRA adapters now raise errors instead of producing noise. That's better, but our LoRA import should surface the message clearly.
- `compute_precision` is a keyword argument on the Z-Image and FLUX.2 initializers (`ComputePrecision`). Krea 2 and Ideogram 4 don't take it. *Added from the build: its only choice is `float16`. There is no bfloat16.*

### Vendored LTX: our pin vs v0.16.3

What `backends/video_mac.py` and `utils/model_files.py` use, compared with upstream v0.16.3 (`ae3b7ea`):

| We use | v0.16.3 |
|---|---|
| `ImageToVideoPipeline` (i2v without upscale) | **Removed** (`493aec2`). i2v is `image=` on any pipeline. |
| `TextToVideoPipeline` (t2v, i2v with upscale, Gemma repo lookup) | **Renamed** to match Lightricks' class names (`d6cc3d1`). The distilled one-stage path lives on `BasePipeline`, and the two-stage path on `DistilledPipeline`. *Corrected: there is no distilled one-stage method upstream, and `BasePipeline` has no generate method.* |
| `generate_and_save(...)` | Only on some pipelines (`DistilledPipeline` has it). `frame_rate` is now a required keyword everywhere (`b35254a`). |
| Our own distilled two-stage upscale (`_generate_upscaled`, about 200 lines), using `_load_text_encoder`, `_encode_text`, `_load_vae_encoder`, `_load_decoders`, `_decode_and_save_video`, `_fuse_pending_loras`, `DISTILLED_SIGMAS`, `STAGE_2_SIGMAS`, `denoise_loop` | **`DistilledPipeline.generate_two_stage`** does the same thing (half-res distilled, 2× spatial upsampler, distilled refine) and matches Lightricks. The private helpers still exist, but upstream has fixed this path several times since our copy: ancestral Euler for 2.5 stage 2, re-applying the conditioning mask in `res2s`, keeping stage-1 audio. *Corrected: those fixes don't apply to our path.* |
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
- **Float16 compute as an option, off by default.** *Dropped after it was implemented and measured; see the correction below.* Add a `compute_precision` key (`bfloat16` | `float16`, default unset) to the image model presets in `config.yaml`. The usual precedence applies (CLI > variant > family > global), and there's a `--compute-precision` flag on `ziv-image`. `backends/image_mac.py` passes it to the Z-Image and FLUX.2 initializers only. For other families it warns once, with `warnings.warn(..., stacklevel=2)`, and ignores the setting. Leaving it off by default keeps output stable. Measure it on two Macs, then decide in a follow-up whether to turn it on for some families.
- **Accept the Z-Image output change** and add a CHANGELOG note: "Z-Image images change slightly for the same seed, and denoising is faster." Users can't pick image precision today (`image_model_loader._PRECISION` is fixed at bfloat16), so there's no old-output mode to keep.
- **Make sure our precision handling doesn't cancel the speedup.** `MfluxBackend.load_model` sets `ModelConfig.precision` and upcasts the VAE (`_upcast_model_weights`). Confirm that Z-Image's transformer actually runs in bfloat16 under 0.22 by timing a step, not just by reading the config.
- Check the callback signatures again (`image_mac.py` progress and preview callbacks, `image_mac_preview.py`). This is where 0.20 broke us. The existing `**kwargs`-tolerant callbacks should hold, but test with live previews in the Web UI.
- Tests: mocked unit tests for the `compute_precision` resolution and for passing it only to supported families. *Dropped with the option.*

### PR 2: re-vendor LTX at v0.16.3

- Set `LTX_COMMIT` to the v0.16.3 commit (`ae3b7ea9583db775bd4a1dfb506fb789585021e7`), not upstream `HEAD`, and run `make update-ltx`. Note the tag in the `Makefile` comment.
- Add `mlx-arsenal>=0.2.4; sys_platform == 'darwin'` to `dependencies`. `make update-ltx` doesn't carry the vendored packages' own dependencies.
- **Port `backends/video_mac.py` to the new pipeline classes:**
  - t2v and i2v without upscale go through the distilled one-stage path, with `image=` for i2v. That replaces both `TextToVideoPipeline` and `ImageToVideoPipeline`. *Corrected: that path doesn't exist upstream. It is composed from public upstream parts in `video_mac.py`.*
  - **Upscale goes through `DistilledPipeline.generate_two_stage`** (built on `DistilledPipeline.generate_and_save`, which wraps it and writes the file). Delete our own `_generate_upscaled` reimplementation and its use of six private helpers. This is the main simplification. It removes about 200 lines that copy pipeline internals, and it picks up upstream's fixes for this exact path. Our `stage1_steps` maps to `stage1_steps=`. Stage 2 always runs the full `STAGE_2_SIGMAS` today, so leave `stage2_steps=` unset to keep that.
  - Pass `frame_rate=` explicitly everywhere, from the job's fps. *Corrected: there is no job fps on macOS. The model default of 24 is used.*
  - Keep LoRAs on `_pending_loras`, the supported hand-off. It still carries `# noqa: SLF001`, but upstream documents it now.
- `utils/model_files.ltx_mlx_text_encoder_repo` reads the `gemma_model_id` default from `BasePipeline.__init__` instead of `TextToVideoPipeline.__init__`. The fallback constant stays.
- **Pass `low_ram_streaming`** when `--low-memory` is set and the machine has 32 GB or less (*corrected: 24 GB or less*), so `ltx-4` fits on smaller Macs with less swapping. Measure peak memory on a 32 GB Mac before choosing the threshold.
- Tests: update the 25 LTX references in `tests/` (mocked pipelines) to the new class names and calls (*corrected: only two test files were coupled to the library*). The tests should check that upscale calls `generate_two_stage` with the mapped step counts, and that i2v passes `image=`.

### Not in this proposal (follow-ups)

- **LTX-2.5 aliases** (`ltx25-4`, `ltx25-8`). Re-vendoring makes them possible, and they're the most visible gain, but they need their own defaults (steps, sizes, auto duration), memory checks and docs. They're also gated, so they depend on the token handling in [first-run.md](first-run.md). Decided 2026-10-10: LTX-2.5 not planned.
- Retake, extend, audio-to-video, keyframe interpolation and HDR in the Web UI.
- Float16 LTX compute (`LTX2_COMPUTE_DTYPE`) and block-sparse attention as user options.

## Verification (Apple Silicon, by hand)

CI runs on `macos-latest`, so it covers unit tests and imports. It can't run models.

- **PR 1:**
  - `zit`, `klein4b`, `klein9b`, `krea2` and `ideo`: q4 and q8, with a LoRA, with live previews, with a 2× upscale, and loading a stored quant.
  - Record seconds per step before and after, with `compute_precision` unset and with `float16`. Note the Mac model and memory in the PR. *The float16 half was run once and dropped.*
  - Run one enhance batch with mlx-lm 0.32.
- **PR 2:**
  - `ltx-4` and `ltx-8`: t2v, i2v, with upscale, with a LoRA, with `--low-memory` on.
  - Compare against clips rendered before the upgrade with the same seed. Expect differences, given the upstream fixes, but no regressions in motion, faces or audio sync.
  - Record peak memory.

## Docs and changelog

- `CHANGELOG.md` `[Unreleased]`:
  - "mflux 0.22: Z-Image is faster, and its images change slightly for the same seed. New `compute_precision` option (float16) for Z-Image and FLUX.2 Klein." *Dropped with the option.*
  - "macOS video uses LTX MLX v0.16.3: closer to the reference output, and lower memory with `--low-memory`."
- Model guide: the `compute_precision` key and flag. *Dropped with the option.*
- `docs/development.md`: the `mlx-arsenal` dependency in the vendoring section, and that `LTX_COMMIT` should point at a release tag. *The section didn't exist; it was added as "Vendored LTX".*

## What changed from the plan

Found while building. The plan above stays as written, with markers where it was wrong.

1. **Upstream has no distilled one-stage API.** In LTX v0.16.3, `--distilled` is two-stage, and `BasePipeline` has no generate method. The no-upscale path (t2v and i2v) is composed in `video_mac.py` from public upstream parts. Upscale uses `DistilledPipeline.generate_and_save`.
2. **Stored quants do not stay valid.** The manifest's `backend_format` is `mflux-<version>`, so every `@q4` and `@q8` copy is rebuilt once after the update. The CHANGELOG says so.
3. **`compute_precision` was implemented and then dropped.** mflux has only `float16` (no bfloat16), and only Z-Image and FLUX.2 Klein take it (not Krea 2 or Ideogram 4). Measured on an M5 with 32 GB it gave 0–4%, which is noise. mflux documents it as aimed at the M1 family, and upstream saw no change on an M5 Max either. See [macos-speed.md](macos-speed.md). Because it is gone, [platform-parity.md](platform-parity.md) no longer carries it as a gap.
4. **`docs/development.md` had no vendoring section.** One was added: "Vendored LTX".
5. **The old vendored LTX had local hand-patches** (step and progress callbacks), which violated the rule against hand-editing `packages/ltx_*`. They are gone. Progress now uses upstream's `on_step` and the `stepwise` slot.
6. **"25 test references" was an overcount.** Only `test_video_upscale.py` and `test_model_files.py` were coupled to the library. A new `tests/test_ltx_mlx_contract.py` pins the upstream internals we use.
7. **Most of the listed upstream fixes don't apply to our path.** The ancestral Euler, `res2s` and stage-1 audio fixes concern code we don't run.
8. **Video output changes.** These upstream changes alter the result: `14ee96f` (terminal sigma kept when steps are below 8), `bfa5755` (image preprocessing) and `2524595` (CRF per generation). So does `transformer-distilled-1.1`, which is now used everywhere. There is no job fps on macOS, so the default of 24 is used.
9. **New runtime dependency:** `mlx-arsenal`.
10. **The low-RAM streaming threshold is 24 GiB, not 32.** Measured with streaming on and off on a 32 GB Mac, 3 interleaved runs each:
    - The lifetime peak dropped only about 1.8 GB, because it is set while the text encoder loads.
    - The denoise footprint went from about 14 GB to 2–3.5 GB.
    - Steps were 0–7% slower, and a LoRA run was 98% slower.
    - There was no swap either way.

    `low_ram_streaming` sets a process-wide MLX cache limit. The backend scopes it and restores it afterwards.
11. **Image output changes.** Z-Image changes slightly for the same seed (bf16 in mflux 0.22). q4 and q8 outputs with a LoRA change on all families, because the bake now folds the LoRA into dense weights before quantizing. Klein output is pixel-identical.

## Results (2026-10-10/11)

Machine: Apple M5, 32 GB. Real models throughout.

**Images** (`zit`, `klein4b`, `klein9b`, `krea2` at q4 and `ideo`; `krea2` q8 was skipped to save disk): all passed.

- Stored quants rebuilt once and then loaded.
- LoRAs: the `zit`, Klein, Krea 2 and Ideogram test LoRAs.
- 2× upscale.
- The prompt enhancer on mlx-lm 0.32.
- Web UI live previews.

**Video** (`ltx-4`): all passed.

- t2v, i2v, upscale t2v and i2v, a LoRA, and `--no-low-memory`.
- The output has 49 frames (it was 48), at 24 fps with audio.
- Web UI progress runs 1 to 11 across both upscale stages.

**Not run:** `ltx-8`.

**Found along the way:** an older bug where MP4s lost their embedded settings. Fixed in `2139801`.

**Checks:** `make check` passes on every commit.

Residual risks:

- The hooks into LTX internals (`stepwise`, `_pre_denoise_flush`, `_pending_loras`) are pinned to v0.16.3 by `tests/test_ltx_mlx_contract.py`. The next re-vendor must re-check them.
- The Web UI memory-fit badge doesn't account for streaming on Macs with 24 GB or less.
- Macs with 16 or 24 GB were not measured.

## Alternatives considered

- **Raise the mflux cap to `<1.0`.** Rejected: 0.20 broke us within a day of release, and mflux changes its internals between minors.
- **Re-vendor LTX at upstream `HEAD`.** Rejected: tags are the commits upstream audits before release (for example `00ca7c0`, "align docs and CLI help with the code before 0.16.3"). HEAD may be mid-change.
- **Keep our own two-stage upscale and only fix the imports.** It would be less work now. But it keeps about 200 lines that depend on six private helpers, and it misses the upstream fixes to the very same path. Rejected.
- **Depend on published packages instead of vendoring.** Not possible today. `ltx-core-mlx` and `ltx-pipelines-mlx` aren't on PyPI (404 on 2026-10-09). A git URL dependency isn't an option either, because PyPI rejects uploads whose metadata contains direct URL dependencies, and we publish to PyPI. Revisit if upstream starts publishing.
- **Turn on float16 compute by default.** Rejected until it's measured on more than one Mac. The gains in the release notes come from one machine each.

## Open questions

- Whether the distilled one-stage path is reachable as a public method in v0.16.3, or only through `DistilledPipeline`'s `generate_two_stage` and `BasePipeline` internals. A one-hour spike against the vendored code settles which class t2v and i2v should use. *Resolved: it isn't reachable. See the correction below.*
- Whether `low_ram_streaming` helps on 16 GB Macs at all with `ltx-4` (the repo is about 60 GB), or only on 32 GB machines. *Partly resolved: measured on 32 GB only. 16 and 24 GB Macs were not measured.*
- mflux 0.22's `transformers>=5.5` and `torch>=2.13` floors tie this PR to the [CUDA proposal](deps-cuda.md). If CUDA verification is slow, PR 1 could land alone by raising the torch floor to `>=2.13`, but then CUDA users get a torch they weren't tested on. The recommended order is CUDA first. *Resolved: CUDA landed first.*
