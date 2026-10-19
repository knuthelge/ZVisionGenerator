# Upscale existing videos 2× on macOS

**Status:** Proposed (2026-10-13). Not started. Replaces the earlier SeedVR2 plan for videos, which research showed won't fit a 32 GB Mac (see [Evidence](#why-not-seedvr2)).

## Problem

The intended video workflow is: generate at `s` (the default), extend the clip until it's right, then enlarge the finished video 2×. Today the last step is impossible:

- The in-run upscale (two-stage LTX) only enlarges LTX's own half-size render inside the same run, and it can't be combined with extend (`video_mac.py`: "Extending a video is not supported together with upscale").
- Extend refuses an upscaled video, so upscaling first and extending afterwards doesn't work either.
- The viewer's Upscale menu is images only.

So a video built up over several extends stays at the size it was generated.

LTX already has everything needed: its two-stage pipeline's second stage takes a latent, upsamples it 2× with the spatial upsampler and refines it in a few distilled steps. Feeding that stage the VAE-encoded finished video, instead of a fresh half-size render, upscales a finished video with the same model, weights and quality as the in-run upscale. No new model, download or dependency.

### Builds on (done)

| Piece | Where | What this reuses |
|---|---|---|
| Two-stage pipeline | Vendored `DistilledPipeline`: `_upsample_latent`, `_stage2` (refine at full size, `stage2_steps` shortens `STAGE_2_SIGMAS` from the tail), `Stage1Result` | Stage 2 as is, fed from an encoded video |
| Swapping stage 1 | `backends/mac/video.py`: `_review_after_stage1` replaces `pipe._stage1` with an instance attribute for the upscale review | The same pattern, or a direct `_upsample_latent` + `_stage2` call |
| Encoding a finished video | Vendored `retake.encode_video_tensor`, `encode_source_audio_latent`, `ltx_core_mlx.utils.image.load_video_frames` (extend's `extend_from_video` encodes a whole source at full size) | Encode the source video and its audio |
| Streamed, tiled decode | `video_mac._decode_window_segment` (`tiled_decode` with `_compute_decode_tiling`, streamed to ffmpeg) | Decode the full-size result without holding it in memory |
| Long-clip tiling | Vendored `VideoModalityTiler` (spatial and temporal DiT tiles with blended overlaps, optional seam frames), passed as `DistilledPipeline(tile_count=…)`. Not used by the app yet | Bounds stage 2's memory for long extended clips |
| Model files | `utils/model_files.ltx_mlx_upsampler_file` | The same upsampler check as the in-run upscale |
| Memory check | `utils/model_memory.clip_memory_fit(…, upscale=True)` | The menu item's high-memory warning |
| Source checks | `utils/video_probe.probe_video`, `utils/video_extend.extend_source_problems` (rotation, frame rate, pixel format, HDR, interlacing, audio) | Refuse sources the frame path can't round-trip |
| Menu, route, job | The image upscale: `upscale` in `web/gallery.py`, `upscaleEntries` in `assetMenu.ts`, the X then 2/4 chord in `AssetViewer.svelte`, `POST /api/upscale`, output name `{stem}_{factor}x` | A 2× item and X then 2 for videos |
| Recorded settings | `utils/provenance.py`: `workflow: "upscale"`, a `source` block that Reuse follows, `generation.upscale`; extend already refuses a video with `generation.upscale` (`EXTEND_UPSCALED_REASON`) | The same keys and the extend refusal |

### Not in scope

- **Windows and Linux.** diffusers has `LTX2LatentUpsamplePipeline`, but the CUDA video backend has no encode-and-refine path yet. Add it to [platform parity](archive/platform-parity.md) when this ships, like extend.
- **Factors other than 2×.** The app only finds 2× spatial upsampler files (`ltx_mlx_upsampler_file`).
- **Frame-rate upscaling.** The vendored `LatentUpsampler` has a `temporal_x2` variant; a separate feature if wanted.
- **Videos from outside the output folder**, like the image upscale. Everything in the output folder was made by LTX, which is what the refine is trained on.
- **A command-line entry point.** Web UI only, like the image upscale.
- **Upscaling during extend.** Extend stays at the source size; upscale the finished clip.

## Evidence

Checked against `main` at 17686fa and the vendored LTX packages, on 2026-10-13.

### LTX stage 2 on a finished video

| Check | Result |
|---|---|
| What stage 2 needs | `_stage2(stage1, video_upscaled, …)` reads `latent_dims`, `video_embeds`, `audio_embeds`, `audio_tokens`, `relay_mask`, `x0_model`, `resolved_images` and `nag` from the `Stage1Result`. All of them can be built without running stage 1: the latent from the VAE encoder, the embeddings from the prompt encoder |
| Refine strength | `STAGE_2_SIGMAS` is `[0.909375, 0.725, 0.421875, 0.0]`. `stage2_steps` keeps the tail, so 3 steps start at 0.91 (most new detail), 2 at 0.725 and 1 at 0.42 (closest to the source). LTX 2.5 packs use their own table |
| Size rules | The source is the half-size input: its sides must be multiples of 32, which every LTX render is. Output is exactly 2× on each side (`s` 768×448 → 1536×896) |
| Frame count | Unchanged. LTX renders and extends are 8k+1 frames, which the VAE takes as is |
| Long clips | Stage 2 holds the whole clip's tokens. A 10 s extended `s` clip (241 frames) is 31 latent frames × 48×28 tokens ≈ 42k tokens at full size. `VideoModalityTiler` splits that into temporal and spatial tiles. Prompt Relay refuses tiling, which doesn't matter here: the upscale uses one prompt |
| Audio | Stage 2 also refines audio tokens. The upscale encodes the source's audio for the pipeline and then muxes the **original** audio track into the output, so the sound is unchanged |
| Memory | The DiT and VAE are the ones a two-stage generation of the same output size loads; stage 1 is skipped and a full-length encode is added. `clip_memory_fit(upscale=True)` measures a two-stage generation; the spike checks it is a safe upper bound |

### Why not SeedVR2

The earlier version of this proposal planned a chunked clip loop around mflux 0.22's SeedVR2 modules. Research before the spike ruled it out:

| Finding | Source |
|---|---|
| mflux's SeedVR2 VAE is single-frame only. `Upsample3D` trims extra frames only when `T == 1`, so n+1 latent frames decode to 4n+4 frames instead of 4n+1, misaligned with the input. `SeedVR2LatentCreator.create_condition` builds its mask with `T = 1` and fails on clips. `apply_color_correction` is a per-frame Lab transfer only | Installed mflux 0.22.0 |
| mlx-gen (MIT, an mflux fork) has a working SeedVR2 video path, measured on an M5 Max with 128 GB: 2× of a 320×240, 149-frame clip peaked at 34.4 GB (3B) and 44.3 GB (7B) and took 454–539 s (about 3–3.6 s per frame). Its safe default is 1× only and it refuses 2× video without `--force-unsafe-video-memory`. At 480×360 it produces "one … corrupted frame in every latent group of four" | [mlx-gen upscaling doc](https://github.com/lpalbou/mlx-gen/blob/main/docs/upscaling.md) |
| Our smallest real case, `s` 768×448 → 1536×896, has over 4× the output pixels of mlx-gen's test, so it can't fit a 32 GB Mac's 26.8 GB budget without spatial tiling | Derived from the row above |
| The working low-memory design is mlx-seedvr2-swift's: Lanczos enlarge first, SeedVR2 at 1:1 in spatial tiles, the causal VAE's state carried across chunk joins, chunks cut at scene changes, chunk length lowered to fit memory. That's a full port, weeks of work | [mlx-seedvr2-swift](https://github.com/xocialize/mlx-seedvr2-swift) |
| The reference ComfyUI node runs on PyTorch MPS; one report has a 16 GB Mac mini taking 35 minutes for a quarter-second clip | [numz node](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), [issue #465](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/issues/465) |

SeedVR2 for images is unaffected and stays as shipped.

## Proposed change

On macOS, the viewer's Upscale menu shows **2×** for videos, with the output size and, when `clip_memory_fit` says the clip is tight or too large, the high-memory marker and warning. The job encodes the video, upsamples it 2× with LTX's spatial upsampler, refines it with the video's own model in a few distilled steps (tiled over time for long clips), decodes it, keeps the original audio and frame rate, and records the source and the upscale settings.

Build it in one worktree, with a commit per step. Each step leaves the app working.

### 0. Spike, with recorded results

A script outside the app that builds a `Stage1Result` from an encoded video and runs `_upsample_latent` → `_stage2` → decode on `DistilledPipeline`. Four app renders with `ltx-4`: a 5 s `s` clip, a 10 s `s` clip made with two extends, a face close-up at `s`, and a 5 s `m` clip. On a 32 GB Mac with `--low-memory`; `ltx-8` and a 64 GB Mac if available. About half a day to a day, most of it machine time.

| Gate | Pass when |
|---|---|
| Fit | 2× of the 10 s extended clip finishes within the budget on a 32 GB Mac. If it doesn't, record the smallest `tile_count` (temporal first, then spatial) that makes it fit, and the cost in time |
| Time | Record seconds per output second at 5 s and 10 s, and the extra time tiling adds |
| Strength | A blind review of 1, 2 and 3 refine steps against a bicubic upscale picks the default. Faces must stay the same person. One step should be visibly better than bicubic, or the feature isn't worth it |
| Extend joins | The joins of the extended clip look no worse after the upscale |
| Prompt | At the chosen strength, the recorded scene and prompt against an empty prompt. If they look the same, videos without a recorded prompt are fine with an empty one |
| Encode | Encoding the full 10 s source at once fits; if not, encode in temporal chunks of 8k+1 frames, the way the VAE's causal frames allow |
| Estimate | `clip_memory_fit(upscale=True)` at the output size is at or above the measured peak. If not, record the term it needs |

Write the numbers, the default step count and the tiling rule into this proposal's Evidence before step 1.

### 1. Backend

- `core/video_types.py` (or next to the extend types): a frozen `VideoUpscaleSource` with the path, probe, size, frame count, frame rate and colour tags.
- `backends/mac/video.py`: `LtxVideoBackend.upscale_video(model, source, *, prompt, seed, steps, output_path, step_callback)`. It loads `DistilledPipeline` (the same load as the in-run upscale, plus `tile_count` from the spike's rule for the clip's length), encodes the video and audio, builds the stage-1 result from the encoded latent and the prompt embeddings, runs `_upsample_latent` and `_stage2` with `stage2_steps=steps`, and decodes through the streamed tiled decode. Progress events reuse `_TwoStageProgress`'s stage-2 phase. It refuses a model without an upsampler file and the diffusion video decoder, like extend.
- The original audio is muxed onto the decoded video with the extend splice helpers (`video_splice.source_audio_chain`), in a work folder beside the output that the gallery skips; the file is moved into place only on success.
- `backends/__init__.py`: `supports_video_file_upscale()`, true on macOS. No other module branches on platform.
- Tests, mocked: the stage-1 result carries the encoded latent's dims and the prompt embeddings; `_stage2` gets `stage2_steps`; the tile count follows the clip length; refusals for a missing upsampler and the diffusion decoder; audio is muxed only when the source has it; a failure leaves no file.

### 2. Runner and recorded settings

- `utils/video_upscale.py`, pure: the output size (2× each side), the step default, and the tiling rule from the spike.
- Config: a `video_upscale:` block with `steps` (default from the spike, 1–3), validated like the other blocks.
- `video_upscale_runner.run_video_upscale(backend, model, request, *, progress_callback, skip_signal)`: emits the same events as the image `run_upscale` for a one-item batch. Quit leaves no file.
- `utils/provenance.py`: `build_video_upscale_payload(...)` starts from the source's recorded payload (prompt, scene, model, seed, LoRAs, `source_video`, `extend_frames`), sets `workflow: "upscale"`, the output width and height, the same `frame_count`, `generation.upscale: {method: "ltx", factor: 2, steps}`, and a `source` block `{path, width, height, workflow}`. `source.workflow` is the source's own workflow, so Reuse regenerates it, and an upscaled extension reuses as Extend from the original's source. Old video records with `generation.upscale` and no `method` stay the in-run upscale.
- Model, prompt and seed: the source's recorded model, LoRAs, scene and prompt, and seed. A video with no recorded model uses the default video model and the prompt rule from the spike.
- Tests: the payload keys; an upscaled extension's `source.workflow` is `extend`; extend still refuses the upscaled video; old records read unchanged; config validation.

### 3. Web server

- `gallery.py`: `upscale` for a video is one 2× option with the output size and, from `clip_memory_fit(upscale=True)`, a `warning` when the fit is tight or too large; `null` when `supports_video_file_upscale()` is false, the size isn't recorded, or the model has no upsampler.
- `POST /api/upscale` for a video asset: only factor 2, no `method` choice. The route probes the source (cached by file identity, like extend's inspection), refuses it with the first source problem, and plans the request. A warned size queues like any other.
- `WebRunner.submit_video_upscale_job`: job type `upscale` with `media_type: "video"`, loading the LTX model the way video jobs do.
- Tests: the gallery JSON on supported and unsupported backends, without a recorded size and without an upsampler; a warned option; the route refuses factor 4 and an unreadable source; the job queues.

### 4. Web UI

- `assetActions.ts`: videos get the server's single 2× option; `upscaleFactors` stays images only. The menu shows one **2×** item with its size, marker and tooltip.
- `AssetViewer.svelte`: X then 2 works on videos; a warned size starts with its toast. X then 4 on a video does nothing, as for any factor without an option.
- Job card: "Upscale 2×", the output size and the stage-2 progress bar.
- Run `make frontend-build` and commit the rebuilt SPA.
- Vitest: a video shows only the 2× item; none when `upscale` is `null`; X then 2 sends the request and X then 4 sends nothing.

### 5. Docs and changelog

- `docs/guides/video.md`: an "Upscale an existing video (Web UI)" section next to "Video Upscale & Audio": the generate-small, extend, then upscale workflow; 2× only; time and memory from the spike; macOS only; audio and frame rate kept; that an upscaled video can't be extended (extend first, upscale last).
- The `video_upscale` key in the video guide's configuration snippet, and `generation.upscale.method: "ltx"` in the embedded-settings list.
- `docs/development.md`: the new modules in Project Structure.
- A `CHANGELOG.md` entry under `[Unreleased]`.
- Move this proposal to `archive/` and update the index.

## Alternatives considered

- **SeedVR2 through a clip loop over mflux's modules** (the earlier plan). Needs a fixed decoder and condition, and doesn't fit a 32 GB Mac at our sizes without a full tiled, streaming port. See [Why not SeedVR2](#why-not-seedvr2).
- **Depend on mlx-gen for SeedVR2 video.** MIT and working, but it refuses 2× video by default for memory reasons and needed 34–44 GB at a quarter of our smallest output size. Worth an hour's benchmark with its CLI only if the LTX result disappoints.
- **Frame-by-frame SeedVR2 `upscale_image`.** Already in the app, but it flickers: the model never sees neighbouring frames, and each frame takes about 25 s at 1408×896.
- **LTX's Pixel Spatial Upscaler IC-LoRA.** Upstream's dedicated video upscaler LoRA; its weights are gated. Revisit if the plain stage-2 refine falls short.
- **Allow extend together with the in-run upscale.** Every extend would run at full size, which is slower and needs more memory at every step instead of once at the end.

## Open questions

None. The spike sets the step default, the tiling rule and the prompt fallback.
