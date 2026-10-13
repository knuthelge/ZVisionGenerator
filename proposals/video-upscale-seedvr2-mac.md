# Upscale existing videos on macOS with SeedVR2

**Status:** Proposed (2026-10-13). Not started. The image work it builds on has shipped; only the video part remains.

## Problem

Images in the gallery can be upscaled 2× or 4× after the fact, and on macOS the plain 2× and 4× items use SeedVR2 ([Upscale existing images on macOS with SeedVR2](archive/image-upscale-seedvr2-mac.md)). Videos can't be upscaled once they're finished. The only way to get a larger video is to choose the two-stage LTX upscale before generating, and that only works on LTX's own half-size render inside the same run. There's no way to enlarge a finished video, for example after picking the best take from a batch of small, fast renders.

SeedVR2 is a video model. mflux 0.22, which we pin, only exposes it for single images, but its transformer and 3D VAE take clips. The remaining work is a clip loop around mflux's modules, frame I/O, chunking, and the video side of the plumbing the image upscale already has. It needs no model port and no new dependency.

### Builds on (done)

The image work shipped the shared foundation. None of this is planned again here:

| Piece | Where | What video reuses |
|---|---|---|
| Backend and Protocol | `core/seedvr2.py` (`SeedVR2Upscaler`: `load_model`, `save_model`, `upscale_image`), `backends/seedvr2_mac.py` (`MfluxSeedVR2Upscaler`, bf16 precision guard, resident weights, buffer-cache cap) | Loading as is. The Protocol gains `upscale_clip` (step 2) |
| Platform gate | `backends/__init__.py`: `get_seedvr2_upscaler()`, `supports_seedvr2()` | The same gate. No separate video flag |
| Tier choice and memory | `utils/seedvr2.py`: `seedvr2:` config block, tiers 7B int8 and 3B int8, `estimate_seedvr2_bytes`, `pick_seedvr2_tier` (per Mac, at the 16 MP reference), `seedvr2_over_budget`, `seedvr2_tier_label` | The same tier, so a Mac keeps one SeedVR2 model. The estimate gains a frames term (step 1) |
| Sizes and warning | `utils/upscale.py`: `seedvr2_output_size` (each side scaled, rounded down to even), `seedvr2_options` (never disabled, `warning` when over budget, no size cap) | The output size as is. The options gain a per-chunk variant |
| Download and stored copy | `seedvr2_loader.py`: `load_seedvr2`, `build_seedvr2_setup`, the 8-bit copy in the hidden `models_dir/.seedvr2/` folder, deletion of the downloaded files; `WebUiConfig.seedvr2`; the Models page row | All of it, unchanged. Video adds no download step |
| Contract test | `tests/test_mflux_seedvr2_contract.py` (constructor, `generate_image`, `save_model`, model configs, file layout) | Extended with the internals the clip loop calls (step 2) |
| Menu and shortcut | `_seedvr2_upscale_json` in `web/gallery.py`; `upscaleEntries` in `assetMenu.ts`; the X then 2/4 chord in `AssetViewer.svelte`, which runs plain 2×/4× and toasts a warned size's warning; plain items labelled `2× → W×H` with a high-memory marker | The same items and chord for videos. Creative stays in the menu only, and images only |
| Job and planning | `POST /api/upscale` with `method`, `plan_seedvr2_upscale`, `WebRunner._load_seedvr2_with_progress` with its download and load phases, `workflows/seedvr2_stages.py`, output name `{stem}_{factor}x` | The route, the load-with-progress helper and the naming rule |
| Recorded settings | `utils/provenance.py`: `generation.upscale` with `method`, `model`, `quantize`, `softness`, `seed`; a `source` block that Reuse follows | The same keys. The video-detection rule needs one change (step 4) |

Decisions carried over: every tier is 8-bit and there's no quantize setting; there's no admission check and no size cap, only a warning; the softness default is 0.75; plain items are named 2× and 4×; videos get no creative upscale.

### Not in scope

- **Windows and Linux.** diffusers 0.41 has no SeedVR2 pipeline. `supports_seedvr2()` is false there, so videos keep no Upscale menu. CUDA belongs to a later proposal or [platform parity](platform-parity.md).
- **Changing the in-run LTX upscale.** `--upscale` and two-stage generation stay as they are.
- **Frame-rate upscaling** (temporal ×2). LTX's temporal upsampler could do it later, as a separate feature.
- **Videos from outside the output folder**, like the image upscale.
- **A command-line entry point.** The image upscale is Web UI only, and so is this. Add `ziv-video` support only if people ask.
- **Factors other than 2× and 4×**, and a creative upscale for videos.

## Evidence

Checked against this branch at 1bb2524 (main's extend, scene, clip-length and clip memory work included) and the installed mflux 0.22.0, on 2026-10-13.

### mflux SeedVR2 on clips

| Check | Result |
|---|---|
| Why the entry point is image-only | `SeedVR2.generate_image` reads one file (`SeedVR2Util.preprocess_image`), and `SeedVR2LatentCreator.create_noise_latents` hard-codes `shape=(B, 16, 1, H, W)` |
| The loop to copy | `generate_image`: `VAEUtil.encode` → `create_condition` → noise → one `transformer(txt, vid, timestep)` call with `SeedVR2TextEmbeddings.load_positive()` → `config.scheduler.step` (`seedvr2_euler`) → `VAEUtil.decode` → crop → `apply_color_correction(decoded, style)` |
| Transformer | `patch_in.py` patches along `T`, and `window.py` builds attention windows over `(t, h, w)` (`wt = ceil(min(t, 30) / …)`), so it accepts multi-frame latents |
| VAE | `Encoder3D`/`Decoder3D` with two temporal down and up blocks. Every conv is `CausalConv3d`, so a clip of 4n+1 frames encodes to n+1 latent frames. Spatial scale 8, 16 latent channels |
| Colour correction | Wavelet reconstruction, then a Lab histogram transfer against the bicubic-resized input. Applied per frame, it may flicker (spike) |
| Tiling | `VAEUtil` and `VAETiler` are image-only. The image path's flat memory floor up to about 8 MP comes from that tiling, so a clip path without tiling holds whole frames in the VAE |

### Community practice

The numz ComfyUI node (`ComfyUI-SeedVR2_VideoUpscaler`) is the reference implementation:

- Frames go through in batches of 4n+1 (`batch_size`, default 5; 21 and 33 are common).
- `temporal_overlap` (0–16 frames) is blended with a Hann window. Optional reversed `prepend_frames` steady the opening.
- VAE encode and decode tile spatially (1024 px tiles, 128 px overlap) only when memory runs out.
- On macOS it runs on PyTorch MPS, not MLX, with BlockSwap off.

### Baseline from the image spike

On a 32 GB M5 Mac (budget 26.80 GB): time is about linear in pixels, 20 s/MP on 3B int8 and 20–24 s/MP on 7B int8. Peak = weights + max(10.79 GB, slope × MP + intercept). At the image rate, frame by frame:

| Job | Output pixels | Time |
|---|---|---|
| 2× of a 5 s 704×448 clip (121 frames at 24 fps) | 1408×896 × 121 = 153 MP | about 50 min |
| 4× of the same clip | 2816×1792 × 121 = 611 MP | about 3.4 h |
| 2× of the xl preset (1408×896, 25 frames) | 2816×1792 × 25 = 126 MP | about 42 min |

The clip path may be cheaper per frame than this, since one transformer call covers a chunk. The spike measures it.

### Video code to reuse

| Need | Existing code |
|---|---|
| Probe size, rotation, frame rate, frame count, audio, colour tags, start offsets | `utils/video_probe.py`: `probe_video` → `VideoProbe` |
| Decode frames as `rgb24`, upright, streamed | `utils/video_tail.py`: `_build_tail_command` (ffmpeg auto-rotation on, `-fps_mode passthrough`), and its streamed read |
| Encode `rgb24` frames at a given size, frame rate and colour tags | `utils/video_splice.py`: `build_segment_command` (libx264, yuv420p, CRF 18, colour matrix from the source's tags) |
| Re-time the source audio onto the video's start | `video_splice.source_audio_chain`; the work folder beside the output (`SPLICE_DIR_PREFIX`, skipped by the gallery) and `_check_joined`'s probe of the result |
| Refuse formats the frame pipe can't round-trip | `utils/video_extend.extend_source_problems`: unsupported rotation, unreadable length or frame rate, variable frame rate, non-square pixels, more than 8 bits, HDR, interlaced, undecodable audio |
| Extending an upscaled video | Already refused: `extend_source_problems` reports `upscaled`, and the gallery shows `EXTEND_UPSCALED_REASON`, whenever `generation.upscale` is recorded |
| Video detection in recorded settings | `provenance._is_video_config`: a `frame_count`, or a `txt2vid`/`img2vid`/`extend` workflow. For videos, `_generation_block` keeps `generation.upscale` without a `method`, so an in-run LTX upscale isn't read as creative |
| Memory note pattern | The SeedVR2 item warning (`seedvr2_over_budget`, at most 1.0× the budget), not the LTX clip note (`clip_memory_fit`, `POST /api/video/memory-fit`): the Upscale menu stays consistent between images and videos |

## Proposed change

On macOS, the viewer's Upscale menu shows **2×** and **4×** for videos too, with their output size and, when a chunk may need more memory than the Mac has, the high-memory marker and warning. The job reads the video in overlapping chunks, upscales each chunk with SeedVR2 through mflux's modules, blends the overlaps, encodes the frames, copies the source's audio and frame rate, and records the source and the upscale settings in the new file.

Build it in one worktree, with a commit per step. Each step leaves the app working.

### 0. Spike, with recorded results

Run SeedVR2 on clips directly against mflux, without the app: a ~50-line loop that copies `generate_image` but encodes a `(1, 3, T, H, W)` clip and creates `T`-frame noise. Use four clips: a slow camera move, fast motion, a face close-up, and an LTX render with fine texture, each 121 frames at 704×448 and 960×512. Measure on the 32 GB Mac, and on a 64 GB or larger one if available.

| Gate | Pass when |
|---|---|
| Single-frame parity | With `T=1`, the same seed and softness, and mflux's VAE tiling off on both sides, the loop matches `generate_image` (PSNR ≥ 60 dB) |
| Temporal consistency | Mean absolute difference between consecutive output frames, divided by the same on the bicubic-upscaled source, is lower for the clip path than for frame-by-frame `generate_image`, and a visual check finds no flicker |
| Colour correction | Per-frame wavelet + Lab correction adds no visible flicker. If it does, compute the Lab statistics once per chunk |
| Softness | 0.75 (the image default) still looks best on video, against 0.5 and 0 |
| Chunk seams | With the chosen chunk and overlap (Hann blend in pixels), a visual check at the seams finds no jump. Record whether one noise tensor for the whole timeline, sliced per chunk, is needed |
| Memory and time | `mx.get_peak_memory()` and seconds per frame on **7B int8** and **3B int8** only, at 2× and 4×, with chunks of 5, 13 and 21 frames and overlaps of 0, 4 and 8 |
| Stability | No Metal command-buffer timeout at the largest chunk that fits |

Then:

- **Parity fails:** stop and fix the loop.
- **Clip path no more stable than per-frame:** the feature isn't worth a video path. Reconsider [the alternatives](#alternatives-considered).
- **2× of a 704×448 clip doesn't fit a 32 GB Mac's budget on its tier at chunk 5 without tiling:** step 2 adds a spatial tiler for `(B, C, T, H, W)` before the feature ships.
- **Metal timeout:** split the transformer call or the VAE decode, the way `_pre_denoise_flush` does for LTX, and measure again.

Write the numbers, the per-chunk memory fit, and the chosen `chunk_frames` and `overlap_frames` into this proposal's Evidence before step 1. Pick the default chunk as the largest that fits a 32 GB Mac's budget on its tier at 2× of a 704×448 clip.

### 1. Frame I/O, chunk planning and per-chunk memory (platform-free)

- `utils/video_frames.py`:
  - Move the `rgb24` decode command out of `video_tail.py` into a shared `build_decode_command(path, start)`, and add `iter_frames(path, width, height)`, which streams frames from one ffmpeg pipe. `video_tail` imports the command instead of keeping its own.
  - `VideoFrameWriter`: a context manager that pipes frames into `video_splice.build_segment_command` at the output size, with the source's frame rate and colour tags. On close it muxes the source's audio (stream copy; re-timed with `source_audio_chain` when the audio and video starts differ) and probes the result: frame count and size as planned, audio only when the source had it. It writes into a work folder beside the output with the `SPLICE_DIR_PREFIX` prefix (the gallery already skips it) and moves the file into place only on success, so a failure or quit leaves nothing behind.
  - Only one chunk is held at full size at a time.
- `utils/video_extend.py`: split the format checks out of `extend_source_problems` into `video_format_problems(probe, *, audio_ok)`, which both extend and upscale call. Extend keeps its own length, grid, size and `upscaled` checks.
- `utils/video_upscale.py`, pure:
  - `plan_chunks(frame_count, chunk_frames, overlap_frames) -> list[ChunkSpan]`: every span is 4n+1 long; a short final chunk is padded by repeating its last frame and the padding is dropped on write; a clip at or under `chunk_frames` is one chunk.
  - `blend_weights(overlap_frames)`: the Hann ramp, adding up to 1 across an overlap.
- `utils/seedvr2.py`: `estimate_seedvr2_clip_bytes(tier, megapixels, frames)`: the image estimate with a per-chunk term fitted in the spike. If the spike shows the image line scales with megapixels × frames, no new tier keys are needed; otherwise each tier gains one constant in `config.yaml`.
- `utils/upscale.py`: `seedvr2_clip_options(width, height, tier, *, chunk_frames, budget_bytes)`: `UpscaleOption`s with `seedvr2_output_size`, never disabled, and a `warning` when the per-chunk estimate is over budget.
- Config: a `video_upscale:` block with `chunk_frames` and `overlap_frames`, defaults from the spike, validated like `seedvr2:` (positive ints, `chunk_frames` of 4n+1, overlap smaller than the chunk). Model, quantization and softness come from `seedvr2:`. Whether a `max_megapixels` key is wanted is an open question.
- Tests: chunk spans (lengths, overlap, padding, a single short clip); blend weights; the clip estimate and the warning just over and at the budget; config validation; `VideoFrameWriter` with a mocked `subprocess.Popen` (audio mapped only when the source has it, a failed encode or a failed check leaves no file); the shared format checks still give extend the same results.

### 2. SeedVR2 clip backend and contract

- `core/seedvr2.py`: add `upscale_clip(model, frames, *, width, height, seed, softness) -> frames` to `SeedVR2Upscaler`, with frames as a uint8 `(T, H, W, 3)` array.
- `backends/seedvr2_mac.py`: implement it. Pre-blur by softness and resize the chunk with bicubic, pad to 16 px, encode with `model.vae` as `(1, 3, T, H, W)`, build the condition with `SeedVR2LatentCreator.create_condition`, create `T`-frame noise (sliced from a timeline-wide tensor if the spike says so), take one transformer step with `load_positive()` and the `seedvr2_euler` scheduler, decode, crop, and apply colour correction against the resized input. Add the 5D spatial tiler here only if the spike requires it; mflux's tilers can't be used.
- Extend `tests/test_mflux_seedvr2_contract.py` to pin, by introspection, what the loop calls: the `vae` and `transformer` attributes, `SeedVR2LatentCreator.create_condition(encoded_latent)`, `SeedVR2TextEmbeddings.load_positive`, `SeedVR2Util.apply_color_correction(content, style, …)`, `SeedVR2Transformer.__call__(txt, vid, timestep)`, and that `"seedvr2_euler"` is a registered scheduler.
- Mocked tests: `upscale_clip` sends 5D tensors with the right `T` to the mocked VAE and transformer, crops the padding, and returns one frame per input frame.

### 3. Runner and recorded settings

- `core/video_types.py`: a frozen `VideoUpscaleRequest` with the source path and probe, the source's recorded payload, factor, output size, tier model, softness, seed, `chunk_frames`, `overlap_frames` and the output path (`{stem}_{factor}x.mp4`, with a counter when taken, the image rule).
- `video_upscale_runner.run_video_upscale(upscaler, model, request, *, progress_callback, skip_signal)`: reads chunk by chunk from `iter_frames`, calls `upscale_clip`, blends overlaps, writes through `VideoFrameWriter`, and embeds the settings. It emits the same events as `run_upscale` does for a one-image batch, with one step per chunk, so the job card's bar moves per chunk. Quit is checked between chunks and leaves no file.
- `utils/provenance.py`:
  - `build_video_upscale_payload(...)` starts from the source's recorded payload, so its prompt, scene, model, family, seed, steps, LoRAs, reference image, `source_video` and `extend_frames` are kept. It sets `workflow: "upscale"`, the output width and height, the same `frame_count`, `generation` with `time`, `audio`, `output_format` and `upscale: {method: "seedvr2", factor, model, quantize: 8, softness, seed, chunk_frames, overlap_frames}`, and a `source` block `{path, width, height, workflow}`. `source.workflow` is the source's own workflow (or its source's, when the source is itself an upscale), so Reuse regenerates it, and an upscaled extension reuses as Extend from the original's source.
  - The video-detection rule: a video record keeps `method` when it is `seedvr2` and still gets no default. A video `generation.upscale` without a method stays the in-run LTX upscale. Add `chunk_frames` and `overlap_frames` to the kept keys.
- Memory: the model loads under the buffer-cache cap (already in `load_model`), and `release_accelerator_memory()` runs after the job, as for images.
- Tests: chunk loop order and blend with a mocked backend and I/O; events for success, quit and failure; the payload keys; an upscaled extension's `source.workflow` is `extend` and Reuse goes to Extend; an upscaled video's `upscaled` flag makes extend refuse it; old video records with `generation.upscale` and no method still read without one.

### 4. Web server

- `gallery.py`: `upscale` for a video is `{factors: [], seedvr2: …}` with `seedvr2_clip_options` from the recorded width and height, or `null` when `web_config.seedvr2` is `None` or the size isn't recorded. `_seedvr2_upscale_json` takes the options as an argument so images and videos share it.
- `POST /api/upscale` for a video asset: `method` must be `seedvr2` (`creative` is refused). The route probes the source (cached by file identity, like `extend_api`'s inspection), refuses it with the first `video_format_problems` message, and builds the request with `plan_seedvr2_video_upscale`, next to `plan_seedvr2_upscale`. A warned size queues like any other.
- `WebRunner.submit_video_upscale_job`: job type `upscale` with `media_type: "video"`. It loads through `_load_seedvr2_with_progress`, changed to take the model name instead of an image request, so the download and load phases are the same.
- Tests: the gallery JSON for a video on supported and unsupported backends and without a recorded size; a warned option; the route refuses `creative` for videos and an unreadable source; it queues the video job.

### 5. Web UI

- `assetActions.ts`: `seedvr2Options` returns options for videos too; `upscaleFactors` stays images only. The menu then lists only 2× and 4× for videos, with the same sizes, marker and tooltip, including the first-use download line.
- `AssetViewer.svelte`: the X then 2/4 chord is active when any plain or creative option exists, so it works on videos, and a warned size still starts with its toast.
- Job card: "Upscale 2×", the output size, the SeedVR2 model that ran, and a bar per chunk. The download and load phase lines already apply.
- Run `make frontend-build` and commit the rebuilt SPA.
- Vitest: a video with `seedvr2` options shows only the plain items; none when `upscale` is `null`; the chord sends `seedvr2` for a video; the job card shows chunk progress for a video upscale.

### 6. Docs and changelog

- `docs/guides/video.md`: an "Upscale an existing video (Web UI)" section next to "Video Upscale & Audio". It covers what SeedVR2 does to a clip, time and memory from the spike, macOS only, audio and frame rate kept, chunks and their warning, and that an upscaled video can't be extended (extend the original). It links to the image guide for the model choice and first-use download rather than repeating them.
- The `video_upscale` keys in the video guide's configuration snippet, and `chunk_frames`/`overlap_frames` in the embedded-settings list.
- `docs/development.md`: the new modules in Project Structure and the extended contract test.
- A `CHANGELOG.md` entry under `[Unreleased]`.
- Move this proposal to `archive/` and update the index.

## Alternatives considered

- **LTX latent upscale or Pixel Spatial Upscaler IC-LoRA on the finished video.** All the pieces are vendored, but it loads the 22B LTX transformer and Gemma, needs a prompt and several steps, follows LTX's frame and size rules, and changes detail more on footage LTX didn't make. Its IC-LoRA weights are gated. Keep it as an option if SeedVR2 is weak on LTX renders.
- **Frame-by-frame `upscale_image`.** No clip loop to write, and the app already has it. But it flickers, because the model never sees neighbouring frames. The spike uses it as the baseline to beat.
- **Contribute `generate_video` to mflux upstream.** Worth offering once the loop works, but the feature shouldn't wait for an upstream release.
- **numz's PyTorch implementation on MPS.** The most tested, but it brings PyTorch into a macOS stack that is MLX only, and it has known MPS allocator problems.
- **FlashVSR or DOVE.** No MLX port, and FlashVSR's speed depends on CUDA sparse-attention kernels.
- **GAN upscalers (Real-ESRGAN and similar).** Cheap, but per frame, so they flicker.

## Open questions

- **`video_upscale.max_megapixels`.** Images settled on no cap and a warning only, and per-chunk memory is bounded by `chunk_frames`, so a cap no longer protects memory. Proposed: drop it. Keep it only if a cap on output size per frame is wanted for time reasons (below).
- **4× and long clips.** At the image rate, 2× of a 5 s 704×448 clip takes about 50 minutes and 4× about 3.4 hours. Decide after the spike whether 4× is offered for videos, and whether the item's tooltip or the job card shows an estimated time (seconds per frame × frames) so a multi-hour job isn't a surprise.
- **A chunk over budget.** When the default chunk doesn't fit at a size, either warn only (as images do, smallest) or plan a smaller 4n+1 chunk that fits, trading some temporal consistency for no swapping. The spike's seam and memory numbers decide.
