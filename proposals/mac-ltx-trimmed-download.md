# Download only the LTX files macOS uses

**Status:** Proposed (2026-10-25)

## Problem

On macOS, the first LTX-2.3 render downloads the whole MLX repo of its quantize level. Those repos hold three full transformers and two copies of the distilled LoRA, but the app loads one transformer and neither LoRA. `ltx-8` downloads 87.5 GB for about 30 GB it uses, and `ltx-4` downloads 59.7 GB for about 20 GB. That is close to 100 GB of wasted disk and download time for a user with both levels, on Macs where internal storage is often the tightest limit.

## Evidence

File sizes from the Hub (2026-10-25):

| File | `dgrauet/ltx-2.3-mlx-q8` | `dgrauet/ltx-2.3-mlx-q4` | Used |
|---|---|---|---|
| `transformer-distilled-1.1.safetensors` | 20.6 GB | 11.3 GB | yes (`ltx_mlx_transformer_file` picks the newest versioned file) |
| `transformer-distilled.safetensors` | 20.6 GB | 11.3 GB | no |
| `transformer-dev.safetensors` | 20.6 GB | 11.3 GB | no (only the vendored two-stage, HQ, a2vid and IC-LoRA pipelines load it; the app uses `DistilledPipeline`) |
| `ltx-2.3-22b-distilled-lora-384{,-1.1}.safetensors` | 2 × 7.6 GB | 2 × 7.6 GB | no (`--distill` takes the LoRA from `Lightricks/LTX-2.3`) |
| `spatial_upscaler_x1_5_v1_0.safetensors` | 1.1 GB | 1.1 GB | no |
| `temporal_upscaler_x2_v1_0.safetensors` | 0.3 GB | 0.3 GB | no |
| `connector`, VAEs, `vocoder`, `audio_vae`, `spatial_upscaler_x2_v1_1`, configs | ~9.1 GB | ~9.1 GB | yes |

- The vendored loader downloads by repo id without patterns (`packages/ltx_pipelines_mlx/utils/_orchestration.py`, `resolve_model_dir`), and `backends/mac/download_plan.video_download_plan` mirrors it with an unfiltered `DownloadTarget`, so the Models page and the pre-run download also take the whole repo.
- A local folder skips the vendored download (`resolve_model_dir` returns an existing path). `backends/mac/video._resolve_pinned_model` already uses that for pinned references: it resolves the snapshot and passes the folder in.
- The LTX-2.5 pack (`MarceloFSO/ltx-2.5-mlx-q4-fast`, 30.5 GB) is mostly used files; only the temporal upscaler (0.3 GB) and possibly one VAE variant are extra. It gets the same treatment, but the saving there is small.

## Proposed change

- Resolve every Hub LTX reference on macOS to a local folder before it reaches the vendored pipeline, by downloading with an allow list. This extends what `_resolve_pinned_model` does for pinned references to all references.
- Keep one allow list, next to the file-name constants in `utils/model_files.py`: the newest versioned distilled transformer (falling back to the unversioned one when a repo has no versioned file), `connector`, the VAE encoder and decoder names the loader looks for, `audio_vae`, `vocoder`, the x2 spatial upsampler names in `_LTX_MLX_UPSAMPLER_STEMS`, the bundled text encoder of a 2.5 pack, and the `*.json` configs.
- `video_download_plan` uses the same list, so the Models page, the pre-run download and the run all agree, and the "downloaded" check reads a trimmed folder as complete.
- The converter's `_BASE_FILES` drops `temporal_upscaler_*`, so it only links files that are downloaded.

For the user: the first LTX render, or the Models page download, fetches about 30 GB (`ltx-8`) or 20 GB (`ltx-4`) instead of 88 or 60 GB. Nothing else changes.

An existing full download keeps working: the folder simply holds more files than needed. Freeing that space is the user's choice, by deleting and re-downloading the model from the Models page. No migration.

## Test

- Unit: the plan for a Hub LTX reference lists exactly the allow-list patterns; the resolver downloads with them and passes a folder to the pipeline; a pinned and an unpinned reference both resolve; a local folder is passed through unchanged.
- Unit: the allow list matches every name the vendored loader opens for text-to-video, image-to-video, Extend and the 2× upscale (checked against the constants it reads, not against a copied list).
- Mac: on a clean cache, download `ltx-4` from the Models page, check its size (~20 GB), then render text-to-video, image-to-video, Extend and a 2× upscale at both levels and with LTX-2.5. Add this to the pending Mac test run rather than a separate trip.

## Alternatives considered

- **Patch the vendored `resolve_model_dir`.** Not allowed: `packages/ltx_*` is updated only by `make update-ltx`.
- **An ignore list (the dev and old transformers, the LoRAs) instead of an allow list.** Shorter, but a file that upstream adds to the repo later would be downloaded again without anyone noticing. An allow list fails loudly when the loader needs a new file, and the Mac test catches that.
- **Leave it.** Wastes 39–58 GB per level on every Mac that runs LTX-2.3.
