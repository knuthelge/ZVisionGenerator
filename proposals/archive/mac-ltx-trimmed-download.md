# Download only the LTX files macOS uses

**Status:** Done (v0.13.0, 2026-10-25). Shipped as the commits "download only the files macOS LTX loads, at pinned revisions" (pinned, trimmed downloads) and "extend with the clip's own model when its downloads are present" (CLI Extend keeps the clip's model), and verified on a Mac (see [Mac results](#mac-results)).

## Problem

On macOS, the first LTX-2.3 render downloads the whole MLX repo of its quantize level. Those repos hold three full transformers and two copies of the distilled LoRA, but the app loads one transformer and neither LoRA. `ltx-8` downloaded 87.5 GB for the 29.75 GB it uses, and `ltx-4` 59.7 GB for the 20.48 GB it uses; the first run of each also fetched the 8.07 GB Gemma 3 text encoder, so 95.6 and 67.8 GB in all. That is close to 100 GB of wasted disk and download time for a user with both levels, on Macs where internal storage is often the tightest limit. Unpinned repo ids made it worse: every missing file sent the vendored loader back to the Hub for the whole repo, and the first Mac test run filled its disk that way twice.

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
- The LTX-2.5 pack (`MarceloFSO/ltx-2.5-mlx-q4-fast`, 30.50 GB) is mostly used files. Without the files the app never loads (the temporal upscaler and the other files it never opens) it is 28.76 GB, a saving of 1.73 GB. It gets the same treatment, but the saving there is small.

## Design as built

- **Pinned revisions.** The q4 and q8 LTX-2.3 packs and the Gemma 3 text encoder are pinned to fixed revisions in `config.yaml` (`mlx_base`, `mlx_text_encoder`); the `ltx-4` and `ltx-8` aliases read the same pins. The LTX-2.5 pack was already pinned.
- **Literal file lists.** Each pack lists the files the vendored loader reads in `mlx_files` in `config.yaml`: the newest versioned distilled transformer, `connector`, the two VAEs, `audio_vae`, `vocoder`, the x2 spatial upsampler and its config, and the `*.json` configs; a 2.5 pack adds its bundled Gemma 4 text encoder, tokenizer files and duration head. A literal list fails loudly when the loader needs a new file, where an ignore list would silently download whatever upstream adds. After changing a pin, the list of the new revision is checked again; the MLX contract test checks the lists against the names the vendored loader opens.
- **One list for the run, the plan and the Models page.** The pre-run download, the download plan and the "downloaded" check (`video_model_downloaded`) all read the same list, so a trimmed folder reads as complete and the Models page dialog shows the size that a run fetches.
- **Every reference resolves to a folder.** A Hub reference is downloaded with the allow list at its pin and passed to the pipeline as a local folder, which skips the vendored `snapshot_download`. Gemma 3 is passed as a folder too.
- **Typed repo ids use the alias.** A known repo id typed on the CLI (`dgrauet/ltx-2.3-mlx-q4`, or an LTX-2.5 pack id) is canonicalized to its alias, so it gets the pin and the list instead of the whole repo. An unknown or ambiguous repo, and an unpinned user `mlx_base`, keep the old behaviour.
- **Converted checkpoints.** The converter's base files are the pack's list minus its transformers, so it only links files that are downloaded.
- **Extend.** The CLI keeps the clip's own model when its downloads are present, instead of falling back to the default video model.

For the user: the first LTX render, or the Models page download, fetches 20.48 GB (`ltx-4`) or 29.75 GB (`ltx-8`) plus 8.07 GB for Gemma 3 once, so about 28.6 and 37.8 GB on a first run, instead of 67.8 and 95.6 GB. `ltx25-4` fetches 28.76 GB. Nothing else changes.

An existing full download keeps working: the folder simply holds more files than needed. Freeing that space is the user's choice, by deleting and downloading the model again from the Models page. No migration.

## Mac results

On an M5 with 32 GiB (2026-10-25), each model deleted and downloaded again from the Models page:

| Model | Dialog | Downloaded | Identity check |
|---|---|---|---|
| `ltx25-4` | 28.8 GB, 1 part | 16 files, exactly the listed ones | Text to video, image to video and 2× upscale frames identical to the full-repo run |
| `ltx-4` | 28.5 GB in 2 parts | 20.5 + 8.1 GB, 10 files plus Gemma 3 (13 files) | Text to video and 2× upscale frames identical to the full-repo run; image to video and Extend complete, Extend keeps `ltx-4` |
| `ltx-8` | 29.75 GB (plus Gemma 3) | 10 files, about 9.5 min | Text to video renders (peak 22.67 GiB, swap Δ 0) |

The snapshots hold only the listed files. Unit tests cover the plan, the resolver, the typed-id canonicalization and the converter's base files.

## Alternatives considered

- **Patch the vendored `resolve_model_dir`.** Not allowed: `packages/ltx_*` is updated only by `make update-ltx`.
- **An ignore list (the dev and old transformers, the LoRAs) instead of an allow list.** Shorter, but a file that upstream adds to the repo later would be downloaded again without anyone noticing. An allow list fails loudly when the loader needs a new file, and the Mac test catches that.
- **Leave it.** Wastes 39–58 GB per level on every Mac that runs LTX-2.3.
