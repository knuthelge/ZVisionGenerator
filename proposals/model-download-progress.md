# Show model download progress in Web UI jobs

**Status:** Proposed (2026-10-10)

## Problem

When a Web UI job uses a model that isn't downloaded yet, the job card says "Loading klein9b..." for the whole download, which can be 16 to 35 GB for an image model and more than 60 GB for video. There is no size, progress, speed or ETA. From the browser, a download that takes 40 minutes looks the same as a load that has hung.

Nothing reports progress today because the download is a side effect of loading the model, inside a library:

- **Image on macOS:** mflux's `PathResolution` calls `snapshot_download(repo_id, allow_patterns=…)` when no complete cached snapshot matches the weight definition's patterns.
- **Image on CUDA:** diffusers `from_pretrained` (`backends/image_win.py:97`, `image_win_quant.py:136`) and our own `snapshot_download` of single components (`image_win_quant.py:166`).
- **Video on macOS:** the vendored LTX pipeline calls `snapshot_download(model_dir)` with no patterns (`packages/ltx_pipelines_mlx/ti2vid_one_stage.py:78`). The Gemma text encoder is a second repo (`utils/model_files.ltx_mlx_text_encoder_repo`).
- **Video on CUDA:** diffusers `from_pretrained` (`backends/video_diffusers.py:184`).

None of these calls back into our code. Web workers also set `HF_HUB_DISABLE_PROGRESS_BARS` and `TQDM_DISABLE` (`web/web_runner.worker_runtime_context`), so even the server terminal shows nothing.

The prompt enhancer already reports a `downloading` phase (`enhancer_loading`), but only as text, with no progress.

## Relationship to other proposals

[First run](first-run.md) §3 plans explicit downloads: `POST /api/models/download`, a download lane beside the job queue, a disk-space check, and progress on the job card when a generation has to download a model first. This proposal covers the job-card part and the shared download helper that the download lane would use. It can ship before the rest of first-run and doesn't depend on the Welcome panel or the ungated mirrors.

## Proposed change

### 1. Download before load, as its own step

Before the backend loads anything, the job runs an **ensure-downloaded** step:

1. Ask the backend which repos and file patterns it will read, through a new method `download_plan(model_ref, **load_options) -> list[DownloadTarget]`. A `DownloadTarget(repo_id, revision, allow_patterns)` is a frozen dataclass, and an empty list means a local path that needs nothing.
   - mflux: the weight definition's `get_download_patterns()`, or the Krea 2 variant patterns we already pass. They have to be **the same patterns** mflux uses, because `PathResolution` only treats a snapshot as cached when it matches them.
   - diffusers (image and video): the component folders from the repo's `model_index.json` (fetched with `hf_hub_download`, a few KB), plus the tokenizer and config files, without root-level single-file checkpoints. This matches `utils/model_files.model_weight_files`.
   - LTX MLX: the model repo (all files, as the vendored code downloads it) and the Gemma text encoder repo.
   - LoRAs from a repo: the single file `converters/lora_import` already resolves.
2. For each target that isn't complete in the cache, call `snapshot_download(..., dry_run=True)` (huggingface_hub 1.33) to get the files still missing and their sizes. If nothing is missing, skip to loading. No network beyond the metadata call, no events.
3. Otherwise download it with `snapshot_download(..., allow_patterns=…, tqdm_class=_JobProgress)` and report progress (§2).
4. Load the model as today. The backend finds the complete snapshot in the cache and doesn't download again.

The step is one pure planner per backend plus one shared function in a new `utils/model_download.py`, which the CLI and the first-run download lane can reuse. Backend selection stays in `backends/__init__.py`. The ensure step only calls the backend's `download_plan`.

If `download_plan` gets a pattern wrong, the backend downloads the missing files itself during the load, as it does today. The job still works. It just shows "Loading…" for that part without progress. Tests pin each backend's patterns against the library's own (mflux's `get_download_patterns()`), so drift shows up in CI.

### 2. Progress source

There are two ways to measure progress, and one spike decides which to use (see Open questions):

- **`tqdm_class`.** A `tqdm` subclass whose `update()` adds bytes to a shared counter instead of drawing a bar. This is exact and costs nothing, but only works if hf_xet's transfers report through it.
- **Polling the cache folder.** A thread sums the sizes of `blobs/*.incomplete` and finished blobs under `utils/model_files.huggingface_cache_repo_dir(repo)` once a second and compares the total with the one from the dry run. This works with any transfer method, as long as xet writes into the blob file as it goes and not all at the end.

Whichever is used, it sits behind one small `DownloadProgress` callback, so the event code doesn't need to know where the numbers come from.

### 3. Event contract

A new job event, `model_downloading`:

| Field | Meaning |
|---|---|
| `mode` | `image` or `video` |
| `model` | Label shown to the user (alias or name) |
| `repo` | Repo being downloaded (`unsloth/FLUX.2-klein-9B`) |
| `part`, `parts` | Index and count when a model needs more than one repo (LTX MLX + Gemma) |
| `bytes_done`, `bytes_total` | Bytes still to download in this run, from the dry run. `bytes_total` is `null` if unknown |
| `bytes_per_sec` | Smoothed speed over the last ~10 s |
| `download_eta_secs` | Remaining time, or `null` |

- At most **one event per second**. Each new `model_downloading` event **replaces the previous one in the job history** instead of appending, so a 40-minute download doesn't push the earlier events out of the 200-event cap, and a client that reconnects still gets the latest numbers.
- The ETA uses its own key. `_publish_event` carries `eta_secs` forward to later events (`record.last_eta_secs`), so a download ETA under that key would show up as the generation ETA.
- After the download finishes, the usual `model_loading` event follows.

### 4. Job card

The job card shows "Downloading klein9b — 12.4 / 34.7 GB · 48 MB/s · about 8 min", with a determinate progress bar, using the existing progress bar component. For multi-repo models it adds "(part 1 of 2)". When `bytes_total` is unknown, it shows the bytes and speed with an indeterminate bar. The enhancer's `downloading` phase can use the same event later.

The UI change gets an HTML preview for approval before it is built.

### 5. Stop during download

**Stop** works during a download. `_JobProgress.update()` (or the polling thread, through a flag checked between files) raises when the control signal is `quit`, and the job ends as cancelled. Partial blobs stay in the cache as `.incomplete` and resume on the next attempt. The job card says so: "Download stopped. It resumes where it left off next time."

### 6. Errors

A failed download (network, disk full, 401/403) fails the job with a message that names the repo and how much was downloaded. Making gated repos clear stays in [First run](first-run.md) §4.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "The Web UI shows download progress when a job needs a model that isn't downloaded yet."
- The Web UI guide's job card section mentions downloads and resuming.

## Alternatives considered

- **Hook into each library's download.** Patch mflux's `PathResolution` or diffusers' download path, or set a process-wide tqdm class. Rejected: it relies on private internals of three libraries plus vendored code we can't edit, and a process-wide hook would also catch the enhancer and other jobs.
- **Only poll the cache folder during load, with no pre-download.** It needs no patterns and works for every backend. Rejected as the main design because without the dry run there is no total, so no percentage and no ETA, and Stop can't interrupt a download running inside a library. It is still the fallback progress source in §2.
- **A separate download job type in the queue.** Rejected for the same reason as in first-run: a 35 GB download would block every queued generation. The first-run download lane covers downloads you start yourself.

## Open questions

- **Spike:** download a small public repo with `snapshot_download(tqdm_class=…)` on hf_xet 1.7 and log the `update()` calls. Do they arrive as bytes move, or once per file? Does `blobs/*.incomplete` grow as it goes? The answers choose the progress source in §2.
- Does a `KeyboardInterrupt`-style exception raised from `tqdm.update()` stop xet transfers cleanly, or do worker threads keep running until the file is done?
- Should the CLI (`ziv-image`, `ziv-video`) also use the ensure step, so it gets one tqdm bar per model instead of the libraries' bars? This would be cheap once the helper exists.

## Out of scope

- The download lane, `POST /api/models/download`, the disk-space check and the Welcome panel ([First run](first-run.md)).
- Download progress for models converted from checkpoints (`converters/convert_checkpoint.py`), which runs as its own Models-page action.
