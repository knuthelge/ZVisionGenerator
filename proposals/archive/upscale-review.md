# Review before upscaling

**Status:** Done (v0.13.0, 2026-10-12). Shipped in `a45e8e9` (feature) and `2741388` (docs): images on every platform, video on macOS through the `_stage1` wrapper in `video_mac`. The manual real-model checks on macOS (step 2) are done. Video review on CUDA moved to [platform parity](platform-parity.md) step 3, since it builds on the two-stage upscale there; until then `supports_video_upscale_review()` is false on CUDA.

## Problem

In-run upscale renders small first and spends most of its time on the full-size pass:

- **Images:** `--upscale 2|4` renders at half or quarter size, then `upscale_stage` enlarges the image and refines it with img2img at full size.
- **Video:** `--upscale 2` runs stage 1 at half resolution, upsamples the latent 2× and refines it at full resolution (stage 2), then decodes.

When the small result is a dud, the full-size pass runs for nothing. Today the only way to catch it is to watch the blurry live preview and press **Next** in time.

The small result already shows what the final output will be. The full-size pass adds detail. It doesn't change composition or motion. Video audio is refined too in stage 2 (`DistilledPipeline._stage2_audio_state` re-noises stage 1's audio and refines it with the video), so the sound can change slightly.

How much time this saves depends on the size, so step 1 and the video spike each record the small render's share of the time for one typical setting.

## Proposed change

A **Review before upscaling** option, the same for images and video, in the Web UI and the CLI. With it on, each generation pauses after the small render and shows it:

- **Approve** continues with the upscale. Everything stays in memory, so the output is the same as without review.
- **Discard** skips this generation. No upscale, no output file, and the preview is gone.

In a batch, every generation pauses. A paused job stays paused until the user decides, including when the browser is closed. **Stop** still ends the job.

The keys are the same everywhere: **Y** approves and **N** discards. N already means "skip this one" (Next), so it keeps its meaning.

### Not in scope

- **Review without upscale.** There's no small render to stop at.
- **Approving later.** A paused generation can't be kept after the job ends. For images, the gallery **Upscale** action already covers that. For video it would need the stage-1 latent saved to disk. See [Alternatives](#alternatives-considered).
- **Upscaling an existing video.** That's a separate feature.

## Shared design

### Decision: a review state on `SkipSignal`

`SkipSignal` (`utils/interactive.py`) gets a review state that is separate from Pause: its own decision slot and event, plus `wait_for_review() -> "approve" | "discard" | "skip" | "quit"`.

- **Start:** if Quit is already queued, it returns `"quit"` at once. A queued `skip` (Next pressed during the small render) is not cleared: it returns `"skip"` at once, stays queued, and ends the review without an upscale, as Next skips an image at any other time (a decision already made wins). A queued `pause` or `repeat` stays queued, and the existing handling after the image applies it, whether the render was approved or discarded.
- **Wait:** it waits in a loop with a short timeout until a decision is set, so Ctrl-C stays responsive on Windows too.
- **CLI keys:** while a review waits, `_on_key` maps `y` to approve, `n` to discard and `q` to quit, before its any-key resume check. Other keys are ignored. Enter does nothing, so a stray keypress can't discard.
- **Web:** `set_review_decision(...)` sets the slot and wakes the waiter. `queue_action("quit")` also wakes a review waiter, so the existing Stop path works.

### Runner contracts

Each runner gives the workflow or backend a `review` callable. That callable shows the small render, emits the events, waits, and reports the decision in whatever way that runner already handles:

1. If the job is in the CLI, write the render to a temporary file outside the output folder and print its path: a PNG for images, the mp4 for video. Then print `Approve upscale? [y] yes  [n] discard  [q] quit`.
2. Emit `job_paused` with a `review` payload (`{ "iteration": ran_iterations }`). For images, the small render goes in as the event's top-level `preview`.
3. `skip_signal.wait_for_review()`.
4. Emit `job_resumed`, and delete the temporary file.

**Images:** `review(image) -> bool`. `review_stage` returns `StageOutcome.skipped` when it gets `False`. On Quit, `wait_for_review` leaves `quit` queued and returns `False`. The runner's existing skipped branch (`image_runner.py`) then emits `generation_finished` `skipped`, sees the queued Quit, and emits `batch_cancelled` before `_QuitBatch`. No new exception path is needed. For a review discard, the skipped branch doesn't print the "Generation skipped by workflow stage" warning.

**Video:** the closure raises `_VideoCancelled` on Quit, which is already handled. On discard it raises a new private `_VideoDiscarded`. `run_video_batch` catches it before `except Exception` and emits `generation_finished` with `status: "skipped"`.

A discard isn't a failure, and an all-discarded batch ends in `batch_completed`. Discards count toward the completed items and the ETA with the time of their small render (the wait excluded). That's real time spent, so the ETA stays honest.

**Timing:** the wait isn't generation time. For video, the stage subtracts it from `generation_time` around the backend call, before `log_video_stage` records it. For images, recorded times already add up only stage compute time. The runner subtracts the wait from its wall-clock `generation_time` in `generation_finished` and from `image_times`.

Reusing `job_paused`/`job_resumed` means the job status, the SSE event list and the snapshot's `paused` flag all work unchanged.

The output file and its recorded settings are the same as without review. Review isn't recorded.

### Web UI

- **Form:** a **Review before upscaling** checkbox under upscale, for images and video, shown only while upscale is on. Help text: "Pauses after the small render. Approve to upscale it or Discard to skip it." For video it adds: "Sound may change slightly after upscaling." Form field `review`. Contract control IDs are `image_upscale_review` and `video_upscale_review`. `build_workflow_contract` hides `video_upscale_review` where `supports_video_upscale_review()` is false, the same way it hides `video_audio`.
- **Request:** `review: bool`. The server ignores it when upscale is off. It refuses it with 400 when it's set for video on a platform without support, since the form never sends it there. The hand-built `argparse.Namespace` in `server.py` and `conftest._make_args` get `review=False`.
- **Controls:** a job with review on lists `approve` and `discard` in its supported controls. `queue_job_control`:
  - accepts `approve`/`discard` only while `record.review` is set;
  - accepts only `approve`, `discard` and `quit` during a review, and refuses `resume` explicitly;
  - clears `record.review` under `record.lock` on the first decision. A second decision, from another tab or a double click, gets a 409.
- **Snapshot:** `review` (`null` or `{ iteration }`), so a reconnecting browser shows the review again. For images the render is the job preview. For video it's the route below. Filesystem paths never go into events or snapshots.
- **Layout (agreed 2026-10-11):**
  - **Stage:** while a review waits, an overlay covers only the Workspace preview area, with a dark tinted backdrop. The settings pane, History strip, navigation and menus stay visible and usable. On the backdrop, centred, sit the small render, a line "Review 3 of 8 · 832×1216 → 1664×2432" (the small size, then the upscaled size), and **Approve (Y)** (primary) and **Discard (N)**, nothing else. The overlay is not modal and does not trap focus; Approve takes focus when it appears. The job card, mascot and queue panel keep their running-job layout under the backdrop, so the card's Cancel is covered; Esc twice (the card's own shortcut) still stops the job.
  - **Video player:** stops on the first frame, and the user presses play. It has a sound control only when Audio is on.
  - **Settings:** the setting is a **Review** sub-row inside the Upscale section, shown while upscale is on. For images it sits above Denoise, for video under the factor. With review on, the section summary reads "2× · Review".
  - **Attention:** while a review waits, the Workspace nav item shows a "Review" badge and the browser tab title starts with "⏸ Review ·". Both clear on the decision.
- **Job card:** the store reads `review` from `job_paused` and the snapshot. It clears it on `job_resumed`, `generation_finished` and terminal events, and takes the image from the `preview_version` that the paused event bumps.
  - While `review` is set, the card shows the message "Review: approve to upscale, or discard." The render, **Approve** (primary) and **Discard** sit in the overlay on the Workspace preview only, not on the card.
  - `jobControls` offers only **Stop** on the card. Resume, Next and Repeat are hidden.
  - The review UI takes precedence over the `'finishing'` activity that the `review` stage name would otherwise give.
  - Shortcuts: **Y** approves and **N** discards during a review. P and R do nothing. Both are added to `APP_SHORTCUTS`.
- **Queue:** jobs queued behind a waiting review wait. Actions that need the generation slot (on-demand enhance, quantize, model and LoRA deletes) are refused while the job is active, so they are unavailable until it ends. Approve starts at once because the model stays loaded.
- **Shutdown:** `WebRunner.shutdown()` sends Quit to the running job and wakes it. Otherwise a waiting review would hang `ziv ui` on Ctrl-C. The same fix covers today's image Pause.
- **Reuse settings and Copy CLI command** carry `review` (`jobSettings.ts`, `cliCommand.ts`).

### CLI

- `--review` for `ziv-image` and `ziv-video`. It is an error without `--upscale` and when stdin isn't a terminal.
- `ziv-image` already runs the key listener.
- `ziv-video` starts a `SkipSignal` only with `--review`, after `ensure_ffmpeg()` and `run_preflight` (both can prompt with `input()`), and stops it in a `finally` so the terminal is restored after Ctrl-C. cbreak mode keeps signals, so prints and Ctrl-C behave as today. `run_video_batch` honours `q`. For video the listener ignores `n`, `p` and `r` outside a review instead of printing "queued", and the batch header prints `Commands: [q] quit`.

## Images

The small render is a decoded PIL image in `artifacts.image` after `text_to_image_stage`, and the model is loaded. Nothing has to be split.

- `build_workflow` adds `review_stage` between `text_to_image_stage` and `upscale_stage` when `args.upscale and args.review`.
- **Preview:** `_publish_event` already pops a top-level `preview` from any event, encodes it as the job JPEG and bumps `preview_version`. The preview survives `job_resumed` and is cleared by `generation_finished`, so no new route is needed. The JPEG is a copy, and `artifacts.image` is untouched.
- The preview is the raw small render, before contrast, saturation and sharpening, which run after upscale.
- `upscale_save_pre` is unchanged. It runs inside `upscale_stage`, so only approved images get it.

## Video

### Backend seam (macOS)

The backends get one optional keyword argument, `review: Callable[[Path], None] | None`, which is valid only with `stage1_steps`. `video_mac` keeps calling `DistilledPipeline.generate_and_save` with or without review. With review, it installs a per-call wrapper over the instance's `_stage1` (restored in a `finally` with `del pipe._stage1`). The wrapper:

1. calls the original `_stage1` and keeps its result unchanged,
2. unpatchifies `video_tokens` with `latent_dims`, loads the decoders and decodes video and audio to a file in a `tempfile.TemporaryDirectory` outside the output folder,
3. frees the decoders again (as `_generate_one_stage` does),
4. calls `review(path)` and returns the stage-1 result.

`generate_and_save` then continues as normal. Its own low-memory frees before the final decode still apply, so the final decode has no new memory cost. There is one code path, and only `_stage1`, the `Stage1Result` fields used, and `_load_decoders` need new pins in `tests/test_ltx_mlx_contract.py` and the "Internals we rely on" list.

**Side effect to neutralize:** under `low_memory`, `_decode_and_save_video` sets `self.dit = None` and `self._loaded = False`. During the preview that frees nothing, because `Stage1Result.x0_model` still holds the DiT. But it would turn off stage-2 sparse attention and break tiling, which both read `self.dit`. The wrapper saves `pipe.dit` and `pipe._loaded` before the preview decode and restores them after.

**Memory during the preview decode:** the DiT, VAE encoder, upsampler and VAE decoder are loaded at the same time, which is a new peak. The spike measures it. If it goes over today's peak on a 32 GB Mac with `--low-memory`, the fallback frees the DiT for the preview and reloads it on Approve. That fallback needs `pipe.load()` plus rebuilding `stage1.x0_model = X0Model(pipe.dit)`, and Approve then waits for the reload. The spike picks one before the Web work starts.

**Audio off:** the MLX pipeline always decodes audio, and `strip_audio_stage` removes it only from the final file. With Audio off, the review player is muted and has no sound control.

### Windows and Linux

Today's diffusers upscale isn't the two-stage flow ([platform parity](platform-parity.md), step 3). Review on CUDA builds on that step. Until then:
- `supports_video_upscale_review()` in `backends/__init__.py` (next to `supports_video_audio()`) returns false on CUDA;
- the form hides the option and the server and CLI refuse it;
- `video_diffusers` raises `ValueError` when it gets `review`, so a missed check can't be silently ignored.

Image review works on every platform.

### Video preview route

`GET /jobs/{id}/review?v=<iteration>` serves the mp4 while that review waits, and 404 otherwise. The path comes from the job record. It uses `Cache-Control: no-store` and the version in the query, so each video in a batch loads fresh. `FileResponse` handles range requests for the player.

## Steps

One worktree, one commit per step, `make check` green at each:

1. **Shared review mechanism and images.** The `SkipSignal` review state, the image runner `review` callable, `review_stage`, discard as `skipped`, timing, image CLI `--review` with the printed preview path. Web: the request key, image form checkbox, controls, latch, snapshot, shutdown fix, job card and Y/N shortcuts. **Done when** an image batch can be approved, discarded and stopped during a review from both the CLI and the Web UI, and a browser reload restores the review.
2. **Video spike (macOS).** The `_stage1` wrapper, measured on a 32 GB Mac with `--low-memory` and on a larger Mac without it. **Go/no-go:**
   - the peak (`mx.get_peak_memory`) during the preview decode and the final decode is at most today's no-review peak, or the reload fallback is chosen;
   - same seed with and without review gives identical frames and audio (`ffmpeg -f framemd5`). First confirm that two no-review runs match each other.
3. **Video backend, runner and CLI.** The `video_mac` wrapper, the `review` kwarg in the Protocol docstring, the `video_diffusers` refusal, `supports_video_upscale_review()`, `_VideoDiscarded`, the stage timing and `ziv-video --review`. **Done when** a macOS CLI video batch can be approved, discarded and stopped during a review, the framemd5 check is identical, and `video_diffusers` refuses `review`.
4. **Video Web UI.** The video form checkbox, the preview route, the player in the job card and the muted player when Audio is off. **Done when** the same works from the Web UI, and a reload restores the review player.
5. **Docs and changelog.** **Done when** the docs below are updated and `make check` passes.

## Tests

- `SkipSignal`:
  - `y`/`n`/`q` decide during a review and other keys are ignored;
  - Web decisions and `queue_action("quit")` wake the waiter;
  - a queued skip ends the review as `"skip"` and stays queued, a decision already made wins over it, and a queued pause or repeat survives;
  - an already queued quit returns at once.
- `review_stage`: approve continues to `upscale_stage`, discard returns `skipped` and runs no upscale or save. It isn't added without upscale or without review.
- Image runner:
  - `job_paused` with `review` and `preview`, then `job_resumed`;
  - discard gives `generation_finished` `skipped` without the skip warning;
  - quit during a review gives `skipped`, then `batch_cancelled`;
  - an all-discarded batch ends `batch_completed`;
  - the wait isn't counted in `generation_time`;
  - Pause queued during the small render applies after the approved upscale, and after a discard;
  - Repeat queued during the small render applies after a discard;
  - Next queued during the small render skips the image without an upscale;
  - a pending pause, repeat or next label on the card survives the review's pause and the decision.
- `run_video_batch`: discard emits `generation_finished` `skipped`, not `failed`. Quit cancels. The stage's `generation_time` excludes the wait.
- CLI (both): the printed preview path exists during the prompt and is gone after it. `--review` without `--upscale`, or with a non-terminal stdin, errors. `ziv-video` restores the terminal after an exception.
- `video_mac` (mocked pipeline):
  - with `review`, `generate_and_save` runs once and `review` gets an existing file after `_stage1`;
  - `pipe.dit` and `_loaded` are restored after the preview decode;
  - `_stage1` is restored after success and after a raise;
  - `review` without `stage1_steps` raises `ValueError`.
- `video_diffusers`: `review` raises `ValueError`.
- `web_runner`:
  - `approve`/`discard` are refused when no review is waiting, and `resume` is refused during one;
  - a second decision gets 409;
  - the snapshot carries `review` and no path;
  - `shutdown()` ends a waiting review;
  - the video route serves only while its review waits.
- Server: `review` is ignored without upscale and refused for video where it isn't supported. The workspace contract hides `video_upscale_review` there.
- Frontend:
  - the card shows the review for images and video, with only Stop besides Approve and Discard;
  - Y/N send `approve`/`discard`, and P/R do nothing during a review;
  - the store clears `review` on resume and on terminal events;
  - the checkbox follows upscale;
  - Reuse settings and Copy CLI command include `--review`.
- `tests/test_ltx_mlx_contract.py`: `_stage1`, the `Stage1Result` fields used and `_load_decoders`.

## Docs

- `docs/guides/image.md` and `docs/guides/video.md`: the option, the Y/N keys, the printed preview path in the CLI, that the queue waits, and for video that the sound may change slightly. Update the video guide's "Pause, Skip and Repeat are not available for video" sentence.
- `docs/reference/cli.md`: `--review` for `ziv-image` and `ziv-video`.
- `docs/development.md`: the review state on `SkipSignal`, the `review` callable, `review_stage`, and the new LTX internals.
- `CHANGELOG.md` `[Unreleased]`: "Review before upscaling: approve or discard each small render before the full-size pass, for images and video (video on macOS)."
- `make frontend-build`, with the rebuilt files committed.

## Alternatives considered

- **Saved drafts that can be approved later.** Save the stage-1 latent, audio, embeddings and conditioning next to a preview, and finish from the gallery. Rejected: it adds sidecar files, gallery state, cleanup and model/LoRA matching. Keeping everything in memory while the job waits is enough.
- **A separate "draft mode"** apart from upscale. A review exists only with upscale, so it belongs to the upscale settings.
- **Reimplementing the two-stage orchestration in `video_mac`** from `_stage1`, `_upsample_latent`, `_stage2` and the decode. Rejected: it adds a second code path and duplicates the vendored low-memory frees. Missing one of those kept the DiT loaded through the final decode.
- **New `review_*` events and a `discarded` status.** Rejected: `job_paused`/`job_resumed` with a payload and the existing `skipped` status already have handling across the runner, SSE client and job card.
- **Reusing Pause's `wait_for_key`/`resume`.** Rejected: any key and Resume would end a review without a decision.
- **`input()` for the CLI prompt.** It conflicts with the image CLI's key listener. One `SkipSignal` mechanism serves both CLIs.
- **A/D shortcuts on the Web.** Rejected in favour of Y/N, so the Web and CLI use the same keys and N keeps meaning "skip this one".
- **Enter as discard in the CLI.** Rejected so a stray keypress can't throw a render away.
