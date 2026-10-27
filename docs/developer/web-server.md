# Web server

The Web UI is a FastAPI server (`web/server.py`) that serves the built Svelte app and a JSON API. This page covers how it runs jobs, stores presets and what it tells the frontend about models.

## Job queue

`WebRunner` (`web/web_runner.py`) runs every Web UI job on a single worker thread. Generation jobs are never refused for being second: `_submit_job` registers the job, publishes `job_submitted`, appends it to the FIFO queue and dispatches it, all under `_jobs_lock`, so the worker receives jobs in queue order. When the worker reaches a job, `_claim` takes it off the queue and publishes `job_started` (status `running`, `started_at`; `elapsed_secs` counts from here). A job removed with `cancel_queued`, `clear_queue` or `shutdown` is no longer queued, so the claim skips it even if its work item was already picked up; it ends with `job_cancelled` and `reason: "removed"`.

The *active* job is the claimed one or, between jobs, the head of the queue; every other unfinished generation job is *queued*, with a 1-based `queue_position`. `GET /api/jobs` and `/api/workspace` return them as `active_job` and `queued_jobs`, and a job is never in both. `DELETE /api/jobs/queue` removes the queued jobs and keeps the active one. Actions that need the model memory to themselves (manual prompt enhancement, saving a quantized copy, deleting models or LoRAs) still wait until no generation job is running or queued.

### Download lane

Downloads started from the Models page take a second path: `DownloadLane` (`web/download_lane.py`) runs them one at a time on its own thread. The lane never touches the job executor or the exclusive slot, so generation jobs are admitted while it downloads. Deleting a model, an enhancer model or a Hugging Face download is refused with 409 while a download touches its repo or folder (`_ensure_not_downloading` in `web/server.py`, using `DownloadFlights.conflict`). See [Model downloads](downloads.md).

## Job log

`web/job_log.py` is the `jobs` component of the [app database](app-database.md): one row per finished job in `jobs_entries_v1`, ordered by `seq`. `server.py` builds it as `JobLog(app_db.app_db_path)`. `WebRunner` takes it as the `job_log` argument and writes nothing without one.

- **Which jobs.** Only image, video and upscale jobs that started (`started_at` is set) are logged. A job removed from the queue never started, and `dummy` jobs are not generation jobs, so neither is logged.
- **Once only.** A job is logged when it first reaches a terminal status (`completed_at` still unset). A later terminal event changes the in-memory status but writes no second entry, so the log keeps the first outcome.
- **Where the write happens.** `_publish_event` copies the plain facts of the job under `record.lock` and writes last, after the subscriber fan-out and the release of the claim, holding no lock. Every logged terminal is published on the worker thread, so the write never blocks the event loop; the routes of the log are plain `def` and run in the threadpool. If a future change logs queue removals, the write must move off the loop.
- **Failures never reach the job.** Building the entry and writing it share one `try/except Exception` that only emits a `UserWarning`, so the terminal event, the claim release and the next queued job do not depend on the log.
- **Partial failures.** The record counts `generation_finished` events with `status: "failed"` (`failed_runs`) and keeps the last reason. A job that still ends `completed` or `cancelled` keeps that status, and the `failed` filter of `GET /api/jobs/log` matches it through `failed_runs`.
- **Storage.** Each row holds `job_id` and the entry's JSON text as `json.dumps(entry, ensure_ascii=False, default=str)` wrote it, so a listing returns what was written (key order, `100.0`). A listing reads every row and filters and pages with `page_entries` / `matches_status`. Each operation uses its own connection, writes run in `app_db.transaction`, and there is no lock. Rows are never trimmed; they stay until `delete` or `clear` removes them.
- **Best effort.** Two `ziv ui` servers on one data directory share the table safely. A server stopped while a job runs may exit before the worker writes that job's entry. A damaged database is quarantined whole and the log restarts empty, with the old rows kept in the `.corrupt-*` file.
- **Routes.** `GET /api/jobs/log` (50 per page, newest first, `status` filter), `DELETE /api/jobs/log/{job_id}` and `DELETE /api/jobs/log` are declared before the `/api/jobs/{job_id}/...` routes. Every storage failure, and a jobs schema newer than this ziv, reaches the routes as `JobLogError` and becomes a 500 with `Could not read or write the job log: <reason>`.
- **Tests.** The autouse `_isolated_app_db` fixture points the app database, and so the job log, at a temporary directory.

## Recent prompts

`web/recent_prompts.py` is the `recent_prompts` component of the [app database](app-database.md): one row per distinct mode, format, prompt, scene and negative prompt in `recent_prompts_entries_v1`, newest by `seq`. It is separate from the job log on purpose: the log writes a job only after it started and ended, truncates the prompt, and its Delete and Clear log would otherwise remove recents.

- **What is recorded.** `_generate_from_form` calls `_remember_prompt` after `POST /api/generate` accepted a job. `recent_prompt_entry` (pure) builds `{format, prompt, scene, negative_prompt, mode, used_at}` for an inline submit: `format: "text"` with the trimmed prompt, the scene for video and the negative prompt for images, or `format: "json"` with the trimmed `json_prompt` as `prompt`. It returns nothing for `prompt_source=file`, or when the prompt (or caption) and scene are both blank. A refused submit (409, 422) never reaches it. The text recorded is the one the form sent, so the user's own text in auto-enhance mode and the Enhanced text when it is used.
- **Dedupe.** The key is `(mode, format, prompt, scene, negative_prompt)`. Recording an existing one is an `INSERT OR REPLACE` that gives the row a new highest `seq`, so it becomes the newest and takes the new `used_at`. Nothing is capped or trimmed.
- **Routes.** `GET /api/recent-prompts?mode=image|video&q=` returns `{"entries": [...]}`: the newest 50 rows of that mode whose prompt or scene contains a non-blank `q`, ignoring case. `DELETE /api/recent-prompts?mode=image|video` removes the rows of that mode and returns `{"deleted_count": n}`. `mode` is required on both and any other value is a 422.
- **Failures.** They never reach the submit: `_remember_prompt` only emits a `UserWarning`, so the response of an accepted job is unchanged. A storage failure, and a schema newer than this ziv, reaches the routes as `RecentPromptsError` and becomes a 500 with `Could not read or write the recent prompts: <reason>`.

## Video upscale jobs

`POST /api/upscale` on a gallery video queues an `upscale` job whose context, submit response and snapshot carry `media_type: "video"` (image jobs leave it out); `WebRunner.submit_video_file_upscale_job` runs it through `video_file_upscale_runner.run_video_file_upscale` after the same SeedVR2 download plan and load as the image upscale. The runner builds the result in a hidden `.ziv-tmp-upscale-*` folder beside the source, which the gallery and `/media` never list or serve, and moves it into place at the end, so a quit or a failure leaves nothing. The result embeds the source's config through `provenance.build_video_upscale_payload`, with `workflow: "upscale"` and a `source` block that follows back to the original.

## Settings presets

Presets are a Web UI feature; the CLI does not know them. A preset is one YAML file, `<data dir>/presets/<mode>/<slug>.yaml` (`mode` is `image` or `video`), with the top-level keys `name`, `model` (or null) and `settings`; the folder decides the mode, so the file does not repeat it. `web/presets.py` owns the format, the slug rules and the store (writes under one lock with `write_text_atomic`).

- **Settings.** The `settings` keys are the Workspace form field names. Unknown keys are ignored when reading and writing, and missing keys count as not set. Prompt, seed, reference image and the workflow are never stored.
- **Routes.** `GET /api/presets` lists both modes sorted by name and skips invalid files. `POST /api/presets` creates one: the server derives the slug from the name and answers 409 when it is taken. `PUT /api/presets/{mode}/{slug}` overwrites (a `mode` in the body is ignored), `POST /api/presets/{mode}/{slug}/rename` renames, `DELETE /api/presets/{mode}/{slug}` removes; an unknown mode or a malformed slug in the path is 422. `POST` takes the mode in its body. Names and slugs are unique within a mode, so image and video presets can share a name.
- **Web apply.** `draft.applyPreset(ctx, preset)` (helpers in `lib/state/presets.ts`) applies a preset the same way in every workflow: keys outside the preset (prompt, seed, reference image, workflow) are untouched, and the capability gates re-run so the new model clears what it cannot use, as a manual model switch does. A model or LoRA that is not installed is skipped without a message.
- **Deselect.** A mode switch, Reset, and an extend or reuse that loads a source's settings clear the applied preset; a preset can be picked again afterwards. The applied slug is a single value in the draft (not per mode); it moves on rename and clears on delete.
- **UI parts.** `PresetMenu.svelte` is the toolbar control and, on a phone, the sheet behind the first chip; `PresetNameDialog.svelte` is the Save as and Rename dialog; `lib/state/presets.svelte.ts` is the list store.
- **Tests.** An autouse fixture in `tests/conftest.py` points the presets folder at a temporary directory, so no test reads or writes the real one.

## Model inventory

A repo-id alias can declare its image family in config via `model_alias_families` (for example `ideo: ideogram4`). The Web UI inventory and workspace bootstrap consult this declared family as an offline fast-path, so a known alias is surfaced without the network `detect_image_model` round-trip that a bare Hugging Face repo id would otherwise require. Content-based detection for arbitrary local model directories is unaffected; only aliases with a declared family skip detection.

`model_labels.<alias>.name` in `config.yaml` is how the Web UI names an alias; `utils.config.model_display_name(config, name)` returns it, or the alias itself when none is set. The server fills `label` on the workspace model options, the Models rows, download plans (and so the lane items) and `model_label` on a job's context and snapshot. The alias stays the id in every API field, CLI flag, prompt file and recorded setting; only the Workspace picker, job card, queue row and Models page show the name.

`model_labels.<alias>.download_size_label` (a `darwin` / `win32` / `linux` map, whole GB) is the one-time download size of an image alias on each platform; the workspace payload carries this platform's value as `download_size_label` on the image options. It is data entered from a dry-run download plan against an empty cache, so update it when an alias's repo or files change. The first-run dialog (`FirstRunDialog.svelte`) opens when every image model is not downloaded, no image download is queued or running and `ui.first_run_skipped` is not set. `first_run` in `config.yaml` names the models it recommends: `web/first_run.recommended_first_run_model` returns `large_model` when the machine's memory (Mac unified memory, CUDA total VRAM) is at least `min_memory_gb` within 5%, else `small_model`, and the workspace payload's `first_run` block carries that and the skip flag. `POST /api/first-run/skip` writes `ui.first_run_skipped` to the user config; it is not a writable config field.

A repo-id alias points at the upstream repo when it is ungated, otherwise at a mirror whose weight files match upstream's sha256.

Per-model Web UI capability flags — `supports_img2img`, `supports_upscale`, `supports_scheduler`, `supports_json_prompt`, `supports_first_sigma`, `dimension_min`, `dimension_max`, and `dimension_step` — are data-driven: they originate in the `model_presets` family entries, flow through `resolve_defaults`, and are emitted into the SPA's `ImageModelDefaults`. Families that do not set them inherit permissive defaults (img2img, upscale and scheduler enabled, JSON caption and first-sigma disabled, `16`/`null`/`16` dimension bounds), so gating is centralized in config rather than branching on family in the web layer.
