# Web server

The Web UI is a FastAPI server (`web/server.py`) that serves the built Svelte app and a JSON API. This page covers how it runs jobs and the rules its subsystems keep. It does not list routes or response fields: run `uv run ziv ui` and open `/docs` for the routes the running server has, and read the module a route calls for what it does. The rule of thumb is that `server.py` parses a request and calls a module in `web/`, which holds plain functions that take what they need and return data.

## Job queue

`WebRunner` (`web/web_runner.py`) runs every job on a single worker thread. Generation jobs are never refused for being second: `_submit_job` registers the job, publishes `job_submitted`, appends it to the FIFO queue and dispatches it, all under one lock, so the worker receives jobs in queue order. When the worker reaches a job, `_claim` takes it off the queue and publishes `job_started`. A job removed with `cancel_queued`, `clear_queue` or `shutdown` is no longer queued, so the claim skips it even if its work item was already picked up; it ends with `job_cancelled` and `reason: "removed"`.

The active job is the claimed one or, between jobs, the head of the queue; every other unfinished generation job is queued with a 1-based position, and a job is never in both. Actions that need the model memory to themselves (manual prompt enhancement, saving a quantized copy, deleting models or LoRAs) wait until no generation job is running or queued.

**Download lane.** Downloads started from the Models page take a second path: `DownloadLane` (`web/download_lane.py`) runs them one at a time on its own thread, never touching the job executor, so generation jobs are admitted while it downloads. Deleting a model or a Hugging Face download is refused with a 409 while a download touches its repo or folder. It also installs LoRAs. See [Model downloads](downloads.md).

**Job log and recent prompts.** `WebRunner` writes finished jobs to the job log, and `_generate_from_form` records the prompt of an accepted job. Both live in the [app database](app-database.md#job-log) and never fail the job.

## Discover LoRAs

The Discover routes serve the curated list ([The curated LoRA list](curated-loras.md)) and, once the user opts in, Hugging Face. Routes are in `server.py`; the logic is in `web/discover_api.py` (response assembly), `web/discover_hub.py` (what is read from the Hub), `web/hub_client.py` (the bounded reader), `web/lora_download.py` (names and the lane item), and `web/lora_details.py` (where files came from and their triggers). Errors are `{"detail": "<text>"}`, and a `detail` is always a sentence the UI can show.

- **Every Hub read is anonymous and bounded.** `hub_client.http_fetch` never sends the user's token or any `Authorization` or `Cookie` header, builds every URL on `https://huggingface.co` on the server (a request value reaches a URL only through `validated_repo`, `validated_file` and `validated_revision`), follows redirects itself and only to `huggingface.co` and `*.hf.co`, and stops reading a body at its cap (2 MB for a safetensors header, 8 MB for JSON). `CachedFetch` keeps answers in memory for five minutes with a hard size bound. A bad `repo` or `file` is a 422 before the Hub is asked.
- **Opt-in.** The server can be bound beyond localhost, so the Hugging Face routes refuse (403) until the user has chosen to show Hugging Face LoRAs (`ui.discover_huggingface`, shared by every device that opens the app).
- **What is never listed.** Gated and private repositories, and ones tagged `not-for-all-audiences`. Unless `ui.discover_content_filter` is `off`, the content filter (`web/listing_filter.py`) also hides repositories whose id, tags, trigger words or file names use a term from `web/listing_filter.json`. It is a best-effort check on names, not a content rating. The term lists are base64-encoded to keep them out of casual reading; decode them to edit, and test with made-up terms.
- **A file's family is checked from its header** by two `Range` requests, never by downloading it: the answer is `match`, `mismatch` or `unknown`.
- **A name is known before Use is pressed.** A repository detail answers each file's `save_as` (a slug of the repository name, numbered when taken on disk or by a running download), so a taken name never surprises the user.
- **Refusals come before any bytes move**: 409 for a curated name held by a different file or download, 403 before the opt-in, 404 for an unknown id or file, 422 for a bad value or a refused repository, 502 or 503 when Hugging Face cannot answer.
- **Installed LoRAs** carry their trigger, its source and, for a Hugging Face download, the facts it kept, so the Installed tab describes a file offline. Deleting a LoRA also deletes its source row, and is refused with a 409 while a job runs or is queued, because the job may be loading the file. See [LoRA sources](app-database.md#lora-sources).
- The memory estimate for a MiniMax H3 job counts a LoRA that is still downloading at the size its download expects, instead of refusing the missing file.

## Video upscale jobs

`POST /api/upscale` on a gallery video queues an `upscale` job whose context carries `media_type: "video"` and `upscale_method` (`seedvr2` or `ltx`), the platform's `video_file_upscale_method()`. `upscale_api.plan_video_upscale` (SeedVR2) or `plan_ltx_video_upscale` (macOS, LTX-2.3) builds the request, and the matching `WebRunner` method downloads, loads and runs it. The runner builds the result in a hidden `.ziv-tmp-upscale-*` folder beside the source, which the gallery and `/media` never list, and moves it into place at the end, so a quit or a failure leaves nothing. The result embeds the source's config with `workflow: "upscale"` and a `source` block that follows back to the original.

## Audio jobs

`POST /api/generate` with `mode=audio` and `POST /api/generate/plan` share `_parse_audio_job`; `_submit_audio_job` queues the result with `WebRunner.submit_audio_request_job` (exclusive). The plan route lists the audio model as one download part with its licence records.

**Plan without files.** The Workspace posts the generate form to `/api/generate/plan` with each upload as an empty part that keeps its field and file name, then posts the real form once to `/api/generate`. `_generate_plan_from_form` swaps every upload for a `PlannedUpload` before it parses, so the plan route never reads or saves a file. File helpers still check extensions, so the user is never asked to approve a download for a job that cannot run.

- **The request is the CLI's.** `web/audio_job.audio_request_from_fields` builds the `AudioGenerationRequest` the way `ziv audio` does, using the same checks in `utils/audio_request.py`, so a form and a command word their refusals alike. The form's field names are the ones the Web UI's Copy CLI reads (`frontend/src/lib/state/cliCommand.ts`, checked against `tests/fixtures/audio_cli_cases.json`). A field that belongs to another kind than the model's is ignored; a blank `tempo`, `key` or `seconds` is Auto.
- **Consent.** A one-off voice recording needs `voice_consent`, unless it is a take made without a reference voice (`audio_voice.voice_needs_consent`). The tick is checked and never saved: the job's recorded settings drop it, so a loaded or retried job comes back unticked. A saved voice needs no tick, because its consent is in its record.
- **Events.** Besides the events of the other runners, an audio job publishes `model_downloading`, `model_loading` and per-take `audio_progress`. A Stop is honoured at every step, also while the model loads. A take that fails ends the job `job_failed`; the takes before it are saved. The job's context carries `media_type: "audio"` and no width or height.
- **The workspace payload lists `audio_models`**, one row per preset that runs on this machine, in config order (`web/audio_models.py`). The workflow contract lists `music`, `speech` and `sound` only for kinds with a runnable model, so a config with no audio preset leaves the Audio tab hidden.
- **The Models page** lists `audio_models` after the video ones (`web/audio_inventory.py`). Deleting one removes its repositories from the Hugging Face cache and is refused while a job runs or a repository downloads. Every audio repository counts as used, so it never appears under Other downloads.

### Audio text enhancement

`POST /api/prompt/enhance` takes `mode: "audio"` beside `image` and `video`. It shares the endpoint's NDJSON stream, the Fast/Smart choice and its admission rule: a running job or another enhancement gives a 409, because the enhancer and the audio models never share the GPU. `web/audio_enhance.py` owns the audio side; the rules of each task live in `utils/prompt_enhance_audio.py` (caption rewrites) and `utils/audio_writers.py` (lyrics, script and fill). A `task` outside `audio_enhance.ENHANCE_TASKS` is a 422, and the request and its checks are the module's to define.

- **Frames.** `status` frames as for images, `text` frames while the enhancer writes, then one `done` frame whose `fields` map each written field to its text. A changed quoted text is a `warning` on the `done` frame. A writer whose answer left the text as it was ends in an `error` frame with `no_change: true`, so the client shows a note instead of an error.
- **The saved record.** `enhance_record` (`settings` holding only non-default options, `model`, `tier`, `format`, `task`) is read by `parse_enhance_record(mode="audio")`; a record that cannot be read is dropped with a warning instead of failing the take. A take with a record saves the user's own text as `prompt_input` and Reuse gives it back. Fill in writes into the user's own fields and leaves no record.

## Saved voices

`web/voices.py` holds the logic of the `/api/voices` routes and `utils/voice_library.py` the store: `<data dir>/voices/<slug>.wav` and a `<slug>.yaml` record. The slug is made once from the name (`utils/slug.slugify`) plus a short random hex suffix, so it is unique and effectively never reused; a rename keeps it. Names are unique ignoring case.

- **Consent lives in the record only.** `audio_voice.consent_for_saving` decides it when a recording is saved: `given` for a tick, `not_needed` only for a speech take of this app made without a reference voice, otherwise refused. A designed voice is always `not_needed`. The routes never return the consent or the path.
- **Nothing is written under the output folder.** The routes serve a voice's WAV by slug only. A recording is probed against the model's voice limits before it is decoded, so a long file is refused, never cut.
- **Delete is a 409 while any job is active**, because the job may be reading the voice.

### Voice design jobs

A design run is its own job, because it makes candidates, not takes: `POST /api/voices/design` queues `WebRunner.submit_voice_design_job` (exclusive), after `/api/voices/design/plan` has listed the VoiceDesign checkpoint in the usual download plan. Its context has `workflow: "voice_design"` and no settings, so it is not logged, cannot be loaded or retried, and offers Cancel only. Candidates are not outputs: they live in `<data dir>/voice-design/` (`utils/voice_candidates.py`), outside the output folder, so no gallery, `/media` or job result lists them. A run adds a batch and never replaces an earlier one; clearing is a 409 while a design job is queued or running.

## Audio in the gallery

Audio extensions are gallery media: `/media` serves them with ranges so a player can seek, and `GET /api/gallery?filter=audio` lists only them. `_read_audio_facts` makes one ffprobe call per file for the embedded settings, duration, channels and sample rate, and never decodes it. The waveform peaks are computed lazily by `GET /api/gallery-peaks`, which decodes the file on the first request and stores the result in the file's index row, so later requests decode nothing; the asset carries `peaks: null` until then. A file with this app's settings is reusable and its Reuse URL never carries the consent. Reuse gives `saved_voice` (the slug) for a take whose recording is in the voices folder, also once the voice is deleted; any other recording comes back as `voice_path` and `voice_text`. A file without settings lists and plays but has no Reuse. The asset JSON is built in `web/gallery_audio.py`.

## Projects

A project is a folder directly under the output root; there is no registry. `web/projects.py` owns the rules: `list_projects` returns the real folders (no files, symlinks or dot-prefixed names) sorted case-insensitively, `create_project` applies the creation rules, and `asset_project` is the first segment of an asset id. Gallery assets carry `project` (a name or `null`), derived from the id.

- **No project is a filter, not a name.** `NO_PROJECT_FILTER` is a value of the gallery's `filter` parameter beside `image`, `video` and `audio`, because a folder can be named anything.
- **Two checks on a name.** `create_project` applies the full creation rules, the same on every platform. `project_dir` (saves and moves) only requires one plain segment, so an existing folder the creation rules would refuse (such as `what?`) still works.
- **Root and save folder are separate.** Generate and plan requests take an optional `project` field. The job's output directory (`args.output`, and an audio job's `output_dir`) is the project's folder, or the root when blank. Uploads, staging, reference resolution and the job context's `output_dir` always use the root, because asset ids and `/media` URLs are relative to it. `project` is not recorded in a file's settings; the folder a job really saves into is the context's `save_project`.
- **Made-from outputs.** An upscale saves beside its source and carries that folder's project, and an extension is saved beside its source (`output_source_folder`).
- **Starred is a flag, not a filter value.** `starred` is a boolean parameter of the listing and of its ID list, applied after the media filter and the project. A star lives in the app database (see [Asset records](app-database.md#asset-records)).
- **Move.** `move_gallery_assets` handles each file on its own and returns what moved and what failed. Each file is `os.replace`d to the next free name from `unique_output_path`; the check and the rename are not atomic, which the single-user, local design accepts. A move keeps mtime, flattens sub-folders, leaves `_preupscale` companions behind and does not rewrite source links in upscales and extensions. `GalleryIndex.rename` re-keys the index rows (see [Gallery index](app-database.md#gallery-index)) and `AssetRecords.rename` moves the star. Each moved entry carries `from_path`, so the client can retarget Workspace slots that held a file it never loaded.
- **Bulk delete returns what it removed.** Each deleted entry carries the file's kind, project, star and path, so the client lowers every list's counts without a map of the files it holds. A file that is already gone counts as deleted.
- **The zip download is a form POST.** A hidden form submits so the browser saves the `attachment` through its own download, and it carries the IDs in one field, which the URL length does not limit. `server.py`'s same-origin guard admits it through the `Origin` header of a form POST, so no `Referrer-Policy: no-referrer` may be added to the app: it would turn `Origin` into `null`. The zip is streamed (`web/gallery_zip.py`) with no temporary file, as deflate level 0 rather than stored, because a deflate stream carries its own end marker, which unzippers need to find an entry's end when the header holds zero sizes and a data descriptor.

## File browser previews

Browse purposes with a `preview` kind (`BrowsePurpose.preview` in `web/path_browser.py`: images, audio tracks, source videos) get preview, info, sprite and media routes, with the logic in `web/browse_media.py`; the media route serves an image at its full size, for the Edit painter, and gallery thumbnails are served by `/thumbs/{asset_id}`, confined like `/media`. They only serve regular files with an extension the purpose accepts. Images are re-encoded without metadata. Pictures are cached in `<data dir>/cache/browse`, and the picker routes refuse cross-site fetches where the browser sends `Sec-Fetch-Site`.

### Video frames

One rule turns a time in a clip into a frame, in `utils/video_point.py`: `frame_at(t) = clamp(floor((t - start) * fps + 1e-3), 0, frames - 1)`, where `start` is the seconds from the file start to the first video frame (audio priming can make it non-zero). `frame_seek_seconds(frame)` is half a frame before the frame, so `ffmpeg -ss` before `-i` decodes that frame first. `frontend/src/lib/state/framePoint.ts` applies the same rule; both are checked against `tests/fixtures/frame_point_cases.json`, so change them together. A clip is assumed to have a constant frame rate.

- **Preview `at`.** The time is snapped as above and goes into the cache key and the ETag, so a time that is not exactly a frame's still names that frame.
- **Sprite.** `utils/video_sprite.py` builds one JPEG contact sheet per clip: a tile per frame for a short clip, evenly spaced tiles for a longer one. It is built on the first request and cached; a sheet that cannot be built is a 404 and is not cached.
- **Colour.** `frame_color_filter` in `utils/video_frame.py` is the one place that decides how a frame is converted to RGB: a clip with no colour matrix tag is read as BT.709, which is what a player shows. Every video still and the sprite use it.
- **Saving a frame.** `POST /api/video/frame` (`web/video_frame_api.py`) saves the frame a time lies in as a full-size PNG under `.web_uploads/` in the output folder, which the Gallery does not list. The route shares the trust model of `/api/extend/source`.

### Extending from a point

An Extend job takes an optional `extend_from` time in the source clip. `_parse_video_job` snaps it with `utils/video_cut.extend_cut_point`, which keeps the frame the time lies in as the last kept frame. A time at or past the last frame keeps the whole clip; a point that keeps fewer than 25 frames, or a negative or non-numeric one, is a 422. The parse only validates: it sets `args.extend_cut` and writes no file, so `/api/generate/plan` stays free of files.

The cut is made when the job runs. `WebRunner._run_video_request` and the CLI wrap the model load and the batch in `applied_extend_cut(args)`, after preflight and downloads. It makes one `.ziv-tmp-cut-<pid>-*` folder in the save folder, cuts the kept frames into it with one ffmpeg call, and checks that the cut file holds exactly the kept frames at the source's size, or the job fails before the model loads. While the block runs, `args.extend_source` is a source rebuilt from the cut file that keeps the original's recorded settings. The folder is removed and the original put back when the block ends, whether the job finished, failed or was stopped; a sweep before the next cut removes the folder of a process that no longer exists. The original file is never touched.

The output records `source_video` as the original clip and `extend_from` as the snapped time, so `frame_at` returns the cut frame from it. When the clip followed an audio track, Start at moves back by the removed frames.

## Reference images

An image job takes references on top of its start image. The Workspace form sends them as one repeated `reference` field, one value per row in prompt order, so the rows stay numbered as the prompt refers to them ("image 1", "image 2"). `_resolve_reference_images` saves uploads like the start image: each is checked and named by the SHA-256 of its content (`.web_uploads/<hash>.<ext>`), and a file already there is left as it is. The picture to edit of an Edit form (`edit_source_file`) is saved the same way, under `.web_uploads/<hash>/<name>`. `_validate_model_capabilities` refuses references for a model whose `max_reference_images` is 0 or below the count.

- **The workflow is derived from the inputs**, not the posted value: a start image or first frame makes it `img2img` or `img2vid`, so the history and the job log record what ran, and a posted `img2img` without an image is refused. No workflow clears the image slots, because a tab change must not drop an image.
- **A recorded reference is kept even if its file is gone.** A gallery image's Reuse URL names its references as recorded, and the Workspace keeps a row whose file is gone, marked missing, because dropping it would renumber the prompt. `POST /api/references/check` answers which paths are no longer files, for every image slot.

## Edit jobs

An Edit form posts `mode=image` and `workflow=edit`, and `_parse_edit_job` in `web/server.py` parses it, with the mask read by `web/edit_job.py`, whose docstring lists the field names. The `edit` workflow is always in the contract, and a model's `supports_mask_edit` in `image_model_defaults` says whether it can edit; the refusals are the CLI's, from `utils/edit_request.check_edit_inputs`. `POST /api/generate/plan` takes the form with empty file parts and reads no file content.

The browser sends the mask at no more than 4 MP; `read_job_paint` scales it to the picture's upright size and keeps the bool array in memory. It travels with the job as `args.edit_paint` (the CLI sets the same attribute from `--mask`), reaches the request as `edit_paint`, and is never recorded or written to disk, so a Reuse has no paint to bring back and the Workspace asks for it again. The result is saved beside the picture when that is an asset under the output folder (`output_source_folder`), otherwise in the project. A gallery Reuse URL carries `workflow=edit`, the picture, the prompt and settings, `edit_repaint=true` when the edit was painted, and the Extend side and amount.

## Settings presets

Presets are a Web UI feature; the CLI does not know them. A preset is one YAML file, `<data dir>/presets/<mode>/<slug>.yaml` (`mode` is `image` or `video`), with `name`, `model` (or null) and `settings`; the folder decides the mode. `web/presets.py` owns the format, the slug rules and the store (writes under one lock with `write_text_atomic`). The `settings` keys are the Workspace form field names; unknown keys are ignored on read and write, and prompt, seed, images and the workflow are never stored. Names and slugs are unique within a mode.

On the frontend, `draft.applyPreset` applies a preset the same way in every workflow: keys outside the preset are untouched and the capability gates re-run, so the new model clears what it cannot use. A mode switch, Reset, and an extend or reuse that loads a source's settings clear the applied preset.

## Model inventory

- **Names and aliases.** The `name` of a model's entry in `model_metadata.yaml` is how the Web UI names it (`utils.model_metadata.model_display_name`). The alias stays the id in every API field, CLI flag, prompt file and recorded setting; only the Workspace picker, job card, queue row and Models page show the name.
- **Declared family.** A built-in image model declares its family in its entry (`ideo: ideogram4`); the Klein entries also declare `distilled` and `size`, the two facts detection reads, so their defaults, LoRA matching and the Edit kind's `supports_mask_edit` are known before the model is downloaded (`declared_image_model`). The inventory consults it as an offline fast path, so a known alias needs no network detection; content-based detection for arbitrary local folders is unaffected.
- **Capabilities are data.** `supports_img2img`, `supports_upscale`, `supports_scheduler`, `supports_json_prompt`, `supports_first_sigma` and the dimension bounds come from the `image_families` entries of the metadata (`utils.model_metadata.model_capabilities`), flow through `resolve_defaults` and reach the SPA as `ImageModelDefaults`. Families that do not set them inherit permissive defaults, so gating is meant to be centralized in the metadata. The web layer is not free of family checks; see the [invariant](architecture.md#invariants) for the known exceptions.
- **Download size.** An image model's `download_size` is its one-time download, entered from a dry-run plan against an empty cache; update it when a model's repo or files change.
- **First run.** The first-run dialog opens when every image model is not downloaded, no image download is queued or running and `ui.first_run_skipped` is not set. `web/first_run.recommended_first_run_model` picks `large_model` or `small_model` of the `first_run` config block from the machine's memory.
