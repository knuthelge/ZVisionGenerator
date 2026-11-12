# Jobs and workflows

How an image or video job runs, from the CLI or the Web UI: a preflight phase that plans and enhances prompts, then a workflow of stages that generates. Audio jobs have no workflow; see [Audio](architecture.md#audio).

## Job phases: preflight and generation

Every image and video job runs in two phases, and only one model is resident at a time:

1. **Preflight** (`preflight.run_preflight`) runs before the generation model loads. It plans every iteration with `plan_iterations` (seed, `{a|b}` expansion, enhancement settings) and, when any iteration asks for auto enhancement, loads the prompt enhancer, rewrites those prompts, closes the enhancer and frees accelerator memory. It returns a frozen `JobPlan` (`core/job_plan.py`).
2. **Generation.** The entry point loads the model and passes the plan to `run_batch` or `run_video_batch` as the required `plan=` argument. Runners raise `ValueError` for a cancelled plan or one whose length differs from the batch, and `RuntimeError` when an iteration is out of order. Retry and repeat keep the planned text and draw a new random seed.

`JobPlan.iterations` holds one `IterationPlan` per iteration in run, set, prompt order, the order the runners loop in. Each carries the seed, the resolved and enhanced prompts and an `enhance_status` (`off`, `enhanced`, `failed`, `skipped`). Video iterations also carry a scene and an enhanced scene, and a video rewrite returns the scene and the action together. `EnhanceSettings.parts` picks what a video rewrite writes (`both` or `action`; auto means `action` when extending a clip that has a scene). A rewrite that keeps the scene stores no `enhanced_scene`, and the saved `scene` is the user's own. No workflow stage calls the LLM: `enhance_prompt_stage` only applies the planned rewrite.

### Controls

Entry points own the control signal: `image_cli.main` and `video_cli.main` create the `SkipSignal` and its key listener before preflight and stop it after the batch, and Web UI jobs pass theirs to both phases. `run_video_batch` mirrors `run_batch` through the shared helpers in `job_controls.py`. During preflight there is a control boundary before each rewrite and a final one after the enhancer is released: `skip` skips the next rewrite, `pause` waits, `quit` cancels and `repeat` is dropped. A rewrite stopped by the user is `skipped`, never `failed`. On Quit, preflight releases the enhancer and memory before it emits `batch_cancelled`.

### Preflight events

None of these carry `eta_secs`, `avg_secs` or `elapsed_secs`. `prompt_started` also carries the iteration's `enhance_status`.

| Event | When | Payload |
|---|---|---|
| `preflight_started` | first event of every preflight | `mode`, `total_iterations`, `total_rewrites` |
| `enhancer_loading` | 0..n, only when the enhancer loads; `cpu` when it runs on the CPU | `mode`, `phase` |
| `prompts_enhancing` | once per rewrite, after its boundary | `mode`, `index`, `total` |
| `prompt_enhance_failed` | a rewrite failed without a user stop | `mode`, `index`, `total`, `message` |
| `job_paused` / `job_resumed` | pause at a boundary | `mode`, `completed_iterations`, `total_iterations` |
| `batch_cancelled` | Quit (web: `job_cancelled`) | `mode`, `completed_iterations`, `total_iterations` |
| `preflight_finished` | last event unless cancelled | `mode`, `total_iterations`, `enhanced`, `failed`, `skipped` |

### Extending preflight

To add pre-load work (LoRA validation, a memory-fit check), add a field to `JobPlan` or `IterationPlan` and a private step in `run_preflight`. There is no stage pipeline or registry; promote `preflight.py` to a package only once it holds several concerns. Tests never run the real memory cleanup; see [Testing](testing.md#protective-fixtures).

## Workflow stages

A stage is a function `(request, artifacts) -> StageOutcome`: `ImageGenerationRequest` and `ImageWorkingArtifacts` (types in `core/image_types.py`, stages in `workflows/image_stages.py`), the video types in `core/video_types.py` and its stages in `workflows/video_stages.py`. Stages are composed per job by `build_workflow()` and `build_video_workflow()` in `workflows/__init__.py`, under the conditions that turn each on. See [Add a workflow stage](how-to.md#add-a-workflow-stage).

- **Prompt.** A video workflow starts with `resolve_prompt_stage`, which composes the scene and the prompt into the text the model receives, then `enhance_prompt_stage`.
- **Extend.** A job whose arguments carry an `extend_source` swaps the generation stage for `extend_video` (the backend makes the new segment only) and `join_clips` (`utils/video_splice.splice_clips` joins it onto the untouched source). It has no `strip_audio` stage, since the joined clip already has the right audio; its provenance `workflow` is `extend`, with `source_video` and `extend_frames` recorded. An extension from a point carries an `ExtendCut`; see [Extending from a point](web-server.md#extending-from-a-point).
- **Audio track.** A request with an `audio_track` swaps `strip_audio` for `audio_track` (or adds `extend_audio_track` before `join_clips`) and records the track's path and range. A track never changes the clip's frame count. The CLI and the web server check once at submit, through `utils/audio_track.require_readable_track`, that the file can be read, before the model loads; a track shorter than the clip is padded with silence.
- **Edit.** A job whose arguments carry an `edit_source` runs `build_edit_workflow()`; see [Editing an image](#editing-an-image).
- **Upscale.** `build_upscale_workflow()` upscales an existing image (`load_source_stage`, `upscale_stage`, `sharpen_stage`, `save_image_stage`); the Web UI runs it through `upscale_runner.run_upscale`, which emits the same events as a one-image batch, without preflight. `build_seedvr2_upscale_workflow()` is the plain upscale, chosen when the request's `upscale_method` is `seedvr2`; it does not sharpen. `utils/seedvr2.pick_seedvr2_tier` picks the tier from the `seedvr2` config block and the machine's memory, and the saved file records `seedvr2` or `creative`.

## Editing an image

`build_edit_workflow()` runs an edit for the CLI and the Web UI: `resolve_prompt_stage`, `load_edit_source_stage`, `prepare_edit_stage`, `edit_stage`, `composite_stage` and `save_image_stage`. It has no sharpen, contrast, saturation or matte stage. `run_batch` picks it when the job's arguments carry an `edit_source`. Each stage is a thin adapter over the pure functions in `processing/edit_region.py`, whose tuning constants are module-level, not configuration. Every edit size goes through `utils/alignment`.

- **A crop, not the whole picture.** The model works on a region. `plan_region` takes the painted box plus a margin, snapped to 16 px and drawn at about one megapixel; a mask over half the picture takes the whole canvas. With no paint and no Extend the crop is the whole picture at its own size, rounded to 16 px (`grid_size`), and `composite_stage` scales the result back when the rounding changed it.
- **Extend draws first, then pads.** The source is drawn at about one megapixel before the strip is added (`extend_draw_size`), so the strip, the source and the blend mask all sit on whole 16 px cells; `EditWork.canvas` keeps the real-size canvas.
- **The reference.** `edit_stage` gives the backend a `reference`: the crop, or for an Extend the drawn canvas with the strip one flat colour, the mean of the source beside it (`extend_reference`), so the strip's fill is never shown to the model and there is no picture in the strip to copy. An Extend also sends the prompt behind `EXTEND_INSTRUCTION`, which tells the model to fill the empty canvas, and passes `fix_join=True` so the strip is colour-matched to the source across the join. With no paint it calls `text_to_image` with the crop as the only reference: the whole-picture edit.
- **Paste-back.** `composite_stage` pastes with `paste_paint` or `paste_strip`. Around a paint the result fades into the canvas, and an Extend's join with the source is hard. A cut-out is edited on a grey backdrop and gets its alpha back at the end.

`utils/edit_request.py` holds what the CLI and the Web UI share, so both refuse with the same words, through `check_edit_inputs`, which reads `supports_mask_edit` from `utils/model_metadata.yaml`; neither entry point branches on a family. Each reads the picture once before a model loads and hands its size to the runner as `edit_size`.

Provenance records `workflow: "edit"`, a `source` block, an `edit` block (`mode`: `paint`, `extend` or `whole`, and the Extend side and amount) and the saved size. It never records the mask. An edit of a cut-out records `generation.transparent: true`. `tests/image/test_edit_workflow.py` and `tests/image/test_edit_region.py` pin the stages and the pure functions.

## Background removal

With `transparent` on, the three image workflow builders append `matte_stage` after the last post-processing stage and before `save_image_stage`. It has to stay last: the stages before it convert to RGB or recolour pixels, which would drop or corrupt the alpha channel.

- **The chain.** The request carries `transparent`, the `matte` chain (`create_matte_chain()`) and `make_room`. The runner owns the chain: it loads on the first cut-out and `run_batch` closes it in a `finally`.
- **The prompt.** `conditioning_prompt(text, *, transparent, extend=False)` in `workflows/generation_prompt.py` appends the background phrase wherever text reaches a model, and with `extend` puts `EXTEND_INSTRUCTION` in front. `_job_prompts` pre-encodes the same text `edit_stage` sends, which a Mac needs, since it frees the text encoder after the batch's prompts. The saved prompt stays without it, so upscaling an output never doubles it. Provenance records `generation.transparent: true`, which the Web UI reads for the checkerboard and Reuse.
- **Upscaling a transparent image.** `load_source_stage` composites the source onto a grey backdrop (`processing/matte.on_backdrop`), because both upscalers take opaque input; the cut-out then runs on the upscaled result.
- **Memory.** `matte_room.MatteRoom` is the request's `make_room`. Before the chain's first load it compares `accelerator_free_bytes()` with `MIN_FREE_BYTES` (2 GiB); when short, it releases the image model through `ImageBackend.release_model` and does so before every later cut-out. `model_for_render` closes the chain and reloads the model before the next render. An unknown free amount never swaps.

## Review before upscaling

`--review` pauses a generation after the small render of an upscale until the user approves or discards it. The review state lives in `SkipSignal` (`utils/interactive.py`), the shared wait in `review.py`, and images pass through `review_stage` before `upscale_stage`. Video passes a `review` keyword to the backend, which honours it after its half-size render. `supports_video_upscale_review()` and the family's `upscale_review` capability (see [Video families](backends.md#video-families)) tell the form, the server and the CLI whether to offer it. Everything stays in memory while the review waits, so an approved output equals one made without review.
