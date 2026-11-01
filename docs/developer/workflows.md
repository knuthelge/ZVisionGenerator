# Jobs and workflows

How a job runs, from the CLI or the Web UI: a preflight phase that plans and enhances prompts, then a workflow of stages that generates.

## Job phases: preflight and generation

Every image and video job (CLI and Web UI) runs in two phases, and only one model is resident at a time:

1. **Preflight** (`preflight.run_preflight`) runs before the generation model loads. It plans every iteration with `plan_iterations` (seed, `{a|b}` expansion, enhancement settings) and, when any iteration asks for auto enhancement, loads the prompt enhancer, rewrites those prompts, closes the enhancer and frees accelerator memory. It returns a frozen `JobPlan` (`core/job_plan.py`).
2. **Generation**: the entry point loads the model and passes the plan to `run_batch` / `run_video_batch` as the required `plan=` argument. Runners raise `ValueError` for a cancelled plan or one whose length differs from the batch, and `RuntimeError` when an iteration is out of order. Retry and repeat keep the planned text and draw a new random seed.

`JobPlan.iterations` holds one `IterationPlan` per iteration in run → set → prompt order, the order the runners loop in. Each carries `seed`, `resolved_prompt`, `enhanced_prompt` and `enhance_status` (`off`, `enhanced`, `failed`, `skipped`); video iterations also carry `scene`, `resolved_scene` and `enhanced_scene`, and a video rewrite returns the scene and the action together. `EnhanceSettings.parts` (`both`, `action` or `None` for auto) picks what a video rewrite writes; `resolve_parts()` turns it into `action` or `both` per job (auto means `action` when extending a clip that has a scene, and a blank scene always means `both`). A rewrite that keeps the scene returns `EnhancePairResult.scene == ""`, so `enhanced_scene` is `None`, no `enhanced_scene` is stored in the metadata, and the provenance `scene` is the user's own, with `scene_input` repeating it. The workflow's `enhance_prompt_stage` only applies the planned rewrite; no workflow stage calls the LLM.

### Controls

Entry points own the control signal: `image_cli.main` and `video_cli.main` create the `SkipSignal` and its key listener before preflight and stop it after the batch, and Web UI jobs pass their control signal to both phases. `run_video_batch` mirrors `run_batch` through the shared helpers in `job_controls.py` (`pause_until_resumed`, `announce_repeat`), and every video job offers Next (a backend with no step hook, such as H3 text to video on a Mac, acts on it when the clip finishes). During preflight there is a control boundary before each rewrite and a final one after the enhancer is released: `skip` skips the next rewrite, `pause` waits, `quit` cancels, and `repeat` is dropped. A rewrite stopped by the user (the `cancelled` callback saw Next or Quit) is `skipped`, never `failed`. On Quit, preflight releases the enhancer and memory before emitting `batch_cancelled`.

### Preflight events

Preflight events (none carry `eta_secs`, `avg_secs` or `elapsed_secs`):

| Event | When | Payload |
|---|---|---|
| `preflight_started` | first event of every preflight | `mode`, `total_iterations`, `total_rewrites` |
| `enhancer_loading` | 0..n, only when the enhancer loads; `cpu` when it runs on the CPU | `mode`, `phase` |
| `prompts_enhancing` | once per rewrite, after its boundary | `mode`, `index`, `total` |
| `prompt_enhance_failed` | a rewrite failed without a user stop | `mode`, `index`, `total`, `message` |
| `job_paused` / `job_resumed` | pause at a boundary | `mode`, `completed_iterations`, `total_iterations` |
| `batch_cancelled` | Quit (web: `job_cancelled`) | `mode`, `completed_iterations`, `total_iterations` |
| `preflight_finished` | last event unless cancelled | `mode`, `total_iterations`, `enhanced`, `failed`, `skipped` |

`prompt_started` also carries the iteration's `enhance_status`.

### Extending preflight

Tests must never run the real memory cleanup: patch `zvisiongenerator.preflight.release_accelerator_memory` (and `zvisiongenerator.web.web_runner.release_accelerator_memory` for web jobs), or pass `release_memory=` to `run_preflight`.

To add pre-load work (for example LoRA validation or a memory-fit check), add a field to `JobPlan` or `IterationPlan` and a private step in `run_preflight`. There is no stage pipeline or registry; promote `preflight.py` to a package only once it holds several concerns.

## Workflow stages

Image stage functions in `workflows/image_stages.py` have the uniform signature `(ImageGenerationRequest, ImageWorkingArtifacts) -> StageOutcome`. Video stage functions in `workflows/video_stages.py` have the signature `(VideoGenerationRequest, VideoWorkingArtifacts) -> StageOutcome`. Stages are composed dynamically by `build_workflow()` and `build_video_workflow()`. A video workflow starts with `resolve_prompt_stage`, which composes the scene and the prompt into the text the model receives (`compose_video_prompt`), and `enhance_prompt_stage` applies a planned scene and action rewrite. A job whose arguments carry an `extend_source` swaps the generation stage for the extend pair: `extend_video` (calls `extend_video` on the backend for the new segment only) and `join_clips` (`utils/video_splice.splice_clips` joins the segment onto the untouched source and removes the temporary segment directory). The extend variant has no `strip_audio` stage, since the joined clip already has the right audio; its provenance `workflow` is `extend`, with `source_video` and `extend_frames` recorded. A request that carries an `audio_track` swaps `strip_audio` for `audio_track` (or adds `extend_audio_track` before `join_clips`). It is recorded as `audio_track_path`, `audio_track_start` and `audio_track_end`, the position in the track where the clip's sound ends (`audio_track_start` plus the new frames over the frame rate, from `VideoGenerationRequest.fps` or the source's), with `generation.audio` true. A track never changes the clip's frame count. The CLI and the Web server check once at submit, through `utils/audio_track.require_readable_track`, that the file can be read, before the model loads; a track shorter than the clip, or a Start at past its end, is padded with silence by `read_track`, so the written sound lasts the whole clip. `build_upscale_workflow()` upscales an existing image (`load_source_stage` → `upscale_stage` → `sharpen_stage` → `save_image_stage`); the Web UI runs it through `WebRunner.submit_upscale_job` and `upscale_runner.run_upscale`, which emits the same events as a one-image batch, without preflight or the batch loop. `build_seedvr2_upscale_workflow()` is the plain upscale (`resolve_prompt_stage` → `load_source_stage` → `seedvr2_stages.upscale_stage` → `save_image_stage`), chosen when the request's `upscale_method` is `seedvr2`; it does not sharpen. The tier is chosen by `utils/seedvr2.pick_seedvr2_tier` from the `seedvr2` config block and the Mac's memory budget, and `generation.upscale.method` records `seedvr2` or `creative` in the saved file.

## Background removal

With `transparent` on, `build_workflow()`, `build_upscale_workflow()` and `build_seedvr2_upscale_workflow()` append `matte_stage` after the last post-processing stage and before `save_image_stage`. It has to stay last: the stages before it convert to RGB or recolour pixels, which would drop or corrupt the alpha channel. It does nothing when `artifacts.image` is `None`, and reports the `image_matte` progress phase.

- **The chain.** The request carries `transparent`, the `matte` chain (`create_matte_chain()`) and `make_room`. The runner owns the chain: it loads on the first `cut_out` and `run_batch` closes it in a `finally`.
- **The prompt.** `conditioning_prompt(text, *, transparent)` in `workflows/generation_prompt.py` appends the background phrase at the four places text reaches a model (`text_to_image_stage`, `upscale_stage`, `_job_prompts`, `upscale_prompts`). The saved and recorded prompt stays without it, so upscaling an output never doubles it. Provenance records `generation.transparent: true`, which the Web UI reads for the checkerboard and Reuse.
- **Upscaling a transparent image.** `load_source_stage` composites the source onto the grey backdrop (`processing/matte.on_backdrop`) because both upscalers take opaque input; the cut-out then runs on the upscaled result.
- **Memory.** `matte_room.MatteRoom` is the request's `make_room`. Before the chain's first load it compares `accelerator_free_bytes()` with `MIN_FREE_BYTES` (2 GiB); when short, it releases the image model through `ImageBackend.release_model` and releases it before every later cut-out. `model_for_render` closes the chain and reloads the model before the next render. An unknown free amount never swaps, and without a `reload` nothing is released.

## Review before upscaling

`--review` pauses a generation after the small render of an upscale until the user approves or discards it. The review state lives in `SkipSignal` (`utils/interactive.py`), the shared wait in `review.py`, and images pass through `review_stage` before `upscale_stage`. Video passes a `review` keyword to the backend, which honours it after its half-size render (the CUDA H3 backend decodes the half-size video, which an unreviewed upscale does not do, and writes it to a temporary video file for the review). `supports_video_upscale_review()` and the `upscale_review` capability of the model's family (see [Video families](backends.md#video-families)) tell the form, server and CLI whether to offer it; MiniMax H3 on macOS has no upscale at all. The Web UI sends approve and discard through the job controls and shows a video review from `GET /jobs/{id}/review`. Everything stays in memory while the review waits, so an approved output equals one made without review.
