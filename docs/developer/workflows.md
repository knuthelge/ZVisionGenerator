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

Image stage functions in `workflows/image_stages.py` have the uniform signature `(ImageGenerationRequest, ImageWorkingArtifacts) -> StageOutcome`. Video stage functions in `workflows/video_stages.py` have the signature `(VideoGenerationRequest, VideoWorkingArtifacts) -> StageOutcome`. Stages are composed dynamically by `build_workflow()` and `build_video_workflow()`. A video workflow starts with `resolve_prompt_stage`, which composes the scene and the prompt into the text the model receives (`compose_video_prompt`), and `enhance_prompt_stage` applies a planned scene and action rewrite. A job whose arguments carry an `extend_source` swaps the generation stage for the extend pair: `extend_video` (calls `extend_video` on the backend for the new segment only) and `join_clips` (`utils/video_splice.splice_clips` joins the segment onto the untouched source and removes the temporary segment directory). The extend variant has no `strip_audio` stage, since the joined clip already has the right audio; its provenance `workflow` is `extend`, with `source_video` and `extend_frames` recorded. `build_upscale_workflow()` upscales an existing image (`load_source_stage` → `upscale_stage` → `sharpen_stage` → `save_image_stage`); the Web UI runs it through `WebRunner.submit_upscale_job` and `upscale_runner.run_upscale`, which emits the same events as a one-image batch, without preflight or the batch loop. `build_seedvr2_upscale_workflow()` is the plain upscale (`resolve_prompt_stage` → `load_source_stage` → `seedvr2_stages.upscale_stage` → `save_image_stage`), chosen when the request's `upscale_method` is `seedvr2`; it does not sharpen. The tier is chosen by `utils/seedvr2.pick_seedvr2_tier` from the `seedvr2` config block and the Mac's memory budget, and `generation.upscale.method` records `seedvr2` or `creative` in the saved file.

## Review before upscaling

`--review` pauses a generation after the small render of an upscale until the user approves or discards it. The review state lives in `SkipSignal` (`utils/interactive.py`), the shared wait in `review.py`, and images pass through `review_stage` before `upscale_stage`. Video passes a `review` keyword to the backend, which honours it after its half-size render (the CUDA H3 backend decodes the half-size video, which an unreviewed upscale does not do, and writes it to a temporary video file for the review). `supports_video_upscale_review()` and the `upscale_review` capability of the model's family (see [Video families](backends.md#video-families)) tell the form, server and CLI whether to offer it; MiniMax H3 on macOS has no upscale at all. The Web UI sends approve and discard through the job controls and shows a video review from `GET /jobs/{id}/review`. Everything stays in memory while the review waits, so an approved output equals one made without review.
