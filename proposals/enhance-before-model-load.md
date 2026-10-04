# Enhance prompts before the model loads

**Status:** In progress (2026-10-04)

## Problem

In a batch with auto prompt enhancement, the first prompt is enhanced but every later prompt silently falls back to the original text. It happens when the image model is a tight memory fit. Video on macOS is likely unaffected: LTX in low-memory mode (the default) drops its text encoder and transformer after each generation (`backends/video_mac.py`). Video still uses the same plan path, so there is one code path. Observed on macOS (MLX) in the Web UI with a prompt file; the server log shows:

```
Prompt enhancement failed (... Insufficient Memory ...); using the original prompt.
```

The failure is invisible in the UI: `apply_prompt_enhancement` (`zvisiongenerator/workflows/enhance_stage.py`) catches every exception, warns, and continues with the original prompt.

## Current behaviour

- Web and CLI jobs load the generation model first, then load the enhancer and keep both resident for the whole job (`web/web_runner.py` `_run_image_request` / `_run_video_request`, `prompt_enhancer_session.job_enhancer`).
- Enhancement runs once per *iteration* (runs × prompts), inside the workflow, seeded with that iteration's seed, after `{a|b|c}` random choices are expanded.

## What we know

- **Tried and failed:** v0.13.0b6 clears MLX's buffer cache (`gc.collect()` + `mx.clear_cache()`) before every rewrite. The error persists, so the memory is held by live arrays, not the cache.
- **mlx-lm loads the enhancer eagerly** (`load(lazy=False)` evaluates all parameters), so its weights are resident from the start.
- **mlx-lm wires memory during generation:** `stream_generate` runs inside `wired_limit()`, which raises the wired limit to the GPU's recommended working set (~75% of RAM). Metal's `kIOGPUCommandBufferCallbackErrorOutOfMemory` is reported against that working set, not total RAM.
- **Confirmed (Step 0):** mflux defers materialising (or quantising) its weights until the first generation. That is why prompt 1 fits and prompt 2 does not: only after the first image are both models fully resident.

## Step 0: confirm the cause

Before building, measure on the Mac with the failing prompt file. Log, temporarily and uncommitted:

- before each enhancement and after each generation: `mx.get_active_memory()`, `mx.get_cache_memory()`, `mx.get_peak_memory()`
- once: `mx.device_info()["max_recommended_working_set_size"]`

| Result | Meaning | Next step |
|---|---|---|
| Active memory jumps by roughly the image model's size between prompt 1 and prompt 2 | Both models do not fit together | Build this proposal |
| No such jump; memory grows gradually | Something accumulates per generation (likely inside mflux) | Find and release it; this proposal may be unnecessary |

**Result (2026-10-04): both models do not fit together; build this proposal.** Measured on the Mac (working set 24.96 GB) with the failing prompt file:

| Point | Active memory |
|---|---|
| Before rewrite 1 | 19.12 GB |
| After image 1 | 30.16 GB (+11.0 GB, one jump) |
| After images 2 and 3 | 30.16 GB (flat, no accumulation) |

Every rewrite after the first failed. A failed rewrite still took about 64 s, against 35 s for the successful one, so the bug also adds about a minute to each iteration.

## Goals

- Every auto-enhanced iteration gets its rewrite, regardless of how tightly the generation model fits.
- The enhancer and the generation model are never resident at the same time.
- Same behaviour on macOS (MLX) and Windows/Linux (CUDA), Web UI and CLI, image and video.
- Enhancement failures are visible in the UI, not only in the server log.

## Non-goals

- Changing enhancer prompts, styles or models.
- The on-demand **Enhance** button (`/enhance` endpoint); it already cannot run during a job.

## Proposal

Split an auto-enhanced job into two phases, each with one model loaded:

1. **Plan.** Before the generation model loads, build one plan entry per iteration: seed, expanded prompt (random choices resolved) and, when that entry is enhanced, the rewritten prompt. Load the enhancer for this phase only.
2. **Release.** Unload the enhancer and free accelerator memory. This must happen before the first generation, not only before `load_model`: mflux (Flux2 Klein, Z-Image) materialises its weights on the first generation, not at load.
3. **Generate.** Load the generation model and run the batch from the plan. The existing stages use the planned values instead of recomputing them.

### Design

- **One pure planning function**, e.g. `plan_iterations(prompts_data, args, config, enhancer) -> list[IterationPlan]`, where `IterationPlan` is a frozen dataclass (run index, set name, prompt index, seed, resolved prompt, enhanced prompt or `None`). Image and video runners iterate the plan instead of picking seeds inline. No second generation code path.
- **Request fields** for the planned values (e.g. `resolved_prompt`, `enhanced_prompt` on the image and video requests). `resolve_prompt_stage` copies the resolved prompt when set; `enhance_prompt_stage` becomes a pass-through that copies the enhanced prompt into `artifacts.resolved_prompt` and `artifacts.metadata["enhanced_prompt"]`, so the embedded config and history keep it.
- **Same rules as today:** skip JSON prompts, apply per-entry `enhance:` settings via `resolve_item_enhance`, expand random choices before enhancing.
- **Model family before load:** `resolve_enhance_ceiling` needs the family. The CLIs already detect it before loading (`detect_image_model` / `detect_video_model`); the web path has `request.model_family` for video and can call `detect_image_model` for image.
- **Load order flips** in `image_cli.py`, `video_cli.py` and `web_runner.py`: enhancer first, generation model second. Side benefit: enhancer errors surface before the slow model load.

### Events and UI

- **New progress for the planning phase**, e.g. `prompts_enhancing` with `index` and `total`, so the UI shows "Enhancing prompt 3 of 30" instead of "Loading prompt enhancer…" for minutes.
- **Keep the per-iteration contract.** `web_runner.py` replaces `record.prompt_progress` on every `prompt_started` and only then attaches `enhanced_prompt`. The generation phase must re-emit `prompt_enhanced` (or carry `enhanced_prompt` on `prompt_started`) for each iteration so the progress panel and reconnecting clients show the right rewrite.
- **Visible failures:** emit an event when an iteration's enhancement fails and the original prompt is used.
- Frontend changes require `make frontend-build` and committing the rebuilt SPA.

### Controls during planning

- **Next** skips the current rewrite; that iteration uses its original prompt.
- **Quit** stops the job before the generation model loads and emits `batch_cancelled`.
- **Pause** waits, as it does during generation.

### Behaviour changes

- **Delay before the first image:** all rewrites (runs × prompts) finish first. A 30-image batch may take a few minutes before generation starts.
- **Retry and repeat reuse the iteration's random choices and rewrite** and only get a new seed. Today they re-expand and re-enhance. Document this in the CHANGELOG.
- With a fixed `--seed`, every run of a prompt gets the same rewrite (unchanged from today).

## Alternatives considered

| Alternative | Verdict |
|---|---|
| Clear MLX cache before each rewrite | Shipped in v0.13.0b6; did not fix it (memory is in live arrays) |
| Unload and reload the enhancer per rewrite | Does not help: it still loads while the generation model is resident |
| Unload and reload the generation model per prompt | Fixes it, but a full model load per image is too slow |
| Enhancer on the CPU (`mx.cpu`) | Likely avoids the Metal error; rewrites 5–20× slower. Best fallback, e.g. only after an out-of-memory error |
| Enhancer in a separate process | Isolates crashes and frees memory on exit, but does not lower peak memory alone |
| Smaller KV cache (`max_kv_size`, `kv_bits`), fewer tokens | KV cache for a rewrite is a few MB; negligible |
| Smaller or more quantised enhancer model | Saves memory but lowers rewrite quality; does not fix the fit in general |
| mflux low-RAM mode / cache limit | Could reduce image-model memory; does not cover LTX video |

Pre-processing is the only option that guarantees the two models are never resident together, and it covers video and CUDA too.

## Test plan

- `plan_iterations`: seeds per iteration, random-choice expansion before enhancing, per-entry settings, JSON prompts skipped, skip/quit during planning, enhancer failure falls back to the original prompt.
- Stages: planned values are copied into artifacts and metadata.
- Runners: generation consumes the plan; retry keeps the rewrite and changes the seed.
- Web: event order and payloads (`prompts_enhancing`, per-iteration `prompt_enhanced`), enhancer released before the model loads.
- Frontend: progress panel shows the planning phase and each iteration's rewrite.
- Manual: the failing prompt file on the Mac enhances every prompt.

## Open questions

- Should the CPU fallback also be added, for when even the enhancer alone does not fit next to leftover memory?
- Should planning be skipped when the job has a single iteration (no benefit, same result)?
