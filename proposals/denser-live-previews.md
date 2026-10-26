# A live preview on every step

**Status:** In progress (2026-10-26)

**Shipped (unreleased):** a preview on every step but the last on both platforms, the macOS preview on the step's own report, the frontend note, docs and CHANGELOG, as measured below.

**Left:** the macOS check. Run one `zit` job in the Web UI on a Mac and confirm the steps report evenly, with a preview from step 1. Archive this proposal once it passes.

## Problem

An image job's live preview updates at 25%, 50% and 75% of the run (`PREVIEW_STEP_FRACTIONS`, `preview_milestone_steps` in `core/progress_events.py`). On the default presets the stage holds one picture for several steps and then jumps:

| Model | Default steps | Previews today | Every step |
|---|---|---|---|
| `zit` | 10 | 3 (steps 3, 5, 8) | 9 |
| `krea2`, `ideo` | 8 | 3 (steps 2, 4, 6) | 7 |
| `klein9b`, `klein4b` | 4 | 3 (steps 1, 2, 3) | 3, no change |

Runs with raised `--steps`, or a 50-step base model, see the longest gaps today.

## Evidence

A preview never runs the VAE. Both backends project the latents to RGB with a fixed linear map:

- **Projection:** `render_latent_preview` (`backends/mac/image_preview.py`, `backends/cuda/image/preview.py`) is one einsum from 16–32 latent channels to 3 at 1/8 of the output size, so 128×128 for a 1024×1024 image.
- **Clean-image estimate:** `estimate_clean_latents` adds one subtract and multiply. It needs the previous step's latents, 0.5–4 MB.
- **JPEG:** `_encode_preview_jpeg` (`web/web_runner.py`) writes a 128×128 JPEG at quality 85. That is a few KB, encoded outside the job record's lock.
- **Delivery:** the event carries only `preview_version`. The browser then fetches `/jobs/{id}/preview` once, and the server always returns the latest JPEG with `no-store`.

Each part is estimated to take well under a millisecond. None of it has been measured yet. A denoising step takes about 0.2 s to several seconds.

What a preview does cost:

- **CUDA:** one blocking wait per preview. `_project_to_image` builds its factor and bias tensors on the device on every call, and `.cpu()` copies the result back. Both wait for the step to finish, so the CPU cannot queue the next step early. The projection and the JPEG encode then run on the generation thread while the GPU idles.
- **macOS:** `call_in_loop` already runs `mx.eval(latents)` on every step, so a preview adds no wait. It is sent as a second `step_progress` report, though, so every-step previews would double the job's events. A job keeps its last 200 events (`max_history_events`), and a tab that reconnects rebuilds the phase timings from them. A 4-image × 30-step job would lose its early events and, on a reload mid-job, its phase durations.

On the frontend, swapping the image source quickly is safe. The current image stays up until the new one has loaded, and replayed events don't trigger fetches. Review before upscaling uses its own `preview_version` and is not affected.

The CLI renders no previews, because it passes no progress callback. Video previews are separate.

## Proposed change

Preview after every step except the last; the finished image follows straight after it.

- **Backend:** `preview_milestone_steps(total_steps)` returns `frozenset(range(1, total_steps))`, and `PREVIEW_STEP_FRACTIONS` goes. Both backends already keep the previous latents only when the next step is a preview step. That now means every step except the last two. Their partial-run offsets (`init_time_step`, `pipe.num_timesteps`) and best-effort handling stay as they are.
- **CUDA:** build the projection's factor and bias tensors once per device, as macOS already does, instead of on every preview. CUDA has no hook before the loop, so step 1 has no previous latents and previews the raw latents: text-to-image jobs open on a frame of noise and sharpen from step 2. That is kept on purpose (see Alternatives). macOS stores the starting latents in `call_before_loop`, so it previews the estimate from step 1.
- **macOS:** attach the preview to the step's own report, as CUDA does, so the event count stays one per step. The second report was added in the same change that made each step evaluate before it is reported. With the latents already evaluated, the projection no longer holds a step back.
- **Frontend:** `JobPreview.svelte` and `JobPhaseLog.svelte` each hard-code `FIRST_PREVIEW_AT = 0.25`. They show "First preview at 25%" and then "No live preview for this model" after 40% of the steps. Move the rule into one shared helper in `$lib/state/jobView`. The note becomes "First preview after the first step", and "No live preview for this model" shows once two steps have finished without a preview. Rebuild the SPA.
- **Docs:** `docs/concepts/images.md` explains the preview, so update it there. Reword the sentences in `docs/web-ui/workspace.md` and `docs/web-ui/first-image.md` and link them to Concepts. Add an `[Unreleased]` entry to `CHANGELOG.md`.

No setting and no throttle.

## Measurement plan

On the 10 GB GPU and 32 GB Linux machine, run the Web UI (`ziv ui --host local`) in its own systemd user unit and submit jobs through its API. CLI runs render no previews. Compare current code with every-step previews: same seed and size, interleaved, three runs each.

1. **Worst case:** a model that fits fully on the card (`klein4b` q4) at 512×512 with `--steps 30`. Steps are short and nothing streams, so the wait is the largest share of a step.
2. **Typical:** `zit` at 1024×1024 with its default 10 steps.
3. **In-process timing:** for runs 1 and 2, time the step-end callback with `perf_counter`. Time it once with `torch.cuda.synchronize()` before the timer starts and once without, to separate the wait for the step from the preview's own work.

The metric is the median interval between `step_progress` timestamps for steps 2..N (step 1 is warm-up), plus the callback times.

Ship rule: ship when the worst case slows steps by less than 2%, judged against the median and spread. Otherwise preview every 10% of the run, which is the same one-line helper change. For runs of 10 steps or fewer, every 10% and every step are the same.

macOS cannot be measured here. Running one `zit` job and checking that steps report evenly is a check before release.

## Measured (2026-10-26)

Setup:
- **Machine:** the 10 GB GPU and 32 GB Linux machine.
- **Harness:** a script that loads the model once and runs five interleaved rounds of no previews, today's milestones and every step, after one warm-up run. Each run goes through `DiffusersBackend.text_to_image` with `LivePreview` and the Web UI's `_encode_preview_jpeg`, which is the same render path a Web UI job takes, without the HTTP layer.
- **Worst-case model:** `zit` at q4 (NF4, fully on the card) in place of `klein4b` q4; it has the same property the worst case needs.
- **Not in the build:** the cached projection tensors were left out, and they only lower the cost.

| Run | No previews | Milestones (3) | Every step | Every step vs none |
|---|---|---|---|---|
| Worst case: `zit` q4, 512×512, 30 steps | 17.81 s (17.29–17.89) | 17.87 s (17.29–17.97) | 17.94 s (17.42–18.03), 29 previews | +0.7%, inside the spread |
| Typical: `zit@q8`, 1024×1024, 10 steps | 26.34 s (26.32–26.45) | 26.38 s (26.28–26.46) | 26.37 s (26.33–26.42), 9 previews | +0.1% |

The table shows the median job time per run, with min–max in brackets.

In-process timing:
- **Preview work:** about 0.44–0.48 ms per preview with the GPU synchronized first.
- **JPEG encode:** 0.10–0.15 ms.
- **Worst case:** the step-end callback waited about 190 ms for the step it was reporting. That is the step's own GPU time, which the next step would have waited for anyway. The real added cost is the preview work, about 0.6 ms × 29 ≈ 17 ms over a 17.8 s job, or 0.1%.
- **Typical:** the streamed `q8` model already syncs every step, so the callback did not wait at all.

Result: the ship rule is met by a wide margin, and every step is the change to make.

## Testing

These tests assert the 25/50/75% steps, or keep the latents only before a milestone, and are rewritten for every step:

- `tests/test_progress_events.py`
- `tests/test_mac_image_preview.py`, including the order of the step and preview reports
- `tests/test_cuda_image_preview.py`
- `tests/test_cuda_image.py`
- `JobPreview.test.ts`

New cases:

- **Step set:** steps 1 to N−1 for N steps; empty for a 1-step run.
- **Partial runs:** previews cover only the steps that run.
- **macOS:** one report per step, with the preview on it.
- **Frontend note:** "No live preview for this model" appears after two steps with no preview.

## Alternatives considered

- **Every 10% of the run:** the fallback if the measurement fails. It differs from every step only on runs longer than 10 steps.
- **A time throttle (at most one preview every 0.5 s):** it would cap the cost on fast runs, but it adds a clock to two backends to solve a cost that hasn't been measured.
- **A setting for preview frequency:** nobody needs to tune this.
- **Keep the second macOS report and coalesce step events in the job history:** this is more code in the runner, and it keeps a split that no longer has a reason.
- **Skip CUDA's step-1 preview, which shows noise:** built, then dropped. On img2img, refine and upscale passes, step 1 is the input image with some noise added, so the skip cost those short 2–3 step passes a useful frame. On text-to-image, the noise frame is part of watching a diffusion model work.
