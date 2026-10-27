# Controls for Web UI video jobs

**Status:** Implemented (2026-10-27); macOS check pending

## Shipped

Quit shipped in v0.13.0b19: Web UI video jobs get a `SkipSignal`, `VIDEO_SUPPORTED_CONTROLS` in `web/job_contract.py` is `("quit",)`, preflight and `run_video_batch` honour Quit, and a running video stops at its next denoising step (the step callback raises). [Review before upscaling](archive/upscale-review.md) then added Approve and Discard decisions for video, and a quit-only key listener in `ziv-video` while `--review` is on.

What remains is **Next** and **Pause**, which image jobs already have (`IMAGE_SUPPORTED_CONTROLS`).

## Proposed change

- Add `next`, `pause` and `resume` to `VIDEO_SUPPORTED_CONTROLS`, so the job card shows them for video jobs.
- Handle them in `run_video_batch` before and after each generation, as `run_batch` does for images. Preflight already honours both during prompt rewrites.
- Next during a running video stops it at its next denoising step, like Quit, but continues with the following video instead of ending the batch.

## Outcome

Implemented in full: video jobs take Next, Pause, Resume, Repeat and Quit, and `ziv-video` always runs the full key listener. Next during a review declines the upscale. Every video job offers Next; on MiniMax H3 text to video on macOS, which has no step hook, Next and Quit act when the clip finishes and the clip is kept. Left: a run on a Mac.

## Open questions (answered)

- Should `ziv-video` run the full key listener (`SkipSignal()` rather than `quit_only=True`) without `--review` too, as `ziv-image` does?
- Is `repeat` useful for video, or only for images?
