# Controls for Web UI video jobs

**Status:** Proposed (2026-10-04)

## Problem

Web UI video jobs cannot be stopped. They get no control signal (`submit_video_request_job` in `web/web_runner.py` passes none, and `VIDEO_SUPPORTED_CONTROLS` in `web/job_contract.py` is empty), and `run_video_batch` has no control handling. The only way to end a long video batch is to stop the server.

This matters more now that jobs run a preflight phase before the model loads (see [Enhance prompts before the model loads](enhance-before-model-load.md)). A video job with auto enhancement rewrites every prompt first, which can take minutes, and that phase cannot be stopped from the browser either. Image jobs already support Next, Pause and Quit during preflight.

## Proposed change

- Give Web UI video jobs a `SkipSignal`, like image jobs.
- Add `quit` to `VIDEO_SUPPORTED_CONTROLS`, so the job card shows **Stop** for video jobs.
- Pass the signal to `run_preflight` (`control=`), which already honours Quit at every rewrite boundary and during a rewrite.
- Honour Quit in `run_video_batch` between iterations: consume the signal before each generation and emit `batch_cancelled` (`job_cancelled` on the web).
- Optionally pass the signal to the video backends so a running denoising loop can stop early, as the image backends do.

## Open questions

- Should video jobs also support **Next** and **Pause**? Preflight already handles both; `run_video_batch` would need the same pre- and post-generation handling as `run_batch`.
- Can the LTX backends (MLX and diffusers) stop mid-generation cleanly, or only between iterations?
- Should `ziv-video` get the same key listener as `ziv-image`?
