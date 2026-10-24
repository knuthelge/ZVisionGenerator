# Step down the video upscale batch when the GPU runs out of memory

**Status:** Proposed (2026-10-24)

## Problem

Upscaling an existing video with SeedVR2 on Windows and Linux picks its batch size from the GPU's memory: the largest 4n+1 batch whose estimated peak fits the card (`video_upscale_batch_size` and `estimate_seedvr2_video_bytes` in `utils/upscale.py`). The estimate's constants are fitted to about 20 measured runs on one card class (RTX 3080 10 GB), one driver and one PyTorch and SeedVR2 core version.

On another setup the real peak can be higher than the estimate: a different card, driver, allocator behaviour or a `make update-seedvr2`. The job then fails in its first chunk with "Not enough GPU memory to upscale to …" (`backends/cuda/seedvr2.py`), although a smaller batch would have fitted. The user has nothing to change in the UI to get past it.

## Evidence

- The batch rule and its fit: `utils/upscale.py` (`_VIDEO_MAX_BATCH`, `_VIDEO_BATCH_STEP`, `_VIDEO_HEADROOM_GB`, `estimate_seedvr2_video_bytes`, `video_upscale_batch_size`). The estimate is at or above every measured peak that finished and classifies every measured out-of-memory case as not fitting, but only for the measured setup.
- An out-of-memory error is turned into a `RuntimeError` per chunk in `backends/cuda/seedvr2.py`. The models stay cached between chunks (`cache_model=True` on upscale and decode), so a retry does not reload them.
- Measured on the reference card: at 3840×2160 a batch of 2, 3 or 5 runs out of memory after 80 s, and batch 1 finishes. An out-of-memory attempt costs 30–80 s before it fails.

## Proposed change

- When a chunk runs out of GPU memory, free the chunk's memory, retry the same chunk with the next smaller 4n+1 batch (down to 1), and keep that batch for the rest of the clip. The frames already written stay; the retried chunk starts from its first frame.
- At batch 1, fail as today with the readable out-of-memory message.
- The job card notes the step-down once (for example "Smaller batches to fit GPU memory"), and the progress total is recounted for the new batch.
- The recorded settings keep the batch that actually ran.

## Test

- Unit: a fake core that raises out-of-memory above a given batch; the runner steps down, keeps the smaller batch, and the frames out equal the frames in.
- GPU: force a budget larger than the card (or patch the estimate low) on a 10 GB card and check that a 1080p→4K and an `m` clip step down and finish.

## Alternatives considered

- **Probe the card before every job** with test batches. Exact, but costs a minute or more on every upscale, even where the estimate is right.
- **A larger fixed headroom.** Avoids some failures, but makes every card slower, and still fails where the estimate is far off.
- **Keep failing.** Fine on the measured setup; leaves users on other cards with a failure they can't fix from the UI.
