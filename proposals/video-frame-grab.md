# Grab a frame from a video

**Status:** Proposed (2026-10-17)

## Problem

A video's loop ends at the viewer. Extend continues a clip from its last frames, but there is no way to take a moment out of a clip as a still, or to start the next image or clip from that moment. "Make a poster from this shot" or "start a new clip from frame 37" needs a screenshot or an external tool, and the result has no provenance and is not in the Gallery.

## Evidence

- The viewer plays videos with the native `<video>` controls, so the current time is known (`currentTime`).
- ffmpeg is already a requirement (`utils/ffmpeg.py`, `ensure_ffmpeg`), `utils/video_probe.py` reads fps, size and frame count, and `utils/video_tail.py` already decodes frames from a chosen start time with ffmpeg for Extend.
- `utils/provenance.py` embeds a config in PNGs (`embed_png_config`) and records a `source` block that the viewer turns into a link to the source asset (`_source_with_asset_id`), as it does for upscales.
- `referenceParams` and `useAsReference` make any image the reference for img2img or img2vid; a grabbed frame only needs to be an image in the output folder to join that path.

## Proposed change

### 1. Frame stepping in the viewer

- `,` and `.` step one frame back or forward while a video is paused (frame duration from the recorded `fps`, or the probe), and `Shift` steps ten. The time and frame number show under the player ("1.54 s · frame 37 of 121").

### 2. Grab frame

- A **Grab frame ▾** button in the viewer (key `G`) and in the tile menu for videos: **Save as image**, **Use for image (img2img)**, **Use for video (img2vid)**. The viewer grabs the paused time; the tile menu offers **First frame** and **Last frame** since no time is chosen there.
- `POST /api/video/frame` with `{asset_id, time_secs}` (or `frame`) runs ffmpeg with an accurate seek and `-frames:v 1`, writes `<stem>-frame-<n>.png` beside the video, and embeds a config: `workflow: frame`, the video's prompt and scene (so Gallery search finds it), and `source: {path, frame, time}`. It returns the new asset's JSON. It runs inline, not as a job: ffmpeg takes well under a second and needs no model, so the queue is not involved.
- For the reference entries the client then applies `referenceParams(newAsset, target)`, exactly as **Use as reference** does for an image, and the toast says "Frame 37 of clip.mp4 is now the reference image".
- The grabbed image's details show "Frame 37 · 1.54 s of clip.mp4" with the usual link to the source.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "Grab any frame of a video as an image, or use it as the reference for the next image or clip."
- The video guide's Web UI part gets a paragraph on frame stepping and Grab frame.

## Alternatives considered

- **Capture from the `<video>` element on the client** with a canvas. No server change, but the browser's decoded frame can be scaled or colour-managed differently from the file, and the image would still have to be uploaded to be a reference. Rejected.
- **Grab as a job** through the queue. The queue is for GPU work; a sub-second ffmpeg call would wait behind a 10-minute generation. Rejected.
- **A dedicated frame picker dialog** with a thumbnail strip. More to build than frame stepping in the player, and the player already shows the frame. Can follow if stepping proves too fiddly.

## Open questions

- Should the grabbed frame be upscaled with the video's upscale settings when the video was generated small and upscaled? The frame comes from the file that was played, which is the final one, so no.
- Seek accuracy on variable-frame-rate sources: the probe reports whether the frame rate is constant (`frame_rates_match`); for others the time is used and the frame number is approximate, which the details say.

## Out of scope

- Trimming or cutting videos.
- Using a frame as the last frame of a clip, which belongs to [First and last frame](archive/video-first-last-frame.md).
