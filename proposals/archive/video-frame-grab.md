# Generate from a point in a video

**Status:** Done (2026-11-10, unreleased; built on `feat/video-frame-grab`)

## Problem

I pause a video at a point and want to carry on from there. Today Extend can only continue from the end of a clip, and a moment inside a clip can only become a first or last frame through a screenshot or another tool.

It is solved when, with a video paused at a point, I can:

- **Extend from that point:** continue the video using the last second before the point as context, keeping everything before it, as Extend does at the end of a clip.
- **Use that frame as the first frame** of a new, standalone video.
- **Use that frame as the last frame** of a video.

The only new thing that appears in the Gallery is the generated video.

## What the user sees and does

### Our own player

The viewer stays as it is. Only its player changes. One player of our own for video and audio replaces the browser's controls. It is wanted in its own right, not only for this feature. It is used in the Gallery viewer, in the Workspace stage where a fresh result first plays, and in the file browser's preview. The review step and the job card's audio button stay as they are.

- **Controls on the picture:** a row along the picture's bottom edge, as wide as the picture, with the timeline just over it. The row holds the time, then Play, Mute and, for a video, Fullscreen. They are laid out from the clip's size in the gallery's data, so nothing moves when the file has been read.
- **Timeline:** a thin line along the foot of the picture. It only seeks. Clicking or dragging never changes whether the media is playing.
- **Scrubbing:** dragging shows the picture live. Sliding up more than 48px during a drag slows the scrub, on any pointer. With a mouse, hovering the line shows a small thumbnail and the time; a finger sees it only while it drags.
- **Big Play:** whenever the media is paused, a large Play sits in the exact middle of the picture, or of a take's waveform, at every size and on every screen. The row's Play is always there beside Mute.
- **Click or tap the picture** to play or pause. A swipe over it pages instead. After a tap or a click, Space plays and pauses, on an iPad too.
- **Volume:** a slider opens beside Mute on hover or with the keys, and is left out where the browser ignores a page's volume (iPhone and iPad), where Mute stands alone.
- **Keys:** a video or take opens with focus on Close, so ←/→ page until you click the picture or timeline, or Tab to the timeline. Then ←/→ step the clip: one frame while paused and 1 s while playing. Shift makes it 5 frames, or 5 s. ↑/↓ set the volume, Home and End go to the clip's start and last frame, and Space plays. With focus anywhere else, ←/→ keep moving between assets. Every other shortcut still fires. Focus rings show only after a key.
- **It keeps what the browser's controls give:** play and pause, time, mute and volume, fullscreen, the poster, and one take playing at a time. It stays usable by keyboard and screen reader.
- **Audio** is the same player with its waveform as the timeline, the time and controls under it, and no frame actions.
- **Fullscreen is for watching.** The frame actions are outside it.
- **Glow:** in the viewer, a video has a soft glow of its own colours behind the picture, which follows each picture as it plays and is left out when the picture covers the viewer.
- **Swipe:** in the viewer the timeline, waveform, playback buttons and time button opt out of the swipe between assets. The picture still pages. On the stage the whole player opts out.
- **Phones:** the player sets `playsinline`, so an iPhone plays in the page and not in Apple's own fullscreen player.

### Pause is the selection

While a video is paused between its start and its end, after playing or seeking, the existing actions act on that frame. A video still at its very start, as every video opens, is not a selection. There is no marker you place. Pausing turns the time in the row into a button with a small arrow, which is a plain time again when playback resumes. The button opens a menu with what you can do with that frame: a popover above it where there is room, a sheet on a phone.

- **Extend** reads "Extend from 3.2 s". While the video plays, sits at its very start or sits at its end, Extend means the end of the clip, as today, and is not disabled. Only a point chosen after the start and under 25 frames into the clip disables Extend. The menu then shows its reason in place.
- **Use as** now appears for videos while paused between the start and the end, or stopped at the end, where it acts on the last frame. It has **First frame** and **Last frame**, the Workspace's own words. It follows the same rules as Use as on images: an entry a model or tab can't use stays visible, with the reason. Only models that take a last frame (LTX and H3 today) can use a last frame.
- **Where the actions sit.** The time button and its menu are in the Gallery viewer and in the Workspace stage, at every width. The stage keeps its ⋯ menu, which offers the same choices. The viewer's own buttons and keys do not change. Its `E` still extends the whole clip.

### What arrives in the Workspace

| Picked | Frame | Scene and Action | Other frame |
|---|---|---|---|
| **Extend from 3.2 s** | None. The clip is the source, with the handle at 3.2 s | Scene becomes the clip's scene and Action is emptied, as Extend does now | Not used |
| **Use as → First frame** | Fills First frame | Scene becomes the clip's recorded scene if it has one, and Action is emptied. With none recorded, the text is left alone. Never a file name | Stays. The toast says so, e.g. "First frame set. Still ending on <name>." |
| **Use as → Last frame** | Fills Last frame | Not touched | The first frame stays |

- Old frames stay because frames from different videos must survive: the start and end can come from different points or different videos.
- While the frame is extracted the time button shows a busy state, as does a closed ⋯ on the stage. If extraction fails, an error toast says so.

### Extend in the Workspace

- The source section shows the clip as a strip of its frames with a "continue from" handle. The part that is dropped is shaded, and the section reads e.g. "Keeps 3.2 s of 6.0 s". A handle at the end means the whole clip. The total length follows the cut.
- The handle stops at 25 frames and says why. **Generate** stays valid.
- The frame shown is the frame at the point, in the source section and in the phone popover, not the clip's last frame. During a drag it is a small preview; on release it is the exact frame.
- The point belongs to the source video: choosing another source resets it. **Retry**, **Load settings** and **Reuse** keep it.
- The result lands beside the original video, as an Extend does now. Extend writes a new file: the clip up to the point, then the new part. What came after the point is not in the new file, and the original is untouched.
- An accidental pause is caught here: the cut is visible, it is one drag from undone, and nothing runs until **Generate**.
- A clip made from an audio track continues the song at the point. Its start moves back by the removed frames, which is visible in **Start at**.
- **CLI:** `ziv video --extend` gets a flag for the point to continue from, and the Web UI's copied command includes it.

### A last frame alone

A last frame needs a first frame. With a last frame set and no first frame, **Generate** is disabled and its reason names the empty First frame slot. Today the Workspace keeps that last frame but silently ignores it and makes a text-to-video clip, so this changes the Workspace for every last frame, not only one from a video.

## How it works

**The point.** The player sends the time of the frame on screen. Where the browser has `requestVideoFrameCallback`, the player takes that time from it, while the video plays and after it pauses, because `currentTime` runs one to three frames ahead of the picture once a video has played. Without it the player sends `currentTime`. The server probes the clip and snaps the time to a frame.

**A frame as a first or last frame.** A small route takes the clip and time, and extracts that frame at full size as a PNG into `.web_uploads`, the staging folder uploads already use, which the Gallery does not list. The PNG stays there, like an upload, and nothing cleans that folder today. The file is named after the clip and the time, for example `<clip>_3.2s.png`, so the slot and the toast show a readable name. The route returns the path, and the Workspace puts it in the slot. The job is an ordinary image-to-video job, and Reuse treats the frame like an upload.

**Hover thumbnails.** The server makes one contact sheet of small frames per clip, the first time it is needed, for the hover thumbnail and the Extend bar; until it is ready, the hover shows the time alone and the bar is plain.

**Extend from a point.** ffmpeg cuts the clip at that frame into a temporary file once per job, sound included, and that shorter clip goes through the existing Extend. Tail reading, audio and joining need no knowledge of the cut.

- The job rebuilds its source from the cut's own timing and keeps the original clip's recorded settings, because the temporary file carries no scene, prompt or audio track.
- The output records the original clip and the cut frame, so Reuse restores both.

## Evidence

- Extend takes only a source path today (`extendParams`), reads its tail and appends to the whole file (`utils/video_splice.py`).
- Video jobs already take `image_path` and `end_image_path`, and `.web_uploads` is a staging folder that is not an asset (`web/gallery.py`).
- The viewer is a bare `<video controls>`. It has no `playsinline`, so an iPhone may play in its fullscreen player, out of reach of our timeline and keys; the new player sets it.

## To verify when building

- **Colour.** Clips made on Windows and Linux carry no colour tags (`cuda/video/encode.py`), and ffmpeg's default for untagged video is BT.601. Check on a real clip from each platform that an extracted frame matches the player, and set the conversion from that. Extend from a point adds no new risk here, because it reads frames as Extend does.
- **Exact frame.** Check with a test clip made by the app, with the frame number encoded in it, that the frame used is the one on screen, in Firefox, and Safari on a Mac and an iPhone. Chromium and WebKit pass.
- **Cut quality.** A stream copy can break the last frames before a cut, so the cut re-encodes the kept part once, at the quality Extend's own re-encodes use (CRF 12). Measure what that costs.
- **Scrubbing feel on a real clip.** Clips are encoded as they are today. Scrubbing is checked on a real clip in a real browser at the end; if it is actually bad we revisit it.

## Alternatives considered

- **A marker placed on the timeline, with an action menu.** Rejected: every seek paused the video and left a marker, and the marker and the picture could disagree. Pausing is the selection instead.
- **A frame picker inside the First frame and Last frame slots.** Not now. It is the largest piece to build, and pausing already does the job.
- **Dragging a frame onto a slot.** Left for the Studio. It is hidden, and it needs a menu anyway.
- **Capture the frame from the `<video>` element with a canvas.** No extraction route, but browsers differ on colour and nothing guarantees exact pixels. Rejected.
- **Teach Extend's tail reader, audio and join about a cut point.** Avoids the temporary file, but spreads the cut through every step of Extend. Rejected.

## Out of scope

- Saving a frame to the Gallery as an image of its own.
- A frame number readout.
- Trimming or cutting videos as an edit of their own.

## How it ties into the Studio

A range in the [Studio](../studio.md) is the same choice made at two points, so its start and end frames can use the frame extraction, and the Workspace's handle can seed a range end. The Studio keeps its frames in its own working folder, so the destination has to be something the caller can choose.
