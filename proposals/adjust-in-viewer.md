# Adjust in the viewer: crop and colour for images, trim for videos

**Status:** In progress (2026-11-15). Adjust for images is done (`feat(web): add Adjust for images in the viewer`, after 0.15.0); Trim and colour or crop on video are not built.

## Shipped: Adjust for images

**Adjust** is in the viewer's toolbar (after Edit), in the ⋯ menu of a tile and the viewer, and on the key `A`. It opens a full-screen layer over the viewer with three parts, **Crop** (shape chips, rotate left and right, flip), **Light & colour** (**Exposure**, **Contrast**, **Highlights**, **Shadows**, **Saturation**, **Vibrance**, **Warmth**, each -100 to 100) and **Colour mixer** (saturation for each of eight colours, each -100 to 100). On a computer the parts are accordion sections, one open at a time and all of them closable, with the last choice remembered; tablets held upright and phones use three tabs. **Compare** shows the original (tap, or hold the button, `M` or the picture) and **Reset all** puts everything back. **Save as copy** writes `<name>_adjusted.png` beside the source; **Replace** overwrites a PNG after a one-time question. Layouts exist for a computer, a tablet in both orientations and a phone in both (a **Save** menu holds the two save actions there). The user-facing description is in the docs under Adjust.

What was decided on the way, and differs from the first draft of this proposal:

- **Seven sliders, not four.** Brightness became **Exposure** (in stops), and **Highlights** and **Shadows** were added, because they are what a flat generated picture needs. **Vibrance** was added after hands-on testing.
- **A Colour mixer and sections, after testing.** Saturation per colour (eight bands placed by Oklab hue, with a fade so a grey is never tinted) was added after hands-on testing, and the computer panel's tabs became an accordion of three sections, so the mixer's eight sliders do not push the others out of view. Only saturation is per colour; hue and luminance per colour are not built.
- **Its own maths, not Post-processing's.** The saved file and the WebGL preview share one definition, built as lookup tables, and a fixture of pixels with expected results (`tests/fixtures/adjust_cases.json`) keeps Python and TypeScript equal. Post-processing's contrast and saturation are unrelated to it and stay as they are.
- **A copy records `copied_from`, not an `adjust` block or `source`.** It keeps the original's generation settings, so **Reuse settings** gives the original's settings and size, and **Made from** links to the original. The sliders are not recorded.
- **Replace was added.** The original proposal had no overwrite. It is limited to PNG, atomic, and asked about once.
- **Re-adjusting is not special.** It starts from the current pixels, as proposed.
- **A synchronous route, not a job** (`POST /api/adjust`), as proposed.

## Remaining: Trim for videos

A clip often has a bad first half second or a wobbly ending, and right now the only way to lose it is outside the app. That matters because **Extend** continues from the end of a clip: a bad ending carries straight into the extension.

### What the user sees

**Trim** takes the same toolbar slot as Adjust on a video, with `A` as its key. It opens a layer like Adjust's. The player's timeline gains a start handle and an end handle, and the parts outside them are dimmed. `I` and `O` set the start and end at the playhead while Trim is open; outside it, `I` still toggles Details. This builds on the frame stepping the player already has. A label shows the kept length, such as `0.4 s – 4.8 s · 4.4 s`. Playing loops the kept part. **Save as copy** writes `<name>_trimmed.mp4` beside the source, with the sound trimmed to the same range, and the viewer shows it as an upscale appears. Details shows **Made from**.

A trimmed copy is an ordinary clip, so Extend, Upscale and Use as all work on it.

- **Computer:** the player stays where it is; a side panel holds the start, end and length and the two "set at playhead" buttons.
- **Tablet:** landscape as on a computer; in portrait the panel becomes a dock under the player.
- **Phone:** full screen, with the handles on the player's timeline and **Start here** and **End here** buttons under it.

### How it works

- `POST /api/trim` with the asset id and a start and end in seconds, snapped to frames by the same rule `utils/video_point` uses for frame grabs.
- **Re-encoded, not stream-copied.** A stream copy can only cut on a keyframe, so the start would land up to a few seconds away from the handle. Trim re-encodes with the settings the app already uses (H.264, CRF 18, yuv420p, AAC) through ffmpeg, which every platform has for video. The embedded config is written as for any video, with a source link like Adjust's `copied_from`.
- A trim of a 10-second clip takes a few seconds of CPU, so it is a plain request with a busy state, like Adjust. ffmpeg failures use the existing tool error so the message says what went wrong.
- Trim is refused for a result shorter than 2 frames, and the handles can't cross.
- **Check:** Extend on a trimmed clip should continue from the trimmed end, because Extend reads the file; verify nothing in `web/extend_api.py` reads a recorded frame count instead.

### Tests and docs

- `tests/web/test_trim_api.py`: a path outside the output folder, bad values, a taken name and the recorded link; a tiny clip made by ffmpeg in the test. Frontend tests for the handle maths and the shortcut handling.
- A "Trim" section in `docs/guides/video.md`, the viewer actions in `docs/web-ui/gallery.md`, the keys in `docs/web-ui/shortcuts.md` and a `CHANGELOG.md` entry. No new config keys.

## Later

- **Colour and crop on video:** the same sliders and crop frame for video, with frames piped through ffmpeg and NumPy so the maths match images. Decide after Adjust has been used on images and Trim has shipped.
- **Remove sound** from a clip: a single ffmpeg flag. Add it to Trim if people ask.

## Alternatives considered

- **Stream-copy trims.** Rejected: cuts land on keyframes, seconds away from the handle.
- **Trim as a job in the queue.** Rejected: it would wait behind a long generation for a few seconds of work.
