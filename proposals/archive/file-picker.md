# Previews and sorting in the file browser

**Status:** Done (2026-11-07, unreleased; built on `feat/file-picker`)

## Problem

When you choose a start image, a reference, an audio track or a source video, the file browser shows an icon and a name per file. Outputs are named like `fox-portrait_2026-11-05_21-38-51.png`, so near-duplicates look the same, long names are cut off with no way to read them, dates have no time, and the list is always in name order, so the newest output is never on top. After you pick, the Workspace slot shows a picture only when the file happens to be a loaded recent output. On a phone the hidden-files switch is missing, the button doesn't say which file it chooses, and a phone held sideways gets the desktop layout squeezed into a sheet.

The browser itself works well: simple navigation like Finder or Explorer. This proposal keeps it and adds previews and sorting to it. Each choice below was picked from mockups of several options; the options not picked are listed under [Not doing](#not-doing).

## What the user sees and does

### Every picker

1. **Sort by clicking a heading.** Name, Size and Modified over the list are buttons. Click one to sort by it, click again to reverse it; a small arrow shows the column and the direction. Folders stay on top in every order (sorted by name when sorting by size). Names compare naturally, so `take-2` comes before `take-10`.
2. **Dates show the time:** "Today 09:12", "Yesterday 21:40", then "2026-11-03 14:05". The Modified column grows from 96 to 124 px. On a phone, which hides the date today, a second line under each name shows the size and date.
3. **Cut-off names can be read:** a tooltip with the full name on a mouse, and the full name wherever the selected file is shown (the footer, the preview, the phone's Choose button).
4. **The order is remembered** in this browser, one setting for every picker, like the hidden-files switch. The first time it is **Newest first**, so the file you just made or downloaded is on top.

This applies to every picker, including model, LoRA, prompt-file and folder pickers. Everything below applies to the **image, audio track and source video** pickers only; the others keep their icons and today's dialog size.

### Image, audio and video pickers

1. **Small thumbnails in the list.** The file icon becomes a thumbnail: an image, a still from a video (a frame 10% in, with a small play badge), or the waveform icon for audio (the real waveform is only in the preview). Folders keep their icon. Thumbnails have a fixed box, so rows never jump while they load, and the browser loads only the ones near the screen (`loading="lazy"`). A file that can't be previewed keeps today's icon.
2. **List or grid.** A List | Grid switch in the toolbar. The grid shows tiles with the picture and, under it, the name (two lines) and the date, without the year when it is this year (no size: the date matters more there). Sorting, filtering, type-to-jump and Enter work the same in both; in the grid the arrow keys move by row and column. The grid has no headings, so above it a "Newest first ⌄" button opens the same six orders as the phone's sort sheet.
3. **A preview you can hide.** A Preview button in the toolbar (also <kbd>Alt+P</kbd>) shows the selected file large (a 768 px picture), its full name (wrapped, never cut) and its facts: pixel size, file size and type, and the full date and time for an image; length, format (type, sample rate, channels) and file size for audio; length, frames, frame rate and pixel size for a video. A folder shows its icon. Arrow down the list and the preview follows. Off, the dialog keeps today's size. The first time it is on.
4. **Play audio and video in the preview.** Audio shows its waveform with a play button and scrubbing (the app's audio player). A video shows its still with the browser's own controls. Nothing plays until you press play; moving the selection or closing the browser stops it; only one thing plays at a time in the app.
5. **The view is remembered** with the order: list or grid, and preview on or off.

### Layouts

The phone's Places button, sort row, Choose label and landscape layout apply to every picker; thumbnails, List | Grid and the preview only to the image, audio and video pickers.

- **Desktop** (wide layout, also a tablet held sideways): the toolbar reads Back, Up, the path, List | Grid, Preview, Filter. With the preview on, a 300 px pane sits right of the list, the dialog grows from 768 to 1120 px wide and the browser is `min(600px, 100dvh − 6rem)` tall (today `min(560px, …)`). Rows stay 30 px with 24 px thumbnails; grid tiles are 132 px. With a finger (touch screen), the heading row grows to 44 px, rows are 44 px with 32 px thumbnails and tiles 150 px.
- **Tablet portrait** (tall layout, 640–1023 px wide): the same dialog; the preview sits in a 236 px strip under the list instead of beside it, and the browser grows taller (`min(780px, 100dvh − 6rem)`) rather than wider. Touch sizes as above.
- **Phone portrait** (compact layout): the browser is a full-height sheet, as today.
  - The row of place chips goes. A **Places** button at the left of the toolbar opens a sheet with Places, Recent (the last five files chosen, as on the desktop) and the **Hidden files** switch.
  - Toolbar: Places, Back, Up, the path (showing its end), List | Grid, Filter. Folder pickers have no List | Grid; their **New folder** is a 40 px icon button after Filter.
  - Above the list, a row says how many items there are and the order ("10 images · Newest first ⌄"). It opens a sort sheet with six orders: Name A–Z, Name Z–A, Newest first, Oldest first, Largest first, Smallest first ("Folders stay on top").
  - Rows are 60 px: a 40 px thumbnail (an icon in other pickers), the name, and a second line with size and date. The grid has three columns.
  - **Preview sheet:** tap a row's thumbnail, tap the selected row again, or press <kbd>Space</kbd> on it, for a sheet with the same picture, player and facts and **Choose this file**. A first tap elsewhere on a row selects it and a second tap on the selected row opens the sheet at once; a double tap does not choose on a phone (the Choose button and the sheet's **Choose this file** do). A double-click chooses on a computer, as today. The thumbnail button is a pointer shortcut only; keyboards and screen readers use the second activation or Space, because a button inside a list option isn't reachable.
  - The main button names the file: "Choose fox-portrait_2…1-38-51.png", cut in the middle, with the full name for screen readers (folder pickers keep **Use "folder"**). The sheet's close button and pulling it down cancel, as today.
- **Phone landscape** (compact layout because it is under 500 px tall): the phone layout, not the desktop one. The sheet is short, so the sort row and the footer fold into one toolbar: Places, Back, Up, the path, a sort button (opens the same sort sheet), List | Grid (or New folder in folder pickers), Filter, and a short **Choose**. The grid has four columns. The Places button is where it is upright, so nothing moves when you turn the phone.

### Workspace slots

Every slot shows a real thumbnail of the file it holds, from any folder, not only recent outputs. Nothing else about the slots changes; there is no playback in a slot.

- **Start image, references and last frame** (the Images section, its phone popover, the phone references list): the picture. An image uploaded from the device shows as today.
- **Audio track and voice recording** (their sections and phone popovers): keep the waveform icon; the real waveform is only in the browser's preview.
- **Source video** (its section and phone popover): the clip's last frame, made by the server, for every path (Extend continues from that frame).
- A file that is gone keeps today's alert state; a file that can't be previewed keeps today's icon.

Sizes stay as today: 40 px on the desktop and tablet, 56 px in phone popovers.

## Not doing

These options were shown and not picked:

- **Sorting:** a "Newest first ⌄" menu button instead of headings in the list (two clicks instead of one; the grid uses it because it has no headings); a Name | Newest switch (no reverse or size order); a date rail beside the list (only pays off in very large folders).
- **Rows:** icons only; thumbnails without a grid (the switch includes them); taller two-line rows (half as many rows fit); a zoom slider (hard to discover).
- **Preview:** no preview; an always-on pane (the hideable one covers it); a Space quick look over the list; a preview in the footer.
- **Audio and video:** length only with no playback; a play button on every row; play on select.
- **Slots:** a thumbnail you press to preview or play; a preview that expands under the slot; showing the input on the stage.
- **Phone:** the chip row with recent files and a ⋯ menu; two columns sideways; all controls in a bottom bar.
- **Remembering:** each picker or each folder keeping its own order; a guessed order with nothing stored.

Also not in this proposal: a Length column for audio and video (the preview shows the length), new places or picker modes, and changes to the path box's completion.

## How it works

### Server

The listing already carries each entry's size and modification time, so **sorting is done in the browser**; `list_directory` and its order stay as they are.

**Which pickers preview is said in one place.** `BrowsePurpose` in `web/path_browser.py` gains `preview: "image" | "audio" | "video" | None` (`reference_image`, `audio_track`, `source_video`; `None` for the rest), and `GET /api/browse/places` returns it in `purpose`. The routes and the frontend switch on it; nothing else lists purposes.

Three GET routes in `server.py` serve what the browser can't read itself. Each takes `purpose` and `path` like `GET /api/browse`:

| Route | Preview kinds | Returns |
|---|---|---|
| `GET /api/browse/preview` | image, video | A WebP picture, at most 384 px on its longest side (768 px with `size=large`, for the preview pane): the image re-encoded, or a video still. A video's still is the frame 10% in, or with `frame=last` its last frame (for the source video slot) |
| `GET /api/browse/info` | image, audio, video | JSON facts: `width`, `height` for an image (after EXIF rotation, so they match the upright preview); `seconds`, `sample_rate`, `channels` and the 100 waveform `peaks` for audio; `width`, `height`, `fps`, `frames`, `seconds` for a video |
| `GET /api/browse/media` | audio, video | The file itself, with range requests, for the players |

The file type shown in the facts comes from the extension. A new `web/browse_media.py` holds the logic, reusing what exists:

- `media_file(purpose, path_text)` is the one check every route goes through (see [Security](#security)).
- **Pictures** go through `utils/lora_preview.make_preview` (WebP, 384 or 768 px, refusing more pixels than Pillow's `Image.MAX_IMAGE_PIXELS`), not a second thumbnailer. It changes in three small ways: it accepts a path and an edge, so the pixel check runs on the header before the file is read; it calls `image.draft("RGB", …)` on a JPEG before converting, so a large photo decodes at reduced size; and it applies `ImageOps.exif_transpose` after shrinking the converted picture, then drops the metadata, so a phone photo isn't shown sideways. (Not `open_image_upright`, which flattens transparency.)
- **Video:** one `ffprobe_json(path, count_packets=False)` call gives the facts, with `frames` from the stream's `nb_frames`, else duration × fps. The 10% still is that probe, then `ffmpeg -ss` for one PNG frame for `make_preview` (two calls); the last frame is one `ffmpeg -sseof` call.
- **Audio:** one `ffprobe_json(path, count_packets=False)` call for `sample_rate` and `channels`, and one decode for the peaks. `utils/audio_peaks` gains a variant of `read_waveform_peaks` that also returns the decoded length (samples ÷ 8000), so `seconds` costs nothing extra and `probe_track_seconds` isn't called. A decode that fails or times out is a 404 like any other failure.
- Every ffmpeg and ffprobe call gets a timeout (10 s).
- **Two sizes.** The app has two picture sizes: 384 px (`PREVIEW_EDGE`: picker tiles and rows, Workspace slots, Discover) and 768 px (`LARGE_PREVIEW_EDGE`: the picker's preview pane, gallery cards and the strip). The Gallery and the Workspace strip load their 768 px thumbnails from `GET /thumbs/{asset_path}`, which confines the path to the output root as `/media` does and answers from the same cache and decode limit; it has no cross-site check.
- **A disk cache** in `<data dir>/cache/browse` (`get_ziv_data_dir()`, so `ZIV_DATA_DIR` moves it) keeps each picture, keyed on the variant (the `frame` and the edge), the resolved path, modification time and size, so a video's two stills and the two sizes don't mix and a changed file is read again. Only successful pictures are cached: a failure answers 404 and the next request tries again. Info answers are not cached on disk. Writes go through `utils/atomic_write`. It is capped at 256 MB; the oldest entries go first. Thumbnails of private files stay in the cache after the files are gone (until they are pushed out) and are part of a backup of the data folder; the data-folder page says so.
- **Revalidation and bounded work.** Responses carry an `ETag` from the cache key with `Cache-Control: private, no-cache`. After the cross-site and file checks, a matching `If-None-Match` is answered 304 from a `stat` alone, before any decode. A picture that is not cached is decoded under a module-level `threading.BoundedSemaphore(2)` taken on the worker thread, in the same `run_in_threadpool` call as the cache read, so at most two decodes run at once. `info` reads the file on each request and takes no slot.

`/media`, `/api/gallery-peaks`, `/api/audio-track/inspect` and `/api/extend/source` stay as they are.

### Security

**Approved position:** the Web UI may show previews of, and play, media files anywhere the browser can list. The listing already reveals names, sizes and dates of any folder to anyone who can reach the server, and anyone who can reach it can already type any path into these fields. The limits:

- **Only the purpose's own file types.** The purpose must have a preview kind the route serves, and the extension must be one the purpose accepts (`IMAGE_EXTENSIONS`, `AUDIO_TRACK_EXTENSIONS`, `SOURCE_VIDEO_EXTENSIONS`); files shown with **Show all** get no preview. Model, LoRA, prompt-file and folder purposes are refused.
- **Regular files only, links resolved and checked again.** The path must be absolute; it is resolved through symlinks, and the resolved target must be a regular file whose own extension the purpose accepts. A `photo.png` link to a key file is refused.
- **Images never leave as their own bytes.** They are decoded and re-encoded as a small WebP with no metadata, so EXIF, GPS and the settings a PNG embeds are not sent. Pictures with more pixels than Pillow's `Image.MAX_IMAGE_PIXELS` are refused before decoding.
- **Raw bytes only for playback**, only for audio and video purposes, with the media type set from the extension and `X-Content-Type-Options: nosniff`, so a file can't be served as a page.
- **Other websites.** None of these routes sends CORS headers, so a page on another site can make your browser load a preview or play a file but can never read the bytes or the facts. On top of that, the three `/api/browse` routes refuse a request whose `Sec-Fetch-Site` is `cross-site` or `same-site` (403), building on `_CROSS_SITE_FETCH_VALUES` in `web/request_guard.py`; that also stops another site embedding them. Browsers send `Sec-Fetch-Site` only to trustworthy origins (HTTPS and `localhost`), so this check works for the default local server. On `ziv ui --host local` over plain HTTP the header is absent and the check does nothing; there only the no-CORS rule holds: other sites can embed, not read.
- **Bounded work:** two decodes at a time, timeouts on ffmpeg and ffprobe, a capped cache.
- With `ziv ui --host local`, everyone on the network can now see previews of and play these files, not only their names. The Other devices page says so beside its existing "a network you trust" warning.

### Frontend

- **`lib/state/pathBrowser.ts`** gains pure helpers: `sortEntries(entries, order)` (folders first, natural names as a tie-break), `formatModified(seconds, now)` (Today, Yesterday, date and time), `formatModifiedFull` (with seconds, for the preview), `middleTruncate(name, max)`, `gridMove(index, key, columns, count)`, and `readBrowseView` / `writeBrowseView` for one `localStorage` key, `ziv-browse-view-v1` = `{sort, dir, view, preview}`, read with defaults when missing or invalid. The hidden-files key stays as it is.
- **`lib/api/browse.ts`** gains `browsePreviewUrl(purpose, path, options)`, `browseMediaUrl` and `browseInfo(purpose, path, signal)`; `BrowsePurpose` in `lib/types` gains `preview`.
- **`molecules/FileThumb.svelte`** (new): the thumbnail box for a host path and purpose: an `<img loading="lazy">` of the preview for images and videos (with the play badge for a 10% still), or the waveform icon for audio (nothing is requested); the file icon, or the icon the slot passes, on failure. Used by the list, the grid and the image and source-video Workspace slots.
- **`molecules/FilePreview.svelte`** (new): the large preview, its full name and facts, with `AudioPlayer` for audio and `<video controls preload="metadata">` with the still as poster for video. Used by the desktop pane, the tablet strip and the phone preview sheet. It fetches `info` 150 ms after the selection settles and aborts the fetch when the selection moves on, so arrowing down a folder doesn't queue a decode per row. It is remounted per file, so a change of selection pauses and releases playback; video joins `claimPlayback` like audio. When the browser can't play the format (for example `.mkv` in Safari), it says so under the still.
- **`molecules/AudioPlayer.svelte`** gets a third `size`, `pane`, for the 300 px pane, the tablet strip and the phone sheet (today `stage` and `viewer`).
- **`molecules/PathBrowser.svelte`**:
  - `visible` is the filtered, sorted listing. **The selection is kept by path, not by position**: today `active` is an index that `navigate` computes against the server's order, so after a re-sort Enter would choose a different file. The index is derived from the selected path, so a re-sort keeps the same file selected and scrolled into view, and type-to-jump walks the sorted order.
  - The heading row becomes buttons in a `role="toolbar"` labelled "Sort by"; the active one has `aria-pressed` and a name that includes the direction. The list keeps `role="listbox"` and `aria-activedescendant` in both views.
  - The phone body is keyed on `layout.compact` (a class on the root), not on `max-width: 639px`, so landscape gets it. The width-keyed `max-sm:hidden` classes (Hidden files switch, extension chip, Cancel, the New folder label) move to that class too. The Places, sort and preview sheets use the existing `Sheet` (`layer="modal-top"`).
  - The dialog uses Modal size `xl` while the pane is shown on the desktop. <kbd>Alt+P</kbd> is matched on `event.code` (`KeyP`), because ⌥P types a character on macOS.
- **Slots:**
  - `ReferenceSlot` and `PopoverImagePicker` take `path` and `purpose` instead of `previewUrl`; `ReferenceThumbContent` shows an uploaded file's object URL, else `FileThumb`. The six `galleryStore.previewUrlFor` calls (ImagesSection ×3, ImagesPopover ×2, PopoverReferences ×1) go, and with them that method and its tests in `lib/state/gallery.test.ts`.
  - The audio slots (`AudioTrackSection`, `VoiceSection` and their popovers) keep the waveform icon.
  - `SourceVideoSection` and `SettingPopover` use `FileThumb` with `frame=last` for every path, outputs included; the hidden `<video>` with a `#t=` seek goes.
- Run `make frontend-build` and commit the rebuilt SPA.

Everything here is the same on macOS, Windows and Linux; ffmpeg is already required for video and audio. Windows paths travel URL-encoded in the query.

## Testing

Mocked, at the contract level, as in AGENTS.md; no real media tools run.

- **`tests/web/test_path_browser.py`:** each purpose's `preview` kind, and `/api/browse/places` returns it.
- **`tests/web/test_browse_media.py`** (new), through the app's test client: each route answers its preview kinds and refuses other purposes (422), and files the purpose doesn't accept, folders and missing files (404); a symlink with an accepted name to another kind of file is refused; `preview` returns `image/webp` no larger than 384 px with no EXIF or PNG text chunks; `frame=last` calls ffmpeg with `-sseof` and is refused for images; `media` answers a `Range` request with 206 and `Content-Range`, sets the media type and `nosniff`, and refuses `reference_image`; `info` returns the documented keys per kind with one mocked ffprobe call and, for audio, one mocked decode; ffmpeg and ffprobe get a timeout and a timeout gives 404; `Sec-Fetch-Site: cross-site` and `same-site` get 403 on the three `/api/browse` routes, `same-origin` and no header pass; a second request is served from the cache, a changed modification time is not, `frame=last` still calls ffmpeg with `-sseof` after the 10% still of the same file is cached, a failure is not cached, and the cache drops its oldest entries over the cap (with a small cap); a matching `If-None-Match` gets 304 without decoding, on `preview` and `info`; `info` is never written to the cache; an audio file whose decode fails is 404.
- **`tests/web/test_web_request_guard.py`:** the cross-site helper for each header value.
- **`tests/common/test_lora_preview.py`:** a path input, an edge chosen by the caller, an EXIF-rotated JPEG comes out upright, a transparent PNG keeps its alpha.
- **`frontend/src/lib/state/pathBrowser.test.ts`:** `sortEntries` keeps folders first in all six orders and in size order, compares names naturally; `formatModified` for today, yesterday and older with a fixed `now`; `readBrowseView` falls back to defaults on bad data; `gridMove`, `middleTruncate`, `downsamplePeaks`.
- **`frontend/src/test-utils/pathBrowser.ts`** gains each purpose's `preview` kind and stubs for `browseInfo`, `browsePreviewUrl` and `browseMediaUrl`.
- **`frontend/src/lib/components/molecules/PathBrowser.test.ts`:** clicking Modified sorts newest first and a second click reverses it; after a re-sort the same file stays selected and Enter chooses it; type-to-jump follows the sorted order; the order, view and preview are written to and read from the one key across purposes; the grid moves by columns; Alt+P (by `code`) toggles the pane; a LoRA picker shows no thumbnails, switch or preview and requests nothing; the compact layout (also with only the height query matching) shows the Places button and the sort row, and the Places sheet holds the Hidden files switch; a folder picker on compact shows New folder in the toolbar; the phone Choose button names the file; the thumbnail, a second activation of the selected row and Space each open the preview sheet; nothing calls `play()` on its own and a change of selection pauses.
- **`FileThumb.test.ts`, `FilePreview.test.ts`** (new): the URL per kind, the icon after an error, the facts per kind, `info` not fetched until the selection settles and aborted when it moves.
- **Slot tests:** `ReferenceSlot` and `PopoverImagePicker` (image), `SourceVideoSection` and `SettingPopover`: a host path outside the output folder gets a preview or the last-frame still; an audio slot keeps the waveform icon; an upload and a missing file behave as today.

## Docs and changelog

- `docs/web-ui/workspace.md`, **Paths on the server**: sorting, list or grid, the preview and <kbd>Alt+P</kbd>, playing in the preview, the phone's Places button; the slot bullets mention the thumbnail. The browser's keys stay on this page with today's <kbd>/</kbd>, <kbd>Ctrl+L</kbd> and <kbd>Backspace</kbd>, and off `docs/web-ui/shortcuts.md` and the <kbd>?</kbd> list, which cover page-level keys; the Preview button's tooltip names Alt+P.
- `docs/web-ui/devices.md`: with `--host`, previews and playback reach other devices too.
- `docs/concepts/data-folder.md`: `cache/` also holds the file browser's previews, up to 256 MB, which outlive the files they show and are part of a backup.
- `docs/developer/web-server.md`: the three routes, the `preview` kind, the checks and the cache. `docs/developer/frontend.md`: `FileThumb`, `FilePreview` and the view key, in the path-browser paragraph.
- `CHANGELOG.md` under `[Unreleased]`: **Added** "The file browser shows thumbnails, a list or grid, and a preview that plays audio and video; Workspace slots show the file they hold"; **Changed** "Sort the file browser by name, size or date; dates show the time"; **Fixed** "On a phone the file browser has Hidden files and Recent in a Places sheet, names the file on its Choose button, and uses the phone layout held sideways".

No config keys or CLI flags change.

## Order of work

1. **Server:** `preview` on `BrowsePurpose`, the cross-site check, `browse_media.py`, the cache, the three routes, the `make_preview` and `audio_peaks` changes, with their tests. *Done when* `uv run pytest tests/web/test_browse_media.py tests/web/test_path_browser.py tests/web/test_web_request_guard.py tests/common/test_lora_preview.py` passes, including the 304, symlink and cross-site cases.
2. **Pure helpers, types and API client**, with `test-utils/pathBrowser.ts`. *Done when* `pathBrowser.test.ts` covers every new helper and passes.
3. **Sorting, dates and names** in `PathBrowser` for every picker, with selection by path. *Done when* the PathBrowser tests for heading sort, reverse, re-sort-then-Enter, type-to-jump and the remembered key pass.
4. **`FileThumb`, the list thumbnails and the grid.** *Done when* `FileThumb.test.ts` and the grid-movement and LoRA-picker-requests-nothing tests pass.
5. **`FilePreview`, the compact `AudioPlayer`, the pane and playback.** *Done when* `FilePreview.test.ts` and the Alt+P, no-autoplay and pause-on-change tests pass.
6. **The phone layout:** `layout.compact`, Places sheet, sort row and sheet, New folder placement, Choose label, preview sheet, landscape toolbar. *Done when* the compact tests pass with only the height query matching.
7. **Slots.** *Done when* the slot tests listed above pass and `galleryStore.previewUrlFor` has no callers left.
8. **Docs, changelog, build.** *Done when* `make frontend-build` leaves the committed SPA current and `make check` passes; then one look on a desktop, a tablet and a phone in both orientations.

## Decisions

Taken with the user on 2026-11-06:

1. **Sorting in the grid** (desktop, tablet and phone): one "Newest first ⌄" button above the grid, opening the six orders of the phone's sort sheet. The list keeps its clickable headings.
2. **No Length column** for audio and video; the preview shows the length.
3. **Preview from the phone grid:** a tap selects the tile; a second tap on the selected tile (or Space) opens the preview sheet, as for a selected row.
4. **The first order is Newest first**, in every picker, until the user picks another; the one shared setting then remembers it.
5. **No path typing on a phone held sideways**, as in the mockups; a path is typed upright.
