# Frontend

The Web UI is a Svelte 5 app in TypeScript, in `frontend/`, built with Vite and pnpm into `zvisiongenerator/web/static/app/`. The built files are committed, so the package works without Node. This page holds the rules the components follow and the reasons for them. For what a component does, read its source; they are small and named for what they are.

## Develop

You need Node.js 22.22.2+ (22.x), 24.15.0+ (24.x) or 26+ and pnpm 11 or newer.

```bash
make frontend-install   # install packages
make frontend-test      # svelte-check and Vitest
make frontend-build     # build into the package; commit the result
```

`make frontend-dev` starts the Vite dev server with hot reload. It proxies the API to a server on port 8765 (`frontend/vite.config.ts`), while `ziv ui` defaults to 8080, so run both:

```bash
uv run ziv ui --port 8765 --no-browser
make frontend-dev
```

Run `make frontend-build` and commit the rebuilt files after every change: `make check` fails when the committed build is out of date. It also fails on a raw colour in a Svelte file (`make frontend-palette-check`) and on a bare `:hover` rule (`make frontend-hover-check`). Tests sit next to their source as `*.test.ts`; see [Testing](testing.md).

| Folder | Holds |
|---|---|
| `src/app/` | The app shell, routing and `global.css` with the design tokens |
| `src/features/` | One folder per page |
| `src/lib/components/` | Shared components, as `atoms`, `molecules` and `organisms` |
| `src/lib/state/` | Stores and pure helpers shared across pages |
| `src/lib/api/` | The API client, one module per area |
| `src/lib/actions/` | Svelte actions for drags, holds and sheets |

| Page | Folder |
|---|---|
| Workspace (Image, Video and Audio tabs) | `src/features/workspace/` |
| Gallery | `src/features/gallery/` |
| Prompts | `src/features/prompts/` |
| Models | `src/features/models/` |
| Config | `src/features/config/` |
| Job log (a drawer) | `src/features/jobLog/` |

A page folder holds its page component, the components only that page uses and `*View.ts` files with the pure logic that decides what the page shows. Put logic that a test should reach without rendering in a pure module and keep the component thin.

## Confirmations, downloads and paths

Ask the user to approve an action with `ConfirmDialog` (`lib/components/molecules/`), never `window.confirm`. Call `requestConfirm({ question, info, confirmLabel })`, which resolves `true` on confirm, or render `<ConfirmDialog>` directly when the dialog needs a body or stays open while the action runs. The dialog has no title: the question, an optional line of information, and a cancel button plus a confirm button named after the action (danger-styled for destructive ones). While it is open, no other key reaches the page or a viewer underneath.

**Download questions.** Every action that would download a model asks through `confirmDownloads(plan)` (`lib/state/confirmDownloads.ts`), which resolves `true` at once for an empty plan and otherwise shows one dialog with a block per part, its size, its licence lines and the free space. A job plan comes from `planGenerate` or `planUpscale`; a Models-page row's plan (`RowDownloadPlan`, which extends `JobDownloadPlan` with a label) goes to `confirmDownloads` as it is. A new action that downloads adds its plan request and this one call.

Let the user choose a path on the server's machine with `PathField`, or `PathInput` and `PathBrowser` where a field needs its own layout. Each takes a browse purpose; the purposes live in `web/path_browser.py`, so add one there for a new kind of path.

## Licence marks

`LicenceMark` shows the terms of a model from its `licences` (the records the server sends with every model option, Models-page row and plan). It renders nothing for an empty list. Without a `label` it is a muted info button; with one, such as `Non-commercial`, a neutral badge button. The card moves to `<body>` through the `popover` action because the Models tables clip. Put the mark beside a control, never inside a button.

## Toasts

Report the outcome of an action with `addToast(message, type, options)` from `$lib/state/toasts.svelte`. One toast shows at a time; the rest wait their turn. A repeat of a waiting or showing message merges into it, an error goes ahead of every non-error and stays until dismissed, and hovering or focusing a toast pauses it. Pass `{ action: { label, run } }` for one follow-up such as Retry. An action that acts on a page must not outlive it: keep the id `addToast` returns and call `dismissToast(id)` in the page's `onDestroy`. The exception is the download lane (`lib/state/downloads.svelte.ts`), whose store raises the done and failed toasts so they stay after the page that began the download is gone. Raise one toast per action, and skip it when the screen already shows the result.

## Requests and errors

Send requests through `api` in `$lib/api/client`, which throws `ApiError` with the HTTP `status` and the server's `detail`. Show a failed request with `errorText(err, fallback)`, never `err.message`, which holds the method, path and raw body. Offer Retry only when `isRetryable(err)`: a 4xx means the server refused the input, and the same request fails the same way again.

## Design system

The Web UI has one design system. Its tokens live in `@theme` in `frontend/src/app/global.css`, and its component classes are the `ui-*` classes in the same file. Build pages from the shared components in `frontend/src/lib/components/` (`Button`, `Input`, `Select`, `Toggle`, `Badge`, `Panel`, `Segmented`, `ActionBar`, `PageHeader`, `KeyValueList`, `Alert`, `EmptyState`, `InspectorSection`, `InspectorRow`) or, where no component fits, the `ui-*` classes, rather than styling elements by hand.

- **Size:** controls are 28px (`h-control`), small ones 22px, rows 30px. Buttons that sit together share one height; inside an `ActionBar` every button is 36px and the one marked `main` takes the remaining width.
- **Type:** four sizes (`text-meta`, `text-ui`, `text-content`, `text-title`), plus `text-display` for the first-run headline only. Nunito (`font-heading`) is for titles, area names and the main action; monospace is for numbers, paths and shortcuts.
- **Labels:** field and row labels are sentence case (`ui-label`). The one small-caps style (`ui-area-label`) names areas: panels, sections, table headers and menu groups.
- **Shape and surfaces:** corners are 4px for chips and badges, 6px for controls and 10px for containers. Overlays (menus, popovers, tooltips, toasts, dialogs) use `bg-overlay`, darker than any page surface, with a faint ring and never a drop shadow.
- **No borders:** containers are separated by shade, and buttons, fields, chips and badges are filled shapes. Draw an outline only where it means something: focus, a field with an error, the selected tile, and the dashed edge of drop zones and add buttons.
- **Main action:** each section (a bar, panel, form, dialog or popover) has at most one main action, a `primary` button. Every other button in the section is secondary.
- **Reference pages:** Config and Models centre their content at `max-w-page` (1200px). Settings use label-left rows (`FormField layout="row"`), and tables use `ui-table-comfortable`.
- **Colour:** neutrals are Graphite greys with no tint. Teal means on, selected or the main action; coral marks `$snippets` and amber `{a|b}` choices; red is for errors and destructive actions. Text is at least 4.5:1 on every surface. Use tokens, never raw palette classes or colour literals.
- **Focus:** every control shows the same 2px teal outline with a 2px offset.
- **Touch is a layer, not a breakpoint.** `@media (pointer: coarse)` in `global.css` raises the size tokens at every width, so an iPad in the desktop layout gets finger-sized targets and a mouse keeps the desktop density. Build sizes on the tokens, never on literal pixels, so they follow. Numbers get stepper buttons (`InspectorNumber`) because a drag scrolls on touch. The `layout` store (`lib/state/layoutMode.svelte.ts`) tells components the composition (`compact`, `tall` or `wide`) and whether the pointer is coarse.
- **Hover only where there is hover.** A touch screen keeps `:hover` on whatever was tapped last, so every `:hover` rule sits inside `@media (hover: hover)` (Tailwind's `hover:` variant already does). A finger gets an `:active` pressed state under `@media (hover: none)` instead, and anything a mouse reveals on hover is always shown there. On a phone, fields are at least 16px so focusing one does not zoom the page.
- **The visible viewport.** `lib/state/viewport.ts` sets `--app-height` and `--app-top` from `window.visualViewport`. The app root and every full-screen layer use `.ui-viewport`, so they sit above an on-screen keyboard and follow Safari's toolbar. Build full-screen layers on `.ui-viewport`, never on `inset: 0` or `100vh`.
- **Three compositions of one Workspace.** The same panes are arranged by `layout.mode`: *wide* (1024px and up) puts the sidebar beside the stage; *tall* (640 to 1023px) puts the stage and strip over Compose and Settings side by side; *compact* (under 640px) makes the stage the screen, with a top strip, a drawer, a strip of setting chips and a composer. Everything stays inside the Workspace form, so every field submits in every composition. Nothing is hidden behind a phone pattern on a wide screen.
- **Stable frames.** Nothing moves under a finger while a page loads. A page draws its complete frame synchronously and requests only fill it; a loading state lives inside the slot it is loading and never changes the height of anything around it. Pages remember what they loaded in `lib/state/pageMemory.ts`, draw it at once and then ask the server again. Anchors (top strip, composer, page bars) keep their place and size. Page switches stay a hard cut.
- **One Gallery.** The strip, the stage's stepping through earlier outputs, the Gallery page and the viewer all read `lib/state/gallery.svelte.ts`, which keeps one list per `filter|sort|starred` query, so `removeAssets`, `mergeOutputs` and `setStarred` reach every list. `starAndReconcile` (`lib/state/starAndReconcile.ts`) is the one way a star is set: it calls the API, then updates the Gallery store and the job's outputs. There is no history route or store.
- **One Select mode.** The Gallery page has one `selectMode` on every layout; `features/gallery/selection.ts` holds its pure rules (what a click means, a Shift range). The selection is a set of IDs that leaves a file only when it is deleted or moved, never because its tile is not loaded: Select all fetches every matching ID, and the bulk actions act on the IDs. A bulk action with no failure leaves the mode; failed files stay selected. `features/gallery/GalleryPage.test.ts` pins it.
- **One viewer, native on each device.** `AssetViewer` follows the open asset by id, never by place, so new outputs or a delete elsewhere never move the view. On touch it pages with a sideways drag, closes with a pull down and opens details with a pull up; a hold or a right-click opens the asset menu. It is an escape layer: Escape closes it.
- **One sheet, and drags that follow the finger.** Every sheet on a phone is `molecules/Sheet.svelte`, and drags are built on `actions/pan.ts`, which decides on the first movement whether a touch is the surface's or the page's (deciding later loses to native scrolling on iOS and Chrome). A press and hold is `actions/hold.ts`. The Prompts page arranges sets and entries with pointer events, not HTML drag and drop, which never fires for a finger.

### State rules

- **One scope table.** Every field of `DraftState` has exactly one owner in `FIELD_SCOPE` (`lib/state/resetScope.ts`); a field added without a scope fails `svelte-check`. A transition is a list of scopes: Reset puts a tab's scopes at `defaultsFor(...)`, and Reuse and Load settings do the same plus the text and the model. Only a Reuse and a Load settings are exact; a Use as, a deep link and a preset apply only what they name. Anything a new field needs beyond that goes into `defaultsFor`, not into a keep list.
- **One settings table.** `SETTING_ROWS` (`features/workspace/settingRows.ts`) lists the Settings pane's rows with their draft keys, phone chip and visibility. The changed dots, the Reset badge and each popover's Reset all read it, and a test compares it to the setting scopes. Image and Video, and Edit and Extend within them, each keep their own text, settings, model with its LoRAs and quantization, and preset, saved when another slot opens. Batch size and Seed belong to the slot as well.
- **One view per slot group.** `imagesView`, `audioTrackView` and `audioView` derive from the draft, the model's capabilities and the held uploads what the open tab uses, and `WorkspacePage` computes each once and passes it down, so what is shown, what gates Generate and what `buildFormData` sends always agree. Uploads persist in IndexedDB (`lib/state/uploadStore.ts`), and Generate waits until they are restored.
- **Tabs.** The top bar has `image`, `video` and `audio` tabs (`navTab(workflow)` in `lib/state/workspaceTab.ts`). Extend is a kind of Video and Edit a kind of Image, each chosen with the Kind control and offered when the contract lists the workflow; `draft.state.imageKind`, `videoKind` and `audioKind` remember the kind a tab was left on. `workspaceTab(workflow)` gives `extend` and `edit` a slot of their own, so nothing done in one of them, a Reset or Reuse included, changes New. `draft.state.workflow` is derived from the tab and its start image. Extend and Edit offer only the models that can do them (`lib/state/workflowModels.ts`). The prompt text is kept per slot.

## Edit in the Web UI

Edit is a workflow of the draft, the way `extend` is: the Image tab's Kind is `draft.state.workflow === 'edit'` or not, and `workspaceTab('edit')` is a slot of its own with its own reset scope (`RESET_SCOPE.edit`, `REUSE_SCOPE.edit`) and held uploads. `workspaceCapabilities` reads the workflow's `visible_controls`, never its name, so a control Edit doesn't list stays hidden without a branch (which is why Edit has no preset, prompt file or Enhance); `isEdit` is the one capability that names it. A prompts file belongs to New, so *Use in Workspace* and *Generate this one* call `draft.setKind('image', 'new', ...)` first. `declinesModel` keeps the workflow's own model when a Reuse or Retry records one that can't edit.

The code is in `features/workspace/edit/`. The draft's `edit…` fields belong to the Edit slot, so a Reset or Reuse of another tab never touches them. The painted mask is held like the other uploads (`heldUploads.editMask`) with a marker in the draft, and strokes and undo live in `features/workspace/edit/editStore.svelte.ts`. The painter is a full-screen layer of its own, not a `Modal` or `Sheet`, whose pull-to-close would take strokes, and its mask canvas is capped at `MASK_MAX_PIXELS` (4 MP), which iOS needs. The draft never holds a Paint and an Extend together: `draft.commitEdit` (what Done calls) stores one mode with its work, `draft.setEditSource()` is the one place that drops that work when the source changes, and `editFormData` sends only the active mode's fields.

`features/workspace/edit/editForm.ts` builds an Edit submit from the draft, not from `new FormData(form)`, which would carry the New form's fields; it sends the fields in [Edit jobs](web-server.md#edit-jobs). A cut-out's transparency is read in the background when the picture is chosen, and an unknown one counts as opaque until it is known.

## Audio in the Web UI

An audio take plays in `MediaPlayer` with `kind="audio"` on the stage, in the viewer and in the file browser's preview. The waveform is the timeline, a `role="slider"` with a canvas inside, drawn from the take's peaks. A gallery listing sends `peaks: null`; `audioPeaks.ensure(asset)` fetches them once per asset. A take never autoplays.

- **The Audio tab.** The kind alone decides what the tab offers (Lyrics for Music, Length for Sound, Voice for Speech); `AUDIO_KINDS` in `lib/types` is the one list. `lib/state/audioDraft.ts` holds the pure parts, `lib/state/musicMode.ts` mirrors the song-or-instrumental rule of `music_mode` (tested against `tests/audio/music_mode_cases.json` with the Python rule), and `features/workspace/audioView.ts` derives what the open kind sends and the one reason Generate is off.
- **What Generate sends.** `audioFormData` builds the fields Copy CLI reads for the open kind only (see [Web server](web-server.md#audio-jobs)). A setting goes only when its control is shown for the open model and kind. A blank `tempo`, `key` or `seconds` is Auto. The plan, the download dialog and the submit all read this one form. The `audio_settings_cases.json` and `audio_cli_cases.json` fixtures pin the round trips in both Vitest and pytest.
- **The speaker's consent.** The tick of a one-off recording lives in `lib/state/voiceConsent.svelte.ts`: in memory only, never in the draft, an address or a job's settings. A new recording, Reset and a prefill that replaces the voice clear it. A saved voice asks for none: the server keeps its consent in the voice's record.
- **The voice.** `voiceMode` is `builtin`, `recording` (a one-off) or `saved`. A saved voice that the list no longer holds keeps the choice and turns Generate off; only this tab's own Delete moves the draft back to the default voice.
- **Use as for takes.** The asset menu and the viewer's Use as list **Audio track** and, for a speech take, **Voice** and **Saved voice**. Both go through the prefill a Reuse uses with `use_as` set, so only that one slot changes. `voice_clone` tells the Speech tab to show the consent tick, because a voice that is itself a clone is a person's; the server decides the same way at submit (`voice_needs_consent`).
- **Job card.** A job with `media_type: "audio"` counts takes and shows `AudioAurora`, whose phases come from `audio_progress` events kept as `job.audio`. The header bar follows `audioFraction`, so it only moves forward, and under `prefers-reduced-motion` the canvas redraws on each event only.
- **Models page.** An audio model's memory fit is the GPU's alone, so the badge words an estimate without `system_gb` as GPU memory only.

## The media player

`molecules/MediaPlayer.svelte` plays video and audio in the viewer (`size="viewer"`), on the stage (`size="stage"`) and in the file browser's preview (`size="pane"`). A caller keys the player on the file (`{#key asset.url}`), so paging builds a new element. The review stage and job preview keep native controls.

- **Time and frames.** A video's facts come from `videoFacts` (one `GET /api/browse/info` per path) and its contact sheet from `videoSprite`; without facts the player works in plain seconds. `currentTime` runs ahead of the picture after a video has played, so the player reads the time of the picture itself from `requestVideoFrameCallback`.
- **Seeking.** The timeline only seeks and never changes playing or paused. One seek is in flight at a time and a newer target replaces one still waiting. The target is the middle of a frame; `fastSeek` is never used, since on a clip with one keyframe it lands on frame 0.
- **Keys.** `playerKeyCommand` (`mediaKeys.ts`) maps the keys. The player marks every key it handles `preventDefault`, and its listeners run before the pages' own, so the viewer and the Workspace stand aside for a key that arrives `defaultPrevented`. The timeline is a `role="slider"` element and never an `<input type="range">`, which `isTyping` counts as typing. The volume slider exists only where the browser keeps a volume a page sets, which iPhone and iPad Safari do not.
- **Touch.** In the viewer the controls carry `data-own-drag` and the root does not, so a swipe or a hold on the picture is the viewer's; on the stage and in the pane the root carries it.
- **Loading.** The video is created with `preload="metadata"` and becomes `auto` after 300 ms or on the first play or seek, so a clip paged past never downloads in full. When the player goes it pauses, removes its `src` and calls `load()` to free the decoder and the connection.

### Pause as the selection

A video paused after its first frame offers that frame: Extend from it, or use it as a first or last frame. `framePoint` (`lib/state/framePoint.ts`) is the only place that decides whether there is a frame; every surface builds its rows from the `FramePoint` it produces, so the surfaces cannot disagree. `molecules/frameMenu.ts` is the one row builder, and a row that cannot be used keeps its place, disabled, with its reason as the hint. `runFrameGrab` posts `{path, time}` to the frame route and ignores a second request for the same video while one runs. A point at the end acts on the last frame for Use as only: Extend stays on the whole clip.

## Projects

The pick is `projectPick.current` (`lib/state/projectPick.svelte.ts`): a project name, or `null` for All files, stored in `localStorage`.

- **`choose` owns the pick.** `projectsStore.choose(name)` is the only place a pick changes: it sets the pick, zeroes the Gallery page's remembered scroll and calls `galleryStore.reset()`. Nothing else may call `projectPick.set`. A pick whose folder is gone returns to All files with a warning.
- **One scoped store.** There is one gallery store, and the pick is part of what it loads, so every reader (page, strip, stage, viewer) shows the picked project unchanged. A pick change resets the store rather than keeping a view per project. `mergeOutputs` adds a finished job's outputs only when `inProject` accepts them, and `completionToast` adds `Saved to …` for a result outside the pick.
- **Not in a project** is a `MediaFilter` value (`no_project`), not a project name; `filterAdmits` is the one rule for what a filter shows.
- **Global reads.** The recent LoRAs and the last seed must not follow the pick, so they read `galleryStore.allRecent`.
- **Move.** `moveAndReconcile` (`lib/state/moveAndReconcile.ts`) is the move the Gallery and the Workspace share. Moves run in turn through one promise chain. A moved file is never added to the Gallery page's deleted-ids filter, because a file moved away and back regains its old id. A moved file follows its slots (`draft.retargetPaths`); a delete empties them (`emptySlotsHolding`).

## Voices

`voicesStore` (`lib/state/voices.svelte.ts`) holds the saved voices and the voice designer's kept candidates; mutations throw so the caller shows the server's reason. `loaded` stays false until the first read, so a draft that names a saved voice is not called deleted before the list is known. The Voice dropdown and its buttons open one **Voices** window, `VoicesWindow.svelte`; whether it is open and on which tab is `lib/state/voiceWindow.svelte.ts`, because several places can open it.

### Preview clips

The Play button of a default voice plays `/app-static/voices/<id>.mp3`, a clip in `frontend/public/voices/` that the build copies into the packaged app. `tests/contracts/test_voice_previews.py` fails when a configured voice id has no clip or a clip has no voice. The clips are made from the model and committed. Each reads the same sentence with seed 1 and is encoded to a 64 kbit/s mono MP3 with the tags removed:

```bash
out=$(mktemp -d)
text="Hello, this is what my voice sounds like. I can read your story, your script, or anything else you write."
for id in ryan aiden vivian serena uncle_fu dylan eric ono_anna sohee; do
  uv run ziv audio -m qwen3_tts --speaker "$id" --prompt "$text" --seed 1 -o "$out"
  ffmpeg -y -i "$out"/speech_*.flac -map_metadata -1 -ac 1 -b:a 64k "frontend/public/voices/$id.mp3"
  rm "$out"/speech_*.flac
done
```

Level each clip with `-af volume=<gain>dB` on the ffmpeg call: one fixed gain per clip that brings it to about -18 LUFS, held back where a peak guard needs it. **Remake the clips whenever the `voices` list, a voice's id or the model's checkpoint changes**: the test only checks that the files exist. Then run `make frontend-build` and commit the rebuilt files.

## Discover LoRAs

The drawer lives in `features/workspace/discover/`: a side `Modal` (a `Sheet` on a phone) mounted by `WorkspacePage`, so it shares the page's draft and download state. `discoverView.ts` holds its pure logic and requests go through `lib/api/discover.ts`. The Models page opens it with the hash `#/workspace?discover=loras`.

- **Downloads.** A LoRA download is a lane item of kind `lora` in `downloadStore`. `lib/state/loraDownloads.ts` turns the lane into what a chip shows and what holds Generate back; a failure wins over a download still on its way. A failed item stays in the store after the server stops listing it, until it is retried or its LoRA deleted, so a red chip does not turn ready by itself.
- **Trigger words.** `features/workspace/triggerMenu.ts` builds the Triggers menu and inserts a chosen word at the start of the text; nothing is written into a prompt without a choice.

## The mascot

Blob is one SVG, `atoms/Mascot.svelte`, with a `mood`, a `tool`, a `size` and an optional `progress` for the creating canvas. Every mood is CSS on the same parts, and a mood change blends from the pose on screen, so an idle mascot costs no script. Below 48px he is drawn in a simpler form, which is also what `public/ziv-icon.svg` shows.

- **Effects.** Mood effects are always mounted and shown by their mood's opacity rule, so the pose blend fades them in and out with everything else. Each has a still frame under `prefers-reduced-motion`.
- **Spots and transitions.** `MascotSpot` wraps him for a place on a page; spots that share a `travelKey` hand him over. A spot with no previous spot is sampled in; a spot that follows another receives him as a stream of pixels along an arc. Call `rememberMascotSpots()` before a change that swaps spots, so the stream starts where he was.
- **Canvas only in transitions.** The SVG is baked with each part's current pose and drawn on a canvas on the `--z-mascot-effect` layer, which is removed at the end. Where the browser cannot animate, and under reduced motion, he is simply there, with no transition.
