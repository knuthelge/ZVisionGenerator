# Touch that feels native, on every screen

**Status:** Done (2026-10-20, v0.13.0)

## Shipped (2026-10-20, v0.13.0, tested on an iPhone)

All eight stages, one commit each. Checked in headless Chrome at a phone size with real touch input through CDP (sheet pulls from the head and the body, scrolled and not, spring-back, popover, drawer, dialogs as sheets, the composer and `--app-height`, no field under 16 px, no page errors) and at desktop size; not committed as a test, since it needs a Playwright dependency. Left out, and open:

- The viewer keeps its own gesture code; it already worked, and moving it onto `pan` would change nothing a user sees.
- No edge swipe to open the drawer: Safari's own back swipe owns the left edge.
- Scrubbing by the event's `pointerType` (a mouse scrubs on a touch screen) is not done; a coarse pointer still means steppers only.
- iOS Safari behaviour (the keyboard, the toolbar, scroll commit) can only be confirmed on a device.

### What the test on an iPhone added

Tested as a home-screen app on iOS 26. Sheets, the stage, the drawer, holds, reorder and the job bar worked; these needed fixing:

- **The bottom band.** With `apple-mobile-web-app-status-bar-style: black-translucent`, iOS 26 draws a home-screen app from the top of the screen but makes its viewport a status bar shorter (WebKit bug 301108), and nothing can paint the strip left at the bottom: sizing the app to `screen.height` only pushed the composer into it. The status bar is now `black` (opaque), which gives the app the whole screen under it and also drops the dark scrim iOS 26.1 draws over a translucent bar. iOS reads the tag when the app is added, so an installed app has to be removed and added again.
- **Opaque bar caveat.** Under an opaque bar, `vh`, `lvh`, `svh` and `h-screen` report the full screen (WebKit bug 316008); only `dvh`, `innerHeight` and `visualViewport` are right.
- **Page zoom.** A pinch or double tap zoomed the whole app: `touch-action: pan-x pan-y` on `html` plus a `gesturestart` guard stop it, the viewer keeps its own pinch, and `maximum-scale=1` stops iOS zooming into a field under 16px on focus.
- **The phone preview.** The image and its caption are one centred block with Reuse and ⋯ on the image; a pill over the stage replaces the history header row.
- **Prompt sets on a phone.** A prompt's buttons and a field's name and × share a row and the text runs the full width under them (a container query on the sets column), and Text or Fields moves into the prompt's menu.

### Considered, not done

- The remaining `vh` and `h-screen` heights (App, ActionMenu, PathInput, RepairView, EnhanceChip) could move to `dvh` because of bug 316008.
- A home-screen app can stay short after the keyboard first closes or after leaving the app's scope; a reported cure is to hide and show the app's root to force a re-measure when `innerHeight` is below the largest seen. Not seen in testing.

## Problem

[A phone layout](phone-layout.md) fixed the *composition*: on a phone the stage is the screen, settings are chips and the prompt is a composer. What it did not fix is the *interaction layer* under it. Sheets that should pull down sometimes don't, a tap leaves buttons looking pressed, a tap on a number zooms the page, a swipe on the stage either does nothing visible or jumps to the Gallery, and some actions only exist on hover. The layout looks like a phone app; the gestures still behave like a desktop page with touch events bolted on.

The same gaps reach past the phone: an iPad in the wide layout has the hover-only actions and the drag-and-drop that a finger cannot use, and a touchscreen laptop gets whichever half of the behaviour its media queries pick.

## Evidence

Read from the code on 2026-10-20.

**Sheets: five implementations, three behaviours.**

| Surface | File | Pull down | Handle | Open/close motion |
|---|---|---|---|---|
| Bottom sheet (All settings, Job) | `molecules/BottomSheet.svelte` | `sheetDrag` | 4 px bar | none on open; slides only when dragged shut |
| Action menu on compact | `molecules/ActionMenu.svelte` | `sheetDrag` | 4 px bar | same |
| Setting popover | `workspace/SettingPopover.svelte` | `sheetDrag` | 4 px bar | same |
| Every `Modal` on compact (prompt chooser, path browser, confirmations, new file) | `molecules/Modal.svelte` | **no** | **none** | none |
| Navigation drawer | `organisms/NavDrawer.svelte` | **no swipe** | — | none |

Each has its own scrim, z-index, Escape wiring and safe-area padding. They look alike and behave differently, which is exactly "slide down sometimes works".

**Why `sheetDrag` itself misses** (`lib/actions/sheetDrag.ts`):

- It waits 8 px before deciding, and does not `preventDefault` until then. Browsers commit to a native scroll or overscroll on the first uncancelled `touchmove`; after that every `touchmove` arrives with `cancelable = false` and `preventDefault` is ignored. The sheet then moves *and* the content, the page or Chrome's pull-to-refresh moves with it, or the browser sends `touchcancel` and the sheet springs back mid-pull.
- Nothing sets `touch-action` on the sheet or its handle, so the browser always believes it owns the gesture.
- `html` and `body` have no `overscroll-behavior`, so a pull that escapes the sheet reaches pull-to-refresh on Android and the rubber band on iOS.
- The flick test uses the average speed of the whole drag, so a slow start and a fast finish reads as slow.
- The drag target is the 4 × 36 px bar plus padding; the header beside it is not a handle unless the body is scrolled to the top.

**Stage swipe** (`WorkspacePage.svelte` `onStageTouchStart/End`): decided only on `touchend`, no `touchmove`, so nothing follows the finger and there is no feedback until the image changes. An upward swipe of 70 px navigates away to the Gallery with no preview of that either, so it happens by accident while scrolling or reaching for the strip. The stage sets no `touch-action`.

**The asset viewer is the one surface that already works** (`AssetViewer.svelte`): `touch-action: none`, pinch, double tap, swipe and pull-down to close with the media following the finger. It is the model for the rest, but its gesture code is private to it.

**Hold Generate** (`WorkspacePage.svelte` `onGeneratePointerDown`): no `user-select: none` or `-webkit-touch-callout: none`, so iOS starts a text selection on the label during the hold; no visual progress during the 480 ms; and the action it opens is invisible until discovered.

**Sticky hover.** About 60 `:hover` rules across `global.css` and the components, one guarded by `@media (hover: hover)` (`AssetTile`). On iOS and Android a tapped element keeps `:hover` until the next tap elsewhere, so buttons stay lit as if still selected. There is almost no `:active` feedback and no `-webkit-tap-highlight-color` rule.

**Hover-only controls** that a finger cannot reach on any width: Prompts entry actions (`.on-hover`, opacity 0), drag grips on entries, sets and snippets, `PromptValueEditor` field remove. On the phone the Prompts page is one column so some of this is hidden by layout; on an iPad in the wide layout it is simply unreachable.

**Drag and drop on Prompts** uses HTML5 `draggable` (`PromptEntryCard`, `PromptSetSection`, `SnippetsPanel`). It does not fire for touch on Android and is unreliable on iPadOS.

**Zoom on focus.** iOS zooms the page when an input with a font size under 16 px is focused, and does not zoom back. The prompt text is 16 px on a coarse pointer (`ComposePane.svelte`), but `InspectorNumber` is 12 px (`--text-ui`) and path fields are 13 px. A tap on a number in a setting popover zooms the whole app.

**The keyboard.** `interactive-widget=resizes-content` in `index.html` is honoured by Chrome only. On iOS Safari the layout viewport keeps its height, so the composer, which is at the foot of an `h-dvh` column, is pushed behind the keyboard or the page scrolls under the fixed top strip. Nothing reads `visualViewport`.

**Target sizes.** The coarse tokens are 36 px controls, 28 px small controls, 32 px row buttons and 32 px setting chips. Apple asks for 44 pt and Material for 48 dp. The small and row sizes are where mis-taps come from (reset buttons beside values, the ✕ on a LoRA).

**The page settles late on iOS.** After moving around near the bottom of the screen, a gap like padding sits under the composer until the first scroll, when the UI "falls into place" (reported 2026-10-20). The root is `h-dvh` and the document itself may scroll: when Safari's toolbar or the keyboard changes the visible area, `dvh` and the document's scroll position update only on the next scroll.

**The job bar runs off the screen.** The live preview (`JobPreview`) sets `height: 100%` and an `aspect-ratio`, so a landscape job on a portrait phone is wider than the stage. `.job-view` is a centred flex item with the default `min-width: auto`, so it grows to that width and both ends of the bar (`left/right: 12px` of the view) are cut off, ⋯ included. Its ⋯ also opens a small menu whose first entry is Job details, one tap more than needed.

**Swipe up to the Gallery** is a useful shortcut but fires only on lift with no feedback, and not at all while a job runs: the handler returns early when the stage cannot step through history.

**Remaining viewport units.** `PathBrowser` uses `100vh`, which on iOS includes the area under the address bar.

## Proposed change

Keep the three compositions from the phone layout; they are sound. Rebuild the interaction layer under them as a small set of shared primitives, and apply them everywhere, so a gesture means the same thing on every surface and every screen.

### Principles

1. **Input decides behaviour, size decides layout.** Gestures key off the *event's* `pointerType` (a touchscreen laptop gets touch behaviour on a touch and mouse behaviour on a mouse); target sizes key off `(any-pointer: coarse)`; layout keys off `layout.mode` as today.
2. **Everything follows the finger.** A drag that changes something shows it while the finger is down, and either commits with momentum or springs back. Nothing happens only on lift.
3. **Every gesture has a visible twin.** Swipe has arrows or a pill, pull-down has ✕, hold has a visible menu entry. Desktop keeps every keyboard shortcut and gains the gestures that have a mouse or trackpad equivalent.
4. **One primitive per pattern.** One sheet, one pan tracker, one press feedback. No component carries its own copy.

### 1. One `Sheet` primitive

A single `Sheet` organism replaces `BottomSheet`, the compact branch of `ActionMenu`, the compact branch of `Modal`, the sheet shell of `SettingPopover` and the Job and Queue sheets. On wide screens it renders as today's centred dialog or anchored popover; on compact it is a bottom sheet. Callers pick *what* (dialog, menu, inspector), not *how*.

- **Drag that cannot lose.** Pointer events with `setPointerCapture`. The grab bar *and the whole header* are a handle with `touch-action: none`, so a pull there always drags, from the first pixel. The body pulls the sheet only when its scroller is at the top; it decides on the first `pointermove`, and sets `touch-action: pan-y` only on the scroller so the browser never owns a downward pull from the top.
- **Momentum from the last 80 ms**, not the average: a flick closes, a slow pull past a quarter closes, anything else springs back with a short spring.
- **Sized to its content:** a sheet is as tall as what it holds and grows to the full height (under the top strip) only when the content needs it; no detents.
- **Motion both ways:** slides in on open and out on ✕, scrim tap and Escape, not only when dragged; off under `prefers-reduced-motion`.
- **Focus, Escape and scroll lock** in one place, using the existing `escapeStack` and the modal stack from `Modal`.
- **Keyboard aware:** the sheet's bottom inset follows `--keyboard-inset` (below).

The `NavDrawer` uses the same tracker on the horizontal axis: swipe left to close, and an edge swipe from the left of the Workspace opens it (off on the stage, which swipes for history).

### 2. A shared pan tracker

Lift the viewer's gesture handling into `lib/actions/pan.ts`: axis lock after the first move, live offset, velocity, rubber band at the ends, pinch passthrough. Pure decision functions (`axisFor`, `outcomeFor`) stay unit-testable as `sheetDragOutcome` is today.

Used by the `Sheet`, the drawer, the viewer, and:

- **The stage.** Swiping sideways moves the image with the finger and commits or springs back; the "2 of 212" pill shows while dragging. Swipe up to the Gallery stays (decided 2026-10-20) and becomes visible: the stage lifts with the finger and a Gallery pill rises from its foot, labelled "Release for the Gallery" past the threshold. It works in every stage state, a running job included.
- **The history strip and Gallery tiles** get press feedback and a long press that opens the asset menu (today only ⋯ does).

### 3. Press feedback and `hold`

Every tappable surface gets an `:active` pressed state under `(hover: none)` (iOS applies `:active` once the document listens for `touchstart`), and the browser's tap flash is off. A `hold` action adds `user-select: none` and `-webkit-touch-callout: none`, a visible ring that fills over the hold time, and a short `navigator.vibrate` where supported. Hold Generate uses it, and its "new seed" action also appears as a visible chevron beside Generate so it is discoverable.

### The job on a phone

- The live preview fits the stage by its aspect ratio (`min(100cqw, 100cqh × ratio)` in a size container) instead of growing past it, and `.job-view` may shrink (`min-width: 0`), so the bar stays on screen on every job shape. Wide screens get the same fit, without the crop `object-fit: cover` gave.
- ⋯ on the job bar opens the Job sheet directly, where Next, Repeat, the queue and Cancel already are. The small menu goes.

### 4. Touch hygiene, enforced

- **Hover only where hover exists:** every `:hover` rule moves under `@media (hover: hover)`, and `make check`'s design-system check fails on a bare `:hover`.
- **No hover-only controls:** anything revealed on hover is always visible under `(hover: none)` (the Prompts row actions, grips, field remove), as `AssetTile` already does.
- **Pointer-based reorder** replaces HTML5 drag and drop on the Prompts page: a drag handle using the pan tracker with a placeholder and auto-scroll. It works with a mouse, a finger and a pen; the keyboard keeps Move up and Move down. The desktop gets a better drag too (no ghost image, no drop flicker).
- **No zoom on focus:** on a coarse pointer every `input`, `textarea` and `select` is at least 16 px, from one rule in `global.css`.
- **The visible viewport:** a `viewport` store reads `visualViewport` and sets `--app-height` and `--app-top`, and the app root is fixed to them. The app is always exactly the visible area: when the keyboard opens it shrinks to the space above it (the composer stays in view on iOS, where `interactive-widget` is ignored), and when Safari's toolbar comes and goes it follows at once instead of on the next scroll. The document is pinned at the top, so it never scrolls under the app. Pinch zoom (scale ≠ 1) leaves the values alone.
- **Targets:** coarse tokens go to 40 px for controls, 32 px for small controls, 44 px for rows and bars, 36 px for row buttons and setting chips.
- **Overscroll:** `overscroll-behavior: none` on `html` and `body`; scrollers keep `contain`.
- **Units:** `dvh`/`svh` everywhere a height is tied to the screen (`PathBrowser`).

### 5. Desktop and tablet, in the same plan

Nothing in 1–4 is phone-only, and the wide layout keeps its density and every control in view. Explicitly:

- **Desktop with a mouse:** sheets stay centred dialogs and anchored popovers; scrubbing, shortcuts and hover arrows stay. New: the same open and close motion as the phone, pointer-based reorder on Prompts, and a two-finger horizontal trackpad swipe on the stage steps history (the desktop twin of the phone swipe, from `wheel` `deltaX`).
- **iPad, wide layout:** gets the touch hygiene (no sticky hover, visible row actions, finger reorder, no zoom on focus, 44 px targets) while keeping the desktop composition. Steppers stay; with a trackpad or mouse attached the event's `pointerType` brings back scrubbing on the label.
- **Touchscreen laptops:** the per-event rule means a mouse scrubs and a finger steps on the same screen.
- **Phone landscape:** compact today by the height rule; the viewport store makes it usable, still not tuned (out of scope).

## Stages

Each stage is one commit that rebuilds the SPA, passes `make check` and keeps every feature reachable on every composition. After each, a test on a real phone (iOS Safari and, if available, Android Chrome) before the next.

1. **Hygiene, no new components:** `overscroll-behavior`, 16 px inputs on a phone, hover rules under `(hover: hover)` with the check in `make check`, pressed states, hover-only controls visible on `(hover: none)`, `dvh` in `PathBrowser`, coarse target tokens.
2. **The visible viewport:** the `viewport` store, the fixed root, the keyboard.
3. **Pan tracker** (`lib/actions/pan.ts`) with pure decision functions.
4. **`Sheet` primitive** with handles, momentum and motion; `BottomSheet`, `ActionMenu`, `SettingPopover`, `Modal` on compact and the drawer move onto it, so every sheet pulls down and the drawer swipes.
5. **The job on a phone:** the preview fit, ⋯ opens the Job sheet.
6. **Stage follows the finger,** the visible Gallery pull in every state, trackpad swipe on desktop.
7. **`hold`** with progress on Generate and its chevron; long press on tiles and history.
8. **Pointer reorder** on Prompts for mouse, finger and pen.

### Tests

- Vitest: the pan tracker's axis and outcome functions (slow pull, flick, spring back, scrolled body), the viewport values from a fake `visualViewport`.
- Playwright (the open item from the phone layout, extended): an iPhone and a Pixel profile with `hasTouch`, dispatching real touch sequences through CDP: pull a sheet from the handle and from a scrolled and an unscrolled body, swipe the stage, focus a number without the page scale changing; and a desktop profile that checks the same flows with a mouse and keys. Gated on Chromium as the frontend tests are.
- A short manual checklist for iOS Safari, which Playwright's Chromium cannot reproduce (scroll commit, keyboard, zoom on focus).

### Docs and changelog

- CHANGELOG `[Unreleased]`, under Changed: sheets pull down everywhere and animate, the stage follows the finger, row actions are visible on touch, drag to reorder works with a finger.
- `docs/development.md` (Web UI Design System): the four principles, `Sheet`, `pan`, `hold`, the hover rule and its check.
- `docs/getting-started.md`: update "From a phone" and "From an iPad" for the gestures.

## Alternatives considered

- **Patch `sheetDrag` and `Modal` only.** Fixes the most visible bug in a day, but leaves five sheets that drift apart again and does nothing for hover, zoom, keyboard or reorder. Stage 1 plus a `sheetDrag` fix is the fallback if the full plan is too much.
- **A gesture library (vaul-style sheets, `@use-gesture`).** Mostly React-first or heavy for what is needed; the viewer already proves a small in-house tracker works, and the decisions stay testable as pure functions.
- **A separate mobile build.** Rejected for the same reason as in the phone layout: two UIs to keep in step.
- **Only sheets on desktop too** (the phone pattern everywhere). Rejected: desktop keeps anchored popovers and dialogs, which put controls next to what they change.

## Decided (2026-10-20)

- Swipe up to the Gallery stays, made visible and working while a job runs.
- Sheets take the height their content needs, up to the full screen; no detents.
- All eight stages are approved.

## Open questions

- Haptics: `navigator.vibrate` works on Android only; worth having for the hold?

## Out of scope

- New features on the phone; this is about how existing ones feel.
- Offline use and a service worker.
- Tuning phone landscape.
