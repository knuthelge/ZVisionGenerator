# A phone layout and an installable app

**Status:** Proposed (2026-10-17)

## Problem

The Web UI is often used from another device: `ziv ui --host local` serves it on the network, and the in-app path browser exists so file fields work "from a phone or another computer". But on a phone the layout is the desktop layout squeezed:

- Below 640 px the Workspace stacks the whole sidebar above the preview (`.workspace-layout` flex column, preview `min-height: 420px`), so Generate sits somewhere in the middle of a long page and the output is below the fold.
- The top bar wraps into two rows plus a third for the workflow tabs below 1000 px.
- The Settings inspector relies on dragging labels to scrub values; on a touch screen a drag scrolls the page.
- The model select, LoRA chips and quantize select share one toolbar row that wraps.
- There is no web app manifest, so the UI cannot be added to a home screen as an app, and it opens with the browser chrome. An `apple-touch-icon` exists, nothing else.

## Evidence

- `WorkspacePage.svelte` has one `@media (max-width: 639px)` rule; `TopNav.svelte` one at 1000 px; `ControlsSidebar.svelte` hides its collapsed strip under 640 px. There is no `pointer: coarse` rule anywhere.
- `InspectorNumber` and `InspectorRow` implement scrubbing by pointer drag on the label with keyboard nudging as the alternative.
- `index.html` has the viewport meta and icons only.
- Chromium and Playwright are available in CI, so a phone viewport can be tested.

## Proposed change

### 1. Installable

- `frontend/public/manifest.webmanifest`: name, short name `ziv`, the existing SVG and PNG icons, `display: standalone`, `start_url: /app`, `background_color` and `theme_color` from the Graphite tokens; a `theme-color` meta; `viewport-fit=cover` so the bars can use the safe-area insets.
- No service worker: the app is useless offline and a cache would serve stale builds.

### 2. Phone Workspace (under 640 px, or any width with a coarse pointer and under 900 px)

- **Bottom tab bar** replaces the top bar: Workspace, Gallery, Prompts, More (Models, Config, shortcuts). The brand and the Review badge move to a slim top strip with the workflow as a dropdown instead of tabs.
- **Stage first.** The preview takes the top ~45 % of the screen: the latest output, the running job card, or the review. The history strip stays under it as a horizontal row, as it is.
- **Prompt next**, full width, with the Enhance and JSON tools where they are.
- **Sticky Generate bar** above the tab bar: Generate (or Add to queue with the count), the model name, and a **Settings** button that opens the Settings inspector as a bottom sheet (a sheet variant of `Modal`, sized to content, swipe or Close to dismiss). The LoRA chips and quantize move into the sheet's top.
- **Touch-friendly numbers.** With a coarse pointer, `InspectorNumber` shows − and + steppers beside the field, since scrubbing is not available; a long press repeats. The scrub hint in the Settings header is hidden there.
- Toasts dock above the Generate bar; the safe-area inset is respected.

### 3. Gallery and viewer on a phone

- The grid is already two columns. The viewer gains swipe left and right between assets, and the action menu opens as a bottom sheet with large rows. The details panel becomes a sheet too.

### 4. Prompts on a phone

- The three columns become one at a time with a segmented switch (Snippets · Sets · Preview) at the top. Drag and drop keeps working within the shown column; moving between columns uses the existing menus. This is the lowest-priority part and can ship after the rest.

### 5. Tests

- Vitest keeps the logic; a small Playwright run at an iPhone-sized viewport checks that the Generate bar, tab bar and sheet render and that the stage is visible on load. It runs in `make check` only when Chromium is present, as the frontend tests already gate on their tools.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "A phone layout for the Web UI, and the app can be added to the home screen."
- The Web UI section of `docs/getting-started.md` gets a "From a phone" paragraph: `--host local`, Add to Home Screen, and the Settings sheet.

## Alternatives considered

- **Responsive tweaks only** (smaller paddings, fewer columns). They keep the sidebar-first order, which is the actual problem. Rejected.
- **A separate mobile build.** Two UIs to keep in step. Rejected; the phone layout reuses every component and only changes composition.
- **Hiding Prompts on phones.** Simpler, but the page still works for small edits; the single-column switch costs little.

## Open questions

- Trigger by width, by `pointer: coarse`, or both? Proposed: width under 640 px always; between 640 and 900 px only with a coarse pointer (small tablets).
- Does https matter here? An installed app on a LAN address is still plain http; it works, but notifications stay unavailable ([Job notifications](job-notifications.md)).

## Out of scope

- Offline use and caching.
- Uploading a reference image from the camera beyond what the existing file input already allows.
