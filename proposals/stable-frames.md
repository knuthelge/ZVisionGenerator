# Stable frames: the frame first, pages remember, anchors do not move

**Status:** In progress (2026-10-24)

## Problem

On a phone, coming back to the Workspace shows the latest image first and then the settings strip and the composer pop in under it, and the image shrinks to make room. The same pattern, a frame built after its data arrives, runs through the other pages: the Config page bar grows by a row when Save appears, and the Gallery, Models and Config pages start from a blank page with a spinner on every visit and forget their scroll position. Page switching itself is instant and nothing else moves, so these pops are what makes the app feel like a web page rather than an installed one. They show on desktop and iPad too, only less, because there is more room for them.

## Evidence

Measured 2026-10-24 in headless Chrome at an iPhone-sized viewport against a local server, recording element positions every frame while switching pages.

- The app frame (the fixed root, the top strip, the scroll offset, the measured viewport height) never moves during a switch. The jumps are inside the pages.
- **Workspace, compact.** `WorkspacePage.svelte` renders `.compact-controls` only inside `{#if authorityReady && context}`, and the context is fetched on every mount (`getWorkspaceCoreContext`, about 200 ms locally, longer over Wi-Fi). The latest image comes from `historyStore`, module-level state that survives page switches, and from the browser cache, so the first frame is the stage at 800 px with the image; 230 ms later the strip (114 px) and composer (61 px) appear, the stage is 686 px and the image re-fits. The wide and tall compositions pass a null context into `ControlsSidebar` and only grey the model field, so compact is the one composition that builds its frame late.
- **Config.** `AdminPageShell.svelte` passes the page bar's actions only when not loading. On compact the actions are a full-width row, so the bar grows by 46 px after the request and the form drops by that much.
- **Gallery, Models, Config.** Their data is `$state` local to the page component (`assets`, `inventory`, `config`); each visit refetches and shows "Loading…" in place of the content. The Workspace (history, draft) and Prompts (the builder) keep theirs in stores and come back at once.
- **Gallery tiles on touch.** `AssetTile.svelte` shows all three overlay actions under `(hover: none)`, so every tile in a two-column phone grid carries three buttons over the image, while tap opens the viewer and press-and-hold the menu.
- **Back on Android.** Sheets and the drawer are not history entries (`escapeStack` only), so the back gesture leaves the page instead of closing the overlay.

## Design direction

Three rules, added to the Web UI Design System so every page follows them in every composition.

1. **The frame first.** A page draws its complete frame synchronously from its composition: bars, strips, panels, grid area, action rows. Requests fill the frame and never add or remove parts of it. A loading state lives inside the slot it is loading (a disabled chip, a greyed row, placeholder tiles) and never changes the height of anything around it.
2. **Pages remember.** Page data lives in stores the way history and the draft already do. Returning to a page shows the last data at once and refreshes in the background. Known events invalidate: a finished job refreshes the Gallery, a download the inventory, a save the Config. Scroll position comes back with the data.
3. **Anchors do not move.** What the thumb aims at stays put: the top strip, the composer, the settings strip, page bars and their actions. Content that changes size, such as an image and its caption, changes inside a box reserved from the start.

Page switches stay a hard cut, which is what a native tab switch does; nothing moves, so nothing needs animating.

## Stages

Each stage is one commit that rebuilds the SPA and passes `make check`.

1. **Workspace frame on the phone.** Render the settings strip and the composer before the context arrives, with chips drawn from the saved draft and disabled until ready, and reserve the caption row under the stage.
2. **Workspace context store.** Keep the loaded context across visits and revalidate in the background, so the chips come back with real values and the model field never says "Loading models…" on a return.
3. **Admin shell.** Always render the page bar's actions, disabled while loading.
4. **Gallery store and scroll restore.** Assets, filter, sort and loaded pages move to a store; a return shows the grid at once and refreshes page one in the background. A first-ever load shows placeholder tiles in the grid instead of a spinner.
5. **Models and Config stores.** The same pattern; the inventory refreshes in the background and after downloads, the config after a save.
6. **Back closes overlays.** Opening a sheet or the drawer pushes a history entry, so the Android back gesture and the browser's back button close it instead of leaving the page.
7. **Quieter tiles on touch.** On a coarse pointer a tile shows only the ⋯ action; expand and reuse live in the menu, on tap and on hold. Desktop hover stays as it is.
8. **Docs and changelog.** The three rules in `docs/development.md`, a changelog entry per stage.

### Tests

Vitest: the stores' stale-while-revalidate and invalidation contracts, the compact Workspace rendering its strip and composer with a null context, the admin shell rendering its actions while loading, the overlay history entries. The frame-by-frame layout measurement stays a manual check, since the repo has no browser-automation dependency.

## Out of scope

Offline caching and a service worker, page transitions, phone landscape tuning, the Models tables as cards on a phone, and caching the model inventory on the server. Each is separable and none is needed for the app to stop moving under a finger.
