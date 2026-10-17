# One Gallery: the Workspace strip is a view of it

**Status:** Proposed (2026-10-17)

## Problem

The same list has two names. The Workspace strip is titled **History**, has its own store (`historyStore`, `/api/history`) and ends with a button that says **Open Gallery**. The Gallery page lists the same files from `/api/gallery`. Toasts after a job say "Refreshed the gallery", the shortcut list says "Focus the history", and the viewer labels the set "History" on one page and "Gallery" on the other. A new user reads two features where there is one.

Two stores also mean two copies of the list that drift: the Workspace merges a job's outputs into its copy as they arrive (`historyStore.mergeOutputs`) and refreshes after the job; the Gallery page refetches on open; a favourite or a delete has to be applied to both.

## Evidence

- `/api/history` and `/api/gallery` in `web/server.py` are the same six lines apart from the name of the media filter parameter (`media_filter` and `filter`).
- `HistoryStrip.svelte`, `history.svelte.ts` and the `historyCollapsed` draft flag carry the name; `GalleryPage.svelte` has its own `assets`, paging and deletion state.
- The `PANES` map in `WorkspacePage.svelte` describes Alt+3 as the history pane; the shortcuts dialog says "Focus the history".

## Proposed change

### 1. One name

- The strip's heading becomes **Gallery**, its count stays, and the button becomes **Show all**. The viewer's set label is "Gallery" on both pages. Shortcut labels say "gallery strip". Toasts keep "gallery".
- Internal names (`historyCollapsed`, test ids) can follow or stay; the draft schema needs no bump if the key stays.

### 2. One store

- `historyStore` becomes `galleryStore` in `lib/state/`, used by the strip and the Gallery page: one list, one paging cursor, one `removeAssets`, one `mergeOutputs`, one favourite toggle when [Search and favourites](gallery-search-favourites.md) lands. The Gallery page's filters and search are a view on top (query params to the server, results kept in the same store keyed by the query), so the strip, which shows the unfiltered newest-first list, is just the empty query.
- `/api/history` is removed; the strip calls `/api/gallery`. Both are internal to the SPA and change together.

### 3. Ordering

Do this before or together with the search proposal, so search is built once, on one store.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "The Workspace's history strip is now called Gallery, as it shows the same list as the Gallery page."
- The Web UI section of `docs/getting-started.md` replaces "history" with "the gallery strip".

## Alternatives considered

- **Keep both names and only share the store.** Cheaper, but the naming is the user-facing problem. Rejected.
- **Rename the Gallery page to History.** "History" suggests jobs and time; the page is a library of outputs with filters. Gallery is the better word.

## Open questions

- Should the strip show the Gallery page's current filter when one is set, or always the full list? Always the full list: it is the "latest work" view beside the stage.

## Out of scope

- The Jobs drawer in [A log of past jobs](job-log.md), which is where "history" in the time sense belongs.
