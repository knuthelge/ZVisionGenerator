# Search and favourites in the Gallery

**Status:** In progress (2026-10-21)

**Shipped:** the asset index (§1) is in the Gallery as a persistent index in the app database `ziv.sqlite3`, replacing the in-memory index first proposed here. Search, filters and favourites remain.

## Problem

The Gallery can narrow the list in two ways: images or videos (`filter`) and newest or oldest (`sort_order`). That is all `filter_and_sort_assets` in `web/gallery.py` accepts. There is no search by prompt, model, workflow, seed or LoRA, no date range, and no way to mark an output worth keeping. Finding "the portrait from last week with the film LoRA" means scrolling through pages of tiles and reading the hover text.

The Workspace history strip has the same limits, so the problem shows on both pages.

## Evidence

- Every asset the server returns already carries `prompt`, `scene`, `model`, `workflow`, `seed`, `lora`, `width`, `height`, `ratio`, `size`, `created_at` and the recorded `details` (`gallery_asset_to_json`). Searching needs no new metadata.
- Before the persistent gallery index shipped, `list_gallery_assets` walked the output folder with `rglob` and read the embedded config of **every** file (PNG `tEXt` chunk or MP4 tag) on **every** request. `/api/gallery` and `/api/history` both called it once per page. Nothing was cached, so a search that fired per keystroke would have re-read every file each time.
- `GalleryPage.svelte` has the two selects in its `PageHeader`, arrow-key grid navigation (`gridNav.ts`) and multi-select delete. The router keeps `#/gallery?…` params, so filters can live in the URL.
- There is no favourite, rating or tag anywhere in the server or the SPA.

## Proposed change

### 1. An asset index on the server

**Shipped.** `web/gallery_index.py` keeps the facts read from each gallery file in the app database `ziv.sqlite3` (table `gallery_assets_v{N}`), keyed by `(root, rel_path)` and validated by `(mtime_ns, ctime_ns, size)`:

- `list_gallery_assets` walks the output folder with `stat` only and reads embedded metadata only for new or changed files. Unchanged files come from the index, including after a restart. A deleted file is never shown, and its row is dropped by the next error-free listing that finds a file.
- Rows hold the raw facts (embedded config and image size), not the derived asset fields, so per-request fields stay live and search can build on the same rows.
- Search filters the full list that `list_gallery_assets` returns, so it needs no further index. If a folder grows large enough for that to be slow, filtering can move into SQL over the same table.

### 2. Search and filters in the API

`GET /api/gallery` (and `/api/history`, or the one endpoint [One Gallery](one-gallery.md) leaves) gains:

| Param | Meaning |
|---|---|
| `q` | Free text. Case-insensitive; every word must match somewhere in prompt, scene, negative prompt, model label, LoRA names, workflow label or filename. A purely numeric word also matches the seed exactly. |
| `model` | Model label, exact |
| `workflow` | One of the `Workflow` values |
| `since` | ISO date or `today`, `7d`, `30d` |
| `favourites` | `1` to list only starred assets |

Matching runs over the full indexed list before paging, so `total_count` and `has_more` stay right. The response also returns the distinct `models` and `workflows` present in the whole folder, so the filter menus list only values that exist.

### 3. Favourites

- A star on every tile (card and compact density) and in the viewer. Keyboard: `F` toggles it on the focused tile and in the viewer.
- Stored in a hidden JSON file in the output folder, `.ziv-gallery.json`, as `{"favourites": ["sub/file.png", …]}` with paths relative to the folder, written with `utils/atomic_write.py`. The file travels with the folder, needs no rewrite of the media files (which would change their modification time and so their order), and is ignored by the scan because `_asset_kind` only accepts media extensions.
- `PUT /api/gallery/{asset_path}/favourite` with `{"favourite": true|false}`. Deleting an asset removes its entry.
- Each asset JSON gains `favourite: boolean`, so the Workspace strip, the viewer and the Gallery agree without a second request.
- Favourites could instead become a component of the app database (a `favourites_*` table next to the gallery index); the sidecar file above is kept because it travels with the folder.

### 4. Gallery page

- A search field in the page bar, debounced (`INPUT_DEBOUNCE_MS`), with `/` focusing it, as the path browser already does. Esc clears it.
- Filter chips next to the existing selects: **Favourites**, **Model ▾**, **Workflow ▾**, **Date ▾** (Today, 7 days, 30 days, All). The media and sort selects stay.
- The whole state lives in the URL (`#/gallery?q=…&model=…&favourites=1`), so Back works and a search can be bookmarked. The page description line says what is shown: "12 of 840 assets match “film noir” · favourites".
- Empty state: "No assets match" with a **Clear search** button, reusing the existing filtered empty state.

### 5. Workspace strip

The strip shows the star on each tile and keeps the full list; no search there. The strip's "Open Gallery" button passes nothing, so the Gallery opens with its last filters.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "Search the Gallery by prompt, model, workflow, seed or LoRA, filter by date, and star favourites."
- The Web UI section of `docs/getting-started.md` gets a short Gallery paragraph (search, filters, the `F` key, and where favourites are stored).

## Alternatives considered

- **Search on the client over loaded pages.** No server change, but it would only search the pages already scrolled into view, which is exactly the case where search is needed. Rejected.
- **An in-memory index.** Removes the per-request reads, but a restart rebuilds it, and the Web UI is restarted often while a cold scan of a large video folder takes minutes. The models also need the RAM. Rejected; the shipped index is persistent.
- **Tags instead of a star.** More expressive, but a star covers the first need (find the keepers) and the same sidecar file can hold tags later.
- **Storing the star inside the media file.** Rewriting a PNG or MP4 changes its modification time and reorders the Gallery, and MP4 rewrites copy the whole file (`embed_mp4_config`). Rejected.

## Open questions

- Should `since` be presets only, or also a custom range? Presets first; a range picker is easy to add if asked for.
- Should the recorded sampling values (steps, guidance) be searchable? Not in the first version; the seed is the one number people remember.

## Out of scope

- Comparing assets side by side.
- Tags, ratings and collections.
- Favourites in the CLI.
