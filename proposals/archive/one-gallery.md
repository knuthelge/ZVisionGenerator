# One Gallery, and a viewer that feels native on every device

**Status:** Done (2026-10-24, unreleased). All six stages shipped, one commit each; the gestures, the flight, the hold and right-click menus and the panel were driven in headless Chrome at phone, iPad and desktop sizes. On a real phone, the swipe-back and the pull-to-close deserve one pass.

## Problem

The same list had two names. The Workspace strip was titled **History**, had its own store and route (`historyStore`, `/api/history`) and ended with **Open Gallery**; the Gallery page listed the same files from `/api/gallery`. Two stores meant two copies that drifted: outputs merged into one, deletes had to be applied to both, and the Gallery page refetched from nothing on every visit.

The viewer is one component already, used by the Gallery, the strip and the stage, with swipe, pinch and a pull-down details sheet on the phone. But it is a full-screen dark layer on every device, it opens and closes with a cut, a swipe moves on release instead of under the finger, the details open from a button rather than a swipe up, and Back leaves the page instead of closing it. On the desktop a page behind a modal panel would read better than a layer that replaces it.

## Evidence (2026-10-24)

- `/api/history` and `/api/gallery` were the same lines apart from the filter parameter's name. The strip used `historyStore`; `GalleryPage.svelte` kept its own `assets`, paging and deletion state, and the page memory from [Stable frames](stable-frames.md) kept a third copy.
- `AssetViewer.svelte` (780 lines) handles swipe left and right, swipe down to close, pinch and double-tap zoom, a details sheet on compact and a details column on wide, a film strip, and keyboard shortcuts. It does not go through the escape stack, so Back does not close it. The swipe decides on `touchend` only; nothing follows the finger.
- `AssetTile.svelte` opens its menu on a hold through `actions/hold.ts`; the viewer's image has no hold.

## What the user sees

**Phone.** Tap a tile and the image grows out of the tile into the screen with the backdrop fading in behind it. Swipe sideways and the neighbouring images move with the finger, settling with the flick. Swipe up and the image rises as the Details sheet comes up under it to half height; pull it further for all of it, or down to put it away. Pull the image down and it shrinks back toward its tile and the viewer closes. Tap the image to hide the bar and the film strip; tap again to bring them back. A hold on the image opens the ⋯ menu at the finger, the same menu a hold on a tile opens. Phone landscape is the same with the sheet from the right.

**iPad and desktop.** Click a tile and a panel grows out of it: the image on the left, the details column on the right, the actions in the panel's top bar, the film strip under the image, a backdrop behind it that closes on click. Arrows, keys and Escape work as today, the details column can be hidden and the choice is remembered. A right-click on a tile or on the image opens the same menu as a hold does on a phone. On an iPad the panel also takes the swipe, pinch, pull-down and hold.

**One Gallery.** The strip is called Gallery with a Show all button. The strip, the stage, the Gallery page and the viewer read one store, so scrolling the viewer to its end loads more from one place, and a delete or a new output lands everywhere at once.

## Stages

Each is one commit that rebuilds the SPA and passes the gate.

1. **One store.** *Done.* `lib/state/gallery.svelte.ts` holds one list per query (filter and sort); the default query is the strip's and the stage's, the Gallery page commits its own query's pages into it, and `removeAssets` and `mergeOutputs` reach every list. The page memory keeps only the query and the scroll position. `/api/history` and `getHistory` are gone; the strip is `GalleryStrip` and says Gallery and Show all.
2. **Viewer on the overlay stack.** The viewer pushes an escape layer, so Escape closes it (the shortcut list first), and the pages stop guarding their keyboard handlers against it by hand.
3. **Open and close from the tile.** The viewer remembers the tile that opened it and animates the image between the tile's box and its own on open and close, on every layout. A pull-down on the phone drives the same path under the finger.
4. **Phone pager, details and hold.** Three panes that follow the finger sideways; the swipe-up details sheet at half and full height; tap to hide the chrome; a hold on the image opens the ⋯ menu, and a right-click on a tile or the image opens it on a desktop.
5. **Desktop and iPad panel.** The modal with the details column and backdrop replaces the full-screen layer on the wide and tall layouts.
6. **Docs and changelog.**

### Tests

Vitest: the store's views, removal and merge contracts; the viewer's escape layer; the pager's decisions (which pane settles, from offset and velocity) as pure helpers; the details sheet's snap points; the hold and context-menu wiring. The animation itself is checked by hand in a browser.

## Out of scope

Favourites and search stay in [their own proposal](../gallery-search-favourites.md) and sit on the same store. A Share action needs a secure context, which a plain http address on the network is not. The film strip stays as it is; it already works as a scrubber on both devices.
