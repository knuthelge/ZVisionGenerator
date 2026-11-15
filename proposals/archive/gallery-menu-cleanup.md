# Tidy the Gallery's and the viewer's buttons and menus

**Status:** Done (2026-11-15; built on `design/gallery-menus`)

Every action on a Gallery file sits in one of several places, each with its own order:

- a Gallery tile's corner buttons, its **⋯** menu, its right-click menu and its press-and-hold menu
- the Workspace preview and strip, which use the same tile
- the viewer's toolbar and the viewer's own menus
- Select mode's bars

This proposal gives them one order and one shared **file panel**. The panel opens where you act: under ⋯, at the pointer, or at the finger. In the viewer, a mouse gets a toolbar with the same items; touch gets the same items in a dock at the foot.

The mockups were kept outside the repo during design; every number below was measured on them.

## What is wrong today

Checked in the running UI on a desktop (1440 × 900, mouse), an iPad both ways (1180 × 820 and 820 × 1180, touch) and a phone both ways (390 × 844 and 844 × 390, touch), with images, videos and audio.

- **The menus are long single columns.** An image's menu is 15 actions, two headings and seven rules. On a desktop it is capped at 22 rem and scrolls, so a right-click on a tile shows neither Download nor Delete. On a phone the sheet fills the screen and Download and Delete need a scroll. Held sideways it shows four rows at a time.
- **The same action sits in a different place on every surface.** The toolbar runs Reuse, Use as, Edit, Upscale, Move, Star, Download, Delete, Details. The menu runs Select, Reuse, Use as, Edit, Upscale, Move, Star, Download, Delete. Copy the prompt is only in the viewer's menus and the C key.
- **Delete is not set apart in the viewer.** It sits between Download and Details.
- **Below 1100 px every toolbar label goes at once,** leaving 11 bare icons. Three of them mislead: Upscale uses the tile's "view full size" arrows, Reuse settings looks like refresh, Use as is a bare picture (also in [the backlog](../backlog.md)).
- **On a phone, Reuse, Edit and Download are two taps away** inside ⋯.
- **Rows do work a smaller shape could do.** Four Use as rows are four choices of one kind; four upscale rows are four sizes.

## The design

### One order

| Group | Contents | Where it goes |
|---|---|---|
| **The file** | Name, ★, how you look at it (View where it applies, Details, shortcuts) | The panel's head; the start of a bar |
| **Make** | **Reuse settings**, then the kind's own: Edit (image), Extend (video), Audio track (audio) | First, with Reuse as the one teal primary |
| **Use as**, **Upscale** | The targets and the sizes | Next, as choices |
| **File** | Download, Move, Copy prompt | The last row, or the right-hand group of a bar |
| **Delete** | Delete | After the file group, past a rule |
| **Close** | ✕ | Alone at the top right of every viewer bar and sheet, never beside Delete in a row of buttons |

### The file panel

One component with one grid, on every surface:

```
┌──────────────────────────────────────────────────┐
│ ▣ poster_2026-11-12_10-48-15.png           [★]   │  head: the file, then View (where it applies) and the star
│   Image · 1216 × 832                              │  as fixed icon buttons
│ [⟲ Reuse settings          ] [✎ Edit             ] │  Make: always a pair
│   Load its settings in Image   Paint over part of it │
│ USE AS ────────────────────────────────────────── │
│ [▣ Start image             ] [▣ Reference        ] │  every choice says what it makes
│   A new image from it          Guide a new image · as image 1
│ [▯ First frame             ] [▯ Last frame       ] │
│   Start a new clip on it       End a new clip on it │
│ UPSCALE ───────────────────────────────────────── │
│ [↗ 2×                      ] [↗ 4× · high memory ] │  the choice is the name,
│   2048 × 2048                  4096 × 4096         │  the size the hint
│ [↗ 2× creative             ] [↗ 4× creative      ] │
│ ──────────────────────────────────────────────── │
│   ⤓ Download    ▭ Move    ⧉ Copy prompt  │  🗑 Delete │  the same row everywhere
└──────────────────────────────────────────────────┘
```

- **One grid:** four columns with 6 px gaps. A choice spans two columns, a file cell one. Rows are at least 44 px with a mouse and 52 px with a finger, and grow together when a hint wraps, so both cells of a row are always the same height. Section labels line up with the choices' left edge.
- **Every choice has two lines,** its name and **what it makes**. That tells *Start image* ("A new image from it") from *First frame* ("Start a new clip on it"). Hints are short enough not to strand one word on a second line in a 127 px text column (a mouse): Extend is "Continue from the end", a paused Extend "Continue from here", Audio track "A new clip to its sound". Reference keeps "Guide a new image · as image 1". Hints use `text-wrap: balance`. Names, hints and greyed reasons each wrap to two lines rather than being cut (so *Extend from 3.2 s* keeps its time), and a finger reads them without a tooltip, as today's reasons do. A longer reason ("Ideogram 4 can't use a start image") still fits in two. Reuse's hint is just "Load its settings": the tab follows the file's kind, as today. The new hints replace the tab names; the ones the code already has stay ("as image 1", "Keep it in your voices").
- **Upscale choices are named by the choice** (*2×*, *4×*, *2× creative*, *4× creative*) with the output size as the hint. A warning goes on the name in amber: *· high memory* or *· very slow*, the two markers `assetMenu.ts` has. The first-use download, a tooltip today, goes in the hint ("1536 × 896 · first use downloads 3.4 GB"), so touch users see it too. A warned size is marked, never asked about, as today.
- **A section with a single choice spans its row,** named in full: a video's one upscale reads *Upscale 2×*. It is the only full-width choice. No other section has one item.
- **Make is always a pair:** Reuse settings with Edit (image), Extend (video) or **Audio track** (every audio take). Audio track is a take's way to make something new, so it moves up from Use as. **Use as** is then only for what a speech take can also be, Voice and Saved voice (`canUseAsVoice` allows speech only); music and sound have no Use as section.
- **The file row is the same everywhere:** Download, Move, Copy prompt, then Delete. Copy prompt is new on tiles (the viewer had it), and greyed with its reason for a file with no recorded text.
- **Head:** the thumbnail, then the name (one line, cut in the middle, see below) with the kind and size under it, then the controls:
  - **View**, in Select mode and on the Workspace preview and strip, where a click on the picture doesn't open the viewer. For a video in the preview, View is the way into the viewer, since the picture is a player.
  - There is **no Select** in the head. An empty box beside the star read as a second toggle, and touch has no tooltip to say otherwise. The tile's own checkbox already starts Select mode in one click or tap, and Space or X does it from the keyboard, so nothing gets further away. The old "⋯ → Select" route goes.
  - The **star**, as a fixed icon button. Only its glyph and colour change when toggled, never its box.
  - **✕**, on sheets only.

  The controls sit 4 px apart as one cluster, 8 px after the name. The name gets the room the Select button took.
- **File names follow one rule everywhere** (bars, panel and sheet heads, tile captions): one line, cut in the middle, with the full name as `title` and the accessible name. The app names files `{set}_{YYYY-MM-DD}_{HH-MM-SS}.ext` (`utils/filename.py`), so the tail kept after the cut is the time: `web_2026-11-14_09-12-33.png` becomes `web_2026-…_09-12-33.png`, and in a narrow tile caption `we…_09-12-33.png`. The tail is the last `_segment` and the extension; for a counted name (`…_09-12-33_2.png`) it is the counter. One shared file-name component does it: it measures with canvas `measureText` (after `document.fonts.ready`, under one shared `ResizeObserver` for every name on screen) and renders a single text node, *start…tail*. A CSS `text-overflow` split into two spans leaves a 1–6 px gap after the ellipsis, so it isn't used.
- **Per kind:** a video has Reuse with Extend, then Upscale 2×. A video paused in the Workspace preview reads *Extend from 3.2 s* and adds "Use frame at 3.2 s as" with First frame and Last frame, as its menu does today. In the viewer, Extend means from the end, as the E key does: the viewer passes no frame today and doesn't start to. A speech take has Reuse with Audio track, then Voice and Saved voice; a song or a sound has Reuse with Audio track and the file row.
- **In Select mode,** ⋯, right-click and press-and-hold still act on that one file, with **View** in the head. The selection's actions are in Select mode's own bar.

### Height: the panel never hides its file row

- Today's `.action-menu` cap (`min(60vh, 22rem)`) does not apply to the panel. It is as tall as its content (423 px for an image with a mouse, 463 px with a finger) and limited only by the room between its anchor and the window's edge. It opens above or below its anchor, wherever there's more room.
- If even that is too little, the body scrolls and the **file row stays pinned at the foot**. A measured case: a 1280 × 440 window gives the panel 424 px, the body scrolls by 36 px, and Download to Delete stay in view.
- On a phone the sheet is as tall as its content (479 px for an image, 359 px for a paused video) up to the screen's height below the status bar, with the same pinning. At 160% text a paused video's sheet still fits.
- **Held sideways** (`layout.short`), the asset sheet ignores `.ui-sheet`'s 520 px cap and spans the screen, padded by `max(12px, env(safe-area-inset-left/right))`; on an iPhone that leaves about 750 px. Sheet's own head row is hidden (the grab bar stays), and the panel draws its head inside its own grid, in the first row beside Make (`align-self: center`), so ✕ stays at the top right. One 8-column grid lines the rows up across both halves; the file row runs along the foot.

### One Delete

- **In a bar** (the viewer's toolbar, Select mode's header): `.ui-btn-danger` with the trash icon and the word Delete, as today, after a 1 px rule.
- **In a row of cells** (the panel, the touch dock, the touch Select bar): the icon over the word in red with **no fill**, tinted while hovered or pressed. A filled red cell would compete with the teal Reuse.
- It is always the last action and always labelled, and it still asks first.

### Tiles

- **Gallery card, mouse:** the corner keeps today's View, Reuse, ⋯, in Select mode too. Only the icons change.
- **Gallery card, touch:** ⋯ in the corner, as today. The caption always shows the star, an outline in full `--color-text-muted` when unstarred (5.36:1 on the caption), so starring is one tap. The caption's name is cut in the middle, so the star never hides the time.
- **Workspace preview** (`AssetTile`, density `stage`): the corner keeps View, Reuse, ★, ⋯. ⋯ opens the panel with View in its head. The star shows in both the corner and the panel's head. Both are the same one-click control: the corner stands in for the caption the preview doesn't have, and the head keeps the panel the same everywhere.
- **Workspace strip** (`GalleryStrip.svelte`, density `compact`): the corner keeps View, Reuse, ⋯ with a mouse and ⋯ on touch. ⋯ opens the panel upwards, with View in its head.

### The viewer with a mouse: the toolbar

```
★ web_2026-11-11_09-12-33.png  ⓘ Details  ?  │ [⟲ Reuse settings] [✎ Edit] [➤ Use as ⌄] [▣ Upscale ⌄] …… [⤓ Download] [▭ Move ⌄] [⧉ Copy prompt] │ [🗑 Delete] │ [✕]
  18 of 19 · Gallery
```

- **The file and how you look at it lead:** the **star first**, at a fixed place at the start of the bar, then the name 4 px after it, then Details and the shortcuts, in that order and style on every device. The star leads so it always sits beside the name and never reads as part of Details. The name block has a set width (so Make starts at a fixed place), and a long name is cut in the middle, keeping the time.
- **Make starts at a fixed place** after the name block and closes up: a kind without an action has no button and no hole. For an image it is Reuse settings, Edit, Use as ▾, Upscale ▾; for a video Reuse settings, Extend, **Upscale 2×**; for a speech take Reuse settings, Audio track, Use as ▾; for a song or a sound Reuse settings, Audio track.
- **One upscale choice means a direct button.** Whenever a file offers exactly one upscale (`offeredUpscaleFactors`, not the file's kind), the bar and dock show *Upscale 2×* with no arrow, which starts it at once. Icon-only, its tooltip is "Upscale 2× (X then 2)". With several choices, it is *Upscale ▾*.
- **File and Delete are anchored to the right,** so Download, Move, Copy prompt and Delete never move.
- **✕ stands alone at the top right,** after a rule. Delete is labelled and red and asks first, so it can't be mistaken for Close.
- **Measured at 1440 px,** Reuse, Download and Delete sit at the same x (413, 955 and 1284 px) for an image, a video and an audio take. Only the Make buttons after Reuse differ, as they must.
- **Copy prompt is visible.** Use as ▾ and Upscale ▾ open the panel's own choices in a popover.
- **Right-click or hold on the picture** opens the whole panel at that point.

**Narrow windows**: Details and the name give way first, then whole groups. Delete always keeps its word, and nothing folds away, so every action stays one click.

| Window | What changes |
|---|---|
| 1330 px and wider | Every label; the name block 240 px (1366 px laptops) |
| 1210–1329 | Details as an icon, the name block 180 px; every action keeps its label (1280 px laptops) |
| 1000–1209 | The File group (Download, Move, Copy prompt) as icons; Move keeps its arrow |
| 855–999 | The Make group after Reuse (Edit, Extend or Audio track; Use as; Upscale) as icons; arrows stay |
| 780–854 | Reuse settings reads *Reuse*; the name block 160 px |
| under 780 | Two rows: the name, Details, ? and ✕ (still at the top right), then the actions. Today's bar wraps below 768 px |

Today every label shows down to 1100 px, then all go at once. Keeping them all to 1100 px isn't possible now that Copy prompt is in the bar, but the common 1280 and 1366 px laptops keep every action's label.

### The viewer on touch: the dock

Touch gets a top bar and a dock, chosen by the pointer, so turning an iPad moves nothing. Which dock is chosen by the same rule that decides where ⋯ menus become sheets today: **the phone dock where `layout.compact` holds** (under 640 px wide or under 500 px tall), **the full dock on every other touch screen**. The top bar has the star, the name (set width, cut in the middle), Details (and ? on a tablet), and ✕ alone at the right.

- **The full dock**: the toolbar's item list at finger size, built like the bar.
  - Make sits flush left and closes up; Download, Move ▾, Copy prompt │ Delete are anchored right; the spare room goes between the groups. So Reuse, Download, Move, Copy prompt and Delete stay put for every kind, and there are no holes mid-row.
  - **Cell sizing, no formula:** every cell is `flex: 0 1 96px; min-width: 70px`, and the spacer between the groups is `flex: 1 0 12px`. The row's fixed parts are 69 px (16 px padding, nine 4 px gaps, the 12 px spacer, a 5 px rule), so at 640 px the eight cells come to 71 px, enough for "Copy prompt", and from about 840 px they reach 96 px. Under 768 px the viewer drops its 24 px inset, so the full dock fits from 640 px, the narrowest screen that isn't compact (an iPad mini upright, Split View, a small Android tablet). Today every touch screen 640 px or wider shows the full toolbar, so nothing becomes two taps.
  - There is no More, so nothing appears twice. Use as ▾ and Upscale ▾ open their choices above their cell.
- **Phone**: four fixed slots, never empty: Reuse, the kind's own Make action (Edit, Extend or Audio track), Download, More. More (drawn with filled dots, so it reads at dock size) and press-and-hold open the panel as a sheet. Use as and Upscale are not in the phone's dock: they'd still cost two taps.
- **Phone held sideways**:
  - The dock becomes a rail on the right with the same four slots. It pads by `env(safe-area-inset-right)`, as the existing bars pad their edges.
  - The top bar lies over the picture on a dark scrim (88% to transparent), so the name reads over a bright image.
  - The film strip stays under the picture, padded by the left and bottom safe-area insets, so jumping to a file is still one tap.

**Gestures:**
- A tap on the picture hides and shows the dock or rail together with the bar and the strip (`chromeHidden`).
- Pulling the picture down fades it with them (`--pull`).
- While the details are a sheet over the picture (a phone, or a screen under 768 px wide), the dock hides as the strip does today.
- Pinch, double-tap and a zoomed pan happen on the picture only; the dock stays and never starts a gesture.

### Select mode

Chosen by the pointer, like the dock:

- **Mouse:** the header keeps today's single group at the right: Done, the count, Select all │ Download, Move ▾ │ Delete, with rules instead of a fixed gap.
- **Touch** (tablet either way and phone, one layout): **the count moves into the header beside Done**, as on a mouse, replacing "Tap files to add or remove". The bar at the foot holds only actions, built like the dock: **Select all**, the spacer, Download, Move ▾, a rule, Delete. Cells are `flex: 0 1 96px; min-width: 66px` and the spacer `flex: 1 0 12px`, so the bar fits from 320 px whatever the count. Every action is a cell with its icon over its label. Select all keeps today's busy state (its label hides under a spinner while the matching files load).

### Icons

One path changes and seven are new in `Icon.svelte` (exact paths under "How it maps onto the code"); the ones that carry a meaning:

- `reuse`: a replay arrow round a play mark, for "run it again". It no longer collides with `sliders` (Sort) or reads as refresh.
- `use-as`: a paper plane, for "send it to a slot". The arrow-into-a-frame read as sign-in.
- `upscale`: a small square anchored in the bottom-left corner of a larger one (the picture grown from its corner), with no arrow. A centred inner square read as a stop or record button beside the player; the earlier arrow read as "open in a new window". Used for plain and creative alike. `sparkle` keeps its meaning, "enhanced prompt".
- `selectall`: two stacked boxes with a tick, for Select all on touch.
- `folder-move`, `image-plus`, `bookmark` and `more-dots` (Move, Reference, Saved voice, the phone dock's More) round out the set.

- **One mark for "opens a choice":** the `chevdown` icon, 2 px after its label, on every button and cell that opens a list (Use as, Upscale with several choices, Move), in the bar, the docks and the touch Select bar. In a cell it hangs from the label (absolutely positioned), so the label stays centred under its icon.


This is the [backlog item](../backlog.md) about the icons. It belongs here because the new bars put View and Upscale side by side.

## Visual spec

Every colour, radius and size is an existing token in `frontend/src/app/global.css`, plus one new size, `--row`, and the spacing scale below. Sizes in brackets are for a coarse pointer, which already raises `--spacing-control` to 40 px.

**Spacing scale.** Six steps from values `global.css` already uses (the `.ui-btn` gap is 6, the action bar's 8, the page bar's padding 12 and 16). Each step does one job, and gaps doing the same job are equal everywhere, measured on every screenshot:

| Step | Job |
|---|---|
| 2 px | A chevron after its label |
| 4 px | Controls in one cluster (the head's View, ★ and ✕; a tile's corner buttons; the name and its star); an icon over its label; cells in a row of cells (dock, rail, file row, touch Select bar) |
| 6 px | An icon beside its label; buttons in a bar; the panel grid's gaps |
| 8 px | Inside a popover and a cell (choice padding, the choice's icon to its text, the section label's indent so it lines up with the choices' icons); a head to its first row; the thumbnail to the name |
| 12 px | Between groups: a 1 px rule with 6 px on each side in a bar, 4 + 2 on each side in a row of cells; the least spare room between Make and File; a bar's ends; a sheet's sides |
| 16 px | A screen edge (page padding, a sheet's foot above the safe area) |

Alignment: the panel's head, grid and file row share one left and right edge, the panel's padding (8 px). Section labels start 8 px in, on the same line as the choices' icons. On a tile, the checkbox is centred on the same line as the corner buttons (12 px in with a mouse, 10 px with a finger).

| Part | Spec |
|---|---|
| Grid | 4 columns `minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.2fr) minmax(0, 0.8fr)` (a choice spans two, so both halves stay equal; the file row gives Copy prompt room and centres Delete's rule in its gap), 6 px gaps, rows `auto` with a 44 px [52] minimum (`--row`); 8 px from the head |
| Head | 40 px thumbnail (`--radius-sm`), 8 px to the name; the name 13 px/600 `--color-text-primary` on one line, cut in the middle; meta 11 px `--color-text-muted`; View (where it applies), ★ and ✕ (sheets) as a 4 px cluster of `.ui-btn-icon`, View and ★ with a transparent fill. In a sheet the head row has 12 px sides (the grid's edge, not `.ui-sheet-head`'s 16/12), its controls a 4 px cluster (not `.ui-sheet-end`'s 6), and **no bottom border**: the grid starts 8 px below, and the panel's own rules do the separating |
| Section label | `.ui-area-label` with a 1 px `--color-border-subtle` rule filling the row |
| Choice (new `.ui-choice`) | padding 6 × 8; icon to text 8; `--radius-sm`; `--color-bg-surface-hover`; 18 px icon `--color-text-secondary`; name 12 px/600/1.3 and hint 11 px/1.3 `--color-text-muted`, each at most 2 lines; warnings in `--color-star` on the name |
| Reuse | `--color-primary-main` fill; `--color-primary-ink` name, hint and icon; name in `--font-heading` 800 |
| File cell (new `.ui-tool`) | 16 px icon (20 in docks) over an 11 px/600 `--color-text-secondary` label; Delete in `--color-error`, no fill, `color-mix(--color-error 14%)` on hover or press, after a 1 px `--color-border-strong` rule |
| Popover | content height, 360 px [400] wide, padding 10, `--color-bg-overlay`, `--radius-md`, the `.ui-overlay` ring and `0 16px 40px var(--color-scrim)` |
| Sheet | the existing `Sheet` with a `head` snippet (above); upright, its usual width (full on a phone, 520 px cap from 640 px); with `layout.short`, the full screen width with `max(12px, safe-area)` sides, Sheet's head row hidden and the panel's head in the grid's first row |
| Toolbar | 52 px; padding 8 × 12; the star first, then the name 4 px after it, in a block of 240 px, then 180 and 160 by step (star included); the name cut in the middle; 6 px between buttons; 1 px `--color-border-strong` rules (6 px each side) after the view controls, before Delete and before ✕ |
| Dock | `--color-bg-surface`, 72 px; padding 6 × 8 (bottom 8 + the safe-area inset); 4 px gaps. Full dock: cells `flex: 0 1 96px; min-width: 70px`, 56 px tall, icon 4 px over its label; Make flush left, File and Delete flush right; spacer `flex: 1 0 12px`. Phone dock: four cells `flex: 1 1 0`. Reuse filled teal |
| Rail | 76 px wide plus `env(safe-area-inset-right)`; cells 60 px tall; the phone's four slots. The sideways film strip pads by the left and bottom insets |
| Sideways scrim | `linear-gradient(rgb(4 4 5 / .88), rgb(4 4 5 / .6) 55%, transparent)` under the top bar |

### States

| State | Choice and cell | Buttons |
|---|---|---|
| Hover (mouse only) | `--color-bg-control-hover`; the hint lifts to `--color-text-secondary` (muted would be 4.40:1 there) | as today |
| Pressed (touch) | `--color-bg-control-hover` while held; Delete takes its red tint | as today |
| Focus | inset 2 px `--color-primary-main` ring | the app's focus ring |
| Open | — | Use as ▾, Upscale ▾, Move ▾, Details and More take the `aria-expanded` teal tint, in the bar and the dock alike |
| Star | glyph filled in `--color-star` when on; the box never changes | the same |
| Disabled | transparent, inset 1 px `--color-border-subtle`, name and icon `--color-text-muted`, the reason as the hint (2 lines); focusable with `aria-disabled`; a tap or key shows the reason as a toast, as today | as today |
| Busy | *Deleting…*, *Extracting the frame…* as today; Select all on touch hides its label under a spinner while the matching files load, as the button does today | as today |

### Contrast (WCAG 2.2, from the tokens)

| Pair | Ratio |
|---|---|
| Name `#ececee` on choice `#232529` | 13.0:1 |
| Hint `#8f939a` on choice `#232529` | 4.98:1 |
| Hint on hover, `#b4b6bc` on `#2c2e33` | 6.7:1 |
| Disabled name `#8f939a` on `#0b0c0d` / `#18191c` | 6.35:1 / 5.70:1 |
| Ink `#0b3b37` on teal `#2dd4bf` | 6.66:1 |
| Delete `#f87171` on `#0b0c0d` / `#18191c` (no fill) | 7.08:1 / 6.35:1 |
| Delete on its pressed tint over `#18191c` | 5.19:1 |
| Outline star `#8f939a` on a caption `#1d1f22` (touch tiles) | 5.36:1 |
| Warning `#fbbf24` on `#232529` | 9.19:1 |
| Focus ring teal on choice | 8.25:1 |

A choice's fill against the panel is 1.3:1, which is fine for controls named by their text (WCAG 1.4.11 asks for 3:1 only where the shape alone identifies the control).

## Behaviour and accessibility

- **Keys in the panel:** it opens with focus on Reuse settings for a mouse or keyboard, and on the panel itself for a finger, as today. Up and Down follow reading order item by item, so nothing is skipped. Left and Right move within a row. Home and End go to the first and last item. Enter and Space act. Escape closes and returns focus to ⋯ or the tile. Tab closes. A greyed item stays reachable.
- **Roles:** the panel is `role="menu"`, named after the file. Each section is a `group` named by its heading, items are `menuitem`, and the head's star is a `menuitemcheckbox`. Names are complete: "Use as start image, make a new image from it"; "Upscale 4×, 4096 by 4096, high memory".
- **Bars, dock, rail:** the Tab order is the visual order. Every icon-only control keeps an `aria-label` and a tooltip with its key.
- **Targets:** 28 px controls with a mouse as today. With a finger, 40 px or more; choices 52 px, dock cells 56 px.
- **Anchors:**
  - A tile's press-and-hold opens the popover under its ⋯, as ⋯ does (`AssetTile.svelte` sets no point for a hold).
  - A right-click opens it at the pointer.
  - In the viewer, a hold or right-click opens it at that point.
  - It opens above or below its anchor, wherever there's more room, and aligns to the anchor's edge that keeps it on screen.
- **Keys:** the viewer's R, M, X 2/4, S, D, C, Delete, I and ?, and the Gallery's, are unchanged. **E** changes for audio only: today it opens Use as for every take; now it starts **Audio track** directly, the action it pairs with Reuse, as E is Extend for a video. On a speech take, Voice and Saved voice stay one click away in Use as ▾, and no new key is added. The shortcuts help text for E becomes "Use as (image), extend (video), audio track (audio)".
- **Reduced motion:** the app's global rule already removes transitions and animations, so popovers, sheets and the chrome appear and hide without moving.

## How it maps onto the code

Build order: the shared pieces first (names, icons, CSS), then the panel, then the bars and docks.

**Shared**
- **`FileName.svelte`** (new atom): one line, cut in the middle. Props: `name`. It renders a single text node, *start…tail*, where the tail is the last `_segment` plus the extension. It measures with a canvas `measureText` in the element's computed font, after `document.fonts.ready`, and re-fits from one shared `ResizeObserver` (a module-level observer that all instances register with). The full name is the `title` and `aria-label`. Used in the viewer bars, panel heads and tile captions (`AssetTile` foot).
- **`Icon.svelte`:** one path changes and seven are new, all on the existing 24 × 24 grid with a 2 px round stroke:

  | Name | Status | Path |
  |---|---|---|
  | `reuse` | changed (a replay arrow round a play mark) | `M4.6 15a8 8 0 1 0 .9-7.4M4 4v4h4M10.5 9v6l4.8-3z` |
  | `use-as` | new, replaces `reference` on Use as (a paper plane) | `M21 3L10 14M21 3l-6.5 18-4-8-8-4z` |
  | `upscale` | new, replaces `expand` on Upscale (a small square in the bottom-left corner of a larger one) | `M5 4h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5a1 1 0 011-1zM4 13h7v7` |
  | `folder-move` | new (Move: a folder with an arrow) | `M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V7zM9 13h7M13.5 10.5L16 13l-2.5 2.5` |
  | `image-plus` | new (the Reference choice) | `M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14M13 4H6a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-7M18 3v6M15 6h6` |
  | `bookmark` | new (the Saved voice choice) | `M6 4h12v17l-6-4-6 4z` |
  | `selectall` | new (Select all on touch: two stacked boxes with a tick) | `M8 8h12v12H8zM4 16V4h12M11 14l2 2 4-4` |
  | `more-dots` | new (the phone dock's More, drawn `solid` so the dots read at dock size) | `M5 10.5a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM12 10.5a1.5 1.5 0 110 3 1.5 1.5 0 010-3zM19 10.5a1.5 1.5 0 110 3 1.5 1.5 0 010-3z` |

  Everything else the panel and bars use already exists: `image` is today's `reference` path (Start image), `frame-first`, `frame-last`, `edit`, `extend`, `waveform`, `speech`, `download`, `copy`, `trash`, `expand` (View), `star`, `info`, `close`, `chevdown`.
- **`global.css`, `@layer components`:**
  - The spacing scale as custom properties (`--sp-2` to `--sp-16`) and `--row` (44 px, 52 px with a coarse pointer).
  - `.ui-choice` and `.ui-tool`, as in the visual spec.
  - `.ui-tool-row`: the dock and touch Select bar layout, with cells `flex: 0 1 96px; min-width: 70px` (66 px in the Select bar), a `flex: 1 0 12px` spacer and 4 px gaps.
  - The hanging chevron: `.ui-tool .chev { position: absolute; left: 100%; margin-left: 2px }`.
- **`lib/state`:** `copyableText` and `copyPrompt` move out of `AssetViewer.svelte` (today around lines 188 and 599) into a shared helper, so tiles get Copy prompt too.

**The panel**
- **`assetMenu.ts`:** `assetMenuEntries` becomes `assetPanel(asset, handlers, …)`. It returns `{ head: { view: boolean }, make: [reuse, sibling], sections: [{ id, label, items }], file: [download, move, copy, delete] }` instead of a flat list.
  - The sibling is Edit (image), Extend (video) or Audio track (every audio take; moved out of `audioUseEntries`).
  - The Use as section holds the image targets (`referenceEntries`) or Voice and Saved voice (speech only; `canUseAsVoice`).
  - The Upscale section comes from `upscaleEntries`, with the choice as the name ("2×", "4× creative"), `warningMarker` on the name, and the size as the hint plus the `plainTitle` download line. A single choice is marked `span: true` and named "Upscale 2×".
  - The frame rows keep `frameEntries` and `extendEntry` (Workspace preview only).
  - The hints become the outcome texts above.
- **`FilePanel.svelte`** (new molecule): draws the head and the grid from `assetPanel()`.
  - The grid is `minmax(0,1fr) minmax(0,1fr) minmax(0,1.2fr) minmax(0,.8fr)`; a choice spans two columns, a file cell one, and a `span` choice the whole row.
  - The head is the thumbnail, `FileName`, the meta, View (when `head.view`), ★ (a `menuitemcheckbox`) and, in a sheet, ✕.
  - When the body is taller than the room it's given, the body scrolls and the file row is a pinned foot.
  - With `layout.short` it lays out 8 columns (the head in row one beside Make, the file row along the foot).
- **`ActionMenu.svelte`:** an optional `content` snippet replaces the item list. The snippet receives `select(entry, event)` and `explain(entry, event)`, so greyed items keep their toasts. Everything else stays: the `popover` action, `Sheet` on a phone, the Escape layer and focus return.
  - For the asset panel, the `.action-menu` `max-height: min(60vh, 22rem)` is lifted. The popover gets the room between its anchor and the window edge (it already flips sides), and `FilePanel` pins its foot inside that.
  - Keys: Left and Right move within a row; Up and Down move in reading order over every item; Home and End go to the first and last; the star is a `menuitemcheckbox`.
- **`Sheet.svelte`:**
  - A `head` snippet replaces the title and meta, keeping Close at the right. The asset sheet's head row has 12 px sides, a 4 px cluster and no bottom border.
  - A `wide` mode (used with `layout.short`) lifts the 520 px cap and pads by `max(12px, env(safe-area-inset-left/right))`. In that mode the head row is hidden (the grab bar stays) and the content draws its own head.

**Bars, docks and tiles**
- **`AssetViewer.svelte`:**
  - **Top bar:** ★ first, then `FileName` in a fixed-width block, then Details and ?. Then Make: Reuse settings, then Edit, Extend or Audio track, then Use as ▾ (images and speech), then Upscale ▾ or a direct Upscale 2× when `offeredUpscaleFactors` has one entry. A flexible spacer follows, then Download, Move ▾ and Copy prompt, a rule, Delete, a rule and ✕.
  - **Label steps:** a `@container` rule on the bar's inline size drops labels by step (the window thresholds above minus the 48 px inset): 1282, 1162, 952, 807 and 732 px of bar.
  - **Docks:** touch (`layout.coarse`) replaces the action buttons with the dock: the four-slot phone dock where `layout.compact` holds, the rail where `layout.coarse && layout.short`, the full dock elsewhere. The top bar keeps ★, the name, Details (? on a tablet) and ✕. Under 768 px wide, the viewer drops its 24 px inset.
  - **Dock visibility:** the dock hides and fades with `chromeHidden` and `--pull`, and hides while the details are a sheet, as the film strip does.
  - **Keys:** E starts Audio track for audio (`viewerShortcuts.ts`; the help text becomes "Use as (image), extend (video), audio track (audio)").
- **`AssetTile.svelte`:**
  - The corner sets are unchanged; only the icons change.
  - On touch, the caption star always shows, an outline in `--color-text-muted` when unstarred.
  - The checkbox is centred on the corner buttons' line (12 px in; 10 px with a coarse pointer).
  - The panel head gets View in Select mode and on preview (`stage`) and strip (`compact`) tiles. The menu's Select lead goes.
- **`GalleryPage.svelte`:** the Select bars are chosen by `layout.coarse` (today `layout.compact`).
  - **Header with a mouse:** Done, the count, Select all │ Download, Move ▾ │ Delete.
  - **Touch:** the header holds Done and the count. The foot bar is a `.ui-tool-row` with Select all (keeping its `aria-busy` spinner), the spacer, Download, Move ▾, a rule and Delete.
- **`GalleryStrip.svelte`:** nothing changes beyond `AssetTile`.

**Tests** check behaviour:
- The panel lists the same actions as today's menu for each kind, plus Copy prompt, minus Select. Audio track is offered for every take, and Use as only for images and speech.
- A single upscale is a direct action.
- Greyed items keep their reasons.
- Up and Down follow reading order; Left and Right stay in a row.
- Delete still asks first.
- E starts Audio track on audio.
- The dock and the select bar follow `layout.coarse` and `layout.compact`.
- `FileName` keeps the tail and puts the full name in `title`.

### Docs and changelog

- `CHANGELOG.md` `[Unreleased]`: "The Gallery and the viewer group their actions the same way everywhere: one compact panel for ⋯, right-click and press-and-hold that says what each choice makes, labels in the viewer's toolbar that drop one step at a time, and on touch the same actions in a dock at the foot."
- Update `docs/web-ui/gallery.md` (the menu, toolbar and Select mode, and stars on touch tiles, which now show unstarred) `docs/web-ui/devices.md` (the docks, the phone held sideways) and `docs/web-ui/shortcuts.md` (E starts Audio track for audio), with new screenshots.

## Review feedback and how it was handled

### Round 6 (final)

| Point | Handling |
|---|---|
| The full dock overflowed ("width over eight" ignored 69 px of fixed parts) | **Done.** No formula: cells `flex: 0 1 96px; min-width: 70px`, spacer `flex: 1 0 12px`. 71 px cells at 640 px; checked for overflow on every frame. |
| The touch Select bar pushed Delete off narrow phones | **Done.** The count moved into the header beside Done; the foot holds the actions with the same flex cells (66 px minimum), fitting from 320 px (1,234 selected on a 360 px phone fits). |
| The sideways sheet didn't fit the real `Sheet` | **Done.** With `layout.short` the asset sheet spans the screen with `max(12px, safe-area)` sides (drawn with an iPhone's 47 px insets, 750 px of grid), hides Sheet's head row, keeps the grab bar, and draws its head in the grid's first row. |
| The viewer star floated away from the name | **Done.** The star leads the bar at a fixed place, the name 4 px after it, on every device and frame. |
| File names: placeholder format; a gap after "…" | **Done.** Every frame uses the app's `{set}_{YYYY-MM-DD}_{HH-MM-SS}.ext` names, so the kept tail is the time. One `FileName` component measures with canvas and renders a single *start…tail* text node, with no CSS gap. |
| Polish: one-word wraps in hints; the file row's rule; the touch star's contrast; the upscale icon; the sheet head; hanging chevrons; "20 px gap" | **Done**, all seven: new hints with `text-wrap: balance`; a 1 / 1 / 1.2 / 0.8 grid with Delete's rule centred in its gap; the star at full muted (5.36:1); the inner square anchored bottom-left; the sheet head at 12 px sides, a 4 px cluster, no border; chevrons hang from their label; the phrase removed. |
| The code map should be complete enough to build from | **Done.** "How it maps onto the code" lists each file, its props or structure, the CSS rules and the measured thresholds. |


### Round 5

| Point | Handling |
|---|---|
| The user: tighten the spacing (gaps between icons, the star and the checkbox) | **Done.** A six-step spacing scale (2, 4, 6, 8, 12, 16 px), each step one job, applied to every surface; measured on all 33 screenshots, gaps doing the same job are equal (the only other values are in the 160% text shot, scaled with the text). The head's controls are a 4 px cluster; the star sits 4 px after the name; the tile checkbox is centred on the corner buttons' line; section labels line up with the choices' icons; rules have 6 px on each side. |
| Viewer bar labels on 1280 and 1366 px laptops | **Done.** Details becomes an icon and the name block narrows to 180 px first, and the gap before ✕ is the rule alone. Every action keeps its label down to 1210 px, then File, then Make give way. Re-measured and redrawn. |
| One rule for file names | **Done.** One line, cut in the middle by CSS (a shrinking start, a fixed tail with the last `_segment` and the extension), everywhere, with the full name in `title` and the accessible name. |
| The head's empty-box Select reads as a second toggle | **Done.** Select is gone from the head (the tile checkbox and Space/X start Select mode in one step); View stays where it applies. The "⋯ → Select" route goes. |
| The touch Select bar floated Delete mid-screen | **Done.** Built like the dock: count and Select all left, Download, Move ▾, rule, Delete right; one layout for tablet and phone. |
| One upscale choice: "Upscale 2×" or "Upscale ▾"? | **Done.** A direct *Upscale 2×* (no arrow) whenever one choice is offered, decided by `offeredUpscaleFactors`; tooltip "Upscale 2× (X then 2)" when icon-only. |
| Mockups: Select-mode panel cut off; header's fixed gap; "▾" typed in docks; the upscale icon | **Done.** The panel opens upwards with "Image · 1024 × 1024"; the header uses rules; `chevdown` everywhere, including Move in the touch Select bar; the upscale icon is a small square inside a larger one. |
| E on a speech take | **Done**, as decided: E starts Audio track; the help text is updated; no new key. |


### Round 4

| Point | Handling |
|---|---|
| Audio track missing from the viewer; Use as ▾ empty for music and sound; E would open nothing | **Done.** Audio track is the second Make action for every take in the bar, both docks and the rail. Use as ▾ appears for images and speech only. E starts Audio track for audio. |
| Which touch screens get the 8-slot dock; 84 px slots overflow small tablets | **Done.** The phone dock where `layout.compact` holds (where ⋯ already becomes a sheet), the full dock elsewhere. Cells are 72–96 px, and under 768 px the viewer drops its inset, so the full dock fits from 640 px. |
| Select all on touch shrank to 12 px text | **Done.** A full cell before Download, keeping today's busy spinner. |
| The bar closes up a missing action, the dock left a hole, the text matched neither | **Done.** The dock is built like the bar: Make flush left and closing up, File and Delete anchored right, the spare room between. The text says so. |
| The label steps left mixed groups; end-cut names hide the counter | **Done.** Labels drop by whole groups in four measured steps (1390, 1160, 940, 860 px), Delete always keeps its word, and names are cut in the middle. (Re-stepped in round 5.) |
| The rail has no safe-area padding | **Done.** The rail pads the right inset; the sideways strip pads the left and bottom. |
| The viewer star moved with the name's length | **Done.** The name block has a set width on every device, with the star pinned at its right edge. |
| The head's Select icon read as "selected" | **Done.** It used the tile's empty box. (Round 5 removed it.) |
| Mockups: ✕ mid-row under 810 px; the preview's corner star; the strip panel covering its tile; the faked large-text scroll; cut choice names; "Load its settings in Image"; nothing selected on screen in tablet Select mode | **Done**, all redrawn. The 160% text sheet now grows and fits. A real overflow was measured in a 1280 × 440 window, and choice names wrap to two lines. |


### Round 3 (Direction A chosen)

| Point | Handling |
|---|---|
| The user's decisions: outline star on touch, the high-memory upscale marked not confirmed, ✕ alone at the top right | **Done**, all three. |
| The tablet dock was a different toolbar, and its More stacked a copy of it | **Done.** The tablet dock is the toolbar's item list at finger size, Use as ▾ and Upscale ▾ included, Delete at the end, no More. The phone keeps its short dock and More. |
| Folding File into More ▾ in a narrow mouse window breaks rule 1 | **Done.** Nothing folds. Labels drop in six measured steps, and under 810 px the bar wraps to two rows (today's wraps under 768). Each step was measured at its narrowest window. (Regrouped in round 4.) |
| The panel is taller than `.action-menu`'s cap; larger text or a paused video overflow the sheet | **Done.** The asset panel isn't capped; it uses the room around its anchor. When that isn't enough the body scrolls and the file row stays pinned. |
| Hints and reasons truncate; Start image and First frame are confusable | **Done.** Hints wrap to two lines and rows grow together. Hints now say what each choice makes. |
| Workspace preview: View in the head; star twice; the strip not designed | **Done:** View is in the head for the preview and the strip, and the strip is designed. **Kept:** the star in both the preview's corner and the head, with the reason given under Tiles. |
| Position drift between kinds | **Done.** File and Delete are anchored to the right of the bar, Make to a fixed start (measured identical for all three kinds). The docks use fixed slots with empty ones kept. (Round 4: the dock now closes up like the bar instead.) |
| Panel shape per kind: audio's banner, orphan half rows, repeated "Image tab" | **Done.** Make is always a pair (audio's is Reuse and Audio track); Voice and Saved voice fill a row; a lone choice spans its row as one named action; the tab names are gone from the hints. |
| A crowded head; the star changes box | **Done.** View and the star are fixed icon buttons. |
| Details and ? differ between desktop and touch | **Done.** Both sit right after the name, in the same quiet style, everywhere. The phone has no ?, since it rarely has a keyboard. |
| Delete filled in touch rows competes with Reuse | **Done.** No fill in rows of cells, tinted when pressed; `.ui-btn-danger` stays in bars. |
| Phone held sideways: no film strip, no scrim, misaligned sheet rows | **Done.** The strip stays; the bar has a scrim; the sheet is one 8-column grid. |
| Icons: reuse clashes with `sliders`; `sparkle` means enhanced; use-as reads as sign-in; `more` too faint | **Done.** Replay for Reuse, a paper plane for Use as, `upscale` for creative too, filled dots for More in the dock. |
| Tablet Select mode used header buttons | **Done.** Select bars follow the pointer: the foot bar on every touch screen. |
| How the dock and rail behave with tap-to-hide, the details and zoom | **Done**, under Gestures. |
| Video upscale as a direct action; "very slow"; the first-use download | **Done.** *Upscale 2×* spans its row; both markers go on the name; the download size goes in the hint. |
| Tablet popover anchoring and "at the finger" | **Done.** A tile's hold anchors under its ⋯, as the code does. The text and mockup now say so, and the popover aligns to its anchor. |
| Missing code details | **Done:** ActionMenu's snippet with `select` and `explain` and the new keys, Sheet's head snippet, and the shared `copyableText` and `copyPrompt`. |

### Round 2

| Point | Handling |
|---|---|
| The file row wasn't stable; Star sat in it and in the title | **Done:** the star is in the head; the file row is the same everywhere. |
| The upscale matrix was the weakest control | **Done:** upscale uses the same choices as Use as. |
| The panel had no single grid | **Done:** one 4-column grid, one row height, left-aligned. |
| Delete was drawn five ways | **Done:** two forms (a bar button, a row cell), same colour and label, always last after a rule. |
| Phone dock slots changed with type; Use as and Upscale in the dock saved nothing | **Done:** fixed slots, Download for every kind, Use as and Upscale left the phone's dock. |
| The phone held sideways wasn't designed | **Done.** |
| Select mode dropped Reuse from the tile corner | **Done:** the corner is unchanged. |
| The Workspace preview was missed | **Done.** |
| Extend had two meanings in the viewer | **Done:** from the end, as the E key. |
| Rotating an iPad moved every action | **Done:** the dock follows the pointer. |
| Too many new styles; "chip" collided with `.ui-chip` | **Done:** no joined groups, no status chip; the component is `.ui-choice`. |
| Down in a 2-D grid skipped the right column | **Done:** Up and Down follow reading order. |
| The held tile's reduced-motion sink | **Out of scope:** it predates this and belongs in the backlog. |

## Decisions for you

None are blocking. The open questions of the last round are settled as you chose.

## Alternatives considered

- **Keep the list menu and only reorder it:** it still scrolls on a desktop and needs about 600 px on a phone.
- **Submenus (Use as ▸, Upscale ▸):** three clicks where today is two, and poor under a finger.
- **A panel docked beside or under the picture, with no menus** (round 2's direction B): dropped by choice in favour of this one.
- **Folding the toolbar's File group into More ▾ in narrow windows:** breaks rule 1; replaced by the label steps.
- **Starting Close at the left of the toolbar:** it kept Delete away from Close, but every reviewer preferred ✕ in the top-right corner, where sheets already put it.

## Out of scope

- The project list behind Move, the paused video's time-chip menu and the Details content.
- A confirmation for the high-memory upscale.
- [Copy a CLI command](../asset-cli-command.md), if built, joins Copy prompt in the file row.
