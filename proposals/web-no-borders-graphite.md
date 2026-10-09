# No borders and Graphite with teal and coral

**Status:** Accepted (2026-10-09; not built yet; branch `worktree-web-design-system`)

Follows [One design system for the Web UI](web-design-system.md). Decided from the previews in `.agent-work/ui-explorations.html` of the worktree (not committed): study 1 (borders), 3 (components) and 4 (Graphite colours).

## Decisions

| Topic | Decision |
|---|---|
| Border style | **No borders.** Panels, cards, the prompt box, section bands, alerts, overlays, the sidebar edge and every control lose their outline. |
| Dividers | Kept: lines between rows, table rows, menu groups, panel heads and footers. |
| Outlines that stay | Only where they mean something: the focus ring, a field with an error, the selected tile, and the dashed edge of drop zones and add buttons. |
| Neutrals | **Graphite**: grey with no tint, replacing today's teal-tinted neutrals. |
| Primary colour | **Teal** `#2dd4bf` (hover `#5eead4`, text on it `#0b3b37`): main actions, selection, focus ring, progress. Unchanged. |
| Secondary colour | **Coral** `#fb8f7c` (Blob's beret): `$snippet` syntax in prompts and the mascot's second colour. |
| Choice syntax | `{a|b}` stays amber `#fbbf24`. |
| Info messages | Stay teal, not coral: a coral info message reads as an error (seen in the preview). |

## Rules for No borders

- **Buttons:** filled, no outline. Secondary buttons use `bg-surface-hover` with primary text; the main action stays teal; danger buttons use a red-tinted fill with red text; quiet buttons stay transparent; add buttons keep a dashed outline.
- **Pressed and selected states** (pressed buttons, chips, enhance options) use a teal-tinted fill instead of a teal outline.
- **Fields and selects:** filled with `bg-surface-hover`; inside panels and overlays, filled with `bg-base` so they still contrast. An error adds a 1px red inset ring.
- **Segmented controls and presets:** a recessed well. The track and unselected options sit on a background darker than their surroundings (overlay mixed with black), unselected text is muted, and the chosen option is a clearly lighter raised pill (text mixed 18% into surface) with primary text. The first preview without this read too flat.
- **Badges and chips:** tinted fills (status colour at about 14%), no outline.
- **Checkboxes:** filled box; checked is teal.
- **Panels, prompt box, queue items, tiles:** separated by shade only. The prompt box uses a fill tone between base and surface. The selected tile keeps a 2px teal ring.
- **Section header bands:** the raised shade, no border above or below.
- **Overlays** (menus, popovers, tooltips, toasts, dialogs): the overlay surface plus a faint 1px ring of text at about 6%, so they don't vanish over dark images.
- **Sidebar and top bar:** the sidebar is a shade between base and surface; the top bar is darker than base. Neither draws an edge line.

## Graphite tokens

| Token | Value |
|---|---|
| `bg-base` | `#111214` |
| `bg-surface` | `#18191c` |
| `bg-surface-hover` | `#232529` |
| `bg-raised` | `#1d1f22` |
| `bg-overlay` | `#0b0c0d` |
| Field / switch track (`zinc-700`) | `#393c42` |
| `border-subtle` (dividers) | `#26282c` |
| `border-strong` | `#33363b` |
| Fill (prompt box) | `#202226` |
| `text-primary` | `#ececee` |
| `text-secondary` | `#b4b6bc` |
| `text-muted` | `#8f939a` |
| Placeholder (`zinc-600`) | `#7a7e85` |
| Secondary accent (replaces `accent-blush` for snippets) | `#fb8f7c` |

The rest of the `zinc-*` scale should be re-derived as neutral greys so nothing keeps the teal tint.

## Contrast (measured)

| Colour | Against | Ratio |
|---|---|---|
| Teal `#2dd4bf` | its button text `#0b3b37` | 6.7:1 (AA) |
| Teal `#2dd4bf` | a Graphite panel `#18191c` | 9.4:1 (AAA) |
| Coral `#fb8f7c` | a Graphite panel | 7.8:1 (AAA) |
| Amber `#fbbf24` | a Graphite panel | 10.5:1 (AAA) |

## Alternatives considered

- **Borders:** today's outlines everywhere (busy, panels and fields look alike); containers without borders but outlined controls (a good middle ground, but still many lines).
- **Neutrals:** Teal (today), Midnight, Warm charcoal, Forest, Light.
- **Pairs on Graphite:** indigo + peach, sky + yellow, violet + mint (the common "AI purple"), white + cyan, lime + lilac, marigold + teal. Teal + coral won because it is Blob's own colours and keeps the app's identity.

## Implementation

One commit on the design-system branch, then screenshots at 1920 × 1100:

1. Replace the neutral and `zinc-*` values in `global.css` with the Graphite tokens; set the secondary accent to coral and point `$snippet` syntax at it.
2. Change the `ui-*` classes to the No borders rules above (buttons, fields, segmented, badges, chips, panels, overlays, alerts, tables, section bands, tiles, checkboxes), plus the scoped styles that still draw outlines (prompt box, InspectorSection, Toggle, Prompts set sections and entries, asset tiles and viewer, queue items, job card).
3. Update `docs/development.md` (Web UI Design System) and the CHANGELOG.
