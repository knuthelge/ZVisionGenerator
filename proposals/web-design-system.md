# One design system for the Web UI

**Status:** Done (2026-10-09, unreleased; branch `worktree-web-design-system`)

## Problem

The Web UI is styled by four overlapping systems that share colour tokens but nothing else:

| System | Where | Look |
|---|---|---|
| Admin forms | Config, Models | `Button`/`Input`/`Select`/`FormField` atoms, 38 px controls, uppercase labels above fields, `.admin-section` cards with 18 px corners |
| Workspace inspector | Workspace sidebar and toolbar | 26 px pill tools, `InspectorRow` rows with the label in a left column, collapsible `InspectorSection`s, a 42 px Generate button |
| Prompts panel | Prompts builder | `.panel-button` (26 px, 6 px corners), joined `.panel-segment`s, heavy small-caps `.panel-label`s |
| Surface utilities | Gallery, asset viewer | `.surface-*` classes on native elements, including native `<select>`s |

The same element therefore looks different from page to page: page headers, dropdowns, buttons, toggles, field labels, card headers, chevrons, page backgrounds and spinners each come in two to five versions. About 60 raw Tailwind palette classes (`text-red-400`, `bg-teal-500/10`) bypass the semantic tokens. `--font-sans` and `--font-mono` name Inter and JetBrains Mono, which are never loaded. Several shared atoms (`NavItem`, `RangeSlider`, `Separator`, `Spinner`, `Textarea`) are unused.

The review also found two bugs that are fixed independently of the migration:

- The Add LoRA popover in the Workspace toolbar does not close on Escape or an outside click, and stays open under other popovers. It does not use the shared `use:popover` action (`WorkspacePage.svelte`).
- Card dividers and table header rules on the Models page use `border-zinc-900`, the same colour as the card, so they are invisible (`ModelsPage.svelte`).

## Decision

Standardise on one compact system, called **F** during design: the Prompts panel's square controls and density, inside the Workspace inspector's panels, prompt box, rows and collapsible sections, with the rules below applied everywhere. A static comparison page of the candidate systems (B, C, E, F) was made during design (not committed).

## Rules

### Size

| Element | Size |
|---|---|
| Controls (buttons, fields, selects, segmented controls, presets) | 28 px |
| Small controls (inline actions, chips) | 22 px |
| Rows (label and value) | 30 px |
| Buttons in an action bar (the sidebar footer) | 36 px, all of them |
| Badges | 18 px |

Buttons that sit together share one height; the context sets the size, not the button. The main action stands out by its teal fill, by taking the remaining width and by a bold Nunito label, never by being taller than its neighbours.

### Type

| Size | Use |
|---|---|
| 11 px | Small-caps area names, badges, helper text, meta |
| 12 px | UI: buttons, fields, rows, menus, tabs |
| 13 px | Content: prompt text, dialog and progress text, the main action label |
| 16 px | Page titles |

- **Nunito** (`--font-heading`) for page and dialog titles, area names and the main action only. Everything else uses the system font.
- **Monospace** only for numbers, paths, sizes and keyboard shortcuts.
- **One small-caps style** (Nunito 800, 11 px, 0.06em tracking, muted) for names of areas: panel titles, section headers, table headers, menu groups. Field labels and row labels are sentence case at 12 px.
- `--font-sans` and `--font-mono` drop Inter and JetBrains Mono, which are not loaded.

### Shape and surfaces

- Corners: 4 px for chips, badges and inline values; 6 px for controls; 10 px for containers (panels, prompt box, dialogs, menus, tiles). Only the switch is a pill.
- Surfaces: `bg-base` for the page and controls, `bg-surface` for panels, a **raised** surface for collapsible section headers, a darker **overlay** surface for every menu, popover, tooltip, toast and dialog so it stands out from any page, and `bg-surface-hover` for hover.
- No drop shadows. Overlays separate from the page by the overlay surface and a stronger border; dialogs add a dark backdrop. The only box shadows left are focus and selection rings.
- Reference pages (Config, Models) centre their content at 1200px, lined up with the page bar; their settings are label-left rows and their tables use content-size text.
- Collapsible section headers are a band with a border above and below; adjacent headers share one line.

### Colour and state

- Teal means on, selected or the main action. Each section (a bar, panel, form, dialog or popover) has at most one main action, shown with a teal fill and a bold Nunito label; a page can have several sections.
- Pink (`--color-accent-blush`) and amber are reserved for prompt syntax (`$snippet`, `{a|b}`). LoRA chips are neutral: the name in normal text, the weight in muted monospace.
- Red means error or destructive. Destructive buttons (Delete, Stop, a dialog's confirm action) are always red text; delete icons in rows are neutral until hover.
- Disabled: 40 % opacity and a not-allowed cursor. A disabled primary button turns neutral instead of fading.
- Focus: one ring everywhere, a 2 px teal outline with a 2 px offset. The prompt box shows it on the box, not the textarea.
- Changed settings show a dot after the label and a reset button at the end of the row.
- Motion: 120 ms colour transitions; no hover lift on buttons or tiles.

## Components

| Component | Today | After |
|---|---|---|
| Button | `Button` atom, `.panel-button`, `.surface-button-*`, `.prompt-tool`, hand-styled buttons | `Button` atom with `primary`, `secondary`, `quiet`, `danger` and `add` variants, `sm` size and an icon-only form; `ActionBar` sets 36 px for its buttons |
| Segmented control, presets | `.surface-toggle-pill`, `.panel-segment`, Workspace preset buttons | One joined `Segmented` component |
| Tabs | Prompt / Enhanced tabs in Compose | Underline tabs |
| Switch, checkbox | `Toggle` atom (36 × 20), Prompts `.switch` (26 × 15), native checkboxes | `Toggle` at 26 × 15; native checkbox with the accent colour |
| Chips | `.surface-chip`, `.prompt-tool` chips, LoRA chips | The `ui-chip` class, pressed through `aria-pressed`; LoRA chips are neutral with a weight field |
| Text field, select, path field | `Input`, `Select`, `FormField`, `PathField`, native `<select>`s | The same atoms at 28 px with sentence-case labels; no native selects |
| Rows, collapsible sections | `InspectorRow`, `InspectorSection`, Prompts preview rows | `InspectorRow`, `InspectorSection` restyled; used on Config too |
| Panel | `.admin-section`, `.surface-card`, Prompts set sections | One `Panel` with a small-caps header |
| Table | Models tables | The Prompts-panel table style, full width |
| Badge, alert, toast | `Badge`, hand-made alerts and error boxes, `Toast` | `Badge`, an `Alert` component, `Toast` on the raised surface |
| Menu, tooltip, dialog | `ActionMenu`, LoRA popover, `Tooltip`, `Modal` | Restyled; the LoRA popover uses `ActionMenu` and `use:popover` |
| Page header | `AdminPageShell`, Gallery's own header, none on Prompts and Workspace | One page header bar on every page |
| Loading, empty state | Five inline spinners, `.surface-empty-state`, Prompts page note | `Spinner` atom; one `EmptyState` |
| Asset tile | `AssetTile` | Restyled (10 px corners, no lift) |

## Migration

One branch, one commit per step. Each commit rebuilds `zvisiongenerator/web/static/app/` with `make frontend-build`, updates `CHANGELOG.md` where users see a change, and passes `make check`.

1. **Fixes:** the Add LoRA popover and the invisible Models dividers.
2. **Tokens:** the raised surface, semantic tints, sizes, radii and type sizes in `@theme`; the font stacks fixed; a `make check` step that reports raw palette classes and hex colours in `.svelte` files (warning only for now).
3. **Atoms and molecules:** rebuild the shared components to the rules above; delete unused ones; tests for the variant contracts.
4. **Prompts page** on the shared components.
5. **Workspace:** sidebar rows, prompt box, toolbar and action bar.
6. **Config and Models:** panels, rows and full-width tables.
7. **Gallery and asset viewer:** page header, selects, buttons, tiles.
8. **Clean-up:** delete the leftover `.admin-*`, `.panel-*` and unused `.surface-*` classes; the palette check fails instead of warning.

## Done when

- Every page uses the shared components; no `.svelte` file has raw palette classes or hex colours, and `make check` enforces it.
- Each component type exists once.
- Before and after screenshots of every page show the same control heights, labels, panels and overlays.

## Alternatives considered

- **Admin forms (A) everywhere.** Clear forms and large targets, but too big for the sidebar and the prompt builder.
- **Workspace inspector (B) as is.** Best panels and sections, but mixes pill buttons, monospace values and caps headings.
- **Prompts panel (C) as is.** The most uniform, but small targets, a weak main action and no form or validation pattern.
- **Surface utilities (D).** A set of classes rather than a system; native selects and no row or section pattern.
- **E (C's controls in B's panels) without the rules.** Kept 26 px buttons next to 28 px fields, three small-caps styles, eight type sizes and drop shadows.
