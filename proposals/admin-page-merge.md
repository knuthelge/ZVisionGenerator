# Merge the Models and Config pages

**Status:** Proposed (2026-10-17)

## Problem

Models and Config are two top-level pages for setup work that happens rarely. They share one shell (`AdminPageShell`) and both show a "Folders and access" panel with overlapping facts (the models and LoRAs folders and the Hugging Face token on Models; `locationItems` on Config). The top bar carries five page tabs plus five workflow tabs; below 1000 px it wraps into two rows, then three. One tab less makes room, and the two pages are easier to find as sections of one.

## Evidence

- `TopNav.svelte`: `navItems` has workspace, prompts, gallery, models, config. `ModelsPage.svelte` and `ConfigPage.svelte` both render `Panel title="Folders and access"`.
- `router.svelte.ts` accepts the five page ids; `startup_view` in the config accepts `workspace`, `gallery` and `config`, but not `models`.
- The shortcuts `G → M` and `G → C` go to the two pages.

## Proposed change

- One page, **Setup** (`#/setup?tab=…`), with a left sub-navigation on wide screens and tabs on narrow ones:
  - **General**: today's Config fields (default models, default size, output folder, enhancer model, startup view).
  - **Models**: the image and video tables and Convert a checkpoint.
  - **LoRAs**: the LoRA table and the two import forms.
  - **Folders and access**: once, with every folder and the token status.
- The `models` and `config` routes redirect to the matching tab, so bookmarks and the Gallery's links keep working; `startup_view` accepts `setup` and maps the old values. `G → M` and `G → C` open the tabs; `G → S` opens the page.
- The Save and Discard actions of General stay in the page bar of that tab only, since the other tabs have their own per-form buttons.
- The top bar shows Workspace · Prompts · Gallery · Setup. On wide screens the workflow tabs then fit beside the brand without wrapping down to 1000 px; the breakpoint is re-measured when the work is done.

### Naming

The Workspace's inspector is already called **Settings**, so the merged page is not. "Setup" says what it is for (models, folders, defaults) without clashing. See open questions.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "The Models and Config pages are now tabs of one Setup page."
- `docs/web-ui/models.md` and `docs/web-ui/config.md` become one page that describes the Setup tabs.

## Alternatives considered

- **A gear menu holding both pages.** Still two pages, and a menu hides them further. Rejected.
- **Rename the Workspace pane to Controls and call the merged page Settings.** Possible, but renames a pane users know for a page they rarely visit.

## Open questions

- "Setup" or "Manage"? Either works; decide when the sub-navigation is mocked up with the design system.
- Should the first-run Welcome panel ([First run](archive/first-run.md)) link to the Models tab? Yes, once both exist.

## Out of scope

- Any change to what the Config fields are.
