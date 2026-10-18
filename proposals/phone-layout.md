# A phone layout and an installable app

**Status:** In progress (2026-10-18)

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

## Decided design (2026-10-18)

Chosen from a design study of four directions, prototyped in HTML (not committed). The first three recomposed the desktop panes for a phone and were rejected: Generate must show where the generation happens, the preview is the point of the app and must never leave the screen, a persistent tab bar spends space on the least used element, and "do not degrade desktop" means nothing hides behind mobile-style menus, not that desktop may not change. The fourth, below, starts from how the app is used.

**One Workspace, three compositions.** The same panes (Compose, Settings, Model, Stage, History) are built once and arranged for the screen; state, the form and the keyboard shortcuts are shared.

| Mode | Screens | Composition |
|---|---|---|
| Wide (≥ 1024 px) | Desktop, iPad landscape | Today's layout: top bar, toolbar, sidebar, stage with the history strip. |
| Tall (640–1023 px) | iPad portrait | The stage and strip full width on top; Compose and Settings side by side under it, Generate at the foot of Settings. |
| Compact (< 640 px, or under 500 px tall) | iPhone | The stage is the screen. A top strip, a settings strip of chips, and a composer with Generate, in reading order. |

**Touch is a layer, not a breakpoint.** `(pointer: coarse)` changes the size tokens (controls 28 → 36 px, rows 30 → 40 px, action bars 36 → 44 px, the switch 26 × 15 → 40 × 24) and gives numbers − and + steppers with press-and-hold, because a label cannot be scrubbed when a drag scrolls. It applies on an iPad in the desktop layout. A mouse keeps the desktop density.

**The phone, top to bottom.**

- *Top strip*, on every page: menu (drawer with the pages), the workflow as a dropdown, the job state (Ready · Step 12 of 32 · Review), Gallery. Other pages show an arrow back and the page name in place of the workflow.
- *Stage*: the latest output, the forming image during a job, or the review. Swipe sideways to step through history (a pill says "2 of 212 · Latest"); swipe up for the Gallery; tap for the viewer; ⋯ for the asset actions.
- *Settings strip*: one 32 px chip per setting with its value and a dot when changed, ordered by how often each changes in the loop: Steps, Guidance, Seed, Batch, then Size, LoRA, Reference or Source where the workflow has one, then Upscale, then Model; video swaps in Length and Audio. A chip opens only that control at thumb size. All, pinned at the right, opens the full inspector as a sheet with the model controls on top.
- *Composer*: a PROMPT label over one line of the prompt and Generate, both 44 px. Expanded: the Inline | File source switch, the Prompt and Enhanced tabs, Enhance and its options, the negative prompt, Scene over Action for video; file mode shows the path, the selected prompts and the chooser. Generate reads Queue with a count while a job runs, with progress on the composer's top edge; press and hold offers a new seed.

**The job, on every screen.** The live preview (the server renders one at 25, 50 and 75 % of the steps) fills the stage. On wide and tall the full card sits beside it as a column: today's header (workflow, notice, Pause and Cancel, headline, time left and elapsed, item bar) and phase log (Ready with timings, Generate with Skip and Repeat, position, prompt, badges, one bar per stepped stage). On a phone a bar over the preview carries the same facts in one row, Next, Repeat and Cancel under ⋯, and Job details opens the full card as a sheet. An in-run upscale is a second stage (two bars; the review's Approve and Discard sit over the small render between them); a standalone upscale from an asset is its own one-stage job with the source as the preview.

**Desktop changes, both additive:** stepping the preview through history with ← → and hover arrows, with the indicator and Latest in the preview header; and the preview filling the stage during a job with the card beside it instead of a thumbnail in the log.

**Shared on a phone:** the viewer swipes between assets and down to close, with details as a sheet; every asset menu is one bottom sheet (stage, tile, viewer); the Gallery gets a Select mode with bulk delete; menus, the queue, the prompt chooser, the path browser and confirmations render as sheets; toasts dock above the composer. The Prompts builder becomes one column at a time with a Snippets · Sets · Preview switch, last. A web app manifest makes the UI installable; no service worker.

## Stages

Each stage is one commit that rebuilds the SPA, passes `make check` and keeps every existing feature reachable. Stages 1 to 10 shipped on 2026-10-18 (unreleased); the Playwright check under Tests is still to do.

1. Touch layer: pointer-coarse size tokens, the switch from tokens, steppers in `InspectorNumber` with hold-to-repeat (off for W × H and Seed), no scrubbing on coarse pointers, a `layout` store (`compact | tall | wide`, `coarse`).
2. Stage stepping through history: a view index on the Workspace, ← → and hover arrows, the indicator and Latest in the preview header, swipe on touch.
3. The job in the stage: the live preview fills the stage; the full card beside it on wide and tall.
4. Tall composition for portrait tablets.
5. Compact shell: top strip, drawer, back arrow, pages; the manifest and `index.html` metas.
6. Compact Workspace: the settings strip, one-setting popovers, the All sheet (a sheet variant of `Modal` hosting `SettingsPane` with the model controls).
7. Composer: prompt source, tabs, the enhancer, prompt files, the negative prompt, Scene and Action.
8. Compact job bar with Job details, the review, the queue as a sheet, swipe up to the Gallery.
9. Viewer swipe and details sheet, menus as sheets, Gallery Select mode and back arrow, confirmations as sheets.
10. Prompts single-column mode.

### Tests

- Vitest: the layout store, the stepper contract, the view index rules, the chip derivation per workflow and model, the sheet's open and close contract.
- One Playwright run at an iPhone-sized viewport checks that the stage is visible on load, a chip opens its popover, and Generate shows the job in the stage. It runs in `make check` only when Chromium is present, as the frontend tests already gate on their tools.

### Docs and changelog

- CHANGELOG `[Unreleased]`: one entry per stage under Added or Changed.
- `docs/development.md` (Web UI Design System): the touch layer and the three compositions.
- `docs/getting-started.md`: a "From a phone" paragraph: `--host local`, Add to Home Screen, the settings strip and All.

## Alternatives considered

- **Responsive tweaks only** (smaller paddings, fewer columns). They keep the sidebar-first order, which is the actual problem. Rejected.
- **A separate mobile build.** Two UIs to keep in step. Rejected; the phone layout reuses every component and only changes composition.
- **Stage on top with Settings as a half-height sheet from the Generate bar; the three panes as swipeable pages; one long page with a stage that shrinks as you scroll.** The first round of the study. All three kept the inspector as the phone's settings UI and a tab bar at the bottom. Rejected for the reasons under Decided design.
- **Hiding Prompts on phones.** Simpler, but the page still works for small edits; the single-column switch costs little.

## Open questions

- Does https matter here? An installed app on a LAN address is still plain http; it works, but notifications stay unavailable ([Job notifications](job-notifications.md)).
- Live preview cadence: with the preview filling the stage, previews more often than 25, 50 and 75 % may be worth their cost. A backend change, measured first.

## Out of scope

- Offline use and caching.
- Uploading a reference image from the camera beyond what the existing file input already allows.
- Phone landscape: it falls into the compact composition by the height rule and works, but is not tuned.
