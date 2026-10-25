# Variations and sweeps from the Workspace

**Status:** Proposed (2026-10-17)

## Problem

The two most common next steps after a good output are "more like this, different seeds" and "the same thing at a few values of one setting". Both need several manual steps today:

- **Variations.** Reuse settings on the output, which loads everything including its seed; unlock the seed; set the batch size; press Generate. Four steps, and Reuse replaces whatever was in the form.
- **Sweeps.** There is no way to ask for steps 4, 6 and 8 or guidance 3, 4 and 5 in one go. Each value is a separate submission, and the outputs give no hint which value made them.

## Evidence

- `reuseParams` in `lib/state/assetActions.ts` turns the server's `reuse_workspace_url` into form params; the server includes `seed` whenever it was recorded (`gallery.py`, reuse params). The Settings pane has a seed lock, a new-seed button and a batch size (`runs`, 1 to 100).
- With no seed, `image_runner.py` draws one per run with `random.randint`, so a batch is already "N different seeds".
- The server queues jobs (`WebRunner._submit_job`); while a job runs, Generate reads **Add to queue**, and `loadQueuedSettings` refills the form from a queued job's `settings`, so settings round-trip through `JobSettings` already.
- `⌘/Ctrl+⇧+↵` generates with a new seed, which shows the "new seed" case is common enough to have a shortcut.
- Provenance (`utils/provenance.py`) embeds a `generation` block in every output; a sweep can record its axis there without a new file format.

## Proposed change

### 1. Variations from any output

- **Variations ▾** in the tile menu and the viewer (key `V`), with 2, 4 and 8 as entries, like the Upscale submenu. For videos the entries are 1 and 2, since each run is minutes.
- It submits **one job**: the asset's reuse params without `seed`, `runs` set to the count, through the same multipart form `handleSubmit` builds. The form serialisation moves out of `WorkspacePage.handleSubmit` into a pure `generateForm(settings)` so both paths share it; the Workspace draft is not touched.
- The toast says "4 variations of `name.png` queued" or "started", and the job card's headline says "Variations of `name.png`" from a `variation_of` entry in the job context.
- Outputs record `generation.variation_of: <source path>` so the viewer can show and link the source, as it does for upscales.

### 2. Sweeps in the Settings pane

- A **Sweep** section at the foot of the Settings inspector, collapsed by default and off unless enabled. It takes one axis, optionally two:
  - Axis: Steps, Guidance, Seed, Scheduler, Quantize, Image strength, First sigma, or the weight of one chosen LoRA.
  - Values: a list (`4, 6, 8`), a range (`3..6`, `0.4..0.8 step 0.1`) or, for Seed, `random × N`.
- The section's summary line shows the count: "Steps × Guidance · 9 jobs". The Generate button reads **Generate 9 jobs**, with a cap of 50.
- On submit the client expands the draft into one job per combination and posts them in order; the queue keeps the order. Each job's context carries `sweep: {id, x: {name, value}, y?: {name, value}}`, and the output's provenance records the same under `generation.sweep`.
- The completed-outputs grid and the viewer details show the axis values under each output ("steps 6 · guidance 4"). The Gallery search from [Search and favourites](gallery-search-favourites.md) can find a whole sweep by `sweep:<id>`.
- The Copy CLI command copies the base command plus a comment line listing the values, since the CLI has no sweep flag; that is recorded as a caveat the way `prompt_options` is.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "Generate variations of any output with new seeds, and sweep one or two settings over a list of values."
- `docs/web-ui/workspace.md` explains Variations and the Sweep section.

## Alternatives considered

- **Expand the sweep on the server** (one request with a `sweep` field; the server enqueues N jobs). One request and one place to validate, and a `--sweep` CLI flag could share it. Rejected for the first version because the client already has everything it needs and the form parsing in `_generate_from_form` stays unchanged; it is the natural second step if the CLI wants sweeps.
- **A contact sheet of the sweep's outputs.** Useful, but it is a compare feature, which was dropped. The per-output labels and the Gallery search cover finding them.
- **Variations as "Reuse, then random seed, then runs = 4" inside the draft.** Rejected: it replaces the form the user is working on, and leaves the draft in a state they did not ask for.

## Open questions

- Should a sweep over `random × N` seeds be the same as Variations from the current draft? Probably yes; Variations is then the asset-side entry to the same path.
- Which prompt variants belong in a sweep: should `{a|b}` choices expand as an axis? Not in the first version.

## Out of scope

- Grid or side-by-side views of the results.
- Sweeping video length or size, which change memory and time per job.
