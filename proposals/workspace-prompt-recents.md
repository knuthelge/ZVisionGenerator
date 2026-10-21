# Recent prompts, rewrite history and "save to a prompt file"

**Status:** In progress (2026-10-21)

## Shipped (unreleased): save a Workspace prompt to a prompt file

- **Save…** sits in the prompt box's tools row (not in a Compose header menu) and offers Recent files, **Browse…** and **New file…**.
- It saves the text the prompt box shows (Original or Enhanced, with the scene for video and the negative prompt when shown) and writes no `enhance:` field.
- A server route, `POST /api/prompt-files/append`, appends the entry atomically; the client no longer loads and saves the document.
- The entry always goes to the end of a set named `prompts`, created at the end of the file when missing, rather than the last set or a set named after the model.
- The route validates the whole resulting file and refuses with a message when it would have errors; the toast offers **Open in Prompts**.

The shipped design differs from the original section 3 in these points. Sections 1 and 2 below are still open.

## Problem

The Workspace remembers one prompt: the one in the box. Typing a new one loses the old one unless a job finished and produced an output to reuse. A prompt whose job failed, was cancelled, or was cleared before Generate is gone. The enhancer keeps one rewrite (`enhancedPrompt`); pressing Enhance again overwrites it, so there is no way back to the previous rewrite. The Prompts page builds prompt files, but nothing carries a Workspace prompt into one.

## Evidence

- `DraftState` holds `prompt`, `scene`, `negativePrompt`, `enhancedPrompt`, `enhancedScene` and `enhancedFrom`, each a single value, saved under one `localStorage` key (`ziv-workspace-draft-v1`).
- The Prompts page already keeps a recent-files list in the browser (`lib/state/promptFileStorage.ts`, `RECENT_KEY`, 8 entries), so the pattern exists.
- The prompt-file API can load and save a document (`POST /api/prompt-files/document`, `PUT /api/prompt-files/document`) and create a file (`POST /api/prompt-files/create`); `NewPromptFileDialog` and the path browser (`purpose: prompt_file`) exist.
- `submittedPair` in `lib/state/promptEnhance.ts` already knows which text a job was submitted with.

## Proposed change

### 1. Recent prompts

- On every Generate and every finished Enhance, the submitted text (prompt, scene, negative prompt, mode, time) is pushed to a browser store, `ziv.workspace.recentPrompts`, deduplicated by text, newest first, at most 50. Pinned entries never fall off.
- A clock button in the Compose header opens a list (the `ActionMenu` pattern with a filter field): first 80 characters, age, a mode badge, a pin. Choosing one fills the prompt and, for video, the scene. `Alt+↑` in the empty prompt opens it too.
- **Clear recents** at the foot of the list.

### 2. Rewrite history

- The draft keeps the last five rewrites of the current text (`enhancedHistory: {prompt, scene, settings, at}[]`) instead of one. The Enhanced tab gets a `‹ 2 of 3 ›` stepper; the shown rewrite is the one Generate uses.
- The history is cleared when the source text changes, which is when the existing "Out of date" badge appears, so the two rules match.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "The Workspace remembers recent prompts and the last rewrites."
- The Web UI section of `docs/getting-started.md` mentions the clock button and that recents are kept in the browser, like the Workspace settings.

## Alternatives considered

- **Keep recents on the server** (`~/.ziv/`), so they follow the user between machines. Tempting, since the UI is often used from another device. Rejected for the first version to stay consistent with the draft, which is browser-local; the store sits behind one small module so it can move to the server later without touching the UI.
- **Recent prompts from the Gallery alone.** Reuse already covers finished outputs. Rejected: the point is the prompts that did not produce an output.
- **Unlimited rewrite history.** Rejected: five covers "the one before was better" without a second list to manage.

## Open questions

- Should a prompt picked from recents also restore the settings used with it? Not in the first version; presets ([Settings presets](settings-presets.md)) are the place for settings.

## Out of scope

- A prompt library page; the Prompts page is that.
- Syncing recents between browsers.
