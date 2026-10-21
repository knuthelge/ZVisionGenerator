# Know when a job is done: title, favicon, sound and notifications

**Status:** Proposed (2026-10-17)

## Problem

A job takes minutes, so the tab is usually in the background when it finishes. The only sign is a toast inside the page and a change of the mascot. The tab title changes only during a review (`documentTitle` in `lib/state/jobView.ts` returns `⏸ Review · …` or the plain title). There is no progress in the title, no mark on the favicon, no sound and no browser notification, so the user has to switch back to check, or misses a failure for an hour.

## Evidence

- `WorkspacePage.svelte` handles completion, failure, loss and cancellation (`handleJobComplete` and friends) by reacting the mascot, refreshing history and adding a toast. The job store exposes lifecycle callbacks (`onComplete`, `onFailed`, `onCancelled`, `onLost` in `lib/state/job.svelte.ts`) and keeps the job across pages (`activeJobStorage.ts`).
- The job state has `currentStep`, `totalSteps` and an ETA for the title to use.
- `Notification` is not used anywhere in the SPA. The index page has no manifest and a static favicon.
- The Web Notifications API needs a secure context. `http://127.0.0.1:8080` and `http://localhost` count as secure; `http://192.168.x.y:8080`, the address used from another machine with `ziv ui --host local`, does not, so `Notification` is undefined there in Chrome and Safari. Any design that depends on notifications alone would not work in the remote case that needs it most.

## Proposed change

All of it runs from one app-level module, `lib/state/jobAlerts.ts`, subscribed in `App.svelte`, so it works on every page, not only the Workspace.

### 1. Title

`documentTitle(base, job)` grows from the review flag to the job state:

| State | Title |
|---|---|
| running, steps known | `37% · Generating · ZVision Generator` |
| running, no steps yet | `Generating · …` |
| review | `⏸ Review · …` (as today) |
| done, tab hidden | `✓ Done · …` until the tab is focused |
| failed, tab hidden | `✗ Failed · …` until focused |

A pure function with tests, like today's.

### 2. Favicon mark

A dot on the favicon while a job runs (teal), after completion (green) and after failure (red), cleared when the tab is focused. The SVG icon is read once and the dot added in a `<circle>`, then set as a data URL on the `link[rel=icon]`. Browsers that ignore SVG favicons fall back to the PNG without the dot, which is fine.

### 3. Sound

A short chime on completion and a lower one on failure, off by default, toggled from a bell menu in the preview header (see below) and stored in the browser with the draft. The sound plays only after the user has interacted with the page, which the browser's autoplay rule requires anyway; the toggle itself counts as interaction.

### 4. Browser notifications

- The bell menu in the preview header has **Notify when done**. Turning it on asks for permission; permission is never requested on load.
- When on, the tab is hidden, and `Notification` is available, completion shows "Generation complete · 4 images" with the first output as the notification image, failure shows the reason, and a review shows "Waiting for your review". Clicking focuses the tab (and opens the review).
- Where the page is not a secure context, the entry is disabled with "Needs a localhost or https address", so the limit is visible instead of silent. The title, favicon and sound cover that case.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "The tab title and favicon show job progress and the result; optional sound and browser notifications when a job finishes."
- The Web UI section of `docs/getting-started.md` documents the bell menu and the localhost/https limit for notifications.

## Alternatives considered

- **Notifications only.** Simplest, but unavailable over plain http from another machine. Rejected as the sole channel.
- **Serve the UI over https** so notifications work on the LAN. Self-signed certificates need a browser exception on every device and a `--tls` option in `ziv ui`. Not rejected, but a separate decision; see open questions.
- **Server-side notifications** (a desktop notification on the machine running `ziv ui`, or a webhook). Useful for unattended batches, but a different feature.
- **App badge** (`navigator.setAppBadge`). Only works for an installed web app; it belongs with [Phone layout](archive/phone-layout.md), which adds the manifest.

## Open questions

- Is an optional `ziv ui --tls` with a self-signed certificate worth it for notifications alone? Probably only together with the installable app in the phone proposal, where https unlocks more.
- Should the sound and notification toggles be per browser (draft) or in `config.yaml`? Per browser: they concern the device that shows them.

## Out of scope

- Notifications for queued jobs starting, or per image in a batch.
- E-mail, Slack or webhook notifications.
