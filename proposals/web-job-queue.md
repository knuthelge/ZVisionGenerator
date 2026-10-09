# Queue jobs in the Web UI and collapse the sidebar

**Status:** Done (2026-10-04, v0.13.0b16)

## Problem

The Web UI runs one generation job at a time and refuses a second one. While a job runs, the user cannot submit another: `WebRunner._submit_job` (`web/web_runner.py`) raises `JobConflictError` (HTTP 409 from `POST /api/generate`), and the workspace disables Generate and most settings through one `busy` flag. To line up a few runs, the user has to wait for each job to finish, change the settings, and press Generate again.

The left sidebar (Compose and Settings) is also fixed at 360 px, so the preview cannot use the full width.

## Decisions

| Topic | Decision |
|---|---|
| Where the queue lives | On the server, in memory. It survives a page reload and every tab sees the same queue. Stopping the server clears the queue and does not wait for queued jobs. |
| Order | First in, first out. No reordering and no "pause the queue" in this version. A paused running job holds the queue until it is resumed or stopped. |
| Editing a queued job | Not supported. A queued job can be removed, or its settings copied back into the form with **Load settings**. |
| Settings of a queued job | Fixed at submit time: form values, prompt files and config are read when the job is submitted, not when it starts. |
| A job fails | The queue carries on with the next job. |
| Clearing | **Clear queue** removes every queued job and never touches the running job. It asks for confirmation (see [Confirmation dialog](#confirmation-dialog)). |
| Queue size | No cap. Accidental submissions are handled by ignoring key repeat, blocking Generate while a submit is in flight, and Clear queue. |
| Preflight (auto-enhance) | Unchanged: each job runs its own preflight when it starts, then generates. |
| Manual Enhance button | Disabled while any job is running or queued, with the note "Available when all jobs have finished." The enhancer and a generation cannot share memory. |
| Generate button | Reads **Add to queue** while a job is running or queued; **Generate** otherwise. |
| Queue placement | Under the job card in the preview area ("Up next"). |
| Job card | Two columns: progress, timing and controls on the left; live preview and outputs on the right, their labels on one line. Stacks on narrow screens. |
| Sidebar | Collapses to a 56 px strip with an expand button, a Generate / Add to queue button and a badge with the queued-job count. No keyboard shortcut. The choice is remembered per browser. |
| Other tabs | Kept current by refreshing the queue list (see F3); no extra event stream. |
| Load settings | Restores every submitted form field. |

A static wireframe of the layout and states was made during design (not committed).

## Definitions

- **Generation job:** a job submitted through `POST /api/generate` (today's `exclusive` jobs).
- **Active job:** the job the worker has claimed (running or paused). When no job is claimed, the job at the head of the queue, waiting for the worker to pick it up.
- **Queued jobs:** every other unfinished generation job, oldest first. A job is never both the active job and a queued job.
- **Queue position:** 1-based position among the queued jobs; `null` for the active job and finished jobs.

## Backend

**B1. One worker.** `WebRunner` uses a single-thread executor (`max_workers=1`) for all jobs. The only other job kind, the dummy job, is used in tests only and is not exposed by the production app; tests that need concurrency pass `max_workers`. Only generation jobs go through the queue order (B2) and the claim step (B5); other jobs (the test-only dummy job) run as before.

**B2. Queue order.** `WebRunner` keeps an ordered `deque` of waiting job ids, guarded by `_jobs_lock`. It is the only source of queue order. `WebRunner` also records the claimed job id (the job the worker is running), under the same lock. Status constants move to `job_contract.py`: `QUEUED_STATUS = "queued"` and an `ACTIVE_STATUSES` set (`queued`, `running`, `paused`).

**B3. Accept instead of refuse.** `_submit_job` no longer raises `JobConflictError` for a second generation job. Under `_jobs_lock` it runs the admission check, registers the record with status `queued` and its full public context, publishes `job_submitted`, appends the id to the deque, and calls `executor.submit`, all before releasing the lock. Holding the lock through dispatch keeps the executor's order equal to the deque order when two tabs submit at once, and `job_submitted` always comes before `job_started`. The admission check (`_reject_while_busy`) still refuses a submission while a manual prompt enhancement runs or a quantized copy is being saved. Both can only start while no job is active, so while jobs are active the check lets the submission through: a busy enhancer then belongs to a job's own preflight.

**B4. Full context at submit.** The public context (workflow, prompt, model, runs, size label (`meta`), created time, output dir and the stored settings from B10) is passed into `_submit_job`, so no reader ever sees a queued job without it. `_generate_from_form` no longer calls `update_job_context` after dispatch, and `WebRunner.update_job_context`, which has no other caller, is removed. The `POST /api/generate` response adds `queue_position` (`null` when the job is the active job).

**B5. Claim step.** `_run_target` begins by claiming the job under `_jobs_lock`: if the id is no longer in the deque (removed, cleared or shut down), it returns at once without running the target, releasing memory or publishing events. Otherwise it removes the id, records it as the claimed job, records `started_at`, and publishes `job_started`, which sets the status to `running`. Status is no longer set to `running` by whatever event comes first. The claimed job id is cleared when the job's terminal event is published.

**B6. Elapsed time.** `elapsed_secs` in events is measured from `started_at` (from `created_at` until the job starts), so time spent waiting in the queue is not counted.

**B7. Remove a queued job.** New `WebRunner.cancel_queued(job_id) -> bool`: under `_jobs_lock`, if the id is in the deque (including the active job while it still waits at the head), remove it; then publish `job_cancelled` (with `reason: "removed"`) outside the lock and return `True`. The record is not marked cancelled before the event: the event sets the status, so every reader sees the status and the terminal event together. `Future.cancel()` is called as a best effort only; the claim step (B5) is what guarantees the job never runs. `POST /api/jobs/{job_id}/cancel` tries `cancel_queued` first, before the supported-controls check, so queued video jobs can be removed too. For a running job it falls back to today's `quit` control; a running video job still cannot be stopped (see [Controls for Web UI video jobs](video-job-controls.md)).

**B8. Clear the queue.** New `DELETE /api/jobs/queue` removes every queued job the same way as B7, never the active job (when no job is claimed, the head of the deque is the active job and is kept), and returns `{"cancelled": [<job ids>]}`. It is declared before any parameterised `/api/jobs/{job_id}` route.

**B9. List jobs.** New `GET /api/jobs` returns `{"active_job": <snapshot or null>, "queued_jobs": [<snapshot>, ...]}` following the definitions above. `GET /api/workspace` adds `queued_jobs` next to its existing `active_job`, which follows the same definition. Queue positions are computed from the deque under `_jobs_lock`, skipping the head when it is the active job, before any `record.lock` is taken (the existing lock order), and passed into `public_job_snapshot`, which gains `queue_position` and `started_at`. Pruned snapshots report `queue_position: null`.

**B10. Stored settings.** Each generation job stores the submitted text form fields verbatim in its context (`settings: dict[str, str | list[str]]`; a field sent more than once, such as `prompt_option_id`, is a list). When an Enhanced prompt override was active, `prompt` holds the enhanced text that was submitted. An uploaded reference image (`image_file`) is replaced by its saved upload path as `image_path`. Snapshots of queued jobs include `settings`, so the frontend can restore the form from them.

**B11. Shutdown.** `WebRunner.shutdown()` empties the deque, publishes `job_cancelled` (`reason: "removed"`) for every queued job and shuts the executor down with `cancel_futures=True`, so stopping the server does not run the rest of the queue. The running job is left as today.

**B12. Existing "busy" checks.** No logic change: `admit_exclusive` (manual enhance, quantized copy) and `_ensure_no_active_job` (deleting models and LoRAs) already treat any unfinished job, queued or not, as busy and refuse with 409. Only their messages change from "the current job" to "all jobs" (for example "Prompt enhancement is available when all jobs have finished.").

**B13. Controls before a job starts.** `POST /jobs/{job_id}/controls/{action}` refuses `pause`, `next` and `repeat` with 409 ("This job has not started yet.") for a job that has not been claimed. Cancelling goes through B7.

**B14. Memory between jobs.** `_run_target` already releases accelerator memory before publishing a job's terminal event, so the next queued job starts on freed memory. Kept as is and covered by a test.

## Frontend

**F1. Unlock the form.** `busy` in `WorkspacePage` covers only the in-flight `POST /api/generate`. Compose, Settings, Reset and the auto-enhance toggle stay usable while jobs run. A separate `jobsActive` flag (an active job exists or the queue is not empty) drives the Generate label ("Add to queue"), the manual Enhance button and its note, and the ⌘E shortcut. Mount no longer sets `busy` after reconnecting to a job. The ⌘↵ handler ignores repeated key events (`event.repeat`). `#ws-busy-note` is shown while `jobsActive` (not `busy`) and reads "A job is running. New runs join the queue."

**F2. Job store.** `lib/state/job.svelte.ts` keeps following the active job and adds `queuedJobs`.

- After a submit, the store decides from the response's `queue_position`, not from its local state: `null` means the new job is the active job and is followed as today; a number means it is queued, so it is added to `queuedJobs`, the store keeps following its current job (if any) and refetches `GET /api/jobs` to pick up an active job another tab started.
- When the followed job reaches a terminal event, the store fetches `GET /api/jobs` and follows the new active job, if any. This happens inside the store, so it also works while the user is on another page.
- `job_started` is added to the SSE event list in `lib/api/sse.ts`; it updates the followed job's status.
- Removing or clearing queued jobs updates `queuedJobs`. It never triggers the "stopped" lifecycle callbacks, toasts or mascot reaction. A `job_cancelled` event with `reason: "removed"` for a followed job is handled the same way, also when it is read from a snapshot's `last_event` during reconnect or stream recovery.
- The existing per-job toasts stay. A failure while jobs are queued adds "Starting the next queued job."

**F3. Keeping tabs current.** The store refetches (the workspace page starts and stops this sync; following the next job after a terminal event works on any page) `GET /api/jobs` after a submit, remove or clear; after the followed job ends; when the tab gains focus or becomes visible; and while the tab is visible: every 4 seconds while `jobsActive`, every 15 seconds otherwise. A visible tab therefore picks up another tab's jobs within 15 seconds (4 seconds once jobs are active). Polling stops while the tab is hidden.

**F4. Queue panel.** An "Up next · N" list under the job card: position, prompt, model, workflow, runs and size label, a "Starts next" tag on the first item, **Load settings** and a remove button per item, and **Clear queue** in the header (opens a `ConfirmDialog`). Hidden when nothing is queued. After the queue is cleared, focus moves to the Generate button.

**F5. Load settings.** Applies the job's `settings` to the draft, covering every field the form submits, through the same path as prefill (`applyPrefill` / `draft.loadFromUrl`), extended where a field is missing. A reference image is restored from `image_path`. If that file has since been deleted, the next submit is refused by the existing reference-image validation, as for any missing path.

**F6. Two-column job card.** `JobCard` places the live preview and outputs to the right of the progress, timing and controls, with the "Live preview" and "Outputs" labels aligned on one line. The columns stack below the existing narrow breakpoint. The P / N / R / Esc job shortcuts are unchanged.

**F7. Collapsible sidebar.**

- A collapse button in the sidebar header and an expand button on the collapsed strip, each with an accessible label. The strip's Generate button submits the same form; it does not reuse `id="ws-submit"`.
- The sidebar stays in the DOM when collapsed (hidden, not removed), because its fields are part of the generate form.
- The form listens for `invalid` in the capture phase. The browser's own report fails for a hidden field, so when the sidebar is collapsed the handler expands it, waits for the DOM update (`await tick()`), then calls `reportValidity()` on the invalid field, the same pattern as `ComposePane.showPromptForValidation`.
- The collapsed state is stored in `localStorage`, with every read and write wrapped in `try`/`catch`.
- Below 640 px, where the sidebar already stacks above the preview, the collapse control is hidden and the sidebar is always shown.

## Confirmation dialog

A new shared `ConfirmDialog` molecule becomes the convention for asking the user to approve an action.

**C1. Content.** No title. The dialog shows the question, an optional line of extra information (or a short body, for the Models delete details), and two buttons: a cancel button (for example **Keep**, **Cancel**) and a confirm button named after the action (for example **Clear queue**, **Delete**), styled as danger for destructive actions.

**C2. Semantics.** `role="alertdialog"`, `aria-modal="true"`, `aria-labelledby` pointing at the question and `aria-describedby` at the information. `Modal` gains props for the role and the labelling ids, and keeps `aria-label={title}` when a title is given.

**C3. Keys.**

- Esc cancels. Enter confirms, except when focus is on the cancel button (where Enter activates Cancel as usual). Repeated key events are ignored, so a held Enter or Del cannot confirm.
- Focus starts on the dialog. Closing returns focus to the opener, or to a fallback the caller names when the opener no longer exists.
- The dialog handles Enter and Esc itself, in a capture-phase listener that stops propagation of every key except Tab. Tab still reaches `Modal`'s focus trap; no other key reaches the page or a viewer underneath. Page shortcuts (Esc to stop a job, P / N / R) are already blocked while a modal is open (`hasOpenModal` / `acceptsPageShortcut`).

**C4. Stacking.** The dialog can open on top of the full-screen `AssetViewer` (deleting from the viewer). It sits above the viewer, and the viewer ignores keys while a confirm dialog is open, the same way it does for its reference popover. Esc closes only the dialog.

**C5. Uses.** All existing confirmations move to it:

- Clear queue (new).
- Deleting assets in the Gallery (one asset and a multi-select) and in the Workspace, which use the browser's `confirm()` today (`GalleryPage.svelte`, `WorkspacePage.svelte`). These deletes become asynchronous, so the Del and ⌫ shortcuts in the Gallery and the viewer ignore key repeat.
- Deleting a model or LoRA on the Models page, which uses its own `Modal` today (`ModelsPage.svelte`). Its details (linked downloads, stored quants, "This cannot be undone.") become the dialog body.

**C6. Convention.** `docs/development.md` documents `ConfirmDialog` as the way to ask for approval; `window.confirm` is not used.

## Tests

| Step | Test |
|---|---|
| B1, B3 | A second generation job is queued (status `queued`, position 1), not refused; jobs run one at a time, in order. |
| B3 | Concurrent submits from two threads run in the order of the deque; `job_submitted` precedes `job_started`. |
| B3 | Submissions are still refused while a prompt enhancement runs or a quantized copy is saved. |
| B4 | A queued job's snapshot has its prompt, model, runs, size label and settings as soon as `/api/generate` returns; the response includes `queue_position`. |
| B5, B7 | A job removed after the worker picked up its work item never runs and publishes only `job_cancelled`; a started job publishes `job_started` and becomes `running`. |
| B6 | `elapsed_secs` of a job that waited in the queue starts near zero. |
| B7 | Cancelling a queued image or video job through `POST /api/jobs/{id}/cancel` removes it; cancelling a running image job still sends `quit`; a running video job still returns 409. |
| B8 | `DELETE /api/jobs/queue` cancels only queued jobs, returns their ids, and leaves the active job, whether running or waiting at the head. |
| B9 | `GET /api/jobs` and `/api/workspace` never list a job in both `active_job` and `queued_jobs`, including between two jobs; positions update as jobs start. |
| B10 | `settings` holds every submitted text field, including several `prompt_option_id` values as a list, with an uploaded image replaced by `image_path`. |
| B13 | `pause`, `next` and `repeat` on a job that has not started return 409. |
| B1, B5 | A dummy job runs through the queue and the claim step. |
| B11 | After `shutdown()`, no queued job starts, and each one is reported cancelled. |
| B12 | Model and LoRA deletion and manual enhancement are refused while jobs are queued, with the updated messages. |
| — | A failed job does not stop the queue; memory is released before the next job starts. |
| F1 | The form stays enabled while a job runs; the button reads "Add to queue"; ⌘↵ key repeat submits once; Enhance and ⌘E are disabled while jobs are active. |
| F2 | A submit answered with a `queue_position` adds to the queue even when the tab has no local job; a removed job read from a snapshot shows no "stopped" toast. |
| F2 | Submitting while a job runs keeps the card on the running job; when it ends the store follows the next job, also when the Workspace page is not mounted; removing a queued job shows no "stopped" toast. |
| F3 | The store refetches the job list on focus; polls every 4 s while jobs are active and 15 s otherwise while visible; stops while hidden. |
| F1 | The busy note shows while jobs are active, not only during a submit. |
| F4 | Queue items render and remove; Clear queue opens the dialog and calls the endpoint only on confirm; focus moves to Generate afterwards. |
| F5 | Load settings round-trips every submitted field, including several prompt-file options and a reference image path. |
| F6 | The job card keeps its controls and P / N / R / Esc shortcuts. |
| F7 | The collapse state persists; a collapsed form submits every field; collapsed with an empty prompt, Generate expands the sidebar and reports the prompt (`reportValidity` after the update). |
| C1–C3 | `ConfirmDialog` is named by its question; Enter confirms; Enter on the cancel button cancels; Esc cancels; key repeat does not confirm; Tab stays inside the dialog. |
| C4 | A confirm dialog over the asset viewer: Esc closes only the dialog; arrow keys do not move the viewer. |
| C5 | Gallery, Workspace and Models deletes go through `ConfirmDialog`; the existing `window.confirm` stubs in `GalleryPage.test.ts` and `WorkspacePage.test.ts` are replaced. |

## Docs and changelog

- `CHANGELOG.md` under `[Unreleased]`: job queue, collapsible sidebar, two-column job card, confirmation dialogs.
- Web UI docs in `docs/`: queueing, Up next, Clear queue, Load settings, collapsing the sidebar, the Enhance note, and that queued jobs use the settings from when they were submitted.
- `docs/development.md`: the `ConfirmDialog` convention and the single job worker.

## Known limitations

- A reference image picked from the gallery can be deleted while its job waits; that job then fails and the queue continues.
- Reloading a model for each job, even when consecutive jobs use the same one.

## Alternatives considered

- **A queue in the browser only.** Simpler on the server, but lost on reload, and two tabs would build separate queues.
- **A live event stream for the queue.** Rejected for now in favour of refreshing the list (F3); it can replace polling later without changing the API.
- **Queue in the sidebar footer, or a popover from the top bar.** The footer is cramped and disappears when the sidebar collapses; a popover hides the queue unless opened.
- **Inline confirmation for Clear queue.** Rejected in favour of one dialog convention for every approval.
- **A queue cap.** Rejected: it limits the user without a technical need; each queued job holds only its settings and prompts.

## Out of scope

- Reordering, pausing the queue, and editing a queued job.
- Keeping a model loaded between queued jobs that use the same model.
- Persisting the queue across server restarts.
- Stopping a running video job ([Controls for Web UI video jobs](video-job-controls.md)).

## Implementation notes

Built on `feat/web-job-queue` in four commits: the confirmation dialog, the collapsible sidebar and two-column job card, the backend queue, and the frontend queue. Differences from the plan above:

- The claim step applies to generation jobs only (B1), so test-only dummy jobs keep running without the queue.
- `_reject_while_busy` skips the enhancer and quantize checks while jobs are active (B3). Without this, a job's own auto-enhance preflight made the enhancer look busy and refused every submission behind it.
