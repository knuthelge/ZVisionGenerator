# A log of past jobs

**Status:** Done (v0.13.0b25, 2026-10-21). The routes are `/api/jobs/log` rather than `/api/jobs/history`, and the drawer is titled Job log. The log later moved from `jobs.jsonl` into the app database.

## Problem

Once a job ends, its card stays only until the next job starts. A failure's reason lives in that card and in a toast, and is gone after that. There is no list of what ran last night, how long each job took, which ones failed and why, or which settings a failed job used. Debugging a failure after the fact means the server log file, which only covers the latest run of `ziv ui` and is cleared on every start.

## Evidence

- `WebRunner` keeps terminal jobs for 300 seconds or 25 jobs, whichever is reached first (`terminal_retention_seconds`, `max_terminal_jobs`), then moves a final snapshot to `_pruned`, capped at 500 and in memory only. A restart loses everything.
- `GET /api/jobs` returns only `active_job` and `queued_jobs` (`list_jobs`).
- Each job record already has what a log needs: `job_type`, `context` (workflow, model, prompt, runs, settings), `created_at`, `started_at`, `completed_at`, `status`, `outputs` and the terminal event with the failure message (`_publish_event`, `_TERMINAL_EVENT_TYPES`).
- `jobSettingsPrefill` in the SPA can refill the form from a job's `settings`, as **Load settings** on a queued job does.

## Proposed change

### 1. Write a line per finished job

When a job reaches a terminal state, `WebRunner` appends one JSON line to `<ZIV data dir>/jobs.jsonl` (next to `config.yaml`, honouring `ZIV_DATA_DIR`): `job_id`, `job_type`, `workflow`, `model`, `prompt` (first 500 characters), `settings`, `status`, `created_at`, `started_at`, `completed_at`, `duration_secs`, `outputs` (asset ids), `error` (the terminal event's message, if any) and `notices`. The file is capped at 2 000 lines by rewriting it with the newest 1 500 when it overflows, through `utils/atomic_write.py`. Writing happens in the worker thread that publishes the terminal event, so it never blocks the event loop.

### 2. API

- `GET /api/jobs/history?page=1&status=failed|completed|cancelled` returns the lines newest first, paged, with `total_count`.
- `DELETE /api/jobs/history/{job_id}` removes one entry by rewriting the file without its line, through `utils/atomic_write.py`. Unknown ids return 404.
- `DELETE /api/jobs/history` clears the whole log.

### 3. Jobs drawer

- A clock icon at the right of the top bar opens a right-hand drawer on any page, using the existing `Modal` for focus handling, as a side variant.
- Each row: status icon, workflow and model, the prompt's first line, duration, and how long ago. A failed row expands to the full error text with a **Copy error** button.
- Row actions: **Load settings** (refills the Workspace form and switches to it) and **Delete** (removes the entry).
- A filter row: All, Failed, Completed.
- **Clear log** at the foot of the drawer purges every entry after a confirm.
- Keyboard: `G → J` opens the drawer, matching the other `G →` chords.

### Docs and changelog

- CHANGELOG `[Unreleased]`: "A Jobs drawer lists past jobs with their duration and failure messages, kept across restarts; load a job's settings, delete an entry or clear the log."
- The Web UI section of `docs/getting-started.md` gets a paragraph on the drawer; the Error Log section mentions `jobs.jsonl` and how to clear it.

## Alternatives considered

- **Keep more jobs in memory.** No restart survival and no bound on memory for settings and outputs. Rejected.
- **Store full event histories per job.** Up to 200 events a job, most of them step ticks; the summary line is what the drawer shows, and the outputs already embed the settings. Rejected.
- **A page instead of a drawer.** A page would need a nav slot, which [Merge the admin pages](../admin-page-merge.md) is trying to free; the drawer is reachable from anywhere, which matters for "the job failed while I was on the Prompts page".

## Open questions

- Should jobs run from the CLI also log here? They would need the same writer in the CLI runners; a natural follow-up, not part of this.
- Keep `prompt` whole rather than 500 characters? JSON captions can be long; the first version truncates and keeps the settings, which carry the full text anyway.

## Out of scope

- Re-running a failed job with one click beyond Load settings and Generate.
- Statistics (average time per model, failure rates).
