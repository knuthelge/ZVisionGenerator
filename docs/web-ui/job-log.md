# Job log

The **Job log** lists the jobs that have finished, newest first. Open it with the clock button in the top bar, or press ++g++ then ++j++.

![The Job log drawer with a failed job expanded](../assets/screenshots/placeholder.svg "job-log")

## Each row

A row shows the workflow, the model, the first line of the prompt, how long the job took and how long ago it ended. It is marked **Completed**, **Failed**, **Cancelled** or **Some runs failed**. A failed row expands to show why.

## Filter

**All**, **Failed** and **Completed** narrow the list. Older jobs are on further pages.

- A job that finished with some runs failed is listed under both **Failed** and **Completed**.
- A cancelled job with failed runs is listed under **Failed** only.

## Actions

| Action | Does |
|---|---|
| **Load settings** | Opens the Workspace with the settings that job used, with the model's defaults around them. Not offered for an upscale. |
| **Copy error** | Copies the failure text. On a page opened over plain `http` from another computer the browser allows no clipboard, so the text is selected for you to copy. |
| **Delete** | Removes one entry. |
| **Clear log** | Removes the whole log, after asking. |

## What is kept

The log is kept on disk in `jobs.jsonl` in the [data folder](../concepts/data-folder.md#job-log), so it survives a restart. Jobs removed from the queue before they started are not recorded.
