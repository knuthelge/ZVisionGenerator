# App database

`ziv.sqlite3` in the data folder is one shared SQLite file for features that keep data between runs. `utils/app_db.py` is stdlib only, so the CLI and the web layer can both use it. It holds no lock and no module-level connection, and it never creates, alters or drops a feature's tables itself. This page is the one home of what is stored in it and why.

## How a feature uses it

- **Components.** A feature declares `Component(name, version, migrate)`. Its tables are named `<name>_*`, and it never reads or writes another component's tables. `connect(components=(...))` brings each listed component to its version before it returns.
- **Versions only go up.** `schema_versions` stores, per component, the highest version that has run on this file, and that number is never lowered. When the stored version differs from the component's, `connect` runs the migration in one `BEGIN IMMEDIATE` transaction; if it raises, the transaction rolls back and the error propagates. A migration also receives a stored version higher than its own, which means a newer ziv wrote the file. A data component raises, so an older ziv leaves a newer component alone. A cache component keeps a table per version instead, so two ziv versions sharing a data folder (a checkout and an installed copy) never rebuild each other's tables.
- **Connections.** Open one short-lived connection per operation, `with closing(app_db.connect(components=(MY_COMPONENT,))) as conn:`. Never keep a module-global connection and never share one across threads. Run every write inside `with transaction(conn):` and keep it short and free of slow I/O. A lock belongs to the feature that needs it. `app_db.run` is the shared connect, recover, close and error-wrapping step.
- **Settings of the file.** WAL, a 5 s `busy_timeout`, `foreign_keys=ON`, `synchronous=NORMAL` and autocommit. The switch to WAL is retried for up to the busy timeout, because SQLite does not apply the timeout to that switch.
- **Corruption.** `recover_if_corrupt(conn, exc)` is the one place the policy lives. A file that is damaged (`SQLITE_CORRUPT`, `SQLITE_NOTADB`) is quarantined: renamed, with its `-wal` and `-shm`, to `ziv.sqlite3.corrupt-<YYYYmmdd-HHMMSS>`, and only if the file at the path still is the one this connection opened, so a file another process already replaced is left alone. The app never deletes the database. Errors that do not mean a damaged file (cannot open, read-only, busy, a directory at the path) never rename anything. Pass the exception to `recover_if_corrupt` before deciding what to do with a failed operation.
- **Tests.** The autouse `_isolated_app_db` fixture points the database at a throwaway folder; see [Testing](testing.md#protective-fixtures).

Two kinds of component exist, and the difference decides how a change is made:

| Kind | Components | If the data is lost |
|---|---|---|
| Cache, rebuildable from files | `gallery`, `loras` | Re-read from disk; nothing is lost |
| Data, kept for the user | `jobs`, `recent_prompts`, `lora_sources`, `assets` | Lost; a migration must copy the rows it keeps, and a quarantine keeps them in the `.corrupt-*` file |

## Job log

`web/job_log.py` is the `jobs` component: one row per finished job in `jobs_entries_v1`, ordered by `seq`. `server.py` builds it as `JobLog(app_db.app_db_path)`, and `WebRunner` writes nothing without one. Each row holds the job id and the entry's JSON exactly as written, so a listing returns what was written.

- **Which jobs.** Only image, video, audio and upscale jobs that started. A job removed from the queue never started, and `dummy` jobs are not generation jobs.
- **Once only.** A job is logged when it first reaches a terminal status. A later terminal event changes the in-memory status but writes no second entry, so the log keeps the first outcome. A job whose runs partly failed keeps its `completed` or `cancelled` status and counts them in `failed_runs`, which the `failed` filter matches.
- **The write never reaches the job.** `_publish_event` copies the facts under the record's lock and writes last, holding no lock, on the worker thread. Building and writing share one `try/except Exception` that only emits a `UserWarning`, so the terminal event, the release of the claim and the next queued job do not depend on the log. If a future change logs queue removals, the write must move off the event loop.
- **Best effort.** Rows are never trimmed; Delete and Clear log remove them. A server stopped while a job runs may exit before its entry is written. Two `ziv ui` servers on one data folder share the table safely.
- **Routes.** `GET /api/jobs/log`, `DELETE /api/jobs/log/{job_id}` and `DELETE /api/jobs/log` are declared before the `/api/jobs/{job_id}/...` routes. A storage failure, or a schema newer than this ziv (`JobLogError`), is a 500 with `Could not read or write the job log: <reason>`.

## Recent prompts

`web/recent_prompts.py` is the `recent_prompts` component: one row per distinct mode, format, prompt, scene and negative prompt in `recent_prompts_entries_v1`, newest by `seq`. It is separate from the job log on purpose: the log writes a job only after it ended, truncates the prompt, and its Delete and Clear would otherwise remove recents.

- **What is recorded.** `_generate_from_form` calls `_remember_prompt` after `POST /api/generate` accepted a job. `recent_prompt_entry` (pure) builds the entry for an inline submit and returns nothing for a prompt file or a blank prompt. A refused submit never reaches it. The text recorded is what the form sent.
- **Modes.** `image`, `video`, and for audio the kind (`music`, `speech`, `sound`), each with a list of its own. A song keeps its lyrics in `scene`.
- **Dedupe.** The key is `(mode, format, prompt, scene, negative_prompt)`. Recording an existing one gives the row a new highest `seq` and the new `used_at`. Nothing is capped or trimmed.
- **Failures never reach the submit.** `_remember_prompt` only emits a `UserWarning`. The routes (`GET` and `DELETE /api/recent-prompts`, `mode` required) turn a storage failure or a newer schema (`RecentPromptsError`) into a 500.

## Gallery index

`web/gallery_index.py` is the gallery component, a cache. `list_gallery_assets` calls it on every listing, so `/api/gallery` and `/api/workspace?include_history=true` read a file's embedded metadata once per file version instead of once per request, including the first request after a restart.

- **Raw facts, not derived fields.** A row stores what was read from the file (the embedded settings and the image, video or audio facts), not the derived `GalleryAsset`. The derivation (recorded settings, filename parsing, prompt fallback) runs on every request, so fields that depend on other files or on the clock are never stale, and a change to the derivation needs no migration. `gallery._read_asset_facts` is the only metadata I/O.
- **Reconciliation.** The root is walked with `stat` only, never descending into staging directories. A row is used only when its `(mtime_ns, ctime_ns, size)` equals the walked stat, so a stale or orphan row costs at most one re-read and can never show a file or wrong metadata.
- **When a read is persisted.** Only when no reader failed, the file's mtime is at least 2 s before now (a file still being written is not cached) and a re-stat after the read returns the same key. Persisted reads are committed every 64 files, so an interrupted cold scan keeps its progress.
- **Pruning.** Rows for files the walk did not see are deleted only by a walk without errors that found at least one media file, because the output folder is created on demand and an unmounted drive looks like an empty folder. A missing root keeps its rows. `delete_gallery_assets` calls `GalleryIndex.forget()`, which removes rows at once.
- **Versioned table.** The table is `gallery_assets_v{N}`, `N = GALLERY_INDEX_VERSION`. Bump the constant when the columns change or when `_read_asset_facts` reads something different. The migration drops lower-versioned tables only when the stored highest version is below `N`, and never a higher one.
- **Never fails the gallery.** An error while opening or loading falls back to reading every file uncached, with a warning, after `recover_if_corrupt` quarantined a corrupt file. A write error keeps the results already read.
- **Locking and moves.** One `threading.Lock` surrounds a reconciliation, so concurrent cold requests read each file once; `forget()` does not take it. `GalleryIndex.rename(root_key, moves)` re-keys rows of files moved between projects, so a moved video is not probed again.
- **Job completion** writes no row. The next listing picks the output up, and `gallery_asset_for_output_path` reads uncached.

## Asset records

`web/asset_records.py` is the `assets` component: one row per file in `assets_entries(asset_id, starred)`, keyed by the output-root-relative path so records survive a moved or renamed output folder. The choices cannot be rebuilt from the files, so it is a data component and not a column of the gallery cache. A new choice is a column with a default, not a new table.

- **Newer versions are accepted.** `migrate` never refuses a stored version above its own, because later versions only add columns, which this code ignores.
- **Moves and deletes.** `AssetRecords.rename` runs with `GalleryIndex.rename` in `gallery._rekey_moved` and replaces any row at the new ID. Any delete calls `AssetRecords.forget`.
- **Listing never fails on it.** `list_gallery_assets` reads the starred set once per listing through `_starred_ids`, which warns and lists every file unstarred on an `AssetRecordsError`. The star route turns the same error into a 500, because a star that did not save must not look saved.
- **Not pruned.** A file deleted or renamed outside ziv keeps its row, and a new file that reuses the name shows the old star.

`tests/web/test_web_asset_records.py` covers the store and `tests/web/test_web_gallery.py` the listing, move and delete.

## LoRA index

`web/lora_index.py` is the `loras` component, a cache of which model family each LoRA file was trained for, so the Workspace lists the LoRAs made for the selected model first. `utils/lora_family.py` holds the pure detector (metadata first, then a key and shape fingerprint); `utils/safetensors_stream.read_header_and_metadata` reads a header in one bounded read, so no tensor data is read.

- **Table and key.** Rows of `loras_entries_v{N}` (`N = LORA_FAMILY_VERSION`) are keyed by `(root, name)` and hold `family`, `size_variant` and the single-caption `caption` of the header beside the stat key. A row is used only when the stat key equals the file's.
- **Lookup.** One `os.stat` per name and one database read. A file with no row or a stale key is detected inline and stored, so each file is read once until it changes; a header that cannot be classified is stored as an unknown family. The same write deletes rows for LoRAs no longer listed, unless the list is empty, so an unreadable folder does not wipe the cache.
- **Cache only.** A database error warns once and every LoRA is reported with an unknown family, so the picker falls back to a flat list.
- **Versioning.** Bump `LORA_FAMILY_VERSION` when the detection rules or the columns change; the next lookup creates the new table, drops the lower ones and re-detects. It never touches `lora_sources`.

## LoRA sources

`web/lora_sources.py` is the `lora_sources` component. It is the only record of where an installed LoRA file came from, so unlike the LoRA index it is not a cache.

- **Rows.** One row per LoRA name in `lora_sources_v1`: the file's `size_bytes` when written, `curated_id`, `repo`, `revision`, `file`, `sha256`, the `auto_trigger` with its `trigger_source` (`curated` or `card`), the `user_trigger` and `downloaded_at`. A row for a file the user only typed a trigger for has no `repo`.
- **Hugging Face downloads** (not curated ones) also keep the repository's `author`, `downloads` and `licence`, and its first preview image, shrunk to a 384-pixel WebP (`utils/lora_preview.py`) in `lora_images_v1`. The Installed tab shows these offline. Fetching the image is best effort and never fails the download; `forget` removes the image with the row.
- **Written by** the lane item's finalize step after a download, `PUT /api/loras/{name}/trigger` and `DELETE /api/loras/{name}`. See [Model downloads](downloads.md#lora-downloads).
- **Trusted only while the size matches.** `web/lora_details.describe_installed` ignores a row whose `size_bytes` differs from the file's current size, so a file replaced by hand is not described by an old row.
- **Order of trust for a trigger** (`resolve_trigger`): the trigger the user typed, the curated entry's (a file is a curated entry when its name and size match), the model card's `instance_prompt` saved at download, then a header `ss_tag_frequency` that holds a single caption. The file name and `ss_output_name` are never a trigger.
