# Projects: work on one chunk at a time

**Status:** Done (2026-11-08, v0.14.0; built on `feat/projects`). Checked with real image jobs through the live queue on Linux, and on a 32 GB M5 on 2026-11-08 with real image, video and audio jobs in Chromium and WebKit, including the phone layout; Windows was reviewed in code only.

## Problem

All work in the app is one chronological stream. There is no way to say "I'm working on this now", see only the files that belong to it, and have new generations join it. When a new piece of work starts, it mixes with everything before it.

A project groups a chunk of work. Pick one and the app shows and saves only that project. Starting a new one takes a name and nothing else.

## What the user sees and does

### Web UI

1. **One project picker, on every page.** It lists **All files**, then each project, then **New project…**, which asks for a name, makes the project and switches to it. The pick is remembered per browser, like other tab settings, so a phone and a desktop can be in different projects.
2. **With a project picked, the app is that project.** The Gallery page, the Workspace strip and stage, and the "latest image" shortcut show only its files. The file picker opens in its folder, and can still go elsewhere to bring in a reference.
3. **New generations save into it.** A job keeps the project it was submitted with, so changing the pick while jobs wait doesn't move them. The job card and queue stay global, but a result from another project doesn't appear on the stage or in the strip. Reuse and Load settings fill in the form and never change the pick.
4. **All files works as the app does today.** It shows everything and saves to the top of the output folder, and each tile shows its project. That is also how to look across projects. "No project" only describes a file that isn't in a project, in Move and in the viewer's details.
5. **Made-from outputs stay with their source.** Upscales already save next to the file they came from. Extend does the same when its source is in the output folder, and otherwise saves to the picked project.
6. **Move to project.** On one file from its **⋯** menu wherever that appears (gallery, viewer, Workspace strip), from a Move button in the desktop viewer's toolbar, and on a gallery selection. It lists the projects, **No project** and **New project…**. This sorts existing files into projects and fixes anything saved in the wrong one.
7. **Where a file is.** The viewer's details show a file's project, which matters when no project is picked.

The queue, job log, recent prompts, recent LoRAs, last seed and uploaded references stay global. When a result lands outside the picked project, a toast says where it went.

Layouts:

- **Desktop:** a labelled menu button right after the logo, before Image, Video and Audio, showing the project name or "All files". It holds the same place on every page.
- **Tablet:** the same button; a long name is cut short with an ellipsis and shown in full in the menu.
- **Phone:** a **Project** row at the top of the drawer, opening the same list as a sheet. The top bar doesn't change. With a project picked, Generate shows its name as a second line. Move to project opens the same kind of sheet.

### CLI

No change. `ziv image -o <output folder>/name` already saves into a project, and it shows up in the Web UI.

## How it works

- **A project is a folder directly under the output folder.** The project list is those folders, so there is no registry to keep in sync and an empty new project just works. Staging folders (`.web_uploads` and the like) are not projects. Folders already there become projects straight away, and files deeper inside one belong to it.
- **Why folders and not labels in the database.** A label would only add putting one file in two projects, which isn't wanted. It would also make the database the only record of the grouping, and moving a file by hand would lose it.
- **Names.** A name becomes the folder name as typed, trimmed. Names with path separators, a leading or trailing dot, or characters or names Windows forbids are refused with a message. Names compare without regard to case, so `Feature Reel` and `feature reel` can't both exist.
- **Gallery.** An asset's project is the first part of its ID under the output folder. `/api/gallery` takes a project and filters on it, in memory, after the scan they already do.
- **One gallery store.** The strip, stage and "latest image" read the store's default view, so the picked project becomes part of the store's query. `mergeOutputs` adds a finished job's outputs only to views whose project matches.
- **Jobs.** Generate and Extend requests carry the project. The server checks the name and resolves the folder under the output root. Upscales need nothing, since they always save next to their source. Copy CLI shows the project's folder as `-o`. Uploaded references stay at the top of the output folder.
- **Move.** A rename within the output folder, so the modified time and the gallery order don't change. A file moves to the top of the target project, without any sub-folder it was in. A name already taken in the target gets a unique one, and nothing is overwritten. A file's gallery ID is its path, so the store replaces the moved assets' IDs.
- **Source links after a move.** Upscales and extensions record their source's path in the file. After a move, the link shows as a missing source, as it does today when a source is deleted, even if both files moved together. The files themselves are not rewritten.
- **A moved file in a Workspace slot.** The slot follows the file to its new path; a deleted file empties it.

## Not in this proposal

- Renaming or deleting projects in the UI. A folder renamed or removed on disk is reflected the next time the list is read.
- Per-project settings, prompts or job history. A project groups files.
- A `--project` CLI flag.
