# Projects: a folder per piece of work

**Status:** Proposed (2026-11-05)

## Idea

Pick a project, and everything you make goes into a folder of that name in the output folder: images, video clips, audio takes and anything made from them. The gallery shows one project, the files outside any project, or everything. With no project picked, nothing changes: outputs land in the output folder itself, beside the project folders.

```
outputs/
├── feature-reel/        ← project "feature-reel"
│   ├── 20261105_101500_ab12.png
│   └── 20261105_102210_cd34.mp4
├── music-video/         ← project "music-video"
│   └── 20261105_110002_ef56.flac
└── 20261104_093000_aa11.png   ← no project
```

## What the user sees and does

### Web UI

1. **A project picker in the top bar**, beside Image · Video · Extend Video · Audio, so it applies to every tab. It lists **No project**, then each project, then **New project…**, which asks for a name and makes the folder. The pick is remembered in the browser, like other tab settings.
2. **Every job saves into the picked project.** The project is taken when the job is submitted, so changing it while jobs wait in the queue doesn't move them.
3. **Made-from outputs stay with their source.** Upscales already save next to the file they came from; Extend Video, and later grabbed frames and variations, do the same, so a project's work stays together whatever project is picked when you make it.
4. **The gallery filters by project.** A Project filter beside the media filter: **All**, **No project** (only files at the top of the output folder) and each project. It opens on the picked project. The Workspace gallery strip shows the picked project too, so the strip matches what you're working on.
5. **Where a file is.** The viewer's details show the project an output belongs to.

Layouts:

- **Desktop:** the picker is a labelled menu button in the top bar showing the project name, or "No project".
- **Tablet:** the same button; a long name is cut short with an ellipsis and shown in full in the menu.
- **Phone:** a folder icon button in the top bar (with a dot when a project is picked) that opens the same list as a sheet. The gallery's Project filter sits with the other filters in its filter row or sheet.

### CLI

`ziv image`, `ziv video` and `ziv audio` take `--project NAME`. Outputs go to `<output folder>/NAME`, where the output folder is `-o` when given and otherwise the Web UI's output folder, so CLI work shows up in the gallery under the same project. The folder is made if it doesn't exist. Without `--project`, `-o` (default: the current folder) works as it does today.

## How it works

- **A project is a folder, nothing more.** The project list is the top-level folders of the output folder; there is no registry to keep in sync. Hidden folders (`.web_uploads`) and staging folders are not projects. Folders people already made there (for example with `-o outputs/tests`) show up as projects straight away.
- **Names.** A name becomes the folder name as typed, trimmed; names with path separators, `..`, a leading dot, or characters Windows forbids in file names are refused with a message. Names compare without regard to case, so `Feature Reel` and `feature reel` can't both exist on macOS and Windows.
- **Gallery.** The gallery already walks subfolders, so nothing needs reindexing. An asset's project is the first part of its path under the output folder; the gallery API takes a `project` parameter (`all`, `none` or a name) and filters on it. Files deeper in a project folder belong to that project.
- **Jobs.** The generate, upscale and extend requests carry the project; the server checks the name and resolves the folder under the output root, as it does for other output paths.

## Not in this proposal

- Renaming, deleting or merging projects in the UI. A folder renamed or removed on disk is simply reflected the next time the list is read.
- Moving outputs between projects from the gallery. Worth adding later as a gallery action if people ask for it.
- Per-project settings (default model, prompt files, presets). A project is a place for files, not a profile.

## Open

- Whether the gallery should open on **All** instead of the picked project. The picked project seems the more useful default while you work; All is one click away.
