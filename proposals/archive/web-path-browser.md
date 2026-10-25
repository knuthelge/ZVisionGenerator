# An in-app path browser

**Status:** Done (v0.13.0, 2026-10-15)

## Problem

Every **Browse** button in the Web UI (output folder, prompt file and its folder, checkpoint, LoRA, source video; the reference image had only a browser upload) calls `POST /api/picker`. That opens a dialog **on the machine running the server**:

- **Linux:** a `tkinter` `filedialog` in a child process. It is the old Tk dialog and looks nothing like GTK or KDE. Without `tkinter` or a display it reports "unsupported".
- **macOS:** `osascript` `choose file`. Native, but only on the Mac itself.
- **Any other device:** with `--host` the UI is reachable from a phone or laptop on the network. Browse there opens the dialog on the host's screen, or nowhere on a headless host.

No native dialog can fix the last point: the dialog has to be drawn by the browser. A browser `<input type="file">` doesn't help either, since it never reveals the host path and the app needs paths (multi-GB checkpoints, output folders), not uploads.

## Decision

Replace the native picker with an in-app browser, drawn by the SPA from a directory listing the server provides. It looks the same on every OS and works from any device. Static mockups of the candidate layouts were made during design (not committed); the chosen layout is a list with a places sidebar (option A) plus path completion in the field itself (option C). A Finder-style column view with a preview pane (option B) was rejected because it needs about 900 px and a second phone design.

Browsing is **not** restricted to some folders. The server only listens on loopback unless started with `--host`, anyone who can reach it can already type any path into these fields, and the listing returns names, sizes and dates, never file contents. A startup warning or root allowlist can follow if the server ever listens on the network by default.

## Design

### API

| Route | Purpose |
|---|---|
| `GET /api/browse?purpose=&path=&hidden=&all=` | List one folder for a purpose. An empty `path` starts at the purpose's own place; a file path lists its folder and names the file as `focus`. Returns `path`, `parent`, `separator`, `focus`, `entries` (`name`, `path`, `kind` `directory`/`file`, `size`, `modified`, `selectable`) and `hidden_count`. Errors: 404 missing, 403 permission denied, 422 unknown purpose or not a folder |
| `GET /api/browse/places?purpose=` | The purpose (label, folder or file, extensions) and the sidebar: Home, the output folder, the models and LoRA folders and, on Windows, each drive. The purpose's own place is flagged `default` |
| `POST /api/browse/folder` `{path, name}` | Create a folder inside `path`, for the folder purposes (New folder). One plain name, no separators |

The purposes stay server-owned (`output_directory`, `prompt_file`, `prompt_file_folder`, `checkpoint_file`, `lora_file`, `source_video`, `reference_image`), each with a kind and an extension list. Files that don't match are counted in `hidden_count` unless `all=true` asks for them (they come back with `selectable: false`). Dotfiles are left out unless `hidden=true`. Folders sort first, then case-insensitive names.

`POST /api/picker`, the Tk and `osascript` code, the picker lock and the `unsupported` status are removed. The prompt-file contract's `trust_boundary.picker` becomes `in_app_browser`.

### Dialog (option A)

- Header: title per purpose ("Choose LoRA file"), the accepted extensions as a chip.
- Toolbar: Back, Up, a breadcrumb that turns into a text box (pencil, <kbd>/</kbd> or <kbd>Ctrl L</kbd>), a filter box, and New folder in folder mode.
- Sidebar: places, then recent paths for this purpose (the last five chosen, kept per purpose in `localStorage`).
- List: name, size, modified. Files that can't be chosen are dimmed. A "N other files hidden · Show all" line when filtering hid any; an empty folder says what was hidden.
- Footer: a Hidden files switch (remembered), the selected name, Cancel and the main action: **Choose file**, or **Use "folder"** in folder mode (the highlighted folder, else the current one).
- Keyboard: arrows move, Enter opens a folder or chooses a file, Backspace goes up, typing a letter jumps, Esc cancels (or leaves the path box). Double-click chooses.
- Opens at the field's value (or its folder), else the purpose's place, else Home. Errors show in place with a way back to the last good folder.
- Below 640 px it fills the screen: places become a row of chips, rows grow to 48 px and the main action sits at the bottom.

### Completion in the field (option C)

The path box in `PathField` and in the Source video section completes as you type. It lists the typed folder (the text up to the last separator) and filters by the rest; folders and matching files show in a popover. Arrow keys move, Tab or Enter on a folder completes it with a trailing separator, Enter on a file takes it, Esc closes. `↓` in an empty box shows recent paths and places; the list never opens on focus alone, so it doesn't cover the form.

## Implementation

- `web/path_browser.py` replaces `web/path_picker.py`: the purposes table, `list_directory`, `browse_places`, `create_folder`, all pure apart from the filesystem reads. Routes in `server.py`.
- Frontend: `lib/api/browse.ts`; `lib/state/pathBrowser.ts` for pure helpers (breadcrumbs, completion split and filter, recent paths); `molecules/PathBrowser.svelte` (the dialog) and `molecules/PathInput.svelte` (the box with completion). `PathField`, `SourceVideoSection` and the reference image in `SettingsPane` use both; the reference image keeps its upload button for a picture on the viewing device.
- Tests for the listing rules, the routes and the frontend helpers and components; the native-picker tests go.
- `docs/` Web UI page and `CHANGELOG.md`.
