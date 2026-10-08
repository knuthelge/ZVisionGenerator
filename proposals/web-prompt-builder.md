# Prompt builder in the Web UI

**Status:** Done (v0.13.0b14, 2026-10-08)

## Problem

The Web UI edits prompt files in a plain `<textarea>` inside a small modal (`PromptFileEditorDialog.svelte`). You can't indent with Tab, there are no line numbers, and errors appear only on save, one at a time and without a line number. Warnings such as a duplicate set name or a bad `enhance:` block are never shown. Saving closes the dialog. Nothing explains what the format means: snippets, structured prompts, `{a|b}` choices and `enhance:` are invisible.

A better text editor would fix the typing but not the last point. Prompt files have a simple structure, so the Web UI should let you build that structure directly instead of typing YAML.

## Proposed change

Add a **Prompts** page: a visual builder for prompt files. You define snippets, then build sets of prompt entries, and see each prompt as the model will receive it. The page reads and writes the same YAML file, so the CLI and hand-edited files keep working. It has no raw-YAML view.

A clickable mockup was made during design (`.agent-work/prompt-builder/index.html`, not committed) and reviewed before this write-up.

### Layout

| Area | Contents |
|---|---|
| File bar | File switcher (recent files, Browse…), **New file**, unsaved-changes dot, save status, **Use in Workspace**, **Save** (⌘S) |
| Snippets (left) | Each snippet with its name, a preview of its value and how many times it is used (unused ones are dimmed). Click to edit, **＋** to add. |
| Sets (centre) | Collapsible sets you can rename, each with "N of M active" and a menu. Entry cards inside. **＋ New set** at the end. |
| Preview (right) | The selected entry with snippets resolved and structure flattened, its prompt id, its enhance settings, its problems, and **🎲 Roll**. |

### Entry cards

Each set is a panel with a header bar; each prompt is one compact row: drag grip, ordinal and **active** switch on the left; the prompt (and its negative, if any) in the middle; **Text / Fields**, ✨ enhance, **▶** (Generate this one) and a menu (Negative, Duplicate, Move up/down, Move to set, Delete) on the right. Every control uses one square button style (`.panel-button`).

- **Prompt text** is shown with `$snippet` references as chips (red when undefined) and `{a|b|c}` choices as pills, with `$refs` inside choices as chips too. Click the text, or press Enter on it, to edit it as plain text. Typing `$` suggests snippet names. The preview updates as you type.
- **Choice pills** open a small editor with one option per line. An empty line is an empty option, so `{|very }` stays as written.
- **Fields** mode edits a structured prompt as key–value rows (`Subject`, `Setting`, `Style`); each value is a prompt text. Switching from Text to Fields puts the text in one unnamed field (no default names; an unnamed field is an error until named). Switching back flattens the fields into the same text the model already received.
- **Structured values the builder can't edit** (lists, nested mappings, non-string values) are shown read-only with their flattened preview and kept exactly as written. **Convert to text** replaces one with its flattened text after asking.
- **Enhance** has full support: **Off**, **Default options** (`enhance: true`), or chosen options on every axis: style, mood, details (any), length, and motion (any, video only). The picker is the same one the Workspace uses, and it refuses a choice that asks for no change, like the server does.
- **▶ Generate this one** saves the file if needed, then queues that one prompt with the current Workspace settings, without changing the prompts selected in the Workspace. It is disabled on inactive entries.

### Arranging

- Drag a set's grip to reorder sets.
- Drag an entry's grip to reorder it, move it into another set, or drop it on a set header to add it at the end (this works on collapsed and empty sets too).
- Drag a snippet to reorder snippets, or drop it on a card to add a `$reference`.
- The menus offer the same moves for keyboard users.

### Names

- Set names must be unique, non-empty and not `snippets`. They become output filename prefixes (made filename-safe as today), so renaming a set says so.
- Snippet names must be valid references (`[A-Za-z_][A-Za-z0-9_]*`) and unique. Renaming a snippet updates every `$reference` in the file; it is refused while a read-only structured value uses it.
- Defaults for new items (`new_set`, `snippet`) and duplicates (`portrait_copy`) get a counter when the name is taken.

### Safety

- **Undo:** ⌘Z undoes the last change (100 steps), back to the last save. Each edit to a text field is one step.
- **Confirmations:** deleting a set that has entries and deleting a snippet that is in use both ask first, through `ConfirmDialog`.
- **Unsaved work** is kept in browser storage per file and revision. Leaving the page, reloading or going back doesn't lose it, and reopening the file offers to restore it.
- **Version check:** a save carries the revision (a SHA-256 of the file) the page loaded. If the file changed on disk in the meantime, the save is refused (HTTP 409), and the page offers **Reload from disk** or **Overwrite**.
- **Save is blocked by errors** (undefined or circular snippets in active entries, empty prompts, bad names), so the file always runs in the CLI. The Save button's tooltip says what is wrong. Warnings (an invalid `enhance:` block, problems in inactive entries) do not block saving.
- **Repair:** a file that isn't valid YAML, or whose structure isn't sets of entries, opens in a repair view. It shows the error with its line, the raw text in an editable box, and **Save and open**. This is the only raw-text view.

### Workspace

- The prompt-file box's **Edit YAML** becomes **Edit**, which opens the Prompts page on that file. `PromptFileEditorDialog` is removed.
- **Use in Workspace** sets the Workspace's prompt source to this file and opens the Workspace.
- **Keeping the selection right.** Prompt ids are `set:index` (`index` counts inactive entries too), so reordering, moving, deleting and renaming change them. Each save returns a map from the old ids to the new ones, and the Workspace selection for that file is remapped, so it never silently points at a different prompt. Prompts that no longer exist, or are now inactive, drop out of the selection.

## Backend

### Comment-preserving round trip

`utils/prompt_document.py` converts between YAML text and a document model, using `ruamel.yaml` in round-trip mode (new dependency) so comments, quoting and block scalars survive. PyYAML stays the parser for meaning: every save is checked with `inspect_prompts_text`, exactly as the CLI loads it.

- **Load** (`load_prompt_document(text)`): returns snippets, sets and entries, each with a positional id (`n0`, `s0`, `s0.e1`).
  - Values become `text` (a string), `fields` (a mapping of strings to strings) or `structured` (anything else, carried as plain data).
  - `active` follows YAML 1.1 truthiness (`no`, `off`), like PyYAML.
  - `enhance` is `None`, `True`, or the mapping as written.
  - A file that isn't a mapping, a set that isn't a list, or an entry that isn't a mapping raises `ValueError`; the page then opens the repair view.
- **Render** (`render_prompt_document(original_text, document)`): applies the document onto the original text.
  - Ids that came from the original reuse its nodes, together with their comments, key order and unknown keys. Only keys that changed are written.
  - New entries are written as `prompt`, `negative`, `active` (only when `false`), `enhance`.
  - Multi-line text is written as a `|` block.
  - `enhance` mappings only write the axes that differ from the defaults.
  - Indentation is normalised to the style of the bundled `prompts.yaml` (2-space mappings, sequences indented under their key). Booleans are written as `true` and `false`.
  - `snippets` is written first.
- **Comments** stay with the item they were attached to. A comment between two items belongs to the item before it, so moving that item moves the comment.

### Routes

| Route | Body | Returns |
|---|---|---|
| `POST /api/prompt-files/document` | `{path}` | `{path, revision, raw_text, enhance_matrix, document}`, or `problem` instead of `document` when the file can't be built (for the repair view) |
| `PUT /api/prompt-files/document` | `{path, revision, document, force?, base_text?}` | `{path, revision, raw_text, document, ids, option_id_map, warnings}`. 409 when the revision is stale and `force` is not set; 422 when validation fails. `force` applies the edits onto `base_text` (the text that was loaded), so edits always land on the version they were made on |
| `POST /api/prompt-files/preview` | `{document, roll_entry_id?}` | `{entries: {id: {prompt, negative}}, problems: [{id, severity, message}], snippet_uses: {id: n}, rolled?}` |
| `POST /api/prompt-files/create` | `{directory, name}` | `{path}`. Creates an empty `<name>.yaml`. Refuses an existing file, a missing folder, or a name with a path separator |

- **Preview** resolves snippets and flattens structure with the same functions the runners use (`resolve_snippets`, `flatten_value`), and rolls choices with `expand_random_choices`. Nothing is read from disk.
- **Problem severity:**
  - Errors: undefined or circular snippets in active entries, empty prompts, and bad set or snippet names.
  - Warnings: an invalid or empty `enhance:` block, and the same resolution problems in inactive entries, which the CLI skips.
- The existing read, write and inspect routes stay. The repair view saves through `PUT /api/prompt-files/write`.
- **File access:** `PROMPT_FILE_CONTRACT.trust_boundary.read_write` becomes `yaml_files_only`. Files are still only on the host; create only makes new `.yaml` files in an existing folder (new picker purpose `prompt_file_folder`).

## Alternatives considered

- **A code editor (CodeMirror 6)** with live diagnostics and completion. It would fix typing, but not understanding the file, and a visual builder makes the format discoverable. Rejected, as was a raw-YAML tab next to the builder.
- **A YAML library in the browser** (`yaml` npm) for the round trip. That would put a second YAML implementation next to PyYAML, in another language. Keeping the format in Python leaves one source of truth. Rejected.
- **Re-dumping the whole file from the model** (no comment preservation). It is simpler, but it would erase comments in hand-written files. Rejected.
- **Generate this one by switching the Workspace selection.** That would quietly change what the next Generate runs. Rejected: the Workspace submits its form with the prompt-file fields pointed at that one prompt, and the selection is never touched.

## Testing

- **`prompt_document`:** load and render round trips that keep comments, quotes and block scalars. Also: reorder and move entries, rename sets and snippets, add and remove keys, `enhance` written minimally, YAML 1.1 `active` values, structured values kept, and invalid structure rejected.
- **Routes:** document load, the repair payload, save with a revision conflict, `force`, the option-id map, validation errors, preview problems and roll, and create.
- **Frontend:** Vitest for the pure helpers (tokenising prompts, moves, unique names, name checks, enhance summaries, id remapping, the undo history). Component tests for the page: load, edit, save, 409, repair, and Edit in the Workspace.
