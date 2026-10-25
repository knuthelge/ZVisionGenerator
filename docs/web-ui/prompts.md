# Prompts page

The **Prompts** page (++g++ ++p++) builds [prompt files](../concepts/prompts.md) without typing YAML. It reads and writes the same files the CLI uses, and keeps their comments, quoting and `|` blocks.

![The Prompts page: snippets on the left, sets of prompts in the middle, the preview on the right](../assets/screenshots/placeholder.svg "prompts-page")

## Open or create a file

Use the file menu to open a prompt file on the server's machine, or **New file…** to create an empty one in a folder you choose.

A file that isn't valid YAML, or isn't shaped like a prompt file, opens in a repair view with the error and the raw text to fix.

## The three columns

### Snippets

Snippets are reusable pieces of text you put in prompts as `$name`.

- Each is listed with how often it's used; unused ones are dimmed.
- Click one to edit it, **＋** to add one.
- Drag one onto a prompt to add a `$reference`.
- Renaming a snippet updates every reference to it.

### Sets and prompts

Sets are panels of prompts, one row each. A set's name starts the file names of what it makes.

- Rename a set in place, and add prompts with **＋**.
- A set's **⋯** menu duplicates, moves, turns all its prompts on or off, or deletes it.
- Drag a set or a prompt by its ⠿ grip to reorder it; drop a prompt on another set's header to move it there.

Each prompt has:

- an **active** switch: inactive prompts are skipped when the file runs;
- ✨ **Enhance**: off, the default rewrite, or chosen [enhance options](../concepts/prompt-enhancer.md#options);
- a **⋯** menu to add or remove a negative prompt, and to duplicate, move or delete it.

**Text / Fields** switches a prompt between plain text and named fields (a structured prompt).

### Preview

The preview shows the selected prompt as the model receives it, with snippets filled in and fields flattened, plus its enhancement and any problems. **🎲 Roll** picks one option of every random choice, as a run would.

## Edit a prompt

Click the text, or press ++enter++ on it, to edit. In the text:

- `$snippets` show as chips, red when undefined. Typing `$` suggests snippet names.
- `{a|b}` random choices show as pills. Click one to edit its options, one per line; an empty line is an empty option.

While typing, ++alt+enter++ works on random choices:

- with the caret in a choice, it edits that choice;
- with words selected, it turns them into a choice;
- anywhere else, it starts a new choice at the caret.

The options box takes focus; ++ctrl+enter++ applies and ++esc++ cancels. ++esc++ or ++ctrl+enter++ also finishes editing the prompt.

## Generate from here

- **▶** on a prompt saves the file, then queues that one prompt with the current Workspace settings.
- **Use in Workspace** saves the file and selects it in the Workspace as the prompt source.

## Save

Press **Save** or ++ctrl+s++.

- Saving is blocked while the file has errors that would stop it from loading: an undefined snippet in an active prompt, an empty prompt, or two sets with the same name. The Save button says what to fix. Warnings, such as an invalid `enhance:` block, don't block saving.
- If the file changed on disk since you opened it, for example in another editor, saving asks before overwriting it.
- Prompts selected in the Workspace keep pointing at the same entries, even after you reorder, move or rename them.

++ctrl+z++ undoes changes back to the last save. Unsaved changes are kept in the browser, so leaving the page or reloading doesn't lose them; reopening the file offers to restore them.

## Save a Workspace prompt

You can also add a prompt to a file from the Workspace. **Save…** in the prompt box's tools row appends the prompt you are looking at to a prompt file. It shows when the prompt source is **Inline**.

- **What is saved:** the text in the box. In the **Prompt** tab (**Your text** for video) that's your own text; in the **Enhanced** tab it's the rewrite. Video also saves the scene, and the negative prompt is saved when it's shown.
- **Where it goes:** the end of the set named `prompts`, added at the end of the file when missing. Choose a **Recent** file, **Browse…** for one, or **New file…**.
- **The file:** comments are kept. A file with errors, such as an undefined `$snippet`, is left unchanged and the message says what to fix.

The confirmation has **Open in Prompts**, which opens the file here.
