# Prompts Guide

Z-Vision Generator supports inline prompts, YAML prompt files with batch generation, prompt variables, structured prompts, and reusable snippets. This guide covers the full prompts system used by both `ziv-image` and `ziv-video`.

## Inline Prompts

Use `--prompt` for quick, one-off generation:

```bash
ziv-image -m my-model --prompt "a beautiful sunset over the ocean"
ziv-video -m ltx-4 --prompt "A cat walking through a garden"
```

When `--prompt` is provided, it overrides `--prompts-file`.

## Prompt Files

Use `--prompts-file` (or `-p`) to load prompts from a YAML file. The default file is `prompts.yaml`.

```bash
ziv-image -m my-model -p prompts.yaml -r 3
ziv-image -m my-model -p my-prompts.yaml
```

Each entry has a set name and a list of prompt objects:

```yaml
gorilla:
  - active: False
    prompt: |
      A gorilla. Detailed fur.
      In a jungle. Rainy day. Moody lighting.

woman:
  - active: True
    prompt: |
      30yo woman, in red dress.
      Walking down a city street. Evening.
```

- `active: True` (default) — prompt is generated. `active: False` — skipped.
- Multiple prompts per set are supported.
- The set name becomes the output filename prefix.

In the Web UI, choose **Prompt file** as the prompt source and type or browse to a YAML file on the machine running the server. Compose then summarizes the selected prompts; click **Prompts…** (or the summary) to open the chooser, where you can filter prompts, select whole sets with **All**/**None**, expand a prompt to see its negative prompt, and confirm with **Use N prompts**. **✨ Enhance** sets the options for **Enhance each image when generating**, and **Edit** opens the file on the [Prompts page](#prompt-builder-web-ui). If you edit the path manually, the visible value is the value submitted; use Enter or Browse when you want the UI to inspect the file and refresh the option list before generating.

## Prompt Builder (Web UI)

The Web UI's **Prompts** page (`G` then `P`) builds prompt files without typing YAML. It reads and writes the same file the CLI uses, and keeps the file's comments, quoting and `|` blocks.

- **Snippets** (left) are listed with how often each is used; unused ones are dimmed. Click one to edit it, **＋** to add one, and drag one onto a prompt to add a `$reference`. Renaming a snippet updates every reference to it.
- **Sets** (centre) are panels of prompts, one row each. Rename a set in place (the name starts the output filenames), add prompts with **＋**, and use a set's **⋯** menu to duplicate, move, turn all prompts on or off, or delete it.
- **Entries** show `$snippets` as chips (red when undefined) and `{a|b}` choices as pills. Click the text, or press `↵` on it, to edit; typing `$` suggests snippet names. Click a pill to edit its options, one per line; an empty line is an empty option. While typing, put the caret in a choice and press `⌥↵` (or **Edit choice** under the text) to edit it the same way, select words and press `⌥↵` (**Make a random choice**) to turn them into a choice, or press `⌥↵` anywhere else (**New random choice**) to type the options of a new choice at the caret, one per line. The options box takes focus; `⌘↵` applies and `Esc` cancels, both returning you to the prompt. Each prompt has an **active** switch and **✨ Enhance**: off, the default options (`enhance: true`), or chosen style, mood, details, length and motion (motion applies to video only). Its **⋯** menu adds or removes a negative prompt, and duplicates, moves or deletes the prompt.
- **Text / Fields** switches a prompt between plain text and named fields (a structured prompt). Switching to fields puts the text in one field for you to name; a field without a name is a problem until you name it. Lists and nested values that the builder can't edit are shown flattened and kept exactly as written; **Convert to text** replaces one with its flattened text.
- **Drag** a set or an entry by its ⠿ grip to reorder it; drop an entry on another set's header to move it there. The menus offer the same moves.
- **Preview** (right) shows the selected entry as the model receives it, with snippets resolved and fields flattened, its prompt id, its enhancement and its problems. **🎲 Roll** picks one option of every choice, as a run would.
- **▶** (Generate this one) saves the file, then queues that one prompt with the current Workspace settings. Your prompt selection in the Workspace doesn't change. **Use in Workspace** saves and selects the file there.

Saving (`⌘S`) is blocked while the file has errors that would stop it from loading (an undefined snippet in an active entry, an empty prompt, two sets with the same name); the Save button says what to fix. Warnings, such as an invalid `enhance:` block, don't block saving. Saving keeps the prompts selected in the Workspace pointing at the same entries, even when you reorder, move or rename them.

`⌘Z` undoes changes back to the last save. Unsaved changes are kept in the browser, so leaving the page or reloading doesn't lose them; reopening the file offers to restore them. If the file changed on disk since you opened it (for example in another editor), saving asks before overwriting it.

A file that isn't valid YAML, or isn't shaped like a prompt file, opens in a repair view with the error and the raw text to fix. **New file…** in the file menu creates an empty prompt file in a folder you choose.

## Prompt Variables

Use `{option1|option2|option3}` syntax for random selection each run:

```
"A {red|blue} car"                         → "A red car" or "A blue car"
"A {big|small} {red|blue} {car|truck}"     → random combo each run
"{Nikon Z9 {50mm|35mm}|Canon EOS 5D}"     → nesting resolves inside-out
```

Variables are resolved independently each run, so repeated runs (`-r 3`) produce different combinations. The `{a|b|c}` random choice syntax works within structured prompt values too.

Pass a JSON object as the value of `--json-prompt` (mutually exclusive with `--prompt`) to provide a literal structured JSON caption inline. The value opts out of `{a|b|c}` expansion and is sent verbatim (used for Ideogram 4). It must be a valid JSON object, or generation is rejected before the model loads.

## Structured Prompts

The `prompt` (and `negative`) field accepts dicts, lists, and nested combinations — not just strings. Structured values are flattened into a single prompt string with `". "` as separator. Dict keys become prefixes; list items are joined.

```yaml
diner_scene:
  - prompt:
      Subjects:
        - Lisa: An old lady with grey hair and kind eyes
        - George is a retired professor with round glasses
        - Nina:
            Hair: Tidy
            Clothes: Red dress
      Setting: Lisa, Nina and George are sharing a meal at the diner
      Style:
        Camera: iPhone
        Color: Warm
```

This flattens to:

```
Subjects: Lisa: An old lady with grey hair and kind eyes. George is a retired professor
with round glasses. Nina: Hair: Tidy. Clothes: Red dress. Setting: Lisa, Nina and George
are sharing a meal at the diner. Style: Camera: iPhone. Color: Warm
```

Prompts can be plain strings, dicts, lists, or nested combinations.

### Flattening Rules

- **Strings** are used as-is.
- **Lists** have each item flattened and joined with `". "`.
- **Dicts** produce `"key: value"` pairs joined with `". "`. Nested dicts/lists are recursively flattened.

## Snippets

Define reusable prompt fragments under a reserved `snippets` top-level key in your prompts file:

```yaml
snippets:
  nina:
    Hair: Tidy
    Clothes: Red dress
  warm_style:
    Camera: iPhone
    Color: Warm
  diner: a cozy 1950s American diner with checkered floors
```

Reference a snippet with `$name`:

```yaml
diner_scene:
  - prompt:
      Subjects:
        - Nina: $nina
      Setting: Lisa and Nina are sharing a meal at $diner
      Style: $warm_style
```

- **Standalone** `$nina` (the entire value) preserves structure — the dict is kept as-is for flattening.
- **Inline** `$diner` within a string is flattened and substituted in place.
- Snippets can reference other snippets.
- Circular references are detected and raise an error.

The `snippets` key is not treated as a prompt set — it is consumed during loading and removed.

## Negative Prompts

The `negative` field in a prompt object specifies what the model should avoid:

```yaml
portrait:
  - prompt: "A woman in a garden"
    negative: "blurry, low quality, bad anatomy"
```

> **Note:** FLUX.2 models do not support negative prompts. If a negative prompt is provided with a FLUX.2 model, a warning is issued and the negative prompt is ignored.

The `negative` field supports the same structured format as `prompt` (strings, dicts, lists).

## Enhancing Prompts

A small local LLM can rewrite a prompt with more visual detail before it is generated. It runs on your machine, and its default model is decensored, so it does not water down your prompt.

| Platform | Default model | First-use download |
|---|---|---|
| macOS | `McG-221/Qwen3.5-4B-heretic-mlx-4Bit` (mlx-lm, 4-bit) | ≈2.4 GB |
| Windows / Linux | `coder3101/Qwen3.5-4B-heretic` (transformers, 4-bit on CUDA) | ≈9 GB |

Without a CUDA GPU, Windows and Linux run the enhancer on the CPU: it works, but each rewrite can take a few minutes and about 10 GB of RAM. The status line says so when this happens. If the machine has an NVIDIA GPU, a warning also suggests updating its driver to 580 or newer, the usual reason CUDA is missing.

Both defaults are pinned to a revision. To use another chat model, set **Prompt Enhancer Model** on the Config page to a Hugging Face `owner/name` (optionally `owner/name@revision`) or a local folder, or pass `--enhance-model REPO[@REVISION]`. Thinking-tuned or heavily merged models are slower and follow the length and style options less reliably.

### Options

| Option | Choose | Values |
|---|---|---|
| Style | one | Keep, Photographic, Candid, Street photography, Analog film, Black & white, Studio portrait, Product shot, Cinematic, Illustration, Anime, Comic, 3D render, Painterly |
| Mood | one | Keep, Serene, Joyful, Romantic, Melancholic, Mysterious, Eerie, Dramatic, Epic, Whimsical, Nostalgic |
| Details | any | Lighting, Composition, Camera & lens, Materials & textures, Color & mood, Environment, Subject |
| Length | one | Shorter (≈50 %), Same, Longer (≈200 %), Extra long (≈300 %) |
| Motion (video only) | any | Action / sequence, Camera movement, Pacing |

Mood sets the feeling of the scene. The enhancer conveys it through light, color, setting and expression instead of naming it. The **Color & mood** detail only adds more description of color and mood; it does not pick one.

Length is a target, not a guarantee. Longer aims for at least 40 words and Extra long for at least 80, so short prompts still grow; Shorter never aims below 12 words, so a prompt of 12 words or fewer keeps about the same length. Results are capped at 300 words (180 for FLUX.1, whose text encoder reads fewer tokens); when a prompt is already at the cap, it keeps about the same length. On short prompts, Same is approximate.

The enhancer keeps your subject and every detail you describe. It adds only things that can be seen: no sounds, smells, or story. It never adds new people, animals, or major objects.

### In the Web UI

- **Enhance**: click ✨ **Enhance** in the prompt box, pick options, and click **Enhance prompt**. The rewrite appears in the **Enhanced** tab. When that tab has text, it is what gets generated; the **used** badge shows which tab that is. Edit it, enhance again, or **Clear** it to go back to your prompt. The button is unavailable while generation jobs are running or queued.
- If you change the prompt or switch between image and video afterwards, the Enhanced prompt is marked **Out of date**. It is still used until you **Re-enhance** or **Clear** it.
- `{a|b}` choices are picked before the rewrite, so the enhancer sees one plain prompt and the Enhanced prompt has no choices left. Enhance again for a different pick, or use **Enhance each image when generating** to keep a fresh pick per image.
- **Enhance each image when generating** rewrites every image's prompt on the server, after its `{a|b}` choices are picked, using that image's seed. It works with inline prompts and prompt files. All rewrites run before the model loads, so the first image starts once every prompt is rewritten; the progress panel shows *Enhancing prompt N of M*. A prompt that could not be enhanced, or was skipped, is marked in the panel and generated from the original prompt.
- Enhance is unavailable while a job runs.

### In Prompt Files and the CLI

Add `enhance:` to an entry to enhance it on every run:

```yaml
woman:
  - prompt: 30yo woman in red dress walking down a city street, evening
    enhance:
      style: cinematic
      mood: mysterious
      details: [lighting, camera]
      length: longer
  - prompt: a quiet street at dawn
    enhance: true        # default options
```

From the command line, `--enhance` enhances every prompt and overrides entries' `enhance:` settings; `--no-enhance` turns all enhancement off:

```bash
ziv-image -m klein9b --prompt "a fox in snow" --enhance
ziv-image -m klein9b -p prompts.yaml --enhance style=photo,mood=serene,details=lighting+camera,length=longer
ziv-video -m ltx-8 --prompt "a fox runs" --enhance style=cinematic,motion=action+camera-move
ziv-image -m klein9b -p prompts.yaml --no-enhance
```

`motion` only applies to video. Prompt files shared between image and video may include it; images ignore it.

The enhanced prompt is printed while prompts are enhanced and again before each generation, and stored in the image or video metadata, so Gallery shows and reuses the prompt that actually rendered.

### While Prompts Are Enhanced

Jobs with auto enhancement rewrite every prompt before the model loads. In image jobs (Web UI controls or `ziv-image` keys):

- **Next** (`n`) skips the current rewrite; that image uses the original prompt.
- **Quit** (`q`) stops the job before the model loads.
- **Pause** (`p`) waits until you resume.
- **Repeat** (`r`) has no effect.

Video jobs have no Web UI controls; `ziv-video` can be stopped with Ctrl+C.

**Repeat** and automatic retries keep the image's `{a|b}` choices and enhanced prompt and only change the seed. The new seed is always random, even when a seed is set.

### Memory

The enhancer and the image or video model are never loaded together: the enhancer is unloaded, and its memory freed, before the model loads. Peak memory is the larger of the two, not their sum. Clicking Enhance loads the enhancer (about 3 GB on macOS) and frees it after two idle minutes, or when a job starts.

## Tips

### Combining Variables with Structured Prompts

Variables work inside structured prompt values, so you can create prompts with controlled randomness:

```yaml
character:
  - prompt:
      Subject: "A {young|old} {man|woman} with {red|blonde|dark} hair"
      Setting: "{A park|A cafe|A beach} on a {sunny|rainy} day"
      Style:
        Camera: "{Nikon Z9|Canon EOS 5D}"
        Color: "{Warm|Cool|Neutral}"
```

### Organizing Large Prompt Files

- Use descriptive set names — they become output filename prefixes.
- Use `active: False` to temporarily disable prompts without deleting them.
- Extract common elements into snippets to avoid duplication.
- Group related prompts in the same set as multiple entries.
