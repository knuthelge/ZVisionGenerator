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
- **Entries** show `$snippets` as chips (red when undefined) and `{a|b}` choices as pills. Click the text, or press `↵` on it, to edit; typing `$` suggests snippet names. Click a pill to edit its options, one per line; an empty line is an empty option. While typing, put the caret in a choice and press `⌥↵` (or **Edit choice** under the text) to edit it the same way, select words and press `⌥↵` (**Make a random choice**) to turn them into a choice, or press `⌥↵` anywhere else (**New random choice**) to type the options of a new choice at the caret, one per line. The options box takes focus; `⌘↵` applies and `Esc` cancels, both returning you to the prompt. Each prompt has an **active** switch and **✨ Enhance**: off, the default options (`enhance: true`), or chosen style, mood, details, length and motion (motion applies to video only). Its popover also has the **Rewrite** choice (Scene and action, or Action only) for video entries, saved as `parts:`. Its **⋯** menu adds or removes a negative prompt, and duplicates, moves or deletes the prompt.
- **Text / Fields** switches a prompt between plain text and named fields (a structured prompt). Switching to fields puts the text in one field for you to name; a field without a name is a problem until you name it. Lists and nested values that the builder can't edit are shown flattened and kept exactly as written; **Convert to text** replaces one with its flattened text.
- **Drag** a set or an entry by its ⠿ grip to reorder it, with a mouse or a finger; drop an entry on another set's header to move it there. A snippet drags by its row with a mouse and by its grip with a finger. The list scrolls when you drag near its top or bottom edge, and the menus offer the same moves.
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

## Scenes for Video

A video prompt can have a **scene** (who and where, look, style, camera setup) apart from its action. Add `scene:` to an entry:

```yaml
snippets:
  courtyard: A knight in a grey castle courtyard, overcast, handheld camera.

castle:
  - scene: $courtyard
    prompt: She draws her sword and turns to the gate.
  - scene: $courtyard
    prompt: She sheathes the sword and walks toward the tower.
  - prompt: A raven lands on the wall.   # no scene: uses --scene or the Workspace Default scene
```

- `scene:` belongs to an entry. To share one scene across a set, put it in a `$snippet` as above; there is no scene on the set itself.
- The video model receives the scene first, then the prompt. An entry with no `scene:` uses `--scene` (or the Workspace's Default scene) if one is given, and otherwise only its prompt.
- A `scene:` takes snippets, structured values and `{a|b}` choices like a prompt does. An entry still needs a `prompt`.
- Image jobs ignore `scene:`, so a file can serve both.
- Old prompt files work unchanged.

With `--extend`, each entry's prompt is the action that follows the source clip, and an entry's `scene:` replaces the source's scene for that entry (see [Extend a Video](video.md#extend-a-video)).

## Enhancing Prompts

A small local LLM can rewrite a prompt with more visual detail before it is generated. It runs on your machine. The default model is an uncensored variant of Qwen3.5-4B, because small instruction-tuned models often refuse or tone down ordinary creative prompts, such as a battle scene, a horror shot or a villain's portrait. The uncensored variant rewrites them faithfully instead.

There are two built-in models. **Fast** is the default. **Smart** is larger and follows your options and wording more closely (it keeps counts, adds fewer things you did not ask for, and writes better JSON captions), but needs more memory and a larger download.

| Platform | Enhancer | Model | First-use download | Needs |
|---|---|---|---|---|
| macOS | Fast | `McG-221/Qwen3.5-4B-heretic-mlx-4Bit` (mlx-lm, 4-bit) | ≈2.4 GB | |
| macOS | Smart | `TheCluster/Qwen3.5-27B-Heretic-MLX-4bit` (mlx-lm, 4-bit) | ≈16 GB | about 20 GB of GPU memory (a Mac with 32 GB) |
| Windows / Linux | Fast | `coder3101/Qwen3.5-4B-heretic` (transformers, 4-bit on CUDA) | ≈9 GB | |
| Windows / Linux | Smart | `coder3101/Qwen3.5-9B-heretic` (transformers, 4-bit on CUDA) | ≈19 GB | about 9 GB of GPU memory |

Choose **Fast** or **Smart** under **Enhancer** in the Enhance popover, or pass `--enhance-model smart` on the command line. The choice applies to the whole job. On a machine where Smart does not fit (too little GPU memory, or no NVIDIA GPU on Windows or Linux), the popover shows why and Smart cannot be picked; on the command line a warning says so and Fast runs. Downloaded models stay in the Hugging Face cache; delete them there to free the disk space.

Without a CUDA GPU, Windows and Linux run the enhancer on the CPU: it works, but each rewrite can take a few minutes and about 10 GB of RAM. The status line says so when this happens. If the machine has an NVIDIA GPU, a warning also suggests updating its driver to 580 or newer, the usual reason CUDA is missing.

All built-in models are pinned to a revision. To use another chat model, set **Prompt Enhancer Model** on the Config page to a Hugging Face `owner/name` (optionally `owner/name@revision`) or a local folder; it then replaces the Fast/Smart choice. `--enhance-model REPO[@REVISION]` does the same for one run. Thinking-tuned or heavily merged models are slower and follow the length and style options less reliably.

### Options

| Option | Choose | Values |
|---|---|---|
| Style | one | Keep, Photographic, Candid, Street photography, Analog film, Black & white, Studio portrait, Product shot, Cinematic, Illustration, Anime, Comic, 3D render, Painterly |
| Mood | one | Keep, Serene, Joyful, Romantic, Melancholic, Mysterious, Eerie, Dramatic, Epic, Whimsical, Nostalgic |
| Details | any | Lighting, Composition, Camera & lens, Materials & textures, Color & mood, Environment, Subject |
| Motion (video only) | any | Action / sequence, Camera movement, Pacing |
| Sound (video with audio on) | any | Ambience, Sound effects, Dialogue, Music |
| Length | one | Shorter (≈50 %), Same, Longer (≈200 %), Extra long (≈300 %) |
| Direction | free text | e.g. *at night, in the rain* or *focus on the dog* (up to 300 characters) |

Mood sets the feeling of the scene. The enhancer conveys it through light, color, setting and expression instead of naming it. The **Color & mood** detail only adds more description of color and mood; it does not pick one.

Length is a target, not a guarantee. Same keeps a prompt's length but grows a short one to about 25 words, since the rewrite adds the details you picked. Longer aims for at least 40 words and Extra long for at least 80; Shorter never aims below 12 words, so a prompt of 12 words or fewer keeps about the same length. Results are capped at 300 words (180 for FLUX.1, whose text encoder reads fewer tokens); when a prompt is already at the cap, it keeps about the same length.

The enhancer keeps your subject and every detail you describe. It adds only things that can be seen, no smells or story, and never adds new people, animals, or major objects.

Text in quotes, such as a sign's `"OPEN 24 HOURS"` or a line of dialogue, is copied exactly, with its case and punctuation. If a rewrite changes it, the enhancer tries once more; if that also changes it, the rewrite is used with a note saying so, so you can fix it in the Enhanced text. In a video with audio off, quoted speech may be dropped, but quoted lettering is kept.

Sound is the exception for video with audio on:

- The enhancer keeps every sound and every quoted line of speech you wrote, word for word, whether or not a Sound option is picked.
- With no Sound option picked, it adds no new sounds.
- **Ambience** and **Sound effects** describe the setting's soundscape and sounds that match the action. **Music** describes music in the scene or a score.
- **Dialogue** keeps your quoted lines. If you wrote none, it writes short lines that fit the scene. Only people already in the scene speak.
- With audio off, and for images, the enhancer describes no sounds and Sound is ignored.

**Direction** is a short instruction the enhancer follows, such as a time of day, weather, or what to focus on. A direction alone is enough to enhance, without picking other options.

The enhancer also knows the job's shape:

- **Frame:** portrait, landscape or square, from the ratio or size, so the composition fits the frame.
- **Clip length (video):** one continuous action for clips up to 5 seconds, two or three beats up to 10 seconds, so the action fits the clip.
- **Starting image:** for image-to-video, the rewrite describes motion and change instead of appearance, which the first frame already shows. For img2img it mentions the subject only briefly. The enhancer does not see the image itself.

### Scene and Action (Video)

For video, one rewrite returns two parts: a **scene** and an **action** (see [Scenes for Video](#scenes-for-video)). The options shape them differently:

| Option | Shapes |
|---|---|
| Style, Mood, Details | mostly the scene |
| Motion, Sound | the action |
| Direction | both |

- **Text to video** splits and expands, so it works when you wrote everything in the prompt or gave only a scene: the setting, look and camera setup go to the scene, and events in time order, camera moves, sound and quoted dialogue go to the action.
- **Image to video** keeps your scene (what the image shows) and writes only the action by default. With a blank scene, or with Rewrite set to Scene and action, it writes a short scene (style, and what the image does not show) and an action that describes the change.
- **Extend** writes the action that happens next, using the previous clip's prompt as context so it does not repeat its actions or spoken lines. From a clip made without a scene, the enhancer takes a scene from the old prompt. See [Extend a Video](video.md#extend-a-video).
- **Length** applies to the scene and the action together.
- If the enhancer does not return a scene and an action, it tries once more, then uses its rewrite as the action and keeps your scene, with a warning.

#### Rewrite: scene and action, or action only

The **Rewrite** option (`parts` in the CLI and prompt files) chooses what the enhancer writes:

| Rewrite | `parts` | What you get |
|---|---|---|
| Scene and action | `both` | A rewritten scene and a rewritten action. With no scene, the enhancer splits your text into both. |
| Action only | `action` | Your scene is kept exactly as written. The enhancer reads it and writes an action that fits it. |

- The default depends on the workflow: **Extend Video** rewrites the action only when the clip has a scene, so a chain of extensions keeps its look, and **Image to Video** does the same when it has a scene, so the scene that describes the starting image is not restyled. **Text to Video** rewrites both.
- Style, Mood and Details shape the scene, so they have no effect while Rewrite is Action only. Motion, Sound, Length and Direction still shape the action.
- With Action only and a scene, Enhance always rewrites the action, even when every other option is left at "keep".
- **Action only needs a scene.** With a blank scene, Action only is unavailable in the Web UI, and the run rewrites both parts. On the command line it does the same and warns once.
- Image workflows ignore Rewrite.

### JSON Captions (Ideogram 4)

Ideogram 4 reads structured JSON captions (see [Ideogram 4](image.md#ideogram-4)). With `ideo`, the Enhance popover has a **Write as** choice: **Text** rewrites your prompt as usual, **JSON** writes a full caption from it instead, with a summary, the look, lighting and colors, the background, one element per thing you named, and the lettering you asked for.

- The same options apply: Style decides photo or artwork, Mood shapes the look and colors, Details and Length decide how much is described, and Direction is followed.
- Quoted text (`"OPEN"`, `'LISBOA'`, `「パン屋」`) and names after *called* or *named* become text elements, copied exactly. On a card or logo, words you wrote without quotes are used too. Signs you mention without their words may get short words that fit, in the language you name.
- The caption appears in the **Enhanced** tab, laid out for reading, and Generate sends it as the structured caption. Edit it there like any Enhanced text.
- With **Enhance each image when generating**, every image gets its own caption.
- If the enhancer returns no usable caption twice, the run reports it and keeps your prompt.
- **Smart** writes noticeably better captions than Fast.

On the command line use `--enhance-format json`; it needs a model that takes JSON captions. With `--prompt` it turns on enhancement by itself. With a prompt file it writes captions for the entries that have their own `enhance:`, keeping their settings, as the Web UI does without auto-enhance; add `--enhance` to rewrite every prompt with the same settings.

### In the Web UI

- **Enhance**: click ✨ **Enhance** in the prompt box, pick options, and click **Enhance prompt**. The rewrite appears in the **Enhanced** tab. The button is unavailable while generation jobs are running or queued. Enhance again always starts from your text, and **Clear** empties the Enhanced text.
- **Video** has a **Rewrite** choice first in the popover: **Scene and action** or **Action only** (see [Rewrite](#rewrite-scene-and-action-or-action-only)). The tabs are **Your text** and **Enhanced**, and each holds a Scene and an **Action** box (the Prompt box of a video is called Action) that switch together. The Rewrite choice is remembered for each video workflow.
- **Each part decides on its own.** For the Scene and for the Action, Generate sends the Enhanced text when that part's Enhanced box has text, and your text otherwise. With Action only, the Enhanced Scene stays empty ("Not enhanced. Generate uses your scene.") and editing your Scene applies directly. Both Enhanced boxes are editable, and typing into an empty one makes that text the part's enhanced text.
- A part that sends its Enhanced text shows **Uses enhanced** beside its label in both tabs, with **Use mine**, which empties that Enhanced box so Generate uses your text again. The **Enhanced** tab shows **used** while either Enhanced box has text.
- If you change the scene or action, or switch workflows, afterwards, the Enhanced text is marked **Out of date**, and each part that uses enhanced text shows the badge too. It is still used until you choose **Use mine**, **Re-enhance** or **Clear**. Putting the text back, for example with **Use the clip's scene** when extending, clears the badge.
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
      note: rain on the pavement
  - prompt: a quiet street at dawn
    enhance: true        # default options
  - prompt: two friends talking at a cafe table
    enhance:
      sound: [ambient, dialogue]   # video with audio on
  - scene: A fisherman on a wooden pier at dawn, mist over the water.
    prompt: he casts his line
    enhance:
      parts: action                # video: keep the scene, rewrite only the action
```

An entry's own `enhance:` that asks for no change at all (style and mood keep, no details, length same, no motion or sound, no direction) is refused when the file loads: a warning appears and that entry is not enhanced. To have the enhancer split a single prompt into scene and action, use `enhance: true` or set at least one option. Job-level `--enhance` and the Web UI accept keep-all settings when the prompt has no scene. `parts: action` counts as a change, so an entry with it is kept even when every other option keeps things as they are.

From the command line, `--enhance` enhances every prompt and overrides entries' `enhance:` settings; `--no-enhance` turns all enhancement off:

```bash
ziv-image -m klein9b --prompt "a fox in snow" --enhance
ziv-image -m klein9b -p prompts.yaml --enhance style=photo,mood=serene,details=lighting+camera,length=longer
ziv-video -m ltx-8 --prompt "a fox runs" --enhance style=cinematic,motion=action+camera-move,sound=ambient+effects
ziv-video -m ltx-8 --scene "A fox in a snowy birch forest at dawn." --prompt "it runs" --enhance parts=action
ziv-image -m klein9b --prompt "a lighthouse" --enhance-note "in a storm, at night"
ziv-image -m klein9b --prompt "a fox in snow" --enhance --enhance-model smart
ziv-image -m ideo --prompt "a travel poster for Lisbon with the text 'LISBOA'" --enhance-format json
ziv-image -m klein9b -p prompts.yaml --no-enhance
```

`--enhance-note` sets the direction and turns on enhancement by itself.

`motion` and `sound` only apply to video, and `sound` only with audio on. Prompt files shared between image and video may include them; images ignore them.

The enhanced prompt is printed while prompts are enhanced and again before each generation, and stored in the image or video metadata, so Gallery shows and reuses the prompt that actually rendered.

### While Prompts Are Enhanced

Jobs with auto enhancement rewrite every prompt before the model loads. In image jobs (Web UI controls or `ziv-image` keys):

- **Next** (`n`) skips the current rewrite; that image uses the original prompt.
- **Quit** (`q`) stops the job before the model loads.
- **Pause** (`p`) waits until you resume.
- **Repeat** (`r`) has no effect.

Web UI video jobs offer only **Cancel**, which also stops the rewrites; `ziv-video` can be stopped with Ctrl+C.

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
