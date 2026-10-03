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

In the Web UI, choose **Prompt file** as the prompt source and type or browse to a YAML file on the machine running the server. Compose then summarizes the selected prompts; click **Prompts…** (or the summary) to open the chooser, where you can filter prompts, select whole sets with **All**/**None**, expand a prompt to see its negative prompt, and confirm with **Use N prompts**. **✨ Enhance** sets the options for **Enhance each image when generating**, and **Edit YAML** opens the file editor. The YAML editor reloads the file from disk each time you open it, so reopening the same path reflects changes made outside the browser. If you edit the path manually, the visible value is the value submitted; use Enter or Browse when you want the UI to inspect the file and refresh the option list before generating.

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

Without a CUDA GPU, Windows and Linux run the enhancer on the CPU: it works, but each rewrite can take a few minutes and about 10 GB of RAM. The status line says so when this happens.

Both defaults are pinned to a revision. To use another chat model, set **Prompt Enhancer Model** on the Config page to a Hugging Face `owner/name` (optionally `owner/name@revision`) or a local folder, or pass `--enhance-model REPO[@REVISION]`. Thinking-tuned or heavily merged models are slower and follow the length and style options less reliably.

### Options

| Option | Choose | Values |
|---|---|---|
| Style | one | Keep, Photographic, Cinematic, Illustration, Anime, 3D render, Painterly |
| Details | any | Lighting, Composition, Camera & lens, Materials & textures, Color & mood, Environment, Subject |
| Length | one | Shorter (≈50 %), Same, Longer (≈200 %), Extra long (≈300 %) |
| Motion (video only) | any | Action / sequence, Camera movement, Pacing |

Length is a target, not a guarantee. Longer aims for at least 40 words and Extra long for at least 80, so short prompts still grow; Shorter never aims below 12 words, so a prompt of 12 words or fewer keeps about the same length. Results are capped at 300 words (180 for FLUX.1, whose text encoder reads fewer tokens); when a prompt is already at the cap, it keeps about the same length. On short prompts, Same is approximate.

The enhancer keeps your subject and every detail you describe. It adds only things that can be seen: no sounds, smells, or story. It never adds new people, animals, or major objects.

### In the Web UI

- **Enhance**: click ✨ **Enhance** in the prompt box, pick options, and click **Enhance prompt**. The rewrite appears in the **Enhanced** tab. When that tab has text, it is what gets generated; the **used** badge shows which tab that is. Edit it, enhance again, or **Clear** it to go back to your prompt.
- If you change the prompt or switch between image and video afterwards, the Enhanced prompt is marked **Out of date**. It is still used until you **Re-enhance** or **Clear** it.
- `{a|b}` choices are picked before the rewrite, so the enhancer sees one plain prompt and the Enhanced prompt has no choices left. Enhance again for a different pick, or use **Enhance each image when generating** to keep a fresh pick per image.
- **Enhance each image when generating** rewrites every image's prompt on the server, after its `{a|b}` choices are picked, using that image's seed. It works with inline prompts and prompt files.
- Enhance is unavailable while a job runs.

### In Prompt Files and the CLI

Add `enhance:` to an entry to enhance it on every run:

```yaml
woman:
  - prompt: 30yo woman in red dress walking down a city street, evening
    enhance:
      style: cinematic
      details: [lighting, camera]
      length: longer
  - prompt: a quiet street at dawn
    enhance: true        # default options
```

From the command line, `--enhance` enhances every prompt and overrides entries' `enhance:` settings; `--no-enhance` turns all enhancement off:

```bash
ziv-image -m klein9b --prompt "a fox in snow" --enhance
ziv-image -m klein9b -p prompts.yaml --enhance style=photo,details=lighting+camera,length=longer
ziv-video -m ltx-8 --prompt "a fox runs" --enhance style=cinematic,motion=action+camera-move
ziv-image -m klein9b -p prompts.yaml --no-enhance
```

`motion` only applies to video. Prompt files shared between image and video may include it; images ignore it.

The enhanced prompt is printed before each generation and stored in the image or video metadata, so Gallery shows and reuses the prompt that actually rendered.

### Memory

On macOS the enhancer shares memory with the image or video model. Clicking Enhance loads it (about 3 GB) and frees it after two idle minutes, or when a job without enhancement starts. **Enhance each image** keeps it loaded for the whole job, in addition to the model; the memory badge does not include it, so on a 16 GB Mac with a large model, leave it off.

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
