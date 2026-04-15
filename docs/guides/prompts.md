# Prompts Guide

Z-Vision Generator supports inline prompts, YAML prompt files with batch generation, prompt variables, structured prompts, and reusable snippets. This guide covers the full prompts system used by both `ziv` and `ziv-video`.

## Inline Prompts

Use `--prompt` for quick, one-off generation:

```bash
ziv -m my-model --prompt "a beautiful sunset over the ocean"
ziv-video -m ltx-4 --prompt "A cat walking through a garden"
```

When `--prompt` is provided, it overrides `--prompts-file`.

## Prompt Files

Use `--prompts-file` (or `-p`) to load prompts from a YAML file. The default file is `prompts.yaml`.

```bash
ziv -m my-model -p prompts.yaml -r 3
ziv -m my-model -p my-prompts.yaml
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

## Prompt Variables

Use `{option1|option2|option3}` syntax for random selection each run:

```
"A {red|blue} car"                         → "A red car" or "A blue car"
"A {big|small} {red|blue} {car|truck}"     → random combo each run
"{Nikon Z9 {50mm|35mm}|Canon EOS 5D}"     → nesting resolves inside-out
```

Variables are resolved independently each run, so repeated runs (`-r 3`) produce different combinations. The `{a|b|c}` random choice syntax works within structured prompt values too.

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
