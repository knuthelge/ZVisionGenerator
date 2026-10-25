# Prompts and prompt files

A prompt describes what you want. You can type one prompt, or keep many in a **prompt file** that the Web UI and the CLI both read.

## Writing a prompt

Describe what is seen: the subject, where it is, the light, the framing and the style.

```text
A red fox sitting in fresh snow, early morning light, close-up, shallow depth of field
```

A short idea is fine too: the [prompt enhancer](prompt-enhancer.md) can add the detail.

## Random choices

`{a|b|c}` picks one option each run:

```text
A {red|blue} car                        → A red car, or A blue car
A {big|small} {red|blue} {car|truck}    → a new combination each run
{Nikon Z9 {50mm|35mm}|Canon EOS 5D}     → nested choices resolve inside out
```

Each run picks again, so several runs give different combinations. An empty option (`{|at night}`) means "sometimes nothing".

## Prompt files

A prompt file is YAML. Each top-level key is a **set**, holding a list of prompts:

```yaml title="prompts.yaml"
gorilla:
  - active: false
    prompt: |
      A gorilla. Detailed fur.
      In a jungle. Rainy day. Moody lighting.

woman:
  - prompt: |
      30yo woman, in red dress.
      Walking down a city street. Evening.
  - prompt: The same woman in a café, morning light
```

- Every prompt with `active: true` (the default) is generated; `active: false` skips it.
- The set's name starts the file names of what it makes: `woman_2026-10-04_14-03-22.png`.
- An entry can also have `negative:`, `scene:` and `enhance:`, described below and in [Prompt enhancer](prompt-enhancer.md#in-prompt-files).

Run one with `ziv-image -p prompts.yaml`, or pick it as the **Prompt file** source in the Workspace. The [Prompts page](../web-ui/prompts.md) edits it without YAML.

## Snippets

Put reusable pieces under the reserved `snippets` key and use them as `$name`:

```yaml
snippets:
  nina:
    Hair: Tidy
    Clothes: Red dress
  diner: a cozy 1950s American diner with checkered floors

diner_scene:
  - prompt:
      Subjects:
        - Nina: $nina
      Setting: Lisa and Nina are sharing a meal at $diner
```

- A snippet that is the whole value (`$nina`) keeps its structure.
- A snippet inside text (`at $diner`) is flattened and put in place.
- Snippets can use other snippets. A loop is reported as an error.
- `snippets` is not a set: it is never generated.

## Structured prompts

A prompt can be a nested set of named fields instead of one string. It is flattened into one prompt, with `. ` between the parts and each field's name before its value:

```yaml
diner_scene:
  - prompt:
      Subjects:
        - Lisa: An old lady with grey hair and kind eyes
        - George is a retired professor with round glasses
      Setting: Lisa and George are sharing a meal at the diner
      Style:
        Camera: iPhone
        Color: Warm
```

becomes

```text
Subjects: Lisa: An old lady with grey hair and kind eyes. George is a retired professor with round glasses. Setting: Lisa and George are sharing a meal at the diner. Style: Camera: iPhone. Color: Warm
```

- **Text** is used as is.
- **Lists** have each item flattened, joined with `. `.
- **Fields** become `name: value`, joined with `. `.

Random choices work inside fields, which makes controlled variety easy:

```yaml
character:
  - prompt:
      Subject: "A {young|old} {man|woman} with {red|blonde|dark} hair"
      Setting: "{A park|A cafe|A beach} on a {sunny|rainy} day"
      Style:
        Camera: "{Nikon Z9|Canon EOS 5D}"
```

## Negative prompts

`negative:` says what the model should avoid:

```yaml
portrait:
  - prompt: A woman in a garden
    negative: blurry, low quality, bad anatomy
```

It takes the same forms as `prompt:`. Only some models use it: Z-Image does, while FLUX.2, Krea 2 and Ideogram 4 ignore it with a warning. See [Models](models.md).

## Scenes for video

A video prompt can split into a **scene** (who and where, look, camera setup) and the **prompt**, which is the action. Add `scene:` to an entry:

```yaml
snippets:
  courtyard: A knight in a grey castle courtyard, overcast, handheld camera.

castle:
  - scene: $courtyard
    prompt: She draws her sword and turns to the gate.
  - scene: $courtyard
    prompt: She sheathes the sword and walks toward the tower.
  - prompt: A raven lands on the wall.   # uses --scene or the Workspace's Default scene, if any
```

- `scene:` belongs to an entry. To share one across a set, put it in a snippet, as above.
- The model receives the scene first, then the prompt.
- An entry without a `scene:` uses `--scene` or the Workspace's **Default scene**, if one is set.
- Image jobs ignore `scene:`, so one file can serve both.

See [Scene and action](video.md#scene-and-action).

## Tips

- Use descriptive set names: they name your files.
- Use `active: false` to park prompts without deleting them.
- Move repeated pieces into snippets.
- Put variations of one idea in the same set.
