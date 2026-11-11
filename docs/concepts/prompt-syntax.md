# Prompt syntax

A prompt says what you want. You can type one, or keep many in a **prompt file** that the Web UI and the command line both read. This page covers the syntax of both: random choices, snippets, structured prompts, negative prompts and the file format. For advice on what to write, see [Get better results](../guides/better-results.md).

## What a prompt is

Describe what is seen: the subject, where it is, the light, the framing and the style.

```text
A red fox sitting in fresh snow, early morning light, close-up, shallow depth of field
```

A short idea is fine too: the [prompt enhancer](prompt-enhancer.md) can add the detail.

## Random choices

`{a|b|c}` randomly picks one option each run:

```text
A {red|blue} car                        → A red car, or A blue car
A {big|small} {red|blue} {car|truck}    → a new combination each run
{Nikon Z9 {50mm|35mm}|Canon EOS 5D}     → nested choices resolve inside out
```

Each run picks again, so the prompt comes out different each time. Use it for either/or choices; several runs give different combinations. An empty option (`{|at night}`) means "sometimes nothing". Random choices work in prompts, scenes, negative prompts and inside the fields of a structured prompt.

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

- Every prompt with `active: true` (the default) is generated; `active: false` skips it, which parks a prompt without deleting it.
- The set's name starts the file names of what it makes, such as `woman_2026-10-04_14-03-22.png`, so give sets descriptive names.
- An entry can also have `negative:`, `scene:` and `enhance:`, described below.
- Put variations of one idea in the same set, and move repeated pieces into snippets.

The [Prompts page](../web-ui/prompts.md) edits a file without YAML. To run one, see [Run batches from prompt files](../guides/batches.md).

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

Random choices inside fields make controlled variety easy:

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

It takes the same forms as `prompt:`. Only some models use it: Z-Image and MOSS-Image SFT (`moss-sft`) do, while FLUX.2, Krea 2, Ideogram 4 and MOSS-Image Turbo (`moss`) ignore it with a warning. See [Models and licences](../reference/models.md).

The Workspace has a **Negative prompt** box for the models that use one. The command line has no flag for it, so use `negative:` in a prompt file.

## Scenes for video

A video prompt splits into a **scene** (who and where, the look, the camera setup) and the **prompt**, which is the action. Add `scene:` to an entry:

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

What a scene is and how to write one is in [Make video](../guides/video.md#scene-and-action).

## The enhance: key { #the-enhance-key }

Add `enhance:` to an entry to run the [prompt enhancer](prompt-enhancer.md) on it every time. `enhance: true` is a general rewrite; a mapping steers it with the options of the enhancer, each written by its key:

```yaml
woman:
  - prompt: 30yo woman in red dress walking down a city street, evening
    enhance:
      style: cinematic
      mood: mysterious
      theme: noir
      details: [lighting, camera]
      length: longer
      note: rain on the pavement
  - prompt: a quiet street at dawn
    enhance: true                  # a general rewrite
  - prompt: a ceramic teapot on a wooden table
    enhance:
      angle: overhead
      palette: custom
      colors: ["#1B3A4B", "#E07A5F"]   # quote hex codes, up to five
  - prompt: two friends talking at a cafe table
    enhance:
      sound: [ambient, dialogue]   # video with audio on
  - scene: A fisherman on a wooden pier at dawn, mist over the water.
    prompt: he casts his line
    enhance:
      parts: action                # video: keep the scene, rewrite the action
```

- The keys are `style`, `mood`, `theme`, `shot`, `angle`, `palette`, `colors`, `details`, `camera`, `pace`, `motion`, `sound`, `length`, `note` and `parts`. Options that take several values (`details`, `motion`, `sound`) take a list. The values are listed in [the spec](../cli/reference.md#the-spec) and what each option does is in [Options](prompt-enhancer.md#options).
- In an image run, an entry whose `enhance:` sets only video options (`camera`, `pace`, `motion`, `sound` or `parts`) is not enhanced, and a warning says so. Video options are ignored for images, so one file can serve both.
- `--enhance` on the command line overrides the entries' settings and `--no-enhance` turns them all off; see [Enhancing every prompt in a job or file](prompt-enhancer.md#enhance-every-prompt).
