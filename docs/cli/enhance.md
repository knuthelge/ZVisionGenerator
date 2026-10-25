# Enhance prompts

The [prompt enhancer](../concepts/prompt-enhancer.md) rewrites a prompt with more visual detail, using a small model on your machine. From the CLI, turn it on with `--enhance`.

## Quick examples

```bash
# A general rewrite: adds detail where the prompt is thin
ziv-image -m klein9b --prompt "a fox in snow" --enhance

# Steer it: a style, a mood, what to describe, how long
ziv-image -m klein9b --prompt "a fox in snow" --enhance style=photo,mood=serene,details=lighting+camera,length=longer

# Add a direction in your own words
ziv-image -m klein9b --prompt "a lighthouse" --enhance-note "in a storm, at night"

# Use the larger Smart model
ziv-image -m klein9b --prompt "a fox in snow" --enhance --enhance-model smart

# Video: camera, pace, motion and sound
ziv-video -m ltx-8 --prompt "a fox runs" --enhance style=cinematic,camera=tracking,pace=fast,motion=action+secondary,sound=ambient+effects

# Video: keep your scene, rewrite only the action
ziv-video -m ltx-8 --scene "A fox in a snowy birch forest at dawn." --prompt "it runs" --enhance parts=action

# Ideogram 4: write a JSON caption
ziv-image -m ideo --prompt "a travel poster for Lisbon with the text 'LISBOA'" --enhance-format json
```

## The spec

`--enhance` takes comma-separated `key=value` pairs; join several values with `+`. Leave a key out to keep its default. With no spec at all you get a general rewrite.

| Key | Values |
|---|---|
| `style` | `keep`; photo: `photo`, `candid`, `street`, `film`, `bw`, `portrait`, `product`, `cinematic`; art: `illustration`, `anime`, `comic`, `3d`, `painterly`, `watercolor`, `gouache`, `sketch`, `pixel`, `concept`, `graphic`, `isometric`, `claymation`, `ukiyo-e`, `papercraft` |
| `mood` | `keep`, `serene`, `cozy`, `joyful`, `romantic`, `whimsical`, `nostalgic`, `melancholic`, `mysterious`, `eerie`, `dramatic`, `epic` |
| `theme` | `keep`, `zen`, `spiritual`, `rustic`, `sensual`, `glamorous`, `gothic`, `noir`, `horror`, `dystopian`, `fairytale`, `mythic`, `surreal`, `retro`, `futuristic`, `cyberpunk`, `steampunk` |
| `details` | Any of `lighting`, `composition`, `camera`, `materials`, `color`, `environment`, `subject`, `expression` |
| `length` | `shorter`, `same` (default), `longer`, `extra` |
| `camera` (video) | `keep`, `auto`, `static`, `push-in`, `pull-back`, `pan`, `tilt`, `tracking`, `orbit`, `handheld`, `aerial` |
| `pace` (video) | `keep`, `slow-motion`, `slow`, `brisk`, `fast` |
| `motion` (video) | Any of `action`, `secondary` |
| `sound` (video with audio) | Any of `ambient`, `effects`, `dialogue`, `music` |
| `parts` (video) | `both` or `action` |

What each option asks for is in [Prompt enhancer options](../concepts/prompt-enhancer.md#options).

## The other flags

| Flag | Does |
|---|---|
| `--enhance-note TEXT` | A direction in your own words, up to 300 characters. Turns on `--enhance` by itself. |
| `--enhance-model` | `fast` (default), `smart`, a Hugging Face `REPO[@REVISION]`, or a local folder (`./smart` for a folder named `smart`). Where Smart doesn't fit, a warning says so and Fast runs. |
| `--enhance-format json` | Write an Ideogram 4 JSON caption instead of a prompt. With `--prompt` it turns on enhancement by itself. |
| `--no-enhance` | Turn enhancement off, including `enhance:` entries in a prompt file. |

## With prompt files

An entry's `enhance:` enhances it on every run:

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
    enhance: true        # a general rewrite
```

`--enhance` on the command line enhances every prompt and overrides the entries' settings; `--no-enhance` turns all of it off. With `--enhance-format json` and a prompt file, only entries with their own `enhance:` get captions, unless you add `--enhance`.

## What you see

The enhanced prompt is printed while prompts are rewritten and again before each image. It's stored in the file with your own prompt and the options, so the Web UI's viewer shows both and **Reuse settings** brings them back.

All rewrites happen before the image or video model loads, and the enhancer is unloaded first, so it adds no memory during generation.
