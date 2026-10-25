# CLI

Every feature of the app is also a command. The CLI suits batch runs, scripts and working over SSH.

| Command | Does |
|---|---|
| `ziv-image` | Generate images |
| `ziv-video` | Generate and extend videos |
| `ziv-model` | Convert checkpoints, import LoRAs, list what's installed |
| `ziv-ui` | Start the [Web UI](../web-ui/index.md) |
| `ziv` | All of the above as subcommands: `ziv image`, `ziv video`, `ziv model`, `ziv ui` |

`ziv` on its own prints help. Add `--help` to any command for its options, or see the [command reference](reference.md).

## Your first image

```bash
ziv-image -m klein4b --prompt "A red fox sitting in fresh snow, early morning light"
```

- `-m` picks the model. `klein4b` is an [alias](../concepts/models.md) for FLUX.2 Klein 4B; it downloads the first time you use it.
- The image is saved in the current folder as `prompt_<date>_<time>.png`. Use `-o` to save somewhere else.

A larger model and a different shape:

```bash
ziv-image -m klein9b --ratio 16:9 --size l --prompt "A lighthouse in a storm"
```

## Your first video

```bash
ziv-video -m ltx-4 --prompt "A cat walking through a garden, birds singing"
```

The clip has sound and is saved as an `.mp4`. The first run downloads the model and, on Windows and Linux, prepares it for your GPU.

Start from an image:

```bash
ziv-video -m ltx-4 --image photo.jpg --prompt "The camera slowly zooms in"
```

## Run many at once

Put prompts in a YAML file and run them all, three times each:

```bash
ziv-image -m klein9b -p prompts.yaml -r 3
```

See [Batches and prompt files](batches.md).

## From the Web UI to the CLI

The terminal button in the Web UI's *Settings* header copies a command with the current model, prompt and settings. Paste it to run the same job here.

## Next

- [Generate images](images.md)
- [Generate videos](video.md)
- [Command reference](reference.md)
