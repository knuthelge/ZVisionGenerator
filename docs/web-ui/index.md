<div class="zb zb--web-ui" markdown>

# Web UI

The Web UI is a browser app that runs on your computer. You write prompts, generate, look through what you made and manage your models, all in one place.

</div>

## Start it

```bash
ziv ui
```

It opens your browser at `http://127.0.0.1:8080/`. If port 8080 is busy it takes the next free port and prints the address in the terminal. Stop it with ++ctrl+c++ in that terminal.

| To… | Run |
|---|---|
| Start without opening a browser | `ziv ui --no-browser` |
| Use another port | `ziv ui --port 9000` |
| Open it from a phone, tablet or another computer | `ziv ui --host local` (see [Other devices](devices.md)) |
| Keep it running after you close the terminal | `ziv ui --background`, and stop it with `ziv ui --background-stop` |

`ziv-ui` is the same as `ziv ui`. All options are in the [command reference](../cli/reference.md#ziv-ui-ziv-ui).

## The screen

![The Workspace: the top bar, Compose and Settings on the left, the preview with the Gallery strip below](../assets/screenshots/workspace.webp)

The **top bar** has the pages and, on the Workspace, the four tabs:

| Tab | Makes |
|---|---|
| Image | An image from a prompt, or from a start image you add in **Images** |
| Video | A video from a prompt, or one that starts on a first frame you add in **Images** |
| Extend Video | More of a video you already have |
| Audio | Music, speech or sound, as FLAC files |

The pages:

| Page | For | Shortcut |
|---|---|---|
| [Workspace](workspace.md) | Writing prompts, choosing settings and generating | ++g++ ++w++ |
| [Prompts](prompts.md) | Building prompt files without typing YAML | ++g++ ++p++ |
| [Gallery](gallery.md) | Everything you made, with its settings | ++g++ ++g++ |
| [Models](models.md) | Downloading, converting and deleting models and LoRAs | ++g++ ++m++ |
| [Config](config.md) | Default models, base resolution and output folder | ++g++ ++c++ |
| [Job log](job-log.md) | Jobs that have finished, and why one failed | ++g++ ++j++ |

Press <kbd>?</kbd> anywhere outside a text field for every [keyboard shortcut](shortcuts.md).

## Next

[Your first image](first-image.md){ .md-button .md-button--primary } [Your first video](first-video.md){ .md-button }
